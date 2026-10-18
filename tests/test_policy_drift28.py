"""Small synthetic arrays/ZIPs; no real Torch import, checkpoint or physics."""
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('policy_drift28_test',ROOT/'tools28/audit_policy_drift28.py')
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


@pytest.fixture
def np():return pytest.importorskip('numpy')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def tensor(value):
    class FakeTensor:
        def __init__(self):self.shape=value.shape;self.dtype=value.dtype;self.device=SimpleNamespace(type='cpu')
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return value
    return FakeTensor()


def library(np,bias=0.):
    state={k:tensor(np.zeros(shape,dtype=np.float32)) for k,shape in r.SHAPES.items()}
    state['action_net.bias']=tensor(np.full(16,bias,dtype=np.float32))
    loads=[];linear=[]
    def load(stream,**kwargs):
        loads.append((stream.read(),kwargs));return state
    def operation(value,weight,bias):
        linear.append((value.shape,weight.shape));return value@weight.T+bias
    t=SimpleNamespace(load=load,float32=np.float32,from_numpy=np.asarray,tanh=np.tanh,
        cat=lambda arrays,dim:np.concatenate(arrays,axis=dim),no_grad=contextlib.nullcontext)
    return SimpleNamespace(np=np,torch=t,functional=SimpleNamespace(linear=operation),
        as_numpy=lambda value:value.numpy().copy() if hasattr(value,'numpy') else value.copy(),
        versions={'numpy':'synthetic','torch':'synthetic'},state=state,loads=loads,linear=linear)


def labels():
    return [{'sequence':r.TASKS[i%3],'window':'drive' if i%2==0 else 'stop'} for i in range(216)]


def test_default_is_plan_only_and_framework_cold():
    code='import importlib.util,sys;from pathlib import Path;p=Path("tools28/audit_policy_drift28.py");s=importlib.util.spec_from_file_location("cold28",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);assert m.main([])==0;assert not any(k.partition(".")[0] in {"torch","numpy","mujoco","gymnasium","stable_baselines3","onnxruntime"} for k in sys.modules)'
    process=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True,text=True)
    value=json.loads(process.stdout)
    assert value['status']=='PLAN_ONLY' and value['fixed_input_rows']==216
    assert value['limits']['actor_batch_calls']==3 and value['limits']['actor_sample_rows']==648


@pytest.mark.parametrize('fault',[None,'shape','nan','extra'])
def test_weights_only_cpu_loader_exact_schema_and_no_optimizer(tmp_path,np,fault):
    path=tmp_path/'policy.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('policy.pth',b'only synthetic state bytes')
    lib=library(np);counts=r.counters_zero()
    if fault=='shape':lib.state['log_std']=tensor(np.zeros(15,dtype=np.float32))
    elif fault=='nan':lib.state['log_std']=tensor(np.full(16,np.nan,dtype=np.float32))
    elif fault=='extra':lib.state['optimizer']=None
    if fault:
        with pytest.raises(ValueError):r.load_state(path,lib,counts)
        assert counts['checkpoint_load_attempts']==1 and counts['checkpoint_load_returns']==0
    else:
        weights=r.load_state(path,lib,counts)
        assert set(weights)==set(r.SHAPES) and counts['checkpoint_load_returns']==1
    assert lib.loads==[(b'only synthetic state bytes',{'weights_only':True,'map_location':'cpu'})]


def test_complete_six_linear_actor_matches_analytic_constant_policy_and_counts(np):
    lib=library(np);weights={k:np.zeros(shape,dtype=np.float32) for k,shape in r.SHAPES.items()}
    weights['action_net.bias']=np.linspace(-2.,2.,16,dtype=np.float32)
    counts=r.counters_zero();result=r.actor_batch(np.zeros((216,627),dtype=np.float32),weights,lib,counts)
    assert np.array_equal(result['raw_mean'][0],weights['action_net.bias'])
    assert np.array_equal(result['clipped_mean'],np.clip(result['raw_mean'],-1,1))
    assert [shape[1] for shape in lib.linear]==[(128,627),(64,128),(3,64),(128,60),(128,128),(16,128)]
    assert counts['actor_batch_returns']==1 and counts['sample_rows_returned']==216 and counts['linear_returns']==6


@pytest.mark.parametrize('fault',['dtype','shape','nan'])
def test_actor_rejects_invalid_batch_before_any_forward(np,fault):
    array=np.zeros((216,627),dtype=np.float32)
    if fault=='dtype':array=array.astype(np.float64)
    elif fault=='shape':array=array[:215]
    else:array[0,0]=np.nan
    counts=r.counters_zero()
    with pytest.raises(ValueError):r.actor_batch(array,{},library(np),counts)
    assert counts['actor_batch_attempts']==counts['linear_attempts']==0


def test_closed_form_gaussian_kl_direction_raw_and_clipped_distinction(np):
    base={'raw_mean':np.zeros((216,16),dtype=np.float32),'clipped_mean':np.zeros((216,16),dtype=np.float32),'log_std':np.zeros(16,dtype=np.float32)}
    target={'raw_mean':np.ones((216,16),dtype=np.float32),'clipped_mean':np.ones((216,16),dtype=np.float32),'log_std':np.zeros(16,dtype=np.float32)}
    result=r.comparisons({'base25':base,'control27':target,'yaw3_27':base},labels(),np)
    assert result['control27']['KL_base25_to_new']['mean']==8.
    assert result['yaw3_27']['KL_base25_to_new']['max']==0.
    # Equal raw means with sigma_new=2: KL(base||new)=16*(log2+1/8-1/2).
    wide={**base,'log_std':np.full(16,np.log(2),dtype=np.float32)}
    result=r.comparisons({'base25':base,'control27':wide,'yaw3_27':base},labels(),np)
    assert result['control27']['KL_base25_to_new']['mean']==pytest.approx(16*(np.log(2)+.125-.5),abs=1e-6)


@pytest.fixture
def bundle(tmp_path,np):
    traces=[]
    for task in r.TASKS:
        for delay in (0,1):
            for seed in (1001,1002,1003):
                path=tmp_path/f'{task}_{delay}_{seed}.npz'
                with path.open('wb') as stream:np.savez_compressed(stream,observation=np.broadcast_to(np.arange(600,dtype=np.float32)[:,None],(600,627)))
                traces.append({'path':str(path),'sequence':task,'delay_ticks':delay,'reset_seed':seed})
    roles={}
    for role in r.ROLES:
        path=tmp_path/(role+'.zip')
        with zipfile.ZipFile(path,'w') as z:z.writestr('policy.pth',role.encode())
        roles[role]={'checkpoint':str(path)}
    return {'traces':traces,'roles':roles,'versions':{'numpy':'synthetic','torch':'synthetic'},
        'inputs':{'synthetic_inputs':'never real model bytes'},'reference_distribution':'synthetic pure fixture'}


def test_fixed_rows_preserve_all_eighteen_cases_and_exact_tick_order(bundle,np):
    batch,rows=r.fixed_inputs(bundle['traces'],np)
    assert batch.shape==(216,627) and len(rows)==216
    assert np.array_equal(batch[:12,0],r.TICKS)
    assert len({(x['sequence'],x['delay_ticks'],x['reset_seed']) for x in rows})==18
    assert sum(x['window']=='drive' for x in rows)==sum(x['window']=='stop' for x in rows)==108


def test_execute_has_exact_three_batches648_rows_no_real_torch_and_refuses_retry(tmp_path,bundle,np):
    lib=library(np)
    result=r.execute(root=tmp_path,input_loader=lambda root:copy.deepcopy(bundle),source_loader=lambda root:{'synthetic_source':True},
        library_loader=lambda:lib,memory_reader=lambda:{'effective_available_bytes':6<<30})
    assert result['status']=='COMPLETE' and result['training_transitions']==result['physics_steps']==0
    assert result['known_actual_counts']==dict(checkpoint_load_attempts=3,checkpoint_load_returns=3,
        actor_batch_attempts=3,actor_batch_returns=3,sample_rows_attempted=648,sample_rows_returned=648,linear_attempts=18,linear_returns=18)
    assert result['qualification_granted'] is False and len(lib.loads)==3
    assert (tmp_path/r.OUTPUT/'distributions.npz').is_file()
    with pytest.raises(ValueError,match='no retry'):r.execute(root=tmp_path)


@pytest.mark.parametrize('fault',['library','weights','ram','drift','wall'])
def test_failure_receipt_preserves_actual_prefix_without_retry(tmp_path,bundle,np,fault):
    lib=library(np);loads=0;clock=[0.];reads=0
    def inputs(root):
        nonlocal loads
        loads+=1
        value=copy.deepcopy(bundle)
        if fault=='drift' and loads>1:value['inputs']={'changed':True}
        return value
    def libs():
        if fault=='library':raise RuntimeError('synthetic import failure')
        if fault=='weights':lib.state['log_std']=tensor(np.zeros(15,dtype=np.float32))
        if fault=='wall':clock[0]=121.
        return lib
    def memory():
        nonlocal reads
        reads+=1
        return {'effective_available_bytes':(4<<30)-1 if fault=='ram' and reads>1 else 6<<30}
    result=r.execute(root=tmp_path,input_loader=inputs,source_loader=lambda root:{'same':True},library_loader=libs,memory_reader=memory,clock=lambda:clock[0])
    assert result['status']=='INCOMPLETE' and result['error']
    assert result['known_actual_counts']['sample_rows_returned']==(648 if fault=='drift' else 0)
    assert (tmp_path/r.OUTPUT/'report.json').is_file()


def test_inspector_accepts_real_local_fixed_metadata_without_tensor_or_nn_only_when_present():
    # Optional local identity check, never a required synthetic-CI substitute.
    if not (ROOT/next(iter(r.PINS))).is_file():pytest.skip('local frozen receipts absent')
    value=r.inspect_inputs()
    assert len(value['traces'])==18 and len(value['roles'])==3
    assert value['qualification_granted'] is False
