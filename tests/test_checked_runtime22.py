"""Standard-library fake pipes/processes only; never creates a worker or model."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools22 import checked_runtime22 as c


def counters():
    return [{'policy_transitions':2,'native_steps':20,'pd_updates':10,'resets':1,'forward_calls':4} for _ in range(4)]


def delays():
    return [{'delay_counts':{'0':1,'1':0,'2':0,'3':0},'last_selection':{
        'mode':'zero','seed':181+rank,'rank':rank,'episode_index':0,'stream':18018,
        'selected_delay_ticks':0,'selection':'zero'}} for rank in range(4)]


class FakeProcess:
    _parent_pid = os.getpid()
    def __init__(self,pid):
        self.pid,self.alive,self.actions=pid,True,[]
    def is_alive(self):
        return self.alive
    def terminate(self):
        self.actions.append('terminate')
        self.alive=False
    def kill(self):
        self.actions.append('kill')
        self.alive=False
    def join(self,timeout=None):
        self.actions.append(('join',timeout))


class FakePipe:
    def __init__(self):
        self.closed=False
    def close(self):
        self.closed=True
    def recv(self):
        pytest.fail('dirty close tried to receive a stale reply')
    def send(self,*args):
        pytest.fail('dirty close tried to send another RPC')


class FakeSubproc:
    def __init__(self,env_fns,start_method):
        assert len(env_fns)==4 and start_method=='forkserver'
        self.calls,self.waiting,self.closed=[],False,False
        self.processes=[FakeProcess(10000+i) for i in range(4)]
        self.remotes=[FakePipe() for _ in range(4)]
        self.work_remotes=[]
        self.response={'counters':counters(),'get_delay_state18':delays()}
        self.observation=object()
        self.fault=None
        self.entered,self.release=threading.Event(),threading.Event()
        self.get_attr('render_mode')  # Same constructor RPC as actual SB3 VecEnv.
    def get_attr(self,name,indices=None):
        self.calls.append(('get_attr',name,indices))
        return [None]*4
    def has_attr(self,name):
        self.calls.append(('has_attr',name))
        return True
    def set_attr(self,name,value,indices=None):
        self.calls.append(('set_attr',name,value,indices))
    def env_is_wrapped(self,cls,indices=None):
        self.calls.append(('env_is_wrapped',cls,indices))
        return [False]*4
    def env_method(self,name,*args,indices=None,**kwargs):
        self.calls.append(('env_method',name))
        if self.fault=='rpc':
            raise EOFError('synthetic channel failure')
        if self.fault=='blocked':
            self.entered.set()
            assert self.release.wait(2),'test forgot to release fake RPC'
        return self.response[name]
    def reset(self):
        self.calls.append(('reset',))
        if self.fault=='reset':
            raise InterruptedError('after reset send, before reply')
        return self.observation
    def step_async(self,actions):
        self.calls.append(('step_async',actions))
        if self.fault=='partial_send':
            raise InterruptedError('after first send, before waiting flag')
        self.waiting=True
    def step_wait(self):
        self.calls.append(('step_wait',))
        if self.fault=='partial_recv':
            raise InterruptedError('after first worker recv')
        self.waiting=False
        return self.observation
    def close(self):
        self.calls.append(('close',))
        assert not self.waiting
        for process in self.processes:
            process.alive=False
        self.closed=True


@pytest.fixture
def vec():
    cls=c.checked_subproc_class22(FakeSubproc)
    return cls([None]*4,mode='zero',seed=181,max_transitions=100)


def test_cold_import_has_no_model_physics_or_numpy_import():
    code='from tools22 import checked_runtime22;import sys;assert not {"torch","mujoco","numpy","onnxruntime","stable_baselines3"}&sys.modules.keys()'
    value=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10,check=False)
    assert value.returncode==0,value.stderr


def test_healthy_path_keeps_original_request_order_and_observation_identity(vec):
    baseline=FakeSubproc([None]*4,'forkserver')
    for target in (baseline,vec):
        assert target.reset() is target.observation
        target.step_async('same_actions')
        assert target.step_wait() is target.observation
        assert target.env_method('counters')==counters()
        assert target.env_method('get_delay_state18')==delays()
        target.close()
    assert vec.calls==baseline.calls
    assert vec.rpc_guard22.snapshot()['rpc_validated']==2
    assert vec.rpc_guard22.snapshot()['closed'] and not vec.rpc_guard22.snapshot()['dirty']


def test_checked_runtime_only_delegates_declared_metadata_methods(vec):
    runtime=c.CheckedRuntime22()
    assert runtime.counters(vec)==counters()
    assert runtime.delay_state(vec)==delays()
    assert runtime.rpc_snapshot(vec)['rpc_validated']==2
    assert runtime.rpc_snapshot(None)=={'state':'NOT_CREATED','dirty':False}


def test_waiting_step_refuses_metadata_without_sending_and_closes_without_rpc(vec):
    vec.step_async('actions')
    previous=list(vec.calls)
    with pytest.raises(c.RpcError22):
        vec.env_method('counters')
    assert vec.calls==previous and vec.rpc_guard22.snapshot()['dirty']
    vec.close()
    assert vec.calls==previous and all(not p.is_alive() for p in vec.processes)
    assert all(pipe.closed for pipe in vec.remotes)
    assert vec.rpc_guard22.snapshot()['cleanup']['rpc_used'] is False


@pytest.mark.parametrize('fault,operation',[('partial_send','step_async'),('partial_recv','step_wait'),('reset','reset')])
def test_interrupted_step_or_reset_poisons_even_before_waiting_flag(vec,fault,operation):
    vec.fault=fault
    if operation=='step_wait':
        vec.step_async('actions')
    with pytest.raises(InterruptedError):
        getattr(vec,operation)(*(['actions'] if operation=='step_async' else []))
    assert vec.rpc_guard22.snapshot()['dirty']
    previous=list(vec.calls)
    with pytest.raises(c.RpcError22):
        vec.env_method('counters')
    vec.close()
    assert vec.calls==previous


@pytest.mark.parametrize('fault',['step_tuple','arraylike','bool','float','negative','missing','impossible','overcap'])
def test_bad_counter_packet_is_never_committed_or_coerced(vec,fault):
    class ArrayLike:
        def tolist(self):
            pytest.fail('coercion would disguise a wrong reply')
    candidate=counters()
    if fault=='step_tuple':
        candidate=[('observation',1.,False,{}, {})]*4
    elif fault=='arraylike':
        candidate=ArrayLike()
    elif fault in ('bool','float','negative'):
        candidate[0]['resets']={'bool':True,'float':1.,'negative':-1}[fault]
    elif fault=='missing':
        del candidate[0]['forward_calls']
    elif fault=='impossible':
        candidate[0]['native_steps']=21
    else:
        candidate[0].update(policy_transitions=101,native_steps=1010,pd_updates=505)
    vec.response['counters']=candidate
    with pytest.raises(c.RpcError22):
        vec.env_method('counters')
    assert vec.rpc_guard22.last_counters is None
    previous=list(vec.calls)
    with pytest.raises(c.RpcError22):
        vec.env_method('get_delay_state18')
    assert vec.calls==previous
    assert c.json_ready22(vec.rpc_guard22.snapshot())['dirty']


def test_json_ready_refuses_nonfinite_and_arraylike_without_conversion():
    for value in ({'nan':float('nan')},object()):
        with pytest.raises((ValueError,TypeError)):
            c.json_ready22(value)


@pytest.mark.parametrize('fault',['rank','seed','episode','stream','histogram','reset_count','mode'])
def test_delay_identity_checked_before_value_can_enter_receipt(vec,fault):
    vec.env_method('counters')
    rows=delays()
    if fault=='rank':rows[1]['last_selection']['rank']=0
    elif fault=='seed':rows[1]['last_selection']['seed']=181
    elif fault=='episode':rows[0]['last_selection']['episode_index']=1
    elif fault=='stream':rows[0]['last_selection']['stream']=999
    elif fault=='histogram':rows[0]['delay_counts']['0']=True
    elif fault=='reset_count':rows[0]['delay_counts']['0']=2
    else:rows[0]['last_selection']['mode']='uniform_0_3'
    vec.response['get_delay_state18']=rows
    with pytest.raises(c.RpcError22):
        vec.env_method('get_delay_state18')
    assert vec.rpc_guard22.snapshot()['rpc_validated']==1 and vec.rpc_guard22.snapshot()['dirty']


def test_late_reply_after_outer_timeout_cannot_commit_or_allow_next_rpc(vec):
    vec.fault='blocked'
    try:
        with pytest.raises(TimeoutError):
            c.bounded_call(lambda:vec.env_method('counters'),timeout=.01)
        assert vec.entered.is_set() and vec.rpc_guard22.snapshot()['inflight']=='counters'
        previous=list(vec.calls)
        with pytest.raises(c.RpcError22):
            vec.env_method('get_delay_state18')
        vec.close()
        assert vec.calls==previous
    finally:
        vec.release.set()
    for _ in range(100):
        if vec.rpc_guard22.snapshot()['inflight'] is None:
            break
        threading.Event().wait(.001)
    assert vec.rpc_guard22.last_counters is None and vec.rpc_guard22.snapshot()['rpc_validated']==0


def test_rpc_error_marks_dirty_and_does_not_send_followup(vec):
    vec.fault='rpc'
    with pytest.raises(EOFError):vec.env_method('counters')
    previous=list(vec.calls)
    with pytest.raises(c.RpcError22):vec.env_method('get_delay_state18')
    assert vec.calls==previous


def test_counter_values_are_json_copies_and_cumulative(vec):
    first=vec.env_method('counters')
    first[0]['resets']=999
    assert vec.rpc_guard22.last_counters[0]['resets']==1
    vec.response['counters'][0]['policy_transitions']=1
    vec.response['counters'][0]['native_steps']=10
    vec.response['counters'][0]['pd_updates']=5
    with pytest.raises(c.RpcError22,match='regressed'):
        vec.env_method('counters')


def test_delays_cannot_use_counter_snapshot_from_an_earlier_step(vec):
    vec.env_method('counters')
    vec.step_async('actions')
    vec.step_wait()
    with pytest.raises(c.RpcError22,match='same idle boundary'):
        vec.env_method('get_delay_state18')


def test_close_checked_preserves_normal_close_and_dirty_close(vec):
    assert c.close_checked22(vec)==(True,None)
    assert vec.rpc_guard22.snapshot()['cleanup']['cleanup_complete'] is True
    assert vec.rpc_guard22.snapshot()['cleanup']['rpc_used'] is True
    assert vec.calls[-1]==('close',)
    assert c.close_checked22(vec)==(True,None)


def test_checked_close_timeout_marks_dirty_and_forces_only_owned_workers(vec,monkeypatch):
    def timeout(function,timeout):
        assert timeout==3.
        raise TimeoutError('synthetic close still in flight')
    monkeypatch.setattr(c,'bounded_call',timeout)
    clean,error=c.close_checked22(vec)
    assert clean and 'TimeoutError' in error and vec.rpc_guard22.snapshot()['dirty']
    assert ('close',) not in vec.calls and all(not process.is_alive() for process in vec.processes)


def test_undeclared_metadata_request_poisons_without_sending(vec):
    previous=list(vec.calls)
    with pytest.raises(c.RpcError22,match='declared'):
        vec.env_method('counters',indices=[0])
    assert vec.calls==previous and vec.rpc_guard22.snapshot()['dirty']
    assert c.close_checked22(vec)==(True,None)
    assert vec.calls==previous


def test_forced_cleanup_never_signals_foreign_process_handle(vec):
    foreign=vec.processes[0]
    foreign._parent_pid=os.getpid()+1
    vec.rpc_guard22.mark_dirty('test')
    result=c.close_owned22(vec)
    assert not result['cleanup_complete'] and foreign.actions==[]


def test_optimizer_hash_excludes_only_learning_rate():
    data={'state':{0:{'step':4,'exp_avg':[.1,.2],'exp_avg_sq':[.3,.4]}},
          'param_groups':[{'params':[0],'lr':3e-4,'betas':(.9,.999)}]}
    class Optimizer:
        def state_dict(self):return data
    optimizer=Optimizer()
    original=c.optimizer_without_lr_sha22(optimizer)
    full=c.optimizer_sha(optimizer)
    data['param_groups'][0]['lr']=1e-4
    assert c.optimizer_without_lr_sha22(optimizer)==original and c.optimizer_sha(optimizer)!=full
    data['state'][0]['exp_avg'][0]+=.001
    assert c.optimizer_without_lr_sha22(optimizer)!=original


def test_runtime_model_factory_is_driver_owned_and_receives_unchanged_hyperparameters():
    hyper={'learning_rate':1e-4}
    marker=object()
    def factory(runtime,vec,plan,parameters,output):
        assert parameters is hyper and vec=='guarded' and plan=={'seed':181}
        runtime.initial_identity={'driver':'explicit'}
        return marker
    runtime=c.CheckedRuntime22(model_factory=factory)
    assert runtime.create_model('guarded',{'seed':181},hyper,Path('.')) is marker
    assert runtime.initial_identity=={'driver':'explicit'} and hyper=={'learning_rate':1e-4}


def test_learning_and_environment_factory_math_are_inherited_or_ast_identical():
    assert c.CheckedRuntime22.learn is c.Runtime18.learn
    assert c.CheckedRuntime22.prepare is c.Runtime18.prepare
    assert c.CheckedRuntime22.policy_sha is c.Runtime18.policy_sha
    def nodes(path,clsname):
        tree=ast.parse(path.read_text())
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==clsname)
        method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='create_vec')
        return {node.name:ast.dump(node,include_attributes=False) for node in method.body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    assert nodes(ROOT/'tools18/train_delay18.py','RealRuntime')==nodes(ROOT/'tools22/checked_runtime22.py','CheckedRuntime22')
