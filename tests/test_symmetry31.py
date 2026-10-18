import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools31.symmetry31 import combine, mirror_action, mirror_observation  # noqa: E402
from tools31 import run_symmetry31 as run  # noqa: E402


def test_mirror_preserves_history_order_and_is_involution():
    obs = np.arange(627,dtype=np.float32)/627
    reflected = mirror_observation(obs)
    assert np.array_equal(mirror_observation(reflected),obs)
    for i in range(11):
        frame,got = obs.reshape(11,57)[i],reflected.reshape(11,57)[i]
        assert got[0]==-frame[0] and got[1]==frame[1] and got[2]==-frame[2]
        assert got[9]==-frame[13] and got[12]==frame[16]
        assert got[41]==-frame[45] and got[44]==frame[48]
    action = np.linspace(-1,1,16,dtype=np.float32)
    assert np.array_equal(mirror_action(mirror_action(action)),action)


def test_mapping_commutes_with_original_observation_encoder():
    spec = importlib.util.spec_from_file_location('interface_sym31',ROOT/'training/d1_rl/interface.py')
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)
    rng = np.random.default_rng(311)
    q = interface.DEFAULT_Q+rng.uniform(-.2,.2,16)
    dq = rng.uniform(-1,1,16)
    last = rng.uniform(-1,1,16).astype(np.float32)
    gyro,previous,command = rng.uniform(-.5,.5,(3,3))
    quat = np.array([.9,.1,.2,.3])
    frame,_ = interface.build_observation(gyro,quat,command,q,dq,last,previous)
    order = [4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11]
    signs = np.tile([-1,1,1,1],4)
    expected,_ = interface.build_observation(gyro*[-1,1,-1],quat*[1,-1,1,-1],command*[1,-1,-1],
        q[order]*signs,dq[order]*signs,mirror_action(last),previous*[-1,1,-1])
    assert np.array_equal(mirror_observation(np.tile(frame,11)),np.tile(expected,11))


def test_actual_wrapper_is_equivariant_and_counts_two_calls_per_tick():
    rng = np.random.default_rng(312)
    weights = rng.normal(0,.01,(16,627)).astype(np.float32)
    bias = rng.normal(0,.1,16).astype(np.float32)
    original = SimpleNamespace(backend={},make_env=lambda *a:None,
                               predict=lambda x:np.tanh(weights@x+bias).astype(np.float32))
    a,b = run.Runtime(original),run.Runtime(original)
    a.make_env('forward_stop',[],0,1001)
    b.make_env('turn_stop',[],3,1003)
    obs = rng.uniform(-1,1,627).astype(np.float32)
    assert np.array_equal(a.predict(mirror_observation(obs)),mirror_action(b.predict(obs)))
    assert a.attempts==a.returns==b.attempts==b.returns==2
    assert len(a.rows)==len(b.rows)==1


def test_projection_does_not_add_a_common_wheel_bias():
    action = np.zeros(16,dtype=np.float32)
    action[[3,7,11,15]] = [.25,-.5,.5,.75]
    result = combine(action,action)
    assert result[3]==result[7] and result[11]==result[15]
    assert result[[3,7,11,15]].mean()==action[[3,7,11,15]].mean()


def test_second_call_failure_preserves_attempt_count():
    calls=[]
    def predict(obs):
        calls.append(obs)
        if len(calls)==2:
            raise RuntimeError('second inference failed')
        return np.zeros(16,np.float32)
    rt=run.Runtime(SimpleNamespace(predict=predict,make_env=lambda *a:None,backend={}))
    rt.make_env('turn_stop',[],3,1001)
    with pytest.raises(RuntimeError):
        rt.predict(np.zeros(627,np.float32))
    assert rt.attempts==2 and rt.returns==1 and not rt.rows


@pytest.mark.parametrize('outcome',['pass','fail','unknown'])
def test_runner_holds_original_gates_and_partial_counts(tmp_path,monkeypatch,outcome):
    plan=tmp_path/'plan.json'
    source={}
    run.atomic_json(plan,{'scope':run.scope(),'inputs':{},'source':source})
    monkeypatch.setattr(run,'PLAN',plan)
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'out')
    monkeypatch.setattr(run,'scope',lambda:run.shared.read_json(plan)['scope'])
    monkeypatch.setattr(run,'source',lambda *a:source)
    monkeypatch.setattr(run.shared,'inputs',lambda:({},{}))
    monkeypatch.setattr(run.os,'getpriority',lambda *a:10)
    clock=iter(range(0,10000,11))
    monkeypatch.setattr(run.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(run.time,'sleep',lambda _:None)
    monkeypatch.setattr(run.shared,'snapshot_memory',lambda _:{'effective_available_bytes':8<<30,'errors':[]})
    stages=[]
    def evaluation(stage,bundle,parent,folder,child,poll,**kwargs):
        stages.append(stage)
        if outcome=='unknown':
            child.update(actual_counts=None,known_actual_counts=run.counts(5),cleanup_complete=False)
            raise RuntimeError('synthetic unknown counter')
        child.update(actual_counts=run.counts(21600),known_actual_counts=run.counts(21600),
            all_36_reference_pass=outcome=='pass',cases=[{'sequence':n,'delay_ticks':d,'reset_seed':s,
                'reference_thresholds_met':outcome=='pass','failure':False,'done_reason':'HORIZON_12S',
                'policy_transitions':600} for n,d,s in run.shared.matrix(stage)])
    monkeypatch.setattr(run.shared,'evaluate',evaluation)
    result=run.execute()
    assert stages==(['formal','holdout'] if outcome=='pass' else ['formal'])
    assert result['python_original_and_holdout_qualified'] is (outcome=='pass')
    if outcome=='unknown':
        assert result['actual_counts'] is None and result['known_actual_counts']==run.counts(5)
    else:
        assert result['actual_counts']==run.counts(43200 if outcome=='pass' else 21600)
