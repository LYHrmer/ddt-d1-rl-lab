import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools34 import cpp_reference34 as cpp  # noqa: E402
from tools34 import run_cpp_reference34 as run  # noqa: E402
from tools33.reference_shape33 import ReferenceShaper, YAW_COLUMNS  # noqa: E402


def test_cpp_receipt_extension_is_required_then_passes_strict_original_wire(monkeypatch):
    response={'format':'d1-cpp-actor19-response-v1','status':'OK','seq':0,'actions':[0.]*16,
        'inference_attempts':1,'inference_returns':1,'effective_yaw_commands':[0.]*11,'effective_input_sha256':'a'*64}
    monkeypatch.setattr(cpp.transport.CppActor19,'_receive',lambda *_:response.copy())
    actor=object.__new__(cpp.CppReferenceActor)
    parsed=actor._receive(1)
    assert set(parsed)=={'format','status','seq','actions','inference_attempts','inference_returns'}
    assert actor.effective_sha=='a'*64 and np.array_equal(actor.effective_yaw,np.zeros(11))
    response.pop('effective_input_sha256')
    with pytest.raises(ValueError,match='evidence missing'):actor._receive(1)


@pytest.mark.parametrize('bad_hash',[False,True])
def test_raw_input_goes_to_cpp_and_only_cpp_action_drives_physics(tmp_path,monkeypatch,bad_hash):
    seen=[]
    class FakeCpp:
        def __init__(self,*args,**kwargs):self.shaper=ReferenceShaper()
        def start(self):return self
        def predict(self,obs):
            seen.append(obs.copy())
            effective=self.shaper(obs)
            self.effective_yaw=effective[YAW_COLUMNS]
            self.effective_sha='invalid' if bad_hash else hashlib.sha256(effective.tobytes()).hexdigest()
            return np.full(16,.100001,np.float32)
    monkeypatch.setattr(cpp,'CppReferenceActor',FakeCpp)
    monkeypatch.setattr(cpp,'sha',lambda _: 'a'*64)
    original=SimpleNamespace(backend={},make_env=lambda *a:None,predict=lambda _:np.full(16,.1,np.float32))
    rt=cpp.Runtime(original,tmp_path,{'binary':'stub','binary_sha256':'b'*64})
    rt.make_env('turn_stop',[],3,1001)
    obs=np.zeros(627,np.float32)
    if bad_hash:
        with pytest.raises(ValueError,match='full effective input'):rt.predict(obs)
    else:
        action=rt.predict(obs)
        assert np.array_equal(action,np.full(16,.100001,np.float32))
        obs[8]=.05;rt.predict(obs)
        assert np.array_equal(seen[-1],obs)
        assert seen[-1][8]!=rt.rows[-1][1][0]
        assert rt.cpp_attempts==rt.cpp_returns==rt.attempts==rt.returns==2


@pytest.mark.parametrize('compile_ok,passes',[(False,False),(True,False),(True,True)])
def test_compile_failure_stops_and_formal_gates_holdout(tmp_path,monkeypatch,compile_ok,passes):
    plan=tmp_path/'plan.json';original_scope=run.scope()
    run.atomic_json(plan,{'scope':original_scope,'inputs':{},'source':{}})
    monkeypatch.setattr(run,'PLAN',plan);monkeypatch.setattr(run,'OUTPUT',tmp_path/'out')
    monkeypatch.setattr(run,'scope',lambda:original_scope);monkeypatch.setattr(run,'source',lambda *a:{})
    monkeypatch.setattr(run,'inputs',lambda:({},{}));monkeypatch.setattr(run.os,'getpriority',lambda *a:10)
    clock=iter(range(0,10000,11));monkeypatch.setattr(run.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(run.time,'sleep',lambda _:None)
    monkeypatch.setattr(run.shared,'snapshot_memory',lambda _:{'effective_available_bytes':8<<30,'errors':[]})
    def build(_):
        if not compile_ok:raise ValueError('synthetic compile failure')
        return {}
    monkeypatch.setattr(run,'build_actor',build)
    from tools17 import evaluate_formal17
    monkeypatch.setattr(evaluate_formal17,'runtime',lambda *args:None)
    class FakeRuntime:
        def __init__(self,*args):
            self.cpp_attempts=self.cpp_returns=self.attempts=self.returns=21600
            self.cpp_cleanup=True
        def close_actor(self):return None
    monkeypatch.setattr(run,'Runtime',FakeRuntime)
    stages=[]
    def evaluate(stage,bundle,parent,folder,child,poll,**kwargs):
        stages.append(stage)
        kwargs['runtime_factory']()
        child.update(actual_counts=run.counts(21600),known_actual_counts=run.counts(21600),all_36_reference_pass=passes,
            cases=[{'sequence':n,'delay_ticks':d,'reset_seed':s,'reference_thresholds_met':passes,'failure':False,
                    'done_reason':'HORIZON_12S','policy_transitions':600} for n,d,s in run.shared.matrix(stage)])
    monkeypatch.setattr(run.shared,'evaluate',evaluate)
    result=run.execute()
    assert stages==([] if not compile_ok else ['formal','holdout'] if passes else ['formal'])
    assert result['new_cpp_closedloop_proven'] is (compile_ok and passes)
    assert result['status']==('COMPLETE' if compile_ok else 'INCOMPLETE')
    with pytest.raises(FileExistsError):run.execute()
