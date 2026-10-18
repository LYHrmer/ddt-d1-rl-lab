import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools32.command_gain32 import compensate, YAW_COLUMNS  # noqa: E402
from tools32.run_command_gain32 import Runtime  # noqa: E402
from tools32 import run_command_gain32 as run  # noqa: E402


def test_only_yaw_command_in_all_frames_changes_without_mutating_input():
    obs=np.linspace(-.8,.8,627,dtype=np.float32)
    obs[YAW_COLUMNS]=np.linspace(-.1,.1,11,dtype=np.float32)
    before=obs.copy()
    result=compensate(obs)
    mask=np.ones(627,bool);mask[YAW_COLUMNS]=False
    assert np.array_equal(result[mask],obs[mask]) and np.array_equal(obs,before)
    assert np.array_equal(result[YAW_COLUMNS],obs[YAW_COLUMNS]*np.float32(1.5))


def test_original_encoded_command_bounds_and_zero_commands_preserved():
    obs=np.zeros(627,np.float32)
    assert np.array_equal(compensate(obs),obs)
    obs[YAW_COLUMNS]=.24
    assert np.array_equal(compensate(obs)[YAW_COLUMNS],np.full(11,.25,np.float32))
    obs[YAW_COLUMNS]=-.24
    assert np.array_equal(compensate(obs)[YAW_COLUMNS],np.full(11,-.25,np.float32))


def test_runtime_calls_original_actor_once_and_records_compensation():
    seen=[]
    def predict(obs):
        seen.append(obs.copy())
        return np.full(16,.3,np.float32)
    rt=Runtime(SimpleNamespace(backend={},make_env=lambda *a:None,predict=predict))
    rt.make_env('turn_stop',[],3,1001)
    obs=np.zeros(627,np.float32);obs[YAW_COLUMNS]=-.05
    actual=rt.predict(obs)
    assert len(seen)==rt.attempts==rt.returns==1
    assert np.array_equal(actual,np.full(16,.3,np.float32))
    assert np.array_equal(seen[0],compensate(obs))
    assert np.array_equal(rt.rows[0][0],obs[YAW_COLUMNS])
    assert np.array_equal(rt.rows[0][1],seen[0][YAW_COLUMNS])


@pytest.mark.parametrize('value',[np.zeros(626,np.float32),np.zeros(627,np.float64),np.full(627,np.nan,np.float32)])
def test_reject_bad_observations_before_inference(value):
    with pytest.raises(ValueError):compensate(value)


@pytest.mark.parametrize('complete',[True,False])
def test_fixed_runner_gates_holdout_and_refuses_output_reuse(tmp_path,monkeypatch,complete):
    plan=tmp_path/'plan.json'
    original_scope=run.scope()
    run.atomic_json(plan,{'scope':original_scope,'inputs':{},'source':{}})
    monkeypatch.setattr(run,'PLAN',plan)
    monkeypatch.setattr(run,'OUTPUT',tmp_path/'out')
    monkeypatch.setattr(run,'scope',lambda:original_scope)
    monkeypatch.setattr(run,'source',lambda *a:{})
    monkeypatch.setattr(run.shared,'inputs',lambda:({},{}))
    monkeypatch.setattr(run.os,'getpriority',lambda *a:10)
    clock=iter(range(0,10000,11))
    monkeypatch.setattr(run.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(run.time,'sleep',lambda _:None)
    monkeypatch.setattr(run.shared,'snapshot_memory',lambda _:{'effective_available_bytes':8<<30,'errors':[]})
    stages=[]
    def evaluation(stage,bundle,parent,folder,child,poll,**kwargs):
        stages.append(stage)
        child.update(actual_counts=run.counts(21600),known_actual_counts=run.counts(21600),
            all_36_reference_pass=complete,cases=[{'sequence':n,'delay_ticks':d,'reset_seed':s,
                'reference_thresholds_met':complete,'failure':False,'done_reason':'HORIZON_12S',
                'policy_transitions':600} for n,d,s in run.shared.matrix(stage)])
    monkeypatch.setattr(run.shared,'evaluate',evaluation)
    result=run.execute()
    assert stages==(['formal','holdout'] if complete else ['formal'])
    assert result['python_original_and_holdout_qualified'] is complete
    assert result['actual_counts']==run.counts(43200 if complete else 21600)
    with pytest.raises(FileExistsError):run.execute()
