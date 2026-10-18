"""Bounded26 fake trainers and first-buffer hashes, without NN or physics."""
from __future__ import annotations

import copy
import importlib.util
import json
import signal
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools26 import train_yaw_reward26 as t

GIB=1<<30
VERSIONS=dict.fromkeys(('python','torch','sb3','numpy','gymnasium','mujoco'),'fake')


def buffer():
    def zeros(shape):
        count=1
        for n in shape:count*=n
        return memoryview(array('f',[0.])*count).cast('B').cast('f',shape)
    return SimpleNamespace(observations={'actor':zeros((4096,1,627)),'velocity_target':zeros((4096,1,3))},
        actions=zeros((4096,1,16)),episode_starts=zeros((4096,1)),values=zeros((4096,1)),log_probs=zeros((4096,1)))


def initial_state(plan):
    return {'num_timesteps':t.INITIAL,'updates':1088,'optimizer_restored':True,
        'policy_state_sha256':'a'*64,'policy_state_sha256_after_checks':'a'*64,
        'optimizer_state_sha256':'b'*64,'optimizer_state_sha256_after_checks':'b'*64,
        'checkpoint_sha256':plan['base_checkpoint']['sha256'],
        'serialized_entries_sha256':dict(plan['baseline_provenance']['serialized_entries_sha256']),
        'seed_before':181,'seed_after':182,'seed_changes_policy_or_optimizer':False,
        'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[1e-4,1e-4],
        'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,
        'rollout_layout':{'backend':'DummyVecEnv','envs':1,'n_steps':4096,'rollout_transitions':4096,
            'buffer_class':'DictRolloutBuffer','actor_shape':[4096,1,627],'velocity_target_shape':[4096,1,3],
            'actions_shape':[4096,1,16],'last_obs_cleared':True,'interprocess_rpc':False,'environment_subprocesses':0}}


def old_fixture():
    spec=importlib.util.spec_from_file_location('train26_reused_pure_fake',ROOT/'tests/test_train_delay25.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.t=t
    return module


OLD=old_fixture()


class Runtime(OLD.Runtime):
    def create_vec(self,plan,transitions,output,root,mode):
        assert transitions==131072 and mode=='uniform_0_3' and plan['seed']==182
        self.mode=mode;self.spawn_calls+=1
        if self.fault=='spawn':raise PermissionError('synthetic owned constructor unavailable')
        self.vec=SimpleNamespace(steps=0,envs=1,waiting=False,dirty=False,closed=False,processes=[])
        if self.fault=='partial_constructor':raise ValueError('partial construction')
        if self.fault=='spawn_signal':signal.raise_signal(signal.SIGTERM)
        return self.vec
    def create_model(self,vec,plan,hyper,output):
        self.initial_identity=initial_state(plan)
        if self.fault=='momentum':self.initial_identity['optimizer_state_sha256_after_checks']='c'*64
        owner=self
        class Model:
            num_timesteps=t.INITIAL
            target_kl=.005
            rollout_buffer=buffer()
            def save(self,path):
                if owner.fault=='save':raise OSError('synthetic save failed')
                Path(path).write_bytes(f'fake:{self.num_timesteps}'.encode())
                if owner.fault=='save_late':owner.clock.now=1201.
        self.model=Model()
        if self.fault=='target_lie':self.model.target_kl=None
        return self.model
    def reward_state(self,vec):
        if vec.waiting or vec.dirty:raise RuntimeError('unsafe reward metadata refused')
        raw=.1*vec.steps;extra=(t.common.ARMS[self.arm]-2.)*raw*.02
        value={'yaw_tracking_weight':t.common.ARMS[self.arm],'steps':vec.steps,
            'original_reward_sum':.1*vec.steps,'training_reward_sum':.1*vec.steps+extra,
            'extra_reward_sum':extra,'raw_wz_tracking_sum':raw}
        if self.fault=='reward_lie':value['extra_reward_sum']+=1.
        return value


@pytest.fixture
def case(tmp_path):
    (tmp_path/'base.zip').write_bytes(b'completed25 random policy and full Adam')
    t.atomic_json(tmp_path/'base_protocol.json',{'hyperparameters':t.HYPERPARAMETERS,'versions':VERSIONS})
    plan={'config':'pilot03d.json','envs':1,'seed':182,'output_root':'evidence/run',
        'stages':{key:dict(t.common.STAGES[key]) for key in ('train_control','train_yaw3')},
        'base_checkpoint':{'path':str(tmp_path/'base.zip'),'sha256':t.sha(tmp_path/'base.zip'),'trained_transitions':t.INITIAL},
        'base_protocol':{'path':str(tmp_path/'base_protocol.json'),'sha256':t.sha(tmp_path/'base_protocol.json')},
        'baseline_provenance':{'serialized_entries_sha256':dict(t.common.BASE_SERIALIZED)},
        'hyperparameters':t.HYPERPARAMETERS,'source_identity':{'git_commit':'fake','sources_sha256':{}},
        'vector_layout':t.common.VECTOR_LAYOUT,'runtime_dependencies_sha256':{'fake.py':'a'*64},
        'resources':{'start_min_effective_bytes':6*GIB,'reserve_bytes':4*GIB,'poll_s':1.}}
    t.atomic_json(tmp_path/'plan.json',plan)
    return tmp_path,tmp_path/'plan.json',plan


def run_fake(case,*,fault=None,memory=None,arm='control'):
    root,path,plan=case;clock=OLD.Clock();runtime=Runtime(clock,fault)
    def close(vec):
        vec.closed=fault!='close_lie'
        return (False,'synthetic close failed') if fault=='close' else (True,None)
    result=t.run(path,arm,root/plan['output_root']/arm/'train',root=root,runtime=runtime,
        loader=lambda p:t.common.read_json(p),launch_validator=lambda *a:None,
        memory_reader=memory or (lambda:{'available_ram_bytes':8*GIB,'effective_available_bytes':7*GIB,'cgroup_remaining_bytes':None}),
        clock=clock,closer=close)
    return result,runtime,root/plan['output_root']/arm/'train'


@pytest.mark.parametrize('arm',['control','yaw3'])
def test_exact32_rollouts_new_offset_same_mode_and_actual_reward_gate(case,arm):
    result,runtime,output=run_fake(case,arm=arm)
    assert result['status']=='COMPLETE' and result['error'] is None and result['cleanup_complete']
    assert result['initial_num_timesteps']==1310720 and result['new_trained_transitions']==131072
    assert result['final_num_timesteps']==1441792 and result['actual_counts']['native_steps']==1310720
    assert result['mode']=='uniform_0_3' and result['yaw_tracking_weight']==t.common.ARMS[arm]
    assert result['reward_change']==t.common.reward_change(arm)
    assert result['initial_state']==initial_state(case[2])
    assert result['eligible_for_evaluation'] and not (output/'partial_policy.zip').exists()
    rows=[json.loads(x) for x in (output/'ledger.jsonl').read_text().splitlines()]
    assert len(rows)==32 and rows[-1]['transitions']==131072
    assert t.common.valid_first_rollout26(result['first_rollout_nonreward'])
    assert result['first_rollout_report_sha256']==t.sha(output/'first_rollout26.json')
    assert runtime.vec.closed and runtime.spawn_calls==runtime.learn_calls==1


@pytest.mark.parametrize('fault',['import','versions','slow_import','spawn','spawn_signal','momentum','target_lie',
    'partial','late','plan_changed','save','save_late','close','close_lie','partial_constructor','dirty_after_close','reward_lie',
    'counter_array','counter_extra','delay_array','waiting_signal'])
def test_failures_keep_partial_ineligible_known_counts_and_owned_cleanup(case,fault):
    prior={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    result,runtime,output=run_fake(case,fault=fault)
    assert result['status']!='COMPLETE' and not result['eligible_for_evaluation']
    assert t.common.read_json(output/'result.json')==result and not (output/'policy.zip').exists()
    assert prior=={s:signal.getsignal(s) for s in prior}
    if fault in ('import','versions','slow_import'):assert runtime.spawn_calls==0
    if fault=='spawn':assert result['actual_counts'] is None and not result['cleanup_complete']
    if fault in ('counter_array','counter_extra','waiting_signal'):
        assert result['actual_counts'] is None and result['known_actual_counts']['policy_transitions']==4096
    if fault=='reward_lie':assert result['reward_state26'] is None
    if fault=='late':assert result['actual_counts']['policy_transitions']==131072


def test_initial_admission_and_nonce_are_cold_and_output_exclusive(case):
    root,path,plan=case;runtime=Runtime(OLD.Clock())
    with pytest.raises(ValueError,match='NOT_READY'):
        t.run(path,'control',root/plan['output_root']/'control/train',root=root,runtime=runtime,
            loader=lambda p:plan,launch_validator=lambda *a:None,memory_reader=lambda:{'effective_available_bytes':5*GIB})
    assert runtime.spawn_calls==0 and not (root/plan['output_root']/'control/train').exists()
    def reject(*args):raise ValueError('missing nonce')
    with pytest.raises(ValueError,match='nonce'):
        t.run(path,'control',root/plan['output_root']/'control/train',root=root,runtime=runtime,
            loader=lambda p:plan,launch_validator=reject)
    run_fake(case)
    with pytest.raises(ValueError,match='exclusive'):run_fake(case)


def test_runtime4gib_stop_is_still_enforced(case):
    calls=[0]
    def memory():
        calls[0]+=1;v=7*GIB if calls[0]<=3 else (4*GIB)-1
        return {'available_ram_bytes':8*GIB,'effective_available_bytes':v}
    result,runtime,output=run_fake(case,memory=memory)
    assert result['status']=='RAM_RESERVE_BELOW_4_GIB' and result['stop_sample']['effective_available_bytes']==(4*GIB)-1
    assert runtime.vec.closed and (output/'stop_sample.json').is_file()


def test_first_rollout_hash_excludes_reward_but_detects_nonreward_change():
    x=buffer();x.rewards=array('f',[.1]);x.returns=array('f',[.2]);x.advantages=array('f',[.3])
    first=t.first_rollout_proof26(x)
    x.rewards[0]=99.;x.returns[0]=88.;x.advantages[0]=77.
    assert first==t.first_rollout_proof26(x)
    x.actions[0,0,0]=1.
    assert first!=t.first_rollout_proof26(x)
    assert t.common.valid_first_rollout26(first)
    bad=copy.deepcopy(first);bad['arrays']['values']['shape']=[True,4096]
    assert not t.common.valid_first_rollout26(bad)


def test_both_arms_have_same_initial_and_first_nonreward_rollout(case):
    control,_,_=run_fake(case,arm='control');yaw,_,_=run_fake(case,arm='yaw3')
    assert control['initial_state']==yaw['initial_state']
    assert control['first_rollout_nonreward']==yaw['first_rollout_nonreward']
    assert control['reward_state26']['extra_reward_sum']==0.
    assert yaw['reward_state26']['extra_reward_sum']>0.
