"""Pure fake29 execution: no tensor libraries, model, environment or physics."""
from __future__ import annotations

import importlib.util
import signal
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools29 import anchor29 as a
from tools29 import train_anchor29 as t

GIB=1<<30
VERSIONS=dict.fromkeys(('python','torch','sb3','numpy','gymnasium','mujoco'),'fake')


def prior_fixture():
    spec=importlib.util.spec_from_file_location('anchor29_prior_pure',ROOT/'tests/test_train_delay25.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.t=t
    return module


OLD=prior_fixture()


def zeros(shape,kind='f'):
    count=1
    for size in shape:count*=size
    return memoryview(array(kind,[0])*count).cast('B').cast(kind,shape)


def buffer():
    return SimpleNamespace(observations={'actor':zeros((4096,1,627)),'velocity_target':zeros((4096,1,3))},
        actions=zeros((4096,1,16)),episode_starts=zeros((4096,1)),values=zeros((4096,1)),log_probs=zeros((4096,1)))


def initial_state(plan):
    return {'num_timesteps':t.INITIAL,'updates':1088,'optimizer_restored':True,
        'policy_state_sha256':'a'*64,'policy_state_sha256_after_checks':'a'*64,
        'optimizer_state_sha256':'b'*64,'optimizer_state_sha256_after_checks':'b'*64,
        'checkpoint_sha256':plan['base_checkpoint']['sha256'],
        'serialized_entries_sha256':dict(plan['baseline_provenance']['serialized_entries_sha256']),
        'seed_before':181,'seed_after':183,'seed_changes_policy_or_optimizer':False,
        'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[1e-4,1e-4],
        'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,
        'rollout_layout':{'backend':'DummyVecEnv','envs':1,'n_steps':4096,'rollout_transitions':4096,
            'buffer_class':'DictRolloutBuffer','actor_shape':[4096,1,627],'velocity_target_shape':[4096,1,3],
            'actions_shape':[4096,1,16],'last_obs_cleared':True,'interprocess_rpc':False,'environment_subprocesses':0}}


def anchor_receipt(coef,rollouts=8):
    records=[]
    if coef==1.:
        for index in range(rollouts):
            records.append({'generation':a.INITIAL+4096*(index+1),'rows':4096,
                'actor_input':a.digest_array29(zeros((4096,627))),
                'teacher_mean':a.digest_array29(zeros((4096,16))),
                'teacher_log_std':a.digest_array29(zeros((16,))),
                'epochs':[{'epoch':0,'permutation':a.digest_array29(memoryview(array('q',range(4096)))),
                           'minibatches_returned':16,'sample_rows_returned':4096}]})
    teacher=a.zero_teacher_counts29() if coef==0. else dict(batch_attempts=rollouts,batch_returns=rollouts,
        rows_attempted=4096*rollouts,rows_returned=4096*rollouts,linear_attempts=6*rollouts,linear_returns=6*rollouts)
    return {'base_actor_sha256':'c'*64,'frozen_actor_sha256_after_training':'c'*64,
        'teacher':teacher,'cache_rollouts':records,'train_calls':rollouts,'train_returns':rollouts,
        'optimizer_updates':rollouts*16,'anchor_loss_statistics':{'batches':0 if coef==0. else rollouts*16,
            'mean':None if coef==0. else .01,'min':None if coef==0. else .0,'max':None if coef==0. else .02}}


class Runtime(OLD.Runtime):
    def create_vec(self,plan,transitions,output,root,mode):
        assert transitions==32768 and mode=='uniform_0_3' and plan['seed']==183
        self.mode=mode;self.spawn_calls+=1
        if self.fault=='spawn':raise PermissionError('synthetic construction failure')
        self.vec=SimpleNamespace(steps=0,envs=1,waiting=False,dirty=False,closed=False,processes=[])
        if self.fault=='partial_constructor':raise ValueError('owned partial construction')
        if self.fault=='spawn_signal':signal.raise_signal(signal.SIGTERM)
        return self.vec
    def create_model(self,vec,plan,hyper,output):
        self.initial_identity=initial_state(plan)
        if self.fault=='momentum':self.initial_identity['optimizer_state_sha256_after_checks']='d'*64
        owner=self
        class Model:
            num_timesteps=t.INITIAL
            target_kl=.005
            anchor_coef=t.common.ARMS[owner.arm]
            rollout_buffer=buffer()
            def anchor_receipt29(self):
                value=anchor_receipt(self.anchor_coef,vec.steps//4096)
                if owner.fault=='teacher_mutated':value['frozen_actor_sha256_after_training']='d'*64
                if owner.fault=='teacher_missing':value['teacher']['rows_returned']=0
                return value
            def save(self,path):
                if owner.fault=='save':raise OSError('synthetic save failed')
                Path(path).write_bytes(f'fake:{self.num_timesteps}'.encode())
                if owner.fault=='save_late':owner.clock.now=601.
        self.model=Model()
        if self.fault=='target_lie':self.model.target_kl=None
        if self.fault=='coef_lie':self.model.anchor_coef=999.
        return self.model
    def reward_state(self,vec):
        if vec.waiting or vec.dirty:raise RuntimeError('metadata refused')
        value={'yaw_tracking_weight':2.,'steps':vec.steps,'original_reward_sum':.1*vec.steps,
            'training_reward_sum':.1*vec.steps,'extra_reward_sum':0.,'raw_wz_tracking_sum':.1*vec.steps}
        if self.fault=='reward_lie':value['extra_reward_sum']=1.
        return value


@pytest.fixture
def case(tmp_path):
    (tmp_path/'base.zip').write_bytes(b'complete random25 plus Adam')
    t.atomic_json(tmp_path/'base_protocol.json',{'hyperparameters':t.HYPERPARAMETERS,'versions':VERSIONS})
    plan={'config':'pilot03d.json','envs':1,'seed':183,'output_root':'evidence/run',
        'stages':{key:dict(t.common.STAGES[key]) for key in ('train_control','train_anchor')},
        'base_checkpoint':{'path':str(tmp_path/'base.zip'),'sha256':t.sha(tmp_path/'base.zip'),'trained_transitions':t.INITIAL},
        'base_protocol':{'path':str(tmp_path/'base_protocol.json'),'sha256':t.sha(tmp_path/'base_protocol.json')},
        'baseline_provenance':{'serialized_entries_sha256':dict(t.common.BASE_SERIALIZED)},
        'hyperparameters':t.HYPERPARAMETERS,'source_identity':{'git_commit':'fake','sources_sha256':{}},
        'vector_layout':t.common.VECTOR_LAYOUT,'runtime_dependencies_sha256':{'fake.py':'a'*64},
        'resources':{'start_min_effective_bytes':4*GIB,'reserve_bytes':2*GIB,'poll_s':1.}}
    t.atomic_json(tmp_path/'plan.json',plan)
    return tmp_path,tmp_path/'plan.json',plan


def run_fake(case,*,fault=None,memory=None,arm='control'):
    root,path,plan=case;clock=OLD.Clock();runtime=Runtime(clock,fault)
    def close(vec):
        vec.closed=fault!='close_lie'
        return (False,'synthetic close failed') if fault=='close' else (True,None)
    result=t.run(path,arm,root/plan['output_root']/arm/'train',root=root,runtime=runtime,
        loader=lambda p:t.common.read_json(p),launch_validator=lambda *args:None,
        memory_reader=memory or (lambda:{'available_ram_bytes':5*GIB,'effective_available_bytes':4*GIB}),
        clock=clock,closer=close)
    return result,runtime,root/plan['output_root']/arm/'train'


@pytest.mark.parametrize('arm',['control','anchor'])
def test_complete_eight_rollouts_same_reward_and_explicit_teacher_counts(case,arm):
    result,runtime,output=run_fake(case,arm=arm)
    assert result['status']=='COMPLETE' and result['error'] is None and result['cleanup_complete']
    assert result['new_trained_transitions']==32768 and result['final_num_timesteps']==1343488
    assert result['actual_counts']['native_steps']==327680 and result['yaw_tracking_weight']==2.
    assert result['anchor_coef']==t.common.ARMS[arm] and result['teacher_policy_used']==(arm=='anchor')
    assert result['anchor_change']==t.common.anchor_change(arm)
    assert a.valid_anchor_receipt29(result['anchor29'],t.common.ARMS[arm],True)
    assert result['anchor29']['teacher']['rows_returned']==(32768 if arm=='anchor' else 0)
    assert t.common.valid_first_rollout29(result['first_rollout_nonreward'])
    assert result['first_rollout_report_sha256']==t.sha(output/'first_rollout29.json')
    assert runtime.vec.closed and runtime.spawn_calls==runtime.learn_calls==1
    assert len((output/'ledger.jsonl').read_text().splitlines())==8


@pytest.mark.parametrize('fault',['import','versions','slow_import','spawn','spawn_signal','momentum','target_lie',
    'partial','late','plan_changed','save','save_late','close','close_lie','partial_constructor','dirty_after_close',
    'reward_lie','counter_array','counter_extra','delay_array','waiting_signal','teacher_mutated','teacher_missing','coef_lie'])
def test_fail_closed_unknown_counters_and_owned_cleanup(case,fault):
    signals={x:signal.getsignal(x) for x in (signal.SIGINT,signal.SIGTERM)}
    result,_,output=run_fake(case,fault=fault,arm='anchor')
    assert result['status']!='COMPLETE' and not result['eligible_for_evaluation']
    assert not (output/'policy.zip').exists() and t.common.read_json(output/'result.json')==result
    assert {x:signal.getsignal(x) for x in signals}==signals
    if fault in ('counter_array','counter_extra','waiting_signal'):
        assert result['actual_counts'] is None and result['known_actual_counts']['policy_transitions']==4096


def test_four_gib_cold_two_gib_running_and_no_retry(case):
    with pytest.raises(ValueError,match='NOT_READY'):
        run_fake(case,memory=lambda:{'effective_available_bytes':(4*GIB)-1})
    calls=[0]
    def memory():
        calls[0]+=1
        return {'available_ram_bytes':5*GIB,'effective_available_bytes':4*GIB if calls[0]==1 else (2*GIB)-1}
    result,runtime,_=run_fake(case,memory=memory)
    assert result['status']!='COMPLETE' and runtime.spawn_calls==0
    with pytest.raises(ValueError,match='exclusive'):run_fake(case)


def test_fair_initial_and_first_nonreward_arrays(case):
    control,_,_=run_fake(case,arm='control');anchor,_,_=run_fake(case,arm='anchor')
    assert control['initial_state']==anchor['initial_state']
    assert control['first_rollout_nonreward']==anchor['first_rollout_nonreward']
    assert control['reward_state29']==anchor['reward_state29']
    assert control['reward_state29']['extra_reward_sum']==0.


def test_first_rollout_hash_excludes_reward_and_detects_action():
    value=buffer();before=t.first_rollout_proof29(value)
    value.rewards=array('f',[99.]);value.advantages=array('f',[88.])
    assert before==t.first_rollout_proof29(value)
    value.actions[0,0,0]=1.
    assert before!=t.first_rollout_proof29(value)
