"""Synthetic26 reward and warm-loader tests; no NN, environment or physics."""
from __future__ import annotations

import ast
import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools26 import runtime26 as r


def terms():
    raw={'vx_tracking':.4,'wz_tracking':.25,'torque_sum_squared':5.,'standstill_tracking':.2}
    weights={'vx_tracking':2.,'wz_tracking':2.,'torque_sum_squared':-1e-5,'standstill_tracking':1.}
    second={k:raw[k]*weights[k] for k in raw}
    return {'raw':raw,'weights':weights,'contributions_per_second':second,
            'contributions_per_tick':{k:v*.02 for k,v in second.items()},'policy_dt_s':.02}


def test_reward_is_actually_changed_only_at_wz_and_control_is_exact_original():
    info={'reward_terms':terms(),'other':object()};before=copy.deepcopy(info['reward_terms'])
    reward=float(sum(before['contributions_per_second'].values())*.02)
    control,ci=r.adjusted_reward26(reward,info,2.)
    assert control is reward and ci['reward_terms'] is info['reward_terms']
    treatment,ti=r.adjusted_reward26(reward,info,3.)
    assert treatment==float((sum(before['contributions_per_second'].values())+.25)*.02)
    assert ti['reward_terms']['raw']==before['raw']
    for group in ('weights','contributions_per_second','contributions_per_tick'):
        assert all(ti['reward_terms'][group][k]==v for k,v in before[group].items() if k!='wz_tracking')
    assert ti['reward_terms']['weights']['wz_tracking']==3.
    assert ti['reward_terms']['contributions_per_second']['wz_tracking']==.75
    assert ti['reward_terms']['contributions_per_tick']['wz_tracking']==.015
    assert info['reward_terms']==before and ti['other'] is info['other']
    assert ti['reward_change26']['extra_reward']==pytest.approx(.005)


@pytest.mark.parametrize('fault',['weight','dt','rawnan','tick','second','total','typedweight'])
def test_untrusted_original_terms_are_rejected(fault):
    info={'reward_terms':terms()};reward=sum(info['reward_terms']['contributions_per_second'].values())*.02
    if fault=='weight':info['reward_terms']['weights']['wz_tracking']=3.
    elif fault=='dt':info['reward_terms']['policy_dt_s']=.04
    elif fault=='rawnan':info['reward_terms']['raw']['wz_tracking']=float('nan')
    elif fault=='tick':info['reward_terms']['contributions_per_tick']['vx_tracking']*=2
    elif fault=='second':info['reward_terms']['contributions_per_second']['vx_tracking']*=2
    elif fault=='typedweight':info['reward_terms']['weights']['wz_tracking']=True
    else:reward+=.1
    with pytest.raises(ValueError):r.adjusted_reward26(reward,info,3.)


def test_wrapper_preserves_step_identity_and_accumulates_actual_reward_before_monitor():
    observation,action=object(),object()
    t=terms();reward=sum(t['contributions_per_second'].values())*.02
    class Base:
        def __init__(self,env):self.env=env
    class Env:
        def __init__(self):self.calls=[]
        def step(self,seen):
            self.calls.append(seen)
            return observation,reward,False,False,{'reward_terms':copy.deepcopy(t),'x':17}
    for arm,weight in r.c.ARMS.items():
        env=Env();wrapper=r.yaw_reward_wrapper26(Base)(env,weight)
        o,new,done,trunc,info=wrapper.step(action)
        assert o is observation and env.calls[-1] is action and done is trunc is False and info['x']==17
        assert new==info['reward_change26']['training_reward']
        assert r.valid_reward_state26(wrapper.get_reward_state26(),arm,1)
    source=(ROOT/'tools26/runtime26.py').read_text()
    assert 'CountedMonitor(self.reward_env26' in source
    assert 'PhaseWrapper(VelocityLabelWrapper(CountedMonitor' in source


def load_old_test():
    spec=importlib.util.spec_from_file_location('runtime26_pure_buffer_fixture',ROOT/'tests/test_low_memory_runtime25.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def warm(monkeypatch,tmp_path):
    fixture=load_old_test();mode=[None];calls=[]
    class Optimizer(fixture.Optimizer):
        def __init__(self):
            super().__init__();self.param_groups[0]['lr']=1e-4;self.momentum['step']=1088
    class Algorithm:
        def __init__(self,vec):
            self.__dict__.update({k:v for k,v in r.c.HYPERPARAMETERS.items() if k not in ('device','cpu_threads','log_std_init','clip_range')})
            self.n_envs,self.seed,self.device,self.num_timesteps,self._n_updates=1,182,'cpu',1310720,1088
            self.lr_schedule,self.clip_range=lambda p:1e-4,lambda p:.2
            self.normalize_advantage,self.clip_range_vf,self.use_sde=True,None,False
            self.policy=SimpleNamespace(optimizer=Optimizer());self.policy_marker='a'*64
            self.vec,self._last_obs,self.rollout_buffer=vec,None,fixture.Buffer()
        @classmethod
        def load(cls,path,**kwargs):
            calls.append((path,kwargs));model=cls(kwargs['env'])
            if mode[0]=='seed':model.seed=181
            elif mode[0]=='envs':model.n_envs=4
            elif mode[0]=='n_steps':model.n_steps=1024
            elif mode[0]=='lr':model.learning_rate=3e-4
            elif mode[0]=='kl':model.target_kl=None
            elif mode[0]=='steps':model.num_timesteps=1048576
            elif mode[0]=='updates':model._n_updates=1024
            elif mode[0]=='buffer':model.rollout_buffer.observations['actor'].shape=(1024,4,627)
            elif mode[0]=='last_obs':model._last_obs='foreign'
            elif mode[0]=='empty_adam':model.policy.optimizer.state_dict=lambda:{'state':{},'param_groups':model.policy.optimizer.param_groups}
            return model
        def get_env(self):return self.vec
        def set_logger(self,logger):self.logger=logger
        def _setup_lr_schedule(self):pytest.fail('26 must not reapply25 LR change')
    monkeypatch.setattr(r,'loading_libraries25',lambda:(Algorithm,fixture.Buffer,lambda *a:None))
    runtime=r.Runtime26();runtime.policy_sha=lambda model:model.policy_marker
    vec=SimpleNamespace(num_envs=1);runtime.vec=vec
    plan={'envs':1,'seed':182,'base_checkpoint':{'path':'completed_random25.zip','sha256':'c'*64,'trained_transitions':1310720},
          'baseline_provenance':{'serialized_entries_sha256':dict(r.c.BASE_SERIALIZED)}}
    return SimpleNamespace(mode=mode,calls=calls,runtime=runtime,vec=vec,plan=plan,output=tmp_path)


@pytest.mark.parametrize('fault',[None,'seed','envs','n_steps','lr','kl','steps','updates','buffer','last_obs','empty_adam'])
def test_completed_random25_warm_start_uses_one_correct_loader_without_lr_kl_changes(warm,fault):
    warm.mode[0]=fault
    if fault:
        with pytest.raises(ValueError):warm.runtime.create_model(warm.vec,warm.plan,r.c.HYPERPARAMETERS,warm.output)
    else:
        model=warm.runtime.create_model(warm.vec,warm.plan,r.c.HYPERPARAMETERS,warm.output)
        proof=warm.runtime.initial_identity
        assert r.initial_identity_valid26(proof,warm.plan)
        assert model.policy.optimizer.momentum['step']==1088
        assert proof['optimizer_state_sha256']==proof['optimizer_state_sha256_after_checks']
        assert proof['policy_state_sha256']==proof['policy_state_sha256_after_checks']
        assert proof['learning_rate_before']==proof['learning_rate_after']==1e-4
        assert proof['target_kl_before']==proof['target_kl_after']==.005
    assert warm.calls==[('completed_random25.zip',{'env':warm.vec,'device':'cpu','seed':182,'force_reset':True,'n_steps':4096})]


@pytest.mark.parametrize('field',['checkpoint_sha256','serialized_entries_sha256','policy_state_sha256_after_checks',
    'optimizer_state_sha256_after_checks','seed_after','learning_rate_before','target_kl_before','updates','rollout_layout'])
def test_initial_proof_cannot_claim_a_different_base_or_training_start(warm,field):
    warm.runtime.create_model(warm.vec,warm.plan,r.c.HYPERPARAMETERS,warm.output)
    initial=copy.deepcopy(warm.runtime.initial_identity);initial[field]=None
    assert not r.initial_identity_valid26(initial,warm.plan)


def test_counter_cap_is_new131072_not_old262144():
    row={'policy_transitions':131072,'native_steps':1310720,'pd_updates':655360,'resets':1,'forward_calls':131074}
    assert r.checked_worker_counts26([row])==row
    row.update(policy_transitions=131073,native_steps=1310730,pd_updates=655365)
    with pytest.raises(ValueError):r.checked_worker_counts26([row])


def test_import_boundary_is_checked_in_a_fresh_process_not_shared_pytest():
    code="import sys;sys.path.insert(0,'.');import tools26.runtime26;assert not any(x.partition('.')[0] in {'torch','mujoco','numpy','gymnasium','stable_baselines3'} for x in sys.modules)"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True)



def test_actual_counted_monitor_constructor_step_and_close_see_forwarded_counts(tmp_path):
    source=(ROOT/'tools09/train_d1_ppo09.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='CountedMonitor')
    class BaseWrapper:
        def __init__(self,env):self.env=env
        def close(self):return self.env.close()
    class Monitor(BaseWrapper):
        def __init__(self,env,filename):super().__init__(env);self.rewards=[]
        def step(self,action):
            out=self.env.step(action);self.rewards.append(out[1]);return out
    counts={'policy_transitions':127,'native_steps':1270,'pd_updates':635,'resets':1,'forward_calls':129}
    class Env:
        def counters(self):return dict(counts)
        def close(self):self.closed=True
        def step(self,action):
            counts.update(policy_transitions=128,native_steps=1280,pd_updates=640,forward_calls=130)
            value=terms();return 'obs',sum(value['contributions_per_second'].values())*.02,False,False,{'reward_terms':value}
    writes=[]
    namespace={'Monitor':Monitor,'Path':Path,'os':os,'atomic_json':lambda path,value:writes.append(value)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual CountedMonitor>','exec'),namespace)  # noqa: S102 -- pure class with fake Monitor/env
    env=Env();wrapped=r.yaw_reward_wrapper26(BaseWrapper)(env,3.)
    monitor=namespace['CountedMonitor'](wrapped,str(tmp_path/'fake.monitor.csv'))
    assert writes[-1]['status']=='created' and writes[-1]['counts']['policy_transitions']==127
    out=monitor.step('action')
    assert monitor.rewards==[out[1]] and out[4]['reward_terms']['weights']['wz_tracking']==3.
    assert writes[-1]['status']=='running' and writes[-1]['counts']['policy_transitions']==128
    monitor.close()
    assert writes[-1]['status']=='closed' and env.closed
