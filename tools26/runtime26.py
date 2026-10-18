"""Independent26 Dummy1 runtime and the sole yaw-reward intervention.

All heavy imports occur only inside the explicitly launched runtime. Config,
commands, delay RNG/FIFO, sensors, action/native loop and observations inherit18.
"""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools16.train_comparison16 import training_phase
from tools18.train_delay18 import optimizer_sha
from tools25.low_memory_runtime25 import (
    LowMemoryRuntime25,
    checked_dummy_class25,
    checked_worker_counts25,
    loading_libraries25,
    validate_layout25,
)
from tools26 import common26 as c

require=c.require


def adjusted_reward26(reward,info,weight):
    """Change only wz weight on audited original terms, with dt applied once."""
    require(type(weight) is float and weight in (2.,3.),'reward26 weight must be exactly control2 or yaw3')
    require(type(reward) in (int,float) and math.isfinite(reward),'nonfinite original26 reward')
    terms=info['reward_terms']
    raw,weights,second,tick=(terms[name] for name in ('raw','weights','contributions_per_second','contributions_per_tick'))
    require(set(raw)==set(weights)==set(second)==set(tick) and 'wz_tracking' in raw
            and type(weights['wz_tracking']) is float and weights['wz_tracking']==2.
            and type(terms['policy_dt_s']) is float and terms['policy_dt_s']==.02,'original26 reward identity differs')
    for name in raw:
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (raw[name],weights[name],second[name],tick[name]))
                and math.isclose(second[name],raw[name]*weights[name],rel_tol=1e-12,abs_tol=1e-12)
                and math.isclose(tick[name],second[name]*.02,rel_tol=1e-12,abs_tol=1e-12),
                'original26 component/dt mismatch: '+name)
    require(0.<=raw['wz_tracking']<=1. and reward==float(sum(second.values())*.02),'original26 total reward mismatch')
    result=reward
    new_info=dict(info)
    if weight!=2.:
        changed=copy.deepcopy(terms)
        changed['weights']['wz_tracking']=weight
        changed['contributions_per_second']['wz_tracking']=raw['wz_tracking']*weight
        changed['contributions_per_tick']['wz_tracking']=raw['wz_tracking']*weight*.02
        result=float(sum(changed['contributions_per_second'].values())*.02)
        new_info['reward_terms']=changed
    extra=result-reward
    require(math.isfinite(result) and math.isclose(extra,(weight-2.)*raw['wz_tracking']*.02,rel_tol=1e-10,abs_tol=1e-12),
            'reward26 intervention changed more than yaw contribution')
    new_info['reward_change26']={'baseline_weight':2.,'weight':weight,'original_reward':reward,
        'training_reward':result,'extra_reward':extra,'raw_wz_tracking':raw['wz_tracking']}
    return result,new_info


def yaw_reward_wrapper26(base_class=None):
    if base_class is None:
        import gymnasium
        base_class=gymnasium.Wrapper
    class YawReward26(base_class):
        def __init__(self,env,weight):
            super().__init__(env)
            require(type(weight) is float and weight in (2.,3.),'unknown26 reward arm')
            self.weight26=weight
            self.reward_state26={'yaw_tracking_weight':weight,'steps':0,'original_reward_sum':0.,
                'training_reward_sum':0.,'extra_reward_sum':0.,'raw_wz_tracking_sum':0.}
        def step(self,action):
            observation,reward,terminated,truncated,info=self.env.step(action)
            new_reward,new_info=adjusted_reward26(reward,info,self.weight26)
            value=new_info['reward_change26'];state=self.reward_state26
            state['steps']+=1
            for key,part in (('original_reward_sum','original_reward'),('training_reward_sum','training_reward'),
                             ('extra_reward_sum','extra_reward'),('raw_wz_tracking_sum','raw_wz_tracking')):
                state[key]+=value[part]
            return observation,new_reward,terminated,truncated,new_info
        def counters(self):
            # CountedMonitor calls its immediate env directly (Gym1.2 does not
            # transparently forward unknown attributes). Preserve exact counts.
            return self.env.counters()
        def get_reward_state26(self):
            return dict(self.reward_state26)
    return YawReward26


def checked_worker_counts26(rows):
    return checked_worker_counts25(rows,transitions=c.TRANSITIONS)


def valid_reward_state26(state,arm,steps=None):
    try:
        require(type(state) is dict and set(state)=={'yaw_tracking_weight','steps','original_reward_sum',
            'training_reward_sum','extra_reward_sum','raw_wz_tracking_sum'} and state['yaw_tracking_weight']==c.ARMS[arm]
            and type(state['yaw_tracking_weight']) is float and type(state['steps']) is int
            and 0<=state['steps']<=c.TRANSITIONS and (steps is None or state['steps']==steps),'reward26 state identity/count differs')
        require(all(type(state[k]) in (int,float) and math.isfinite(state[k]) for k in state if k not in ('steps','yaw_tracking_weight')),
                'nonfinite26 reward totals')
        require(0<=state['raw_wz_tracking_sum']<=state['steps'] and math.isclose(state['extra_reward_sum'],
            (c.ARMS[arm]-2.)*state['raw_wz_tracking_sum']*.02,rel_tol=1e-9,abs_tol=1e-8)
            and math.isclose(state['training_reward_sum'],state['original_reward_sum']+state['extra_reward_sum'],rel_tol=1e-9,abs_tol=1e-8),
            'reward26 totals do not join actual intervention')
        return True
    except (ValueError,KeyError,TypeError):return False


def validate_settings26(model,hyper):
    for key in ('aux_coef','n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef',
                'max_grad_norm','learning_rate','target_kl'):
        require(getattr(model,key)==hyper[key],'runtime26 PPO differs: '+key)
    require(str(model.device)=='cpu' and type(model.seed) is int and model.seed==c.SEED
            and model.normalize_advantage is True and model.clip_range_vf is None and model.use_sde is False
            and model.clip_range(0.)==model.clip_range(1.)==.2
            and model.lr_schedule(0.)==model.lr_schedule(1.)==1e-4,'runtime26 CPU/schedule/seed differs')


def initial_identity_valid26(initial,plan):
    if not isinstance(initial,dict):return False
    fixed={'num_timesteps':c.INITIAL,'updates':c.BASE_UPDATES,'optimizer_restored':True,
        'checkpoint_sha256':plan['base_checkpoint']['sha256'],
        'serialized_entries_sha256':plan['baseline_provenance']['serialized_entries_sha256'],
        'seed_before':181,'seed_after':c.SEED,'seed_changes_policy_or_optimizer':False,
        'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[1e-4,1e-4],
        'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,
        'rollout_layout':{'backend':'DummyVecEnv','envs':1,'n_steps':4096,'rollout_transitions':4096,
            'buffer_class':'DictRolloutBuffer','actor_shape':[4096,1,627],'velocity_target_shape':[4096,1,3],
            'actions_shape':[4096,1,16],'last_obs_cleared':True,'interprocess_rpc':False,'environment_subprocesses':0}}
    try:
        require(all(type(initial.get(k)) is type(v) and initial[k]==v for k,v in fixed.items()),'initial26 fixed fields differ')
        require(json.dumps(initial['rollout_layout'],sort_keys=True,allow_nan=False)==json.dumps(fixed['rollout_layout'],sort_keys=True,allow_nan=False),
                'initial26 layout typed fields differ')
        require(all(isinstance(initial.get(k),str) and re.fullmatch('[0-9a-f]{64}',initial[k]) for k in (
            'policy_state_sha256','optimizer_state_sha256','policy_state_sha256_after_checks','optimizer_state_sha256_after_checks'))
            and initial['policy_state_sha256']==initial['policy_state_sha256_after_checks']
            and initial['optimizer_state_sha256']==initial['optimizer_state_sha256_after_checks'],'initial26 policy/Adam not preserved')
        return True
    except (ValueError,KeyError,TypeError):return False


class Runtime26(LowMemoryRuntime25):
    def create_vec(self,plan,transitions,output,root,mode):
        require(plan['envs']==1 and plan['seed']==c.SEED and transitions==c.TRANSITIONS and mode==c.MODE
                and self.arm in c.ARMS,'runtime26 environment scope differs')
        import gymnasium as gym
        from d1_rl.delay_env18 import D1Delay18Env
        from d1_rl.velocity_labels15 import VelocityLabelWrapper
        sys.path.insert(0,str(root/'tools09'))
        from train_d1_ppo09 import CountedMonitor
        reward_class=yaw_reward_wrapper26(gym.Wrapper)
        class PhaseWrapper(gym.Wrapper):
            def step(self,action):
                base=self.env.unwrapped
                phase=training_phase(base._training_schedule,base._episode_steps)
                obs,reward,terminated,truncated,info=self.env.step(action)
                return obs,reward,terminated,truncated,dict(info,training_phase=phase)
            def get_delay_state18(self):
                return self.env.unwrapped.get_delay_state18()
        def make():
            base=D1Delay18Env(training_delay_mode=mode,config=root/plan['config'],arm='B',seed=c.SEED,
                              rank=0,max_transitions=transitions)
            self.reward_env26=reward_class(base,c.ARMS[self.arm])
            return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(self.reward_env26,str(output/'env_00.monitor.csv'))))
        cls=checked_dummy_class25();self.vec=cls.__new__(cls)
        self.vec.__init__([make],mode=mode,seed=c.SEED,max_transitions=transitions)
        return self.vec

    def create_model(self,vec,plan,hyperparameters,output):
        require(plan['envs']==1 and plan['seed']==c.SEED and hyperparameters==c.HYPERPARAMETERS
                and plan['base_checkpoint']['trained_transitions']==c.INITIAL,'runtime26 warm-load scope differs')
        algorithm,buffer_type,configure=loading_libraries25()
        self._memory25('before_model_load')
        # Same SB3.load strict parameter+optimizer restoration as25; the bound
        # ZIP members and audited save_util/base_class bytes identify the source.
        # Re-seeding is performed before set_parameters, never after Adam restore.
        model=algorithm.load(plan['base_checkpoint']['path'],env=vec,device='cpu',seed=c.SEED,force_reset=True,n_steps=4096)
        self._memory25('after_model_load')
        require(type(model) is algorithm and type(model.num_timesteps) is int and model.num_timesteps==c.INITIAL
                and type(model._n_updates) is int and model._n_updates==c.BASE_UPDATES,'runtime26 completed25 offset/update differs')
        layout=validate_layout25(model,vec,buffer_type)
        validate_settings26(model,hyperparameters)
        optimizer=model.policy.optimizer
        require(bool(optimizer.param_groups) and all(group['lr']==1e-4 for group in optimizer.param_groups)
                and bool(optimizer.state_dict().get('state')),'runtime26 did not restore trained Adam state')
        initial={'num_timesteps':model.num_timesteps,'updates':model._n_updates,
            'policy_state_sha256':self.policy_sha(model),'optimizer_state_sha256':optimizer_sha(optimizer),
            'optimizer_restored':True,'checkpoint_sha256':plan['base_checkpoint']['sha256'],
            'serialized_entries_sha256':copy.deepcopy(plan['baseline_provenance']['serialized_entries_sha256']),
            'seed_before':181,'seed_after':c.SEED,'seed_changes_policy_or_optimizer':False,
            'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[model.lr_schedule(0.),model.lr_schedule(1.)],
            'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,'rollout_layout':layout}
        # No LR/KL setter or new optimizer is used: both are already correct25.
        initial['policy_state_sha256_after_checks']=self.policy_sha(model)
        initial['optimizer_state_sha256_after_checks']=optimizer_sha(optimizer)
        require(initial_identity_valid26(initial,plan),'runtime26 initial restoration proof incomplete')
        self.initial_identity=initial
        model.set_logger(configure(str(output),['csv']))
        return model

    def counters(self,vec):
        rows=vec.env_method('counters');checked_worker_counts26(rows);return rows

    def reward_state(self,vec):
        def validate(value):
            require(valid_reward_state26(value,self.arm),'invalid runtime26 reward provenance')
        return vec.single25.call('reward26',self.reward_env26.get_reward_state26,validator=validate,metadata=True)
