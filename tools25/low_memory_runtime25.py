"""Single-process25 runtime; importing this file never imports NN/physics libraries.

The frozen task/wrappers and original complete policy/Adam are retained. Only
the vector layout becomes DummyVecEnv(1) / 4096 steps per rollout; LR/KL changes
are explicit and checked independently of restoring optimizer momentum.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools16.train_comparison16 import (
    bounded_call,
    count_totals,
    training_phase,
)
from tools18.train_delay18 import RealRuntime as OriginalRuntime
from tools18.train_delay18 import optimizer_sha
from tools22.checked_runtime22 import (
    RpcGuard22,
    optimizer_without_lr_sha22,
)


def require(value, message):
    if not value:
        raise ValueError(message)


def contract25():
    from tools25 import common25
    return common25


def checked_worker_counts25(rows, transitions=None):
    """One real cumulative row; partial native/PD interruption remains visible."""
    if transitions is None:
        transitions = contract25().TRANSITIONS
    require(type(rows) is list and len(rows) == 1, 'single25 needs exactly one environment counter row')
    total = count_totals(rows)
    require(set(total) == {'policy_transitions', 'native_steps', 'pd_updates', 'resets', 'forward_calls'},
            'single25 counter schema differs')
    p, n, d = (total[key] for key in ('policy_transitions', 'native_steps', 'pd_updates'))
    require(type(transitions) is int and 0 <= p <= transitions and max(0, p-1)*10 <= n <= p*10
            and max(0, p-1)*5 <= d <= p*5 and 0 <= 2*d-n <= 2, 'single25 counters exceed cap/physical interval')
    json.dumps(rows, allow_nan=False)
    return total


def checked_dummy_class25(base_class=None):
    """Reuse22's pure metadata validators; no pipe, server or worker is created."""
    if base_class is None:
        import stable_baselines3
        require(stable_baselines3.__version__ == '2.7.0', 'single25 requires SB3 2.7.0')
        from stable_baselines3.common.vec_env import DummyVecEnv
        base_class = DummyVecEnv

    class CheckedDummyVecEnv25(base_class):
        def __init__(self, env_fns, *, mode, seed, max_transitions):
            require(len(env_fns) == 1, 'single25 forbids additional environment instances')
            self.envs = []
            self.waiting = False
            self.single25 = RpcGuard22(self, 1, mode, seed, max_transitions=max_transitions)
            self.construction_complete25 = False
            try:
                super().__init__(env_fns)
                require(self.num_envs == 1 and not getattr(self, 'processes', ())
                        and not getattr(self, 'remotes', ()), 'single25 unexpectedly owns IPC/processes')
                self.construction_complete25 = True
            except BaseException as exc:
                self.single25.mark_dirty('construction failed: '+type(exc).__name__)
                raise

        def step_async(self, actions):
            result = self.single25.call('step_async', lambda:super(CheckedDummyVecEnv25,self).step_async(actions))
            self.waiting = True
            return result

        def step_wait(self):
            require(self.waiting is True, 'single25 step_wait without queued action')
            def step():
                result = super(CheckedDummyVecEnv25,self).step_wait()
                self.waiting = False
                return result
            return self.single25.call('step_wait', step, allow_waiting=True, advances=True)

        def reset(self):
            return self.single25.call('reset', lambda:super(CheckedDummyVecEnv25,self).reset(), advances=True)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            require(method_name in ('counters','get_delay_state18') and not args and not kwargs
                    and indices is None, 'single25 allows only complete direct metadata snapshots')
            guard = self.single25
            validator = guard.validate_counters if method_name == 'counters' else guard.validate_delays
            return guard.call('counters' if method_name == 'counters' else 'delays',
                lambda:super(CheckedDummyVecEnv25,self).env_method(method_name, indices=indices),
                validator=validator, metadata=True)

        def close(self):
            guard = self.single25
            with guard._lock:
                if guard.closed:
                    return
                require(guard.inflight is None, 'single25 cannot close during in-process operation')
            try:
                super().close()
                require(self.construction_complete25 is True, 'single25 construction ownership incomplete')
            except BaseException as exc:
                guard.mark_dirty('close failed: '+type(exc).__name__)
                raise
            with guard._lock:
                self.waiting = False
                guard.closed = True

    return CheckedDummyVecEnv25


def close_single25(vec):
    if vec is None:
        return True, None
    try:
        bounded_call(vec.close, timeout=3.)
        require(vec.single25.snapshot()['closed'] is True, 'single25 close did not complete')
        return True, None
    except BaseException as exc:  # noqa: BLE001 -- interrupted or timed-out close must not claim cleanup
        if getattr(vec, 'single25', None) is not None:
            vec.single25.mark_dirty('close: '+type(exc).__name__)
        # A timed-out in-process call cannot be "killed" as an env subprocess.
        # The external owned training-process supervisor provides that boundary.
        return False, type(exc).__name__+': '+str(exc)


def loading_libraries25():
    from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
    from stable_baselines3.common.buffers import DictRolloutBuffer
    from stable_baselines3.common.logger import configure
    return AuxiliaryVelocityPPO, DictRolloutBuffer, configure


def self_memory25(path=Path('/proc/self/smaps_rollup')):
    """One process's RSS/PSS, never a sum of possibly shared worker RSS."""
    result = {'pid':os.getpid(),'rss_bytes':None,'pss_bytes':None,'error':None}
    try:
        rows = {key:value.split() for line in path.read_text().splitlines()
                if ':' in line for key,value in [line.split(':',1)]}
        for name,key in (('Rss','rss_bytes'),('Pss','pss_bytes')):
            require(len(rows[name]) == 2 and rows[name][1] == 'kB' and int(rows[name][0]) >= 0,
                    'invalid smaps_rollup '+name)
            result[key] = int(rows[name][0])*1024
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result.update(rss_bytes=None,pss_bytes=None,error=type(exc).__name__+': '+str(exc))
    return result


def validate_layout25(model, vec, buffer_type):
    require(type(model.n_envs) is int and model.n_envs == vec.num_envs == 1
            and type(model.n_steps) is int and model.n_steps == 4096 and model.get_env() is vec
            and model._last_obs is None, 'single25 warm-load did not rebuild/reset single environment')
    buffer = model.rollout_buffer
    require(type(buffer) is buffer_type and buffer.buffer_size == 4096 and buffer.n_envs == 1
            and buffer.pos == 0 and buffer.full is False and buffer.generator_ready is False
            and set(buffer.observations) == {'actor','velocity_target'}, 'single25 exact fresh DictRolloutBuffer required')
    arrays = {**buffer.observations, 'actions':buffer.actions,
              **{key:getattr(buffer,key) for key in ('rewards','returns','episode_starts','values','log_probs','advantages')}}
    shapes = {'actor':(4096,1,627), 'velocity_target':(4096,1,3), 'actions':(4096,1,16)}
    require(all(array.shape == shapes.get(key,(4096,1)) and str(array.dtype) == 'float32'
                for key,array in arrays.items()), 'single25 buffer shape/dtype differs')
    return {'backend':'DummyVecEnv', 'envs':1, 'n_steps':4096, 'rollout_transitions':4096,
            'buffer_class':'DictRolloutBuffer', 'actor_shape':[4096,1,627],
            'velocity_target_shape':[4096,1,3], 'actions_shape':[4096,1,16],
            'last_obs_cleared':True, 'interprocess_rpc':False, 'environment_subprocesses':0}


def validate_settings25(model, hyper, *, baseline):
    expected = dict(hyper, learning_rate=3e-4, target_kl=None) if baseline else hyper
    for key in ('aux_coef','n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef',
                'max_grad_norm','learning_rate','target_kl'):
        require(getattr(model,key) == expected[key], 'single25 loaded PPO setting differs: '+key)
    require(str(model.device) == 'cpu' and model.seed == 181 and model.normalize_advantage is True
            and model.clip_range_vf is None and model.use_sde is False
            and model.clip_range(0.) == model.clip_range(1.) == .2
            and model.lr_schedule(0.) == model.lr_schedule(1.) == expected['learning_rate'],
            'single25 loaded CPU/schedule/PPO semantics differ')


def initial_identity_valid25(initial):
    common = contract25()
    if not isinstance(initial, dict):
        return False
    base = common.BASE_INITIAL_STATE
    fixed = {'optimizer_momentum_preserved':True, 'learning_rate_before':3e-4, 'learning_rate_after':1e-4,
             'lr_schedule_endpoints':[1e-4,1e-4], 'target_kl_before':None, 'target_kl_after':.005,
             'target_kl_changes_optimizer_state':False,
             'rollout_layout':{'backend':'DummyVecEnv', 'envs':1, 'n_steps':4096, 'rollout_transitions':4096,
                'buffer_class':'DictRolloutBuffer', 'actor_shape':[4096,1,627], 'velocity_target_shape':[4096,1,3],
                'actions_shape':[4096,1,16], 'last_obs_cleared':True, 'interprocess_rpc':False, 'environment_subprocesses':0}}
    try:
        return (all(type(initial.get(k)) is type(v) and initial[k] == v for k,v in {**base,**fixed}.items())
            and json.dumps(initial['rollout_layout'],sort_keys=True,allow_nan=False)
                == json.dumps(fixed['rollout_layout'],sort_keys=True,allow_nan=False)
            and initial['optimizer_state_sha256_before_lr'] == base['optimizer_state_sha256']
            and initial['optimizer_state_sha256_before_lr'] != initial['optimizer_state_sha256_after_lr']
            and initial['optimizer_without_lr_sha256_before'] == initial['optimizer_without_lr_sha256_after']
            and all(type(initial[k]) is str and len(initial[k]) == 64 and set(initial[k]) <= set('0123456789abcdef')
                    for k in ('optimizer_state_sha256_after_lr','optimizer_without_lr_sha256_before'))
            and initial['optimizer_state_sha256_before_target_kl'] == initial['optimizer_state_sha256_after_lr']
            and initial['optimizer_state_sha256_after_target_kl'] == initial['optimizer_state_sha256_after_lr'])
    except (KeyError,TypeError):
        return False


class LowMemoryRuntime25(OriginalRuntime):
    def __init__(self):
        self.vec = None
        self.memory_samples25 = []

    def _memory25(self,phase):
        self.memory_samples25.append(dict(self_memory25(),phase=phase,
            vec_num_envs=None if self.vec is None else getattr(self.vec,'num_envs',None),
            environment_process_count=len(getattr(self.vec,'processes',())) if self.vec is not None else 0))

    def create_vec(self, plan, transitions, output, root, mode):
        common = contract25()
        require(plan['envs'] == 1 and plan['seed'] == 181 and transitions == common.TRANSITIONS
                and mode in ('zero','uniform_0_3'), 'single25 environment scope differs')
        import gymnasium as gym
        from d1_rl.delay_env18 import D1Delay18Env
        from d1_rl.velocity_labels15 import VelocityLabelWrapper
        sys.path.insert(0,str(root/'tools09'))
        from train_d1_ppo09 import CountedMonitor

        class PhaseWrapper(gym.Wrapper):
            def step(self, action):
                base = self.env.unwrapped
                phase = training_phase(base._training_schedule,base._episode_steps)
                obs,reward,terminated,truncated,info = self.env.step(action)
                return obs,reward,terminated,truncated,dict(info,training_phase=phase)

            def get_delay_state18(self):
                return self.env.unwrapped.get_delay_state18()

        def make():
            base = D1Delay18Env(training_delay_mode=mode,config=root/plan['config'],arm='B',
                seed=plan['seed'],rank=0,max_transitions=transitions)
            return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(base,str(output/'env_00.monitor.csv'))))
        cls = checked_dummy_class25()
        self.vec = cls.__new__(cls)
        self.vec.__init__([make],mode=mode,seed=plan['seed'],max_transitions=transitions)
        return self.vec

    def create_model(self, vec, plan, hyperparameters, output):
        common = contract25()
        require(plan['envs'] == 1 and plan['seed'] == 181 and hyperparameters == common.HYPERPARAMETERS
                and hyperparameters['n_steps'] == 4096 and hyperparameters['target_kl'] == .005
                and plan['base_checkpoint']['trained_transitions'] == common.INITIAL,
                'single25 warm-load scope differs')
        algorithm, buffer_type, configure = loading_libraries25()
        # SB3.load writes n_envs from vec, then applies n_steps before _setup_model
        # builds the buffer; set_parameters afterwards restores policy AND Adam.
        self._memory25('before_model_load')
        model = algorithm.load(plan['base_checkpoint']['path'],env=vec,device='cpu',seed=181,
                               force_reset=True,n_steps=4096)
        self._memory25('after_model_load')
        require(type(model) is algorithm and model.num_timesteps == common.INITIAL
                and type(model.num_timesteps) is int and type(model._n_updates) is int,
                'single25 baseline algorithm/timestep identity differs')
        layout = validate_layout25(model,vec,buffer_type)
        validate_settings25(model,hyperparameters,baseline=True)
        initial = {'num_timesteps':model.num_timesteps, 'policy_state_sha256':self.policy_sha(model),
            'optimizer_state_sha256':optimizer_sha(model.policy.optimizer), 'optimizer_restored':True,
            'updates':model._n_updates}
        require(initial == common.BASE_INITIAL_STATE, 'single25 original complete policy/Adam/update state differs')
        optimizer = model.policy.optimizer
        before, without_before = optimizer_sha(optimizer), optimizer_without_lr_sha22(optimizer)
        require(bool(optimizer.param_groups) and all(group['lr'] == 3e-4 for group in optimizer.param_groups),
                'single25 restored Adam LR differs')
        model.learning_rate = hyperparameters['learning_rate']
        model._setup_lr_schedule()
        for group in optimizer.param_groups:
            group['lr'] = hyperparameters['learning_rate']
        after, without_after = optimizer_sha(optimizer), optimizer_without_lr_sha22(optimizer)
        require(before != after and without_before == without_after
                and self.policy_sha(model) == initial['policy_state_sha256']
                and all(group['lr'] == 1e-4 for group in optimizer.param_groups),
                'single25 LR altered policy/Adam momentum')
        model.target_kl = hyperparameters['target_kl']
        kl_after = optimizer_sha(optimizer)
        require(after == kl_after and self.policy_sha(model) == initial['policy_state_sha256'],
                'single25 KL setting changed policy/Adam')
        validate_settings25(model,hyperparameters,baseline=False)
        initial.update(rollout_layout=layout, optimizer_state_sha256_before_lr=before,
            optimizer_state_sha256_after_lr=after, optimizer_without_lr_sha256_before=without_before,
            optimizer_without_lr_sha256_after=without_after, optimizer_momentum_preserved=True,
            learning_rate_before=3e-4,learning_rate_after=1e-4,lr_schedule_endpoints=[model.lr_schedule(0.),model.lr_schedule(1.)],
            target_kl_before=None,target_kl_after=.005,target_kl_changes_optimizer_state=False,
            optimizer_state_sha256_before_target_kl=after,optimizer_state_sha256_after_target_kl=kl_after)
        require(initial_identity_valid25(initial), 'single25 incomplete warm-load transition proof')
        self.initial_identity = initial
        model.set_logger(configure(str(output),['csv']))
        return model

    def counters(self, vec):
        require(getattr(vec,'single25',None) is not None, 'single25 unguarded environment refused')
        rows = vec.env_method('counters')
        checked_worker_counts25(rows)
        return rows

    def delay_state(self, vec):
        require(getattr(vec,'single25',None) is not None, 'single25 unguarded delay source refused')
        return vec.env_method('get_delay_state18')

    def snapshot(self, vec=None):
        vec = self.vec if vec is None else vec
        if vec is None:
            return {'backend':'DummyVecEnv','state':'NOT_CREATED','interprocess_rpc':False,
                    'environment_subprocesses':0,'dirty':False,'closed':False,'waiting':False,
                    'inflight':None,'reason':None,'generation':0,'metadata_attempts':0,
                    'metadata_returns':0,'metadata_validated':0,'construction_complete':False,
                    'process_memory_samples':list(self.memory_samples25)}
        raw = vec.single25.snapshot()
        return {'backend':'DummyVecEnv','state':'CLOSED' if raw['closed'] else 'ACTIVE',
                'interprocess_rpc':False,'environment_subprocesses':0,
                **{key:raw[key] for key in ('dirty','closed','reason','inflight','waiting','generation')},
                'metadata_attempts':raw['rpc_attempts'],'metadata_returns':raw['rpc_returns'],
                'metadata_validated':raw['rpc_validated'], 'construction_complete':vec.construction_complete25,
                'process_memory_samples':list(self.memory_samples25)}

    def close(self, vec=None):
        result = close_single25(self.vec if vec is None else vec)
        self._memory25('after_close')
        return result
