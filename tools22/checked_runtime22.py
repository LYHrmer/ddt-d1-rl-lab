"""Single-flight, fail-closed metadata RPC around the unchanged runtime18.

Only prepare/create_vec loads SB3 or physical dependencies. Import and the guard
itself use the standard library. Learning-rate configuration belongs to driver22.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools16.train_comparison16 import bounded_call, training_phase
from tools18.train_delay18 import RealRuntime as Runtime18
from tools18.train_delay18 import optimizer_sha

COUNTER_KEYS = {'policy_transitions', 'native_steps', 'pd_updates', 'resets', 'forward_calls'}
DELAY_KEYS = {'mode', 'seed', 'rank', 'episode_index', 'stream', 'selected_delay_ticks', 'selection'}


class RpcError22(RuntimeError):
    """A channel is unsafe; no further RPC may be attempted on this VecEnv."""


def require(value, message):
    if not value:
        raise RpcError22(message)


def json_ready22(value):
    """Validate, never coerce ndarray/NumPy scalars into purported provenance."""
    return json.loads(json.dumps(value, allow_nan=False))


def optimizer_without_lr_sha22(optimizer):
    """Original optimizer hash arithmetic, excluding only param-group lr."""
    state = optimizer.state_dict()
    filtered = {'state': state['state'], 'param_groups': [
        {key: value for key, value in group.items() if key != 'lr'} for group in state['param_groups']]}
    return optimizer_sha(SimpleNamespace(state_dict=lambda: filtered))


class RpcGuard22:
    """Own one VecEnv's RPC state, including a timed-out caller still in flight."""

    def __init__(self, vec, envs, mode, seed, *, max_transitions=None):
        require(type(envs) is int and envs > 0 and mode in ('zero','uniform_0_3') and type(seed) is int,
                'invalid RPC identity')
        self.vec, self.envs, self.mode, self.seed = vec, envs, mode, seed
        self.max_transitions = max_transitions
        self._lock = threading.Lock()
        self.dirty = self.closed = False
        self.reason = self.inflight = None
        self.attempts = self.returns = self.validated = 0
        self.generation = 0
        self.last_counters = self.counter_generation = None
        self.cleanup = None

    def _dirty(self, reason):
        self.dirty = True
        if self.reason is None:
            self.reason = str(reason)[:240]

    def mark_dirty(self, reason):
        with self._lock:
            self._dirty(reason)

    def snapshot(self):
        with self._lock:
            return json_ready22({'dirty': self.dirty, 'closed': self.closed, 'reason': self.reason,
                'inflight': self.inflight, 'waiting': getattr(self.vec, 'waiting', False) is True,
                'rpc_attempts': self.attempts, 'rpc_returns': self.returns, 'rpc_validated': self.validated,
                'generation': self.generation, 'cleanup': self.cleanup})

    def call(self, name, function, *, validator=None, allow_waiting=False, advances=False, metadata=False):
        with self._lock:
            unsafe = (self.dirty or self.closed or self.inflight is not None
                      or getattr(self.vec, 'waiting', False) is True and not allow_waiting)
            if unsafe:
                self._dirty('RPC refused: '+name+'; dirty/closed/inflight/waiting')
                raise RpcError22(self.reason)
            self.inflight = name
            if metadata:
                self.attempts += 1
        try:
            candidate = function()
            if metadata:
                with self._lock:
                    self.returns += 1
            if validator is not None:
                validator(candidate)
            checked = json_ready22(candidate) if metadata else candidate
            with self._lock:
                require(not self.dirty and not self.closed and self.inflight == name, 'late reply on dirty/closed channel')
                if name == 'counters':
                    self.last_counters = copy.deepcopy(checked)
                    self.counter_generation = self.generation
                if advances:
                    self.generation += 1
                if metadata:
                    self.validated += 1
            return checked
        except BaseException as exc:
            self.mark_dirty(name+': '+type(exc).__name__)
            raise
        finally:
            with self._lock:
                if self.inflight == name:
                    self.inflight = None

    def validate_counters(self, rows):
        require(type(rows) is list and len(rows) == self.envs, 'counter worker count/type differs')
        for rank, row in enumerate(rows):
            require(type(row) is dict and set(row) == COUNTER_KEYS
                    and all(type(value) is int and value >= 0 for value in row.values()), 'counter schema differs')
            p, n, d = row['policy_transitions'], row['native_steps'], row['pd_updates']
            require(max(0,p-1)*10 <= n <= p*10 and max(0,p-1)*5 <= d <= p*5 and 0 <= 2*d-n <= 2,
                    'counter physics is infeasible')
            require(self.max_transitions is None or p <= self.max_transitions, 'counter cap exceeded')
            if self.last_counters is not None:
                require(all(row[key] >= self.last_counters[rank][key] for key in COUNTER_KEYS), 'cumulative counters regressed')

    def validate_delays(self, rows):
        require(type(rows) is list and len(rows) == self.envs, 'delay worker count/type differs')
        require(self.last_counters is not None and self.counter_generation == self.generation,
                'delay provenance requires counters from this same idle boundary')
        for rank, row in enumerate(rows):
            require(type(row) is dict and set(row) == {'last_selection','delay_counts'}, 'delay row schema differs')
            counts, last = row['delay_counts'], row['last_selection']
            require(type(counts) is dict and set(counts) == {'0','1','2','3'}
                    and all(type(value) is int and value >= 0 for value in counts.values()), 'delay histogram differs')
            resets = sum(counts.values())
            require(resets == self.last_counters[rank]['resets'], 'delay/reset count differs')
            if last is None:
                require(resets == 0, 'missing last delay selection')
                continue
            require(type(last) is dict and set(last) == DELAY_KEYS and resets > 0
                    and all(type(last[key]) is int for key in ('seed','rank','episode_index','stream','selected_delay_ticks'))
                    and last['mode'] == self.mode and last['seed'] == self.seed+rank and last['rank'] == rank
                    and last['episode_index'] == resets-1 and last['stream'] == 18018
                    and last['selected_delay_ticks'] in (0,1,2,3) and last['selection'] == self.mode
                    and counts[str(last['selected_delay_ticks'])] > 0, 'delay selection identity differs')
            if self.mode == 'zero':
                require(last['selected_delay_ticks'] == 0 and all(counts[key] == 0 for key in ('1','2','3')),
                        'zero arm selected a delay')


def close_owned22(vec, *, clock=time.monotonic, terminate_s=3., kill_s=2.):
    """No RPC/recv: stop only the processes and pipes created by this VecEnv."""
    started, errors, signals = clock(), [], []
    processes = tuple(getattr(vec, 'processes', ()))
    owned = []
    for process in processes:
        if getattr(process, '_parent_pid', os.getpid()) != os.getpid():
            errors.append('foreign process handle refused')
        else:
            owned.append(process)
    for action, deadline in (('terminate', started+terminate_s), ('kill', started+terminate_s+kill_s)):
        for process in owned:
            try:
                if process.is_alive():
                    getattr(process, action)()
                    signals.append({'pid': process.pid, 'action': action})
            except (OSError, ValueError, AssertionError) as exc:
                errors.append(type(exc).__name__+': '+str(exc))
        for process in owned:
            try:
                process.join(timeout=max(0.,deadline-clock()))
            except (OSError, ValueError, AssertionError) as exc:
                errors.append(type(exc).__name__+': '+str(exc))
    remaining = []
    for process in owned:
        try:
            if process.is_alive():
                remaining.append(process.pid)
        except (OSError, ValueError, AssertionError):
            remaining.append(process.pid)
    for remote in (*getattr(vec,'remotes',()), *getattr(vec,'work_remotes',())):
        try:
            remote.close()
        except (OSError, ValueError) as exc:
            errors.append('pipe close: '+type(exc).__name__)
    receipt = {'cleanup_complete': not remaining and not errors, 'remaining_pids': remaining,
               'errors': errors, 'signals': signals, 'elapsed_s': clock()-started, 'rpc_used': False}
    guard = getattr(vec,'rpc_guard22',None)
    if guard is not None:
        with guard._lock:
            guard.cleanup = receipt
            guard.closed = receipt['cleanup_complete']
    if receipt['cleanup_complete']:
        vec.closed = True
    return receipt


def close_checked22(vec):
    """Driver finalizer: preserve normal SB3 close, bound it, then force if needed."""
    error = None
    try:
        bounded_call(vec.close, timeout=3.)
    except BaseException as exc:  # noqa: BLE001 - timeout/interruption must poison the owned channel before cleanup
        error = 'checked close: '+type(exc).__name__+': '+str(exc)
        vec.rpc_guard22.mark_dirty(error)
        receipt = close_owned22(vec)
        return receipt['cleanup_complete'], error
    snapshot = vec.rpc_guard22.snapshot()
    return snapshot['closed'], None if snapshot['closed'] else 'closed state unverified'


def checked_subproc_class22(base_class=None):
    """Lazy SB3 subclass; fake base injection exercises the actual methods."""
    if base_class is None:
        import stable_baselines3
        require(stable_baselines3.__version__ == '2.7.0', 'checked RPC supports local SB3 2.7.0 only')
        from stable_baselines3.common.vec_env import SubprocVecEnv
        base_class = SubprocVecEnv

    class CheckedSubprocVecEnv22(base_class):
        def __init__(self, env_fns, *, mode, seed, max_transitions=None, start_method='forkserver'):
            self.rpc_guard22 = RpcGuard22(self,len(env_fns),mode,seed,max_transitions=max_transitions)
            try:
                super().__init__(env_fns,start_method=start_method)
            except BaseException:
                self.rpc_guard22.mark_dirty('constructor interrupted')
                close_owned22(self)
                raise

        def step_async(self, actions):
            return self.rpc_guard22.call('step_async',lambda:super(CheckedSubprocVecEnv22,self).step_async(actions))

        def step_wait(self):
            if getattr(self,'waiting',False) is not True:
                self.rpc_guard22.mark_dirty('step_wait without pending step')
                raise RpcError22('step_wait without pending step')
            return self.rpc_guard22.call('step_wait',lambda:super(CheckedSubprocVecEnv22,self).step_wait(),
                                         allow_waiting=True,advances=True)

        def reset(self):
            return self.rpc_guard22.call('reset',lambda:super(CheckedSubprocVecEnv22,self).reset(),advances=True)

        def env_method(self, method_name, *args, indices=None, **kwargs):
            if indices is not None or args or kwargs or method_name not in ('counters','get_delay_state18'):
                self.rpc_guard22.mark_dirty('undeclared metadata RPC')
                raise RpcError22('only the declared whole-vector metadata RPC is allowed')
            validator = self.rpc_guard22.validate_counters if method_name == 'counters' else self.rpc_guard22.validate_delays
            return self.rpc_guard22.call('counters' if method_name=='counters' else 'delay_state',
                lambda:super(CheckedSubprocVecEnv22,self).env_method(method_name,*args,indices=indices,**kwargs),
                validator=validator,metadata=True)

        def get_attr(self, name, indices=None):
            return self.rpc_guard22.call('get_attr',lambda:super(CheckedSubprocVecEnv22,self).get_attr(name,indices=indices))

        def has_attr(self, name):
            return self.rpc_guard22.call('has_attr',lambda:super(CheckedSubprocVecEnv22,self).has_attr(name))

        def set_attr(self, name, value, indices=None):
            return self.rpc_guard22.call('set_attr',lambda:super(CheckedSubprocVecEnv22,self).set_attr(name,value,indices=indices),advances=True)

        def env_is_wrapped(self, cls, indices=None):
            return self.rpc_guard22.call('env_is_wrapped',lambda:super(CheckedSubprocVecEnv22,self).env_is_wrapped(cls,indices=indices))

        def close(self):
            state = self.rpc_guard22.snapshot()
            if state['closed']:
                return
            if state['dirty'] or state['inflight'] is not None or getattr(self,'waiting',False) is True:
                self.rpc_guard22.mark_dirty('forced close: dirty/inflight/waiting')
                require(close_owned22(self)['cleanup_complete'],'owned vector cleanup incomplete')
                return
            try:
                self.rpc_guard22.call('close',lambda:super(CheckedSubprocVecEnv22,self).close())
                receipt = close_owned22(self)
                receipt.update(rpc_used=True, method='original_sb3_close_then_owned_pipe_cleanup')
                require(receipt['cleanup_complete'],'normal close left owned resources')
            except BaseException:
                self.rpc_guard22.mark_dirty('normal close interrupted')
                close_owned22(self)
                raise
            return

    return CheckedSubprocVecEnv22


class CheckedRuntime22(Runtime18):
    """Reuse18 learning; only RPC safety and explicit driver model factory differ."""

    def __init__(self, *, model_factory=None):
        self.model_factory = model_factory

    def create_vec(self, plan, transitions, output, root, mode):
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

        def factory(rank):
            def make():
                base = D1Delay18Env(training_delay_mode=mode,config=root/plan['config'],arm='B',
                                   seed=plan['seed'],rank=rank,max_transitions=transitions//plan['envs'])
                return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(base,str(output/f'env_{rank:02d}.monitor.csv'))))
            return make
        return checked_subproc_class22()([factory(rank) for rank in range(plan['envs'])],mode=mode,seed=plan['seed'],
                                         max_transitions=transitions//plan['envs'],start_method='forkserver')

    def create_model(self, vec, plan, hyperparameters, output):
        if self.model_factory is not None:
            return self.model_factory(self,vec,plan,hyperparameters,output)
        return super().create_model(vec,plan,hyperparameters,output)

    def counters(self, vec):
        require(isinstance(getattr(vec,'rpc_guard22',None),RpcGuard22),'unguarded vector refused')
        return vec.env_method('counters')

    def delay_state(self, vec):
        require(isinstance(getattr(vec,'rpc_guard22',None),RpcGuard22),'unguarded vector refused')
        return vec.env_method('get_delay_state18')

    def rpc_snapshot(self, vec):
        return {'state':'NOT_CREATED','dirty':False} if vec is None else vec.rpc_guard22.snapshot()
