"""Independent29 fixed-base KL arm; original completed25 weights and Adam."""
from __future__ import annotations
import argparse
import hashlib
import json
import signal
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools16.train_comparison16 import StopRequested, Watchdog, bounded_call, complete_counts, termination_scope
from tools18.train_delay18 import valid_delay_state
from tools29 import common29 as common
from tools29.anchor29 import valid_anchor_receipt29
(INITIAL, TRANSITIONS) = (common.INITIAL, common.TRANSITIONS)
HYPERPARAMETERS = common.HYPERPARAMETERS
TARGET_KL = 0.005
(require, sha, atomic_json) = (common.require, common.sha, common.atomic_json)

class Watchdog29(Watchdog):
    """Retain guarded accounting; hash the first existing buffer before PPO."""
    first_rollout_nonreward = None

    def on_rollout_end(self, transitions):
        if self.first_rollout_nonreward is None:
            self.first_rollout_nonreward = first_rollout_proof29(self.model.rollout_buffer)
            atomic_json(self.output / 'first_rollout29.json', self.first_rollout_nonreward)
        return super().on_rollout_end(transitions)

    def poll(self, *, force=False, phase='training'):
        elapsed = self.clock() - self.started
        if force or elapsed - self.last_poll >= self.release['resources']['poll_s']:
            sample = dict(self.memory_reader(), elapsed_s=elapsed, phase=phase)
            require(all((type(sample.get(key)) is int and sample[key] >= 0 for key in ('available_ram_bytes', 'effective_available_bytes'))), 'invalid RAM accounting')
            (self.last_poll, self.last_sample) = (elapsed, sample)
            self.min_available = sample['available_ram_bytes'] if self.min_available is None else min(self.min_available, sample['available_ram_bytes'])
            self.min_effective = sample['effective_available_bytes'] if self.min_effective is None else min(self.min_effective, sample['effective_available_bytes'])
            with (self.output / 'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False) + '\n')
            if sample['effective_available_bytes'] < self.release['resources']['reserve_bytes'] and self.reason is None:
                (self.reason, self.stop_sample) = ('RAM_RESERVE_BELOW_2_GIB', sample)
        if elapsed >= self.release['stages'][self.stage]['max_wall_s'] and self.reason is None:
            (self.reason, self.stop_sample) = ('WALL_LIMIT', self.last_sample)
        if self.reason:
            atomic_json(self.output / 'stop_sample.json', dict(self.stop_sample or {}, reason=self.reason))
        return self.reason is None

def json_ready(value):
    json.dumps(value, allow_nan=False)
    return value

def checked_worker_counts(rows):
    from tools29.anchor29 import checked_worker_counts29
    return checked_worker_counts29(rows)

def initial_identity_valid(initial, plan):
    from tools29.anchor29 import initial_identity_valid29
    return initial_identity_valid29(initial, plan)

def first_rollout_proof29(buffer):
    arrays = {**{'observations.' + key: value for (key, value) in buffer.observations.items()}, **{key: getattr(buffer, key) for key in ('actions', 'episode_starts', 'values', 'log_probs')}}
    require(set(arrays) == {'observations.actor', 'observations.velocity_target', 'actions', 'episode_starts', 'values', 'log_probs'}, 'first29 rollout fields differ')
    identities = {}
    for (key, array) in sorted(arrays.items()):
        view = memoryview(array)
        require(view.c_contiguous, 'first29 buffer is not contiguous')
        identities[key] = {'shape': list(view.shape), 'format': view.format, 'nbytes': view.nbytes, 'sha256': hashlib.sha256(view.cast('B')).hexdigest()}
    raw = json.dumps(identities, sort_keys=True, allow_nan=False).encode()
    return {'format': 'd1-anchor29-first-rollout-v1', 'rollout_transitions': 4096, 'arrays': identities, 'sha256': hashlib.sha256(raw).hexdigest()}

def memory_ok(sample, minimum):
    value = sample.get('effective_available_bytes')
    require(type(value) is int and value >= 0, 'effective RAM unknown')
    return value >= minimum

from tools25.low_memory_runtime25 import LowMemoryRuntime25


class Runtime29(LowMemoryRuntime25):
    """Keep checked Dummy1 ownership; all scope/model gates are explicit29."""
    def _memory25(self,phase):
        from tools25.low_memory_runtime25 import self_memory25
        sample=common.read_memory()
        self.memory_samples25.append(dict(self_memory25(),phase=phase,
            vec_num_envs=None if self.vec is None else getattr(self.vec,'num_envs',None),
            environment_process_count=0,resource_sample=sample))
        require(memory_ok(sample,common.RESOURCES['reserve_bytes']),'RAM_RESERVE_BELOW_2_GIB '+phase)

    def create_vec(self,plan,transitions,output,root,mode):
        require(plan['envs']==1 and plan['seed']==common.SEED and transitions==TRANSITIONS
                and mode==common.MODE and self.arm in common.ARMS,'runtime29 scope differs')
        import gymnasium as gym
        from d1_rl.delay_env18 import D1Delay18Env
        from d1_rl.velocity_labels15 import VelocityLabelWrapper
        from tools16.train_comparison16 import training_phase
        from tools25.low_memory_runtime25 import checked_dummy_class25
        from tools26.runtime26 import yaw_reward_wrapper26
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
            base=D1Delay18Env(training_delay_mode=mode,config=root/plan['config'],arm='B',
                              seed=common.SEED,rank=0,max_transitions=transitions)
            # Exactly original reward2 in both arms, inside the Monitor.
            self.reward_env29=reward_class(base,2.)
            return PhaseWrapper(VelocityLabelWrapper(CountedMonitor(self.reward_env29,str(output/'env_00.monitor.csv'))))
        cls=checked_dummy_class25();self.vec=cls.__new__(cls)
        self.vec.__init__([make],mode=mode,seed=common.SEED,max_transitions=transitions)
        return self.vec

    def create_model(self,vec,plan,hyperparameters,output):
        import copy
        from stable_baselines3.common.buffers import DictRolloutBuffer
        from stable_baselines3.common.logger import configure
        from tools18.train_delay18 import optimizer_sha
        from tools25.low_memory_runtime25 import validate_layout25
        from tools29.ppo_anchor29 import AnchoredVelocityPPO29
        require(plan['envs']==1 and plan['seed']==common.SEED and hyperparameters==HYPERPARAMETERS
                and plan['base_checkpoint']['trained_transitions']==INITIAL,'runtime29 base/scope differs')
        self._memory25('before_model_load')
        model=AnchoredVelocityPPO29.load(plan['base_checkpoint']['path'],env=vec,device='cpu',
            seed=common.SEED,force_reset=True,n_steps=4096,anchor_coef=common.ARMS[self.arm])
        self._memory25('after_model_load')
        require(type(model) is AnchoredVelocityPPO29 and type(model.num_timesteps) is int and model.num_timesteps==INITIAL
                and type(model._n_updates) is int and model._n_updates==common.BASE_UPDATES,'runtime29 base offsets differ')
        layout=validate_layout25(model,vec,DictRolloutBuffer)
        for key in ('aux_coef','n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef',
                    'max_grad_norm','learning_rate','target_kl'):
            require(getattr(model,key)==hyperparameters[key],'runtime29 PPO differs: '+key)
        require(str(model.device)=='cpu' and type(model.seed) is int and model.seed==common.SEED
                and model.normalize_advantage is True and model.clip_range_vf is None and model.use_sde is False
                and model.clip_range(0.)==model.clip_range(1.)==.2
                and model.lr_schedule(0.)==model.lr_schedule(1.)==1e-4,'runtime29 CPU/schedule differs')
        optimizer=model.policy.optimizer
        require(bool(optimizer.param_groups) and all(row['lr']==1e-4 for row in optimizer.param_groups)
                and bool(optimizer.state_dict().get('state')),'runtime29 Adam not restored')
        initial={'num_timesteps':model.num_timesteps,'updates':model._n_updates,
            'policy_state_sha256':self.policy_sha(model),'optimizer_state_sha256':optimizer_sha(optimizer),
            'optimizer_restored':True,'checkpoint_sha256':plan['base_checkpoint']['sha256'],
            'serialized_entries_sha256':copy.deepcopy(plan['baseline_provenance']['serialized_entries_sha256']),
            'seed_before':181,'seed_after':common.SEED,'seed_changes_policy_or_optimizer':False,
            'learning_rate_before':1e-4,'learning_rate_after':1e-4,'lr_schedule_endpoints':[model.lr_schedule(0.),model.lr_schedule(1.)],
            'target_kl_before':.005,'target_kl_after':.005,'optimizer_momentum_preserved':True,'rollout_layout':layout}
        model.configure_anchor29()
        initial['policy_state_sha256_after_checks']=self.policy_sha(model)
        initial['optimizer_state_sha256_after_checks']=optimizer_sha(optimizer)
        require(initial_identity_valid(initial,plan),'runtime29 incomplete original policy/Adam proof')
        self.initial_identity=initial
        model.set_logger(configure(str(output),['csv']))
        return model

    def counters(self,vec):
        rows=vec.env_method('counters');checked_worker_counts(rows);return rows

    def reward_state(self,vec):
        from tools29.anchor29 import valid_reward_state29
        def validate(value):
            require(valid_reward_state29(value),'runtime29 original reward changed')
        return vec.single25.call('reward29',self.reward_env29.get_reward_state26,validator=validate,metadata=True)


def run(plan_path, arm, output, *, runtime=None, loader=common.load_plan, launch_validator=common.validate_launch, memory_reader=common.read_memory, clock=time.monotonic, root=ROOT, closer=None):
    started = clock()
    (root, plan_path, output) = (Path(root).resolve(), Path(plan_path).resolve(), Path(output).absolute())
    plan = loader(plan_path)
    require(arm in common.ARMS, 'unknown29 arm')
    (stage, mode) = ('train_' + arm, common.MODE)
    spec = plan['stages'][stage]
    require(spec['mode'] == mode and spec['transitions'] == TRANSITIONS and (plan['envs'] == 1) and (plan['seed'] == common.SEED) and (spec['anchor_coef'] == common.ARMS[arm]) and (plan.get('hyperparameters') == HYPERPARAMETERS), 'new29 training scope differs')
    require(output == root / plan['output_root'] / arm / 'train' and (not output.exists()) and output.is_relative_to(root), 'exclusive fixed29 training output required')
    cursor = output
    while cursor != root:
        require(not cursor.is_symlink(), 'symlink training output forbidden')
        cursor = cursor.parent
    launch_validator(plan, plan_path, stage, output)
    (baseline, base_protocol) = (plan['base_checkpoint'], plan['base_protocol'])
    (checkpoint, protocol_path) = (Path(baseline['path']), Path(base_protocol['path']))
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
    protocol_path = protocol_path if protocol_path.is_absolute() else root / protocol_path
    require(sha(checkpoint) == baseline['sha256'] and baseline['trained_transitions'] == INITIAL and (sha(protocol_path) == base_protocol['sha256']), 'baseline identity changed')
    original = common.read_json(protocol_path)
    require(original['hyperparameters'] == HYPERPARAMETERS, 'completed25 PPO settings changed')
    require(memory_ok(memory_reader(), plan['resources']['start_min_effective_bytes']), 'NOT_READY: insufficient start RAM')
    plan_sha = sha(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'format': 'd1-anchor29-training-protocol-v1', 'plan_sha256': plan_sha, 'stage': stage, 'arm': arm, 'mode': mode, 'seed': plan['seed'], 'envs': plan['envs'], 'planned_new_transitions': TRANSITIONS, 'planned_native_steps': TRANSITIONS * 10, 'base_checkpoint': baseline, 'base_protocol': base_protocol, 'source_identity': plan['source_identity'], 'hyperparameters': HYPERPARAMETERS, 'baseline_hyperparameters': HYPERPARAMETERS, 'versions': None, 'warm_start': True, 'teacher_policy_used': common.ARMS[arm] != 0.0, 'reset_num_timesteps': False, 'checkpoint_selection': 'final_only', 'delay_sampling': mode + '; inherited stream18 and FIFO; original paired commands', 'yaw_tracking_weight': 2.0, 'anchor_change': common.anchor_change(arm), 'target_kl_change': {'before': TARGET_KL, 'after': TARGET_KL, 'changed': False, 'normal_batch_stop_strict_above': 1.5 * TARGET_KL}, 'comparison_between_arms': 'only fixed-base KL coefficient differs; original reward2 and shared uniform_0_3 delay', 'lr_change': {'before': 0.0001, 'after': 0.0001, 'changed': False, 'schedule': 'constant', 'all_optimizer_param_groups': True}, 'single_variable_effect_against22_claimed': False, 'single_variable_effect_against24_claimed': False, 'vector_layout': plan['vector_layout'], 'runtime_dependencies_sha256': plan['runtime_dependencies_sha256'], 'anchor_coef': common.ARMS[arm]}
    atomic_json(output / 'protocol.json', protocol)
    vec = model = watcher = workers = totals = delays = initial = reward_state = None
    rpc = {'status': 'not_created'}
    (status, error, cleanup, unchanged, spawned) = ('ERROR', None, True, False, False)
    checkpoint_sha = startup_stop = None

    def unchanged_inputs():
        loader(plan_path)
        require(sha(plan_path) == plan_sha and sha(checkpoint) == baseline['sha256'] and (sha(protocol_path) == base_protocol['sha256']), 'plan or baseline changed')
    with termination_scope() as termination:
        try:
            if runtime is None:
                runtime = Runtime29()
            runtime.arm = arm
            runtime.prepare()
            protocol['versions'] = json_ready(runtime.versions)
            atomic_json(output / 'protocol.json', protocol)
            require(runtime.versions == original['versions'], 'runtime versions differ from original completed baseline')
            unchanged_inputs()
            require(memory_ok(memory_reader(), plan['resources']['reserve_bytes']), 'RAM_RESERVE_BELOW_2_GIB after imports')
            require(clock() - started < spec['max_wall_s'], 'WALL_LIMIT before spawn')
            (termination['defer'], spawned) = (True, True)
            try:
                vec = runtime.create_vec(plan, TRANSITIONS, output, root, mode)
            finally:
                termination['defer'] = False
            if termination['received']:
                raise StopRequested(termination['received'])
            sample = dict(memory_reader(), elapsed_s=clock() - started, phase='after_spawn')
            if not memory_ok(sample, plan['resources']['reserve_bytes']):
                startup_stop = dict(sample, reason='RAM_RESERVE_BELOW_2_GIB')
                atomic_json(output / 'stop_sample.json', startup_stop)
                raise StopRequested(startup_stop['reason'])
            model = runtime.create_model(vec, plan, HYPERPARAMETERS, output)
            candidate = runtime.initial_identity
            require(initial_identity_valid(candidate, plan) and getattr(model, 'target_kl', None) == TARGET_KL, 'original optimizer/LR/KL transition proof or actual model target incomplete')
            initial = json_ready(candidate)
            protocol['initial_state'] = initial
            atomic_json(output / 'protocol.json', protocol)
            watcher = Watchdog29(output, plan, stage, started, runtime, vec, model, memory_reader, clock)
            model._anchor_poll29 = lambda phase, force=False: watcher.poll(force=force, phase=phase)
            runtime.learn(model, TRANSITIONS, watcher)
            watcher.poll(force=True, phase='after_learn')
            status = watcher.reason or ('COMPLETE' if model.num_timesteps == INITIAL + TRANSITIONS else 'PARTIAL')
        except BaseException as exc:
            status = watcher.reason if watcher and watcher.reason else str(exc) if isinstance(exc, StopRequested) else 'ERROR'
            error = f'{type(exc).__name__}: {exc}'
            (output / 'error.txt').write_text(traceback.format_exc())
        finally:
            termination['defer'] = True
            for number in (signal.SIGINT, signal.SIGTERM):
                signal.signal(number, signal.SIG_IGN)
            if vec is None and runtime is not None:
                vec = getattr(runtime, 'vec', None)
            if vec is not None:
                try:
                    candidate = bounded_call(lambda : runtime.counters(vec))
                    candidate_totals = checked_worker_counts(candidate)
                    (workers, totals) = (candidate, candidate_totals)
                except BaseException as exc:
                    error = (error or '') + f'; counter collection failed: {type(exc).__name__}: {exc}'
                try:
                    candidate = bounded_call(lambda : runtime.delay_state(vec))
                    require(valid_delay_state(candidate, 1, mode), 'invalid delay provenance')
                    delays = json_ready(candidate)
                except BaseException as exc:
                    error = (error or '') + f'; delay provenance failed: {type(exc).__name__}: {exc}'
                try:
                    from tools29.anchor29 import valid_reward_state29
                    candidate = bounded_call(lambda : runtime.reward_state(vec))
                    require(valid_reward_state29(candidate, totals['policy_transitions'] if totals else None), 'invalid actual yaw reward provenance')
                    reward_state = json_ready(candidate)
                except BaseException as exc:
                    error = (error or '') + f'; reward provenance failed: {type(exc).__name__}: {exc}'
                try:
                    if closer is None:
                        closer = runtime.close
                    (cleanup, close_error) = closer(vec)
                    if close_error:
                        error = (error or '') + '; ' + close_error
                except BaseException as exc:
                    (cleanup, error) = (False, (error or '') + f'; worker cleanup {type(exc).__name__}: {exc}')
            elif spawned:
                try:
                    (cleanup, close_error) = runtime.close(None)
                    if close_error:
                        error = (error or '') + '; ' + close_error
                except BaseException as exc:
                    (cleanup, error) = (False, (error or '') + f'; construction cleanup {exc}')
            try:
                if runtime is not None:
                    rpc = json_ready(runtime.snapshot(vec))
                    if spawned and vec is None:
                        rpc.update(state='CONSTRUCTION_FAILED', ownership_unknown=True)
                    if rpc.get('dirty') is True:
                        error = (error or '') + '; in-process metadata state dirty'
            except BaseException as exc:
                rpc = {'status': 'unknown', 'error': str(exc)}
                (cleanup, error) = (False, (error or '') + '; guard diagnostic unavailable')
            try:
                unchanged_inputs()
                unchanged = True
            except BaseException as exc:
                error = (error or '') + f'; inputs changed: {exc}'
            final = getattr(model, 'num_timesteps', None)
            new = final - INITIAL if type(final) is int and final >= INITIAL else None
            anchor = None
            if model is not None:
                try:
                    anchor = json_ready(model.anchor_receipt29())
                    require(valid_anchor_receipt29(anchor, common.ARMS[arm], status == 'COMPLETE'), 'invalid anchor29 receipt')
                except BaseException as exc:
                    error = (error or '') + '; anchor29 receipt failed: ' + str(exc)
            guard_complete = rpc.get('backend') == 'DummyVecEnv' and rpc.get('interprocess_rpc') is False and (type(rpc.get('environment_subprocesses')) is int) and (rpc['environment_subprocesses'] == 0) and (rpc.get('dirty') is False) and (rpc.get('closed') is True) and (rpc.get('waiting') is False) and (rpc.get('inflight') is None) and (rpc.get('construction_complete') is True)
            if status == 'COMPLETE' and (error or not cleanup or (not unchanged) or (not guard_complete) or (not complete_counts(workers, 1, TRANSITIONS)) or (not watcher) or (watcher.rollouts * 4096 != TRANSITIONS) or (not valid_delay_state(delays, 1, mode)) or (getattr(model, 'target_kl', None) != TARGET_KL) or (not valid_reward_state29(reward_state, TRANSITIONS)) or (watcher.first_rollout_nonreward is None) or (not valid_anchor_receipt29(anchor, common.ARMS[arm], True)) or (getattr(model, 'anchor_coef', None) != common.ARMS[arm])):
                status = 'INCOMPLETE'
            if termination['received']:
                status = termination['received']
            if clock() - started >= spec['max_wall_s']:
                (status, error) = ('WALL_LIMIT', (error or '') + '; final wall limit')
            if model is not None:
                saved = output / ('policy.zip' if status == 'COMPLETE' else 'partial_policy.zip')
                try:
                    model.save(str(saved))
                    unchanged_inputs()
                    checkpoint_sha = sha(saved)
                except BaseException as exc:
                    (status, unchanged) = ('ERROR', False)
                    error = (error or '') + f'; checkpoint save/identity failed: {exc}'
                if clock() - started >= spec['max_wall_s']:
                    (status, error) = ('WALL_LIMIT', (error or '') + '; checkpoint final wall limit')
                if status != 'COMPLETE' and (output / 'policy.zip').exists():
                    (output / 'policy.zip').replace(output / 'partial_policy.zip')
            known = totals
            if known is None:
                known = common.expected_counts(0)
                try:
                    row = common.read_json(output / 'live_status.json')
                    candidate = checked_worker_counts(row['per_worker_counts'])
                    require(complete_counts(row['per_worker_counts'], 1, row['transitions']) and row['actual_counts'] == candidate, 'invalid completed prefix')
                    known = candidate
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            result = {'format': 'd1-anchor29-training-result-v1', 'plan_sha256': plan_sha, 'protocol_sha256': sha(output / 'protocol.json'), 'stage': stage, 'arm': arm, 'mode': mode, 'seed': common.SEED, 'envs': 1, 'aux_coef': 0.0, 'status': status, 'source_identity': plan['source_identity'], 'versions': protocol['versions'], 'hyperparameters': HYPERPARAMETERS, 'base_checkpoint_sha256': baseline['sha256'], 'new_trained_transitions': new, 'initial_num_timesteps': initial['num_timesteps'] if initial else None, 'final_num_timesteps': final, 'initial_state': initial, 'actual_counts': totals, 'known_actual_counts': known, 'worker_counts': workers, 'per_worker_counts': workers, 'delay_state18': delays, 'single_process_guard29': rpc, 'yaw_tracking_weight': 2.0, 'anchor_change': common.anchor_change(arm), 'reward_state29': reward_state, 'first_rollout_nonreward': watcher.first_rollout_nonreward if watcher else None, 'first_rollout_report_sha256': sha(output / 'first_rollout29.json') if (output / 'first_rollout29.json').is_file() else None, 'inputs_unchanged': unchanged, 'cleanup_complete': cleanup, 'error': error, 'checkpoint_sha256': checkpoint_sha, 'eligible_for_evaluation': status == 'COMPLETE', 'elapsed_total_s': clock() - started, 'stop_sample': watcher.stop_sample if watcher else startup_stop, 'min_available_ram_bytes': watcher.min_available if watcher else None, 'min_effective_available_bytes': watcher.min_effective if watcher else None, 'checkpoint_selection': 'final_only', 'preflight_weights_used_for_training': False, 'single_variable_effect_against22_claimed': False, 'single_variable_effect_against24_claimed': False, 'vector_layout': plan['vector_layout'], 'runtime_dependencies_sha256': plan['runtime_dependencies_sha256'], 'anchor_coef': common.ARMS[arm], 'teacher_policy_used': common.ARMS[arm] != 0.0, 'anchor29': anchor}
            atomic_json(output / 'result.json', result)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(common.ARMS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan, args.arm, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'REFUSED_BEFORE_SPAWN', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'COMPLETE' else 1
if __name__ == '__main__':
    raise SystemExit(main())
