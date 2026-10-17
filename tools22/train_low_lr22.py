"""One independent low-LR continuation; frozen attempts are never resumed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools22 import common22 as common  # noqa: E402
from tools18.train_delay18 import (  # noqa: E402
    HYPERPARAMETERS as ORIGINAL_HYPERPARAMETERS, RealRuntime as OriginalRuntime,
    optimizer_sha, valid_delay_state, validate_loaded_model,
)
from tools16.train_comparison16 import (  # noqa: E402
    StopRequested, Watchdog, bounded_call, complete_counts, count_totals, termination_scope,
)

INITIAL, TRANSITIONS = common.INITIAL, common.TRANSITIONS
HYPERPARAMETERS = common.HYPERPARAMETERS
require, sha, atomic_json = common.require, common.sha, common.atomic_json


def load_low_lr_model22(runtime, vec, plan, hyper, output):
    """Restore the original Adam, then change only its LR and schedule."""
    from tools22.checked_runtime22 import optimizer_without_lr_sha22
    require(hyper == HYPERPARAMETERS, 'new22 hyperparameters differ')
    model = OriginalRuntime.create_model(runtime, vec, plan, ORIGINAL_HYPERPARAMETERS, output)
    initial = dict(runtime.initial_identity)
    optimizer = model.policy.optimizer
    before, without_before = optimizer_sha(optimizer), optimizer_without_lr_sha22(optimizer)
    require(initial['optimizer_state_sha256'] == before and initial['num_timesteps'] == INITIAL,
            'restored baseline optimizer identity differs')
    require(model.learning_rate == 3e-4 and model.lr_schedule(0.) == model.lr_schedule(1.) == 3e-4
            and bool(optimizer.param_groups) and all(group['lr'] == 3e-4 for group in optimizer.param_groups),
            'baseline learning rate differs')
    model.learning_rate = hyper['learning_rate']
    model._setup_lr_schedule()
    for group in optimizer.param_groups:
        group['lr'] = hyper['learning_rate']
    validate_loaded_model(model, hyper, INITIAL)
    after, without_after = optimizer_sha(optimizer), optimizer_without_lr_sha22(optimizer)
    require(before != after and without_before == without_after
            and runtime.policy_sha(model) == initial['policy_state_sha256']
            and model.lr_schedule(0.) == model.lr_schedule(1.) == 1e-4
            and all(group['lr'] == 1e-4 for group in optimizer.param_groups),
            'LR change altered the policy/Adam state or schedule')
    initial.update(optimizer_state_sha256_before_lr=before, optimizer_state_sha256_after_lr=after,
        optimizer_without_lr_sha256_before=without_before, optimizer_without_lr_sha256_after=without_after,
        optimizer_momentum_preserved=True, learning_rate_before=3e-4, learning_rate_after=1e-4,
        lr_schedule_endpoints=[model.lr_schedule(0.), model.lr_schedule(1.)])
    runtime.initial_identity = initial
    return model


class Watchdog22(Watchdog):
    """Keep frozen rollout accounting; apply the new honest 4-GiB label."""
    def poll(self, *, force=False, phase='training'):
        elapsed = self.clock()-self.started
        if force or elapsed-self.last_poll >= self.release['resources']['poll_s']:
            sample = dict(self.memory_reader(), elapsed_s=elapsed, phase=phase)
            require(all(type(sample.get(key)) is int and sample[key] >= 0
                        for key in ('available_ram_bytes', 'effective_available_bytes')), 'invalid RAM accounting')
            self.last_poll, self.last_sample = elapsed, sample
            self.min_available = sample['available_ram_bytes'] if self.min_available is None else min(self.min_available, sample['available_ram_bytes'])
            self.min_effective = sample['effective_available_bytes'] if self.min_effective is None else min(self.min_effective, sample['effective_available_bytes'])
            with (self.output/'resource_samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            if sample['effective_available_bytes'] < self.release['resources']['reserve_bytes'] and self.reason is None:
                self.reason, self.stop_sample = 'RAM_RESERVE_BELOW_4_GIB', sample
        if elapsed >= self.release['stages'][self.stage]['max_wall_s'] and self.reason is None:
            self.reason, self.stop_sample = 'WALL_LIMIT', self.last_sample
        if self.reason:
            atomic_json(self.output/'stop_sample.json', dict(self.stop_sample or {}, reason=self.reason))
        return self.reason is None


def json_ready(value):
    json.dumps(value, allow_nan=False)
    return value


def checked_worker_counts(rows):
    totals = count_totals(rows)
    require(len(rows) == 4, 'wrong counter worker count')
    for row in rows:
        p, n, d = (row[key] for key in ('policy_transitions', 'native_steps', 'pd_updates'))
        require(p <= TRANSITIONS//4 and max(0, p-1)*10 <= n <= p*10
                and max(0, p-1)*5 <= d <= p*5 and 0 <= 2*d-n <= 2, 'counter physics/cap differs')
    json_ready(rows)
    return totals


def initial_identity_valid(initial):
    hashes = ('policy_state_sha256', 'optimizer_state_sha256', 'optimizer_state_sha256_before_lr',
              'optimizer_state_sha256_after_lr', 'optimizer_without_lr_sha256_before', 'optimizer_without_lr_sha256_after')
    return (isinstance(initial, dict) and initial.get('num_timesteps') == INITIAL
        and initial.get('optimizer_restored') is True and initial.get('optimizer_momentum_preserved') is True
        and all(isinstance(initial.get(key), str) and len(initial[key]) == 64
                and set(initial[key]) <= set('0123456789abcdef') for key in hashes)
        and initial['optimizer_state_sha256'] == initial['optimizer_state_sha256_before_lr']
        and initial['optimizer_state_sha256_before_lr'] != initial['optimizer_state_sha256_after_lr']
        and initial['optimizer_without_lr_sha256_before'] == initial['optimizer_without_lr_sha256_after']
        and initial.get('learning_rate_before') == 3e-4 and initial.get('learning_rate_after') == 1e-4
        and initial.get('lr_schedule_endpoints') == [1e-4, 1e-4])


def memory_ok(sample, minimum):
    value = sample.get('effective_available_bytes')
    require(type(value) is int and value >= 0, 'effective RAM unknown')
    return value >= minimum


def run(plan_path, output, *, runtime=None, loader=common.load_plan, launch_validator=common.validate_launch,
        memory_reader=common.read_memory, clock=time.monotonic, root=ROOT, closer=None):
    started = clock()
    root, plan_path, output = Path(root).resolve(), Path(plan_path).resolve(), Path(output).absolute()
    plan = loader(plan_path)
    spec, stage, mode = plan['stages']['train'], 'train', 'zero'
    require(spec['mode'] == mode and spec['transitions'] == TRANSITIONS and plan['envs'] == 4 and plan['seed'] == 181
            and plan.get('hyperparameters') == HYPERPARAMETERS, 'new22 training scope differs')
    require(output == root/plan['output_root']/stage and not output.exists() and output.is_relative_to(root),
            'exclusive fixed22 training output required')
    cursor = output
    while cursor != root:
        require(not cursor.is_symlink(), 'symlink training output forbidden')
        cursor = cursor.parent
    launch_validator(plan, plan_path, stage, output)
    baseline, base_protocol = plan['base_checkpoint'], plan['base_protocol']
    checkpoint, protocol_path = Path(baseline['path']), Path(base_protocol['path'])
    checkpoint = checkpoint if checkpoint.is_absolute() else root/checkpoint
    protocol_path = protocol_path if protocol_path.is_absolute() else root/protocol_path
    require(sha(checkpoint) == baseline['sha256'] and baseline['trained_transitions'] == INITIAL
            and sha(protocol_path) == base_protocol['sha256'], 'baseline identity changed')
    original = common.read_json(protocol_path)
    require(original['hyperparameters'] == ORIGINAL_HYPERPARAMETERS, 'original PPO settings changed')
    require(memory_ok(memory_reader(), plan['resources']['start_min_effective_bytes']), 'NOT_READY: insufficient start RAM')
    plan_sha = sha(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'format': 'd1-low-lr22-training-protocol-v1', 'plan_sha256': plan_sha, 'stage': stage, 'mode': mode,
        'seed': plan['seed'], 'envs': plan['envs'], 'planned_new_transitions': TRANSITIONS,
        'planned_native_steps': TRANSITIONS*10, 'base_checkpoint': baseline, 'base_protocol': base_protocol,
        'source_identity': plan['source_identity'], 'hyperparameters': HYPERPARAMETERS,
        'baseline_hyperparameters': ORIGINAL_HYPERPARAMETERS, 'versions': None, 'warm_start': True,
        'teacher_policy_used': False, 'reset_num_timesteps': False, 'checkpoint_selection': 'final_only',
        'delay_sampling': 'zero; inherited stream18 and FIFO; original paired commands and rewards',
        'lr_change': {'before': 3e-4, 'after': 1e-4, 'schedule': 'constant', 'all_optimizer_param_groups': True},
        'lr_causal_effect_identifiable_against18': False}
    atomic_json(output/'protocol.json', protocol)
    vec = model = watcher = workers = totals = delays = initial = None
    rpc = {'status': 'not_created'}
    status, error, cleanup, unchanged, spawned = 'ERROR', None, True, False, False
    checkpoint_sha = startup_stop = None

    def unchanged_inputs():
        loader(plan_path)
        require(sha(plan_path) == plan_sha and sha(checkpoint) == baseline['sha256']
                and sha(protocol_path) == base_protocol['sha256'], 'plan or baseline changed')

    with termination_scope() as termination:
        try:
            if runtime is None:
                from tools22.checked_runtime22 import CheckedRuntime22
                runtime = CheckedRuntime22(model_factory=load_low_lr_model22)
            runtime.prepare()
            protocol['versions'] = json_ready(runtime.versions)
            atomic_json(output/'protocol.json', protocol)
            require(runtime.versions == original['versions'], 'runtime versions differ from original completed baseline')
            unchanged_inputs()
            require(memory_ok(memory_reader(), plan['resources']['start_min_effective_bytes']), 'NOT_READY: RAM after imports')
            require(clock()-started < spec['max_wall_s'], 'WALL_LIMIT before spawn')
            termination['defer'], spawned = True, True
            try:
                vec = runtime.create_vec(plan, TRANSITIONS, output, root, mode)
            finally:
                termination['defer'] = False
            if termination['received']:
                raise StopRequested(termination['received'])
            sample = dict(memory_reader(), elapsed_s=clock()-started, phase='after_spawn')
            if not memory_ok(sample, plan['resources']['reserve_bytes']):
                startup_stop = dict(sample, reason='RAM_RESERVE_BELOW_4_GIB')
                atomic_json(output/'stop_sample.json', startup_stop)
                raise StopRequested(startup_stop['reason'])
            model = runtime.create_model(vec, plan, HYPERPARAMETERS, output)
            candidate = runtime.initial_identity
            require(initial_identity_valid(candidate), 'original optimizer/LR transition proof incomplete')
            initial = json_ready(candidate)
            protocol['initial_state'] = initial
            atomic_json(output/'protocol.json', protocol)
            watcher = Watchdog22(output, plan, stage, started, runtime, vec, model, memory_reader, clock)
            runtime.learn(model, TRANSITIONS, watcher)
            watcher.poll(force=True, phase='after_learn')
            status = watcher.reason or ('COMPLETE' if model.num_timesteps == INITIAL+TRANSITIONS else 'PARTIAL')
        except BaseException as exc:
            status = watcher.reason if watcher and watcher.reason else str(exc) if isinstance(exc, StopRequested) else 'ERROR'
            error = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
        finally:
            termination['defer'] = True
            for number in (signal.SIGINT, signal.SIGTERM):
                signal.signal(number, signal.SIG_IGN)
            if vec is not None:
                try:
                    candidate = bounded_call(lambda: runtime.counters(vec))
                    candidate_totals = checked_worker_counts(candidate)
                    workers, totals = candidate, candidate_totals
                except BaseException as exc:
                    error = (error or '')+f'; counter collection failed: {type(exc).__name__}: {exc}'
                try:
                    candidate = bounded_call(lambda: runtime.delay_state(vec))
                    require(valid_delay_state(candidate, 4, mode), 'invalid delay provenance')
                    delays = json_ready(candidate)
                except BaseException as exc:
                    error = (error or '')+f'; delay provenance failed: {type(exc).__name__}: {exc}'
                try:
                    if closer is None:
                        from tools22.checked_runtime22 import close_checked22
                        closer = close_checked22
                    cleanup, close_error = closer(vec)
                    if close_error:
                        error = (error or '')+'; '+close_error
                except BaseException as exc:
                    cleanup, error = False, (error or '')+f'; worker cleanup {type(exc).__name__}: {exc}'
            elif spawned:
                cleanup = False
            try:
                if runtime is not None:
                    rpc = json_ready(runtime.rpc_snapshot(vec))
                    if spawned and vec is None:
                        rpc.update(state='CONSTRUCTION_FAILED', ownership_unknown=True)
                    if rpc.get('dirty') is True:
                        error = (error or '')+'; RPC channel dirty'
            except BaseException as exc:
                rpc = {'status': 'unknown', 'error': str(exc)}
                cleanup, error = False, (error or '')+'; guard diagnostic unavailable'
            try:
                unchanged_inputs()
                unchanged = True
            except BaseException as exc:
                error = (error or '')+f'; inputs changed: {exc}'
            final = getattr(model, 'num_timesteps', None)
            new = final-INITIAL if type(final) is int and final >= INITIAL else None
            if status == 'COMPLETE' and (error or not cleanup or not unchanged
                    or not complete_counts(workers, 4, TRANSITIONS) or not watcher or watcher.rollouts*4096 != TRANSITIONS
                    or not valid_delay_state(delays, 4, mode)):
                status = 'INCOMPLETE'
            if termination['received']:
                status = termination['received']
            if clock()-started >= spec['max_wall_s']:
                status, error = 'WALL_LIMIT', (error or '')+'; final wall limit'
            if model is not None:
                saved = output/('policy.zip' if status == 'COMPLETE' else 'partial_policy.zip')
                try:
                    model.save(str(saved))
                    unchanged_inputs()
                    checkpoint_sha = sha(saved)
                except BaseException as exc:
                    status, unchanged = 'ERROR', False
                    error = (error or '')+f'; checkpoint save/identity failed: {exc}'
                if clock()-started >= spec['max_wall_s']:
                    status, error = 'WALL_LIMIT', (error or '')+'; checkpoint final wall limit'
                if status != 'COMPLETE' and (output/'policy.zip').exists():
                    (output/'policy.zip').replace(output/'partial_policy.zip')
            known = totals
            if known is None:
                known = common.expected_counts(0)
                try:
                    row = common.read_json(output/'live_status.json')
                    candidate = checked_worker_counts(row['per_worker_counts'])
                    require(complete_counts(row['per_worker_counts'], 4, row['transitions'])
                            and row['actual_counts'] == candidate, 'invalid completed prefix')
                    known = candidate
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            result = {'format': 'd1-low-lr22-training-result-v1', 'plan_sha256': plan_sha,
                'protocol_sha256': sha(output/'protocol.json'), 'stage': stage, 'mode': mode, 'seed': 181, 'envs': 4,
                'aux_coef': 0., 'status': status, 'source_identity': plan['source_identity'], 'versions': protocol['versions'],
                'hyperparameters': HYPERPARAMETERS, 'base_checkpoint_sha256': baseline['sha256'],
                'new_trained_transitions': new, 'initial_num_timesteps': initial['num_timesteps'] if initial else None,
                'final_num_timesteps': final, 'initial_state': initial, 'actual_counts': totals, 'known_actual_counts': known,
                'worker_counts': workers, 'per_worker_counts': workers, 'delay_state18': delays, 'rpc_guard22': rpc,
                'inputs_unchanged': unchanged, 'cleanup_complete': cleanup, 'error': error, 'checkpoint_sha256': checkpoint_sha,
                'eligible_for_evaluation': status == 'COMPLETE', 'elapsed_total_s': clock()-started,
                'stop_sample': watcher.stop_sample if watcher else startup_stop,
                'min_available_ram_bytes': watcher.min_available if watcher else None,
                'min_effective_available_bytes': watcher.min_effective if watcher else None,
                'checkpoint_selection': 'final_only', 'preflight_weights_used_for_training': False,
                'lr_causal_effect_identifiable_against18': False}
            atomic_json(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.plan, args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'REFUSED_BEFORE_SPAWN', 'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
