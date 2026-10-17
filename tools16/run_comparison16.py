#!/usr/bin/env python3
"""Plan or supervise one authorised, sequential local comparison. No automatic retries."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time


ORDER = ('preflight_control', 'preflight_velocity', 'train_control', 'train_velocity',
         'eval_control', 'eval_velocity', 'compare')


def contract_api():
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module('contract16')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('receipt must be a JSON object')
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def commands(release, release_path, root):
    root, release_path = Path(root).resolve(), Path(release_path).resolve()
    output = root / release['output_root']
    python = [sys.executable, '-B']
    result = []
    for name in ORDER:
        if name == 'compare':
            target = output / 'comparison.json'
            argv = [*python, str(root / 'tools16/compare_results16.py'), '--release', str(release_path),
                    '--control', str(output / 'eval_control/report.json'),
                    '--velocity', str(output / 'eval_velocity/report.json'), '--output', str(target)]
            maximum = release['stages']['compare']['max_wall_s'] + release['resources']['process_wall_grace_s']
        else:
            stage, arm = name.split('_', 1)
            target = output / name
            argv = [*python, str(root / ('tools16/evaluate_comparison16.py' if stage == 'eval'
                                        else 'tools16/train_comparison16.py')),
                    '--release', str(release_path), '--arm', arm]
            if stage == 'eval':
                argv += ['--checkpoint', str(output / ('train_' + arm) / 'policy.zip')]
            else:
                argv += ['--stage', stage]
            argv += ['--output', str(target)]
            maximum = release['stages'][stage]['max_wall_s'] + release['resources']['process_wall_grace_s']
        result.append({'name': name, 'argv': argv, 'output': str(target), 'process_wall_limit_s': maximum})
    return result


def validate_stage_launch(release, release_path, stage_name, output, root=None):
    """A child must belong to the runner's currently claimed stage, before heavy imports."""
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    release_path = Path(release_path).resolve()
    specs = {item['name']: item for item in commands(release, release_path, root)}
    if stage_name not in specs or Path(output).resolve() != Path(specs[stage_name]['output']):
        raise ValueError('stage output is outside the fixed comparison order')
    report = read_json(root / release['output_root'] / 'report.json')
    current = report.get('stages', {}).get(stage_name, {})
    token = os.environ.get('D1_COMPARISON16_LAUNCH_TOKEN')
    if (report.get('format') != 'd1-comparison16-execution-v1' or report.get('status') != 'RUNNING'
            or report.get('release_sha256') != sha(release_path)
            or report.get('runner_sha256') != sha(__file__)
            or type(report.get('supervisor_pid')) is not int or report['supervisor_pid'] != os.getppid()
            or report.get('active_stage') != stage_name or current.get('status') != 'STARTED'
            or current.get('command') != specs[stage_name]['argv']
            or os.environ.get('D1_COMPARISON16_STAGE') != stage_name
            or not token or current.get('launch_token') != token
            or any(report.get('stages', {}).get(name, {}).get('status') != 'COMPLETE'
                   for name in ORDER[:ORDER.index(stage_name)])):
        raise ValueError('stage requires its active runner journal and launch environment')


def plan(release_path, api=None):
    api = api or contract_api()
    release_path = Path(release_path).resolve()
    release = api.load_release(release_path, require_authorized=False)
    api.validate_sources(release)
    return {'format': 'd1-comparison16-plan-v1', 'status': 'PLAN_ONLY',
            'release_status': release['status'], 'release_sha256': sha(release_path),
            'execution_authorized': release['status'] == 'AUTHORIZED_FROZEN',
            'training_started': False, 'child_processes_started': 0,
            'physics_steps_executed_by_supervisor': 0,
            'output_root': str(Path(api.ROOT) / release['output_root']),
            'stages': commands(release, release_path, api.ROOT)}


class Interrupted(Exception):
    pass


class ResourceReadError(Exception):
    def __init__(self, sample):
        super().__init__(sample['error'])
        self.sample = sample


@contextmanager
def termination_guard():
    state = {'defer': False, 'received': None}
    def handler(signum, frame):
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        state['received'] = signal.Signals(signum).name
        if not state['defer']:
            raise Interrupted(state['received'])
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, handler)
        yield state
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def live_group_members(pgid):
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def cleanup(process, resources, natural_exit=False):
    """Only the new session created for this Popen is eligible for signalling."""
    sent = []
    def alive():
        return process.poll() is None or bool(live_group_members(process.pid))
    if natural_exit:
        deadline = time.monotonic() + resources['kill_grace_s']
        while alive() and time.monotonic() < deadline:
            time.sleep(.02)
    for sig, grace in ((signal.SIGINT, resources['interrupt_grace_s']),
                       (signal.SIGTERM, resources['terminate_grace_s']),
                       (signal.SIGKILL, resources['kill_grace_s'])):
        process.poll()
        if not alive():
            break
        try:
            if sig == signal.SIGINT and process.returncode is None:
                process.send_signal(sig)  # Give the trainer its counter/worker-close path first.
            else:
                os.killpg(process.pid, sig)
            sent.append(sig.name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while alive() and time.monotonic() < deadline:
            process.poll()
            time.sleep(.02)
    process.poll()
    return sent, not alive()


def memory_sample(api, name, phase, started):
    value = dict(api.read_memory())
    host, capped, effective = (value.get(key) for key in
                               ('available_ram_bytes', 'cgroup_remaining_bytes', 'effective_available_bytes'))
    if (type(host) is not int or host < 0 or type(effective) is not int or effective < 0
            or (capped is not None and (type(capped) is not int or capped < 0))
            or effective != (min(host, capped) if capped is not None else host)):
        raise ValueError('invalid host/cgroup/effective resource sample')
    return dict(value, stage=name, phase=phase, elapsed_s=time.monotonic() - started, sample_valid=True)


def expected_counts(transitions):
    return {'policy_transitions': transitions, 'native_steps': transitions * 10, 'pd_updates': transitions * 5}


def checked_counts(value, maximum, exact=False):
    if (not isinstance(value, dict) or not set(expected_counts(0)) <= set(value)
            or any(type(item) is not int or item < 0 for item in value.values())):
        return False
    transitions = value['policy_transitions']
    return (transitions == maximum if exact else transitions <= maximum) and all(
        value[key] == count for key, count in expected_counts(transitions).items())


def partial_counts(value, maximum):
    """An interrupted interval can have real native/PD attempts below the full ratio."""
    return (isinstance(value, dict) and set(expected_counts(0)) <= set(value)
            and all(type(item) is int and item >= 0 for item in value.values())
            and all(type(value[key]) is int and 0 <= value[key] <= limit
                    for key, limit in expected_counts(maximum).items())
            and value['native_steps'] <= value['policy_transitions'] * 10
            and value['pd_updates'] <= value['policy_transitions'] * 5)


def ledger_rows(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            break  # A child may currently be writing its last line.
        row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite ledger')))
        if not isinstance(row, dict) or not isinstance(row.get('deterministic'), dict):
            raise ValueError('ledger lacks the deterministic comparison fields')
        rows.append(row)
    return rows


def compare_prefix(preflight, train):
    reference, actual = ledger_rows(preflight), ledger_rows(train)
    if len(reference) != 8:
        raise ValueError('preflight must contain exactly eight complete rollout rows')
    compared = 0
    for before, after in zip(reference, actual[:8]):
        if before['deterministic'] != after['deterministic']:
            fields = set(before['deterministic']) | set(after['deterministic'])
            return compared, {'rollout': compared + 1, 'differing_fields': sorted(
                key for key in fields if before['deterministic'].get(key) != after['deterministic'].get(key))}
        compared += 1
    return compared, None


def minimum_receipt_identity(name, child, release, release_sha, output):
    """Partial counters are usable only when bound to this stage and its protocol bytes."""
    if not isinstance(child, dict) or name == 'compare':
        return False
    stage, arm = name.split('_', 1)
    try:
        protocol_sha = sha(Path(output) / 'protocol.json')
    except OSError:
        return False
    return (child.get('format') == ('d1-comparison16-evaluation-result-v1' if stage == 'eval'
                                   else 'd1-comparison16-training-result-v1')
            and child.get('release_sha256') == release_sha and child.get('arm') == arm
            and (stage == 'eval' or child.get('stage') == stage)
            and child.get('seed') == release['seed'] and child.get('envs') == release['envs']
            and child.get('aux_coef') == release['arms'][arm]['aux_coef']
            and child.get('protocol_sha256') == protocol_sha)


def validate_completion(spec, child, release, release_sha, execution):
    name = spec['name']
    if child.get('status') != 'COMPLETE' or child.get('release_sha256') != release_sha:
        raise ValueError('child did not complete under this release')
    if name == 'compare':
        if (child.get('format') != 'd1-comparison16-comparison-v1'
                or child.get('inputs_sha256') != {arm: sha(execution / ('eval_' + arm) / 'report.json')
                                                  for arm in ('control', 'velocity')}):
            raise ValueError('wrong comparison receipt format')
        return
    stage, arm = name.split('_', 1)
    output = Path(spec['output'])
    if (not minimum_receipt_identity(name, child, release, release_sha, output)
            or child.get('inputs_unchanged') is not True or child.get('cleanup_complete') is not True
            or child.get('error')):
        raise ValueError('child identity, protocol, source or cleanup receipt differs')
    transitions = release['stages'][stage]['transitions']
    if not checked_counts(child.get('actual_counts'), transitions, exact=stage != 'eval'):
        raise ValueError('child actual counts are incomplete or inconsistent')
    if stage == 'eval':
        directory = str(Path(__file__).resolve().parent)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        evaluator = importlib.import_module('evaluate_comparison16')
        train = execution / ('train_' + arm)
        cases = child.get('cases')
        if (child.get('format') != 'd1-comparison16-evaluation-result-v1'
                or child.get('checkpoint_sha256') != sha(train / 'policy.zip')
                or child.get('training_protocol_sha256') != sha(train / 'protocol.json')
                or child.get('training_result_sha256') != sha(train / 'result.json')
                or not isinstance(cases, list) or len(cases) != 3 or not all(isinstance(item, dict) for item in cases)
                or [item.get('sequence') for item in cases] != list(evaluator.TASKS)
                or not all(evaluator.valid_case_completion(item) for item in cases)
                or any(sum(item['actual_counts'][key] for item in cases) != child['actual_counts'][key]
                       for key in expected_counts(0))
                or child.get('full_horizons_complete') is not (
                    child['actual_counts']['policy_transitions'] == transitions
                    and all(item['done_reason'] == 'HORIZON_12S' for item in cases))):
            raise ValueError('evaluation does not bind the completed training and three cases')
        return
    workers = child.get('per_worker_counts')
    counts = child['actual_counts']
    if (child.get('format') != 'd1-comparison16-training-result-v1' or child.get('stage') != stage
            or child.get('trained_transitions') != transitions
            or child.get('eligible_for_evaluation') is not (stage == 'train')
            or child.get('checkpoint_sha256') != sha(output / 'policy.zip')
            or not isinstance(workers, list) or len(workers) != release['envs']
            or not all(checked_counts(worker, transitions // release['envs'], exact=True) for worker in workers)
            or not all(type(value) is int and value >= 0 and all(set(worker) == set(counts) for worker in workers)
                       and sum(worker[key] for worker in workers) == value for key, value in counts.items())):
        raise ValueError('training result, final checkpoint or per-worker counts are incomplete')
    rows = ledger_rows(output / 'ledger.jsonl')
    if len(rows) != transitions // 4096 or any(
            row.get('rollout') != index or row.get('transitions') != index * 4096
            or row['deterministic'].get('rollout') != index
            or row['deterministic'].get('transitions') != index * 4096 for index, row in enumerate(rows, 1)):
        raise ValueError('training ledger does not contain the exact complete rollouts')
    if name == 'preflight_velocity':
        control = ledger_rows(execution / 'preflight_control' / 'ledger.jsonl')
        if not control or control[0]['deterministic'] != rows[0]['deterministic']:
            raise ValueError('paired arms differ before the first PPO update')
    if stage == 'train':
        compared, mismatch = compare_prefix(execution / ('preflight_' + arm) / 'ledger.jsonl', output / 'ledger.jsonl')
        if compared != 8 or mismatch:
            raise ValueError('formal training does not match its own preflight prefix')


def receipt_path(spec):
    output = Path(spec['output'])
    return output if spec['name'] == 'compare' else output / ('report.json' if spec['name'].startswith('eval_') else 'result.json')


def artifacts(spec):
    if spec['name'] == 'compare':
        paths = [Path(spec['output'])]
    else:
        output = Path(spec['output'])
        paths = [receipt_path(spec), output / 'protocol.json']
        if not spec['name'].startswith('eval_'):
            paths += [output / 'policy.zip', output / 'ledger.jsonl']
    return {str(path): sha(path) for path in paths}


def run_stage(spec, release, release_path, release_sha, execution, api, observe, control, token, first_sample=None):
    resources = release['resources']
    record = {'status': 'STARTED', 'command': spec['argv'], 'launch_token': token, 'process_started': False,
              'returncode': None, 'stop_reason': None, 'stop_sample': None, 'error': None,
              'cleanup_signals': [], 'cleanup_complete': True, 'child_receipt': None,
              'rollouts_compared': 0, 'first_mismatch': None}
    process = None
    started = time.monotonic()
    log = execution / 'supervision' / spec['name']
    log.mkdir(parents=True, exist_ok=False)
    try:
        sample = first_sample if first_sample is not None else observe(spec['name'], 'before_spawn')
        if sample['effective_available_bytes'] < resources['start_min_effective_bytes']:
            record.update(stop_reason='STAGE_START_RAM_HEADROOM', stop_sample=sample)
        elif Path(spec['output']).exists() or Path(spec['output']).is_symlink():
            record['stop_reason'] = 'STAGE_OUTPUT_ALREADY_EXISTS'
        else:
            env = dict(os.environ, PYTHONPATH=f"{api.ROOT / 'training'}:{api.ROOT / '.python-deps'}",
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                       D1_COMPARISON16_STAGE=spec['name'], D1_COMPARISON16_LAUNCH_TOKEN=token)
            with (log / 'stdout.txt').open('x') as stdout, (log / 'stderr.txt').open('x') as stderr:
                control['defer'] = True
                try:
                    process = subprocess.Popen(spec['argv'], cwd=api.ROOT, env=env, stdout=stdout,
                                               stderr=stderr, start_new_session=True)
                    record['process_started'] = True
                finally:
                    control['defer'] = False
                if control['received']:
                    raise Interrupted(control['received'])
                while process.poll() is None:
                    sample = observe(spec['name'], 'running')
                    if sample['effective_available_bytes'] < resources['reserve_bytes']:
                        record.update(stop_reason='RAM_RESERVE_BELOW_3_GIB', stop_sample=sample)
                        break
                    if time.monotonic() - started >= spec['process_wall_limit_s']:
                        record.update(stop_reason='PROCESS_WALL_LIMIT', stop_sample=sample)
                        break
                    if spec['name'].startswith('train_'):
                        arm = spec['name'].split('_', 1)[1]
                        compared, mismatch = compare_prefix(execution / ('preflight_' + arm) / 'ledger.jsonl',
                                                            Path(spec['output']) / 'ledger.jsonl')
                        record.update(rollouts_compared=compared, first_mismatch=mismatch)
                        if mismatch:
                            record['stop_reason'] = 'PREFLIGHT_PREFIX_MISMATCH'
                            break
                    time.sleep(resources['poll_s'])
    except BaseException as exc:
        if isinstance(exc, ResourceReadError):
            record.update(stop_reason='RESOURCE_READ_FAILED', stop_sample=exc.sample)
        record['stop_reason'] = record['stop_reason'] or ('SUPERVISOR_INTERRUPTED' if isinstance(exc, (Interrupted, KeyboardInterrupt)) else 'SUPERVISOR_ERROR')
        record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        # Repeated termination cannot interrupt ownership-limited bounded cleanup.
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                if process.poll() not in (None, 0):
                    record['stop_reason'] = record['stop_reason'] or 'CHILD_EXIT_NONZERO'
                try:
                    sent, clean = cleanup(process, resources, natural_exit=record['stop_reason'] is None)
                    record.update(cleanup_signals=sent, cleanup_complete=clean)
                    if not clean:
                        record['stop_reason'] = record['stop_reason'] or 'PROCESS_GROUP_CLEANUP_INCOMPLETE'
                    elif sent and record['stop_reason'] is None:
                        record['stop_reason'] = 'RESIDUAL_PROCESS_GROUP'
                except Exception as exc:
                    record.update(cleanup_complete=False, error=f'cleanup {type(exc).__name__}: {exc}')
                    record['stop_reason'] = record['stop_reason'] or 'PROCESS_GROUP_CLEANUP_INCOMPLETE'
                record['returncode'] = process.returncode
            try:
                observe(spec['name'], 'after_exit')
            except Exception as exc:
                record['stop_reason'] = record['stop_reason'] or 'RESOURCE_READ_FAILED'
                if isinstance(exc, ResourceReadError) and record['stop_sample'] is None:
                    record['stop_sample'] = exc.sample
                record['error'] = record['error'] or f'{type(exc).__name__}: {exc}'
        finally:
            for sig, old in previous.items():
                signal.signal(sig, old)
    try:
        record['child_receipt'] = read_json(receipt_path(spec))
    except (OSError, ValueError) as exc:
        record['receipt_error'] = f'{type(exc).__name__}: {exc}'
    if record['stop_reason'] is None:
        if record['returncode'] != 0:
            record['stop_reason'] = 'CHILD_EXIT_NONZERO'
        else:
            try:
                validate_completion(spec, record['child_receipt'] or {}, release, release_sha, execution)
                if spec['name'].startswith('train_'):
                    record['rollouts_compared'] = 8
                record['artifacts_sha256'] = artifacts(spec)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(stop_reason='CHILD_COMPLETION_INVALID', error=f'{type(exc).__name__}: {exc}')
    record['status'] = 'COMPLETE' if record['stop_reason'] is None else 'FAIL'
    record['wall_s'] = time.monotonic() - started
    return record


def execute(release_path, api=None):
    api = api or contract_api()
    release_path = Path(release_path).resolve()
    release = api.load_release(release_path, require_authorized=True)
    api.validate_sources(release)
    release_sha = sha(release_path)
    execution = Path(api.ROOT) / release['output_root']
    if execution.exists() or execution.is_symlink():
        raise FileExistsError('execution already claimed; no retry, resume or environment-count fallback')
    started = time.monotonic()
    try:
        initial = memory_sample(api, ORDER[0], 'before_spawn', started)
        refusal_error = None
    except Exception as exc:
        initial = None
        refusal_error = f'{type(exc).__name__}: {exc}'
    if initial is None or initial['effective_available_bytes'] < release['resources']['start_min_effective_bytes']:
        directory = execution.parent / 'readiness_checks'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'run_{time.time_ns()}_{os.getpid()}.json'
        value = {'format': 'd1-comparison16-readiness-v1', 'status': 'NOT_READY',
                 'release_sha256': release_sha, 'sample': initial, 'error': refusal_error,
                 'execution_claimed': False, 'child_processes_started': 0,
                 'training_started': False, 'physics_steps_executed_by_supervisor': 0}
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        return 2
    with termination_guard() as control:
        execution.mkdir(parents=True, exist_ok=False)
        report = {'format': 'd1-comparison16-execution-v1', 'status': 'RUNNING',
                  'release_sha256': release_sha, 'runner_sha256': sha(__file__), 'stage_order': list(ORDER),
                  'supervisor_pid': os.getpid(), 'active_stage': None,
                  'stages': {}, 'failed_stage': None, 'error': None,
                  'minimum_observed_effective_bytes': initial['effective_available_bytes'],
                  'resource_reservation_guaranteed': False, 'retry_or_resume_allowed': False,
                  'physics_steps_executed_by_supervisor': 0}
        path = execution / 'report.json'
        atomic_json(path, report)
        with (execution / 'resources.jsonl').open('x') as stream:
            stream.write(json.dumps(initial) + '\n')
            stream.flush()
            def observe(name, phase):
                try:
                    sample = memory_sample(api, name, phase, started)
                except Exception as exc:
                    failure = {'sample_valid': False, 'stage': name, 'phase': phase,
                               'elapsed_s': time.monotonic() - started, 'memory': None,
                               'error': f'{type(exc).__name__}: {exc}'}
                    stream.write(json.dumps(failure) + '\n')
                    stream.flush()
                    raise ResourceReadError(failure) from exc
                report['minimum_observed_effective_bytes'] = min(
                    report['minimum_observed_effective_bytes'], sample['effective_available_bytes'])
                stream.write(json.dumps(sample, allow_nan=False) + '\n')
                stream.flush()
                return sample

            def unchanged():
                if sha(release_path) != release_sha:
                    raise ValueError('frozen release changed during execution')
                current = api.load_release(release_path, require_authorized=True)
                api.validate_sources(current)
                for old in report['stages'].values():
                    for name, digest in old.get('artifacts_sha256', {}).items():
                        if sha(name) != digest:
                            raise ValueError('a completed stage artifact changed: ' + name)

            name = None
            try:
                for index, spec in enumerate(commands(release, release_path, api.ROOT)):
                    name = spec['name']
                    unchanged()
                    token = secrets.token_hex(16)
                    report['active_stage'] = name
                    report['stages'][name] = {'status': 'STARTED', 'command': spec['argv'], 'launch_token': token}
                    atomic_json(path, report)  # Durable claim before Popen.
                    record = run_stage(spec, release, release_path, release_sha, execution, api, observe,
                                       control, token, first_sample=initial if index == 0 else None)
                    report['stages'][name] = record
                    atomic_json(path, report)
                    if record['status'] != 'COMPLETE':
                        report.update(status='FAIL', failed_stage=name)
                        break
                    unchanged()
                else:
                    report['status'] = 'COMPLETE'
            except BaseException as exc:
                report.update(status='FAIL', failed_stage=name, error=f'{type(exc).__name__}: {exc}')
            finally:
                report['child_processes_started'] = sum(stage.get('process_started') is True
                                                        for stage in report['stages'].values())
                report['active_stage'] = None
                known = expected_counts(0)
                unknown = []
                for stage, record in report['stages'].items():
                    if stage == 'compare' or not record.get('process_started'):
                        continue
                    counts = (record.get('child_receipt') or {}).get('actual_counts')
                    maximum = release['stages'][stage.split('_', 1)[0]]['transitions']
                    if (minimum_receipt_identity(stage, record.get('child_receipt'), release, release_sha,
                                                 execution / stage) and partial_counts(counts, maximum)):
                        for key in known:
                            known[key] += counts[key]
                    else:
                        unknown.append(stage)
                report['known_actual_counts'] = known
                report['count_basis'] = 'child-reported counters; supervisor does not measure physical steps'
                report['unknown_count_stages'] = unknown
                report['actual_counts'] = None if unknown else known
                report['wall_s'] = time.monotonic() - started
                atomic_json(path, report)
    return 0 if report['status'] == 'COMPLETE' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='default: validate and list commands only')
    mode.add_argument('--execute', action='store_true', help='requires a separately authorised frozen release')
    args = parser.parse_args(argv)
    if args.execute:
        return execute(args.release)
    print(json.dumps(plan(args.release), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
