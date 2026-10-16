#!/usr/bin/env python3
"""Contract 10 supervision; importing this module starts neither models nor physics.

This sidecar observes resources, rather than reserving host RAM. It cannot recover a
transient sample seen only by the frozen trainer's callback. New runs require a new
release and exclusive contract-10 paths; this tool never resumes contract 09.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))
from run_scale08 import compare_rollout, complete_lines  # noqa: E402

RESERVE = 3*1024**3
RELEASE_FORMAT = 'd1-contract10-supervisor-release-v1'
REPORT_FORMAT = 'd1-contract10-supervisor-report-v1'
COMPARATOR = 'tools08/run_scale08.py'


class SupervisorTerminated(Exception):
    pass


@contextmanager
def handle_termination():
    control = {'defer': False, 'received': None}
    def interrupted(signum, _frame):
        for handled in (signal.SIGINT, signal.SIGTERM):
            signal.signal(handled, signal.SIG_IGN)  # Allow the bounded cleanup to finish.
        control['received'] = signal.Signals(signum).name
        if not control['defer']:
            raise SupervisorTerminated(control['received'])
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        yield control
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_memory():
    """Host MemAvailable and the tightest visible cgroup-v2 hard-limit headroom."""
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(meminfo['MemAvailable'].split()[0])*1024
    remaining = []
    scope = next((line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::')), None)
    if scope is not None:
        base = Path('/sys/fs/cgroup')
        current = base/scope.lstrip('/')
        if not current.is_dir():
            raise RuntimeError('the current cgroup-v2 scope is not visible')
        while current.is_relative_to(base):
            maximum = current/'memory.max'
            if maximum.exists():
                limit = maximum.read_text().strip()
                if limit != 'max':
                    remaining.append(max(0, int(limit)-int((current/'memory.current').read_text())))
            if current == base:
                break
            current = current.parent
    return {'available_ram_bytes': available,
            'cgroup_remaining_bytes': min(remaining) if remaining else None}


def load_release(path, root=ROOT):
    root = Path(root).resolve()
    release = json.loads(Path(path).read_text())
    if (release.get('format') != RELEASE_FORMAT or release.get('contract_id') != 10
            or release.get('status') != 'FROZEN_BEFORE_TRAINING'
            or release.get('supervisor_sha256') != sha(__file__)):
        raise SystemExit('a new, matching contract 10 frozen release is required')
    frozen = release['frozen_inputs']
    required = {release['entry'], release['config'], release['contract'], COMPARATOR,
                *(release['plans'][kind]['frozen_ledger'] for kind in ('preflight', 'scale'))}
    if not required.issubset(frozen) or release['contract'] != 'docs/career_execution_contract_10.md':
        raise SystemExit('release must freeze the entry, config, ledger, comparator and new contract')
    for name, digest in frozen.items():
        target = (root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha(target) != digest:
            raise SystemExit(f'frozen input changed or missing: {name}')
    if type(release['seed']) is not int or type(release['envs']) is not int or release['envs'] not in (4, 8):
        raise SystemExit('invalid seed or worker count')
    outputs = []
    for kind in ('preflight', 'scale'):
        plan = release['plans'][kind]
        output = (root/plan['output']).resolve()
        if output.parent != root/'evidence/career_contract_10' or output.name == 'readiness_checks':
            raise SystemExit('outputs must use the new contract 10 evidence directory')
        if (type(plan['steps']) is not int or plan['steps'] <= 0 or plan['steps'] % 4096
                or type(plan['compare_rollouts']) is not int
                or not 0 < plan['compare_rollouts'] <= plan['steps']//4096
                or isinstance(plan['max_wall'], bool)
                or not math.isfinite(plan['max_wall']) or plan['max_wall'] <= 0):
            raise SystemExit('invalid frozen plan')
        outputs.append(output)
    if outputs[0] == outputs[1]:
        raise SystemExit('preflight and scale need distinct, exclusive outputs')
    resources = release['resources']
    if (type(resources['start_min_available_ram_bytes']) is not int
            or resources['start_min_available_ram_bytes'] < RESERVE):
        raise SystemExit('the start headroom cannot lower the 3 GiB RAM reserve')
    for field in ('poll_s', 'interrupt_grace_s', 'terminate_grace_s', 'kill_grace_s', 'process_wall_grace_s'):
        value = resources[field]
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise SystemExit(f'invalid frozen resource field: {field}')
    if not 0 < resources['poll_s'] <= 2 or resources['kill_grace_s'] == 0:
        raise SystemExit('poll_s must be in (0, 2]; kill_grace_s must be positive')
    return release


def result_complete(result, plan, envs):
    """A COMPLETE label alone does not establish completion or correct accounting."""
    if not isinstance(result, dict):
        return False
    counts, workers = result.get('actual_counts'), result.get('per_worker_counts')
    if (result.get('status') != 'COMPLETE' or result.get('inputs_unchanged') is not True
            or type(result.get('trained_transitions')) is not int
            or result['trained_transitions'] != plan['steps'] or result.get('error')
            or not isinstance(counts, dict) or not isinstance(workers, list) or len(workers) != envs):
        return False
    expected = {'policy_transitions': plan['steps'], 'native_steps': plan['steps']*10,
                'pd_updates': plan['steps']*5}
    if any(type(counts.get(key)) is not int or counts[key] != value for key, value in expected.items()):
        return False
    return all(isinstance(worker, dict) and set(worker) == set(counts)
               and all(type(value) is int and value >= 0 for value in worker.values())
               and worker['policy_transitions'] == plan['steps']//envs
               and worker['native_steps'] == worker['policy_transitions']*10
               and worker['pd_updates'] == worker['policy_transitions']*5 for worker in workers) and all(
        type(value) is int and value >= 0 and sum(worker[key] for worker in workers) == value
        for key, value in counts.items())


def receipt_dir(output):
    return output.with_name(output.name+'_supervisor')


def resource_sample(memory_reader, phase, started):
    value = dict(memory_reader(), phase=phase, elapsed_s=time.monotonic()-started)
    available, capped = value['available_ram_bytes'], value.get('cgroup_remaining_bytes')
    if (type(available) is not int or available < 0
            or (capped is not None and (type(capped) is not int or capped < 0))):
        raise ValueError('invalid resource sample')
    value['effective_available_bytes'] = min(available, capped) if capped is not None else available
    return value


def not_ready(kind, release_sha, initial, error, root):
    """A readiness refusal is recorded without claiming the single spawn directory."""
    directory = root/'evidence/career_contract_10/readiness_checks'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{kind}_{time.time_ns()}_{os.getpid()}.json'
    report = {'format': REPORT_FORMAT, 'kind': kind, 'status': 'NOT_READY',
              'stop_reason': 'RESOURCE_READ_FAILED' if error else 'START_RAM_HEADROOM',
              'error': error, 'stop_sample': initial, 'release_sha256': release_sha,
              'trainer_returncode': None, 'trainer_result': None, 'training_started': False,
              'minimum_observed_available_ram_bytes': initial['available_ram_bytes'] if initial else None,
              'physics_steps_executed_by_supervisor': 0, 'resource_reservation_guaranteed': False}
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'kind': kind, 'status': 'NOT_READY', 'readiness_report': str(path)}))
    return 2


def require_preflight(path, release, release_sha, root=ROOT):
    output = (Path(root)/release['plans']['preflight']['output']).resolve()
    if path is None or Path(path).resolve() != receipt_dir(output)/'report.json':
        raise SystemExit('the scale run needs this release\'s own preflight report')
    report = json.loads(Path(path).read_text())
    result_path = output/'result.json'
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if (report.get('format') != REPORT_FORMAT or report.get('kind') != 'preflight'
            or report.get('status') != 'PASS' or report.get('equivalence') != 'PASS'
            or report.get('trainer_returncode') != 0 or report.get('stop_reason') is not None
            or report.get('release_sha256') != release_sha
            or report.get('supervisor_sha256') != sha(__file__)
            or report.get('inputs_unchanged') is not True or report.get('cleanup_complete') is not True
            or report.get('plan') != release['plans']['preflight']
            or report.get('rollouts_compared') != release['plans']['preflight']['compare_rollouts']
            or report.get('rollouts_required') != release['plans']['preflight']['compare_rollouts']
            or report.get('first_mismatch') is not None
            or report.get('trainer_result') != result
            or not result_complete(result, release['plans']['preflight'], release['envs'])):
        raise SystemExit('preflight was incomplete, failed, changed, or belongs to a different release')


def live_group_members(pgid):
    """Ignore already-dead zombies; signal only the session this supervisor created."""
    members = []
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                members.append(int(stat.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_group(process, resources):
    signals = []
    for sig, grace in ((signal.SIGINT, resources['interrupt_grace_s']),
                       (signal.SIGTERM, resources['terminate_grace_s']),
                       (signal.SIGKILL, resources['kill_grace_s'])):
        process.poll()
        if not live_group_members(process.pid):
            break
        try:
            if sig == signal.SIGINT and process.returncode is None:
                process.send_signal(sig)  # Let the trainer collect counters and close its workers.
            else:
                os.killpg(process.pid, sig)
            signals.append(sig.name)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+grace
        while time.monotonic() < deadline:
            process.poll()
            if not live_group_members(process.pid):
                break
            time.sleep(min(.02, max(0, deadline-time.monotonic())))
    process.poll()
    return signals, not live_group_members(process.pid)


def run(kind, release_path, preflight_receipt=None, python=sys.executable, root=ROOT,
        memory_reader=read_memory):
    with handle_termination() as termination:
        return _run(kind, release_path, preflight_receipt, python, root, memory_reader, termination)


def _run(kind, release_path, preflight_receipt, python, root, memory_reader, termination):
    root, release_path = Path(root).resolve(), Path(release_path).resolve()
    release = load_release(release_path, root)
    release_sha = sha(release_path)
    if kind == 'scale':
        require_preflight(preflight_receipt, release, release_sha, root)
    plan, resources = release['plans'][kind], release['resources']
    output = (root/plan['output']).resolve()
    if output.exists():
        raise SystemExit(f'output already exists: {output}')
    sidecar = receipt_dir(output)
    if sidecar.exists():
        raise SystemExit(f'single spawn attempt already exists: {sidecar}')
    started = time.monotonic()
    try:
        initial = resource_sample(memory_reader, 'before_spawn', started)
    except Exception as exc:
        return not_ready(kind, release_sha, None, f'{type(exc).__name__}: {exc}', root)
    if initial['effective_available_bytes'] < resources['start_min_available_ram_bytes']:
        return not_ready(kind, release_sha, initial, None, root)
    sidecar.mkdir(parents=True, exist_ok=False)
    command = [python, '-B', str(root/release['entry']), '--output', str(output),
               '--config', str(root/release['config']), '--arm', 'B', '--seed', str(release['seed']),
               '--envs', str(release['envs']), '--steps', str(plan['steps']),
               '--max-wall', str(plan['max_wall']), '--kind', kind]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=f"{root/'.python-deps'}:{root/'training'}")
    process = None
    compared, mismatch, minimum, stop_sample = 0, None, None, None
    reason, error, cleanup_signals, cleanup_complete = None, None, [], True

    def compare_new():
        nonlocal compared, mismatch
        ledger = output/'ledger.jsonl'
        if mismatch is None and compared < len(frozen) and ledger.is_file():
            rows = complete_lines(ledger.read_text())
            while compared < min(len(rows), len(frozen)):
                problems = compare_rollout(json.loads(rows[compared]), frozen[compared])
                if problems:
                    mismatch = {'rollout': compared+1, 'problems': problems[:10]}
                    break
                compared += 1

    with (sidecar/'resources.jsonl').open('x') as observations:
        def sample(phase):
            nonlocal minimum
            value = resource_sample(memory_reader, phase, started)
            available = value['available_ram_bytes']
            minimum = available if minimum is None else min(minimum, available)
            observations.write(json.dumps(value, allow_nan=False)+'\n'); observations.flush()
            return value

        try:
            frozen = [json.loads(line) for line in complete_lines((root/plan['frozen_ledger']).read_text())]
            if len(frozen) < plan['compare_rollouts']:
                raise ValueError('frozen ledger has too few rollouts')
            frozen = frozen[:plan['compare_rollouts']]
            observations.write(json.dumps(initial, allow_nan=False)+'\n'); observations.flush()
            minimum = initial['available_ram_bytes']
            with (sidecar/'trainer_stdout.txt').open('x') as stdout:
                termination['defer'] = True  # Keep the handle if INT/TERM arrives during Popen.
                try:
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    termination['defer'] = False
                if termination['received']:
                    raise SupervisorTerminated(termination['received'])
                while process.poll() is None:
                    current = sample('running')
                    if current['effective_available_bytes'] < RESERVE:
                        reason, stop_sample = 'RAM_RESERVE_BELOW_3_GIB', current
                        break
                    if time.monotonic()-started >= plan['max_wall']+resources['process_wall_grace_s']:
                        reason, stop_sample = 'PROCESS_WALL_LIMIT', current
                        break
                    compare_new()
                    if mismatch:
                        reason = 'EQUIVALENCE_MISMATCH'
                        break
                    time.sleep(resources['poll_s'])
        except BaseException as exc:
            interrupted = isinstance(exc, (KeyboardInterrupt, SupervisorTerminated))
            reason = reason or ('SUPERVISOR_INTERRUPTED' if interrupted else 'SUPERVISOR_ERROR')
            error = f'{type(exc).__name__}: {exc}'
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)  # Finish bounded cleanup and its receipt.
            if process is not None:
                cleanup_signals, cleanup_complete = stop_group(process, resources)
                if cleanup_signals and reason is None:
                    reason = 'RESIDUAL_PROCESS_GROUP'
            try:
                sample('after_exit')
                compare_new()
            except BaseException as exc:
                reason = reason or 'SUPERVISOR_ERROR'
                error = error or f'{type(exc).__name__}: {exc}'
    result_path = output/'result.json'
    try:
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
    except (OSError, ValueError) as exc:
        result = None
        error = error or f'{type(exc).__name__}: {exc}'
    try:
        unchanged = (sha(__file__) == release['supervisor_sha256'] and sha(release_path) == release_sha
                     and all(sha(root/name) == digest for name, digest in release['frozen_inputs'].items()))
    except OSError as exc:
        unchanged = False
        error = error or f'{type(exc).__name__}: {exc}'
    equivalence = 'FAIL' if mismatch else 'PASS' if compared == plan['compare_rollouts'] else 'INCOMPLETE'
    returncode = process.returncode if process is not None else None
    if reason is None:
        if not cleanup_complete:
            reason = 'PROCESS_GROUP_CLEANUP_INCOMPLETE'
        elif returncode != 0:
            reason = 'TRAINER_EXIT_NONZERO'
        elif not unchanged:
            reason = 'FROZEN_INPUTS_CHANGED'
        elif equivalence != 'PASS':
            reason = 'EQUIVALENCE_MISMATCH' if mismatch else 'EQUIVALENCE_INCOMPLETE'
        elif not result_complete(result, plan, release['envs']):
            reason = 'TRAINING_INCOMPLETE'
    passed = (reason is None and error is None and cleanup_complete and returncode == 0
              and unchanged and equivalence == 'PASS' and result_complete(result, plan, release['envs']))
    report = {'format': REPORT_FORMAT, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'kind': kind, 'status': 'PASS' if passed else 'FAIL', 'plan': plan, 'command': command,
              'resources': resources, 'ram_reserve_bytes': RESERVE, 'training_started': process is not None,
              'process_group_id': process.pid if process is not None else None,
              'stop_reason': reason, 'error': error, 'trainer_returncode': returncode, 'trainer_result': result,
              'equivalence': equivalence, 'rollouts_compared': compared,
              'rollouts_required': plan['compare_rollouts'], 'first_mismatch': mismatch,
              'minimum_observed_available_ram_bytes': minimum, 'stop_sample': stop_sample,
              'cleanup_signals': cleanup_signals, 'cleanup_complete': cleanup_complete,
              'inputs_unchanged': unchanged, 'release_sha256': release_sha,
              'supervisor_sha256': release['supervisor_sha256'], 'wall_s': time.monotonic()-started,
              'resource_reservation_guaranteed': False,
              'sampling_scope': 'before_spawn, running and after_exit observations; reserve enforced while running; '
                                'not the frozen trainer callback samples',
              'physics_steps_executed_by_supervisor': 0}
    atomic_json(sidecar/'report.json', report)
    print(json.dumps({key: report[key] for key in ('kind', 'status', 'stop_reason', 'trainer_returncode')}))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('preflight', 'scale'))
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--preflight-receipt', type=Path)
    args = parser.parse_args()
    return run(args.kind, args.release, args.preflight_receipt)


if __name__ == '__main__':
    raise SystemExit(main())
