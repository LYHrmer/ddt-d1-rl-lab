#!/usr/bin/env python3
"""Prepare, plan, or execute the single bounded D1 closeout; never trains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools17 import provenance17 as provenance  # noqa: E402
from tools16.contract16 import read_memory  # noqa: E402
from tools16.run_comparison16 import cleanup, termination_guard  # noqa: E402

ORDER = ('export', 'cpp', 'formal')
BUDGET = {'policy_transitions': 21600, 'native_steps': 216000, 'pd_updates': 108000}
RESOURCES = {'start_min_effective_bytes': 6 * 2**30, 'reserve_bytes': 3 * 2**30,
             'poll_s': 1., 'process_wall_grace_s': 30,
             'interrupt_grace_s': 10, 'terminate_grace_s': 5, 'kill_grace_s': 2}
LIMITS = {'export': 180, 'cpp': 90, 'formal': 600}
OUTPUT_ROOT = 'evidence/closeout17/execution_01'
FORMAT = 'd1-closeout17-authorized-v1'


def prepare(authorization_path, output):
    authorization_path, output = Path(authorization_path).resolve(), Path(output).resolve()
    provenance.require(output.is_relative_to(ROOT / 'evidence/closeout17'), 'prepare under closeout17 evidence')
    auth = provenance.read_json(authorization_path)
    provenance.require(auth.get('approved') is True and auth.get('owner_statement') == '完成收尾工作'
                       and auth.get('budget') == BUDGET and auth.get('no_training') is True
                       and auth.get('one_attempt') is True and bool(auth.get('owner_message_reference')),
                       'record explicit owner closeout scope and bounded validation budget')
    summary = provenance.read_json(ROOT / provenance.SUMMARY)
    training_root = Path(summary['replacement']['root'])
    _, _, result, _ = provenance.load_completed_control(training_root)
    release = {'format': FORMAT, 'status': 'AUTHORIZED_FROZEN',
               'owner_statement': auth['owner_statement'],
               'authorization_path': str(authorization_path), 'authorization_sha256': provenance.sha256(authorization_path),
               'source_identity': provenance.source_identity(), 'training_root': str(training_root),
               'checkpoint_sha256': result['checkpoint_sha256'], 'budget': BUDGET,
               'resources': RESOURCES, 'stage_limits_s': LIMITS, 'stage_order': list(ORDER),
               'output_root': OUTPUT_ROOT, 'training_allowed': False, 'retry_or_resume_allowed': False,
               'formal_backend': 'python_ort_cpu_complete_export',
               'formal_scenarios': 36, 'episode_ticks': 600}
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.exclusive_json(output, release)
    return release


def load_contract(path):
    path = Path(path).resolve()
    provenance.require(path.is_relative_to(ROOT / 'evidence/closeout17'), 'release must be in closeout17 evidence')
    release = provenance.read_json(path)
    expected = {'format': FORMAT, 'status': 'AUTHORIZED_FROZEN', 'owner_statement': '完成收尾工作',
                'budget': BUDGET, 'resources': RESOURCES, 'stage_limits_s': LIMITS,
                'stage_order': list(ORDER), 'output_root': OUTPUT_ROOT,
                'training_allowed': False, 'retry_or_resume_allowed': False,
                'formal_backend': 'python_ort_cpu_complete_export', 'formal_scenarios': 36, 'episode_ticks': 600}
    provenance.require(all(release.get(key) == value for key, value in expected.items()), 'fixed closeout scope differs')
    auth_path = Path(release['authorization_path'])
    provenance.require(auth_path.resolve().is_relative_to(ROOT / 'evidence/closeout17')
                       and provenance.sha256(auth_path) == release['authorization_sha256'], 'owner record changed')
    auth = provenance.read_json(auth_path)
    provenance.require(auth.get('approved') is True and auth.get('owner_statement') == '完成收尾工作'
                       and auth.get('budget') == BUDGET and auth.get('no_training') is True
                       and auth.get('one_attempt') is True and bool(auth.get('owner_message_reference')),
                       'closeout authorization differs')
    provenance.check_sources(release['source_identity'])
    _, _, result, _ = provenance.load_completed_control(release['training_root'])
    provenance.require(result['checkpoint_sha256'] == release['checkpoint_sha256'], 'selected checkpoint changed')
    return release


def commands(release, release_path):
    output = ROOT / release['output_root']
    common = [sys.executable, '-B']
    return [
        {'name': 'export', 'argv': [*common, str(ROOT / 'tools17/export_completed17.py'),
                                  '--training-root', release['training_root'], '--output', str(output / 'export')]},
        {'name': 'cpp', 'argv': [*common, str(ROOT / 'tools17/verify_cpp17.py'),
                               '--export-dir', str(output / 'export'), '--output', str(output / 'cpp')]},
        {'name': 'formal', 'argv': [*common, str(ROOT / 'tools17/evaluate_formal17.py'),
                                  '--release', str(Path(release_path).resolve()),
                                  '--export-dir', str(output / 'export'),
                                  '--cpp-report', str(output / 'cpp/report.json'), '--output', str(output / 'formal')]},
    ]


def validate_launch(release, release_path, output):
    execution = ROOT / release['output_root']
    provenance.require(Path(output).resolve() == execution / 'formal', 'formal output outside the fixed attempt')
    report = provenance.read_json(execution / 'report.json')
    stage = report.get('stages', {}).get('formal', {})
    expected_command = commands(release, release_path)[2]['argv']
    provenance.require(report.get('status') == 'RUNNING' and report.get('active_stage') == 'formal'
                       and report.get('supervisor_pid') == os.getppid()
                       and report.get('release_sha256') == provenance.sha256(release_path)
                       and stage.get('status') == 'STARTED' and stage.get('command') == expected_command
                       and os.environ.get('D1_CLOSEOUT17_STAGE') == 'formal'
                       and stage.get('launch_token') == os.environ.get('D1_CLOSEOUT17_LAUNCH_TOKEN')
                       and bool(stage.get('launch_token'))
                       and all(report['stages'].get(name, {}).get('status') == 'COMPLETE' for name in ('export', 'cpp')),
                       'formal evaluation requires the claimed supervisor stage and completed numeric checks')


def valid_counts(counts):
    if not (isinstance(counts, dict) and set(counts) == set(BUDGET)
            and all(type(counts.get(k)) is int and 0 <= counts[k] <= cap for k, cap in BUDGET.items())):
        return False
    policy, native, pd = (counts[key] for key in ('policy_transitions', 'native_steps', 'pd_updates'))
    return (max(0, policy - 1) * 10 <= native <= policy * 10
            and max(0, policy - 1) * 5 <= pd <= policy * 5 and 0 <= pd * 2 - native <= 2)


def minimum_formal_identity(child, release, release_sha, output):
    if not isinstance(child, dict):
        return False
    try:
        path = Path(output) / 'protocol.json'
        protocol = provenance.read_json(path)
        return (child.get('format') == 'd1-closeout17-formal-result-v1'
                and child.get('release_sha256') == release_sha
                and child.get('checkpoint_sha256') == release['checkpoint_sha256']
                and child.get('source_identity') == release['source_identity']
                and child.get('protocol_sha256') == provenance.sha256(path)
                and protocol.get('format') == 'd1-closeout17-formal-protocol-v1'
                and protocol.get('release_sha256') == release_sha
                and protocol.get('checkpoint_sha256') == release['checkpoint_sha256']
                and protocol.get('source_identity') == release['source_identity'])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def cleanup_owned(process, natural_exit):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            sent, clean = cleanup(process, RESOURCES, natural_exit=natural_exit)
            return sent, clean, None
        except BaseException as exc:
            return [], False, f'cleanup {type(exc).__name__}: {exc}'
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def artifact_hashes(directory):
    return {str(path): provenance.sha256(path) for path in Path(directory).rglob('*') if path.is_file()}


def unchanged_stages(report):
    for record in report['stages'].values():
        if record['status'] == 'COMPLETE':
            provenance.require(bool(record.get('artifacts_sha256')), 'completed stage artifact manifest missing')
            for name, digest in record['artifacts_sha256'].items():
                provenance.require(provenance.sha256(name) == digest, 'completed stage artifact changed: ' + name)


def execute(release_path):
    release_path = Path(release_path).resolve()
    release = load_contract(release_path)
    execution = ROOT / release['output_root']
    provenance.require(not execution.exists(), 'attempt already claimed; no retry or resume')
    try:
        sample, readiness_error = read_memory(), None
    except Exception as exc:
        sample, readiness_error = None, f'{type(exc).__name__}: {exc}'
    if sample is None or sample['effective_available_bytes'] < RESOURCES['start_min_effective_bytes']:
        directory = execution.parent / 'readiness_checks'
        directory.mkdir(exist_ok=True)
        provenance.exclusive_json(directory / f'{time.time_ns()}.json',
                                  {'status': 'NOT_READY', 'execution_claimed': False, 'sample': sample,
                                   'error': readiness_error})
        return 2
    started = time.monotonic()
    execution.mkdir(parents=True, exist_ok=False)
    release_sha = provenance.sha256(release_path)
    report = {'format': 'd1-closeout17-execution-v1', 'status': 'RUNNING', 'active_stage': None,
              'release_sha256': release_sha, 'source_identity': release['source_identity'],
              'supervisor_pid': os.getpid(), 'stages': {}, 'error': None,
              'minimum_effective_bytes': sample['effective_available_bytes'], 'actual_counts': None,
              'known_actual_counts': dict.fromkeys(BUDGET, 0), 'unknown_count_stages': [],
              'training_transitions': 0, 'retry_or_resume_allowed': False}
    report_path = execution / 'report.json'
    provenance.atomic_json(report_path, report)
    with (execution / 'resources.jsonl').open('x') as resource_log, termination_guard() as control:
        resource_log.write(json.dumps(dict(sample, stage='export', phase='initial', elapsed_s=0., sample_valid=True)) + '\n')
        resource_log.flush()
        def observe(name, phase):
            try:
                memory = read_memory()
            except Exception as exc:
                resource_log.write(json.dumps({'sample_valid': False, 'stage': name, 'phase': phase,
                    'elapsed_s': time.monotonic() - started, 'error': f'{type(exc).__name__}: {exc}'}) + '\n')
                resource_log.flush()
                raise
            entry = dict(memory, stage=name, phase=phase, elapsed_s=time.monotonic() - started, sample_valid=True)
            resource_log.write(json.dumps(entry, allow_nan=False) + '\n')
            resource_log.flush()
            report['minimum_effective_bytes'] = min(report['minimum_effective_bytes'], memory['effective_available_bytes'])
            return memory['effective_available_bytes']

        try:
            for spec in commands(release, release_path):
                name = spec['name']
                provenance.require(provenance.sha256(release_path) == release_sha, 'frozen release changed')
                load_contract(release_path)
                unchanged_stages(report)
                provenance.require(observe(name, 'before_spawn') >= RESOURCES['start_min_effective_bytes'],
                                   'stage start effective RAM below 6 GiB')
                token = secrets.token_hex(16)
                record = {'status': 'STARTED', 'command': spec['argv'], 'launch_token': token,
                          'process_started': False, 'returncode': None, 'cleanup_complete': True,
                          'stop_reason': None, 'receipt': None, 'error': None}
                report['stages'][name] = record
                report['active_stage'] = name
                provenance.atomic_json(report_path, report)
                process, stage_started = None, time.monotonic()
                directory = execution / 'supervision' / name
                directory.mkdir(parents=True)
                try:
                    env = dict(os.environ, PYTHONPATH=f'{ROOT}:{ROOT / "training"}:{ROOT / ".python-deps"}',
                               PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                               MKL_NUM_THREADS='1', D1_CLOSEOUT17_STAGE=name, D1_CLOSEOUT17_LAUNCH_TOKEN=token)
                    with (directory / 'stdout.txt').open('x') as stdout, (directory / 'stderr.txt').open('x') as stderr:
                        control['defer'] = True
                        try:
                            process = subprocess.Popen(spec['argv'], cwd=ROOT, env=env, stdout=stdout,
                                                       stderr=stderr, start_new_session=True)
                            record['process_started'] = True
                        finally:
                            control['defer'] = False
                        if control['received']:
                            raise RuntimeError('supervisor interrupted during spawn')
                        while process.poll() is None:
                            provenance.require(observe(name, 'running') >= RESOURCES['reserve_bytes'],
                                               'RAM reserve below 3 GiB')
                            provenance.require(time.monotonic() - stage_started < LIMITS[name] + 30,
                                               'stage process wall limit')
                            time.sleep(RESOURCES['poll_s'])
                    provenance.require(time.monotonic() - stage_started < LIMITS[name] + 30,
                                       'stage process wall limit after exit')
                    provenance.require(process.returncode == 0, 'child exited nonzero')
                except BaseException as exc:
                    record.update(stop_reason='STAGE_FAILED', error=f'{type(exc).__name__}: {exc}')
                finally:
                    if process is not None:
                        signals, clean, error = cleanup_owned(process, natural_exit=record['stop_reason'] is None)
                        record.update(cleanup_complete=clean, cleanup_signals=signals, returncode=process.returncode)
                        if error:
                            record['error'] = error
                        if not clean or signals:
                            record['stop_reason'] = record['stop_reason'] or 'CLEANUP_INCOMPLETE_OR_RESIDUAL'
                path = execution / name / 'report.json'
                if path.is_file():
                    record['receipt'] = provenance.read_json(path)
                    record['receipt_sha256'] = provenance.sha256(path)
                if record['stop_reason'] is None:
                    try:
                        child = record['receipt'] or {}
                        provenance.require(child.get('source_identity') == release['source_identity'], 'child source proof differs')
                        provenance.require(child.get('status') == ('COMPLETE' if name == 'formal' else 'PASS'),
                                           'child receipt does not prove completion')
                        provenance.require(child.get('checkpoint_sha256') == release['checkpoint_sha256'],
                                           'child checkpoint identity differs')
                        if name == 'formal':
                            provenance.require(minimum_formal_identity(child, release, release_sha, execution / 'formal')
                                               and child.get('inputs_unchanged') is True
                                               and child.get('cleanup_complete') is True
                                               and child.get('error') is None and len(child.get('cases', [])) == 36
                                               and valid_counts(child.get('actual_counts')), 'formal receipt is incomplete')
                        record['artifacts_sha256'] = artifact_hashes(execution / name)
                    except BaseException as exc:
                        record.update(stop_reason='CHILD_COMPLETION_INVALID', error=f'{type(exc).__name__}: {exc}')
                record['status'] = 'COMPLETE' if record['stop_reason'] is None else 'FAIL'
                provenance.atomic_json(report_path, report)
                if record['status'] != 'COMPLETE':
                    report.update(status='FAIL', failed_stage=name)
                    break
            else:
                load_contract(release_path)
                unchanged_stages(report)
                report['status'] = 'COMPLETE'
        except BaseException as exc:
            report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}', failed_stage=report['active_stage'])
        finally:
            formal = report['stages'].get('formal', {})
            counts = (formal.get('receipt') or {}).get('actual_counts')
            if formal.get('process_started'):
                bound = minimum_formal_identity(formal.get('receipt'), release, release_sha, execution / 'formal')
                report['actual_counts'] = counts if bound and valid_counts(counts) else None
                report['known_actual_counts'] = counts if bound and valid_counts(counts) else dict.fromkeys(BUDGET, 0)
                report['unknown_count_stages'] = [] if bound and valid_counts(counts) else ['formal']
            else:
                report['actual_counts'] = dict.fromkeys(BUDGET, 0)
            report['active_stage'] = None
            report['wall_s'] = time.monotonic() - started
            provenance.atomic_json(report_path, report)
    return 0 if report['status'] == 'COMPLETE' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands_parser = parser.add_subparsers(dest='command', required=True)
    preparation = commands_parser.add_parser('prepare')
    preparation.add_argument('--authorization', type=Path, required=True)
    preparation.add_argument('--output', type=Path, required=True)
    for name in ('plan', 'execute'):
        command = commands_parser.add_parser(name)
        command.add_argument('--release', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        value = prepare(args.authorization, args.output)
        print(json.dumps({'status': value['status'], 'budget': value['budget'], 'execution_started': False}))
        return 0
    if args.command == 'plan':
        release = load_contract(args.release)
        print(json.dumps({'status': 'PLAN_ONLY', 'budget': release['budget'],
                          'commands': commands(release, args.release), 'execution_started': False}, indent=2))
        return 0
    return execute(args.release)


if __name__ == '__main__':
    raise SystemExit(main())
