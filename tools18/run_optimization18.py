"""One new, serial, resource-bounded optimization attempt; never resumes."""
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
from tools18 import common18 as common  # noqa: E402
from tools16.run_comparison16 import cleanup, compare_prefix, termination_guard  # noqa: E402


def command(plan_path, stage, output):
    kind = common.STAGES[stage]['kind']
    filename = {'train': 'train_delay18.py', 'evaluate': 'evaluate_delay18.py',
                'export': 'export_delay18.py', 'cpp': 'export_delay18.py'}[kind]
    return [sys.executable, '-B', str(ROOT/'tools18'/filename), '--plan', str(plan_path),
            '--stage', stage, '--output', str(output)]


def artifacts(directory):
    return {str(path): common.sha(path) for path in Path(directory).rglob('*') if path.is_file()}


def unchanged_stages(report):
    for record in report['stages'].values():
        if record['status'] == 'COMPLETE':
            for name, digest in record['artifacts_sha256'].items():
                common.require(common.sha(name) == digest, 'completed stage artifact changed: '+name)


def child_receipt(output, kind):
    return Path(output)/('result.json' if kind == 'train' else 'report.json')


def known_counts(report):
    """Rebuild once from unique stage facts; interrupted aggregation is idempotent."""
    values = [record['actual_counts'] for record in report['stages'].values()
              if record.get('actual_counts') is not None]
    return {key: sum(value[key] for value in values) for key in common.expected_counts(0)}


def valid_counts(value, cap, *, complete, envs):
    if not isinstance(value, dict) or not all(type(value.get(key)) is int and 0 <= value[key] <= cap[key] for key in cap):
        return False
    p, native, pd = (value[key] for key in ('policy_transitions', 'native_steps', 'pd_updates'))
    if complete:
        return native == 10*p and pd == 5*p
    return (max(0, p-envs)*10 <= native <= 10*p and max(0, p-envs)*5 <= pd <= 5*p
            and 0 <= 2*pd-native <= 2*envs)


def verify_child(plan, plan_path, stage, output, child):
    spec = plan['stages'][stage]
    common.require(child.get('status') == ('PASS' if spec['kind'] in ('export', 'cpp') else 'COMPLETE')
                   and child.get('cleanup_complete') is True and child.get('error') is None,
                   'child stage incomplete or failed')
    common.require(child.get('plan_sha256') == common.sha(plan_path)
                   and child.get('source_identity') == plan['source_identity'] and child.get('stage') == stage,
                   'child identity mismatch')
    cap = common.expected_counts(spec['transitions'])
    common.require(valid_counts(child.get('actual_counts'), cap, complete=True, envs=plan['envs']),
                   'child physical counters invalid')
    if spec['kind'] == 'train':
        common.require(child.get('actual_counts', {}).get('policy_transitions') == spec['transitions']
                       and child.get('new_trained_transitions') == spec['transitions']
                       and child.get('inputs_unchanged') is True, 'training steps incomplete')
        if stage.startswith('train_'):
            mode = stage.removeprefix('train_')
            preflight = ROOT/plan['output_root']/('preflight_'+mode)/'ledger.jsonl'
            compared, difference = compare_prefix(preflight, Path(output)/'ledger.jsonl')
            common.require(compared == 8 and difference is None, 'preflight/full training prefix differs')
        other = {'preflight_random': 'preflight_zero', 'train_random': 'train_zero'}.get(stage)
        if other:
            before = common.read_json(ROOT/plan['output_root']/other/'result.json')
            common.require(child['initial_state'] == before['initial_state'], 'paired arms did not start from same policy/Adam')
    elif spec['kind'] == 'evaluate':
        from tools18.evaluate_delay18 import matrix
        expected = matrix(stage)
        common.require(child.get('inputs_unchanged') is True
                       and [(item['sequence'], item['delay_ticks'], item['reset_seed']) for item in child['cases']] == expected,
                       'evaluation matrix incomplete')


def execute(plan_path, *, loader=common.load_plan, memory_reader=common.read_memory, clock=time.monotonic):
    plan_path = Path(plan_path).resolve()
    plan = loader(plan_path)
    execution = ROOT/plan['output_root']
    common.require(not execution.exists(), 'attempt already claimed; no resume or retry')
    admission = memory_reader()
    if admission['effective_available_bytes'] < plan['resources']['start_min_effective_bytes']:
        return {'status': 'NOT_READY', 'execution_started': False, 'resource_sample': admission}
    execution.mkdir(parents=True, exist_ok=False)
    started = clock()
    report = {'format': 'd1-optimization18-execution-v1', 'status': 'RUNNING', 'active_stage': None,
              'plan_sha256': common.sha(plan_path), 'source_identity': plan['source_identity'],
              'supervisor_pid': os.getpid(), 'stages': {}, 'actual_counts': None,
              'known_actual_counts': common.expected_counts(0), 'unknown_count_stages': [],
              'minimum_effective_bytes': admission['effective_available_bytes'], 'cleanup_complete': True,
              'error': None, 'new_training_transitions': 0, 'selected_arm': None,
              'full_delay_qualification': False}
    common.atomic_json(execution/'report.json', report)
    process = None
    current = None

    def sample(stage, phase):
        row = memory_reader()
        common.require(type(row.get('effective_available_bytes')) is int and row['effective_available_bytes'] >= 0,
                       'invalid resource sample')
        row.update(elapsed_s=clock()-started, stage=stage, phase=phase)
        report['minimum_effective_bytes'] = min(report['minimum_effective_bytes'], row['effective_available_bytes'])
        with (execution/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        return row

    with termination_guard() as termination:
        try:
            sample(None, 'initial')
            for stage in plan['stage_order']:
                spec = plan['stages'][stage]
                loader(plan_path)
                common.require(common.sha(plan_path) == report['plan_sha256'], 'plan changed')
                unchanged_stages(report)
                if stage == 'holdout':
                    from tools18.evaluate_delay18 import selection
                    report['selected_arm'] = selection(plan)
                    if report['selected_arm'] is None:
                        report['stages'][stage] = {'status': 'SKIPPED_NO_QUALIFIED_CANDIDATE',
                                                   'actual_counts': common.expected_counts(0)}
                        continue
                output = execution/stage
                row = sample(stage, 'before_launch')
                common.require(row['effective_available_bytes'] >= plan['resources']['start_min_effective_bytes'],
                               'stage start RAM below6GiB')
                record = {'status': 'STARTED', 'nonce': secrets.token_hex(24),
                          'command': command(plan_path, stage, output), 'actual_counts': None}
                current = stage
                report['stages'][stage] = record
                report['active_stage'] = stage
                common.atomic_json(execution/'report.json', report)
                env = dict(os.environ, D1_STAGE18_NONCE=record['nonce'],
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                           PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT))))
                stage_started = clock()
                try:
                    with (execution/(stage+'.stdout.log')).open('x') as stdout:
                        termination['defer'] = True
                        try:
                            process = subprocess.Popen(record['command'], cwd=ROOT, env=env, stdout=stdout,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                        finally:
                            termination['defer'] = False
                        record['pid'] = process.pid
                        common.require(termination['received'] is None,
                                       'termination requested: '+str(termination['received']))
                        common.atomic_json(execution/'report.json', report)
                        while process.poll() is None:
                            row = sample(stage, 'running')
                            common.require(row['effective_available_bytes'] >= plan['resources']['reserve_bytes'],
                                           'RAM reserve below3GiB')
                            common.require(clock()-stage_started < spec['max_wall_s']+plan['resources']['process_wall_grace_s'],
                                           'stage process wall limit')
                            time.sleep(plan['resources']['poll_s'])
                        common.require(clock()-stage_started < spec['max_wall_s']+plan['resources']['process_wall_grace_s'],
                                       'stage process exceeded deadline before observed exit')
                        record['exit_code'] = process.returncode
                finally:
                    termination['defer'] = True
                    if process is not None:
                        try:
                            signals_sent, clean = cleanup(process, plan['resources'], natural_exit=process.poll() is not None)
                            record.update(cleanup_complete=clean, cleanup_signals=signals_sent)
                            report['cleanup_complete'] &= clean
                        except BaseException as exc:
                            record.update(cleanup_complete=False, cleanup_error=str(exc))
                            report['cleanup_complete'] = False
                            raise
                    termination['defer'] = False
                common.require(termination['received'] is None, 'termination requested: '+str(termination['received']))
                path = child_receipt(output, spec['kind'])
                child = common.read_json(path)
                record['receipt_sha256'] = common.sha(path)
                common.require(record['exit_code'] == 0 and record['cleanup_complete'], 'child exit or process cleanup failed')
                verify_child(plan, plan_path, stage, output, child)
                record['actual_counts'] = {key: child['actual_counts'][key] for key in plan['budget']}
                record['status'] = 'COMPLETE'
                record['artifacts_sha256'] = artifacts(output)
                report['known_actual_counts'] = known_counts(report)
                if spec['kind'] == 'train':
                    report['new_training_transitions'] += child['new_trained_transitions']
                if stage == 'holdout':
                    report['full_delay_qualification'] = child['all_36_reference_pass'] is True
                common.atomic_json(execution/'report.json', report)
                print(json.dumps({'stage': stage, 'status': 'COMPLETE',
                                  'reference_pass': child.get('reference_pass_cases'),
                                  'actual_counts': record['actual_counts']}), flush=True)
                process, current = None, None
            loader(plan_path)
            common.require(common.sha(plan_path) == report['plan_sha256'], 'plan changed during final verification')
            unchanged_stages(report)
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['status'] = 'STOPPED'
            report['error'] = f'{type(exc).__name__}: {exc}'
            if current is not None:
                record = report['stages'].get(current)
                if record is not None:
                    record['status'] = 'INCOMPLETE'
                try:
                    spec = plan['stages'][current]
                    child = common.read_json(child_receipt(execution/current, spec['kind']))
                    common.require(child.get('plan_sha256') == report['plan_sha256']
                                   and child.get('source_identity') == plan['source_identity']
                                   and child.get('stage') == current, 'foreign partial receipt')
                    if spec['kind'] in ('train', 'evaluate'):
                        protocol_path = execution/current/'protocol.json'
                        protocol = common.read_json(protocol_path)
                        common.require(child.get('protocol_sha256') == common.sha(protocol_path)
                                       and protocol.get('plan_sha256') == report['plan_sha256']
                                       and protocol.get('source_identity') == plan['source_identity']
                                       and protocol.get('stage') == current, 'foreign partial protocol')
                    counts = child['actual_counts']
                    common.require(valid_counts(counts, common.expected_counts(spec['transitions']),
                                                complete=False, envs=plan['envs']), 'unknown partial counts')
                    record['actual_counts'] = {key: counts[key] for key in plan['budget']}
                    report['known_actual_counts'] = known_counts(report)
                except BaseException:
                    report['unknown_count_stages'].append(current)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            report['known_actual_counts'] = known_counts(report)
            report['actual_counts'] = None if report['unknown_count_stages'] else report['known_actual_counts']
            report['active_stage'] = None
            report['elapsed_s'] = clock()-started
            common.atomic_json(execution/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'execute'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        value = common.load_plan(args.plan)
        print(json.dumps({'status': 'PLAN_ONLY', 'budget': value['budget'], 'stages': value['stages']}, indent=2))
        return 0
    report = execute(args.plan)
    print(json.dumps({key: report.get(key) for key in ('status', 'actual_counts', 'full_delay_qualification', 'error')}, indent=2))
    return 0 if report['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
