#!/usr/bin/env python3
"""Run the approved frozen sequence once; by default only print its commands.

The journal permits continuation after finished, SHA-bound stages. A started
stage without its final receipt blocks continuation instead of rerunning physics.
Resource refusals are the only stages that may be admitted again.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ('tools10', 'tools11', 'tools12'):
    sys.path.insert(0, str(ROOT/directory))
import supervise_scale10 as supervisor  # noqa: E402
from preflight_deployment11 import run_checked  # noqa: E402
from verify_contract10 import training_complete, load_release  # noqa: E402
from summarize_export10 import build as export_summary  # noqa: E402

sha = supervisor.sha
BASE = Path('evidence/career_contract_10')
OUTPUTS = {name: BASE/path for name, path in {
    'export': 'scale10_export_01', 'cpp': 'scale10_cpp_golden_01',
    'evaluation': 'scale10_eval_01', 'replay': 'scale10_replay_01',
    'verdict': 'scale10_verdict_01'}.items()}


class Journal:
    def __init__(self, directory, release_sha):
        self.directory = Path(directory)
        self.path = self.directory/'journal.json'
        identity = {'release_sha256': release_sha, 'runner_sha256': sha(__file__)}
        if self.directory.exists():
            self.data = json.loads(self.path.read_text())
            if any(self.data.get(key) != value for key, value in identity.items()):
                raise ValueError('journal identity differs from this release or runner')
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.data = {'format': 'd1-contract10-execution-v1', **identity, 'stages': {}}
            self.save()

    def save(self):
        supervisor.atomic_json(self.path, self.data)

    def stage(self, name, execute, artifacts):
        previous = self.data['stages'].get(name)
        if previous and previous['status'] != 'NOT_READY':
            if previous['status'] != 'FINISHED':
                raise ValueError(f'{name} already attempted without a final receipt; do not retry')
            for path, digest in previous['artifacts'].items():
                if not Path(path).is_file() or sha(path) != digest:
                    raise ValueError(f'{name} completed artifact changed or missing: {path}')
            return previous
        self.data['stages'][name] = {'status': 'STARTED'}
        self.save()  # Durable before entering any real stage.
        result = execute()
        receipt = {'status': 'NOT_READY' if result.get('not_ready') else 'FINISHED',
                   'result': result, 'artifacts': {str(path): sha(path) for path in artifacts()}}
        self.data['stages'][name] = receipt
        self.save()
        return receipt


def commands(release, release_path, root=ROOT):
    root = Path(root)
    py = [sys.executable, '-B']
    train = root/release['plans']['scale']['output']
    preflight = root/release['plans']['preflight']['output']
    export, cpp, evaluation, replay, verdict = (root/OUTPUTS[key] for key in OUTPUTS)
    return {
        'preflight': [*py, str(root/'tools10/supervise_scale10.py'), 'preflight', '--release', str(release_path)],
        'scale': [*py, str(root/'tools10/supervise_scale10.py'), 'scale', '--release', str(release_path),
                  '--preflight-receipt', str(supervisor.receipt_dir(preflight)/'report.json')],
        'export': [*py, str(root/'tools10/export_single_sample10.py'), '--checkpoint', str(train/'policy.zip'),
                   '--config', str(root/release['config']), '--output', str(export)],
        'cpp': [str(root/'runtime03/build/d1_runtime03'), '--schema', str(root/'runtime03/schema03d.json'),
                '--onnx-golden', str(export/'golden_fixtures.json'), '--policy', str(export/'policy.onnx'),
                '--expected-policy-sha256', 'FROM_SUCCESSFUL_EXPORT', '--output', str(cpp)],
        'evaluation': [*py, str(root/'tools/evaluate_d1_ppo.py'), '--checkpoint', str(train/'policy.zip'),
                       '--config', str(root/release['config']), '--phase', 'pilot', '--max-episodes', '3',
                       '--max-native-steps', '18000', '--output', str(evaluation)],
        'replay': [*py, str(root/'tools08/replay_training_episodes08.py'), '--train-dir', str(train),
                   '--verify-episodes', '0', '--output', str(replay)],
        'verdict': [*py, str(root/'tools12/verify_contract10.py'), '--release', str(release_path),
                    '--supervisor-report', str(supervisor.receipt_dir(train)/'report.json'),
                    '--replay', str(replay/'report.json'), '--eval-dir', str(evaluation), '--output', str(verdict)]}


def accounting(release, stages, root=ROOT):
    root = Path(root)
    known = {'transitions': 0, 'native': 0}
    pools, unknown = {}, []
    for name in ('preflight', 'scale', 'evaluation'):
        if name not in stages or stages[name].get('status') == 'NOT_READY':
            continue
        path = root/OUTPUTS['evaluation']/'summary.json' if name == 'evaluation' else (
            root/release['plans'][name]['output']/'result.json')
        if name == 'evaluation' and not path.exists():
            path = path.with_name('error.json')
        try:
            value = json.loads(path.read_text()) if path.is_file() else {}
            if not isinstance(value, dict):
                value = {}
        except (OSError, ValueError):
            value = {}
        if name == 'evaluation':
            native, transitions = value.get('actual_native_steps'), 0
            cap = 18000
        else:
            counts = value.get('actual_counts') or {}
            if not isinstance(counts, dict):
                counts = {}
            native, transitions = counts.get('native_steps'), counts.get('policy_transitions')
            cap = release['plans'][name]['steps']*10
        valid = (type(native) is int and 0 <= native <= cap and type(transitions) is int
                 and transitions >= 0 and (name == 'evaluation' or native == transitions*10))
        if valid:
            known['native'] += native
            known['transitions'] += transitions
            pools[name] = {'native': native, 'transitions': transitions, 'receipt_sha256': sha(path)}
        # Only a final receipt from a cleanly finished child closes its pool.
        result = stages[name].get('result', {})
        if (not valid or stages[name].get('status') != 'FINISHED'
                or result.get('cleanup_complete') is not True
                or (name == 'evaluation' and value.get('status') != 'COMPLETE')):
            unknown.append(name)
    before = release['project_cumulative_before']
    return {'format': 'd1-contract10-accounting-v1', 'approved_cap': release['proposed_budget'],
            'known_actual': known, 'pools': pools, 'unknown_actual_pools': unknown,
            'actual_complete': not unknown, 'project_cumulative_before': before,
            'project_known_at_least': {key: before.get(key, 0)+known[key] for key in known},
            'unspent_budget_is_not_retry_authorisation': True}


def execute(release_path, root=ROOT):
    root, release_path = Path(root).resolve(), Path(release_path).resolve()
    release = load_release(release_path, root)
    journal = Journal(root/BASE/'execution_01', sha(release_path))
    argv = commands(release, release_path, root)
    env = dict(os.environ, PYTHONPATH=f"{root/'.python-deps'}:{root/'training'}",
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')

    def unchanged():
        if sha(release_path) != journal.data['release_sha256']:
            raise ValueError('frozen release changed')
        load_release(release_path, root)

    def snapshot(directory):
        return sorted(path for path in directory.rglob('*') if path.is_file()) if directory.exists() else []

    def run_training(kind):
        unchanged()
        output = root/release['plans'][kind]['output']
        report_path = supervisor.receipt_dir(output)/'report.json'
        receipt = root/release['plans']['preflight']['output']
        code = supervisor.run(kind, release_path, supervisor.receipt_dir(receipt)/'report.json'
                              if kind == 'scale' else None, root=root)
        if code == 2 and not report_path.exists() and not output.exists():
            return {'returncode': code, 'not_ready': True, 'cleanup_complete': True}
        report = json.loads(report_path.read_text())
        return {'returncode': code, 'cleanup_complete': report['cleanup_complete'],
                'supervisor_report': str(report_path)}

    def training_files(kind):
        output = root/release['plans'][kind]['output']
        return snapshot(output)+snapshot(supervisor.receipt_dir(output))

    def run_child(name, timeout):
        unchanged()
        if (root/OUTPUTS[name]).exists():
            raise ValueError(f'{name} output already exists without this journal; do not retry')
        return run_checked(argv[name], journal.directory/name, timeout, env)

    try:
        for kind in ('preflight', 'scale'):
            record = journal.stage(kind, lambda k=kind: run_training(k), lambda k=kind: training_files(k))
            if record['status'] == 'NOT_READY':
                journal.data['status'] = 'WAITING_RAM'
                return 2
            if record['result']['returncode'] != 0 or record['result']['cleanup_complete'] is not True:
                journal.data['status'] = 'TRAINING_STOPPED'
                if kind == 'scale':
                    # The judge records NOT_EVALUABLE without reading eval/replay.
                    argv['verdict'] = argv['verdict'][:argv['verdict'].index('--replay')]+[
                        '--output', str(root/OUTPUTS['verdict'])]
                    journal.stage('verdict', lambda: run_child('verdict', 120),
                                  lambda: snapshot(root/OUTPUTS['verdict']))
                return 1
        train = root/release['plans']['scale']['output']
        report = json.loads((supervisor.receipt_dir(train)/'report.json').read_text())
        if (report.get('format') != supervisor.REPORT_FORMAT or report.get('kind') != 'scale'
                or report.get('release_sha256') != sha(release_path)
                or report.get('supervisor_sha256') != release['supervisor_sha256']
                or report.get('plan') != release['plans']['scale']
                or report.get('resources') != release['resources']):
            raise ValueError('foreign supervisor report; refusing downstream evaluation')
        if not training_complete(report, release, json.loads((train/'result.json').read_text())):
            raise ValueError('final-policy path requires verified complete training')
        exported = journal.stage('export', lambda: run_child('export', 600), lambda: snapshot(root/OUTPUTS['export']))
        journal.data['numerical_deployment_chain'] = {'status': 'FAIL', 'reason': 'export failed or has no passing sidecar'}
        if exported['result']['cleanup_complete'] is not True:
            raise ValueError('export process cleanup incomplete')
        sidecar = root/OUTPUTS['export']/'single_sample_export.json'
        if exported['result']['returncode'] == 0 and sidecar.is_file():
            info = json.loads(sidecar.read_text())
            if info.get('status') != 'PASS' or info.get('checkpoint_sha256') != sha(train/'policy.zip'):
                raise ValueError('export does not bind this final policy')
            command = argv['cpp']
            command[command.index('FROM_SUCCESSFUL_EXPORT')] = sha(root/OUTPUTS['export']/'policy.onnx')
            cpp = journal.stage('cpp', lambda: run_child('cpp', 120), lambda: snapshot(root/OUTPUTS['cpp']))
            if cpp['result']['cleanup_complete'] is not True:
                raise ValueError('C++ process cleanup incomplete')
            try:
                if cpp['result']['returncode'] != 0:
                    raise ValueError('C++ numerical check failed')
                journal.data['numerical_deployment_chain'] = export_summary(
                    root/OUTPUTS['export'], root/OUTPUTS['cpp']/'report.json', root/'reports/contract08_summary.json')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                journal.data['numerical_deployment_chain'] = {'status': 'FAIL', 'reason': str(exc)}
        for name, timeout in (('evaluation', 900), ('replay', 300), ('verdict', 120)):
            record = journal.stage(name, lambda n=name, t=timeout: run_child(n, t),
                                   lambda n=name: snapshot(root/OUTPUTS[n]))
            if record['result']['returncode'] != 0 or record['result']['cleanup_complete'] is not True:
                journal.data['status'] = name.upper()+'_FAILED'
                return 1
        journal.data['status'] = 'COMPLETE'
        return 0
    finally:
        journal.data['accounting'] = accounting(release, journal.data['stages'], root)
        journal.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='run this approved frozen sequence once')
    args = parser.parse_args()
    release = load_release(args.release.resolve(), ROOT)
    if not args.execute:
        print(json.dumps({'mode': 'PLAN_ONLY', 'physics_steps_executed': 0,
                          'commands': commands(release, args.release.resolve())}, indent=2))
        return 0
    return execute(args.release)


if __name__ == '__main__':
    raise SystemExit(main())
