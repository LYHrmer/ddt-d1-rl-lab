"""Training-only, read-only value/gradient audit of all six complete69 models."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(directory))
from tools53.run_pilot53 import require, sha, write_json

SOURCE69 = Path('evidence/body_jvp69/execution_01')
PLAN69 = Path('docs/experiments69/body_jvp69_preregistered.json')
TRAIN = Path('evidence/model_gradient53/execution_01/train_transitions.npz')
SUMMARY53 = Path('reports/model_gradient53_results_20261101.json')
LABELS68 = Path('evidence/body_response68/admission_01')
ZERO_SOURCES = {'58': Path('evidence/train_action_response58/collection_01'),
                '62': Path('evidence/clean_history62/execution_01')}
SEEDS, ARMS = (64001, 64002, 64003), ('control', 'active')
SPEC = {
    'model_seeds': list(SEEDS), 'arms': list(ARMS), 'training_transitions': 7200,
    'label_sources': [58, 62], 'label_rows': 96, 'label_centers': 24,
    'state_windows': 6828, 'state_horizon': 8, 'chunk_size': 64, 'last_chunk': 44,
    'state_reduction': 'equal windows; F32 per-chunk parameter gradients accumulated in F64 with n/6828',
    'gradients': ['state', 'yaw', 'body', 'anchor'],
    'anchor': 'mean24 square((predicted yaw28 at original a0 - zero-perturbation target yaw28)/PHYSICAL_SCALE[28])',
    'objective_dtype': 'float32; physical zero truth separately retained float64',
    'geometry_dtype': 'float64', 'zero_norm_cosine': None,
    'value_report': 'all45 coordinates, seven frozen groups, yaw28 and body_vxy; all24 centers retained',
    'expected_forward_rows_per_model': 54840, 'expected_forward_calls_per_model': 859,
    'total_forward_rows': 329040, 'total_forward_calls': 5154,
    'state_forward_rows': 327744, 'actor_calls': 0, 'physics_steps': 0, 'updates': 0,
    'selection': 'no seed, center, epsilon or loss-weight search; diagnostics only',
    'interpretation': 'Euclidean local derivatives at final parameters, not Adam/clip prediction or proof of capacity',
    'candidate_automatically_authorized': False, 'source69_qualification_changed': False,
}


def required_inputs():
    paths = {str(PLAN69), str(SOURCE69 / 'report.json'), str(TRAIN), str(SUMMARY53),
             str(LABELS68 / 'report.json'), str(LABELS68 / 'body_labels96.npz'),
             'tools71/run71.py', 'tools71/audit71.py', 'tests/test_run71.py', 'tests/test_audit71.py',
             'docs/training_gradient71_plan_20261101.md',
             'tools52/dynamics52.py', 'tools53/run_pilot53.py', 'tools54/numerics54.py',
             'tools56/fit56.py', 'tools59/fit59.py', 'tools64/fit64.py', 'tools69/fit69.py'}
    for seed in SEEDS:
        for arm in ARMS:
            base = SOURCE69 / f'{arm}_seed{seed}'
            paths.update(str(base / name) for name in ('final_state_dict.pt', 'fit_receipt.json',
                                                       'final_body_training_diagnostic.json'))
            paths.update(str(base / f'label_diagnostic_{source}/final_train_labels.json') for source in ZERO_SOURCES)
    paths.update(str(directory / f'branch_{point:02d}_{branch:02d}.npz')
                 for directory in ZERO_SOURCES.values() for point in range(12) for branch in (0, 1))
    return paths


def verify71(plan):
    require(plan['schema'] == 'd1-training-gradient71-v1' and plan['decision_spec'] == SPEC, '71 contract changed')
    require(plan['cpu_seconds'] == 120 and plan['wall_seconds'] == 180, '71 budget changed')
    require(required_inputs() == set(plan['source_sha256']), '71 binding set differs')
    prior = json.loads((ROOT / PLAN69).read_text())['source_sha256']
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '71 source/input changed: ' + name)
        if name in prior:
            require(digest == prior[name], 'inherited69 binding differs: ' + name)
    original = json.loads((ROOT / SUMMARY53).read_text())['raw_manifest']['execution_01/train_transitions.npz']
    require(sha(ROOT / TRAIN) == original['sha256'] and (ROOT / TRAIN).stat().st_size == original['bytes'],
            'original53 training data differs')


def read_zero_targets(labels):
    import numpy as np
    targets = []
    for source_index, (source, directory) in enumerate(ZERO_SOURCES.items()):
        for point in range(12):
            rows = []
            for branch in (0, 1):
                with np.load(ROOT / directory / f'branch_{point:02d}_{branch:02d}.npz', allow_pickle=False) as raw:
                    rows.append({k: raw[k].copy() for k in ('s', 'applied_action', 'target')})
            for key, shape in (('s', (1, 45)), ('applied_action', (1, 16)), ('target', (1, 45))):
                require(all(row[key].shape == shape and row[key].dtype == np.float64
                            and np.isfinite(row[key]).all() for row in rows), 'invalid raw zero branch')
                require(np.array_equal(rows[0][key], rows[1][key]), 'repeated zero branches differ')
            start = source_index * 48 + point * 4
            require(np.all(labels['source_id'][start:start + 4] == int(source)), 'center source order differs')
            for k, raw_key in (('s', 's'), ('a0', 'applied_action')):
                require(np.array_equal(labels[k][start:start + 4], np.repeat(rows[0][raw_key], 4, axis=0)),
                        'zero branch not same complete label center')
            targets.append(rows[0]['target'][0])
    return np.stack(targets)


def complete_source(source):
    require(source['status'] == source['verification_after'] == 'COMPLETE'
            and source['completed_model_fits'] == 6 and source['model_updates'] == 6000,
            'six complete69 models required')
    require([(m['seed'], m['arm']) for m in source['models']] == [(s, a) for s in SEEDS for a in ARMS],
            'all six ordered models required')
    for m in source['models']:
        fit = m['fit']
        require(m['status'] == fit['status'] == 'COMPLETE' and fit['updates'] == fit['updates_attempted'] == 1000
                and fit['checkpoint_selection'] == 'final_only' and fit['eligible_for_evaluation'],
                'partial model cannot be audited as complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, wall = time.process_time(), time.monotonic()
    report = {'schema': 'd1-training-gradient71-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'models': [], 'completed_models': 0, 'model_updates': 0, 'policy_updates': 0,
              'actor_calls': 0, 'physics_steps': 0, 'source69_qualification_changed': False}
    write_json(args.output / 'initial_incomplete.json', report)
    events = (args.output / 'events.jsonl').open('x', buffering=1)
    plan = None
    def event(name, **detail):
        events.write(json.dumps({'event': name, 'cpu_seconds': time.process_time() - started, **detail}) + '\n')
    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (119, 120))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'single-thread CPU environment required')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify71(plan)
        import numpy as np
        import torch

        from tools52 import dynamics52
        from tools56 import fit56
        from tools69 import fit69
        from tools71.audit71 import audit_model
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report['versions'] = {k: importlib.metadata.version(k) for k in ('numpy', 'torch')}
        report['torch_runtime_version'] = torch.__version__
        require(report['versions'] == plan['versions'], 'registered package versions differ')
        source = json.loads((ROOT / SOURCE69 / 'report.json').read_text())
        complete_source(source)
        require(source['plan_sha256'] == sha(ROOT / PLAN69), 'source69 plan identity differs')
        require(all(report['versions'][k] == source['versions'][k] == source['versions_after'][k]
                    for k in report['versions']) and torch.__version__ == source['torch_runtime_version'],
                'runtime differs from original69')
        labels = fit69.load_labels68(ROOT / LABELS68)
        zero = read_zero_targets(labels)
        with np.load(ROOT / TRAIN, allow_pickle=False) as raw:
            train = dynamics52.validate_data({k: raw[k].copy() for k in fit56.DATA_KEYS})
        require(len(train['s']) == 7200 and np.array_equal(np.unique(train['episode']), np.arange(12)),
                'original complete training set required')
        training_hash, labels_hash = fit56.mapping_sha256(train), fit69.label_hashes(labels)
        for point in range(12):
            row = point * 4
            index = (int(labels['seed'][row]) - 53101) * 600 + int(labels['tick'][row])
            require(np.array_equal(train['s'][index], labels['s'][row])
                    and np.array_equal(train['a'][index], labels['a0'][row]), '58 center not original training transition')
        with (args.output / 'zero_targets24.npz').open('xb') as stream:
            np.savez_compressed(stream, target=zero, **{k: labels[k][::4] for k in ('s', 'a0', 'seed', 'tick', 'source_id')})
        for original in source['models']:
            seed, arm = original['seed'], original['arm']
            base = ROOT / SOURCE69 / original['directory']
            fit = original['fit']
            require(fit['training_data_sha256'] == training_hash and all(fit[k] == v for k, v in labels_hash.items()),
                    '69 training identities differ')
            weights = base / 'final_state_dict.pt'
            require(sha(weights) == source['sealed_artifacts_sha256'][original['directory'] + '/final_state_dict.pt'],
                    'weights not original sealed artifact')
            model = dynamics52.Dynamics52()
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            model.eval().requires_grad_(False)
            require(fit56.mapping_sha256(model.state_dict()) == fit['final_model_state_sha256'], 'final mapping differs')
            row = {'seed': seed, 'arm': arm, 'status': 'INCOMPLETE', 'source_model_sha256': fit['final_model_state_sha256']}
            report['models'].append(row)
            artifacts, progress = {}, {}
            event('MODEL_START', seed=seed, arm=arm)
            try:
                details, _ = audit_model(model, train, labels, zero, 118., arm=arm,
                                         artifacts=artifacts, progress=progress)
                row['diagnostic'] = details
                row['saved_training_responses'] = {
                    'yaw': {s: json.loads((base / f'label_diagnostic_{s}/final_train_labels.json').read_text())
                            for s in ZERO_SOURCES},
                    'body': json.loads((base / 'final_body_training_diagnostic.json').read_text())}
                require(row['saved_training_responses'] == {
                    'yaw': {s: original['final_training_label_diagnostics']['sources'][s] for s in ZERO_SOURCES},
                    'body': original['final_training_label_diagnostics']['body']}, 'saved69 training diagnostic changed')
                row['status'] = 'COMPLETE'
                report['completed_models'] += 1
                event('MODEL_COMPLETE', seed=seed, arm=arm)
            finally:
                row['progress'] = progress
                with (args.output / f'{arm}_seed{seed}.npz').open('xb') as stream:
                    np.savez_compressed(stream, **artifacts)
                write_json(args.output / f'{arm}_seed{seed}.json', row)
            del model
        require(report['completed_models'] == 6, 'incomplete71 models')
        report.update(status='COMPLETE')
    except BaseException as error:
        report.update(status='FAILED_OR_PARTIAL', failure=type(error).__name__ + ': ' + str(error))
        event('FAILED', failure=report['failure'])
        raise
    finally:
        late_failure = None
        report['verification_after'] = 'NOT_REACHED'
        if plan is not None:
            try:
                verify71(plan)
                require(sha(args.plan) == report['plan_sha256'], 'plan file changed during execution')
                report['versions_after'] = {k: importlib.metadata.version(k) for k in plan['versions']}
                require(report['versions_after'] == plan['versions'], 'packages changed during execution')
                if 'torch_runtime_version' in report:
                    require(sys.modules['torch'].__version__ == report['torch_runtime_version'], 'torch runtime changed')
                report['verification_after'] = 'COMPLETE'
            except BaseException as error:  # noqa: BLE001 - retain verification error while the original interruption propagates
                report['verification_after'] = 'FAILED'
                report['verification_failure'] = type(error).__name__ + ': ' + str(error)
                if report['status'] == 'COMPLETE':
                    late_failure = report['verification_failure']
                    report.update(status='FAILED_OR_PARTIAL', failure=late_failure)
        report.update(cpu_seconds=time.process_time() - started, wall_seconds=time.monotonic() - wall)
        write_json(args.output / 'report.json', report)
        events.close()
        if late_failure is not None:
            raise RuntimeError(late_failure)


if __name__ == '__main__':
    main()
