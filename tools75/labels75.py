"""Admit archived world-vz responses without model, actor or physics calls.

All original68 arrays remain byte-for-byte identical in dtype, shape and row
order. The additional target is raw world-frame state25, not rotated body-vz.
Rejected channels remain in the report; only complete admission publishes the
trainable96-row NPZ. Source verification precedes that publication.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools74.geometry74 import EPSILONS, scalar_qualification
from tools74.runtime74 import run_preserving

SOURCES = {'58': 'evidence/train_action_response58/collection_01',
           '62': 'evidence/clean_history62/execution_01'}
SEEDS, TICKS = (53103, 53106, 53109, 53112), (110, 260, 410)
PLAN68 = 'docs/experiments68/body_response68_preregistered.json'
REPORT68 = 'evidence/body_response68/admission_01/report.json'
LABEL68 = 'evidence/body_response68/admission_01/body_labels96.npz'
PLAN68_SHA = 'b8ca3c2a3d89e9dda50c6b9df263910a0f7d2f722c9e00993e3ba641d956405e'
LABEL_FILE = 'world_vz_labels96.npz'
ORIGINAL_FIELDS = ('s', 'a0', 'a_plus', 'a_minus', 'v', 'seed', 'tick', 'epsilon',
                   'target_yaw', 'target_body', 'source_id', 'source_point_index', 'body_channel_qualified')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_inputs():
    paths = {PLAN68, REPORT68, LABEL68, 'tools75/labels75.py', 'tests/test_labels75.py',
             'tools74/geometry74.py', 'tools74/runtime74.py', 'tools68/labels68.py', 'tools55/probe_local55.py'}
    for folder in SOURCES.values():
        paths.add(folder + '/report.json')
        paths.update(f'{folder}/response_{i:02d}.npz' for i in range(12))
        paths.update(f'{folder}/branch_{i:02d}_{b:02d}.npz' for i in range(12) for b in range(10))
    return paths


def plan_spec():
    return {'schema': 'd1-world-vz75-labels-plan-v1', 'sources': list(SOURCES),
            'train_seeds': list(SEEDS), 'ticks': list(TICKS), 'eps': list(EPSILONS),
            'coordinate': 25, 'frame': 'world', 'expected_states': 24, 'expected_label_rows': 96,
            'cpu_seconds': 30, 'wall_seconds': 60, 'numpy_version': '2.2.6',
            'model_calls': 0, 'actor_calls': 0, 'physics_steps': 0, 'updates': 0,
            'qualification': 'tools74.geometry74.scalar_qualification; all four eps and all four wheels',
            'retry_permitted': False, 'resume_permitted': False, 'overwrite_permitted': False}


def build_plan(root=ROOT):
    root = Path(root)
    return {**plan_spec(), 'source_sha256': {name: sha(root / name) for name in sorted(required_inputs())}}


def verify(plan, root=ROOT):
    root = Path(root)
    require({k: v for k, v in plan.items() if k != 'source_sha256'} == plan_spec(), 'labels75 protocol changed')
    require(set(plan['source_sha256']) == required_inputs(), 'labels75 bindings must contain exactly its independent inputs')
    require(sha(root / PLAN68) == PLAN68_SHA, 'frozen68 plan changed')
    old = json.loads((root / PLAN68).read_text())
    for name, digest in plan['source_sha256'].items():
        require(sha(root / name) == digest, 'labels75 input changed: ' + name)
        if name in old['source_sha256']:
            require(digest == old['source_sha256'][name], 'original68 input binding changed: ' + name)
        if any(name.startswith(folder + '/') for folder in SOURCES.values()):
            require(name in old['source_sha256'], 'raw input lacks original68 binding: ' + name)


def read_npz(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key].copy() for key in raw.files}


def validate_original(labels):
    import numpy as np
    require(tuple(labels) == ORIGINAL_FIELDS, 'original68 field order changed')
    widths = {'s': 45, 'a0': 16, 'a_plus': 16, 'a_minus': 16, 'v': 16,
              'target_body': 2, 'body_channel_qualified': 2}
    for key, value in labels.items():
        dtype = np.float32 if key == 'v' else np.int64 if key in (
            'seed', 'tick', 'source_id', 'source_point_index') else np.bool_ if key == 'body_channel_qualified' else np.float64
        shape = (96, widths[key]) if key in widths else (96,)
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'original68 array differs: ' + key)
    expected = [(int(source), seed, tick, epsilon, i) for source in SOURCES
                for i, (seed, tick) in enumerate((s, t) for s in SEEDS for t in TICKS) for epsilon in EPSILONS]
    actual = list(zip(*(labels[key].tolist() for key in ('source_id', 'seed', 'tick', 'epsilon', 'source_point_index')), strict=True))
    require(actual == expected, 'all96 original rows in source/state/epsilon order required')
    require(len(np.unique(labels['s'], axis=0)) == 24 and labels['body_channel_qualified'].all(), 'original68 admission differs')
    for start in range(0, 96, 4):
        require(all(np.array_equal(labels[k][start:start + 4], np.repeat(labels[k][start:start + 1], 4, axis=0))
                    for k in ('s', 'a0')), 'four-epsilon center differs')
    return labels


def action_ulps(reference, perturbed):
    """Frozen55 ULP definition, evaluated without importing its model dependencies."""
    import numpy as np
    reference, perturbed = np.asarray(reference, dtype=np.float32), np.asarray(perturbed, dtype=np.float32)
    require(reference.shape == perturbed.shape == (16,), 'action16 required')
    require(np.isfinite(reference).all() and np.isfinite(perturbed).all(), 'nonfinite action')
    delta = perturbed.astype(np.float64) - reference.astype(np.float64)
    ulp = np.spacing(np.abs(reference)).astype(np.float64)
    ratios, nonzero = np.abs(delta) / ulp, delta != 0
    wheels = np.array([3, 7, 11, 15])
    return {'reference': reference.tolist(), 'perturbed': perturbed.tolist(), 'delta': delta.tolist(),
            'reference_float32_ulp': ulp.tolist(), 'delta_ulps': ratios.tolist(),
            'nonzero_count': int(nonzero.sum()), 'wheel_nonzero_count': int(nonzero[wheels].sum()),
            'all_four_wheels_nonzero': bool(nonzero[wheels].all()),
            'min_nonzero_delta_ulps': float(ratios[nonzero].min()) if nonzero.any() else None,
            'maxabs': float(np.abs(delta).max())}


def read_point(folder, source, index, labels, old, row):
    """Mutate the current report row incrementally so catchable failure keeps it."""
    import numpy as np
    folder = Path(folder)
    require(old['status'] == 'COMPLETE' and old['label_admitted'] and len(old['branches']) == 10,
            'complete admitted training source point required')
    deltas = [0., 0., *(sign * e for e in EPSILONS for sign in (1, -1))]
    require([b['delta'] for b in old['branches']] == deltas and all(
        b['status'] == 'COMPLETE' and b['issued_equals_applied'] for b in old['branches']), 'complete original action grid required')
    branches = []
    for order in range(10):
        data = read_npz(folder / f'branch_{index:02d}_{order:02d}.npz')
        branch = {key: data[key] for key in ('s', 'target', 'applied_action', 'issued_action')}
        for key, width in (('s', 45), ('target', 45), ('applied_action', 16), ('issued_action', 16)):
            require(branch[key].shape == (1, width) and branch[key].dtype == np.float64
                    and np.isfinite(branch[key]).all(), 'raw finite F64 branch required: ' + key)
        require(np.array_equal(branch['applied_action'], branch['issued_action']), 'issued/applied action differs')
        require(np.array_equal(branch['applied_action'], branch['applied_action'].astype(np.float32))
                and np.abs(branch['applied_action']).max() <= 1., 'actual clipped F32 action required')
        branches.append(branch)
        row['branches_read'] = len(branches)
    require(all(np.array_equal(branches[0][k], branches[1][k]) for k in branches[0])
            and all(b['zero_exact'] for b in old['branches'][:2]), 'duplicate zero branch differs')
    require(all(np.array_equal(b['s'], branches[0]['s']) for b in branches), 'branches must share one pre-state')
    raw = read_npz(folder / f'response_{index:02d}.npz')
    keys = ('central_state45_fd', 'forward_state45_fd', 'backward_state45_fd')
    require(all(raw[k].shape == (4, 45) and raw[k].dtype == np.float64 and np.isfinite(raw[k]).all() for k in keys),
            'complete original F64 response grid required')
    a0, zero = branches[0]['applied_action'][0], branches[0]['target'][0]
    direction = np.zeros(16, dtype=np.float64)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    offset = (0 if source == '58' else 48) + 4 * index
    for ep, epsilon in enumerate(EPSILONS):
        plus, minus = branches[2 + 2 * ep:4 + 2 * ep]
        for key, value in zip(keys, ((plus['target'][0] - minus['target'][0]) / (2 * epsilon),
                                    (plus['target'][0] - zero) / epsilon, (zero - minus['target'][0]) / epsilon), strict=True):
            require(np.array_equal(value, raw[key][ep]), 'original raw45 endpoint FD differs: ' + key)
            row['raw_fd_scalar_checks'] += 45
        n = offset + ep
        for key, value in (('s', branches[0]['s'][0]), ('a0', a0),
                           ('a_plus', plus['applied_action'][0]), ('a_minus', minus['applied_action'][0])):
            require(np.array_equal(labels[key][n], value), 'original68 center/action differs: ' + key)
        require(labels['target_yaw'][n] == raw['central_state45_fd'][ep, 28], 'original68 yaw target differs')
        require(old['yaw28'][ep] == {'epsilon': epsilon, **{name: float(raw[key][ep, 28])
                for name, key in zip(('central', 'forward', 'backward'), keys, strict=True)}}, 'old yaw JSON differs')
        action_rows = []
        for sign, branch, order in ((1, plus, 2 + 2 * ep), (-1, minus, 3 + 2 * ep)):
            action = branch['applied_action'][0]
            require(np.array_equal(action, np.clip(a0 + sign * epsilon * direction, -1., 1.).astype(np.float32)),
                    'absolute wheel action perturbation differs')
            ulps = action_ulps(a0, action)
            require(ulps == old['branches'][order]['actual_action_delta'], 'archived action ULP evidence differs')
            action_rows.append(ulps)
        row['action_checks'].append({'epsilon': epsilon, 'plus': action_rows[0], 'minus': action_rows[1]})
        v = ((plus['applied_action'][0] - minus['applied_action'][0]) / (2 * epsilon)).astype(np.float32)
        require(np.array_equal(v, labels['v'][n]), 'actual-action F64-difference v32 differs')
        require(np.array_equal((plus['applied_action'][0] + minus['applied_action'][0]) / 2, a0), 'action midpoint differs')
        row['eps'].append({'epsilon': epsilon, **{name: float(raw[key][ep, 25])
                         for name, key in zip(('central', 'forward', 'backward'), keys, strict=True)}})
    row['qualification'] = scalar_qualification(row['eps'])
    row['all_four_wheels_nonzero'] = all(side['all_four_wheels_nonzero']
        for check in row['action_checks'] for side in (check['plus'], check['minus']))
    require(row['all_four_wheels_nonzero'] == old['all_four_wheels_nonzero'], 'old wheel qualification differs')
    row.update(status='COMPLETE', qualified=row['qualification']['stable'] and row['all_four_wheels_nonzero'])
    return raw['central_state45_fd'][:, 25].copy()


def admission(points):
    expected = [(source, seed, tick) for source in SOURCES for seed in SEEDS for tick in TICKS]
    require([(p['source'], p['seed'], p['tick']) for p in points] == expected
            and all(p['status'] == 'COMPLETE' and len(p['eps']) == 4 for p in points), 'all24 complete ordered states required')
    require(all(p['qualification'] == scalar_qualification(p['eps']) and p['qualified'] == (
        p['qualification']['stable'] and p['all_four_wheels_nonzero']) for p in points), 'stored qualification disagrees with full evidence')
    qualified = sum(p['qualified'] for p in points)
    return {'status': 'ADMITTED' if qualified == 24 else 'REJECTED_WORLD_VZ_QUALIFICATION',
            'admitted': qualified == 24, 'qualified_states': qualified, 'required_states': 24,
            'rows_dropped': 0, 'label_rows': 96, 'selected_epsilon': None,
            'frame': 'world', 'coordinate': 25, 'fits_authorized_by_this_result': 0}


def assemble_labels(original, points, targets):
    import numpy as np
    validate_original(original)
    require(admission(points)['admitted'], 'all24 world-vz states must be admitted before publishing labels')
    require(len(targets) == 24 and all(v.shape == (4,) and v.dtype == np.float64 and np.isfinite(v).all()
                                    for v in targets), '24 raw F64 target grids required')
    require(all(np.array_equal(target, [row['central'] for row in point['eps']])
                for target, point in zip(targets, points, strict=True)), 'world-vz target differs from admitted response')
    labels = {key: value.copy() for key, value in original.items()}
    labels['target_world_vz'] = np.concatenate(targets)
    labels['world_vz_channel_qualified'] = np.repeat(np.asarray([p['qualified'] for p in points], dtype=np.bool_), 4)
    require(all(labels[k].dtype == original[k].dtype and labels[k].shape == original[k].shape
                and labels[k].tobytes() == original[k].tobytes() for k in original), 'original68 arrays changed')
    return labels


def calculate(report, root=ROOT, *, wall_deadline=None):
    root = Path(root)
    old68 = json.loads((root / REPORT68).read_text())
    require(old68['status'] == old68['verification_after'] == 'COMPLETE' and old68['admission']['admitted']
            and old68['admission']['qualified_state_channels'] == 48 and old68['label_rows'] == 96,
            'complete admitted68 source required')
    require(sha(root / LABEL68) == old68['label_file_sha256'], '68 label file differs from its complete report')
    labels = validate_original(read_npz(root / LABEL68))
    targets = []
    for source, folder in SOURCES.items():
        old = json.loads((root / folder / 'report.json').read_text())
        require(old['status'] == old['verification_after'] == 'COMPLETE' and old['labels_admitted'] == 12
                and len(old['points']) == 12, 'complete58/62 source required')
        for index, (seed, tick) in enumerate((s, t) for s in SEEDS for t in TICKS):
            require(time.process_time() < 25., 'labels75 CPU budget exhausted')
            require(wall_deadline is None or time.monotonic() < wall_deadline, 'labels75 wall budget exhausted')
            require((old['points'][index]['seed'], old['points'][index]['tick']) == (seed, tick), 'source identity/order differs')
            row = {'source': source, 'seed': seed, 'tick': tick, 'status': 'INCOMPLETE',
                   'eps': [], 'branches_read': 0, 'raw_fd_scalar_checks': 0, 'action_checks': []}
            report['points'].append(row)
            targets.append(read_point(root / folder, source, index, labels, old['points'][index], row))
            report['completed_points'] += 1
    result = admission(report['points'])
    result['admitted'] = False
    if result['qualified_states'] == 24:
        result['status'] = 'PENDING_FINAL_VERIFICATION'
    report['admission'] = result
    report['label_rows'] = 96
    report['original68_fields_preserved'] = list(labels)
    report['raw_fd_scalar_checks'] = sum(p['raw_fd_scalar_checks'] for p in report['points'])
    report['wheel_branch_checks'] = sum(2 * len(p['action_checks']) for p in report['points'])
    # Publication and admitted=true occur only after the final verification.
    return labels, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    context = {}

    def body(report):
        context['report'] = report
        report.update(schema='d1-world-vz75-labels-v1', completed_points=0, label_rows=0,
                      label_file_sha256=None, admission={'status': 'INCOMPLETE', 'admitted': False,
                      'qualified_states': 0, 'required_states': 24, 'rows_dropped': 0},
                      source_decisions_changed=False, heldout_rows_read=0, checkpoints_read=0)
        resource.setrlimit(resource.RLIMIT_CPU, (28, 30))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1, 'Nice10/singleCPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'single-thread/no-GPU environment required')
        context['started_wall'] = time.monotonic()
        context['plan'] = json.loads(args.plan.read_text())
        context['plan_sha'] = report['plan_sha256'] = sha(args.plan)
        verify(context['plan'])
        report['numpy_version'] = importlib.metadata.version('numpy')
        require(report['numpy_version'] == context['plan']['numpy_version'], 'original NumPy version required')
        context['labels'], context['targets'] = calculate(report, wall_deadline=context['started_wall'] + 55.)

    def after():
        report = context.get('report')
        if report is not None and report['status'] != 'COMPLETE':
            report['admission'].update(status='INCOMPLETE', admitted=False)
        try:
            require('plan' in context, 'labels75 plan not loaded')
            verify(context['plan'])
            require(sha(args.plan) == context['plan_sha'], 'labels75 plan changed')
            require(importlib.metadata.version('numpy') == context['plan']['numpy_version'], 'NumPy version changed')
            require(time.monotonic() - context['started_wall'] < 55., 'labels75 wall budget exhausted')
            if report['status'] == 'COMPLETE':
                result = admission(report['points'])
                if result['admitted']:
                    import numpy as np
                    labels = assemble_labels(context['labels'], report['points'], context['targets'])
                    path = args.output / LABEL_FILE
                    with path.open('xb') as stream:
                        np.savez_compressed(stream, **labels)
                    restored = read_npz(path)
                    require(list(restored) == list(labels) and all(restored[k].dtype == v.dtype
                            and restored[k].shape == v.shape and restored[k].tobytes() == v.tobytes()
                            for k, v in labels.items()), 'saved label arrays differ')
                    report['label_file_sha256'] = sha(path)
                report['admission'] = result
        except BaseException:
            if report is not None:
                report['admission'].update(status='INCOMPLETE', admitted=False)
            raise

    report = run_preserving(args.output, body, after)
    print(json.dumps({k: report.get(k) for k in ('status', 'failure', 'admission', 'label_rows', 'cpu_seconds')}, indent=2))
    return int(report['status'] != 'COMPLETE' or not report['admission']['admitted'])


if __name__ == '__main__':
    raise SystemExit(main())
