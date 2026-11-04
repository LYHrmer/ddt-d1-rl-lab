"""Independent83 entry with corrected cold source admission only.

The original zero-call failed execution/source/plan remain frozen. Geometry,
objectives, selected D points and the2016-call cap are copied unchanged.
--validate-only performs no tensor import or archive deserialization.

The two-halfspace Euclidean projection is an exact active-set calculation,
not randomized PCGrad. Feasibility alone is uninformative because zero is
always feasible. All gains and the row-L1 bound concern a local learned-model
proxy; they are neither a physical improvement nor a five-percent certificate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import resource
import signal
import struct
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SEEDS = (64001, 64002, 64003)
WHEEL_ROWS = (3, 7, 11, 15)
SOURCE75 = 'evidence/world_vz75/execution_01'
SOURCE80 = 'evidence/steady_sampling80/execution_01'
SOURCE81 = 'evidence/objective81/execution_01'
PLAN53 = 'docs/experiments53/model_gradient53_preregistered.json'
PLAN81 = 'docs/experiments81/objective81_preregistered.json'
SEAL80 = 'development_sealed_before_training.json'
SEAL81 = 'development_sealed_before_updates.json'
ACTOR_SHA = '8fb00b77019c6bce7247a5f3e9774e63064e48307f620fc56ac7f0ef02a35106'
KKT_ATOL, KKT_RTOL = 1e-12, 1e-10
CHECKED_SCHEMA = 'd1-policy-geometry83-checked-v1'
CHECKED_RESULT_SCHEMA = 'd1-policy-geometry83-checked-results-v1'
NEW_SOURCE_NAMES = frozenset({
    'tools83/policy_geometry83_checked.py', 'tests/test_policy_checked83.py',
    'tools83/source_admission83.py', 'tests/test_source_admission83.py',
    'docs/experiments83/policy_geometry83_preregistered.json',
    'evidence/policy_geometry83/execution_01/report.json', '.github/workflows/project83.yml',
})
SPEC = {
    'model_seeds': list(MODEL_SEEDS), 'source_model_arm': 'control',
    'yaw': {'seeds': [81107, 81108, 81109], 'ticks': [175, 325],
            'objective': 'policy81_Q6_mean_reset_max_positive_negative_yaw_mae'},
    'forward': {'seeds': [80101, 80102, 80103], 'ticks': [110, 260],
                'objective': 'mean_body_vx_mae_over_six_H32_starts'},
    'stop': {'seeds': list(range(80101, 80110)), 'ticks': [410],
             'objective': 'mean_frozen54_six_component_C_over_nine_post_motion_starts'},
    'horizon': 32, 'starts_per_model': 21, 'wheel_rows': list(WHEEL_ROWS),
    'head_coordinates': 516, 'candidate_step_descriptive_only': .009,
    'per_wheel_row_l1_bound': .01, 'kkt_atol': KKT_ATOL, 'kkt_rtol': KKT_RTOL,
    'model_forward_cap': 2016, 'actor_head_call_cap': 2016,
    'physics_steps': 0, 'model_updates': 0, 'policy_updates': 0,
    'cpu_seconds': 120, 'wall_seconds': 180, 'artifact_cap_bytes': 16 << 20,
    'finite_difference_or_endpoint_rollouts': 0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(values, name):
    require(isinstance(values, (list, tuple)) and values, name + ': nonempty vector required')
    require(all(type(v) in (int, float) for v in values), name + ': real scalars required')
    try:
        result = [float(v) for v in values]
    except OverflowError as exc:
        raise ValueError(name + ': finite binary64 required') from exc
    require(all(math.isfinite(v) for v in result), name + ': finite vector required')
    return result


def dot(a, b):
    require(len(a) == len(b), 'dot dimensions differ')
    value = math.fsum(x * y for x, y in zip(a, b, strict=True))
    require(math.isfinite(value), 'dot overflow')
    return value


def project_two_halfspaces(yaw, forward, stop):
    """Minimize .5 ||d + yaw||^2 subject to forward.d, stop.d <= 0.

    Positive rescaling of the constraint normals does not change the problem.
    Enumerate all four active sets, verify primal/dual/stationarity/complementary
    residuals, and retain the minimum certified objective. Singular double
    active sets are skipped; one-active candidates cover redundant/opposite
    parallel normals. Uncertified near-singular cases raise, never silently
    regularize or declare success. Tolerances certify arithmetic, not safety.
    """
    gradients = [_vector(v, n) for v, n in
                 ((yaw, 'yaw'), (forward, 'forward'), (stop, 'stop'))]
    yaw, forward, stop = gradients
    require(len({len(v) for v in gradients}) == 1, 'gradient dimensions differ')
    initial = [-v for v in yaw]
    norms = [math.sqrt(dot(v, v)) for v in (forward, stop)]
    normals = [[x / norm for x in row] if norm else [0.] * len(row)
               for row, norm in zip((forward, stop), norms, strict=True)]
    scale = max(math.sqrt(dot(initial, initial)), 1e-300)
    tolerance = KKT_ATOL + KKT_RTOL * scale
    available = [i for i, norm in enumerate(norms) if norm > 0.]
    candidates, rejected = [], []
    for size in range(3):
        for active in itertools.combinations(available, size):
            multipliers = [0., 0.]
            if size == 1:
                i = active[0]
                multipliers[i] = dot(normals[i], initial) / dot(normals[i], normals[i])
            elif size == 2:
                aa, bb, ab = dot(normals[0], normals[0]), dot(normals[1], normals[1]), dot(normals[0], normals[1])
                determinant = aa * bb - ab * ab
                if determinant <= 0.:
                    rejected.append({'active': list(active), 'reason': 'singular_gram'})
                    continue
                rhs0, rhs1 = dot(normals[0], initial), dot(normals[1], initial)
                multipliers = [(bb * rhs0 - ab * rhs1) / determinant,
                               (aa * rhs1 - ab * rhs0) / determinant]
            direction = [v - math.fsum(multipliers[i] * normals[i][j] for i in (0, 1))
                         for j, v in enumerate(initial)]
            require(all(math.isfinite(x) for x in (*direction, *multipliers)), 'projection overflow')
            constraint_dots = [dot(n, direction) for n in normals]
            stationarity = [d - d0 + math.fsum(multipliers[i] * normals[i][j] for i in (0, 1))
                            for j, (d, d0) in enumerate(zip(direction, initial, strict=True))]
            residual = {'primal': max(0., *constraint_dots),
                        'dual': max(0., *(-v for v in multipliers)),
                        'stationarity_linf': max(map(abs, stationarity)),
                        'complementarity': max(abs(m * c) for m, c in zip(multipliers, constraint_dots, strict=True))}
            passed = (max(residual[k] for k in ('primal', 'dual', 'stationarity_linf')) <= tolerance
                      and residual['complementarity'] <= tolerance * max(scale, 1.))
            if not passed:
                rejected.append({'active': list(active), 'reason': 'kkt_not_certified', 'residual': residual})
                continue
            delta = [d - d0 for d, d0 in zip(direction, initial, strict=True)]
            candidates.append({'active': list(active), 'direction': direction,
                'multipliers_unit_normals': multipliers, 'objective': .5 * dot(delta, delta),
                'constraint_dots_unit_normals': constraint_dots, 'kkt': residual})
    require(candidates, 'no numerically certified projection; geometry unresolved')
    best = min(candidates, key=lambda c: (c['objective'], len(c['active']), c['active']))
    projected = best['direction']
    norm2, yaw_norm2 = dot(projected, projected), dot(yaw, yaw)
    gain = -dot(yaw, projected)
    return {'status': 'COMPLETE', **best, 'constraint_norms': norms,
            'zero_constraints': [name for name, norm in zip(('forward', 'stop'), norms, strict=True) if norm == 0.],
            'raw_constraint_dots': [dot(forward, projected), dot(stop, projected)],
            'unconstrained_raw_constraint_dots': [dot(forward, initial), dot(stop, initial)],
            'numerical_tolerance': tolerance, 'rejected_active_sets': rejected,
            'yaw_linear_descent': gain, 'projected_norm': math.sqrt(norm2),
            'retained_unscaled_yaw_descent_ratio': gain / yaw_norm2 if yaw_norm2 else None,
            'cone_projection_identity_error': abs(gain - norm2),
            'strict_descent_resolved': gain > tolerance * max(math.sqrt(yaw_norm2), 1e-300),
            'zero_is_always_feasible': True}


def row_geometry(yaw, forward, stop, projection, *, row_width=129):
    """Normalize directions only; never construct an actor or change weights."""
    yaw, forward, stop = [_vector(v, name) for v, name in
                          ((yaw, 'yaw'), (forward, 'forward'), (stop, 'stop'))]
    require(type(row_width) is int and row_width > 0 and len(yaw) == 4 * row_width,
            'exactly four wheel rows required')
    require(len(forward) == len(stop) == len(yaw), 'gradient dimensions differ')
    projected = _vector(projection['direction'], 'projected')
    require(len(projected) == len(yaw), 'projected dimensions differ')
    rows = [yaw[i:i + row_width] for i in range(0, len(yaw), row_width)]
    upper = .01 * math.fsum(max(map(abs, row)) for row in rows)
    result = {'row_l1_bound': .01, 'descriptive_step': .009,
              'unconstrained_row_l1_linear_descent_upper_bound': upper,
              'upper_bound_is_local_proxy_only': True}
    for name, values in (('unconstrained', [-v for v in yaw]), ('projected', projected)):
        row_norms = [math.fsum(map(abs, values[i:i + row_width])) for i in range(0, len(values), row_width)]
        norm = max(row_norms)
        normalized = [v / norm for v in values] if norm else [0.] * len(values)
        gains = [dot(g, normalized) for g in (yaw, forward, stop)]
        unit_guard_dots = [dot([v / math.sqrt(dot(g, g)) for v in g], normalized)
                           if dot(g, g) else 0. for g in (forward, stop)]
        tolerance = KKT_ATOL + KKT_RTOL * math.sqrt(dot(normalized, normalized))
        resolved = norm > 0. and projection['strict_descent_resolved'] if name == 'projected' else norm > 0.
        result[name] = {'direction': normalized, 'normalization_max_row_l1': norm,
            'normalized_wheel_row_l1': [math.fsum(map(abs, normalized[i:i + row_width]))
                                      for i in range(0, len(values), row_width)],
            'yaw_forward_stop_directional_derivatives': gains,
            'predicted_yaw_reduction_at_0_009': -.009 * gains[0],
            'post_normalization_unit_guard_dots': unit_guard_dots,
            'post_normalization_kkt_tolerance': tolerance,
            'post_normalization_constraints_certified': max(unit_guard_dots) <= tolerance,
            'nonzero_direction_resolved': resolved}
    # Tiny raw projection residues must not acquire an apparently usable
    # direction when normalized to unit row-L1. Preserve the arrays but flag it.
    result['projected']['usable_local_descent'] = (
        result['projected']['nonzero_direction_resolved']
        and result['projected']['post_normalization_constraints_certified']
        and result['projected']['yaw_forward_stop_directional_derivatives'][0] < 0.)
    return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read(path):
    return json.loads(Path(path).read_text())


def _episodes():
    rows = [(SOURCE81, 'turn_stop', seed) for seed in (81107, 81108, 81109)]
    rows += [(SOURCE80, task, seed) for task, seeds in
             (('forward_stop', range(80101, 80104)), ('reverse_stop', range(80104, 80107)),
              ('turn_stop', range(80107, 80110))) for seed in seeds]
    return rows


def validate_episode_receipt(receipt, task, seed):
    require(receipt.get('status') == 'COMPLETE' and receipt.get('steps') == 600
            and receipt.get('task') == task and receipt.get('seed') == seed, 'partial/wrong D episode')
    counts = receipt.get('actual_counts', {})
    require(all(type(counts.get(k)) is int and counts[k] == value for k, value in
                (('policy_transitions', 600), ('native_steps', 6000), ('pd_updates', 3000), ('resets', 1))),
            'complete D collection counts required')


def _admission_helper():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools83 import source_admission83
    require(source_admission83.IMPLEMENTATION_PHASE == 'GREEN', 'source regression must be green first')
    return source_admission83


def _trusted_sources():
    admission = _admission_helper()
    parent = admission.read_pinned_json(ROOT / PLAN81, admission.PLAN81_SHA)
    archived = admission.read_pinned_json(ROOT / admission.FAILED83_PLAN, admission.FAILED83_PLAN_SHA)
    require(parent['schema'] == 'd1-objective81-preregistered-v2'
            and archived['schema'] == 'd1-policy-geometry83-v1', 'fixed original schemas differ')
    require(len(parent['source_sha256']) == 1390 and len(archived['source_sha256']) == 1446,
            'fixed original closure sizes differ')
    require(archived['spec'] == SPEC, 'science spec differs from original83')
    return parent, archived


def _parent_identity():
    admission = _admission_helper()
    return {'original81_plan_sha256': admission.PLAN81_SHA,
            'failed83_plan': admission.FAILED83_PLAN, 'failed83_plan_sha256': admission.FAILED83_PLAN_SHA,
            'correction': 'source_admission_only_exact_inherited_absolute_paths',
            'original_failure_status': 'INCOMPLETE', 'original_model_calls': 0, 'original_actor_calls': 0}


def required_inputs():
    _parent, archived = _trusted_sources()
    return set(archived['source_sha256']) | NEW_SOURCE_NAMES


def build_plan():
    """Build a separately sealed manifest; never mutate/reuse the failed plan."""
    _parent, archived = _trusted_sources()
    bindings = dict(archived['source_sha256'])
    for name in sorted(NEW_SOURCE_NAMES):
        require(name not in bindings, 'new registered input overlaps archived source')
        bindings[name] = sha(ROOT / name)
    return {'schema': CHECKED_SCHEMA, 'spec': copy.deepcopy(SPEC),
            'source_admission_parent': _parent_identity(), 'source_sha256': bindings}


def validate_structure(plan, parent, archived, registered_new_hashes):
    """Pure structural check; parent/archived must first pass fixed byte pins."""
    admission = _admission_helper()
    require(plan.get('schema') == CHECKED_SCHEMA and plan.get('spec') == SPEC
            and archived.get('spec') == SPEC, 'registered science spec or checked schema differs')
    require(plan.get('source_admission_parent') == _parent_identity(), 'source correction identity differs')
    require(set(registered_new_hashes) == NEW_SOURCE_NAMES, 'new-source whitelist differs')
    inherited = parent['source_sha256']
    original_additions = {k: v for k, v in archived['source_sha256'].items() if k not in inherited}
    require(all(archived['source_sha256'].get(k) == v for k, v in inherited.items()), 'archived parent rebinding')
    require(not (original_additions.keys() & registered_new_hashes.keys()), 'new source overlaps old additions')
    return admission.validate_manifest(plan['source_sha256'], inherited,
                                       original_additions | registered_new_hashes)


def _npy_header(stream):
    require(stream.read(6) == b'\x93NUMPY', 'invalid NPY magic')
    version = stream.read(2)
    require(len(version) == 2 and tuple(version) in ((1, 0), (2, 0), (3, 0)), 'unsupported NPY header')
    size_bytes = 2 if version[0] == 1 else 4
    raw_size = stream.read(size_bytes)
    require(len(raw_size) == size_bytes, 'truncated NPY length')
    size = struct.unpack('<H' if size_bytes == 2 else '<I', raw_size)[0]
    require(0 < size <= 65536, 'NPY header length exceeds bound')
    raw_header = stream.read(size)
    require(len(raw_header) == size, 'truncated NPY header')
    header = ast.literal_eval(raw_header.decode('utf-8' if version[0] == 3 else 'latin1'))
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'NPY metadata differs')
    return header


def _point_headers(path, count, include_height):
    shapes = {'state': ((45,), '<f8'), 'frame': ((57,), '<f4'), 'history': ((10, 57), '<f4'),
              'filtered': ((3,), '<f8'), 'last': ((16,), '<f4'), 'fifo': ((0, 16), '<f8'),
              'commands': ((33, 3), '<f8')}
    if include_height:
        shapes['nominal_height'] = ((), '<f8')
    expected = {f'{index}_{name}.npy' for index in range(count) for name in shapes}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == expected,
                'D point NPZ member grid differs')
        for index in range(count):
            for name, (shape, dtype) in shapes.items():
                with archive.open(f'{index}_{name}.npy') as stream:
                    header = _npy_header(stream)
                require(header == {'descr': dtype, 'fortran_order': False, 'shape': shape},
                        'D point shape/dtype metadata differs: ' + name)
    return len(expected)


def _cold_prerequisites(bindings):
    admission = _admission_helper()
    original = admission.read_pinned_json(ROOT / PLAN53, bindings[PLAN53])
    versions = admission.check_version_metadata(original['versions'])
    source = _read(ROOT / SOURCE75 / 'report.json')
    require(source['status'] == source['verification_after'] == 'COMPLETE'
            and source['completed_model_fits'] == 6, 'incomplete75 source')
    for seed in MODEL_SEEDS:
        old = [m for m in source['models'] if m['seed'] == seed and m['arm'] == 'control']
        require(len(old) == 1 and old[0]['status'] == old[0]['fit']['status'] == 'COMPLETE'
                and old[0]['fit']['updates'] == old[0]['fit']['updates_attempted'] == 1000,
                'partial/missing75 control checkpoint')
        member = old[0]['directory'] + '/final_state_dict.pt'
        require(bindings[SOURCE75 + '/' + member] == source['sealed_artifacts_sha256'][member],
                '75 checkpoint file binding differs')
    checkpoint = ROOT / original['checkpoint']
    require(checkpoint.is_file() and not checkpoint.is_symlink(), '25 checkpoint must be regular non-symlink')
    with zipfile.ZipFile(checkpoint) as archive:
        require(archive.namelist().count('policy.pth') == 1, '25 checkpoint policy member differs')
    seals = {}
    for base, sealname in ((SOURCE80, SEAL80), (SOURCE81, SEAL81)):
        report = _read(ROOT / base / 'report.json')
        require(report['status'] == report['verification_after'] == 'COMPLETE', 'partial D source')
        expected = report['development_seal_sha256'] if base == SOURCE80 else report['seals'][sealname]
        require(bindings[base + '/' + sealname] == expected, 'original D seal binding differs')
        seal = _read(ROOT / base / sealname)
        if base == SOURCE80:
            require(seal['actor_mapping_sha256'] == ACTOR_SHA, 'D actor mapping differs')
        seals[base] = seal['artifact_sha256']
    headers = 0
    for base, task, seed in _episodes():
        folder = f'development/{task}_{seed}'
        for name in ('receipt.json', 'point_arrays.npz', 'integration_context_archive.pkl', 'full_or_partial_trace.npz'):
            require(bindings[base + '/' + folder + '/' + name] == seals[base][folder + '/' + name],
                    'D original artifact binding differs')
        receipt = _read(ROOT / base / folder / 'receipt.json')
        validate_episode_receipt(receipt, task, seed)
        ticks = ([175, 325] if base == SOURCE81 else
                 [110, 175, 260, 325, 410] if task == 'turn_stop' else [110, 260, 410])
        require(receipt['points'] == [{'seed': seed, 'tick': tick, 'task': task} for tick in ticks],
                'D receipt point identities differ')
        require(receipt['trace_sha256'] == bindings[base + '/' + folder + '/full_or_partial_trace.npz'],
                'D trace binding differs')
        headers += _point_headers(ROOT / base / folder / 'point_arrays.npz', len(ticks), base == SOURCE81)
    failure = _read(ROOT / admission.FAILED83_REPORT)
    require(failure['status'] == 'INCOMPLETE' and failure['model_calls'] == failure['actor_calls'] == 0
            and failure['failure'] == 'ValueError: relative binding required'
            and failure['plan_sha256'] == admission.FAILED83_PLAN_SHA, 'original zero-call failure changed')
    return {'versions': versions, 'complete_D_episodes': 12, 'point_headers_checked': headers,
            'model_checkpoint_receipts_checked': 3, 'original_model_calls': 0, 'original_actor_calls': 0,
            'model_mapping_recomputed': False, 'pickle_or_tensor_deserialized': False}


def verify_plan(plan):
    """All metadata/file prerequisites run before any tensor import or pickle."""
    admission = _admission_helper()
    parent, archived = _trusted_sources()
    new_hashes = {name: sha(ROOT / name) for name in sorted(NEW_SOURCE_NAMES)}
    structure = validate_structure(plan, parent, archived, new_hashes)
    require(structure['inherited_absolute_count'] == 6, 'fixed inherited absolute set differs')
    new_names = set(plan['source_sha256']) - set(parent['source_sha256'])
    admission.check_file_hashes(plan['source_sha256'], ROOT, new_names)
    return {'manifest': structure, 'cold_prerequisites': _cold_prerequisites(plan['source_sha256'])}


def _execution_environment():
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
            'Nice10 and single CPU affinity required')
    require(all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'single-thread CPU/no GPU required')
    return {'nice': 10, 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'gpu_visible': False}


def _load_points(plan):
    """Only deserialize our locally generated, already SHA-admitted archives."""
    import numpy as np

    from tools59.run59 import load_own_points

    grouped = {'yaw': [], 'forward': [], 'stop': []}
    seals = {}
    for base, sealname in ((SOURCE80, SEAL80), (SOURCE81, SEAL81)):
        report = _read(ROOT / base / 'report.json')
        require(report['status'] == report['verification_after'] == 'COMPLETE', 'partial development source')
        expected = (report['development_seal_sha256'] if base == SOURCE80
                    else report['seals'][sealname])
        require(sha(ROOT / base / sealname) == expected, 'original development seal changed')
        seal = _read(ROOT / base / sealname)
        if base == SOURCE80:
            require(seal['actor_mapping_sha256'] == ACTOR_SHA, '80 development actor differs')
        seals[base] = seal['artifact_sha256']
    for base, task, seed in _episodes():
        relative = f'development/{task}_{seed}'
        folder = ROOT / base / relative
        for name in ('receipt.json', 'point_arrays.npz', 'integration_context_archive.pkl', 'full_or_partial_trace.npz'):
            require(sha(folder / name) == seals[base][relative + '/' + name], 'D member not originally sealed')
            require(sha(folder / name) == plan['source_sha256'][base + '/' + relative + '/' + name], 'D binding changed')
        receipt = _read(folder / 'receipt.json')
        validate_episode_receipt(receipt, task, seed)
        require(receipt['trace_sha256'] == sha(folder / 'full_or_partial_trace.npz'), 'D trace receipt differs')
        points = load_own_points(folder / 'integration_context_archive.pkl')
        expected_ticks = ([175, 325] if base == SOURCE81 else
                          [110, 175, 260, 325, 410] if task == 'turn_stop' else [110, 260, 410])
        require([(p['seed'], p['tick']) for p in points] == [(seed, t) for t in expected_ticks], 'D archive grid differs')
        with np.load(folder / 'point_arrays.npz', allow_pickle=False) as arrays:
            for index, point in enumerate(points):
                for key, shape in (('state', (45,)), ('frame', (57,)), ('history', (10, 57)),
                                   ('filtered', (3,)), ('last', (16,)), ('fifo', (0, 16)), ('commands', (33, 3))):
                    value = np.asarray(point[key])
                    require(value.shape == shape and np.isfinite(value).all()
                            and np.array_equal(value, arrays[f'{index}_{key}']), 'D numeric context differs: ' + key)
                require(math.isfinite(float(point['nominal_height'])), 'invalid nominal height')
                role = ('yaw' if base == SOURCE81 else 'stop' if point['tick'] == 410
                        else 'forward' if task == 'forward_stop' else None)
                if role:
                    expected_command = ([0., 0., .2 if point['tick'] == 175 else -.2] if role == 'yaw'
                                        else [.2, 0., 0.] if role == 'forward' else [0., 0., 0.])
                    require(np.array_equal(point['commands'], np.tile(expected_command, (33, 1))), 'D command interval differs')
                    grouped[role].append(point | {'task': task, 'status': 'COMPLETE', 'source': base})
    for role, count in (('yaw', 6), ('forward', 6), ('stop', 9)):
        grouped[role].sort(key=lambda p: (p['seed'], p['tick']))
        expected = [(seed, tick) for seed in SPEC[role]['seeds'] for tick in SPEC[role]['ticks']]
        require(len(grouped[role]) == count and [(p['seed'], p['tick']) for p in grouped[role]] == expected,
                'complete registered D role required: ' + role)
    return grouped


def _audit_model(model, anchor, groups, row, folder, report, cpu_deadline, wall_deadline):
    import torch

    from tools54 import numerics54 as numeric
    from tools56.fit56 import mapping_sha256
    from tools79.policy79 import (
        _gradient_hash,
        _head,
        _protected_hash,
        _validate_frozen,
    )
    from tools81.policy81 import _objective_summary, objective_values

    _validate_frozen(model)
    _validate_frozen(anchor)
    originals = {'model': model, 'actor': anchor}
    original_hashes = {k: mapping_sha256(v.state_dict()) for k, v in originals.items()}
    gradient_hashes = {k: _gradient_hash(v) for k, v in originals.items()}
    work_model, actor = copy.deepcopy(model).double(), copy.deepcopy(anchor).double()
    head = _head(actor).requires_grad_(True)
    require(isinstance(actor.policy_net[-1], torch.nn.Tanh), 'bounded policy latent required for row-L1 claim')
    work_hashes = {'model': mapping_sha256(work_model.state_dict()), 'actor': mapping_sha256(actor.state_dict())}
    protected = _protected_hash(actor)
    hooks = []

    def before(kind, module, inputs):
        del module, inputs
        if time.process_time() >= cpu_deadline or time.monotonic() >= wall_deadline:
            raise TimeoutError('83 CPU/wall deadline reached')
        require(report[kind + '_calls_attempted'] < 2016, kind + ' call cap reached')
        report[kind + '_calls_attempted'] += 1

    def after(kind, module, inputs, output):
        del module, inputs, output
        report[kind + '_calls'] += 1

    def save_point(role, index, value, trace, point, extra=None):
        item = {'role': role, 'index': index, 'seed': point['seed'], 'tick': point['tick'],
                'task': point['task'], 'source': point['source'], 'status': 'COMPLETE',
                'objective': float(value.detach()), 'frozen54_cost': float(trace['costs'].mean().detach()),
                'cost_components_mean': trace['cost_components'].detach().mean(0).tolist(),
                'clipped_actions': int((trace['actions'].detach().abs() >= 1.).sum()), **(extra or {})}
        row['points'].append(item)
        _json(folder / f'{role}_{index:02d}.json', item)

    row.update(source_mapping_sha256=original_hashes, source_gradient_sha256=gradient_hashes,
               protected_actor_sha256=protected, parameter_order='wheel rows3,7,11,15; each128 weights then bias')
    try:
        for module, kind in ((work_model, 'model'), (head, 'actor')):
            hooks.append(module.register_forward_pre_hook(lambda m, x, k=kind: before(k, m, x)))
            hooks.append(module.register_forward_hook(lambda m, x, y, k=kind: after(k, m, x, y)))

        def save_yaw(index, point_row):
            save_point('yaw', index, point_row['yaw_mae'], point_row['trace'], groups['yaw'][index],
                       {'exact_zero_errors': int((point_row['yaw_error'].detach() == 0.).sum())})

        yaw = objective_values(work_model, actor, groups['yaw'], cpu_deadline=cpu_deadline, on_point=save_yaw)
        row['yaw_definition_receipt'] = _objective_summary(yaw)
        objectives = {'yaw': yaw['Q6']}
        for role in ('forward', 'stop'):
            values = []
            for index, point in enumerate(groups[role]):
                value, trace = numeric.imaginary(work_model, actor, point, torch.zeros((), dtype=torch.float64),
                                                 return_trace=True, horizon=32)
                require(all(bool(torch.isfinite(trace[key]).all()) for key in ('states', 'actions', 'costs')),
                        'nonfinite83 imagined trace')
                extra = {}
                if role == 'forward':
                    body = numeric._rotate_inverse(trace['states'][:, 3:7], trace['states'][:, 23:26])
                    commands = torch.as_tensor(point['commands'], dtype=torch.float64)
                    error = body[:, 0] - commands[:32, 0]
                    value = error.abs().mean()
                    extra = {'exact_zero_errors': int((error.detach() == 0.).sum()),
                             'minimum_absolute_error': float(error.detach().abs().min())}
                values.append(value)
                save_point(role, index, value, trace, point, extra)
            objectives[role] = torch.stack(values).mean()
        gradients = {}
        for role, value in objectives.items():
            require(value.dtype == torch.float64 and bool(torch.isfinite(value)), 'finite F64 objective required')
            weight, bias = torch.autograd.grad(value, (head.weight, head.bias))
            gradients[role] = torch.cat((weight[list(WHEEL_ROWS)], bias[list(WHEEL_ROWS), None]), dim=1).detach().reshape(-1).tolist()
            _vector(gradients[role], role)
        row['objectives'] = {key: float(value.detach()) for key, value in objectives.items()}
        row['gradients'] = gradients
        row['projection'] = project_two_halfspaces(gradients['yaw'], gradients['forward'], gradients['stop'])
        row['row_geometry'] = row_geometry(gradients['yaw'], gradients['forward'], gradients['stop'], row['projection'])
        q = row['objectives']['yaw']
        row['upper_bound_div_current_Q6'] = (
            row['row_geometry']['unconstrained_row_l1_linear_descent_upper_bound'] / q if q > 0. else None)
        row['gradient_norms'] = {k: math.sqrt(dot(v, v)) for k, v in gradients.items()}
        row['gradient_cosines'] = {a + '_' + b: (dot(gradients[a], gradients[b]) /
            (row['gradient_norms'][a] * row['gradient_norms'][b])
            if row['gradient_norms'][a] * row['gradient_norms'][b] else None)
            for a, b in itertools.combinations(gradients, 2)}
        row['status'] = 'COMPLETE'
    finally:
        for handle in hooks:
            handle.remove()
        unchanged = ({k: mapping_sha256(v.state_dict()) for k, v in originals.items()} == original_hashes
                     and {k: _gradient_hash(v) for k, v in originals.items()} == gradient_hashes
                     and mapping_sha256(work_model.state_dict()) == work_hashes['model']
                     and mapping_sha256(actor.state_dict()) == work_hashes['actor']
                     and _protected_hash(actor) == protected)
        row['verification_after'] = 'COMPLETE' if unchanged else 'INCOMPLETE'
        if not unchanged:
            row['status'] = 'INCOMPLETE'
        _json(folder / 'receipt.json', row)
        require(unchanged, '83 parameters/buffers/source gradients changed')


def validate_only(plan_path, output):
    """Run the actual checked entry's source admission without tensor imports."""
    for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
        sys.path.insert(0, str(directory))
    from tools74.runtime74 import run_preserving

    plan_path, output = Path(plan_path), Path(output)
    plan_sha = sha(plan_path)
    plan = _read(plan_path)
    live = {}

    def no_numeric_modules():
        require(not any(name.split('.')[0] in ('torch', 'numpy', 'mujoco') for name in sys.modules),
                'cold validation must not import Torch/NumPy/MuJoCo')

    def body(report):
        nonlocal live
        live = report
        report.update(schema='d1-policy-geometry83-checked-admission-v1', plan_sha256=plan_sha,
                      source_compatible=False, model_updates=0, policy_updates=0,
                      model_calls_attempted=0, actor_calls_attempted=0, formal_audit_run=False,
                      scientific_question='UNANSWERED', original_execution_status='INCOMPLETE',
                      global_candidate_accepted=False, physical_validation='NOT_RUN')
        no_numeric_modules()
        report['execution_environment'] = _execution_environment()
        report['admission'] = verify_plan(plan)
        no_numeric_modules()
        report['source_compatible'] = True

    def after():
        try:
            require(sha(plan_path) == plan_sha, 'checked admission plan changed')
            verify_plan(plan)
            no_numeric_modules()
        except BaseException:
            # Revoke provisional source qualification on late failure.
            live['source_compatible'] = False
            raise

    def stop(_signum, _frame):
        raise TimeoutError('checked cold admission wall budget')

    previous = signal.signal(signal.SIGALRM, stop)
    signal.alarm(60)
    try:
        report = run_preserving(output, body, after)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return report


def audit(plan_path, output):
    """Execute once under the root-owned serial resource guard; preserve failure."""
    for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
        sys.path.insert(0, str(directory))
    from tools74.runtime74 import run_preserving

    plan_path, output = Path(plan_path), Path(output)
    original_plan_sha = sha(plan_path)
    plan = _read(plan_path)
    cpu_started, wall_started = time.process_time(), time.monotonic()
    live_report = {}

    def verify_after():
        require(sha(plan_path) == original_plan_sha, '83 plan changed during run')
        verify_plan(plan)
        for name, digest in live_report.get('artifact_sha256', {}).items():
            require(sha(output / name) == digest, '83 generated artifact changed: ' + name)
        require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) < SPEC['artifact_cap_bytes'],
                '83 artifact cap exceeded')

    def body(report):
        nonlocal live_report
        live_report = report
        report.update(schema=CHECKED_RESULT_SCHEMA, plan_sha256=original_plan_sha,
                      spec=copy.deepcopy(SPEC), model_calls_attempted=0, actor_calls_attempted=0,
                      model_updates=0, policy_updates=0, global_candidate_accepted=False,
                      physical_validation='NOT_RUN', diagnostics_only=True, models=[],
                      actor_calls_semantics='action_head forward calls, including each imagined policy step')
        report['source_admission_before'] = verify_plan(plan)
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10 and single CPU affinity required')
        require(all(os.environ.get(k) == '1' for k in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'single-thread CPU/no GPU required')
        source, original = _read(ROOT / SOURCE75 / 'report.json'), _read(ROOT / PLAN53)
        require(source['status'] == source['verification_after'] == 'COMPLETE'
                and source['completed_model_fits'] == 6, 'complete75 source required')
        versions = {key: importlib.metadata.version(key) for key in original['versions']}
        require(versions == original['versions'], '83 runtime metadata differs')
        report['versions'] = versions
        import torch

        from tools52 import adapter52, dynamics52
        from tools56.fit56 import mapping_sha256

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == source['torch_runtime_version'], '83 Torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        anchor = adapter52.load_actor(ROOT / original['checkpoint'])
        require(mapping_sha256(anchor.state_dict()) == ACTOR_SHA, 'original25 actor mapping differs')
        report['actor_mapping_sha256'] = ACTOR_SHA
        groups = _load_points(plan)
        report['point_identities'] = {k: [{'seed': p['seed'], 'tick': p['tick'], 'task': p['task'], 'source': p['source']}
                                        for p in points] for k, points in groups.items()}
        for seed in MODEL_SEEDS:
            old = [m for m in source['models'] if m['seed'] == seed and m['arm'] == 'control']
            require(len(old) == 1, 'one75 control model per registered initialization required')
            old = old[0]
            require(old['status'] == old['fit']['status'] == 'COMPLETE'
                    and old['fit']['updates'] == old['fit']['updates_attempted'] == 1000, 'partial source checkpoint')
            relative = old['directory'] + '/final_state_dict.pt'
            path = ROOT / SOURCE75 / relative
            require(sha(path) == source['sealed_artifacts_sha256'][relative], '75 weight artifact differs')
            model = dynamics52.Dynamics52()
            model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
            model.eval().requires_grad_(False)
            require(mapping_sha256(model.state_dict()) == old['fit']['final_model_state_sha256'], '75 model mapping differs')
            row = {'seed': seed, 'status': 'INCOMPLETE', 'points': [],
                   'source_weight_file_sha256': sha(path), 'source_path': str(path.relative_to(ROOT))}
            report['models'].append(row)
            folder = output / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            _json(folder / 'initial_incomplete.json', row)
            _audit_model(model, anchor, groups, row, folder, report, cpu_started + 110., wall_started + 170.)
        require(report['model_calls'] == report['model_calls_attempted'] == 2016
                and report['actor_calls'] == report['actor_calls_attempted'] == 2016, '83 forward accounting differs')
        require(all(row['status'] == row['verification_after'] == 'COMPLETE' for row in report['models']),
                'all three model audits must complete')
        report['artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        report['interpretation'] = ('Local learned-model D-gradient geometry only. Zero is always feasible. '
            'No actor/model update, finite endpoint, physical validation or complete-candidate requalification. '
            'Stop means post-motion zero-command protection, not independent cold-start standing. '
            'Abs/max objectives can be nonsmooth; selected AD subgradients are not certified physical derivatives.')

    return run_preserving(output, body, verify_after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true',
                        help='cold source/spec/hash/metadata admission only; zero model calls')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30) if args.validate_only else (119, 120))
    report = validate_only(args.plan, args.output) if args.validate_only else audit(args.plan, args.output)
    print(json.dumps({'status': report['status'], 'verification_after': report.get('verification_after'),
                      'validate_only': args.validate_only,
                      'model_calls': report['model_calls'], 'actor_calls': report['actor_calls'],
                      'physics_steps': report['physics_steps'], 'updates': report['updates'],
                      'models_completed': sum(m['status'] == 'COMPLETE' for m in report.get('models', [])),
                      'report': str(args.output / 'report.json')}, allow_nan=False))
    return 0 if report['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
