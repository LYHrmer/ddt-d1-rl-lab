"""Independent stdlib audit of the separately sealed checked83 readback.

Never imports the producer, Torch, NumPy, pickle, a model or a simulator.
KKT/duality certify the saved convex projection without rerunning its active-set
solver. Hook counts are cross-checked against the complete point ledger, not
re-observed at runtime. Original83's zero-call failure stays INCOMPLETE. Only
the six exact absolute names inherited from the byte-pinned81 plan are allowed.
All physical/performance qualifications remain absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import time
from pathlib import Path

ROOT = Path('/home/lyh/ddt_ros2_ws')
DEFAULT_SOURCE = ROOT / 'evidence/policy_geometry83/execution_checked_01'
MODEL_SEEDS = (64001, 64002, 64003)
SOURCE75 = 'evidence/world_vz75/execution_01'
SOURCE80 = 'evidence/steady_sampling80/execution_01'
SOURCE81 = 'evidence/objective81/execution_01'
PLAN81 = 'docs/experiments81/objective81_preregistered.json'
PLAN81_SHA = '987db250df14869b9f8ce72ab3522d3356de646233341c9a66c4bbccb3d6cf20'
FAILED83_PLAN = 'docs/experiments83/policy_geometry83_preregistered.json'
FAILED83_PLAN_SHA = '616f93d1d337cea4392710b2e66edd05926773c8ed5294e7596c82be1ac3bcce'
FAILED83_REPORT = 'evidence/policy_geometry83/execution_01/report.json'
CHECKED_PLAN = 'docs/experiments83/policy_geometry83_checked_preregistered.json'
CHECKED_PLAN_SHA = '2ddfec95f472022ad24139d930009a54bc2605541ff81b92e1a250d6031eeee3'
REGISTERED_NEW_NAMES = frozenset({
    'tools83/policy_geometry83_checked.py', 'tests/test_policy_checked83.py',
    'tools83/source_admission83.py', 'tests/test_source_admission83.py',
    FAILED83_PLAN, FAILED83_REPORT, '.github/workflows/project83.yml',
})
ACTOR_SHA = '8fb00b77019c6bce7247a5f3e9774e63064e48307f620fc56ac7f0ef02a35106'
EMPTY_MAPPING_SHA = hashlib.sha256().hexdigest()
ATOL, RTOL = 1e-12, 1e-10
OUTPUT_CAP = 2 << 20


def require(ok, message):
    if not ok:
        raise ValueError(message)


def scalar(value, name):
    require(type(value) in (int, float), name + ': real scalar required')
    value = float(value)
    require(math.isfinite(value), name + ': nonfinite')
    return value


def vector(value, size, name):
    require(isinstance(value, list) and len(value) == size, name + ': wrong vector shape')
    return [scalar(v, name) for v in value]


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON value')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def dot(left, right):
    require(len(left) == len(right), 'dot dimensions differ')
    return scalar(math.fsum(a * b for a, b in zip(left, right, strict=True)), 'dot')


def norm(row):
    return math.hypot(*row)


def expected_points():
    return {
        'yaw': [{'seed': seed, 'tick': tick, 'task': 'turn_stop', 'source': SOURCE81}
                for seed in (81107, 81108, 81109) for tick in (175, 325)],
        'forward': [{'seed': seed, 'tick': tick, 'task': 'forward_stop', 'source': SOURCE80}
                    for seed in (80101, 80102, 80103) for tick in (110, 260)],
        'stop': [{'seed': seed, 'tick': 410, 'task': task, 'source': SOURCE80}
                 for task, seeds in (('forward_stop', range(80101, 80104)),
                                     ('reverse_stop', range(80104, 80107)),
                                     ('turn_stop', range(80107, 80110))) for seed in seeds],
    }


def verify_development_files(check, bindings):
    """Hash only; never deserialize a pickle or load numerical arrays."""
    episodes = {(p['source'], p['task'], p['seed']) for rows in expected_points().values() for p in rows}
    seals = {}
    for base, seal_name in ((SOURCE80, 'development_sealed_before_training.json'),
                            (SOURCE81, 'development_sealed_before_updates.json')):
        require(base + '/report.json' in bindings and base + '/' + seal_name in bindings,
                'original development provenance missing from plan')
        report = check.load(ROOT / base / 'report.json', bindings[base + '/report.json'])
        require(report['status'] == report['verification_after'] == 'COMPLETE', 'incomplete D source')
        digest = report['development_seal_sha256'] if base == SOURCE80 else report['seals'][seal_name]
        require(bindings[base + '/' + seal_name] == digest, 'D seal binding differs from original report')
        seals[base] = check.load(ROOT / base / seal_name, digest)['artifact_sha256']
    for base, task, seed in sorted(episodes):
        folder = f'development/{task}_{seed}'
        for name in ('receipt.json', 'point_arrays.npz', 'integration_context_archive.pkl', 'full_or_partial_trace.npz'):
            relative = folder + '/' + name
            require(base + '/' + relative in bindings, 'D source member omitted from plan')
            require(bindings[base + '/' + relative] == seals[base][relative], 'D plan/original member SHA differs')
            check.bind(ROOT / base / relative, seals[base][relative])
        receipt = check.load(ROOT / base / folder / 'receipt.json')
        require(receipt['status'] == 'COMPLETE' and receipt['steps'] == 600
                and receipt['task'] == task and receipt['seed'] == seed, 'partial/wrong D episode')
        require(all(receipt['actual_counts'][key] == value for key, value in
                    (('policy_transitions', 600), ('native_steps', 6000), ('pd_updates', 3000), ('resets', 1))),
                'D collection counts incomplete')
        require(receipt['trace_sha256'] == seals[base][folder + '/full_or_partial_trace.npz'], 'D trace receipt differs')
    return len(episodes)


def verify_checked_sources(check, plan, raw):
    """Pin81 before interpreting any inherited absolute name; no imports."""
    parent = check.load(ROOT / PLAN81, PLAN81_SHA)
    archived = check.load(ROOT / FAILED83_PLAN, FAILED83_PLAN_SHA)
    inherited = parent['source_sha256']
    original = archived['source_sha256']
    require(parent['schema'] == 'd1-objective81-preregistered-v2'
            and archived['schema'] == 'd1-policy-geometry83-v1', 'fixed parent schema differs')
    require(len(inherited) == 1390 and len(original) == 1446, 'original closure count differs')
    require(archived['spec'] == plan['spec'] == raw['spec'], 'scientific spec changed during source correction')
    expected_parent = {
        'original81_plan_sha256': PLAN81_SHA, 'failed83_plan': FAILED83_PLAN,
        'failed83_plan_sha256': FAILED83_PLAN_SHA,
        'correction': 'source_admission_only_exact_inherited_absolute_paths',
        'original_failure_status': 'INCOMPLETE', 'original_model_calls': 0, 'original_actor_calls': 0,
    }
    require(plan['source_admission_parent'] == expected_parent, 'checked parent identity differs')
    bindings = plan['source_sha256']
    require(set(bindings) == set(original) | REGISTERED_NEW_NAMES and len(bindings) == 1453,
            'checked source whitelist contains an extra/missing binding')
    require(all(original.get(name) == value for name, value in inherited.items()), 'archived81 closure rebinding')
    require(all(bindings.get(name) == value for name, value in original.items()), 'checked archived-source rebinding')
    allowed_absolute = {name: value for name, value in inherited.items() if Path(name).is_absolute()}
    require(len(allowed_absolute) == 6, 'fixed81 absolute set must contain exactly six paths')
    actual_absolute = {name: value for name, value in bindings.items() if Path(name).is_absolute()}
    require(actual_absolute == allowed_absolute, 'new/rebound absolute path rejected')
    for name, digest in bindings.items():
        require(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
                and set(digest) <= set('0123456789abcdef'), 'invalid source path/digest')
        path = Path(name)
        if path.is_absolute():
            require(name in allowed_absolute and digest == allowed_absolute[name], 'absolute source not inherited exactly')
        else:
            require(name not in ('', '.') and '\\' not in name and '\0' not in name
                    and '..' not in path.parts and path.as_posix() == name, 'unsafe relative source binding')
            if name not in inherited:
                require((ROOT / path).resolve().is_relative_to(ROOT), 'new relative source resolves outside repository')
        check.bind(ROOT / name, digest)
    failure = check.load(ROOT / FAILED83_REPORT, bindings[FAILED83_REPORT])
    require(failure['status'] == 'INCOMPLETE' and failure['model_calls'] == failure['actor_calls'] == 0
            and failure['plan_sha256'] == FAILED83_PLAN_SHA
            and failure['failure'] == 'ValueError: relative binding required', 'original83 failure was replaced')
    admission = raw['source_admission_before']
    manifest = admission['manifest']
    require(manifest == {'inherited_entries': 1390, 'registered_new_entries': 63,
                        'total_entries': 1453, 'inherited_absolute_count': 6,
                        'inherited_absolute_paths': sorted(allowed_absolute)}, 'producer source admission ledger differs')
    cold = admission['cold_prerequisites']
    require(cold['complete_D_episodes'] == 12 and cold['point_headers_checked'] == 279
            and cold['model_checkpoint_receipts_checked'] == 3
            and cold['original_model_calls'] == cold['original_actor_calls'] == 0
            and cold['model_mapping_recomputed'] is False and cold['pickle_or_tensor_deserialized'] is False,
            'producer cold prerequisite receipt differs')
    plan53_name = 'docs/experiments53/model_gradient53_preregistered.json'
    versions = check.load(ROOT / plan53_name, inherited[plan53_name])['versions']
    require(cold['versions'] == raw['versions'] == versions, 'recorded package metadata differs')
    check.report['preserved_original_failure'] = {key: failure[key] for key in
                                                 ('status', 'failure', 'model_calls', 'actor_calls', 'plan_sha256')}
    check.report['absolute_sources_admitted_by_fixed81'] = allowed_absolute
    check.report['checked_parent_pins'] = {PLAN81: PLAN81_SHA, FAILED83_PLAN: FAILED83_PLAN_SHA,
                                         CHECKED_PLAN: CHECKED_PLAN_SHA}


class Audit:
    def __init__(self, report, cpu_start, wall_start):
        self.report = report
        self.cpu_start, self.wall_start = cpu_start, wall_start

    def deadline(self):
        if time.process_time() - self.cpu_start >= 25 or time.monotonic() - self.wall_start >= 55:
            raise TimeoutError('independent readback CPU/wall budget')

    def digest(self, path):
        self.deadline()
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
                self.deadline()
        return digest.hexdigest()

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        digest = self.digest(path)
        previous = self.report['inputs_sha256'].get(str(path))
        require(previous is None or previous == digest, 'input changed while reading: ' + str(path))
        require(expected is None or expected == digest, 'input SHA mismatch: ' + str(path))
        self.report['inputs_sha256'][str(path)] = digest
        return digest

    def load(self, path, expected=None):
        self.bind(path, expected)
        result = json.loads(Path(path).read_text())
        finite_tree(result)
        return result

    def close(self, actual, expected, label):
        actual, expected = scalar(actual, label), scalar(expected, label)
        gap = abs(actual - expected)
        self.report['scalar_checks'] += 1
        self.report['maximum_comparison_absolute_gap'] = max(
            self.report['maximum_comparison_absolute_gap'], gap)
        require(gap <= ATOL + RTOL * abs(expected), f'{label}: inconsistent saved value ({gap!r})')

    def array_close(self, actual, expected, label):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': shape differs')
        for a, b in zip(actual, expected, strict=True):
            self.close(a, b, label)

    def optional_close(self, actual, expected, label):
        if expected is None:
            require(actual is None, label + ': expected undefined zero-denominator result')
        else:
            self.close(actual, expected, label)


def audit_geometry(check, row, summary):
    gradients = {role: vector(row['gradients'][role], 516, role) for role in ('yaw', 'forward', 'stop')}
    require(set(row['gradients']) == set(gradients), 'unexpected gradient role')
    yaw, forward, stop = (gradients[k] for k in ('yaw', 'forward', 'stop'))
    norms = {k: norm(v) for k, v in gradients.items()}
    for key, value in norms.items():
        check.close(row['gradient_norms'][key], value, 'gradient norm')
    for a, b in (('yaw', 'forward'), ('yaw', 'stop'), ('forward', 'stop')):
        denominator = norms[a] * norms[b]
        value = dot(gradients[a], gradients[b]) / denominator if denominator else None
        check.optional_close(row['gradient_cosines'][a + '_' + b], value, 'gradient cosine')

    projected = row['projection']
    require(projected['status'] == 'COMPLETE' and projected['zero_is_always_feasible'] is True,
            'projection receipt incomplete')
    direction = vector(projected['direction'], 516, 'projected direction')
    multipliers = vector(projected['multipliers_unit_normals'], 2, 'KKT multipliers')
    unit = [[v / norms[role] for v in gradients[role]] if norms[role] else [0.] * 516
            for role in ('forward', 'stop')]
    raw_constraints = [dot(forward, direction), dot(stop, direction)]
    constraints = [dot(normal, direction) for normal in unit]
    original = [-v for v in yaw]
    original_constraints = [dot(forward, original), dot(stop, original)]
    stationarity = [direction[j] + yaw[j] + math.fsum(m * n[j] for m, n in zip(multipliers, unit, strict=True))
                    for j in range(516)]
    residuals = {'primal': max(0., *constraints), 'dual': max(0., *(-v for v in multipliers)),
                 'stationarity_linf': max(map(abs, stationarity)),
                 'complementarity': max(abs(m * c) for m, c in zip(multipliers, constraints, strict=True))}
    tolerance = ATOL + RTOL * max(norms['yaw'], 1e-300)
    check.close(projected['numerical_tolerance'], tolerance, 'KKT tolerance')
    for key, value in residuals.items():
        check.close(projected['kkt'][key], value, 'KKT residual ' + key)
    require(max(residuals[k] for k in ('primal', 'dual', 'stationarity_linf')) <= tolerance,
            'saved projection lacks primal/dual/stationarity certificate')
    require(residuals['complementarity'] <= tolerance * max(norms['yaw'], 1.),
            'saved projection lacks complementarity certificate')
    active = projected['active']
    require(isinstance(active, list) and active == sorted(set(active)) and set(active) <= {0, 1},
            'invalid active set')
    require(all(abs(multipliers[i]) <= tolerance for i in (0, 1) if i not in active), 'inactive multiplier nonzero')
    check.array_close(projected['constraint_norms'], [norms['forward'], norms['stop']], 'constraint norms')
    check.array_close(projected['raw_constraint_dots'], raw_constraints, 'raw guard derivative')
    check.array_close(projected['constraint_dots_unit_normals'], constraints, 'unit guard derivative')
    check.array_close(projected['unconstrained_raw_constraint_dots'], original_constraints, 'original guard derivative')
    require(projected['zero_constraints'] == [k for k in ('forward', 'stop') if norms[k] == 0.],
            'zero constraint classification differs')

    # Independent convex dual certificate, without calling/enumerating the
    # producer's projection solver. A small certified primal-dual gap verifies
    # global optimality for this local quadratic problem, not robot behaviour.
    correction = [math.fsum(m * n[j] for m, n in zip(multipliers, unit, strict=True)) for j in range(516)]
    shift = [d + g for d, g in zip(direction, yaw, strict=True)]
    primal_value = .5 * dot(shift, shift)
    dual_value = dot(multipliers, [dot(n, original) for n in unit]) - .5 * dot(correction, correction)
    dual_gap = primal_value - dual_value
    require(abs(dual_gap) <= 8 * tolerance * max(norms['yaw'], 1.), 'projection primal-dual gap unresolved')
    check.close(projected['objective'], primal_value, 'projection objective')
    descent = -dot(yaw, direction)
    pnorm = norm(direction)
    retained = descent / dot(yaw, yaw) if dot(yaw, yaw) else None
    identity_error = abs(descent - dot(direction, direction))
    check.close(projected['yaw_linear_descent'], descent, 'raw projected yaw descent')
    check.close(projected['projected_norm'], pnorm, 'projection norm')
    check.optional_close(projected['retained_unscaled_yaw_descent_ratio'], retained, 'retained raw descent')
    check.close(projected['cone_projection_identity_error'], identity_error, 'cone identity error')
    resolved = descent > tolerance * max(norms['yaw'], 1e-300)
    require(projected['strict_descent_resolved'] is resolved, 'raw descent qualification differs')

    geometry = row['row_geometry']
    require(geometry['row_l1_bound'] == .01 and geometry['descriptive_step'] == .009
            and geometry['upper_bound_is_local_proxy_only'] is True, 'row bound or interpretation differs')
    upper = .01 * math.fsum(max(abs(v) for v in yaw[i:i + 129]) for i in range(0, 516, 129))
    check.close(geometry['unconstrained_row_l1_linear_descent_upper_bound'], upper, 'row-L1 dual upper bound')
    q = scalar(row['objectives']['yaw'], 'current Q6')
    upper_fraction = upper / q if q > 0. else None
    check.optional_close(row['upper_bound_div_current_Q6'], upper_fraction, 'upper bound fraction')
    directions = {}
    for name, raw in (('unconstrained', original), ('projected', direction)):
        entry = geometry[name]
        row_lengths = [math.fsum(abs(v) for v in raw[i:i + 129]) for i in range(0, 516, 129)]
        scaling = max(row_lengths)
        normalized = [v / scaling for v in raw] if scaling else [0.] * 516
        check.array_close(entry['direction'], normalized, 'normalized ' + name)
        check.close(entry['normalization_max_row_l1'], scaling, 'normalization ' + name)
        normalized_lengths = [math.fsum(abs(v) for v in normalized[i:i + 129]) for i in range(0, 516, 129)]
        check.array_close(entry['normalized_wheel_row_l1'], normalized_lengths, 'normalized row L1')
        derivatives = [dot(gradients[k], normalized) for k in ('yaw', 'forward', 'stop')]
        check.array_close(entry['yaw_forward_stop_directional_derivatives'], derivatives, 'normalized derivatives')
        normalized_guards = [dot(n, normalized) for n in unit]
        normalized_tolerance = ATOL + RTOL * norm(normalized)
        check.array_close(entry['post_normalization_unit_guard_dots'], normalized_guards, 'normalized guards')
        check.close(entry['post_normalization_kkt_tolerance'], normalized_tolerance, 'normalized tolerance')
        certified = max(normalized_guards) <= normalized_tolerance
        nonzero = scaling > 0. and (resolved if name == 'projected' else True)
        require(entry['post_normalization_constraints_certified'] is certified
                and entry['nonzero_direction_resolved'] is nonzero, 'normalized classification differs')
        decrease = -.009 * derivatives[0]
        check.close(entry['predicted_yaw_reduction_at_0_009'], decrease, 'descriptive .009 yaw change')
        if name == 'projected':
            require(entry['usable_local_descent'] is (nonzero and certified and derivatives[0] < 0.),
                    'local descent availability differs')
        directions[name] = {
            'yaw_forward_stop_derivatives': derivatives,
            'descriptive_0_009_changes': [.009 * value for value in derivatives],
            'yaw_reduction_fraction_of_current_Q6': decrease / q if q > 0. else None,
            'guard_positive_raw_signs': [value > 0. for value in derivatives[1:]],
            'normalized_constraints_certified': certified,
            'nonzero_direction_resolved': nonzero,
        }
    original_decrease = -directions['unconstrained']['descriptive_0_009_changes'][0]
    projected_decrease = -directions['projected']['descriptive_0_009_changes'][0]
    summary.update(
        status='COMPLETE', gradient_shape=[3, 516], gradient_norms=norms,
        source_mapping_sha256=row['source_mapping_sha256'],
        objectives=row['objectives'], raw_unconstrained_guard_dots=original_constraints,
        raw_unconstrained_guard_conflicts={name: value > 0. for name, value in
                                           zip(('forward', 'stop'), original_constraints, strict=True)},
        projected_raw_guard_dots=raw_constraints, active_constraints=active,
        independently_computed_kkt=residuals, kkt_tolerance=tolerance,
        primal_value=primal_value, dual_value=dual_value, primal_dual_gap=dual_gap,
        retained_unscaled_yaw_descent_ratio=retained,
        normalized_yaw_reduction_retention_ratio=(projected_decrease / original_decrease
                                                 if original_decrease else None),
        normalized_yaw_reduction_lost_at_0_009=original_decrease - projected_decrease,
        directions=directions, unconstrained_0_01_row_l1_linear_yaw_upper_bound=upper,
        upper_bound_fraction_of_current_Q6=upper_fraction,
        projected_usable_local_descent=geometry['projected']['usable_local_descent'],
        local_proxy_only=True, physical_evidence='NOT_RUN', performance_certificate=False,
    )


def run(check, source, plan_path):
    result = check.report
    raw = check.load(source / 'report.json')
    require(raw['schema'] == 'd1-policy-geometry83-checked-results-v1', 'wrong checked source schema')
    require(raw['status'] == raw['verification_after'] == 'COMPLETE' and raw.get('failure') is None,
            'partial/failed source cannot be audited as complete')
    require(raw['global_candidate_accepted'] is False and raw['diagnostics_only'] is True
            and raw['physical_validation'] == 'NOT_RUN', 'physical/performance qualification was changed')
    require(raw['plan_sha256'] == CHECKED_PLAN_SHA, 'source does not use the separately sealed checked plan')
    plan = check.load(plan_path, CHECKED_PLAN_SHA)
    require(plan['schema'] == 'd1-policy-geometry83-checked-v1' and plan['spec'] == raw['spec'],
            'checked plan/report spec differs')
    spec = plan['spec']
    for key, expected in {'model_seeds': list(MODEL_SEEDS), 'source_model_arm': 'control',
                          'horizon': 32, 'starts_per_model': 21, 'head_coordinates': 516,
                          'wheel_rows': [3, 7, 11, 15], 'model_forward_cap': 2016,
                          'actor_head_call_cap': 2016, 'per_wheel_row_l1_bound': .01,
                          'candidate_step_descriptive_only': .009, 'kkt_atol': ATOL, 'kkt_rtol': RTOL,
                          'physics_steps': 0, 'model_updates': 0, 'policy_updates': 0,
                          'finite_difference_or_endpoint_rollouts': 0, 'cpu_seconds': 120,
                          'wall_seconds': 180}.items():
        require(spec[key] == expected, 'registered scope differs: ' + key)
    require(raw['point_identities'] == expected_points(), 'source points are not the fixed D-only grid')
    for key in ('model_calls', 'model_calls_attempted', 'actor_calls', 'actor_calls_attempted'):
        require(type(raw[key]) is int and raw[key] == 2016, 'actual hook accounting differs: ' + key)
    for key in ('physics_steps', 'updates', 'model_updates', 'policy_updates'):
        require(type(raw[key]) is int and raw[key] == 0, 'nonzero work outside audit: ' + key)
    require(0. <= scalar(raw['cpu_seconds'], 'CPU') <= 120.
            and 0. <= scalar(raw['wall_seconds'], 'wall') <= 180., 'producer resource budget exceeded')
    require(raw['actor_mapping_sha256'] == ACTOR_SHA, 'actor mapping differs')

    verify_checked_sources(check, plan, raw)
    result['complete_D_episodes_verified'] = verify_development_files(check, plan['source_sha256'])
    source75 = check.load(ROOT / SOURCE75 / 'report.json')
    require(source75['status'] == source75['verification_after'] == 'COMPLETE', 'incomplete75 checkpoint provenance')
    require(raw['torch_runtime_version'] == source75['torch_runtime_version'], 'recorded Torch build differs')
    required_artifacts = {'initial_incomplete.json'}
    for seed in MODEL_SEEDS:
        required_artifacts |= {f'seed{seed}/initial_incomplete.json', f'seed{seed}/receipt.json'}
        required_artifacts |= {f'seed{seed}/{role}_{i:02d}.json'
                               for role, points in expected_points().items() for i in range(len(points))}
    require(set(raw['artifact_sha256']) == required_artifacts, 'unexpected/missing formal artifacts')
    for name, digest in raw['artifact_sha256'].items():
        check.bind(source / name, digest)
    result['source_verification_before'] = 'COMPLETE'
    require([row['seed'] for row in raw['models']] == list(MODEL_SEEDS), 'missing/duplicate/reordered model initialization')
    point_count = 0
    for row in raw['models']:
        seed = row['seed']
        summary = {'seed': seed, 'status': 'INCOMPLETE', 'points_verified': 0}
        result['models'].append(summary)
        require(row['status'] == row['verification_after'] == 'COMPLETE', 'partial model audit')
        receipt = check.load(source / f'seed{seed}/receipt.json')
        require(receipt == row, 'final model receipt/report row differs')
        old = [m for m in source75['models'] if m['seed'] == seed and m['arm'] == 'control']
        require(len(old) == 1, 'one original75 control model required')
        old = old[0]
        require(old['status'] == old['fit']['status'] == 'COMPLETE'
                and old['fit']['updates'] == old['fit']['updates_attempted'] == 1000, 'partial75 model')
        expected_path = SOURCE75 + '/' + old['directory'] + '/final_state_dict.pt'
        require(row['source_path'] == expected_path, 'wrong source model path')
        file_sha = check.bind(ROOT / expected_path, row['source_weight_file_sha256'])
        require(file_sha == source75['sealed_artifacts_sha256'][old['directory'] + '/final_state_dict.pt'],
                'published75 weight file SHA differs')
        require(row['source_mapping_sha256'] == {'model': old['fit']['final_model_state_sha256'], 'actor': ACTOR_SHA},
                'published/source mapping identity differs')
        require(row['source_gradient_sha256'] == {'model': EMPTY_MAPPING_SHA, 'actor': EMPTY_MAPPING_SHA},
                'source parameter gradients not empty')
        require(row['parameter_order'] == 'wheel rows3,7,11,15; each128 weights then bias', 'gradient coordinates differ')
        expected = [(role, index, point) for role, points in expected_points().items()
                    for index, point in enumerate(points)]
        require(len(row['points']) == len(expected) == 21, 'incomplete model point grid')
        grouped_values = {'yaw': [], 'forward': [], 'stop': []}
        for point, (role, index, identity) in zip(row['points'], expected, strict=True):
            require(point['status'] == 'COMPLETE' and point['role'] == role and point['index'] == index
                    and all(point[k] == v for k, v in identity.items()), 'non-D/partial/wrong point identity')
            saved = check.load(source / f'seed{seed}/{role}_{index:02d}.json')
            require(saved == point, 'point JSON/report differs')
            objective = scalar(point['objective'], 'point objective')
            require(objective >= 0., 'negative nonnegative objective')
            grouped_values[role].append(objective)
            costs = vector(point['cost_components_mean'], 6, 'cost components')
            check.close(point['frozen54_cost'], math.fsum(costs), 'frozen54 component mean sum')
            point_count += 1
            summary['points_verified'] += 1
        q = [max(grouped_values['yaw'][i:i + 2]) for i in (0, 2, 4)]
        check.close(row['objectives']['yaw'], math.fsum(q) / 3., 'Q6 reduction')
        check.array_close(row['yaw_definition_receipt']['per_seed_q'], q, 'per-reset Q6')
        check.close(row['yaw_definition_receipt']['Q6'], math.fsum(q) / 3., 'Q6 definition receipt')
        for role in ('forward', 'stop'):
            check.close(row['objectives'][role], math.fsum(grouped_values[role]) / len(grouped_values[role]), role + ' mean')
        audit_geometry(check, row, summary)
    require(point_count == 63 and all(r['status'] == 'COMPLETE' for r in result['models']), 'complete three-model audit required')
    result.update(points_verified=point_count, gradients_verified=9, gradient_scalars_verified=9 * 516,
                  original_artifacts_verified=len(required_artifacts),
                  source_binding_count=len(plan['source_sha256']),
                  producer_cpu_seconds=raw['cpu_seconds'], producer_wall_seconds=raw['wall_seconds'],
                  call_accounting={'recorded_completed_model_calls': raw['model_calls'],
                                   'recorded_attempted_model_calls': raw['model_calls_attempted'],
                                   'recorded_completed_actor_head_calls': raw['actor_calls'],
                                   'recorded_attempted_actor_head_calls': raw['actor_calls_attempted'],
                                   'independent_point_horizon_count': point_count * 32,
                                   'original_failed_execution_model_calls': 0,
                                   'original_failed_execution_actor_calls': 0,
                                   'cumulative_completed_model_calls': raw['model_calls'],
                                   'cumulative_completed_actor_head_calls': raw['actor_calls'],
                                   'meaning': 'saved actual hook receipts agree with all63 fixed H32 points; no runtime replay'},
                  original_candidate_status_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--plan', type=Path, default=ROOT / CHECKED_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'output exists; no overwrite/resume')
    initial = args.output.with_name(args.output.name + '.initial.json')
    cpu_start, wall_start = time.process_time(), time.monotonic()
    result = {'schema': 'policy-geometry83-checked-independent-audit-v1', 'status': 'INCOMPLETE',
              'verification_after': 'NOT_RUN', 'failure': None, 'inputs_sha256': {}, 'models': [],
              'scalar_checks': 0, 'maximum_comparison_absolute_gap': 0.,
              'new_model_calls': 0, 'new_actor_calls': 0, 'new_physics_steps': 0, 'new_updates': 0,
              'global_candidate_accepted': False,
              'interpretation_limits': [
                  'Independent audit of saved gradients and convex geometry, not a new AD/FD or physical check.',
                  'Parameter file hashes and published mapping receipts are checked; tensor mappings are not recomputed.',
                  'A feasible zero vector is not evidence of useful descent; all three initializations are retained.',
                  'The .009 response and .01 row-L1 bound are local linear proxies, not a physical5% certificate.',
                  'Stop refers to post-motion D histories, not independently sampled cold-start standing.',
                  'Shared D resets are not multiplied by three model initializations into independent physical seeds.',
                  'Original83 zero-call failure remains INCOMPLETE; this checks only the separately sealed checked execution.',
                  'Allowed absolute sources are precisely the six names and hashes inherited from pinned81; no blanket exception.',
              ]}
    check = Audit(result, cpu_start, wall_start)
    script_sha = check.bind(Path(__file__).resolve())
    with initial.open('x') as stream:
        json.dump({'status': 'INCOMPLETE', 'script_sha256': script_sha}, stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

    def stop(signum, _frame):
        raise InterruptedError('audit interrupted: signal ' + str(signum))

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU, signal.SIGALRM):
        signal.signal(signum, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    signal.alarm(60)
    try:
        run(check, args.source.resolve(), args.plan.resolve())
        result['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve completed prefix on catchable termination.
        result['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            for name, digest in result['inputs_sha256'].items():
                require(check.digest(name) == digest, 'input changed after readback: ' + name)
            result['verification_after'] = 'COMPLETE'
        except BaseException as exc:  # noqa: BLE001 - late identity failure revokes completion.
            result['status'] = 'INCOMPLETE'
            result['verification_after'] = 'INCOMPLETE'
            result['verification_failure'] = f'{type(exc).__name__}: {exc}'
        result['cpu_seconds'] = time.process_time() - cpu_start
        result['wall_seconds'] = time.monotonic() - wall_start
        serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
        require(len(serialized.encode()) + initial.stat().st_size <= OUTPUT_CAP, 'audit output exceeds2MiB')
        with args.output.open('x') as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        signal.alarm(0)
    print(json.dumps({'status': result['status'], 'verification_after': result['verification_after'],
                      'points_verified': result.get('points_verified', 0), 'scalar_checks': result['scalar_checks'],
                      'cpu_seconds': result['cpu_seconds'], 'output': str(args.output)}))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
