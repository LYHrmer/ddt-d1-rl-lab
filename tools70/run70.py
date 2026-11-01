"""Read-only H1 cost finite-difference decomposition of complete69 archives.

No weights, model/actor calls, environment, fitting or policy changes. The two
finite-difference intervals differ; their algebraic attribution is not causal
identification of a local dynamics Jacobian. Original69 decisions are retained.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import require, sha, write_json
from tools59.run59 import load_grid, load_own_points

MODEL_SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
ROLES = ('old56', 'old64', 'fresh')
TICKS = (110, 260, 410)
EPSILONS = (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4)
MODEL_EPS = .001
ATOL, RTOL = 1e-10, 1e-8
SOURCE69 = Path('evidence/body_jvp69/execution_01')
PLAN69 = Path('docs/experiments69/body_jvp69_preregistered.json')
SUMMARY69 = Path('reports/body_jvp69_results_20261101.json')
DIRECTORIES = {'old56': Path('evidence/paired_horizon56/execution_01'),
               'old64': Path('evidence/mixed_sign64/execution_01'), 'fresh': SOURCE69}
RESET_SEEDS = {'old56': (56301, 56302, 56303), 'old64': (64301, 64302, 64303),
               'fresh': (69301, 69302, 69303)}
METRICS = ('fd_absolute_error', 'midpoint_absolute_mass', 'zero_bias_absolute_mass',
           'midpoint_offset_absolute_mass', 'response_absolute_mass')
DECISION_SPEC = {
    'model_seeds': list(MODEL_SEEDS), 'arms': list(ARMS), 'roles': list(ROLES),
    'reset_seeds': {role: list(values) for role, values in RESET_SEEDS.items()},
    'ticks': list(TICKS), 'physical_eps': list(EPSILONS), 'model_eps': MODEL_EPS,
    'model_points': 162, 'paired_intervals': 648, 'cost_component_pairs': 3888,
    'residual_coordinates': 22, 'horizon': 1, 'parity_atol': ATOL, 'parity_rtol': RTOL,
    'decomposition': 'symmetric midpoint and secant response terms for weighted squared residuals',
    'midpoint_split': 'zero-point difference plus endpoint midpoint-minus-zero offset difference; not a curvature estimate',
    'AD_minus_FD': 'total only; 162 different model-point values each reused at four physical eps',
    'arithmetic_residuals': 'total summation, residual-coordinate regrouping and identity residual separately',
    'primary_scope': 'all27 points, equal role/point/epsilon weight; all six cost components retained',
    'pair_comparison': 'per-seed active minus control component FD MAE; all3 decrease/increase/equal/mixed descriptive only',
    'absolute_mass': 'sum absolute algebraic terms; not additive percentages of net error or causal attribution',
    'failure_appendix': 'all69 H1 direction failures and all69 new H1 regressions, no selection',
    'source69_qualification_unchanged': True, 'unknown_reference_rejudged': False,
    'H32_decomposition': False, 'model_call_count': 0, 'actor_call_count': 0,
    'physics_steps': 0, 'updates': 0, 'new_candidate_automatically_authorized': False,
    'different_epsilon_intervals_include_nonlinearity_and_execution_precision_difference': True,
    'zero_point_difference_is_not_pure_model_state_bias': True,
}


def required_inputs():
    paths = {str(PLAN69), str(SUMMARY69), 'tools70/run70.py', 'tools70/decompose70.py',
             'tests/test_run70.py', 'tests/test_decompose70.py', 'tools53/run_pilot53.py',
             'tools54/numerics54.py', 'tools59/run59.py', 'tools69/outputs69.py'}
    for directory in DIRECTORIES.values():
        paths.update({str(directory / 'report.json'), str(directory / 'probe_snapshots.pkl')})
        paths.update(str(directory / f'grid_{point:02d}_{branch:02d}.npz')
                     for point in range(9) for branch in range(10))
    paths.update(str(SOURCE69 / f'{arm}_seed{seed}/cost_h1_{role}/point_{point:02d}.npz')
                 for seed in MODEL_SEEDS for arm in ARMS for role in ROLES for point in range(9))
    return paths


def inherited_bindings():
    return json.loads((ROOT / PLAN69).read_text())['source_sha256']


def verify70(plan):
    require(plan['schema'] == 'd1-cost-decomposition70-v1' and plan['decision_spec'] == DECISION_SPEC,
            '70 diagnostic contract changed')
    require(plan['cpu_seconds'] == 30 and plan['wall_seconds'] == 60, '70 resource budget changed')
    require(required_inputs() <= set(plan['source_sha256']), '70 inputs not all bound')
    require(all(plan['source_sha256'].get(name) == digest for name, digest in inherited_bindings().items()),
            '70 must retain every69 source binding unchanged')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '70 input/source changed: ' + name)


def close(a, b):
    import numpy as np
    left, right = np.asarray(a, np.float64), np.asarray(b, np.float64)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(),
            'invalid parity operands')
    gap = np.abs(left - right)
    return {'passed': bool(np.all(gap <= ATOL + RTOL * np.abs(right))),
            'max_abs_gap': float(gap.max(initial=0.)), 'atol': ATOL, 'rtol': RTOL}


def check_equal(a, b, name, checks):
    receipt = close(a, b)
    checks[name] = receipt
    require(receipt['passed'], '70 parity failure: ' + name)


def model_trace(path, start, source, checks):
    import numpy as np
    import torch

    from tools54 import numerics54 as frozen
    from tools70 import decompose70 as algebra
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    result = {}
    for branch in ('theta0', 'plus', 'minus'):
        state, action, applied, parts, cost = [data[branch + '_' + key] for key in
                                               ('states', 'actions', 'applied_actions', 'cost_components', 'costs')]
        require(all(array.dtype == np.float64 and np.isfinite(array).all()
                    for array in (state, action, applied, parts, cost)), 'model trace must be raw F64')
        require(state.shape == (1, 45) and action.shape == applied.shape == (1, 16)
                and parts.shape == (1, 6) and cost.shape == (1,)
                and np.array_equal(action, applied) and bool((np.abs(action) <= 1).all()),
                'H1 model trace shape or zero-delay action differs')
        command = np.asarray(start['commands'][0:1], np.float64)
        previous = np.asarray(start['last'][None], np.float64)
        with torch.no_grad():
            tensors = [torch.from_numpy(value) for value in (state, command, action, previous)]
            recomputed = frozen.cost(*tensors, start['nominal_height']).numpy()
            recomputed_parts = frozen.cost_components(*tensors, start['nominal_height']).numpy()
            coordinates = algebra.residuals(*tensors, start['nominal_height'])
        check_equal(cost, recomputed, branch + '_saved_total', checks)
        check_equal(parts, recomputed_parts, branch + '_saved_components', checks)
        check_equal(coordinates['components'], parts, branch + '_residual_components', checks)
        result[branch] = {'coordinates': coordinates['coordinates'][0], 'components': parts[0], 'cost': float(cost[0])}
    require(source['status'] == 'COMPLETE' and source['model_fd_epsilon'] == MODEL_EPS,
            'complete original H1 prediction required')
    check_equal(result['theta0']['cost'], source['cost'], 'saved_zero_to_receipt', checks)
    check_equal((result['plus']['cost'] - result['minus']['cost']) / (2 * MODEL_EPS),
                source['model_central_difference'], 'saved_fd_to_receipt', checks)
    return result


def physical_trace(start, grid, source, reference, checks):
    import numpy as np
    import torch

    from tools54 import numerics54 as frozen
    from tools70 import decompose70 as algebra
    require(len(grid) == 10 and np.asarray(start['fifo']).shape == (0, 16), 'ten zero-delay branches required')
    state = np.stack([branch['states'][0] for branch in grid])
    require(state.dtype == np.float64, 'physical state F64 required')
    action = np.stack([branch['observations'][0, 41:57] for branch in grid]).astype(np.float64)
    command = np.broadcast_to(np.asarray(start['commands'][0], np.float64), (10, 3)).copy()
    previous = np.broadcast_to(np.asarray(start['last'], np.float64), (10, 16)).copy()
    with torch.no_grad():
        tensors = [torch.from_numpy(value) for value in (state, command, action, previous)]
        cost = frozen.cost(*tensors, start['nominal_height']).numpy()
        parts = frozen.cost_components(*tensors, start['nominal_height']).numpy()
        coordinates = algebra.residuals(*tensors, start['nominal_height'])
    check_equal(cost, np.asarray([row['per_step_cost64'][0] for row in source['branches']]),
                'physical_cost_to_original', checks)
    check_equal(coordinates['components'], parts, 'physical_residual_components', checks)
    require(np.array_equal(coordinates['coordinates'][0], coordinates['coordinates'][1])
            and cost[0] == cost[1], 'repeated physical zero differs')
    require([row['epsilon'] for row in reference['eps']] == list(EPSILONS), 'all original epsilon references required')
    fd = [(cost[2 + 2 * ep] - cost[3 + 2 * ep]) / (2 * eps) for ep, eps in enumerate(EPSILONS)]
    check_equal(np.asarray(fd), np.asarray([row['central64'] for row in reference['eps']]),
                'physical_fd_to69_reference', checks)
    return {'coordinates': coordinates['coordinates'], 'components': parts, 'costs': cost}


def interval(model, physical, source_ad, index, epsilon, *, receipt=None, arrays=None):
    import numpy as np

    from tools70 import decompose70 as algebra
    plus, minus = 2 + 2 * index, 3 + 2 * index
    result = algebra.decompose(model['theta0']['coordinates'], model['plus']['coordinates'], model['minus']['coordinates'],
        physical['coordinates'][0], physical['coordinates'][plus], physical['coordinates'][minus],
        eps_model=MODEL_EPS, eps_true=epsilon)
    model_total_fd = (model['plus']['cost'] - model['minus']['cost']) / (2 * MODEL_EPS)
    true_total_fd = (physical['costs'][plus] - physical['costs'][minus]) / (2 * epsilon)
    model_parts_fd = (model['plus']['components'] - model['minus']['components']) / (2 * MODEL_EPS)
    true_parts_fd = (physical['components'][plus] - physical['components'][minus]) / (2 * epsilon)
    ad_fd = source_ad - model_total_fd
    total_rounding = (model_total_fd - math.fsum(model_parts_fd)) - (true_total_fd - math.fsum(true_parts_fd))
    coordinate_rounding = (model_parts_fd - result['model_fd']) - (true_parts_fd - result['true_fd'])
    reconstructed = math.fsum(result['midpoint_contribution']) + math.fsum(result['response_contribution'])
    reconstructed += ad_fd + total_rounding + math.fsum(coordinate_rounding) + math.fsum(result['identity_residual'])
    row = {} if receipt is None else receipt
    array_output = {} if arrays is None else arrays
    row.update(epsilon_true=epsilon, epsilon_model=MODEL_EPS, status='INCOMPLETE',
               model_total_fd=model_total_fd, true_total_fd=float(true_total_fd),
               model_AD_minus_true_FD=float(source_ad - true_total_fd),
               model_AD_minus_model_FD=float(ad_fd), total_summation_rounding_difference=float(total_rounding),
               coordinate_regrouping_rounding_difference=coordinate_rounding.tolist(),
               model_saved_component_fd=model_parts_fd.tolist(), true_frozen_component_fd=true_parts_fd.tolist(),
               reconstruction_error=float(source_ad - true_total_fd - reconstructed), checks={})
    for name, value in result.items():
        if isinstance(value, np.ndarray):
            array_output[name] = value
            if value.shape == (6,):
                row[name] = value.tolist()
    array_output.update({'scalar22_' + name: value for name, value in result['scalar22'].items()})
    row['coordinate_absolute_mass'] = {
        label: [float(np.abs(result['scalar22'][label + '_contribution'][list(indices)]).sum())
                for indices in algebra.GROUPS.values()]
        for label in ('midpoint', 'zero_bias', 'midpoint_offset', 'response')}
    checks = row['checks']
    check_equal(result['midpoint_contribution'] + result['response_contribution'],
                result['model_fd'] - result['true_fd'], 'component_identity', checks)
    check_equal(result['zero_bias_contribution'] + result['midpoint_offset_contribution'],
                result['midpoint_contribution'], 'midpoint_split', checks)
    check_equal(result['model_fd'], model_parts_fd, 'model_coordinate_fd', checks)
    check_equal(result['true_fd'], true_parts_fd, 'true_coordinate_fd', checks)
    check_equal(reconstructed, source_ad - true_total_fd, 'AD_total_reconstruction', checks)
    checks['scalar_identity'] = result['scalar_identity_check']
    require(result['scalar_identity_check']['passed'], '70 parity failure: scalar identity')
    row['status'] = 'COMPLETE'
    return row, array_output


def summaries(points, component_names):
    require(tuple(component_names) == ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness'),
            'all six original cost components required')
    require(len(points) == 162 and all(row['status'] == 'COMPLETE' and len(row['eps']) == 4
            and all(ep['status'] == 'COMPLETE' for ep in row['eps'])
            and [ep['epsilon_true'] for ep in row['eps']] == list(EPSILONS) for row in points),
            'complete162 points required for summary')
    require({(p['model_seed'], p['arm'], p['role'], p['seed'], p['tick']) for p in points} ==
            {(seed, arm, role, reset, tick) for seed in MODEL_SEEDS for arm in ARMS
             for role in ROLES for reset in RESET_SEEDS[role] for tick in TICKS},
            'all model/state identities required without duplicates')
    rows = []
    groups = (*ROLES, 'all27')
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            for role in groups:
                chosen = [p for p in points if p['model_seed'] == seed and p['arm'] == arm
                          and (role == 'all27' or p['role'] == role)]
                require(len(chosen) == (27 if role == 'all27' else 9), 'summary point grid differs')
                for column, component in enumerate(component_names):
                    values = {name: [] for name in METRICS}
                    for point in chosen:
                        for ep in point['eps']:
                            values['fd_absolute_error'].append(abs(ep['model_saved_component_fd'][column]
                                                                 - ep['true_frozen_component_fd'][column]))
                            for label in ('midpoint', 'zero_bias', 'midpoint_offset', 'response'):
                                values[label + '_absolute_mass'].append(ep['coordinate_absolute_mass'][label][column])
                    rows.append({'model_seed': seed, 'arm': arm, 'role': role, 'component': component,
                                 'points': len(chosen), 'intervals': 4 * len(chosen),
                                 **{key: math.fsum(value) / len(value) for key, value in values.items()}})
    paired = []
    for role in groups:
        for component in component_names:
            differences = []
            for seed in MODEL_SEEDS:
                left, right = [next(r for r in rows if r['model_seed'] == seed and r['arm'] == arm
                                   and r['role'] == role and r['component'] == component) for arm in ARMS]
                delta = right['fd_absolute_error'] - left['fd_absolute_error']
                differences.append({'model_seed': seed, 'control_fd_MAE': left['fd_absolute_error'],
                                    'active_fd_MAE': right['fd_absolute_error'], 'active_minus_control': delta})
            deltas = [row['active_minus_control'] for row in differences]
            category = ('ALL_THREE_NUMERICALLY_LOWER' if all(value < 0 for value in deltas) else
                        'ALL_THREE_NUMERICALLY_HIGHER' if all(value > 0 for value in deltas) else
                        'ALL_EQUAL' if all(value == 0 for value in deltas) else 'MIXED_OR_TIED')
            paired.append({'role': role, 'component': component, 'pairs': differences, 'classification': category})
    return {'metric_rows': rows, 'paired_component_FD_MAE': paired,
            'absolute_mass_is_not_causal_or_additive_fraction': True,
            'statistical_significance_claimed': False, 'controller_performance_claimed': False,
            'source69_rejudged': False, 'automatic_new_candidate': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-cost-decomposition70-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_calls': 0, 'actor_calls': 0, 'physics_steps': 0, 'model_updates': 0,
              'policy_updates': 0, 'control_updates': 0, 'completed_model_points': 0,
              'completed_intervals': 0, 'completed_component_pairs': 0, 'points': [],
              'source69_rejudged': False, 'summary': {'status': 'NOT_RUN'}}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall, plan = time.process_time(), time.monotonic(), None
    deadline = cpu + 28.

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    with (args.output / 'events.jsonl').open('x', buffering=1) as events:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
            require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                    'Nice10/single CPU required')
            require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                    'single thread/CPU environment required')
            plan = json.loads(args.plan.read_text())
            report['plan_sha256'] = sha(args.plan)
            verify70(plan)
            original = json.loads((ROOT / SOURCE69 / 'report.json').read_text())
            published = json.loads((ROOT / SUMMARY69).read_text())
            require(original == published['formal'] and original['status'] == original['verification_after'] == 'COMPLETE',
                    'complete unchanged69 source required')
            require(original['plan_sha256'] == sha(ROOT / PLAN69), '69 preregistration identity differs')
            report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
            require(report['versions'] == original['versions'], 'numeric package versions changed')
            import numpy as np
            import torch

            from tools70 import decompose70 as algebra
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            require(torch.__version__ == original['torch_runtime_version'], 'torch runtime differs')
            report['torch_runtime_version'] = torch.__version__
            require(algebra.ATOL == ATOL and algebra.RTOL == RTOL, 'algebra tolerance changed')
            names = list(algebra.NAMES)
            report['components'] = names
            report['source69_comparison'] = original['comparison']
            report['physical'] = {}
            contexts, truths = {}, {}
            for role, directory in DIRECTORIES.items():
                require(time.process_time() < deadline, 'CPU deadline before physical archive read')
                saved_report = json.loads((ROOT / directory / 'report.json').read_text())
                require(saved_report['status'] == 'COMPLETE', 'physical source incomplete: ' + role)
                contexts[role] = load_own_points(ROOT / directory / 'probe_snapshots.pkl')
                require([(p['seed'], p['tick']) for p in contexts[role]] ==
                        [(seed, tick) for seed in RESET_SEEDS[role] for tick in TICKS], 'snapshot grid differs')
                grids = load_grid(ROOT / directory, saved_report['physical_probes'])
                truths[role] = []
                report['physical'][role] = []
                for index, (start, grid, source) in enumerate(zip(contexts[role], grids, saved_report['physical_probes'], strict=True)):
                    checks = {}
                    reference = original['head_references'][role][index]
                    require((reference['seed'], reference['tick']) == (start['seed'], start['tick']), 'reference identity differs')
                    true = physical_trace(start, grid, source, reference['cost_h1'], checks)
                    truths[role].append(true)
                    report['physical'][role].append({'seed': start['seed'], 'tick': start['tick'], 'checks': checks,
                                                    'source69_cost_h1_reference': reference['cost_h1']})
            require({(m['seed'], m['arm']) for m in original['models']} ==
                    {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'all six original models required')
            by_key = {(m['seed'], m['arm']): m for m in original['comparison']['models']}
            for model in original['models']:
                require(model['status'] == model['fit']['status'] == 'COMPLETE', 'complete69 model receipt required')
                for role in ROLES:
                    source_points = model['cost_h1'][role]['points']
                    require(len(source_points) == 9 and model['cost_h1'][role]['status'] == 'COMPLETE', 'incomplete69 H1 predictions')
                    for index, (start, truth, source) in enumerate(zip(contexts[role], truths[role], source_points, strict=True)):
                        require(time.process_time() < deadline, 'CPU deadline before H1 decomposition')
                        require((source['seed'], source['tick']) == (start['seed'], start['tick']), 'model-point identity differs')
                        old = by_key[model['seed'], model['arm']]['heads'][role]['points'][index]
                        require((old['seed'], old['tick']) == (start['seed'], start['tick']), 'comparison identity differs')
                        row = {'model_seed': model['seed'], 'arm': model['arm'], 'role': role,
                               'seed': start['seed'], 'tick': start['tick'], 'status': 'INCOMPLETE',
                               'source69_h1': old['cost_h1'], 'checks': {}, 'eps': []}
                        report['points'].append(row)
                        name = f'{model["arm"]}_seed{model["seed"]}_{role}_{index:02d}'
                        arrays = {}
                        try:
                            path = SOURCE69 / model['directory'] / ('cost_h1_' + role) / f'point_{index:02d}.npz'
                            require(sha(ROOT / path) == original['sealed_artifacts_sha256'][str(path.relative_to(SOURCE69))],
                                    'model H1 trace is not original sealed output')
                            trace = model_trace(ROOT / path, start, source, row['checks'])
                            for branch, values in trace.items():
                                arrays['model_' + branch + '_coordinates'] = values['coordinates']
                            arrays['physical_coordinates'] = truth['coordinates']
                            for ep, epsilon in enumerate(EPSILONS):
                                receipt, values = {'status': 'INCOMPLETE', 'epsilon_true': epsilon}, {}
                                row['eps'].append(receipt)
                                try:
                                    interval(trace, truth, source['gradient'], ep, epsilon, receipt=receipt, arrays=values)
                                finally:
                                    arrays.update({f'epsilon_{ep}_{key}': value for key, value in values.items()})
                                report['completed_intervals'] += 1
                                report['completed_component_pairs'] += 6
                            row['status'] = 'COMPLETE'
                            report['completed_model_points'] += 1
                        except BaseException as exc:
                            row['failure'] = f'{type(exc).__name__}: {exc}'
                            raise
                        finally:
                            with (args.output / (name + '.npz')).open('xb') as stream:
                                np.savez_compressed(stream, **arrays)
                            write_json(args.output / (name + '.json'), row)
                            events.write(json.dumps({'point': name, 'status': row['status'],
                                'completed_points': report['completed_model_points'],
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
            require(report['completed_model_points'] == 162 and report['completed_intervals'] == 648
                    and report['completed_component_pairs'] == 3888, 'incomplete70 grid')
            report['summary'] = {'status': 'COMPLETE', **summaries(report['points'], names)}
            report['all69_H1_failures'] = [{'model_seed': p['model_seed'], 'arm': p['arm'], 'role': p['role'],
                'seed': p['seed'], 'tick': p['tick'], 'source69_h1': p['source69_h1']} for p in report['points']
                if p['source69_h1']['valid_match'] is False]
            report['all69_new_H1_regressions'] = [dict(model_seed=pair['seed'], **failure)
                for pair in original['comparison']['paired'] for failure in pair['new_resolved_cost_failures']
                if failure['horizon'] == 1]
            require(time.process_time() < deadline, 'CPU deadline before completion')
            report['status'] = 'COMPLETE'
        except BaseException as exc:  # noqa: BLE001 - retain any incomplete diagnostic without retry.
            report['failure'] = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if plan is not None:
                    verify70(plan)
                    require(sha(args.plan) == report['plan_sha256'], '70 plan changed')
                if 'versions' in report:
                    report['versions_after'] = {name: importlib.metadata.version(name) for name in report['versions']}
                    require(report['versions_after'] == report['versions'], 'numeric package versions changed during70')
                if 'torch_runtime_version' in report:
                    require(sys.modules['torch'].__version__ == report['torch_runtime_version'], 'torch runtime changed during70')
                report['verification_after'] = 'COMPLETE' if plan is not None else 'NOT_RUN'
            except Exception as exc:  # noqa: BLE001 - preserve verification and original failure separately.
                report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=f'{type(exc).__name__}: {exc}')
            if report['status'] != 'COMPLETE' and report['summary'].get('status') == 'COMPLETE':
                report['summary']['status'] = 'INCOMPLETE'
            report['cpu_seconds'] = time.process_time() - cpu
            report['wall_seconds'] = time.monotonic() - wall
            report['raw_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
            write_json(args.output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'failure', 'completed_model_points',
          'completed_intervals', 'completed_component_pairs', 'cpu_seconds', 'wall_seconds', 'verification_after')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
