"""65: fixed64 models, all fresh points, and cost-prefix diagnostics only.

No environment, model fit, policy update, oracle correction or horizon selection.
Physical finite differences retain all four original theta epsilons. Prefix
references are new descriptive diagnostics; the64 H32 decision is immutable.
The six cost derivatives describe the full closed-loop theta path, not separate
state/action Jacobians or a causal attribution to a particular dynamics term.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import pickle
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import require, sha, verify, write_json
from tools55.probe_local55 import EPSILONS, local_stability

SOURCE64 = Path('evidence/mixed_sign64/execution_01')
PLAN64 = Path('docs/experiments64/mixed_sign64_preregistered.json')
PLAN53 = Path('docs/experiments53/model_gradient53_preregistered.json')
MODEL_SEEDS = (64001, 64002, 64003)
PROBE_SEEDS = (64301, 64302, 64303)
ARMS = ('control', 'active')
TICKS = (110, 260, 410)
HORIZONS = (1, 2, 4, 8, 16, 32)
COMPONENTS = ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness')
MODEL_EPS = .001
ATOL = RTOL = 1e-10


def required_inputs():
    """Explicit inputs read by65, in addition to original53 frozen identities."""
    paths = {str(PLAN53), str(PLAN64), str(SOURCE64 / 'report.json'),
             str(SOURCE64 / 'probe_snapshots.pkl'), 'tools65/diagnose65.py',
             'tools54/numerics54.py', 'tools52/adapter52.py', 'tools52/control52.py',
             'tools52/dynamics52.py', 'tools52/run_pilot52.py', 'tools53/run_pilot53.py',
             'tools55/probe_local55.py', 'tools64/run64.py', 'tools64/fit64.py',
             'evidence/low_memory_delay25/execution_01/random/export/golden_fixtures.json'}
    paths.update(str(SOURCE64 / f'grid_{point:02d}_{branch:02d}.{suffix}')
                 for point in range(9) for branch in range(10) for suffix in ('npz', 'json'))
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            base = SOURCE64 / f'{arm}_seed{seed}'
            paths.add(str(base / 'final_state_dict.pt'))
            paths.update(str(base / f'fresh/point_{point:02d}.{suffix}')
                         for point in range(9) for suffix in ('npz', 'json'))
    return paths


def verify65(plan):
    require(plan['schema'] == 'd1-cost-prefix65-diagnostic-v1', 'wrong65 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS) and plan['horizons'] == list(HORIZONS)
            and plan['eps'] == list(EPSILONS), 'registered diagnostic grid changed')
    require(plan['cpu_seconds'] == 180 and plan['wall_seconds'] == 300
            and plan['expected_model_points'] == 54 and plan['new_physics_steps'] == 0
            and plan['model_updates'] == 0 and plan['policy_updates'] == 0,
            '65 work budget changed')
    require(plan['parity_atol'] == ATOL and plan['parity_rtol'] == RTOL
            and plan['model_fd_epsilon'] == MODEL_EPS, 'fixed numeric tolerances changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required65 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '65 source/input changed: ' + name)


def close_receipt(actual, expected):
    """Elementwise F64 parity; this tolerance never qualifies physical gradients."""
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'invalid parity arrays')
    gap = np.abs(a - b)
    return {'passed': bool(np.all(gap <= ATOL + RTOL * np.abs(b))),
            'max_abs_gap': float(gap.max(initial=0.)), 'atol': ATOL, 'rtol': RTOL}


def prefix_values(values):
    """[32,...] -> cumulative sums and prefix means, with frozen Torch reduction.

    Recompute each prefix mean instead of dividing cumsum: H32 then uses the
    exact same mean reduction as the original physical/model scalar cost.
    """
    import numpy as np
    import torch
    array = np.asarray(values, dtype=np.float64)
    require(array.shape[0] == 32 and np.isfinite(array).all(), 'finite complete32-step values required')
    tensor = torch.tensor(array, dtype=torch.float64)
    return {'cumulative': tensor.cumsum(0).numpy(),
            'prefix_mean': torch.stack([tensor[:h].mean(0) for h in range(1, 33)]).numpy()}


def prefix_differences(zero, plus, minus, epsilon):
    """Per-step, cumulative and prefix-mean central/one-sided FDs; no eps choice."""
    import numpy as np
    require(math.isfinite(epsilon) and epsilon > 0., 'positive finite epsilon required')
    z, p, m = [np.asarray(value, dtype=np.float64) for value in (zero, plus, minus)]
    require(z.shape == p.shape == m.shape, 'FD endpoint shapes differ')
    endpoint = [{'per_step': value, **prefix_values(value)} for value in (z, p, m)]
    arrays = {}
    for name in ('per_step', 'cumulative', 'prefix_mean'):
        z, p, m = [item[name] for item in endpoint]
        arrays.update({name + '_central': (p - m) / (2 * epsilon),
                       name + '_forward': (p - z) / epsilon,
                       name + '_backward': (z - m) / epsilon})
    return arrays


def fd_roundoff_bounds(plus_components, minus_components, epsilon):
    """Conservative absolute F64 reconciliation bounds, never stability gates.

    For at most32*6 components at each endpoint, gamma1024 bounds the addition,
    subtraction and division roundoff of the two equivalent reductions. S is
    the sum of endpoint absolute component values, with a floor1. Cumulative
    bound=gamma1024*S/(2eps); prefix-mean bound divides this by H. The per-step
    bound uses only that step. No change is made to the55C eligibility rule.
    """
    import numpy as np
    p, m = [np.asarray(value, dtype=np.float64) for value in (plus_components, minus_components)]
    require(p.shape == m.shape == (32, 6) and np.isfinite(p).all() and np.isfinite(m).all()
            and math.isfinite(epsilon) and epsilon > 0, 'invalid reconciliation inputs')
    unit = 2. ** -53
    gamma = 1024 * unit / (1 - 1024 * unit)
    scale = np.abs(p).sum(-1) + np.abs(m).sum(-1)
    per_step = gamma * np.maximum(1., scale) / (2 * epsilon)
    cumulative = gamma * np.maximum(1., scale.cumsum()) / (2 * epsilon)
    return {'per_step': per_step, 'cumulative': cumulative,
            'prefix_mean': cumulative / np.arange(1, 33)}


def prefix_stability(fd_arrays, *, action_resolved):
    """All32 prefix references, each using all four fixed physical theta scales."""
    require(len(fd_arrays) == 4, 'all four epsilon arrays required')
    rows = []
    for tick in range(32):
        gradients = [{'epsilon': eps, 'central64': float(fd['prefix_mean_central'][tick]),
                      'forward64': float(fd['prefix_mean_forward'][tick]),
                      'backward64': float(fd['prefix_mean_backward'][tick])}
                     for eps, fd in zip(EPSILONS, fd_arrays, strict=True)]
        stable = local_stability(gradients)
        rows.append({'horizon': tick + 1, 'registered_summary_horizon': tick + 1 in HORIZONS,
                     'gradients': gradients, 'local_stability': stable,
                     'action_resolved': bool(action_resolved),
                     'resolved': bool(action_resolved and stable['locally_stable']),
                     'reference_gradient': math.fsum(row['central64'] for row in gradients) / 4,
                     'changes64_decision': False})
    return rows


def selfcheck(analytic, numerical):
    """Frozen52/54 AD-vs-model-FD rule, independently for total and six terms."""
    require(math.isfinite(analytic) and math.isfinite(numerical), 'nonfinite derivative')
    return {'gradient': analytic, 'model_central_difference': numerical,
            'autograd_agrees': abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4,
            'direction_identifiable': abs(analytic) >= 1e-5 and abs(numerical) >= 1e-5
            and analytic * numerical > 0}


def decompose_model(rollout, *, expected_trace=None, expected_receipt=None, arrays=None):
    """Frozen F64 rollout(theta)->(mean,trace); scalar AD plus all-cost JVP.

    Functional JVP uses reverse-over-reverse, one fixed implementation. The
    scalar mean AD and theta0 trace are admitted before decomposition. Only
    immutable model weights enter; no observed future state enters rollout.
    Optional expectations exist solely for asset-free synthetic unit tests.
    """
    import numpy as np
    import torch
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value, trace = rollout(theta)
    analytic, = torch.autograd.grad(value, theta)
    require(value.dtype == analytic.dtype == torch.float64
            and trace['cost_components'].shape == (32, 6) and trace['costs'].shape == (32,),
            'complete F64 graph required')
    arrays = {} if arrays is None else arrays
    arrays.update({f'theta0_{name}': item.detach().numpy() for name, item in trace.items()})
    parity = {}
    if expected_trace is not None:
        require(set(trace) <= set(expected_trace), 'saved64 trace lacks fields')
        parity['trace'] = {name: close_receipt(item.detach().numpy(), expected_trace[name])
                           for name, item in trace.items()}
        require(all(row['passed'] for row in parity['trace'].values()), 'zero theta trace differs from64')
    if expected_receipt is not None:
        parity['cost'] = close_receipt(float(value.detach()), expected_receipt['cost'])
        parity['gradient'] = close_receipt(float(analytic), expected_receipt['gradient'])
        require(parity['cost']['passed'] and parity['gradient']['passed'], 'scalar mean AD differs from64')

    def outputs(current):
        _, result = rollout(current)
        return result['costs'], result['cost_components']

    (values, parts), (cost_jvp, component_jvp) = torch.autograd.functional.jvp(
        outputs, torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64),
        create_graph=False, strict=False)
    parity['jvp_primal_cost'] = close_receipt(values.detach().numpy(), arrays['theta0_costs'])
    parity['jvp_primal_components'] = close_receipt(parts.detach().numpy(), arrays['theta0_cost_components'])
    parity['jvp_scalar_mean'] = close_receipt(float(cost_jvp.mean()), float(analytic))
    parity['components_to_total_value'] = close_receipt(parts.sum(-1).numpy(), values.numpy())
    parity['components_to_total_derivative'] = close_receipt(component_jvp.sum(-1).numpy(), cost_jvp.numpy())
    require(all(row['passed'] for name, row in parity.items() if name != 'trace'), 'JVP decomposition identity failed')
    arrays.update(per_step_cost_derivative=cost_jvp.numpy(), per_step_component_derivative=component_jvp.numpy())
    for prefix, derivative in (('cost_derivative', cost_jvp), ('component_derivative', component_jvp)):
        arrays.update({name + '_' + prefix: data for name, data in prefix_values(derivative.numpy()).items()})
    endpoint = []
    with torch.no_grad():
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            scalar, current = rollout(torch.tensor(sign * MODEL_EPS, dtype=torch.float64))
            endpoint.append((float(scalar), current))
            arrays.update({label + '_' + key: current[key].numpy() for key in ('costs', 'cost_components')})
    model_fd = (endpoint[0][0] - endpoint[1][0]) / (2 * MODEL_EPS)
    if expected_receipt is not None:
        parity['model_fd'] = close_receipt(model_fd, expected_receipt['model_central_difference'])
        require(parity['model_fd']['passed'], 'original model FD differs from64')
    for label, key in (('cost', 'costs'), ('component', 'cost_components')):
        fd = prefix_differences(arrays['theta0_' + key], arrays['plus_' + key], arrays['minus_' + key], MODEL_EPS)
        arrays.update({label + '_model_' + name: data for name, data in fd.items()})
    require(all(np.isfinite(array).all() for array in arrays.values()), 'nonfinite model decomposition')
    rows = []
    for tick in range(32):
        total = selfcheck(float(arrays['prefix_mean_cost_derivative'][tick]),
                          float(arrays['cost_model_prefix_mean_central'][tick]))
        components = {name: selfcheck(float(arrays['prefix_mean_component_derivative'][tick, index]),
                                     float(arrays['component_model_prefix_mean_central'][tick, index]))
                      for index, name in enumerate(COMPONENTS)}
        rows.append({'horizon': tick + 1, 'registered_summary_horizon': tick + 1 in HORIZONS,
                     **total, 'components': components,
                     'all_component_selfchecks': all(row['autograd_agrees'] for row in components.values())})
    return {'cost': float(value.detach()), 'gradient': float(analytic), 'model_central_difference': model_fd,
            'parity': parity, 'prefixes': rows, 'model_fd_epsilon': MODEL_EPS,
            'all_prefix_total_selfchecks': all(row['autograd_agrees'] for row in rows),
            'all_prefix_component_selfchecks': all(row['all_component_selfchecks'] for row in rows)}, arrays


def physical_decomposition(start, index, old, directory, *, arrays=None):
    """Read exact completed64 branches; cost uses poststate/current command.

    Observation41:57 is the actual issued F32 action. Original FIFO is empty,
    so issued==applied. First previous action is snapshot.last; later values
    are this branch's preceding issued action. No actor is evaluated here.
    """
    import numpy as np
    import torch

    from tools54 import numerics54 as numeric
    require(old['status'] == 'COMPLETE' and len(old['branches']) == 10, 'partial64 physical point')
    require(np.asarray(start['fifo']).shape == (0, 16), 'actual issued actions require original zero delay')
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    arrays = {} if arrays is None else arrays
    raw, totals, components, parity = [], [], [], []
    for order, theta in enumerate(sequence):
        stem = directory / f'grid_{index:02d}_{order:02d}'
        row = json.loads(stem.with_suffix('.json').read_text())
        require(row == old['branches'][order] and row['status'] == 'COMPLETE' and row['steps'] == 32
                and row['theta'] == theta and not row['unsafe_steps'], 'partial/different64 branch metadata')
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as stored:
            branch = {key: stored[key].copy() for key in ('states', 'observations', 'rewards')}
        arrays.update({f'branch_{order}_{key}': value for key, value in branch.items()})
        require(branch['states'].shape == (32, 45) and branch['observations'].shape == (32, 627)
                and branch['rewards'].shape == (32,) and branch['states'].dtype == np.float64
                and branch['observations'].dtype == np.float32
                and all(np.isfinite(value).all() for value in branch.values()), 'invalid64 branch arrays')
        state = torch.tensor(branch['states'], dtype=torch.float64)
        action = torch.tensor(branch['observations'][:, 41:57], dtype=torch.float64)
        require(bool((action.abs() <= 1.).all()), 'unclipped physical action')
        previous = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), action[:-1]))
        command = torch.tensor(start['commands'][:32], dtype=torch.float64)
        with torch.no_grad():
            cost = numeric.cost(state, command, action, previous, start['nominal_height'])
            parts = numeric.cost_components(state, command, action, previous, start['nominal_height'])
        require(np.array_equal(cost.numpy(), row['per_step_cost64']) and float(cost.mean()) == row['cost64'],
                'physical cost does not exactly reproduce64')
        reconciliation = close_receipt(parts.sum(-1).numpy(), cost.numpy())
        require(reconciliation['passed'], 'physical cost components do not restore total')
        parity.append({'branch': order, 'cost_per_step_exact': True, 'cost_mean_exact': True,
                       'components_to_total': reconciliation})
        raw.append(branch)
        totals.append(cost.numpy())
        components.append(parts.numpy())
    require(all(np.array_equal(raw[0][key], raw[1][key]) for key in raw[0]), 'repeated physical zero differs')
    arrays.update(branch_costs=np.stack(totals), branch_components=np.stack(components),
                  branch_actual_actions=np.stack([branch['observations'][:, 41:57] for branch in raw]),
                  commands=np.asarray(start['commands'][:32], dtype=np.float64),
                  initial_previous_action=np.asarray(start['last'], dtype=np.float64))
    fd_arrays, component_fd_arrays, reconciliations = [], [], []
    for ep, epsilon in enumerate(EPSILONS):
        plus, minus = 2 + 2 * ep, 3 + 2 * ep
        cost_fd = prefix_differences(totals[0], totals[plus], totals[minus], epsilon)
        part_fd = prefix_differences(components[0], components[plus], components[minus], epsilon)
        fd_arrays.append(cost_fd)
        component_fd_arrays.append(part_fd)
        arrays.update({f'epsilon_{ep}_total_{key}': value for key, value in cost_fd.items()})
        arrays.update({f'epsilon_{ep}_component_{key}': value for key, value in part_fd.items()})
        check = {'epsilon': epsilon, 'bound_is_not_a_stability_tolerance': True}
        for direction, left, right, denominator_eps in (
                ('central', components[plus], components[minus], epsilon),
                ('forward', components[plus], components[0], epsilon / 2),
                ('backward', components[0], components[minus], epsilon / 2)):
            bounds = fd_roundoff_bounds(left, right, denominator_eps)
            for prefix, bound in bounds.items():
                gap = np.abs(part_fd[prefix + '_' + direction].sum(-1) - cost_fd[prefix + '_' + direction])
                require(bool((gap <= bound).all()), 'physical FD component reconciliation exceeds roundoff bound')
                check[prefix + '_' + direction] = {'max_abs_gap': float(gap.max()),
                    'max_bound': float(bound.max()), 'max_gap_over_bound': float((gap / bound).max())}
                arrays[f'epsilon_{ep}_{prefix}_{direction}_reconciliation_bound'] = bound
        reconciliations.append(check)
    prefixes = prefix_stability(fd_arrays, action_resolved=old['all_four_wheels_resolved'])
    component_prefixes = {name: prefix_stability(
        [{key: value[:, column] for key, value in fd.items()} for fd in component_fd_arrays],
        action_resolved=old['all_four_wheels_resolved']) for column, name in enumerate(COMPONENTS)}
    for derived, saved in zip(prefixes[-1]['gradients'], old['gradients'], strict=True):
        require(all(derived[key] == saved[key] for key in ('epsilon', 'central64', 'forward64', 'backward64')),
                'H32 physical gradients differ from64')
    return {'seed': start['seed'], 'tick': start['tick'], 'status': 'COMPLETE', 'parity': parity,
            'prefixes': prefixes, 'component_prefixes': component_prefixes, 'reconciliations': reconciliations,
            'actual_action_scope': 'issued F32 recovered from observation41:57; original zero delay'}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-cost-prefix65-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'new_physics_steps': 0, 'model_updates': 0, 'policy_updates': 0,
              'source64_decision_changed': False, 'selected_horizon': None, 'selected_seed': None,
              'diagnostics_complete': False, 'completed_model_points': 0, 'completed_physical_points': 0,
              'components': list(COMPONENTS), 'summary_horizons': list(HORIZONS), 'eps': list(EPSILONS),
              'physical': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN'}
                           for seed in PROBE_SEEDS for tick in TICKS],
              'models': [{'seed': seed, 'arm': arm, 'status': 'NOT_RUN', 'points': [
                  {'seed': reset, 'tick': tick, 'status': 'NOT_RUN'} for reset in PROBE_SEEDS for tick in TICKS]}
                  for seed in MODEL_SEEDS for arm in ARMS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    plan, original = None, None
    progress = {'phase': 'admission', 'model': None, 'point': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress, 'completed_model_points': report['completed_model_points'],
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
        events.flush()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (179, 180))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify65(plan)
        original = json.loads((ROOT / PLAN53).read_text())
        verify(original)
        source = json.loads((ROOT / SOURCE64 / 'report.json').read_text())
        require(source['status'] == 'COMPLETE' and source['verification_after'] == 'COMPLETE'
                and source['completed_model_fits'] == 6 and source['model_updates'] == 6000
                and source['counts']['policy_transitions'] == 4680
                and source['plan_sha256'] == sha(ROOT / PLAN64), 'complete admitted64 source required')
        report['source64_comparison'] = source['comparison']
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == source['versions'], 'package versions differ')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools54 import numerics54 as numeric
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == source['torch_runtime_version'], 'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        actor = copy.deepcopy(actor32).double().eval().requires_grad_(False)
        # Own-generated local snapshot bytes are SHA-admitted before unpickling.
        with (ROOT / SOURCE64 / 'probe_snapshots.pkl').open('rb') as stream:
            stored = pickle.load(stream)
        points = [point | {'shaper': SimpleNamespace(**point['python_state']['shaper'])} for point in stored]
        require([(point['seed'], point['tick']) for point in points] ==
                [(seed, tick) for seed in PROBE_SEEDS for tick in TICKS], '64 point identities differ')
        event('ADMITTED')
        physical_output = args.output / 'physical'
        physical_output.mkdir(exist_ok=False)
        for index, start in enumerate(points):
            require(time.process_time() < 178., 'CPU deadline before physical readback')
            progress.update(phase='physical_readback', point=index)
            row, arrays = report['physical'][index], {}
            row['status'] = 'INCOMPLETE'
            try:
                result, arrays = physical_decomposition(start, index, source['physical_probes'][index],
                                                        ROOT / SOURCE64, arrays=arrays)
                reference = source['cost_references']['fresh'][index]
                require((reference['seed'], reference['tick']) == (start['seed'], start['tick'])
                        and reference['resolved'] == result['prefixes'][-1]['resolved'], 'H32 reference status differs')
                row.update(result, source64_reference=reference, source64_decision_changed=False)
                report['completed_physical_points'] += 1
            except BaseException as exc:
                row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                shared.save_npz(physical_output / f'point_{index:02d}.npz', arrays)
                write_json(physical_output / f'point_{index:02d}.json', row)
                event('PHYSICAL_SAVED')
        for model_row in report['models']:
            require(time.process_time() < 178., 'CPU deadline before model')
            model_row['status'] = 'INCOMPLETE'
            directory = f'{model_row["arm"]}_seed{model_row["seed"]}'
            output = args.output / directory
            output.mkdir(exist_ok=False)
            progress.update(phase='model', model=directory, point=None)
            old_model = next(row for row in source['models']
                             if (row['arm'], row['seed']) == (model_row['arm'], model_row['seed']))
            require(old_model['status'] == 'COMPLETE' and old_model['fit']['status'] == 'COMPLETE'
                    and old_model['fit']['updates'] == 1000, 'complete64 final model required')
            weights = directory + '/final_state_dict.pt'
            require(sha(ROOT / SOURCE64 / weights) == source['sealed_artifacts_sha256'][weights],
                    '64 final weight differs')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(ROOT / SOURCE64 / weights, map_location='cpu', weights_only=True))
            model = model.double().eval().requires_grad_(False)
            for index, start in enumerate(points):
                require(time.process_time() < 178., 'CPU deadline before numeric point')
                row, arrays = model_row['points'][index], {}
                row['status'] = 'INCOMPLETE'
                progress.update(phase='model_point', point=index)
                event('POINT_BEGIN')
                try:
                    old_receipt = old_model['diagnostics']['fresh']['points'][index]
                    old_path = ROOT / SOURCE64 / directory / f'fresh/point_{index:02d}'
                    require(json.loads(old_path.with_suffix('.json').read_text()) == old_receipt
                            and old_receipt['status'] == 'COMPLETE'
                            and (old_receipt['seed'], old_receipt['tick']) == (start['seed'], start['tick'])
                            and sha(old_path.with_suffix('.npz')) == old_receipt['trace_sha256'], '64 model trace identity differs')
                    with np.load(old_path.with_suffix('.npz'), allow_pickle=False) as raw:
                        old_trace = {key: raw[key].copy() for key in raw.files}
                    result, arrays = decompose_model(
                        lambda theta, model=model, start=start: numeric.imaginary(
                            model, actor, start, theta, return_trace=True),
                        expected_trace=old_trace, expected_receipt=old_receipt, arrays=arrays)
                    for prefix, physical in zip(result['prefixes'], report['physical'][index]['prefixes'], strict=True):
                        resolved = physical['resolved']
                        prefix.update(physical_reference_resolved=resolved,
                            physical_reference_gradient=physical['reference_gradient'],
                            sign_matches_reference=prefix['gradient'] * physical['reference_gradient'] > 0 if resolved else None,
                            valid_direction_match=bool(prefix['autograd_agrees'] and prefix['direction_identifiable']
                                and prefix['gradient'] * physical['reference_gradient'] > 0) if resolved else None)
                        for name, component in prefix['components'].items():
                            reference = report['physical'][index]['component_prefixes'][name][prefix['horizon'] - 1]
                            component.update(physical_reference_resolved=reference['resolved'],
                                physical_reference_gradient=reference['reference_gradient'],
                                sign_matches_reference=component['gradient'] * reference['reference_gradient'] > 0
                                    if reference['resolved'] else None,
                                valid_direction_match=bool(component['autograd_agrees'] and component['direction_identifiable']
                                    and component['gradient'] * reference['reference_gradient'] > 0)
                                    if reference['resolved'] else None)
                    row.update(result, status='COMPLETE', source64_reference=source['cost_references']['fresh'][index],
                               source64_decision_changed=False)
                    report['completed_model_points'] += 1
                except BaseException as exc:
                    row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    shared.save_npz(output / f'point_{index:02d}.npz', arrays)
                    write_json(output / f'point_{index:02d}.json', row)
                    event('POINT_SAVED')
            model_row['status'] = 'COMPLETE'
            model_row['summary_by_horizon'] = {str(h): {
                'resolved_references': sum(point['prefixes'][h - 1]['physical_reference_resolved'] for point in model_row['points']),
                'valid_direction_matches': sum(point['prefixes'][h - 1]['valid_direction_match'] is True for point in model_row['points']),
                'total_selfchecks': sum(point['prefixes'][h - 1]['autograd_agrees'] for point in model_row['points']),
                'component_selfchecks': sum(point['prefixes'][h - 1]['all_component_selfchecks'] for point in model_row['points'])}
                for h in HORIZONS}
            write_json(output / 'result.json', model_row)
        require(report['completed_model_points'] == 54 and report['completed_physical_points'] == 9,
                'all6x9 model points and all9 physical readbacks required')
        report.update(status='COMPLETE', diagnostics_complete=True)
    except BaseException as exc:  # noqa: BLE001 - partial is retained, never resumed or substituted for64.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], '65 plan changed')
                verify65(plan)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - preserve original and final admission failures.
            report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=str(exc))
        if report['status'] != 'COMPLETE':
            report['diagnostics_complete'] = False
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'completed_model_points', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
