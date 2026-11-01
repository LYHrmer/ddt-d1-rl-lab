"""67: one-step model joint responses along existing physical64 trajectories.

No environment, actor inference, fit or policy update. At t0 the actual input
state direction is zero, identifying B only along the recorded action direction.
Later A/B are model-internal contributions: the real correlated state/action
directions do not separately identify the physical A or B. All four scales and
all32 steps are retained; physical qualification is inherited unchanged from65.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import require, sha, verify, write_json
from tools65 import diagnose65 as prior
from tools66 import rebase66

SOURCE64 = prior.SOURCE64
SOURCE65 = rebase66.SOURCE65
SOURCE66 = Path('evidence/cost_rebase66/diagnostic_01')
PLAN66 = Path('docs/experiments66/cost_rebase66_preregistered.json')
MODEL_SEEDS, PROBE_SEEDS, ARMS = prior.MODEL_SEEDS, prior.PROBE_SEEDS, prior.ARMS
TICKS, HORIZONS, COMPONENTS, EPSILONS = prior.TICKS, prior.HORIZONS, prior.COMPONENTS, prior.EPSILONS
LINE_FD_STEP = .001


def required_inputs():
    paths = rebase66.required_inputs() | {str(PLAN66), str(SOURCE66 / 'report.json'),
                                        'tools67/joint67.py', 'tools66/rebase66.py'}
    paths.update(str(SOURCE66 / f'{arm}_seed{seed}/point_{point:02d}.json')
                 for seed in MODEL_SEEDS for arm in ARMS for point in range(9))
    return paths


def verify67(plan):
    require(plan['schema'] == 'd1-joint-response67-diagnostic-v1', 'wrong67 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS) and plan['horizons'] == list(HORIZONS)
            and plan['eps'] == list(EPSILONS), '67 diagnostic grid changed')
    require(plan['cpu_seconds'] == 180 and plan['wall_seconds'] == 300
            and plan['expected_model_points'] == 54 and plan['new_physics_steps'] == 0
            and plan['model_updates'] == 0 and plan['policy_updates'] == 0, '67 budget changed')
    require(plan['parity_atol'] == prior.ATOL and plan['parity_rtol'] == prior.RTOL
            and plan['line_fd_step'] == LINE_FD_STEP, '67 fixed numeric tolerances changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required67 sources/inputs not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '67 source/input changed: ' + name)


def build_directions(start, branches):
    """Ten complete branches -> true zero inputs and four finite chord directions.

    Returned arrays are F64. Actual issued actions originate from F32 obs41:57;
    zero delay makes them applied actions too. Pre-state is initial+post[:-1],
    and previous-action is snapshot.last+issued[:-1], separately on each branch.
    No input normalization or extra quaternion projection is applied.
    """
    import numpy as np
    require(len(branches) == 10 and np.asarray(start['fifo']).shape == (0, 16), 'complete zero-delay grid required')
    for branch in branches:
        require(branch['states'].shape == (32, 45) and branch['states'].dtype == np.float64
                and branch['observations'].shape == (32, 627) and branch['observations'].dtype == np.float32
                and all(np.isfinite(branch[key]).all() for key in ('states', 'observations')), 'invalid physical branch')
    require(all(np.array_equal(branches[0][key], branches[1][key]) for key in ('states', 'observations')),
            'repeated physical zero differs')
    post = np.stack([branch['states'] for branch in branches])
    action = np.stack([branch['observations'][:, 41:57] for branch in branches]).astype(np.float64)
    require(bool((np.abs(action) <= 1).all()), 'physical action exceeds executed clip')
    pre = np.concatenate((np.broadcast_to(np.asarray(start['state'], np.float64), (10, 1, 45)), post[:, :-1]), axis=1)
    previous = np.concatenate((np.broadcast_to(np.asarray(start['last'], np.float64), (10, 1, 16)), action[:, :-1]), axis=1)
    values = {'pre_state': pre, 'action': action, 'next_state': post, 'previous_action': previous}
    result = {'commands': np.asarray(start['commands'][:32], dtype=np.float64)}
    for name, array in values.items():
        result['zero_' + name] = array[0].copy()
        result['branches_' + name] = array
        result['delta_' + name] = np.stack([(array[2 + 2 * index] - array[3 + 2 * index]) / (2 * eps)
                                           for index, eps in enumerate(EPSILONS)])
        result['midpoint_drift_' + name] = np.stack([
            (array[2 + 2 * index] + array[3 + 2 * index]) / 2 - array[0] for index in range(4)])
    require(np.count_nonzero(result['delta_pre_state'][:, 0]) == 0
            and np.count_nonzero(result['delta_previous_action'][:, 0]) == 0, 'initial input directions must be exactly zero')
    return result


def group_rms(array):
    """Seven normalized state groups; only reporting uses PHYSICAL_SCALE."""
    import numpy as np

    from tools52 import dynamics52 as dynamics
    normalized = np.asarray(array, np.float64) / dynamics.PHYSICAL_SCALE.astype(np.float64)
    return {name: float(np.sqrt(np.mean(normalized[:, list(indices)] ** 2)))
            for name, indices in dynamics.GROUPS.items()}


def joint_response(model, directions, *, deadline=math.inf, arrays=None):
    """Batch1 A*ds, B*da, joint JVP and fixed .001 model-line FD for all4x32.

    The line parameter is distinct from each physical theta epsilon: it moves
    along that epsilon's actual chord direction at the true zero input. It is
    not an endpoint-model FD at the saved asymmetric physical plus/minus inputs.
    """
    import numpy as np
    import torch

    from tools52 import dynamics52 as dynamics
    arrays = {} if arrays is None else arrays
    require(all(not parameter.requires_grad for parameter in model.parameters()), 'model parameters must be frozen')
    tensor = {key: torch.tensor(value, dtype=torch.float64) for key, value in directions.items()
              if key in ('zero_pre_state', 'zero_action', 'delta_pre_state', 'delta_action')}
    scale = dynamics.PHYSICAL_SCALE.astype(np.float64)
    rows = []
    for ep, epsilon in enumerate(EPSILONS):
        collected = {key: [] for key in ('model_zero_next', 'A_delta_state', 'B_delta_action', 'joint_jvp', 'model_line_fd')}
        parity = []
        for tick in range(32):
            require(time.process_time() < deadline, 'CPU deadline during joint response')
            state = tensor['zero_pre_state'][tick:tick + 1]
            action = tensor['zero_action'][tick:tick + 1]
            ds = tensor['delta_pre_state'][ep, tick:tick + 1]
            da = tensor['delta_action'][ep, tick:tick + 1]
            base_a, a = torch.autograd.functional.jvp(model, (state, action), (ds, torch.zeros_like(da)),
                                                     create_graph=False, strict=False)
            base_b, b = torch.autograd.functional.jvp(model, (state, action), (torch.zeros_like(ds), da),
                                                     create_graph=False, strict=False)
            baseline, joint = torch.autograd.functional.jvp(model, (state, action), (ds, da),
                                                           create_graph=False, strict=False)
            checks = {'sum': prior.close_receipt((a + b).numpy(), joint.numpy()),
                      'primal_a': prior.close_receipt(base_a.numpy(), baseline.numpy()),
                      'primal_b': prior.close_receipt(base_b.numpy(), baseline.numpy())}
            require(all(check['passed'] for check in checks.values()), 'A+B or primal JVP identity failed')
            with torch.no_grad():
                plus = model(state + LINE_FD_STEP * ds, action + LINE_FD_STEP * da)
                minus = model(state - LINE_FD_STEP * ds, action - LINE_FD_STEP * da)
                line_fd = (plus - minus) / (2 * LINE_FD_STEP)
            for name, value in (('model_zero_next', baseline), ('A_delta_state', a),
                                ('B_delta_action', b), ('joint_jvp', joint), ('model_line_fd', line_fd)):
                collected[name].append(value[0].numpy())
                arrays[f'epsilon_{ep}_{name}'] = np.stack(collected[name])
            parity.append(checks)
        joint = arrays[f'epsilon_{ep}_joint_jvp']
        line = arrays[f'epsilon_{ep}_model_line_fd']
        normalized_gap = np.abs((joint - line) / scale)
        tolerance = .1 * np.abs(line / scale) + 1e-4
        checks = normalized_gap <= tolerance
        actual = directions['delta_next_state'][ep]
        arrays.update({f'epsilon_{ep}_state_error': joint - actual,
                       f'epsilon_{ep}_line_selfcheck': checks,
                       f'epsilon_{ep}_normalized_line_gap': normalized_gap,
                       f'epsilon_{ep}_normalized_line_tolerance': tolerance})
        require(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite joint diagnostic')
        require(np.count_nonzero(arrays[f'epsilon_{ep}_A_delta_state'][0]) == 0,
                't0 A contribution must be exactly zero')
        rows.append({'epsilon': epsilon, 'status': 'COMPLETE', 'sum_and_primal_parity': parity,
                     'line_fd_step': LINE_FD_STEP, 'state_line_selfchecks': int(checks.sum()),
                     'state_line_selfcheck_channels': int(checks.size), 'all_state_line_selfchecks': bool(checks.all()),
                     'normalized_rms_error': group_rms(joint - actual),
                     'normalized_rms_actual_response': group_rms(actual),
                     'normalized_rms_model_response': group_rms(joint),
                     'zero_prediction_bias_rms': group_rms(arrays[f'epsilon_{ep}_model_zero_next'] - directions['zero_next_state']),
                     'first_step_yaw28': {'actual': float(actual[0, 28]), 'model_B': float(arrays[f'epsilon_{ep}_B_delta_action'][0, 28]),
                                          'model_joint': float(joint[0, 28]), 'state_line_selfcheck': bool(checks[0, 28])}})
    return {'status': 'COMPLETE', 'epsilons': rows,
            't0_only_actual_action_direction_identifiable': True, 'later_physical_A_B_separately_identifiable': False}, arrays


def projection_comparison(reference, physical_fd, actual_projection, model_projection, state_numeric_valid):
    """Report local direction only when reference and numerical evidence permit.

    Actual-tangent projection versus physical FD is a first-order consistency
    diagnostic, not AD/FD of the same function and not a new physical gate.
    """
    require(len(physical_fd) == len(actual_projection) == len(model_projection) == 4, 'all four projection scales required')
    require(all(math.isfinite(float(value)) for series in (physical_fd, actual_projection, model_projection)
                for value in series), 'nonfinite cost projection')
    p, a, m = [[float(value) for value in series] for series in (physical_fd, actual_projection, model_projection)]
    agrees = [abs(left - right) <= .1 * abs(right) + 1e-4 for left, right in zip(a, p, strict=True)]
    actual_identifiable = all(abs(left) >= 1e-5 and left * right > 0 for left, right in zip(a, p, strict=True))
    model_magnitude = all(abs(value) >= 1e-5 for value in m)
    model_same_sign = all(value > 0 for value in m) or all(value < 0 for value in m)
    signs = [left * right > 0 for left, right in zip(m, p, strict=True)] if reference['resolved'] else [None] * 4
    if not reference['resolved']:
        status = 'REFERENCE_UNRESOLVED'
    elif not state_numeric_valid:
        status = 'NUMERIC_UNRESOLVED'
    elif not all(agrees) or not actual_identifiable:
        status = 'ACTUAL_COST_TANGENT_UNRESOLVED'
    elif not model_magnitude or not model_same_sign:
        status = 'MODEL_DIRECTION_UNRESOLVED'
    else:
        status = 'LOCAL_DIRECTION_MATCH' if all(signs) else 'LOCAL_DIRECTION_MISMATCH'
    return {'status': status, 'physical_reference_resolved': reference['resolved'],
            'physical_fd_all_eps': p, 'actual_tangent_projection_all_eps': a, 'model_joint_projection_all_eps': m,
            'actual_projection_minus_physical_fd': [left - right for left, right in zip(a, p, strict=True)],
            'actual_tangent_first_order_agreement': agrees, 'actual_tangent_direction_identifiable': actual_identifiable,
            'state_line_selfchecks_passed': bool(state_numeric_valid), 'model_all_abs_ge_1e_5': model_magnitude,
            'model_all_four_same_sign': model_same_sign, 'model_raw_sign_matches_all_eps': signs,
            'valid_local_direction_match': True if status == 'LOCAL_DIRECTION_MATCH' else
                False if status == 'LOCAL_DIRECTION_MISMATCH' else None,
            'physical_eligibility_changed': False}


def project_cost_response(start, directions, response, physical_row, physical_arrays, *, arrays=None):
    """True-post-state cost Jacobian applied to actual and model next-state tangents.

    Both projections use actual issued-action and previous-action tangents. Their
    difference must therefore equal C_state times the model joint state error.
    The physical chord is finite: its projected cost need not equal physical FD.
    """
    import numpy as np
    import torch

    from tools54 import numerics54 as numeric
    arrays = {} if arrays is None else arrays
    state = torch.tensor(directions['zero_next_state'], dtype=torch.float64)
    action = torch.tensor(directions['zero_action'], dtype=torch.float64)
    previous_action = torch.tensor(directions['zero_previous_action'], dtype=torch.float64)
    command = torch.tensor(directions['commands'], dtype=torch.float64)

    def cost_outputs(s, a, p):
        return (numeric.cost(s, command, a, p, start['nominal_height']),
                numeric.cost_components(s, command, a, p, start['nominal_height']))

    parity = []
    for ep, epsilon in enumerate(EPSILONS):
        actual_state = torch.tensor(directions['delta_next_state'][ep], dtype=torch.float64)
        model_state = torch.tensor(response[f'epsilon_{ep}_joint_jvp'], dtype=torch.float64)
        da = torch.tensor(directions['delta_action'][ep], dtype=torch.float64)
        dp = torch.tensor(directions['delta_previous_action'][ep], dtype=torch.float64)
        derivatives = {}
        for kind, ds, va, vp in (('actual', actual_state, da, dp), ('model', model_state, da, dp),
                                ('state_error', model_state - actual_state, torch.zeros_like(da), torch.zeros_like(dp))):
            values, tangent = torch.autograd.functional.jvp(cost_outputs, (state, action, previous_action), (ds, va, vp),
                                                            create_graph=False, strict=False)
            require(np.array_equal(values[0].numpy(), physical_arrays['branch_costs'][0])
                    and np.array_equal(values[1].numpy(), physical_arrays['branch_components'][0]),
                    'cost Jacobian base does not exactly restore65 physical baseline')
            check = prior.close_receipt(tangent[1].sum(-1).numpy(), tangent[0].numpy())
            require(check['passed'], 'projected component sum does not restore total')
            derivatives[kind] = tangent
            for label, value in zip(('total', 'component'), tangent, strict=True):
                raw = value.numpy()
                arrays[f'epsilon_{ep}_{kind}_{label}_per_step'] = raw
                arrays.update({f'epsilon_{ep}_{kind}_{label}_{name}': item for name, item in prior.prefix_values(raw).items()})
                if kind != 'state_error':
                    for prefix in ('per_step', 'cumulative', 'prefix_mean'):
                        arrays[f'epsilon_{ep}_{kind}_{label}_{prefix}_minus_physical_fd'] = (
                            arrays[f'epsilon_{ep}_{kind}_{label}_{prefix}']
                            - physical_arrays[f'epsilon_{ep}_{label}_{prefix}_central'])
        identities = {label: prior.close_receipt((derivatives['model'][index] - derivatives['actual'][index]).numpy(),
                                                 derivatives['state_error'][index].numpy())
                      for index, label in enumerate(('total', 'component'))}
        require(all(check['passed'] for check in identities.values()), 'cost response residual identity failed')
        parity.append({'epsilon': epsilon, 'model_minus_actual_equals_Cstate_state_error': identities})
    rows = []
    for tick in range(32):
        numeric_valid = all(bool(response[f'epsilon_{ep}_line_selfcheck'][:tick + 1].all()) for ep in range(4))
        row = {'horizon': tick + 1, 'registered_summary_horizon': tick + 1 in HORIZONS,
               'all_prefix_state_line_selfchecks': numeric_valid, 'components': {}}
        for label, column, reference in [('total', None, physical_row['prefixes'][tick])] + [
                ('component', index, physical_row['component_prefixes'][name][tick]) for index, name in enumerate(COMPONENTS)]:
            coordinate = tick if column is None else (tick, column)
            values = [[float(source[f'epsilon_{ep}_{key}'][coordinate]) for ep in range(4)] for source, key in (
                (physical_arrays, label + '_prefix_mean_central'),
                (arrays, 'actual_' + label + '_prefix_mean'), (arrays, 'model_' + label + '_prefix_mean'))]
            comparison = projection_comparison(reference, *values, numeric_valid)
            if column is None:
                row['total'] = comparison
            else:
                row['components'][COMPONENTS[column]] = comparison
        rows.append(row)
    return {'status': 'COMPLETE', 'cost_baseline_exact': True, 'residual_identity': parity, 'prefixes': rows}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-joint-response67-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'new_physics_steps': 0, 'model_updates': 0, 'policy_updates': 0, 'actor_evaluations': 0,
              'source_decisions_changed': False, 'diagnostics_complete': False, 'is_controller_repair': False,
              'selected_seed': None, 'selected_horizon': None, 'completed_model_points': 0,
              'physical': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN'} for seed in PROBE_SEEDS for tick in TICKS],
              'registered_failure_appendix': [],
              'models': [{'seed': seed, 'arm': arm, 'status': 'NOT_RUN', 'points': [
                  {'seed': reset, 'tick': tick, 'status': 'NOT_RUN'} for reset in PROBE_SEEDS for tick in TICKS]}
                  for seed in MODEL_SEEDS for arm in ARMS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    plan, original, plan66 = None, None, None
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
        verify67(plan)
        original = json.loads((ROOT / prior.PLAN53).read_text())
        verify(original)
        plan66 = json.loads((ROOT / PLAN66).read_text())
        rebase66.verify66(plan66)
        source64 = json.loads((ROOT / SOURCE64 / 'report.json').read_text())
        source65 = json.loads((ROOT / SOURCE65 / 'report.json').read_text())
        source66 = json.loads((ROOT / SOURCE66 / 'report.json').read_text())
        require(all(source['status'] == 'COMPLETE' and source['verification_after'] == 'COMPLETE'
                    for source in (source64, source65, source66))
                and source65['completed_model_points'] == source66['completed_model_points'] == 54
                and source64['completed_model_fits'] == 6 and source64['model_updates'] == 6000
                and source66['plan_sha256'] == sha(ROOT / PLAN66), 'complete64/65/66 sources required')
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(all(source['versions'] == report['versions'] == original['versions'] for source in (source64, source65, source66)),
                'package versions differ')
        import numpy as np
        import torch

        from tools52 import dynamics52 as dynamics
        from tools52.run_pilot52 import save_npz
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(all(source['torch_runtime_version'] == torch.__version__ for source in (source64, source65, source66)),
                'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        with (ROOT / SOURCE64 / 'probe_snapshots.pkl').open('rb') as stream:
            points = pickle.load(stream)  # Own-generated, exact SHA-admitted local snapshots only.
        require([(point['seed'], point['tick']) for point in points] ==
                [(seed, tick) for seed in PROBE_SEEDS for tick in TICKS], 'point identities differ')
        physical_output = args.output / 'physical'
        physical_output.mkdir(exist_ok=False)
        directions, references, physical_data = [], [], []
        for index, start in enumerate(points):
            progress.update(phase='physical_readback', point=index)
            row, arrays = report['physical'][index], {}
            row['status'] = 'INCOMPLETE'
            try:
                reference = source65['physical'][index]
                require(reference == json.loads((ROOT / SOURCE65 / f'physical/point_{index:02d}.json').read_text())
                        and reference['status'] == 'COMPLETE'
                        and (reference['seed'], reference['tick']) == (start['seed'], start['tick'])
                        and reference['source64_reference'] == source64['cost_references']['fresh'][index], '65 reference differs')
                with np.load(ROOT / SOURCE65 / f'physical/point_{index:02d}.npz', allow_pickle=False) as raw:
                    saved = {key: raw[key].copy() for key in raw.files}
                branches = []
                for order, theta in enumerate([0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]):
                    stem = ROOT / SOURCE64 / f'grid_{index:02d}_{order:02d}'
                    metadata = json.loads(stem.with_suffix('.json').read_text())
                    require(metadata == source64['physical_probes'][index]['branches'][order]
                            and metadata['status'] == 'COMPLETE' and metadata['steps'] == 32
                            and metadata['theta'] == theta and not metadata['unsafe_steps'], 'partial/different physical branch')
                    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as raw:
                        branch = {key: raw[key].copy() for key in ('states', 'observations', 'rewards')}
                    require(all(np.array_equal(branch[key], saved[f'branch_{order}_{key}']) for key in branch),
                            '65 physical raw differs from64')
                    branches.append(branch)
                arrays = build_directions(start, branches)
                directions.append(arrays)
                references.append(reference)
                physical_data.append(saved)
                row.update(status='COMPLETE', source64_reference=reference['source64_reference'],
                    t0_prestate_direction_zero=True, t0_previous_action_direction_zero=True,
                    midpoint_drift_max_abs={name: [float(np.abs(arrays['midpoint_drift_' + name][ep]).max()) for ep in range(4)]
                                           for name in ('pre_state', 'action', 'next_state', 'previous_action')})
            except BaseException as exc:
                row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                save_npz(physical_output / f'point_{index:02d}.npz', arrays)
                write_json(physical_output / f'point_{index:02d}.json', row)
                event('PHYSICAL_SAVED')
        for model_row in report['models']:
            require(time.process_time() < 178., 'CPU deadline before model')
            model_row['status'] = 'INCOMPLETE'
            directory = f'{model_row["arm"]}_seed{model_row["seed"]}'
            output = args.output / directory
            output.mkdir(exist_ok=False)
            progress.update(phase='model', model=directory, point=None)
            old64 = next(row for row in source64['models'] if (row['arm'], row['seed']) == (model_row['arm'], model_row['seed']))
            old66 = next(row for row in source66['models'] if (row['arm'], row['seed']) == (model_row['arm'], model_row['seed']))
            require(old64['status'] == old66['status'] == 'COMPLETE' and old64['fit']['status'] == 'COMPLETE'
                    and old64['fit']['updates'] == 1000, 'complete final model required')
            weights = directory + '/final_state_dict.pt'
            require(sha(ROOT / SOURCE64 / weights) == source64['sealed_artifacts_sha256'][weights], '64 final weights changed')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(ROOT / SOURCE64 / weights, map_location='cpu', weights_only=True))
            model = model.double().eval().requires_grad_(False)
            for index, start in enumerate(points):
                require(time.process_time() < 178., 'CPU deadline before point')
                row, response_arrays, cost_arrays = model_row['points'][index], {}, {}
                row['status'] = 'INCOMPLETE'
                progress.update(phase='model_point', point=index)
                event('POINT_BEGIN')
                try:
                    old = old66['points'][index]
                    require(old == json.loads((ROOT / SOURCE66 / directory / f'point_{index:02d}.json').read_text())
                            and old['status'] == 'COMPLETE' and (old['seed'], old['tick']) == (start['seed'], start['tick']),
                            '66 point identity differs')
                    row['joint_response'], _ = joint_response(model, directions[index], deadline=178., arrays=response_arrays)
                    row['cost_projection'], _ = project_cost_response(start, directions[index], response_arrays,
                        references[index], physical_data[index], arrays=cost_arrays)
                    row.update(source64_reference=references[index]['source64_reference'], status='COMPLETE')
                    for horizon in (1, 32):
                        old_free = old['free']['prefixes'][horizon - 1]
                        if model_row['arm'] == 'active' and old_free['valid_direction_match'] is False:
                            report['registered_failure_appendix'].append({'model_seed': model_row['seed'],
                                'probe_seed': start['seed'], 'tick': start['tick'], 'horizon': horizon,
                                'free65': old_free, 'cost_rebased66': old['cost_rebased']['prefixes'][horizon - 1],
                                'local_joint67': row['cost_projection']['prefixes'][horizon - 1]['total']})
                    report['completed_model_points'] += 1
                except BaseException as exc:
                    row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    save_npz(output / f'point_{index:02d}.npz', response_arrays | cost_arrays)
                    write_json(output / f'point_{index:02d}.json', row)
                    event('POINT_SAVED')
            model_row['status'] = 'COMPLETE'
            model_row['summary_by_horizon'] = {str(horizon): {
                status: sum(point['cost_projection']['prefixes'][horizon - 1]['total']['status'] == status
                            for point in model_row['points']) for status in (
                    'LOCAL_DIRECTION_MATCH', 'LOCAL_DIRECTION_MISMATCH', 'REFERENCE_UNRESOLVED',
                    'NUMERIC_UNRESOLVED', 'ACTUAL_COST_TANGENT_UNRESOLVED', 'MODEL_DIRECTION_UNRESOLVED')}
                for horizon in HORIZONS}
            write_json(output / 'result.json', model_row)
        require(report['completed_model_points'] == 54 and len(report['registered_failure_appendix']) == 5,
                'all54 points and frozen five-case appendix required')
        report.update(status='COMPLETE', diagnostics_complete=True)
    except BaseException as exc:  # noqa: BLE001 - partial is retained, never resumed.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], '67 plan changed')
                verify67(plan)
            if plan66 is not None:
                rebase66.verify66(plan66)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            report['verification_after'] = 'COMPLETE' if original is not None and plan66 is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - preserve both original and final verification failures.
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
