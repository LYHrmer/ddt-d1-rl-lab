"""69 extra output checks: real-action body response and free first-step cost.

H1 free predictions are written before fresh physical perturbations. Body
diagnostics condition on the actual recorded actions, like the inherited yaw
first-step check; they are not a free closed-loop policy gradient. All four
theta scales and each component's own reference qualification are retained.
"""
from __future__ import annotations

import math
import time

from tools53.run_pilot53 import require, write_json
from tools55.probe_local55 import EPSILONS, action_delta_ulps, local_stability
from tools65.diagnose65 import MODEL_EPS, close_receipt, selfcheck

CHANNELS = ('vx', 'vy')
# Exact F64 promotion of frozen52's float32 PHYSICAL_SCALE[23]. The executing
# model path checks this identity; gate-only tests can remain standard-library.
BODY_SCALE = 0.30000001192092896


def body_output(state):
    from tools54 import numerics54 as numeric
    return numeric._rotate_inverse(state[..., 3:7], state[..., 23:26])[..., :2]


def scalar_reference(zero, plus_minus, actions_resolved):
    require(len(plus_minus) == 4, 'all four theta scales required')
    rows = [{'epsilon': epsilon, 'central64': (plus - minus) / (2 * epsilon),
             'forward64': (plus - zero) / epsilon, 'backward64': (zero - minus) / epsilon}
            for epsilon, (plus, minus) in zip(EPSILONS, plus_minus, strict=True)]
    require(all(math.isfinite(row[key]) for row in rows for key in ('central64', 'forward64', 'backward64')),
            'nonfinite output reference')
    stability = local_stability(rows)
    return {'eps': rows, 'stability': stability, 'all_four_wheels_resolved': bool(actions_resolved),
            'resolved': bool(actions_resolved and stability['locally_stable']),
            'gradient': math.fsum(row['central64'] for row in rows) / 4}


def head_references(points, grids):
    import numpy as np
    import torch

    from tools54 import numerics54 as numeric
    require(len(points) == len(grids) == 9, 'all nine head reference points required')
    references = []
    for start, grid in zip(points, grids, strict=True):
        require(len(grid) == 10 and np.asarray(start['fifo']).shape == (0, 16), 'complete zero-delay grid required')
        states = torch.tensor(np.stack([branch['states'][0] for branch in grid]), dtype=torch.float64)
        actions = torch.tensor(np.stack([branch['observations'][0, 41:57] for branch in grid]), dtype=torch.float64)
        previous = torch.tensor(start['last'], dtype=torch.float64).expand_as(actions)
        commands = torch.tensor(start['commands'][0], dtype=torch.float64).expand(10, 3)
        with torch.no_grad():
            body = body_output(states).numpy()
            cost = numeric.cost(states, commands, actions, previous, start['nominal_height']).numpy()
        require(np.array_equal(body[0], body[1]) and cost[0] == cost[1], 'repeated zero outputs differ')
        actions_resolved = all(action_delta_ulps(grid[0]['observations'][0, 41:57],
            branch['observations'][0, 41:57])['all_four_wheels_nonzero'] for branch in grid[2:])
        row = {'seed': start['seed'], 'tick': start['tick'], 'body': {},
               'coordinate': 'original policy theta; actual first F32 issued actions at zero delay'}
        for column, name in enumerate(CHANNELS):
            row['body'][name] = scalar_reference(float(body[0, column]),
                [(float(body[2 + 2 * ep, column]), float(body[3 + 2 * ep, column])) for ep in range(4)], actions_resolved)
        row['cost_h1'] = scalar_reference(float(cost[0]),
            [(float(cost[2 + 2 * ep]), float(cost[3 + 2 * ep])) for ep in range(4)], actions_resolved)
        references.append(row)
    return references


def evaluate_h1_costs(model, actor, points, output, ledger, deadline):
    """Seal these free H1 AD/FD predictions before fresh perturbed physical truth."""
    import torch

    from tools52.run_pilot52 import save_npz
    from tools54 import numerics54 as numeric
    require(len(points) == 9, 'all nine H1 free predictions required')
    output.mkdir(parents=True, exist_ok=False)
    ledger.update(status='INCOMPLETE', points=[], completed_points=0)
    try:
        for index, start in enumerate(points):
            require(time.process_time() < deadline, 'CPU deadline before H1 prediction')
            row = {'seed': start['seed'], 'tick': start['tick'], 'status': 'INCOMPLETE'}
            ledger['points'].append(row)
            arrays = {}
            try:
                theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
                value, trace = numeric.imaginary(model, actor, start, theta, return_trace=True, horizon=1)
                gradient, = torch.autograd.grad(value, theta)
                arrays.update({'theta0_' + name: item.detach().numpy() for name, item in trace.items()})
                endpoints = []
                with torch.no_grad():
                    for sign, name in ((1, 'plus'), (-1, 'minus')):
                        cost, current = numeric.imaginary(model, actor, start, torch.tensor(sign * MODEL_EPS,
                            dtype=torch.float64), return_trace=True, horizon=1)
                        endpoints.append(float(cost))
                        arrays.update({name + '_' + key: item.numpy() for key, item in current.items()})
                fd = (endpoints[0] - endpoints[1]) / (2 * MODEL_EPS)
                check = selfcheck(float(gradient), fd)
                reconciliation = close_receipt(trace['cost_components'].sum(-1).detach().numpy(),
                                               trace['costs'].detach().numpy())
                require(reconciliation['passed'] and trace['states'].shape == (1, 45), 'H1 cost trace differs')
                row.update(**check, cost=float(value.detach()), model_fd_epsilon=MODEL_EPS,
                           physical_perturbed_data_used=False, components_to_total=reconciliation, status='COMPLETE')
                ledger['completed_points'] += 1
            except BaseException as exc:
                row['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                save_npz(output / f'point_{index:02d}.npz', arrays)
                write_json(output / f'point_{index:02d}.json', row)
        ledger['status'] = 'COMPLETE'
    finally:
        write_json(output / 'result.json', ledger)
    return ledger


def evaluate_heads(model, points, grids, cost_h1_predictions, output, ledger, deadline):
    """Conditional body response uses the same actual-action seam as first-yaw."""
    import numpy as np
    import torch

    from tools52.dynamics52 import PHYSICAL_SCALE
    from tools52.run_pilot52 import save_npz
    require(BODY_SCALE == float(PHYSICAL_SCALE[23]), 'frozen body output scale changed')
    require(len(points) == len(grids) == len(cost_h1_predictions) == 9, 'all nine head points required')
    output.mkdir(parents=True, exist_ok=False)
    ledger.update(status='INCOMPLETE', completed_points=0, points=[])
    try:
        for index, (start, grid, cost) in enumerate(zip(points, grids, cost_h1_predictions, strict=True)):
            require(time.process_time() < deadline, 'CPU deadline before body output diagnostic')
            require(cost['status'] == 'COMPLETE' and (cost['seed'], cost['tick']) == (start['seed'], start['tick']),
                    'precomputed H1 prediction identity differs')
            require(np.asarray(start['fifo']).shape == (0, 16), 'body output requires original zero delay')
            row = {'seed': start['seed'], 'tick': start['tick'], 'status': 'INCOMPLETE',
                   'body': {name: {'eps': []} for name in CHANNELS}, 'cost_h1': dict(cost)}
            ledger['points'].append(row)
            arrays = {}
            try:
                state = torch.tensor(start['state'][None], dtype=torch.float64)
                center = torch.tensor(grid[0]['observations'][0:1, 41:57], dtype=torch.float64)

                def output_body(action, state=state):
                    return body_output(model(state, action))

                for ep, epsilon in enumerate(EPSILONS):
                    require(time.process_time() < deadline, 'CPU deadline during body output diagnostic')
                    plus = torch.tensor(grid[2 + 2 * ep]['observations'][0:1, 41:57], dtype=torch.float64)
                    minus = torch.tensor(grid[3 + 2 * ep]['observations'][0:1, 41:57], dtype=torch.float64)
                    direction = (plus - minus) / (2 * epsilon)
                    primal, jvp = torch.autograd.functional.jvp(output_body, center, direction,
                                                               create_graph=False, strict=False)
                    with torch.no_grad():
                        body_plus, body_minus = output_body(plus), output_body(minus)
                        fd = (body_plus - body_minus) / (2 * epsilon)
                        line_fd = (output_body(center + MODEL_EPS * direction)
                                   - output_body(center - MODEL_EPS * direction)) / (2 * MODEL_EPS)
                    require(all(bool(torch.isfinite(item).all()) for item in (jvp, fd, line_fd)), 'nonfinite body prediction')
                    arrays.update({f'epsilon_{ep}_{name}': value.numpy() for name, value in (
                        ('actual_action_plus', plus), ('actual_action_minus', minus), ('actual_action_v', direction),
                        ('action_midpoint_drift', (plus + minus) / 2 - center), ('body_zero', primal),
                        ('body_plus', body_plus), ('body_minus', body_minus), ('body_fd', fd), ('body_jvp', jvp),
                        ('body_line_fd', line_fd))})
                    for column, name in enumerate(CHANNELS):
                        analytic, actual_fd, straight_fd = (float(item[0, column]) for item in (jvp, fd, line_fd))
                        line_check, endpoint_check = selfcheck(analytic, straight_fd), selfcheck(analytic, actual_fd)
                        row['body'][name]['eps'].append({'epsilon': epsilon, 'model_fd': actual_fd,
                            'model_jvp': analytic, 'model_line_fd': straight_fd, 'line_selfcheck': line_check,
                            'endpoint_selfcheck': endpoint_check, 'numeric_valid': bool(line_check['autograd_agrees']
                                and endpoint_check['autograd_agrees'] and line_check['direction_identifiable']
                                and endpoint_check['direction_identifiable'])})
                row['status'] = 'COMPLETE'
                ledger['completed_points'] += 1
            except BaseException as exc:
                row['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                save_npz(output / f'point_{index:02d}.npz', arrays)
                write_json(output / f'point_{index:02d}.json', row)
        ledger['status'] = 'COMPLETE'
    finally:
        write_json(output / 'result.json', ledger)
    return ledger


def head_comparison(predictions, references):
    require(len(predictions) == len(references) == 9, 'all nine comparison points required')
    result = {'status': 'COMPLETE', 'points': [], 'body_valid_matches': 0, 'body_resolved_count': 0,
              'cost_h1_valid_matches': 0, 'cost_h1_resolved_count': 0, 'numeric_valid': True}
    squared_errors = []
    for prediction, reference in zip(predictions, references, strict=True):
        require(prediction['status'] == 'COMPLETE'
                and (prediction['seed'], prediction['tick']) == (reference['seed'], reference['tick']), 'head identities differ')
        row = {'seed': prediction['seed'], 'tick': prediction['tick'], 'body': {}}
        for name in CHANNELS:
            pred, ref = prediction['body'][name]['eps'], reference['body'][name]
            require([x['epsilon'] for x in pred] == [x['epsilon'] for x in ref['eps']] == list(EPSILONS),
                    'complete body epsilon grid required')
            eps = []
            for p, q in zip(pred, ref['eps'], strict=True):
                same = p['model_fd'] * q['central64'] > 0 if ref['resolved'] else None
                valid = bool(p['numeric_valid'] and abs(p['model_fd']) >= 1e-5 and same) if ref['resolved'] else None
                result['numeric_valid'] &= p['numeric_valid']
                squared_errors.append(((p['model_fd'] - q['central64']) / BODY_SCALE) ** 2)
                eps.append({'epsilon': p['epsilon'], 'model_fd': p['model_fd'], 'true_fd': q['central64'],
                            'same_sign': same, 'numeric_valid': p['numeric_valid'], 'valid_match': valid})
            passed = all(p['valid_match'] is True for p in eps) if ref['resolved'] else None
            row['body'][name] = {'resolved': ref['resolved'], 'eps': eps, 'passed': passed}
            result['body_resolved_count'] += ref['resolved']
            result['body_valid_matches'] += passed is True
        pred, ref = prediction['cost_h1'], reference['cost_h1']
        numeric = pred['autograd_agrees'] and pred['direction_identifiable']
        valid = bool(numeric and pred['gradient'] * ref['gradient'] > 0) if ref['resolved'] else None
        row['cost_h1'] = {'reference_resolved': ref['resolved'], 'prediction': pred['gradient'],
                         'reference': ref['gradient'], 'valid_match': valid, 'numeric_valid': numeric}
        result['cost_h1_resolved_count'] += ref['resolved']
        result['cost_h1_valid_matches'] += valid is True
        result['numeric_valid'] &= numeric
        result['points'].append(row)
    require(len(squared_errors) == 9 * 4 * 2 and all(math.isfinite(value) for value in squared_errors),
            'complete finite body error grid required')
    mse = math.fsum(squared_errors) / len(squared_errors)
    result.update(body_normalized_mse=mse, body_normalized_rmse=math.sqrt(mse),
                  body_error_scalar_count=72, body_error_includes_unresolved_reference_values=True,
                  body_passed_18_of18=result['body_resolved_count'] == result['body_valid_matches'] == 18,
                  cost_h1_passed_9_of9=result['cost_h1_resolved_count'] == result['cost_h1_valid_matches'] == 9,
                  all_references_resolved=result['body_resolved_count'] == 18 and result['cost_h1_resolved_count'] == 9)
    return result
