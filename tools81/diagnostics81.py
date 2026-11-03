"""Cross-score sealed81 traces without changing candidate selection or gates."""
from __future__ import annotations

import copy
import math
from fractions import Fraction

PREFIXES = (1, 8, 16, 32)
SCALES = (.009, .0045, .00225, .001125)


def physical_cost_parts(point, arrays):
    import torch

    from tools54 import numerics54 as numeric

    state = torch.as_tensor(arrays['states'], dtype=torch.float64)
    action = torch.as_tensor(arrays['observations'][:, 41:57], dtype=torch.float64)
    previous = torch.cat((torch.as_tensor(point['last'][None], dtype=torch.float64), action[:-1]))
    command = torch.as_tensor(arrays['commands'], dtype=torch.float64)
    with torch.no_grad():
        return numeric.cost_components(state, command, action, previous, point['nominal_height']).numpy()


def trace_metrics(arrays):
    import numpy as np

    yaw = arrays['body_wz'] if 'body_wz' in arrays else arrays['states'][:, 28]
    errors = yaw - arrays['commands'][:, 2]
    actions = arrays['issued_actions'] if 'issued_actions' in arrays else arrays['actions']
    assert len(errors) == len(arrays['costs']) == 32
    assert np.isfinite(errors).all() and np.isfinite(arrays['costs']).all()
    return {'yaw_mae': float(np.mean(np.abs(errors))),
            'prefix_metrics': {str(h): {'cost': float(np.mean(arrays['costs'][:h])),
                'yaw_mae': float(np.mean(np.abs(errors[:h]))),
                'cost_components': np.mean(arrays['cost_components'][:h], axis=0).tolist(),
                'zero_yaw_error_count': int(np.count_nonzero(errors[:h] == 0.)),
                'minimum_abs_yaw_error': float(np.min(np.abs(errors[:h])))} for h in PREFIXES},
            'action_at_bound_coordinates': int(np.count_nonzero(np.abs(actions) >= 1.)),
            'action_at_bound_semantics': 'observed saturation boundary, not evidence of preclip excess'}


def validation_ad(model64, anchor, points, directions, model_seed, objective_values, cpu_deadline):
    """Differentiate V anchor objectives, projecting only onto presealed D directions."""
    import torch

    actor64 = copy.deepcopy(anchor).double().eval().requires_grad_(False)
    actor64.action_head.requires_grad_(True)
    values = objective_values(model64, actor64, points, cpu_deadline=cpu_deadline)
    point_rows = values['points']
    seeds = sorted({p['seed'] for p in point_rows})
    result = []
    for index, seed in enumerate(seeds):
        selected = [p for p in point_rows if p['seed'] == seed]
        assert len(selected) == 2
        scalars = {'C': (selected[0]['cost'] + selected[1]['cost']) / 2,
                   'Q': values['per_seed_q'][index]}
        for metric, scalar in scalars.items():
            grads = torch.autograd.grad(scalar, tuple(actor64.action_head.parameters()), retain_graph=True)
            for arm, direction in directions.items():
                derivative = float(sum((g * d).sum() for g, d in zip(grads, direction, strict=True)))
                assert math.isfinite(derivative)
                result.append({'model_seed': model_seed, 'arm': arm, 'seed': seed, 'metric': metric,
                    'anchor_value': float(scalar.detach()), 'model_directional_ad': derivative,
                    'horizon': 32, 'direction_source': 'D_sealed_direction',
                    'physical_ad': None, 'V_used_for_updates': False})
    return result


def finite_response(zero, plus, minus, h):
    """Exact difference of stored binary64 scalars; no physical precision guarantee."""
    assert all(math.isfinite(x) for x in (zero, plus, minus, h)) and h > 0
    z, p, m, step = map(Fraction.from_float, (zero, plus, minus, h))
    return {'zero': zero, 'plus': plus, 'minus': minus,
            'plus_delta': float(p - z), 'minus_delta': float(m - z),
            'central': float((p - m) / (2 * step)),
            'forward': float((p - z) / step), 'backward': float((z - m) / step),
            'strong_monotone': plus < zero < minus,
            'stored_plus_equals_zero': plus == zero, 'stored_minus_equals_zero': minus == zero}


def _pair(rows, horizon):
    assert len(rows) == 2 and {r['tick'] for r in rows} == {175, 325}
    metrics = [r['prefix_metrics'][str(horizon)] for r in rows]
    yaw = [m['yaw_mae'] for m in metrics]
    return {'C': sum(m['cost'] for m in metrics) / 2, 'Q': max(yaw),
            'max_ticks': [r['tick'] for r, value in zip(rows, yaw, strict=True) if value == max(yaw)]}


def cross_readback(predictions, branches, ad_rows, output):
    """Read only persisted traces after all physical branches; no new forward calls."""
    import numpy as np

    from tools55.probe_local55 import action_delta_ulps

    model_seeds = (64001, 64002, 64003)
    resets = (81207, 81208, 81209)
    ad = {(r['model_seed'], r['arm'], r['seed'], r['metric']): r['model_directional_ad'] for r in ad_rows}
    result = {'schema': 'd1-objective81-cross-readback-v1', 'status': 'COMPLETE',
              'new_model_calls': 0, 'new_physics_steps': 0, 'selected_scale': None,
              'physical_ad': None, 'responses': [], 'actual_action_response': [],
              'model_ad_fd_comparison': 'AD follows ideal D direction; FD uses stored F32 endpoints; difference includes quantization and finite-step effects',
              'fd_semantics': 'fixed endpoints; stored scalar rounding only, no confidence or Jacobian qualification'}
    for model_seed in model_seeds:
        for arm in ('C6', 'Q6'):
            for seed in resets:
                for k, h in enumerate(SCALES):
                    grouped = {}
                    for label, rows in (('model', predictions), ('physical', branches)):
                        local = [r for r in rows if r['seed'] == seed and r['tick'] in (175, 325)
                                 and (label == 'model' or r['repeat'] == 0)]
                        anchor = [r for r in local if r['arm'] == 'anchor'
                                  and (label == 'physical' or r['model_seed'] == model_seed)]
                        endpoints = {sign: [r for r in local if r['arm'] == arm
                            and r['model_seed'] == model_seed and r['scale_index'] == k and r['sign'] == sign]
                            for sign in (1, -1)}
                        grouped[label] = (anchor, endpoints)
                    for horizon in PREFIXES:
                        row = {'model_seed': model_seed, 'arm': arm, 'seed': seed,
                               'scale_index': k, 'h': h, 'horizon': horizon}
                        for label, (anchor, endpoints) in grouped.items():
                            z = _pair(anchor, horizon)
                            p, m = (_pair(endpoints[sign], horizon) for sign in (1, -1))
                            row[label] = {metric: finite_response(z[metric], p[metric], m[metric], h)
                                          for metric in ('C', 'Q')}
                            row[label]['max_ticks'] = {'zero': z['max_ticks'], 'plus': p['max_ticks'], 'minus': m['max_ticks']}
                            row[label]['max_tick_switch'] = z['max_ticks'] != p['max_ticks'] or z['max_ticks'] != m['max_ticks']
                        for metric in ('C', 'Q'):
                            row.setdefault('gain_ratio', {})[metric] = None
                            row.setdefault('plus_delta_same_nonzero_sign', {})[metric] = (
                                row['model'][metric]['plus_delta'] * row['physical'][metric]['plus_delta'] > 0)
                        row['gain_ratio_null_reason'] = 'no independently qualified prediction-resolution rule; raw deltas retained'
                        row['model_ad'] = {metric: ad[(model_seed, arm, seed, metric)] for metric in ('C', 'Q')} if horizon == 32 else None
                        result['responses'].append(row)
    # First physical policy input is independently archived; do not infer parity from snapshot names.
    anchors = {(r['seed'], r['tick']): r for r in branches if r['arm'] == 'anchor' and r['repeat'] == 0}
    preds = {(r['model_seed'], r['arm'], r['scale_index'], r['sign'], r['seed'], r['tick']): r for r in predictions}
    for row in branches:
        if row['arm'] == 'anchor':
            continue
        baseline = anchors[(row['seed'], row['tick'])]
        pred = preds[(row['model_seed'], row['arm'], row['scale_index'], row['sign'], row['seed'], row['tick'])]
        model_anchor = preds[(row['model_seed'], 'anchor', None, 0, row['seed'], row['tick'])]
        with np.load(output / baseline['trace_path'], allow_pickle=False) as z, \
             np.load(output / row['trace_path'], allow_pickle=False) as p, \
             np.load(output / pred['trace_path'], allow_pickle=False) as predicted, \
             np.load(output / model_anchor['trace_path'], allow_pickle=False) as predicted_zero:
            zerr, perr = z['body_wz'] - z['commands'][:, 2], p['body_wz'] - p['commands'][:, 2]
            mzerr = predicted_zero['states'][:, 28] - predicted_zero['commands'][:, 2]
            mperr = predicted['states'][:, 28] - predicted['commands'][:, 2]
            result['actual_action_response'].append({k: row[k] for k in
                ('model_seed', 'arm', 'scale_index', 'sign', 'seed', 'tick')} | {
                'first_action': action_delta_ulps(z['issued_actions'][0], p['issued_actions'][0]),
                'physical_first_actor_input_bit_exact': np.array_equal(z['first_actor_input'], p['first_actor_input']),
                'model_physical_first_actor_input_max_abs': float(np.max(np.abs(
                    predicted['actor_inputs'][0] - p['first_actor_input'].astype(np.float64)))),
                'yaw_error_sign_switches_from_anchor': int(np.count_nonzero(np.sign(zerr) != np.sign(perr))),
                'model_yaw_error_sign_switches_from_anchor': int(np.count_nonzero(np.sign(mzerr) != np.sign(mperr)))})
    return result
