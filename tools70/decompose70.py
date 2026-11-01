"""F64 endpoint identities for the frozen54 six-term H1 cost.

These are finite-interval accounting identities, not causal interventions or
identification of a dynamics Jacobian. No model or physical rollout is called.
The caller retains saved-cost rounding and model AD-minus-FD separately.
"""
from __future__ import annotations

import math

NAMES = ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness')
GROUPS = {'vx': (0,), 'yaw': (1,), 'lateral': (2,), 'tilt': (3, 4),
          'height': (5,), 'action_smoothness': tuple(range(6, 22))}
COORDINATE_NAMES = ('vx_error_over_02', 'yaw_error_over_02', 'vy_over_02',
                    'gravity_x', 'gravity_y', 'height_error',
                    *(f'action_delta_{index:02d}' for index in range(16)))
WEIGHTS = (1., 1., 1., 6., 6., 20., *((.1 / 16,) * 16))
ATOL, RTOL = 1e-10, 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _f64(value, name, width=None):
    import numpy as np

    array = np.asarray(value)
    _require(array.dtype == np.dtype('float64'), name + ' must retain float64 inputs')
    _require(array.size > 0 and bool(np.isfinite(array).all()), name + ' must be nonempty and finite')
    if width is not None:
        _require(array.ndim >= 1 and array.shape[-1] == width, name + f' must end in {width} coordinates')
    return array


def component_costs(coordinates):
    """Six weighted-square groups in the original54 within-group order.

The first three coordinates already contain the division by .2. Tilt retains
6*sum(square), and smooth retains .1*mean(square); neither is replaced by a
different weighting/reduction order. The total cost's left-associated sum
belongs to the caller's frozen54 reconciliation.
    """
    import numpy as np

    values = _f64(coordinates, 'coordinates', 22)
    with np.errstate(over='ignore', invalid='ignore'):
        square = values * values
        groups = np.stack((square[..., 0], square[..., 1], square[..., 2],
                           6 * square[..., 3:5].sum(axis=-1), 20 * square[..., 5],
                           .1 * square[..., 6:22].mean(axis=-1)), axis=-1)
    return _f64(groups, 'component costs', 6)


def residuals(state, command, action, previous, nominal_height):
    """Return F64 coordinates[...,22] and components[...,6], without rollout.

All array inputs must already be float64. Promote actual F32 issued actions
before calling; do not compute their differences in float32. Leading shapes
must match. A finite scalar nominal height may broadcast over those shapes.
Quaternion normalization and inverse rotation use the frozen54 implementation.
    """
    import numpy as np
    import torch

    from tools54 import numerics54 as frozen

    state = _f64(state, 'state', 45)
    command = _f64(command, 'command', 3)
    action = _f64(action, 'action', 16)
    previous = _f64(previous, 'previous', 16)
    shape = state.shape[:-1]
    _require(all(value.shape[:-1] == shape for value in (command, action, previous)),
             'state, command, action and previous leading shapes must match')
    height = _f64(nominal_height, 'nominal_height')
    try:
        height = np.broadcast_to(height, shape)
    except ValueError as exc:
        raise ValueError('nominal_height does not broadcast to state shape') from exc
    _require(bool((np.abs(state[..., 3:7]).max(axis=-1) > 0).all()), 'nonzero quaternion required')
    # Copies isolate readback from caller buffers, including read-only NPZ views.
    tensor = torch.tensor(state, dtype=torch.float64, device='cpu')
    with torch.no_grad():
        velocity = frozen._rotate_inverse(tensor[..., 3:7], tensor[..., 23:26]).numpy()
        world_gravity = tensor.new_tensor([0., 0., -1.]).expand_as(tensor[..., 23:26])
        gravity = frozen._rotate_inverse(tensor[..., 3:7], world_gravity).numpy()
    values = np.concatenate((((velocity[..., 0] - command[..., 0]) / .2)[..., None],
                             ((state[..., 28] - command[..., 2]) / .2)[..., None],
                             (velocity[..., 1] / .2)[..., None], gravity[..., :2],
                             (state[..., 2] - height)[..., None], action - previous), axis=-1)
    values = _f64(values, 'computed coordinates', 22)
    return {'coordinates': values, 'components': component_costs(values)}


def _group_sum(weighted_coordinates):
    import numpy as np

    return np.stack([weighted_coordinates[..., indices].sum(axis=-1)
                     for indices in GROUPS.values()], axis=-1)


def _identity(left, right):
    import numpy as np

    gap = np.abs(left - right)
    limit = ATOL + RTOL * np.maximum(np.abs(left), np.abs(right))
    return {'passed': bool((gap <= limit).all()), 'max_abs_gap': float(gap.max()),
            'max_tolerance_ratio': float((gap / limit).max()), 'atol': ATOL, 'rtol': RTOL}


def decompose(model_zero, model_plus, model_minus, true_zero, true_plus, true_minus,
              *, eps_model=.001, eps_true):
    """Symmetric six-component error decomposition of six residual arrays.

For either endpoint pair, m=(r+ + r-)/2 and d=(r+ - r-)/(2*epsilon).
The coordinate cost FD is 2*w*m*d. Its model-minus-truth difference is
2*w*((mM-mT)*mean(dM,dT) + (dM-dT)*mean(mM,mT)). The midpoint term is also
split into zero-point residual mismatch and the difference of each endpoint
midpoint relative to its zero point. The zero-point term is not pure world-model
bias: the saved paths may also differ in nominal actions and coordinate transforms.

Different epsilons are allowed: the identity then compares different finite
intervals. The midpoint-offset term may include F32 action quantization,
clamping and coordinate transforms; it is not an estimate of curvature.
All returned FD values use each pair's own epsilon; nothing is rescaled by an
assumed ideal action direction. Identity-check failure is returned for archival,
not hidden by a model-gradient acceptance threshold or a numerical retry.
    """
    import numpy as np

    for name, value in (('eps_model', eps_model), ('eps_true', eps_true)):
        _require(not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
                 and np.asarray(value).dtype.kind in 'fiu' and math.isfinite(float(value))
                 and float(value) > 0, name + ' must be finite and positive')
    eps_model, eps_true = float(eps_model), float(eps_true)
    labels = ('model_zero', 'model_plus', 'model_minus', 'true_zero', 'true_plus', 'true_minus')
    arrays = [_f64(value, name, 22) for name, value in zip(labels,
        (model_zero, model_plus, model_minus, true_zero, true_plus, true_minus), strict=True)]
    _require(all(value.shape == arrays[0].shape for value in arrays), 'all six residual shapes must match')
    mz, mp, mm, tz, tp, tm = arrays
    weight = np.asarray(WEIGHTS, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        model_midpoint, true_midpoint = (mp + mm) / 2, (tp + tm) / 2
        model_response, true_response = (mp - mm) / (2 * eps_model), (tp - tm) / (2 * eps_true)
        average_response = (model_response + true_response) / 2
        average_midpoint = (model_midpoint + true_midpoint) / 2
        midpoint_error, response_error = model_midpoint - true_midpoint, model_response - true_response
        zero_bias = mz - tz
        model_offset, true_offset = model_midpoint - mz, true_midpoint - tz
        midpoint_offset_difference = model_offset - true_offset
        scalar = {
            'model_midpoint': model_midpoint, 'true_midpoint': true_midpoint,
            'model_response': model_response, 'true_response': true_response,
            'average_response': average_response, 'average_midpoint': average_midpoint,
            'midpoint_error': midpoint_error, 'response_error': response_error,
            'zero_bias': zero_bias, 'model_midpoint_offset': model_offset,
            'true_midpoint_offset': true_offset, 'midpoint_offset_difference': midpoint_offset_difference,
            'model_fd': weight * (mp * mp - mm * mm) / (2 * eps_model),
            'true_fd': weight * (tp * tp - tm * tm) / (2 * eps_true),
            'midpoint_contribution': 2 * weight * midpoint_error * average_response,
            'response_contribution': 2 * weight * response_error * average_midpoint,
            'zero_bias_contribution': 2 * weight * zero_bias * average_response,
            'midpoint_offset_contribution': 2 * weight * midpoint_offset_difference * average_response,
        }
    for name, value in scalar.items():
        _f64(value, 'scalar22.' + name, 22)
    model_fd = (component_costs(mp) - component_costs(mm)) / (2 * eps_model)
    true_fd = (component_costs(tp) - component_costs(tm)) / (2 * eps_true)
    result = {'model_fd': model_fd, 'true_fd': true_fd, 'difference_fd': model_fd - true_fd}
    for name in ('midpoint_contribution', 'response_contribution', 'zero_bias_contribution', 'midpoint_offset_contribution'):
        result[name] = _group_sum(scalar[name])
    result['identity_residual'] = result['difference_fd'] - (result['midpoint_contribution'] + result['response_contribution'])
    result['midpoint_identity_residual'] = result['midpoint_contribution'] - (result['zero_bias_contribution'] + result['midpoint_offset_contribution'])
    # Distinguish original within-group endpoint arithmetic from scalar weighting.
    result['model_scalar_grouping_roundoff'] = model_fd - _group_sum(scalar['model_fd'])
    result['true_scalar_grouping_roundoff'] = true_fd - _group_sum(scalar['true_fd'])
    scalar['identity_residual'] = (scalar['model_fd'] - scalar['true_fd']) - (
        scalar['midpoint_contribution'] + scalar['response_contribution'])
    scalar['midpoint_identity_residual'] = scalar['midpoint_contribution'] - (
        scalar['zero_bias_contribution'] + scalar['midpoint_offset_contribution'])
    for name, value in result.items():
        _f64(value, name, 6)
    for name in ('identity_residual', 'midpoint_identity_residual'):
        _f64(scalar[name], 'scalar22.' + name, 22)
    result['identity_check'] = _identity(result['difference_fd'],
        result['midpoint_contribution'] + result['response_contribution'])
    result['midpoint_identity_check'] = _identity(result['midpoint_contribution'],
        result['zero_bias_contribution'] + result['midpoint_offset_contribution'])
    result['scalar_identity_check'] = _identity(scalar['model_fd'] - scalar['true_fd'],
        scalar['midpoint_contribution'] + scalar['response_contribution'])
    result['scalar22'] = scalar
    result['eps_model'], result['eps_true'] = eps_model, eps_true
    return result
