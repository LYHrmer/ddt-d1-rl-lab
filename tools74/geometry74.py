"""Pure standard-library74 endpoint geometry and unfiltered response diagnostics.

Python floats retain the saved F64 endpoints. Geometry follows54's normalized
wxyz inverse rotation, not an Euler-angle approximation; raw quaternion signs
are never flipped. These functions neither execute models nor authorize fits.
"""
from __future__ import annotations

import math

EPSILONS = (.000003125, .00000625, .0000125, .000025)
GROUPS = {
    'position': tuple(range(3)), 'quaternion': tuple(range(3, 7)),
    'joint_position': tuple(range(7, 23)), 'base_linear_velocity': tuple(range(23, 26)),
    'base_angular_velocity': tuple(range(26, 29)),
    'leg_velocity': (29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43),
    'wheel_velocity': (32, 36, 40, 44),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f'{name} must be a real number')
    result = float(value)
    _require(math.isfinite(result), f'{name} must be finite')
    return result


def _vector(values, width, name):
    _require(isinstance(values, (list, tuple)) and len(values) == width, f'{name} must have length {width}')
    return [_number(value, name) for value in values]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _rotate_inverse(quaternion, vector):
    norm = math.hypot(*quaternion)
    _require(math.isfinite(norm) and norm > 0., 'quaternion norm must be finite and positive')
    q = [value / norm for value in quaternion]
    cross = _cross(q[1:], vector)
    second = _cross(q[1:], cross)
    return [vector[index] - 2 * q[0] * cross[index] + 2 * second[index] for index in range(3)]


def endpoint_outputs(state45):
    """F64 state45 -> body vx/vy, projected-gravity x/y and world height.

    Apply this to each saved endpoint BEFORE forming finite differences.
    Formula is54's v - 2*w*cross(q_xyz,v) + 2*cross(q_xyz,cross(q_xyz,v)).
    """
    state = _vector(state45, 45, 'state45')
    body = _rotate_inverse(state[3:7], state[23:26])
    gravity = _rotate_inverse(state[3:7], [0., 0., -1.])
    result = dict(zip(('body_vx', 'body_vy', 'gravity_x', 'gravity_y', 'height'),
                      (*body[:2], *gravity[:2], state[2]), strict=True))
    _require(all(math.isfinite(value) for value in result.values()), 'nonfinite transformed endpoint')
    return result


def quaternion_audit(plus45, minus45):
    """Report endpoint norms and normalized dot; require dot>0 without sign repair.

    Zero norms or opposing/orthogonal representations are explicit failures.
    The original state vectors are never changed, including raw q coordinates.
    """
    plus, minus = _vector(plus45, 45, 'plus45')[3:7], _vector(minus45, 45, 'minus45')[3:7]
    pnorm, mnorm = math.hypot(*plus), math.hypot(*minus)
    _require(math.isfinite(pnorm) and math.isfinite(mnorm), 'nonfinite quaternion norm')
    valid = pnorm > 0 and mnorm > 0
    dot = math.fsum((p / pnorm) * (m / mnorm) for p, m in zip(plus, minus, strict=True)) if valid else None
    reasons = []
    if not valid:
        reasons.append('QUATERNION_NORM_INVALID')
    if dot is None or dot <= 0:
        reasons.append('NORMALIZED_DOT_NOT_POSITIVE')
    return {'plus_norm': pnorm, 'minus_norm': mnorm, 'normalized_dot': dot,
            'norms_valid': valid, 'same_hemisphere': dot is not None and dot > 0,
            'passed': not reasons, 'failure_reasons': reasons, 'sign_repaired': False}


def scalar_qualification(rows):
    """Exact58 scalar gate for all four absolute-action eps; independent reasons.

    Each row supplies epsilon/central/forward/backward. One-sided sign and gap
    failures are retained independently of central sign/magnitude/spread.
    """
    _require(isinstance(rows, (list, tuple)) and len(rows) == 4, 'all four training epsilon rows required')
    _require(all(isinstance(row, dict) and {'epsilon', 'central', 'forward', 'backward'} <= row.keys()
                 for row in rows), 'missing scalar response field')
    values = [{key: _number(row[key], key) for key in ('epsilon', 'central', 'forward', 'backward')} for row in rows]
    _require([row['epsilon'] for row in values] == list(EPSILONS), 'incomplete action-epsilon grid')
    central = [row['central'] for row in values]
    scale = max(abs(value) for value in central)
    spread = (max(central) - min(central)) / scale if scale else None
    _require(spread is None or math.isfinite(spread), 'nonfinite central relative spread')
    same_sign = all(value > 0 for value in central) or all(value < 0 for value in central)
    floor = all(abs(value) >= 1e-5 for value in central)
    reasons, sides = [], []
    if not same_sign:
        reasons.append('CENTRAL_SIGN_INCONSISTENT_OR_ZERO')
    if not floor:
        reasons.append('CENTRAL_MAGNITUDE_BELOW_1E_5')
    if spread is None:
        reasons.append('CENTRAL_SPREAD_UNDEFINED')
    elif spread > .25:
        reasons.append('CENTRAL_SPREAD_ABOVE_0_25')
    for row in values:
        forward, backward = row['forward'], row['backward']
        magnitude = max(abs(forward), abs(backward))
        gap = abs(forward - backward) / magnitude if magnitude else None
        _require(gap is None or math.isfinite(gap), 'nonfinite one-sided relative gap')
        sign = forward * backward > 0
        failures = []
        if not sign:
            failures.append('ONE_SIDED_SIGN_INCONSISTENT_OR_ZERO')
        if gap is None:
            failures.append('ONE_SIDED_GAP_UNDEFINED')
        elif gap > .25:
            failures.append('ONE_SIDED_GAP_ABOVE_0_25')
        reasons.extend(reason for reason in failures if reason not in reasons)
        sides.append({'epsilon': row['epsilon'], 'same_sign': sign, 'relative_gap': gap,
                      'stable': sign and gap is not None and gap <= .25, 'failure_reasons': failures})
    return {'central_same_sign': same_sign, 'all_central_abs_ge_1e_5': floor,
            'central_whole_grid_relative_spread': spread, 'one_sided': sides,
            'stable': same_sign and floor and spread is not None and spread <= .25
            and all(row['stable'] for row in sides), 'failure_reasons': reasons, 'selected_epsilon': None,
            'coordinate': 'absolute action delta along fixed wheel direction; no yaw_error multiplier'}


def group_squared_errors(pred45, true45, scales45, groups):
    """Seven frozen normalized MSEs of full45 responses; never a feature mask."""
    predicted, actual = _vector(pred45, 45, 'pred45'), _vector(true45, 45, 'true45')
    scales = _vector(scales45, 45, 'scales45')
    _require(all(value > 0 for value in scales), 'all45 scales must be positive')
    _require(isinstance(groups, dict) and set(groups) == set(GROUPS)
             and all(tuple(groups[name]) == indices for name, indices in GROUPS.items()),
             'exact original seven groups covering all45 coordinates required')
    error = [(p - t) / scale for p, t, scale in zip(predicted, actual, scales, strict=True)]
    squared = [value * value for value in error]
    _require(all(math.isfinite(value) for value in squared), 'nonfinite normalized squared error')
    return {name: math.fsum(squared[index] for index in indices) / len(indices)
            for name, indices in GROUPS.items()}
