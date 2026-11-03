"""Fixed endpoint substitutions for77; no model, policy or physical execution.

Quaternions are normalized wxyz. Body preservation is a real-arithmetic
identity; floating-point body coordinates are only tolerance-consistent.
The reported gaps do not create a qualification threshold.
"""
from __future__ import annotations

import math

MODES = ('control', 'active', 'world_only', 'other_only', 'body_protected')


def _state(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 45:
        raise ValueError(name + ' must contain exactly45 scalars')
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(name + ' requires finite real scalars')
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(name + ' scalar exceeds binary64') from exc
        if not math.isfinite(value):
            raise ValueError(name + ' contains a nonfinite scalar')
        result.append(value)
    _unit(result[3:7])
    return result


def _unit(quaternion):
    norm = math.hypot(*quaternion)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('quaternion norm must be finite and positive')
    return [value / norm for value in quaternion]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _rotate(quaternion, vector, *, inverse):
    q = _unit(quaternion)
    cross = _cross(q[1:], vector)
    second = _cross(q[1:], cross)
    sign = -1 if inverse else 1
    # The inverse is frozen54's normalized v - 2*w*cross + 2*crosscross.
    result = [vector[i] + sign * 2 * q[0] * cross[i] + 2 * second[i] for i in range(3)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError('rotation produced a nonfinite value')
    return result


def compose(control45, active45, mode):
    """Compose a fresh45-list in one of MODES, leaving both inputs unchanged.

    Protected velocity is v_C + R(q_C)e_z*(A25-C25)/Rzz. Require
    abs(Rzz)>=0.5, without clamping or fallback. Assign25=A25 explicitly for
    bit-exact world vz; this does not promise bit-exact body vx/vy.
    """
    control, active = _state(control45, 'control45'), _state(active45, 'active45')
    if mode not in MODES:
        raise ValueError('unknown endpoint composition mode')
    result = list(active if mode in ('active', 'other_only') else control)
    if mode == 'world_only':
        result[25] = active[25]
    elif mode == 'other_only':
        result[25] = control[25]
    elif mode == 'body_protected':
        column = _rotate(control[3:7], [0., 0., 1.], inverse=False)
        if abs(column[2]) < .5:
            raise ValueError('body_protected requires abs(Rzz)>=0.5')
        delta = active[25] - control[25]
        # Preserve exact identity, including signed zeros, when delta is zero.
        if delta != 0:
            scale = delta / column[2]
            for index in range(2):
                result[23 + index] = control[23 + index] + column[index] * scale
        result[25] = active[25]
        if not all(math.isfinite(value) for value in result):
            raise ValueError('body_protected produced a nonfinite velocity')
    return result


def preservation_receipt(control45, active45, composed45):
    """Describe preservation with no inferred pass/fail for floating body gaps.

    Nonvelocity means all coordinates except23,24,25, compared to control.
    World vz compares to active. Exact comparisons include signed-zero bits.
    Rzz is descriptive here; only compose(...,'body_protected') enforces0.5.
    """
    control, active, composed = [_state(values, name) for values, name in
                                 ((control45, 'control45'), (active45, 'active45'), (composed45, 'composed45'))]
    before = _rotate(control[3:7], control[23:26], inverse=True)
    after = _rotate(composed[3:7], composed[23:26], inverse=True)
    gaps = [abs(after[i] - before[i]) for i in range(2)]
    if not all(math.isfinite(value) for value in gaps):
        raise ValueError('body preservation gap is nonfinite')
    return {'unchanged_nonvelocity': all(control[i].hex() == composed[i].hex()
                                        for i in range(45) if i not in (23, 24, 25)),
            'world_vz_exact': active[25].hex() == composed[25].hex(),
            'body_xy_abs_gap': gaps,
            'Rzz': _rotate(control[3:7], [0., 0., 1.], inverse=False)[2]}
