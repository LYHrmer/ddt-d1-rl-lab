"""Pure scalar support diagnostics; no fitting, decisions or threshold search.

Distances retain the frozen seven-group denominators after feature masking.
Quaternion coordinates are compared as recorded, without sign alignment or
normalization. Association summaries always retain all 36 registered rows.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

ZERO_FEATURES = (0, 1, 10, 14, 18, 22)
WHEEL_AXIS = tuple(1.0 if i in (3, 11) else -1.0 if i in (7, 15) else 0.0 for i in range(16))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool), name + ' must be a real scalar')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + ' must be finite') from exc
    _require(math.isfinite(result), name + ' must be finite')
    return result


def _vector(values, length, name):
    _require(not isinstance(values, (str, bytes, Mapping)), name + ' must be a vector')
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(name + ' must be a vector') from exc
    _require(len(items) == length, name + ' has wrong length')
    return [_finite(value, name) for value in items]


def _sum(values, name):
    try:
        return _finite(math.fsum(values), name)
    except OverflowError as exc:
        raise ValueError(name + ' overflow') from exc


def _norm(values, name):
    return _finite(math.hypot(*values), name)


def masked_state_distance(a45, b45, scales45, groups7, zero_indices):
    """Return sqrt(mean of seven scaled group MSEs), with frozen masks.

    ``groups7`` maps seven names to disjoint index sequences covering all45
    coordinates. Masked coordinates contribute zero and remain in their group
    denominators; no quaternion sign or phase wrapping is applied.
    """
    a, b = _vector(a45, 45, 'state a'), _vector(b45, 45, 'state b')
    scales = _vector(scales45, 45, 'physical scales')
    _require(all(value > 0 for value in scales), 'physical scales must be positive')
    _require(isinstance(groups7, Mapping) and len(groups7) == 7, 'seven state groups required')
    groups = {}
    for name, indices in groups7.items():
        _require(isinstance(name, str) and bool(name), 'group names must be nonempty strings')
        try:
            indices = tuple(indices)
        except TypeError as exc:
            raise ValueError('group indices must be sequences') from exc
        _require(indices and all(type(i) is int and 0 <= i < 45 for i in indices), 'invalid state group indices')
        groups[name] = indices
    all_indices = [i for indices in groups.values() for i in indices]
    _require(sorted(all_indices) == list(range(45)), 'seven groups must partition all45 coordinates')
    try:
        zeros = tuple(zero_indices)
    except TypeError as exc:
        raise ValueError('mask indices must be a sequence') from exc
    _require(len(zeros) == 6 and all(type(i) is int for i in zeros)
             and set(zeros) == set(ZERO_FEATURES), 'the six frozen ZERO_FEATURES are required')
    delta = [0.0 if i in zeros else _finite((a[i] - b[i]) / scales[i], 'scaled state difference') for i in range(45)]
    group_mse = {name: _sum((delta[i] * delta[i] for i in indices), 'group squared error') / len(indices)
                 for name, indices in groups.items()}
    mean = _sum(group_mse.values(), 'seven-group error') / 7
    return {'distance': math.sqrt(mean), 'group_mse': group_mse}


def action_distance(a16, b16):
    """Root mean squared raw action difference over all16 coordinates."""
    a, b = _vector(a16, 16, 'action a'), _vector(b16, 16, 'action b')
    differences = [_finite(x - y, 'action difference') for x, y in zip(a, b, strict=True)]
    return _norm(differences, 'action distance norm') / 4


def action_direction(plus16, minus16, epsilon, axis16):
    """Analyze the recorded endpoint direction along the fixed wheel axis.

    A zero response has zero norm, projection and orthogonal norm; normalized
    direction, relative orthogonal norm and cosine are undefined (None).
    """
    plus, minus = _vector(plus16, 16, 'plus action'), _vector(minus16, 16, 'minus action')
    axis = _vector(axis16, 16, 'wheel axis')
    _require(tuple(axis) == WHEEL_AXIS, 'fixed wheel + - + - axis required')
    epsilon = _finite(epsilon, 'epsilon')
    _require(epsilon > 0, 'epsilon must be positive')
    denominator = _finite(2 * epsilon, 'epsilon denominator')
    v = [_finite((p - m) / denominator, 'action direction') for p, m in zip(plus, minus, strict=True)]
    norm = _norm(v, 'action direction norm')
    axis_square = _sum((x * x for x in axis), 'axis squared norm')
    dot = _sum((x * y for x, y in zip(v, axis, strict=True)), 'axis projection')
    gain = dot / axis_square
    orthogonal = [_finite(x - gain * y, 'orthogonal direction') for x, y in zip(v, axis, strict=True)]
    orthogonal_norm = _norm(orthogonal, 'orthogonal norm')
    cosine = dot / norm / math.sqrt(axis_square) if norm else None
    return {'v': v, 'norm': norm, 'signed_projection_gain': gain, 'orthogonal_norm': orthogonal_norm,
            'relative_orthogonal_norm': orthogonal_norm / norm if norm else None,
            'cosine_to_axis': max(-1.0, min(1.0, cosine)) if cosine is not None else None,
            'normalized_direction': [x / norm for x in v] if norm else None}


def _average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(x, y):
    """Spearman correlation with average ties; zero rank variance gives None."""
    try:
        x, y = list(x), list(y)
    except TypeError as exc:
        raise ValueError('rank inputs must be vectors') from exc
    _require(len(x) == len(y) and bool(x), 'rank inputs require equal nonzero lengths')
    a, b = _average_ranks(_vector(x, len(x), 'rank x')), _average_ranks(_vector(y, len(y), 'rank y'))
    center = (len(a) + 1) / 2
    a, b = [v - center for v in a], [v - center for v in b]
    norm_a, norm_b = _norm(a, 'rank x norm'), _norm(b, 'rank y norm')
    if norm_a == 0 or norm_b == 0:
        return None
    value = _sum((u / norm_a * v / norm_b for u, v in zip(a, b, strict=True)), 'rank correlation')
    return max(-1.0, min(1.0, value))


def summarize_distance_association(rows, distance_key, error_change_key):
    """Describe all36 rows for one model seed's scalar error-change column.

    Rows have unique ``(role, seed, tick)`` identities. The largest9 distances
    use that identity in ascending order for ties; the other27 remain in the
    comparison. This fixed split is descriptive, never an eligibility gate.
    """
    _require(isinstance(distance_key, str) and isinstance(error_change_key, str), 'column names must be strings')
    rows = list(rows)
    _require(len(rows) == 36, 'all36 registered rows required')
    points = []
    for row in rows:
        _require(isinstance(row, Mapping), 'each point must be a mapping')
        _require(isinstance(row['role'], str) and bool(row['role']) and type(row['seed']) is int
                 and type(row['tick']) is int and row['seed'] >= 0 and row['tick'] >= 0, 'invalid point identity')
        distance = _finite(row[distance_key], 'support distance')
        _require(distance >= 0, 'support distance cannot be negative')
        points.append({'role': row['role'], 'seed': row['seed'], 'tick': row['tick'], 'distance': distance,
                       'error_change': _finite(row[error_change_key], 'error change')})
    identities = [(p['role'], p['seed'], p['tick']) for p in points]
    _require(len(set(identities)) == 36, '36 unique point identities required')
    ordered = sorted(points, key=lambda p: (-p['distance'], p['role'], p['seed'], p['tick']))

    def summarize(part):
        return {'count': len(part), 'points': part,
                'mean_distance': _sum((p['distance'] / len(part) for p in part), 'mean distance'),
                'mean_error_change': _sum((p['error_change'] / len(part) for p in part), 'mean error change')}

    high, other = summarize(ordered[:9]), summarize(ordered[9:])
    difference = _finite(high['mean_error_change'] - other['mean_error_change'], 'between-group error-change difference')
    return {'count': 36, 'distance_key': distance_key, 'error_change_key': error_change_key,
            'spearman_rho': spearman([p['distance'] for p in ordered], [p['error_change'] for p in ordered]),
            'largest9': high, 'other27': other,
            'mean_error_change_difference_largest9_minus_other27': difference,
            'boundary_tied': ordered[8]['distance'] == ordered[9]['distance'],
            'all_distances_equal': ordered[0]['distance'] == ordered[-1]['distance'],
            'split_rule': 'descending distance; ascending (role, seed, tick) for exact ties; fixed9 versus27',
            'descriptive_only': True, 'changes_candidate_eligibility': False, 'threshold_selected_from_results': False}
