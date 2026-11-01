"""Stored binary64 endpoint arithmetic and one-neighbor sensitivity only.

This is not an error bound for the model or its whole computation. It neither
selects epsilon nor changes any frozen qualification. Exact fractions describe
already rounded endpoints, not the mathematical outputs that produced them.
"""
from __future__ import annotations

import math
from fractions import Fraction

FROZEN_FLOOR = 1e-5


def endpoint_precision(plus, minus, epsilon, saved_fd):
    """Return JSON-safe diagnostics for four finite binary64 scalar inputs.

    Fractions have string numerator/denominator and a nullable nearest float.
    ``endpoint_gap_exact`` and its ULP ratio are absolute. Signed arithmetic
    residuals are stored minus exact, replay minus saved, or rounded minus exact.
    Interval flags use exact rationals; ``crosses`` excludes a mere touch.
    The interval independently moves each endpoint at most one adjacent F64
    value. It supplies no assumption about the endpoint computation's accuracy.
    """
    values = {}
    for name, value in zip(('plus', 'minus', 'epsilon', 'saved_fd'),
                           (plus, minus, epsilon, saved_fd), strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(name + ' must be a finite binary64 scalar')
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(name + ' is outside binary64') from exc
        if not math.isfinite(value):
            raise ValueError(name + ' must be finite')
        values[name] = value
    plus, minus, epsilon, saved_fd = values.values()
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    unavailable = {}

    def nullable(value, path):
        if math.isfinite(value):
            return value
        unavailable[path] = 'nonfinite derived binary64 value'
        return None

    def packed(value, path):
        try:
            approximate = nullable(float(value), path + '.float')
        except OverflowError:
            approximate = None
            unavailable[path + '.float'] = 'exact rational exceeds finite binary64 range'
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
                'float': approximate}

    def absent(path, reason):
        unavailable[path] = reason

    fp, fm, fe, fsaved = map(Fraction.from_float, (plus, minus, epsilon, saved_fd))
    denominator = 2 * fe
    difference = fp - fm
    exact = difference / denominator
    ulp_sum = Fraction.from_float(math.ulp(plus)) + Fraction.from_float(math.ulp(minus))
    subtraction = plus - minus
    replay_denominator = 2 * epsilon
    replay = subtraction / replay_denominator
    replay_finite = math.isfinite(replay)
    sub_finite = math.isfinite(subtraction)
    denominator_finite = math.isfinite(replay_denominator)
    arithmetic = {
        'subtraction': nullable(subtraction, 'python_replay.subtraction'),
        'denominator': nullable(replay_denominator, 'python_replay.denominator'),
        'fd': nullable(replay, 'python_replay.fd'),
        'bitwise_equal_saved': replay.hex() == saved_fd.hex() if replay_finite else
            absent('python_replay.bitwise_equal_saved', 'replay is nonfinite'),
        'minus_saved': packed(Fraction.from_float(replay) - fsaved, 'python_replay.minus_saved')
            if replay_finite else absent('python_replay.minus_saved', 'replay is nonfinite'),
        'subtraction_roundoff': packed(Fraction.from_float(subtraction) - difference,
                                      'python_replay.subtraction_roundoff') if sub_finite else
            absent('python_replay.subtraction_roundoff', 'subtraction is nonfinite'),
        'division_roundoff': packed(Fraction.from_float(replay) - Fraction.from_float(subtraction)
            / Fraction.from_float(replay_denominator), 'python_replay.division_roundoff')
            if replay_finite and sub_finite and denominator_finite else
            absent('python_replay.division_roundoff', 'division operands or result are nonfinite'),
    }
    # nextafter respects asymmetric spacing at binade boundaries and signed zero.
    neighbors = {'plus_down': math.nextafter(plus, -math.inf),
                 'plus_up': math.nextafter(plus, math.inf),
                 'minus_down': math.nextafter(minus, -math.inf),
                 'minus_up': math.nextafter(minus, math.inf)}
    bounds = {}
    for name, left, right in (('lower', 'plus_down', 'minus_up'),
                              ('upper', 'plus_up', 'minus_down')):
        if math.isfinite(neighbors[left]) and math.isfinite(neighbors[right]):
            bounds[name] = (Fraction.from_float(neighbors[left])
                            - Fraction.from_float(neighbors[right])) / denominator
        else:
            bounds[name] = None
    interval = {'lower': packed(bounds['lower'], 'one_neighbor_interval.lower')
                if bounds['lower'] is not None else absent('one_neighbor_interval.lower', 'neighbor is infinite'),
                'upper': packed(bounds['upper'], 'one_neighbor_interval.upper')
                if bounds['upper'] is not None else absent('one_neighbor_interval.upper', 'neighbor is infinite')}
    for name, boundary in (('zero', Fraction(0)), ('positive_floor', Fraction.from_float(FROZEN_FLOOR)),
                           ('negative_floor', -Fraction.from_float(FROZEN_FLOOR))):
        if all(value is not None for value in bounds.values()):
            lower, upper = bounds['lower'], bounds['upper']
            interval['crosses_' + name] = lower < boundary < upper
            interval['touches_or_crosses_' + name] = lower <= boundary <= upper
        else:
            for prefix in ('crosses_', 'touches_or_crosses_'):
                key = prefix + name
                interval[key] = absent('one_neighbor_interval.' + key, 'a neighbor bound is unavailable')
    interval['contains_zero'] = interval['touches_or_crosses_zero']
    return {
        'schema': 'binary64-endpoint-precision76-v1',
        'scope': 'stored-endpoint arithmetic; one-adjacent-value sensitivity, not a full-chain error bound',
        'changes_epsilon_or_qualification': False,
        'inputs': {name: {'value': value, 'hex': value.hex()} for name, value in values.items()},
        'frozen_magnitude_floor': FROZEN_FLOOR,
        'endpoint_gap_exact': packed(abs(difference), 'endpoint_gap_exact'),
        'endpoint_gap_over_combined_ulp': packed(abs(difference) / ulp_sum, 'endpoint_gap_over_combined_ulp'),
        'one_ulp_fd_scale': packed(ulp_sum / denominator, 'one_ulp_fd_scale'),
        'exact_fd': packed(exact, 'exact_fd'),
        'saved_minus_exact': packed(fsaved - exact, 'saved_minus_exact'),
        'python_replay': arithmetic,
        'one_neighbor_interval': interval,
        'derived_unavailable_reasons': unavailable,
    }
