"""Analytic invariants and degenerate cases for scalar support diagnostics."""
from __future__ import annotations

import math

import pytest

from tools73.support73 import (
    WHEEL_AXIS,
    ZERO_FEATURES,
    action_direction,
    action_distance,
    masked_state_distance,
    spearman,
    summarize_distance_association,
)

GROUPS = {'position': (0, 1, 2), 'quaternion': (3, 4, 5, 6),
          'joint_position': tuple(range(7, 23)), 'base_linear_velocity': (23, 24, 25),
          'base_angular_velocity': (26, 27, 28),
          'leg_velocity': (29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43),
          'wheel_velocity': (32, 36, 40, 44)}


def _state(a, b, scales=None, groups=None, mask=ZERO_FEATURES):
    return masked_state_distance(a, b, [1.] * 45 if scales is None else scales,
                                 GROUPS if groups is None else groups, mask)


def test_masked_coordinates_vanish_without_removing_original_group_denominators():
    a, b = [0.] * 45, [0.] * 45
    for i in ZERO_FEATURES:
        a[i] = 1000.
    assert _state(a, b)['distance'] == 0.
    a[2] = 1.
    result = _state(a, b)
    assert result['group_mse']['position'] == pytest.approx(1 / 3)
    assert result['distance'] == pytest.approx(math.sqrt(1 / 21))
    assert a[0] == 1000.  # No in-place masking of the input.


def test_quaternion_sign_is_recorded_not_silently_aligned():
    a, b = [0.] * 45, [0.] * 45
    a[3], b[3] = 1., -1.
    result = _state(a, b)
    assert result['group_mse']['quaternion'] == 1.
    assert result['distance'] == pytest.approx(math.sqrt(1 / 7))


def test_groups_are_equally_weighted_and_physical_rescaling_is_invariant():
    zero, position, wheel = [0.] * 45, [0.] * 45, [0.] * 45
    position[2], wheel[32] = math.sqrt(3), 2.
    assert _state(position, zero)['distance'] == pytest.approx(_state(wheel, zero)['distance'])
    assert _state(position, zero)['distance'] == pytest.approx(_state(zero, position)['distance'])
    scaled = [2 * x for x in position]
    assert _state(scaled, zero, scales=[2.] * 45)['distance'] == pytest.approx(_state(position, zero)['distance'])


@pytest.mark.parametrize('problem', ('scale_zero', 'scale_nan', 'short_state', 'duplicate_group', 'missing_mask'))
def test_invalid_state_geometry_is_rejected(problem):
    a, b, scales, groups, mask = [0.] * 45, [0.] * 45, [1.] * 45, dict(GROUPS), ZERO_FEATURES
    if problem == 'scale_zero':
        scales[0] = 0.
    elif problem == 'scale_nan':
        scales[3] = math.nan
    elif problem == 'short_state':
        a.pop()
    elif problem == 'duplicate_group':
        groups['position'] = (0, 0, 2)
    else:
        mask = ZERO_FEATURES[:-1]
    with pytest.raises(ValueError):
        _state(a, b, scales, groups, mask)


def test_action_distance_retains_all16_coordinates_and_raw_scale():
    a, b = [0.] * 16, [0.] * 16
    a[0] = 1.
    assert action_distance(a, b) == .25
    assert action_distance(a, b) == action_distance(b, a)
    assert action_distance([2.] * 16, b) == 2.


def test_projection_decomposes_parallel_and_orthogonal_response():
    v = [2 * value for value in WHEEL_AXIS]
    v[0] = 3.
    result = action_direction([.25 * x for x in v], [-.25 * x for x in v], .25, WHEEL_AXIS)
    assert result['v'] == v
    assert result['norm'] == 5.
    assert result['signed_projection_gain'] == 2.
    assert result['orthogonal_norm'] == 3.
    assert result['relative_orthogonal_norm'] == .6
    assert result['cosine_to_axis'] == .8
    assert math.hypot(*result['normalized_direction']) == pytest.approx(1.)
    reverse = action_direction([-.25 * x for x in v], [.25 * x for x in v], .25, WHEEL_AXIS)
    assert reverse['signed_projection_gain'] == -2.
    assert reverse['cosine_to_axis'] == -.8


def test_actual_endpoint_quantization_is_used_instead_of_an_ideal_axis():
    plus, minus = [0.] * 16, [0.] * 16
    plus[3], minus[3] = .5001, .4999
    result = action_direction(plus, minus, .001, WHEEL_AXIS)
    assert result['v'][3] == pytest.approx(.1)
    assert result['signed_projection_gain'] == pytest.approx(.025)
    assert result['orthogonal_norm'] == pytest.approx(math.sqrt(.0075))
    assert result['v'][7] == 0.


def test_zero_action_direction_is_explicitly_undefined():
    result = action_direction([.5] * 16, [.5] * 16, .001, WHEEL_AXIS)
    assert result['norm'] == result['signed_projection_gain'] == result['orthogonal_norm'] == 0.
    assert result['normalized_direction'] is None
    assert result['cosine_to_axis'] is None
    assert result['relative_orthogonal_norm'] is None


@pytest.mark.parametrize('epsilon', (0., -1., math.inf, math.nan))
def test_action_direction_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        action_direction([0.] * 16, [0.] * 16, epsilon, WHEEL_AXIS)


def test_invalid_action_shape_axis_and_nonfinite_values_are_rejected():
    with pytest.raises(ValueError):
        action_distance([0.] * 15, [0.] * 16)
    with pytest.raises(ValueError):
        action_direction([0.] * 16, [0.] * 16, .1, [0.] * 16)
    with pytest.raises(ValueError):
        action_distance([math.inf] * 16, [0.] * 16)


def test_spearman_known_tied_ranks_and_monotone_transform():
    assert spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(math.sqrt(3) / 2)
    assert spearman([1, 2, 3, 4], [9, 8, 7, 6]) == pytest.approx(-1.)
    assert spearman([1, 2, 3, 4], [1, 8, 27, 64]) == pytest.approx(1.)


def test_spearman_zero_variance_is_none_not_a_zero_correlation():
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1, 2, 3], [7, 7, 7]) is None
    assert spearman([3], [4]) is None
    with pytest.raises(ValueError):
        spearman([], [])
    with pytest.raises(ValueError):
        spearman([1], [1, 2])
    with pytest.raises(ValueError):
        spearman([math.nan], [1])


def _rows():
    return [{'role': 'registered', 'seed': i, 'tick': 110, 'support': float(i), 'change64001': 2. * i}
            for i in range(36)]


def test_fixed_all36_association_and_largest9_are_descriptive_only():
    result = summarize_distance_association(_rows(), 'support', 'change64001')
    assert result['count'] == 36
    assert [p['seed'] for p in result['largest9']['points']] == list(range(35, 26, -1))
    assert result['largest9']['count'] == 9 and result['other27']['count'] == 27
    assert result['spearman_rho'] == pytest.approx(1.)
    assert result['largest9']['mean_error_change'] == pytest.approx(62.)
    assert result['other27']['mean_error_change'] == pytest.approx(26.)
    assert result['mean_error_change_difference_largest9_minus_other27'] == pytest.approx(36.)
    assert result['descriptive_only'] and not result['changes_candidate_eligibility']
    assert not result['threshold_selected_from_results']


def test_tied_boundary_has_identity_tiebreak_independent_of_input_order():
    rows = _rows()
    for row in rows:
        row['support'] = 1.
    first = summarize_distance_association(rows, 'support', 'change64001')
    second = summarize_distance_association(list(reversed(rows)), 'support', 'change64001')
    assert first == second
    assert [p['seed'] for p in first['largest9']['points']] == list(range(9))
    assert first['boundary_tied'] and first['all_distances_equal']
    assert first['spearman_rho'] is None


@pytest.mark.parametrize('problem', ('partial', 'duplicate', 'nonfinite', 'negative_distance'))
def test_association_refuses_missing_duplicate_or_invalid_rows(problem):
    rows = _rows()
    if problem == 'partial':
        rows.pop()
    elif problem == 'duplicate':
        rows[-1] = dict(rows[0])
    elif problem == 'nonfinite':
        rows[0]['change64001'] = math.nan
    else:
        rows[0]['support'] = -1.
    with pytest.raises(ValueError):
        summarize_distance_association(rows, 'support', 'change64001')
