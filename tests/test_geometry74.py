"""Small standard-library74 geometry and strict qualification regressions."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools74 import geometry74 as geometry


def state(quaternion=(1., 0., 0., 0.), velocity=(1., 2., 3.)):
    result = [0.] * 45
    result[2], result[3:7], result[23:26] = .5, list(quaternion), list(velocity)
    return result


def rows(central, forward=None, backward=None):
    forward = central if forward is None else forward
    backward = central if backward is None else backward
    return [{'epsilon': epsilon, 'central': c, 'forward': f, 'backward': b}
            for epsilon, c, f, b in zip(geometry.EPSILONS, central, forward, backward, strict=True)]


def test_inverse_yaw_rotation_normalizes_nonunit_wxyz_and_preserves_raw_state():
    original = state((2., 0., 0., 2.))  # +90 degrees body-to-world yaw.
    before = original.copy()
    result = geometry.endpoint_outputs(original)
    assert result == pytest.approx({'body_vx': 2., 'body_vy': -1., 'gravity_x': 0., 'gravity_y': 0., 'height': .5})
    assert original == before
    opposite = original.copy()
    opposite[3:7] = [-value for value in original[3:7]]
    assert geometry.endpoint_outputs(opposite) == pytest.approx(result)
    audit = geometry.quaternion_audit(original, opposite)
    assert audit['normalized_dot'] == pytest.approx(-1.) and not audit['passed']
    assert not audit['sign_repaired'] and opposite[3] == -2.


def test_roll_gravity_and_endpoint_transform_precede_finite_difference():
    rolled = state((math.sqrt(.5), math.sqrt(.5), 0., 0.))
    assert geometry.endpoint_outputs(rolled)['gravity_y'] == pytest.approx(-1.)
    plus, minus = state((1., 0., 0., .1)), state((1., 0., 0., -.1))
    delta = (geometry.endpoint_outputs(plus)['body_vy'] - geometry.endpoint_outputs(minus)['body_vy']) / .2
    assert delta == pytest.approx(-2 / 1.01)
    assert geometry.quaternion_audit(plus, minus)['passed']


@pytest.mark.parametrize('quat', [(0., 0., 0., 0.), (0., 1., 0., 0.)])
def test_zero_or_orthogonal_quaternion_is_not_admitted(quat):
    other = state(quat)
    audit = geometry.quaternion_audit(state(), other)
    assert not audit['passed'] and 'NORMALIZED_DOT_NOT_POSITIVE' in audit['failure_reasons']
    if not any(quat):
        assert audit['normalized_dot'] is None and 'QUATERNION_NORM_INVALID' in audit['failure_reasons']
        with pytest.raises(ValueError, match='norm'):
            geometry.endpoint_outputs(other)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_qualification_preserves_negative_labels_and_inclusive_boundaries(sign):
    floor = geometry.scalar_qualification(rows([sign * 1e-5] * 4))
    assert floor['stable'] and floor['failure_reasons'] == []
    boundary = geometry.scalar_qualification(rows([sign * value for value in (3., 4., 3., 4.)],
                                                 [sign * 3.] * 4, [sign * 4.] * 4))
    assert boundary['stable'] and boundary['central_whole_grid_relative_spread'] == .25
    assert all(row['relative_gap'] == .25 for row in boundary['one_sided'])


def test_low_zero_sign_and_spread_failures_are_not_collapsed_into_one_reason():
    result = geometry.scalar_qualification(rows([1e-6, -1e-6, 0., 2e-6], [1e-6] * 4, [-1e-6] * 4))
    assert not result['stable']
    assert set(result['failure_reasons']) == {'CENTRAL_SIGN_INCONSISTENT_OR_ZERO', 'CENTRAL_MAGNITUDE_BELOW_1E_5',
        'CENTRAL_SPREAD_ABOVE_0_25', 'ONE_SIDED_SIGN_INCONSISTENT_OR_ZERO', 'ONE_SIDED_GAP_ABOVE_0_25'}
    zero = geometry.scalar_qualification(rows([0.] * 4))
    assert not zero['stable'] and zero['central_whole_grid_relative_spread'] is None
    assert 'CENTRAL_SPREAD_UNDEFINED' in zero['failure_reasons'] and 'ONE_SIDED_GAP_UNDEFINED' in zero['failure_reasons']
    low = geometry.scalar_qualification(rows([9e-6] * 4))
    assert low['failure_reasons'] == ['CENTRAL_MAGNITUDE_BELOW_1E_5']


def test_full45_group_error_keeps_xy_and_wheel_phase_coordinates():
    predicted = [0.] * 45
    predicted[0], predicted[1], predicted[10], predicted[22], predicted[32] = 2., -3., 4., 2., 5.
    result = geometry.group_squared_errors(predicted, [0.] * 45, [1.] * 45, geometry.GROUPS)
    assert result['position'] == pytest.approx(13 / 3)
    assert result['joint_position'] == 20 / 16 and result['wheel_velocity'] == 25 / 4
    assert len(result) == 7 and result['quaternion'] == 0.
    masked = dict(geometry.GROUPS, position=(2,))
    with pytest.raises(ValueError, match='all45'):
        geometry.group_squared_errors(predicted, [0.] * 45, [1.] * 45, masked)


def test_shapes_nonfinite_values_and_wrong_epsilon_are_rejected():
    with pytest.raises(ValueError, match='length 45'):
        geometry.endpoint_outputs(state()[:-1])
    invalid = state()
    invalid[23] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        geometry.endpoint_outputs(invalid)
    wrong = rows([1.] * 4)
    wrong[0]['epsilon'] *= 10
    with pytest.raises(ValueError, match='epsilon'):
        geometry.scalar_qualification(wrong)
    with pytest.raises(ValueError, match='positive'):
        geometry.group_squared_errors([0.] * 45, [0.] * 45, [0.] * 45, geometry.GROUPS)
