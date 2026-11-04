"""Analytic cone geometry and source completeness; no simulator/model needed."""
from __future__ import annotations

import copy
import math

import pytest

from tools83.policy_geometry83 import (
    dot,
    project_two_halfspaces,
    row_geometry,
    validate_episode_receipt,
)


def test_already_safe_direction_is_unchanged():
    result = project_two_halfspaces([1., 2.], [1., 0.], [0., 1.])
    assert result['direction'] == [-1., -2.]
    assert result['objective'] == 0.
    assert result['retained_unscaled_yaw_descent_ratio'] == 1.


def test_one_active_constraint_preserves_unconstrained_tangent():
    result = project_two_halfspaces([-1., -2.], [1., 0.], [0., -1.])
    assert result['direction'] == [0., 2.]
    assert result['active'] == [0]
    assert result['raw_constraint_dots'] == [0., -2.]
    assert result['yaw_linear_descent'] == 4.


def test_two_active_constraints_leave_third_coordinate():
    result = project_two_halfspaces([-1., -2., -3.], [1., 0., 0.], [0., 1., 0.])
    assert result['direction'] == [0., 0., 3.]
    assert result['active'] == [0, 1]
    assert result['multipliers_unit_normals'] == [1., 2.]
    assert result['yaw_linear_descent'] == 9.
    assert result['cone_projection_identity_error'] == 0.


def test_oblique_constraints_have_joint_not_sequential_projection():
    # Intersection of x<=0 and x+y<=0; closest point to (1,2) is (-.5,.5).
    result = project_two_halfspaces([-1., -2.], [1., 0.], [1., 1.])
    assert result['direction'] == pytest.approx([-.5, .5], abs=1e-12)
    assert result['raw_constraint_dots'][0] < 0.
    assert result['objective'] == pytest.approx(2.25)


def test_two_nonorthogonal_active_constraints_satisfy_kkt():
    result = project_two_halfspaces([-2., -1., -3.], [1., 0., 0.], [1., 1., 0.])
    assert result['active'] == [0, 1]
    assert result['direction'] == pytest.approx([0., 0., 3.], abs=1e-12)
    assert result['multipliers_unit_normals'] == pytest.approx([1., math.sqrt(2.)], abs=1e-12)
    assert result['cone_projection_identity_error'] <= 1e-12


@pytest.mark.parametrize('second', ([4., 0.], [-4., 0.]))
def test_parallel_redundant_or_opposite_normals_are_not_a_singular_failure(second):
    result = project_two_halfspaces([-2., -3.], [2., 0.], second)
    assert result['direction'] == [0., 3.]
    assert all(v <= 1e-12 for v in result['raw_constraint_dots'])


def test_positive_constraint_rescaling_does_not_change_geometry():
    a = project_two_halfspaces([-1., -2., -3.], [1., 1., 0.], [0., 1., 1.])
    b = project_two_halfspaces([-1., -2., -3.], [7., 7., 0.], [0., .125, .125])
    assert a['direction'] == pytest.approx(b['direction'], abs=1e-12)


def test_zero_is_feasible_but_not_a_yaw_improvement():
    yaw, forward, stop = [-1., 0., 0., 0.], [1., 0., 0., 0.], [0.] * 4
    projected = project_two_halfspaces(yaw, forward, stop)
    details = row_geometry(yaw, forward, stop, projected, row_width=1)
    assert projected['direction'] == [0.] * 4
    assert projected['zero_is_always_feasible']
    assert not projected['strict_descent_resolved']
    assert not details['projected']['usable_local_descent']
    assert details['unconstrained_row_l1_linear_descent_upper_bound'] == .01


def test_zero_yaw_gradient_is_reported_without_inventing_a_direction():
    result = project_two_halfspaces([0.] * 4, [1.] * 4, [0.] * 4)
    details = row_geometry([0.] * 4, [1.] * 4, [0.] * 4, result, row_width=1)
    assert result['retained_unscaled_yaw_descent_ratio'] is None
    assert not details['projected']['usable_local_descent']


def test_row_l1_dual_bound_is_attainable_and_normalization_keeps_constraints():
    yaw = [1., -3., 0., 2., -4., 1., 1., 1.]
    forward = [-1., 0., 0., 0., 0., 0., 0., 0.]
    stop = [0.] * 8
    projection = project_two_halfspaces(yaw, forward, stop)
    result = row_geometry(yaw, forward, stop, projection, row_width=2)
    # One extremal coordinate per wheel row exhausts each .01 L1 budget.
    extremal_step = [0., .01, 0., -.01, .01, 0., -.01, 0.]
    assert -dot(yaw, extremal_step) == pytest.approx(.1)
    assert result['unconstrained_row_l1_linear_descent_upper_bound'] == pytest.approx(.1)
    p = result['projected']
    assert max(p['normalized_wheel_row_l1']) == pytest.approx(1.)
    assert p['post_normalization_constraints_certified']
    assert p['yaw_forward_stop_directional_derivatives'][0] < 0.
    assert p['yaw_forward_stop_directional_derivatives'][1] <= 1e-12


@pytest.mark.parametrize('bad', ([math.nan], [math.inf], [True], [], ['1']))
def test_invalid_gradient_is_rejected(bad):
    with pytest.raises(ValueError):
        project_two_halfspaces(bad, [1.], [1.])


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError, match='dimensions'):
        project_two_halfspaces([1., 2.], [1.], [1., 2.])
    with pytest.raises(ValueError, match='four wheel rows'):
        row_geometry([1.] * 3, [1.] * 3, [0.] * 3, {'direction': [-1.] * 3}, row_width=1)


def test_complete_episode_cannot_be_replaced_by_partial_or_wrong_identity():
    receipt = {'status': 'COMPLETE', 'steps': 600, 'task': 'forward_stop', 'seed': 80101,
               'actual_counts': {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000, 'resets': 1}}
    validate_episode_receipt(receipt, 'forward_stop', 80101)
    for key, value in (('status', 'INCOMPLETE'), ('steps', 599), ('seed', 81207)):
        changed = copy.deepcopy(receipt)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_episode_receipt(changed, 'forward_stop', 80101)
    changed = copy.deepcopy(receipt)
    changed['actual_counts']['pd_updates'] = 2999
    with pytest.raises(ValueError):
        validate_episode_receipt(changed, 'forward_stop', 80101)
