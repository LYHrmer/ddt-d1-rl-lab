"""Analytic endpoint identities and frozen-cost alignment; no rollout or fit."""
from __future__ import annotations

import pytest

np = pytest.importorskip('numpy')

from tools70.decompose70 import component_costs, decompose, residuals


def _one(value, coordinate=0):
    result = np.zeros(22, dtype=np.float64)
    result[coordinate] = value
    return result


def _linear(model_base, model_response, true_base, true_response, em=.001, et=.00025):
    mz, md, tz, td = map(_one, (model_base, model_response, true_base, true_response))
    return decompose(mz, mz + em * md, mz - em * md,
                     tz, tz + et * td, tz - et * td, eps_model=em, eps_true=et)


@pytest.mark.parametrize(('bases', 'expected'), [((2., 3., 1., 3.), (6., 0.)),
                                               ((2., 3., 2., 1.), (0., 8.)),
                                               ((2., 1., 1., 2.), (3., -3.))])
def test_baseline_response_and_cancelling_errors_have_independent_known_terms(bases, expected):
    result = _linear(*bases)
    assert result['midpoint_contribution'][0] == pytest.approx(expected[0], abs=1e-10)
    assert result['response_contribution'][0] == pytest.approx(expected[1], abs=1e-10)
    assert result['difference_fd'][0] == pytest.approx(sum(expected), abs=1e-10)
    assert result['zero_bias_contribution'][0] == pytest.approx(expected[0], abs=1e-10)
    np.testing.assert_allclose(result['midpoint_offset_contribution'], 0., atol=1e-10)
    assert result['identity_check']['passed'] and result['midpoint_identity_check']['passed']


def test_different_intervals_of_the_same_curved_function_are_not_zero_point_bias():
    em, et = .01, .02
    def point(theta):
        return _one(1 + 2 * theta + 3 * theta * theta)
    result = decompose(point(0.), point(em), point(-em), point(0.), point(et), point(-et),
                       eps_model=em, eps_true=et)
    expected = 12 * (em * em - et * et)
    assert result['difference_fd'][0] == pytest.approx(expected, abs=1e-11)
    assert result['midpoint_offset_contribution'][0] == pytest.approx(expected, abs=1e-11)
    np.testing.assert_allclose(result['zero_bias_contribution'], 0., atol=1e-12)
    np.testing.assert_allclose(result['response_contribution'], 0., atol=1e-11)


def test_linear_response_can_have_endpoint_midpoint_offset_without_curvature():
    # Both paths use r(action)=1+2*action. The true perturbed endpoints share an
    # action offset, while the saved zero still uses action=0. This is not a
    # curved transfer function and must not be labelled a curvature estimate.
    em, et, offset = .001, .00025, .03
    def point(action):
        return _one(1 + 2 * action)
    result = decompose(point(0.), point(em), point(-em), point(0.),
                       point(offset + et), point(offset - et), eps_model=em, eps_true=et)
    assert result['difference_fd'][0] == pytest.approx(-8 * offset, abs=1e-11)
    assert result['midpoint_offset_contribution'][0] == pytest.approx(-8 * offset, abs=1e-11)
    assert result['scalar22']['true_midpoint_offset'][0] == pytest.approx(2 * offset)
    np.testing.assert_allclose(result['zero_bias_contribution'], 0., atol=1e-12)
    np.testing.assert_allclose(result['response_contribution'], 0., atol=1e-11)
    assert result['identity_check']['passed'] and result['midpoint_identity_check']['passed']


def test_smooth_is_a_sixteen_coordinate_mean_and_leading_batches_are_retained():
    center = np.zeros((2, 22), dtype=np.float64)
    center[:, 6:] = 2.
    direction = np.zeros_like(center)
    direction[:, 6:] = np.arange(1., 17.)
    epsilon = .001
    zero = np.zeros_like(center)
    result = decompose(center, center + epsilon * direction, center - epsilon * direction,
                       zero, zero, zero, eps_true=.00025)
    assert result['model_fd'].shape == (2, 6)
    np.testing.assert_allclose(result['model_fd'][:, 5], 3.4, rtol=0., atol=1e-11)
    np.testing.assert_allclose(component_costs(center)[:, 5], .4, rtol=0., atol=1e-14)
    np.testing.assert_array_equal(result['model_fd'][:, :5], 0.)
    assert result['scalar22']['model_response'].shape == (2, 22)


def test_residual_coordinates_normalize_quaternion_and_align_with_frozen_cost():
    torch = pytest.importorskip('torch')
    from tools54 import numerics54 as frozen

    state = np.zeros((2, 45), dtype=np.float64)
    state[:, 2] = (.42, .51)
    state[:, 3:7] = ((2., 0., 0., 0.), (2., 1., -.5, .25))
    state[:, 23:26] = ((.3, -.2, .1), (.4, .1, -.2))
    state[:, 28] = (.1, -.15)
    command = np.asarray(((.2, 0., -.1), (.3, 0., .2)), dtype=np.float64)
    action = np.linspace(-.3, .4, 32, dtype=np.float64).reshape(2, 16)
    previous = action / 2
    result = residuals(state, command, action, previous, .45)
    assert result['coordinates'].shape == (2, 22) and result['coordinates'].dtype == np.float64
    assert result['coordinates'][0, 0] == pytest.approx(.5)
    assert result['coordinates'][0, 2] == pytest.approx(-1.)
    np.testing.assert_array_equal(result['coordinates'][0, 3:5], (0., 0.))
    with torch.no_grad():
        expected = frozen.cost_components(*(torch.tensor(x, dtype=torch.float64)
            for x in (state, command, action, previous)), .45).numpy()
    np.testing.assert_allclose(result['components'], expected, rtol=1e-14, atol=1e-14)
    scaled = state.copy()
    scaled[:, 3:7] *= 3.
    np.testing.assert_allclose(residuals(scaled, command, action, previous, .45)['coordinates'],
                               result['coordinates'], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('position', range(6))
@pytest.mark.parametrize('bad', (float('nan'), float('inf'), -float('inf')))
def test_every_zero_and_endpoint_must_be_finite(position, bad):
    args = [_one(1.) for _ in range(6)]
    args[position][0] = bad
    with pytest.raises(ValueError, match='finite'):
        decompose(*args, eps_true=.00025)


@pytest.mark.parametrize('epsilon', (0., -.001, float('nan'), float('inf'), True))
@pytest.mark.parametrize('name', ('eps_model', 'eps_true'))
def test_invalid_finite_difference_interval_is_rejected(name, epsilon):
    kwargs = {'eps_true': .00025, name: epsilon}
    with pytest.raises(ValueError, match='finite and positive'):
        decompose(*[_one(1.) for _ in range(6)], **kwargs)


@pytest.mark.parametrize('bad', (np.zeros(23, dtype=np.float64), np.zeros((1, 22), dtype=np.float64),
                               np.zeros(22, dtype=np.float32)))
def test_no_silent_shape_broadcasting_or_float32_endpoints(bad):
    args = [_one(1.) for _ in range(6)]
    args[4] = bad
    with pytest.raises(ValueError):
        decompose(*args, eps_true=.00025)


def test_zero_quaternion_is_rejected_before_inverse_rotation():
    pytest.importorskip('torch')
    with pytest.raises(ValueError, match='nonzero quaternion'):
        residuals(np.zeros(45, dtype=np.float64), np.zeros(3, dtype=np.float64),
                  np.zeros(16, dtype=np.float64), np.zeros(16, dtype=np.float64), .45)
