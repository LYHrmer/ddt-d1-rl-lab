"""Pure checks for the archived wheel-contact diagnostic. No robot model required."""
import numpy as np
import pytest

from tools88.contact88 import LoadedPoint, compare_positions, tangent_speeds, window_summary


def test_slip_uses_material_velocity_on_incline_and_ignores_penetration_speed():
    angle = np.deg2rad(4.)
    forward = np.array([np.cos(angle), 0., np.sin(angle)])
    normal = np.array([-np.sin(angle), 0., np.cos(angle)])
    lateral = np.array([0., 1., 0.])
    radius = .085
    omega = np.array([0., .2/radius, 0.])
    # Rolling wheel center at 0.2 m/s and instantaneous bottom material point.
    rolling_velocity = .2*forward + np.cross(omega, -radius*normal)
    assert tangent_speeds(rolling_velocity, normal) == pytest.approx((0., 0.), abs=1e-12)
    sliding = -.03*forward + .04*lateral + .5*normal
    for oriented_normal in (normal, -normal):
        assert tangent_speeds(sliding, oriented_normal) == pytest.approx((-.03, .05))
    # A vertical face normal parallel to world +x gives no longitudinal tangent.
    speed, magnitude = tangent_speeds(np.array([1., 2., 0.]), np.array([1., 0., 0.]))
    assert speed is None and magnitude == 2.


def test_load_weighted_slip_does_not_cancel_opposite_directions_or_hide_unloaded_ticks():
    sample = [[LoadedPoint(.61, 9., -.1, .1, 1., .8, 3),
               LoadedPoint(.59, 1., .2, .2, .5, .8, 3)], []]
    row = window_summary(sample, [.5, .51], .6)
    assert row["mean_axle_normal_force_n"] == 5.
    assert row["normal_load_fraction_past_toe"] == .9
    assert row["longitudinal_slip_signed_m_s"] == pytest.approx(-.07)
    assert row["longitudinal_slip_abs_m_s"] == pytest.approx(.11)
    assert row["sum_tangential_force_magnitude_over_sum_normal_force"] == .15
    assert row["loaded_boundaries"] == 1 and row["boundaries"] == 2
    assert window_summary([[]], [.45], .6)["longitudinal_slip_abs_m_s"] is None
    assert window_summary([], [], .6)["mean_axle_normal_force_n"] is None


def test_comparison_checks_missingness_and_values_separately():
    out = compare_positions([np.nan, .5, .6], [.51, .5001, .6000001])
    assert not out["passed"] and out["common_finite"] == 2
    assert out["missingness_mismatches"] == 1 and out["differences_over_tolerance"] == 1
    assert compare_positions([np.nan, .5], [np.nan, .5000001])["passed"]
