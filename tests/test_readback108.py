import numpy as np
import pytest

from tools108.readback108 import _cell, _check_packets, _contact_surface, _episode, _motion_series
from tools108.preview108 import BEAMS
from tools94.common94 import read


def test_contact_clock_uses_first_front_load_and_a_full_32_decision_window():
    steps = 40
    spec = dict(ticks=steps, horizon=32, trigger_n=1., pulse_ticks=16, pulse_rad=-.08)
    time = np.arange(steps+1)*.02
    force = np.zeros((10*steps, 4))
    force[0, 2] = 40.
    force[2, 0] = .9999
    force[10, 1] = 1.
    knees = np.zeros((steps, 2))
    knees[2:18] = -.08
    data = dict(time_s=time, axle_x_m=time*.5, base_x_m=time*.25, step_force_n=force, knee_offset_rad=knees,
                fallen=np.zeros(steps, dtype=bool), illegal_contact=np.zeros(steps, dtype=bool))
    row = dict(seed=1, arm="flexion", ticks=steps, status="COMPLETE", sha256="synthetic", trigger_tick=2,
               first_contact_s=.022, max_front_contact_force_n=1., window_complete=True, fallen=False,
               illegal_contact=False, changed_knee_ticks=16, final_axle_x_m=.4, final_base_x_m=.2,
               axle_mm=320., base_mm=160.)
    result = _episode(data, row, "flexion", spec)
    assert result["k"] == 2 and result["safe"]
    np.testing.assert_allclose(result["path"], .01*np.arange(1, 33))

    with pytest.raises(ValueError, match="first_contact_s"):
        _episode(data, {**row, "first_contact_s": .020}, "flexion", spec)
    with pytest.raises(ValueError, match="pulse outside"):
        _episode({**data, "knee_offset_rad": np.roll(knees, -1, axis=0)}, row, "flexion", spec)


def test_oblique_corner_load_differs_from_horizontal_top_and_keeps_nonwheel_support_separate():
    spec = dict(trigger_n=1., horizon=1, geometry=dict(front_x_m=.45, height_m=.03))
    raw = np.array([
        [0, 0, 22, .8, .45, .1, .03, 0, 0, 1],
        [0, 0, 22, 12., .45, .1, .03, -.8, 0, .6],
        [1, 1, 29, 20., .451, -.1, .03, 0, 0, 1],
        [1, 2, 34, 5., .451, -.1, .020, 0, 0, 1],
        [1, -1, 3, 4., .451, 0., .03, 0, 0, 1],
    ])
    forces = np.array([[12.8, 0., 0., 0.], [0., 20., 5., 0.]])
    result = _contact_surface(raw, forces, spec)
    assert result["top_by_wheel"] == [False, True, False, False]
    assert result["max_nonwheel_step_force_n"] == 4. and result["loaded_front_events"] == 2
    np.testing.assert_allclose(result["front_x_minus_edge_mm"], [0., 1.], atol=1e-12)
    assert result["front_normal_z"] == [.6, 1.]
    with pytest.raises(AssertionError):
        _contact_surface(raw, forces*2, spec)
    raw[raw[:, 0] == 1, 0] = 10
    extended = np.zeros((20, 4))
    extended[[0, 10]] = forces
    window = _contact_surface(raw, extended, spec, 1)["window"]
    assert (window["window_front_wheel_loaded_contact_records"], window["window_top_loaded_records"]) == (1, 1)
    assert window["window_other_contact_unit_normal_z"] is None
    assert window["window_peak_front_wheel_step_normal"]["native_substep"] == 10


def test_motion_peak_channels_decode_wxyz_pitch_and_keep_millimetres_separate():
    angle = np.deg2rad(25.)
    quat = np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.])
    data = dict(calves_rad=np.array([[-1.2, -1.4], [-1.2, -1.4]]), wheel_z_m=np.full((2, 2), .14),
                base_z_m=np.array([.32, .35]), base_quat=np.stack([quat, -quat]))
    curves = _motion_series(data, np.array([.10, .12]))
    np.testing.assert_allclose(curves["base_pitch_deg"], 25.)
    np.testing.assert_allclose(curves["mean_front_calf_angle_rad"], -1.3)
    np.testing.assert_allclose(curves["front_axle_forward_x_mm"], [100., 120.])
    np.testing.assert_allclose(curves["mean_front_wheel_center_height_mm"], 140.)
    np.testing.assert_allclose(curves["base_height_mm"], [320., 350.])


def test_only_an_already_delivered_range_packet_can_match_the_precontact_actor_input():
    sensor = read(BEAMS)
    spec = dict(geometry=dict(front_x_m=.45, length_m=1., half_width_m=.7, height_m=.03))
    pair = dict(k=5, contact_s=.084, hashes=["native", "paired"])
    pose = np.array([[0., 0., .3, 1., 0., 0., 0.]])
    obs = np.zeros(643, dtype=np.float32)
    motion = dict(s_packet_ticks=np.array([[0, 2]]), s_range_features=np.zeros((1, 16), np.float32),
                  s_capture_pose=pose.copy(), s_delivery_pose=pose.copy(), s_base_z_m=np.full(5, .3),
                  s_base_quat=np.tile(pose[:, 3:], (5, 1)))
    measured = dict(s_base_x_m=np.zeros(5))
    row = dict(paired_precontact=True, first_contact_ms=84, original_trace_sha256="native",
               precontact_packets=[[0, 2, 44, 0, 0, 0, 0, 0, 8]], packets_with_top_both_sides=0,
               counts=dict(top=0, floor=0, front_riser=0, unclassified_valid=0, invalid=8),
               first_top_packet=None, last_top_packet=None, max_capture_pose_labeled_surface_residual_um=None,
               max_delivery_pose_same_surface_residual_mm=None)
    _check_packets(row, sensor, spec, pair, "s", motion, measured, obs)
    with pytest.raises(AssertionError):
        _check_packets(row, sensor, spec, pair, "s", {**motion, "s_packet_ticks":np.array([[0, 3]])}, measured, obs)
    obs[627] = 1.
    with pytest.raises(AssertionError):
        _check_packets(row, sensor, spec, pair, "s", motion, measured, obs)


@pytest.mark.parametrize(("predicted", "reference", "reasons"), [
    (5., 4., []),
    (5.001, 4., ["outside_native_25pct"]),
    (.7, .7, ["below_engineering_resolution"]),
    (1., -1., ["direction_mismatch", "outside_native_25pct"]),
    (0., 0., ["below_engineering_resolution", "direction_mismatch"]),
])
def test_blind_score_distinguishes_resolution_direction_and_magnitude(predicted, reference, reasons):
    pred, native = np.array([0., predicted]), np.array([0., reference])
    result = _cell(pred, native, 2., pred*.001, 1., .25)
    assert result["reasons"] == reasons
    assert result["passed"] is (not reasons)
    assert result["train_mean_error_mm"] == abs(2.-reference)
    assert result["zero_error_mm"] == abs(reference)
    assert result["relative_error"] == (None if reference == 0 else abs(predicted-reference)/abs(reference))
    assert result["trajectory_mae_mm"] == abs(predicted-reference)/2.
