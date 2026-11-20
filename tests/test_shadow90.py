"""Deterministic geometry, causal selection and accounting without archived rollouts or QP solves."""
from dataclasses import dataclass

import numpy as np
import pytest

from tools90 import shadow90 as shadow


def test_two_delayed_lanes_resolve_a_crease_and_mark_extrapolation():
    xs = np.array([0., .25, .5, .75])
    points = np.stack([np.column_stack((xs, np.full(4, y), .1*np.maximum(xs-.25, 0.) + y*.02))
                       for y in (.16, -.16)])
    terrain = shadow.TerrainPatches()
    assert terrain.at_wheel(0, np.zeros(3)) is None
    terrain.add(points, np.ones((2, 4), bool), 7)
    for lane in range(2):
        y = points[lane, 0, 1]
        back = terrain.at_wheel(lane, np.array([.1, y, .1]))
        front = terrain.at_wheel(lane, np.array([.6, y, .1]))
        assert back[2:4] == front[2:4] == ("range_bracket", 7)
        np.testing.assert_allclose(back[1], shadow._upward(np.array([0., -.02, 1.])))
        np.testing.assert_allclose(front[1], shadow._upward(np.array([-.1, -.02, 1.])))
        assert terrain.at_wheel(lane, np.array([-.1, y, 0.]))[2] == "range_extension"
        assert terrain.at_wheel(lane, np.array([-.16, y, 0.])) is None
    blank = shadow.TerrainPatches()
    blank.add(points, np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=bool), 9)
    assert blank.at_wheel(0, np.array([.1, .16, .1])) is None


def test_footprint_uses_individual_encoder_geometry_and_rim_not_body_attitude():
    points = np.array([[.3, .15, .06], [.3, -.15, 0], [-.3, .15, 0], [-.3, -.15, 0]])
    normals = shadow.footprint_normals(points)
    assert normals.shape == (4, 3)
    np.testing.assert_allclose(normals[3], [0, 0, 1])
    assert len(np.unique(normals.round(6), axis=0)) == 4
    axes = np.tile([0., 1., 0.], (4, 1))
    rim = shadow.rim_points(points, axes, np.full(4, .09), np.full(4, .03), normals)
    actual = ((rim-points)*normals).sum(axis=1)
    np.testing.assert_allclose(actual, -.09*np.sqrt(1-normals[:, 1]**2) - .03*abs(normals[:, 1]))


def test_front_contact_labels_select_the_following_action_and_cannot_cross_a_reset():
    n = 80
    actor = np.zeros((3*n, 627))
    for start in (0, n, 2*n):
        actor[start+50:start+n, 6] = .4
    past = np.zeros((3*n, 4), bool)
    past[2*n+52:2*n+58, 0] = True
    arrays = dict(episode_tick=np.tile(np.arange(1, n+1), 3), terrain=np.repeat([0, 1, 1], n),
                  slope_start=np.repeat([False, True, False], n), loaded_wheels_past_toe=past, proprio627=actor)
    source = dict(flat=[[40, 59]], slope_stand=[[n, n+9], [n+40, n+49]], front_contact=[[2*n+53, 2*n+72]])
    selected = shadow.selected_windows(source, arrays)
    assert selected["front_contact"] == (2*n, list(range(2*n+53, 2*n+73)))
    with pytest.raises(ValueError):
        shadow.selected_windows(dict(source, front_contact=[[2*n+52, 2*n+71]]), arrays)
    arrays["episode_tick"][2*n+60] = 1
    with pytest.raises(ValueError):
        shadow.selected_windows(source, arrays)


def test_a_rejected_qp_does_not_become_an_executable_target_or_poison_the_next_frame():
    @dataclass
    class State:
        joint_q: np.ndarray
        joint_dq: np.ndarray

    class Estimator:
        def update(self, tick, proprio, ranges):
            return State(np.zeros(16), np.zeros(16)), np.zeros(3), {}

    class Controller:
        def __init__(self):
            self.n = 0

        def correct(self, state, nominal, command, enabled=True):
            if enabled:
                self.n += 1
                if self.n == 1:
                    return shadow._rejected("deliberate test rejection")
                return dict(accepted=True, status="accepted", action=nominal.tolist(), violations={"max": 1e-9})
            return dict(accepted=True, status="disabled", action=nominal.tolist())

    model, estimator = Controller(), Estimator()
    action = np.arange(16, dtype=np.float32) / 16
    action[0] = -0.
    rows = []
    for tick in (0, 1):
        row, error = shadow._test_frame(estimator, model, lambda *_: action, tick, np.zeros(627), np.zeros(16), None)
        assert error is None and row["bypass_bitwise_equal"] and row["elapsed_ms"] >= row["estimate_ms"]
        rows.append(row)
    assert rows[0]["enabled"]["action"] is None
    assert rows[1]["enabled"]["accepted"] and rows[1]["solver_attempted"]


def test_delivered_packet_uses_the_capture_attitude_and_wheel_encoders_only():
    """Three synthetic measurements, analytic flat returns and no mj_step or solver call."""
    from types import SimpleNamespace
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    from training.d1_rl.interface import DEFAULT_Q, WHEEL_INDICES

    # Fixed encoder geometry and a zero relative rim velocity isolate the sensor clock.
    # No MuJoCo model, native integration or QP backend is constructed by this test.
    centers = np.array([[.3, .15, -.25], [.3, -.15, -.25], [-.3, .15, -.25], [-.3, -.15, -.25]])
    wheel_axis_frame = Rotation.from_euler("x", 90, degrees=True).as_matrix()

    def kinematics(model, data):
        attitude = Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_matrix()
        data.geom_xpos = centers @ attitude.T
        data.geom_xmat = np.tile((attitude @ wheel_axis_frame).ravel(), (4, 1))

    estimator = object.__new__(shadow.SensorEstimator)
    estimator.mj = SimpleNamespace(mj_kinematics=kinematics, mj_comPos=lambda *_: None,
                                   mj_jac=lambda model, data, jac, *args: jac.fill(0))
    estimator.model, estimator.rotation = None, Rotation
    estimator.data = SimpleNamespace(qpos=np.zeros(23), qvel=np.zeros(22))
    estimator.qadr, estimator.dadr = np.arange(7, 23), np.arange(6, 22)
    estimator.bodies = estimator.geoms = np.arange(4)
    estimator.radius, estimator.half_width = np.full(4, .09), np.full(4, .02)
    estimator.jac = np.empty((3, 22))
    estimator.patches = shadow.TerrainPatches()
    estimator.capture_pose = estimator.last_delivery = None
    estimator.last_tick, estimator.yaw = -1, 0.
    estimator.velocity, estimator.position = np.zeros(3), np.zeros(3)
    estimator.wheel_angles, estimator.previous_rates = np.zeros(4), np.zeros(4)
    estimator.support, estimator.normals = np.ones(4, dtype=bool), np.tile(shadow.UP, (4, 1))
    estimator.mounts = np.array([[.355, .19, .055], [.355, -.19, .055]])
    depression, max_range = np.deg2rad([82., 56., 43., 33.]), 1.45
    estimator.directions = np.column_stack((np.cos(depression), np.zeros(4), -np.sin(depression)))
    estimator.range_max, estimator.delay = max_range, 2
    estimator.capture_due = lambda tick: tick == 0
    frame = np.zeros(627)
    frame[5] = -1.
    frame[9:25] = -DEFAULT_Q
    frame[25+WHEEL_INDICES] = .05
    for tick in (0, 1):
        state, _, evidence = estimator.update(tick, frame, np.zeros(16))
        assert evidence["latest_delivered_capture_tick"] is None
        assert evidence["normal_sources"] == ["kinematic_prior"]*4
    frame[3:6] = [np.sin(.3), 0, -np.cos(.3)]
    encoded = np.tile(.45 / np.sin(depression) / max_range, 2)
    state, _, evidence = estimator.update(2, frame, np.r_[encoded, np.ones(8)])
    assert evidence["latest_delivered_capture_tick"] == 0
    # Tilt changed after acquisition. Applying the pitch at delivery would tip this flat patch.
    for patches in estimator.patches.lanes:
        np.testing.assert_allclose(patches[:, 6:9], np.tile(shadow.UP, (3, 1)), atol=1e-7)
    np.testing.assert_allclose(state.joint_q[WHEEL_INDICES], .04)
    with pytest.raises(ValueError, match="uninterrupted"):
        estimator.update(4, frame, np.zeros(16))
