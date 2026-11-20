"""Zero-model checks of executable D1 targets, moving contacts and QP acceptance."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools90 import wbc90 as w
from training.d1_rl.interface import DEFAULT_Q, TORQUE_LIMIT, WHEEL_INDICES, action_to_torque


def test_reachable_servo_bounds_include_saturation_and_match_frozen_contract():
    config = Path(__file__).resolve().parents[1] / "training/configs/pilot03d.json"
    np.testing.assert_array_equal(w.PILOT03D_SCALE, json.loads(config.read_text())["action_scales"])
    rng = np.random.default_rng(102)
    q, dq = DEFAULT_Q + rng.uniform(-3, 3, 16), rng.uniform(-40, 40, 16)
    offset, lower, upper = w.reachable_torques(q, dq)
    for sign, expected in ((-1, lower), (1, upper)):
        np.testing.assert_allclose(action_to_torque(
            np.full(16, sign), q, dq, w.PILOT03D_SCALE)[0], expected, atol=1e-12)
    assert np.any(lower == upper)
    assert np.any(upper - lower > 0)
    nominal = rng.uniform(-1, 1, 16).astype(np.float32)
    for fraction in np.linspace(0, 1, 11):
        desired = lower + fraction * (upper - lower)
        action = w.executable_action(desired, offset, nominal)
        actual, _ = action_to_torque(action, q, dq, w.PILOT03D_SCALE)
        assert action.dtype == np.float32 and np.max(np.abs(action)) <= 1
        np.testing.assert_allclose(actual, desired, atol=2e-6, rtol=0)


def test_bypass_preserves_float32_bits_and_uses_original_servo():
    action = np.linspace(-1, 1, 16, dtype=np.float32)
    action[5] = -0.0
    executable = w.passthrough_action(action)
    assert executable is action and executable.dtype == np.float32
    assert executable.tobytes() == action.tobytes()
    result = w.bypass(action, DEFAULT_Q, np.zeros(16))
    assert result["status"] == "disabled" and result["accepted"]
    assert np.asarray(result["action"], dtype=action.dtype).tobytes() == action.tobytes()
    np.testing.assert_array_equal(result["servo_torque_nm"],
                                  action_to_torque(action, DEFAULT_Q, np.zeros(16), w.PILOT03D_SCALE)[0])


def test_tilted_cylinder_support_and_frame_derivatives():
    axis = np.array([0.1, 0.96, 0.25])
    axis /= np.linalg.norm(axis)
    normal = np.array([-.08, 0, 1])
    normal /= np.linalg.norm(normal)
    omega = np.array([.9, -1.2, .45])
    speed = np.linalg.norm(omega)
    direction = omega / speed

    def contact_at(t):
        angle = t * speed
        rotated = (np.cos(angle) * axis + np.sin(angle) * np.cross(direction, axis)
                   + (1 - np.cos(angle)) * direction * (direction @ axis))
        return w._cylinder_contact(rotated, normal, omega, .087, .020)

    dt = 1e-6
    before, after = contact_at(-dt), contact_at(dt)
    offset, offset_dot, frame, frame_dot = contact_at(0)
    np.testing.assert_allclose(offset @ normal, -(.087 * np.sqrt(1 - (normal @ axis)**2)
                                                 + .020 * abs(normal @ axis)), atol=1e-13)
    np.testing.assert_allclose((after[0] - before[0]) / (2 * dt), offset_dot, atol=1e-9)
    np.testing.assert_allclose((after[2] - before[2]) / (2 * dt), frame_dot, atol=1e-9)
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-14)


def test_rolling_wheel_rim_has_no_material_centripetal_acceleration():
    omega = np.array([0., 3., 0.])
    normal = np.array([0., 0., 1.])
    offset, offset_dot, frame, frame_dot = w._cylinder_contact(
        np.array([0., 1., 0.]), normal, omega, .087, .020)
    velocity = np.array([.087 * 3, 0., 0.]) + np.cross(omega, offset)
    material_bias = np.cross(omega, np.cross(omega, offset))
    geometric_bias = material_bias + np.cross(omega, offset_dot - np.cross(omega, offset))
    np.testing.assert_allclose(frame @ velocity, 0)
    assert material_bias[2] > .7
    np.testing.assert_allclose(frame @ geometric_bias + frame_dot @ velocity, 0)


def _fixture():
    support = np.array([True, False, False, False])
    normal = np.tile([0., 0., 1.], (4, 1))
    state = w.EstimatedState90(np.array([1., 0., 0., 0.]), np.array([0., 0., .5]),
                               np.zeros(3), np.zeros(3), DEFAULT_Q.copy(), np.zeros(16),
                               support, normal)
    contacts = w._Contacts(np.zeros((4, 3, 22)), np.tile(np.eye(3), (4, 1, 1)),
                           np.zeros((4, 3)), np.zeros((4, 3)))
    contacts.jac[0, :, :3] = np.eye(3)
    drive = np.zeros((22, 16))
    drive[6:] = np.eye(16)
    bias = np.r_[0., 0., 10., np.zeros(19)]
    return state, contacts, drive, bias


def test_audit_measures_each_hard_constraint_and_permits_lateral_scrub():
    state, contacts, drive, bias = _fixture()
    forces = np.zeros((4, 3))
    forces[0, 2] = 10.
    args = (np.eye(22), bias, drive, contacts, state.support, -TORQUE_LIMIT, TORQUE_LIMIT)
    qdd, torque, action = np.zeros(22), np.zeros(16), np.zeros(16)
    assert w.audit_solution(*args, 20., qdd, forces, torque, action, torque)["max"] == 0.
    qdd[:3], forces[0], forces[1] = [2., 3., -1.], [6., 3., -2.], [0., 0., 1.]
    servo = torque.copy()
    servo[0], action[1] = 81., 1.5
    violations = w.audit_solution(*args, 20., qdd, forces, torque, action, servo)
    assert violations == pytest.approx({
        "dynamics": 81, "rolling": 2, "normal": 1, "friction": 10.1, "normal_force": 2,
        "inactive_force": 1, "reachable_torque": 1, "torque_limit": 1, "action_bounds": .5,
        "servo_roundtrip": 81, "max": 81})
    qdd, servo = np.zeros(22), np.zeros(16)
    qdd[1], forces[:] = 1., 0.
    forces[0] = [0., 1., 10.]
    assert w.audit_solution(*args, 20., qdd, forces, torque, np.zeros(16), servo)["max"] == 0.


def test_velocity_attitude_and_all_ppo_targets_drive_distinct_objective_rows():
    transform = pytest.importorskip("scipy.spatial.transform")
    state, contacts, _, _ = _fixture()
    controller = w.WBC90.__new__(w.WBC90)
    controller._base, controller._weight_n, controller._nominal_height = 0, 500., .5
    controller._rotation = transform.Rotation
    controller._qadr, controller._dadr = np.arange(7, 23), np.arange(6, 22)
    controller.data = SimpleNamespace(xmat=np.eye(3).reshape(1, 9),
                                      qpos=np.r_[state.base_position_m, state.base_quat_wxyz, DEFAULT_Q],
                                      qvel=np.zeros(22))
    _, base = controller._cost(contacts, state.support, np.zeros(3), np.zeros(16))
    _, changed = controller._cost(contacts, state.support, [0.2, 0.1, 0.3], np.ones(16))
    assert np.all(changed[[0, 1, 5]] < base[[0, 1, 5]])
    assert np.all(changed[6:22] < base[6:22])
    controller.data.xmat[0] = transform.Rotation.from_rotvec([.1, 0., 0.]).as_matrix().ravel()
    _, tilted = controller._cost(contacts, state.support, np.zeros(3), np.zeros(16))
    assert tilted[3] > base[3]


def test_synthetic_qp_returns_executable_servo_target_and_rejects_impossible_contact():
    osqp = pytest.importorskip("osqp")
    sparse = pytest.importorskip("scipy.sparse")
    state, contacts, drive, bias = _fixture()
    controller = w.WBC90.__new__(w.WBC90)
    controller._osqp, controller._sparse = osqp, sparse
    controller._mass, controller._drive, controller._force_limit = np.eye(22), drive, 20.
    controller._reconstruct = lambda state: bias
    controller._contacts = lambda support, normals: contacts
    controller._cost = lambda *args: (np.eye(50), np.r_[np.zeros(34), np.full(16, -200.)])
    result = controller.correct(state, np.zeros(16, dtype=np.float32), np.zeros(3))
    assert result["status"] == "accepted" and result["violations"]["max"] <= w.HARD_TOLERANCE
    assert result["solve_ms"] >= result["osqp_ms"] > 0.
    np.testing.assert_allclose(result["contact_force_world_n"][0], [0, 0, 10], atol=1e-6)
    np.testing.assert_allclose(np.array(result["servo_torque_nm"])[WHEEL_INDICES], 2.5, atol=1e-6)
    controller._force_limit = 0.5
    rejected = controller.correct(state, np.zeros(16), np.zeros(3))
    assert not rejected["accepted"] and rejected["action"] is None
    assert "infeasible" in rejected["solver_status"]
