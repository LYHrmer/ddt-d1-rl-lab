"""Cold, sensor-conditioned whole-body inverse dynamics for the D1 offline panel.

The optimization variables are 22 generalized accelerations, four world contact
forces and 16 servo torques. All kinematics belong to a private nominal-model
data object. Outputs are prospective same-state targets at the P/V interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from training.d1_rl.interface import (
    DEFAULT_Q, JOINT_NAMES, KD, KP, LEG_INDICES, TORQUE_LIMIT, WHEEL_INDICES,
    action_to_torque,
)


PILOT03D_SCALE = np.tile([0.5, 0.5, 0.5, 5.0], 4)
SERVO_GAIN = KP * PILOT03D_SCALE
SERVO_GAIN[WHEEL_INDICES] = KD[WHEEL_INDICES] * PILOT03D_SCALE[WHEEL_INDICES]
MU = 0.55
HARD_TOLERANCE = 1e-4
FRICTION_ROWS = np.array([[1, 1, -MU], [1, -1, -MU], [-1, 1, -MU], [-1, -1, -MU]])


@dataclass(frozen=True)
class EstimatedState90:
    """World velocities and position refer to the free-joint origin; gyro is body-local."""

    base_quat_wxyz: np.ndarray
    base_position_m: np.ndarray
    base_v_world_m_s: np.ndarray
    base_omega_body_rad_s: np.ndarray
    joint_q: np.ndarray
    joint_dq: np.ndarray
    support: np.ndarray
    normals_world: np.ndarray


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _unit(vector, name):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError(f"undefined {name}")
    return vector / norm


def _nominal_action(value):
    action = _finite(value, (16,), "nominal action")
    if (np.abs(action) > 1.0).any():
        raise ValueError("nominal PPO targets must already lie in [-1,1]")
    return action


def reachable_torques(q, dq):
    """Exact image of the bounded policy action under the saturated pilot03d servo."""
    q = _finite(q, (16,), "joint positions")
    dq = _finite(dq, (16,), "joint velocities")
    offset = KP * (DEFAULT_Q - q) - KD * dq
    return (offset, np.clip(offset - SERVO_GAIN, -TORQUE_LIMIT, TORQUE_LIMIT),
            np.clip(offset + SERVO_GAIN, -TORQUE_LIMIT, TORQUE_LIMIT))


def executable_action(torque, offset, nominal):
    """Use the inverse on the linear region and the closest PPO target on a plateau."""
    raw_action = (torque - offset) / SERVO_GAIN
    # On a physical saturation plateau several actions produce the same torque.
    raw_action = np.where(torque >= TORQUE_LIMIT, np.maximum(raw_action, nominal), raw_action)
    raw_action = np.where(torque <= -TORQUE_LIMIT, np.minimum(raw_action, nominal), raw_action)
    return np.clip(raw_action, -1.0, 1.0).astype(np.float32)


def passthrough_action(nominal_action):
    """The disabled execution path returns the original PPO array, including signed zero."""
    original = np.asarray(nominal_action)
    _nominal_action(original)
    return original


def bypass(nominal_action, joint_q, joint_dq):
    """Serialize only after the unchanged executable target has reached the P/V mapping."""
    nominal = passthrough_action(nominal_action)
    torque, _ = action_to_torque(nominal, joint_q, joint_dq, PILOT03D_SCALE)
    return {"accepted": True, "status": "disabled", "action": nominal.tolist(),
            "action_dtype": str(nominal.dtype), "servo_torque_nm": torque.tolist(), "violations": None}


def _cylinder_contact(axis, normal, omega, radius, half_width):
    """Instantaneous support point on a cylinder and its moving-rim derivatives.

    At an exactly level axle the support line is a segment; use its center.
    During a tilted contact the finite half-width selects the supporting edge.
    The estimated plane normal is held constant over this instantaneous solve.
    """
    dot = axis @ normal
    projection = normal - dot * axis
    radial = _unit(projection, "projected wheel radius")
    axial = -half_width * np.sign(dot)
    offset = -radius * radial + axial * axis
    axis_dot = np.cross(omega, axis)
    projection_dot = -axis_dot * dot - axis * (axis_dot @ normal)
    radial_dot = (projection_dot - radial * (radial @ projection_dot)) / np.linalg.norm(projection)
    offset_dot = -radius * radial_dot + axial * axis_dot
    roll_raw = np.cross(axis, normal)
    rolling = _unit(roll_raw, "rolling direction")
    roll_raw_dot = np.cross(axis_dot, normal)
    rolling_dot = (roll_raw_dot - rolling * (rolling @ roll_raw_dot)) / np.linalg.norm(roll_raw)
    frame = np.stack((rolling, np.cross(normal, rolling), normal))
    rate = np.stack((rolling_dot, np.cross(normal, rolling_dot), np.zeros(3)))
    return offset, offset_dot, frame, rate


@dataclass
class _Contacts:
    jac: np.ndarray                 # [wheel, world xyz, dof]
    frame: np.ndarray               # rows = rolling, lateral, normal
    target: np.ndarray              # desired components of J qdd in that frame
    point: np.ndarray


def _constraints(mass, bias, drive, contacts, support, lower_tau, upper_tau, force_limit):
    """Keep the full Newton–Euler equality, even for zero estimated supports."""
    active = np.flatnonzero(support)
    matrix = np.zeros((50 + 7 * len(active), 50))
    lower = np.zeros(len(matrix))
    upper = np.zeros(len(matrix))
    matrix[:22, :22] = mass
    matrix[:22, 22:34] = -contacts.jac.reshape(12, 22).T
    matrix[:22, 34:] = -drive
    lower[:22] = upper[:22] = -bias
    matrix[22:50, 22:50] = np.eye(28)
    force_bound = np.repeat(np.where(support, np.inf, 0.0), 3)
    lower[22:34], upper[22:34] = -force_bound, force_bound
    lower[34:50], upper[34:50] = lower_tau, upper_tau
    for block, wheel in enumerate(active):
        row = 50 + 7 * block
        force = slice(22 + 3 * wheel, 25 + 3 * wheel)
        frame = contacts.frame[wheel]
        matrix[row:row + 4, force] = FRICTION_ROWS @ frame
        lower[row:row + 4] = -np.inf
        matrix[row + 4, force] = frame[2]
        upper[row + 4] = force_limit
        matrix[row + 5:row + 7, :22] = frame[[0, 2]] @ contacts.jac[wheel]
        lower[row + 5:row + 7] = upper[row + 5:row + 7] = contacts.target[wheel, [0, 2]]
    return matrix, lower, upper


def _max(value):
    return float(np.max(value, initial=0.0))


def audit_solution(mass, bias, drive, contacts, support, lower_tau, upper_tau,
                   force_limit, qdd, force, qp_torque, action, servo_torque):
    """Recompute physical equations using the independently replayed float32 action."""
    generalized_force = np.einsum("wij,wi->j", contacts.jac, force)
    dynamics = mass @ qdd + bias - generalized_force - drive @ servo_torque
    active = np.flatnonzero(support)
    components = np.einsum("wij,wj->wi", contacts.frame[active], force[active])
    acceleration = np.einsum("wij,j->wi", contacts.jac[active], qdd)
    residual = (np.einsum("wij,wj->wi", contacts.frame[active], acceleration)
                - contacts.target[active])
    normal = components[:, 2]
    metrics = {
        "dynamics": _max(np.abs(dynamics)),
        "rolling": _max(np.abs(residual[:, 0])), "normal": _max(np.abs(residual[:, 2])),
        "friction": _max(np.sum(np.abs(components[:, :2]), axis=1) - MU * normal),
        "normal_force": max(_max(-normal), _max(normal - force_limit)),
        "inactive_force": _max(np.abs(force[~support])),
        "reachable_torque": max(_max(lower_tau - servo_torque), _max(servo_torque - upper_tau)),
        "torque_limit": _max(np.abs(servo_torque) - TORQUE_LIMIT),
        "action_bounds": _max(np.abs(action) - 1.0),
        "servo_roundtrip": _max(np.abs(qp_torque - servo_torque)),
    }
    metrics["max"] = max(metrics.values())
    return metrics


class WBC90:
    """Own an isolated MjData; only estimated state and PPO/velocity commands enter."""

    def __init__(self, model):
        import mujoco
        import osqp
        from scipy import sparse
        from scipy.spatial.transform import Rotation

        if mujoco.__version__ != "3.3.0" or (model.nq, model.nv, model.nu) != (23, 22, 16):
            raise ValueError("expected the frozen floating-base D1 with MuJoCo 3.3.0")
        self.model, self.data = model, mujoco.MjData(model)
        self._mj, self._osqp, self._sparse, self._rotation = mujoco, osqp, sparse, Rotation
        joints = np.array([model.joint(name).id for name in JOINT_NAMES])
        self._qadr, self._dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        self._base = model.body("base_link").id
        self._axles = joints[WHEEL_INDICES]
        self._wheels = model.jnt_bodyid[self._axles]
        cylinders = []
        for body in self._wheels:
            ids = np.flatnonzero((model.geom_bodyid == body)
                                 & (model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER)
                                 & (model.geom_contype != 0))
            if len(ids) != 1:
                raise ValueError("expected one D1 collision cylinder per wheel")
            cylinders.append(ids[0])
        self._geoms = np.array(cylinders)
        if (not np.array_equal(model.actuator_trnid[:, 0], joints)
                or not np.allclose(model.actuator_gear[:, 0], 1)
                or model.jnt_qposadr[model.joint("floating_base_joint").id] != 0):
            raise ValueError("D1 joint-to-motor transmission contract changed")
        self._drive = np.zeros((22, 16))
        self._drive[self._dadr, np.arange(16)] = 1.0
        self._mass = np.empty((22, 22))
        self._weight_n = float(model.body_mass.sum() * np.linalg.norm(model.opt.gravity))
        self._force_limit = 0.65 * self._weight_n
        self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        self.data.qpos[self._qadr] = DEFAULT_Q
        mujoco.mj_kinematics(model, self.data)
        axes = self.data.geom_xmat[self._geoms].reshape(4, 3, 3)[:, 2, 2]
        radius, half_width = model.geom_size[self._geoms, :2].T
        extents = radius * np.sqrt(np.maximum(0, 1 - axes**2)) + half_width * np.abs(axes)
        self._nominal_height = float(np.max(extents - self.data.geom_xpos[self._geoms, 2]))

    def _reconstruct(self, state):
        data, model, mj = self.data, self.model, self._mj
        data.qpos[:3] = _finite(state.base_position_m, (3,), "base position")
        data.qpos[3:7] = _unit(_finite(state.base_quat_wxyz, (4,), "attitude"), "attitude")
        data.qpos[self._qadr] = _finite(state.joint_q, (16,), "joint positions")
        data.qvel[:3] = _finite(state.base_v_world_m_s, (3,), "base origin velocity")
        data.qvel[3:6] = _finite(state.base_omega_body_rad_s, (3,), "gyroscope")
        data.qvel[self._dadr] = _finite(state.joint_dq, (16,), "encoder rates")
        # No force/constraint solve or actual contact read. Rigid-body terms are
        # evaluated by the nominal model at the estimated generalized state.
        mj.mj_fwdPosition(model, data)
        mj.mj_fwdVelocity(model, data)
        mj.mj_fullM(model, self._mass, data.qM)
        # Native dry friction is constraint-based, outside qfrc_passive. Use
        # sliding Coulomb friction and select zero from the static interval.
        return (data.qfrc_bias - data.qfrc_passive
                + model.dof_frictionloss * np.sign(data.qvel))

    def _contacts(self, support, normals):
        model, data, mj = self.model, self.data, self._mj
        result = _Contacts(np.zeros((4, 3, 22)), np.zeros((4, 3, 3)),
                           np.zeros((4, 3)), np.zeros((4, 3)))
        for i in np.flatnonzero(support):
            n = _unit(normals[i], "contact normal")
            if n[2] <= 0.0:
                raise ValueError("support normal must point up")
            body, geom = int(self._wheels[i]), self._geoms[i]
            axis = data.xaxis[self._axles[i]]
            rotational = np.empty((3, 22))
            mj.mj_jac(model, data, None, rotational, data.geom_xpos[geom], body)
            omega = rotational @ data.qvel
            offset, offset_dot, frame, frame_dot = _cylinder_contact(
                axis, n, omega, *model.geom_size[geom, :2])
            point = data.geom_xpos[geom] + offset
            jacobian = result.jac[i]
            mj.mj_jac(model, data, jacobian, None, point, body)
            material_jdot = np.empty_like(jacobian)
            mj.mj_jacDot(model, data, material_jdot, None, point, body)
            # mj_jacDot holds one material point fixed on the wheel. The point
            # selected geometrically moves across the rotating surface.
            bias = material_jdot @ data.qvel + np.cross(omega, offset_dot - np.cross(omega, offset))
            velocity = jacobian @ data.qvel
            result.target[i] = frame @ (-bias - 20.0 * velocity) - frame_dot @ velocity
            result.frame[i], result.point[i] = frame, point
        return result

    def _cost(self, contacts, support, command, nominal):
        data = self.data
        rotation = data.xmat[self._base].reshape(3, 3)
        normal = (_unit(contacts.frame[support, 2].mean(axis=0), "support normal")
                  if np.any(support) else np.array([0., 0., 1.]))
        heading = rotation[:, 0]
        forward = _unit(heading - normal * (normal @ heading), "ground-tangent heading")
        left = np.cross(normal, forward)
        orientation = np.column_stack((forward, left, normal))
        velocity_error = command[0] * forward + command[1] * left - data.qvel[:3]
        desired_linear = 3.0 * (velocity_error - normal * (normal @ velocity_error))
        if np.any(support):
            height = np.einsum("ij,ij->i", contacts.frame[support, 2],
                               data.qpos[:3] - contacts.point[support]).mean()
            desired_linear += normal * (80.0 * (self._nominal_height - height)
                                        - 18.0 * (normal @ data.qvel[:3]))
        attitude_error = self._rotation.from_matrix(orientation @ rotation.T).as_rotvec()
        desired_angular = (80.0 * attitude_error + 16.0 * (
            normal * command[2] - rotation @ data.qvel[3:6]))
        desired_joints = np.empty(16)
        q, dq = data.qpos[self._qadr], data.qvel[self._dadr]
        desired_joints[LEG_INDICES] = (
            30.0 * (DEFAULT_Q + PILOT03D_SCALE * nominal - q) - 6.0 * dq)[LEG_INDICES]
        desired_joints[WHEEL_INDICES] = (10.0 * (
            PILOT03D_SCALE * nominal - dq))[WHEEL_INDICES]
        jac = np.zeros((22, 50))
        jac[:3, :3], jac[3:6, 3:6] = np.eye(3), rotation
        jac[6 + np.arange(16), self._dadr] = 1
        weights = np.array([1, 1, 5, 10, 10, 2] + [0.05] * 16)
        reference = np.r_[desired_linear, desired_angular, desired_joints]
        hessian = jac.T @ (weights[:, None] * jac)
        gradient = -jac.T @ (weights * reference)
        diagonal = np.r_[np.full(22, 1e-7), np.full(12, 1e-4 / self._weight_n**2),
                         1e-3 / TORQUE_LIMIT**2]
        hessian.flat[::51] += diagonal
        for wheel in np.flatnonzero(support):
            lateral = contacts.frame[wheel, 1] @ contacts.jac[wheel]
            hessian[:22, :22] += np.outer(lateral, lateral)
            gradient[:22] -= lateral * contacts.target[wheel, 1]
        return hessian, gradient

    def correct(self, state, nominal_action, command, enabled=True):
        start = perf_counter()
        record = {"accepted": False, "status": "rejected", "reason": None,
                  "action": None, "violations": None, "servo_torque_nm": None,
                  "solver_status": None, "osqp_ms": None}
        try:
            if enabled:
                self._solve_frame(record, state, nominal_action, command)
            else:
                record.update(bypass(nominal_action, state.joint_q, state.joint_dq))
        except (ValueError, RuntimeError, ArithmeticError, self._osqp.OSQPException) as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        record["solve_ms"] = (perf_counter() - start) * 1000.0
        return record

    def _solve_frame(self, record, state, nominal_action, command):
        nominal = _nominal_action(nominal_action)
        command = _finite(command, (3,), "velocity/yaw command")
        support = np.asarray(state.support)
        normals = np.asarray(state.normals_world, dtype=float)
        if support.shape != (4,) or not np.all(np.isin(support, [False, True])):
            raise ValueError("expected four estimated contact indicators")
        support = support.astype(bool)
        if normals.shape != (4, 3) or not np.all(np.isfinite(normals[support])):
            raise ValueError("active wheels require finite estimated normal vectors")
        bias = self._reconstruct(state)
        contacts = self._contacts(support, normals)
        offset, lo, hi = reachable_torques(state.joint_q, state.joint_dq)
        hard = _constraints(self._mass, bias, self._drive, contacts, support, lo, hi, self._force_limit)
        hessian, gradient = self._cost(contacts, support, command, nominal)
        record.update(reachable_lo_nm=lo.tolist(), reachable_hi_nm=hi.tolist())
        solver = self._osqp.OSQP()
        solve_start = perf_counter()
        solver.setup(P=self._sparse.csc_matrix(np.triu(hessian)), q=gradient,
                     A=self._sparse.csc_matrix(hard[0]), l=hard[1], u=hard[2], verbose=False,
                     eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, polishing=True)
        solution = solver.solve(raise_error=False)
        info = solution.info
        record.update(osqp_ms=(perf_counter() - solve_start) * 1000.0,
                      solver_status=info.status, iterations=int(info.iter),
                      primal_residual=float(info.prim_res), dual_residual=float(info.dual_res))
        if info.status_val not in (1, 2) or solution.x is None:
            record["reason"] = f"OSQP: {info.status}"
            return
        if not np.all(np.isfinite(solution.x)):
            record["reason"] = "nonfinite QP solution"
            return
        qdd, forces, torque = solution.x[:22], solution.x[22:34].reshape(4, 3), solution.x[34:]
        action = executable_action(torque, offset, nominal)
        servo, _ = action_to_torque(action, state.joint_q, state.joint_dq, PILOT03D_SCALE)
        violations = audit_solution(self._mass, bias, self._drive, contacts, support, lo, hi,
                                    self._force_limit, qdd, forces, torque, action, servo)
        record.update(qdd=qdd.tolist(), contact_force_world_n=forces.tolist(),
                      qp_torque_nm=torque.tolist(), servo_torque_nm=servo.tolist(), violations=violations)
        if not all(np.isfinite(value) for value in violations.values()) or violations["max"] > HARD_TOLERANCE:
            record.update(reason="executed-target constraint residual", rejected_action=action.tolist())
            return
        record.update(accepted=True, status="accepted", action=action.tolist())
