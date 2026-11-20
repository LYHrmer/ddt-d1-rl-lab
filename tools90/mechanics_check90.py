"""Independent readback of saved WBC targets without a QP solve or a physics step."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.d1_rl.interface import DEFAULT_Q, JOINT_NAMES, KD, KP, LEG_INDICES, TORQUE_LIMIT, WHEEL_INDICES


FD_DT = 1e-6
HARD_TOLERANCE = 1e-4
FIELDS = ("dynamics", "rolling", "normal", "friction", "normal_force", "inactive_force",
          "reachable_torque", "torque_limit", "action_bounds", "servo_roundtrip", "max")


def _max(values):
    return float(np.max(values, initial=0.))


def _array(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"invalid {label} in numeric WBC record")
    return array


class _Readback:
    def __init__(self):
        import mujoco
        from tools87.terrain87 import load_model

        self.mj, self.model = mujoco, load_model("flat")
        if mujoco.__version__ != "3.3.0":
            raise ValueError("mechanics readback requires the tested MuJoCo 3.3.0 binding")
        model = self.model
        self.data = mujoco.MjData(model)
        self.perturbed = mujoco.MjData(model)
        self.joints = [model.joint(name).id for name in JOINT_NAMES]
        self.qadr = model.jnt_qposadr[self.joints]
        self.dadr = model.jnt_dofadr[self.joints]
        self.geoms = []
        for name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            body = model.body(name).id
            ids = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)
                                 & (model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))
            if len(ids) != 1:
                raise ValueError(f"expected one contact cylinder on {name}")
            self.geoms.append(int(ids[0]))
        config = Path(__file__).resolve().parents[1] / "training/configs/pilot03d.json"
        self.scale = _array(json.loads(config.read_bytes())["action_scales"], (16,), "pilot03d action scale")
        self.upper_normal = .65 * model.body_mass.sum() * np.linalg.norm(model.opt.gravity)

    def _servo(self, q, dq, action):
        position = DEFAULT_Q.copy()
        position[LEG_INDICES] += self.scale[LEG_INDICES] * action[LEG_INDICES]
        velocity = np.zeros(16)
        velocity[WHEEL_INDICES] = self.scale[WHEEL_INDICES] * action[WHEEL_INDICES]
        return np.clip(KP * (position - q) + KD * (velocity - dq), -TORQUE_LIMIT, TORQUE_LIMIT)

    def _rim(self, data, active, normals, reference_edges=None):
        """Get slip components from the cylinder's spatial velocity at its center."""
        mj, model = self.mj, self.model
        components, frames, points, edges, transport = [], [], [], [], []
        for idx, wheel in enumerate(active):
            geom = self.geoms[wheel]
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            normal = normals[wheel]
            roll = np.cross(axis, normal)
            roll /= np.linalg.norm(roll)
            lateral = np.cross(normal, roll)
            frame = np.stack((roll, lateral, normal))
            edge = float(np.sign(axis @ normal))
            selected = edge if reference_edges is None else reference_edges[idx]
            radius, width = model.geom_size[geom, :2]
            offset = -radius * np.cross(roll, axis) - width * selected * axis
            spatial = np.empty(6)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_GEOM, geom, spatial, 0)
            slip_world = spatial[3:] + np.cross(spatial[:3], offset)
            components.append(frame @ slip_world)
            transport.append(frame @ np.cross(spatial[:3], slip_world))
            frames.append(frame)
            points.append(data.geom_xpos[geom] + offset)
            edges.append(edge)
        return (np.array(components).reshape(-1, 3), np.array(frames).reshape(-1, 3, 3),
                np.array(points).reshape(-1, 3), np.array(edges), np.array(transport).reshape(-1, 3))

    def _symmetric_slip(self, qdd, active, normals, edges):
        """Sample configuration and instantaneous velocity on a local kinematic curve."""
        mj, model, data, trial = self.mj, self.model, self.data, self.perturbed
        slips, actual_edges = [], []
        for sign in (-1, 1):
            dt = sign * FD_DT
            trial.qpos[:] = data.qpos
            mj.mj_integratePos(model, trial.qpos, data.qvel + 0.5 * dt * qdd, dt)
            trial.qvel[:] = data.qvel + dt * qdd
            mj.mj_kinematics(model, trial)
            mj.mj_comPos(model, trial)
            mj.mj_comVel(model, trial)
            # At a switching edge the derivative of the support point is
            # discontinuous. Compare the derivative on the recorded branch
            # and count any perturbation which naturally selects another edge.
            components, _, _, sample_edges, _ = self._rim(trial, active, normals, edges)
            slips.append(components)
            actual_edges.append(sample_edges)
        switches = np.count_nonzero((actual_edges[0] != edges) | (actual_edges[1] != edges))
        return (slips[1] - slips[0]) / (2 * FD_DT), int(switches)

    def evaluate(self, frame):
        model, data, mj = self.model, self.data, self.mj
        estimate, solution = frame["estimated_state"], frame["enabled"]
        command = _array(frame["command"], (3,), "velocity and yaw-rate command")
        q = _array(estimate["joint_q"], (16,), "joint positions")
        dq = _array(estimate["joint_dq"], (16,), "joint rates")
        data.qpos[:3] = _array(estimate["base_position_m"], (3,), "base position")
        quat = _array(estimate["base_quat_wxyz"], (4,), "attitude")
        quat_norm = np.linalg.norm(quat)
        if quat_norm == 0:
            raise ValueError("zero saved orientation quaternion")
        data.qpos[3:7] = quat / quat_norm
        data.qpos[self.qadr] = q
        data.qvel[:3] = _array(estimate["base_v_world_m_s"], (3,), "linear velocity")
        data.qvel[3:6] = _array(estimate["base_omega_body_rad_s"], (3,), "body rate")
        data.qvel[self.dadr] = dq
        support = np.asarray(estimate["support"])
        if support.shape != (4,) or not np.all(np.isin(support, [True, False])):
            raise ValueError("invalid four-wheel support flags")
        support = support.astype(bool)
        active = np.flatnonzero(support)
        normals = np.asarray(estimate["normals_world"], dtype=float).copy()
        if normals.shape != (4, 3) or not np.all(np.isfinite(normals[support])):
            raise ValueError("invalid saved active contact normal")
        lengths = np.linalg.norm(normals[support], axis=1, keepdims=True)
        if (lengths == 0).any():
            raise ValueError("zero saved active contact normal")
        normals[support] /= lengths
        qdd = _array(solution["qdd"], (22,), "generalized acceleration")
        forces = _array(solution["contact_force_world_n"], (4, 3), "world contact forces")
        offered = solution["action"] if solution["accepted"] else solution["rejected_action"]
        action = _array(np.asarray(offered, dtype=np.float32), (16,), "executable float32 target")
        torque = self._servo(q, dq, action)
        low, high = self._servo(q, dq, -np.ones(16)), self._servo(q, dq, np.ones(16))
        raw_torque = _array(solution["qp_torque_nm"], (16,), "proposed torques")

        mj.mj_fwdPosition(model, data)
        mj.mj_fwdVelocity(model, data)
        data.qacc[:] = qdd
        generalized = np.empty(22)
        mj.mj_rne(model, data, 1, generalized)
        # mj_rne(1) includes body accelerations and bias; rotor inertia is
        # added by mj_crb to the joint-space mass but is outside the RNE tree.
        generalized += model.dof_armature * qdd - data.qfrc_passive
        generalized += model.dof_frictionloss * np.sign(data.qvel)
        generalized[self.dadr] -= torque
        slip, frames, points, edges, transport = self._rim(data, active, normals)
        contact_loads = np.zeros(22)
        for idx, wheel in enumerate(active):
            mj.mj_applyFT(model, data, forces[wheel], np.zeros(3), points[idx],
                          int(model.geom_bodyid[self.geoms[wheel]]), contact_loads)
        derivative, switches = self._symmetric_slip(qdd, active, normals, edges)
        stabilized = derivative + 20. * slip
        projected = np.einsum("wij,wj->wi", frames, forces[support])
        fn = projected[:, 2]
        violations = {
            "dynamics": _max(np.abs(generalized - contact_loads)),
            "rolling": _max(np.abs(stabilized[:, 0])), "normal": _max(np.abs(stabilized[:, 2])),
            "friction": _max(np.abs(projected[:, 0]) + np.abs(projected[:, 1]) - .55 * fn),
            "normal_force": max(_max(-fn), _max(fn - self.upper_normal)),
            "inactive_force": _max(np.abs(forces[~support])),
            "reachable_torque": max(_max(low - torque), _max(torque - high)),
            "torque_limit": _max(np.abs(torque) - TORQUE_LIMIT), "action_bounds": _max(np.abs(action) - 1.),
            "servo_roundtrip": _max(np.abs(raw_torque - torque)),
        }
        violations["max"] = max(violations.values())
        if not np.all(np.isfinite(list(violations.values()))):
            raise ValueError("nonfinite independent mechanics residual")
        saved = _array(solution["servo_torque_nm"], (16,), "recorded servo torque")
        # MuJoCo 3.3.0's spatial jacDot term needs omega x v_material
        # to recover the classical acceleration of a moving material point.
        # Subtract this one term from the independent derivative and compare
        # with the stored analytic residual; this is a diagnostic, never a
        # correction to the archived targets or to their acceptance gate.
        omitted = stabilized - transport
        omission_gap = {key: abs(_max(np.abs(omitted[:, idx])) - solution["violations"][key])
                        for key, idx in (("rolling", 0), ("normal", 2))}
        return (violations, switches, command, _max(np.abs(saved - torque)), omission_gap,
                {"active_wheels": active.tolist(), "stabilized_slip_acceleration_m_s2": stabilized.tolist(),
                 "omega_cross_material_velocity_in_contact_frame_m_s2": transport.tolist()})


def check_solutions(frames):
    """Summarize every saved numeric solution, including rejected numeric candidates."""
    result = dict(numeric_rows=0, accepted_rows=0, rejected_numeric_rows=0, skipped_non_numeric_rows=0,
                  max_violations=None, max_abs_difference_from_solver_record=None, maxima_at={},
                  finite_difference_dt_s=FD_DT, rim_edge_switches=0, servo_receipt_difference_nm=None,
                  command_min=None, command_max=None, native_steps=0, qp_solves=0,
                  original_hard_tolerance=HARD_TOLERANCE, numeric_within_tolerance=0,
                  numeric_over_tolerance=0, accepted_within_tolerance=0, accepted_over_tolerance=0,
                  per_constraint_counts={field: dict(within=0, over=0, accepted_within=0, accepted_over=0)
                                         for field in FIELDS},
                  max_gap_after_omitting_transport_from_fd_m_s2=None, rows=[])
    readback = None
    for frame in frames:
        solution = frame["enabled"]
        if "qdd" not in solution:
            if solution["accepted"]:
                raise ValueError("accepted WBC frame missing numeric acceleration")
            result["skipped_non_numeric_rows"] += 1
            continue
        if readback is None:
            readback = _Readback()
            result["max_violations"] = {name: 0. for name in FIELDS}
            result["max_abs_difference_from_solver_record"] = {name: 0. for name in FIELDS}
            result["servo_receipt_difference_nm"] = 0.
            result["max_gap_after_omitting_transport_from_fd_m_s2"] = {"rolling": 0., "normal": 0.}
        values, switches, command, servo_delta, omission_gap, wheel_data = readback.evaluate(frame)
        key = {name: frame[name] for name in ("seed", "phase", "source_row")}
        accepted = bool(solution["accepted"])
        differences = result["max_abs_difference_from_solver_record"]
        past = []
        for name, value in values.items():
            if name not in result["maxima_at"] or value > result["max_violations"][name]:
                result["max_violations"][name], result["maxima_at"][name] = value, key
            delta = abs(value - float(solution["violations"][name]))
            if not np.isfinite(delta):
                raise ValueError("nonfinite stored WBC residual")
            differences[name] = max(differences[name], delta)
            which = "within" if value <= HARD_TOLERANCE else "over"
            result["per_constraint_counts"][name][which] += 1
            if accepted:
                result["per_constraint_counts"][name]["accepted_" + which] += 1
            if which == "over" and name != "max":
                past.append(name)
        result["numeric_over_tolerance" if past else "numeric_within_tolerance"] += 1
        if accepted:
            result["accepted_over_tolerance" if past else "accepted_within_tolerance"] += 1
        gaps = result["max_gap_after_omitting_transport_from_fd_m_s2"]
        for name, gap in omission_gap.items():
            gaps[name] = max(gaps[name], gap)
        result["rows"].append(dict(**key, emitted=accepted, within_original_hard_tolerance=not past,
                                   over_tolerance_fields=past, violations=values, rim_edge_switches=switches,
                                   gap_after_omitting_transport_from_fd_m_s2=omission_gap, **wheel_data))
        result["rim_edge_switches"] += switches
        result["servo_receipt_difference_nm"] = max(result["servo_receipt_difference_nm"], servo_delta)
        if not result["numeric_rows"]:
            result["command_min"] = result["command_max"] = command.tolist()
        result["command_min"] = np.minimum(result["command_min"], command).tolist()
        result["command_max"] = np.maximum(result["command_max"], command).tolist()
        result["numeric_rows"] += 1
        result["accepted_rows" if solution["accepted"] else "rejected_numeric_rows"] += 1
    return result
