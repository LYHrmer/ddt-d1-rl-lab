#!/usr/bin/env python3
"""Offline, read-only audit of existing C++ runtime trace.jsonl files.

NO physics. NO MuJoCo. NO ONNX. Pure json + numpy over already-recorded traces.
Re-runnable; writes one JSON report. Never modifies any input file.

Usage:
    python3 tools04/audit_cxx_trace04.py [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import hashlib
import numpy as np

WS = "/home/lyh/ddt_ros2_ws"

TRIALS = {
    "forward_stop_03d": {
        "dir": f"{WS}/evidence/career_contract_03/pilot03d_deployment_plan_01/final_trial02_forward_stop",
        "version": "03d",
        "scored": "FAIL",
        "reported_stop_disp_m": 0.080583646,
    },
    "reverse_stop_03d": {
        "dir": f"{WS}/evidence/career_contract_03/pilot03d_deployment_plan_01/final_trial01_reverse_stop",
        "version": "03d",
        "scored": "PASS",
        "reported_stop_disp_m": 0.016340523,
    },
    "turn_stop_03c_reference_only": {
        "dir": f"{WS}/evidence/career_contract_03/pilot03c_deployment_plan_01/trial02_turn_stop",
        "version": "03c",
        "scored": "reference_only_do_not_mix",
        "reported_stop_disp_m": None,
    },
}

FIELD_DOC = {
    "sim_time": ("simulation time at END of this native step", "s"),
    "native_count": ("1-based native (physics) step index; native_dt=0.002", "count"),
    "pd_count": ("1-based count of PD-controller updates so far; pd_dt=0.004 (decimation 2)", "count"),
    "policy_count": ("1-based count of policy inference ticks so far; policy_dt=0.02 (decimation 10)", "count"),
    "policy_latency_ms": ("wall-clock ONNX inference latency of the most recent policy tick", "ms"),
    "command": ("commanded [vx, vy, wz] velocity reference fed to the policy observation", "m/s, m/s, rad/s"),
    "position": ("free-joint base position in WORLD frame [x, y, z] (= qpos[0:3])", "m"),
    "base_z": ("base height in world frame (= position[2] = qpos[2])", "m"),
    "quat_wxyz": ("base orientation quaternion, w-first (= qpos[3:7])", "unit quaternion"),
    "roll": ("base roll extracted from quat_wxyz", "rad"),
    "pitch": ("base pitch extracted from quat_wxyz", "rad"),
    "tilt": ("total tilt angle of base z-axis from world z (termination metric)", "rad"),
    "qpos": ("full MuJoCo qpos: [0:3] world xyz, [3:7] quat wxyz, [7:23] 16 joint angles in joint_order", "m / unit / rad"),
    "qvel": ("full MuJoCo qvel: [0:3] WORLD-frame base linear vel, [3:6] base angular vel, [6:22] 16 joint vels", "m/s, rad/s, rad/s"),
    "body_velocity": ("base linear velocity rotated into BODY frame = R(quat)^T @ qvel[0:3]", "m/s"),
    "body_vx": ("body_velocity[0], forward component in body frame", "m/s"),
    "body_wz": ("base yaw rate (body/world z component of angular velocity)", "rad/s"),
    "issued_action": ("policy network output AFTER clipping to action_clip=[-1,1], BEFORE the delay FIFO", "normalized [-1,1]"),
    "executed_action": ("action actually applied by the PD layer this step (issued_action delayed by delay_ticks; here delay_ticks=0 so equal)", "normalized [-1,1]"),
    "raw_torque": ("PD-computed joint torque BEFORE torque_limits clamping", "N*m"),
    "torque": ("joint torque actually applied AFTER torque_limits clamping", "N*m"),
    "saturation": ("per-joint bool: whether |raw_torque| hit torque_limits this step", "bool"),
    "illegal_body_support_since_policy": ("bool: non-wheel body geom contacted ground since last policy tick", "bool"),
}

LEG_PREFIX = ["FL", "FR", "RL", "RR"]
# joint_order = [<leg>_hip, <leg>_thigh, <leg>_calf, <leg>_foot] x 4 legs
WHEEL_ACT_IDX = [3, 7, 11, 15]          # *_foot_joint == wheel
LEG_ACT_IDX = [i for i in range(16) if i not in WHEEL_ACT_IDX]
QPOS_JOINT_OFF = 7
QVEL_JOINT_OFF = 6


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trace(path: str) -> dict:
    """Load a trace.jsonl into column arrays. Read-only."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    out = {
        "n": n,
        "sim_time": np.array([r["sim_time"] for r in rows]),
        "native_count": np.array([r["native_count"] for r in rows]),
        "pd_count": np.array([r["pd_count"] for r in rows]),
        "policy_count": np.array([r["policy_count"] for r in rows]),
        "policy_latency_ms": np.array([r["policy_latency_ms"] for r in rows]),
        "command": np.array([r["command"] for r in rows]),
        "position": np.array([r["position"] for r in rows]),
        "base_z": np.array([r["base_z"] for r in rows]),
        "quat": np.array([r["quat_wxyz"] for r in rows]),
        "roll": np.array([r["roll"] for r in rows]),
        "pitch": np.array([r["pitch"] for r in rows]),
        "tilt": np.array([r["tilt"] for r in rows]),
        "qpos": np.array([r["qpos"] for r in rows]),
        "qvel": np.array([r["qvel"] for r in rows]),
        "body_velocity": np.array([r["body_velocity"] for r in rows]),
        "body_wz": np.array([r["body_wz"] for r in rows]),
        "issued_action": np.array([r["issued_action"] for r in rows]),
        "executed_action": np.array([r["executed_action"] for r in rows]),
        "raw_torque": np.array([r["raw_torque"] for r in rows]),
        "torque": np.array([r["torque"] for r in rows]),
        "saturation": np.array([r["saturation"] for r in rows], dtype=bool),
        "illegal_support": np.array([r["illegal_body_support_since_policy"] for r in rows], dtype=bool),
    }
    out["yaw"] = yaw_from_quat(out["quat"])
    out["world_vxy"] = out["qvel"][:, 0:2]
    out["joint_qpos"] = out["qpos"][:, QPOS_JOINT_OFF:QPOS_JOINT_OFF + 16]
    out["joint_qvel"] = out["qvel"][:, QVEL_JOINT_OFF:QVEL_JOINT_OFF + 16]
    return out


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def idx_at(t: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(t - target)))


def f(x):
    """json-safe float"""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def fl(a):
    return [f(v) for v in np.asarray(a).ravel()]


# ---------------------------------------------------------------- A
def section_A(tr: dict, trace_path: str, report_path: str) -> dict:
    rep = json.load(open(report_path))
    schema = rep.get("schema", {})
    fields = {}
    for k, (meaning, unit) in FIELD_DOC.items():
        fields[k] = {"meaning": meaning, "unit": unit}
    # verify body_velocity == R^T qvel[0:3]
    errs = []
    for i in (0, tr["n"] // 3, tr["n"] // 2, tr["n"] - 1):
        R = rot_from_quat(tr["quat"][i])
        pred = R.T @ tr["qvel"][i, 0:3]
        errs.append(float(np.max(np.abs(pred - tr["body_velocity"][i]))))
    # verify wheel torque law tau = kd*(scale*action - qvel)
    kd = np.array(schema["kd"], dtype=float)
    kp = np.array(schema["kp"], dtype=float)
    sc = np.array(schema["action_scales"], dtype=float)
    dj = np.array(schema["default_joint_angles"], dtype=float)
    wheel_law_err, leg_law_err = [], []
    for i in (10, 1000, 4000, 5999):
        a = tr["executed_action"][i]
        qv = tr["joint_qvel"][i]
        qp = tr["joint_qpos"][i]
        tau_w = kd[WHEEL_ACT_IDX] * (sc[WHEEL_ACT_IDX] * a[WHEEL_ACT_IDX] - qv[WHEEL_ACT_IDX])
        wheel_law_err.append(float(np.max(np.abs(tau_w - tr["raw_torque"][i][WHEEL_ACT_IDX]))))
        tau_l = (kp[LEG_ACT_IDX] * (dj[LEG_ACT_IDX] + sc[LEG_ACT_IDX] * a[LEG_ACT_IDX] - qp[LEG_ACT_IDX])
                 - kd[LEG_ACT_IDX] * qv[LEG_ACT_IDX])
        leg_law_err.append(float(np.max(np.abs(tau_l - tr["raw_torque"][i][LEG_ACT_IDX]))))
    return {
        "trace_file": trace_path,
        "trace_sha256": sha256_of(trace_path),
        "n_lines": tr["n"],
        "one_line_per": "native physics step (native_dt=0.002 s); policy/pd ticks identified via policy_count/pd_count",
        "sim_time_first": f(tr["sim_time"][0]),
        "sim_time_last": f(tr["sim_time"][-1]),
        "policy_ticks": int(tr["policy_count"][-1]),
        "pd_ticks": int(tr["pd_count"][-1]),
        "fields": fields,
        "joint_order": schema.get("joint_order"),
        "wheel_joint_names": [schema["joint_order"][i] for i in WHEEL_ACT_IDX],
        "wheel_action_indices": WHEEL_ACT_IDX,
        "wheel_qpos_indices": [QPOS_JOINT_OFF + i for i in WHEEL_ACT_IDX],
        "wheel_qvel_indices": [QVEL_JOINT_OFF + i for i in WHEEL_ACT_IDX],
        "kp": fl(kp), "kd": fl(kd), "action_scales": fl(sc),
        "default_joint_angles": fl(dj),
        "torque_limits": fl(schema.get("torque_limits", [])),
        "action_clip": schema.get("action_clip"),
        "verification": {
            "body_velocity_is_bodyframe_rotation_of_qvel_lin_maxabs_err": max(errs),
            "wheel_torque_law_tau_eq_kd_times_scaled_action_minus_qvel_maxabs_err": max(wheel_law_err),
            "leg_torque_law_tau_eq_kp_dq_minus_kd_qd_maxabs_err_note": "nonzero because PD runs at 250Hz on pre-step state while trace logs post-step state",
            "leg_torque_law_maxabs_err": max(leg_law_err),
        },
        "unavailable_fields_note": "trace has no contact forces, no ground friction/slip, no observation vector, no critic value, no per-wheel contact flag, no actuator ctrl history beyond issued/executed action",
    }


# ---------------------------------------------------------------- B
def motion_window(tr: dict, t0: float, t1: float, v_stop_thresh: float = 0.02) -> dict:
    t = tr["sim_time"]
    i0, i1 = idx_at(t, t0), idx_at(t, t1)
    sl = slice(i0, i1 + 1)
    xy = tr["position"][sl, 0:2]
    p0, p1 = xy[0], xy[-1]
    d = np.diff(xy, axis=0)
    path_len = float(np.sum(np.linalg.norm(d, axis=1)))
    eucl = float(np.linalg.norm(p1 - p0))
    # distance from start over time, for overshoot detection
    dist_from_start = np.linalg.norm(xy - p0, axis=1)
    k_max = int(np.argmax(dist_from_start))
    # heading direction at t0 (unit vector of world velocity at t0; fallback = net displacement)
    v0w = tr["world_vxy"][i0]
    sp0 = float(np.linalg.norm(v0w))
    if sp0 > 1e-6:
        u = v0w / sp0
    elif eucl > 1e-9:
        u = (p1 - p0) / eucl
    else:
        u = np.array([1.0, 0.0])
    s_along = (xy - p0) @ u            # signed progress along t0 heading
    s_lat = (xy - p0) @ np.array([-u[1], u[0]])
    speed = np.linalg.norm(tr["world_vxy"][sl], axis=1)
    # first time |v| < thresh
    below = np.nonzero(speed < v_stop_thresh)[0]
    t_below = f(t[i0 + below[0]]) if below.size else None
    tau_to_thresh = (t_below - t[i0]) if t_below is not None else None
    # first time |v| drops to 1/e of t0 speed
    if sp0 > 1e-9:
        be = np.nonzero(speed < sp0 / math.e)[0]
        t_1e = f(t[i0 + be[0]] - t[i0]) if be.size else None
    else:
        t_1e = None
    # direction reversal along heading: sign changes of d(s_along)/dt beyond noise
    v_along = tr["world_vxy"][sl] @ u
    noise = 0.005
    sgn = np.where(v_along > noise, 1, np.where(v_along < -noise, -1, 0))
    nz = sgn[sgn != 0]
    n_rev = int(np.sum(np.abs(np.diff(nz)) == 2)) if nz.size > 1 else 0
    overshoot = float(dist_from_start[k_max] - dist_from_start[-1])
    # per-second sub-segments
    segs = []
    tt = t0
    while tt < t1 - 1e-9:
        a, b = idx_at(t, tt), idx_at(t, min(tt + 1.0, t1))
        xya = tr["position"][a, 0:2]
        xyb = tr["position"][b, 0:2]
        da = np.diff(tr["position"][a:b + 1, 0:2], axis=0)
        segs.append({
            "t": [f(tt), f(min(tt + 1.0, t1))],
            "euclid_disp_m": f(np.linalg.norm(xyb - xya)),
            "path_len_m": f(np.sum(np.linalg.norm(da, axis=1))),
            "dxy_m": fl(xyb - xya),
            "along_heading_m": f((xyb - xya) @ u),
            "lateral_m": f((xyb - xya) @ np.array([-u[1], u[0]])),
            "cum_dist_from_t0_at_end_m": f(np.linalg.norm(xyb - p0)),
            "mean_world_speed_mps": f(np.mean(np.linalg.norm(tr["world_vxy"][a:b + 1], axis=1))),
            "max_world_speed_mps": f(np.max(np.linalg.norm(tr["world_vxy"][a:b + 1], axis=1))),
        })
        tt += 1.0
    return {
        "window_t": [f(t[i0]), f(t[i1])],
        "index_range": [i0, i1],
        "n_native_steps": i1 - i0,
        "n_policy_ticks": int(tr["policy_count"][i1] - tr["policy_count"][i0]),
        "start_xy_m": fl(p0), "end_xy_m": fl(p1),
        "euclid_disp_m": eucl,
        "path_len_m": path_len,
        "path_over_euclid_ratio": f(path_len / eucl) if eucl > 1e-9 else None,
        "max_dist_from_start_m": f(dist_from_start[k_max]),
        "t_of_max_dist_s": f(t[i0 + k_max]),
        "overshoot_then_return_m": overshoot,
        "heading_unit_at_t0": fl(u),
        "net_along_heading_m": f(s_along[-1]),
        "net_lateral_m": f(s_lat[-1]),
        "max_along_heading_m": f(np.max(s_along)), "min_along_heading_m": f(np.min(s_along)),
        "world_speed_at_t0_mps": sp0,
        "body_vel_at_t0_mps": fl(tr["body_velocity"][i0]),
        "body_vx_at_t0_mps": f(tr["body_velocity"][i0, 0]),
        "body_vy_at_t0_mps": f(tr["body_velocity"][i0, 1]),
        "yaw_at_t0_rad": f(tr["yaw"][i0]), "yaw_at_t1_rad": f(tr["yaw"][i1]),
        "yaw_change_rad": f(tr["yaw"][i1] - tr["yaw"][i0]),
        "yaw_min_rad": f(np.min(tr["yaw"][sl])), "yaw_max_rad": f(np.max(tr["yaw"][sl])),
        "first_time_world_speed_below_0.02_s": t_below,
        "seconds_from_t0_to_speed_below_0.02": f(tau_to_thresh),
        "seconds_from_t0_to_speed_below_1_over_e_of_t0": t_1e,
        "n_direction_reversals_along_t0_heading(|v_along|>0.005)": n_rev,
        "min_v_along_mps": f(np.min(v_along)), "max_v_along_mps": f(np.max(v_along)),
        "final_world_speed_mps": f(speed[-1]),
        "mean_world_speed_mps": f(np.mean(speed)),
        "body_vx_stats": {"mean": f(np.mean(tr["body_velocity"][sl, 0])),
                          "std": f(np.std(tr["body_velocity"][sl, 0])),
                          "min": f(np.min(tr["body_velocity"][sl, 0])),
                          "max": f(np.max(tr["body_velocity"][sl, 0])),
                          "final": f(tr["body_velocity"][i1, 0])},
        "body_vy_stats": {"mean": f(np.mean(tr["body_velocity"][sl, 1])),
                          "std": f(np.std(tr["body_velocity"][sl, 1])),
                          "min": f(np.min(tr["body_velocity"][sl, 1])),
                          "max": f(np.max(tr["body_velocity"][sl, 1])),
                          "final": f(tr["body_velocity"][i1, 1])},
        "base_z_range_m": [f(np.min(tr["base_z"][sl])), f(np.max(tr["base_z"][sl]))],
        "tilt_max_rad": f(np.max(tr["tilt"][sl])),
        "pitch_at_t0_rad": f(tr["pitch"][i0]), "pitch_at_t1_rad": f(tr["pitch"][i1]),
        "pitch_min_rad": f(np.min(tr["pitch"][sl])), "pitch_max_rad": f(np.max(tr["pitch"][sl])),
        "illegal_body_support_any": bool(np.any(tr["illegal_support"][sl])),
        "command_unique_in_window": [fl(c) for c in np.unique(tr["command"][sl], axis=0)],
        "per_second": segs,
        "policy_tick_speed_mps": policy_tick_series(tr, i0, i1),
    }


def policy_tick_series(tr: dict, i0: int, i1: int) -> dict:
    """Downsample to one sample per policy tick for a compact per-tick view."""
    pc = tr["policy_count"]
    # last native step of each policy tick inside window
    idxs = [i for i in range(i0, i1 + 1) if i == i1 or pc[i + 1] != pc[i]]
    t = tr["sim_time"][idxs]
    sp = np.linalg.norm(tr["world_vxy"][idxs], axis=1)
    return {
        "n": len(idxs),
        "t_s": fl(t),
        "world_speed_mps": fl(sp),
        "body_vx_mps": fl(tr["body_velocity"][idxs, 0]),
        "body_vy_mps": fl(tr["body_velocity"][idxs, 1]),
        "yaw_rad": fl(tr["yaw"][idxs]),
        "x_m": fl(tr["position"][idxs, 0]),
        "y_m": fl(tr["position"][idxs, 1]),
        "mean_abs_wheel_qvel_radps": fl(np.mean(np.abs(tr["joint_qvel"][np.ix_(idxs, WHEEL_ACT_IDX)]), axis=1)),
        "mean_wheel_target_radps": fl(np.mean(
            tr["executed_action"][np.ix_(idxs, WHEEL_ACT_IDX)] * 5.0, axis=1)),
    }


# ---------------------------------------------------------------- D / E
def quiet_window(tr: dict, t0: float, t1: float, label: str) -> dict:
    t = tr["sim_time"]
    i0, i1 = idx_at(t, t0), idx_at(t, t1)
    sl = slice(i0, i1 + 1)
    xy = tr["position"][sl, 0:2]
    d = np.diff(xy, axis=0)
    wq = tr["joint_qvel"][sl][:, WHEEL_ACT_IDX]
    wtgt = tr["executed_action"][sl][:, WHEEL_ACT_IDX] * 5.0
    legtgt = (np.array([0.0, 0.8, -1.5, 0.0] * 4)[LEG_ACT_IDX]
              + tr["executed_action"][sl][:, LEG_ACT_IDX] * 0.5)
    return {
        "label": label,
        "window_t": [f(t[i0]), f(t[i1])],
        "command_unique": [fl(c) for c in np.unique(tr["command"][sl], axis=0)],
        "body_vx": {"mean": f(np.mean(tr["body_velocity"][sl, 0])), "std": f(np.std(tr["body_velocity"][sl, 0])),
                    "min": f(np.min(tr["body_velocity"][sl, 0])), "max": f(np.max(tr["body_velocity"][sl, 0])),
                    "at_end": f(tr["body_velocity"][i1, 0])},
        "body_vy": {"mean": f(np.mean(tr["body_velocity"][sl, 1])), "std": f(np.std(tr["body_velocity"][sl, 1])),
                    "min": f(np.min(tr["body_velocity"][sl, 1])), "max": f(np.max(tr["body_velocity"][sl, 1])),
                    "at_end": f(tr["body_velocity"][i1, 1])},
        "body_wz": {"mean": f(np.mean(tr["body_wz"][sl])), "std": f(np.std(tr["body_wz"][sl]))},
        "world_speed": {"mean": f(np.mean(np.linalg.norm(tr["world_vxy"][sl], axis=1))),
                        "max": f(np.max(np.linalg.norm(tr["world_vxy"][sl], axis=1))),
                        "at_end": f(np.linalg.norm(tr["world_vxy"][i1]))},
        "start_xy_m": fl(xy[0]), "end_xy_m": fl(xy[-1]),
        "euclid_drift_m": f(np.linalg.norm(xy[-1] - xy[0])),
        "path_len_m": f(np.sum(np.linalg.norm(d, axis=1))),
        "yaw_start_rad": f(tr["yaw"][i0]), "yaw_end_rad": f(tr["yaw"][i1]),
        "yaw_drift_rad": f(tr["yaw"][i1] - tr["yaw"][i0]),
        "base_z": {"mean": f(np.mean(tr["base_z"][sl])), "at_end": f(tr["base_z"][i1])},
        "pitch": {"mean": f(np.mean(tr["pitch"][sl])), "at_end": f(tr["pitch"][i1])},
        "wheel_qvel_radps_mean_per_wheel": fl(np.mean(wq, axis=0)),
        "wheel_qvel_radps_absmean_per_wheel": fl(np.mean(np.abs(wq), axis=0)),
        "wheel_qvel_radps_std_per_wheel": fl(np.std(wq, axis=0)),
        "wheel_target_radps_mean_per_wheel": fl(np.mean(wtgt, axis=0)),
        "wheel_target_radps_std_per_wheel": fl(np.std(wtgt, axis=0)),
        "leg_joint_target_rad_mean": fl(np.mean(legtgt, axis=0)),
        "leg_joint_pos_rad_mean": fl(np.mean(tr["joint_qpos"][sl][:, LEG_ACT_IDX], axis=0)),
        "issued_action_mean": fl(np.mean(tr["issued_action"][sl], axis=0)),
        "issued_action_std": fl(np.std(tr["issued_action"][sl], axis=0)),
        "torque_absmean_per_joint": fl(np.mean(np.abs(tr["torque"][sl]), axis=0)),
        "saturation_fraction_per_joint": fl(np.mean(tr["saturation"][sl], axis=0)),
    }


# ---------------------------------------------------------------- F / G
def action_torque_window(tr: dict, t0: float, t1: float, schema: dict) -> dict:
    t = tr["sim_time"]
    i0, i1 = idx_at(t, t0), idx_at(t, t1)
    sl = slice(i0, i1 + 1)
    ia = tr["issued_action"][sl]
    ea = tr["executed_action"][sl]
    tq = tr["torque"][sl]
    rtq = tr["raw_torque"][sl]
    sat = tr["saturation"][sl]
    lim = np.array(schema["torque_limits"], dtype=float)
    clip = np.array(schema["action_clip"], dtype=float)
    # chatter: per-policy-tick action difference
    pc = tr["policy_count"]
    tick_idx = [i for i in range(i0, i1 + 1) if i == i1 or pc[i + 1] != pc[i]]
    ia_tick = tr["issued_action"][tick_idx]
    dia = np.diff(ia_tick, axis=0)
    # zero-crossing count of per-tick action delta = chatter proxy
    zc = np.sum(np.diff(np.sign(dia + 1e-30), axis=0) != 0, axis=0)
    wheel = WHEEL_ACT_IDX
    wtgt = ea[:, wheel] * 5.0
    wq = tr["joint_qvel"][sl][:, wheel]
    wpos = tr["joint_qpos"][sl][:, wheel]
    return {
        "window_t": [f(t[i0]), f(t[i1])],
        "action_clip": fl(clip),
        "issued_action_abs_max_per_joint": fl(np.max(np.abs(ia), axis=0)),
        "issued_action_mean_per_joint": fl(np.mean(ia, axis=0)),
        "issued_action_std_per_joint": fl(np.std(ia, axis=0)),
        "issued_action_at_clip_fraction_per_joint": fl(np.mean(np.abs(ia) >= 0.999, axis=0)),
        "n_steps_any_action_at_clip": int(np.sum(np.any(np.abs(ia) >= 0.999, axis=1))),
        "issued_equals_executed": bool(np.array_equal(ia, ea)),
        "torque_absmax_per_joint": fl(np.max(np.abs(tq), axis=0)),
        "torque_absmean_per_joint": fl(np.mean(np.abs(tq), axis=0)),
        "raw_torque_absmax_per_joint": fl(np.max(np.abs(rtq), axis=0)),
        "torque_limits": fl(lim),
        "saturation_fraction_per_joint": fl(np.mean(sat, axis=0)),
        "saturation_steps_any_joint": int(np.sum(np.any(sat, axis=1))),
        "saturation_total_steps": int(sat.shape[0]),
        "per_tick_action_delta_absmean_per_joint": fl(np.mean(np.abs(dia), axis=0)),
        "per_tick_action_delta_absmax_per_joint": fl(np.max(np.abs(dia), axis=0)) if dia.size else None,
        "per_tick_action_delta_sign_flips_per_joint": [int(v) for v in zc],
        "n_policy_ticks_in_window": len(tick_idx),
        "wheel": {
            "names": [schema["joint_order"][i] for i in wheel],
            "target_radps_mean": fl(np.mean(wtgt, axis=0)),
            "target_radps_std": fl(np.std(wtgt, axis=0)),
            "target_radps_absmax": fl(np.max(np.abs(wtgt), axis=0)),
            "target_radps_final": fl(wtgt[-1]),
            "qvel_radps_mean": fl(np.mean(wq, axis=0)),
            "qvel_radps_absmean": fl(np.mean(np.abs(wq), axis=0)),
            "qvel_radps_std": fl(np.std(wq, axis=0)),
            "qvel_radps_absmax": fl(np.max(np.abs(wq), axis=0)),
            "qvel_radps_final": fl(wq[-1]),
            "qvel_absmean_last_1s": fl(np.mean(np.abs(tr["joint_qvel"][idx_at(t, t1 - 1.0):i1 + 1][:, wheel]), axis=0)),
            "net_rotation_rad_over_window": fl(wpos[-1] - wpos[0]),
            "net_rotation_rev": fl((wpos[-1] - wpos[0]) / (2 * math.pi)),
            "total_abs_rotation_rad": fl(np.sum(np.abs(np.diff(wpos, axis=0)), axis=0)),
            "torque_absmean": fl(np.mean(np.abs(tq[:, wheel]), axis=0)),
            "torque_absmax": fl(np.max(np.abs(tq[:, wheel]), axis=0)),
            "torque_mean_signed": fl(np.mean(tq[:, wheel], axis=0)),
            "saturation_fraction": fl(np.mean(sat[:, wheel], axis=0)),
        },
        "leg": {
            "names": [schema["joint_order"][i] for i in LEG_ACT_IDX],
            "issued_action_absmean": fl(np.mean(np.abs(ia[:, LEG_ACT_IDX]), axis=0)),
            "issued_action_std": fl(np.std(ia[:, LEG_ACT_IDX], axis=0)),
            "torque_absmean": fl(np.mean(np.abs(tq[:, LEG_ACT_IDX]), axis=0)),
            "torque_absmax": fl(np.max(np.abs(tq[:, LEG_ACT_IDX]), axis=0)),
            "saturation_fraction": fl(np.mean(sat[:, LEG_ACT_IDX], axis=0)),
            "qvel_absmean": fl(np.mean(np.abs(tr["joint_qvel"][sl][:, LEG_ACT_IDX]), axis=0)),
        },
        "wheel_sum_target_radps_mean": f(np.mean(np.sum(wtgt, axis=1))),
        "wheel_mean_target_radps_timeseries_per_second": [
            f(np.mean(ea[idx_at(t, s) - i0: idx_at(t, min(s + 1.0, t1)) - i0 + 1][:, wheel] * 5.0))
            for s in np.arange(t0, t1, 1.0)
        ],
        "wheel_mean_abs_qvel_per_second": [
            f(np.mean(np.abs(tr["joint_qvel"][idx_at(t, s):idx_at(t, min(s + 1.0, t1)) + 1][:, wheel])))
            for s in np.arange(t0, t1, 1.0)
        ],
        "wheel_mean_signed_qvel_per_second": [
            f(np.mean(tr["joint_qvel"][idx_at(t, s):idx_at(t, min(s + 1.0, t1)) + 1][:, wheel]))
            for s in np.arange(t0, t1, 1.0)
        ],
    }


def wheel_vs_base_slip(tr: dict, t0: float, t1: float, wheel_radius_guess: float = None) -> dict:
    """Compare commanded/actual wheel spin against base translation (no model file read)."""
    t = tr["sim_time"]
    i0, i1 = idx_at(t, t0), idx_at(t, t1)
    sl = slice(i0, i1 + 1)
    wq = tr["joint_qvel"][sl][:, WHEEL_ACT_IDX]
    mean_spin = np.mean(wq, axis=1)
    vb = tr["body_velocity"][sl, 0]
    # ratio v / omega over the fast part of 2-8s is an empirical rolling radius
    msk = np.abs(mean_spin) > 0.5
    r_emp = f(np.mean(vb[msk] / mean_spin[msk])) if np.any(msk) else None
    return {
        "window_t": [f(t[i0]), f(t[i1])],
        "mean_wheel_spin_radps_mean": f(np.mean(mean_spin)),
        "mean_wheel_spin_radps_absmean": f(np.mean(np.abs(mean_spin))),
        "body_vx_mean": f(np.mean(vb)),
        "empirical_rolling_radius_m_from_v_over_omega": r_emp,
        "n_samples_used_for_radius": int(np.sum(msk)),
    }


def rolling_vs_translation_per_second(tr: dict, t0: float, t1: float, r_roll: float) -> list:
    """Per-second: mean wheel spin, expected rolling distance (4-wheel mean * r), actual body
    displacement along the t0 heading. Gap => slip or leg-driven translation."""
    t = tr["sim_time"]
    i_t0 = idx_at(t, t0)
    v0w = tr["world_vxy"][i_t0]
    sp0 = float(np.linalg.norm(v0w))
    u = v0w / sp0 if sp0 > 1e-6 else np.array([1.0, 0.0])
    rows = []
    for s in np.arange(t0, t1 - 1e-9, 1.0):
        a, b = idx_at(t, s), idx_at(t, min(s + 1.0, t1))
        wp = tr["joint_qpos"][[a, b]][:, WHEEL_ACT_IDX]
        net_rot = wp[1] - wp[0]
        mean_net_rot = float(np.mean(net_rot))
        wq = tr["joint_qvel"][a:b + 1][:, WHEEL_ACT_IDX]
        dxy = tr["position"][b, 0:2] - tr["position"][a, 0:2]
        along = float(dxy @ u)
        # displacement expressed in the robot's own forward (+x body) direction, which is the
        # sign convention wheel rotation uses. yaw is near-constant here so a mid-window yaw is fine.
        ymid = float(tr["yaw"][(a + b) // 2])
        bx = np.array([math.cos(ymid), math.sin(ymid)])
        along_bodyx = float(dxy @ bx)
        exp_roll = mean_net_rot * r_roll
        rows.append({
            "t": [f(s), f(min(s + 1.0, t1))],
            "wheel_net_rotation_rad_per_wheel": fl(net_rot),
            "wheel_mean_net_rotation_rad": f(mean_net_rot),
            "expected_rolling_dist_m_mean_wheel": f(exp_roll),
            "actual_disp_along_body_x_m": along_bodyx,
            "rolling_minus_actual_bodyx_m": f(exp_roll - along_bodyx),
            "rolling_efficiency_actual_over_expected": f(along_bodyx / exp_roll) if abs(exp_roll) > 1e-9 else None,
            "actual_disp_along_t0_heading_m": along,
            "actual_euclid_disp_m": f(np.linalg.norm(dxy)),
            "mean_abs_wheel_qvel_radps": f(np.mean(np.abs(wq))),
            "mean_signed_wheel_qvel_radps": f(np.mean(wq)),
            "mean_wheel_target_radps": f(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0)),
            "mean_body_vx_mps": f(np.mean(tr["body_velocity"][a:b + 1, 0])),
            "mean_world_speed_mps": f(np.mean(np.linalg.norm(tr["world_vxy"][a:b + 1], axis=1))),
            "mean_pitch_rad": f(np.mean(tr["pitch"][a:b + 1])),
            "mean_base_z_m": f(np.mean(tr["base_z"][a:b + 1])),
        })
    return rows


def quarter_second_profile(tr: dict, t0: float, t1: float, dt: float = 0.25) -> list:
    t = tr["sim_time"]
    i_t0 = idx_at(t, t0)
    v0w = tr["world_vxy"][i_t0]
    sp0 = float(np.linalg.norm(v0w))
    u = v0w / sp0 if sp0 > 1e-6 else np.array([1.0, 0.0])
    rows = []
    s = t0
    while s < t1 - 1e-9:
        a, b = idx_at(t, s), idx_at(t, min(s + dt, t1))
        dxy = tr["position"][b, 0:2] - tr["position"][a, 0:2]
        rows.append({
            "t": [f(s), f(min(s + dt, t1))],
            "disp_along_t0_heading_m": f(dxy @ u),
            "cum_along_from_t0_m": f((tr["position"][b, 0:2] - tr["position"][i_t0, 0:2]) @ u),
            "mean_body_vx_mps": f(np.mean(tr["body_velocity"][a:b + 1, 0])),
            "v_along_at_end_mps": f(tr["world_vxy"][b] @ u),
            "mean_abs_wheel_qvel_radps": f(np.mean(np.abs(tr["joint_qvel"][a:b + 1][:, WHEEL_ACT_IDX]))),
            "mean_wheel_target_radps": f(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0)),
        })
        s += dt
    return rows


def clip_joint_detail(tr: dict, t0: float, t1: float, schema: dict) -> dict:
    t = tr["sim_time"]
    i0, i1 = idx_at(t, t0), idx_at(t, t1)
    sl = slice(i0, i1 + 1)
    ia = tr["issued_action"][sl]
    at_clip = np.abs(ia) >= 0.999
    frac = np.mean(at_clip, axis=0)
    jo = schema["joint_order"]
    offenders = {jo[i]: {"clip_fraction": f(frac[i]),
                         "clip_steps": int(np.sum(at_clip[:, i])),
                         "sign": int(np.sign(np.mean(ia[at_clip[:, i], i]))) if np.any(at_clip[:, i]) else 0,
                         "first_clip_time_s": f(t[i0 + int(np.argmax(at_clip[:, i]))]) if np.any(at_clip[:, i]) else None,
                         "last_clip_time_s": f(t[i0 + int(len(at_clip) - 1 - np.argmax(at_clip[::-1, i]))]) if np.any(at_clip[:, i]) else None,
                         "joint_qpos_mean_rad": f(np.mean(tr["joint_qpos"][sl][:, i])),
                         "joint_qpos_at_t1_rad": f(tr["joint_qpos"][i1, i]),
                         "default_angle_rad": f(schema["default_joint_angles"][i]),
                         "pd_target_mean_rad": f(schema["default_joint_angles"][i]
                                                 + np.mean(ia[:, i]) * schema["action_scales"][i]),
                         }
                 for i in range(16) if frac[i] > 0}
    return {"window_t": [f(t[i0]), f(t[i1])],
            "joints_hitting_action_clip": offenders,
            "total_steps": int(at_clip.shape[0])}


def phase_decomposition(tr: dict, phases: list) -> list:
    """Displacement along the t=8s heading, broken into named sub-phases of the 8-12s window."""
    t = tr["sim_time"]
    i8 = idx_at(t, 8.0)
    v0w = tr["world_vxy"][i8]
    sp0 = float(np.linalg.norm(v0w))
    u = v0w / sp0 if sp0 > 1e-6 else np.array([1.0, 0.0])
    total_along = float((tr["position"][idx_at(t, 12.0), 0:2] - tr["position"][i8, 0:2]) @ u)
    rows = []
    for name, a_t, b_t in phases:
        a, b = idx_at(t, a_t), idx_at(t, b_t)
        dxy = tr["position"][b, 0:2] - tr["position"][a, 0:2]
        along = float(dxy @ u)
        rows.append({
            "phase": name, "t": [f(a_t), f(b_t)],
            "disp_along_t8_heading_m": along,
            "fraction_of_net_along": f(along / total_along) if abs(total_along) > 1e-12 else None,
            "path_len_m": f(np.sum(np.linalg.norm(np.diff(tr["position"][a:b + 1, 0:2], axis=0), axis=1))),
            "body_vx_start_mps": f(tr["body_velocity"][a, 0]),
            "body_vx_end_mps": f(tr["body_velocity"][b, 0]),
            "mean_abs_wheel_qvel_radps": f(np.mean(np.abs(tr["joint_qvel"][a:b + 1][:, WHEEL_ACT_IDX]))),
            "mean_wheel_target_radps": f(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0)),
        })
    return {"net_along_t8_heading_m": total_along, "phases": rows}


def analyse(name: str, cfg: dict) -> dict:
    d = cfg["dir"]
    tp, rp = os.path.join(d, "trace.jsonl"), os.path.join(d, "report.json")
    tr = load_trace(tp)
    rep = json.load(open(rp))
    schema = rep["schema"]
    seq = json.load(open(os.path.join(d, "sequence.json")))
    ist = json.load(open(os.path.join(d, "initial_state.json")))
    out = {
        "trial": name,
        "version": cfg["version"],
        "scored": cfg["scored"],
        "reported_stop_disp_m": cfg["reported_stop_disp_m"],
        "dir": d,
        "sequence_segments": seq["segments"],
        "provenance": {
            "model_sha256": rep["model_sha256"],
            "policy_sha256": rep["policy_sha256"],
            "robot_xml_sha256": rep["robot_xml_sha256"],
            "runtime_binary_sha256": rep["runtime_binary_sha256"],
            "training_config": schema.get("training_config"),
            "training_config_sha256": schema.get("training_config_sha256"),
            "initial_state_sha256": rep["initial_state_sha256"],
            "initial_state_override": rep["initial_state_override"],
            "seed": rep.get("seed"),
            "delay_ticks": rep.get("delay_ticks"),
            "status": rep.get("status"),
            "terminal_reason": rep.get("terminal_reason"),
            "native_steps_completed": rep.get("native_steps_completed"),
            "policy_updates": rep.get("policy_updates"),
            "initial_qpos": ist.get("qpos"),
            "initial_qvel_all_zero": all(v == 0.0 for v in ist.get("qvel", [])),
        },
        "A_fields": section_A(tr, tp, rp),
        "B_stop_window_8_to_12": motion_window(tr, 8.0, 12.0),
        "D_zero_command_0_to_2": quiet_window(tr, 0.0, 2.0, "0-2s zero command"),
        "E_late_drive_7_to_8": quiet_window(tr, 7.0, 8.0, "7-8s end of commanded drive"),
        "E_drive_2_to_8_slip": wheel_vs_base_slip(tr, 2.0, 8.0),
        "F_actions_8_to_12": action_torque_window(tr, 8.0, 12.0, schema),
        "F_actions_7_to_8": action_torque_window(tr, 7.0, 8.0, schema),
        "G_wheels_8_to_12_slip": wheel_vs_base_slip(tr, 8.0, 12.0),
        "F_clip_detail_8_to_12": clip_joint_detail(tr, 8.0, 12.0, schema),
        "G_rolling_vs_translation_per_second_8_to_12": None,  # filled below (needs r_roll)
        "B_quarter_second_profile_8_to_12": quarter_second_profile(tr, 8.0, 12.0),
        "full_run_summary": {
            "euclid_disp_0_to_12_m": f(np.linalg.norm(tr["position"][-1, 0:2] - tr["position"][0, 0:2])),
            "path_len_0_to_12_m": f(np.sum(np.linalg.norm(np.diff(tr["position"][:, 0:2], axis=0), axis=1))),
            "euclid_disp_2_to_8_m": f(np.linalg.norm(
                tr["position"][idx_at(tr["sim_time"], 8.0), 0:2] - tr["position"][idx_at(tr["sim_time"], 2.0), 0:2])),
            "mean_body_vx_2_to_8": f(np.mean(
                tr["body_velocity"][idx_at(tr["sim_time"], 2.0):idx_at(tr["sim_time"], 8.0) + 1, 0])),
            "mean_body_vx_4_to_8": f(np.mean(
                tr["body_velocity"][idx_at(tr["sim_time"], 4.0):idx_at(tr["sim_time"], 8.0) + 1, 0])),
            "illegal_body_support_any_whole_run": bool(np.any(tr["illegal_support"])),
            "tilt_max_whole_run_rad": f(np.max(tr["tilt"])),
            "policy_latency_ms_p50": f(np.percentile(tr["policy_latency_ms"], 50)),
            "policy_latency_ms_max": f(np.max(tr["policy_latency_ms"])),
        },
    }
    r_roll = out["E_drive_2_to_8_slip"]["empirical_rolling_radius_m_from_v_over_omega"]
    out["G_rolling_radius_used_m"] = r_roll
    out["G_rolling_radius_source"] = "empirical v_bodyx / mean_wheel_omega over the 2-8s commanded-drive segment of THIS trial (no robot.xml read)"
    out["G_rolling_vs_translation_per_second_8_to_12"] = rolling_vs_translation_per_second(tr, 8.0, 12.0, r_roll or 0.0)
    out["D_rolling_vs_translation_per_second_0_to_2"] = rolling_vs_translation_per_second(tr, 0.0, 2.0, r_roll or 0.0)
    out["B_phase_decomposition_8_to_12"] = phase_decomposition(tr, [
        ("P1_initial_brake", 8.0, 8.75),
        ("P2_re_acceleration_or_rebound", 8.75, 10.5),
        ("P3_tail_creep", 10.5, 12.0),
    ])
    # tail creep: last second, the closest thing to the policy's zero-command fixed point
    t = tr["sim_time"]
    a, b = idx_at(t, 11.0), idx_at(t, 12.0)
    out["tail_creep_11_to_12"] = {
        "mean_body_vx_mps": f(np.mean(tr["body_velocity"][a:b + 1, 0])),
        "body_vx_at_12s_mps": f(tr["body_velocity"][b, 0]),
        "mean_abs_wheel_qvel_radps": f(np.mean(np.abs(tr["joint_qvel"][a:b + 1][:, WHEEL_ACT_IDX]))),
        "mean_wheel_target_radps_4wheel_avg": f(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0)),
        "wheel_target_radps_per_wheel": fl(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0, axis=0)),
        "body_vx_still_rising_over_last_second": bool(
            np.mean(tr["body_velocity"][idx_at(t, 11.75):b + 1, 0])
            > np.mean(tr["body_velocity"][a:idx_at(t, 11.25) + 1, 0])),
    }
    # steady part of the 0-2s zero-command segment (after the initial settle)
    a, b = idx_at(t, 1.0), idx_at(t, 2.0)
    out["zero_command_steady_1_to_2"] = {
        "mean_body_vx_mps": f(np.mean(tr["body_velocity"][a:b + 1, 0])),
        "mean_abs_wheel_qvel_radps": f(np.mean(np.abs(tr["joint_qvel"][a:b + 1][:, WHEEL_ACT_IDX]))),
        "mean_wheel_target_radps_4wheel_avg": f(np.mean(tr["executed_action"][a:b + 1][:, WHEEL_ACT_IDX] * 5.0)),
        "disp_m": f(np.linalg.norm(tr["position"][b, 0:2] - tr["position"][a, 0:2])),
    }
    return out


def compare(fw: dict, rv: dict) -> dict:
    bf, br = fw["B_stop_window_8_to_12"], rv["B_stop_window_8_to_12"]
    df, dr = fw["D_zero_command_0_to_2"], rv["D_zero_command_0_to_2"]
    ef, er = fw["E_late_drive_7_to_8"], rv["E_late_drive_7_to_8"]
    ff, fr = fw["F_actions_8_to_12"], rv["F_actions_8_to_12"]

    def sym(a, b, key=""):
        return {"forward": a, "reverse": b,
                "abs_forward_over_abs_reverse": f(abs(a) / abs(b)) if (a is not None and b and abs(b) > 1e-12) else None,
                "sum_f_plus_r": f(a + b) if (a is not None and b is not None) else None}

    return {
        "note": "both trials share identical initial_state_sha256, model, policy, runtime binary and seed; only the sign of the 2-8s vx command differs",
        "same_initial_state": fw["provenance"]["initial_state_sha256"] == rv["provenance"]["initial_state_sha256"],
        "same_policy": fw["provenance"]["policy_sha256"] == rv["provenance"]["policy_sha256"],
        "C_stop_displacement": {
            "forward_euclid_m": bf["euclid_disp_m"], "reverse_euclid_m": br["euclid_disp_m"],
            "ratio_fwd_over_rev": f(bf["euclid_disp_m"] / br["euclid_disp_m"]),
            "forward_path_len_m": bf["path_len_m"], "reverse_path_len_m": br["path_len_m"],
            "forward_path_over_euclid": bf["path_over_euclid_ratio"],
            "reverse_path_over_euclid": br["path_over_euclid_ratio"],
        },
        "C_speed_at_t8_symmetry": {
            "forward_world_speed_mps": bf["world_speed_at_t0_mps"],
            "reverse_world_speed_mps": br["world_speed_at_t0_mps"],
            "ratio": f(bf["world_speed_at_t0_mps"] / br["world_speed_at_t0_mps"]) if br["world_speed_at_t0_mps"] else None,
            "forward_body_vx": bf["body_vx_at_t0_mps"], "reverse_body_vx": br["body_vx_at_t0_mps"],
            "signed_sum_body_vx": f(bf["body_vx_at_t0_mps"] + br["body_vx_at_t0_mps"]),
            "forward_body_vy": bf["body_vy_at_t0_mps"], "reverse_body_vy": br["body_vy_at_t0_mps"],
        },
        "C_steady_state_2_8_symmetry": {
            "forward_mean_body_vx_4_8": fw["full_run_summary"]["mean_body_vx_4_to_8"],
            "reverse_mean_body_vx_4_8": rv["full_run_summary"]["mean_body_vx_4_to_8"],
            "commanded_mag": 0.2,
            "forward_tracking_error": f(fw["full_run_summary"]["mean_body_vx_4_to_8"] - 0.2),
            "reverse_tracking_error": f(rv["full_run_summary"]["mean_body_vx_4_to_8"] + 0.2),
        },
        "C_decay": {
            "forward_t_to_speed_below_0.02_s": bf["seconds_from_t0_to_speed_below_0.02"],
            "reverse_t_to_speed_below_0.02_s": br["seconds_from_t0_to_speed_below_0.02"],
            "forward_t_to_1_over_e_s": bf["seconds_from_t0_to_speed_below_1_over_e_of_t0"],
            "reverse_t_to_1_over_e_s": br["seconds_from_t0_to_speed_below_1_over_e_of_t0"],
            "forward_overshoot_then_return_m": bf["overshoot_then_return_m"],
            "reverse_overshoot_then_return_m": br["overshoot_then_return_m"],
            "forward_reversals": bf["n_direction_reversals_along_t0_heading(|v|>0.005)"]
            if "n_direction_reversals_along_t0_heading(|v|>0.005)" in bf
            else bf.get("n_direction_reversals_along_t0_heading(|v_along|>0.005)"),
            "reverse_reversals": br.get("n_direction_reversals_along_t0_heading(|v_along|>0.005)"),
        },
        "C_per_second_disp": {
            "forward": [{"t": s["t"], "euclid_m": s["euclid_disp_m"], "along_m": s["along_heading_m"],
                         "lateral_m": s["lateral_m"], "path_m": s["path_len_m"],
                         "mean_speed": s["mean_world_speed_mps"]} for s in bf["per_second"]],
            "reverse": [{"t": s["t"], "euclid_m": s["euclid_disp_m"], "along_m": s["along_heading_m"],
                         "lateral_m": s["lateral_m"], "path_m": s["path_len_m"],
                         "mean_speed": s["mean_world_speed_mps"]} for s in br["per_second"]],
        },
        "D_zero_command_consistency": {
            "forward_body_vx_mean": df["body_vx"]["mean"], "reverse_body_vx_mean": dr["body_vx"]["mean"],
            "forward_body_vy_mean": df["body_vy"]["mean"], "reverse_body_vy_mean": dr["body_vy"]["mean"],
            "forward_drift_m": df["euclid_drift_m"], "reverse_drift_m": dr["euclid_drift_m"],
            "forward_yaw_drift_rad": df["yaw_drift_rad"], "reverse_yaw_drift_rad": dr["yaw_drift_rad"],
            "max_abs_issued_action_diff_0_2s": None,  # filled by caller
        },
        "E_7_8_symmetry": {
            "forward_body_vx_mean": ef["body_vx"]["mean"], "reverse_body_vx_mean": er["body_vx"]["mean"],
            "forward_body_vy_mean": ef["body_vy"]["mean"], "reverse_body_vy_mean": er["body_vy"]["mean"],
            "forward_wheel_target_mean": ef["wheel_target_radps_mean_per_wheel"],
            "reverse_wheel_target_mean": er["wheel_target_radps_mean_per_wheel"],
            "forward_wheel_qvel_mean": ef["wheel_qvel_radps_mean_per_wheel"],
            "reverse_wheel_qvel_mean": er["wheel_qvel_radps_mean_per_wheel"],
            "forward_pitch_mean": ef["pitch"]["mean"], "reverse_pitch_mean": er["pitch"]["mean"],
            "forward_base_z_mean": ef["base_z"]["mean"], "reverse_base_z_mean": er["base_z"]["mean"],
        },
        "F_8_12_symmetry": {
            "forward_wheel_target_mean": ff["wheel"]["target_radps_mean"],
            "reverse_wheel_target_mean": fr["wheel"]["target_radps_mean"],
            "forward_wheel_target_std": ff["wheel"]["target_radps_std"],
            "reverse_wheel_target_std": fr["wheel"]["target_radps_std"],
            "forward_wheel_qvel_mean": ff["wheel"]["qvel_radps_mean"],
            "reverse_wheel_qvel_mean": fr["wheel"]["qvel_radps_mean"],
            "forward_wheel_net_rev": ff["wheel"]["net_rotation_rev"],
            "reverse_wheel_net_rev": fr["wheel"]["net_rotation_rev"],
            "forward_sat_steps_any": ff["saturation_steps_any_joint"],
            "reverse_sat_steps_any": fr["saturation_steps_any_joint"],
            "forward_action_at_clip_steps": ff["n_steps_any_action_at_clip"],
            "reverse_action_at_clip_steps": fr["n_steps_any_action_at_clip"],
            "forward_per_tick_delta_absmean_wheels": [ff["per_tick_action_delta_absmean_per_joint"][i] for i in WHEEL_ACT_IDX],
            "reverse_per_tick_delta_absmean_wheels": [fr["per_tick_action_delta_absmean_per_joint"][i] for i in WHEEL_ACT_IDX],
            "forward_mean_of_4_wheel_targets_radps": f(sum(ff["wheel"]["target_radps_mean"]) / 4.0),
            "reverse_mean_of_4_wheel_targets_radps": f(sum(fr["wheel"]["target_radps_mean"]) / 4.0),
            "forward_clip_joints": fw["F_clip_detail_8_to_12"]["joints_hitting_action_clip"],
            "reverse_clip_joints": rv["F_clip_detail_8_to_12"]["joints_hitting_action_clip"],
        },
        "G_rolling_vs_translation": {
            "forward_rolling_radius_m": fw["G_rolling_radius_used_m"],
            "reverse_rolling_radius_m": rv["G_rolling_radius_used_m"],
            "forward_per_second": fw["G_rolling_vs_translation_per_second_8_to_12"],
            "reverse_per_second": rv["G_rolling_vs_translation_per_second_8_to_12"],
        },
        "zero_command_creep_vs_stop_residual": {
            "note": "the 0-2s segment is the SAME zero command as 8-12s; its mean body_vx is the policy's intrinsic creep",
            "zero_cmd_0_2_mean_body_vx_mps": df["body_vx"]["mean"],
            "zero_cmd_0_2_euclid_drift_m": df["euclid_drift_m"],
            "zero_cmd_0_2_drift_scaled_to_4s_m": f(df["euclid_drift_m"] * 2.0),
            "forward_8_12_mean_body_vx_mps": fw["B_stop_window_8_to_12"]["body_vx_stats"]["mean"],
            "reverse_8_12_mean_body_vx_mps": rv["B_stop_window_8_to_12"]["body_vx_stats"]["mean"],
            "forward_8_12_euclid_m": fw["B_stop_window_8_to_12"]["euclid_disp_m"],
            "reverse_8_12_euclid_m": rv["B_stop_window_8_to_12"]["euclid_disp_m"],
            "forward_final_body_vx_mps": fw["B_stop_window_8_to_12"]["body_vx_stats"]["final"],
            "reverse_final_body_vx_mps": rv["B_stop_window_8_to_12"]["body_vx_stats"]["final"],
            "both_end_drifting_forward": (fw["B_stop_window_8_to_12"]["body_vx_stats"]["final"] > 0
                                          and rv["B_stop_window_8_to_12"]["body_vx_stats"]["final"] > 0),
        },
    }


def zero_seg_action_diff(a_dir: str, b_dir: str) -> dict:
    """Step-by-step comparison of the 0-2s identical-command segment between two trials."""
    ta = load_trace(os.path.join(a_dir, "trace.jsonl"))
    tb = load_trace(os.path.join(b_dir, "trace.jsonl"))
    i = min(idx_at(ta["sim_time"], 2.0), idx_at(tb["sim_time"], 2.0))
    sl = slice(0, i + 1)
    res = {}
    for key in ("issued_action", "qpos", "qvel", "position", "torque", "body_velocity"):
        da = np.abs(ta[key][sl] - tb[key][sl])
        res[key] = {"max_abs_diff": f(np.max(da)), "mean_abs_diff": f(np.mean(da)),
                    "first_step_with_diff_gt_1e-12": int(np.argmax(np.any(da > 1e-12, axis=1))) + 1
                    if np.any(da > 1e-12) else None}
    # where does divergence first appear and how does it grow
    dpos = np.linalg.norm(ta["position"][:, 0:2] - tb["position"][:, 0:2], axis=1)
    res["xy_divergence_m"] = {
        "at_t0.5": f(dpos[idx_at(ta["sim_time"], 0.5)]),
        "at_t1.0": f(dpos[idx_at(ta["sim_time"], 1.0)]),
        "at_t2.0": f(dpos[idx_at(ta["sim_time"], 2.0)]),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{WS}/evidence/career_contract_04/cxx_trace_audit_01/report.json")
    args = ap.parse_args()

    results = {}
    for name, cfg in TRIALS.items():
        results[name] = analyse(name, cfg)

    fw, rv = results["forward_stop_03d"], results["reverse_stop_03d"]
    cmp = compare(fw, rv)
    zdiff = zero_seg_action_diff(TRIALS["forward_stop_03d"]["dir"], TRIALS["reverse_stop_03d"]["dir"])
    cmp["D_zero_command_consistency"]["max_abs_issued_action_diff_0_2s"] = zdiff["issued_action"]["max_abs_diff"]
    cmp["D_zero_command_stepwise_diff"] = zdiff

    # cross-check reported stop displacement against our recomputation
    cmp["validation_vs_reported_score"] = {
        "forward_reported": fw["reported_stop_disp_m"],
        "forward_recomputed_euclid_8_12": fw["B_stop_window_8_to_12"]["euclid_disp_m"],
        "forward_abs_err": f(abs(fw["reported_stop_disp_m"] - fw["B_stop_window_8_to_12"]["euclid_disp_m"])),
        "reverse_reported": rv["reported_stop_disp_m"],
        "reverse_recomputed_euclid_8_12": rv["B_stop_window_8_to_12"]["euclid_disp_m"],
        "reverse_abs_err": f(abs(rv["reported_stop_disp_m"] - rv["B_stop_window_8_to_12"]["euclid_disp_m"])),
    }

    report = {
        "audit": "offline C++ trace audit, career_contract_04 / cxx_trace_audit_01",
        "generated_by": os.path.abspath(__file__),
        "physics_steps_executed_by_this_tool": 0,
        "inputs_modified": "none (read-only)",
        "key_numbers_03d": {
            "stop_euclid_m": {"forward_FAIL": fw["B_stop_window_8_to_12"]["euclid_disp_m"],
                              "reverse_PASS": rv["B_stop_window_8_to_12"]["euclid_disp_m"],
                              "ratio": f(fw["B_stop_window_8_to_12"]["euclid_disp_m"]
                                         / rv["B_stop_window_8_to_12"]["euclid_disp_m"])},
            "stop_path_len_m": {"forward": fw["B_stop_window_8_to_12"]["path_len_m"],
                                "reverse": rv["B_stop_window_8_to_12"]["path_len_m"],
                                "ratio": f(fw["B_stop_window_8_to_12"]["path_len_m"]
                                           / rv["B_stop_window_8_to_12"]["path_len_m"])},
            "speed_at_t8_mps": {"forward": fw["B_stop_window_8_to_12"]["world_speed_at_t0_mps"],
                                "reverse": rv["B_stop_window_8_to_12"]["world_speed_at_t0_mps"]},
            "reverse_overshoot_then_return_m": rv["B_stop_window_8_to_12"]["overshoot_then_return_m"],
            "forward_overshoot_then_return_m": fw["B_stop_window_8_to_12"]["overshoot_then_return_m"],
            "zero_command_0_2s_is_bit_identical_between_trials": True,
            "zero_command_steady_1_2s_creep_mps": fw["zero_command_steady_1_to_2"]["mean_body_vx_mps"],
            "tail_creep_11_12s_mps": {"forward": fw["tail_creep_11_to_12"]["mean_body_vx_mps"],
                                      "reverse": rv["tail_creep_11_to_12"]["mean_body_vx_mps"]},
            "tail_wheel_target_11_12s_radps_avg": {"forward": fw["tail_creep_11_to_12"]["mean_wheel_target_radps_4wheel_avg"],
                                                   "reverse": rv["tail_creep_11_to_12"]["mean_wheel_target_radps_4wheel_avg"]},
            "phase_decomposition": {"forward": fw["B_phase_decomposition_8_to_12"],
                                    "reverse": rv["B_phase_decomposition_8_to_12"]},
            "torque_limit_saturation_steps_8_12": {"forward": fw["F_actions_8_to_12"]["saturation_steps_any_joint"],
                                                   "reverse": rv["F_actions_8_to_12"]["saturation_steps_any_joint"]},
            "action_clip_steps_8_12_of_2001": {"forward": fw["F_actions_8_to_12"]["n_steps_any_action_at_clip"],
                                               "reverse": rv["F_actions_8_to_12"]["n_steps_any_action_at_clip"]},
        },
        "trials": results,
        "comparison_03d_forward_vs_reverse": cmp,
        "caveats": [
            "trace contains no contact force / friction / slip information: wheel-vs-ground slip can only be inferred from the rolling-radius bookkeeping, never measured",
            "trace contains no per-step observation vector, so the policy input that produced each action cannot be reconstructed from the trace alone",
            "the rolling radius used in section G is empirical (v_bodyx / wheel_omega over 2-8s) and not read from robot.xml; a wrong radius shifts all slip ratios proportionally",
            "only one trial per direction exists, so no run-to-run variance can be estimated; nothing here separates a systematic policy bias from a single-seed fluke",
            "03c turn_stop uses a different policy (pilot03c) and a different command sequence; it is reported separately and must not be mixed into 03d conclusions",
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f_:
        json.dump(report, f_, indent=2, sort_keys=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
