"""Verify the native compensation cases using public trajectories and onboard inputs."""
import math

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools111.replay import check_episode
from tools112.surface import PUBLIC as SURFACE_RESULTS, SEEDS, bindings


def _commands(encoders):
    """Reconstruct each actor request without the implementation that ran the controller."""
    n = len(encoders)
    requests = np.array([command(k) for k in range(n)])
    measured = .087 * np.median(encoders, axis=1)
    base = requests * SPEED_GAIN
    issued = np.empty(n)
    increments = np.empty(n)
    bias = cruise = previous = increment = 0.
    held = 0
    weight = -math.expm1(-.02/.40)
    for k, request in enumerate(requests):
        held = held + 1 if k and request == requests[k-1] else 0
        if held >= 10:
            bias += weight * (previous - measured[k] - bias)
            desired = np.clip((1.025 - SPEED_GAIN) * request + bias, 0., .18)
            cruise += np.clip(desired - cruise, -.0036, .0036)
        cap = .18 * np.clip(request / 1.5, 0., 1.) + max(0., -measured[k])
        increment += np.clip(min(cruise, cap) - increment, -.0036, .0036)
        increments[k] = increment
        issued[k] = previous = base[k] + increment
    return measured, base, increments, issued


def verify_episode(path, row, reference_path, reference_row):
    seed = row["seed"]
    require(seed in SEEDS and reference_row["seed"] == seed, "unmatched initial state")
    public = read(SURFACE_RESULTS / "results.json")
    require(public["sources"] == bindings(), "original native controller or floor changed")
    case, = [c for c in public["cases"] if c["seed"] == seed]
    vendor = read(ROOT / "reports/speed110/results.json")["preflight"]
    require(case["case_status"] == row["status"] == "COMPLETE" and case["episode"] == reference_row
            and reference_row["motion"]["ticks"] == TICKS and row["trim"] == "brake_aware"
            and row["layout"] == "flat" and row["surface"] == "level_heightfield"
            and row["arm"] == "heading_on" and row["gain"] == SPEED_GAIN
            and row["actor_sha256"] == vendor["model_sha256"]
            and row["actuator_config_sha256"] == vendor["config_sha256"]
            and row["mujoco_version"] == reference_row["mujoco_version"] == "3.3.0"
            and row["reference_sha256"] == reference_row["trace_sha256"] == digest(reference_path)
            and row["trace_sha256"] == digest(path), "invalid paired source or actuator contract")
    with np.load(path, allow_pickle=False) as trace, np.load(reference_path, allow_pickle=False) as baseline:
        data, off = dict(trace), dict(baseline)
    check_episode(off, reference_row)
    n = len(data["time"]) - 1
    require(1 <= n <= TICKS and data["encoder_wheel_rad_s"].shape == (n, 4), "invalid trajectory clock")
    for key in ("qpos", "qvel", "imu_quat", "encoder_wheel_rad_s", "range_m", "raw", "issued", "applied",
                "first_torque_nm", "actor_frame_command"):
        np.testing.assert_array_equal(data[key][0], off[key][0], err_msg=key)
    for key in ("time", "base_cmd", "capture_tick"):
        np.testing.assert_array_equal(data[key], off[key][:len(data[key])], err_msg=key)
    measured, base, increments, issued = _commands(data["encoder_wheel_rad_s"])
    for key in ("preview_actor_vx", "actor_vx", "velocity_trim", "encoder_forward_m_s"):
        require(data[key].shape == (n,), f"incorrect interval count for {key}")
    np.testing.assert_array_equal(data["encoder_forward_m_s"], measured)
    np.testing.assert_array_equal(data["preview_actor_vx"], base)
    np.testing.assert_array_equal(base, off["actor_vx"][:n])
    np.testing.assert_allclose(data["velocity_trim"], increments, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(data["actor_vx"], issued, atol=1e-12, rtol=0.)
    frame = data["actor_frame_command"]
    require(frame.shape == (n, 3) and frame.dtype == np.float32, "invalid actor command triplets")
    np.testing.assert_array_equal(frame[:, 0], (2.*data["actor_vx"]).astype(np.float32))

    # Reconstruct the original preview from its real ranges and IMU. Keep the
    # measured states, issued actions and servo torques for the actuator audit.
    preview_frame = frame.copy()
    preview_frame[:, 0] = (2.*base).astype(np.float32)
    motion, _ = check_episode(dict(data, actor_vx=base, actor_frame_command=preview_frame), row)
    ended = data["fallen"] | data["illegal"]
    require(not ended[:-1].any() and (n == TICKS or ended[-1]), "unexplained early terminal")
    changed = np.flatnonzero(frame[:, 0] != preview_frame[:, 0])
    first = int(changed[0]) if len(changed) else None
    prefix = n if first is None else first
    for key in ("qpos", "qvel"):
        np.testing.assert_array_equal(data[key][:prefix+1], off[key][:prefix+1], err_msg=key)
    np.testing.assert_array_equal(data["raw"][:prefix], off["raw"][:prefix])
    if n < TICKS:
        require(not motion["passed"] and motion["stop_tail_max_m_s"] is None, "truncated motion qualified")
        return None

    headings = data["imu_yaw"][575:]
    axes = np.column_stack((np.cos(headings), np.sin(headings)))
    forward = np.sum(data["vxy"][575:-1] * axes, axis=1)
    distance = np.sum(np.diff(data["xy"][575:], axis=0) * axes)
    return dict(cruise_m_s=motion["speed_m_s"], stop_peak_m_s=motion["stop_tail_max_m_s"],
                stop_mean_forward_m_s=float(forward.mean()), stop_forward_displacement_mm=float(distance*1000.),
                first_trim_tick=first)
