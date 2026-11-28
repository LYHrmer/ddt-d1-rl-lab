"""Read back both registered compound-route interventions from their measured native trajectories."""
from pathlib import Path

import numpy as np

from tools91.launch91 import digest, read, require
from tools110.course import TICKS, command
from tools111.control import SPEED_GAIN
from tools111.course import ground, score
from tools111.replay import check_episode
from tools114.closed_results import parking_stop
from tools116.response_replay import INDEX, INDEX_SHA, ORIGINAL_SLEW, RELEASE_SLEW, trim_at_recorded_states
from tools116.score import SEEDS, _observe, first_failed_section, lidar_effects
from tools116.wiring import ArchiveGeometry, _check_leg_targets


def _first(values):
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None


def _complete_file(path):
    require(path.name == "trace.npz" and not any((path.parent / name).exists() for name in
            ("partial.npz", "partial_counts.json", "worker_partial.json", "partial.json")),
            "a partial run cannot qualify as a native trajectory")


def _verify_reference(path, seed, parking):
    require(seed in SEEDS and digest(INDEX) == INDEX_SHA, "unregistered start or changed R116 archive")
    source, = [c for c in read(INDEX)["cases"] if c["seed"] == seed]
    require(source["stage_status"] == "COMPLETE" and source["complete_horizon"], "incomplete paired R116 start")
    sha = digest(path)
    if parking:
        receipt = read(path.parent / "report.json")
        require(receipt["stage"] == f"speed_{seed}" and receipt["status"] == "COMPLETE"
                and receipt["files"]["trace.npz"] == sha, "parking reference must be its completed speed arm")
    else:
        require(source["trace_sha256"] == sha, "wrong source trace for the paired release start")
    return sha


def _speed_signals(z):
    wheels, preview = z["encoder_wheel_rad_s"], z["preview_actor_vx"]
    n = len(z["raw"])
    require(wheels.shape == (n, 4) and preview.shape == (n,) and np.isfinite(preview).all(),
            "invalid encoder or preview clock")
    increments = trim_at_recorded_states(wheels, preview, RELEASE_SLEW)
    actual = preview + increments
    frame = z["actor_frame_command"]
    require(frame.dtype == np.float32, "changed actor input precision")
    np.testing.assert_array_equal(z["velocity_trim"], increments)
    np.testing.assert_array_equal(z["actor_vx"], actual)
    np.testing.assert_array_equal(z["encoder_forward_m_s"], .087 * np.median(wheels, axis=1))
    np.testing.assert_array_equal(z["trim_previous_actor_vx"], np.r_[0., actual[:-1]])
    np.testing.assert_array_equal(frame[:, 0], (2. * actual).astype(np.float32))
    previous_raw = np.vstack((np.zeros_like(z["raw"][0]), z["raw"][:-1]))
    history = np.maximum(0, np.arange(n)[:, None] + np.arange(-10, 0))
    np.testing.assert_array_equal(z["actor_previous_raw"], previous_raw)
    np.testing.assert_array_equal(z["actor_history_previous_raw"], previous_raw[history])
    np.testing.assert_array_equal(z["actor_history_frame_command"], frame[history])
    original = trim_at_recorded_states(wheels, preview, ORIGINAL_SLEW)
    return increments, _first(frame[:, 0] != (2. * (preview + original)).astype(np.float32))


def _projected_wheels(z, parking):
    wheels = z["raw"][:, 3::4].astype(float)
    expected = wheels.copy()
    if parking:
        # A zero request before the first forward command does not arm the projection.
        requests = np.array([command(k) for k in range(len(wheels))])
        active = (requests == 0.) & np.maximum.accumulate(requests > 0.)
        expected[active] -= wheels[active].mean(axis=1, keepdims=True)
        # The original leg adapter returns the actor's dtype when it has no lift request.
        no_lift = ~np.any(z["lift_demand"] != 0., axis=1)
        expected[no_lift] = expected[no_lift].astype(z["raw"].dtype)
    np.testing.assert_array_equal(z["issued"][:, 3::4], expected)
    return _first(np.any(expected != wheels, axis=1))


def _paired_prefix(z, old, parking, first_trigger):
    n = len(z["raw"])
    require(len(old["raw"]) == TICKS and n <= TICKS, "uncertain paired horizon")
    first_input = _first(z["actor_frame_command"][:, 0] != old["actor_frame_command"][:n, 0])
    first_issue = _first(np.any(z["issued"] != old["issued"][:n], axis=1))
    first_apply = _first(np.any(z["applied"] != old["applied"][:n], axis=1))
    delayed = None if first_issue is None or first_issue + 1 == n else first_issue + 1
    require(first_apply == delayed, "paired actuator difference at the wrong FIFO tick")
    if parking:
        require(first_issue == first_trigger, "paired action changed outside the parking intervention")
    else:
        require(first_input == first_trigger and (first_issue is None or
                (first_input is not None and first_input <= first_issue)), "paired action preceded the release")
    issued = n if first_issue is None else first_issue
    applied = n if first_apply is None else first_apply
    inputs = min(n, applied + 1)
    raw = inputs if parking or first_input is None else first_input
    for key in ("qpos", "qvel", "xy", "vxy", "wheels_xy", "measured_relative_z", "front_extent_x", "time"):
        np.testing.assert_array_equal(z[key][:applied+1], old[key][:applied+1], err_msg=key)
    for key in ("applied", "applied_lift_fk", "first_torque_nm", "fallen", "illegal", "front_platform_touch",
                "top_contact", "exit_contact", "wheel_saturation"):
        np.testing.assert_array_equal(z[key][:applied], old[key][:applied], err_msg=key)
    for key in ("issued", "issued_lift_fk"):
        np.testing.assert_array_equal(z[key][:issued], old[key][:issued], err_msg=key)
    for key in ("raw", "actor_frame_command", "actor_previous_raw", "actor_history_previous_raw",
                "actor_history_frame_command"):
        np.testing.assert_array_equal(z[key][:raw], old[key][:raw], err_msg=key)
    for key in ("imu_quat", "encoder_wheel_rad_s", "wheel_query_xy", "base_cmd", "preview_actor_vx",
                "actor_wz", "lift_demand", "capture_used", "coverage", "odom_travel"):
        np.testing.assert_array_equal(z[key][:inputs], old[key][:inputs], err_msg=key)
    packets = int((z["capture_tick"] < inputs).sum())
    for key in ("capture_tick", "range_m", "range_valid", "self_hit"):
        np.testing.assert_array_equal(z[key][:packets], old[key][:packets], err_msg=key)
    return dict(first_changed_input_tick=first_input, first_changed_issue_tick=first_issue,
                first_changed_applied_tick=first_apply, equal_state_endpoints=applied+1,
                equal_issued_ticks=issued, equal_applied_ticks=applied, equal_raw_actor_outputs=raw,
                equal_lidar_packets=packets)


def score_trace(path, reference_path, seed, parking=False):
    """Score the full trace and verify the release (or parking) against its paired control."""
    path, reference_path = Path(path), Path(reference_path)
    for source in (path, reference_path):
        _complete_file(source)
    reference_sha = _verify_reference(reference_path, seed, parking)
    with np.load(path, allow_pickle=False) as archive, np.load(reference_path, allow_pickle=False) as prior:
        z, old = dict(archive), dict(prior)
    counts = _observe(z)
    increment, first_release = _speed_signals(z)
    first_projection = _projected_wheels(z, parking)
    height, pose = float(z["nominal_base_height_m"]), z["qpos"][1:]
    require(.25 < height < .8 and height == float(old["nominal_base_height_m"]), "changed initial root height")
    tilt = np.arccos(np.clip(1.-2.*(pose[:, 4]**2 + pose[:, 5]**2), -1., 1.))
    clearance = pose[:, 2] - ground("development", pose[:, 0], pose[:, 1])
    np.testing.assert_array_equal(z["fallen"], (clearance < height-.18) | (tilt > .6))

    frame = z["actor_frame_command"].copy()
    frame[:, 0] = (2. * z["preview_actor_vx"]).astype(np.float32)
    preview = dict(z, actor_vx=z["preview_actor_vx"], actor_frame_command=frame)
    if parking:
        # R111's replay expects unmodified wheels between its input and the leg adapter.
        # Present the wheels independently checked above at that boundary. The actual raw
        # policy history, delayed target and unsynthesized servo torque are still checked.
        adapted = z["raw"].astype(float)
        adapted[:, 3::4] = z["issued"][:, 3::4]
        preview["raw"] = adapted
    motion, effects = score(preview, "development")
    check_episode(preview, dict(layout="development", arm="both", gain=SPEED_GAIN, counts=counts,
                                motion=motion, effects=effects))
    geometry = _check_leg_targets(z, ArchiveGeometry())
    lidar = lidar_effects(z, increment)
    prefix = _paired_prefix(z, old, parking, first_projection if parking else first_release)
    stop = parking_stop(z["qpos"], z["qvel"])
    if stop is not None:
        require(stop["peak_m_s"] == motion["stop_tail_max_m_s"]
                and ("stopped" in motion["failed"]) == (stop["endpoints_above_limit"] > 0),
                "inconsistent 26-endpoint stop score")
    reference_motion, _ = score(old, "development")
    return dict(seed=seed, trace_sha256=digest(path), reference_sha256=reference_sha, counts=counts,
                complete_horizon=motion["ticks"] == TICKS, driving_pass=lidar["passed"] and
                all(k == "stopped" for k in motion["failed"]), gate_pass=motion["passed"] and lidar["passed"],
                motion=motion, effects=effects, lidar=lidar, stop=stop, stop_endpoints=26 if stop is not None else 0,
                first_failed_section=first_failed_section(motion), prefix=prefix, geometry=geometry,
                reference_motion=reference_motion, reference_stop=parking_stop(old["qpos"], old["qvel"]))
