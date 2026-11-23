"""Paired readback of a 40 ms point-cloud delivery delay using sealed D1 packets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.score100 import estimate_delta_mm, grade
from tools100.sensor100 import _grid
from tools101.common101 import PUBLIC as PACKETS, protocol
from tools101.score101 import METADATA, audit_geometry, validate_holdout


PUBLIC = ROOT / "reports/motion104"
FORECAST = ROOT / "reports/action_window103"
WHEEL_RADIUS_M = 0.087  # Four collision cylinders in the native D1 MJCF.
DT_S = 0.02
ARMS = ("original", "wheel_imu", "reference_pose")


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def heading(pose):
    w, x, y, z = pose[3:] / np.linalg.norm(pose[3:])
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def reference_motion(poses):
    """Translation in the capture heading frame and change in yaw, for audit only."""
    capture, delivery = poses
    yaw_c, yaw_d = heading(capture), heading(delivery)
    xy = (delivery[:2] - capture[:2]) @ rotation(yaw_c)
    return np.r_[xy, np.arctan2(np.sin(yaw_d - yaw_c), np.cos(yaw_d - yaw_c))]


def measured_motion(gyro, gravity, wheels):
    """Integrate two 20 ms intervals with the three samples available by delivery."""
    up = -gravity / np.linalg.norm(gravity, axis=1, keepdims=True)
    projection = np.sum(up[:, 1:] ** 2, axis=1)
    require(np.all(projection > 1e-8), "undefined horizontal heading")
    # Angular rate of the body's forward direction projected on the horizontal plane.
    rates = np.sum(up[:, 1:] * gyro[:, 1:], axis=1) / projection
    yaw = np.r_[0., np.cumsum(0.5 * DT_S * (rates[:-1] + rates[1:]))]
    speed = WHEEL_RADIUS_M * wheels.mean(axis=1)
    velocities = speed[:, None] * np.column_stack([np.cos(yaw), np.sin(yaw)])
    xy = 0.5 * DT_S * (velocities[:-1] + velocities[1:]).sum(axis=0)
    return np.r_[xy, yaw[-1]]


def capture_query(query, motion):
    """Express the delivered wheel queries in the capture frame of the points."""
    return query @ rotation(motion[2]).T + motion[:2]


def extract_windows(path):
    from training.d1_rl.interface import WHEEL_INDICES

    plan = protocol()
    source, = [s for s in read(PACKETS / "data.json")["sources"] if s["source_seed"] == 89037]
    trace = ROOT / source["path"]
    expected_hash = plan["data"]["trace_sha256"]["89037"]
    require(digest(trace) == source["trace_sha256"] == expected_hash, "native trace changed")
    keys, poses, gyro, gravity, wheels = [], [], [], [], []
    with np.load(trace, allow_pickle=False) as a:
        qpos, actor = a["qpos"], a["proprio627"]
        for episode in source["admitted_flat_episodes"]:
            start, stop = episode["first_archive_row"], episode["end_archive_row_exclusive"]
            for c in plan["capture_ticks"]:
                d = c + plan["delay_ticks"]
                require(start + d < stop, "motion window crosses a reset")
                # In one received actor row the current frame comes first, preceded in
                # time by the history stored oldest to newest after the current frame.
                frames = actor[start+d].reshape(11, 57)[[-2, -1, 0]]
                require(np.array_equal(frames, actor[start+c:start+d+1, :57]), "causal history mismatch")
                frames = frames.astype(np.float64)
                keys.append((episode["episode"], c, d))
                poses.append(qpos[[start+c-1, start+d-1], :7])  # interval-end pose
                gyro.append(frames[:, :3] / 0.25)
                gravity.append(frames[:, 3:6])
                wheels.append(frames[:, 25 + WHEEL_INDICES] / 0.05)
    arrays(path, dict(keys=keys, reference_poses=poses, gyro_body=gyro, gravity_body=gravity,
                      wheel_rad_s=wheels, source_sha256=expected_hash))


def batch_stats(ids, truth, estimates):
    stats, original = {}, None
    for name in ARMS:
        values = estimates[name][ids]
        valid = np.isfinite(values)
        passed = np.array([grade(t, v if ok else None)
                           for t, v, ok in zip(truth[ids], values, valid, strict=True)])
        if original is None:
            original = passed
        stats[name] = dict(passed=int(passed.sum()), missing=int((~valid).sum()),
                           recovered=int((~original & passed).sum()), regressed=int((original & ~passed).sum()),
                           mae_mm=round(float(np.mean(np.abs(truth[ids][valid] - values[valid]))), 6)
                           if valid.any() else None)
    return dict(count=len(ids), arms=stats)


def score(windows):
    plan = protocol()
    packet_file = PACKETS / "holdout.npz"
    forecast_file = FORECAST / "forecast.npz"
    identity = read(FORECAST / "report.json")
    require(identity["status"] == "COMPLETE" and identity["holdout_sha256"] == digest(packet_file)
            and identity["forecast_sha256"] == digest(forecast_file), "sealed development packets changed")
    with np.load(packet_file, allow_pickle=False) as archive:
        packets = {k: archive[k] for k in (*METADATA, "geometry_depth_m64", "valid", "query_xy",
                                            "world_query_xy", "capture_body_to_level")}
    validate_holdout(packets)
    audit_geometry(packet_file, packets, plan)
    with np.load(forecast_file, allow_pickle=False) as f:
        starts = f["row_id"]
    expected = np.flatnonzero(packets["series_id"][1:] == packets["series_id"][:-1])
    require(np.array_equal(starts, expected), "incomplete or reordered transition coverage")
    keys = list(zip(packets["episode"], packets["capture_tick"], packets["delivery_tick"], strict=True))
    with np.load(windows, allow_pickle=False) as w:
        roster = list(map(tuple, w["keys"]))
        require(roster == sorted(set(keys)) and len(roster) == 80
                and w["source_sha256"].item() == plan["data"]["trace_sha256"]["89037"],
                "incomplete motion windows or incorrect native source")
        poses, gyro, gravity, wheels = (w[k] for k in ("reference_poses", "gyro_body", "gravity_body", "wheel_rad_s"))
    require([v.shape for v in (poses, gyro, gravity, wheels)] == [(80, 2, 7), (80, 3, 3), (80, 3, 3), (80, 3, 4)]
            and all(np.isfinite(v).all() for v in (poses, gyro, gravity, wheels)), "invalid motion samples")
    lookup = {k: j for j, k in enumerate(roster)}
    motions = {"original": np.zeros((len(roster), 3)),
               "wheel_imu": np.array([measured_motion(g, u, v) for g, u, v in zip(gyro, gravity, wheels, strict=True)]),
               "reference_pose": np.array([reference_motion(pair) for pair in poses])}
    geom = read(ROOT / plan["data"]["sensor_base"])["geometry"]
    directions, _, _ = _grid(geom)
    mounts = np.repeat(geom["mounts_m"], len(directions), axis=0)
    directions = np.tile(directions, (2, 1))
    estimates = {name: np.full(len(keys), np.nan) for name in ARMS}
    for i, k in enumerate(keys):
        j = lookup[k]
        capture, delivery = poses[j]
        query = packets["query_xy"][i]
        world = packets["world_query_xy"][i]  # Used only to verify the independent pose oracle.
        require(np.allclose(query @ rotation(heading(delivery)).T + delivery[:2], world, rtol=0, atol=1e-9)
                and np.allclose(capture_query(query, motions["reference_pose"][j]),
                                (world - capture[:2]) @ rotation(heading(capture)), rtol=0, atol=1e-9),
                "reference transform and published world queries disagree")
        level = packets["capture_body_to_level"][i]
        require(np.allclose(level[:, 2], -gravity[j, 0], rtol=0, atol=1e-6), "capture gravity clock mismatch")
        points = (mounts + packets["geometry_depth_m64"][i, :, None] * directions) @ level
        valid_points = points[packets["valid"][i]]
        for name in ARMS:
            value = estimate_delta_mm(valid_points, capture_query(query, motions[name][j]))
            if value is not None:
                estimates[name][i] = value
    require(np.allclose(estimates["original"], packets["geometric_mm"], rtol=0, atol=1e-8, equal_nan=True),
            "baseline disagrees with the original geometric estimator")

    truth = packets["target_mm"]
    results, edge_rows = {}, []
    for visibility in ("masked", "clear"):
        selected = np.flatnonzero(packets["visibility"] == visibility)
        s = starts[packets["visibility"][starts] == visibility]
        changed = truth[s] != truth[s+1]
        edge = s[changed] + 1
        groups = {"all_arrivals": selected, "switch": edge, "stable": s[~changed] + 1,
                  "enter": edge[truth[edge] != 0], "exit": edge[truth[edge] == 0]}
        results[visibility] = {name: batch_stats(ids, truth, estimates) for name, ids in groups.items()}
        if visibility == "masked":
            for i in edge:
                values = {name: round(float(v[i]), 6) if np.isfinite(v[i]) else None for name, v in estimates.items()}
                edge_rows.append(dict(row_id=int(i), episode=int(packets["episode"][i]), layout=str(packets["layout"][i]),
                                      capture_tick=int(packets["capture_tick"][i]), delivery_tick=int(packets["delivery_tick"][i]),
                                      target_mm=float(truth[i]), measured_mm=values))
    reference, measured = motions["reference_pose"], motions["wheel_imu"]
    shifts = 1000 * np.linalg.norm(reference[:, :2], axis=1)
    errors = 1000 * np.linalg.norm(measured[:, :2] - reference[:, :2], axis=1)
    return dict(status="COMPLETE", holdout_sha256=identity["holdout_sha256"], forecast_sha256=identity["forecast_sha256"],
                native_trace_sha256=plan["data"]["trace_sha256"]["89037"], windows_sha256=digest(windows),
                episodes=5, source_seeds=[89037], physical_clouds=160, motion_windows=len(roster),
                new_physics_steps=0, new_rays=0, model_updates=0, policy_updates=0,
                motion_mm=dict(reference_min=round(float(shifts.min()), 6), reference_median=round(float(np.median(shifts)), 6),
                               reference_max=round(float(shifts.max()), 6), wheel_imu_median_error=round(float(np.median(errors)), 6),
                               wheel_imu_max_error=round(float(errors.max()), 6)),
                results=results, masked_switch_arrivals=edge_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", type=Path, help="extract from the local native trace into a new directory")
    mode.add_argument("--check", type=Path, default=PUBLIC, help="replay the published diagnostic")
    args = parser.parse_args()
    if args.record is not None:
        args.record.mkdir(parents=True, exist_ok=False)
        path = args.record / "windows.npz"
        extract_windows(path)
        result = score(path)
        write(args.record / "report.json", result)
    else:
        result = score(args.check / "windows.npz")
        require(result == read(args.check / "report.json"), "published motion diagnostic does not reproduce")
    for view, groups in result["results"].items():
        for label in ("switch", "stable"):
            group = groups[label]
            print(view, label, group["count"], {k: v["passed"] for k, v in group["arms"].items()})
