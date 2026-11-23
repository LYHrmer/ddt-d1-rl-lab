"""Identify the measured surfaces at failed wheel queries, using world coordinates only for diagnosis."""
import json

import numpy as np

from tools94.common94 import ROOT, digest, read
from tools100.score100 import RADIUS_M, TIE_ATOL_M, estimate_delta_mm, grade
from tools100.sensor100 import _grid
from tools101.common101 import protocol
from tools104.motion104 import PACKETS, PUBLIC, capture_query, heading, measured_motion, rotation


def surface(point, front, rear, center_y, half_y, top):
    x, y, z = point
    tol = 1e-7
    if abs(z) < tol:
        return "ground"
    if front-tol <= x <= rear+tol and abs(y - center_y) <= half_y+tol:
        if abs(z - top) < tol:
            return "top"
        if 0 < z < top:
            if abs(x - front) < tol:
                return "front_face"
            if abs(x - rear) < tol:
                return "rear_face"
            if abs(abs(y - center_y) - half_y) < tol:
                return "side_face"
    return "unknown"


def evidence():
    report = read(PUBLIC / "report.json")
    path, windows = PACKETS / "holdout.npz", PUBLIC / "windows.npz"
    if digest(path) != report["holdout_sha256"] or digest(windows) != report["windows_sha256"]:
        raise ValueError("the returns must belong to the scored packet and motion archives")
    plan = protocol()["data"]
    geom = read(ROOT / plan["sensor_base"])["geometry"]
    fixture, = plan["layouts"]["holdout"]
    front = fixture["box_center_x_m"] - fixture["box_half_x_m"]
    rear = fixture["box_center_x_m"] + fixture["box_half_x_m"]
    top = geom["box_height_m"]
    directions, _, _ = _grid(geom)
    mounts = np.repeat(geom["mounts_m"], len(directions), axis=0)
    directions = np.tile(directions, (2, 1))
    with np.load(path, allow_pickle=False) as a, np.load(windows, allow_pickle=False) as w:
        p = {k: a[k] for k in ("episode", "capture_tick", "delivery_tick", "valid", "geometry_depth_m64",
                               "visibility", "query_xy", "world_query_xy", "capture_body_to_level",
                               "geometric_mm", "target_mm", "side", "series_id")}
        lookup = {tuple(key): j for j, key in enumerate(w["keys"])}
        poses = w["reference_poses"]
        capture_up = -w["gravity_body"][:, 0]
        motions = [measured_motion(*vals) for vals in zip(w["gyro_body"], w["gravity_body"], w["wheel_rad_s"], strict=True)]
    failures, mismatches = [], []
    for i in np.flatnonzero(p["visibility"] == "masked"):
        key = p["episode"][i], p["capture_tick"][i], p["delivery_tick"][i]
        j = lookup[key]
        pose = poses[j, 0]
        raw = (mounts + p["geometry_depth_m64"][i, :, None] * directions)[p["valid"][i]]
        local = raw @ p["capture_body_to_level"][i]
        world = local.copy()
        world[:, :2] = local[:, :2] @ rotation(heading(pose)).T + pose[:2]
        world[:, 2] += pose[2]
        query = p["world_query_xy"][i]
        truth = p["target_mm"][i]
        up = capture_up[j] / np.linalg.norm(capture_up[j])
        forward = np.array([1., 0., 0.]) - up[0] * up
        forward /= np.linalg.norm(forward)
        imu_level = np.column_stack((forward, np.cross(up, forward), up))
        if not np.allclose(imu_level, p["capture_body_to_level"][i], atol=1e-6, rtol=0):
            raise ValueError("IMU gravity does not reproduce the archived leveling transform")
        wheel_query = capture_query(p["query_xy"][i], motions[j])
        measured = estimate_delta_mm(raw @ imu_level, wheel_query)
        archived_level = estimate_delta_mm(local, wheel_query)
        if (measured is None) != (archived_level is None) or (
            measured is not None and abs(measured - archived_level) > 1e-4
        ):
            raise ValueError("IMU-only leveling changes the measured wheel-track height")
        exact = estimate_delta_mm(world, query)
        if grade(truth, measured) != grade(truth, exact):
            mismatches.append(int(i))
        if grade(truth, exact):
            continue
        side = p["side"][i]
        xy = query[0 if side == "left" else 1]
        target_z = top if truth != 0 else 0.
        distances = np.linalg.norm(world[:, :2] - xy, axis=1)
        nearest = distances.min()
        if nearest > RADIUS_M:
            raise ValueError("unsupported failed query")
        closest = world[distances <= nearest + TIE_ATOL_M]
        center_y = fixture["box_abs_y_m"] * (1 if side == "left" else -1)
        on_surface = (np.abs(world[:, 2] - target_z) < 1e-7) & (distances <= RADIUS_M)
        previous = i > 0 and p["series_id"][i-1] == p["series_id"][i]
        failures.append(dict(row_id=int(i), truth_mm=float(truth), estimate_mm=round(exact, 3),
                             original_passed=grade(truth, p["geometric_mm"][i]),
                             switched=bool(previous and truth != p["target_mm"][i-1]),
                             side=str(side), nearest_z_mm=np.round(1000 * closest[:, 2], 3).tolist(),
                             nearest_surfaces=[surface(pt, front, rear, center_y, fixture["box_half_y_m"], top)
                                               for pt in closest],
                             nearest_xy_mm=round(1000 * float(nearest), 3),
                             expected_surface_nearest_xy_mm=round(1000 * float(distances[on_surface].min()), 3)
                             if on_surface.any() else None,
                             query_past_front_mm=round(1000 * float(xy[0] - front), 3),
                             query_past_rear_mm=round(1000 * float(xy[0] - rear), 3)))
    expected = report["results"]["masked"]["all_arrivals"]
    if len(failures) != expected["count"] - expected["arms"]["reference_pose"]["passed"]:
        raise ValueError("world-coordinate replay disagrees with capture-coordinate replay")
    return dict(checked=expected["count"], wheel_imu_vs_reference_grade_mismatches=mismatches, failures=failures)


if __name__ == "__main__":
    print(json.dumps(evidence(), ensure_ascii=False, indent=2))
