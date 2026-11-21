"""Translate the two equal angular shutters using only a nominal standing D1 calibration."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from tools100.sensor100 import _grid


CAPTURE_TICKS = (25, 34, 42, 50, 59, 67, 75, 84, 92, 100, 109, 117, 125, 134, 142, 150)
ACTIVE_MOD4 = (2, 3)
WINDOW_SHAPE = (6, 3)
SIDES = ("left", "right")
ROBOT_SHA256 = "64b7ba2887d011cfaefe22fe589c71596c0fa2ba0c0d64c5533c26fbe7e64757"
ROBOT_PATH = "src/ddt_ros2_control/urdfs/d1_description/mujoco/robot.xml"
NOMINAL_HEIGHT = 0.45237399982329246
NOMINAL_FRONT = np.array([[0.16921315742449886, 0.22031016496273878, -0.36536608664461967],
                          [0.16921471783598130, -0.2203141364527048, -0.36536566812213195]])


def verify_nominal_robot(path):
    """Check the distributed nominal geometry using the original private MJCF, without loading a simulator."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != ROBOT_SHA256:
        raise ValueError("nominal standing calibration requires the registered D1 MJCF")
    base = ElementTree.fromstring(data).find(".//body[@name='base_link']")
    centers, lowest = [], []
    for leg in ("FL", "FR", "RL", "RR"):
        node, pos, rotation = base, np.zeros(3), np.eye(3)
        for part, joint_angle in zip(("hip", "thigh", "calf", "foot"), (0., .8, -1.5, 0.), strict=True):
            node = node.find(f"body[@name='{leg}_{part}']")
            pos += rotation @ np.fromstring(node.get("pos"), sep=" ")
            q = np.fromstring(node.get("quat", "1 0 0 0"), sep=" ")
            w, x, y, z = q / np.linalg.norm(q)
            skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
            body_rotation = np.eye(3) + 2 * (w * skew + skew @ skew)
            if node.find("joint").get("axis") != "0 0 1":
                raise ValueError("unexpected D1 joint axis")
            c, s = np.cos(joint_angle), np.sin(joint_angle)
            rotation = rotation @ body_rotation @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        wheel = node.find("geom[@type='cylinder'][@class='collision']")
        radius, half_length = np.fromstring(wheel.get("size"), sep=" ")
        centers.append(pos)
        extent_z = radius * np.hypot(*rotation[2, :2]) + half_length * abs(rotation[2, 2])
        lowest.append(pos[2] - extent_z)
    if not (np.allclose(centers[:2], NOMINAL_FRONT, rtol=0, atol=1e-12)
            and abs(-min(lowest) - NOMINAL_HEIGHT) < 1e-12):
        raise ValueError("nominal wheel centers or static base height differ from public calibration")


def _window_start(angles, size, target):
    # Computing every candidate window handles the nonuniform depression grid and field-of-view edges.
    centers = np.lib.stride_tricks.sliding_window_view(angles, size).mean(axis=1)
    return int(np.argmin(np.abs(centers - target)))


def calibration(geometry):
    """Compute the shutter locations from a level default pose on flat ground. Indices are zero-based."""
    azimuth = np.asarray(geometry["azimuth_deg"], dtype=float)
    depression = np.asarray(geometry["depression_deg"], dtype=float)
    mounts = np.asarray(geometry["mounts_m"], dtype=float)
    if (azimuth.shape != (13,) or depression.shape != (20,) or mounts.shape != (2, 3)
            or not np.isfinite(mounts).all() or not np.all(np.diff(azimuth) > 0)
            or not np.all(np.diff(depression) > 0)):
        raise ValueError("expected D1's two mounts and sorted 20 by 13 ray grid")
    _, _, keep = _grid(geometry)
    old_mask = (~keep).reshape(2, 20, 13)
    blocks = []
    height, width = WINDOW_SHAPE
    for eye in old_mask:
        rr, cc = np.nonzero(eye)
        r, c = int(rr.min()), int(cc.min())
        if rr.size != height * width or not eye[r:r+height, c:c+width].all():
            raise ValueError("original shutter does not have the specified rectangular shape")
        blocks.append([r, c])

    query = NOMINAL_FRONT.copy()
    query[:, 0] = NOMINAL_FRONT[:, 0].mean() + geometry["lookahead_m"]
    query[:, 2] = -NOMINAL_HEIGHT
    angles, moved = {}, {}
    for side, point in zip(SIDES, query, strict=True):
        dx, dy, dz = (point - mounts).T
        yaw = np.rad2deg(np.arctan2(dy, dx))
        tilt = np.rad2deg(np.arctan2(-dz, np.hypot(dx, dy)))
        angles[side] = np.column_stack((yaw, tilt)).tolist()
        moved[side] = [[_window_start(depression, height, d), _window_start(azimuth, width, a)]
                       for a, d in zip(yaw, tilt, strict=True)]
    return dict(window_shape=list(WINDOW_SHAPE), original_starts=blocks, moved_starts=moved,
                active_mod4=list(ACTIVE_MOD4), nominal_headings_azimuth_depression_deg=angles,
                query_in_nominal_base_m=query.tolist(), nominal_front_wheel_centers_m=NOMINAL_FRONT.tolist(),
                nominal_base_height_m=NOMINAL_HEIGHT, original_mjcf=ROBOT_PATH, original_mjcf_sha256=ROBOT_SHA256,
                nominal_height_record="reports/local_comparison16_comparison.json",
                default_leg_q_rad=[0., .8, -1.5, 0.])


def _blocked(starts):
    blocked = np.zeros((2, 20, 13), dtype=bool)
    height, width = WINDOW_SHAPE
    for eye, (row, col) in zip(blocked, starts, strict=True):
        eye[row:row+height, col:col+width] = True
    return blocked.ravel()


def packet_masks(capture_tick, target_side, geometry):
    """Return (original blocked, moved blocked, active) for N rows. True blocked means ray suppressed."""
    ticks, side = np.asarray(capture_tick), np.asarray(target_side)
    if side.ndim == 0:
        side = np.full(ticks.shape, side.item())
    if (ticks.ndim != 1 or ticks.dtype.kind not in "iu" or side.shape != ticks.shape
            or not np.isin(side, SIDES).all() or not np.isin(ticks, CAPTURE_TICKS).all()):
        raise ValueError("expected registered 6 Hz acquisition ticks and an explicit wheel side per row")
    tick_index = np.searchsorted(CAPTURE_TICKS, ticks)
    active = np.isin(tick_index % 4, ACTIVE_MOD4)
    info = calibration(geometry)
    fixed = np.tile(_blocked(info["original_starts"]), (len(ticks), 1))
    moved = fixed.copy()
    for name in SIDES:
        moved[active & (side == name)] = _blocked(info["moved_starts"][name])
    return fixed, moved, active


def level_points(range_m64, valid, capture_body_to_level, geometry):
    """Recover measured XYZ in the same gravity-aligned frame as round 101; masked ranges become NaNs."""
    ranges = np.asarray(range_m64)
    valid = np.asarray(valid)
    rotations = np.asarray(capture_body_to_level)
    if (ranges.ndim != 2 or ranges.shape[1] != 520 or ranges.dtype.kind != "f"
            or valid.dtype != bool or valid.shape != ranges.shape or rotations.shape != (len(ranges), 3, 3)
            or not np.isfinite(rotations).all() or not np.isfinite(ranges[valid]).all()
            or np.any(ranges[valid] < geometry["range_min_m"]) or np.any(ranges[valid] > geometry["range_max_m"])):
        raise ValueError("invalid ray ranges, validity or attitude transform")
    directions, _, _ = _grid(geometry)
    mounts = np.repeat(np.asarray(geometry["mounts_m"]), len(directions), axis=0)
    points = (mounts + np.where(valid, ranges, 0.)[..., None] * np.tile(directions, (2, 1))) @ rotations
    points[~valid] = np.nan
    return points
