"""D1 first-return point clouds at a recorded pose, without integration."""

from __future__ import annotations

from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np


def _add_box(world, geometry, layout):
    if layout not in ("flat", "left", "right"):
        raise ValueError(f"unknown replay layout {layout!r}")
    if layout == "flat":
        return
    half_z = geometry["box_height_m"] / 2
    y = geometry["box_abs_y_m"] * (1 if layout == "left" else -1)
    ET.SubElement(world, "geom", name="bump100", type="box",
                  pos=f"{geometry['box_center_x_m']} {y} {half_z}",
                  size=f"{geometry['box_half_x_m']} {geometry['box_half_y_m']} {half_z}")


def _load_model(mj, geometry, layout):
    from tools87.terrain87 import _set_road, _write_scene

    with TemporaryDirectory(prefix="d1-pointcloud100-") as directory:
        path = _write_scene(directory)
        tree = ET.parse(path)
        _add_box(tree.getroot().find("worldbody"), geometry, layout)
        tree.write(path, encoding="utf-8", xml_declaration=True)
        model = mj.MjModel.from_xml_path(str(path))
    _set_road(model, "flat")
    return model


def _wheel_cylinders(model, cylinder_type):
    """Use the same four physical wheel surfaces as the native D1 recorder."""
    wheels = []
    for foot in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        body = model.body(foot).id
        ids = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)
                             & (model.geom_type == cylinder_type))
        if len(ids) != 1:
            raise ValueError(f"expected one collision cylinder on {foot}")
        wheels.append(ids[0])
    return np.array(wheels)


def _grid(geometry):
    """Per mount, order rows by depression, then by azimuth; keep exact native rays."""
    azimuth, depression = np.meshgrid(geometry["azimuth_deg"], geometry["depression_deg"])
    azimuth, depression = azimuth.ravel(), depression.ravel()
    a, d = np.deg2rad(azimuth), np.deg2rad(depression)
    directions = np.column_stack((np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), -np.sin(d)))
    sparse = []
    for angle in geometry["sparse_depression_deg"]:
        selected = np.flatnonzero((azimuth == 0) & (depression == angle))
        if len(selected) != 1:
            raise ValueError(f"native beam at azimuth 0, depression {angle} must exist exactly once")
        sparse.append(selected[0])
    blocked_min, blocked_max = geometry["mask_depression_deg"]
    blocked = ((np.abs(azimuth) <= geometry["mask_abs_azimuth_max_deg"])
               & (depression >= blocked_min) & (depression <= blocked_max))
    n = len(directions)
    sparse = np.array(sparse, dtype=int)
    return directions, np.concatenate((sparse, sparse + n)), np.tile(~blocked, 2)


def _yaw_basis(rotation):
    """Columns are horizontal forward, left and gravity-up unit axes in world space."""
    fx, fy = rotation[:2, 0]
    length = np.hypot(fx, fy)
    if length < 1e-8:
        raise ValueError("body forward direction has no horizontal projection")
    fx, fy = fx / length, fy / length
    return np.array([[fx, -fy, 0.], [fy, fx, 0.], [0., 0., 1.]])


def _rays(mj, model, data, origin, directions, include_static):
    # Use the same primitive as the 87 sensor for every ray, including rays
    # skimming small robot parts. The identity of the hit never leaves here.
    geom_id = np.empty(1, dtype=np.int32)
    distances = np.empty(len(directions))
    for index, direction in enumerate(directions):
        distances[index] = mj.mj_ray(model, data, origin, direction, None, include_static, -1, geom_id)
    return distances


class Sensor100:
    """Shared dense acquisition; sparse and masked views select these same rays."""

    def __init__(self, plan, layout):
        import mujoco

        self.mj = mujoco
        self.geometry = plan["geometry"]
        self.model = _load_model(mujoco, self.geometry, layout)
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("base_link").id
        self.wheels = _wheel_cylinders(self.model, mujoco.mjtGeom.mjGEOM_CYLINDER)
        self.mounts = np.asarray(self.geometry["mounts_m"], dtype=float)
        if self.mounts.shape != (2, 3):
            raise ValueError("D1 replay requires two three-dimensional mount offsets")
        self.directions, self.sparse_indices, self.angle_keep = _grid(self.geometry)
        self.heading = None

    def pose(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("invalid D1 replay pose")
        self.data.qpos[:] = qpos
        self.mj.mj_fwdPosition(self.model, self.data)
        rotation = self.data.xmat[self.base].reshape(3, 3)
        self.heading = _yaw_basis(rotation)
        base_position = self.data.xpos[self.base]
        centers = self.data.geom_xpos[self.wheels]
        front_local = (centers[:2] - base_position) @ self.heading
        query = front_local[:, :2].copy()
        query[:, 0] = front_local[:, 0].mean() + self.geometry["lookahead_m"]
        world_query = query @ self.heading[:2, :2].T + base_position[:2]

        # The cylinder's local z is its axle. The support function gives its
        # maximum extent along world x, including the wheel's contact margin.
        x_components = self.data.geom_xmat[self.wheels, :3]
        radius, half_length = self.model.geom_size[self.wheels, :2].T
        extent = radius * np.hypot(x_components[:, 0], x_components[:, 1])
        extent += half_length * np.abs(x_components[:, 2]) + self.model.geom_margin[self.wheels]
        if self.data.time != 0:
            raise RuntimeError("the replay clock advanced")
        return {"query_xy": query, "world_query_xy": world_query,
                "wheel_max_x_bound": float(np.max(centers[:, 0] + extent))}

    def capture(self):
        if self.heading is None:
            raise RuntimeError("a pose is required before casting rays")
        rotation = self.data.xmat[self.base].reshape(3, 3)
        world_directions = np.ascontiguousarray(self.directions @ rotation.T)
        limits = self.geometry["range_min_m"], self.geometry["range_max_m"]
        ranges, self_hits, points = [], [], []
        for mount in self.mounts:
            origin = self.data.xpos[self.base] + mount @ rotation.T
            measured = _rays(self.mj, self.model, self.data, origin, world_directions, True)
            robot = _rays(self.mj, self.model, self.data, origin, world_directions, False)
            self_hit = (robot >= 0.) & (np.abs(robot - measured) <= 1e-8)
            valid = (measured >= limits[0]) & (measured <= limits[1]) & ~self_hit
            body_points = mount + self.directions * measured[:, None]
            level_points = body_points @ rotation.T @ self.heading
            level_points[~valid] = np.nan
            ranges.append(measured)
            self_hits.append(self_hit)
            points.append(level_points)
        return {"range_m": np.concatenate(ranges), "self_hit": np.concatenate(self_hits),
                "points": np.concatenate(points)}
