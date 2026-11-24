"""Fixed D1 optical sites and a common rail for a longitudinal sensor-position comparison."""
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np

from tools94.common94 import ROOT, read
from tools100.sensor100 import Sensor100, _add_box, _grid, _rays, _wheel_cylinders


CONFIG = ROOT / "docs/experiments107/mount.json"


def mount_positions(arm):
    """Optical centers in base_link coordinates, left then right."""
    config = read(CONFIG)
    if arm not in config["optical_x_m"]:
        raise ValueError(f"Unknown optical position {arm!r}")
    x, y, z = config["optical_x_m"][arm], config["optical_abs_y_m"], config["optical_z_m"]
    return np.array([[x, y, z], [x, -y, z]], dtype=float)


origins = mount_positions


def parts(arm):
    """Named bracket and housing boxes as (lower, upper) corners in base_link meters."""
    config = read(CONFIG)
    result = {}
    for name, corners in config["paired_fixed_boxes_left_min_max_m"].items():
        lo, hi = np.asarray(corners, dtype=float)
        right_lo, right_hi = lo.copy(), hi.copy()
        right_lo[1], right_hi[1] = -hi[1], -lo[1]
        result[f"mount107_{name}_left"] = (lo, hi)
        result[f"mount107_{name}_right"] = (right_lo, right_hi)
    result["mount107_bridge"] = tuple(np.asarray(config["bridge_min_max_m"], dtype=float))
    relative = np.asarray(config["housing_min_max_relative_to_site_m"], dtype=float)
    for side, location in zip(("left", "right"), mount_positions(arm), strict=True):
        result[f"mount107_housing_{side}"] = tuple(relative + location)
    return result


def _vec(values):
    return " ".join(format(float(v), ".12g") for v in values)


def _save_new(tree, path):
    with path.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)


def _mounted_scene(source_scene, geometry, layout, arm):
    """Write uniquely named siblings in the temporary scene, leaving the robot.xml symlink untouched."""
    source_scene = Path(source_scene)
    scene = ET.parse(source_scene)
    include, = scene.getroot().findall("include[@file='robot.xml']")
    original = source_scene.parent / include.attrib["file"]
    robot = ET.parse(original)
    base, = robot.getroot().findall("./worldbody/body[@name='base_link']")
    for name, (lo, hi) in parts(arm).items():
        # Non-contact visual geoms on the free base remain visible to both mj_ray passes.
        ET.SubElement(base, "geom", name=name, type="box", pos=_vec((lo + hi) / 2), size=_vec((hi - lo) / 2),
                      contype="0", conaffinity="0", group="1", density="0", margin="0", rgba="0.18 0.25 0.32 1")
    config = read(CONFIG)
    for name, origin in zip(config["site_names"], mount_positions(arm), strict=True):
        ET.SubElement(base, "site", name=name, pos=_vec(origin), quat=_vec(config["site_quat_wxyz"]),
                      size="0.001", rgba="0.2 0.8 1 1")
    mounted_robot = source_scene.with_name("robot107.xml")
    _save_new(robot, mounted_robot)
    include.set("file", mounted_robot.name)
    _add_box(scene.find("worldbody"), geometry, layout)
    mounted = source_scene.with_name("scene107.xml")
    _save_new(scene, mounted)
    return mounted


class _RayGeomIds:
    """Observe the geom ID written by each original mj_ray call."""

    def __init__(self, mj):
        self.mj = mj
        self.ids = []

    def mj_ray(self, *args):
        distance = self.mj.mj_ray(*args)
        self.ids.append(int(args[-1][0]) if distance >= 0. else -1)
        return distance


def _box_separations(centers, rotations, half_sizes, static_centers, static_half_sizes):
    """Lower bound on box-to-box Euclidean distances from separating-axis projection gaps."""
    common = np.broadcast_to(np.eye(3), (len(centers), 3, 3))
    local = rotations.swapaxes(1, 2)
    cross = np.cross(common[:, :, None], local[:, None, :]).reshape(len(centers), 9, 3)
    norms = np.linalg.norm(cross, axis=2, keepdims=True)
    cross = np.divide(cross, norms, out=np.zeros_like(cross), where=norms > 1e-12)
    axes = np.concatenate((common, local, cross), axis=1)
    robot_projection = np.einsum("gak,gk->ga", np.abs(axes @ rotations), half_sizes)
    rig_projection = np.einsum("gak,rk->gra", np.abs(axes), static_half_sizes)
    delta = centers[:, None] - static_centers[None, :]
    gaps = np.abs(np.einsum("grk,gak->gra", delta, axes)) - robot_projection[:, None] - rig_projection
    return np.maximum(gaps.max(axis=2), 0.)


class MountedSensor(Sensor100):
    """Pose-only round-100 sensor with two actual emitter sites in a derived scene."""

    def __init__(self, geometry, layout, arm):
        import mujoco
        from tools87.terrain87 import _set_road, _write_scene

        self.mj = mujoco
        self.geometry = geometry
        self.mounts = mount_positions(arm)
        with TemporaryDirectory(prefix="d1-mount107-") as directory:
            scene = _mounted_scene(_write_scene(directory), geometry, layout, arm)
            self.model = mujoco.MjModel.from_xml_path(str(scene))
        _set_road(self.model, "flat")
        self.data = mujoco.MjData(self.model)
        self.base = self.model.body("base_link").id
        self.wheels = _wheel_cylinders(self.model, mujoco.mjtGeom.mjGEOM_CYLINDER)
        self.directions, self.sparse_indices, self.angle_keep = _grid(geometry)
        self.heading = None
        self.site_ids = np.array([self.model.site(name).id for name in read(CONFIG)["site_names"]])
        self.rig_names = tuple(parts(arm))
        self.rig_ids = np.array([self.model.geom(name).id for name in self.rig_names])
        np.testing.assert_allclose(self.model.site_pos[self.site_ids], self.mounts, rtol=0., atol=1e-12)
        if not np.all(self.model.site_bodyid[self.site_ids] == self.base) or not np.all(
                self.model.geom_bodyid[self.rig_ids] == self.base):
            raise ValueError("mount sites and structural parts must be rigidly attached to base_link")
        boxes = np.asarray(list(parts(arm).values()))
        self._rig_centers = boxes.mean(axis=1)
        self._rig_half_sizes = (boxes[:, 1] - boxes[:, 0]) / 2
        self._moving_geometry()

    def _moving_geometry(self):
        m, mj = self.model, self.mj
        leg_parts = {f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR")
                     for part in ("hip", "thigh", "calf", "foot")}
        body_ids = [m.body(name).id for name in sorted(leg_parts)]
        self._moving = np.flatnonzero(np.isin(m.geom_bodyid, body_ids))
        center, half_sizes, names, visual = [], [], [], []
        for idx in self._moving:
            kind = m.geom_type[idx]
            is_mesh = kind == mj.mjtGeom.mjGEOM_MESH
            if is_mesh:
                mesh = m.geom_dataid[idx]
                start = m.mesh_vertadr[mesh]
                vertices = m.mesh_vert[start:start + m.mesh_vertnum[mesh]].astype(np.float64)
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                center.append((lo + hi) / 2)
                half_sizes.append((hi - lo) / 2)
                label = "mesh"
            else:
                center.append(np.zeros(3))
                if kind == mj.mjtGeom.mjGEOM_BOX:
                    half_sizes.append(m.geom_size[idx].copy())
                    label = "box"
                elif kind == mj.mjtGeom.mjGEOM_CYLINDER:
                    radius, half_length = m.geom_size[idx, :2]
                    half_sizes.append([radius, radius, half_length])
                    label = "cylinder"
                else:
                    raise ValueError(f"unsupported moving geometry {idx}")
            if is_mesh != (m.geom_contype[idx] == 0 and m.geom_conaffinity[idx] == 0):
                raise ValueError("native D1 visual/collision geometry classes changed")
            names.append(f"{m.body(m.geom_bodyid[idx]).name}/{label}#{idx}")
            visual.append(is_mesh)
        self._geom_centers, self._geom_half = np.array(center), np.array(half_sizes)
        self._geom_names, self._visual = np.array(names), np.array(visual)
        if self._visual.sum() != len(body_ids) or self._visual.all():
            raise ValueError("incomplete D1 moving geometry")

    def clearance_m(self):
        """Bound clearance to the complete moving leg/wheel geometry at the current archived pose."""
        if self.heading is None:
            raise ValueError("set a replay pose before inspecting clearance")
        rot = self.data.xmat[self.base].reshape(3, 3)
        orientations = rot.T @ self.data.geom_xmat[self._moving].reshape(-1, 3, 3)
        centers = (self.data.geom_xpos[self._moving] - self.data.xpos[self.base]) @ rot
        centers += np.einsum("gij,gj->gi", orientations, self._geom_centers)
        separation = _box_separations(centers, orientations, self._geom_half, self._rig_centers, self._rig_half_sizes)
        clearance = np.maximum(0., separation - self.model.geom_margin[self._moving, None])
        result = {"minimum_m": float(clearance.min())}
        for name, mask in (("visual", self._visual), ("collision", ~self._visual)):
            ids = np.flatnonzero(mask)
            i, j = np.unravel_index(np.argmin(clearance[ids]), (len(ids), len(self.rig_names)))
            result[f"{name}_m"] = float(clearance[ids[i], j])
            result[f"closest_{name}"] = [str(self._geom_names[ids[i]]), self.rig_names[j]]
        return result

    def capture(self):
        if self.heading is None or self.data.time != 0.:
            raise RuntimeError("a zero-time replay pose is required before capture")
        ranges, self_hits, bracket_hits, points = [], [], [], []
        for site in self.site_ids:
            origin = self.data.site_xpos[site]
            rotation = self.data.site_xmat[site].reshape(3, 3)
            directions = np.ascontiguousarray(self.directions @ rotation.T)
            ids = _RayGeomIds(self.mj)
            measured = _rays(ids, self.model, self.data, origin, directions, True)
            robot = _rays(self.mj, self.model, self.data, origin, directions, False)
            self_hit = (robot >= 0.) & (np.abs(robot - measured) <= 1e-8)
            valid = (measured >= self.geometry["range_min_m"]) & (measured <= self.geometry["range_max_m"]) & ~self_hit
            leveled = (origin - self.data.xpos[self.base] + measured[:, None] * directions) @ self.heading
            leveled[~valid] = np.nan
            ranges.append(measured)
            self_hits.append(self_hit)
            bracket_hits.append(self_hit & np.isin(ids.ids, self.rig_ids))
            points.append(leveled)
        return dict(range_m=np.concatenate(ranges), self_hit=np.concatenate(self_hits),
                    bracket_hit=np.concatenate(bracket_hits), points=np.concatenate(points))
