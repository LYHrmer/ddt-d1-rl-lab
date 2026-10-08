#!/usr/bin/env python3
"""Static, reproducible MuJoCo 3.3 validation of D1 terrain course 03.

Loads and forwards models only. No integration, viewer, or policy inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "src/ddt_ros2_control/urdfs/d1_description/mujoco"
DEFAULT_SCENE = SCENE_DIR / "scene_training_course_03.xml"
DEFAULT_REFERENCE = SCENE_DIR / "scene_contract_plane_02.xml"
DEFAULT_MANIFEST = SCENE_DIR / "training_course_03_assets/layout_manifest.json"
DEFAULT_OUTPUT = ROOT / "evidence/terrain_course_03/static_validation.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def model_names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) for i in range(count)]


def compare_arrays(reference, terrain, fields):
    differences = {}
    for field in fields:
        before = np.asarray(getattr(reference, field))
        after = np.asarray(getattr(terrain, field))
        if before.shape != after.shape or not np.array_equal(before, after):
            differences[field] = {"reference_shape": list(before.shape),
                                  "terrain_shape": list(after.shape),
                                  "max_abs_difference": (float(np.max(np.abs(before - after)))
                                                         if before.shape == after.shape and
                                                         np.issubdtype(before.dtype, np.number) else None)}
    return differences


def geom(model, name):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if index < 0:
        raise ValueError(f"Missing geom: {name}")
    return index


def top_end(data, model, name, side):
    """World xyz at a tilted box's upper face, at its -x or +x endpoint."""
    i = geom(model, name)
    size = model.geom_size[i]
    local = np.array([side * size[0], 0.0, size[2]])
    return data.geom_xpos[i] + data.geom_xmat[i].reshape(3, 3) @ local


def box_top(data, model, name, side):
    return top_end(data, model, name, side)


def footprint_distance_and_lane(model, data, manifest):
    lane = manifest["config"]["straight_lane"]["bounds_xy_m"]
    spawn_radius = manifest["spawn_clear_radius_m"]
    nearest = (math.inf, None)
    lane_hits = []
    for item in manifest["geometries"]:
        if not item["collision"]:
            continue
        name = item["name"]
        i = geom(model, name)
        position = data.geom_xpos[i]
        size = model.geom_size[i]
        kind = item["type"]
        if kind == "box":
            rotation = np.abs(data.geom_xmat[i].reshape(3, 3))
            half_x = float(rotation[0] @ size)
            half_y = float(rotation[1] @ size)
        elif kind == "cylinder":
            half_x = half_y = float(size[0])
        elif kind == "hfield":
            asset = model.geom_dataid[i]
            half_x, half_y = map(float, model.hfield_size[asset, :2])
        else:
            raise ValueError(f"Unexpected obstacle geom type: {kind}")
        xlo, xhi = float(position[0] - half_x), float(position[0] + half_x)
        ylo, yhi = float(position[1] - half_y), float(position[1] + half_y)
        distance = math.hypot(max(xlo, -xhi, 0), max(ylo, -yhi, 0))
        if distance < nearest[0]:
            nearest = (distance, name)
        if xhi > lane[0] and xlo < lane[2] and yhi > lane[1] and ylo < lane[3]:
            lane_hits.append(name)
    return {"nearest_obstacle_m": round(nearest[0], 9),
            "nearest_obstacle_name": nearest[1],
            "required_clear_radius_m": spawn_radius,
            "straight_lane_length_m": lane[2] - lane[0],
            "straight_lane_obstacle_names": lane_hits,
            "pass": nearest[0] >= spawn_radius and lane[2] - lane[0] >= 16 and not lane_hits}


def inspect(scene, reference, manifest_path):
    if mujoco.__version__ != "3.3.0":
        raise RuntimeError(f"MuJoCo 3.3.0 required, got {mujoco.__version__}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terrain = mujoco.MjModel.from_xml_path(str(scene))
    baseline = mujoco.MjModel.from_xml_path(str(reference))
    data = mujoco.MjData(terrain)
    mujoco.mj_forward(terrain, data)
    checks = {}

    source_root = scene.parent.parent
    source_mismatch = []
    for relative, expected in manifest["source_asset_sha256_before_and_after"].items():
        path = source_root / relative
        if not path.is_file() or digest(path) != expected:
            source_mismatch.append(relative)
    checks["source_assets"] = {"pass": not source_mismatch,
                               "count": len(manifest["source_asset_sha256_before_and_after"]),
                               "mismatches": source_mismatch}

    body_fields = ["body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                   "body_ipos", "body_iquat", "body_mocapid", "body_weldid"]
    body_diff = compare_arrays(baseline, terrain, body_fields)
    body_names_equal = model_names(baseline, mujoco.mjtObj.mjOBJ_BODY, baseline.nbody) == model_names(
        terrain, mujoco.mjtObj.mjOBJ_BODY, terrain.nbody)
    checks["robot_bodies"] = {"pass": baseline.nbody == terrain.nbody and body_names_equal and not body_diff,
                              "count": terrain.nbody, "names_equal": body_names_equal,
                              "different_fields": body_diff}

    joint_fields = ["jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
                    "dof_damping", "dof_armature", "dof_frictionloss"]
    joint_diff = compare_arrays(baseline, terrain, joint_fields)
    joint_names_equal = model_names(baseline, mujoco.mjtObj.mjOBJ_JOINT, baseline.njnt) == model_names(
        terrain, mujoco.mjtObj.mjOBJ_JOINT, terrain.njnt)
    checks["robot_joints"] = {"pass": baseline.njnt == terrain.njnt and joint_names_equal and not joint_diff,
                              "count": terrain.njnt, "names_equal": joint_names_equal,
                              "different_fields": joint_diff}

    sensor_fields = ["sensor_type", "sensor_objtype", "sensor_objid", "sensor_reftype",
                     "sensor_refid", "sensor_dim", "sensor_cutoff", "sensor_noise"]
    sensor_diff = compare_arrays(baseline, terrain, sensor_fields)
    sensor_names_equal = model_names(baseline, mujoco.mjtObj.mjOBJ_SENSOR, baseline.nsensor) == model_names(
        terrain, mujoco.mjtObj.mjOBJ_SENSOR, terrain.nsensor)
    checks["robot_sensors"] = {"pass": baseline.nsensor == terrain.nsensor and sensor_names_equal and not sensor_diff,
                               "count": terrain.nsensor, "names_equal": sensor_names_equal,
                               "different_fields": sensor_diff}

    option_fields = ["timestep", "gravity", "integrator", "solver", "cone", "jacobian",
                     "impratio", "iterations", "ls_iterations", "noslip_iterations",
                     "tolerance", "density", "viscosity", "wind", "magnetic",
                     "o_margin", "o_solref", "o_solimp", "disableflags", "enableflags"]
    option_diff = compare_arrays(baseline.opt, terrain.opt, option_fields)
    checks["physics_options"] = {"pass": not option_diff, "different_fields": option_diff,
                                 "timestep_s": float(terrain.opt.timestep),
                                 "gravity_m_s2": native(terrain.opt.gravity)}

    reference_robot_geoms = np.flatnonzero(baseline.geom_bodyid != 0)
    terrain_robot_geoms = np.flatnonzero(terrain.geom_bodyid != 0)
    robot_geom_fields = ["geom_friction", "geom_margin", "geom_solref", "geom_solimp",
                         "geom_contype", "geom_conaffinity", "geom_condim"]
    geom_diff = {}
    for field in robot_geom_fields:
        a = np.asarray(getattr(baseline, field))[reference_robot_geoms]
        b = np.asarray(getattr(terrain, field))[terrain_robot_geoms]
        if not np.array_equal(a, b):
            geom_diff[field] = {"reference_shape": list(a.shape), "terrain_shape": list(b.shape)}
    floor_same = np.array_equal(baseline.geom_friction[geom(baseline, "floor")],
                                terrain.geom_friction[geom(terrain, "course03_floor")])
    checks["original_contact_parameters"] = {
        "pass": len(reference_robot_geoms) == len(terrain_robot_geoms) and not geom_diff and floor_same,
        "robot_geom_count": len(terrain_robot_geoms), "different_fields": geom_diff,
        "floor_friction_equal": floor_same,
        "ground_friction": native(terrain.geom_friction[geom(terrain, "course03_floor")])}

    marker_bad = []
    obstacle_bad = []
    for item in manifest["geometries"]:
        i = geom(terrain, item["name"])
        if item["collision"]:
            if terrain.geom_contype[i] != 1 or terrain.geom_conaffinity[i] != 1:
                obstacle_bad.append(item["name"])
        elif terrain.geom_contype[i] != 0 or terrain.geom_conaffinity[i] != 0:
            marker_bad.append(item["name"])
    checks["contact_flags"] = {"pass": not marker_bad and not obstacle_bad,
                               "noncolliding_mark_count": manifest["noncolliding_marking_geom_count"],
                               "colliding_obstacle_count": manifest["colliding_course_geom_count"],
                               "bad_marks": marker_bad, "bad_obstacles": obstacle_bad}

    ramps = []
    for x0, angle in zip(manifest["config"]["ramps"]["start_x_m"],
                         manifest["config"]["ramps"]["angles_deg"]):
        run = manifest["config"]["ramps"]["run_m"]
        deck = manifest["config"]["ramps"]["platform_length_m"]
        h = run * math.tan(math.radians(angle))
        points = [top_end(data, terrain, f"ramp_{angle}deg_up", -1),
                  top_end(data, terrain, f"ramp_{angle}deg_up", 1),
                  box_top(data, terrain, f"ramp_{angle}deg_deck", -1),
                  box_top(data, terrain, f"ramp_{angle}deg_deck", 1),
                  top_end(data, terrain, f"ramp_{angle}deg_down", -1),
                  top_end(data, terrain, f"ramp_{angle}deg_down", 1)]
        errors = [np.linalg.norm(points[0][[0, 2]] - [x0, 0]),
                  np.linalg.norm(points[1] - points[2]),
                  np.linalg.norm(points[3] - points[4]),
                  np.linalg.norm(points[5][[0, 2]] - [x0 + 2 * run + deck, 0]),
                  abs(points[1][2] - h)]
        ramps.append({"angle_deg": angle, "rise_m": round(h, 9),
                      "maximum_endpoint_or_joint_error_m": float(max(errors)),
                      "pass": max(errors) <= 1e-6})
    checks["ramp_surface_continuity"] = {"pass": all(r["pass"] for r in ramps),
                                         "tolerance_m": 1e-6, "ramps": ramps}

    steps = []
    for expected in manifest["config"]["single_steps"]["heights_m"]:
        i = geom(terrain, f"step_{int(round(expected * 1000)):02d}mm")
        measured = float(data.geom_xpos[i, 2] + terrain.geom_size[i, 2])
        steps.append({"expected_m": expected, "compiled_m": measured,
                      "pass": abs(measured - expected) <= 1e-9})
    staircases = []
    for riser in manifest["config"]["staircases"]["riser_m"]:
        heights = []
        for tread in range(1, 2 * manifest["config"]["staircases"]["up_treads"] + 1):
            i = geom(terrain, f"stairs_{int(round(riser * 1000))}mm_tread_{tread}")
            heights.append(float(data.geom_xpos[i, 2] + terrain.geom_size[i, 2]))
        n = manifest["config"]["staircases"]["up_treads"]
        target = [min(k + 1, 2 * n - k) * riser for k in range(2 * n)]
        staircases.append({"riser_m": riser, "top_heights_m": heights,
                           "pass": bool(np.allclose(heights, target, rtol=0, atol=1e-9))})
    checks["step_heights"] = {"pass": all(s["pass"] for s in steps + staircases),
                              "single_steps": steps, "staircases": staircases}

    gates = []
    for index, expected in enumerate(manifest["config"]["gates"]["clear_width_m"], 1):
        south = geom(terrain, f"gate_{index}_south")
        north = geom(terrain, f"gate_{index}_north")
        clear = float(data.geom_xpos[north, 1] - data.geom_xpos[south, 1] -
                      terrain.geom_size[north, 0] - terrain.geom_size[south, 0])
        gates.append({"expected_m": expected, "compiled_m": clear,
                      "pass": abs(clear - expected) <= 1e-9})
    checks["gate_clear_widths"] = {"pass": all(g["pass"] for g in gates), "gates": gates}

    fields = []
    for index, expected in enumerate(manifest["config"]["rough_tiles"]["max_height_m"], 1):
        name = f"rough_{int(round(expected * 1000))}mm"
        field_id = mujoco.mj_name2id(terrain, mujoco.mjtObj.mjOBJ_HFIELD, name)
        start, rows, cols = map(int, (terrain.hfield_adr[field_id], terrain.hfield_nrow[field_id],
                                      terrain.hfield_ncol[field_id]))
        values = terrain.hfield_data[start:start + rows * cols].reshape(rows, cols)
        image = np.asarray(Image.open(scene.parent / "training_course_03_assets" / f"{name}.png").convert("L"))
        edges_zero = bool(np.all(values[0] == 0) and np.all(values[-1] == 0) and
                          np.all(values[:, 0] == 0) and np.all(values[:, -1] == 0))
        image_edges_zero = bool(np.all(image[0] == 0) and np.all(image[-1] == 0) and
                                np.all(image[:, 0] == 0) and np.all(image[:, -1] == 0))
        minimum, maximum = float(values.min()), float(values.max())
        height = float(terrain.hfield_size[field_id, 2])
        fields.append({"name": name, "resolution": [rows, cols], "normalized_min": minimum,
                       "normalized_max": maximum, "border_zero": edges_zero,
                       "png_border_zero": image_edges_zero, "physical_height_m": height,
                       "pass": minimum == 0 and maximum == 1 and edges_zero and image_edges_zero
                       and abs(height - expected) <= 1e-9})
    checks["colliding_heightfields"] = {"pass": all(f["pass"] for f in fields), "fields": fields}

    checks["spawn_and_straight_lane"] = footprint_distance_and_lane(terrain, data, manifest)
    report = {
        "description": "Static model validation only; mj_forward was called, no mj_step or policy traversal",
        "mujoco_version": mujoco.__version__,
        "inputs_sha256": {"scene": digest(scene), "reference": digest(reference),
                          "manifest": digest(manifest_path)},
        "compiled_counts": {"terrain_geoms_including_robot": terrain.ngeom,
                            "reference_geoms_including_robot": baseline.ngeom,
                            "bodies": terrain.nbody, "joints": terrain.njnt,
                            "sensors": terrain.nsensor, "heightfields": terrain.nhfield},
        "checks": checks,
        "pass": all(check["pass"] for check in checks.values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = inspect(args.scene.resolve(), args.reference.resolve(), args.manifest.resolve())
    serialized = json.dumps(report, ensure_ascii=False, indent=2, default=native) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(f"Wrote {args.output}: pass={report['pass']}")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
