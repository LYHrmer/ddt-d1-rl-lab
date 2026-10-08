#!/usr/bin/env python3
"""Generate a deterministic, partitioned D1 MuJoCo terrain course.

This script only writes new files. It does not load MuJoCo or run physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("course03_layout.json")
DEFAULT_SCENE_DIR = ROOT / "src/ddt_ros2_control/urdfs/d1_description/mujoco"

COLORS = {
    "easy": "0.22 0.75 0.51 1",
    "moderate": "0.96 0.68 0.18 1",
    "hard": "0.94 0.33 0.28 1",
    "neutral": "0.47 0.58 0.67 1",
    "marking": "0.75 0.83 0.92 1",
}


def fmt(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config):
    for key in ("scene_name", "asset_directory"):
        value = config[key]
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError(f"{key} must be a single filename/directory name")
    if config["scene_name"] != "scene_training_course_03.xml":
        raise ValueError("This generator only creates scene_training_course_03.xml")
    if config["asset_directory"] != "training_course_03_assets":
        raise ValueError("This generator only creates training_course_03_assets")
    lane = config["straight_lane"]
    x0, y0, x1, y1 = lane["bounds_xy_m"]
    if x1 - x0 < 16 or y0 >= 0 or y1 <= 0 or lane["length_m"] != x1 - x0:
        raise ValueError("straight lane must be at least 16 m long and centered across y=0")
    for section, keys in (("single_steps", ("centers_x_m", "heights_m")),
                          ("ramps", ("start_x_m", "angles_deg")),
                          ("staircases", ("start_x_m", "riser_m")),
                          ("rough_tiles", ("center_x_m", "max_height_m")),
                          ("gates", ("center_x_m", "clear_width_m"))):
        if len(config[section][keys[0]]) != len(config[section][keys[1]]):
            raise ValueError(f"{section} has unequal {keys[0]} and {keys[1]} lengths")
    if config["spawn_clear_radius_m"] < 1.2:
        raise ValueError("spawn clear radius must be at least 1.2 m")
    if config["rough_tiles"]["resolution_px"] < 3:
        raise ValueError("heightfield image resolution must be at least 3 pixels")


def protected_assets(scene_dir):
    robot = scene_dir / "robot.xml"
    if not robot.is_file():
        raise ValueError(f"robot.xml not found: {robot}")
    files = [robot]
    original_scene = scene_dir / "scene.xml"
    if original_scene.is_file():
        files.append(original_scene)
    compiler = ET.parse(robot).getroot().find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for mesh in ET.parse(robot).getroot().findall(".//mesh[@file]"):
        mesh_path = (scene_dir / meshdir / mesh.get("file")).resolve()
        if not mesh_path.is_file():
            raise ValueError(f"Referenced robot mesh not found: {mesh_path}")
        files.append(mesh_path)
    return {os.path.relpath(path, scene_dir.parent.resolve()): sha256(path)
            for path in sorted(set(files))}


def bounds(cx, cy, sx, sy):
    return [round(cx - sx / 2, 4), round(cy - sy / 2, 4),
            round(cx + sx / 2, 4), round(cy + sy / 2, 4)]


def height_image(seed, pixels, fade):
    """Smooth deterministic bumps with an exactly flat zero-height border."""
    rng = np.random.default_rng(seed)
    field = np.zeros((pixels, pixels), dtype=float)
    for n, weight in ((7, 0.55), (15, 0.30), (31, 0.15)):
        coarse = (rng.random((n, n)) * 255).astype(np.uint8)
        resized = Image.fromarray(coarse).resize((pixels, pixels), Image.Resampling.BICUBIC)
        field += weight * np.asarray(resized, dtype=float) / 255.0
    axis = np.linspace(-1, 1, pixels)
    xx, yy = np.meshgrid(axis, axis)
    distance_to_edge = 1.0 - np.maximum(np.abs(xx), np.abs(yy))
    edge = np.clip(distance_to_edge / fade, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    field *= edge
    field /= field.max()
    image = np.rint(field * 255).astype(np.uint8)
    image[0, :] = image[-1, :] = image[:, 0] = image[:, -1] = 0
    return Image.fromarray(image, "L")


def build(config, scene_dir, source_hashes):
    asset_name = config["asset_directory"]
    root = ET.Element("mujoco", model="D1 training course 03")
    root.append(ET.Comment("Generated terrain; original robot and contact parameters are unchanged."))
    ET.SubElement(root, "include", file="robot.xml")
    ET.SubElement(root, "statistic", center="8 0 0.2", extent="18")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.12 0.18 0.26 1")
    ET.SubElement(visual, "global", azimuth="-125", elevation="-50", fovy="45")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0.02 0.04 0.08", width="512", height="3072")
    ET.SubElement(assets, "texture", type="2d", name="course03_grid",
                  builtin="checker", mark="edge", rgb1="0.13 0.19 0.25",
                  rgb2="0.17 0.24 0.30", markrgb="0.28 0.37 0.44",
                  width="256", height="256")
    ET.SubElement(assets, "material", name="course03_ground", texture="course03_grid",
                  texuniform="true", texrepeat="25 25", reflectance="0.12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="6 -5 12", dir="0 0 -1", directional="true")
    ET.SubElement(world, "light", pos="12 8 10", dir="0 0 -1", directional="true", diffuse="0.35 0.38 0.45")
    ET.SubElement(world, "geom", name="course03_floor", type="plane", size="0 0 0.05", material="course03_ground")
    geoms = []
    zones = []
    images = {}

    def geom(name, kind, pos, size=None, *, color="neutral", collide=True, **extra):
        props = {"name": name, "type": kind, "pos": fmt(pos),
                 "rgba": COLORS.get(color, color),
                 "contype": "1" if collide else "0",
                 "conaffinity": "1" if collide else "0"}
        if size is not None:
            props["size"] = fmt(size)
        props.update({key: str(value) for key, value in extra.items()})
        ET.SubElement(world, "geom", **props)
        geoms.append({"name": name, "type": kind, "pos_m": list(pos),
                      "size_m": list(size) if size is not None else None,
                      "collision": collide})

    def zone(name, zone_bounds, difficulty, direction, notes, parameters=None):
        zones.append({"name": name, "bounds_xy_m": [round(v, 4) for v in zone_bounds],
                      "difficulty": difficulty, "recommended_approach": direction,
                      "notes": notes, "parameters": parameters or {}})

    def line(name, x, y, sx, sy, color="marking"):
        geom(name, "box", (x, y, 0.002), (sx, sy, 0.001), color=color, collide=False)

    # Flat route along +x and a broad, unencumbered starting area.
    lane = config["straight_lane"]
    xmin, ymin, xmax, ymax = lane["bounds_xy_m"]
    for y, side in ((ymin, "south"), (ymax, "north")):
        line(f"lane_{side}_edge", (xmin + xmax) / 2, y, (xmax - xmin) / 2, 0.02)
    for x in range(math.ceil(xmin) + 1, math.floor(xmax), 2):
        line(f"lane_tick_{x}", x, 0, 0.16, 0.025)
    for x, label in ((xmin + 0.35, "start"), (xmax - 0.35, "finish")):
        line(f"lane_{label}", x, 0, 0.025, 0.45, color="easy")
    zone("straight_acceleration_braking", [xmin, ymin, xmax, ymax], "easy", "+x",
         "Level ground; 20 m acceleration and braking lane with non-colliding marks.",
         {"length_m": lane["length_m"], "width_m": ymax - ymin})

    steps = config["single_steps"]
    for index, (x, h) in enumerate(zip(steps["centers_x_m"], steps["heights_m"]), 1):
        y = steps["center_y_m"]
        geom(f"step_{int(round(h * 1000)):02d}mm", "box", (x, y, h / 2),
             (steps["length_m"] / 2, steps["width_m"] / 2, h / 2),
             color="easy" if h <= 0.015 else ("moderate" if h <= 0.03 else "hard"))
        line(f"step_entry_{index}", x - 0.65, y, 0.012, steps["width_m"] / 2, color="easy")
        zone(f"single_step_{int(round(h * 1000))}mm",
             bounds(x, y, 1.65, 2.1), "easy" if h <= 0.015 else ("moderate" if h <= 0.03 else "hard"),
             "+x; approach each bay separately from level ground",
             "Discrete solid box step; exit is level ground.", {"height_m": h})

    def ramp(name, x0, x1, y, z0, z1, width, thick, color):
        run = x1 - x0
        slope = math.atan2(z1 - z0, run)
        # Upper face endpoints meet the ground or deck precisely.
        center_x = (x0 + x1) / 2 + math.sin(slope) * thick / 2
        center_z = (z0 + z1) / 2 - math.cos(slope) * thick / 2
        quat = (math.cos(slope / 2), 0, -math.sin(slope / 2), 0)
        geom(name, "box", (center_x, y, center_z),
             (math.hypot(run, z1 - z0) / 2, width / 2, thick / 2),
             color=color, quat=fmt(quat))

    ramps = config["ramps"]
    for index, (x0, angle) in enumerate(zip(ramps["start_x_m"], ramps["angles_deg"]), 1):
        run, deck_len = ramps["run_m"], ramps["platform_length_m"]
        y, width, thick = ramps["center_y_m"], ramps["width_m"], ramps["thickness_m"]
        h = run * math.tan(math.radians(angle))
        color = "easy" if angle == 3 else ("moderate" if angle == 6 else "hard")
        ramp(f"ramp_{angle}deg_up", x0, x0 + run, y, 0, h, width, thick, color)
        geom(f"ramp_{angle}deg_deck", "box", (x0 + run + deck_len / 2, y, h - thick / 2),
             (deck_len / 2, width / 2, thick / 2), color=color)
        ramp(f"ramp_{angle}deg_down", x0 + run + deck_len,
             x0 + 2 * run + deck_len, y, h, 0, width, thick, color)
        zone(f"ramp_{angle}deg", bounds(x0 + run + deck_len / 2, y,
             2 * run + deck_len + 0.4, width + 0.8), color, "+x",
             "Continuous top surfaces at entrance, deck joins and exit.",
             {"angle_deg": angle, "rise_m": round(h, 5), "run_m": run,
              "deck_length_m": deck_len})

    stairs = config["staircases"]
    n = stairs["up_treads"]
    for x0, riser in zip(stairs["start_x_m"], stairs["riser_m"]):
        y, tread, width = stairs["center_y_m"], stairs["tread_m"], stairs["width_m"]
        difficulty = "easy" if riser == 0.015 else ("moderate" if riser == 0.03 else "hard")
        for i in range(2 * n):
            level = i + 1 if i < n else 2 * n - i
            h = level * riser
            geom(f"stairs_{int(round(riser * 1000))}mm_tread_{i + 1}", "box",
                 (x0 + (i + 0.5) * tread, y, h / 2),
                 (tread / 2, width / 2, h / 2), color=difficulty)
        zone(f"stairs_{int(round(riser * 1000))}mm_riser",
             bounds(x0 + n * tread, y, 2 * n * tread + 0.4, width + 0.8),
             difficulty, "+x", "Three rising and three descending treads; first and final risers match the named height.",
             {"riser_m": riser, "tread_m": tread, "peak_height_m": n * riser})

    rough = config["rough_tiles"]
    robot_compiler = ET.parse(scene_dir / "robot.xml").getroot().find("compiler")
    meshdir = robot_compiler.get("meshdir", "") if robot_compiler is not None else ""
    for i, (x, height) in enumerate(zip(rough["center_x_m"], rough["max_height_m"]), 1):
        name = f"rough_{int(round(height * 1000))}mm"
        filename = name + ".png"
        images[filename] = height_image(config["seed"] + i, rough["resolution_px"], rough["edge_fade_fraction"])
        relative = os.path.relpath(scene_dir / asset_name / filename, scene_dir / meshdir)
        sx, sy = rough["size_xy_m"]
        ET.SubElement(assets, "hfield", name=name, file=relative,
                      size=fmt((sx / 2, sy / 2, height, 0.005)))
        geom(f"{name}_surface", "hfield", (x, rough["center_y_m"], 0),
             color="easy" if i == 1 else ("moderate" if i == 2 else "hard"), hfield=name)
        zone(name, bounds(x, rough["center_y_m"], sx + 0.5, sy + 0.5),
             "easy" if i == 1 else ("moderate" if i == 2 else "hard"), "+x",
             "Colliding PNG heightfield with a zero-height border; maximum height is nominal.",
             {"max_height_m": height, "resolution_px": rough["resolution_px"], "image": filename})

    plates = config["broken_plates"]
    px0, py0, px1, py1 = plates["bounds_xy_m"]
    rng = np.random.default_rng(config["seed"] + 100)
    # Two offset rows leave a path among independently colliding fixed plates.
    for i in range(plates["count"]):
        row, column = divmod(i, (plates["count"] + 1) // 2)
        x = px0 + 0.4 + column * 0.57 + float(rng.uniform(-0.07, 0.07))
        y = py0 + 0.43 + row * 1.13 + float(rng.uniform(-0.12, 0.12))
        length = float(rng.uniform(*plates["length_range_m"]))
        width = float(rng.uniform(*plates["width_range_m"]))
        h = float(rng.uniform(*plates["height_range_m"]))
        angle = float(rng.uniform(-0.35, 0.35))
        geom(f"fixed_plate_{i + 1:02d}", "box", (x, y, h / 2),
             (length / 2, width / 2, h / 2), color="moderate",
             quat=fmt((math.cos(angle / 2), 0, 0, math.sin(angle / 2))))
    zone("fixed_broken_plates", plates["bounds_xy_m"], "moderate", "+x or -x",
         "Sparse, fixed, individually colliding low boxes with flat gaps; deterministic from seed.",
         {"count": plates["count"], "height_range_m": plates["height_range_m"]})

    slalom = config["slalom"]
    for i in range(slalom["count"]):
        x = slalom["start_x_m"] + i * slalom["spacing_x_m"]
        y = slalom["center_y_m"] + (-1 if i % 2 else 1) * slalom["offset_y_m"]
        h = slalom["post_height_m"]
        geom(f"slalom_post_{i + 1:02d}", "cylinder", (x, y, h / 2),
             (slalom["post_radius_m"], h / 2), color="moderate")
    zone("slalom", bounds(slalom["start_x_m"] + (slalom["count"] - 1) *
         slalom["spacing_x_m"] / 2, slalom["center_y_m"], 8.3, 2.3),
         "moderate", "+x or -x", "Alternating fixed round posts with open bypass on both sides.",
         {"count": slalom["count"], "post_spacing_m": slalom["spacing_x_m"]})

    gates = config["gates"]
    for i, (x, clear) in enumerate(zip(gates["center_x_m"], gates["clear_width_m"]), 1):
        for side in (-1, 1):
            y = gates["center_y_m"] + side * (clear / 2 + gates["post_radius_m"])
            h = gates["post_height_m"]
            geom(f"gate_{i}_{'south' if side < 0 else 'north'}", "cylinder",
                 (x, y, h / 2), (gates["post_radius_m"], h / 2),
                 color="easy" if i == 1 else ("moderate" if i == 2 else "hard"))
        zone(f"gate_{int(round(clear * 1000))}mm_clear", bounds(x, gates["center_y_m"],
             1.8, clear + 0.8), "easy" if i == 1 else ("moderate" if i == 2 else "hard"),
             "+x or -x", "Clear width is the free space between post surfaces; no overhead bar.",
             {"clear_width_m": clear})

    # Non-colliding zone edge cues stop well short of the open connection corridors.
    for label, y in (("north_inner", 2.05), ("south_inner", -2.05),
                     ("north_outer", 5.05), ("south_outer", -5.05),
                     ("north_far", 8.15), ("south_far", -8.15)):
        for segment, x in enumerate((3.0, 8.5, 14.0), 1):
            line(f"zone_guide_{label}_{segment}", x, y, 1.7, 0.015)

    zone("flat_connectors", [-2.0, -11.0, 18.0, 10.8], "easy", "from central lane via open side corridors",
         "The plane remains open between all zones; painted guides have no contact.")
    manifest = {
        "description": "Deterministic D1 partitioned terrain training course",
        "config": config,
        "source_asset_sha256_before_and_after": source_hashes,
        "spawn_from_original_robot_xml_m": [0, 0, 0.8],
        "spawn_clear_radius_m": config["spawn_clear_radius_m"],
        "control_capability": "not verified; geometry only, no traversal or policy claim",
        "size_convention": "MuJoCo half extents for boxes, radius and half height for cylinders, half xy/elevation/base depth for hfields",
        "zones": zones,
        "geometries": geoms,
        "colliding_course_geom_count": sum(g["collision"] for g in geoms),
        "noncolliding_marking_geom_count": sum(not g["collision"] for g in geoms),
        "ground_geom_count": 1,
    }
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n", manifest, images


README = """# D1 training course 03

Generated from `tools/course03_layout.json` by `tools/generate_training_course.py`.
The scene includes the original `robot.xml`; the generator does not alter source
assets, contact friction, robot parameters or weights. The manifest records SHA256
hashes of the included robot, its meshes and the original scene where present.

The center lane is 20 m of open level ground. South bays contain independent
5/15/30/50 mm steps, 15/30/60 mm riser staircases, fixed low broken plates and
slalom posts. North bays contain 3/6/10 degree continuous ramps, 10/25/50 mm
colliding heightfields and gates with 1.5/1.1/0.75 m clear openings. Floor guides
and entry marks are visual only (`contype=0`, `conaffinity=0`); obstacles collide.
The original `(0, 0, 0.8)` spawn is unobstructed. Open ground connects every bay.

The scene is for staged training and manual inspection. No controller success or
traversal has been verified. Colors show intended relative difficulty, not a
measured capability threshold. Geometry uses solid boxes/cylinders and PNG hfields.
MuJoCo resolves hfield paths relative to the robot compiler's `meshdir`, hence
the generated `../mujoco/training_course_03_assets/...` references.

Regeneration requires a new, empty destination: the generator refuses to replace
the scene or asset directory. It does not start a viewer or step a simulation.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    scene_dir = args.scene_dir.resolve()
    scene_path = scene_dir / config["scene_name"]
    asset_dir = scene_dir / config["asset_directory"]
    if scene_path.exists() or asset_dir.exists():
        parser.error("Refusing to overwrite existing scene or asset directory")
    original_hashes = protected_assets(scene_dir)
    xml, manifest, images = build(config, scene_dir, original_hashes)
    if original_hashes != protected_assets(scene_dir):
        parser.error("A source asset changed while building the scene")
    # Exclusive creation makes repeated generation an explicit error.
    asset_dir.mkdir()
    for name, image in images.items():
        with (asset_dir / name).open("xb") as stream:
            image.save(stream, format="PNG")
    with (asset_dir / "layout_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (asset_dir / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(README)
    with scene_path.open("x", encoding="utf-8") as stream:
        stream.write(xml)
    if original_hashes != protected_assets(scene_dir):
        raise RuntimeError("Source asset hash changed during output writing")
    print(f"Created {scene_path}")
    print(f"Created {asset_dir}")
    print(f"{manifest['colliding_course_geom_count']} colliding course geoms; "
          f"{manifest['noncolliding_marking_geom_count']} non-colliding marks; 1 ground plane")


if __name__ == "__main__":
    main()
