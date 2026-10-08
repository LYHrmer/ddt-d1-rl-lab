#!/usr/bin/env python3
"""Create an estimated video-inspired D1 course without changing source assets.

Uses NumPy and Pillow only; never imports MuJoCo or starts a simulation.
All generated destinations must be absent. Existing files are never replaced.
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


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_DIR = (
    WORKSPACE / "src/ddt_ros2_control/urdfs/d1_description/mujoco"
)
ASSET_DIR_NAME = "video_course_assets"
SEED = 20261008


def numbers(values):
    return " ".join(f"{float(value):.7g}" for value in values)


def rotation_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def height_images():
    """Grayscale PNG samples; MuJoCo normalizes each image to [0, 1]."""
    rng = np.random.default_rng(SEED)
    side = 129
    axis = np.linspace(-1.0, 1.0, side)
    x, y = np.meshgrid(axis, axis)
    # Every boundary sample is exactly zero: no vertical step into either tile.
    edge = np.clip((1 - np.maximum(np.abs(x), np.abs(y))) / 0.23, 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    noise = np.zeros((side, side))
    for resolution, weight in ((7, 0.65), (15, 0.25), (29, 0.10)):
        small = (rng.random((resolution, resolution)) * 255).astype(np.uint8)
        smooth = Image.fromarray(small).resize(
            (side, side), Image.Resampling.BICUBIC
        )
        noise += weight * np.asarray(smooth, dtype=float) / 255
    rough = (0.10 + noise) * edge

    # Meandering raised lines approximate the patterned tile in the recording.
    ridges = np.zeros_like(x)
    for offset, phase in ((-0.55, 0.2), (-0.05, 1.7), (0.45, 3.1)):
        path = offset + 0.11 * np.sin(8 * x + phase) + 0.05 * np.sin(19 * x)
        ridges += np.exp(-((y - path) / 0.035) ** 2)
    for offset in (-0.55, 0.55):
        path = offset + 0.06 * np.sin(13 * y)
        ridges += 0.6 * np.exp(-((x - path) / 0.03) ** 2)
    patterned = (0.02 + ridges) * edge
    return {
        "rough_height.png": Image.fromarray(
            np.rint(255 * rough / rough.max()).astype(np.uint8)
        ),
        "pattern_height.png": Image.fromarray(
            np.rint(255 * patterned / patterned.max()).astype(np.uint8)
        ),
    }


def build_scene(scene_dir):
    robot = scene_dir / "robot.xml"
    compiler = ET.parse(robot).getroot().find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    root = ET.Element("mujoco", model="D1 estimated video course")
    root.append(ET.Comment(
        " Visual/geometric approximation of the supplied recording. "
        "All course dimensions are estimated; see video_course_assets/README.md. "
    ))
    ET.SubElement(root, "include", file="robot.xml")
    ET.SubElement(root, "statistic", center="4.1 1.0 0.4", extent="9.2")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6",
                  ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="-130", elevation="-48", fovy="42")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="video_ground_texture",
                  builtin="checker", mark="edge", rgb1="0.2 0.3 0.4",
                  rgb2="0.1 0.2 0.3", markrgb="0.55 0.65 0.75",
                  width="300", height="300")
    ET.SubElement(asset, "material", name="video_ground",
                  texture="video_ground_texture", texuniform="true",
                  texrepeat="5 5", reflectance="0.2")
    ET.SubElement(asset, "material", name="video_concrete",
                  rgba="0.67 0.67 0.64 1", specular="0.08", shininess="0.12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="2 -3 9", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="video_floor", type="plane",
                  size="0 0 0.05", material="video_ground")
    geometry = []

    def geom(name, kind, position, size, *, rgba=None, quat=None, **extra):
        attributes = dict(name=name, type=kind, pos=numbers(position),
                          size=numbers(size), contype="1", conaffinity="1")
        if rgba is None:
            attributes["material"] = "video_concrete"
        else:
            attributes["rgba"] = numbers(rgba)
        if quat is not None:
            attributes["quat"] = numbers(quat)
        attributes.update(extra)
        ET.SubElement(world, "geom", **attributes)
        geometry.append({"name": name, "type": kind, "position_m": position,
                         "mujoco_size_m": size, "estimated": True,
                         "collision_enabled": True,
                         **({"quaternion_wxyz": quat} if quat else {}),
                         **({"rgba": rgba} if rgba else {}), **extra})

    # Ordered near-to-far: orange, purple, green, yellow, cyan.
    colors = [
        ("orange", 0.8, 0.20, (1.0, 0.38, 0.08, 1)),
        ("purple", 2.6, 0.28, (0.64, 0.18, 0.88, 1)),
        ("green", 4.4, 0.36, (0.12, 0.83, 0.35, 1)),
        ("yellow", 6.2, 0.44, (0.95, 0.96, 0.06, 1)),
        ("cyan", 8.0, 0.52, (0.06, 0.86, 0.81, 1)),
    ]
    for color, x, height, rgba in colors:
        geom(f"video_block_{color}", "box", (x, -3.0, height / 2),
             (0.50, 0.50, height / 2), rgba=rgba)

    # Broad gray staircase and an elevated deck. These are display obstacles,
    # not a claim that the loaded policy can negotiate these heights.
    for index in range(7):
        height = 0.12 * (index + 1)
        geom(f"video_main_stair_{index + 1}", "box",
             (4.5 + 0.30 * index, 4.0, height / 2), (0.15, 1.0, height / 2))
    geom("video_main_platform", "box", (6.95, 4.0, 0.42), (0.50, 1.0, 0.42))

    def ramp(name, start, end, width, thickness=0.05):
        dx, dy, dz = np.subtract(end, start)
        horizontal = math.hypot(dx, dy)
        pitch = -math.atan2(dz, horizontal)
        yaw = math.atan2(dy, dx)
        quat = rotation_quaternion(0, pitch, yaw)
        # start/end describe the upper surface centerline, not the box center.
        normal = (math.cos(yaw) * math.sin(pitch),
                  math.sin(yaw) * math.sin(pitch), math.cos(pitch))
        center = tuple((a + b) / 2 - thickness / 2 * n
                       for a, b, n in zip(start, end, normal))
        geom(name, "box", center,
             (math.sqrt(horizontal**2 + dz**2) / 2, width / 2, thickness / 2),
             quat=quat)

    ramp("video_long_ramp", (7.45, 4.0, 0.84), (10.1, 4.0, 0.02), 1.25)
    ramp("video_cross_ramp", (8.55, -1.0, 0.02), (8.55, 2.6, 0.78), 0.8)
    geom("video_cross_deck", "box", (8.55, 2.9, 0.755), (0.40, 0.30, 0.025))
    # Secondary low stairs, a round platform and two gray access panels.
    for index in range(5):
        height = 0.10 * (index + 1)
        geom(f"video_secondary_stair_{index + 1}", "box",
             (6.3 + 0.28 * index, 1.6, height / 2), (0.14, 0.80, height / 2))
    geom("video_cylinder_base", "box", (6.6, 0.1, 0.06), (0.85, 0.72, 0.06))
    geom("video_round_platform", "cylinder", (6.6, 0.1, 0.30), (0.53, 0.18))
    ramp("video_wide_access_panel", (4.7, 4.0, 0.015), (6.5, 2.8, 0.72), 1.30)

    hfields = [
        ("video_rough", "rough_height.png", (2.35, 0.65, 0), (1.10, 1.10, 0.025, 0.015)),
        ("video_pattern", "pattern_height.png", (4.3, 2.05, 0), (1.05, 1.05, 0.035, 0.015)),
    ]
    for name, filename, position, size in hfields:
        # MuJoCo 3.3 resolves hfields through compiler.meshdir, just like meshes.
        relative = os.path.relpath(scene_dir / ASSET_DIR_NAME / filename,
                                   scene_dir / meshdir)
        ET.SubElement(asset, "hfield", name=name, file=relative, size=numbers(size))
        geom(name + "_surface", "hfield", position, (), hfield=name)
        geometry[-1]["hfield_size_m"] = size
        geometry[-1]["height_png"] = ASSET_DIR_NAME + "/" + filename
        geometry[-1]["max_surface_height_m"] = size[2]

    # Fixed overlapping thin plates approximate debris, with individual box
    # collision shapes; no single convex hull conceals the gaps between pieces.
    geom("video_debris_base", "box", (0.15, 3.3, 0.012), (1.10, 1.10, 0.012))
    rng = np.random.default_rng(SEED + 1)
    for index in range(35):
        center = (0.15 + float(rng.uniform(-0.78, 0.78)),
                  3.3 + float(rng.uniform(-0.78, 0.78)),
                  float(rng.uniform(0.06, 0.14)))
        size = (float(rng.uniform(0.20, 0.48)),
                float(rng.uniform(0.07, 0.20)), float(rng.uniform(0.008, 0.018)))
        quat = rotation_quaternion(float(rng.uniform(-0.13, 0.13)),
                                   float(rng.uniform(-0.13, 0.13)),
                                   float(rng.uniform(-math.pi, math.pi)))
        geom(f"video_debris_plate_{index + 1:02d}", "box", center, size, quat=quat)

    manifest = {
        "description": "Estimated visual/geometric reconstruction, not original colleague assets",
        "all_course_dimensions_estimated": True, "units": "metres",
        "seed": SEED, "source_recording": "/home/lyh/下载/录屏.webm",
        "reference_frames": "ddt_ros2_ws/evidence/reference/reference_01.png through reference_04.png",
        "robot_include": "robot.xml",
        "robot_xml_sha256": hashlib.sha256(robot.read_bytes()).hexdigest(),
        "robot_spawn_m_from_original_xml": [0, 0, 0.8],
        "suggested_gentle_entry_xy_m": [[0, 0], [0.8, 0], [1.3, 0.2], [2.0, 0.4]],
        "entry_note": "Geometric suggestion only; no control command or traversal validation.",
        "heightfield_resolution_px": [129, 129],
        "camera_estimated": {"center": [4.1, 1.0, 0.4], "extent": 9.2,
                             "azimuth_deg": -130, "elevation_deg": -48, "fovy_deg": 42},
        "size_convention": "MuJoCo half-extents for boxes; radius and half-height for cylinders; hfield half-x, half-y, elevation range, base depth.",
        "geometry": geometry,
    }
    # Hfield geom.size is supplied by its asset; an empty size attribute is invalid.
    for element in world.findall("geom"):
        if element.get("type") == "hfield":
            element.attrib.pop("size")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n", manifest


README = """# Estimated video course / 录屏场地近似重建

这是一份依据录屏外观制作的可选视觉与几何近似，**不是同事原始场地参数**。
所有障碍尺寸、位置、颜色和相机参数均为 estimated；没有从录屏恢复真实尺度。
原 scene.xml、robot.xml、STL、机器人出生点、关节、执行器、策略、物理时间步均未修改。

内容：蓝色棋盘、青黄绿紫橙5色块、灰色台阶/平台/长坡、圆柱、碎板区、
平缓粗糙起伏区和纹路地块。碎板为35个固定薄箱体，属于形状简化；所有障碍
都有静态碰撞几何。两个起伏地块使用真正的 PNG hfield 碰撞，不是凸包网格。
高度场边界为零，粗糙区最大抬升25毫米，纹路区最大抬升35毫米。

机器人沿用原始 (0, 0, 0.8) 初始位置，周围留空。layout_estimated.json 中提供一条
驶向平缓粗糙区的几何路线，仅供人工低速操作参考，不代表控制策略已通过验证。
高台、长坡、彩块用于展示；不应把它们作为现有策略默认必须通关的考核。

本目录由 tools/generate_video_course.py 生成；该程序拒绝覆盖任何已有输出。
生成器不调用 MuJoCo、不启动仿真。场景加载与viewer验证由主任务另行完成。
相机设置意在一次显示全场及起点机器人，实际窗口比例/视距可由viewer手动调整。

复用 robot.xml 的 compiler.meshdir 时，MuJoCo 3.3 同样通过 meshdir 解析 hfield.file，
故PNG引用可能写作 ../mujoco/video_course_assets/*.png；这是相对于 ../meshes 的路径。
本地schema依据：mujoco/test/user/testdata/hfield_png.xml 与
mujoco/src/user/user_objects.cc 中 mjCHField::Compile / LoadPNG。

layout_estimated.json 记录每个碰撞体、高度场尺度、参考帧、随机种子和机器人XML散列。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    args = parser.parse_args()
    scene_dir = args.scene_dir.resolve()
    if not (scene_dir / "robot.xml").is_file():
        parser.error(f"robot.xml not found in {scene_dir}")
    scene_path = scene_dir / "scene_video_reconstruction.xml"
    assets = scene_dir / ASSET_DIR_NAME
    if scene_path.exists() or assets.exists():
        parser.error("Refusing to replace existing scene or asset directory; all outputs must be new.")
    # Prepare everything before creating outputs. The original files are read-only.
    xml, manifest = build_scene(scene_dir)
    images = height_images()
    assets.mkdir()
    for filename, image in images.items():
        with (assets / filename).open("xb") as stream:
            image.save(stream, format="PNG")
    with (assets / "layout_estimated.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with (assets / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(README)
    with scene_path.open("x", encoding="utf-8") as stream:
        stream.write(xml)
    print(scene_path)
    print(assets)
    print("Estimated course generated; no model was loaded and no simulation was run.")


if __name__ == "__main__":
    main()
