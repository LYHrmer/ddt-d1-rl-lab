#!/usr/bin/env python3
"""Render four still views of the unchanged D1 terrain scene with MuJoCo EGL.

Only mj_forward is used. The images are labeled as static terrain previews.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_training_course_03.xml"
DEFAULT_OUTPUT_DIR = ROOT / "evidence/terrain_course_03"

VIEWS = [
    ("overview.png", (8.0, 0.0, 0.0), 35.0, -90, -62, "OVERVIEW · STATIC TERRAIN PREVIEW"),
    ("south_obstacles.png", (8.2, -6.8, 0.0), 14.0, -80, -52, "SOUTH · STEPS / STAIRS / PLATES / SLALOM"),
    ("north_ramps.png", (8.3, 3.5, 0.0), 13.0, -70, -43, "NORTH · 3° / 6° / 10° RAMPS"),
    ("north_rough_gates.png", (8.1, 7.9, 0.0), 14.5, -90, -50,
     "NORTH · ROUGH TILES / GATES"),
]


def font(size):
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def label(image, title):
    canvas = ImageDraw.Draw(image)
    width, _ = image.size
    canvas.rectangle((0, 0, width, 68), fill=(12, 20, 29))
    canvas.text((28, 17), title, font=font(27), fill=(237, 244, 249))
    canvas.rectangle((0, image.height - 47, width, image.height), fill=(12, 20, 29))
    canvas.text((28, image.height - 36), "D1 COURSE 03  |  mj_forward only  |  NO POLICY TRAVERSAL",
                font=font(18), fill=(172, 202, 218))


def render(scene, output_dir):
    if mujoco.__version__ != "3.3.0":
        raise RuntimeError(f"MuJoCo 3.3.0 required, got {mujoco.__version__}")
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Set MUJOCO_GL=egl for reproducible headless rendering")
    paths = [output_dir / name for name, *_ in VIEWS]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing image(s): " + ", ".join(existing))
    model = mujoco.MjModel.from_xml_path(str(scene))
    # Allocate the requested image size in the renderer without editing the scene.
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 1100
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=1100, width=1600)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path, lookat, distance, azimuth, elevation, title in VIEWS:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = azimuth
            camera.elevation = elevation
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            label(image, title)
            with (output_dir / path).open("xb") as stream:
                image.save(stream, format="PNG")
            print(f"Wrote {output_dir / path} ({image.width}x{image.height})")
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    render(args.scene.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
