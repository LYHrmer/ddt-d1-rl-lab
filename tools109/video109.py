"""Play an archived native flat-course trajectory with a fixed metric ground reference."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from tools87.video87 import chinese_font
from tools94.common94 import ROOT, digest, read
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, rim_marks, verify_video
from tools109.run109 import BASE, completed


LABELS = {"actor5": "全身策略 · 原轮速量程", "actor24": "全身策略 · 宽轮速量程",
          "reference24": "策略控制腿部 · 四轮使用速度参考"}


def meter_marks(mj, scene, position):
    """Render-only marks every metre on the ground beside the path."""
    for x in range(int(np.floor(position))-3, int(np.ceil(position))+5):
        mark = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(mark, mj.mjtGeom.mjGEOM_BOX, np.array([.013, .38, .002]),
                        np.array([x, -.72, .002]), np.eye(3).ravel(), np.array([.95, .82, .42, 1.], np.float32))
        mark.category = int(mj.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def render(out: Path):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    from tools109.cruise109 import BLIND, DEV

    driver = probe_cpu_egl(out)
    entries = [("actors", "actor24", DEV[0], "开发对照")]
    if any(BASE.glob("reference_??/report.json")):
        entries.append(("reference", "reference24", DEV[0], "同一起点"))
    if any(BASE.glob("blind_??/report.json")):
        _, blind, _ = completed("blind")
        entries.append(("blind", blind["arms"][0], BLIND[0], "独立盲测"))
    scene = ROOT / read(ROOT/"training/configs/pilot03d.json")["scene"]
    model = mj.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth, model.vis.global_.offheight = 720, 348
    data, cam = mj.MjData(model), mj.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 108., -13., 3.4
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    title, label_font = chinese_font(36), chinese_font(25)
    frames, clips = 0, []
    movie = out/"D1_平地1.5m每秒_加速与稳定性对照_109.mp4"
    proc = encoder(movie)
    try:
        with mj.Renderer(model, height=348, width=720) as renderer:
            for stage, arm, seed, label in entries:
                parent, report, _ = completed(stage)
                record, = [r for r in report["episodes"] if r["arm"] == arm and r["seed"] == seed]
                path = parent / f"{seed}_{arm}" / "trace.npz"
                if record["trace_sha256"] != digest(path):
                    raise ValueError("invalid archived speed109 trajectory")
                with np.load(path, allow_pickle=False) as archive:
                    poses = archive["qpos"].copy()
                end = len(poses)-1
                clips.append({"stage": stage, "seed": seed, "arm": arm, "source_trace_sha256": record["trace_sha256"],
                              "passed": record["score"]["passed"], "end_tick": end})
                for tick in range(end+1):
                    # Cruise in real time, acceleration and braking at half-speed.
                    if 150 <= tick < 275 and tick % 2 and tick != end:
                        continue
                    data.qpos[:] = poses[tick]
                    mj.mj_fwdPosition(model, data)
                    cam.lookat[:] = [float(poses[tick, 0])+.24, float(poses[tick, 1]), .20]
                    renderer.update_scene(data, camera=cam)
                    meter_marks(mj, renderer.scene, poses[tick, 0])
                    rim_marks(mj, renderer.scene, model, data, wheels)
                    view = Image.fromarray(renderer.render()).resize((WIDTH, 616), Image.Resampling.BILINEAR)
                    img = Image.new("RGB", (WIDTH, HEIGHT), "#16222B")
                    img.paste(view, (0, 60))
                    draw = ImageDraw.Draw(img)
                    draw.text((20, 8), f"D1 平地 1.5 m/s 调试   {LABELS[arm]}", font=title, fill="#F3F6F6")
                    phase = ("起步" if tick < 50 else "加速" if tick < 150 else "巡航" if tick < 275
                             else "减速" if tick < 350 else "停车")
                    rate = "原速" if 150 <= tick < 275 else "半速"
                    caption = f"{label} {seed}   ·   实测轨迹回放   ·   {phase}，{rate}   ·   地面黄线间隔 1 米"
                    if tick == end and end < 400:
                        caption += "   ·   提前终止"
                    draw.text((20, 679), caption, font=label_font, fill="#E0E9EB")
                    copies = 2*FPS if tick == end else 1
                    pixels = img.tobytes()
                    for _ in range(copies):
                        proc.stdin.write(pixels)
                    frames += copies
        proc.stdin.close()
        if proc.wait(timeout=90):
            raise RuntimeError("ffmpeg encoder failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return {"filename": movie.name, "sha256": digest(movie), "bytes": movie.stat().st_size,
            "clips": clips, "frames": frames, "stream": verify_video(movie, frames),
            "driver": driver, "new_physics_substeps": 0}
