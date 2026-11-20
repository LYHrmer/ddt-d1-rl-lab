"""Render one paired D1 slope trial from saved states at its original speed."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DT, DURATION, FPS = 0.02, 6.0, 25
FRAMES = int(DURATION * FPS)
PANE_W, PANE_H, GAP = 604, 306, 4
WIDTH, HEIGHT, SCENE_TOP = 2 * PANE_W + GAP, 590, 91
BACKGROUND, TEXT, MUTED = "#142330", "#f0f3f5", "#b4c3cb"
ACCENTS = ("#83baf1", "#64d5b8")
TERRAIN_LABELS = {"up": "4° 上坡", "down": "4° 下坡", "flat": "平地"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Trace:
    meta: dict
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    speed: np.ndarray
    command: np.ndarray
    valid_beams: np.ndarray
    hashes: dict


def read_trace(folder, arm):
    metrics, trace = folder / "metrics.json", folder / "trace.npz"
    meta = json.loads(metrics.read_text(encoding="utf-8"))
    if meta["status"] != "COMPLETE" or meta["arm"] != arm or meta["terrain"] not in TERRAIN_LABELS:
        raise ValueError(f"expected a complete {arm} recording on the registered course in {metrics}")
    with np.load(trace, allow_pickle=False) as saved:
        t, qpos, qvel, speed = [saved[key] for key in ("time_s", "qpos", "qvel", "body_vx_m_s")]
        command, ranges = saved["command_vx_m_s"], saved["raw_ranges16"]
    n = len(t)
    if not (2 <= n <= 301 and t.shape == speed.shape == command.shape == (n,)
            and qpos.shape == (n, 23) and qvel.shape == (n, 22)
            and ranges.shape == (n, 16) and np.isin(ranges[:, 8:], (0., 1.)).all()
            and all(np.isfinite(a).all() for a in (t, qpos, qvel, speed, command, ranges))
            and np.allclose(t, np.arange(n) * DT, atol=1e-6, rtol=0)):
        raise ValueError(f"incomplete or invalid 50 Hz state stream in {trace}")
    if bool(meta["full_horizon"]) != (n == 301) or (n < 301 and not meta["terminated"]):
        raise ValueError(f"recorded end time disagrees with episode outcome in {metrics}")
    hashes = {"trace.npz": digest(trace), "metrics.json": digest(metrics)}
    if meta["trace_sha256"] != hashes["trace.npz"]:
        raise ValueError(f"trajectory digest differs from {metrics}")
    return Trace(meta, t, qpos, qvel, speed, command, ranges[:, 8:].sum(axis=1), hashes)


def read_pair(baseline, perceptive):
    left, right = read_trace(baseline, "baseline"), read_trace(perceptive, "range")
    common = min(len(left.time), len(right.time))
    if (left.meta["terrain"] != right.meta["terrain"] or left.meta["seed"] != right.meta["seed"]
            or left.meta["protocol_sha256"] != right.meta["protocol_sha256"]
            or not np.array_equal(left.command[:common], right.command[:common])):
        raise ValueError("both trajectories must share the protocol, terrain, evaluation seed and command")
    return left, right


def source_tick(frame):
    # Centre of each 40 ms display interval; use the exact endpoints in the boundary frames.
    if frame == 0:
        return 0
    if frame == FRAMES - 1:
        return round(DURATION / DT)
    return 2 * frame + 1


def chinese_font(size, bold=False):
    from PIL import ImageFont

    name = "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc"
    return ImageFont.truetype(name, size, index=2)


def add_marks(mujoco, scene, terrain, entry_x, ground_height, x_min, x_max, y):
    marks = [(i / 4, False) for i in range(math.floor(x_min * 4), math.ceil(x_max * 4) + 1)]
    if terrain != "flat":
        marks = [(x, corner) for x, corner in marks if abs(x - entry_x) > .025]
        marks.append((entry_x, True))
    for x, corner in marks:
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("render-only marks exceed available geometry slots")
        z = float(ground_height(terrain, x)) + .003
        geom = scene.geoms[scene.ngeom]
        rgba = (.97, .52, .12, 1) if corner else (.48, .52, .54, 1)
        kind = mujoco.mjtGeom.mjGEOM_CAPSULE
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                           np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, kind, .006 if corner else .0016,
                            np.array((x, y - .68, z)), np.array((x, y + .68, z)))
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def draw_speed(draw, traces, tick, fonts):
    left, right, top, bottom = 86, WIDTH - 30, 478, 550
    low = math.floor(min(0, *(r.speed.min() for r in traces)) * 10) / 10
    high = math.ceil(max(.30, *(r.speed.max() for r in traces)) * 10) / 10
    x = lambda t: left + t * (right - left) / DURATION
    y = lambda v: bottom - (v - low) * (bottom - top) / (high - low)
    for value in (low, high):
        draw.line((left, y(value), right, y(value)), fill="#455661")
        draw.text((left - 9, y(value)), f"{value:.1f}", font=fonts["small"], fill=MUTED, anchor="rm")
    command = max(traces, key=lambda r: len(r.time))
    path = [(x(t), y(v)) for t, v in zip(command.time, command.command)]
    draw.line(path, fill="#d4d5cd", width=1)
    for i, r in enumerate(traces):
        end = min(tick + 1, len(r.time))
        points = [(x(t), y(v)) for t, v in zip(r.time[:end], r.speed[:end])]
        if len(points) >= 2:
            draw.line(points, fill=ACCENTS[i], width=2)
        px, py = points[-1]
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=ACCENTS[i])
    for second in (0, 2, 4, 6):
        draw.text((x(second), bottom + 2), f"{second} s", font=fonts["small"],
                  fill=MUTED, anchor="mt")


def render(baseline, perceptive, output):
    if output.suffix.lower() != ".mp4" or output.exists():
        raise FileExistsError(f"choose a new .mp4 file: {output}")
    left, right = traces = read_pair(baseline, perceptive)
    output.parent.mkdir(exist_ok=True, parents=True)
    receipt = output.with_suffix(".json")
    report = {"status": "INCOMPLETE", "physics_steps": 0, "policy_calls": 0,
              "terrain": right.meta["terrain"], "seed": right.meta["seed"],
              "training_seed": right.meta["training_seed"], "baseline": left.hashes, "range": right.hashes,
              "source_end_s": [float(r.time[-1]) for r in traces], "frames_written": 0}
    with receipt.open("x", encoding="utf-8") as journal:
        try:
            _write_video(traces, output, report)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name",
                 "-of", "json", str(output)], check=True, capture_output=True, text=True, timeout=90)
            stream, = json.loads(probe.stdout)["streams"]
            if (stream["width"], stream["height"], int(stream["nb_read_frames"]), stream["codec_name"]
                ) != (WIDTH, HEIGHT, FRAMES, "h264") or abs(float(stream["duration"]) - DURATION) > 1e-4:
                raise ValueError(f"encoded stream differs from 6 s playback: {stream}")
            if output.stat().st_size > 5_000_000:
                raise ValueError("video exceeds 5 MB")
            for directory, record in zip((baseline, perceptive), traces):
                if any(digest(directory / name) != sha for name, sha in record.hashes.items()):
                    raise ValueError(f"recording changed during visual playback: {directory}")
            report.update(status="COMPLETE", bytes=output.stat().st_size, sha256=digest(output),
                          fps=FPS, stream=stream)
        except BaseException as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(report, journal, allow_nan=False, indent=2)
            journal.write("\n")
    return report


def _write_video(traces, output, report):
    from PIL import Image, ImageDraw

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("LP_NUM_THREADS", "1")
    if os.environ["MUJOCO_GL"] != "egl" or os.environ["LIBGL_ALWAYS_SOFTWARE"] != "1":
        raise RuntimeError("recorded playback requires software offscreen EGL")
    sys.path[:0] = [str(ROOT), str(ROOT / ".python-deps"), str(ROOT / "training")]
    import mujoco
    from tools87.terrain87 import RAMP_START_X_M, ground_height, load_model

    terrain = report["terrain"]
    model = load_model(terrain)
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    data, camera, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    options.geomgroup[3] = 0  # The robot's collision boxes overlap its visual meshes.
    positions = np.concatenate([r.qpos[:, :3] for r in traces])
    low, high = positions.min(axis=0), positions.max(axis=0)
    cx, cy = (low[:2] + high[:2]) / 2
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -13.
    camera.distance = max(1.58, .75 * (1.18 + high[0] - low[0]) + .29 * (high[1] - low[1]))
    camera.lookat[:] = (cx, cy, float(ground_height(terrain, cx)) + .21)
    fonts = {"title": chinese_font(29, True), "label": chinese_font(21, True),
             "data": chinese_font(19), "small": chinese_font(15)}
    args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-threads", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
            "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-maxrate", "2600k", "-bufsize", "1500k", "-threads", "1", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output)]
    labels = ["原全身策略　·　本体反馈", f"前视测距 PPO　·　训练种子 {report['training_seed']}"]
    reference = max(traces, key=lambda r: len(r.time))
    with mujoco.Renderer(model, width=PANE_W, height=PANE_H) as renderer:
        with subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as encoder:
            try:
                for frame in range(FRAMES):
                    tick = source_tick(frame)
                    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                    draw = ImageDraw.Draw(canvas)
                    draw.text((18, 8), f"D1 轮足　·　{TERRAIN_LABELS[terrain]}", font=fonts["title"], fill=TEXT)
                    reference_speed = reference.command[min(tick, len(reference.time) - 1)]
                    caption = (f"指令 {reference_speed:.2f} m/s　　评测种子 {report['seed']}"
                               f"　　1× 原速　　{tick * DT:.2f} s")
                    draw.text((WIDTH - 18, 18), caption, font=fonts["data"], fill=MUTED, anchor="ra")
                    for side, r in enumerate(traces):
                        i = min(tick, len(r.time) - 1)
                        x = side * (PANE_W + GAP)
                        draw.rectangle((x, 54, x + 5, SCENE_TOP - 2), fill=ACCENTS[side])
                        draw.text((x + 17, 54), labels[side], font=fonts["label"], fill=ACCENTS[side])
                        if side == 1:
                            draw.text((x + PANE_W - 15, 59), f"有效测距 {int(r.valid_beams[i])}/8",
                                      font=fonts["small"], fill=MUTED, anchor="ra")
                        data.qpos[:], data.qvel[:], data.time = r.qpos[i], r.qvel[i], r.time[i]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(data, camera=camera, scene_option=options)
                        add_marks(mujoco, renderer.scene, terrain, RAMP_START_X_M, ground_height,
                                  low[0] - 1.4, high[0] + 1.4, cy)
                        canvas.paste(Image.fromarray(renderer.render()), (x, SCENE_TOP))
                        distance = r.qpos[i, 0] - r.qpos[0, 0]
                        line = f"实测 {r.speed[i]:+.2f} m/s　　前进 {distance:+.2f} m"
                        draw.text((x + 17, SCENE_TOP + PANE_H + 7), line, font=fonts["data"], fill=TEXT)
                        if r.meta["terminated"] and tick >= len(r.time) - 1:
                            reason = "跌倒" if r.meta["fallen"] else "终止"
                            draw.text((x + PANE_W - 13, SCENE_TOP + PANE_H + 8),
                                      f"{r.time[-1]:.2f} s {reason} · 定格", font=fonts["small"],
                                      fill="#ffc793", anchor="ra")
                    caption = "地面细线间隔 0.25 m" if terrain == "flat" else "橙线标记坡脚　·　地面细线间隔 0.25 m"
                    draw.text((18, 438), "机身前向速度（m/s）　·　白线为指令", font=fonts["small"], fill=MUTED)
                    draw.text((WIDTH - 18, 438), caption, font=fonts["small"], fill=MUTED, anchor="ra")
                    draw_speed(draw, traces, tick, fonts)
                    encoder.stdin.write(canvas.tobytes())
                    report["frames_written"] += 1
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"ffmpeg exited with {encoder.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--perceptive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(render(options.baseline, options.perceptive, options.output), indent=2))
