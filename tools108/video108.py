"""Render three paired D1 step trials from sealed native poses, without advancing the simulator."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

from tools87.video87 import chinese_font
from tools94.common94 import ROOT, digest, read


FPS, NATIVE_HZ, WIDTH, HEIGHT = 25, 50, 1280, 720
PANE_W, PANE_H = 624, 514
RASTER_W, RASTER_H = 432, 356
PANE_X, PANE_Y = (12, 644), 117
SLOW_WINDOW_TICKS = 15
BG, INK, MUTED = "#18252E", "#F3F5F5", "#C6D1D5"
ACCENTS = ("#A8C7F3", "#70D4BC")
STEP_COLOR = (.78, .45, .13, 1.)


@dataclass(frozen=True)
class Shot:
    tick: int
    frames: int
    speed: float
    hold: str = ""


def timeline(last_tick, contact_ticks):
    """Publish every recorded 50 Hz boundary, and never invent an intermediate pose."""
    if last_tick < 0 or any(c is not None and not 1 <= c <= last_tick for c in contact_ticks):
        raise ValueError("invalid native endpoint or wheel-to-step onset")
    if last_tick == 0:
        yield Shot(0, 2*FPS, 0., "仅存初态")
        return
    contacts = sorted({int(c) for c in contact_ticks if c is not None})
    pauses = [c for i, c in enumerate(contacts) if i == 0 or c-contacts[i-1] >= 5]
    for tick in range(last_tick+1):
        slow = any(abs(tick-c) <= SLOW_WINDOW_TICKS for c in contacts)
        normal = Shot(tick, 2 if slow else 1, .25 if slow else .5)
        yield normal
        if tick == 0:
            yield Shot(tick, 12, 0., "起点定格")
        if tick in pauses:
            yield Shot(tick, FPS, 0., "首次接触定格")
        if tick == last_tick:
            yield Shot(tick, FPS, 0., "末态定格")


def first_contact(force):
    """Return physical end time and next recorded 50 Hz boundary for front-wheel/step force."""
    force = np.asarray(force)
    if (force.ndim != 2 or force.shape[1] != 4 or len(force) % 10 or not np.isfinite(force).all()
            or (force < 0).any()):
        raise ValueError("expected nonnegative step-specific 500 Hz FL/FR/RL/RR forces")
    hits = np.flatnonzero((force[:, :2] >= 1.).any(axis=1))
    if len(hits):
        endpoint = int(hits[0])+1
        return endpoint/500, (endpoint+9)//10
    return None, None


@dataclass
class Arm:
    qpos: np.ndarray
    time: np.ndarray
    first_contact_s: float | None
    contact_tick: int | None
    stop_reason: str
    path: Path
    sha256: str

    @property
    def last_tick(self):
        return len(self.time)-1


def _status(arm, tick):
    if tick >= arm.last_tick and arm.stop_reason:
        return arm.stop_reason + " · 末态定格"
    if arm.contact_tick is not None and tick >= arm.contact_tick:
        return "前轮已触台"
    if tick >= arm.last_tick:
        return "窗内前轮未触台"
    return "前轮尚未触台"


def _canvas(i, seed, split, shot, arms, pictures, fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(frame)
    title, pane_label, label, small = fonts
    d.text((19, 7), "D1 前轮遇到 3 厘米台阶", font=title, fill=INK)
    source = "独立盲测" if split == "blind" else "开发采样"
    d.text((WIDTH-18, 16), f"{source}起点 {i+1}/3　·　{seed}", font=label, fill=MUTED, anchor="ra")
    timing = shot.hold or ("0.25 倍慢放" if shot.speed == .25 else "0.5 倍慢放")
    d.text((20, 53), "原生闭环仿真　·　全身策略与关节 PD", font=small, fill=MUTED)
    d.text((WIDTH-18, 54), f"仿真 {shot.tick/NATIVE_HZ:.2f} 秒　·　{timing}", font=small, fill=INK, anchor="ra")
    for side, (x, arm, picture) in enumerate(zip(PANE_X, arms, pictures, strict=True)):
        d.rectangle((x, 85, x+5, 110), fill=ACCENTS[side])
        d.text((x+14, 79), ("原动作", "前腿目标微调")[side], font=pane_label, fill=ACCENTS[side])
        d.text((x+PANE_W-5, 87), _status(arm, shot.tick), font=small, fill=INK, anchor="ra")
        frame.paste(picture, (x, PANE_Y))
    right = arms[1]
    pulsing = right.contact_tick is not None and right.contact_tick <= shot.tick < min(right.last_tick, right.contact_tick+16)
    caption = ("右侧此时请求两前膝目标各减 0.08 rad，持续 0.32 秒。" if pulsing else
               "看浅色轮缘刻线和橙色台沿，比较前轮触台后的进退。")
    d.text((18, 647), caption, font=label, fill=INK)
    d.text((18, 688), "画面取自实测闭环姿态。世界模型的拟合单独离线进行。", font=small, fill=MUTED)
    return frame.tobytes()


def rim_marks(mj, scene, model, data, wheels):
    """Draw a short stripe on each front tire using the archived wheel rotation."""
    for geom_id in wheels[:2]:
        geom_pos = data.geom_xpos[geom_id]
        basis = data.geom_xmat[geom_id].reshape(3, 3)
        radius, half_width = model.geom_size[geom_id, :2]
        for sign in (-1, 1):
            face = geom_pos + sign*(half_width+.0015)*basis[:, 2]
            if scene.ngeom == scene.maxgeom:
                raise ValueError("insufficient render-only geometry slots")
            mark = scene.geoms[scene.ngeom]
            kind = mj.mjtGeom.mjGEOM_CAPSULE
            mj.mjv_initGeom(mark, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                            np.array((.98, .93, .71, 1.), dtype=np.float32))
            mj.mjv_connector(mark, kind, .0035, face + .31*radius*basis[:, 0], face + .87*radius*basis[:, 0])
            mark.category = int(mj.mjtCatBit.mjCAT_DECOR)
            scene.ngeom += 1


def encoder(path):
    return subprocess.Popen(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-maxrate", "900k", "-bufsize", "900k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def verify_video(path, frames):
    cmd = ["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,pix_fmt,r_frame_rate",
           "-of", "json", str(path)]
    stream, = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout)["streams"]
    if ((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"], stream["r_frame_rate"])
            != (WIDTH, HEIGHT, "h264", "yuv420p", f"{FPS}/1") or int(stream["nb_read_frames"]) != frames
            or abs(float(stream["duration"]) - frames/FPS) > .001):
        raise ValueError("encoded stream does not match the native playback schedule")
    return stream


def _render_pair(mj, model, output, report, number, seed, arms, fonts):
    from PIL import Image
    from OpenGL import GL
    from tools100.sensor100 import _wheel_cylinders

    step, floor = model.geom("step108").id, model.geom("floor").id
    if model.geom_type[step] != mj.mjtGeom.mjGEOM_BOX or not np.isclose(model.geom_size[step, 2]*2, .03):
        raise ValueError("movie needs the registered 30 mm step")
    model.vis.global_.offwidth, model.vis.global_.offheight = RASTER_W, RASTER_H
    model.vis.quality.offsamples = 0
    model.vis.quality.shadowsize = 128
    model.vis.headlight.ambient[:] = .42
    model.vis.headlight.diffuse[:] = .52
    model.geom_matid[[floor, step]] = -1
    model.geom_rgba[floor] = (.36, .40, .40, 1.)
    model.geom_rgba[step] = STEP_COLOR
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    for wheel in wheels:
        visual = (model.geom_bodyid == model.geom_bodyid[wheel]) & (model.geom_group == 2)
        model.geom_rgba[visual] = (.14, .17, .18, 1.)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -9.
    options = mj.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    data = mj.MjData(model)
    last = max(a.last_tick for a in arms)
    positions = [a.qpos[np.minimum(np.arange(last+1), a.last_tick), :2] for a in arms]
    camera.distance = 1.32 + .8*np.linalg.norm(positions[0]-positions[1], axis=1).max()
    centers = .5*(positions[0] + positions[1])
    source_tick = -1
    with mj.Renderer(model, width=RASTER_W, height=RASTER_H) as renderer:
        driver = (GL.glGetString(GL.GL_RENDERER) or b"").decode()
        if "llvmpipe" not in driver.lower() or report["renderer"] != driver:
            raise ValueError(f"unexpected render device {driver}")
        for shot in timeline(last, [a.contact_tick for a in arms]):
            if shot.tick != source_tick:
                views = []
                camera.lookat[:] = (*centers[shot.tick], .23)
                camera.lookat[0] += .10
                previous_pose = None
                for arm in arms:
                    at = min(shot.tick, arm.last_tick)
                    pose = arm.qpos[at]
                    if previous_pose is not None and np.array_equal(previous_pose, pose):
                        # Paired resets share this exact physical pose and camera before the intervention.
                        views.append(views[-1])
                        continue
                    data.qpos[:], data.time = pose, arm.time[at]
                    mj.mj_fwdPosition(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=options)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    rim_marks(mj, renderer.scene, model, data, wheels)
                    raw = Image.fromarray(renderer.render())
                    views.append(raw.resize((PANE_W, PANE_H), Image.Resampling.BILINEAR))
                    previous_pose = pose
                source_tick = shot.tick
            pixels = _canvas(number, seed, report["split"], shot, arms, views, fonts)
            for _ in range(shot.frames):
                output.write(pixels)
            report["frames"] += shot.frames


def load_arm(root, seed, tag):
    directory = root / f"seed{seed}_{tag}"
    meta = read(directory/"report.json")
    name = meta["trace_file"]
    if (meta["seed"] != seed or meta["arm"] != tag or meta["status"] not in ("COMPLETE", "PARTIAL")
            or name not in ("trace.npz", "partial_trace.npz") or (meta["status"] == "PARTIAL") != (name == "partial_trace.npz")):
        raise ValueError(f"invalid native receipt in {directory}")
    path = directory / name
    sha = digest(path)
    if sha != meta["trace_sha256"]:
        raise ValueError(f"recording digest mismatch in {directory}")
    with np.load(path, allow_pickle=False) as a:
        qpos, t, force = a["qpos"], a["time_s"], a["step_force_n"]
        fallen, illegal = a["fallen"], a["illegal_contact"]
    n = len(qpos)-1
    if (not 0 <= n <= 200 or n != meta["ticks_completed"] or qpos.shape != (n+1,23) or t.shape != (n+1,)
            or force.shape != (n*10,4) or fallen.shape != (n,) or illegal.shape != (n,)
            or not np.isfinite(qpos).all() or not np.allclose(t, np.arange(n+1)/NATIVE_HZ, atol=1e-6, rtol=0)
            or (meta["status"] == "COMPLETE" and (n != 200 or not meta["full_horizon"]))):
        raise ValueError(f"invalid recorded prefix or 50 Hz clock in {directory}")
    when, at = first_contact(force)
    if at != meta["trigger_tick"] or (None if when is None else round(when*500)-1) != meta["first_contact_substep"]:
        raise ValueError(f"archived force and trigger disagree in {directory}")
    reasons = []
    if fallen.any():
        reasons.append("跌倒")
    if illegal.any():
        reasons.append("非法接触")
    if meta["status"] == "PARTIAL":
        reasons.append("部分记录")
    return Arm(qpos, t, when, at, " · ".join(reasons), path, sha), meta


def render(episodes, output, split="blind", receipt=None):
    """Run inside the admitted Mesa unit, with a fresh movie and adjacent JSON receipt."""
    from tools102.mesa_video102 import GL_ENV, probe_cpu_egl

    episodes, output = Path(episodes), Path(output)
    receipt = Path(receipt) if receipt else output.with_suffix(".json")
    if output.suffix.lower() != ".mp4" or output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("supply new .mp4 and receipt paths")
    plan = read(ROOT/"docs/experiments108/contact.json")
    if split not in ("blind", "development"):
        raise ValueError("choose a registered group of three resets")
    seeds = plan[split]
    index_path = episodes/"index.json"
    index = read(index_path)
    if (index["reset_seeds"] != seeds or index["arms"] != plan["arms"] or len(index["episodes"]) != 6
            or index["checkpoint_sha256"] != plan["checkpoint_sha256"]):
        raise ValueError("incomplete or unregistered three-pair recording matrix")
    report = dict(status="PARTIAL", file=output.name, split=split, seeds=seeds, fps=FPS, frames=0, cover_frames=0,
                  renderer=None, native_decisions=0, physics_substeps=0, new_rays=0, model_steps=0, policy_updates=0,
                  index_sha256=digest(index_path), pairs=[])
    with receipt.open("x", encoding="utf-8") as journal:
        try:
            pairs = []
            for number, seed in enumerate(seeds):
                pair = []
                entry = dict(seed=seed, arms=[])
                for tag in plan["arms"]:
                    arm, meta = load_arm(episodes, seed, tag)
                    name = f"seed{seed}_{tag}"
                    if (meta["geometry"] != plan["geometry"] or meta["checkpoint_sha256"] != plan["checkpoint_sha256"]
                            or index["episodes"][2*number+len(pair)] != {"name": name, **meta}):
                        raise ValueError(f"track provenance differs from recorded matrix: {name}")
                    entry["arms"].append(dict(name=tag, trace=str(arm.path.relative_to(episodes)), sha256=arm.sha256,
                          status=meta["status"], terminal_reason=meta["terminal_reason"], ticks=arm.last_tick,
                          first_front_step_contact_s=arm.first_contact_s, first_visible_pose=arm.contact_tick))
                    pair.append(arm)
                if not np.array_equal(pair[0].qpos[0], pair[1].qpos[0]):
                    raise ValueError("arm initial poses differ")
                pairs.append(pair)
                report["pairs"].append(entry)
            if not all(os.environ.get(k) == value for k, value in GL_ENV.items()):
                raise RuntimeError("set the registered software Mesa/EGL environment before replay")
            with TemporaryDirectory(prefix="d1-video108-egl-") as temporary:
                driver = probe_cpu_egl(Path(temporary))
            report.update(renderer=driver["renderer"], software_egl=driver)
            import mujoco as mj
            from tools108.contact108 import load_contact_model

            model = load_contact_model(plan["geometry"])
            fonts = (chinese_font(33, True), chinese_font(25, True), chinese_font(23), chinese_font(18))
            with encoder(output) as proc:
                try:
                    for number, (seed, pair) in enumerate(zip(seeds, pairs, strict=True)):
                        entry = report["pairs"][number]
                        entry["first_frame"] = report["frames"]
                        _render_pair(mj, model, proc.stdin, report, number, seed, pair, fonts)
                        entry["last_frame"] = report["frames"]-1
                finally:
                    proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg exited {proc.returncode}")
            frames = sum(s.frames for pair in pairs for s in timeline(max(a.last_tick for a in pair),
                                                                      [a.contact_tick for a in pair]))
            if (report["frames"] != frames or digest(index_path) != report["index_sha256"]
                    or any(digest(a.path) != a.sha256 for p in pairs for a in p)):
                raise ValueError("source archive or shown frame schedule changed")
            stream = verify_video(output, frames)
            report.update(status="COMPLETE", stream=stream, sha256=digest(output), bytes=output.stat().st_size,
                          all_episodes_complete=all(a["status"] == "COMPLETE" for e in report["pairs"] for a in e["arms"]))
        except BaseException as err:
            report["error"] = f"{type(err).__name__}: {err}"
            raise
        finally:
            json.dump(report, journal, ensure_ascii=False, indent=2, allow_nan=False)
            journal.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("blind", "development"), default="blind")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    render(args.episodes, args.output, args.split, args.receipt)
