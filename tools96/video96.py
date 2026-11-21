"""Play one measured D1 trajectory alongside two sealed world-model forecasts."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import add_marks, chinese_font, digest
from tools94.video94 import DT, FPS, BG, INK, MUTED, Trace
from tools95.score95 import HORIZON, SEEDS
from tools95.video95 import Arm, load_arm


RESETS = (96201, 96202, 96203)
KINDS = ("native", "control", "paired")
LABELS = ("原生 MuJoCo 实测", "原损失模型预测", "配对损失模型预测")
COLORS = ("#F4CF8D", "#A8CEFF", "#80E1C2")
WIDTH, HEIGHT, PANE_W, PANE_H = 1536, 566, 504, 280
SCENE_Y, HOLD = 82, 18


def sealed_input(root, path, seal, used):
    name = str(path.relative_to(root))
    expected = seal["files"].get(name)
    if expected is None and not path.exists():
        return None
    if expected is None or not path.is_file() or digest(path) != expected:
        raise ValueError(f"source differs from its stage seal: {path}")
    used[path] = expected
    return expected


def model_arm(root, folder, tag, seal, used, seed, reset):
    receipt = folder / f"{tag}.json"
    rhash = sealed_input(root, receipt, seal, used)
    info = dict(receipt=str(receipt.relative_to(root)), receipt_sha256=rhash, source=None, source_sha256=None)
    if rhash is None:
        return Arm(None, {**info, "status": "NO_RECEIPT"})
    row = json.loads(receipt.read_bytes())
    if (row.get("model") != folder.name or row.get("alpha") != (1. if tag == "p1" else 0.)
            or row.get("seed") != seed or row.get("reset") != reset):
        raise ValueError(f"forecast arm, policy seed or reset mismatch: {receipt}")
    path = folder / f"{tag}.npz"
    sha = sealed_input(root, path, seal, used)
    if sha != row.get("state_sha256"):
        raise ValueError(f"forecast receipt SHA differs: {path}")
    safe = row["safe"]
    info.update(safe=safe, reported_return_m=row.get("return_m"), error=row.get("error"))
    if sha is None:
        if safe:
            raise ValueError(f"missing nominally complete forecast: {path}")
        return Arm(None, {**info, "status": "NO_RECORDED_STATES"})
    info.update(source=str(path.relative_to(root)), source_sha256=sha)
    with np.load(path, allow_pickle=False) as archive:
        states = archive["state45"]
    if (states.ndim != 2 or states.shape[1] != 45 or not 0 <= len(states) <= HORIZON+1
            or row["ticks"] != max(0, len(states)-1)):
        raise ValueError(f"forecast endpoints disagree with receipt: {path}")
    info["recorded_ticks"] = row["ticks"]
    valid = np.isfinite(states).all(axis=1) & (np.abs(np.linalg.norm(states[:, 3:7], axis=1) - 1.) <= 1e-4)
    invalid = np.flatnonzero(~valid)
    if safe and (len(states) != HORIZON+1 or len(invalid) or info["error"]):
        raise ValueError(f"nominally complete forecast has missing or invalid poses: {path}")
    error = str(info["error"] or "").lower()
    note = ""
    if len(invalid):
        note = "预测姿态无效"
    elif not safe:
        note = "预测跌倒" if "predicted fall" in error else "预测中止"
    if len(invalid):
        states = states[:int(invalid[0])]
    if not len(states):
        return Arm(None, {**info, "status": "NO_VALID_POSE", "stop_reason": note})
    trace = Trace(path, sha, states, note)
    return Arm(trace, {**info, "status": "RECORDED" if safe else "PARTIAL", "ticks": trace.ticks, "stop_reason": note})


def native_arm(root, folder, reset, seal, used):
    source = load_arm(root, folder, reset)
    for field, sha_field in (("receipt", "receipt_sha256"), ("source", "source_sha256")):
        if source.info.get(field):
            path = root / source.info[field]
            if sealed_input(root, path, seal, used) != source.info.get(sha_field):
                raise ValueError(f"native fingerprint changed before replay: {path}")
    path = folder / "native.json"
    if source.info.get("receipt_sha256"):
        row = json.loads(path.read_bytes())
        source.info.update(safe=row["safe"], reported_return_m=row.get("return_m"))
    return source


@dataclass
class Panel:
    p1: Arm
    zero: Arm
    gains: np.ndarray

    def summary(self):
        full = len(self.gains) == HORIZON
        return dict(p1=self.p1.info, zero=self.zero.info, common_ticks=len(self.gains),
                    h32_gain_mm=float(self.gains[-1]) if full else None,
                    h32_both_safe=bool(full and self.p1.info.get("safe") and self.zero.info.get("safe")))


def panel(p1, zero, physical_start, native):
    for arm in (p1, zero):
        if arm.trace is None:
            continue
        expected = physical_start if native else physical_start.astype(np.float32)
        if not np.array_equal(arm.trace.states[0], expected):
            raise ValueError(f"recording starts from a different physical reset: {arm.trace.path}")
        ret = arm.info.get("reported_return_m")
        if arm.trace.ticks == HORIZON and ret is not None:
            calc = -float(np.abs(arm.trace.states[1:, 0].astype(np.float64)-physical_start[0]).mean())
            if not math.isclose(calc, ret, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"H32 return or physical origin differs: {arm.trace.path}")
    n = min((arm.trace.ticks if arm.trace else 0) for arm in (p1, zero))
    gains = np.empty(0, dtype=np.float64)
    if n:
        a, b = (np.abs(arm.trace.states[1:n+1, 0].astype(np.float64) - physical_start[0]) for arm in (p1, zero))
        gains = 1000 * np.cumsum(b-a, dtype=np.float64) / np.arange(1, n+1)
    return Panel(p1, zero, gains)


@dataclass
class Cell:
    seed: int
    reset: int
    panels: list[Panel]

    @property
    def last_tick(self):
        return max((p.p1.trace.ticks for p in self.panels if p.p1.trace), default=0)


def load_cells(native, predictions):
    used, seals = {}, []
    for root in (native, predictions):
        path = root / "seal.json"
        raw = path.read_bytes()
        seals.append(json.loads(raw))
        used[path] = sha256(raw).hexdigest()
    if not seals[0].get("protocol_sha256") or seals[0]["protocol_sha256"] != seals[1]["protocol_sha256"]:
        raise ValueError("native and forecast stages use different protocols")
    starts = {}
    for reset in RESETS:
        path = predictions / f"start_{reset}.npz"
        if sealed_input(predictions, path, seals[1], used):
            with np.load(path, allow_pickle=False) as saved:
                start = saved["state45"]
            if start.shape != (1, 45) or start.dtype != np.float64 or not np.isfinite(start).all():
                raise ValueError(f"invalid physical reset: {path}")
            starts[reset] = start[0]
    cells = []
    for seed in SEEDS:
        for reset in RESETS:
            rel = Path(f"seed_{seed}/reset_{reset}")
            pair = [native_arm(native, native / rel / t, reset, seals[0], used) for t in ("p1", "zero")]
            start = starts.get(reset)
            if start is None:
                start = next((a.trace.states[0] for a in pair if a.trace), None)
            panels = [panel(*pair, start, True)]
            for kind in KINDS[1:]:
                pair = [model_arm(predictions, predictions/rel/kind, t, seals[1], used, seed, reset)
                        for t in ("p1", "zero")]
                if start is None and any(a.trace for a in pair):
                    raise ValueError(f"no physical origin for forecast of {seed}/{reset}")
                panels.append(panel(*pair, start, False))
            cells.append(Cell(seed, reset, panels))
    # Keep all nine, opening on a measured robot when available.
    cells.sort(key=lambda c: (c.panels[0].p1.trace is None, not any(p.p1.trace for p in c.panels)))
    return cells, used, seals[0]["protocol_sha256"]


def frame(cell, index, tick, picture, fonts, scale):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title, large, small = fonts
    draw.text((14, 3), "D1  4° 驻坡 · 同一步策略调整，模型能算准收益吗？", font=title, fill=INK)
    draw.text((WIDTH-14, 8), f"{index}/9    策略 {cell.seed}    起点 {cell.reset}    {tick*DT:.2f} s    0.5×",
              font=small, fill=MUTED, anchor="ra")
    for side, p in enumerate(cell.panels):
        x = 4 + 512 * side
        color = COLORS[side]
        draw.text((x+8, 45), LABELS[side], font=large, fill=color)
        t = p.p1.trace
        if t:
            at = min(tick, t.ticks)
            image.paste(picture(t, at), (x, SCENE_Y))
            drift = t.drift_mm(at)
            motion = "后溜" if drift <= 0 else "前移"
            draw.text((x+8, 370), f"+1 步　当前{motion} {abs(drift):.1f} mm", font=small, fill=INK)
            if tick >= t.ticks and t.stop_reason:
                draw.text((x+PANE_W-6, 370), f"{t.stop_reason} · {at*DT:.2f} s 定格", anchor="ra", font=small, fill="#FFD6A6")
        else:
            draw.rectangle((x, SCENE_Y, x+PANE_W-1, SCENE_Y+PANE_H-1), fill="#273741")
            draw.text((x+PANE_W/2, SCENE_Y+PANE_H/2), "缺少可回放的轨迹", anchor="mm", font=large, fill=MUTED)
        g = p.gains[tick-1] if 0 < tick <= len(p.gains) else None
        metric = "32 拍收益" if tick == HORIZON else f"前 {tick} 拍收益"
        label = "等待首拍" if tick == 0 else f"{metric}缺测"
        if g is not None:
            label = f"{metric}   {g:+.3f} mm"
        draw.text((x+8, 405), label, font=large, fill=color)
        center, extent = x+PANE_W//2, PANE_W//2-16
        draw.line((center-extent, 463, center+extent, 463), width=2, fill="#536773")
        if g is not None and g != 0:
            draw.line((center, 463, center + extent*g/scale, 463), width=9, fill=color)
        draw.line((center, 451, center, 475), width=2, fill=INK)
        if p.zero.trace and p.zero.trace.stop_reason and tick >= p.zero.trace.ticks:
            draw.text((x+8, 478), f"零步对照{p.zero.trace.stop_reason} · {p.zero.trace.ticks*DT:.2f} s", font=small, fill="#FFD6A6")
    draw.text((14, 520), "收益是相对本栏零步对照少偏离驻车点的毫米数。橙线是驻车起点。", font=small, fill=INK)
    draw.text((WIDTH-14, 520), f"三栏同一刻度　← 多偏离　　少偏离 →　 ±{scale:g} mm", font=small, fill=MUTED, anchor="ra")
    return image


def encode(target, cells, report):
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        raise RuntimeError("software offscreen EGL is required")
    from PIL import Image
    import mujoco
    from tools87.terrain87 import ground_height, load_model

    model = load_model("up")
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor], model.geom_rgba[floor] = -1, (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    xy = np.concatenate([p.p1.trace.states[:, :2] for c in cells for p in c.panels if p.p1.trace])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (lo+hi)/2
    data, camera, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    options.geomgroup[3] = 0
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 100., -13., max(1.48, 1.22 + .85*(hi[0]-lo[0]) + .3*(hi[1]-lo[1]))
    camera.lookat[:] = (cx, cy, float(ground_height("up", cx))+.22)
    fonts = chinese_font(28, True), chinese_font(24, True), chinese_font(18)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
            "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-maxrate", "1000k", "-bufsize", "1000k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(trace, tick):
            state = trace.states[tick]
            data.qpos[:], data.qvel[:], data.time = state[:23], state[23:], tick*DT
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            add_marks(mujoco, renderer.scene, "up", trace.states[0, 0], ground_height, lo[0]-.45, hi[0]+.45, cy)
            report["rendered_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            try:
                for i, cell in enumerate(cells):
                    item = report["cells"][i]
                    item["first_frame"] = report["frames"]
                    for tick in range(cell.last_tick+1):
                        pixels = frame(cell, i+1, tick, picture, fonts, report["bar_scale_mm"]).tobytes()
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    item["endpoint_frame"] = report["frames"]-1
                    for _ in range(HOLD):
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    item["last_frame"] = report["frames"]-1
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"ffmpeg exited {encoder.returncode}")


def render(native, predictions, output, receipt):
    if output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("use unused video and receipt paths")
    report = dict(status="PARTIAL", file=output.name, fps=FPS, playback_speed=.5, source_dt_s=DT, frames=0,
                  cover_frames=0, endpoint_hold_frames=HOLD, new_physics_steps=0, model_steps=0,
                  policy_calls=0, training_updates=0, rendered_poses=0, displayed_alpha=1., reference_alpha=0.,
                  gain="R(+1) minus R(0) in millimetres about the physical reset", cells=[])
    with receipt.open("x", encoding="utf-8") as journal:
        try:
            cells, used, protocol = load_cells(native, predictions)
            report.update(protocol_sha256=protocol, native_seal_sha256=used[native/"seal.json"],
                          predictions_seal_sha256=used[predictions/"seal.json"])
            report["cells"] = [dict(seed=c.seed, reset=c.reset, panels={k: p.summary() for k,p in zip(KINDS,c.panels,strict=True)})
                               for c in cells]
            report["displayed_panels"] = sum(p.p1.trace is not None for c in cells for p in c.panels)
            report["matched_cells"] = sum(
                bool(c.panels[0].p1.trace and c.panels[0].p1.trace.ticks and
                     any(p.p1.trace and p.p1.trace.ticks for p in c.panels[1:])) for c in cells)
            if not report["matched_cells"]:
                report.update(status="NO_MATCHED_NATIVE_AND_MODEL_MOTION", file=None, video_sha256=None)
                return report
            maxima = [float(np.abs(p.gains).max()) for c in cells for p in c.panels if len(p.gains)]
            report["bar_scale_mm"] = math.ceil(max([1., *maxima]))
            encode(output, cells, report)
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate", "-of",
                                    "json", str(output)], check=True, capture_output=True, text=True, timeout=100)
            stream, = json.loads(probe.stdout)["streams"]
            frames = sum(c.last_tick + 1 + HOLD for c in cells)
            if ((stream["width"], stream["height"], stream["codec_name"], stream["r_frame_rate"]) != (WIDTH, HEIGHT, "h264", f"{FPS}/1")
                    or int(stream["nb_read_frames"]) != frames or report["frames"] != frames
                    or abs(float(stream["duration"]) - frames/FPS) > .0001 or not 0 < output.stat().st_size <= 5_000_000):
                raise ValueError(f"encoded video readback failed: {stream}")
            if any(digest(p) != sha for p,sha in used.items()):
                raise ValueError("sealed inputs changed while rendering")
            report.update(status="COMPLETE", video_sha256=digest(output), bytes=output.stat().st_size,
                          duration_s=frames/FPS, stream=stream)
        except BaseException as error:
            report["reason"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            json.dump(report, journal, ensure_ascii=False, allow_nan=False, indent=2)
            journal.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("native", "predictions", "output", "receipt"):
        parser.add_argument("--"+key, type=Path, required=True)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k: result.get(k) for k in ("status", "frames", "duration_s", "video_sha256")}))
