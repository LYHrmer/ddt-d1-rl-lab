"""Replay the two signed whole-policy perturbations from measured D1 states."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools87.video87 import add_marks, chinese_font, digest
from tools94.common94 import SEEDS
from tools94.video94 import ACCENTS, BG, DT, FPS, INK, MUTED, PANE_H, PANE_W, PANE_X, SCENE_Y, WIDTH, Trace, _trace


RESETS, HORIZON = (95101, 95102, 95103), 32
HEIGHT, HOLD = 548, 16
LABELS = ("逆着模型建议　−1 步", "顺着模型建议　+1 步")


@dataclass
class Arm:
    trace: Trace | None
    info: dict


def mean_drift_mm(trace, tick):
    if tick == 0:
        return 0.
    return float(1000 * np.abs(trace.states[1:tick + 1, 0] - trace.states[0, 0]).mean())


def load_arm(run, folder, reset):
    receipt = folder / "native.json"
    info = {"receipt": str(receipt.relative_to(run)), "source": None, "source_sha256": None}
    if not receipt.is_file():
        return Arm(None, {**info, "status": "NO_NATIVE_RECEIPT"})
    info["receipt_sha256"] = digest(receipt)
    native = json.loads(receipt.read_text(encoding="utf-8"))
    name = native.get("trace_file")
    if name is None:
        return Arm(None, {**info, "status": "NO_RECORDED_TRACE"})
    if name not in ("native/trace.npz", "native/partial_trace.npz"):
        raise ValueError(f"unexpected native recording path in {receipt}")
    path = folder / name
    source_hash = digest(path)
    if source_hash != native["trace_sha256"]:
        raise ValueError(f"recorded native digest differs from {path}")
    info.update(source=str(path.relative_to(run)), source_sha256=source_hash)
    with np.load(path, allow_pickle=False) as saved:
        n = int(saved["completed_transitions"])
        if name.endswith("partial_trace.npz") and len(saved["state45"]) == n == 0:
            return Arm(None, {**info, "status": "PARTIAL_WITHOUT_RESET_STATE"})
        if n < 0 or not np.allclose(saved["time_s"], np.arange(n+1) * DT, rtol=0, atol=1.e-6):
            raise ValueError(f"invalid 50 Hz native time base in {path}")
        truncated = bool(saved["truncated"].any())
    trace = _trace(path, reset, 0., HORIZON)
    if trace.ticks != n or trace.sha256 != source_hash:
        raise ValueError(f"native endpoint count or bytes changed in {path}")
    partial = name.endswith("partial_trace.npz")
    if not trace.stop_reason and (truncated or partial):
        trace.stop_reason = "提前截断" if truncated else "记录中止"
    return Arm(trace, {**info, "status": "PARTIAL_RECORDING" if partial else "RECORDED",
                       "ticks": n, "stop_reason": trace.stop_reason,
                       "end_displacement_mm": trace.drift_mm(n), "mean_absolute_drift_mm": mean_drift_mm(trace, n)})


@dataclass
class Pair:
    seed: int
    reset: int
    left: Arm
    right: Arm
    zero: Arm

    @property
    def shown(self):
        return self.left, self.right

    @property
    def last_tick(self):
        return max((arm.trace.ticks for arm in self.shown if arm.trace), default=0)

    def summary(self):
        return dict(seed=self.seed, reset=self.reset, left=self.left.info, right=self.right.info, zero=self.zero.info)


def load_pairs(run):
    if not run.is_dir():
        raise FileNotFoundError(run)
    pairs = []
    for seed in SEEDS:
        for reset in RESETS:
            folder = run / f"seed_{seed}" / f"reset_{reset}"
            arms = [load_arm(run, folder / tag, reset) for tag in ("m1", "p1", "zero")]
            available = [a.trace for a in arms if a.trace]
            if any(not np.array_equal(t.states[0], available[0].states[0]) for t in available):
                raise ValueError(f"the three recordings do not share their physical reset: {seed}/{reset}")
            pairs.append(Pair(seed, reset, *arms))
    # Open on physical footage if any exists; retain every missing comparison at the end.
    return sorted(pairs, key=lambda p: not any(a.trace for a in p.shown))


def draw_frame(pair, index, tick, picture, fonts):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title, label, small = fonts
    draw.text((16, 5), "D1  4° 驻坡 · 全身策略参数方向实测", font=title, fill=INK)
    draw.text((WIDTH-16, 9), f"{index}/9    策略种子 {pair.seed}    起点 {pair.reset}", anchor="ra", font=label, fill=MUTED)
    for i, arm in enumerate(pair.shown):
        x, trace = PANE_X[i], arm.trace
        draw.text((x+12, 48), LABELS[i], font=label, fill=ACCENTS[i])
        if trace is None:
            draw.rectangle((x, SCENE_Y, x+PANE_W-1, SCENE_Y+PANE_H-1), fill="#273741")
            draw.text((x+PANE_W/2, SCENE_Y+PANE_H/2), "缺少已核验的原生轨迹", font=label, fill=MUTED, anchor="mm")
            continue
        at = min(tick, trace.ticks)
        image.paste(picture(trace, at), (x, SCENE_Y))
        drift = trace.drift_mm(at)
        motion = "后溜" if drift <= 0 else "前移"
        draw.text((x+12, 442), f"当前{motion}  {abs(drift):.1f} mm", font=label, fill=INK)
        draw.text((x+12, 475), f"平均偏离  {mean_drift_mm(trace, at):.3f} mm", font=label, fill=ACCENTS[i])
        if at == trace.ticks:
            reason = trace.stop_reason or "到达时限"
            draw.text((x+PANE_W-12, 446), f"{reason} · {at*DT:.2f} s 定格", anchor="ra", font=small, fill="#ffd6a6")
    measured = any(a.trace for a in pair.shown)
    time_text = f"仿真 {tick*DT:.2f} s   ·   0.5× 播放" if measured else "该组缺少原生对照画面"
    draw.text((WIDTH-16, 52), time_text, anchor="ra", font=small, fill=MUTED)
    draw.text((16, 518), "橙线是驻车起点，平均偏离越小，驻坡越好。", font=small, fill=MUTED)
    base = pair.zero.trace
    if base and measured and tick <= base.ticks:
        zero_label = f"原参数同期平均偏离  {mean_drift_mm(base, tick):.3f} mm"
        if tick == base.ticks and base.stop_reason:
            zero_label += f" · {base.stop_reason}"
    elif base and measured:
        zero_label = f"原参数止于 {base.ticks*DT:.2f} s · {base.stop_reason or '记录结束'}"
    else:
        zero_label = "缺少同期原参数对照"
    draw.text((WIDTH-16, 518), zero_label, anchor="ra", font=small, fill=MUTED)
    return image


def encode(output, pairs, report):
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        raise RuntimeError("software offscreen EGL is required")
    import mujoco
    from PIL import Image
    from tools87.terrain87 import ground_height, load_model

    model = load_model("up")
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    xy = np.concatenate([a.trace.states[:, :2] for p in pairs for a in p.shown if a.trace])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (lo + hi) / 2
    data, camera, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    options.geomgroup[3] = 0
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -13.
    camera.distance = max(1.48, 1.22 + (hi[0]-lo[0])*.85 + (hi[1]-lo[1])*.3)
    camera.lookat[:] = (cx, cy, float(ground_height("up", cx)) + .22)
    fonts = chinese_font(29, True), chinese_font(23, True), chinese_font(19)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
            "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-maxrate", "850k",
            "-bufsize", "850k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(trace, tick):
            state = trace.states[tick]
            data.qpos[:], data.qvel[:], data.time = state[:23], state[23:], tick * DT
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            add_marks(mujoco, renderer.scene, "up", trace.states[0, 0], ground_height, lo[0]-.45, hi[0]+.45, cy)
            report["rendered_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            try:
                for i, pair in enumerate(pairs):
                    item = report["pairs"][i]
                    item["first_frame"] = report["frames"]
                    for tick in range(pair.last_tick+1):
                        pixels = draw_frame(pair, i+1, tick, picture, fonts).tobytes()
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


def render(run, output, receipt):
    run, output, receipt = map(Path, (run, output, receipt))
    if output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("video and receipt require unused output paths")
    report = dict(status="PARTIAL", file=output.name, fps=FPS, playback_speed=.5, source_dt_s=DT, frames=0,
                  cover_frames=0, endpoint_hold_frames=HOLD, rendered_poses=0, new_physics_steps=0, model_steps=0,
                  policy_calls=0, training_updates=0, left_alpha=-1., right_alpha=1., reference_alpha=0., pairs=[])
    with receipt.open("x", encoding="utf-8") as journal:
        try:
            pairs = load_pairs(run)
            report["pairs"] = [pair.summary() for pair in pairs]
            report["measured_panels"] = sum(a.trace is not None for p in pairs for a in p.shown)
            if not report["measured_panels"]:
                report.update(status="NO_MEASURED_ARMS", file=None, video_sha256=None)
                return report
            encode(output, pairs, report)
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of", "json", str(output)], check=True, capture_output=True, text=True, timeout=100)
            stream, = json.loads(probe.stdout)["streams"]
            frames = sum(1 + pair.last_tick + HOLD for pair in pairs)
            if ((stream["width"], stream["height"], stream["codec_name"], stream["r_frame_rate"]) != (WIDTH, HEIGHT, "h264", f"{FPS}/1")
                    or int(stream["nb_read_frames"]) != frames or report["frames"] != frames
                    or abs(float(stream["duration"]) - frames/FPS) > 1e-4 or not 0 < output.stat().st_size <= 5_000_000):
                raise ValueError(f"encoded video readback failed: {stream}")
            for p in pairs:
                for arm in (*p.shown, p.zero):
                    for field, sha_field in (("source", "source_sha256"), ("receipt", "receipt_sha256")):
                        if arm.info.get(sha_field) and digest(run / arm.info[field]) != arm.info[sha_field]:
                            raise ValueError(f"replay input changed: {arm.info[field]}")
            report.update(status="COMPLETE", video_sha256=digest(output), bytes=output.stat().st_size,
                          duration_s=frames/FPS, stream=stream)
        except BaseException as exc:
            report["reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(report, journal, ensure_ascii=False, allow_nan=False, indent=2)
            journal.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "output", "receipt"):
        parser.add_argument("--" + key, type=Path, required=True)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k: result.get(k) for k in ("status", "measured_panels", "frames", "video_sha256")}))
