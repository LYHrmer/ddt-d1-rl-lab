"""Replay measured D1 cold-start trajectories, directly from their first frame."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import add_marks, chinese_font, digest
from tools94.common94 import CALIBRATION, EVALUATION, SEEDS


ROOT = Path(__file__).resolve().parents[1]
DT, FPS, END_HOLD = 0.02, 25, 8
WIDTH, HEIGHT, PANE_W, PANE_H = 1280, 500, 632, 350
PANE_X, SCENE_Y = (4, 644), 86
INK, MUTED, BG = "#F4F7F7", "#B7C7D4", "#17222B"
ACCENTS = ("#9CCAFF", "#87E2C2")
ARMS = {
    "calibration": ("四轮目标各减 0.20 rad/s", "四轮目标各加 0.20 rad/s"),
    "policy": ("训练前 · 随机初始化", "训练后 · 世界模型更新策略"),
}


@dataclass
class Trace:
    path: Path
    sha256: str
    states: np.ndarray
    stop_reason: str

    @property
    def ticks(self):
        return len(self.states) - 1

    def drift_mm(self, tick):
        return float((self.states[tick, 0] - self.states[0, 0]) * 1000)


@dataclass
class Pair:
    seed: int
    reset: int
    round: int
    kind: str
    status: str
    status_label: str
    arms: tuple[Trace, Trace]

    @property
    def last_tick(self):
        return max(a.ticks for a in self.arms)


def _trace(path, reset, offset, horizon):
    before = digest(path)
    with np.load(path, allow_pickle=False) as saved:
        states = saved["state45"]
        n = len(states) - 1
        if (states.shape != (n + 1, 45) or not 0 <= n <= horizon or not np.isfinite(states).all()
                or not np.allclose(np.linalg.norm(states[:, 3:7], axis=1), 1., rtol=0, atol=1e-5)):
            raise ValueError(f"invalid native trajectory in {path}")
        if int(saved["reset_seed"].item()) != reset or abs(float(saved["wheel_offset_rad_s"].item()) - offset) > 1e-7:
            raise ValueError(f"reset or wheel intervention does not match the manifest: {path}")
        if int(saved["requested_ticks"].item()) != horizon or int(saved["explore_seed"].item()) != -1:
            raise ValueError(f"video needs a measured, deterministic calibration or evaluation recording: {path}")
        events = {key: saved[key] for key in ("fallen", "illegal", "terminated", "truncated")}
        if any(flag.shape != (n,) or flag.dtype != bool for flag in events.values()):
            raise ValueError(f"invalid episode event records: {path}")
        if any(np.any(flag[:-1]) for flag in events.values()):
            raise ValueError(f"trajectory continues after its terminal event: {path}")
    reasons = [label for key, label in (("fallen", "跌倒"), ("illegal", "非法接触")) if np.any(events[key])]
    if not reasons and np.any(events["terminated"]):
        reasons.append("回合终止")
    if not reasons and n < horizon:
        reasons.append("提前截断" if np.any(events["truncated"]) else "记录中止")
    return Trace(path, before, states, " / ".join(reasons))


def load_pairs(manifest):
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    pairs = []
    sources = set()
    cells = set()
    for row in sorted(payload["pairs"], key=lambda r: (r["seed"], r["reset"])):
        seed, reset, round_number, kind = row["seed"], row["reset"], row["round"], row["kind"]
        if seed not in SEEDS or kind not in ARMS or (seed, reset) in cells or round_number not in (1, 2):
            raise ValueError(f"invalid or repeated display cell: {seed}/{reset}")
        eligible = CALIBRATION[round_number-1] if kind == "calibration" else EVALUATION
        if reset not in eligible:
            raise ValueError(f"reset {reset} outside the registered {kind} group")
        horizon = 100 if kind == "policy" else 50
        offsets = (0., 0.) if kind == "policy" else (-.20, .20)
        arms = []
        for side, offset in zip(("left", "right"), offsets, strict=True):
            path = (manifest.parent / row[side]).resolve(strict=True)
            if path in sources:
                raise ValueError(f"duplicate native recording: {path}")
            sources.add(path)
            arm = _trace(path, reset, offset, horizon)
            if f"{side}_sha256" in row and row[f"{side}_sha256"] != arm.sha256:
                raise ValueError(f"manifest hash differs from native source: {path}")
            arms.append(arm)
        if not np.array_equal(arms[0].states[0], arms[1].states[0]):
            raise ValueError(f"the comparison does not share a measured initial state: {seed}/{reset}")
        pairs.append(Pair(seed, reset, round_number, kind, str(row["status"]), str(row.get("status_label", "")), tuple(arms)))
        cells.add((seed, reset))
    if not pairs:
        raise ValueError("no recorded physical pairs in the manifest")
    return pairs


def _source_metadata(trace):
    path = trace.path.relative_to(ROOT) if trace.path.is_relative_to(ROOT) else trace.path
    return dict(trace=str(path), sha256=trace.sha256, ticks=trace.ticks, stop_reason=trace.stop_reason,
                source_end_s=trace.ticks * DT, final_displacement_mm=trace.drift_mm(trace.ticks),
                first_second_rollback_mm=max(0., -trace.drift_mm(50)) if trace.ticks >= 50 else None)


def _frame(pair, index, total, tick, picture, fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(frame)
    title = f"第 {pair.round} 轮轮地响应校准" if pair.kind == "calibration" else "策略训练前后"
    if pair.status_label:
        title += f"  ·  {pair.status_label}"
    draw.text((16, 5), f"D1  4° 驻坡  ·  {title}", fill=INK, font=fonts[0])
    draw.text((WIDTH-16, 8), f"{index}/{total}    种子 {pair.seed}    起点 {pair.reset}",
              anchor="ra", fill=MUTED, font=fonts[1])
    labels = ARMS[pair.kind]
    for side, arm in enumerate(pair.arms):
        x = PANE_X[side]
        draw.text((x+11, 48), labels[side], fill=ACCENTS[side], font=fonts[1])
        at = min(tick, arm.ticks)
        frame.paste(picture(arm, at), (x, SCENE_Y))
        delta = arm.drift_mm(at)
        motion = "后溜" if delta <= 0 else "前移"
        draw.text((x+11, 446), f"{motion}  {abs(delta):.1f} mm", fill=ACCENTS[side], font=fonts[0])
        if at == arm.ticks and arm.stop_reason:
            note = f"{arm.stop_reason} · 止于 {at*DT:.2f} s"
        elif tick > 50:
            note = f"首秒后溜 {max(0., -arm.drift_mm(50)):.1f} mm" if arm.ticks >= 50 else ""
        else:
            note = "橙线为机身起点"
        draw.text((x+PANE_W-11, 456), note, fill=MUTED, anchor="ra", font=fonts[2])
    speed = "+0.20" if tick > 50 else "0"
    draw.text((WIDTH-16, 51), f"指令 {speed} m/s   ·   {tick*DT:.2f} s   ·   0.5×", anchor="ra",
              fill=MUTED, font=fonts[2])
    return frame


def _encode(output, pairs, report):
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
        raise RuntimeError("use software offscreen EGL for measured-state playback")
    import mujoco
    from PIL import Image
    from tools87.terrain87 import ground_height, load_model

    model = load_model("up")
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    data, camera, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    options.geomgroup[3] = 0
    xy = np.concatenate([a.states[:, :2] for pair in pairs for a in pair.arms])
    low, high = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (low + high) / 2
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -13.
    camera.distance = max(1.48, 1.22 + (high[0]-low[0]) * .85 + (high[1]-low[1]) * .3)
    camera.lookat[:] = (cx, cy, float(ground_height("up", cx)) + .22)
    fonts = (chinese_font(29, True), chinese_font(23, True), chinese_font(19))
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
               "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-maxrate", "850k", "-bufsize", "850k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(arm, tick):
            state = arm.states[tick]
            data.qpos[:], data.qvel[:], data.time = state[:23], state[23:], tick * DT
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            add_marks(mujoco, renderer.scene, "up", arm.states[0, 0], ground_height, low[0]-.45, high[0]+.45, cy)
            report["rendered_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as encoder:
            try:
                for index, pair in enumerate(pairs, 1):
                    start = report["frames"]
                    for tick in range(pair.last_tick + 1):
                        pixels = _frame(pair, index, len(pairs), tick, picture, fonts).tobytes()
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    for _ in range(END_HOLD):
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    report["pairs"].append(dict(seed=pair.seed, reset=pair.reset, round=pair.round, kind=pair.kind,
                                                status=pair.status, status_label=pair.status_label,
                                                first_frame=start, endpoint_frame=start+pair.last_tick,
                                                last_frame=report["frames"]-1, left=_source_metadata(pair.arms[0]),
                                                right=_source_metadata(pair.arms[1])))
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"video encoder failed ({encoder.returncode})")


def run(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "video.mp4"
    report = dict(status="PARTIAL", file=target.name, frames=0, fps=FPS, playback_speed=.5, source_dt_s=DT,
                  cover_frames=0, endpoint_hold_frames=END_HOLD, new_physics_steps=0, model_steps=0,
                  policy_calls=0, training_updates=0, rendered_poses=0, pairs=[])
    try:
        report["manifest_sha256"] = digest(manifest)
        pairs = load_pairs(manifest)
        _encode(target, pairs, report)
        probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                "-of", "json", str(target)], check=True, capture_output=True, text=True, timeout=100)
        stream, = json.loads(probe.stdout)["streams"]
        frames = sum(pair.last_tick + 1 + END_HOLD for pair in pairs)
        if ((stream["width"], stream["height"], stream["codec_name"], stream["r_frame_rate"]) != (WIDTH, HEIGHT, "h264", "25/1")
                or int(stream["nb_read_frames"]) != frames or report["frames"] != frames
                or abs(float(stream["duration"]) - frames/FPS) > .0001 or not 0 < target.stat().st_size <= 5_000_000):
            raise ValueError(f"encoded video readback failed: {stream}")
        if digest(manifest) != report["manifest_sha256"] or any(digest(a.path) != a.sha256 for pair in pairs for a in pair.arms):
            raise ValueError("measured source or manifest changed during playback")
        report.update(status="COMPLETE", bytes=target.stat().st_size, sha256=digest(target), duration_s=frames/FPS, stream=stream)
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        with (output / "video.json").open("x", encoding="utf-8") as journal:
            json.dump(report, journal, ensure_ascii=False, allow_nan=False, indent=2)
            journal.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.output)
    print(json.dumps({k: result[k] for k in ("status", "frames", "duration_s", "sha256")}, ensure_ascii=False))
