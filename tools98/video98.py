"""Replay new endpoint-supervision holdouts using their saved poses and closed-loop forecasts."""
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
from tools96 import video96 as view


KINDS = ("native", "control", "endpoint")
LABELS = ("原生 MuJoCo 实测", "对照模型预测", "末端损失模型预测")
COLORS = ("#FFE0A0", "#A6CFFF", "#79DDB7")
WIDTH, HEIGHT, PANE_W, PANE_H = 1440, 702, 508, 232
FPS, HORIZON, HOLD = 25, 32, 18


def require(valid, message):
    if not valid:
        raise ValueError(message)


@dataclass
class Case:
    seed: int
    reset: int
    panels: dict[str, view.Panel]

    @property
    def last_tick(self):
        return max((arm.trace.ticks for p in self.panels.values() for arm in (p.p1, p.zero) if arm.trace), default=0)

    def snapshot(self):
        actual = self.panels["native"].p1.trace
        rows = {kind: p.summary() for kind, p in self.panels.items()}
        for kind in KINDS[1:]:
            prediction = self.panels[kind].p1.trace
            error = None
            if (actual and prediction and self.panels["native"].p1.info.get("h16_scoreable")
                    and self.panels[kind].p1.info.get("h16_scoreable")):
                estimate = prediction.states[16, :3].astype(np.float64)
                error = float(1000 * np.linalg.norm(estimate - actual.states[16, :3]))
            rows[kind]["h16_p1_body_position_error_mm"] = error
        return {"seed": self.seed, "reset": self.reset, "panels": rows}


def accepted_h16(arm, forecast):
    if not arm.trace or arm.trace.ticks < 16:
        return False
    if forecast["ticks"] > 16 or forecast["safe"]:
        return True
    # The old rollout stored a proposed endpoint before checking it. At tick 16
    # itself, require evidence that it passed validation before the next step.
    error = forecast.get("error") or ""
    rejected = "predicted fall" in error or "invalid predicted quaternion" in error
    return not rejected and forecast.get("attempted_model_steps", 0) > 16


def load_cases(native, predictions, seeds, resets, protocol_sha):
    used, seals, hashes = {}, {}, {}
    fit = native.parent / "fit_01"
    for stage, root in (("fit", fit), ("predictions", predictions), ("native", native)):
        path = root / "seal.json"
        data = path.read_bytes()
        hashes[stage] = used[path] = sha256(data).hexdigest()
        seals[stage] = json.loads(data)
        require(seals[stage]["protocol_sha256"] == protocol_sha and seals[stage]["stage"] == stage,
                f"incorrect {stage} experiment or seal")
    require(seals["native"]["previous_seal_sha256"] == hashes["predictions"], "native stage used different blind forecasts")
    require(seals["predictions"]["previous_seal_sha256"] == hashes["fit"], "forecasts used a different fit stage")
    checkpoints = {}
    for seed in seeds:
        for kind in KINDS[1:]:
            folder = fit / f"seed_{seed}" / kind
            path = folder / "receipt.json"
            if view.sealed_input(fit, path, seals["fit"], used):
                trained = json.loads(path.read_bytes())
                if trained["status"] == "COMPLETE":
                    key = view.sealed_input(fit, folder/"world.pt", seals["fit"], used)
                    require(key and key == trained["checkpoint"]["sha256"] and trained["updates_completed"] == 25,
                            "forecast checkpoint is not the sealed 25-update model")
                    checkpoints[seed, kind] = key
    starts = {}
    for reset in resets:
        file = predictions / f"start_{reset}.npz"
        if view.sealed_input(predictions, file, seals["predictions"], used):
            with np.load(file, allow_pickle=False) as saved:
                state = saved["state45"]
                require(state.shape == (1, 45) and state.dtype == np.float64 and np.isfinite(state).all(), "invalid reset")
                starts[reset] = state[0].copy()
    cases = []
    for seed in seeds:
        for reset in resets:
            rel = Path(f"seed_{seed}/reset_{reset}")
            native_pair = [view.native_arm(native, native / rel / tag, reset, seals["native"], used)
                           for tag in ("p1", "zero")]
            for arm in native_pair:
                scoreable = arm.trace and arm.trace.ticks >= 16 and arm.info["status"] == "RECORDED"
                if scoreable:
                    scoreable = not json.loads((native / arm.info["receipt"]).read_bytes()).get("error")
                arm.info["h16_scoreable"] = bool(scoreable)
            start = starts.get(reset)
            if start is None:
                start = next((a.trace.states[0] for a in native_pair if a.trace), None)
            panels = {"native": view.panel(*native_pair, start, True)}
            for kind in KINDS[1:]:
                predicted = [view.model_arm(predictions, predictions / rel / kind, tag, seals["predictions"], used,
                                            seed, reset) for tag in ("p1", "zero")]
                for arm in predicted:
                    if arm.info.get("receipt_sha256"):
                        data = json.loads((predictions / arm.info["receipt"]).read_bytes())
                        key = checkpoints.get((seed, kind))
                        require(key and data["checkpoint_sha256"] == key, "forecast names a different world checkpoint")
                        arm.info["checkpoint_sha256"] = key
                        arm.info["h16_scoreable"] = accepted_h16(arm, data)
                require(start is not None or not any(a.trace for a in predicted), "missing physical reset for forecast")
                panels[kind] = view.panel(*predicted, start, False)
            cases.append(Case(seed, reset, panels))
    return cases, used, hashes


def gain_scale(cases):
    maxima = [float(np.abs(p.gains).max()) for case in cases for p in case.panels.values() if len(p.gains)]
    peak = max([0.01, *maxima])
    unit = 10. ** math.floor(math.log10(peak))
    return math.ceil(peak / unit) * unit


def arm_note(arm, tick):
    if not arm.trace:
        return "缺测"
    if tick >= arm.trace.ticks:
        if arm.trace.stop_reason:
            return f"{arm.trace.stop_reason} · {arm.trace.ticks/50:.2f} s 止"
        if arm.trace.ticks < HORIZON:
            return f"{arm.trace.ticks/50:.2f} s 记录中止"
    return ""


def frame(case, summary, tick, index, picture, fonts, scale):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), view.BG)
    draw = ImageDraw.Draw(image)
    title, label, small = fonts
    draw.text((16, 5), "D1  4° 驻坡 · 世界模型的第 16 拍与第 32 拍盲测", font=title, fill=view.INK)
    draw.text((WIDTH-16, 12), f"{index}/9　策略 {case.seed}　新起点 {case.reset}　{tick/50:.2f} s · 0.5×",
              anchor="ra", font=small, fill=view.MUTED)
    native = case.panels["native"]
    for row, (arm, tag) in enumerate(((native.zero, "原参数（零步）"), (native.p1, "参数沿原方向 +1 步"))):
        y = 60 + 280*row
        draw.text((16, y), "原生实测 · " + tag, font=label, fill=COLORS[0])
        trace = arm.trace
        if trace:
            at = min(tick, trace.ticks)
            drift = trace.drift_mm(at)
            draw.text((16+PANE_W, y+4), f"{'后溜' if drift <= 0 else '前移'} {abs(drift):.1f} mm",
                      font=small, fill=view.MUTED, anchor="ra")
            image.paste(picture(trace, at), (16, y+34))
            note = arm_note(arm, tick)
            if note:
                box = draw.textbbox((27, y+39), note, font=small)
                draw.rectangle((box[0]-4, box[1]-3, box[2]+4, box[3]+3), fill=view.BG)
                draw.text((27, y+39), note, font=small, fill="#FFD5A1")
        else:
            draw.rectangle((16, y+34, 15+PANE_W, y+33+PANE_H), fill="#2A3740")
            draw.text((16+PANE_W/2, y+34+PANE_H/2), "缺少原生状态", font=small, fill=view.MUTED, anchor="mm")
    draw.text((16, 620), "橙线是驻车起点，两段实测使用同一相机。", font=small, fill=view.MUTED)

    draw.text((552, 59), "走 +1 步，平均能少偏离驻车点多少？（毫米）", font=label, fill=view.INK)
    x0, x1, y0, y1 = 608, 1405, 112, 325

    def chart_point(at, value):
        return x0 + at * (x1-x0)/HORIZON, (y0+y1)/2 - value*(y1-y0)/(2*scale)

    for val in (-scale, -scale/2, 0, scale/2, scale):
        _, y = chart_point(0, val)
        draw.line((x0, y, x1, y), fill="#74858D" if val == 0 else "#354651", width=1)
        draw.text((x0-8, y), f"{val:g}", font=small, fill=view.MUTED, anchor="rm")
    for at in (0, 8, 16, 24, 32):
        x, _ = chart_point(at, 0)
        if at == 16:
            draw.line((x, y0, x, y1), fill="#798B95", width=1)
        draw.text((x, y1+8), f"{at} 拍", font=small, fill=view.MUTED, anchor="ma")
    for i, kind in enumerate(KINDS):
        p = case.panels[kind]
        count = min(tick, len(p.gains))
        points = [chart_point(j, p.gains[j-1]) for j in range(1, count+1)]
        if len(points) >= 2:
            draw.line(points, fill=COLORS[i], width=4)
        if points:
            x, y = points[-1]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS[i])
        y = 375 + 39*i
        draw.line((555, y+13, 589, y+13), fill=COLORS[i], width=4)
        draw.text((602, y), LABELS[i], font=label, fill=COLORS[i])
        current = "等待首拍" if tick == 0 else "缺少同期对照"
        if 0 < tick <= len(p.gains):
            current = f"{tick} 拍收益   {p.gains[tick-1]:+.3f} mm"
        draw.text((847, y), current, font=label, fill=COLORS[i])
        first_note, base_note = arm_note(p.p1, tick), arm_note(p.zero, tick)
        notes = [prefix+note for prefix, note in (("+1 ", first_note), ("零步 ", base_note)) if note]
        if first_note and first_note == base_note:
            notes = ["+1/零步 " + first_note]
        if notes:
            draw.text((WIDTH-16, y+5), "  ".join(notes), font=small, fill="#FFD5A1", anchor="ra")
    draw.line((552, 504, WIDTH-16, 504), fill="#40515C")
    draw.text((552, 519), "第 16 拍 · +1 分支的机身三维位置误差", font=label, fill=view.INK)
    if tick < 16:
        draw.text((552, 560), "待原生和预测轨迹到达第 16 拍", font=small, fill=view.MUTED)
    else:
        for i, kind in enumerate(KINDS[1:]):
            error = summary["panels"][kind]["h16_p1_body_position_error_mm"]
            text = f"{error:.2f} mm" if error is not None else "缺测"
            draw.text((552+i*420, 558), f"{LABELS[i+1]}　{text}", font=label, fill=COLORS[i+1])
    draw.text((552, 611), "正收益表示 +1 比同一栏的零步对照更靠近驻车点。", font=small, fill=view.MUTED)
    draw.line((16, 652, WIDTH-16, 652), fill="#40515C")
    draw.text((16, 662), "两种模型从复位开始，各自闭环预测整段轨迹。第 16 拍的实测状态仅用于事后评分。", font=small, fill=view.INK)
    return image


def encode(target, cases, report):
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
            "software offscreen EGL is required")
    from PIL import Image
    import mujoco
    from tools87.terrain87 import ground_height, load_model

    model = load_model("up")
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    positions = np.concatenate([a.trace.states[:, :2] for c in cases
                                for a in (c.panels["native"].zero, c.panels["native"].p1) if a.trace])
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    cx, cy = (lo+hi)/2
    data, camera, opt = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[3] = 0
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -13.
    camera.distance = max(1.48, 1.22+.85*(hi[0]-lo[0])+.3*(hi[1]-lo[1]))
    camera.lookat[:] = cx, cy, float(ground_height("up", cx)) + .22
    fonts = chinese_font(28, True), chinese_font(22, True), chinese_font(18)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
            "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-maxrate", "1050k", "-bufsize", "1050k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(trace, tick):
            state = trace.states[tick]
            data.qpos[:], data.qvel[:], data.time = state[:23], state[23:], tick/50.
            mujoco.mj_forward(model, data)  # Position the recorded geometry without advancing time.
            renderer.update_scene(data, camera=camera, scene_option=opt)
            add_marks(mujoco, renderer.scene, "up", trace.states[0,0], ground_height, lo[0]-.45, hi[0]+.45, cy)
            report["rendered_native_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
            try:
                for i, (case, row) in enumerate(zip(cases, report["cells"], strict=True), 1):
                    row["first_frame"] = report["frames"]
                    for tick in range(case.last_tick+1):
                        pixels = frame(case, row, tick, i, picture, fonts, report["gain_scale_mm"]).tobytes()
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    row["endpoint_frame"] = report["frames"]-1
                    for _ in range(HOLD):
                        encoder.stdin.write(pixels)
                        report["frames"] += 1
                    row["last_frame"] = report["frames"]-1
            finally:
                encoder.stdin.close()
            require(encoder.wait() == 0, "video encoder failed")


def render(output, receipt):
    from tools98.common98 import BASE, PLAN_SHA, RESETS, SEEDS, protocol

    if output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("video and receipt require unused paths")
    protocol()
    report = dict(status="PARTIAL", file=output.name, protocol_sha256=PLAN_SHA, frames=0, fps=FPS, source_dt_s=.02,
                  playback_speed=.5, cover_frames=0, endpoint_hold_frames=HOLD, rendered_native_poses=0,
                  rendered_model_poses=0, new_physics_steps=0, model_steps=0, policy_calls=0, training_updates=0,
                  displayed_native_alphas=[0., 1.], gain="R(+1) minus R(0) in mm, physical origin, ticks 1 through current",
                  h16_display_metric="Euclidean error of the +1 predicted base xyz versus the measured base xyz, mm")
    with receipt.open("x", encoding="utf-8") as journal:
        try:
            cases, used, hashes = load_cases(BASE/"native_01", BASE/"predictions_01", SEEDS, RESETS, PLAN_SHA)
            report.update(cells=[case.snapshot() for case in cases], stage_seals_sha256=hashes, gain_scale_mm=gain_scale(cases))
            report["matched_cells"] = sum(bool(c.panels["native"].gains.size and
                                              any(c.panels[k].gains.size for k in KINDS[1:])) for c in cases)
            if not report["matched_cells"]:
                report.update(status="NO_MATCHED_NATIVE_AND_MODEL_MOTION", file=None, video_sha256=None)
                return report
            encode(output, cases, report)
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of", "json", str(output)], check=True, capture_output=True, text=True, timeout=100)
            stream, = json.loads(probe.stdout)["streams"]
            frames = sum(1+c.last_tick+HOLD for c in cases)
            require((stream["width"], stream["height"], stream["codec_name"], stream["r_frame_rate"])
                    == (WIDTH, HEIGHT, "h264", "25/1") and int(stream["nb_read_frames"]) == frames == report["frames"]
                    and abs(float(stream["duration"])-frames/FPS) < .0001 and 0 < output.stat().st_size <= 5_000_000,
                    "encoded video readback differs")
            require(all(digest(path) == expected for path, expected in used.items()), "sealed inputs changed during playback")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k: result.get(k) for k in ("status", "frames", "duration_s", "video_sha256")}))
