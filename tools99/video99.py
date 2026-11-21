"""Replay fresh D1 holdouts and compare full and late return gains on the same scale."""
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
from tools98.video98 import arm_note, require


KINDS = ("native", "control", "late")
TAGS = ("p1", "zero")
LABELS = ("原生实测", "对照模型预测", "后半段损失模型预测")
COLORS = ("#FFE0A0", "#A6CFFF", "#79DDB7")
CUT, END, FPS, HOLD = 16, 32, 25, 18
WIDTH, HEIGHT, PANE_W, PANE_H = 1460, 704, 496, 232


def suffix_gains(panel, origin):
    """R(+1) minus R(0) from tick 17 onward; both departures use the physical reset."""
    n = len(panel.gains) - CUT
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    plus, zero = (arm.trace.states[CUT+1:CUT+1+n, 0].astype(np.float64) for arm in (panel.p1, panel.zero))
    reduction = np.abs(zero - origin) - np.abs(plus - origin)
    return 1000 * np.cumsum(reduction, dtype=np.float64) / np.arange(1, n+1)


@dataclass
class Case:
    seed: int
    reset: int
    origin: float | None
    panels: dict[str, view.Panel]
    late: dict[str, np.ndarray]

    @property
    def end(self):
        return max((arm.trace.ticks for p in self.panels.values() for arm in (p.p1, p.zero) if arm.trace), default=0)

    def summary(self):
        rows = {}
        for kind, panel in self.panels.items():
            late = self.late[kind]
            rows[kind] = {**panel.summary(), "full_gain_mm_by_tick": panel.gains.tolist(),
                          "late_gain_mm_by_tick": late.tolist(), "late_common_ticks": len(late),
                          "late_gain_mm": float(late[-1]) if len(late) == END-CUT else None}
        return dict(seed=self.seed, reset=self.reset, physical_origin_x_m=self.origin, panels=rows)


def read_stages(base, plan_sha):
    folders, seals, used = {}, {}, {}
    previous = None
    for stage in ("fit", "predictions", "native"):
        folder = folders[stage] = base / (stage + "_01")
        path = folder / "seal.json"
        raw = path.read_bytes()
        sig = used[path] = sha256(raw).hexdigest()
        record = seals[stage] = json.loads(raw)
        require(record["stage"] == stage and record["protocol_sha256"] == plan_sha, f"wrong sealed {stage} experiment")
        if previous:
            require(record["previous_seal_sha256"] == previous, f"{stage} does not follow its sealed predecessor")
        previous = sig
    return folders, seals, used


def fit_checkpoints(folder, seal, used, seeds):
    checkpoints = {}
    for seed in seeds:
        for kind in KINDS[1:]:
            arm = folder / f"seed_{seed}" / kind
            receipt = arm / "receipt.json"
            if not view.sealed_input(folder, receipt, seal, used):
                continue
            fit = json.loads(receipt.read_bytes())
            if fit["status"] == "COMPLETE":
                key = view.sealed_input(folder, arm/"world.pt", seal, used)
                require(key and fit["updates_completed"] == 25 and fit["checkpoint"]["sha256"] == key,
                        f"unverified final world checkpoint: {arm}")
                checkpoints[seed, kind] = key
    return checkpoints


def load_cases(base, seeds, resets, plan_sha):
    folders, seals, used = read_stages(base, plan_sha)
    checkpoints = fit_checkpoints(folders["fit"], seals["fit"], used, seeds)
    forecast_dir, native_dir = folders["predictions"], folders["native"]
    starts = {}
    for reset in resets:
        path = forecast_dir / f"start_{reset}.npz"
        if view.sealed_input(forecast_dir, path, seals["predictions"], used):
            with np.load(path, allow_pickle=False) as archive:
                state = archive["state45"]
            require(state.shape == (1, 45) and state.dtype == np.float64 and np.isfinite(state).all(), "invalid physical reset")
            starts[reset] = state[0]
    cases = []
    for seed in seeds:
        for reset in resets:
            relative = Path(f"seed_{seed}/reset_{reset}")
            native_arms = [view.native_arm(native_dir, native_dir/relative/tag, reset, seals["native"], used) for tag in TAGS]
            start = starts.get(reset)
            if start is None:
                start = next((a.trace.states[0] for a in native_arms if a.trace), None)
            panels = {"native": view.panel(*native_arms, start, True)}
            for kind in KINDS[1:]:
                arms = [view.model_arm(forecast_dir, forecast_dir/relative/kind, tag, seals["predictions"], used,
                                       seed, reset) for tag in TAGS]
                for arm in arms:
                    if arm.info.get("receipt_sha256"):
                        row = json.loads((forecast_dir/arm.info["receipt"]).read_bytes())
                        expected = checkpoints.get((seed, kind))
                        require(expected and row["checkpoint_sha256"] == expected, "forecast refers to a different model fit")
                        arm.info["checkpoint_sha256"] = expected
                require(start is not None or not any(a.trace for a in arms), "forecast has no measured reset")
                panels[kind] = view.panel(*arms, start, False)
            origin = float(start[0]) if start is not None else None
            late = {k: suffix_gains(p, origin) for k, p in panels.items()}
            cases.append(Case(seed, reset, origin, panels, late))
    hashes = {stage: used[folder/"seal.json"] for stage, folder in folders.items()}
    return cases, used, hashes


def shared_scale(cases):
    all_series = [series for c in cases for k, p in c.panels.items() for series in (p.gains, c.late[k]) if len(series)]
    peak = max([.01, *(float(np.abs(s).max()) for s in all_series)])
    unit = 10. ** math.floor(math.log10(peak))
    return math.ceil(peak/unit)*unit


def plot(draw, bounds, first, tick, lines, fonts, scale):
    x0, y0, x1, y1 = bounds
    origin_tick = first-1
    _, _, small = fonts

    def point(t, v):
        return x0 + (t-origin_tick)*(x1-x0)/(END-origin_tick), (y0+y1)/2 - v*(y1-y0)/(2*scale)

    for value in (-scale, -scale/2, 0, scale/2, scale):
        _, y = point(first, value)
        draw.line((x0, y, x1, y), fill="#75858D" if value == 0 else "#374855")
        draw.text((x0-7, y), f"{value:g}", font=small, anchor="rm", fill=view.MUTED)
    for t in (origin_tick, (END+origin_tick)//2, END):
        x, _ = point(t, 0)
        draw.text((x, y1+8), f"{t} 拍", font=small, anchor="ma", fill=view.MUTED)
    if tick < first:
        draw.text(((x0+x1)/2, (y0+y1)/2), f"等待第 {first} 拍", font=small, anchor="mm", fill=view.MUTED)
    for kind, color in zip(KINDS, COLORS, strict=True):
        data = lines[kind]
        length = min(max(0, tick-origin_tick), len(data))
        points = [point(origin_tick+j+1, value) for j, value in enumerate(data[:length])]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        if points:
            x, y = points[-1]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)


def reading(series, count):
    if count <= 0:
        return "待计分"
    return f"{series[count-1]:+.3f} mm" if count <= len(series) else "缺少同期对照"


def frame(case, tick, index, picture, fonts, scale):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), view.BG)
    draw = ImageDraw.Draw(image)
    title, label, small = fonts
    draw.text((16, 4), "D1  4° 驻坡 · 监督后半段动作响应后，预测更准吗？", font=title, fill=view.INK)
    draw.text((WIDTH-16, 11), f"{index}/9　策略 {case.seed}　新起点 {case.reset}　{tick/50:.2f} s · 0.5×",
              anchor="ra", font=small, fill=view.MUTED)
    native = case.panels["native"]
    for i, (arm, heading) in enumerate(((native.zero, "原参数（零步）"), (native.p1, "沿原方向 +1 步"))):
        y = 60 + 280*i
        draw.text((16, y), "原生实测 · " + heading, font=label, fill=COLORS[0])
        if arm.trace is None:
            draw.rectangle((16, y+34, 15+PANE_W, y+33+PANE_H), fill="#2B3943")
            draw.text((16+PANE_W/2, y+34+PANE_H/2), "缺少实测状态", anchor="mm", font=small, fill=view.MUTED)
            continue
        at = min(tick, arm.trace.ticks)
        distance = arm.trace.drift_mm(at)
        direction = "后溜" if distance <= 0 else "前移"
        draw.text((16+PANE_W, y+4), f"{direction} {abs(distance):.1f} mm", anchor="ra", font=small, fill=view.MUTED)
        image.paste(picture(arm.trace, at), (16, y+34))
        note = arm_note(arm, tick)
        if note:
            box = draw.textbbox((26, y+40), note, font=small)
            draw.rectangle((box[0]-4, box[1]-3, box[2]+4, box[3]+3), fill=view.BG)
            draw.text((26, y+40), note, font=small, fill="#FFD5A1")
    draw.text((16, 618), "橙线标记原始驻车点，两段实测共用相机。", font=small, fill=view.MUTED)
    full = {k: p.gains for k, p in case.panels.items()}
    draw.text((548, 58), "全程 · 第 1～32 拍", font=label, fill=view.INK)
    draw.text((1002, 58), "后半程 · 第 17～32 拍", font=label, fill=view.INK)
    draw.text((WIDTH-18, 61), "相同纵轴（毫米）", anchor="ra", font=small, fill=view.MUTED)
    plot(draw, (582, 116, 937, 328), 1, tick, full, fonts, scale)
    plot(draw, (1045, 116, 1422, 328), CUT+1, tick, case.late, fonts, scale)
    draw.text((548, 377), "相对零步少偏离多少？", font=small, fill=view.MUTED)
    draw.text((877, 377), f"全程至第 {tick} 拍", font=small, fill=view.MUTED)
    draw.text((1148, 377), "后半程累计", font=small, fill=view.MUTED)
    for i, kind in enumerate(KINDS):
        y = 415 + 44*i
        draw.line((550, y+12, 577, y+12), fill=COLORS[i], width=4)
        draw.text((590, y), LABELS[i], font=label, fill=COLORS[i])
        draw.text((877, y), reading(full[kind], tick), font=label, fill=COLORS[i])
        draw.text((1148, y), reading(case.late[kind], tick-CUT), font=label, fill=COLORS[i])
    notes = []
    for k, name in zip(KINDS[1:], ("对照", "新损失"), strict=True):
        panel = case.panels[k]
        for arm, tag in ((panel.p1, "+1"), (panel.zero, "零步")):
            note = arm_note(arm, tick)
            if note:
                notes.append(f"{name}{tag} {note}")
    for i in range(0, len(notes), 2):
        draw.text((548, 551 + 24*(i//2)), "　".join(notes[i:i+2]), font=small, fill="#FFD5A1")
    draw.text((548, 617), "正数表示 +1 步比本栏零步更靠近原始驻车点。", font=small, fill=view.MUTED)
    draw.line((16, 650, WIDTH-16, 650), fill="#42515C")
    draw.text((16, 662), "盲测图比较 +1 步和零步。训练新增的损失比较正、反两步的逐拍纵向位移。两种模型均从本轮新起点独立预测。", font=small, fill=view.INK)
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
    xy = np.concatenate([arm.trace.states[:, :2] for c in cases
                         for arm in (c.panels["native"].p1, c.panels["native"].zero) if arm.trace])
    low, high = xy.min(axis=0), xy.max(axis=0)
    cx, cy = (low+high)/2
    data, camera, option = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 100., -13.
    camera.distance = max(1.48, 1.22 + .85*(high[0]-low[0]) + .3*(high[1]-low[1]))
    camera.lookat[:] = cx, cy, float(ground_height("up",cx))+.22
    fonts = chinese_font(28,True), chinese_font(22,True), chinese_font(18)
    command = ["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-n","-threads","1",
               "-f","rawvideo","-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),"-i","-",
               "-an","-filter_threads","1","-c:v","libx264","-preset","veryfast","-crf","21",
               "-maxrate","1050k","-bufsize","1050k","-threads","1","-x264-params","threads=1:lookahead-threads=1",
               "-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(trace, tick):
            state = trace.states[tick]
            data.qpos[:], data.qvel[:], data.time = state[:23], state[23:], tick/50.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            add_marks(mujoco, renderer.scene, "up", trace.states[0,0], ground_height, low[0]-.45, high[0]+.45, cy)
            report["rendered_native_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for index, (case, row) in enumerate(zip(cases, report["cells"], strict=True), 1):
                    row["first_frame"] = report["frames"]
                    for tick in range(case.end+1):
                        pixels = frame(case, tick, index, picture, fonts, report["gain_scale_mm"]).tobytes()
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
    from tools99.common99 import BASE, PLAN_SHA, RESETS, SEEDS, protocol

    if output.exists() or receipt.exists() or output == receipt:
        raise FileExistsError("video and receipt require unused output paths")
    protocol()
    report = dict(status="PARTIAL", file=output.name, protocol_sha256=PLAN_SHA, frames=0, fps=FPS, source_dt_s=.02,
                  playback_speed=.5, cover_frames=0, endpoint_hold_frames=HOLD, rendered_native_poses=0, rendered_model_poses=0,
                  new_physics_steps=0, model_steps=0, policy_calls=0, training_updates=0, displayed_native_alphas=[0.,1.],
                  full_ticks=[1,END], late_ticks=[CUT+1,END], plotted_gain="R(+1) minus R(0) about the original physical reset, mm")
    with receipt.open("x",encoding="utf-8") as journal:
        try:
            cases, used, hashes = load_cases(BASE, SEEDS, RESETS, PLAN_SHA)
            report.update(cells=[case.summary() for case in cases], stage_seals_sha256=hashes, gain_scale_mm=shared_scale(cases))
            report["matched_cells"] = sum(bool(c.panels["native"].gains.size and
                                              any(c.panels[k].gains.size for k in KINDS[1:])) for c in cases)
            if not report["matched_cells"]:
                report.update(status="NO_MATCHED_NATIVE_AND_MODEL_MOTION", file=None, video_sha256=None)
                return report
            encode(output, cases, report)
            probe = subprocess.run(["ffprobe","-v","error","-threads","1","-count_frames","-select_streams","v:0",
                                    "-show_entries","stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of","json",str(output)],check=True,capture_output=True,text=True,timeout=100)
            stream, = json.loads(probe.stdout)["streams"]
            frames = sum(c.end+1+HOLD for c in cases)
            require((stream["width"],stream["height"],stream["codec_name"],stream["r_frame_rate"]) == (WIDTH,HEIGHT,"h264","25/1")
                    and int(stream["nb_read_frames"]) == frames == report["frames"] and
                    abs(float(stream["duration"])-frames/FPS) < .0001 and 0 < output.stat().st_size <= 5_000_000,
                    "invalid encoded frame count, dimensions, duration or size")
            require(all(digest(p) == expected for p,expected in used.items()), "sealed source changed during playback")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k:result.get(k) for k in ("status","frames","duration_s","video_sha256")}))
