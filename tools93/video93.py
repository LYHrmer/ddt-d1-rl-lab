"""Render the nine measured standing pairs from the recorded states at half speed."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import add_marks, chinese_font
from tools93.common93 import ARMS, BASE, PARENTS, PLAN_SHA, RESETS, ROOT, checked_plan, digest, read, require, write


FPS, TICKS, DT = 25, 50, .02
INTRO_FRAMES, ENDPOINT_FRAMES, SUMMARY_FRAMES = 75, 32, 125
WIDTH, HEIGHT, PANE_W, PANE_H = 1280, 720, 608, 310
LEFT_X, RIGHT_X, SCENE_Y = 24, 648, 128
BACKGROUND, TEXT, MUTED, GRID = "#12202C", "#F5F7F7", "#B8C8D3", "#455867"
COLORS = ("#9CCAFF", "#79E0BD")
OLD = ROOT / "evidence/wbc90/stand_01"


@dataclass
class Arm:
    q: np.ndarray
    v: np.ndarray
    rollback_mm: np.ndarray
    path: Path
    sha: str


@dataclass
class Pair:
    parent: int
    reset: int
    arms: tuple

    def effect_mm(self, tick):
        return float(self.arms[0].rollback_mm[tick] - self.arms[1].rollback_mm[tick])


def frame_counts():
    total = INTRO_FRAMES + len(PARENTS)*len(RESETS)*(TICKS + ENDPOINT_FRAMES) + SUMMARY_FRAMES
    return dict(intro=INTRO_FRAMES, movement_per_pair=TICKS, endpoint_per_pair=ENDPOINT_FRAMES,
                summary=SUMMARY_FRAMES, total=total)


def _rows_by_key(rows, fields):
    indexed = {tuple(row[k] for k in fields): row for row in rows}
    require(len(indexed) == len(rows), "duplicate video source identifier")
    return indexed


def _arm(folder, row, horizon, policy_sha):
    require(row["status"] == "COMPLETE" and row["clean"] and row["ticks"] == horizon
            and row["policy_sha256"] == policy_sha, "incomplete or mismatched physical video source")
    relative = Path(row["trace"])
    path = folder / relative
    require(not relative.is_absolute() and path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            and path.is_file() and digest(path) == row["trace_sha256"], "physical video source hash or location")
    with np.load(path, allow_pickle=False) as saved:
        q, v = saved["qpos"], saved["qvel"]
        t, command = saved["time_s"], saved["command"]
        require(q.shape == (horizon, 23) and v.shape == (horizon, 22) and t.shape == (horizon,)
                and command.shape == (horizon, 3) and not np.any(command[:TICKS])
                and np.allclose(t[:TICKS], np.arange(1, TICKS+1)*DT, atol=1e-9, rtol=0),
                "video requires a measured first second of zero-command endpoints")
        q = np.vstack((saved["initial_qpos"], q[:TICKS]))
        v = np.vstack((saved["initial_qvel"], v[:TICKS]))
    require(q.shape == (TICKS+1, 23) and v.shape == (TICKS+1, 22) and q[0, 0] == 1.10 and not v[0].any()
            and np.isfinite(q).all() and np.isfinite(v).all(), "invalid slope start or recorded state")
    rollback = np.maximum(0., q[0, 0] - q[:, 0])
    require(abs(float(rollback[-1]) - row["rollback_m"]) < 1e-12, "physical arm readout differs from its receipt")
    return Arm(q, v, rollback*1000, path, row["trace_sha256"])


def load_pairs(plan, score, old_folder, new_folder):
    old, new = read(old_folder / "report.json"), read(new_folder / "seal.json")
    require(old["status"] == new["status"] == "COMPLETE" and new["protocol_sha256"] == PLAN_SHA
            and score["protocol_sha256"] == PLAN_SHA, "video inputs are incomplete or unregistered")
    fields = ("parent_seed", "reset_seed", "arm")
    historical = _rows_by_key(old["cases"], fields)
    acquired = _rows_by_key(new["cases"], fields)
    official = _rows_by_key(score["cells"], fields[:2])
    matrix = [(p, r) for p in PARENTS for r in RESETS]
    diagonal = set(zip(PARENTS, RESETS, strict=True))
    require(official.keys() == set(matrix)
            and historical.keys() == {(p, r, a) for p, r in diagonal for a in ARMS}
            and acquired.keys() == {(p, r, a) for p, r in matrix if (p, r) not in diagonal for a in ARMS},
            "video requires all nine physical pairs and six reused diagonal arms")
    starts, pairs = {}, []
    for parent, reset in matrix:
        older = (parent, reset) in diagonal
        rows, folder, horizon = (historical, old_folder, 100) if older else (acquired, new_folder, TICKS)
        arms = tuple(_arm(folder, rows[parent, reset, arm], horizon, plan["policies_sha256"][str(parent)])
                     for arm in ARMS)
        start = starts.setdefault(reset, arms[0].q[0])
        require(all(np.array_equal(a.q[0], start) for a in arms), "different starting pose under a shared reset")
        pair = Pair(parent, reset, arms)
        scores = official[parent, reset]
        require(np.allclose([a.rollback_mm[-1]/1000 for a in arms], scores["truth"], atol=1e-12, rtol=0)
                and abs(pair.effect_mm(TICKS)/1000 - scores["true_effect_m"]) < 1e-12,
                "video readout differs from the scored physical pair")
        pairs.append(pair)
    return pairs


def _intro(fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(frame)
    draw.text((54, 55), "D1 轮足机器人  /  4° 上坡", font=fonts["label"], fill=MUTED)
    draw.text((54, 119), "速度指令为零，机器人在坡上会退多远？", font=fonts["title"], fill=TEXT)
    draw.text((54, 220), "每组从同一姿态起步，用同一份感知 PPO 和同一套关节伺服。", font=fonts["label"], fill=TEXT)
    for y, color, name, detail in ((325, COLORS[0], "左侧   原策略", "直接执行全身策略给出的轮速目标"),
                                    (416, COLORS[1], "右侧   同一策略", "抑制四轮目标中共同倒转的部分")):
        draw.rectangle((55, y, 62, y+44), fill=color)
        draw.text((81, y-1), name, fill=color, font=fonts["label"])
        draw.text((370, y-1), detail, fill=TEXT, font=fonts["label"])
    draw.line((55, 557, WIDTH-55, 557), fill=GRID, width=2)
    draw.text((55, 584), "3 份策略 × 3 个重置  ·  9 组全部播放  ·  每组实测 1 秒，以 0.5 倍速回放",
              fill=MUTED, font=fonts["body"])
    return frame


def _summary(pairs, fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(frame)
    d.text((35, 23), "驻坡首秒的实际后溜", fill=TEXT, font=fonts["title"])
    d.text((36, 85), "每格先列原策略与修正后的距离，再给出减少量。单位均为毫米。", fill=MUTED, font=fonts["body"])
    grid_x, grid_y, cell_w, cell_h = 229, 207, 335, 135
    for j, reset in enumerate(RESETS):
        d.text((grid_x + j*cell_w + cell_w//2, 170), f"重置 {reset}", anchor="ma", font=fonts["label"], fill=TEXT)
    for i, parent in enumerate(PARENTS):
        y = grid_y + i*cell_h
        d.text((34, y+35), f"策略 {parent}", fill=TEXT, font=fonts["label"])
        for j in range(len(RESETS)):
            pair = pairs[i*len(RESETS) + j]
            a, b = (arm.rollback_mm[-1] for arm in pair.arms)
            x = grid_x + j*cell_w
            d.rectangle((x, y, x+cell_w-9, y+cell_h-8), fill="#203342")
            d.text((x+17, y+15), f"{a:.1f}  →  {b:.1f}", font=fonts["number"], fill=TEXT)
            d.text((x+18, y+71), f"少退 {pair.effect_mm(TICKS):+.2f} mm", fill=COLORS[1], font=fonts["body"])
    d.text((36, 652), "正数表示抑制共同倒转后少退，负数表示多退。", font=fonts["body"], fill=MUTED)
    return frame


def _chart(draw, pair, tick, limit, fonts):
    left, right, top, bottom = 73, 905, 555, 652
    draw.text((26, 508), "相对起点的后溜距离（mm）", font=fonts["small"], fill=MUTED)
    for value in (0, limit // 2, limit):
        y = bottom - value / limit * (bottom - top)
        draw.line((left, y, right, y), fill=GRID)
        draw.text((left-8, y), str(value), font=fonts["small"], anchor="rm", fill=MUTED)
    for i, arm in enumerate(pair.arms):
        points = [(left+k/TICKS*(right-left), bottom-v/limit*(bottom-top))
                  for k, v in enumerate(arm.rollback_mm[:tick+1])]
        if len(points) > 1:
            draw.line(points, fill=COLORS[i], width=3)
        x, y = points[-1]
        draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS[i])
    draw.text((left, bottom+3), "0.00 s", fill=MUTED, font=fonts["small"], anchor="la")
    draw.text((right, bottom+3), "1.00 s", fill=MUTED, font=fonts["small"], anchor="ra")


def _pair_frame(pair, index, tick, limit, picture, fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(frame)
    d.text((24, 9), "D1 坡上驻车", font=fonts["title"], fill=TEXT)
    d.text((WIDTH-24, 17), f"{index}/9    策略 {pair.parent}    重置 {pair.reset}",
           anchor="ra", font=fonts["label"], fill=TEXT)
    stamp = "首秒读数 · 定格" if tick == TICKS else "0.5× 慢放"
    d.text((25, 63), f"4° 上坡    指令 0 m/s    {stamp}    实测 {tick*DT:.2f} s", fill=MUTED, font=fonts["body"])
    for i, (x, label, arm) in enumerate(zip((LEFT_X, RIGHT_X),
                                           ("原策略 · 原轮速目标", "同一策略 · 抑制四轮共同倒转"), pair.arms, strict=True)):
        d.text((x+4, 93), label, font=fonts["label"], fill=COLORS[i])
        frame.paste(picture(arm, tick), (x, SCENE_Y))
        d.text((x+2, SCENE_Y+PANE_H+9), "当前后溜", font=fonts["body"], fill=MUTED)
        d.text((x+119, SCENE_Y+PANE_H+3), f"{arm.rollback_mm[tick]:.1f} mm", font=fonts["number"], fill=COLORS[i])
    _chart(d, pair, tick, limit, fonts)
    d.line((941, 529, 941, 657), fill=GRID, width=2)
    d.text((964, 537), "两侧相差多少", fill=MUTED, font=fonts["body"])
    d.text((963, 574), f"{pair.effect_mm(tick):+.2f} mm", fill=TEXT, font=fonts["number"])
    d.text((964, 629), "正数表示右侧少退", fill=MUTED, font=fonts["small"])
    d.text((25, 688), "橙线固定在共同起点 x = 1.10 m，两侧使用同一固定相机。", fill=MUTED, font=fonts["small"])
    return frame


def _encode(output, pairs, report):
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
            "use software offscreen EGL playback")
    import mujoco
    from PIL import Image
    from tools87.terrain87 import ground_height, load_model

    fonts = {"title": chinese_font(35, True), "label": chinese_font(25, True), "body": chinese_font(23),
             "number": chinese_font(32, True), "small": chinese_font(17)}
    model = load_model("up")
    model.vis.global_.offwidth, model.vis.global_.offheight = PANE_W, PANE_H
    floor = model.geom("floor").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.71, .74, .75, 1)
    model.vis.headlight.ambient[:] = .48
    data, cam, options = mujoco.MjData(model), mujoco.MjvCamera(), mujoco.MjvOption()
    options.geomgroup[3] = 0
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 100., -13., 1.48
    cam.lookat[:] = (1.00, 0., float(ground_height("up", 1.0)) + .22)
    limit = max(50, math.ceil(max(float(np.max(a.rollback_mm)) for pair in pairs for a in pair.arms)/50)*50)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-filter_threads", "1",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-maxrate", "900k", "-bufsize", "1000k",
           "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(output)]
    with mujoco.Renderer(model, height=PANE_H, width=PANE_W) as renderer:
        def picture(arm, tick):
            data.qpos[:], data.qvel[:], data.time = arm.q[tick], arm.v[tick], tick*DT
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam, scene_option=options)
            add_marks(mujoco, renderer.scene, "up", 1.10, ground_height, .45, 1.80, 0.)
            report["rendered_poses"] += 1
            return Image.fromarray(renderer.render())

        with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as encoder:
            def emit(frame, repeats=1):
                pixels = frame.tobytes()
                for _ in range(repeats):
                    encoder.stdin.write(pixels)
                    report["frames"] += 1
            try:
                emit(_intro(fonts), INTRO_FRAMES)
                for i, pair in enumerate(pairs, 1):
                    for tick in range(TICKS):
                        emit(_pair_frame(pair, i, tick, limit, picture, fonts))
                    emit(_pair_frame(pair, i, TICKS, limit, picture, fonts), ENDPOINT_FRAMES)
                emit(_summary(pairs, fonts), SUMMARY_FRAMES)
            finally:
                encoder.stdin.close()
            require(encoder.wait() == 0, "video encoder did not finish")


def run(out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    output = out / "stand93.mp4"
    report = dict(status="PARTIAL", protocol_sha256=PLAN_SHA, file=output.name, physical_cases=0, frames=0,
                  fps=FPS, playback_speed=.5, source_dt_s=DT, frame_counts=frame_counts(), rendered_poses=0,
                  new_physics_steps=0, policy_calls=0, model_steps=0, training_updates=0, cells=[])
    try:
        plan = checked_plan()
        scoring = BASE / "score_01/report.json"
        new_folder = BASE / "collect_01"
        pairs = load_pairs(plan, read(scoring), OLD, new_folder)
        report.update(score_sha256=digest(scoring), physical_seal_sha256=digest(new_folder / "seal.json"),
                      diagonal_report_sha256=digest(OLD / "report.json"), physical_cases=2*len(pairs))
        for i, pair in enumerate(pairs):
            frame0 = INTRO_FRAMES + i*(TICKS+ENDPOINT_FRAMES)
            report["cells"].append(dict(parent_seed=pair.parent, reset_seed=pair.reset,
                                        clip_first_frame=frame0, endpoint_first_frame=frame0+TICKS,
                                        clip_last_frame=frame0+TICKS+ENDPOINT_FRAMES-1, true_effect_mm=pair.effect_mm(TICKS),
                                        rollback_mm={k: float(a.rollback_mm[-1]) for k, a in zip(ARMS, pair.arms, strict=True)},
                                        source_sha256={k: a.sha for k, a in zip(ARMS, pair.arms, strict=True)}))
        _encode(output, pairs, report)
        probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
                                "-of", "json", str(output)], capture_output=True, text=True, check=True, timeout=100)
        stream, = json.loads(probe.stdout)["streams"]
        count = report["frame_counts"]["total"]
        require(stream["codec_name"] == "h264" and stream["r_frame_rate"] == "25/1"
                and (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
                and int(stream["nb_read_frames"]) == report["frames"] == count
                and abs(float(stream["duration"]) - count/FPS) <= 1e-4 and report["rendered_poses"] == 18*(TICKS+1)
                and 0 < output.stat().st_size <= 5_000_000, "encoded timeline, pose count, codec or size differs")
        require(all(digest(a.path) == a.sha for pair in pairs for a in pair.arms), "physical source changed during display")
        report.update(status="COMPLETE", bytes=output.stat().st_size, sha256=digest(output), stream=stream)
    except BaseException as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(out / "video.json", report)
    return report
