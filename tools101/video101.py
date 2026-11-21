"""Render the two frozen perception readouts on the same delivered sensor packets."""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import chinese_font
from tools94.common94 import digest, read
from tools100.score100 import grade
from tools101.common101 import PLAN_SHA
from tools101.score101 import LAYOUTS, VISIBILITIES, qualifying, read_inputs, require, score_archives


WIDTH, HEIGHT, FPS = 1120, 660, 12
FRAMES_PER_PACKET, END_HOLD, REGRESSION_HOLD, SUMMARY_HOLD = 2, 4, 36, 54
BG, SURFACE, INK, MUTED, GRID = "#11212A", "#1C303A", "#EEF4F4", "#B4C7CA", "#36525E"
TRUTH, BASE, PRED, BAD = "#FFF3DA", "#8BBEFF", "#79DEBA", "#FFAB91"


@dataclass
class Clip:
    model: int
    rows: list[int]
    regression: int | None = None


def ordered_sequence(rows, episode, layout, visibility):
    selected = np.flatnonzero((rows["episode"] == episode) & (rows["layout"] == layout)
                              & (rows["visibility"] == visibility))
    return selected[np.argsort(rows["delivery_tick"][selected])].tolist()


def clips_for(rows, seeds, baseline, predictive):
    first_episode = int(np.min(rows["episode"]))
    clips = [Clip(model, ordered_sequence(rows, first_episode, layout, vis))
             for model in range(len(seeds)) for layout in LAYOUTS for vis in ("masked", "clear")]
    truth = rows["target_mm"]
    # Use registration order to select the first regression. Keeping this deterministic
    # allows a reader to locate the exact packet even if the arrays have been re-ordered.
    ids = [i for episode in sorted(set(rows["episode"].tolist())) for layout in LAYOUTS for vis in VISIBILITIES
           for i in ordered_sequence(rows, episode, layout, vis)]
    for model, (b, p) in enumerate(zip(baseline, predictive, strict=True)):
        regressed = qualifying(truth, b) & ~qualifying(truth, p)
        first = next((i for i in ids if regressed[i]), None)
        if first is not None:
            seq = ordered_sequence(rows, rows["episode"][first], rows["layout"][first], rows["visibility"][first])
            clips.append(Clip(model, seq, seq.index(first)))
            break
    return clips


def side(delta):
    if not np.isfinite(delta):
        return "无估计"
    if delta > 10:
        return "左边较高"
    if delta < -10:
        return "右边较高"
    return "接近齐平"


def value(delta, places=1):
    if not np.isfinite(delta):
        return "无估计"
    if abs(delta) < .5 * 10**(-places):
        return f"{0:.{places}f} mm"
    return f"{delta:+.{places}f} mm"


def sensor_views(image, draw, depth, valid, fonts):
    from PIL import Image

    _, _, small, tiny = fonts
    draw.rounded_rectangle((22, 82, 478, 205), radius=12, fill=SURFACE)
    draw.text((35, 89), "两组使用的同一包测距", font=small, fill=INK)
    eyes = depth.reshape(2, 20, 13)
    eye_valid = valid.reshape(eyes.shape)
    lo = np.array([246, 200, 123])
    hi = np.array([73, 121, 172])
    for k, x in enumerate((37, 125)):
        fraction = np.clip((eyes[k] - .14) / (1.45 - .14), 0., 1.)[..., None]
        rgb = np.rint(lo + fraction * (hi-lo)).astype(np.uint8)
        rgb[~eye_valid[k]] = (42, 53, 58)
        bitmap = Image.fromarray(rgb).resize((65, 68), resample=Image.Resampling.NEAREST)
        image.paste(bitmap, (x, 123))
        draw.text((x+71, 144), "左" if k == 0 else "右", font=tiny, fill=MUTED)
    draw.text((240, 122), f"有效回波　{int(valid.sum())} / 520", font=small, fill=INK)
    draw.text((240, 158), "暖色近，冷色远，深灰无回波", font=tiny, fill=MUTED)


def arm_panel(draw, x, label, color, estimate, truth, fonts):
    title, body, small, tiny = fonts
    ok = grade(float(truth), float(estimate) if np.isfinite(estimate) else None)
    draw.rounded_rectangle((x, 222, x+526, 363), radius=12, fill=SURFACE)
    draw.text((x+17, 229), label, font=small, fill=color)
    draw.text((x+17, 264), value(estimate), font=title, fill=INK)
    draw.text((x+288, 269), side(estimate), font=body, fill=INK)
    err = f"绝对误差 {abs(float(estimate)-float(truth)):.1f} mm" if np.isfinite(estimate) else "无输出计为失败"
    draw.text((x+18, 325), err, font=tiny, fill=MUTED)
    draw.text((x+507, 318), "本包合格" if ok else "本包失败", anchor="rt", font=small, fill=PRED if ok else BAD)


def trace_plot(draw, rows, clip, position, baseline, predictive, bound, fonts):
    _, _, small, tiny = fonts
    draw.rounded_rectangle((22, 380, WIDTH-22, 586), radius=12, fill=SURFACE)
    draw.text((37, 388), "连续新包上的高差", font=small, fill=INK)
    for x, text, color in ((569, "地形真值", TRUTH), (735, "只学高差", BASE), (905, "多学下一包", PRED)):
        draw.line((x, 407, x+22, 407), fill=color, width=3)
        draw.text((x+30, 391), text, font=small, fill=color)
    left, right, top, bottom = 103, 1065, 427, 542
    t = rows["delivery_tick"][clip.rows] / 50.
    xx = lambda v: left + (right-left)*(v - t[0])/(t[-1]-t[0])
    yy = lambda v: (bottom+top)/2 - v*(bottom-top)/(2*bound)
    for y in (-30, 0, 30):
        draw.line((left, yy(y), right, yy(y)), fill=GRID, width=1)
        draw.text((left-9, yy(y)), f"{y:+d}" if y else "0", anchor="rm", font=tiny, fill=MUTED)
    draw.text((40, 424), "mm", font=tiny, fill=MUTED)
    for at in (0, len(t)//2, len(t)-1):
        x = xx(t[at])
        draw.text((x, bottom+3), f"{t[at]:.2f} s", anchor="mt", font=tiny, fill=MUTED)
    draw.line((xx(t[position]), top, xx(t[position]), bottom), fill="#799198", width=1)
    for series, color, radius in ((rows["target_mm"], TRUTH, 5), (baseline[clip.model], BASE, 4),
                                  (predictive[clip.model], PRED, 2)):
        previous = None
        for j, row in enumerate(clip.rows[:position+1]):
            if not np.isfinite(series[row]):
                previous = None
                continue
            x, y = xx(t[j]), yy(float(series[row]))
            if previous is not None:
                draw.line((previous, (x, y)), fill=color, width=2)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
            previous = x, y


def packet_frame(rows, depth, valid, seeds, baseline, predictive, score, clip, position, bound, fonts):
    from PIL import Image, ImageDraw

    row = clip.rows[position]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title, body, small, tiny = fonts
    bad_example = clip.regression is not None
    heading = "首次从合格变为失败的包" if bad_example else "同一包点云，能看准前方哪边更高吗？"
    draw.text((22, 4), heading, font=body, fill=BAD if bad_example else INK)
    v = "视野遮挡" if rows["visibility"][row] == "masked" else "视野清晰"
    layout = "左轮迹隆起" if rows["side"][row] == "left" else "右轮迹隆起"
    draw.text((24, 49), f"{layout}　·　{v}　·　原生回合 {rows['episode'][row]}　·　拟合种子 {seeds[clip.model]}", font=small, fill=MUTED)
    draw.text((WIDTH-25, 52), f"第 {position+1}/{len(clip.rows)} 包　送达 {rows['delivery_tick'][row]/50:.2f} s",
              font=tiny, fill=MUTED, anchor="rt")
    sensor_views(image, draw, depth[row], valid[row], fonts)
    draw.rounded_rectangle((491, 82, WIDTH-22, 205), radius=12, fill=SURFACE)
    truth = float(rows["target_mm"][row])
    draw.text((509, 90), "前轮前方 35 cm　·　地形真值", font=small, fill=TRUTH)
    draw.text((508, 129), value(truth, 0), font=title, fill=TRUTH)
    draw.text((795, 137), side(truth), font=body, fill=TRUTH)
    arm_panel(draw, 22, "只学本包高差　λ = 0", BASE, baseline[clip.model, row], truth, fonts)
    arm_panel(draw, 572, "同时学习预测下一包特征　λ = 1", PRED, predictive[clip.model, row], truth, fonts)
    trace_plot(draw, rows, clip, position, baseline, predictive, bound, fonts)
    draw.text((25, 592), "高差取左减右。±10 mm 内判为齐平，有高侧时还须误差不超过 10 mm。", font=small, fill=INK)
    if bad_example:
        note = f"全部三种子共 {score['overall']['regressed']} 次从合格变为失败。此帧按登记顺序取第一次。"
    else:
        note = "两组都看相同的点云、本体和过去动作。视频回放接触前的传感器记录。"
    draw.text((25, 626), note, font=tiny, fill=MUTED)
    return image


def metric(value):
    return "—" if value is None else f"{value:.2f}"


def summary_frame(score, fonts):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    _, body, small, tiny = fonts
    draw.text((25, 19), "加入下一包预测后，三个种子各自表现如何？", font=body, fill=INK)
    draw.text((27, 75), f"{score['native_episodes']} 个留出原生回合，左右镜像，清晰与遮挡各评分一次。每个模型检验 {score['heldout_arrivals']} 包。",
              font=small, fill=MUTED)
    draw.text((28, 128), "每格按“只学高差 → 多学下一包”排列。误差越小越好，合格包数越多越好。", font=small, fill=INK)
    xs = (38, 183, 405, 663, 884)
    headings = ("拟合种子", "遮挡误差 / mm", "遮挡合格包", "清晰误差 / mm", "清晰合格包")
    for x, label in zip(xs, headings, strict=True):
        draw.text((x, 194), label, font=small, fill=MUTED)
    for ordinal, entry in enumerate(score["by_seed"]):
        y = 242 + 73*ordinal
        draw.rounded_rectangle((22, y, WIDTH-22, y+56), radius=8, fill=SURFACE)
        cells = [str(entry["seed"])]
        for vis in ("masked", "clear"):
            part = entry[vis]
            cells += [f"{metric(part['baseline_mae_both_mm'])} → {metric(part['predictive_mae_both_mm'])}",
                      f"{part['baseline']['passed']} → {part['predictive']['passed']} / {part['baseline']['packets']}"]
        for x, word in zip(xs, cells, strict=True):
            draw.text((x, y+9), word, font=small, fill=INK)
    draw.text((27, 489), f"达到全部登记条件　{score['passing_seeds']} / {len(score['fit_seeds'])} 个拟合种子",
              font=body, fill=PRED if score["passed"] else BAD)
    draw.text((29, 552), "要求每个种子遮挡误差下降、清晰误差不增加，两种条件的合格数及可估计包数均不减少。",
              font=small, fill=INK)
    draw.text((29, 600), "本轮只更新感知网络。高差估计结果能否帮助机器人越障，还要用运控闭环验证。", font=tiny, fill=MUTED)
    return image


def render(holdout_npz, prediction_path, report_path, output_folder):
    holdout, predictions, report, out = map(Path, (holdout_npz, prediction_path, report_path, output_folder))
    archive, movie = out / "video.json", out / "jeplo101.mp4"
    require(not archive.exists() and not movie.exists(), "preserving existing movie or receipt")
    rec = dict(status="PARTIAL", fps=FPS, cover_frames=0, frames=0, new_rays=0, physics_substeps=0,
               perception_updates=0, world_updates=0, policy_updates=0, clips=[], protocol_sha256=PLAN_SHA)
    with archive.open("x", encoding="utf-8") as journal:
        try:
            inputs = {"holdout_sha256": holdout, "predictions_sha256": predictions, "report_sha256": report}
            rec.update({k: digest(v) for k, v in inputs.items()})
            report_data = read(report)
            require(report_data["status"] == "COMPLETE" and report_data["protocol_sha256"] == PLAN_SHA
                    and all(report_data[k] == rec[k] for k in ("holdout_sha256", "predictions_sha256"))
                    and report_data["score"] == score_archives(holdout, predictions),
                    "video score or registration differs from sealed measurements")
            score = report_data["score"]
            rows, seeds, baseline, predictive = read_inputs(holdout, predictions)
            with np.load(holdout, allow_pickle=False) as data:
                depth, valid = data["depth_m"], data["valid"]
            clips = clips_for(rows, seeds, baseline, predictive)
            finite_estimates = np.concatenate((baseline[np.isfinite(baseline)], predictive[np.isfinite(predictive)]))
            bound = math.ceil(max(35., float(np.max(np.abs(finite_estimates), initial=0)))/10.)*10.
            fonts = chinese_font(39, True), chinese_font(29, True), chinese_font(21), chinese_font(16)
            args = ["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-threads", "1", "-f", "rawvideo",
                    "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
                    "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-maxrate", "650k", "-bufsize", "650k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)]
            with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
                def emit(frame, times):
                    pixels = frame.tobytes()
                    for _ in range(times):
                        encoder.stdin.write(pixels)
                        rec["frames"] += 1
                try:
                    for clip in clips:
                        indices = range(len(clip.rows)) if clip.regression is None else (clip.regression,)
                        shot = {"seed": seeds[clip.model], "first_frame": rec["frames"],
                                "first_regression": clip.regression is not None, "row_ids": [clip.rows[i] for i in indices]}
                        rec["clips"].append(shot)
                        for i in indices:
                            image = packet_frame(rows, depth, valid, seeds, baseline, predictive, score, clip, i, bound, fonts)
                            times = REGRESSION_HOLD if clip.regression is not None else FRAMES_PER_PACKET + END_HOLD*(i == len(clip.rows)-1)
                            emit(image, times)
                        shot["last_frame"] = rec["frames"]-1
                    rec["summary_first_frame"] = rec["frames"]
                    emit(summary_frame(score, fonts), SUMMARY_HOLD)
                finally:
                    encoder.stdin.close()
                require(encoder.wait() == 0, "video encoder failed")
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,codec_name,r_frame_rate,nb_read_frames,duration",
                                    "-of", "json", str(movie)], text=True, capture_output=True, check=True, timeout=90)
            track, = json.loads(probe.stdout)["streams"]
            expected = SUMMARY_HOLD + sum((len(c.rows)*FRAMES_PER_PACKET + END_HOLD) if c.regression is None
                                          else REGRESSION_HOLD for c in clips)
            require(rec["frames"] == expected == int(track["nb_read_frames"]) and track["codec_name"] == "h264"
                    and (track["width"], track["height"], track["r_frame_rate"]) == (WIDTH, HEIGHT, f"{FPS}/1")
                    and abs(float(track["duration"])-expected/FPS) <= 1e-5 and 0 < movie.stat().st_size <= 5_000_000,
                    "decoded movie differs from expected stream or exceeds 5 MB")
            require(all(rec[k] == digest(path) for k, path in inputs.items()), "input changed while rendering")
            rec.update(status="COMPLETE", video_sha256=digest(movie), bytes=movie.stat().st_size, stream=track,
                       duration_s=expected/FPS)
        except BaseException as exc:
            rec["reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(rec, journal, indent=2, ensure_ascii=False, allow_nan=False)
            journal.write("\n")
    return rec


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("holdout", "predictions", "report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    render(args.holdout, args.predictions, args.report, args.output)
