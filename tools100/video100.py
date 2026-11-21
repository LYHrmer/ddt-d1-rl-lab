"""Show the two registered sensors looking at the same wheel tracks from recorded returns."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import chinese_font, digest
from tools100.score100 import RADIUS_M, TIE_ATOL_M, estimate_delta_mm, grade, score_rows


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments100/pointcloud_density.json"
WIDTH, HEIGHT, FPS = 1280, 720, 12
PACKET_FRAMES, END_HOLD = 2, 8
LAYOUTS = ("left", "right", "flat")
VISIBILITIES = ("masked", "clear")
LAYOUT_LABEL = {"left": "左轮迹隆起", "right": "右轮迹隆起", "flat": "平地"}
VISIBILITY_LABEL = {"masked": "部分视野遮挡", "clear": "视野清晰"}
BG, CARD, GRID = "#111D26", "#192C38", "#34505D"
INK, MUTED, TARGET = "#EEF5F3", "#AEC2CA", "#F4E3B0"
GOOD, BAD = "#8AE2B9", "#FFAF9B"
LOW_RGB, HIGH_RGB = np.array((92, 156, 192)), np.array((255, 171, 73))


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Packet:
    capture: int
    delivery: int
    query: np.ndarray
    returns: tuple[np.ndarray, np.ndarray]
    truth: float
    estimates: tuple[float | None, float | None]

    def summary(self):
        return dict(capture_tick=self.capture, delivery_tick=self.delivery, truth_mm=self.truth,
                    sparse_mm=self.estimates[0], dense_mm=self.estimates[1],
                    passed=[bool(grade(self.truth, v)) for v in self.estimates],
                    points_shown=[len(points) for points in self.returns])


@dataclass
class Case:
    seed: int
    layout: str
    visibility: str
    packets: list[Packet]


def load_cases(archive, report, protocol=PROTOCOL):
    plan = json.loads(protocol.read_bytes())
    source = json.loads(report.read_bytes())
    require(source["protocol_sha256"] == digest(protocol), "video protocol checksum differs from the report")
    require(source["cloud_sha256"] == digest(archive), "video cloud checksum differs from the report")
    require(source.get("status") == "COMPLETE", "incomplete input report")
    seeds, ticks = plan["seeds"], plan["capture_ticks"]
    budget = plan["budget"]
    require(set(plan["layouts"]) == set(LAYOUTS) and set(plan["visibilities"]) == set(VISIBILITIES)
            and plan["policy_hz"] == 50 and plan["sensor_hz"] == 6 and plan["delay_ticks"] == 2
            and plan["geometry"]["lookahead_m"] == .35 and plan["estimator"]["radius_m"] == .15
            and plan["score"]["tolerance_mm"] == 10, "video labels differ from the registered experiment")
    with np.load(archive, allow_pickle=False) as data:
        xyz, query = data["points"], data["query_xy"]
        seed, layout, tick = (data[k] for k in ("cloud_seed", "cloud_layout", "capture_tick"))
        sparse, keep = data["sparse_indices"], data["angle_keep"]
    n, directions = budget["independent_clouds"], budget["directions_per_cloud"]
    require(xyz.shape == (n, directions, 3) and xyz.dtype == np.float64 and query.shape == (n, 2, 2)
            and np.isfinite(query).all() and (query[:, 0, 1] > query[:, 1, 1]).all()
            and seed.shape == layout.shape == tick.shape == (n,), "invalid recorded clouds or wheel queries")
    require((np.isfinite(xyz).all(axis=-1) | np.isnan(xyz).all(axis=-1)).all(), "invalid point validity encoding")
    require(sparse.shape == (8,) and np.issubdtype(sparse.dtype, np.integer) and len(np.unique(sparse)) == 8
            and (sparse >= 0).all() and (sparse < directions).all()
            and keep.shape == (directions,) and keep.dtype == np.bool_, "invalid ray membership")
    lookup = {(int(s), str(l), int(t)): i for i, (s, l, t) in enumerate(zip(seed, layout, tick, strict=True))}
    keys = {(s, l, t) for s in seeds for l in LAYOUTS for t in ticks}
    require(len(lookup) == n == len(keys) and set(lookup) == keys, "missing, duplicate or unregistered clouds")
    rows = {(r["seed"], r["layout"], r["visibility"], r["capture_tick"]): r for r in source["rows"]}
    expect = {(s, l, v, t) for s, l, t in keys for v in VISIBILITIES}
    require(len(source["rows"]) == len(rows) == budget["paired_rows"] and set(rows) == expect,
            "missing, duplicate or unregistered score rows")
    require(source["score"] == score_rows(source["rows"], plan), "published verdict differs from the packet rows")
    cases = []
    for s in seeds:
        for l in LAYOUTS:
            for v in VISIBILITIES:
                packets = []
                for t in ticks:
                    row, i = rows[s, l, v, t], lookup[s, l, t]
                    truth = row["truth_mm"]
                    require(np.isfinite(truth) and row["delivery_tick"] == t + plan["delay_ticks"]
                            and truth == rows[s, l, "clear", t]["truth_mm"], "inconsistent packet truth or delay")
                    valid = np.isfinite(xyz[i]).all(axis=1)
                    if v == "masked":
                        valid &= keep
                    returns = (xyz[i, sparse[valid[sparse]]], xyz[i, valid])
                    estimates = (row["sparse_mm"], row["dense_mm"])
                    for points, estimate in zip(returns, estimates, strict=True):
                        actual = estimate_delta_mm(points, query[i])
                        require((actual is None and estimate is None) or
                                (actual is not None and estimate is not None and np.isfinite(estimate)
                                 and abs(actual - estimate) <= 1e-8), "displayed estimate differs from recorded returns")
                    packets.append(Packet(t, row["delivery_tick"], query[i], returns, truth, estimates))
                cases.append(Case(s, l, v, packets))
    return cases


def wheel_origin(query):
    return query.mean(axis=0) - np.array([.35, 0.])


def common_extent(cases):
    # Reference every packet to its front axle. All cases then share the same metric axes.
    near, far, half_width = -.12, .65, .45
    for case in cases:
        for p in case.packets:
            xy = np.concatenate((p.returns[1][:, :2], p.query), axis=0) - wheel_origin(p.query)
            near = min(near, float(xy[:, 0].min()) - .025)
            far = max(far, float(xy[:, 0].max()) + .025)
            half_width = max(half_width, float(np.abs(xy[:, 1]).max()) + .035)
    return math.floor(near * 5) / 5, math.ceil(far * 5) / 5, math.ceil(half_width * 5) / 5


def side_text(delta):
    if delta is None:
        return "无估计"
    if abs(delta) <= 10:
        return f"近似齐平  {delta:+.1f} mm" if abs(delta) >= .05 else "齐平  0.0 mm"
    return f"{'左' if delta > 0 else '右'}高  {abs(delta):.1f} mm"


def nearest_mask(points, query):
    if not len(points):
        return np.zeros(0, dtype=bool)
    offset = points[:, :2] - query
    distances = np.hypot(offset[:, 0], offset[:, 1])
    nearest = distances.min()
    return (distances <= RADIUS_M) & (distances - nearest <= TIE_ATOL_M)


def panel(draw, edge, arm, packet, extent, counts, length, fonts):
    title, body, small = fonts
    draw.rounded_rectangle((edge, 108, edge + 605, 612), radius=12, fill=CARD)
    label = "八束测距" if arm == 0 else "稠密点云"
    points = packet.returns[arm]
    draw.text((edge + 20, 119), label, font=title, fill=INK)
    draw.text((edge + 584, 133), f"有效回波 {len(points)}", font=small, fill=MUTED, anchor="rt")
    near, far, half = extent
    left, right, top, bottom = edge + 76, edge + 571, 173, 482
    scale = min((bottom-top)/(far-near), (right-left)/(2*half))
    origin = wheel_origin(packet.query)

    def pixel(x, y):
        return (left+right)/2 - y*scale, (top+bottom)/2 - (x-(near+far)/2)*scale

    for value in np.arange(round(5*near), round(5*far)+1) / 5:
        lo, hi = pixel(value, half), pixel(value, -half)
        draw.line((lo, hi), fill=GRID)
        draw.text((lo[0]-6, lo[1]), f"{value:.1f}", font=small, fill=MUTED, anchor="rm")
    for value in np.arange(-round(5*half), round(5*half)+1) / 5:
        draw.line((pixel(near, value), pixel(far, value)), fill=GRID)

    targets = packet.query - origin
    wheels = [pixel(x-.35, y) for x,y in targets]
    draw.line(wheels, fill=MUTED, width=2)
    for px, py in wheels:
        draw.rounded_rectangle((px-5, py-9, px+5, py+9), radius=3, fill=MUTED)

    # Both arms use the same measured vertical reference, including when sparse has no return.
    if len(points):
        min_z = packet.returns[1][:, 2].min()
        fraction = np.clip((points[:, 2] - min_z) / .03, 0, 1)
        colors = np.rint(LOW_RGB + fraction[:, None] * (HIGH_RGB-LOW_RGB)).astype(int)
        radius = 4.8 if arm == 0 else 1.9
        for p, color in zip(points, colors, strict=True):
            px, py = pixel(*(p[:2] - origin))
            draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=tuple(map(int, color)))

    for side, (q, local) in enumerate(zip(packet.query, targets, strict=True)):
        cx, cy = pixel(*local)
        radius_px = .15 * scale
        draw.ellipse((cx-radius_px, cy-radius_px, cx+radius_px, cy+radius_px), outline="#839797", width=1)
        for p in points[nearest_mask(points, q)]:
            px, py = pixel(*(p[:2] - origin))
            draw.line((cx, cy, px, py), fill=TARGET, width=2)
            draw.ellipse((px-4,py-4,px+4,py+4), outline=TARGET, width=2)
        draw.line((cx-7,cy,cx+7,cy), fill=TARGET, width=2)
        draw.line((cx,cy-7,cx,cy+7), fill=TARGET, width=2)
        draw.text((cx, cy-radius_px-2), "左" if side == 0 else "右", font=small, fill=TARGET, anchor="mb",
                  stroke_width=2, stroke_fill=CARD)

    draw.line((edge+16, 501, edge+589, 501), fill=GRID)
    delta = packet.estimates[arm]
    ok = grade(packet.truth, delta)
    draw.text((edge+20, 513), "估计　" + side_text(delta), font=body, fill=INK)
    draw.text((edge+582, 513), "合格" if ok else "失败", font=body, fill=GOOD if ok else BAD, anchor="rt")
    error = "查询区缺少回波" if delta is None else f"绝对误差  {abs(delta-packet.truth):.1f} mm"
    draw.text((edge+21, 561), error, font=small, fill=MUTED)
    draw.text((edge+582, 561), f"本段合格 {counts[arm]} / {length} 包", font=small, fill=MUTED, anchor="rt")


def frame(case, index, shot, total, extent, counts, fonts):
    from PIL import Image, ImageDraw

    p = case.packets[index]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    title, body, small = fonts
    draw.text((23, 9), "D1 · 前轮前方哪边地面更高？", font=title, fill=INK)
    draw.text((503, 13), "此刻真值　" + side_text(p.truth), font=body, fill=TARGET)
    draw.text((WIDTH-23, 22), f"工况 {shot}/{total}　复位 {case.seed}", font=small, fill=MUTED, anchor="rt")
    draw.text((24, 64), f"{LAYOUT_LABEL[case.layout]}　·　{VISIBILITY_LABEL[case.visibility]}", font=body, fill=INK)
    draw.text((WIDTH-23, 68), f"第 {index+1}/{len(case.packets)} 包　采集 {p.capture/50:.2f} s → 送达 {p.delivery/50:.2f} s",
              font=small, fill=MUTED, anchor="rt")
    for arm, edge in enumerate((24, 651)):
        panel(draw, edge, arm, p, extent, counts, len(case.packets), fonts)
    draw.text((24, 617), "俯视，车头朝上。十字在前轴前 35 cm，圆圈半径 15 cm。连线标出估计使用的最近回波。", font=small, fill=INK)
    draw.text((24, 649), "纵轴为距前轴的米数。点色由蓝至橙表示比本包最低回波高 0～30 mm。采集坐标保持 40 ms 后送达。",
              font=small, fill=MUTED)
    draw.text((24, 681), "高差取左减右，±10 mm 内判为齐平。其余须判对高侧且误差不超过 10 mm。无估计计失败。", font=small, fill=INK)
    return image


def encode(output, cases, journal):
    fonts = chinese_font(30, True), chinese_font(23, True), chinese_font(17)
    extent = common_extent(cases)
    journal["shared_extent_m"] = extent
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-threads", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
           "-an", "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
           "-maxrate", "550k", "-bufsize", "550k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        try:
            for shot, case in enumerate(cases, 1):
                rows = [p.summary() for p in case.packets]
                counts = [sum(r["passed"][a] for r in rows) for a in (0, 1)]
                published = dict(seed=case.seed, layout=case.layout, visibility=case.visibility, passed_counts=counts,
                                 first_frame=journal["frames"], packets=rows)
                journal["cases"].append(published)
                for k in range(len(rows)):
                    pixels = frame(case, k, shot, len(cases), extent, counts, fonts).tobytes()
                    times = PACKET_FRAMES + (END_HOLD if k == len(rows)-1 else 0)
                    for _ in range(times):
                        encoder.stdin.write(pixels)
                        journal["frames"] += 1
                published["last_frame"] = journal["frames"]-1
        finally:
            encoder.stdin.close()
        require(encoder.wait() == 0, "point-cloud movie encoder failed")


def render(archive, report, output, receipt, protocol=PROTOCOL):
    paths = (archive, report, protocol, output, receipt)
    if len({p.resolve() for p in paths}) != len(paths) or output.exists() or receipt.exists():
        raise FileExistsError("video and receipt require separate unused outputs")
    journal = dict(status="PARTIAL", file=output.name, frames=0, fps=FPS, frames_per_packet=PACKET_FRAMES,
                   endpoint_hold_frames=END_HOLD, cover_frames=0, cases=[], new_rays=0, new_physics_steps=0,
                   policy_updates=0, world_updates=0, perception_updates=0)
    with receipt.open("x", encoding="utf-8") as stream:
        try:
            hashes = [digest(p) for p in (archive, report, protocol)]
            journal.update(cloud_sha256=hashes[0], report_sha256=hashes[1], protocol_sha256=hashes[2])
            cases = load_cases(archive, report, protocol)
            encode(output, cases, journal)
            probe = subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,nb_read_frames,duration,codec_name,r_frame_rate",
                                    "-of", "json", str(output)], check=True, capture_output=True, text=True, timeout=100)
            track, = json.loads(probe.stdout)["streams"]
            expected = sum(len(c.packets)*PACKET_FRAMES + END_HOLD for c in cases)
            require((track["width"], track["height"], track["codec_name"], track["r_frame_rate"]) ==
                    (WIDTH, HEIGHT, "h264", f"{FPS}/1") and int(track["nb_read_frames"]) == journal["frames"] == expected
                    and abs(float(track["duration"])-expected/FPS) < .0001 and 0 < output.stat().st_size <= 5_000_000,
                    "movie dimensions, frame count, duration or size differs")
            require(hashes == [digest(p) for p in (archive, report, protocol)], "sealed movie inputs changed")
            journal.update(status="COMPLETE", duration_s=expected/FPS, bytes=output.stat().st_size,
                           video_sha256=digest(output), stream=track)
        except BaseException as exc:
            journal["reason"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            json.dump(journal, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    return journal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "report", "output", "receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    result = render(**vars(parser.parse_args()))
    print(json.dumps({k: result[k] for k in ("status", "frames", "duration_s", "video_sha256")}))
