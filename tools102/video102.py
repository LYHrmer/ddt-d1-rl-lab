"""Slow D1 pose replay with registered, co-timed wheel-track occlusions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools87.video87 import chinese_font
from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read
from tools100.common100 import protocol as sensor_protocol
from tools101.common101 import PUBLIC


WIDTH, HEIGHT = 1152, 652
SCENE_W, SCENE_H = 764, 452
FPS, FRAMES_PER_TICK = 20, 4
SOURCE, EPISODE, SIDE = 89037, 0, "left"
HOLD_SECONDS = {0: 4, 6: 7, 7: 7, 8: 6, 15: 4}
BG, PANEL, INK, MUTED = "#112128", "#1A3038", "#F3F4EF", "#B4C9CC"
KEPT, HIDDEN, QUERY, GRID = "#82D7CB", "#FF8ABA", "#FFDA77", "#365059"


def clocks(deliveries):
    """Hold a delivered packet until its successor; preserve every original 50 Hz robot pose."""
    ticks = np.asarray(deliveries)
    require(ticks.shape == (16,) and ticks[0] == 27 and ticks[-1] == 152
            and np.all(np.diff(ticks) > 0) and list(ticks[6:9]) == [77, 86, 94],
            "unexpected clocks for the registered movie")
    for tick in range(int(ticks[0]), int(ticks[-1]) + 1):
        packet = int(np.searchsorted(ticks, tick, side="right") - 1)
        extra = FPS * HOLD_SECONDS.get(packet, 0) if tick == ticks[packet] else 0
        yield tick, packet, extra


@dataclass
class Packet:
    capture: int
    delivery: int
    cloud: int
    active: bool
    valid: np.ndarray
    old: np.ndarray
    moved: np.ndarray
    points_ego: np.ndarray
    points_world: np.ndarray
    query_ego: np.ndarray
    query_world: np.ndarray


def original_pose_stream():
    manifest = read(PUBLIC / "manifest.json")["files"]
    index = PUBLIC / "data.json"
    require(digest(index) == manifest[str(index.relative_to(ROOT))], "native episode listing changed")
    archive = read(index)
    source, = (s for s in archive["sources"] if s["source_seed"] == SOURCE)
    native = ROOT / source["path"]
    roster = archive["splits"]["holdout"]["episode_roster"]
    case, = (row for row in roster if (row["source_seed"], row["episode"]) == (SOURCE, EPISODE))
    require(digest(native) == source["trace_sha256"], "native training trajectory changed")
    with np.load(native, allow_pickle=False) as trace:
        start, stop = case["first_archive_row"], case["end_archive_row_exclusive"]
        poses = trace["qpos"][start:stop]
    pose_hash = hashlib.sha256(poses.tobytes()).hexdigest()
    require(poses.shape == (300, 23) and pose_hash == case["physical_episode_qpos_sha256"],
            "did not recover the selected complete native pose series")
    return poses, native, pose_hash


def load_packets(path, scene, poses):
    """Use original ray ranges at acquisition and original wheel queries at delivery."""
    holdout = PUBLIC / "holdout.npz"
    require(digest(holdout) == read(PUBLIC / "manifest.json")["files"][str(holdout.relative_to(ROOT))],
            "source clouds changed")
    with np.load(path, allow_pickle=False) as pulses, np.load(holdout, allow_pickle=False) as original:
        take = np.flatnonzero((pulses["episode"] == EPISODE) & (pulses["layout"] == "unseen_left")
                              & (pulses["mask_side"] == SIDE))
        take = take[np.argsort(pulses["capture_tick"][take])]
        require(len(take) == 16, "missing movie packet series")
        src = pulses["clear_row"][take]
        depths = original["geometry_depth_m64"][src]
        valid = original["valid"][src]
        body_to_level = original["capture_body_to_level"][src]
        query, world = original["query_xy"][src], original["world_query_xy"][src]
        for name in ("cloud_id", "episode", "layout", "capture_tick", "delivery_tick"):
            require(np.array_equal(pulses[name][take], original[name][src]), f"misaligned movie {name}")
        require(np.all(original["visibility"][src] == "clear") and np.all(original["source_seed"][src] == SOURCE)
                and np.array_equal(pulses["query_xy"][take], query), "cloud or wheel query mismatch")
        ids, captures, deliveries = (pulses[k][take] for k in ("cloud_id", "capture_tick", "delivery_tick"))
        old, moved, active = (pulses[k][take] for k in ("original_keep", "moved_keep", "active"))
    require(old.shape == moved.shape == valid.shape == (16, 520) and old.dtype == moved.dtype == valid.dtype == bool
            and np.array_equal(active, np.arange(16) % 4 >= 2) and np.all(old.sum(axis=1) == 484)
            and np.all(moved.sum(axis=1) == 484) and np.array_equal(old[~active], moved[~active])
            and np.array_equal(deliveries, captures + 2), "incorrect compared masks or pulses")
    directions = np.tile(scene.directions, (2, 1))
    mounts = np.repeat(scene.mounts, len(scene.directions), axis=0)
    rows = []
    for j, (cap, delivery) in enumerate(zip(captures, deliveries, strict=True)):
        scene.pose(poses[cap - 1])
        rotation = scene.data.xmat[scene.base].reshape(3, 3)
        local = mounts + directions * depths[j, :, None]
        ego = local @ body_to_level[j]
        points = local @ rotation.T + scene.data.xpos[scene.base]
        require(np.allclose(rotation.T @ scene.heading, body_to_level[j], atol=1e-10, rtol=0),
                "saved ranges and native acquisition pose disagree")
        delivered = scene.pose(poses[delivery - 1])
        require(np.allclose(delivered["world_query_xy"], world[j], atol=1e-9, rtol=0)
                and np.allclose(delivered["query_xy"], query[j], atol=1e-9, rtol=0),
                "native pose does not reproduce this packet's wheel queries")
        rows.append(Packet(int(cap), int(delivery), int(ids[j]), bool(active[j]), valid[j],
                           old[j], moved[j], ego, points, query[j], world[j]))
    return rows


def _add_line(mj, scene, a, b, radius, rgba):
    require(scene.ngeom < scene.maxgeom, "render-only line capacity exceeded")
    kind = mj.mjtGeom.mjGEOM_CAPSULE
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    mj.mjv_connector(geom, kind, radius, np.asarray(a), np.asarray(b))
    geom.category = int(mj.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def overlay_3d(sensor, mj_scene, packet):
    """Yellow wheel-track ring and actual hidden returns, strictly render-only geometry."""
    mj, geometric = sensor.mj, sensor.geometry
    center = np.array([geometric["box_center_x_m"], geometric["box_abs_y_m"]])
    half = np.array([geometric["box_half_x_m"], geometric["box_half_y_m"]])

    def level(xy):
        return .004 + geometric["box_height_m"] * bool(np.all(np.abs(xy - center) <= half))

    # A render-only spoke follows the archived wheel orientation on each outside face.
    for wheel in sensor.wheels:
        rotation = sensor.data.geom_xmat[wheel].reshape(3, 3)
        radius, half_width = sensor.model.geom_size[wheel, :2]
        hub = sensor.data.geom_xpos[wheel]
        axis = rotation[:, 2]
        side = 1. if np.dot(axis, hub-sensor.data.xpos[sensor.base]) >= 0. else -1.
        face = hub + side * (half_width + .003) * axis
        _add_line(mj, mj_scene, face, face + radius * .76 * rotation[:, 0], .0045, (.85, .84, .73, 1.))
    for wheel, query in enumerate(packet.query_world):
        rgba = (.97, .73, .13, 1.) if wheel == 0 else (.4, .49, .5, 1.)
        z = level(query) + .006
        circle = query + .036 * np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 17)])
        for a, b in zip(circle[:-1], circle[1:], strict=True):
            _add_line(mj, mj_scene, [*a, z], [*b, z], .004 if wheel == 0 else .002, rgba)
        wheel_center = sensor.data.geom_xpos[sensor.wheels[wheel], :2]
        _add_line(mj, mj_scene, [*wheel_center, .008], [*query, z], .002, rgba)
    for point in packet.points_world[packet.valid & ~packet.moved]:
        require(mj_scene.ngeom < mj_scene.maxgeom, "render-only return capacity exceeded")
        geom = mj_scene.geoms[mj_scene.ngeom]
        position = point + [0., 0., .005]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_SPHERE, np.full(3, .005), position,
                        np.eye(3).ravel(), np.asarray((.94, .12, .43, 1.), np.float32))
        geom.category = int(mj.mjtCatBit.mjCAT_DECOR)
        mj_scene.ngeom += 1


def point_map(draw, packet, keep, top, title, fonts):
    """Two equal-scale magnifications centered on the selected left wheel's query."""
    small, tiny = fonts[2:]
    left, right, bottom = 790, WIDTH-16, top+222
    draw.rounded_rectangle((left, top, right, bottom), 9, fill=PANEL)
    draw.text((left+12, top+7), title, font=small, fill=INK)
    # Local x runs right and local y up. The same physical scale and center
    # apply to both panels, including when the available beam count changes.
    px0, py0, scale = (left+right)/2, top+130, 445
    origin = packet.query_ego[0]
    xy = packet.points_ego[:, :2]
    pixels = np.column_stack((px0 + (xy[:, 0]-origin[0])*scale, py0 - (xy[:, 1]-origin[1])*scale))
    y0, y1 = top+42, bottom-5
    x0, x1 = left+9, right-9
    for offset in (-.2, -.1, .0, .1, .2):
        x, y = px0 + offset*scale, py0 + offset*scale
        draw.line((x, y0, x, y1), fill=GRID)
        if y0 <= y <= y1:
            draw.line((x0, y, x1, y), fill=GRID)
    for px, py in pixels[packet.valid & keep]:
        if x0 <= px <= x1 and y0 <= py <= y1:
            draw.ellipse((px-1.4, py-1.4, px+1.4, py+1.4), fill=KEPT)
    for px, py in pixels[packet.valid & ~keep]:
        if x0 <= px <= x1 and y0 <= py <= y1:
            draw.ellipse((px-2.6, py-2.6, px+2.6, py+2.6), outline=HIDDEN, width=2)
    draw.ellipse((px0-10, py0-10, px0+10, py0+10), outline=QUERY, width=2)
    draw.line((px0-4, py0, px0+4, py0), fill=QUERY, width=1)
    draw.line((px0, py0-4, px0, py0+4), fill=QUERY, width=1)
    draw.text((right-13, top+12), "左轮迹俯视　车头 →", font=tiny, fill=MUTED, anchor="ra")


def caption(packet, ordinal, tick, pause):
    if ordinal == 0 and pause:
        return "左前轮前面的黄圈标出了查询点。两幅俯视图使用同一包点云。"
    if ordinal == 6 and pause:
        return "看右侧两图。上图维持原位置，下图的粉色回波移到了左轮迹附近。"
    if ordinal == 7 and pause:
        return "同一侧连续遮住第二包。车身仍按存档动作前进，粉圈标出被屏蔽的回波。"
    if ordinal == 8 and pause:
        return "遮挡回到原位置，两图逐束相同。黄圈随每次新包的送达更新。"
    if ordinal == 15 and pause:
        return "本段在车轮接触凸台前结束。画面回放原来在平地训练时记录的姿态。"
    if packet.active:
        return "上图保留原遮挡，下图将同样数量的遮挡移向左前轮查询区。"
    return "当前两组采用原遮挡。青点是保留的回波，粉圈是从同一包中隐藏的回波。"


def canvas(scene_image, packet, ordinal, tick, pause, fonts):
    from PIL import Image, ImageDraw

    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(frame)
    title, body, small, tiny = fonts
    draw.text((15, 7), "把遮挡移到左前轮将要经过的地面", font=title, fill=INK)
    draw.text((17, 55), "D1 平地训练的记录姿态　·　回放场景在左侧加入 3 厘米凸台", font=small, fill=MUTED)
    timing = f"画面定格　{tick/50:.2f} s" if pause else f"0.1 倍慢动作　{tick/50:.2f} s"
    draw.text((WIDTH-18, 22), timing, font=small, fill=QUERY, anchor="ra")
    frame.paste(scene_image, (14, 94))
    point_map(draw, packet, packet.old, 94, "原遮挡", fonts)
    point_map(draw, packet, packet.moved, 324, "轮迹遮挡" if packet.active else "轮迹组 · 原位置", fonts)
    draw.text((18, 559), caption(packet, ordinal, tick, pause), font=body, fill=INK)
    draw.text((18, 610), "黄圈　左前轮查询点", font=small, fill=QUERY)
    draw.text((252, 610), "粉圈　隐藏的回波", font=small, fill=HIDDEN)
    draw.text((468, 610), "青点　保留的回波", font=small, fill=KEPT)
    draw.text((WIDTH-17, 614), f"测距包 {ordinal+1}/16　·　原始时钟 50 Hz", font=tiny, fill=MUTED, anchor="ra")
    return frame


def _film(folder, out, rec, inputs):
    import mujoco
    from PIL import Image
    from OpenGL import GL
    from tools100.sensor100 import Sensor100
    from tools101.common101 import protocol as parent_protocol

    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
            "movie requires the registered software EGL renderer")
    poses, native_path, pose_hash = original_pose_stream()
    inputs["native_trace_sha256"] = native_path
    fixture, = parent_protocol()["data"]["layouts"]["holdout"]
    geometry = {**sensor_protocol()["geometry"], **fixture}
    sensor = Sensor100({"geometry": geometry}, SIDE)
    packets = load_packets(folder / "packets.npz", sensor, poses)
    archived_ticks = np.arange(packets[0].capture, packets[-1].delivery + 1)
    shown_ticks = np.arange(packets[0].delivery, packets[-1].delivery + 1)
    snapshot = out.with_name("poses.npz")
    arrays(snapshot, dict(qpos=poses[archived_ticks - 1], tick=archived_ticks, shown_ticks=shown_ticks,
                          capture_ticks=np.array([p.capture for p in packets]),
                          delivery_ticks=np.array([p.delivery for p in packets]),
                          source_seed=np.array(SOURCE), episode=np.array(EPISODE)))
    rec["poses_sha256"] = digest(snapshot)
    model, options, camera = sensor.model, mujoco.MjvOption(), mujoco.MjvCamera()
    options.geomgroup[3] = 0
    model.vis.global_.offwidth, model.vis.global_.offheight = SCENE_W, SCENE_H
    model.vis.quality.shadowsize = 1024
    model.vis.headlight.ambient[:] = .3
    model.vis.headlight.diffuse[:] = .5
    floor, box = model.geom("floor").id, model.geom("bump100").id
    model.geom_matid[floor] = -1
    model.geom_rgba[floor] = (.26, .31, .32, 1.)
    model.geom_rgba[box] = (.19, .52, .51, 1.)
    for wheel in sensor.wheels:
        visible = (model.geom_bodyid == model.geom_bodyid[wheel]) & (model.geom_group != 3)
        model.geom_rgba[visible] = (.13, .18, .19, 1.)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 108., -32., 1.50
    camera.lookat[:] = (.34, .025, .19)
    fonts = [chinese_font(35, True), chinese_font(23), chinese_font(20), chinese_font(15)]
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-n", "-threads", "1", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an",
           "-filter_threads", "1", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-maxrate", "640k",
           "-bufsize", "640k", "-threads", "1", "-x264-params", "threads=1:lookahead-threads=1",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    rec.update(native_source_seed=SOURCE, native_episode=EPISODE, native_episode_qpos_sha256=pose_hash,
               layout="unseen_left", mask_side=SIDE, capture_ticks=[p.capture for p in packets],
               delivery_ticks=[p.delivery for p in packets], cloud_ids=[p.cloud for p in packets],
               active=[p.active for p in packets], sensor_delay_ticks=2, native_rate_hz=50,
               playback_rate=.1, hold_events=[], inputs={k: digest(p) for k, p in inputs.items()})
    with mujoco.Renderer(model, width=SCENE_W, height=SCENE_H) as renderer:
        driver = GL.glGetString(GL.GL_RENDERER)
        require(driver and any(s in driver.lower() for s in (b"llvmpipe", b"softpipe", b"swrast")),
                f"expected a CPU rasterizer, received {driver!r}")
        rec["renderer"] = driver.decode()
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
            def emit(image, count):
                pixels = image.tobytes()
                for _ in range(count):
                    encoder.stdin.write(pixels)
                    rec["frames"] += 1
            try:
                for tick, ordinal, hold in clocks(rec["delivery_ticks"]):
                    row = packets[ordinal]
                    sensor.pose(poses[tick-1])
                    renderer.update_scene(sensor.data, camera=camera, scene_option=options)
                    overlay_3d(sensor, renderer.scene, row)
                    robot = Image.fromarray(renderer.render())
                    if hold:
                        rec["hold_events"].append(dict(packet=ordinal+1, delivery_tick=tick, cloud_id=row.cloud,
                                                       first_frame=rec["frames"], frames=hold))
                        emit(canvas(robot, row, ordinal, tick, True, fonts), hold)
                    emit(canvas(robot, row, ordinal, tick, False, fonts), FRAMES_PER_TICK)
            finally:
                encoder.stdin.close()
            require(encoder.wait() == 0, "video encoding failed")
    require(sensor.data.time == 0 and rec["frames"] == 1064, "unexpected native or video clock")


def render(replay_folder, output_folder, plan):
    from tools102.common102 import PLAN_SHA, protocol

    require(plan == protocol(), "unexpected movie registration")
    folder, destination = Path(replay_folder), Path(output_folder)
    out, receipt = destination / "occlusion102.mp4", destination / "video.json"
    require(not out.exists() and not receipt.exists(), "preserving previous movie attempt")
    inputs = {"packets_sha256": folder / "packets.npz", "report_sha256": folder / "report.json",
              "holdout_sha256": PUBLIC / "holdout.npz", "source_index_sha256": PUBLIC / "data.json"}
    rec = dict(status="PARTIAL", protocol_sha256=PLAN_SHA, frames=0, fps=FPS, cover_frames=0, physics_substeps=0,
               policy_updates=0, perception_updates=0, world_updates=0, policy_calls=0, new_rays=0)
    with receipt.open("x") as log:
        try:
            _film(folder, out, rec, inputs)
            stream, = json.loads(subprocess.run(["ffprobe", "-v", "error", "-threads", "1", "-count_frames",
                "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration",
                "-of", "json", str(out)], capture_output=True, check=True, text=True).stdout)["streams"]
            nbytes = out.stat().st_size
            require((stream["codec_name"], stream["width"], stream["height"], stream["r_frame_rate"]) ==
                    ("h264", WIDTH, HEIGHT, f"{FPS}/1") and int(stream["nb_read_frames"]) == rec["frames"]
                    and abs(float(stream["duration"]) - rec["frames"]/FPS) < 1e-5 and 0 < nbytes <= 5_000_000,
                    "decoded movie disagrees with registered frames or size")
            require(rec["inputs"] == {k: digest(p) for k, p in inputs.items()}, "movie sources changed")
            rec.update(status="COMPLETE", bytes=nbytes, duration_s=rec["frames"]/FPS, stream=stream,
                       video_sha256=digest(out), **rec["inputs"])
        except BaseException as exc:
            rec["error"] = repr(exc)
            raise
        finally:
            json.dump(rec, log, allow_nan=False, indent=2)
            log.write("\n")
    return rec
