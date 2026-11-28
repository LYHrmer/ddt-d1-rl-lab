"""Replay the registered speed comparison between the 5 m and 6 m gates."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import digest, read, require
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, rim_marks, verify_video
from tools110.course import assess
from tools112.surface_video import ground_line
from tools113.video import STOP_TICK, load_track
from tools114.parking_video import axle_motion_mm, marker
from tools116.response_replay import INDEX, INDEX_SHA
from tools116.video import cameras, first_tick


SEED = 111101
NAME = "compound117.mp4"
DESKTOP_NAME = "D1_台阶段超速_补偿回落前后对照_117.mp4"
PANELS = ("左 · 原编码器补偿", "右 · 补偿触顶后加快回落")


def sources(candidate_dir):
    require(digest(INDEX) == INDEX_SHA, "different round-116 index")
    reference, = [c for c in read(INDEX)["cases"] if c["seed"] == SEED]
    require(reference["stage_status"] == "COMPLETE" and reference["complete_horizon"], "incomplete reference")
    tracks, readings, receipts = [], [], []
    cases = ((INDEX.parent / f"native_{SEED}", f"native_{SEED}"), (Path(candidate_dir), f"speed_{SEED}"))
    for folder, stage in cases:
        receipt = folder / "report.json"
        report = read(receipt)
        require(report["status"] == "COMPLETE" and report["stage"] == stage and report["result"]["seed"] == SEED
                and not any(folder.glob("*partial*")), "expected complete native speed pair")
        path = folder / "trace.npz"
        track = load_track(path)
        require(track.sha256 == report["result"]["trace_sha256"] == report["files"]["trace.npz"]
                and (track.last == 600 or track.end_reason != "记录结束"), "changed or truncated motion source")
        with np.load(path, allow_pickle=False) as trace:
            motion = assess(trace, "step")
            contact = first_tick(trace["front_platform_touch"])
        gates = motion["gate_times_s"]
        readings.append(dict(entry_s=gates[2] if len(gates) >= 3 else None,
                             exit_s=gates[3] if len(gates) == 4 else None,
                             section_m_s=motion["sections_m_s"][-1] if len(gates) == 4 else None,
                             contact_tick=None if contact is None else contact+1))
        tracks.append(track)
        receipts.append(digest(receipt))
    require(tracks[0].sha256 == reference["trace_sha256"], "unregistered baseline")
    np.testing.assert_array_equal(tracks[0].pose[0], tracks[1].pose[0])
    info = dict(seed=SEED, trace_sha256=[t.sha256 for t in tracks], report_sha256=receipts,
                native_report_sha256=receipts[1], last_ticks=[t.last for t in tracks],
                end_reasons=[t.end_reason for t in tracks], measurements=readings,
                renderer_sha256=digest(Path(__file__)))
    return tracks, info


def timeline(ends, readings):
    """Slow both cars on the same clock; hold one joint result frame after the last crossing."""
    points, contacts, exits = set(), set(), []
    for end, reading in zip(ends, readings, strict=True):
        for field in ("entry_s", "exit_s"):
            if reading[field] is not None:
                tick = int(np.ceil(50 * reading[field] - 1e-9))
                require(0 <= tick <= end, "gate after end of recording")
                points.add(tick)
                if field == "exit_s":
                    exits.append(tick)
        if reading["contact_tick"] is not None:
            contacts.add(reading["contact_tick"])
    require(len(ends) == 2 and len(exits) >= 1 and contacts, "missing registered baseline events")
    points |= contacts
    start, finish = max(0, min(points)-18), min(max(ends), max(points)+14)
    step_ticks = points | set(range(start, finish+1, 2)) | {start, finish}
    # A terminal is displayed at its measured time even when it lies outside the selected metre.
    early_ends = {end for end in ends if end < STOP_TICK}
    step_ticks |= early_ends
    tail = set(range(STOP_TICK, max(ends)+1, 2)) | {end for end in ends if end >= STOP_TICK}
    shots, frames = [], 0
    for phase, ticks, holds, rate in (("step", step_ticks, {min(contacts), max(exits)} | early_ends, 2),
                                     ("tail", tail, {STOP_TICK, max(ends)}, 1)):
        ordered = sorted(ticks)
        for index, tick in enumerate(ordered):
            gap = min(2, ordered[index+1]-tick) if index+1 < len(ordered) else 1
            count = gap * rate
            if tick in holds:
                count += 2*FPS if phase == "step" and tick == max(exits) else FPS
            shots.append(dict(tick=tick, phase=phase, frames=count, first_frame=frames))
            frames += count
    return shots


def caption(track, reading, tick, phase, mark):
    if tick >= track.last and track.end_reason != "记录结束":
        return f"{track.end_reason}于 {track.last/50:.2f} 秒"
    if phase == "tail":
        offset = axle_motion_mm(track, tick, mark)
        if abs(offset) < .5:
            return "前轴位于橙线处"
        return f"前轴相对橙线{'后移' if offset < 0 else '前移'} {abs(offset):.0f} mm"
    now = tick / 50
    if reading["exit_s"] is not None and now >= reading["exit_s"] - 1e-12:
        return f"通过这一米的实测均速  {reading['section_m_s']:.4f} m/s"
    if reading["entry_s"] is not None and now >= reading["entry_s"] - 1e-12:
        return f"从 5 米线起已用时  {now-reading['entry_s']:.3f} 秒"
    return "机身中心正在接近 5 米线"


def encode(movie, tracks, info, shots, views, marks, temp):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools87.video87 import chinese_font
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    from tools111.course import install_heightfield, make_scene

    driver = probe_cpu_egl(temp)
    model = mj.MjModel.from_xml_path(str(make_scene(temp, "development")))
    install_heightfield(model, "development")
    model.vis.global_.offwidth, model.vis.global_.offheight = 432, 336
    model.vis.quality.offsamples, model.vis.quality.shadowsize = 0, 128
    data, camera = mj.MjData(model), mj.MjvCamera()
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    title, label, foot = chinese_font(34, True), chinese_font(26), chinese_font(22)
    with mj.Renderer(model, width=432, height=336) as renderer:
        proc = encoder(movie)
        try:
            for shot in shots:
                tick, phase = shot["tick"], shot["phase"]
                step = phase == "step"
                heading = "台阶前后这一米，机身用多久通过？" if step else "驶离台阶后 · 原驻车控制下的末半秒"
                sheet = Image.new("RGB", (WIDTH, HEIGHT), "#17252e")
                draw = ImageDraw.Draw(sheet)
                draw.text((16, 8), heading, font=title, fill="#f3f5ef")
                draw.text((WIDTH-18, 14), f"仿真 {tick/50:.2f} 秒", anchor="ra", font=label, fill="#e2edeb")
                for side, track in enumerate(tracks):
                    left = side*WIDTH//2
                    draw.text((left+15, 58), PANELS[side], font=label, fill="#e2edeb")
                    if tick <= track.last:
                        view = views[phase][side]
                        camera.lookat[:] = view["lookat_m"]
                        camera.distance = view["distance_m"]
                        camera.azimuth, camera.elevation = view["azimuth_deg"], view["elevation_deg"]
                        data.qpos[:] = track.pose[tick]
                        mj.mj_fwdPosition(model, data)
                        np.testing.assert_allclose(data.geom_xpos[wheels[:2], :2].mean(axis=0), track.axle[tick],
                                                   atol=1e-8, rtol=0.)
                        renderer.update_scene(data, camera=camera)
                        scene = renderer.scene
                        scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                        rim_marks(mj, scene, model, data, wheels)
                        if step:
                            for x, color in ((5., (.20, .92, .75, 1.)), (6., (.60, .52, 1., 1.))):
                                ground_line(mj, scene, (x, 0.), 0., .80)
                                scene.geoms[scene.ngeom-1].rgba[:] = color
                        else:
                            mark = marks[side]
                            ground_line(mj, scene, mark["xy_m"], mark["yaw_rad"], .60)
                        panel = Image.fromarray(renderer.render()).resize((WIDTH//2-2, 496), Image.Resampling.BILINEAR)
                        sheet.paste(panel, (left, 96))
                    text = caption(track, info["measurements"][side], tick, phase, marks[side])
                    draw.text((left+15, 611), text, font=label, fill="#f6dfa8")
                note = ("绿线 5 米，紫线 6 米 · 该段验收范围 1.425～1.575 m/s" if step else
                        "橙线标出各车 11.50 秒时的前轴位置")
                draw.text((16, 672), f"闭环仿真存档 · 起点 {SEED}　　{note}", font=foot, fill="#d7e4e5")
                pixels = sheet.tobytes()
                for _ in range(shot["frames"]):
                    proc.stdin.write(pixels)
            proc.stdin.close()
            require(proc.wait(timeout=90) == 0, "video encoder failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return driver


def render(out, candidate_dir):
    out = Path(out)
    movie = out / NAME
    require(out.is_dir() and not movie.exists(), "speed film requires a fresh output")
    tracks, info = sources(candidate_dir)
    shots = timeline(info["last_ticks"], info["measurements"])
    marks = [marker(track) for track in tracks]
    views = cameras(tracks, shots, marks)
    with TemporaryDirectory(prefix="d1-step117-film-") as tmp:
        driver = encode(movie, tracks, info, shots, views, marks, Path(tmp))
    frames = sum(s["frames"] for s in shots)
    stream = verify_video(movie, frames)
    size = movie.stat().st_size
    require(0 < size <= 5_000_000 and sources(candidate_dir)[1] == info, "film size or sources changed")
    return dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **info, sha256=digest(movie),
                frames=frames, fps=FPS, duration_s=float(stream["duration"]), bytes=size, stream=stream, driver=driver,
                shots=shots, cameras=views, ground_markers=marks, panels=list(PANELS), physics=0, cover_frames=0)
