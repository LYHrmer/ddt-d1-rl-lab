"""Show the first prespecified independent reset on the original compound course."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import digest, read, require
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, rim_marks, verify_video
from tools110.course import assess
from tools111.course import ground
from tools112.surface_video import ground_line
from tools113.video import STOP_TICK, load_track
from tools114.closed_results import parking_stop
from tools114.parking_video import END, ZERO, axle_motion_mm, marker
from tools116.video import first_tick


SEED = 2087901246
NAME = "compound118.mp4"
DESKTOP_NAME = "D1_复合路独立起点盲测_118.mp4"
LAYOUT = "development"


def source(native_dir):
    folder = Path(native_dir)
    report_path = folder / "report.json"
    report = read(report_path)
    row = report["result"]
    require(report["stage"] == f"blind_{SEED}" and report["status"] == row["status"] == "COMPLETE"
            and row["seed"] == SEED and row["layout"] == LAYOUT and not any(folder.glob("*partial*")),
            "expected the completed first blind trial on the original route")
    path = folder / "trace.npz"
    track = load_track(path)
    require(track.sha256 == row["trace_sha256"] == report["files"]["trace.npz"]
            and (track.last == END or track.end_reason != "记录结束"), "changed or truncated video source")
    with np.load(path, allow_pickle=False) as trace:
        motion = assess(trace, "step")
        stop = parking_stop(trace["qpos"], trace["qvel"])
        contact = first_tick(trace["front_platform_touch"])
        signals = {k:trace[k] for k in ("capture_tick", "range_m", "range_valid", "capture_used")}
    signals["qpos"] = track.pose
    fields = ("ticks", "gate_times_s", "speed_m_s", "sections_m_s", "stop_tail_max_m_s")
    require(all(motion[k] == row["motion"][k] for k in fields) and stop == row["stop"],
            "video measurements disagree with the recorded motion")
    info = dict(seed=SEED, layout=LAYOUT, trace_sha256=track.sha256, source_report_sha256=digest(report_path),
                renderer_sha256=digest(Path(__file__)), last_tick=track.last, end_reason=track.end_reason,
                gate_times_s=motion["gate_times_s"], speed_m_s=motion["speed_m_s"], sections_m_s=motion["sections_m_s"],
                contact_tick=None if contact is None else contact+1, stop=stop, ground_marker=marker(track))
    return track, signals, info


def timeline(last_tick, gate_times_s, contact_tick):
    """Display only measured poses, selecting terrain and parking windows on their native clock."""
    require(1 <= last_tick <= END and 0 <= len(gate_times_s) <= 4, "invalid source clock")
    gates = [int(np.ceil(sec*50 - 1e-9)) for sec in gate_times_s]
    require(all(0 <= t <= last_tick for t in gates) and all(a < b for a,b in zip(gates, gates[1:]))
            and (contact_tick is None or 1 <= contact_tick <= last_tick), "unmeasured gate or contact")
    drive_end = min(last_tick, 500, gates[-1]+12 if len(gates) == 4 else 500)
    start = max(0, min(drive_end, gates[0])-18) if gates else max(0, drive_end-55)
    ticks = {*range(start, drive_end+1, 2), start, drive_end, last_tick, *gates}
    if last_tick >= 500:
        ticks.update(range(500, min(STOP_TICK, last_tick)+1, 3))
    if last_tick >= ZERO:
        ticks.add(ZERO)
    if last_tick >= STOP_TICK:
        ticks.update(range(STOP_TICK, last_tick+1, 2))
    if contact_tick is not None:
        ticks.add(contact_tick)
    step_start = min([*gates[2:3], *([contact_tick] if contact_tick is not None else [])], default=END+10)-8
    holds = {last_tick: 2*FPS}
    for k in (contact_tick, ZERO if last_tick >= ZERO else None, STOP_TICK if last_tick >= STOP_TICK else None,
              gates[3] if len(gates) == 4 else None):
        if k is not None:
            holds[k] = max(holds.get(k, 0), FPS)
    ordered, shots, frames = sorted(ticks), [], 0
    for i, k in enumerate(ordered):
        phase = "tail" if k >= STOP_TICK else "braking" if k >= 500 else "step" if k >= step_start else "cruise"
        gap = min(3, ordered[i+1]-k) if i+1 < len(ordered) else 1
        count = (2 if phase in ("step", "tail") else 1)*gap + holds.get(k, 0)
        shots.append(dict(phase=phase, tick=k, frames=count, first_frame=frames))
        frames += count
    return shots


def caption(info, k, phase, axle_offset_mm=None):
    if k == info["last_tick"] and info["end_reason"] != "记录结束":
        return f"{info['end_reason']} · 末个实测姿态定格"
    if phase == "tail":
        if k == END and info["stop"] is not None:
            stop = info["stop"]
            delta = stop["forward_displacement_mm"]
            return (f"末半秒机身水平速率峰值 {stop['peak_m_s']:.4f} m/s"
                    f"　·　机身{'后移' if delta < 0 else '前移'} {abs(delta):.1f} mm")
        if abs(axle_offset_mm) < .5:
            return "橙线标出 11.50 秒的前轴位置"
        return f"前轴相对橙线{'后移' if axle_offset_mm < 0 else '前移'} {abs(axle_offset_mm):.0f} mm"
    if phase == "braking":
        return "驾驶指令已归零" if k >= ZERO else "驾驶员正在减速"
    reached = sum(t <= k/50 + 1e-12 for t in info["gate_times_s"])
    if reached == 4:
        return f"3～6 米实测均速 {info['speed_m_s']:.4f} m/s　·　最后一米 {info['sections_m_s'][-1]:.4f} m/s"
    if k == info["contact_tick"]:
        return "前轮首次与台阶发生有载接触"
    if reached:
        return f"机身已通过 {reached+2} 米线"
    return "机身正在接近测速区"


def encode(movie, track, signals, info, shots, temp):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools87.video87 import chinese_font
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    from tools111.course import install_heightfield, make_scene
    from tools111.video import dots, world_returns

    driver = probe_cpu_egl(temp)
    model = mj.MjModel.from_xml_path(str(make_scene(temp, LAYOUT)))
    install_heightfield(model, LAYOUT)
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 300
    model.vis.quality.offsamples, model.vis.quality.shadowsize = 0, 128
    data, cam = mj.MjData(model), mj.MjvCamera()
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    clouds = world_returns(mj, signals)
    mark = info["ground_marker"]
    title, label, small = chinese_font(33, True), chinese_font(27), chinese_font(22)
    titles = dict(cruise="上下坡与交错起伏", step="30 毫米台阶与测速出口",
                  braking="驾驶员减速与归零", tail="停车末半秒 · 前轮近景")
    with mj.Renderer(model, width=640, height=300) as renderer:
        proc = encoder(movie)
        try:
            for shot in shots:
                k, phase = shot["tick"], shot["phase"]
                pos = track.pose[k, :3]
                cam.azimuth, cam.elevation = 98., -17.
                if phase == "tail":
                    cam.lookat[:] = (*mark["xy_m"], .14)
                    cam.distance = 1.12
                elif phase == "step" and 4.55 <= pos[0] <= 6.55:
                    cam.lookat[:] = (5.54, track.pose[min(info["contact_tick"] or k, track.last), 1], .22)
                    cam.distance = 2.55
                else:
                    cam.lookat[:] = (pos[0]+.16, pos[1], .23+float(ground(LAYOUT, *pos[:2])))
                    cam.distance = 2.40
                data.qpos[:] = track.pose[k]
                mj.mj_fwdPosition(model, data)
                np.testing.assert_allclose(data.geom_xpos[wheels[:2], :2].mean(axis=0), track.axle[k], rtol=0., atol=1e-8)
                renderer.update_scene(data, camera=cam)
                scene = renderer.scene
                scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                rim_marks(mj, scene, model, data, wheels)
                if phase in ("cruise", "step"):
                    for x in (3., 4., 5., 6.):
                        ground_line(mj, scene, (x, 0.), 0., .80)
                        line = scene.geoms[scene.ngeom-1]
                        line.pos[2] = .004 + float(ground(LAYOUT, x, 0.))
                        line.rgba[:] = (.60, .52, 1., 1.) if x == 6. else (.20, .92, .75, 1.)
                    if k:
                        dots(mj, scene, clouds.get(int(signals["capture_used"][k-1]), ()))
                if phase == "tail":
                    ground_line(mj, scene, mark["xy_m"], mark["yaw_rad"], .60)
                image = Image.fromarray(renderer.render()).resize((WIDTH, 600), Image.Resampling.BILINEAR)
                sheet = Image.new("RGB", (WIDTH, HEIGHT), "#17252e")
                sheet.paste(image, (0, 58))
                draw = ImageDraw.Draw(sheet)
                draw.text((18, 8), f"D1 独立起点 · {titles[phase]}", font=title, fill="#f3f5ef")
                draw.text((WIDTH-18, 14), f"仿真 {k/50:.2f} 秒", anchor="ra", font=label, fill="#e2edeb")
                delta = axle_motion_mm(track, k, mark) if phase == "tail" else None
                draw.text((18, 670), caption(info, k, phase, delta), font=label, fill="#f6dfa8")
                source_label = "闭环存档 · 金点为雷达回波" if phase in ("step", "cruise") else "闭环存档"
                draw.text((WIDTH-18, 676), f"{source_label} · {SEED}", font=small, anchor="ra", fill="#d7e4e5")
                pixels = sheet.tobytes()
                for _ in range(shot["frames"]):
                    proc.stdin.write(pixels)
            proc.stdin.close()
            require(proc.wait(timeout=90) == 0, "blind film encoder failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return driver


def verify(record, movie, native_dir):
    _, _, info = source(native_dir)
    shots = timeline(info["last_tick"], info["gate_times_s"], info["contact_tick"])
    frames = sum(s["frames"] for s in shots)
    movie = Path(movie)
    expected = dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **info, shots=shots,
                    frames=frames, fps=FPS, duration_s=frames/FPS, bytes=movie.stat().st_size, sha256=digest(movie),
                    physics=0, cover_frames=0)
    require(all(record[k] == v for k,v in expected.items()) and 0 < record["bytes"] <= 5_000_000
            and "llvmpipe" in record["driver"]["renderer"].lower(), "wrong blind replay or measured evidence")
    stream = record["stream"]
    require((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"], stream["r_frame_rate"])
            == (WIDTH, HEIGHT, "h264", "yuv420p", f"{FPS}/1") and int(stream["nb_read_frames"]) == frames
            and abs(float(stream["duration"])-frames/FPS) <= .001, "wrong encoded blind stream")
    return record


def render(out, native_dir):
    movie = Path(out) / NAME
    require(movie.parent.is_dir() and not movie.exists(), "blind film requires a fresh output")
    track, signals, info = source(native_dir)
    shots = timeline(track.last, info["gate_times_s"], info["contact_tick"])
    with TemporaryDirectory(prefix="d1-blind118-film-") as temp:
        driver = encode(movie, track, signals, info, shots, Path(temp))
    frames = sum(s["frames"] for s in shots)
    stream = verify_video(movie, frames)
    result = dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **info, sha256=digest(movie),
                  frames=frames, fps=FPS, bytes=movie.stat().st_size, duration_s=float(stream["duration"]),
                  stream=stream, driver=driver, shots=shots, physics=0, cover_frames=0)
    return verify(result, movie, native_dir)
