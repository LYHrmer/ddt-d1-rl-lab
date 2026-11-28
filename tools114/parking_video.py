"""Replay the prespecified round-113/114 pair on a common, ground-fixed camera and clock."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools87.video87 import chinese_font
from tools91.launch91 import digest, read, require, write
from tools100.sensor100 import _wheel_cylinders
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, rim_marks, verify_video
from tools110.course import scene
from tools112.surface_video import ground_line
from tools113.video import STOP_TICK, load_track
from tools114.common import DEV, baseline


NAME = "parking114.mp4"
DESKTOP_NAME = "D1_停车修复_等高网格_114.mp4"
START, ZERO, END = 500, 525, 600
LABELS = ("左 · 第 113 轮保留四轮平均轮速目标", "右 · 第 114 轮停车时清零四轮平均目标")


def sources(native_dir):
    seed = DEV[0]
    old, old_row = baseline(seed)
    path, receipt = native_dir / "trace.npz", native_dir / "report.json"
    report = read(receipt)
    require(report["status"] == "COMPLETE", "native worker did not complete")
    row = report["result"]
    require(row["seed"] == seed == 111011 and row["status"] == "COMPLETE" and row["cohort"] == "development"
            and row["projection"] == "parked_wheel_common_zero"
            and row["reference_sha256"] == old_row["trace_sha256"] and row["trace_sha256"] == report["files"]["trace.npz"]
            and all(row[k] == old_row[k] for k in ("actor_sha256", "actuator_config_sha256", "layout", "surface",
                                                   "arm", "gain", "trim", "mujoco_version")),
            "expected the registered, matched round-113/114 native pair")
    tracks = [load_track(p) for p in (old, path)]
    require([t.sha256 for t in tracks] == [old_row["trace_sha256"], row["trace_sha256"]]
            and [t.last for t in tracks] == [END, row["motion"]["ticks"]]
            and (tracks[1].last == END or tracks[1].end_reason != "记录结束"), "wrong source or unexplained terminal")
    # The common controller and one-tick motor delay preserve the same physical
    # endpoints up to and including the first tick after the zero driver request.
    stop = min(tracks[1].last, ZERO+1) + 1
    np.testing.assert_array_equal(tracks[0].pose[:stop], tracks[1].pose[:stop])
    provenance = dict(seed=seed, trace_sha256=[t.sha256 for t in tracks], native_report_sha256=digest(receipt),
                      actor_sha256=row["actor_sha256"], last_ticks=[t.last for t in tracks],
                      renderer_sha256=digest(Path(__file__)))
    return tracks, provenance


def timeline(last_ticks):
    require(len(last_ticks) == 2 and all(0 < k <= END for k in last_ticks), "invalid endpoint pair")
    last = max(last_ticks)
    # Preserve any early terminal on camera before cutting to the stopping window.
    early = sorted({k for end in last_ticks if end < START for k in (max(0, end-6), end)})
    wide_end = min(last, STOP_TICK)
    wide = set(range(START, wide_end+1, 3)) | {k for k in (START, ZERO, wide_end, *last_ticks)
                                              if START <= k <= wide_end}
    periods = [("overview", sorted(set(early) | wide))]
    if last >= STOP_TICK:
        periods.append(("front_axle", sorted(set(range(STOP_TICK, last+1, 2)) |
                                             {STOP_TICK, last, *(k for k in last_ticks if k >= STOP_TICK)})))
    shots = []
    for phase, ticks in periods:
        for i, k in enumerate(ticks):
            gap = min(3, ticks[i+1]-k) if i+1 < len(ticks) else 1
            frames = gap * (2 if phase == "front_axle" else 1)
            if not shots or k == ZERO or (phase == "front_axle" and k == STOP_TICK):
                frames += FPS
            if k == last and phase == periods[-1][0]:
                frames += 2*FPS
            elif k in last_ticks and k != END:
                frames += FPS
            shots.append(dict(phase=phase, tick=k, frames=frames))
    return shots


def camera_setup(tracks, shots):
    cameras = {}
    for phase in dict.fromkeys(s["phase"] for s in shots):
        points = np.array([t.axle[s["tick"]] for s in shots if s["phase"] == phase
                           for t in tracks if s["tick"] <= t.last])
        lo, hi = points.min(axis=0), points.max(axis=0)
        cx, cy = .5 * (lo + hi)
        close = phase == "front_axle"
        cameras[phase] = dict(lookat_m=[cx-(.035 if close else .12), cy, .13 if close else .24],
                              distance_m=(.85 if close else 1.60) + 1.5*np.linalg.norm(hi-lo),
                              azimuth_deg=98., elevation_deg=-17.)
    return cameras


def marker(track):
    if track.last < STOP_TICK:
        return None
    w, x, y, z = track.pose[STOP_TICK, 3:7]
    yaw = float(np.arctan2(2.*(w*z+x*y), 1.-2.*(y*y+z*z)))
    return dict(xy_m=track.axle[STOP_TICK].tolist(), yaw_rad=yaw)


def axle_motion_mm(track, tick, mark):
    offset = track.axle[tick] - mark["xy_m"]
    yaw = mark["yaw_rad"]
    return float(1000. * offset @ [np.cos(yaw), np.sin(yaw)])


def caption(track, k, mark):
    if k > track.last or (k == track.last and track.end_reason != "记录结束"):
        return f"{track.end_reason}于 {track.last / 50:.2f} 秒"
    if k < ZERO:
        return "驾驶员前进指令正在减小"
    if k < STOP_TICK:
        return "驾驶员前进指令为零"
    travel = axle_motion_mm(track, k, mark)
    if abs(travel) < .5:
        return "前轴在标线处（约 0 mm）"
    return f"前轴相对标线{'后移' if travel < 0 else '前移'} {abs(travel):.0f} mm"


def robot_panel(mj, renderer, model, data, camera, wheels, track, k, mark):
    from PIL import Image

    data.qpos[:] = track.pose[k]
    mj.mj_fwdPosition(model, data)
    np.testing.assert_allclose(data.geom_xpos[wheels[:2], :2].mean(axis=0), track.axle[k], rtol=0., atol=1e-8)
    renderer.update_scene(data, camera=camera)
    view = renderer.scene
    view.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
    view.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
    rim_marks(mj, view, model, data, wheels)
    if mark is not None and k >= STOP_TICK:
        ground_line(mj, view, mark["xy_m"], mark["yaw_rad"], .60)
    return Image.fromarray(renderer.render()).resize((WIDTH//2-2, 494), Image.Resampling.BILINEAR)


def encode(movie, tracks, shots, cameras, markers, temp):
    from PIL import Image, ImageDraw
    import mujoco as mj
    from tools102.mesa_video102 import probe_cpu_egl
    from tools112.surface_native import flatten

    driver = probe_cpu_egl(temp)
    model = mj.MjModel.from_xml_path(str(scene(temp, "ridge")))
    flatten(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 422, 326
    model.vis.quality.offsamples, model.vis.quality.shadowsize = 0, 128
    data, camera = mj.MjData(model), mj.MjvCamera()
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    title, label, small = chinese_font(33, True), chinese_font(26), chinese_font(22)
    with mj.Renderer(model, width=422, height=326) as renderer:
        proc = encoder(movie)
        try:
            for shot in shots:
                k, phase = shot["tick"], shot["phase"]
                cam = cameras[phase]
                camera.lookat[:] = cam["lookat_m"]
                camera.distance, camera.azimuth, camera.elevation = cam["distance_m"], cam["azimuth_deg"], cam["elevation_deg"]
                sheet = Image.new("RGB", (WIDTH, HEIGHT), "#17252e")
                draw = ImageDraw.Draw(sheet)
                if phase == "front_axle":
                    heading = "停车近景 · 观察前轮相对橙线的位置"
                elif k == STOP_TICK:
                    heading = "在前轴正下方的地面设置橙线"
                elif k == ZERO:
                    heading = "10.50 秒 · 前进指令归零"
                else:
                    heading = "等高网格减速停车" if k < ZERO else "前进指令归零后的车身运动"
                draw.text((17, 8), heading, font=title, fill="#f4f5ed")
                draw.text((WIDTH-18, 12), f"仿真 {k / 50:.2f} 秒", anchor="ra", font=label, fill="#e3edeb")
                for side, (t, mark) in enumerate(zip(tracks, markers)):
                    left = side * (WIDTH//2)
                    draw.text((left+15, 57), LABELS[side], font=label, fill="#e3edeb")
                    if k <= t.last:
                        sheet.paste(robot_panel(mj, renderer, model, data, camera, wheels, t, k, mark), (left, 96))
                    draw.text((left+15, 608), caption(t, k, mark), font=label, fill="#f6dfa8")
                note = "两侧橙线分别标出本车 11.50 秒的前轴位置" if k >= STOP_TICK else "同一地面坐标、同一视角、同一仿真时刻"
                draw.text((17, 670), f"MuJoCo 闭环轨迹回放　·　开发起点 {DEV[0]}　·　{note}", font=small, fill="#d6e4e5")
                pixels = sheet.tobytes()
                for _ in range(shot["frames"]):
                    proc.stdin.write(pixels)
            proc.stdin.close()
            if proc.wait(timeout=90):
                raise RuntimeError("ffmpeg could not finish the parking movie")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return driver


def render(native_dir, output_dir):
    native_dir, out = Path(native_dir), Path(output_dir)
    tracks, inputs = sources(native_dir)
    shots = timeline(inputs["last_ticks"])
    cameras = camera_setup(tracks, shots)
    markers = [marker(t) for t in tracks]
    frames = 0
    for s in shots:
        s["first_frame"] = frames
        frames += s["frames"]
    out.mkdir(parents=True, exist_ok=False)
    movie = out / NAME
    try:
        with TemporaryDirectory(prefix="d1-parking114-render-") as folder:
            driver = encode(movie, tracks, shots, cameras, markers, Path(folder))
        size = movie.stat().st_size
        require(0 < size <= 5_000_000, "film exceeds the 5 MB limit")
        require(inputs == sources(native_dir)[1], "source changed during playback")
        stream = verify_video(movie, frames)
        result = dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **inputs, sha256=digest(movie),
                      bytes=size, fps=FPS, frames=frames, duration_s=float(stream["duration"]), stream=stream,
                      driver=driver, panels=list(LABELS), marker_tick=STOP_TICK, driver_zero_tick=ZERO,
                      ground_markers=markers, cameras=cameras, shots=shots,
                      final_front_axle_offset_mm=[axle_motion_mm(t, t.last, m) if m else None
                                                  for t, m in zip(tracks, markers)],
                      cover_frames=0, physics=0)
        write(out / "video.json", result)
        return result
    except BaseException as error:
        write(out / "partial.json", dict(status="PARTIAL", **inputs, error=repr(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="fresh, dedicated output directory")
    args = parser.parse_args()
    receipt = render(args.native_dir, args.output_dir)
    print(json.dumps({k: receipt[k] for k in ("filename", "bytes", "sha256", "frames", "duration_s")}))
