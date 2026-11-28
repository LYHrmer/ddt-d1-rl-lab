"""Show the same measured D1 stop with the original and projected wheel targets."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import ROOT, digest, read, require, write
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, verify_video
from tools113.video import STOP_TICK, load_track
from tools114.closed_results import parking_stop
from tools114.parking_video import END, ZERO, camera_setup, caption, marker, robot_panel, timeline
from tools117.run import PUBLIC, completed


SEED = 111101
STAGES = (f"speed_{SEED}", f"parking_{SEED}")
NAME = "compound117_parking.mp4"
DESKTOP_NAME = "D1_复合路停车_四轮共同目标归零前后_117.mp4"
PANELS = ("左 · 原轮速目标", "右 · 驾驶指令归零后，去掉四轮的共同目标")


def sources():
    folders = [PUBLIC / stage for stage in STAGES]
    if all((folder / "report.json").is_file() for folder in folders):
        for folder in folders:
            guard = read(folder / "guard.json")
            require(guard["status"] == "COMPLETE" and guard["child_reaped"] and guard["returncode"] == 0,
                    "incomplete published guard")
    else:
        records = [completed(stage) for stage in STAGES]
        require(all(records), "missing complete parking video inputs")
        folders = [record[0] for record in records]
    tracks, reports, stops, trace_paths = [], [], [], []
    for stage, folder in zip(STAGES, folders, strict=True):
        report = read(folder / "report.json")
        row = report["result"]
        require(report["stage"] == stage and report["status"] == row["status"] == "COMPLETE"
                and row["seed"] == SEED and row["parking"] == (stage == STAGES[1])
                and row["layout"] == "development" and row["arm"] == "both" and row["motion"]["ticks"] == END
                and not any(folder.glob("*partial*")), "expected the two complete, matched compound parking runs")
        path = folder / "trace.npz"
        track = load_track(path)
        require(track.last == END and track.sha256 == row["trace_sha256"] == report["files"]["trace.npz"],
                "wrong pose archive or clock")
        with np.load(path, allow_pickle=False) as trace:
            stop = parking_stop(trace["qpos"], trace["qvel"])
            require(trace["base_cmd"][ZERO-1] > 0 and np.all(trace["base_cmd"][ZERO:] == 0),
                    "wrong driver-zero tick")
        require(stop == row["stop"], "different measured stopping motion")
        tracks.append(track)
        stops.append(stop)
        reports.append(row)
        trace_paths.append(path)
    require(reports[1]["reference_sha256"] == tracks[0].sha256 and all(reports[1][k] == reports[0][k]
            for k in ("gain", "actor_sha256", "actuator_config_sha256", "mujoco_version")), "unpaired parking sources")
    with np.load(trace_paths[0], allow_pickle=False) as old, np.load(trace_paths[1], allow_pickle=False) as new:
        first_applied = int(np.flatnonzero(np.any(old["applied"] != new["applied"], axis=1))[0])
        require(first_applied == ZERO + 1, "wrong effective intervention tick")
        np.testing.assert_array_equal(old["qpos"][:first_applied+1], new["qpos"][:first_applied+1])
    receipts = [digest(f / "report.json") for f in folders]
    info = dict(seed=SEED, trace_sha256=[track.sha256 for track in tracks], report_sha256=receipts,
                native_report_sha256=receipts[1], renderer_sha256=digest(Path(__file__)), last_ticks=[END, END],
                driver_zero_tick=ZERO, first_changed_applied_tick=first_applied, tail_endpoints=[STOP_TICK, END],
                stop=stops, ground_markers=[marker(t) for t in tracks])
    return tracks, info


def shots_for(tracks):
    shots = timeline([t.last for t in tracks])
    frames = 0
    for shot in shots:
        shot["first_frame"] = frames
        frames += shot["frames"]
    return shots


def encode(movie, tracks, info, shots, views, temp):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools87.video87 import chinese_font
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    from tools111.course import install_heightfield, make_scene

    driver = probe_cpu_egl(temp)
    model = mj.MjModel.from_xml_path(str(make_scene(temp, "development")))
    install_heightfield(model, "development")
    model.vis.global_.offwidth, model.vis.global_.offheight = 422, 326
    model.vis.quality.offsamples, model.vis.quality.shadowsize = 0, 128
    data, camera = mj.MjData(model), mj.MjvCamera()
    wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
    title, label, small = chinese_font(32, True), chinese_font(26), chinese_font(22)
    with mj.Renderer(model, width=422, height=326) as renderer:
        proc = encoder(movie)
        try:
            for shot in shots:
                phase, k = shot["phase"], shot["tick"]
                camera.lookat[:] = views[phase]["lookat_m"]
                camera.distance = views[phase]["distance_m"]
                camera.azimuth, camera.elevation = views[phase]["azimuth_deg"], views[phase]["elevation_deg"]
                sheet = Image.new("RGB", (WIDTH, HEIGHT), "#17252e")
                draw = ImageDraw.Draw(sheet)
                heading = "复合路停车 · 驾驶员减速" if k < ZERO else (
                    "10.50 秒 · 驾驶指令归零" if k == ZERO else
                    "停车末半秒 · 观察前轮与橙线" if k >= STOP_TICK else "驾驶指令归零后的车身运动")
                draw.text((16, 8), heading, font=title, fill="#f3f5ef")
                draw.text((WIDTH-18, 14), f"仿真 {k/50:.2f} 秒", anchor="ra", font=label, fill="#e2edeb")
                for side, (track, mark) in enumerate(zip(tracks, info["ground_markers"], strict=True)):
                    left = side*WIDTH//2
                    draw.text((left+16, 58), PANELS[side], font=label, fill="#e2edeb")
                    sheet.paste(robot_panel(mj, renderer, model, data, camera, wheels, track, k, mark), (left, 96))
                    draw.text((left+16, 611), caption(track, k, mark), font=label, fill="#f6dfa8")
                note = ("两侧橙线分别标出本车 11.50 秒时的前轴位置" if k >= STOP_TICK else
                        "同一次前进请求，两侧共用地面坐标和仿真时钟")
                draw.text((16, 672), f"闭环仿真存档 · 起点 {SEED}　　{note}", font=small, fill="#d7e4e5")
                pixels = sheet.tobytes()
                for _ in range(shot["frames"]):
                    proc.stdin.write(pixels)
            proc.stdin.close()
            require(proc.wait(timeout=90) == 0, "parking encoder failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return driver


def _verify(report, movie_path):
    tracks, info = sources()
    shots = shots_for(tracks)
    frames = sum(s["frames"] for s in shots)
    movie = Path(movie_path)
    expected = dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **info,
                    shots=shots, cameras=camera_setup(tracks, shots), frames=frames, duration_s=frames/FPS,
                    panels=list(PANELS), fps=FPS, physics=0, cover_frames=0, sha256=digest(movie), bytes=movie.stat().st_size)
    require(all(report[k] == v for k, v in expected.items()) and 0 < expected["bytes"] <= 5_000_000
            and "llvmpipe" in report["driver"]["renderer"].lower(), "unbound parking replay")
    stream = report["stream"]
    require((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"], stream["r_frame_rate"])
            == (WIDTH, HEIGHT, "h264", "yuv420p", f"{FPS}/1") and int(stream["nb_read_frames"]) == frames
            and abs(float(stream["duration"])-frames/FPS) <= .001, "wrong parking movie stream receipt")
    return report


def verify(receipt_path, movie_path):
    return _verify(read(Path(receipt_path)), movie_path)


def render(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        tracks, info = sources()
        shots = shots_for(tracks)
        views = camera_setup(tracks, shots)
        movie = out / NAME
        with TemporaryDirectory(prefix="d1-stop117-film-") as temp:
            driver = encode(movie, tracks, info, shots, views, Path(temp))
        frames = sum(s["frames"] for s in shots)
        stream = verify_video(movie, frames)
        result = dict(status="COMPLETE", filename=NAME, desktop_filename=DESKTOP_NAME, **info,
                      shots=shots, cameras=views, fps=FPS, frames=frames, duration_s=float(stream["duration"]),
                      panels=list(PANELS), physics=0, cover_frames=0, driver=driver, stream=stream,
                      sha256=digest(movie), bytes=movie.stat().st_size)
        _verify(result, movie)
        write(out / "video.json", result)
    except BaseException as exc:
        write(out / "partial.json", dict(status="PARTIAL", error=repr(exc)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="fresh worker output directory")
    mode.add_argument("--verify", type=Path, help="published video.json")
    parser.add_argument("--movie", type=Path, default=ROOT/"docs/assets"/NAME)
    args = parser.parse_args()
    if args.output:
        render(args.output)
    else:
        result = verify(args.verify, args.movie)
        require(verify_video(args.movie, result["frames"]) == result["stream"], "encoded frames differ")
