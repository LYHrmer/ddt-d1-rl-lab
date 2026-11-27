"""Show the registered grid runs on a common clock using their recorded robot poses."""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools87.video87 import chinese_font
from tools91.launch91 import digest
from tools100.sensor100 import _wheel_cylinders
from tools108.video108 import FPS, HEIGHT, WIDTH, encoder, rim_marks, verify_video
from tools110.course import scene
from tools112.surface_video import ground_line


NAME = "D1_高速世界模型_等高网格巡航与停车_113.mp4"
PANELS = ("速度补偿关闭", "速度补偿开启")
STOP_TICK = 575


@dataclass
class Track:
    pose: np.ndarray
    axle: np.ndarray
    distance: np.ndarray
    gate: int | None
    end_reason: str
    sha256: str

    @property
    def last(self):
        return len(self.pose) - 1


def load_track(path):
    with np.load(path, allow_pickle=False) as trace:
        pose, times, xy = trace["qpos"], trace["time"], trace["xy"]
        yaw, wheels = trace["imu_yaw"], trace["wheels_xy"]
        n = len(times) - 1
        if not (0 <= n <= 600 and pose.shape == (n + 1, 23)
                and wheels.shape == (n + 1, 4, 2) and yaw.shape == (n,)):
            raise ValueError("incomplete native endpoints in the playback source")
        np.testing.assert_allclose(times, np.arange(n + 1) / 50, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(pose[:, :2], xy)
        if not all(np.isfinite(v).all() for v in (pose, wheels, yaw)):
            raise ValueError("nonfinite playback source")
        direction = np.column_stack((np.cos(yaw), np.sin(yaw)))
        distance = np.r_[0., np.cumsum((np.diff(xy, axis=0) * direction).sum(axis=1))]
        crossings = np.flatnonzero((xy[:-1, 0] < 6.) & (xy[1:, 0] >= 6.))
        reason = "跌倒终止" if np.any(trace["fallen"]) else (
            "非法接触终止" if np.any(trace["illegal"]) else "记录结束")
    return Track(pose, wheels[:, :2].mean(axis=1), distance,
                 int(crossings[0]) + 1 if len(crossings) else None, reason, digest(path))


def shots(tracks):
    """Group nearby crossings, show any early termination, then slow down the stop."""
    last = max(t.last for t in tracks)
    gates = {t.gate for t in tracks if t.gate is not None and t.gate < 520}
    early_ends = {t.last for t in tracks if t.last < 520}
    events = sorted(gates | early_ends or {min(450, last)})
    intervals = []
    for tick in events:
        first, end = max(0, tick - 30), min(last, 519, tick + 12)
        if intervals and first <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((first, end))
    phases = [("cruise", first, end, set(events)) for first, end in intervals]
    if last >= 520:
        phases.append(("braking", 520, last, {525, STOP_TICK, last, *(t.last for t in tracks)}))
    for phase, first, end, landmarks in phases:
        ticks = sorted({*range(first, end + 1, 4), first, end,
                        *(k for k in landmarks if first <= k <= end)})
        for i, k in enumerate(ticks):
            dt = ticks[i + 1] - k if i + 1 < len(ticks) else 1
            dwell = FPS if k in landmarks and k != 525 else 0
            yield phase, k, dt * (2 if phase == "braking" and k >= STOP_TICK else 1) + dwell


def draw_robot(mj, renderer, model, data, camera, wheels, track, tick, braking):
    from PIL import Image

    data.qpos[:] = track.pose[tick]
    mj.mj_fwdPosition(model, data)
    camera.azimuth, camera.elevation = 98., -15.
    if braking:
        anchor = track.axle[min(STOP_TICK, track.last)]
        camera.lookat[:] = (anchor[0] - .12, anchor[1], .25)
        camera.distance = 1.55
    else:
        x = track.pose[tick, 0]
        camera.lookat[:] = (np.clip(5.85, x - .9, x + .9), track.pose[tick, 1], .22)
        camera.distance = 3.6
    renderer.update_scene(data, camera=camera)
    view = renderer.scene
    view.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
    view.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
    rim_marks(mj, view, model, data, wheels)
    if not braking:
        ground_line(mj, view, (6., 0.), 0., .85)
    elif tick >= STOP_TICK:
        w, x, y, z = track.pose[STOP_TICK, 3:7]
        angle = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        ground_line(mj, view, track.axle[STOP_TICK], angle, .53)
    return Image.fromarray(renderer.render()).resize((WIDTH // 2 - 2, 494), Image.Resampling.BILINEAR)


def reading(track, tick, braking):
    if tick == track.last and (track.last < 600 or track.end_reason != "记录结束"):
        return f"{track.end_reason}，仿真 {tick / 50:.2f} 秒"
    if not braking:
        return f"车身位置  {track.pose[tick, 0]:.2f} m"
    if tick < STOP_TICK:
        return "驾驶请求已归零" if tick >= 525 else "正在减速"
    travel = (track.distance[tick] - track.distance[STOP_TICK]) * 1000
    if abs(travel) < .5:
        return "车身位移约 0 mm"
    return f"车身{'后退' if travel < 0 else '前移'}  {abs(travel):.0f} mm"


def render(out: Path, seed: int, old_path: Path, new_path: Path):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools102.mesa_video102 import probe_cpu_egl
    from tools112.surface_native import flatten

    movie = out / NAME
    if movie.exists():
        raise FileExistsError(movie)
    paths = (old_path, new_path)
    tracks = [load_track(path) for path in paths]
    np.testing.assert_array_equal(tracks[0].pose[0], tracks[1].pose[0])
    driver = probe_cpu_egl(out)
    title, label, caption = chinese_font(34, True), chinese_font(28), chinese_font(23)
    schedule = list(shots(tracks))
    frames = 0
    with TemporaryDirectory(prefix="d1-film113-") as directory:
        model = mj.MjModel.from_xml_path(str(scene(directory, "ridge")))
        flatten(model)
        model.vis.global_.offwidth, model.vis.global_.offheight = 422, 326
        model.vis.quality.offsamples, model.vis.quality.shadowsize = 0, 128
        data, camera = mj.MjData(model), mj.MjvCamera()
        wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
        with mj.Renderer(model, width=422, height=326) as renderer:
            proc = encoder(movie)
            try:
                for phase, tick, repeat in schedule:
                    braking = phase == "braking"
                    sheet = Image.new("RGB", (WIDTH, HEIGHT), "#16242d")
                    draw = ImageDraw.Draw(sheet)
                    heading = "减速与停车 · 观察前轮和地面标线" if braking else "等高网格巡航 · 观察车身何时通过 6 米线"
                    draw.text((18, 7), heading, font=title, fill="white")
                    for side, track in enumerate(tracks):
                        left = side * (WIDTH // 2)
                        draw.text((left + 16, 58), PANELS[side], font=label, fill="#e6eded")
                        if tick > track.last:
                            status = f"{track.end_reason}于 {track.last / 50:.2f} 秒"
                            draw.text((left + 30, 318), status, font=label, fill="#edc68d")
                            continue
                        view = draw_robot(mj, renderer, model, data, camera, wheels, track, tick, braking)
                        sheet.paste(view, (left, 96))
                        draw.text((left + 16, 611), reading(track, tick, braking), font=label, fill="#f6e1ad")
                    marker = "11.50 秒在本车前轴位置设置橙线" if braking else "橙线标出 6 米位置"
                    draw.text((18, 672), f"原生闭环存档　·　起点 {seed}　·　仿真 {tick / 50:.2f} 秒　·　{marker}",
                              font=caption, fill="#deeaed")
                    pixels = sheet.tobytes()
                    for _ in range(repeat):
                        proc.stdin.write(pixels)
                    frames += repeat
                proc.stdin.close()
                if proc.wait(timeout=90):
                    raise RuntimeError("video encoder failed")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    size = movie.stat().st_size
    if not 0 < size <= 5_000_000 or [digest(path) for path in paths] != [t.sha256 for t in tracks]:
        raise ValueError("film size or playback sources failed verification")
    return dict(filename=NAME, seed=seed, bytes=size, sha256=digest(movie), frames=frames,
                trace_sha256=[t.sha256 for t in tracks], last_ticks=[t.last for t in tracks],
                gate_ticks=[t.gate for t in tracks],
                shown_ticks={phase: [tick for name, tick, _ in schedule if name == phase]
                             for phase in ("cruise", "braking")},
                stream=verify_video(movie, frames), driver=driver, physics=0, cover_frames=0)
