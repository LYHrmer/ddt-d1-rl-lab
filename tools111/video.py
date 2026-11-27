"""Replay two registered physical trajectories at the same simulation time, with delivered lidar returns."""
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import ROOT, digest, read
from tools100.sensor100 import _grid, _wheel_cylinders
from tools109.video109 import FPS, HEIGHT, WIDTH, chinese_font, encoder, rim_marks, verify_video
from tools111.course import SEEDS, ground, install_heightfield, make_scene
from tools111.run import BASE, complete


def dots(mj, scene, points):
    for pos in points:
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_SPHERE, np.array([.009, 0., 0.]),
                        pos+[0., 0., .005], np.eye(3).ravel(), np.array([1., .70, .15, 1.], np.float32))
        geom.category = int(mj.mjtCatBit.mjCAT_DECOR)
        scene.ngeom += 1


def world_returns(mj, z):
    spec = read(ROOT/"docs/experiments100/pointcloud_density.json")["geometry"]
    directions = _grid(spec)[0]
    table = {}
    for i, k in enumerate(z["capture_tick"]):
        q = z["qpos"][k]
        matrix = np.empty(9)
        mj.mju_quat2Mat(matrix, q[3:7])
        local = np.repeat(np.array(spec["mounts_m"]), len(directions), axis=0)
        local += np.tile(directions, (2, 1))*z["range_m"][i, :, None]
        table[int(k)] = local[z["range_valid"][i]][::12] @ matrix.reshape(3, 3).T + q[:3]
    return table


def render(out):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools102.mesa_video102 import probe_cpu_egl
    driver = probe_cpu_egl(out)
    compound = any(BASE.glob("development_??/report.json"))
    stage = "development" if compound else "flat"
    folder, result, _ = complete(stage)
    seed = SEEDS[stage][0]
    arms = ("neither", "both") if compound else ("heading_off", "heading_on")
    names = ("雷达仅记录", "雷达调速和腿高") if compound else ("原厂角速度指令", "IMU 航向反馈")
    layout = stage if compound else "flat"
    name = "D1_1.5m每秒_复合地形_激光雷达调速与腿高_111.mp4" if compound else "D1_1.5m每秒_航向反馈平地对照_111.mp4"
    movie = out/name
    clips, frames = [], 0
    title, label, small = chinese_font(35, True), chinese_font(28), chinese_font(23)
    with ExitStack() as stack:
        directory = stack.enter_context(TemporaryDirectory(prefix="d1-111-movie-"))
        model = mj.MjModel.from_xml_path(str(make_scene(directory, layout)))
        install_heightfield(model, layout)
        model.vis.global_.offwidth, model.vis.global_.offheight = 512, 416
        renderer = stack.enter_context(mj.Renderer(model, width=512, height=416))
        data, cam = mj.MjData(model), mj.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 103., -14., 2.45
        wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
        traces, clouds, ends = [], [], []
        for arm in arms:
            row, = (r for r in result["episodes"] if r["seed"] == seed and r["arm"] == arm)
            trace = stack.enter_context(np.load(folder/f"{seed}_{arm}"/"trace.npz", allow_pickle=False))
            z = {key: trace[key] for key in ("qpos", "capture_tick", "range_m", "range_valid", "capture_used",
                                              "actor_vx", "base_cmd", "applied_lift_fk", "actor_wz")}
            traces.append(z)
            ends.append(len(z["qpos"])-1)
            clouds.append(world_returns(mj, z) if compound else {})
            clips.append(dict(stage=stage, seed=seed, arm=arm, trace_sha256=row["trace_sha256"],
                              end_tick=ends[-1], motion_passed=row["motion"]["passed"]))
        first_edge = row["effects"].get("first_front_edge_tick") if compound else None
        freeze = first_edge if first_edge is not None else min(ends[-1], 290)
        proc = encoder(movie)
        try:
            for k in range(max(ends)+1):
                x = traces[1]["qpos"][min(k, ends[1]), 0]
                slow = (2.5 <= x <= 6.35) or k >= max(ends)-30
                if not slow and k%2 and k not in (freeze, max(ends)):
                    continue
                frame = Image.new("RGB", (WIDTH, HEIGHT), "#16242d")
                draw = ImageDraw.Draw(frame)
                road = "上下坡、交错起伏、台阶" if compound else "平地"
                draw.text((18, 9), f"D1  ·  {road}  ·  目标 1.5 m/s", font=title, fill="white")
                for side, z in enumerate(traces):
                    index = min(k, ends[side])
                    tick = min(max(index-1, 0), ends[side]-1)
                    pose = z["qpos"][index]
                    data.qpos[:] = pose
                    mj.mj_fwdPosition(model, data)
                    cam.lookat[:] = (pose[0]+.27, pose[1], .20+float(ground(layout, *pose[:2])))
                    renderer.update_scene(data, camera=cam)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    rim_marks(mj, renderer.scene, model, data, wheels)
                    if compound:
                        dots(mj, renderer.scene, clouds[side].get(int(z["capture_used"][tick]), ()))
                    view = Image.fromarray(renderer.render()).resize((WIDTH//2-2, 516), Image.Resampling.BILINEAR)
                    left = side*(WIDTH//2)
                    frame.paste(view, (left, 97))
                    draw.text((left+16, 57), names[side], font=label, fill="#e5edef")
                    if compound:
                        drop = z["base_cmd"][tick]-z["actor_vx"][tick]
                        lift = max(z["applied_lift_fk"][tick, :2])*1000
                        message = f"雷达减速指令 {drop:.3f} m/s　前腿抬高目标 {lift:.0f} mm"
                    else:
                        message = f"指令 {z['actor_vx'][tick]:.2f} m/s　转向指令 {z['actor_wz'][tick]:+.2f} rad/s"
                    draw.text((left+14, 626), message, font=small, fill="#f0e4c5")
                phase = "前轮到达台沿 · 定格" if first_edge is not None and k == freeze else ("半速" if slow else "原速")
                source = f"同一起点 {seed}  ·  原生闭环轨迹回放  ·  {phase}"
                if compound:
                    source += "  ·  金点是最近送达的地面回波"
                draw.text((18, 676), source, font=small, fill="#d8e5eb")
                repeat = FPS if k in (freeze, max(ends)) else 1
                pixels = frame.tobytes()
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
    return dict(filename=name, bytes=movie.stat().st_size, sha256=digest(movie), frames=frames, clips=clips,
                stream=verify_video(movie, frames), driver=driver, physical_substeps=0)
