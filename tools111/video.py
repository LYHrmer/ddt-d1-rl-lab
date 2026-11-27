"""Replay two registered physical trajectories at the same simulation time, with delivered lidar returns."""
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import ROOT, digest, read
from tools100.sensor100 import _grid, _wheel_cylinders
from tools109.video109 import FPS, HEIGHT, WIDTH, chinese_font, encoder, rim_marks, verify_video
from tools111.course import SEEDS, ground, install_heightfield, make_scene
from tools111.results import audit_for_video
from tools111.run import BASE


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


def segment(x):
    for end, name in ((2.5,"坡前"),(3.25,"上坡"),(4.,"下坡"),(5.,"错相起伏"),(5.12,"台阶前"),(5.88,"台面")):
        if x < end:
            return name
    return "驶出台阶"


def render(out):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools102.mesa_video102 import probe_cpu_egl
    driver = probe_cpu_egl(out)
    validated_sha = audit_for_video(out)
    cases = read(out/"validated.json")["cases"]
    layout = "development"
    seed = SEEDS[layout][0]
    arms = ("neither", "both")
    names = ("左 · 雷达仅记录", "右 · 雷达调速和腿高")
    name = "D1_1.5m每秒_复合地形_激光雷达调速与腿高_111.mp4"
    movie = out/name
    clips, frames = [], 0
    first_shown, last_crossing = 170, None
    title, label, small = chinese_font(35, True), chinese_font(28), chinese_font(23)
    with ExitStack() as stack:
        directory = stack.enter_context(TemporaryDirectory(prefix="d1-111-movie-"))
        model = mj.MjModel.from_xml_path(str(make_scene(directory, layout)))
        install_heightfield(model, layout)
        model.vis.global_.offwidth, model.vis.global_.offheight = 432, 352
        renderer = stack.enter_context(mj.Renderer(model, width=432, height=352))
        data, cam = mj.MjData(model), mj.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 103., -14., 2.45
        wheels = _wheel_cylinders(model, mj.mjtGeom.mjGEOM_CYLINDER)
        traces, clouds, ends = [], [], []
        for arm in arms:
            case, = (r for r in cases if r["seed"] == seed and r["arm"] == arm and r["case_status"] == "COMPLETE")
            row = case["episode"]
            trace = stack.enter_context(np.load(BASE/case["unit"]/f"{seed}_{arm}"/"trace.npz", allow_pickle=False))
            z = {key: trace[key] for key in ("qpos", "capture_tick", "range_m", "range_valid", "capture_used",
                                              "actor_vx", "base_cmd", "applied_lift_fk", "actor_wz")}
            traces.append(z)
            ends.append(len(z["qpos"])-1)
            clouds.append(world_returns(mj, z))
            clips.append(dict(stage=layout, seed=seed, arm=arm, trace_sha256=row["trace_sha256"],
                              unit_status=case["unit_status"], end_tick=ends[-1],
                              speed_m_s=row["motion"]["speed_m_s"], stop_tail_max_m_s=row["motion"]["stop_tail_max_m_s"],
                              motion_passed=row["motion"]["passed"], front_edge_tick=row["effects"]["first_front_edge_tick"]))
        freeze = clips[1]["front_edge_tick"]
        proc = encoder(movie)
        try:
            for k in range(max(ends)+1):
                x = traces[1]["qpos"][min(k, ends[1]), 0]
                if k < first_shown or (x > 6.6 and k < max(ends)-25):
                    continue
                slow = 2.5 <= x <= 6.35 or k >= max(ends)-25
                if not slow and k%2 and k not in (freeze, max(ends)):
                    continue
                if x <= 6.6:
                    last_crossing = k
                frame = Image.new("RGB", (WIDTH, HEIGHT), "#16242d")
                draw = ImageDraw.Draw(frame)
                draw.text((18, 9), "D1  ·  上下坡、交错起伏、台阶  ·  目标 1.5 m/s", font=title, fill="white")
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
                    dots(mj, renderer.scene, clouds[side].get(int(z["capture_used"][tick]), ()))
                    view = Image.fromarray(renderer.render()).resize((WIDTH//2-2, 516), Image.Resampling.BILINEAR)
                    left = side*(WIDTH//2)
                    frame.paste(view, (left, 97))
                    terrain = "停车" if k >= max(ends)-25 else segment(pose[0])
                    draw.text((left+16, 57), names[side]+"  ·  "+terrain, font=label, fill="#e5edef")
                    if k == max(ends):
                        message = f"中央实测 {clips[side]['speed_m_s']:.3f} m/s　停车峰值 {clips[side]['stop_tail_max_m_s']:.3f} m/s"
                    else:
                        drop = z["base_cmd"][tick]-z["actor_vx"][tick]
                        lift = max(z["applied_lift_fk"][tick, :2])*1000
                        message = f"雷达减速 {drop:.3f} m/s　前轮抬高目标 {lift:.0f} mm"
                    draw.text((left+14, 626), message, font=small, fill="#f0e4c5")
                phase = "右车前轮到台沿 · 定格" if k == freeze else ("半速" if slow else "原速")
                source = f"起点 {seed}  ·  仿真 {k/50:.2f} 秒  ·  {phase}  ·  金点是最近送达的回波"
                if k == max(ends):
                    source = "两车均驶过台阶。中央三米均速和停车余速均未达标。"
                draw.text((18, 676), source, font=small, fill="#d8e5eb")
                repeat = 2*FPS if k == max(ends) else FPS if k == freeze else 1
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
                shown_tick_ranges=[[first_shown,last_crossing],[max(ends)-25,max(ends)]],
                stream=verify_video(movie, frames), driver=driver, physical_substeps=0, validated_sha256=validated_sha)
