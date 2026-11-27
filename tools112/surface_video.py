"""Replay the first matched plane/heightfield pair at the same native times."""
from contextlib import ExitStack
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import digest
from tools100.sensor100 import _wheel_cylinders
from tools109.video109 import FPS, HEIGHT, WIDTH, chinese_font, encoder, rim_marks, verify_video
from tools110.course import scene
from tools112.run_surface import recorded
from tools112.surface import SEEDS, reference


NAME = "D1_等高地面_巡航失速与停车后退_112.mp4"


def ground_line(mj, view, xy, yaw, half_width):
    marker = view.geoms[view.ngeom]
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    mj.mjv_initGeom(marker,mj.mjtGeom.mjGEOM_BOX,np.array([.008,half_width,.003]),
                    np.array([*xy,.004]),rotation.ravel(),np.array([1.,.55,.08,1.],np.float32))
    marker.category = int(mj.mjtCatBit.mjCAT_DECOR)
    view.ngeom += 1


def render(out):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools102.mesa_video102 import probe_cpu_egl
    from tools112.surface_native import flatten
    driver = probe_cpu_egl(out)
    seed = SEEDS[0]
    plane_path, plane = reference(seed)
    field_folder, _, field_record = recorded(str(seed))
    field = field_record["episode"]
    cases = ((plane_path,plane,"解析平面"),(field_folder/"trace.npz",field,"等高网格地面"))
    gates = [int(np.ceil(r["motion"]["gate_times_s"][-1]*50.)) for _,r,_ in cases]
    cruise = range(min(gates)-32,max(gates)+9)
    brake = range(520,601)
    title, label, caption = chinese_font(35,True), chinese_font(28), chinese_font(24)
    frames = 0
    movie = out/NAME
    with ExitStack() as stack:
        panels = []
        for side,(path,row,name) in enumerate(cases):
            if digest(path) != row["trace_sha256"]:
                raise ValueError("video source differs from the paired measurement")
            folder = stack.enter_context(TemporaryDirectory(prefix="d1-surface112-film-"))
            model = mj.MjModel.from_xml_path(str(scene(folder,"ridge" if side else "flat")))
            if side:
                flatten(model)
            model.vis.global_.offwidth, model.vis.global_.offheight = 422, 326
            renderer = stack.enter_context(mj.Renderer(model,width=422,height=326))
            with np.load(path,allow_pickle=False) as z:
                pose = z["qpos"]
                axle = z["wheels_xy"][575,:2].mean(axis=0)
                yaw = z["imu_yaw"]
                heading = np.column_stack((np.cos(yaw),np.sin(yaw)))
                ds = (np.diff(z["xy"],axis=0)*heading).sum(axis=1)
                distance = np.r_[0.,np.cumsum(ds)]
            panels.append((name,model,mj.MjData(model),renderer,mj.MjvCamera(),
                           _wheel_cylinders(model,mj.mjtGeom.mjGEOM_CYLINDER),pose,axle,yaw[575],distance))
        proc = encoder(movie)
        try:
            for segment, ticks in (("cruise",cruise),("braking",brake)):
                for k in ticks:
                    cruise_view = segment == "cruise"
                    frame = Image.new("RGB",(WIDTH,HEIGHT),"#16242d")
                    draw = ImageDraw.Draw(frame)
                    text = "同一速度指令 · 通过 6 米线" if cruise_view else "停车 · 观察前轮相对地面标线的位置"
                    draw.text((18,8),text,font=title,fill="white")
                    for side,(name,model,data,renderer,cam,wheels,pose,axle,yaw,dist) in enumerate(panels):
                        data.qpos[:] = pose[k]
                        mj.mj_fwdPosition(model,data)
                        cam.azimuth, cam.elevation = 98., -15.
                        cam.distance = 4.35 if cruise_view else 1.55
                        cam.lookat[:] = (5.85,0.,.22) if cruise_view else (axle[0]-.12,axle[1],.25)
                        renderer.update_scene(data,camera=cam)
                        view = renderer.scene
                        view.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                        view.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                        rim_marks(mj,view,model,data,wheels)
                        if cruise_view:
                            ground_line(mj,view,(6.,0.),0.,.85)
                        elif k >= 575:
                            ground_line(mj,view,axle,yaw,.53)
                        left = side*(WIDTH//2)
                        raster = Image.fromarray(renderer.render()).resize((WIDTH//2-2,494),Image.Resampling.BILINEAR)
                        frame.paste(raster,(left,97))
                        draw.text((left+17,57),name,font=label,fill="#e2ebee")
                        message = f"车身位置 {pose[k,0]:.2f} m" if cruise_view else (
                            f"末半秒后退 {1000*(dist[575]-dist[k]):.0f} mm" if k >= 575 else
                            ("前进指令已归零" if k >= 525 else "正在制动"))
                        draw.text((left+16,612),message,font=label,fill="#f5e5bf")
                    guide = "橙线固定在 6 米处" if cruise_view else "橙线标出本车 11.50 秒的前轴位置"
                    draw.text((18,673),f"起点 {seed}  ·  仿真 {k/50:.2f} 秒  ·  {guide}  ·  原生轨迹慢放",
                              font=caption,fill="#dce8ea")
                    repeat = FPS if k in (gates[0],575,600) else 2 if k>575 else 1
                    pixels = frame.tobytes()
                    for _ in range(repeat):
                        proc.stdin.write(pixels)
                    frames += repeat
            proc.stdin.close()
            if proc.wait(timeout=90):
                raise RuntimeError("encoder returned nonzero")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return dict(filename=NAME,seed=seed,bytes=movie.stat().st_size,sha256=digest(movie),frames=frames,
                shown_ticks=[[cruise.start,cruise.stop-1],[brake.start,brake.stop-1]],driver=driver,
                trace_sha256=[r["trace_sha256"] for _,r,_ in cases],stream=verify_video(movie,frames),physics=0)
