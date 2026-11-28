"""Show the registered compound-route pair at common native times, using recorded poses."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import digest, read, require
from tools108.video108 import FPS, HEIGHT, WIDTH
from tools111.course import step_bounds
from tools113.video import STOP_TICK, load_track
from tools114.parking_video import axle_motion_mm, marker
from tools116.score import SEEDS, reference


NAME = "compound116.mp4"
DESKTOP_NAME = "D1_复合地形_雷达调速与编码器补偿_116.mp4"
PANELS = ("左 · 雷达调速与抬腿", "右 · 相同雷达控制，加轮速编码器补偿")
DRIVER_ZERO = 525


def first_tick(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def sources(native_dir):
    seed = SEEDS[0]
    old, old_row = reference(seed)
    native_dir = Path(native_dir)
    candidate, receipt = native_dir/"trace.npz", native_dir/"report.json"
    report = read(receipt)
    row = report["result"]
    require(seed == 111101 and report["status"] == row["status"] == "COMPLETE" and row["seed"] == seed
            and row["trim"] == "brake_aware_after_lidar" and row["reference_sha256"] == old_row["trace_sha256"]
            and row["trace_sha256"] == report["files"]["trace.npz"]
            and all(row[key] == old_row[key] for key in ("layout","arm","gain")), "wrong compound video pair")
    require(not any((native_dir/name).exists() for name in ("partial.npz","partial_counts.json","worker_partial.json")),
            "partial video source")
    tracks, signals, events = [], [], []
    for path, result in ((old,old_row),(candidate,row)):
        track = load_track(path)
        require(track.sha256 == result["trace_sha256"] and track.last == result["motion"]["ticks"]
                and (track.last == 600 or track.end_reason != "记录结束"), "unexplained video endpoint or changed trace")
        with np.load(path,allow_pickle=False) as archive:
            z = {key:archive[key] for key in ("capture_tick","capture_used","range_m","range_valid","applied_lift_fk",
                                             "front_platform_touch")}
            z["qpos"] = track.pose
            wheels = archive["wheels_xy"]
            rear = first_tick(np.min(wheels[:,2:,0],axis=1) > step_bounds("development")[1])
        contact = first_tick(z["front_platform_touch"])
        tracks.append(track)
        signals.append(z)
        events.append(dict(crest=first_tick(track.pose[:,0] >= 3.25),
                           ripple=first_tick(track.pose[:,0] >= 4.35),
                           front_contact=None if contact is None else contact+1, rear_clear=rear))
    equal = row["paired_prefix"]["equal_state_endpoints"]
    require(1 <= equal <= tracks[1].last+1, "invalid matched prefix")
    np.testing.assert_array_equal(tracks[0].pose[:equal],tracks[1].pose[:equal])
    info = dict(seed=seed, trace_sha256=[track.sha256 for track in tracks], native_report_sha256=digest(receipt),
                last_ticks=[track.last for track in tracks], end_reasons=[track.end_reason for track in tracks],
                events=events, renderer_sha256=digest(Path(__file__)))
    return tracks,signals,info


def timeline(ends, events):
    """Select both arms' measured terrain events and show the same tick on each side."""
    require(len(ends) == len(events) == 2 and all(0 < n <= 600 for n in ends), "invalid compound replay lengths")
    ticks, holds, contacts = set(), set(), []
    last = max(ends)
    def window(start, stop, stride):
        start, stop = max(0,start), min(last,stop)
        if start <= stop:
            ticks.update(range(start,stop+1,stride))
            ticks.update((start,stop))

    for end, landmarks in zip(ends,events):
        for name,before,after in (("crest",26,27),("ripple",9,8),("front_contact",17,19),("rear_clear",8,11)):
            k = landmarks[name]
            if k is not None:
                require(0 <= k <= end, "terrain event after the end of its source")
                window(k-before,min(end,k+after),3)
                ticks.add(k)
                if name == "front_contact":
                    contacts.append(k)
                    holds.add(k)
        if end < 600:
            window(end-6,end,2)
            holds.add(end)
    window(500,STOP_TICK-1,3)
    window(STOP_TICK,last,2)
    for k in (DRIVER_ZERO,STOP_TICK,last):
        if k <= last:
            ticks.add(k)
            holds.add(k)
    first_step = min(contacts, default=600)-17
    order = sorted(ticks)
    shots, frames = [], 0
    for i,k in enumerate(order):
        phase = "tail" if k >= STOP_TICK else "braking" if k >= 500 else "step" if k >= first_step else "slope"
        gap = min(3,order[i+1]-k) if i+1 < len(order) else 1
        # Half speed on the slope, quarter speed around the step and in the final half-second.
        repeat = gap*(2 if phase in ("step","tail") else 1)
        if k in holds:
            repeat += 2*FPS if k == last else FPS
        shots.append(dict(tick=k,phase=phase,first_frame=frames,frames=repeat))
        frames += repeat
    return shots


def cameras(tracks, shots, markers):
    views = {}
    for phase in dict.fromkeys(s["phase"] for s in shots):
        ticks = [s["tick"] for s in shots if s["phase"] == phase]
        if phase == "tail":
            # Equal magnification, each camera fixed at its own measured 11.50 s axle.
            views[phase] = [None if mark is None else dict(
                lookat_m=[mark["xy_m"][0]-.04, mark["xy_m"][1], .13],
                distance_m=1.05,azimuth_deg=98.,elevation_deg=-16.) for mark in markers]
        else:
            points = np.asarray([track.pose[k,:2] for track in tracks for k in ticks if k <= track.last])
            low, high = points.min(axis=0),points.max(axis=0)
            cx,cy = (low+high)/2.
            view = dict(lookat_m=[float(cx),float(cy),.26],distance_m=float(1.50+np.linalg.norm(high-low)),
                        azimuth_deg=97.,elevation_deg=-17.)
            views[phase] = [view,view]
    return views


def caption(track, event, k, phase, lift, mark):
    if k >= track.last and track.end_reason != "记录结束":
        return f"{track.end_reason}于 {track.last/50:.2f} 秒"
    if phase == "tail":
        travel = axle_motion_mm(track,k,mark)
        if abs(travel) < .5:
            return "前轴仍在地面标线处"
        return f"前轴相对橙线{'后移' if travel < 0 else '前移'} {abs(travel):.0f} mm"
    if phase == "braking":
        return "前进指令已归零" if k >= DRIVER_ZERO else "驾驶员正在减速"
    if event["front_contact"] == k:
        return "前轮接触台阶 · 定格"
    if event["rear_clear"] is not None and k >= event["rear_clear"]:
        return "两后轮轮心已越过台阶后沿"
    if event["front_contact"] is not None and k >= event["front_contact"]:
        return "前轮已接触台阶"
    if lift > .005:
        return "前轮抬高目标已送到伺服"
    x = track.pose[k,0]
    for end, terrain in ((2.5,"接近上坡"),(3.25,"上坡"),(4.,"下坡"),(5.,"经过交错起伏"),
                         (5.12,"接近台阶"),(5.88,"台阶区域")):
        if x < end:
            return terrain
    return "车身已在台阶后方"


def encode(movie, tracks, signals, info, shots, views, marks, temp):
    from PIL import Image, ImageDraw
    import mujoco as mj
    from tools87.video87 import chinese_font
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    from tools108.video108 import encoder, rim_marks
    from tools111.course import install_heightfield, make_scene
    from tools111.video import dots, world_returns
    from tools112.surface_video import ground_line

    driver = probe_cpu_egl(temp)
    model = mj.MjModel.from_xml_path(str(make_scene(temp,"development")))
    install_heightfield(model,"development")
    model.vis.global_.offwidth,model.vis.global_.offheight = 432,336
    model.vis.quality.offsamples,model.vis.quality.shadowsize = 0,128
    data,camera = mj.MjData(model),mj.MjvCamera()
    wheels = _wheel_cylinders(model,mj.mjtGeom.mjGEOM_CYLINDER)
    clouds = [world_returns(mj,z) for z in signals]
    title, label, foot = chinese_font(34,True), chinese_font(26), chinese_font(22)
    titles = dict(slope="上下坡与交错起伏",step="30 mm 台阶",braking="驾驶指令减小与归零",tail="停车末半秒 · 看前轮与地面标线")
    with mj.Renderer(model,width=432,height=336) as renderer:
        proc = encoder(movie)
        try:
            for shot in shots:
                k,phase = shot["tick"],shot["phase"]
                sheet = Image.new("RGB",(WIDTH,HEIGHT),"#17252e")
                draw = ImageDraw.Draw(sheet)
                draw.text((16,8),titles[phase],font=title,fill="#f3f5ef")
                draw.text((WIDTH-18,14),f"仿真 {k/50:.2f} 秒",anchor="ra",font=label,fill="#e2edeb")
                for i,(track,z) in enumerate(zip(tracks,signals)):
                    left = i*WIDTH//2
                    draw.text((left+15,58),PANELS[i],font=label,fill="#e2edeb")
                    if k > track.last:
                        draw.text((left+35,315),f"{track.end_reason}于 {track.last/50:.2f} 秒",font=label,fill="#f5d491")
                        continue
                    view = views[phase][i]
                    camera.lookat[:] = view["lookat_m"]
                    camera.distance,camera.azimuth,camera.elevation = view["distance_m"],view["azimuth_deg"],view["elevation_deg"]
                    data.qpos[:] = track.pose[k]
                    mj.mj_fwdPosition(model,data)
                    np.testing.assert_allclose(data.geom_xpos[wheels[:2],:2].mean(axis=0),track.axle[k],atol=1e-8,rtol=0.)
                    renderer.update_scene(data,camera=camera)
                    scene = renderer.scene
                    scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    rim_marks(mj,scene,model,data,wheels)
                    # A pose at boundary k is the result of interval k-1.
                    tick = max(0,k-1)
                    if phase in ("slope","step"):
                        dots(mj,scene,clouds[i].get(int(z["capture_used"][tick]),()))
                    if phase == "tail":
                        ground_line(mj,scene,marks[i]["xy_m"],marks[i]["yaw_rad"],.60)
                    panel = Image.fromarray(renderer.render()).resize((WIDTH//2-2,496),Image.Resampling.BILINEAR)
                    sheet.paste(panel,(left,96))
                    text = caption(track,info["events"][i],k,phase,z["applied_lift_fk"][tick,:2].max(),marks[i])
                    draw.text((left+15,611),text,font=label,fill="#f6dfa8")
                note = ("等比例近景 · 橙线分别标出 11.50 秒的前轴位置" if phase == "tail" else
                        "两侧共用地面坐标与固定视角" if phase == "braking" else "两侧同一地面坐标 · 金点是最近送达的雷达回波")
                draw.text((16,672),f"闭环仿真存档回放 · 起点 {info['seed']}　　{note}",font=foot,fill="#d7e4e5")
                pixels = sheet.tobytes()
                for _ in range(shot["frames"]):
                    proc.stdin.write(pixels)
            proc.stdin.close()
            if proc.wait(timeout=90):
                raise RuntimeError("compound movie encoder failed")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return driver


def render(out, native_dir):
    from tools108.video108 import verify_video

    out,native_dir = Path(out),Path(native_dir)
    movie = out/NAME
    require(out.is_dir() and not movie.exists(), "compound film requires a fresh output")
    tracks,signals,info = sources(native_dir)
    shots = timeline(info["last_ticks"],info["events"])
    marks = [marker(t) for t in tracks]
    views = cameras(tracks,shots,marks)
    with TemporaryDirectory(prefix="d1-compound116-video-") as temp:
        driver = encode(movie,tracks,signals,info,shots,views,marks,Path(temp))
    size = movie.stat().st_size
    require(0 < size <= 5_000_000 and info == sources(native_dir)[2], "film size or sources changed")
    frames = sum(s["frames"] for s in shots)
    stream = verify_video(movie,frames)
    return dict(status="COMPLETE",filename=NAME,desktop_filename=DESKTOP_NAME,**info,sha256=digest(movie),
                frames=frames,fps=FPS,bytes=size,duration_s=float(stream["duration"]),stream=stream,driver=driver,
                shots=shots,cameras=views,ground_markers=marks,panels=list(PANELS),physics=0,cover_frames=0)
