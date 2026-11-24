"""Software-rendered replay of measured D1 poses, with course gates at their world coordinates."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools109.video109 import FPS, HEIGHT, WIDTH, chinese_font, encoder, rim_marks, verify_video
from tools110.course import SEEDS, TERRAINS, height, scene, set_floor
from tools110.run import BASE, complete


LABEL = {"flat": "平地资格", "ridge": "连续 4° 上下坡", "ripple": "左右错相起伏", "step": "30 mm 台阶"}


def marks(mj, scn, terrain, robot_x):
    for pos in range(max(-1,int(robot_x)-3),min(16,int(robot_x)+5)):
        for side in (-1.,1.):
            mark=scn.geoms[scn.ngeom]
            y=side*.78
            h=float(height(terrain,pos,y))
            color=[.98,.72,.22,1.] if pos in (3,4,5,6) else [.68,.76,.76,1.]
            mj.mjv_initGeom(mark,mj.mjtGeom.mjGEOM_BOX,np.array([.014,.14,.003]),
                            np.array([pos,y,h+.006]),np.eye(3).ravel(),np.array(color,np.float32))
            mark.category=int(mj.mjtCatBit.mjCAT_DECOR)
            scn.ngeom+=1


def render(out):
    import mujoco as mj
    from PIL import Image, ImageDraw
    from tools100.sensor100 import _wheel_cylinders
    from tools102.mesa_video102 import probe_cpu_egl
    driver=probe_cpu_egl(out)
    if any(BASE.glob("development_??/report.json")):
        dev=complete("development")
        blind=complete("blind") if any(BASE.glob("blind_??/report.json")) else None
        chosen=[(blind if blind and t in blind[1]["terrains"] else dev,t) for t in TERRAINS]
    else:
        stage="calibration" if any(BASE.glob("calibration_??/report.json")) else "flat"
        chosen=[(complete(stage),"flat")]
    movie=out/"D1_1.5m每秒_原厂全身策略_实测回放_110.mp4"
    proc=encoder(movie)
    frames=0
    clips=[]
    head,small=chinese_font(34,True),chinese_font(25)
    try:
        for (folder,report,_),terrain in chosen:
            seed=SEEDS[report["stage"]][0]
            row,=[r for r in report["episodes"] if r["terrain"]==terrain and r["seed"]==seed]
            with np.load(folder/f"{terrain}_{seed}"/"trace.npz",allow_pickle=False) as z:
                pose=z["qpos"].copy()
            end=len(pose)-1
            clips.append(dict(seed=seed,terrain=terrain,stage=report["stage"],end_tick=end,first_frame=frames,
                              source_sha256=row["trace_sha256"],passed=row["score"]["passed"]))
            with TemporaryDirectory(prefix="d1-110-replay-") as d:
                m=mj.MjModel.from_xml_path(str(scene(d,terrain)))
                set_floor(m,terrain)
                m.vis.global_.offwidth,m.vis.global_.offheight=640,308
                data,cam=mj.MjData(m),mj.MjvCamera()
                wheel=_wheel_cylinders(m,mj.mjtGeom.mjGEOM_CYLINDER)
                cam.azimuth,cam.elevation,cam.distance=107.,-15.,2.65
                with mj.Renderer(m,width=640,height=308) as rend:
                    for k in range(0,end+1):
                        x,y=pose[k,:2]
                        slow=(end-k<30 or (terrain=="step" and 3.75<x<4.5) or (terrain!="flat" and 2.8<x<3.3))
                        if not slow and k%2 and k!=end:
                            continue
                        data.qpos[:]=pose[k]
                        mj.mj_fwdPosition(m,data)
                        cam.lookat[:]=(x+.28,y,float(height(terrain,x,y))+.24)
                        rend.update_scene(data,camera=cam)
                        rend.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
                        rend.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=0
                        marks(mj,rend.scene,terrain,x)
                        rim_marks(mj,rend.scene,m,data,wheel)
                        pic=Image.fromarray(rend.render()).resize((WIDTH,616),Image.Resampling.BILINEAR)
                        frame=Image.new("RGB",(WIDTH,HEIGHT),"#17252d")
                        frame.paste(pic,(0,60))
                        draw=ImageDraw.Draw(frame)
                        draw.text((22,8),f"D1 原厂全身策略  ·  {LABEL[terrain]}  ·  指令 1.5 m/s",font=head,fill="#f6f8f8")
                        source="盲测" if report["stage"]=="blind" else "开发"
                        state="进入测速区" if x<3. else "穿越测速区" if x<6. else "驶出测速区"
                        reason="达到本段验收" if row["score"]["passed"] else "本段未通过验收"
                        caption=f"{source}复位 {seed}  ·  实测回放  ·  {state}  ·  {'半速' if slow else '原速'}  ·  黄线间距 1 米"
                        if k==end:
                            caption=f"{source}复位 {seed}  ·  终态定格  ·  {reason}"
                        draw.text((22,680),caption,font=small,fill="#e6eded")
                        repeat=FPS if k==end else 1
                        pixels=frame.tobytes()
                        for _ in range(repeat):
                            proc.stdin.write(pixels)
                        frames+=repeat
        proc.stdin.close()
        if proc.wait(timeout=90):
            raise RuntimeError("video encoder failed")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    from tools94.common94 import digest
    return dict(filename=movie.name,sha256=digest(movie),bytes=movie.stat().st_size,clips=clips,frames=frames,
                stream=verify_video(movie,frames),driver=driver,physics_steps=0)
