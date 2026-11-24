"""Run the registered whole-body reference and terrain gates in one-shot, single-CPU units."""
import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA


BASE=ROOT/"evidence/speed110"
PLAN=ROOT/"docs/speed110_proposal.md"
PLAN_SHA="786205ceac382f0293cf7cb5c010e3c1b7a8983f1bb2299142c31c8b6c1cac4d"
STAGES=("preflight","flat","calibration","development","blind","video")


def paths(stage,attempt):
    require(stage in STAGES and 1<=attempt<=99,"invalid speed110 stage or attempt")
    tag=f"{stage}_{attempt:02d}"
    return BASE/tag, BASE/("guard_"+tag), f"d1-speed110-{stage}-{attempt:02d}"


def bindings():
    return {p:digest(ROOT/p) for p in ("tools110/course.py","tools110/native.py","tools110/run.py",
            "training/configs/pilot03d.json","training/d1_rl/interface.py","training/d1_rl/env.py",
            "training/d1_rl/delay_env18.py","training/d1_rl/fast_checks.py")}


def complete(stage):
    from tools88.export88 import resources
    results=[]
    for report in sorted(BASE.glob(f"{stage}_??/report.json")):
        out,guard,_=paths(stage,int(report.parent.name[-2:]))
        if not (guard/"report.json").is_file() or read(guard/"report.json")["status"]!="COMPLETE":
            continue
        r=read(report)
        require(r["status"]=="COMPLETE" and r["stage"]==stage and r["sources"]==bindings()
                and r["plan"]==digest(PLAN)==PLAN_SHA, "unbound speed110 stage")
        for row in r.get("episodes",[]):
            p=out/f'{row["terrain"]}_{row["seed"]}'
            require(read(p/"report.json")==row and row["status"]=="COMPLETE"
                    and digest(p/"trace.npz")==row["trace_sha256"],"invalid speed110 episode")
        results.append((out,r,resources(guard)))
    require(len(results)==1,f"no unique complete {stage} stage")
    return results[0]


def locked_gain():
    from tools110.course import next_gain
    r=complete("flat")[1]["episodes"]
    gain=next_gain(r)
    if all(e["score"]["passed"] for e in r):
        return 1.
    if gain is not None:
        v=complete("calibration")[1]["episodes"]
        if len(v)==3 and all(e["score"]["passed"] and e["gain"]==gain for e in v):
            return gain
    return None


def eligible_terrains():
    from tools110.course import SEEDS,TERRAINS
    rows=complete("development")[1]["episodes"]
    return [t for t in TERRAINS if [r["seed"] for r in rows if r["terrain"]==t]==list(SEEDS["development"])
            and all(r["score"]["passed"] for r in rows if r["terrain"]==t)]


def preflight():
    from tempfile import TemporaryDirectory
    import mujoco as mj
    import numpy as np
    from tools110.course import TERRAINS,scene,set_floor,measured_terrain
    from tools110.native import load_policy,torque,POSE,SCALE,KP,KD,MODEL_SHA,YAML_SHA
    from training.d1_rl.interface import TORQUE_LIMIT,WHEEL_INDICES

    geometry={}
    for t in ("flat",*TERRAINS):
        with TemporaryDirectory(prefix="d1-110-check-") as d:
            m=mj.MjModel.from_xml_path(str(scene(d,t)))
            set_floor(m,t)
            geometry[t]=measured_terrain(m,t)
    s=load_policy()
    frame=np.zeros((1,57),np.float32)
    frame[0,5]=-1.
    history=np.tile(frame,(1,10,1))
    a=s.run(["nn_output"],{"nn_input0":frame,"nn_input1":history})[0][0]
    assert a.shape==(16,) and np.isfinite(a).all()
    worst=0.
    for v in (a, np.linspace(-4.,4.,16), -a):
        for dq in (np.zeros(16),np.linspace(-19.,22.,16)):
            q=np.linspace(-1.2,1.2,16)
            vendor=KP*(SCALE*v+POSE-q)-KD*dq
            vendor[WHEEL_INDICES]=KP[WHEEL_INDICES]*SCALE[WHEEL_INDICES]*v[WHEEL_INDICES]-KD[WHEEL_INDICES]*dq[WHEEL_INDICES]
            expected=np.clip(vendor,-TORQUE_LIMIT,TORQUE_LIMIT)
            err=float(np.max(np.abs(torque(v,q,dq)[0]-expected)))
            worst=max(worst,err)
    require(worst<1e-12,"vendor wheel torque conversion failed")
    return dict(terrain_rays=geometry, max_torque_error_nm=worst, model_sha256=MODEL_SHA, config_sha256=YAML_SHA,
                physics_steps=0,policy_forwards=1)


def worker(stage,attempt):
    from tools27.resource_guard27 import read_memory
    from tools110.course import SEEDS,TERRAINS,next_gain
    from tools110.native import episode,load_policy
    out,_,unit=paths(stage,attempt)
    scope=scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip()=="0::"+scope and len(os.sched_getaffinity(0))==1
            and os.getpriority(os.PRIO_PROCESS,0)==10 and all(os.environ.get(k)=="1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES")=="" and read_memory(scope)["effective_available_bytes"]>=4<<30
            and digest(PLAN)==PLAN_SHA,"speed110 worker outside the admitted unit")
    out.mkdir(exist_ok=False)
    head=dict(stage=stage,plan=PLAN_SHA,sources=bindings())
    write(out/"inputs.json",head)
    try:
        if stage=="preflight":
            payload=preflight()
        elif stage=="video":
            from tools110.video import render
            payload=render(out)
        else:
            complete("preflight")
            gain=1.
            terrains=("flat",)
            if stage=="calibration":
                first=complete("flat")[1]["episodes"]
                gain=next_gain(first)
                require(gain is not None and not all(r["score"]["passed"] for r in first),"calibration gate closed")
            if stage in ("development","blind"):
                gain=locked_gain()
                require(gain is not None,"flat qualification gate closed")
                terrains=TERRAINS if stage=="development" else eligible_terrains()
                require(bool(terrains),"no terrain qualified for blind tests")
            session=load_policy()
            rows=[episode(out/f"{t}_{seed}",seed,t,gain,session) for t in terrains for seed in SEEDS[stage]]
            payload=dict(episodes=rows,gain=gain,terrains=terrains)
        write(out/"report.json",dict(status="COMPLETE",**head,**payload))
    except BaseException as e:
        write(out/"partial.json",dict(status="PARTIAL",error=repr(e),**head))
        raise


def launch(stage,attempt):
    out,guard,unit=paths(stage,attempt)
    require(digest(PLAN)==PLAN_SHA and digest(PRESSURE)==GUARD_SHA,"speed110 registration changed")
    BASE.mkdir(parents=True,exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not out.exists() and not guard.exists(),"preserve the existing attempt")
        require(not any(BASE.glob(f"{stage}_??/*/reservation.json")),"native budget was already used")
        env=dict.fromkeys(THREADS,"1")
        env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))),
                   CUDA_VISIBLE_DEVICES="",MUJOCO_GL="disable",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1",PYTHONHASHSEED="0")
        if stage=="video":
            from tools102.mesa_video102 import GL_ENV
            env.update(GL_ENV)
        cmd=["systemd-run","--user","--wait","--collect","--pipe","--slice=app.slice",f"--unit={unit}",
             "-p","Nice=10","-p","CPUWeight=25","-p","OOMPolicy=kill","-p","MemoryHigh=2G","-p","MemoryMax=3G",
             "-p","MemorySwapMax=0",f"--working-directory={ROOT}",*(f"--setenv={k}={v}" for k,v in env.items()),
             sys.executable,"-B",str(PRESSURE),"--scope",scope_for(unit),"--output",str(guard),"--timeout","900",
             "--",sys.executable,"-B","-m","tools110.run","worker",stage,"--attempt",str(attempt)]
        subprocess.run(cmd,check=True)
        require(read(guard/"report.json")["status"]=="COMPLETE" and (out/"report.json").is_file(),"invalid receipt")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("worker","launch")); p.add_argument("stage",choices=STAGES)
    p.add_argument("--attempt",type=int,default=1)
    args=p.parse_args()
    {"worker":worker,"launch":launch}[args.mode](args.stage,args.attempt)
