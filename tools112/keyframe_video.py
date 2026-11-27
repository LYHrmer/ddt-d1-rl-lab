"""Export a separate keyframe film from complete native trajectories after the first render was interrupted."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools112.surface import BASE


OUTPUT = BASE/"video_keyframes"
INITIAL = BASE/"guard_video_01/report.json"


def worker(unit):
    from tools27.resource_guard27 import read_memory
    from tools112.surface_video import render
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope
            and os.getpriority(os.PRIO_PROCESS,0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4<<30, "render worker outside admitted unit")
    OUTPUT.mkdir(exist_ok=False)
    header = dict(unit=unit,initial_guard_sha256=digest(INITIAL),renderer_sha256=digest(ROOT/"tools112/surface_video.py"))
    write(OUTPUT/"reservation.json",header)
    try:
        film = render(OUTPUT)
        write(OUTPUT/"report.json",dict(status="COMPLETE",film=film,**header))
    except BaseException as error:
        write(OUTPUT/"partial.json",dict(status="PARTIAL",error=repr(error),**header))
        raise


def completed():
    report = read(OUTPUT/"report.json")
    guard = BASE/("guard_"+report["unit"])
    require(report["status"] == read(guard/"report.json")["status"] == "COMPLETE"
            and report["initial_guard_sha256"] == digest(INITIAL)
            and report["renderer_sha256"] == digest(ROOT/"tools112/surface_video.py"),"unbound video receipt")
    return OUTPUT,guard,report


def launch():
    from tools102.mesa_video102 import GL_ENV
    from tools111.remaining import ready, wait_for_admission
    initial = read(INITIAL)
    require(digest(PRESSURE) == GUARD_SHA and initial["status"] == "INCOMPLETE"
            and initial["owned_cgroup_children_empty"], "initial render is not terminal")
    env = dict.fromkeys(THREADS,"1")
    env.update(GL_ENV,CUDA_VISIBLE_DEVICES="",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))))
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (OUTPUT/"report.json").is_file():
            completed()
            return
        require(not OUTPUT.exists(),"preserve the independent export")
        for attempt in range(1,100):
            unit = f"d1-surface112-keyframes-{attempt:02d}"
            guard = BASE/("guard_"+unit)
            if guard.exists():
                require(not read(guard/"report.json")["child_started"],"preserve a started guard")
                continue
            if not ready():
                wait_for_admission()
            cmd = ["systemd-run","--user","--wait","--collect","--pipe","--slice=app.slice",f"--unit={unit}",
                   "-p","Nice=10","-p","CPUWeight=25","-p","OOMPolicy=kill","-p","MemoryHigh=2G",
                   "-p","MemoryMax=3G","-p","MemorySwapMax=0",f"--working-directory={ROOT}",
                   *(f"--setenv={k}={v}" for k,v in env.items()),sys.executable,"-B",str(PRESSURE),
                   "--scope",scope_for(unit),"--output",str(guard),"--timeout","300","--",
                   sys.executable,"-B","-m","tools112.keyframe_video","--worker",unit]
            status = subprocess.run(cmd).returncode
            if status == 0:
                completed()
                return
            receipt = read(guard/"report.json")
            require(not receipt["child_started"] and not OUTPUT.exists(),f"preserve ended render: {receipt['error']}")
            require(any(k in str(receipt["error"]) for k in ("effective memory","PSI","admission not sustained")),
                    "render guard failed outside admission")
        raise RuntimeError("no unused render admission")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        for key in THREADS:
            os.environ[key] = "1"
        launch()
