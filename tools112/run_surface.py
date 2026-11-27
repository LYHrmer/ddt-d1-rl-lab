"""Launch the registered three-case surface diagnosis in independent one-shot CPU units."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
if __name__ == "__main__" and sys.argv[1:2] == ["launch"]:
    for thread_variable in THREADS:
        os.environ[thread_variable] = "1"

from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import ready, wait_for_admission
from tools112.surface import BASE, PLAN_SHA, REFERENCE_SHA, SEEDS, bindings


CASES = tuple(str(s) for s in SEEDS)
TASKS = ("survey",*CASES,"video")


def paths(task, attempt):
    require(task in TASKS and 1 <= attempt <= 99,"unknown diagnostic unit")
    tag = f"{task}_{attempt:02d}"
    return BASE/tag, BASE/("guard_"+tag), "d1-surface112-"+tag.replace("_","-")


def recorded(task):
    reports = []
    for path in sorted(BASE.glob(task+"_??/report.json")):
        out, guard, _ = paths(task,int(path.parent.name[-2:]))
        receipt = guard/"report.json"
        if not receipt.is_file() or read(receipt)["status"] != "COMPLETE":
            continue
        r = read(path)
        require(r["task"] == task and r["status"] == "COMPLETE" and r["sources"] == bindings()
                and r["proposal_sha256"] == PLAN_SHA and r["reference_manifest_sha256"] == REFERENCE_SHA,
                "unbound diagnostic receipt")
        if task in CASES:
            require(r["episode"]["seed"] == int(task) and digest(out/"trace.npz") == r["episode"]["trace_sha256"],
                    "unbound measured trajectory")
        reports.append((out,guard,r))
    require(len(reports) <= 1,"duplicate completed task")
    return reports[0] if reports else None


def worker(task, attempt):
    from tools27.resource_guard27 import read_memory
    out, _, unit = paths(task,attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope
            and os.getpriority(os.PRIO_PROCESS,0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4<<30,"inadmissible diagnostic worker")
    sources = bindings()
    require(not any(BASE.glob(task+"_??/reservation.json")),"task was already reserved")
    if task in CASES:
        require(recorded("survey") is not None,"equal-height gate closed")
    if task == "video":
        require(all(recorded(k) is not None for k in CASES),"video requires all three completed pairs")
    out.mkdir(exist_ok=False)
    header = dict(task=task,sources=sources,proposal_sha256=PLAN_SHA,reference_manifest_sha256=REFERENCE_SHA)
    write(out/"reservation.json",header)
    try:
        if task == "survey":
            from tools112.surface_native import survey
            data = dict(survey=survey())
        elif task == "video":
            from tools112.surface_video import render
            data = dict(film=render(out),renderer_sha256=digest(ROOT/"tools112/surface_video.py"))
        else:
            from tools112.surface_native import episode
            data = dict(episode=episode(out,int(task)))
        write(out/"report.json",dict(status="COMPLETE",**header,**data))
    except BaseException as error:
        write(out/"partial.json",dict(status="PARTIAL",error=repr(error),**header))
        raise


def launch_one(task):
    require(digest(PRESSURE) == GUARD_SHA,"pressure supervisor changed")
    bindings()
    BASE.mkdir(parents=True,exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if recorded(task) is not None:
            return
        require(not any(BASE.glob(task+"_??/reservation.json")),"preserve incomplete task "+task)
        if task in CASES:
            require(recorded("survey") is not None,"static survey did not pass")
        for attempt in range(1,100):
            out, guard, unit = paths(task,attempt)
            if out.exists() or guard.exists():
                previous = read(guard/"report.json")
                require(not out.exists() and not previous["child_started"],"preserve a started worker")
                continue
            if not ready():
                wait_for_admission()
            env = dict.fromkeys(THREADS,"1")
            env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))),
                       CUDA_VISIBLE_DEVICES="",MUJOCO_GL="disable",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1")
            if task == "video":
                from tools102.mesa_video102 import GL_ENV
                env.update(GL_ENV)
            cmd = ["systemd-run","--user","--wait","--collect","--pipe","--slice=app.slice",f"--unit={unit}",
                   "-p","Nice=10","-p","CPUWeight=25","-p","OOMPolicy=kill","-p","MemoryHigh=2G",
                   "-p","MemoryMax=3G","-p","MemorySwapMax=0",f"--working-directory={ROOT}",
                   *(f"--setenv={k}={v}" for k,v in env.items()),sys.executable,"-B",str(PRESSURE),
                   "--scope",scope_for(unit),"--output",str(guard),"--timeout","600","--",
                   sys.executable,"-B","-m","tools112.run_surface","worker",task,"--attempt",str(attempt)]
            status = subprocess.run(cmd).returncode
            receipt = read(guard/"report.json")
            if status == 0:
                require(recorded(task) is not None,"missing completed diagnostic receipt")
                return
            if receipt["child_started"] or out.exists() or not any(s in str(receipt["error"]) for s in
                    ("effective memory","admission not sustained","PSI")):
                raise RuntimeError(f"{unit} ended with {receipt['error']}")
            wait_for_admission()
        raise RuntimeError(f"no further admission directories for {task}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("launch","worker"))
    p.add_argument("task",choices=(*TASKS,"diagnose"))
    p.add_argument("--attempt",type=int,default=1)
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.task,a.attempt)
    elif a.task == "diagnose":
        for task in ("survey",*CASES):
            launch_one(task)
    else:
        launch_one(a.task)
