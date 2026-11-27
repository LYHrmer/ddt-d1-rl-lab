"""Run untouched, already registered development cases in independent units after a resource interruption."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.course import SEEDS
from tools111.run import BASE, PLAN, PLAN_SHA, complete, flat_choice, sources


ARMS = ("neither", "speed", "legs", "both")


def paths(seed, arm, attempt):
    require(seed in SEEDS["development"] and arm in ARMS and 1 <= attempt <= 99, "unregistered development case")
    tag = f"individual_{seed}_{arm}_{attempt:02d}"
    return BASE/tag, BASE/("guard_"+tag), "d1-lidar111-"+tag.replace("_","-")


def origin():
    archive = BASE/"development_01"
    guard = read(BASE/"guard_development_01/report.json")
    require(guard["status"] == "INCOMPLETE" and guard["owned_cgroup_children_empty"]
            and guard["error"] == "RuntimeError: effective memory below original2GiB reserve",
            "original unit must be terminal from the resource guard")
    require(read(archive/"inputs.json")["sources"] == sources() and digest(PLAN) == PLAN_SHA, "physical code changed")
    return archive


def unstarted(seed, arm):
    name = f"{seed}_{arm}"
    require(not any(BASE.glob(f"development_??/{name}/reservation.json"))
            and not any(BASE.glob(f"individual_*/{name}/reservation.json")), f"{name} already reserved")


def worker(seed, arm, attempt):
    from tools27.resource_guard27 import read_memory
    from tools110.native import load_policy
    from tools111.native import episode
    out, _, unit = paths(seed, arm, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope and os.getpriority(os.PRIO_PROCESS,0) == 10
            and len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4<<30,
            "independent case outside the admitted unit")
    origin()
    unstarted(seed, arm)
    complete("preflight")
    gain, safe, qualified = flat_choice()
    require(safe, "compound safety gate closed")
    out.mkdir(exist_ok=False)
    head = dict(seed=seed, arm=arm, layout="development", plan_sha256=PLAN_SHA, sources=sources(), flat_passed=qualified)
    write(out/"inputs.json",head)
    try:
        row = episode(out/f"{seed}_{arm}",seed,"development",arm,gain,load_policy())
        write(out/"report.json",dict(status="COMPLETE",**head,episode=row))
    except BaseException as error:
        write(out/"partial.json",dict(status="PARTIAL",error=repr(error),**head))
        raise


def launch(seed, arm, attempt):
    out, guard, unit = paths(seed, arm, attempt)
    require(digest(PRESSURE) == GUARD_SHA and digest(PLAN) == PLAN_SHA, "guard or registration changed")
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        origin()
        unstarted(seed, arm)
        require(not out.exists() and not guard.exists(), "preserve previous independent case")
        env = dict.fromkeys(THREADS,"1")
        env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))),
                   CUDA_VISIBLE_DEVICES="",MUJOCO_GL="disable",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1")
        cmd = ["systemd-run","--user","--wait","--collect","--pipe","--slice=app.slice",f"--unit={unit}",
               "-p","Nice=10","-p","CPUWeight=25","-p","OOMPolicy=kill","-p","MemoryHigh=2G",
               "-p","MemoryMax=3G","-p","MemorySwapMax=0",f"--working-directory={ROOT}",
               *(f"--setenv={k}={v}" for k,v in env.items()),sys.executable,"-B",str(PRESSURE),
               "--scope",scope_for(unit),"--output",str(guard),"--timeout","300","--",sys.executable,
               "-B","-m","tools111.individual","worker",str(seed),arm,"--attempt",str(attempt)]
        subprocess.run(cmd,check=True)
        require(read(guard/"report.json")["status"] == "COMPLETE" and (out/"report.json").is_file(), "missing independent receipt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("worker","launch"))
    p.add_argument("seed",type=int)
    p.add_argument("arm",choices=ARMS)
    p.add_argument("--attempt",type=int,default=1)
    a = p.parse_args()
    {"worker":worker,"launch":launch}[a.mode](a.seed,a.arm,a.attempt)
