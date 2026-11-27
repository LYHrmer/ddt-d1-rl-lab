"""Run each registered high-speed forecast and native comparison once in its own guarded CPU unit."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools91.launch91 import PRESSURE, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import ready, wait_for_admission
from tools113.common import BASE, PLAN, ROOT, SEEDS, grid_case, registration, roster


FITS = tuple(f"fit_{s}" for s in SEEDS)
MOTIONS = tuple(f"native_{s}" for s in SEEDS)
STAGES = (*FITS,"forecast",*MOTIONS,"video")
SEAL = PLAN.parent/"forecast_seal.json"


def paths(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "unregistered 113 stage")
    tag = f"{stage}_{attempt:02d}"
    return BASE/tag, BASE/("guard_"+tag), "d1-speed113-"+tag.replace("_","-")


def outputs(out):
    return {p.name:digest(p) for p in sorted(out.iterdir())
            if p.is_file() and p.name not in ("report.json","reservation.json")}


def recorded(stage):
    found = []
    for f in BASE.glob(stage+"_??/report.json"):
        out, guard, _ = paths(stage,int(f.parent.name[-2:]))
        if not (guard/"report.json").is_file() or read(guard/"report.json")["status"] != "COMPLETE":
            continue
        report = read(f)
        require(report["status"] == "COMPLETE" and report["stage"] == stage
                and report["registration_sha256"] == registration() and report["files"] == outputs(out),
                f"unbound {stage} receipt")
        found.append((out,guard,report))
    require(len(found) <= 1,"duplicate completed "+stage)
    return found[0] if found else None


def dependency_stages(stage):
    if stage == "forecast":
        return FITS
    if stage in MOTIONS:
        return ("forecast",)
    if stage == "video":
        return (MOTIONS[0],)
    return ()


def dependencies(stage):
    parents = {}
    for parent in dependency_stages(stage):
        item = recorded(parent)
        require(item is not None,"missing fully guarded "+parent)
        parents[parent] = item
    return parents


def forecast_seal():
    out, _, r = dependencies(MOTIONS[0])["forecast"]
    return {"registration_sha256":registration(),"forecast_report_sha256":digest(out/"report.json"),
            "sealed_at_unix_ns":r["completed_at_unix_ns"],"forecast_inputs":r["inputs"],
            "rows":r["result"]["rows"],"actor_sha256":r["result"]["actor_sha256"]}


def write_seal():
    proof = forecast_seal()
    if SEAL.exists():
        require(proof == read(SEAL),"preserve prior prospective predictions")
        return
    require(not list(BASE.glob("native_*_??/reservation.json")),"cannot retrospectively seal a forecast")
    write(SEAL,proof)


def worker(stage, attempt):
    from tools27.resource_guard27 import read_memory
    out, _, unit = paths(stage,attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope and os.getpriority(os.PRIO_PROCESS,0) == 10
            and len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(k)=="1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4<<30,
            "worker outside the registered systemd limits")
    sha = registration()
    parents = dependencies(stage)
    if stage in MOTIONS:
        require(SEAL.is_file() and read(SEAL) == forecast_seal(),"missing prospective seal before physics")
    require(not any(BASE.glob(stage+"_??/reservation.json")), "preserve the original reservation")
    out.mkdir(exist_ok=False)
    header = {"stage":stage,"registration_sha256":sha,"reserved_at_unix_ns":time.time_ns(),
              "inputs":{p:digest(v[0]/"report.json") for p,v in parents.items()}}
    write(out/"reservation.json",header)
    try:
        if stage in FITS:
            from tools113.model import fit_fold
            result = fit_fold(out,roster(),int(stage.split("_")[1]))
        elif stage == "forecast":
            from tools113.forecast import make_forecast
            result = make_forecast(out,{s:parents[f"fit_{s}"][0]/"action.npz" for s in SEEDS})
        elif stage in MOTIONS:
            from tools113.native import episode
            result = episode(out,int(stage.split("_")[1]))
        else:
            from tools113.video import render
            seed = SEEDS[0]
            path, _ = grid_case(seed)
            result = render(out,seed,path,parents[MOTIONS[0]][0]/"trace.npz")
        write(out/"report.json",dict(**header,status="COMPLETE",files=outputs(out),result=result,
                                     completed_at_unix_ns=time.time_ns()))
    except BaseException as exc:
        write(out/"worker_partial.json",dict(**header,status="PARTIAL",error=repr(exc)))
        raise


def launch(stage):
    require(digest(PRESSURE) == GUARD_SHA,"pressure supervisor changed")
    registration()
    BASE.mkdir(parents=True,exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if recorded(stage) is not None:
            return
        dependencies(stage)
        if stage in MOTIONS:
            require(SEAL.is_file() and read(SEAL) == forecast_seal(),"seal the predictions first")
        require(not any(BASE.glob(stage+"_??/reservation.json")),"preserve the interrupted stage "+stage)
        env = dict.fromkeys(THREADS,"1")
        env.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))),
                   CUDA_VISIBLE_DEVICES="",MUJOCO_GL="disable",MPLBACKEND="Agg",PYTHONDONTWRITEBYTECODE="1")
        if stage == "video":
            from tools102.mesa_video102 import GL_ENV
            env.update(GL_ENV)
        for attempt in range(1,100):
            out, guard, unit = paths(stage,attempt)
            if out.exists() or guard.exists():
                require(not out.exists() and not read(guard/"report.json")["child_started"], "preserve a started worker")
                continue
            if not ready():
                wait_for_admission()
            cmd = ["systemd-run","--user","--wait","--collect","--pipe","--slice=app.slice",f"--unit={unit}",
                   "-p","Nice=10","-p","CPUWeight=25","-p","OOMPolicy=kill","-p","MemoryHigh=2G",
                   "-p","MemoryMax=3G","-p","MemorySwapMax=0",f"--working-directory={ROOT}",
                   *(f"--setenv={k}={v}" for k,v in env.items()),sys.executable,"-B",str(PRESSURE),
                   "--scope",scope_for(unit),"--output",str(guard),"--timeout","600","--",
                   sys.executable,"-B","-m","tools113.run","worker",stage,"--attempt",str(attempt)]
            print(f"dispatch {stage} in {unit}",flush=True)
            rc = subprocess.run(cmd).returncode
            receipt = read(guard/"report.json")
            if rc == 0:
                require(recorded(stage) is not None,"missing completed "+stage)
                return
            if receipt["child_started"] or out.exists() or not any(key in str(receipt["error"]) for key in
                                                                   ("effective memory","admission not sustained","PSI")):
                raise RuntimeError(f"{unit}: {receipt['error']}")
            wait_for_admission()
        raise RuntimeError("exhausted fresh admission directories")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("worker","launch","seal"))
    parser.add_argument("stage",nargs="?",choices=(*STAGES,"learn","validate"))
    parser.add_argument("--attempt",type=int,default=1)
    args = parser.parse_args()
    if args.mode == "seal":
        write_seal()
    elif args.mode == "worker":
        worker(args.stage,args.attempt)
    else:
        for key in THREADS:
            os.environ[key] = "1"
        sequence = (*FITS,"forecast") if args.stage=="learn" else MOTIONS if args.stage=="validate" else (args.stage,)
        for stage in sequence:
            launch(stage)
