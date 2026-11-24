"""Run the registered native contacts, sealed offline fits and software video in separate serial units."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write


PLAN = ROOT / "docs/experiments108/contact.json"
PLAN_SHA = "8781cedd4f3203adbbf42dbba094d9b9d780bb7eda373b37ad78b8859a7e5961"
BASE = ROOT / "evidence/contact108"
STAGES = ("development", "fit", "blind", "score", "video")
NATIVE_FILES = (
    "tools108/contact108.py", "tools87/terrain87.py", "tools89/course89.py", "tools89/train89.py",
    "training/d1_rl/env.py", "training/d1_rl/interface.py", "training/d1_rl/delay_env18.py",
    "training/configs/pilot03d.json")


def protocol():
    require(digest(PLAN) == PLAN_SHA, "registered plan changed")
    return read(PLAN)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid stage/attempt")
    name = f"{stage}_{attempt:02d}"
    return BASE / name, BASE / ("guard_" + name), f"d1-contact108-{stage}-{attempt:02d}"


def manifest(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file() and p != folder/"seal.json"}


def completed(stage):
    from tools88.export88 import resources

    valid = []
    for seal_file in sorted(BASE.glob(f"{stage}_??/seal.json")):
        folder = seal_file.parent
        attempt = int(folder.name.split("_")[-1])
        _, guard, unit = locations(stage, attempt)
        r = read(guard/"report.json")
        seal = read(seal_file)
        if r["status"] != "COMPLETE":
            continue
        require(r["scope"] == scope_for(unit) and r["supervisor_sha256"] == protocol()["resources"]["supervisor_sha256"]
                and seal["stage"] == stage and seal["status"] == "COMPLETE" and seal["plan_sha256"] == PLAN_SHA
                and seal["files"] == manifest(folder), "invalid stage evidence")
        valid.append((folder, resources(guard)))
    require(len(valid) == 1, f"expected one complete guarded stage for {stage}")
    return valid[0]


def work(stage, out, plan):
    if stage in ("development", "blind"):
        from tools108.contact108 import collect
        if stage == "blind":
            fits, _ = completed("fit")
            dev, _ = completed("development")
            require(read(dev/"inputs.json")["native_sources"] == read(out/"inputs.json")["native_sources"],
                    "physics or actor sources changed across splits")
            write(out/"fit_binding.json", {"fit_seal_sha256": digest(fits/"seal.json")})
        collect(out/"episodes", {**plan, "reset_seeds": plan[stage]})
        if stage == "development":
            from tools108.score108 import collect_summary
            collect_summary(out/"episodes", out, plan, stage)
        return
    if stage == "video":
        from tools108.video108 import render
        split = "blind" if any(BASE.glob("blind_??/seal.json")) else "development"
        src, _ = completed(split)
        render(src/"episodes", out/"step108.mp4", split, out/"video.json")
        return
    from tools108 import model108
    if stage == "fit":
        dev, _ = completed("development")
        require(read(dev/"contacts.json")["data_eligible"], "insufficient matched contact windows for a fit")
        model108.fit_models(model108.load_pairs(dev/"episodes", split="train"), out/"fit")
        return
    if stage == "score":
        from tools108.score108 import collect_summary
        blind, _ = completed("blind")
        fits, _ = completed("fit")
        require(read(fits/"inputs.json")["model_sha256"] == digest(Path(model108.__file__)), "fit/score source mismatch")
        # Only the original arm's pre-contact observation reaches forecast. Score labels open afterwards.
        forecasts = out/"forecast"
        model108.predict_blind(model108.load_blind_starts(blind/"episodes"), fits/"fit", forecasts)
        summary = collect_summary(blind/"episodes", out, plan, "blind")
        if summary["data_eligible"]:
            write(out/"score.json", model108.score_blind(model108.load_pairs(blind/"episodes", split="blind"), forecasts))
        else:
            write(out/"score.json", {"status": "INSUFFICIENT_BLIND_PAIRS", "policy_updates": 0})


def worker(stage, attempt):
    from tools27.resource_guard27 import read_memory
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    plan = protocol()
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4<<30,
            "worker requires the registered single-CPU unit")
    out.mkdir(exist_ok=False)
    try:
        inputs = {"plan_sha256": PLAN_SHA, "launcher_sha256": digest(Path(__file__)), "stage": stage}
        if stage in ("development", "blind"):
            inputs["native_sources"] = {name: digest(ROOT/name) for name in NATIVE_FILES}
        if stage in ("fit", "score"):
            inputs["model_sha256"] = digest(ROOT/"tools108/model108.py")
        write(out/"inputs.json", inputs)
        work(stage, out, plan)
        write(out/"seal.json", {"status":"COMPLETE", "stage":stage, "plan_sha256":PLAN_SHA, "files":manifest(out)})
    except BaseException as error:
        write(out/"partial.json", {"status":"PARTIAL", "stage":stage, "error":repr(error)})
        raise


def launch(stage, attempt):
    out, guard, unit = locations(stage, attempt)
    limits = protocol()["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "unregistered supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists() and not guard.exists(), "preserving prior attempt")
        if stage in ("development", "blind"):
            require(not any(BASE.glob(f"{stage}_??/episodes/seed*/report.json")), "reserved rollout matrix already used")
        if stage == "fit":
            require(not any(BASE.glob("fit_??/fit/*")), "fit budget has already started")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
                   MPLBACKEND="Agg", PYTHONPATH=os.pathsep.join(map(str,(ROOT/".python-deps",ROOT/"training",ROOT))))
        if stage == "video":
            from tools102.mesa_video102 import GL_ENV
            env.update(GL_ENV)
        command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
                   "--property=Nice=10", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
                   "--property=MemoryHigh=2147483648", "--property=MemoryMax=3221225472", "--property=MemorySwapMax=0",
                   f"--working-directory={ROOT}", *(f"--setenv={k}={v}" for k,v in env.items()), sys.executable, "-B",
                   str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard), "--timeout", "900", "--",
                   sys.executable, "-B", "-m", "tools108.run108", "worker", stage, "--attempt", str(attempt)]
        subprocess.run(command, check=True)
        completed(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker"))
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    {"launch":launch, "worker":worker}[args.action](args.stage, args.attempt)
