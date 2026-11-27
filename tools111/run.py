"""One-shot, serial D1 lidar stages under the shared pressure supervisor."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA


BASE = ROOT / "evidence/lidar_compound111"
PLAN = ROOT / "docs/lidar_compound111_proposal.md"
PLAN_SHA = "cf67f221a834e487f96e8f0ff583ad15fdb1143925ddb61d70d928ab5727f9e0"
STAGES = ("preflight", "flat", "calibration", "development", "blind", "video")


def paths(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid lidar stage")
    tag = f"{stage}_{attempt:02d}"
    return BASE/tag, BASE/("guard_"+tag), f"d1-lidar111-{stage}-{attempt:02d}"


def sources():
    from tools110.run import bindings
    files = ("tools111/control.py", "tools111/sensing.py", "tools111/course.py", "tools111/native.py",
             "tools111/preflight.py", "tools111/run.py", "tools100/sensor100.py", "tools87/terrain87.py",
             "docs/experiments100/pointcloud_density.json")
    return {**bindings(), **{name: digest(ROOT/name) for name in files}}


def complete(stage):
    from tools88.export88 import resources
    verified = []
    for f in sorted(BASE.glob(stage+"_??/report.json")):
        out, guard, _ = paths(stage, int(f.parent.name[-2:]))
        if not (guard/"report.json").is_file() or read(guard/"report.json")["status"] != "COMPLETE":
            continue
        record = read(f)
        require(record["status"] == "COMPLETE" and record["sources"] == sources() and record["stage"] == stage
                and record["plan_sha256"] == digest(PLAN) == PLAN_SHA, "unbound lidar stage")
        for row in record.get("episodes", []):
            trial = out/f"{row['seed']}_{row['arm']}"
            require(read(trial/"report.json") == row and row["status"] == "COMPLETE"
                    and digest(trial/"trace.npz") == row["trace_sha256"], "incomplete lidar trajectory")
        verified.append((out, record, resources(guard)))
    require(len(verified) == 1, f"no unique complete {stage}")
    return verified[0]


def flat_choice():
    from tools111.control import SPEED_GAIN
    from tools111.course import calibration_gain
    initial = complete("flat")[1]["episodes"]
    gain = calibration_gain(initial)
    rows = ([r for r in initial if r["arm"] == "heading_on"] if gain is None
            else complete("calibration")[1]["episodes"])
    safe = all(not set(r["motion"]["failed"]) & {"horizon","stopped","fall_free","legal_support"} for r in rows)
    return gain if gain is not None else SPEED_GAIN, safe, all(r["motion"]["passed"] for r in rows)


def both_qualifies(rows):
    group = [r for r in rows if r["arm"] == "both"]
    return len(group) == 3 and all(r["motion"]["passed"] and r["effects"]["speed_active"]
                                   and r["effects"]["lift_active"] for r in group)


def measure(out, stage):
    from tools110.native import load_policy
    from tools111.control import SPEED_GAIN
    from tools111.course import SEEDS, calibration_gain
    from tools111.native import episode
    complete("preflight")
    gain, layout, arms = SPEED_GAIN, "flat", ("heading_off", "heading_on")
    if stage == "calibration":
        gain = calibration_gain(complete("flat")[1]["episodes"])
        require(gain is not None, "calibration gate closed")
        arms = ("heading_on",)
    if stage in ("development", "blind"):
        gain, safe, flat_passed = flat_choice()
        require(safe, "compound diagnostic safety gate closed")
        layout = stage
        arms = ("neither", "speed", "legs", "both")
        if stage == "blind":
            require(flat_passed and both_qualifies(complete("development")[1]["episodes"]), "blind gate closed")
            arms = ("both",)
    session = load_policy()
    rows = [episode(out/f"{seed}_{arm}", seed, layout, arm, gain, session) for seed in SEEDS[stage] for arm in arms]
    return dict(episodes=rows)


def worker(stage, attempt):
    from tools27.resource_guard27 import read_memory
    out, _, unit = paths(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::"+scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4<<30
            and digest(PLAN) == PLAN_SHA, "inadmissible lidar worker")
    out.mkdir(exist_ok=False)
    head = dict(stage=stage, plan_sha256=PLAN_SHA, sources=sources())
    write(out/"inputs.json", head)
    try:
        if stage == "preflight":
            from tools111.preflight import run
            result = run()
        elif stage == "video":
            from tools111.video import render
            result = render(out)
        else:
            result = measure(out, stage)
        write(out/"report.json", dict(status="COMPLETE", **head, **result))
    except BaseException as error:
        write(out/"partial.json", dict(status="PARTIAL", error=repr(error), **head))
        raise


def launch(stage, attempt):
    out, guard, unit = paths(stage, attempt)
    require(digest(PLAN) == PLAN_SHA and digest(PRESSURE) == GUARD_SHA, "lidar protocol changed")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT/"evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not out.exists() and not guard.exists() and not any(BASE.glob(f"{stage}_??/*/reservation.json"))
                and not any(BASE.glob(f"{stage}_??/report.json")), "preserve existing complete or native attempt")
        env = dict.fromkeys(THREADS, "1")
        env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT/".python-deps",ROOT/"training",ROOT))),
                   CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        if stage == "video":
            from tools102.mesa_video102 import GL_ENV
            env.update(GL_ENV)
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "OOMPolicy=kill", "-p", "MemoryHigh=2G",
               "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", f"--working-directory={ROOT}",
               *(f"--setenv={k}={v}" for k,v in env.items()), sys.executable, "-B", str(PRESSURE),
               "--scope", scope_for(unit), "--output", str(guard), "--timeout", "1200", "--", sys.executable,
               "-B", "-m", "tools111.run", "worker", stage, "--attempt", str(attempt)]
        subprocess.run(cmd, check=True)
        require(read(guard/"report.json")["status"] == "COMPLETE" and (out/"report.json").is_file(), "missing lidar receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch","worker"))
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    {"launch":launch, "worker":worker}[args.mode](args.stage, args.attempt)
