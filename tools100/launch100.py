"""Run the paired sensing experiment or its movie in an isolated CPU unit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools95.run95 import files_for_seal
from tools100.common100 import BASE, PLAN_SHA, producer, protocol


def stage_path(stage):
    return BASE / f"{stage}_01"


def unit_name(stage):
    return f"d1-pointcloud100-{stage}-01"


def resource_summary(stage, limits):
    guard = BASE / f"guard_{stage}_01"
    receipt = read(guard / "report.json")
    scope = scope_for(unit_name(stage))
    require(receipt["status"] == "COMPLETE" and receipt["returncode"] == 0
            and receipt["owned_cgroup_children_empty"] and receipt["child_started"] and receipt["scope"] == scope
            and receipt["supervisor_sha256"] == limits["supervisor_sha256"]
            and len(receipt["cpu_affinity"]) == limits["cpu_affinity_count"], "resource supervisor incomplete")
    rows = [json.loads(line) for line in (guard / "resources.jsonl").read_text().splitlines()]
    startup = [r["effective_available_bytes"] for r in rows if r["stage"] == "startup"]
    runtime = [r["effective_available_bytes"] for r in rows if r["stage"] == "runtime"]
    require(startup[-1] >= limits["admission_min_bytes"] and min(runtime) >= limits["runtime_min_bytes"],
            "memory reserve not met")
    for row in rows:
        owned = row["owned"]
        require(owned["path"] == scope and owned["high_bytes"] == limits["memory_high_bytes"]
                and owned["max_bytes"] == limits["memory_max_bytes"]
                and owned["swap_max_bytes"] == limits["swap_max_bytes"] and owned["swap_current_bytes"] == 0,
                "unexpected cgroup limits")
    return {"startup_effective_bytes": startup[-1], "minimum_running_bytes": min(runtime),
            "peak_group_bytes": receipt["peak_group_bytes"], "elapsed_s": receipt["elapsed_s"],
            "final_memory_events": receipt["final_memory_events"], "cpu_affinity": receipt["cpu_affinity"],
            "supervisor_sha256": receipt["supervisor_sha256"], "scope": scope,
            "resource_log_sha256": digest(guard / "resources.jsonl"), "receipt_sha256": digest(guard / "report.json")}


def completed(stage, plan):
    folder = stage_path(stage)
    seal = read(folder / "seal.json")
    require(seal["status"] == "COMPLETE" and seal["protocol_sha256"] == PLAN_SHA
            and seal["sources"] == producer(stage) and seal["files"] == files_for_seal(folder),
            f"unsealed or changed {stage} artifacts")
    return folder, resource_summary(stage, plan["resources"])


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    plan = protocol(private=True)
    limits = plan["resources"]
    scope = scope_for(unit)
    require(unit == unit_name(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"],
            "expected admitted single-threaded unit")
    folder = stage_path(stage)
    folder.mkdir()
    start = {"stage": stage, "unit": unit, "protocol_sha256": PLAN_SHA, "sources": producer(stage)}
    write(folder / "start.json", start)
    try:
        if stage == "replay":
            from tools100.replay100 import replay
            replay(folder, plan)
        else:
            source, _ = completed("replay", plan)
            subprocess.run([sys.executable, "-B", "-m", "tools100.video100",
                            "--archive", str(source / "clouds.npz"), "--report", str(source / "report.json"),
                            "--output", str(folder / "pointcloud100.mp4"), "--receipt", str(folder / "video.json")],
                           check=True)
        require(start["sources"] == producer(stage), "producer changed during run")
        write(folder / "seal.json", dict(**start, status="COMPLETE", files=files_for_seal(folder)))
    except BaseException as exc:
        write(folder / "partial.json", dict(**start, status="PARTIAL", error=repr(exc)))
        raise


def launch(stage):
    plan = protocol(private=True)
    limits = plan["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "resource supervisor changed")
    BASE.mkdir(parents=True, exist_ok=True)
    unit = unit_name(stage)
    guard = BASE / f"guard_{stage}_01"
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.exists() or stage_path(stage).exists():
            raise FileExistsError(f"preserve the existing {stage} attempt")
        if stage == "video":
            completed("replay", plan)
        producer(stage)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}", f"--property=MemorySwapMax={limits['swap_max_bytes']}",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd.extend(f"--setenv={k}={v}" for k, v in env.items())
        cmd.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                    "--timeout", "900", "--", sys.executable, "-B", "-m", "tools100.launch100", stage, "--worker-unit", unit])
        subprocess.run(cmd, check=True)
        completed(stage, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "video"))
    parser.add_argument("--worker-unit")
    arguments = parser.parse_args()
    if arguments.worker_unit:
        worker(arguments.stage, arguments.worker_unit)
    else:
        launch(arguments.stage)
