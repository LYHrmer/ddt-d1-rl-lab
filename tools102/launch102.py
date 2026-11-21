"""Run the archive replay and the desktop movie in separate bounded CPU units."""
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
from tools102.common102 import BASE, PLAN_SHA, STAGES, protocol, sources


def folder(stage):
    return BASE / f"{stage}_01"


def guard(stage):
    return BASE / f"guard_{stage}_01"


def unit(stage):
    return f"d1-occlusion102-{stage}-01"


def resource_receipt(stage, limits):
    where = guard(stage)
    receipt = read(where / "report.json")
    scope = scope_for(unit(stage))
    require(receipt["status"] == "COMPLETE" and receipt["returncode"] == 0 and receipt["child_started"]
            and receipt["owned_cgroup_children_empty"] and receipt["scope"] == scope
            and len(receipt["cpu_affinity"]) == limits["cpu_affinity_count"]
            and receipt["supervisor_sha256"] == limits["supervisor_sha256"], "incomplete resource guard")
    samples = [json.loads(line) for line in (where / "resources.jsonl").read_text().splitlines()]
    acquired = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    running = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    require(acquired and running and acquired[-1] >= limits["admission_min_bytes"]
            and min(running) >= limits["runtime_min_bytes"], "insufficient memory reserve")
    for sample in samples:
        row = sample["owned"]
        require(row["path"] == scope and row["high_bytes"] == limits["memory_high_bytes"]
                and row["max_bytes"] == limits["memory_max_bytes"] and row["swap_current_bytes"] == 0
                and row["swap_max_bytes"] == limits["swap_max_bytes"], "unexpected unit limits")
    return dict(scope=scope, receipt_sha256=digest(where / "report.json"), log_sha256=digest(where / "resources.jsonl"),
                startup_bytes=acquired[-1], minimum_running_bytes=min(running), cpu_affinity=receipt["cpu_affinity"],
                peak_group_bytes=receipt["peak_group_bytes"], elapsed_s=receipt["elapsed_s"],
                final_memory_events=receipt["final_memory_events"])


def completed(stage, plan):
    output = folder(stage)
    record = read(output / "seal.json")
    require(record["status"] == "COMPLETE" and record["protocol_sha256"] == PLAN_SHA
            and record["sources"] == sources(stage) and record["files"] == files_for_seal(output),
            f"partial or changed {stage} output")
    return output, resource_receipt(stage, plan["resources"])


def worker(stage, name):
    from tools27.resource_guard27 import read_memory

    plan = protocol()
    limits = plan["resources"]
    scope = scope_for(name)
    require(name == unit(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"],
            "worker requires the registered admitted single-core unit")
    output = folder(stage)
    output.mkdir()
    start = dict(unit=name, stage=stage, sources=sources(stage), protocol_sha256=PLAN_SHA)
    if stage == "video":
        start["replay_seal_sha256"] = digest(folder("replay") / "seal.json")
    write(output / "start.json", start)
    try:
        if stage == "replay":
            import torch
            from tools102.replay102 import replay

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            replay(output, plan)
        else:
            from tools102.mask102 import ROBOT_PATH, verify_nominal_robot
            from tools102.video102 import render
            verify_nominal_robot(ROOT / ROBOT_PATH)
            render(folder("replay"), output, plan)
        require(sources(stage) == start["sources"], "producer changed during stage")
        write(output / "seal.json", dict(status="COMPLETE", files=files_for_seal(output), **start))
    except BaseException as error:
        write(output / "partial.json", dict(status="PARTIAL", error=repr(error), **start))
        raise


def launch(stage):
    plan = protocol()
    limits = plan["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "resource supervisor changed")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard(stage).exists() or folder(stage).exists():
            raise FileExistsError(f"preserving previous {stage} attempt")
        if stage == "video":
            completed("replay", plan)
        sources(stage)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        if stage == "video":
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless",
                       GALLIUM_DRIVER="llvmpipe", LIBGL_ALWAYS_SOFTWARE="1")
        name = unit(stage)
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={name}",
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}",
               f"--property=MemorySwapMax={limits['swap_max_bytes']}", "--property=CPUWeight=25",
               "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd.extend(f"--setenv={k}={v}" for k, v in env.items())
        cmd.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(name), "--output", str(guard(stage)),
                    "--timeout", "1200", "--", sys.executable, "-B", "-m", "tools102.launch102",
                    stage, "--worker-unit", name])
        subprocess.run(cmd, check=True)
        completed(stage, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        launch(args.stage)
