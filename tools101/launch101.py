"""Serial one-shot collection, fitting, blind evaluation and sensor replay."""
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
from tools101.common101 import BASE, PLAN_SHA, STAGES, producer, protocol


DEPENDENCIES = {"prepare": (), "fit": ("prepare",),
                "evaluate": ("prepare", "fit"), "video": ("prepare", "evaluate")}


def target(stage):
    return BASE / f"{stage}_01"


def guard_path(stage):
    return BASE / f"guard_{stage}_01"


def unit_name(stage):
    return f"d1-jeplo101-{stage}-01"


def resources(stage, limits):
    guard = guard_path(stage)
    receipt = read(guard / "report.json")
    scope = scope_for(unit_name(stage))
    require(receipt["status"] == "COMPLETE" and receipt["returncode"] == 0
            and receipt["child_started"] and receipt["owned_cgroup_children_empty"]
            and receipt["scope"] == scope and receipt["supervisor_sha256"] == limits["supervisor_sha256"]
            and len(receipt["cpu_affinity"]) == limits["cpu_affinity_count"], "incomplete supervisor receipt")
    samples = [json.loads(row) for row in (guard / "resources.jsonl").read_text().splitlines()]
    startup = [r["effective_available_bytes"] for r in samples if r["stage"] == "startup"]
    runtime = [r["effective_available_bytes"] for r in samples if r["stage"] == "runtime"]
    require(startup and runtime and startup[-1] >= limits["admission_min_bytes"]
            and min(runtime) >= limits["runtime_min_bytes"], "memory reserves were not observed")
    for sample in samples:
        owned = sample["owned"]
        require(owned["path"] == scope and owned["high_bytes"] == limits["memory_high_bytes"]
                and owned["max_bytes"] == limits["memory_max_bytes"]
                and owned["swap_max_bytes"] == limits["swap_max_bytes"] and owned["swap_current_bytes"] == 0,
                "resource limits differ from registration")
    return dict(startup_bytes=startup[-1], minimum_running_bytes=min(runtime), scope=scope,
                peak_group_bytes=receipt["peak_group_bytes"], cpu_affinity=receipt["cpu_affinity"],
                elapsed_s=receipt["elapsed_s"], memory_events=receipt["final_memory_events"],
                receipt_sha256=digest(guard / "report.json"), log_sha256=digest(guard / "resources.jsonl"))


def completed(stage, plan):
    folder = target(stage)
    seal = read(folder / "seal.json")
    require(seal["status"] == "COMPLETE" and seal["protocol_sha256"] == PLAN_SHA
            and seal["sources"] == producer(stage) and seal["files"] == files_for_seal(folder),
            f"{stage} output incomplete or changed")
    return folder, resources(stage, plan["resources"])


def execute(stage, folder, plan):
    data = target("prepare")
    if stage in ("fit", "evaluate"):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        write(folder / "runtime.json", dict(torch=torch.__version__, threads=torch.get_num_threads(),
                                            interop=torch.get_num_interop_threads(),
                                            deterministic=torch.are_deterministic_algorithms_enabled()))
    if stage == "prepare":
        from tools101.data101 import prepare
        prepare(folder, plan)
    elif stage == "fit":
        from tools101.fit101 import fit
        fit(data / "train.npz", folder, plan)
    elif stage == "evaluate":
        from tools101.fit101 import predict
        from tools101.score101 import score_archives
        predict(data / "holdout.npz", target("fit"), folder, plan)
        predictions = folder / "predictions.npz"
        bindings = {"holdout_sha256": digest(data / "holdout.npz"),
                    "fit_sha256": digest(target("fit") / "fit.json"),
                    "predictions_sha256": digest(predictions)}
        write(folder / "forecast_seal.json", bindings)
        outcome = score_archives(data / "holdout.npz", predictions, plan["seeds"])
        write(folder / "report.json", dict(status="COMPLETE", protocol_sha256=PLAN_SHA,
                                           score=outcome, **bindings))
    else:
        from tools101.video101 import render
        results = target("evaluate")
        render(data / "holdout.npz", results / "predictions.npz", results / "report.json", folder)


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    plan = protocol()
    limits = plan["resources"]
    scope = scope_for(unit)
    require(unit == unit_name(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"],
            "worker requires the registered admitted systemd unit")
    folder = target(stage)
    folder.mkdir()
    parents = {s: digest(target(s) / "seal.json") for s in DEPENDENCIES[stage]}
    start = dict(stage=stage, unit=unit, protocol_sha256=PLAN_SHA, sources=producer(stage), parents=parents)
    write(folder / "start.json", start)
    try:
        execute(stage, folder, plan)
        require(start["sources"] == producer(stage), "producer changed during stage")
        write(folder / "seal.json", dict(status="COMPLETE", files=files_for_seal(folder), **start))
    except BaseException as exc:
        write(folder / "partial.json", dict(status="PARTIAL", error=repr(exc), **start))
        raise


def launch(stage):
    plan = protocol()
    limits = plan["resources"]
    require(digest(PRESSURE) == limits["supervisor_sha256"], "supervisor differs from registration")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard_path(stage).exists() or target(stage).exists():
            raise FileExistsError(f"preserving prior {stage} attempt")
        for parent in DEPENDENCIES[stage]:
            completed(parent, plan)
        producer(stage)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        unit = unit_name(stage)
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}",
               f"--property=MemorySwapMax={limits['swap_max_bytes']}", "--property=CPUWeight=25",
               "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd.extend(f"--setenv={k}={v}" for k, v in env.items())
        cmd.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard_path(stage)),
                    "--timeout", "1200", "--", sys.executable, "-B", "-m", "tools101.launch101",
                    stage, "--worker-unit", unit])
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
