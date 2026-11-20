"""Run the one-shot offline replay and its independent readback under the serial D1 guard."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools91.launch91 import PRESSURE, THREADS
from tools92.common92 import BASE, PLAN_SHA, ROOT, checked_plan, digest, read, require


STAGES = ("preflight", "prepare", "replay", "score", "plot", "tests")
RECEIPTS = {"prepare": "seal.json", "replay": "seal.json", "score": "report.json", "plot": "effect.png"}


def scope_for(unit):
    uid = os.getuid()
    return f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.service"


def source_receipts():
    source_files = ("common92.py", "replay92.py")
    return {name: digest(ROOT / "tools92" / name) for name in source_files}


def completed_stage(stage):
    out = BASE / f"{stage}_01"
    guard = BASE / f"guard_{stage}_01"
    receipt = read(guard / "report.json")
    require(receipt["status"] == "COMPLETE" and receipt["child_started"] and receipt["returncode"] == 0
            and receipt["owned_cgroup_children_empty"] and len(receipt["cpu_affinity"]) == 1, f"incomplete {stage} guard")
    last = json.loads((guard / "stdout.log").read_text().strip().splitlines()[-1])
    require(last["protocol_sha256"] == PLAN_SHA and last["receipt_sha256"] == digest(out / RECEIPTS[stage])
            and last["producer_sha256"] == source_receipts(), f"{stage} output or predictor changed")
    samples = [json.loads(line) for line in (guard / "resources.jsonl").read_text().splitlines()]
    admission = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"][-1]
    running_min = min(s["effective_available_bytes"] for s in samples if s["stage"] == "runtime")
    require(admission >= 4 << 30 and running_min >= 2 << 30, f"{stage} memory reserve")
    return {**{k: receipt[k] for k in ("elapsed_s", "cpu_affinity", "peak_group_bytes", "final_memory_events")},
            "admitted_effective_bytes": admission, "minimum_running_effective_bytes": running_min,
            "guard_sha256": digest(guard / "report.json"), "receipt_sha256": last["receipt_sha256"]}


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4 << 30,
            "worker requires admitted single-CPU unit")
    plan = checked_plan()
    require(digest(PRESSURE) == plan["resources"]["supervisor_sha256"], "unregistered supervisor")
    if stage in ("preflight", "tests"):
        tests = ["tests/test_replay92.py", "tests/test_score92.py"]
        if stage == "tests":
            tests += ["tests/test_public92.py"]
        subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *tests], check=True)
        return
    require(unit == f"d1-fixed92-{stage}-01", "unregistered stage attempt")
    folder = BASE / f"{stage}_01"
    if stage == "prepare":
        from tools92.replay92 import prepare
        prepare(folder)
    elif stage == "replay":
        from tools92.replay92 import predict
        completed_stage("prepare")
        predict(folder, BASE / "prepare_01")
    elif stage == "score":
        from tools92.score92 import run
        resources = {name: completed_stage(name) for name in ("prepare", "replay")}
        run(folder, BASE / "replay_01", BASE / "prepare_01", resources=resources)
    elif stage == "plot":
        completed_stage("score")
        folder.mkdir()
        subprocess.run([sys.executable, "-B", "-m", "tools92.plot92", BASE / "score_01/report.json", folder / "effect.png"], check=True)
    print(json.dumps({"protocol_sha256": PLAN_SHA, "stage": stage, "receipt_sha256": digest(folder / RECEIPTS[stage]),
                      "producer_sha256": source_receipts()}), flush=True)


def launch(stage):
    tag = str(time.time_ns()) if stage in ("preflight", "tests") else "01"
    unit = f"d1-fixed92-{stage}-{tag}"
    guard = BASE / f"guard_{stage}_{tag}"
    if guard.exists() or (BASE / f"{stage}_{tag}").exists():
        raise FileExistsError(f"preserve existing {stage} attempt")
    settings = dict.fromkeys(THREADS, "1")
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                    CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
                    MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
    command += [f"--setenv={key}={value}" for key, value in settings.items()]
    command += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard), "--timeout", "900",
                "--", sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit]
    subprocess.run(command, check=True)
    receipt = read(guard / "report.json")
    require(receipt["status"] == "COMPLETE" and receipt["owned_cgroup_children_empty"], "incomplete supervisor receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        BASE.mkdir(exist_ok=True, parents=True)
        with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.stage)
