"""Exclusive serial stages for physical collection, fixed-action inference and readback."""
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
from tools92.launch92 import scope_for
from tools93.common93 import BASE, PLAN_SHA, ROOT, checked_plan, digest, read, require


MODULE = {"coverage": "coverage93", "collect": "collect93", "inputs": "model93", "predict": "model93",
          "score": "score93", "video": "video93"}
OUTPUT = {"coverage": "coverage.json", "collect": "seal.json", "inputs": "seal.json", "predict": "seal.json",
          "score": "report.json", "video": "video.json"}


def receipt(stage):
    root = BASE / f"guard_{stage}_01"
    record = read(root / "report.json")
    require(record["status"] == "COMPLETE" and record["returncode"] == 0 and record["child_started"]
            and record["owned_cgroup_children_empty"] and len(record["cpu_affinity"]) == 1, f"incomplete {stage} unit")
    last = json.loads((root / "stdout.log").read_text().strip().splitlines()[-1])
    sha = digest(BASE / f"{stage}_01" / OUTPUT[stage])
    require(last == stage_record(stage, sha), f"{stage} receipt or producer changed")
    snapshots = [json.loads(s) for s in (root / "resources.jsonl").read_text().splitlines()]
    entry = [s["effective_available_bytes"] for s in snapshots if s["stage"] == "startup"][-1]
    reserve = min(s["effective_available_bytes"] for s in snapshots if s["stage"] == "runtime")
    require(entry >= 4 << 30 and reserve >= 2 << 30, f"insufficient {stage} memory reserve")
    return dict(guard_sha256=digest(root / "report.json"), receipt_sha256=sha, source_sha256=last["source_sha256"],
                admitted_effective_bytes=entry, minimum_running_effective_bytes=reserve,
                **{k: record[k] for k in ("cpu_affinity", "peak_group_bytes", "elapsed_s", "final_memory_events")})


def stage_record(stage, sha):
    return dict(stage=stage, protocol_sha256=PLAN_SHA, receipt_sha256=sha,
                source_sha256=digest(ROOT / "tools93" / (MODULE[stage] + ".py")))


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4 << 30,
            "requires admitted single-CPU unit")
    plan = checked_plan()
    require(digest(PRESSURE) == plan["resources"]["supervisor_sha256"], "resource supervisor differs")
    if stage in ("preflight", "tests"):
        tests = [ROOT / "tests" / "test_collect93.py", ROOT / "tests" / "test_model93.py"]
        if stage == "tests":
            tests = sorted((ROOT / "tests").glob("test*93.py"))
        subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *tests], check=True)
        return
    require(unit == f"d1-data93-{stage}-01", "unregistered attempt")
    out = BASE / f"{stage}_01"
    if stage == "coverage":
        from tools93.coverage93 import run
        run(out)
    elif stage == "collect":
        from tools87.ppo87 import single_cpu
        from tools93.collect93 import run
        single_cpu()
        run(out)
    elif stage == "inputs":
        from tools93.model93 import prepare
        receipt("collect")
        prepare(out, BASE / "collect_01")
    elif stage == "predict":
        from tools93.model93 import predict
        receipt("inputs")
        predict(out, BASE / "inputs_01")
    elif stage == "score":
        from tools93.score93 import run
        resources = {k: receipt(k) for k in ("coverage", "collect", "inputs", "predict")}
        run(out, resources)
    elif stage == "video":
        from tools93.video93 import run
        receipt("score")
        run(out)
    print(json.dumps(stage_record(stage, digest(out / OUTPUT[stage]))), flush=True)


def launch(stage):
    tag = str(time.time_ns()) if stage in ("preflight", "tests") else "01"
    unit = f"d1-data93-{stage}-{tag}"
    guard = BASE / f"guard_{stage}_{tag}"
    if guard.exists() or (BASE / f"{stage}_{tag}").exists():
        raise FileExistsError(f"preserve {stage} attempt: {guard}")
    settings = dict.fromkeys(THREADS, "1")
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                    CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
                    MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if stage == "video":
        settings.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
    command += [f"--setenv={k}={v}" for k, v in settings.items()]
    command += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                "--timeout", "1200", "--", sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit]
    subprocess.run(command, check=True)
    result = read(guard / "report.json")
    require(result["status"] == "COMPLETE" and result["owned_cgroup_children_empty"], "unit incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*MODULE, "preflight", "tests"])
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        BASE.mkdir(exist_ok=True, parents=True)
        with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.stage)
