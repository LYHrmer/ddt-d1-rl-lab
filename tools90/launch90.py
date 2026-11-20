"""Run the preregistered stand pair or offline WBC under the existing pressure supervisor."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/wbc90"
PLAN = ROOT / "docs/experiments90/stand_wbc.json"
PLAN_SHA256 = "b6e4f94c46fc7fde0a1a3f6892c09162c498ff4988260a43a0b0aefcef71a8bd"
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS", "BLIS_NUM_THREADS")
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    scope = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    expected = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}.service"
    if (not unit.startswith(f"d1-wbc90-{stage}-") or scope != expected or len(os.sched_getaffinity(0)) != 1
            or os.getpriority(os.PRIO_PROCESS, 0) != 10 or any(os.environ.get(k) != "1" for k in THREAD_KEYS)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or read_memory(scope)["effective_available_bytes"] < 4 << 30):
        raise RuntimeError("worker requires the admitted one-CPU pressure-supervised unit")
    plan = json.loads(PLAN.read_bytes())
    if sha(PLAN) != PLAN_SHA256 or any(sha(ROOT / plan[k]) != plan[k + "_sha256"] for k in ("parent87", "parent89")):
        raise ValueError("registered experiment or parent has changed")
    if stage == "stand":
        from tools90.stand90 import run
    elif stage == "shadow":
        import osqp
        if osqp.__version__ != "1.1.3":
            raise RuntimeError("use the registered OSQP 1.1.3 dependency")
        from tools90.shadow90 import run
    elif stage == "readback":
        from tools90.readback90 import run
    else:
        tests = sorted((ROOT / "tests").glob("test*90.py"))
        if not tests:
            raise FileNotFoundError("missing experiment tests")
        pytest = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"]
        subprocess.run([*pytest, *tests], check=True)
        subprocess.run([*pytest, "-s", ROOT / "tools90/check_rolling_bias90.py"], check=True)
        return
    suffix = unit.removeprefix(f"d1-wbc90-{stage}-")
    if stage != 'readback' and suffix != '01':
        raise ValueError('each scientific stage has one registered attempt')
    run(BASE / f"{stage}_{suffix}")


def launch(stage, audit_id):
    if audit_id != '01' and (stage != 'readback' or not re.fullmatch(r'[a-z0-9_-]{1,32}', audit_id)):
        raise ValueError('an independent readback ID is only valid for a fresh offline audit')
    suffix = str(time.time_ns()) if stage == "tests" else audit_id
    unit = f"d1-wbc90-{stage}-{suffix}"
    guard = BASE / f"guard_{stage}_{suffix}"
    if guard.exists() or (stage != "tests" and (BASE / f"{stage}_{suffix}").exists()):
        raise FileExistsError(f"attempt exists: {stage}. Retain evidence and partials in place.")
    env = dict.fromkeys(THREAD_KEYS, "1")
    env.update(PYTHONPATH=os.pathsep.join(map(str, (BASE / "deps_01", ROOT / ".python-deps", ROOT / "training", ROOT))),
               CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="disable",
               MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    scope = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}.service"
    pressure = ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
               "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
               "--property=OOMPolicy=kill", "--property=CPUWeight=25", f"--working-directory={ROOT}"]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    command.extend([sys.executable, "-B", str(pressure), "--scope", scope, "--output", str(guard), "--timeout", "1200", "--",
                    sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit])
    subprocess.run(command, check=True)
    result = json.loads((guard / "report.json").read_bytes())
    if result["status"] != "COMPLETE" or not result["owned_cgroup_children_empty"]:
        raise RuntimeError(f"resource guard did not complete: {guard}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stand", "shadow", "tests", "readback"))
    parser.add_argument("--worker-unit")
    parser.add_argument("--audit-id", default="01")
    args = parser.parse_args()
    if args.worker_unit:
        run_worker(args.stage, args.worker_unit)
    else:
        BASE.mkdir(exist_ok=True, parents=True)
        with (ROOT / "evidence/uphill89/serial.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.stage, args.audit_id)
