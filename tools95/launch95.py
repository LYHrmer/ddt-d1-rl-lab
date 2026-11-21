"""Launch one exclusive direction-diagnostic stage under the shared D1 pressure guard."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools91.launch91 import PRESSURE, THREADS
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read

BASE = ROOT / "evidence/policy_direction95"
PLAN = ROOT / "docs/experiments95/policy_direction.json"
PLAN_SHA = "ed410b01fd20da7b57c54435231783a80193eed51f1c63f9bd8c3fddcbbca9af"
STAGES = ("directions", "predictions", "native", "video")


def guard_receipt(stage):
    output = BASE / f"guard_{stage}_01"
    receipt = read(output / "report.json")
    if not (receipt["status"] == "COMPLETE" and receipt["child_started"] and
            receipt["returncode"] == 0 and receipt["owned_cgroup_children_empty"] and
            len(receipt["cpu_affinity"]) == 1):
        raise RuntimeError(f"incomplete {stage} guard")
    return receipt


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory
    scope = scope_for(unit)
    if (unit != f"d1-direction95-{stage}-01" or Path("/proc/self/cgroup").read_text().strip() != "0::" + scope
            or len(os.sched_getaffinity(0)) != 1 or os.getpriority(os.PRIO_PROCESS, 0) != 10
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in THREADS)
            or read_memory(scope)["effective_available_bytes"] < 4 << 30
            or digest(PLAN) != PLAN_SHA or digest(PRESSURE) != read(PLAN)["resources"]["supervisor_sha256"]):
        raise RuntimeError("expected the registered single-CPU supervised worker")
    previous = STAGES.index(stage) - 1
    if previous >= 0:
        guard_receipt(STAGES[previous])
    folder = BASE / f"{stage}_01"
    folder.mkdir()
    if stage == "video":
        subprocess.run([sys.executable, "-B", "-m", "tools95.video95", "--run", str(BASE / "native_01"),
                        "--output", str(folder / "policy_direction95.mp4"),
                        "--receipt", str(folder / "video.json")], check=True)
        return
    import torch
    from tools95.run95 import run_stage
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run_stage(stage, folder)


def launch(stage):
    BASE.mkdir(parents=True, exist_ok=True)
    unit = f"d1-direction95-{stage}-01"
    guard = BASE / f"guard_{stage}_01"
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.exists() or (BASE / f"{stage}_01").exists():
            raise FileExistsError(f"preserve the existing {stage} attempt")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        if stage == "video":
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit=" + unit,
                   "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G",
                   "--property=MemorySwapMax=0", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
                   "--working-directory=" + str(ROOT)]
        command += [f"--setenv={key}={value}" for key, value in env.items()]
        command += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                    "--timeout", "1800", "--", sys.executable, "-B", str(Path(__file__).resolve()), stage,
                    "--worker-unit", unit]
        subprocess.run(command, check=True)
        guard_receipt(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        launch(args.stage)
