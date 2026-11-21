"""Run one new stage of the paired world-model update under the serial D1 supervisor."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools91.launch91 import PRESSURE, THREADS
from tools92.launch92 import scope_for
from tools96.common96 import BASE, ROOT, STAGES, checked, digest, protocol, read


def unit_name(stage):
    return f"d1-paired96-{stage}-01"


def completed_guard(stage):
    row = read(BASE / f"guard_{stage}_01/report.json")
    if not (row["status"] == "COMPLETE" and row["child_started"] and row["returncode"] == 0
            and row["owned_cgroup_children_empty"] and len(row["cpu_affinity"]) == 1
            and row["scope"] == scope_for(unit_name(stage))):
        raise RuntimeError(f"incomplete {stage} supervisor")


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory
    limits = protocol()["resources"]
    scope = scope_for(unit)
    if (unit != unit_name(stage) or Path("/proc/self/cgroup").read_text().strip() != "0::" + scope
            or len(os.sched_getaffinity(0)) != 1 or os.getpriority(os.PRIO_PROCESS, 0) != limits["nice"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(os.environ.get(k) != "1" for k in THREADS)
            or read_memory(scope)["effective_available_bytes"] < limits["admission_min_bytes"]
            or digest(PRESSURE) != limits["supervisor_sha256"]):
        raise RuntimeError("expected the registered supervised worker and admission reserve")
    index = STAGES.index(stage)
    if index:
        completed_guard(STAGES[index-1])
    folder = BASE / f"{stage}_01"
    folder.mkdir()
    if stage == "video":
        predictions, native = checked("predictions"), checked("native")
        subprocess.run([sys.executable, "-B", "-m", "tools96.video96", "--native", str(native),
                        "--predictions", str(predictions), "--output", str(folder / "paired_response96.mp4"),
                        "--receipt", str(folder / "video.json")], check=True)
    else:
        import torch
        from tools96.run96 import run_stage
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        run_stage(stage, folder)


def launch(stage):
    limits = protocol()["resources"]
    BASE.mkdir(parents=True, exist_ok=True)
    guard = BASE / f"guard_{stage}_01"
    unit = unit_name(stage)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.exists() or (BASE / f"{stage}_01").exists():
            raise FileExistsError(f"preserve the original {stage} attempt")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        if stage == "video":
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit=" + unit,
               "--property=Nice=" + str(limits["nice"]), "--property=MemoryHigh=" + str(limits["memory_high_bytes"]),
               "--property=MemoryMax=" + str(limits["memory_max_bytes"]),
               "--property=MemorySwapMax=" + str(limits["swap_max_bytes"]),
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", "--working-directory=" + str(ROOT)]
        cmd += [f"--setenv={k}={v}" for k, v in env.items()]
        cmd += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                "--timeout", "1800", "--", sys.executable, "-B", str(Path(__file__).resolve()), stage,
                "--worker-unit", unit]
        subprocess.run(cmd, check=True)
        completed_guard(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        launch(args.stage)
