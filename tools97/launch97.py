"""Run the two immutable suffix probes and their video in serial constrained D1 units."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import digest, read, write
from tools95.run95 import files_for_seal
from tools97.common97 import BASE, PLAN_SHA, ROOT, STAGES, checked_stage, previous_seal, protocol, resource_record, sources_for, unit_name


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    limits = protocol(private=True)["resources"]
    scope = scope_for(unit)
    require(unit == unit_name(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"]
            and len(os.sched_getaffinity(0)) == limits["cpu_affinity_count"]
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1" for k in THREADS)
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"]
            and digest(PRESSURE) == limits["supervisor_sha256"], "run inside the admitted CPU supervisor")
    predecessors = STAGES[:STAGES.index(stage)]
    for previous in predecessors:
        resource_record(previous)
        checked_stage(previous)
    inputs = digest(previous_seal(stage))
    producers = sources_for(stage)
    folder = BASE / f"{stage}_01"
    folder.mkdir()
    start = {"stage": stage, "protocol_sha256": PLAN_SHA, "previous_seal_sha256": inputs,
             "sources": producers, "unit": unit}
    write(folder / "start.json", start)
    if stage == "video":
        subprocess.run([sys.executable, "-B", "-m", "tools97.video97", "--output", str(folder / "suffix_replay97.mp4"),
                        "--receipt", str(folder / "video.json")], check=True)
        video = read(folder / "video.json")
        totals = {"status": video["status"]}
    else:
        import torch
        from tools97.replay97 import run

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        totals = run(stage, folder)
        require(0 <= totals["completed_model_steps"] <= totals["attempted_model_steps"]
                <= protocol()["budget"]["max_model_steps_per_stage"], "suffix model-step budget exceeded")
    require(inputs == digest(previous_seal(stage)) and producers == sources_for(stage), "stage sources changed during replay")
    write(folder / "seal.json", {**start, "totals": totals, "files": files_for_seal(folder)})


def launch(stage):
    limits = protocol(private=True)["resources"]
    unit = unit_name(stage)
    guard = BASE / f"guard_{stage}_01"
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.exists() or (BASE / f"{stage}_01").exists():
            raise FileExistsError(f"preserve the existing {stage} attempt")
        for previous in STAGES[:STAGES.index(stage)]:
            resource_record(previous)
            checked_stage(previous)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="disable",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MPLBACKEND="Agg")
        if stage == "video":
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit=" + unit,
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}", f"--property=MemorySwapMax={limits['swap_max_bytes']}",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd += [f"--setenv={key}={value}" for key, value in env.items()]
        cmd += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard), "--timeout", "900",
                "--", sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit]
        subprocess.run(cmd, check=True)
        resource_record(stage)
        checked_stage(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        launch(args.stage)
