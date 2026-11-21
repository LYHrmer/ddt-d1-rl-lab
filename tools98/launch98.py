"""Execute one stage of the registered endpoint training and validation on a single CPU."""
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
from tools98.common98 import (BASE, PLAN_SHA, ROOT, STAGES, checked_stage, previous_seal,
                              protocol, resource_record, sources_for, unit_name)


def verify_predecessors(stage):
    for prior in STAGES[:STAGES.index(stage)]:
        resource_record(prior)
        checked_stage(prior)


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory

    plan = protocol(private=True)
    limits = plan["resources"]
    scope = scope_for(unit)
    require(unit == unit_name(stage) and Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == limits["nice"] and len(os.sched_getaffinity(0)) == 1
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and all(os.environ.get(k) == "1" for k in THREADS)
            and read_memory(scope)["effective_available_bytes"] >= limits["admission_min_bytes"]
            and digest(PRESSURE) == limits["supervisor_sha256"], "expected admitted single-CPU endpoint worker")
    verify_predecessors(stage)
    folder = BASE / f"{stage}_01"
    folder.mkdir()
    start = {"stage": stage, "protocol_sha256": PLAN_SHA, "sources": sources_for(stage), "unit": unit,
             "previous_seal_sha256": digest(previous_seal(stage)),
             "direction_seal95_sha256": plan["local_parent_seals_sha256"]["evidence/policy_direction95/directions_01/seal.json"]}
    write(folder / "start.json", start)
    if stage == "video":
        subprocess.run([sys.executable, "-B", "-m", "tools98.video98", "--output", str(folder / "endpoint98.mp4"),
                        "--receipt", str(folder / "video.json")], check=True)
        result = {"totals": {"status": read(folder / "video.json")["status"]}}
    else:
        import torch
        from tools98 import run98

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result = {"fit": run98.fit, "predictions": run98.predict, "native": run98.native}[stage](folder)
    require(start["sources"] == sources_for(stage) and start["previous_seal_sha256"] == digest(previous_seal(stage)),
            "producer or sealed input changed during stage")
    write(folder / "seal.json", {**start, **result, "files": files_for_seal(folder)})


def launch(stage):
    limits = protocol(private=True)["resources"]
    BASE.mkdir(parents=True, exist_ok=True)
    guard = BASE / f"guard_{stage}_01"
    unit = unit_name(stage)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if guard.exists() or (BASE / f"{stage}_01").exists():
            raise FileExistsError(f"preserve existing {stage} attempt")
        verify_predecessors(stage)
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                   MUJOCO_GL="disable", MPLBACKEND="Agg")
        if stage == "video":
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit=" + unit,
               f"--property=Nice={limits['nice']}", f"--property=MemoryHigh={limits['memory_high_bytes']}",
               f"--property=MemoryMax={limits['memory_max_bytes']}", f"--property=MemorySwapMax={limits['swap_max_bytes']}",
               "--property=CPUWeight=25", "--property=OOMPolicy=kill", f"--working-directory={ROOT}"]
        cmd += [f"--setenv={k}={v}" for k, v in env.items()]
        cmd += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                "--timeout", "1800", "--", sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker-unit", unit]
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
