"""Run one cold-start seed or its diagnostics under the established single-CPU guard."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools91.launch91 import PRESSURE, THREADS
from tools92.launch92 import scope_for
from tools94.common94 import BASE, PLAN, ROOT, SEEDS, digest, read


STAGES = ("preflight", *(f"seed{s}" for s in SEEDS), "readback", "video", "verify")


def verify_start():
    import torch
    from tools89.course89 import D1Course89
    from tools94.actor94 import make_actor, predict
    from tools94.native94 import _endpoint, _reset_uphill
    from tools94.learn94 import _Ranges, _measure_packet

    # Verify the actual interface without advance, optimization or a calibration/evaluation sample.
    env = D1Course89("up", 947000, curriculum=True, perception=True, max_transitions=1)
    try:
        observation = _reset_uphill(env, 947000)
        endpoint = _endpoint(env, observation, 0)
        trace = {name: value[None] for name, value in endpoint.items()}
        ranges = _Ranges(trace, 0)
        if (env.data.time != 0. or observation.shape != (643,)
                or not (ranges.pending[1] == _measure_packet(endpoint["state45"])).all()):
            raise ValueError("cold-start reset or sensor interface failed")
        for seed in SEEDS:
            actor = make_actor(seed)
            action = predict(actor, observation)
            if action.shape != (16,) or not torch.isfinite(torch.tensor(action)).all():
                raise ValueError("cold actor export failed")
    finally:
        env.close()


def worker(stage, unit):
    from tools27.resource_guard27 import read_memory
    scope = scope_for(unit)
    if (Path("/proc/self/cgroup").read_text().strip() != "0::" + scope or len(os.sched_getaffinity(0)) != 1
            or os.getpriority(os.PRIO_PROCESS, 0) != 10 or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or any(os.environ.get(k) != "1" for k in THREADS)
            or read_memory(scope)["effective_available_bytes"] < 4 << 30
            or digest(PRESSURE) != read(PLAN)["resource_supervisor_sha256"]):
        raise RuntimeError("cold start requires the admitted, single-CPU, verified supervisor")
    if stage.startswith("seed") or stage == "preflight":
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if stage in ("preflight", "verify"):
        tests = sorted((ROOT / "tests").glob("test_cold94*.py"))
        if stage == "preflight":
            verify_start()
            tests = [t for t in tests if t.name != "test_cold94_public.py"]
        subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *tests], check=True)
    elif stage.startswith("seed"):
        from tools94.run94 import run
        report = run(int(stage.removeprefix("seed")))
        print(stage, report["status"], report["totals"], flush=True)
    elif stage == "readback":
        from tools94.readback94 import run
        run(BASE / "readback_01")
    elif stage == "video":
        subprocess.run([sys.executable, "-B", "-m", "tools94.video94", "--manifest",
                        str(BASE / "readback_01/video_manifest.json"), "--output", str(BASE / "video_01")], check=True)


def launch(stage):
    tag = str(time.time_ns()) if stage in ("preflight", "verify") else "01"
    unit = f"d1-cold94-{stage}-{tag}"
    guard = BASE / f"guard_{stage}_{tag}"
    if guard.exists() or (BASE / f"{stage}_{tag}").exists():
        raise FileExistsError(f"preserve the existing {stage} attempt")
    env = dict.fromkeys(THREADS, "1")
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               PYTHONPATH=os.pathsep.join(map(str, (ROOT/".python-deps", ROOT/"training", ROOT))),
               MUJOCO_GL="disable", MPLBACKEND="Agg", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    if stage == "video":
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit="+unit,
               "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G",
               "--property=MemorySwapMax=0", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
               "--working-directory="+str(ROOT)]
    command.extend(f"--setenv={k}={v}" for k,v in env.items())
    command.extend([sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                    "--timeout", "1800", "--", sys.executable, "-B", str(Path(__file__).resolve()), stage,
                    "--worker-unit", unit])
    subprocess.run(command, check=True)
    report = read(guard / "report.json")
    if report["status"] != "COMPLETE" or not report["owned_cgroup_children_empty"]:
        raise RuntimeError("incomplete resource guard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--worker-unit")
    args = parser.parse_args()
    if args.worker_unit:
        worker(args.stage, args.worker_unit)
    else:
        BASE.mkdir(parents=True, exist_ok=True)
        with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch(args.stage)
