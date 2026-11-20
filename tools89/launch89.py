"""Serial, resource-limited entry points for the D1 paired curriculum smoke."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools89.common89 import BASE, check_worker, complete_stage, read


def run_unit(stage):
    name = f"d1-curriculum89-{stage}"
    guard = BASE / f"guard_{stage}_01"
    if guard.exists():
        raise FileExistsError(f"keeping existing {stage} attempt at {guard}")
    env = {"PYTHONPATH": os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "MUJOCO_GL": "disable",
           "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    if stage == "video":
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1")
    env.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                              "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS", "BLIS_NUM_THREADS"), "1"))
    scope = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{name}.service"
    limit = ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"
    cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={name}",
           "--property=Nice=10", "--property=MemoryHigh=2G", "--property=MemoryMax=3G", "--property=MemorySwapMax=0",
           "--property=OOMPolicy=kill", "--property=CPUWeight=25", f"--working-directory={ROOT}"]
    cmd.extend(f"--setenv={k}={v}" for k, v in env.items())
    cmd.extend([sys.executable, "-B", str(limit), "--scope", scope, "--output", str(guard), "--timeout", "1200",
                "--", sys.executable, "-B", str(Path(__file__).resolve()), stage, "--worker"])
    subprocess.run(cmd, check=True)
    receipt = read(guard / "report.json")
    if receipt["status"] != "COMPLETE" or not receipt["owned_cgroup_children_empty"]:
        raise RuntimeError(f"unfinished resource or child cleanup receipt for {stage}")


def work(stage):
    check_worker(stage)
    if stage == "roadcheck":
        from tools89.roadcheck89 import run
        run()
        return
    if stage in ("admission", "smoke", "training"):
        from tools89.run89 import run
        run(stage)
        return
    if stage in ("evaluate", "readback"):
        complete_stage("training")
        if stage == "readback":
            complete_stage("evaluate")
        args = ["-m", "tools89.eval89", "run" if stage == "evaluate" else "readback", "--training",
                str(BASE / "training_01"), "--root", str(BASE / "evaluate_01")]
        if stage == "readback":
            args.extend(["--output", str(BASE / "readback_01.json")])
    elif stage == "video":
        complete_stage("evaluate")
        args = ["-m", "tools89.video89"]
    else:
        args = ["-m", "pytest", "--noconftest", "-p", "no:cacheprovider", "-q",
                *(str(p) for p in sorted((ROOT / "tests").glob("test*89.py")))]
    subprocess.run([sys.executable, "-B", *args], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("roadcheck", "admission", "smoke", "training", "evaluate", "readback", "tests", "video"))
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        work(args.stage)
    else:
        BASE.mkdir(exist_ok=True, parents=True)
        with (BASE / "serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_unit(args.stage)
