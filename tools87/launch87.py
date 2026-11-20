"""Start the full-body smoke stages serially in resource-limited user services."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"
BASE = ROOT / "evidence/full_body87"
TESTS = ("tests/test_fullbody_ppo87.py", "tests/test_whole_body_terrain87.py",
         "tests/test_full_body_video87.py", "tests/test_fullbody_scores87.py")


def run_unit(stage, worker, timeout=900):
    name = f"d1-body87-{stage}-01"
    scope = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{name}.service"
    guard = BASE / f"guard_{stage}_01"
    if guard.exists():
        raise FileExistsError(f"preserving previous service evidence at {guard}")
    env = dict(PYTHONPATH=os.pathsep.join(str(x) for x in (ROOT / ".python-deps", ROOT / "training", ROOT)),
               PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", MUJOCO_GL="egl",
               PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", LIBGL_ALWAYS_SOFTWARE="1",
               CUDA_VISIBLE_DEVICES="")
    env.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS",
                              "BLIS_NUM_THREADS"), "1"))
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice",
               f"--unit={name}", "--property=Nice=10", "--property=MemoryHigh=2G",
               "--property=MemoryMax=3G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
               "--property=CPUWeight=25", f"--working-directory={ROOT}"]
    command.extend(f"--setenv={k}={v}" for k, v in env.items())
    command.extend([sys.executable, "-B", str(SUPERVISOR), "--scope", scope, "--output",
                    str(guard), "--timeout", str(timeout), "--", sys.executable, "-B", *map(str, worker)])
    subprocess.run(command, check=True)
    receipt = json.loads((guard / "report.json").read_text())
    if receipt["status"] != "COMPLETE" or not receipt["owned_cgroup_children_empty"]:
        raise RuntimeError(f"failed resource or cleanup receipt for {stage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "experiment", "verify", "tests"))
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "tests":
            run_unit("tests", ["-m", "pytest", "--noconftest", "-p", "no:cacheprovider", "-q", *TESTS])
            return
        if args.action == "verify":
            run_unit("readback", ["-m", "tools87.readback87", "--root", BASE,
                                  "--output", BASE / "readback_01.json"])
            run_unit("tests", ["-m", "pytest", "--noconftest", "-p", "no:cacheprovider", "-q", *TESTS])
            run_unit("preview", ["-m", "tools87.preview87", "--root", BASE])
            return
        entry = ["-m", "tools87.run87"]
        preflight, base, training, heldout = (BASE / f"{s}_01" for s in
                                               ("preflight", "baseline", "training", "evaluate"))
        if args.action == "preflight":
            run_unit("preflight", [*entry, "preflight", "--output", preflight])
            return
        for receipt in (preflight / "report.json", BASE / "guard_preflight_01/report.json"):
            if json.loads(receipt.read_text())["status"] != "COMPLETE":
                raise ValueError("complete preflight required")
        run_unit("baseline", [*entry, "baseline", "--output", base])
        run_unit("training", [*entry, "training", "--baseline", base, "--preflight", preflight,
                              "--output", training], timeout=1800)
        run_unit("evaluate", [*entry, "evaluate", "--training", training, "--output", heldout])
        selected = json.loads((heldout / "report.json").read_text())["video_selection"]
        run_unit("video", ["-m", "tools87.video87", "--baseline", heldout / selected["baseline"],
                           "--perceptive", heldout / selected["range"], "--output", BASE / "showcase_01.mp4"])


if __name__ == "__main__":
    main()
