"""Launch the registered D1 uphill probe and its separate archive readers serially."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/uphill88"
ARCHIVE = ROOT / "evidence/full_body87"
SUPERVISOR = ROOT.parent / "wheel-legged-control-lab/scripts/run_pressure83.py"


def run(stage):
    name = f"d1-uphill88-{stage}"
    guard = BASE / f"guard_{stage}_01"
    if guard.exists():
        raise FileExistsError(guard)
    scope = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{name}.service"
    env = {"PYTHONPATH": os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "MUJOCO_GL": "disable",
           "MPLBACKEND": "Agg", "CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LP_NUM_THREADS",
                              "BLIS_NUM_THREADS"), "1"))
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice",
               f"--unit={name}", "--property=Nice=10", "--property=MemoryHigh=2G",
               "--property=MemoryMax=3G", "--property=MemorySwapMax=0", "--property=OOMPolicy=kill",
               "--property=CPUWeight=25", f"--working-directory={ROOT}"]
    command.extend(f"--setenv={k}={v}" for k, v in env.items())
    if stage == "probe":
        worker = ["-m", "tools88.probe88"]
    elif stage == "archive":
        worker = [str(Path(__file__).resolve()), "read_archives"]
    else:
        worker = ["-m", "pytest", "--noconftest", "-p", "no:cacheprovider", "-q",
                  *(str(path) for path in sorted((ROOT / "tests").glob("test*88.py")))]
    command.extend([sys.executable, "-B", str(SUPERVISOR), "--scope", scope, "--output", str(guard),
                    "--timeout", "900", "--", sys.executable, "-B", *worker])
    subprocess.run(command, check=True)
    receipt = json.loads((guard / "report.json").read_text())
    if receipt["status"] != "COMPLETE" or not receipt["owned_cgroup_children_empty"]:
        raise RuntimeError(f"incomplete resource or cleanup receipt for {stage}")


def read_archives():
    folder = BASE / "readback_01"
    folder.mkdir(exist_ok=False)
    failures = []
    for module in ("drivetrain88", "contact88", "policy88", "verify88", "plot88"):
        root = BASE / "probe_01" if module in ("verify88", "plot88") else ARCHIVE
        dest = folder / (module + (".png" if module == "plot88" else ".json"))
        cmd = [sys.executable, "-B", "-m", "tools88." + module, "--root", str(root), "--output", str(dest)]
        if subprocess.run(cmd).returncode:
            failures.append(module)
    if failures:
        raise RuntimeError("archive readers failed: " + ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "archive", "tests", "read_archives"))
    stage = parser.parse_args().stage
    if stage == "read_archives":
        if not Path("/proc/self/cgroup").read_text().strip().endswith("/d1-uphill88-archive.service"):
            parser.error("use the archive stage to start the guarded readers")
        read_archives()
    else:
        BASE.mkdir(parents=True, exist_ok=True)
        with (BASE / "serial.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(stage)
