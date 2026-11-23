"""One serial workstation readback and publication of the frozen D1 support-surface comparison."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools105 import score105
from tools105.data105 import build_windows


BASE = ROOT / "evidence/surface105"
RECORD, GUARD = BASE / "readback_01", BASE / "guard_01"
UNIT = "d1-surface105-readback-01"
FILES = ("input_audit.json", "windows.npz", "predictions.npz", "report.json")


def registration():
    plan = read(score105.PROTOCOL)
    for relative, sha in plan["frozen_files_sha256"].items():
        require(digest(ROOT / relative) == sha, f"registered source changed: {relative}")
    return plan


def worker():
    from tools27.resource_guard27 import read_memory
    registration()
    scope = scope_for(UNIT)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and os.getpriority(os.PRIO_PROCESS, 0) == 10
            and len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4 << 30,
            "readback requires the admitted single-CPU unit")
    RECORD.mkdir(parents=True, exist_ok=False)
    write(RECORD / "input_audit.json", build_windows(RECORD / "windows.npz"))
    report = score105.record(RECORD)
    write(RECORD / "seal.json", dict(status=report["status"], protocol_sha256=digest(score105.PROTOCOL),
                                    files={name: digest(RECORD / name) for name in FILES}))


def launch():
    plan = registration()
    require(digest(PRESSURE) == plan["pressure_supervisor_sha256"], "resource supervisor changed")
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not RECORD.exists() and not GUARD.exists(), "preserving the previous readback attempt")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={UNIT}",
               "--property=Nice=10", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
               "--property=MemoryHigh=2147483648", "--property=MemoryMax=3221225472", "--property=MemorySwapMax=0",
               f"--working-directory={ROOT}"]
        cmd += [f"--setenv={k}={v}" for k, v in env.items()]
        cmd += [sys.executable, "-B", str(PRESSURE), "--scope", scope_for(UNIT), "--output", str(GUARD),
                "--timeout", "600", "--", sys.executable, "-B", "-m", "tools105.run105", "worker"]
        subprocess.run(cmd, check=True)


def verify_seal(folder):
    registration()
    seal = read(folder / "seal.json")
    require(seal == dict(status="COMPLETE", protocol_sha256=digest(score105.PROTOCOL),
                         files={name: digest(folder / name) for name in FILES}), "incomplete or changed readback")
    audit = read(folder / "input_audit.json")
    require(audit["windows_sha256"] == seal["files"]["windows.npz"] and audit["native_first_returns"]["file_count"] == 32
            and audit["native_first_returns"]["public_float32_equals_rounded_native"]
            and audit["native_first_returns"]["validity_and_mask_exact"], "incomplete range audit")


def publish():
    from tools88.export88 import resources

    verify_seal(RECORD)
    usage = resources(GUARD)
    require(read(GUARD / "report.json")["scope"] == scope_for(UNIT), "unexpected unit")
    dest = score105.PUBLIC
    require(not dest.exists(), "preserving the public readback")
    with TemporaryDirectory(dir=dest.parent, prefix=".surface105-") as temp:
        stage = Path(temp) / "complete"
        stage.mkdir()
        for name in (*FILES, "seal.json"):
            shutil.copyfile(RECORD / name, stage / name)
        verify_seal(stage)
        write(stage / "resources.json", {**usage, "unit": UNIT, "nice": 10, "memory_high_bytes": 2 << 30,
                                        "memory_max_bytes": 3 << 30, "swap_max_bytes": 0,
                                        "guard_sha256": digest(GUARD / "report.json")})
        stage.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "publish", "check"))
    action = parser.parse_args().action
    if action == "check":
        verify_seal(score105.PUBLIC)
        score105.check(score105.PUBLIC)
    else:
        {"worker": worker, "launch": launch, "publish": publish}[action]()
