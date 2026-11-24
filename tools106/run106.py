"""Run the single-CPU archive readback, publish a complete attempt, or verify it from public data."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import PRESSURE, THREADS, require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read, write
from tools106 import readback106 as bench


BASE = ROOT / "evidence/history106"
MANIFEST = bench.PLAN.with_name("sources.sha256")
FILES = ("measured.npz", "reference101.npz", "reference100.npz", "input_audit.json", "predictions.npz", "report.json")


def registration():
    for line in MANIFEST.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        require(digest(ROOT / relative) == expected, f"source changed: {relative}")


def locations(attempt):
    require(1 <= attempt <= 99, "invalid attempt")
    return (BASE / f"readback_{attempt:02d}", BASE / f"guard_{attempt:02d}", f"d1-terrain-history106-{attempt:02d}")


def worker(attempt):
    from tools27.resource_guard27 import read_memory
    from tools106.geometry100 import extract_reference100
    from tools106.history106 import extract_measured, extract_oracle
    folder, _, unit = locations(attempt)
    scope = scope_for(unit)
    registration()
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and os.getpriority(os.PRIO_PROCESS, 0) == 10
            and len(os.sched_getaffinity(0)) == 1 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4 << 30,
            "worker must start in the admitted single-CPU unit")
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "input_audit.json", dict(measured=extract_measured(folder / "measured.npz"),
          reference101=extract_oracle(folder / "reference101.npz"), reference100=extract_reference100(folder / "reference100.npz")))
    bench.record(folder)
    write(folder / "seal.json", seal(folder))


def launch(attempt):
    folder, guard, unit = locations(attempt)
    registration()
    plan = read(bench.PLAN)
    require(digest(PRESSURE) == plan["pressure_supervisor_sha256"], "pressure supervisor changed")
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not folder.exists() and not guard.exists(), "previous attempt preserved")
        env = dict.fromkeys(THREADS, "1")
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", f"--unit={unit}",
                   "--property=Nice=10", "--property=CPUWeight=25", "--property=OOMPolicy=kill",
                   "--property=MemoryHigh=2147483648", "--property=MemoryMax=3221225472", "--property=MemorySwapMax=0",
                   f"--working-directory={ROOT}", *(f"--setenv={k}={v}" for k, v in env.items()),
                   sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                   "--timeout", str(plan["budget"]["worker_timeout_s"]), "--",
                   sys.executable, "-B", "-m", "tools106.run106", "worker", "--attempt", str(attempt)]
        subprocess.run(command, check=True)


def seal(folder):
    return dict(status="COMPLETE", manifest_sha256=digest(MANIFEST), files={f: digest(folder / f) for f in FILES})


def publish(attempt):
    from tools88.export88 import resources
    folder, guard, unit = locations(attempt)
    registration()
    require(read(folder / "seal.json") == seal(folder) and read(folder / "report.json")["status"] == "COMPLETE",
            "unsealed readback")
    usage = resources(guard)
    require(read(guard / "report.json")["scope"] == scope_for(unit) and usage["minimum_effective_bytes"] >= 2 << 30,
            "incorrect unit or runtime reserve")
    require(not bench.PUBLIC.exists(), "public readback exists")
    with TemporaryDirectory(dir=bench.PUBLIC.parent, prefix=".history106-") as temporary:
        stage = Path(temporary) / "complete"
        shutil.copytree(folder, stage)
        require(read(stage / "seal.json") == seal(stage), "incomplete publication")
        write(stage / "resources.json", {**usage, **read(bench.PLAN)["resource_limits"], "unit": unit,
                                         "guard_sha256": digest(guard / "report.json")})
        stage.rename(bench.PUBLIC)


def check():
    folder = bench.PUBLIC
    registration()
    require(read(folder / "seal.json") == seal(folder), "incomplete public readback")
    usage = read(folder / "resources.json")
    require(usage["startup_effective_bytes"] >= 4 << 30 and usage["minimum_effective_bytes"] >= 2 << 30
            and len(usage["cpu_affinity"]) == 1, "invalid archive resource receipt")
    saved = read(folder / "report.json")
    pred = bench.measured_predictions(folder / "measured.npz")
    with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(bench.COLUMNS), "unexpected sealed columns")
        for k in bench.COLUMNS:
            if k.endswith("_mm"):
                np.testing.assert_allclose(archive[k], pred[k], atol=1e-7, rtol=0., equal_nan=True)
            else:
                np.testing.assert_array_equal(archive[k], pred[k])
    output = bench.diagnose(folder, pred)
    for k in ("per_source", "retrospective_checks", "checked_sources_passed", "original_gate_against_nearest"):
        require(output[k] == saved[k], f"changed result: {k}")
    rows = output["old_masked_exit_footprints"]
    require(len(rows) == len(saved["old_masked_exit_footprints"]), "changed audit coverage")
    for a, b in zip(rows, saved["old_masked_exit_footprints"], strict=True):
        for k, v in a.items():
            if isinstance(v, float):
                np.testing.assert_allclose(v, b[k], atol=1e-7, rtol=0.)
            else:
                require(v == b[k], f"changed audit: {k}")
    require(saved["external_reference_pose_only"] == bench.external_scores(folder / "reference100.npz"), "external readback changed")
    require(saved["predictions_sha256"] == digest(folder / "predictions.npz") and
            all(saved[k] == 0 for k in ("native_decisions", "new_rays", "model_updates", "policy_updates")), "invalid counters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "publish", "check"))
    parser.add_argument("--attempt", default=1, type=int)
    args = parser.parse_args()
    if args.action == "check":
        check()
    else:
        {"launch": launch, "worker": worker, "publish": publish}[args.action](args.attempt)
