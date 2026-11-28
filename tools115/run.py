"""Dispatch the registered parking probes once each, under the shared D1 resource guard."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools27.resource_guard27 import read_memory
from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import wait_for_admission
from tools114.common import BLIND, DEV


BASE = ROOT / "evidence/parking115"
PLANS = ROOT / "docs/experiments115"
SEEDS = (*DEV, *BLIND)
STAGES = ("policy", *(f"native_{s}" for s in SEEDS))
TIMEOUT = 600


def protocol(stage):
    require(stage in STAGES, "unregistered stage")
    family = stage.split("_")[0]
    method = PLANS / "verification.json"
    registered = PLANS / f"{family}.json"
    design, implementation = read(method), read(registered)
    require(design["paired_starts"] == list(SEEDS) and design["supervisor_sha256"] == GUARD_SHA
            and implementation["family"] == family and implementation["design_sha256"] == digest(method),
            "changed study or stage registration")
    for item in (design, implementation):
        for name, sha in item["sources_sha256"].items():
            require(digest(ROOT / name) == sha, "changed registered source: " + name)
    return digest(registered)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "unregistered stage or attempt")
    tag = f"{stage}_{attempt:02d}"
    return BASE / tag, BASE / ("guard_" + tag), "d1-parking115-" + tag.replace("_", "-")


def worker(stage, attempt):
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4 << 30, "worker admission mismatch")
    header = dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=protocol(stage),
                  reserved_at_unix_ns=time.time_ns())
    out.mkdir(exist_ok=False)
    write(out / "reservation.json", header)
    start = time.monotonic()
    try:
        if stage == "policy":
            from tools115.policy_probe import BUNDLE, IMU_RECEIPT, JOURNAL, run_probe
            result = run_probe(out)
            files = (BUNDLE, IMU_RECEIPT, JOURNAL)
        else:
            from tools115.native import episode
            result = episode(out, int(stage.split("_")[1]))
            files = ("trace.npz",)
            require(result["trace_sha256"] == digest(out / files[0]), "trace digest differs from episode")
        write(out / "report.json", dict(**header, status="COMPLETE", result=result,
                                        files={n: digest(out / n) for n in files},
                                        completed_at_unix_ns=time.time_ns(), elapsed_s=time.monotonic() - start))
    except BaseException as exc:
        write(out / "partial.json", dict(**header, status="PARTIAL", error=repr(exc)))
        raise


def launch(stage):
    registered = protocol(stage)
    require(digest(PRESSURE) == GUARD_SHA, "changed resource supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    settings = dict.fromkeys(THREADS, "1")
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
                    CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(1, 100):
            out, guard, unit = locations(stage, attempt)
            admission_file = BASE / f"admission_{out.name}.json"
            if guard.exists():
                prior = read(guard / "report.json")
                if prior["child_started"] or out.exists():
                    return prior["status"] == "COMPLETE" and (out / "report.json").is_file()
                continue
            require(not out.exists(), "preserve existing worker " + out.name)
            if admission_file.exists():
                continue
            print("Waiting for headroom", unit, flush=True)
            try:
                wait_for_admission()
            except TimeoutError as exc:
                write(admission_file, dict(stage=stage, attempt=attempt, unit=unit, child_started=False,
                                           registration_sha256=registered, error=str(exc), time_ns=time.time_ns()))
                continue
            command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice",
                       "--unit=" + unit, "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "OOMPolicy=kill",
                       "-p", "MemoryHigh=2G", "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0",
                       f"--working-directory={ROOT}", *(f"--setenv={k}={v}" for k, v in settings.items()),
                       sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                       "--timeout", str(TIMEOUT), "--", sys.executable, "-B", "-m", "tools115.run", "worker",
                       stage, "--attempt", str(attempt)]
            print("Dispatch", unit, flush=True)
            status = subprocess.run(command).returncode
            record = read(guard / "report.json")
            if status == 0:
                require(record["status"] == "COMPLETE" and record["child_started"] and record["returncode"] == 0
                        and record["child_reaped"] and record["owned_cgroup_children_empty"]
                        and read(out / "report.json")["registration_sha256"] == registered,
                        "missing guarded completion for " + stage)
                return True
            if record["child_started"] or out.exists():
                print("Retained started case", stage, record.get("error"), flush=True)
                return False
            require(any(term in str(record["error"]) for term in ("effective memory", "admission not sustained", "PSI")),
                    "supervisor failed before work")
    raise RuntimeError("no fresh admission directory for " + stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    parser.add_argument("stage", choices=(*STAGES, "all_native"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.stage, args.attempt)
    else:
        requested = STAGES[1:] if args.stage == "all_native" else (args.stage,)
        attempted = [launch(stage) for stage in requested]
        raise SystemExit(0 if all(attempted) else 2)
