"""Run the authorized parking comparison once per seed, under the original resource guard."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools27.resource_guard27 import read_memory
from tools91.launch91 import PRESSURE, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import wait_for_admission
from tools114.common import BASE, BLIND, DEV, ROOT, baseline, registration


PLAN = ROOT / "docs/experiments114/closed_loop.json"
STAGES = tuple(f"native_{seed}" for seed in (*DEV, *BLIND)) + ("film",)


def protocol():
    plan = read(PLAN)
    require(plan["parking_registration_sha256"] == registration()
            and plan["development"] == list(DEV) and plan["conditional_blind"] == list(BLIND)
            and plan["supervisor_sha256"] == GUARD_SHA, "parking cohort or parent protocol changed")
    for name, sha in plan["sources_sha256"].items():
        require(digest(ROOT / name) == sha, "parking source changed: " + name)
    return digest(PLAN)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid parking stage or attempt")
    tag = f"{stage}_{attempt:02d}"
    return BASE / stage, BASE / ("guard_" + tag), "d1-parking114-" + tag.replace("_", "-")


def completed(stage):
    out = BASE / stage
    if not (out / "report.json").is_file():
        return False
    row = read(out / "report.json")
    _, guard, unit = locations(stage, row["attempt"])
    if not (guard / "report.json").is_file():
        return False
    supervisor = read(guard / "report.json")
    if supervisor["status"] != "COMPLETE":
        return False
    header = read(out / "reservation.json")
    require(row["status"] == "COMPLETE" and row["stage"] == stage and row["unit"] == unit
            and row["registration_sha256"] == protocol() and all(row[k] == v for k, v in header.items())
            and supervisor["child_started"] and supervisor["returncode"] == 0 and supervisor["child_reaped"]
            and supervisor["owned_cgroup_children_empty"] and supervisor["scope"] == scope_for(unit)
            and supervisor["supervisor_sha256"] == GUARD_SHA, "unbound parking worker receipt")
    for name, sha in row["files"].items():
        path = out / name
        require(path.parent == out and digest(path) == sha, "changed " + name)
    return True


def development_passed():
    from tools114.closed_results import assess
    for seed in DEV:
        stage = f"native_{seed}"
        if not completed(stage):
            return False
        motion = read(BASE / stage / "report.json")["result"]["motion"]
        if motion["ticks"] != 600 or not motion["passed"] or motion["failed"]:
            return False
        if not assess(seed, BASE / stage)["gate_pass"]:
            return False
    return True


def eligible(stage):
    if stage == "film":
        return completed(f"native_{DEV[0]}")
    if int(stage.split("_")[1]) in BLIND:
        return development_passed()
    return True


def worker(stage, attempt):
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4 << 30, "worker admission mismatch")
    sha = protocol()
    require(eligible(stage), "parking prerequisites unavailable")
    header = dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=sha,
                  reserved_at_unix_ns=time.time_ns())
    if stage != "film":
        seed = int(stage.split("_")[1])
        if seed in DEV:
            header["baseline_sha256"] = baseline(seed)[1]["trace_sha256"]
    if stage == "film":
        header["input_report_sha256"] = digest(BASE / f"native_{DEV[0]}" / "report.json")
    out.mkdir(exist_ok=False)
    write(out / "reservation.json", header)
    try:
        if stage == "film":
            media = out / "media"
            subprocess.run([sys.executable, "-B", "-m", "tools114.parking_video", "--native-dir",
                            str(BASE / f"native_{DEV[0]}"), "--output-dir", str(media)], check=True)
            result = read(media / "video.json")
            # Keep the original per-file names in the worker receipt.
            files = {p.name: digest(p) for p in media.iterdir() if p.is_file()}
            for name in files:
                (media / name).rename(out / name)
            media.rmdir()
        else:
            from tools114.native import episode
            result = episode(out, seed)
            files = {"trace.npz": digest(out / "trace.npz")}
            require(result["trace_sha256"] == files["trace.npz"], "trace digest differs from episode")
        write(out / "report.json", dict(**header, status="COMPLETE", result=result, files=files,
                                        completed_at_unix_ns=time.time_ns()))
    except BaseException as exc:
        write(out / "worker_partial.json", dict(**header, status="PARTIAL", error=repr(exc)))
        raise


def launch(stage):
    protocol()
    require(digest(PRESSURE) == GUARD_SHA, "changed resource supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    env = dict.fromkeys(THREADS, "1")
    env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
               CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    if stage == "film":
        from tools102.mesa_video102 import GL_ENV
        env.update(GL_ENV)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed(stage):
            return True
        if not eligible(stage):
            print("Closed gate:", stage, flush=True)
            return False
        for attempt in range(1, 100):
            out, guard, unit = locations(stage, attempt)
            if guard.exists():
                previous = read(guard / "report.json")
                if previous["child_started"]:
                    print("Retained started case:", stage, previous["status"], flush=True)
                    return False
                continue
            require(not out.exists(), "preserve existing reservation " + stage)
            print("Waiting for admission:", unit, flush=True)
            wait_for_admission()
            cmd = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice",
                   "--unit=" + unit, "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "OOMPolicy=kill",
                   "-p", "MemoryHigh=2G", "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0",
                   f"--working-directory={ROOT}", *(f"--setenv={k}={v}" for k, v in env.items()),
                   sys.executable, "-B", str(PRESSURE), "--scope", scope_for(unit), "--output", str(guard),
                   "--timeout", "600", "--", sys.executable, "-B", "-m", "tools114.closed_run", "worker",
                   stage, "--attempt", str(attempt)]
            print("Dispatch", unit, flush=True)
            code = subprocess.run(cmd).returncode
            receipt = read(guard / "report.json")
            if code == 0:
                require(completed(stage), "incomplete worker return")
                return True
            if receipt["child_started"] or out.exists():
                print("Retained failed case:", stage, receipt.get("error"), flush=True)
                return False
            require(any(term in str(receipt["error"]) for term in
                        ("effective memory", "admission not sustained", "PSI")), "supervisor failed before work")
    raise RuntimeError("no fresh admission directory for " + stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.stage, args.attempt)
    else:
        raise SystemExit(0 if launch(args.stage) else 2)
