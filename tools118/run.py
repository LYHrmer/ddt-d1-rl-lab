"""Run the fixed D1 controller in three one-shot, single-CPU blind trials."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools27.resource_guard27 import read_memory
from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import wait_for_admission
from tools117.run import protocol as parent_protocol


PARENT = "621ba6dcc7cd633ffb11485708efb6b1202d9c28"
PARENT_RESULTS = ROOT / "reports/compound117/results.json"
BASE, PUBLIC = ROOT / "evidence/compound118", ROOT / "reports/compound118"
PLAN = ROOT / "docs/experiments118/registration.json"
SEEDS = tuple(int.from_bytes(hashlib.sha256(f"D1-compound118:{PARENT}:{i}".encode("ascii")).digest()[:4], "big")
              for i in (1, 2, 3))
NATIVE_STAGES = tuple(f"blind_{s}" for s in SEEDS)
STAGES = (*NATIVE_STAGES, "video")
TIMEOUT = 600


def protocol(stage):
    require(stage in STAGES, "unregistered stage")
    reg = read(PLAN)
    require(reg["controller_commit"] == PARENT and reg["seeds"] == list(SEEDS) and len(set(SEEDS)) == 3
            and reg["parent_results_sha256"] == digest(PARENT_RESULTS)
            and reg["parent_registration_sha256"] == parent_protocol("parking_111101")
            and reg["supervisor_sha256"] == GUARD_SHA, "changed locked controller or blind roster")
    parent = read(ROOT / "docs/experiments117/registration.json")
    require(reg["thresholds"] == parent["thresholds"] and reg["bounds"] == parent["bounds"]
            and reg["layout"] == parent["layout"] and reg["preselected_video_seed"] == SEEDS[0],
            "changed course, acceptance criteria, resource bounds or display start")
    for filename, sha in reg["sources_sha256"].items():
        require(digest(ROOT / filename) == sha, "registered source changed: " + filename)
    return digest(PLAN)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid blind attempt")
    tag = f"{stage}_{attempt:02d}"
    return BASE / tag, BASE / ("guard_" + tag), "d1-compound118-" + tag.replace("_", "-")


def completed(stage):
    for path in sorted(BASE.glob(stage + "_??/report.json")):
        attempt = int(path.parent.name.rsplit("_", 1)[1])
        out, guard, unit = locations(stage, attempt)
        if not (guard / "report.json").is_file():
            continue
        supervisor = read(guard / "report.json")
        if supervisor["status"] != "COMPLETE":
            continue
        record, header = read(path), read(out / "reservation.json")
        require(record["status"] == "COMPLETE" and header["stage"] == stage and header["attempt"] == attempt
                and header["unit"] == unit and header["registration_sha256"] == protocol(stage)
                and all(record[k] == v for k, v in header.items()) and supervisor["scope"] == scope_for(unit)
                and supervisor["returncode"] == 0 and supervisor["child_reaped"] and supervisor["child_started"]
                and supervisor["owned_cgroup_children_empty"] and supervisor["supervisor_sha256"] == GUARD_SHA,
                "incomplete blind stage or supervisor")
        for name, sha in record["files"].items():
            require(Path(name).name == name and digest(out / name) == sha, "changed stage output")
        return out, record
    return None


def worker(stage, attempt):
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope and len(os.sched_getaffinity(0)) == 1
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and all(os.environ.get(k) == "1" for k in THREADS)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "" and read_memory(scope)["effective_available_bytes"] >= 4 << 30,
            "worker outside admitted single-CPU unit")
    header = dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=protocol(stage),
                  reserved_at_unix_ns=time.time_ns())
    out.mkdir(exist_ok=False)
    write(out / "reservation.json", header)
    start = time.monotonic()

    def interrupted(signum, _frame):
        raise InterruptedError(f"worker interrupted by signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        if stage == "video":
            from tools118.video import render
            source = completed(NATIVE_STAGES[0])
            require(source is not None, "missing preselected video source")
            result = render(out, source[0])
        else:
            from tools118.native import episode
            result = episode(out, int(stage.split("_")[1]))
        files = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "reservation.json"}
        write(out / "report.json", dict(**header, status="COMPLETE", result=result, files=files,
                                        elapsed_s=time.monotonic()-start, completed_at_unix_ns=time.time_ns()))
    except BaseException as error:
        write(out / "worker_partial.json", dict(**header, status="PARTIAL", error=repr(error)))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def guarded_command(stage, attempt):
    _, guard, unit = locations(stage, attempt)
    env = dict.fromkeys(THREADS, "1")
    env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / ".python-deps", ROOT / "training", ROOT))),
               PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="disable", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    if stage == "video":
        from tools102.mesa_video102 import GL_ENV
        env.update(GL_ENV)
    return ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit="+unit,
            "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "OOMPolicy=kill", "-p", "MemoryHigh=2G",
            "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", f"--working-directory={ROOT}",
            *(f"--setenv={k}={v}" for k, v in env.items()), sys.executable, "-B", str(PRESSURE),
            "--scope", scope_for(unit), "--output", str(guard), "--timeout", str(TIMEOUT), "--",
            sys.executable, "-B", "-m", "tools118.run", "worker", stage, "--attempt", str(attempt)]


def launch(stage):
    registration = protocol(stage)
    require(digest(PRESSURE) == GUARD_SHA, "changed resource supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed(stage):
            return True
        if stage == "video" and not completed(NATIVE_STAGES[0]):
            return False
        for attempt in range(1, 100):
            out, guard, unit = locations(stage, attempt)
            admission = BASE / ("admission_" + out.name + ".json")
            if guard.exists():
                prior = read(guard / "report.json")
                if prior["child_started"] or out.exists():
                    return False
                continue
            require(not out.exists(), "preserve existing reservation: " + out.name)
            if admission.exists():
                continue
            print("Waiting for sustained headroom:", unit, flush=True)
            try:
                wait_for_admission()
            except TimeoutError as error:
                write(admission, dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=registration,
                                      child_started=False, error=str(error), time_ns=time.time_ns()))
                continue
            code = subprocess.run(guarded_command(stage, attempt)).returncode
            receipt = read(guard / "report.json")
            if code == 0:
                require(completed(stage) is not None, "missing worker completion")
                print("Complete:", stage, flush=True)
                return True
            if receipt["child_started"] or out.exists():
                return False
            require(any(t in str(receipt["error"]) for t in ("effective memory", "admission not sustained", "PSI")),
                    "unexpected error before admission")
    raise RuntimeError("no unreserved admission slot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    parser.add_argument("stage", choices=(*STAGES, "blind"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.stage, args.attempt)
    else:
        stages = NATIVE_STAGES if args.stage == "blind" else (args.stage,)
        completed_stages = [launch(stage) for stage in stages]
        raise SystemExit(0 if all(completed_stages) else 2)
