"""Launch the preregistered compound-route stages on one guarded CPU."""
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
from tools116.run import SEEDS, protocol as parent_protocol
from tools116.response_replay import INDEX, INDEX_SHA


BASE = ROOT / "evidence/compound117"
PUBLIC = ROOT / "reports/compound117"
PLAN = ROOT / "docs/experiments117/registration.json"
SPEED_STAGES = tuple(f"speed_{s}" for s in SEEDS)
PARKING_STAGES = tuple(f"parking_{s}" for s in SEEDS)
STAGES = (*SPEED_STAGES, *PARKING_STAGES, "video")
TIMEOUT = 600


def protocol(stage):
    require(stage in STAGES, "unknown compound117 stage")
    registration = read(PLAN)
    require(registration["seeds"] == list(SEEDS) and registration["supervisor_sha256"] == GUARD_SHA
            and registration["parent_native_registration_sha256"] == parent_protocol("native_111101")
            and registration["parent_results_sha256"] == INDEX_SHA == digest(INDEX), "different parent protocol")
    for name, sha in registration["sources_sha256"].items():
        require(digest(ROOT / name) == sha, "changed registration input: " + name)
    return digest(PLAN)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, "invalid compound117 attempt")
    tag = f"{stage}_{attempt:02d}"
    return BASE / tag, BASE / ("guard_" + tag), "d1-compound117-" + tag.replace("_", "-")


def completed(stage):
    for path in sorted(BASE.glob(stage + "_??/report.json")):
        number = int(path.parent.name.rsplit("_", 1)[1])
        out, guard, unit = locations(stage, number)
        if not (guard / "report.json").is_file():
            continue
        receipt = read(guard / "report.json")
        if receipt["status"] != "COMPLETE":
            continue
        record, header = read(path), read(out / "reservation.json")
        require(record["status"] == "COMPLETE" and header["stage"] == stage and header["unit"] == unit
                and header["attempt"] == number and header["registration_sha256"] == protocol(stage)
                and all(record[k] == v for k, v in header.items()) and receipt["child_started"]
                and receipt["returncode"] == 0 and receipt["child_reaped"] and receipt["owned_cgroup_children_empty"]
                and receipt["scope"] == scope_for(unit) and receipt["supervisor_sha256"] == GUARD_SHA,
                "unbound compound117 stage completion")
        for name, sha in record["files"].items():
            require(Path(name).name == name and digest(out / name) == sha, "changed worker output " + name)
        return out, record
    return None


def parking_required(records):
    """Demand every driving and perception gate, then require at least one outstanding stop."""
    return (len(records) == len(SEEDS) and [r["seed"] for r in records] == list(SEEDS)
            and all(r["status"] == "COMPLETE" and not r["parking"] and r["driving_pass"]
                    and r["motion"]["ticks"] == 600 for r in records) and any(not r["gate_pass"] for r in records))


def eligible(stage):
    if stage in SPEED_STAGES:
        return True
    if stage == "video":
        return completed(SPEED_STAGES[0]) is not None
    require(stage in PARKING_STAGES, "unknown native stage")
    sources = [completed(s) for s in SPEED_STAGES]
    return all(sources) and parking_required([source[1]["result"] for source in sources])


def reference_path(seed, parking):
    if parking:
        source = completed(f"speed_{seed}")
        require(source is not None and source[1]["result"]["driving_pass"], "ineligible paired speed run")
        return source[0] / "trace.npz"
    case, = [r for r in read(INDEX)["cases"] if r["seed"] == seed]
    path = INDEX.parent / f"native_{seed}/trace.npz"
    require(digest(INDEX) == INDEX_SHA and case["stage_status"] == "COMPLETE" and case["complete_horizon"]
            and digest(path) == case["trace_sha256"], "incomplete or mismatched round-116 reference")
    return path


def worker(stage, attempt):
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    require(Path("/proc/self/cgroup").read_text().strip() == "0::" + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == "1" for k in THREADS) and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and read_memory(scope)["effective_available_bytes"] >= 4 << 30, "guarded worker admission mismatch")
    require(eligible(stage), "closed stage gate")
    header = dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=protocol(stage),
                  reserved_at_unix_ns=time.time_ns())
    out.mkdir(exist_ok=False)
    write(out / "reservation.json", header)
    start = time.monotonic()
    try:
        if stage == "video":
            from tools117.video import render
            result = render(out, completed(SPEED_STAGES[0])[0])
        else:
            from tools117.native import episode
            phase, seed = stage.split("_")
            seed, parking = int(seed), phase == "parking"
            result = episode(out, seed, reference_path(seed, parking), parking)
        files = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != "reservation.json"}
        write(out / "report.json", dict(**header, status="COMPLETE", result=result, files=files,
                                        elapsed_s=time.monotonic()-start, completed_at_unix_ns=time.time_ns()))
    except BaseException as exc:
        write(out / "worker_partial.json", dict(**header, status="PARTIAL", error=repr(exc)))
        raise


def launch(stage):
    registration = protocol(stage)
    require(digest(PRESSURE) == GUARD_SHA, "unexpected resource supervisor")
    BASE.mkdir(parents=True, exist_ok=True)
    settings = dict.fromkeys(THREADS, "1")
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT/".python-deps", ROOT/"training", ROOT))),
                    PYTHONDONTWRITEBYTECODE="1", MUJOCO_GL="disable", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    if stage == "video":
        from tools102.mesa_video102 import GL_ENV
        settings.update(GL_ENV)
    with (ROOT / "evidence/uphill89/serial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed(stage):
            return True
        if not eligible(stage):
            print("Stage gate closed:", stage, flush=True)
            return False
        for attempt in range(1, 100):
            out, guard, unit = locations(stage, attempt)
            admission = BASE / f"admission_{out.name}.json"
            if guard.exists():
                old = read(guard / "report.json")
                require(not old["child_started"] and not out.exists(), "preserve started stage: " + stage)
                continue
            require(not out.exists(), "preserve existing attempt: " + out.name)
            if admission.exists():
                continue
            print("Waiting for sustained headroom:", unit, flush=True)
            try:
                wait_for_admission()
            except TimeoutError as exc:
                write(admission, dict(stage=stage, attempt=attempt, unit=unit, child_started=False,
                                      registration_sha256=registration, error=str(exc), time_ns=time.time_ns()))
                continue
            args = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--slice=app.slice", "--unit="+unit,
                    "-p", "Nice=10", "-p", "CPUWeight=25", "-p", "OOMPolicy=kill", "-p", "MemoryHigh=2G",
                    "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", f"--working-directory={ROOT}",
                    *(f"--setenv={k}={v}" for k,v in settings.items()), sys.executable, "-B", str(PRESSURE),
                    "--scope", scope_for(unit), "--output", str(guard), "--timeout", str(TIMEOUT), "--",
                    sys.executable, "-B", "-m", "tools117.run", "worker", stage, "--attempt", str(attempt)]
            result = subprocess.run(args)
            require((guard / "report.json").is_file(), f"{unit} did not return a resource receipt")
            receipt = read(guard / "report.json")
            if result.returncode == 0:
                require(completed(stage) is not None, "missing worker receipt")
                print("Complete:", stage, flush=True)
                return True
            if receipt["child_started"] or out.exists():
                print("Retained started attempt:", stage, receipt["error"], flush=True)
                return False
            require(any(t in str(receipt["error"]) for t in ("effective memory", "admission not sustained", "PSI")),
                    "unexpected error before start")
    raise RuntimeError("no unreserved attempt for " + stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker"))
    parser.add_argument("stage", choices=(*STAGES, "speed", "parking"))
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.stage, args.attempt)
    else:
        stages = {"speed": SPEED_STAGES, "parking": PARKING_STAGES}.get(args.stage, (args.stage,))
        raise SystemExit(0 if all(launch(s) for s in stages) else 2)
