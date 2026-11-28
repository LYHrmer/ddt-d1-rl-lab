"""Publish and independently verify the registered parking episodes and their source-bound film."""
import argparse
import json
import re
from pathlib import Path

from tools91.launch91 import digest, read, require
from tools109.run109 import GUARD_SHA
from tools111.results import install
from tools112.surface_results import install_json
from tools113.results import compare_tree
from tools114.closed_results import assess, development_passed, verify_public
from tools114.closed_run import PLAN, STAGES, locations, protocol
from tools114.common import BASE, BLIND, DEV, PUBLIC, ROOT, baseline
from tools114.results import resource_summary


MOVIE = ROOT / "docs/assets/parking114.mp4"
INDEX = PUBLIC / "closed_loop.json"
LIMIT = dict(decisions=600, servo=3000, physics=6000, packets=72, ray_queries=74880)


def _guards(root):
    entries = {}
    folders = sorted((*root.glob("guard_native_*"), *root.glob("guard_film_*")))
    for folder in folders:
        stage, suffix = folder.name[6:].rsplit("_", 1)
        require(stage in STAGES and re.fullmatch(r"[0-9]{2}", suffix), "unregistered parking guard")
        attempt = int(suffix)
        _, _, unit = locations(stage, attempt)
        receipt = read(folder / "report.json")
        scope = re.fullmatch(r"/user\.slice/user-(\d+)\.slice/user@(\d+)\.service/app\.slice/"
                             + re.escape(unit) + r"\.service", receipt["scope"])
        command = receipt["command"]
        require(scope and scope[1] == scope[2] and receipt["supervisor_sha256"] == GUARD_SHA
                and receipt["schema"] == "pressure-supervisor83-v1" and receipt["timeout_s"] == 600
                and Path(command[0]).name in ("python3", "python3.10")
                and command[1:] == ["-B", "-m", "tools114.closed_run", "worker", stage, "--attempt", str(attempt)],
                "guard does not belong to the registered worker: " + unit)
        files = {p.name: digest(p) for p in (folder/"report.json", folder/"resources.jsonl") if p.is_file()}
        started = receipt["child_started"]
        require(type(started) is bool and receipt["status"] in ("COMPLETE", "INCOMPLETE"), "invalid guard receipt")
        resources = None
        if receipt["status"] == "COMPLETE":
            require(started and receipt["error"] is None, "successful guard without a clean worker")
            resources = resource_summary(folder, receipt)
            samples = [json.loads(line) for line in (folder/"resources.jsonl").read_text().splitlines()]
            require(all(s["self_scope"] == s["owned"]["path"] == receipt["scope"] for s in samples)
                    and resources["minimum_effective_bytes"] == min(s["effective_available_bytes"] for s in samples),
                    "resource samples differ from their guard scope or minimum")
        if started:
            require("resources.jsonl" in files and receipt.get("child_reaped") and
                    receipt.get("owned_cgroup_children_empty"), "worker cleanup or resource journal missing")
        entries[folder.name] = dict(stage=stage, attempt=attempt, status=receipt["status"], child_started=started,
                                    error=receipt["error"], files=files, resources=resources)
    return entries


def _stage(stage, root, guards, movie, registration):
    attempts = [(key, g) for key, g in guards.items() if g["stage"] == stage]
    require([g["attempt"] for _, g in attempts] == list(range(1, len(attempts)+1)), "missing guard attempt")
    started = [(key, g) for key, g in attempts if g["child_started"]]
    require(len(started) <= 1 and (not started or started[-1] == attempts[-1]), "repeated a started case")
    folder = root / stage
    if not started:
        require(not folder.exists(), "stage output without a started guard")
        return dict(status="NOT_STARTED", guard=None, files={}, reserved_at_unix_ns=None, completed_at_unix_ns=None)

    key, guard = started[0]
    finished = guard["status"] == "COMPLETE"
    status = "COMPLETE" if finished else "PARTIAL"
    files = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.suffix != ".mp4"}
    if stage == "film" and movie.parent != folder:
        require(not any(folder.rglob("*.mp4")), "duplicate film in the public receipt directory")
    if finished and stage == "film":
        files[MOVIE.name] = digest(movie)
    expected = {"reservation.json", "report.json", "trace.npz"} if stage != "film" else {
        "reservation.json", "report.json", "video.json", MOVIE.name}
    if finished:
        require(set(files) == expected, "missing or additional completed stage files: " + stage)
    if "reservation.json" not in files:
        require(not finished and not files, "unreserved stage contains output")
        return dict(status="INTERRUPTED_BEFORE_RESERVATION", guard=key, files={}, reserved_at_unix_ns=None,
                    completed_at_unix_ns=None)
    header = read(folder / "reservation.json")
    _, _, unit = locations(stage, guard["attempt"])
    require(header["stage"] == stage and header["attempt"] == guard["attempt"] and header["unit"] == unit
            and header["registration_sha256"] == registration and header["reserved_at_unix_ns"] > 0,
            "unbound worker reservation")
    if stage != "film":
        seed = int(stage.split("_")[1])
        require(header.get("baseline_sha256") == (baseline(seed)[1]["trace_sha256"] if seed in DEV else None),
                "reservation names another baseline")
    completed_ns = None
    if "report.json" in files:
        report = read(folder / "report.json")
        require(all(report.get(k) == v for k, v in header.items()) and report["status"] == "COMPLETE"
                and report["completed_at_unix_ns"] >= header["reserved_at_unix_ns"], "unbound worker report")
        if finished:
            require(report["files"] == {n: files[n] for n in expected - {"reservation.json", "report.json"}},
                    "worker output digest differs")
            completed_ns = report["completed_at_unix_ns"]
    if "worker_partial.json" in files:
        partial = read(folder / "worker_partial.json")
        require(not finished and partial["status"] == "PARTIAL" and
                all(partial.get(k) == v for k,v in header.items()), "unbound partial receipt")
    return dict(status=status, guard=key, files=files, reserved_at_unix_ns=header["reserved_at_unix_ns"],
                completed_at_unix_ns=completed_ns)


def _case(seed, stage, folder, guards, expected=None):
    if stage["status"] == "COMPLETE":
        return assess(seed, folder) if expected is None else verify_public(folder, seed, expected)
    counts = None
    files = stage["files"]
    reason = guards[stage["guard"]]["error"] if stage["guard"] else None
    if "partial_counts.json" in files:
        partial = read(folder/"partial_counts.json")
        counts, reason = partial["counts"], partial["error"]
    elif "report.json" in files:
        counts = read(folder/"report.json")["result"]["counts"]
    elif "worker_partial.json" in files:
        reason = read(folder/"worker_partial.json")["error"]
    return dict(seed=seed, stage_status=stage["status"], complete=False, paired=seed in DEV, gate_pass=False,
                failed=[stage["status"].lower()], motion=None, stop=None, comparison=None, wiring=None,
                counts=counts, error=reason)


def _cohorts(rows, stages):
    dev = rows[:len(DEV)]
    require(all(r["stage_status"] != "NOT_STARTED" for r in dev), "development has an unstarted case")
    opened = development_passed(dev)
    for row in rows[len(DEV):]:
        stage = stages[f'native_{row["seed"]}']
        if opened:
            require(stage["status"] != "NOT_STARTED", "the authorized blind cohort is still incomplete")
            if stage["reserved_at_unix_ns"] is not None:
                last = max(stages[f"native_{seed}"]["completed_at_unix_ns"] for seed in DEV)
                require(stage["reserved_at_unix_ns"] >= last, "blind native case preceded three development passes")
        else:
            require(stage["status"] == "NOT_STARTED", "blind case started while development gate was closed")
            row.update(stage_status="CONDITIONAL_NOT_OPENED", failed=["development_gate_closed"])
    return opened


def _budgets(rows, stages, plan):
    measured = dict.fromkeys(LIMIT, 0)
    unknown = []
    reserved = 0
    for row in rows:
        stage = stages[f'native_{row["seed"]}']
        if stage["reserved_at_unix_ns"] is None:
            continue
        reserved += 1
        counts = row["counts"]
        if counts is None:
            unknown.append(row["seed"])
            continue
        require(counts.keys() == LIMIT.keys() and all(type(v) is int and 0 <= v <= LIMIT[k] for k,v in counts.items()),
                "native counts outside the registered per-case budget")
        for name in measured:
            measured[name] += counts[name]
    upper = {k: v + LIMIT[k]*len(unknown) for k,v in measured.items()}
    limits = plan["maximum_totals"]
    require(reserved <= limits["episodes"] and all(upper[k] <= limits[k] for k in upper if k != "packets"),
            "total native budget exceeded")
    return dict(reserved_episodes=reserved, counted=measured, unknown_counts_seeds=unknown, upper_bound=upper,
                world_model_steps=0, world_updates=0, policy_updates=0)


def _film(root, movie, stage, first):
    if first["status"] != "COMPLETE":
        require(stage["status"] == "NOT_STARTED", "film without the prescribed native case")
        return dict(status="SOURCE_UNAVAILABLE")
    require(stage["status"] != "NOT_STARTED", "the prescribed native case still needs its film")
    if stage["status"] != "COMPLETE":
        return dict(status=stage["status"])
    source_report = read(root / f"native_{DEV[0]}" / "report.json")
    row = source_report["result"]
    receipt = read(root / "film/report.json")
    info = read(root / "film/video.json")
    require(stage["reserved_at_unix_ns"] >= first["completed_at_unix_ns"] and receipt["result"] == info
            and receipt["input_report_sha256"] == info["native_report_sha256"] == first["files"]["report.json"]
            and info["trace_sha256"] == [baseline(DEV[0])[1]["trace_sha256"], first["files"]["trace.npz"]]
            and info["seed"] == DEV[0] and info["last_ticks"] == [600, row["motion"]["ticks"]]
            and info["actor_sha256"] == row["actor_sha256"]
            and info["renderer_sha256"] == digest(ROOT/"tools114/parking_video.py")
            and info["status"] == "COMPLETE" and info["filename"] == movie.name == MOVIE.name
            and info["sha256"] == stage["files"][MOVIE.name]
            and 0 < info["bytes"] == movie.stat().st_size <= 5_000_000 and info["cover_frames"] == info["physics"] == 0,
            "film bytes, worker, renderer or matched source pair differ")
    stream = info["stream"]
    require(info["fps"] == 25 and stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and (stream["width"], stream["height"]) == (1280,720) and stream["r_frame_rate"] == "25/1"
            and int(stream["nb_read_frames"]) == info["frames"] == sum(s["frames"] for s in info["shots"])
            and info["frames"] > 0 and abs(info["duration_s"] - info["frames"]/25) < .05,
            "incorrect recorded playback stream")
    return {k: info[k] for k in ("status", "filename", "desktop_filename", "seed", "trace_sha256",
                                   "sha256", "bytes", "frames", "duration_s", "native_report_sha256")}


def _index(root, guard_root, movie, expected=None):
    registration = protocol()
    plan = read(PLAN)
    guards = _guards(guard_root)
    stages = {s: _stage(s, root, guards, movie, registration) for s in STAGES}
    intervals = sorted((v["reserved_at_unix_ns"],v["completed_at_unix_ns"]) for v in stages.values()
                       if v["completed_at_unix_ns"] is not None)
    require(all(left[1] <= right[0] for left,right in zip(intervals,intervals[1:])), "registered workers overlapped")
    if movie.parent != root/"film" and stages["film"]["status"] != "COMPLETE":
        require(not movie.exists(), "unrelated public film for an incomplete stage")
    seeds = (*DEV,*BLIND)
    rows = [_case(seed, stages[f"native_{seed}"], root/f"native_{seed}", guards,
                  expected["cases"][i] if expected is not None else None) for i,seed in enumerate(seeds)]
    opened = _cohorts(rows, stages)
    if not opened:
        require(not any(g["stage"] in {f"native_{s}" for s in BLIND} for g in guards.values()),
                "blind admissions attempted with a closed development gate")
    film = _film(root, movie, stages["film"], stages[f"native_{DEV[0]}"])
    summary = dict(development_passes=sum(r["gate_pass"] for r in rows[:len(DEV)]), blind_opened=opened,
                   blind_passes=sum(r["gate_pass"] for r in rows[len(DEV):]),
                   partial_stages=[s for s,e in stages.items() if e["status"] in ("PARTIAL", "INTERRUPTED_BEFORE_RESERVATION")],
                   admission_failures=[k for k,g in guards.items() if not g["child_started"]],
                   film_delivered=film["status"] == "COMPLETE", budget=_budgets(rows,stages,plan))
    return dict(registration_sha256=registration, stages=stages, guards=guards, cases=rows, summary=summary, film=film)


def publish():
    record = _index(BASE, BASE, BASE/"film"/MOVIE.name)
    for stage, entry in record["stages"].items():
        for name, sha in entry["files"].items():
            dest = MOVIE if stage == "film" and name == MOVIE.name else PUBLIC/stage/name
            install(BASE/stage/name, dest, sha)
    for name, entry in record["guards"].items():
        for file, sha in entry["files"].items():
            install(BASE/name/file, PUBLIC/"resources"/name/file, sha)
    verify(record)
    install_json(INDEX, record)


def verify(record=None):
    record = read(INDEX) if record is None else record
    rebuilt = _index(PUBLIC, PUBLIC/"resources", MOVIE, record)
    compare_tree(rebuilt, record)
    return rebuilt["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
