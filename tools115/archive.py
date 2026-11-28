"""Package one-shot policy and native parking evidence and recompute each result from the public files."""
import argparse
import json
import math
from pathlib import Path

from tools91.launch91 import ROOT, digest, read, require
from tools109.run109 import GUARD_SHA
from tools111.results import install
from tools112.surface_results import install_json
from tools113.results import compare_tree
from tools114.common import BLIND, DEV
from tools114.results import resource_summary


BASE = ROOT / "evidence/parking115"
PUBLIC = ROOT / "reports/parking115"
INDEX = PUBLIC / "results.json"
SEEDS = (*DEV, *BLIND)
STAGES = ("policy", *(f"native_{s}" for s in SEEDS))
POLICY_FILES = {"imu_contract.json", "policy_probe.npz", "policy_calls.jsonl"}
LIMIT = dict(decisions=600, servo=3000, physics=6000, packets=72, ray_queries=74880)


def _key(tag):
    stage, sep, digits = tag.rpartition("_")
    require(sep and stage in STAGES and len(digits) == 2 and digits.isascii() and digits.isdecimal()
            and int(digits) > 0, "unknown parking attempt " + tag)
    return stage, int(digits)


def _status(guard, reserved, admission):
    if guard is None:
        return "PRE_ADMISSION_FAILURE" if admission and not reserved else "PARTIAL_WITHOUT_GUARD"
    require(type(guard["child_started"]) is bool and guard["status"] in ("COMPLETE", "INCOMPLETE"),
            "invalid supervisor outcome")
    if not guard["child_started"]:
        require(guard["status"] != "COMPLETE" and not reserved, "unstarted worker has a reservation or completion")
        return "PRE_ADMISSION_FAILURE"
    if guard["status"] == "COMPLETE":
        require(reserved, "successful worker without reservation")
        return "COMPLETE"
    return "PARTIAL" if reserved else "INTERRUPTED_BEFORE_RESERVATION"


def _relative(tag, status):
    return Path(_key(tag)[0]) if status == "COMPLETE" else Path("attempts") / tag


def _collect():
    tags = set()
    for stage in STAGES:
        tags.update(p.name for p in BASE.glob(stage + "_??") if p.is_dir())
        tags.update(p.name[6:] for p in BASE.glob("guard_" + stage + "_??") if p.is_dir())
        tags.update(p.stem[10:] for p in BASE.glob("admission_" + stage + "_??.json"))
    attempts = {}
    for tag in sorted(tags):
        _key(tag)
        stage_dir, guard = BASE / tag, BASE / ("guard_" + tag)
        files = {p.name: p for p in stage_dir.glob("*") if p.is_file()}
        for origin, name in ((guard/"report.json", "guard.json"), (guard/"resources.jsonl", "resources.jsonl"),
                             (BASE/("admission_" + tag + ".json"), "admission.json")):
            if origin.is_file():
                require(name not in files, "duplicate worker/guard artifact")
                files[name] = origin
        result = read(files["guard.json"]) if "guard.json" in files else None
        status = _status(result, "reservation.json" in files, "admission.json" in files)
        relative = _relative(tag, status)
        kept, partial = {}, {}
        for name, origin in sorted(files.items()):
            sha = digest(origin)
            if status != "COMPLETE" and origin.suffix not in (".json", ".jsonl"):
                partial[name] = sha
            else:
                install(origin, PUBLIC / relative / name, sha)
                kept[name] = sha
        attempts[tag] = dict(status=status, files=kept, local_partial_sha256=partial)
    return attempts


def _ledger(tag, entry, registration):
    from tools115.run import TIMEOUT, locations
    stage, number = _key(tag)
    unit = locations(stage, number)[2]
    folder = PUBLIC / _relative(tag, entry["status"])
    files = entry["files"]
    require(set(files) == {p.name for p in folder.glob("*") if p.is_file()}, "unaccounted attempt file")
    for name, sha in files.items():
        require(Path(name).name == name and digest(folder/name) == sha, "changed parking evidence " + name)
    partial = entry["local_partial_sha256"]
    require(not set(partial) & set(files) and all(Path(k).name == k and len(sha) == 64 for k, sha in partial.items()),
            "invalid partial artifact identity")
    guard = read(folder/"guard.json") if "guard.json" in files else None
    header = read(folder/"reservation.json") if "reservation.json" in files else None
    dispatch = read(folder/"admission.json") if "admission.json" in files else None
    status = _status(guard, header is not None, dispatch is not None)
    require(status == entry["status"], "attempt status differs from evidence")
    usage = None
    if header is not None:
        require(header["stage"] == stage and header["attempt"] == number and header["registration_sha256"] == registration
                and header["reserved_at_unix_ns"] > 0 and header["unit"] == unit, "unbound parking reservation")
    if guard is not None:
        cmd = guard["command"]
        require(guard["supervisor_sha256"] == GUARD_SHA and guard["new_physics_steps_in_supervisor"] == 0
                and guard["schema"] == "pressure-supervisor83-v1" and guard["timeout_s"] == TIMEOUT
                and cmd[1:] == ["-B", "-m", "tools115.run", "worker", stage, "--attempt", str(number)]
                and guard["scope"].endswith("/" + unit + ".service"), "wrong parking guard or worker")
        if status == "COMPLETE":
            usage = resource_summary(folder, guard)
            samples = [json.loads(line) for line in (folder/"resources.jsonl").read_text().splitlines()]
            require(all(s["self_scope"] == s["owned"]["path"] == guard["scope"] for s in samples)
                    and min(s["effective_available_bytes"] for s in samples) == usage["minimum_effective_bytes"],
                    "unbound memory samples")
    error = guard.get("error") if guard else None
    if dispatch is not None:
        require(dispatch["stage"] == stage and dispatch["attempt"] == number and dispatch["unit"] == unit
                and dispatch["registration_sha256"] == registration and dispatch["child_started"] is False
                and header is None and guard is None, "unbound pre-admission failure")
        error = dispatch.get("error") or dispatch.get("reason") or error or "pre-admission failure"
    ended = None
    if status == "COMPLETE":
        report = read(folder/"report.json")
        evidence = POLICY_FILES if stage == "policy" else {"trace.npz"}
        require(set(files) == evidence | {"reservation.json", "guard.json", "resources.jsonl", "report.json"}
                and not partial and report["status"] == "COMPLETE" and all(report.get(k) == v for k,v in header.items())
                and report["files"] == {k: files[k] for k in evidence}
                and math.isfinite(report["elapsed_s"]) and report["elapsed_s"] >= 0
                and report["completed_at_unix_ns"] >= header["reserved_at_unix_ns"], "wrong stage or output digest")
        ended = report["completed_at_unix_ns"]
    elif "partial.json" in files:
        failed = read(folder/"partial.json")
        require(header is not None and all(failed.get(k) == v for k,v in header.items())
                and failed["status"] == "PARTIAL", "unbound partial receipt")
        error = failed.get("error") or error
    return dict(stage=stage, attempt=number, status=status, resources=usage, error=error,
                reserved_at_unix_ns=header["reserved_at_unix_ns"] if header else None, completed_at_unix_ns=ended)


def _select(attempts):
    stages = {}
    for name in STAGES:
        sequence = sorted((k for k, s in attempts.items() if s["stage"] == name), key=lambda k: attempts[k]["attempt"])
        require([attempts[k]["attempt"] for k in sequence] == list(range(1, len(sequence)+1)),
                "missing prior admission attempt")
        started = [k for k in sequence if attempts[k]["status"] != "PRE_ADMISSION_FAILURE"]
        require(len(started) <= 1 and (not started or started[0] == sequence[-1]), "repeated a started stage")
        tag = started[0] if started else None
        stages[name] = dict(status=attempts[tag]["status"] if tag else "NOT_STARTED", attempt=tag,
                            admission_failures=[k for k in sequence if k != tag])
    return stages


def _policy(stage):
    if stage["status"] != "COMPLETE":
        return None
    from tools115.policy_probe import verify
    folder = PUBLIC / "policy"
    row = read(folder/"report.json")["result"]
    checked = verify(folder, row)
    require(checked["new_forwards"] == 0 and checked["archived_forwards"] == row["actor_forwards"] <= 18,
            "wrong inference budget")
    return row


def _previous_wheel_gaps():
    """Descriptive check of the previous raw commands seen at the observed decision."""
    import numpy as np
    from tools115.policy_probe import BUNDLE, FEEDBACK_COLUMNS, TICK
    with np.load(PUBLIC / "policy" / BUNDLE, allow_pickle=False) as data:
        # The original arm's 11 ascending frames each carry the preceding raw action.
        original = data["inputs"][:, 0]
        wheels = 5. * original[:, :, FEEDBACK_COLUMNS].astype(np.float64)
    gaps = wheels[:, :, 1] - wheels[:, :, [0, 2, 3]].mean(axis=2)
    return dict(first_previous_action_tick=TICK - gaps.shape[1], last_previous_action_tick=TICK - 1,
                samples=int(gaps.size), samples_with_negative_gap=int((gaps < 0.).sum()),
                minimum_gap_rad_s=float(gaps.min()), maximum_gap_rad_s=float(gaps.max()))


def _native(seed, stage):
    if stage["status"] != "COMPLETE":
        counts = None
        if stage["attempt"]:
            directory = PUBLIC / "attempts" / stage["attempt"]
            if (directory/"partial_counts.json").is_file():
                counts = read(directory/"partial_counts.json")["counts"]
            elif (directory/"report.json").is_file():
                counts = read(directory/"report.json").get("result", {}).get("counts")
        return dict(seed=seed, stage_status=stage["status"], complete_horizon=False, gate_pass=False,
                    motion=None, tilt=None, counts=counts)
    from tools115.native_compare import score_candidate
    folder = PUBLIC / f"native_{seed}"
    row = read(folder/"report.json")["result"]
    score = score_candidate(folder/"trace.npz", seed)
    require(row["seed"] == seed and row["status"] == "COMPLETE" and row["cohort"] == "paired_reused_round114"
            and row["intervention"] == "release_rf_contrast_once" and row["trace_sha256"] == score["trace_sha256"]
            and row["reference_sha256"] == score["reference_sha256"], "wrong native intervention or pair")
    for name in ("motion", "counts"):
        compare_tree(score[name], row[name])
    compare_tree(score, row["assessment"])
    return dict(stage_status="COMPLETE", **score)


def tilt_verdict(cases, threshold, required):
    values = [c["tilt"]["delta_peak_deg"] for c in cases if c["tilt"] is not None and c["tilt"]["window_complete"]]
    above = sum(v >= threshold for v in values)
    below = sum(v <= -threshold for v in values)
    if len(values) < required:
        decision = "MISSING_WINDOWS"
    elif above == required:
        decision = "ORIGINAL_CONTRAST_REDUCED_PEAK_TILT"
    elif below == required:
        decision = "ORIGINAL_CONTRAST_INCREASED_PEAK_TILT"
    else:
        decision = "DIRECTION_OR_MAGNITUDE_DID_NOT_PASS"
    return dict(windows=len(values), required=required, threshold_deg=threshold,
                increased_at_least_threshold=above, decreased_at_least_threshold=below, outcome=decision)


def _index(artifacts):
    from tools115.run import protocol
    from tools115.signals import audit
    signals = PUBLIC / "signals.json"
    compare_tree(audit(), read(signals))
    registrations = {s: protocol(s) for s in STAGES}
    attempts = {tag: _ledger(tag, entry, registrations[_key(tag)[0]]) for tag, entry in artifacts.items()}
    stages = _select(attempts)
    intervals = sorted((a["reserved_at_unix_ns"], a["completed_at_unix_ns"]) for a in attempts.values()
                       if a["completed_at_unix_ns"] is not None)
    require(all(a[1] <= b[0] for a,b in zip(intervals, intervals[1:])), "overlapping completed workers")
    policy = _policy(stages["policy"])
    cases = [_native(s, stages[f"native_{s}"]) for s in SEEDS]
    counts = dict.fromkeys(LIMIT, 0)
    unknown = []
    for c in cases:
        if c["counts"] is None:
            if c["stage_status"] != "NOT_STARTED":
                unknown.append(c["seed"])
            continue
        require(c["counts"].keys() == LIMIT.keys() and all(type(v) is int and 0 <= v <= LIMIT[k]
                for k,v in c["counts"].items()), "native counts outside one registered case")
        for name in counts:
            counts[name] += c["counts"][name]
    valid = [c for c in cases if c["tilt"] is not None and c["tilt"]["window_complete"]]
    native_plan = read(ROOT/"docs/experiments115/verification.json")["native"]
    tilt = tilt_verdict(cases, native_plan["minimum_relevant_degrees"], native_plan["required_pairs"])
    gate_names = ("horizon", "forward_speed", "one_metre_speeds", "lane", "all_wheels_exited",
                  "step_top_support", "stopped", "fall_free", "legal_support")
    gates = {k: sum(c["motion"] is not None and k not in c["motion"]["failed"] for c in cases) for k in gate_names}
    upper = {k: v+LIMIT[k]*len(unknown) for k,v in counts.items()}
    for k, maximum in (("decisions","max_actor_decisions"), ("servo","max_servo_ticks"),
                       ("physics","max_physics_steps"), ("ray_queries","max_ray_queries")):
        require(upper[k] <= native_plan[maximum], "native budget exceeded")
    summary = dict(registered_pairs=len(SEEDS), completed_native_stages=sum(c["stage_status"]=="COMPLETE" for c in cases),
                   full_native_horizons=sum(c["complete_horizon"] for c in cases), motion_passes=sum(c["gate_pass"] for c in cases),
                   tilt_windows=len(valid), peak_tilt_increased=sum(c["tilt"]["delta_peak_deg"] > 0. for c in valid),
                   original_motion_gate_passes=gates, tilt_verdict=tilt,
                   confirmed_native_counts=counts, unknown_partial_native_seeds=unknown,
                   native_counts_upper_bound=upper, policy_probe_forwards=policy["actor_forwards"] if policy else None,
                   policy_updates=0, world_updates=0, world_model_steps=0,
                   admission_failures=[k for k,v in attempts.items() if v["status"] == "PRE_ADMISSION_FAILURE"],
                   partial_stages=[k for k,v in stages.items() if v["status"] not in ("COMPLETE", "NOT_STARTED")])
    return dict(registrations=registrations, signals_sha256=digest(signals), attempts=artifacts,
                previous_wheel_gaps=_previous_wheel_gaps() if policy is not None else None,
                attempt_results=attempts, stages=stages, policy=policy,
                cases=cases, summary=summary,
                scope="Six reused paired round-114 starts on level terrain. The short tilt window measures a local "
                      "response to the one-decision action change. The fixed-input probe measures policy sensitivity.")


def publish():
    artifacts = _collect()
    require(artifacts, "no experiment receipts to publish")
    record = _index(artifacts)
    require(all(v["status"] != "NOT_STARTED" or v["admission_failures"] for v in record["stages"].values()),
            "registered stages have no outcome yet")
    install_json(INDEX, record)


def verify():
    saved = read(INDEX)
    rebuilt = _index(saved["attempts"])
    compare_tree(rebuilt, saved)
    return rebuilt["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    print(json.dumps(verify(), indent=2, allow_nan=False))
