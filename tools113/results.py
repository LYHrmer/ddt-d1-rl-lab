"""Package frozen models, their prospective predictions and the independent native measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools91.launch91 import digest, read, require
from tools111.results import install
from tools112.surface_results import install_json
from tools113 import model
from tools113.audit import verify_episode
from tools113.common import BASE, PLAN, PUBLIC, ROOT, SEEDS, grid_case, registration, roster


METRICS = ("cruise_m_s", "stop_peak_m_s")
PHASES = ("cruise", "braking")
STAGES = (*(f"fit_{s}" for s in SEEDS), "forecast", *(f"native_{s}" for s in SEEDS), "video")
MOVIE = ROOT / "docs/assets/speed_world113.mp4"
MIN_CHANGE = .001
MAX_RELATIVE_ERROR = .25


def score_change(predicted, observed):
    result = {"predicted_m_s": predicted, "observed_m_s": observed,
              "sign_correct": None, "relative_error": None, "passed": False}
    if any(v is not None and not math.isfinite(v) for v in (predicted, observed)):
        raise ValueError("nonfinite signed speed change")
    if predicted is None or observed is None:
        return dict(result, status="UNAVAILABLE")
    if abs(observed) < MIN_CHANGE:
        return dict(result, status="UNIDENTIFIABLE")
    sign = predicted * observed > 0.
    relative = abs(predicted - observed) / max(abs(predicted), abs(observed))
    passed = sign and relative <= MAX_RELATIVE_ERROR
    return dict(result, status="PASS" if passed else "FAIL", passed=passed,
                sign_correct=sign, relative_error=relative)


def metric_difference(on, off):
    return {k: None if on.get(k) is None or off.get(k) is None else on[k] - off[k] for k in METRICS}


def compare_metrics(actual, stored):
    for key in METRICS:
        if actual[key] is None or stored[key] is None:
            require(actual[key] is stored[key], f"missing {key}")
        else:
            np.testing.assert_allclose(actual[key], stored[key], rtol=0., atol=1e-8, err_msg=key)


def _item(stage, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ("", ".", ".."),
            "expected an archive filename")
    return MOVIE if stage == "video" and name.endswith(".mp4") else PUBLIC / stage / name


def copy_stage(stage, record):
    if record is None:
        reservations = sorted(BASE.glob(stage + "_??/reservation.json"))
        require(len(reservations) <= 1, f"duplicate reserved {stage}")
        if not reservations:
            started = [p.parent for p in BASE.glob(f"guard_{stage}_??/report.json") if read(p)["child_started"]]
            if started:
                require(len(started) == 1, f"multiple interrupted workers for {stage}")
                return {"status": "INTERRUPTED_BEFORE_RESERVATION", "unit": started[0].name.removeprefix("guard_")}
            return {"status": "NOT_STARTED"}
        out = reservations[0].parent
        status = "PARTIAL"
        guard = None
    else:
        out, guard, _ = record
        status = "COMPLETE"
    files = {}
    for path in sorted(out.iterdir()):
        if not path.is_file():
            continue
        if status == "PARTIAL" and path.suffix == ".mp4":
            # Its checksum preserves the local interrupted stream identity without distributing it as the film.
            continue
        sha = digest(path)
        install(path, _item(stage, path.name), sha)
        files[path.name] = sha
    require("reservation.json" in files, f"missing {stage} reservation")
    result = {"status": status, "unit": out.name, "files": files}
    if guard is not None:
        result["guard"] = guard.name
    elif stage == "video":
        result["partial_streams"] = {p.name: digest(p) for p in sorted(out.glob("*.mp4"))}
    return result


def copy_resources():
    guards = {}
    for folder in sorted(BASE.glob("guard_*")):
        report = folder / "report.json"
        files = {}
        for name in ("report.json", "resources.jsonl"):
            path = folder / name
            if path.is_file():
                sha = digest(path)
                install(path, PUBLIC / "resources" / folder.name / name, sha)
                files[name] = sha
        if report.is_file():
            entry = read(report)
            guards[folder.name] = {"status": entry["status"], "child_started": entry["child_started"],
                                   "error": entry.get("error"), "files": files}
        else:
            guards[folder.name] = {"status": "NO_FINAL_RECEIPT", "child_started": None,
                                   "error": None, "files": files}
    return guards


def verify_resources(guards, stages):
    completed = {s["guard"] for s in stages.values() if s["status"] == "COMPLETE"}
    require(completed <= set(guards), "missing completed guard")
    for name, entry in guards.items():
        unit = name.removeprefix("guard_")
        stage, attempt = unit.rsplit("_", 1)
        require(name == "guard_" + unit and stage in STAGES and len(attempt) == 2 and attempt.isdecimal(),
                "unregistered guard")
        directory = PUBLIC / "resources" / name
        require(set(entry["files"]) <= {"report.json", "resources.jsonl"}, "unexpected resource files")
        for filename, sha in entry["files"].items():
            require(digest(directory / filename) == sha, "changed resource journal")
        if "report.json" not in entry["files"]:
            require(name not in completed and entry["status"] == "NO_FINAL_RECEIPT" and entry["child_started"] is None,
                    "missing final resource receipt")
            continue
        r = read(directory / "report.json")
        require((r["status"], r["child_started"], r.get("error")) ==
                (entry["status"], entry["child_started"], entry["error"]), "invalid resource result")
        if name not in completed:
            require(r["status"] != "COMPLETE", "unaccounted completed worker")
            if r["child_started"]:
                require(stages[stage]["status"] in ("PARTIAL", "INTERRUPTED_BEFORE_RESERVATION")
                        and stages[stage]["unit"] == unit,
                        "started interrupted worker absent from partial reservations")
            continue
        require(stages[stage]["unit"] == unit and stages[stage]["guard"] == name and
                r["status"] == "COMPLETE" and r["returncode"] == 0 and r["child_started"] and
                r["child_reaped"] and r["owned_cgroup_children_empty"] and len(r["cpu_affinity"]) == 1,
                "invalid completed resource receipt")
        readings = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
        startup = [s for s in readings if s["stage"] == "startup"]
        require(startup and startup[-1]["effective_available_bytes"] >= 4 << 30, "startup reserve")
        for sample in readings:
            own = sample["owned"]
            require(not sample["errors"] and sample["effective_available_bytes"] >= 2 << 30
                    and (own["high_bytes"], own["max_bytes"], own["swap_max_bytes"],
                         own["swap_current_bytes"]) == (2 << 30, 3 << 30, 0, 0)
                    and not own["events"]["oom_kill"], "runtime reserve or cgroup limits")


def stage_report(index, stage):
    s = index[stage]
    require(s["status"] in ("COMPLETE", "PARTIAL", "INTERRUPTED_BEFORE_RESERVATION", "NOT_STARTED"),
            "unknown stage status")
    for file, sha in s.get("files", {}).items():
        require(digest(_item(stage, file)) == sha, f"changed {stage}/{file}")
    if s["status"] == "NOT_STARTED":
        require("files" not in s and "unit" not in s, "unstarted stage has evidence")
        return None
    if s["status"] == "INTERRUPTED_BEFORE_RESERVATION":
        require("files" not in s and s["unit"].startswith(stage + "_"), "invalid unreserved interruption")
        return None
    reservation = read(_item(stage, "reservation.json"))
    require(reservation["stage"] == stage and reservation["registration_sha256"] == registration(),
            "reservation differs from registration")
    if s["status"] == "PARTIAL":
        return None
    row = read(_item(stage, "report.json"))
    outputs = {name: sha for name, sha in s["files"].items() if name not in ("report.json", "reservation.json")}
    require(row["stage"] == stage and row["status"] == "COMPLETE" and
            row["registration_sha256"] == reservation["registration_sha256"] and
            row["inputs"] == reservation["inputs"] and row["files"] == outputs, "stage provenance mismatch")
    return row


def verify_inputs(stages):
    for stage in STAGES:
        if stages[stage]["status"] in ("NOT_STARTED", "INTERRUPTED_BEFORE_RESERVATION"):
            continue
        inputs = read(_item(stage, "reservation.json"))["inputs"]
        if stage == "forecast":
            required = {f"fit_{seed}" for seed in SEEDS}
        elif stage.startswith("native_"):
            required = {"forecast"}
        elif stage == "video":
            required = {f"native_{SEEDS[0]}"}
        else:
            required = set()
        predecessors = set(STAGES[:STAGES.index(stage)])
        if stage.startswith("fit_"):
            predecessors = set()
        require(required <= set(inputs) <= predecessors, f"missing or circular dependencies for {stage}")
        for dependency, sha in inputs.items():
            require(stages[dependency]["status"] == "COMPLETE"
                    and sha == stages[dependency]["files"]["report.json"], f"unsealed input to {stage}")
        if stage.startswith("native_"):
            seal = read(PLAN.parent / "forecast_seal.json")
            forecast = read(_item("forecast", "report.json"))
            require(seal["forecast_report_sha256"] == inputs["forecast"] and
                    seal["registration_sha256"] == registration() and seal["rows"] == forecast["result"]["rows"] and
                    seal["forecast_inputs"] == forecast["inputs"] and seal["actor_sha256"] == forecast["result"]["actor_sha256"],
                    "public forecasts differ from the prospective seal")
            require(seal["sealed_at_unix_ns"] == forecast["completed_at_unix_ns"] <
                    read(_item(stage, "reservation.json"))["reserved_at_unix_ns"], "late native forecast seal")


def verify_fit(seed, stage, fit, cases):
    directory = PUBLIC / stage
    ref, _ = grid_case(seed)
    train = sorted((r for r in cases if r[0] != seed), key=lambda r: (r[1], r[0], Path(r[2]).name))
    sources = [dict(seed=s, domain=d, file=Path(p).name, sha256=digest(p)) for s, d, p in train]
    require(len(train) == 9 and fit["training"] == sources and fit["heldout_seed"] == seed
            and fit["heldout_sha256"] == digest(ref) and fit["world_updates"] == 100
            and fit["policy_updates"] == fit["physics_steps"] == 0, "invalid training split or budget")
    train_traces = [model.read_trajectory(*r) for r in train]
    initial, _, _ = model._initial(train_traces, seed)
    plan = model._batches(train_traces, seed)
    require(fit["sample_schedule_sha256"] == hashlib.sha256(plan.tobytes()).hexdigest()
            and fit["sample_seed"] == 1130000 + seed, "changed paired sampling")
    log = directory / "fit_updates.jsonl"
    archive = directory / "heldout_h16.npz"
    require(digest(log) == fit["updates_sha256"] and digest(archive) == fit["holdout_scores_sha256"],
            "fit journal or held-out outputs differ")
    updates = [json.loads(line) for line in log.read_text().splitlines()]
    expected = [(arm, k, event) for arm in model.ARMS for k in range(1, 51) for event in ("attempt", "complete")]
    require([(r["arm"], r["update"], r["event"]) for r in updates] == expected, "wrong update roster")
    test = model.read_trajectory(seed, "grid", ref)
    with np.load(archive, allow_pickle=False) as stored:
        for arm in model.ARMS:
            receipt = fit["arms"][arm]
            done = [r for r in updates if r["arm"] == arm and r["event"] == "complete"]
            require(all(math.isfinite(r["loss"]) and r["loss"] >= 0. and
                        math.isfinite(r["gradient_norm"]) and r["gradient_norm"] >= 0. for r in done)
                    and (receipt["first_batch_loss"], receipt["last_batch_loss"]) ==
                        (done[0]["loss"], done[-1]["loss"]), "invalid optimizer journal")
            path = directory / (arm + ".npz")
            require(receipt["updates"] == 50 and receipt["checkpoint_sha256"] == digest(path), "invalid checkpoint")
            world = model.load_model(path)
            require(world.heldout == seed and world.uses_action == (arm == "action"), "wrong world model arm")
            for attr in ("reference_state", "reference_target", "target_scale", "state_scale"):
                np.testing.assert_allclose(getattr(world, attr), getattr(initial, attr), rtol=0., atol=1e-10)
            signals, scores = model.heldout_predictions(world, test)
            for key, values in signals.items():
                np.testing.assert_allclose(stored[arm if key == "prediction" else key], values, rtol=0., atol=1e-8)
            for phase in PHASES:
                original = receipt["holdout"][phase]
                require(scores[phase]["windows"] == original["windows"] > 0, "invalid held-out windows")
                np.testing.assert_allclose(scores[phase]["mae_m_s"], original["mae_m_s"], rtol=0., atol=1e-8)
    a, b = (fit["arms"][arm]["holdout"] for arm in ("action", "masked"))
    passed = {p: a[p]["mae_m_s"] < b[p]["mae_m_s"] for p in PHASES}
    require(fit["passed"] == passed and fit["both_phases_passed"] == all(passed.values()), "invalid H16 verdict")
    return {"windows": {p: a[p]["windows"] for p in PHASES},
            "errors_m_s": {arm: {p: fit["arms"][arm]["holdout"][p]["mae_m_s"] for p in PHASES}
                           for arm in model.ARMS}, "passed": passed, "both_passed": all(passed.values())}


def verify_forecasts(record, fits):
    if record is None:
        return {}
    from tools113.forecast import replay_prediction
    result = record["result"]
    rows = result["rows"]
    require([r["seed"] for r in rows] == list(SEEDS) and result["actor_forwards"] == 3600, "forecast roster")
    for row in rows:
        seed = row["seed"]
        require(seed in fits, "forecast without sealed fit")
        checkpoint = PUBLIC / f"fit_{seed}/action.npz"
        path = PUBLIC / "forecast" / f"pred_{seed}.npz"
        require(row["model_sha256"] == digest(checkpoint) and row["file"] == path.name and
                row["sha256"] == digest(path), "wrong prospective model or trace")
        reference, _ = grid_case(seed)
        reconstructed = replay_prediction(path, checkpoint, reference, seed)
        for name in ("off", "on", "delta"):
            compare_metrics(reconstructed[name], row[name])
        compare_metrics(metric_difference(row["on"], row["off"]), row["delta"])
    return {r["seed"]: r for r in rows}


def native_case(seed, record, forecast):
    ref, ref_row = grid_case(seed)
    off = {"cruise_m_s": ref_row["motion"]["speed_m_s"], "stop_peak_m_s": ref_row["motion"]["stop_tail_max_m_s"]}
    on = None
    motion = None
    if record is not None:
        require(forecast is not None, "native data without a frozen prospective forecast")
        row = record["result"]
        require(row["seed"] == seed, "native seed mismatch")
        on = verify_episode(PUBLIC / f"native_{seed}/trace.npz", row, ref, ref_row)
        motion = row["motion"]
    observed = metric_difference(on or {}, off)
    predicted = {k: None for k in METRICS} if forecast is None else forecast["delta"]
    scores = {k: score_change(predicted[k], observed[k]) for k in METRICS}
    return {"seed": seed, "baseline": off, "actual": on, "observed_delta": observed,
            "predicted_off": None if forecast is None else forecast["off"],
            "predicted_on": None if forecast is None else forecast["on"],
            "predicted_delta": predicted, "scores": scores, "motion": motion,
            "control_passed": motion is not None and motion["ticks"] == 600 and motion["passed"],
            "model_passed": all(v["passed"] for v in scores.values())}


def calculate(stages):
    require(set(stages) == set(STAGES), "registered stages missing or duplicated")
    records = {k: stage_report(stages, k) for k in STAGES}
    verify_inputs(stages)
    inputs = roster()
    fits = {}
    totals = dict(world_updates=0, policy_updates=0, decisions=0, servo=0, physics=0, packets=0, ray_queries=0)
    for seed in SEEDS:
        stage = f"fit_{seed}"
        if records[stage] is not None:
            fit = records[stage]["result"]
            fits[seed] = verify_fit(seed, stage, fit, inputs)
            totals["world_updates"] += fit["world_updates"]
    forecasts = verify_forecasts(records["forecast"], fits)
    cases = []
    for seed in SEEDS:
        stage = f"native_{seed}"
        record = records[stage]
        case = native_case(seed, record, forecasts.get(seed))
        case["stage_status"] = stages[stage]["status"]
        cases.append(case)
        if record is not None:
            for k, amount in record["result"]["counts"].items():
                totals[k] += amount
    partial = {k for k in STAGES if stages[k]["status"] == "PARTIAL"}
    reserved = totals.copy()
    reserved["world_updates"] += 100 * sum(k.startswith("fit_") for k in partial)
    for counter, upper in dict(decisions=600, servo=3000, physics=6000, packets=72, ray_queries=74880).items():
        reserved[counter] += upper * sum(k.startswith("native_") for k in partial)
    require(reserved["world_updates"] <= 300 and reserved["physics"] <= 18000 and reserved["decisions"] <= 1800
            and reserved["ray_queries"] <= 224640, "reserved work exceeds budget")
    if records["video"] is not None:
        require(records[f"native_{SEEDS[0]}"] is not None, "missing paired video source")
        video = records["video"]["result"]
        ref, _ = grid_case(SEEDS[0])
        native = records[f"native_{SEEDS[0]}"]["result"]
        require(video["seed"] == SEEDS[0] and video["trace_sha256"] == [digest(ref), native["trace_sha256"]]
                and video["physics"] == video["cover_frames"] == 0 and video["sha256"] == digest(MOVIE)
                and 0 < video["bytes"] == MOVIE.stat().st_size <= 5_000_000
                and video["frames"] == int(video["stream"]["nb_read_frames"])
                and video["stream"]["codec_name"] == "h264" and "llvmpipe" in video["driver"]["renderer"].lower(),
                "invalid measured film")
    else:
        require(not MOVIE.exists(), "public video without a completed render")
    summary = {"fit_folds": len(fits), "fit_both_passed": sum(v["both_passed"] for v in fits.values()),
               "forecast_seeds": len(forecasts), "complete_12s_pairs": sum(c["actual"] is not None for c in cases),
               "control_passed": sum(c["control_passed"] for c in cases), "model_passed": sum(c["model_passed"] for c in cases),
               "predicted_sign_correct": {k: sum(c["scores"][k]["sign_correct"] is True for c in cases) for k in METRICS},
               "prediction_passed": {k: sum(c["scores"][k]["passed"] for c in cases) for k in METRICS},
               "complete_native_stages": sum(records[f"native_{s}"] is not None for s in SEEDS),
               "unidentifiable": {k: sum(c["scores"][k]["status"] == "UNIDENTIFIABLE" for c in cases) for k in METRICS},
               "partial_stages": sorted(partial),
               "interrupted_before_reservation": [k for k in STAGES if stages[k]["status"] == "INTERRUPTED_BEFORE_RESERVATION"],
               "unstarted_stages": [k for k in STAGES if stages[k]["status"] == "NOT_STARTED"],
               "confirmed_counts": totals, "including_partial_upper_bounds": reserved}
    return {"fits": {str(s): v for s, v in fits.items()}, "cases": cases, "summary": summary}


def publish():
    from tools113.run import recorded
    stages = {stage: copy_stage(stage, recorded(stage)) for stage in STAGES}
    guards = copy_resources()
    verify_resources(guards, stages)
    record = {"registration_sha256": registration(), "stages": stages, "guards": guards, **calculate(stages)}
    install_json(PUBLIC / "results.json", record)


def verify():
    record = read(PUBLIC / "results.json")
    require(record["registration_sha256"] == registration(), "registration hash differs")
    verify_resources(record["guards"], record["stages"])
    rebuilt = calculate(record["stages"])
    for key, value in rebuilt.items():
        compare_tree(value, record[key])
    return rebuilt["summary"]


def compare_tree(actual, stored):
    if isinstance(actual, float):
        np.testing.assert_allclose(actual, stored, rtol=0., atol=1e-8)
    elif isinstance(actual, dict):
        require(actual.keys() == stored.keys(), "different public score fields")
        for key, value in actual.items():
            compare_tree(value, stored[key])
    elif isinstance(actual, list):
        require(len(actual) == len(stored), "different public score counts")
        for left, right in zip(actual, stored):
            compare_tree(left, right)
    else:
        require(actual == stored, "different public score or source")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))
