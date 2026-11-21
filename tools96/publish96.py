"""Publish a sealed round 96 or recompute its public comparisons without a simulator or learned model."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools95.publish95 import FLAGS, RESET_FIELDS, checked, require, reset_hash, same_value
from tools96.common96 import BASE, PLAN, PLAN_SHA, files_for_seal, sources_for
from tools96.score96 import ARMS, HORIZON, MODELS, RESETS, SEEDS, TRAIN_RESETS, reward_m, score_models, summarize


REPORT = ROOT / "reports/paired_response96.json"
CURVES = ROOT / "reports/paired_response96_curves.npz"
STAGES = ("collection", "fit", "predictions", "native")
NATIVE_COUNTS = ("native_decisions", "servo_updates", "physics_substeps")
MODEL_COUNTS = ("attempted_model_steps", "completed_model_steps", "backward_attempted", "backward_completed")
FIT_COUNTS = ("attempted_model_steps", "completed_model_steps", "updates_attempted", "updates_completed")


def stage_record(stage):
    folder = BASE / (stage + "_01")
    path = folder / "seal.json"
    if not path.is_file():
        return {"status": "UNSEALED" if folder.is_dir() else "NOT_RUN", "seal_sha256": None}
    seal = read(path)
    files = files_for_seal(folder)
    require(files == seal["files"] and seal["protocol_sha256"] == digest(PLAN) == PLAN_SHA and seal["sources"] == sources_for(stage),
            f"changed {stage} files, producers or protocol")
    return {"status": "SEALED", "seal_sha256": digest(path), "file_count": len(files),
            "producer_sha256": seal["sources"], "record": {k: v for k, v in seal.items() if k not in ("files", "sources")}}


def guard_record(stage, policy):
    folder = BASE / ("guard_" + stage + "_01")
    path, log = folder / "report.json", folder / "resources.jsonl"
    if not path.is_file():
        return {"status": "NO_GUARD_RECEIPT"}
    record = read(path)
    samples = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    startup = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    runtime = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    limits = [policy[k] for k in ("memory_high_bytes", "memory_max_bytes", "swap_max_bytes")]
    matched = bool(samples) and all([s["owned"][k] for k in ("high_bytes", "max_bytes", "swap_max_bytes")] == limits
                                   and s["owned"]["swap_current_bytes"] == 0 and s["owned"]["path"] == record["scope"] for s in samples)
    row = {k: record.get(k) for k in ("status", "error", "scope", "returncode", "child_started", "owned_cgroup_children_empty",
           "cpu_affinity", "peak_group_bytes", "elapsed_s", "supervisor_sha256", "final_memory_events")}
    row.update(receipt_sha256=digest(path), resources_sha256=digest(log) if log.exists() else None, limits_verified=matched,
               startup_effective_bytes=startup[-1] if startup else None, runtime_min_effective_bytes=min(runtime, default=None))
    return row


def verify_guard(stage, row, policy):
    if row["status"] == "COMPLETE":
        require(row["limits_verified"] is True and row["returncode"] == 0 and row["child_started"] and row["owned_cgroup_children_empty"]
                and len(row["cpu_affinity"]) == policy["cpu_affinity_count"] and row["supervisor_sha256"] == policy["supervisor_sha256"]
                and row["scope"].endswith(f"/d1-paired96-{stage}-01.service") and len(row["resources_sha256"]) == len(row["receipt_sha256"]) == 64
                and row["startup_effective_bytes"] is not None and row["startup_effective_bytes"] >= policy["admission_min_bytes"]
                and row["runtime_min_effective_bytes"] is not None and row["runtime_min_effective_bytes"] >= policy["runtime_min_bytes"],
                f"{stage} resource guard differs")


def missing(ref, counts, status):
    return {"curve_key": ref, "status": status, "safe": False, "return_m": None, "ticks": 0,
            "source_sha256": None, "receipt_sha256": None, "error": None, **dict.fromkeys(counts, 0)}


def native_arm(stage, seed, reset, tag, starts, first, curves):
    folder = BASE / f"{stage}_01/seed_{seed}/reset_{reset}/{tag}"
    path = folder / "native.json"
    ref = f"{stage}_{seed}_{reset}_{tag}_x"
    row = missing(ref, NATIVE_COUNTS, "PARTIAL" if folder.exists() else "NOT_RUN")
    row.update(flags_key=ref + "_flags", origin_x_m=None, reset_sha256=None, first_endpoint_sha256=None)
    curves[ref], curves[row["flags_key"]] = np.empty(0, np.float64), np.empty((0, len(FLAGS)), bool)
    if not path.is_file():
        return row
    raw = read(path)
    row.update(receipt_sha256=digest(path), error=raw.get("error"), status="PARTIAL")
    file = raw.get("trace_file")
    if file is None:
        return row
    require(file in ("native/trace.npz", "native/partial_trace.npz"), f"unexpected native trace {path}")
    partial = file.endswith("partial_trace.npz") or raw.get("error") is not None
    with np.load(checked(folder / file, raw["trace_sha256"]), allow_pickle=False) as t:
        states, applied, issued = t["state45"], t["applied16"], t["issued16"]
        n = len(applied)
        require(int(t["reset_seed"]) == reset and int(t["requested_ticks"]) == HORIZON and int(t["completed_transitions"]) == n
                and n <= HORIZON and issued.shape == applied.shape == (n, 16) and states.ndim == 2 and states.shape[1] == 45
                and (len(states) == n+1 or (partial and n == len(states) == 0)) and np.isfinite(states).all()
                and states.dtype == np.float64 and np.isfinite(issued).all() and (np.abs(issued) <= 1).all() and int(t["explore_seed"]) == -1
                and not t["command3"].any() and not t["exploration16"].any() and float(t["wheel_offset_rad_s"]) == 0.,
                f"invalid native trace {path}")
        if len(states):
            reset_state = {k: t[k][:1] for k in RESET_FIELDS}
            signature = reset_hash(reset_state)
            require(starts.setdefault(reset, signature) == signature, "within-reset initial observation differs")
            row.update(reset_sha256=signature, origin_x_m=float(states[0, 0]))
        if n:
            require(not applied[0].any() and np.array_equal(applied[1:], issued[:-1]) and np.array_equal(issued, t["raw16"])
                    and np.array_equal(t["obs643"][1:, 41:57], issued), "native action FIFO differs")
            signature = reset_hash({"state45": states[1:2]})
            require(first.setdefault(reset, signature) == signature, "shared first physical endpoint differs")
            row["first_endpoint_sha256"] = signature
        require(all(t[k].shape == (n,) and t[k].dtype == bool for k in FLAGS), "native flag records differ")
        flags = np.stack([t[k] for k in FLAGS], axis=1)
        require(not (t["terminated"] | t["truncated"])[:-1].any(), "native continued through terminal")
        counts = [int(t[k]) for k in ("policy_steps", "servo_steps", "native_steps")]
        expected = [n, 5*n, 10*n]
        require(all(a >= b for a, b in zip(counts, expected)) if partial else counts == expected, "native clocks differ")
        curves[ref] = states[:, 0].copy()
        curves[row["flags_key"]] = flags
    safe = n == HORIZON and not flags.any() and not partial
    result = reward_m(curves[ref], row["origin_x_m"]) if row["origin_x_m"] is not None else None
    require(raw["ticks"] == n and all(bool(flags[:, i].any()) == raw[k] for i, k in enumerate(FLAGS)) and
            safe == raw["safe"] and same_value(result, raw["return_m"]), "native receipt and raw trajectory disagree")
    row.update(dict(zip(NATIVE_COUNTS, counts)), source_sha256=raw["trace_sha256"], safe=safe, ticks=n, return_m=result,
               status="PARTIAL" if partial else ("COMPLETE" if safe else "TERMINAL"))
    return row


def model_arm(seed, reset, model, tag, start, curves):
    folder = BASE / f"predictions_01/seed_{seed}/reset_{reset}/{model}"
    path = folder / f"{tag}.json"
    ref = f"{model}_{seed}_{reset}_{tag}_x"
    row = missing(ref, MODEL_COUNTS, "PARTIAL" if (folder/f"{tag}.npz").exists() else "NOT_RUN")
    row["directional_derivative"] = None
    curves[ref] = np.empty(0, np.float32)
    if path.is_file():
        raw = read(path)
        require((raw["seed"], raw["reset"], raw["alpha"], raw["file"], raw["model"]) == (seed, reset, ARMS[tag], tag + ".npz", model), "model arm identity")
        with np.load(checked(folder / raw["file"], raw["state_sha256"]), allow_pickle=False) as t:
            states = t["state45"]
        require(states.ndim == 2 and states.shape[1] == 45 and np.isfinite(states).all(), "invalid model states")
        if len(states):
            require(start is not None and np.array_equal(states[0], start["state45"][0].astype(np.float32)), "model initial state differs")
        x = curves[ref] = states[:, 0].copy()
        n = max(0, len(x)-1)
        result = reward_m(x, float(start["state45"][0, 0])) if start is not None else None
        ad = raw["directional_derivative"]
        require((ad is None or (tag == "zero" and math.isfinite(ad))) and raw["ticks"] == n and same_value(result, raw["return_m"])
                and n <= raw["completed_model_steps"] <= raw["attempted_model_steps"] <= HORIZON
                and (not raw["safe"] or (n == HORIZON and not raw.get("error"))), "inconsistent model receipt")
        row.update({k: raw[k] for k in MODEL_COUNTS}, safe=raw["safe"], status="COMPLETE" if raw["safe"] else "PARTIAL",
                   error=raw.get("error"), return_m=result, ticks=n, directional_derivative=ad, receipt_sha256=digest(path),
                   source_sha256=raw["state_sha256"])
    if tag == "zero":
        ad = row["directional_derivative"]
        row["ad_key"] = ref + "_ad"
        curves[row["ad_key"]] = np.array([] if ad is None else [ad], np.float64)
    return row


def fit_row(seed):
    folder = BASE / f"fit_01/seed_{seed}"
    path = folder / "receipt.json"
    if not path.exists():
        return {"seed": seed, "status": "PARTIAL" if folder.exists() else "NOT_RUN", "arms": {}, "receipt_sha256": None}
    row = read(path)
    require(row["seed"] == seed, "wrong fitted seed")
    row["receipt_sha256"] = digest(path)
    for model, arm in row.get("arms", {}).items():
        require(model in MODELS, "unexpected fitted model")
        if arm["status"] == "COMPLETE":
            require(arm["updates_attempted"] == arm["updates_completed"] == 50 and
                    arm["attempted_model_steps"] == arm["completed_model_steps"] == 12800, "incomplete checkpoint labelled complete")
            checkpoint = arm["checkpoint"]
            require(checkpoint["file"] == "world.pt" and len(checkpoint["weights_sha256"]) == len(checkpoint["adam_sha256"]) == 64,
                    "fitted checkpoint identity")
            checked(folder / model / checkpoint["file"], checkpoint["sha256"])
    if row["status"] == "COMPLETE":
        require(all(row["arms"].get(m, {}).get("status") == "COMPLETE" for m in MODELS), "missing comparison checkpoint")
    for pair in row.get("source", {}).get("new_pairs", []):
        for arm in pair.get("arms", {}).values():
            raw_path = Path(arm["path"])
            if raw_path.is_absolute():
                arm["path"] = str(raw_path.relative_to(ROOT))
    return row


def verify_fit(fit, collection, curves, plan):
    for name, arm in fit.get("arms", {}).items():
        require(name in MODELS and arm["lambda_pair"] == plan["arms"][name] and
                0 <= arm["updates_completed"] <= arm["updates_attempted"] <= 50 and
                0 <= arm["completed_model_steps"] <= arm["attempted_model_steps"] <= 12800, "fit arm counters or loss weight")
        if arm["status"] == "COMPLETE":
            require(arm["updates_attempted"] == arm["updates_completed"] == 50 and arm["checkpoint"] and
                    arm["attempted_model_steps"] == arm["completed_model_steps"] == 12800, "incomplete fit")
    if fit["status"] != "COMPLETE":
        return
    require(all(fit["arms"][m]["status"] == "COMPLETE" for m in MODELS) and
            fit["policy_updates"] == fit["ppo_updates"] == 0 and fit["batch"]["rng_seed"] == fit["seed"]+960000 and
            len(fit["batch"]["indices_sha256"]) == 64, "paired training receipts disagree with design")
    for key in ("initial_weights_sha256", "initial_adam_sha256"):
        require(fit["arms"]["control"][key] == fit["arms"]["paired"][key] and len(fit["arms"]["control"][key]) == 64, "different initial weights or Adam state")
    # The two original public reports identify the old training roster, checkpoint and fixed policy direction.
    previous = read(checked(ROOT / "reports/policy_direction95.json", plan["direction_report95_sha256"]))
    direction = next(d for d in previous["directions"] if d["seed"] == fit["seed"])
    summary94 = read(checked(ROOT / "reports/cold_start94.json", direction["parents"]["summary94_sha256"]))
    old_seed = next(s for s in summary94["seeds"] if s["seed"] == fit["seed"])
    old = old_seed["rounds"][0]
    source = fit["source"]
    require(source["world94"] == old["world_checkpoint"] and source["policy94"] == old_seed["initial"] and
            source["direction95_seal_sha256"] == plan["direction_seal95_sha256"] and
            source["direction95_receipt_sha256"] == direction["receipt_sha256"] and
            source["direction95_tensors_sha256"] == direction["delta_tensors_sha256"] and
            source["original_adam_sha256"] == fit["arms"]["control"]["initial_adam_sha256"] and
            source["world94"]["weights_sha256"] == fit["arms"]["control"]["initial_weights_sha256"], "wrong parent model, optimizer or direction")
    require([(r["reset"], r["sha256"]) for r in source["old_traces"]] == [(r["reset"], r["sha256"]) for r in old["fit_episodes"]]
            and len(source["old_traces"]) == 12, "old training data differs from the round-94 fit")
    new = {p["reset"]: p for p in source["new_pairs"]}
    require(len(source["new_pairs"]) == len(new) == 3 and set(new) == set(TRAIN_RESETS), "paired fit has an unexpected training reset")
    targets, new_roster = [], []
    for c in collection:
        if c["training_seed"] != fit["seed"]:
            continue
        minus, plus = (c["arms"][tag] for tag in ("m1", "p1"))
        require(all(a["status"] in ("COMPLETE", "TERMINAL") for a in (minus, plus)), "exceptional or missing train arm used")
        n = min(minus["ticks"], plus["ticks"])
        require(n >= 8, "training pair has no common eight-tick window")
        roster = new[c["reset"]]
        require(roster["windows"] == n-7 and all(roster["arms"][tag]["sha256"] == c["arms"][tag]["source_sha256"]
                for tag in ("m1", "p1")), "new training sources do not match shared native collection")
        x, y = (curves[a["curve_key"]] for a in (minus, plus))
        for tick in range(n-7):
            targets.append((y[tick+1:tick+9] - y[tick]) - (x[tick+1:tick+9] - x[tick]))
            new_roster.append((c["reset"], tick))
    target = np.stack(targets)
    norm = fit["normalization"]
    mse = float(np.mean(target**2))
    floor = plan["fit"]["denominator_floor_m2"]
    require(norm["pair_windows"] == fit["batch"]["paired_windows"] == len(targets) and norm["physical_samples"] == target.size and
            norm["floor_m2"] == floor and math.isclose(norm["target_mse_m2"], mse, rel_tol=1e-12, abs_tol=0.) and
            math.isclose(norm["denominator_m2"], max(mse, floor), rel_tol=1e-12, abs_tol=0.) and
            reset_hash({"plus_minus_x_m": target}) == norm["response_targets_sha256"], "native paired-response normalization differs")
    old_roster = [(r["reset"], i) for r in old["fit_episodes"] for i in range(r["ticks"]-7)]
    require(len(old_roster) == fit["batch"]["old_windows"] and len(new_roster) == fit["batch"]["paired_windows"], "batch population differs")
    rng = np.random.default_rng(fit["seed"] + 960000)
    old_rows, new_rows = [], []
    for _ in range(plan["fit"]["updates_per_arm"]):
        old_rows.append(rng.integers(len(old_roster), size=plan["fit"]["old_windows_per_batch"], dtype=np.int64))
        new_rows.append(rng.integers(len(new_roster), size=plan["fit"]["paired_windows_per_batch"], dtype=np.int64))
    batch_hash = reset_hash({"old_reset_start": np.array(old_roster, np.int64), "new_reset_start": np.array(new_roster, np.int64),
                             "old_rows": np.stack(old_rows), "paired_rows": np.stack(new_rows)})
    require(batch_hash == fit["batch"]["indices_sha256"], "shared batch schedule differs from registered sampler")


def work_counts(collection, fits, cells):
    train = [a for c in collection for a in c["arms"].values()]
    blind = [a["native"] for c in cells for a in c["arms"].values()]
    models = [a[m] for c in cells for a in c["arms"].values() for m in MODELS]
    fitted = [arm for f in fits for arm in f.get("arms", {}).values()]
    total = lambda rows, k: sum(r.get(k, 0) for r in rows)
    out = {k: total(train + blind, k) for k in NATIVE_COUNTS}
    for name, rows in (("collection", train), ("blind", blind)):
        out[name + "_native_episodes"] = sum(a["receipt_sha256"] is not None for a in rows)
        out[name + "_native_decisions"] = total(rows, "native_decisions")
    out["native_episodes"] = out["collection_native_episodes"] + out["blind_native_episodes"]
    out.update(fit_model_steps=total(fitted, "attempted_model_steps"), fit_model_steps_completed=total(fitted, "completed_model_steps"),
               world_updates=total(fitted, "updates_attempted"), world_updates_completed=total(fitted, "updates_completed"),
               validation_model_steps=total(models, "attempted_model_steps"), validation_model_steps_completed=total(models, "completed_model_steps"),
               validation_backwards=total(models, "backward_attempted"), validation_backwards_completed=total(models, "backward_completed"),
               persistent_policy_updates=total(fits, "policy_updates"), ppo_updates=total(fits, "ppo_updates"))
    out["model_steps_total"] = out["fit_model_steps"] + out["validation_model_steps"]
    return out


def verify_arm(row, curves, seen, ref, physical_origin, native, neutral=False):
    require(row["curve_key"] == ref, f"curve mapped to wrong arm {ref}")
    x = curves[ref]
    seen.add(ref)
    n = max(0, len(x)-1)
    require(x.ndim == 1 and len(x) <= HORIZON+1 and np.isfinite(x).all() and row["ticks"] == n, f"invalid prefix {ref}")
    if len(x):
        require(physical_origin is not None and x[0] == (physical_origin if native else np.float32(physical_origin)), "physical origin differs")
    result = reward_m(x, physical_origin) if physical_origin is not None else None
    require(result == row["return_m"], f"H32 differs for {ref}")
    status = row["status"]
    require(status in ("COMPLETE", "PARTIAL", "TERMINAL", "NOT_RUN") and row["safe"] is (status == "COMPLETE"), "status and safety disagree")
    if status == "COMPLETE":
        require(n == HORIZON and result is not None and row["source_sha256"] and row["receipt_sha256"] and not row["error"], "incomplete data marked safe")
    if status == "NOT_RUN":
        require(not len(x) and row["receipt_sha256"] is None, "unrun arm contains samples")
    if native:
        require(row["flags_key"] == ref+"_flags", "native flags mapped to wrong arm")
        seen.add(row["flags_key"])
        flags = curves[row["flags_key"]]
        require(flags.shape == (n, len(FLAGS)) and flags.dtype == bool and not flags[:-1, 2:].any(), "invalid native flag prefix")
        require(row["safe"] == (n == HORIZON and not flags.any() and status != "PARTIAL" and not row["error"]), "native flags and safety disagree")
        if status == "TERMINAL":
            require(flags[-1, 2:].any(), "terminal status lacks terminal flag")
    elif neutral:
        require(row["ad_key"] == ref+"_ad", "analytic derivative mapped to wrong arm")
        seen.add(row["ad_key"])
        values = curves[row["ad_key"]]
        require(values.shape in ((1,), (0,)) and np.isfinite(values).all() and
                row["directional_derivative"] == (float(values[0]) if len(values) else None), "archived analytic derivative differs")
        require(0 <= row["backward_completed"] <= row["backward_attempted"] <= 1 and
                (not len(values) or row["backward_completed"] == 1), "invalid neutral backward counter")
    else:
        require(row["directional_derivative"] is None and row["backward_attempted"] == row["backward_completed"] == 0, "unexpected derivative for nonzero multiplier")


def provenance(report):
    return (report["summary"]["candidate_supported_by_scores"] and
            all(report["stages"][s]["status"] == "SEALED" and report["guards"][s]["status"] == "COMPLETE" for s in STAGES) and
            all(f["status"] == "COMPLETE" and all(f["arms"][m]["status"] == "COMPLETE" for m in MODELS) for f in report["fits"]))


def verify_public(report, curves):
    plan = read(PLAN)
    require(report["protocol_sha256"] == digest(PLAN) == PLAN_SHA and report["budget"] == plan["budget"], "protocol or accounting budget changed")
    require(report["native_flag_columns"] == list(FLAGS) and [f["seed"] for f in report["fits"]] == list(SEEDS), "data schema or fit roster differs")
    seen = set()
    coords, first = {}, {}
    for stage, rows, resets, tags in (("collection", report["collection"], TRAIN_RESETS, ("m1", "p1")),
                                      ("native", report["cells"], RESETS, tuple(ARMS))):
        require([(c["training_seed"], c["reset"]) for c in rows] == [(s, r) for s in SEEDS for r in resets], "incomplete or duplicated cells")
        for c in rows:
            s, r = c["training_seed"], c["reset"]
            require(set(c["arms"]) == set(tags), "arm omitted from published denominator")
            for tag, arm in c["arms"].items():
                n = arm if stage == "collection" else arm["native"]
                origin = n["origin_x_m"] if stage == "collection" else c["origin_x_m"]
                verify_arm(n, curves, seen, f"{stage}_{s}_{r}_{tag}_x", origin, True)
                if n["reset_sha256"] is not None:
                    require(coords.setdefault(r, n["reset_sha256"]) == n["reset_sha256"], "resets not paired")
                if n["first_endpoint_sha256"] is not None:
                    require(first.setdefault(r, n["first_endpoint_sha256"]) == n["first_endpoint_sha256"], "FIFO endpoints not paired")
                if stage == "native":
                    require(arm["alpha"] == ARMS[tag], "policy multiplier differs")
                    for m in MODELS:
                        verify_arm(arm[m], curves, seen, f"{m}_{s}_{r}_{tag}_x", origin, False, tag == "zero")
            if stage == "native":
                require(c["scores"] == score_models(c["arms"]), f"cell score differs for {s}/{r}")
    require(seen == set(curves), "unaccounted published curves")
    for fit in report["fits"]:
        verify_fit(fit, report["collection"], curves, plan)
    require(report["summary"] == summarize(report["cells"]), "world counts, per-seed medians, or attribution differs")
    require(report["qualified_with_provenance"] == provenance(report), "qualification differs from records")
    counts = work_counts(report["collection"], report["fits"], report["cells"])
    require(counts == report["counts"] and all(0 <= counts[k] <= cap for k, cap in report["budget"].items()), "work counters differ or exceed budget")
    for stage in STAGES:
        stored = report["stages"][stage]
        verify_guard(stage, report["guards"][stage], plan["resources"])
        if stored["status"] != "SEALED":
            continue
        record = stored["record"]
        require(record["protocol_sha256"] == report["protocol_sha256"] and len(stored["seal_sha256"]) == 64 and
                record["direction_seal95_sha256"] == plan["direction_seal95_sha256"] and stored["producer_sha256"] == sources_for(stage), "stage protocol identity differs")
        if stage != "collection":
            require(record["previous_seal_sha256"] == report["stages"][STAGES[STAGES.index(stage)-1]]["seal_sha256"], "broken stage chain")
        totals = record["totals"]
        if stage in ("collection", "native"):
            rows = [a if stage == "collection" else a["native"] for c in (report["collection"] if stage == "collection" else report["cells"]) for a in c["arms"].values()]
            require(record["native_episodes"] == sum(x["receipt_sha256"] is not None for x in rows) and
                    all(totals[k] == sum(x[k] for x in rows) for k in NATIVE_COUNTS) and
                    all(totals[k] == 0 for k in ("world_updates", "policy_updates", "ppo_updates")), "native seal clocks differ")
        elif stage == "fit":
            require([totals[k] for k in FIT_COUNTS] == [counts[k] for k in ("fit_model_steps", "fit_model_steps_completed", "world_updates", "world_updates_completed")], "fit seal clocks differ")
        else:
            require([totals[k] for k in MODEL_COUNTS] == [counts[k] for k in ("validation_model_steps", "validation_model_steps_completed", "validation_backwards", "validation_backwards_completed")], "prediction seal clocks differ")


def publish():
    if REPORT.exists() or CURVES.exists():
        raise FileExistsError("round-96 output already exists; use --check to audit")
    plan = read(PLAN)
    stages = {s: stage_record(s) for s in STAGES}
    guards = {s: guard_record(s, plan["resources"]) for s in STAGES}
    curves, starts, first = {}, {}, {}
    collection = []
    for seed in SEEDS:
        for reset in TRAIN_RESETS:
            arms = {tag: native_arm("collection", seed, reset, tag, starts, first, curves) for tag in ("m1", "p1")}
            collection.append({"training_seed": seed, "reset": reset, "arms": arms})
    fits = [fit_row(s) for s in SEEDS]
    preview = {}
    for r in RESETS:
        path = BASE / f"predictions_01/start_{r}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as src:
                preview[r] = dict(src)
                starts[r] = reset_hash({k: preview[r][k] for k in RESET_FIELDS})
    cells = []
    for seed in SEEDS:
        for reset in RESETS:
            start = preview.get(reset)
            origin = float(start["state45"][0, 0]) if start is not None else None
            arms = {}
            for tag, alpha in ARMS.items():
                native = native_arm("native", seed, reset, tag, starts, first, curves)
                if origin is None:
                    origin = native["origin_x_m"]
                arms[tag] = {"alpha": alpha, "native": native,
                             **{m: model_arm(seed, reset, m, tag, start, curves) for m in MODELS}}
            cells.append({"training_seed": seed, "reset": reset, "origin_x_m": origin, "arms": arms, "scores": score_models(arms)})
    report = {"schema": "d1-paired-response96-v1", "protocol_sha256": digest(PLAN), "budget": plan["budget"],
              "reward": plan["validation"]["reward"], "native_flag_columns": list(FLAGS), "horizon_ticks": HORIZON,
              "attribution_definition": {"slope_error": "For each seed take the median of all six abs(AD-native FD)/max(abs(AD),abs(native FD)) errors on the same physical curves; a zero denominator is undefined.",
                                         "baseline_error": "For each seed take the median of all three absolute neutral H32 return prediction errors in metres."},
              "stages": stages, "guards": guards, "fits": fits, "collection": collection, "cells": cells,
              "counts": work_counts(collection, fits, cells), "summary": summarize(cells)}
    report["qualified_with_provenance"] = provenance(report)
    verify_public(report, curves)
    arrays(CURVES, curves)
    report["curves_sha256"] = digest(CURVES)
    write(REPORT, report)
    return report


def check():
    report = read(REPORT)
    with np.load(checked(CURVES, report["curves_sha256"]), allow_pickle=False) as data:
        verify_public(report, dict(data))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify published numbers and gates without private traces")
    result = check() if parser.parse_args().check else publish()
    print(json.dumps({"qualified_with_provenance": result["qualified_with_provenance"], "counts": result["counts"],
                      "gates": result["summary"]["gates"]}, indent=2))
