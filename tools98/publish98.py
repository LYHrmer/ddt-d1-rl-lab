"""Publish the paired blind endpoints, or recompute both acceptance tests using public NumPy arrays."""

import argparse
import json
import math

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools95.publish95 import FLAGS, RESET_FIELDS, checked, require, reset_hash, same_value
from tools95.score95 import HORIZON, reward_m
from tools96.publish96 import FIT_COUNTS, MODEL_COUNTS, NATIVE_COUNTS, missing, verify_arm
from tools98.common98 import BASE, PLAN_SHA, checked_stage, protocol, resource_record, sources_for
from tools98.score98 import ARMS, GROUPS, MODELS, RESETS, SEEDS, TICK, h32_scores, state_error, summarize


REPORT = ROOT / "reports/endpoint98.json"
CURVES = ROOT / "reports/endpoint98_curves.npz"
STAGES = ("fit", "predictions", "native")


def h16_status(row, state, native):
    if state.shape != (45,) or not np.isfinite(state).all() or row["ticks"] < TICK:
        return "UNAVAILABLE"
    if native:
        return "OBSERVED" if row["status"] in ("COMPLETE", "TERMINAL") and not row["error"] else "EXCEPTIONAL_NATIVE"
    if row["status"] == "COMPLETE" or row["ticks"] > TICK:
        return "ACCEPTED"
    # The frozen rollout appends the proposed endpoint before checking quaternion and fall limits.
    error = row.get("error") or ""
    if row["ticks"] == TICK and any(message in error for message in ("predicted fall", "invalid predicted quaternion")):
        return "REJECTED_AT_16"
    if row["ticks"] == TICK and row["attempted_model_steps"] > TICK:
        return "ACCEPTED"  # The next model step was attempted after validation at 16.
    return "ACCEPTANCE_UNCONFIRMED"


def set_state(row, states, curves, native):
    key = row["curve_key"] + "_h16_state45"
    vector = states[TICK].copy() if len(states) > TICK else np.empty(0, np.float64 if native else np.float32)
    curves[key] = vector
    row.update(h16_key=key, h16_status=h16_status(row, vector, native))


def native_arm(seed, reset, tag, signatures, first, curves):
    folder = BASE / f"native_01/seed_{seed}/reset_{reset}/{tag}"
    path = folder / "native.json"
    ref = f"native_{seed}_{reset}_{tag}_x"
    row = missing(ref, NATIVE_COUNTS, "PARTIAL" if folder.exists() else "NOT_RUN")
    row.update(flags_key=ref+"_flags", origin_x_m=None, reset_sha256=None, first_endpoint_sha256=None)
    states = np.empty((0, 45), np.float64)
    flags = np.empty((0, len(FLAGS)), bool)
    if path.exists():
        raw = read(path)
        row.update(status="PARTIAL", receipt_sha256=digest(path), error=raw.get("error"))
        if raw.get("trace_file"):
            file = raw["trace_file"]
            require(file in ("native/trace.npz", "native/partial_trace.npz"), f"unexpected native file {path}")
            with np.load(checked(folder/file, raw["trace_sha256"]), allow_pickle=False) as data:
                states, applied, issued = data["state45"], data["applied16"], data["issued16"]
                n = len(applied)
                exceptional = bool(row["error"]) or file.endswith("partial_trace.npz")
                require(states.dtype == np.float64 and states.ndim == 2 and states.shape[1] == 45 and np.isfinite(states).all() and
                        (len(states) == n+1 or (exceptional and len(states) == n == 0)) and n <= HORIZON and
                        int(data["reset_seed"]) == reset and int(data["requested_ticks"]) == HORIZON and int(data["completed_transitions"]) == n and
                        int(data["explore_seed"]) == -1 and not data["exploration16"].any() and not data["command3"].any() and
                        float(data["wheel_offset_rad_s"]) == 0 and applied.shape == issued.shape == (n, 16), "invalid native sample")
                require(all(data[k].shape == (n,) and data[k].dtype == bool for k in FLAGS), "invalid native flags")
                flags = np.stack([data[k] for k in FLAGS], axis=1)
                if len(states):
                    signature = reset_hash({k: data[k][:1] for k in RESET_FIELDS})
                    require(signatures.setdefault(reset, signature) == signature, "native observation differs from sealed reset")
                    row.update(origin_x_m=float(states[0, 0]), reset_sha256=signature)
                if n:
                    require(not applied[0].any() and np.array_equal(applied[1:], issued[:-1]) and np.array_equal(data["raw16"], issued) and
                            np.array_equal(data["obs643"][1:, 41:57], issued), "applied action FIFO differs")
                    signature = reset_hash({"state45": states[1:2]})
                    require(first.setdefault(reset, signature) == signature, "physical first endpoint differs between arms")
                    row["first_endpoint_sha256"] = signature
                counts = [int(data[k]) for k in ("policy_steps", "servo_steps", "native_steps")]
                required = [n, 5*n, 10*n]
                require(all(x >= y for x, y in zip(counts, required)) if exceptional else counts == required, "native counters differ")
            safe = not exceptional and n == HORIZON and not flags.any()
            value = reward_m(states[:, 0], row["origin_x_m"]) if len(states) else None
            require(safe == raw["safe"] and n == raw["ticks"] and all(raw[k] == bool(flags[:, i].any()) for i, k in enumerate(FLAGS))
                    and same_value(value, raw["return_m"]), "native receipt differs from trace")
            row.update(dict(zip(NATIVE_COUNTS, counts)), status="PARTIAL" if exceptional else ("COMPLETE" if safe else "TERMINAL"),
                       ticks=n, safe=safe, return_m=value, source_sha256=raw["trace_sha256"])
    curves[ref], curves[row["flags_key"]] = states[:, 0].copy(), flags
    set_state(row, states, curves, True)
    return row


def model_arm(seed, reset, model, tag, start, curves):
    folder = BASE / f"predictions_01/seed_{seed}/reset_{reset}/{model}"
    path = folder / (tag+".json")
    ref = f"{model}_{seed}_{reset}_{tag}_x"
    row = missing(ref, MODEL_COUNTS, "PARTIAL" if (folder/(tag+".npz")).exists() else "NOT_RUN")
    states = np.empty((0, 45), np.float32)
    ad = None
    if path.is_file():
        raw = read(path)
        require((raw["seed"], raw["reset"], raw["model"], raw["alpha"], raw["file"]) == (seed, reset, model, ARMS[tag], tag+".npz"), "wrong blind forecast")
        with np.load(checked(folder/raw["file"], raw["state_sha256"]), allow_pickle=False) as data:
            states = data["state45"]
        require(states.dtype == np.float32 and states.ndim == 2 and states.shape[1] == 45 and np.isfinite(states).all(), "invalid saved forecast states")
        n = max(0, len(states)-1)
        if len(states):
            require(start is not None and np.array_equal(states[0], start["state45"][0].astype(np.float32)), "forecast reset differs")
        value = reward_m(states[:, 0], float(start["state45"][0, 0])) if start is not None else None
        ad = raw["directional_derivative"]
        require(raw["ticks"] == n and n <= raw["completed_model_steps"] <= raw["attempted_model_steps"] <= HORIZON and same_value(value, raw["return_m"]), "forecast receipt differs")
        row.update({k: raw[k] for k in MODEL_COUNTS}, source_sha256=raw["state_sha256"], receipt_sha256=digest(path), checkpoint_sha256=raw["checkpoint_sha256"],
                   status="COMPLETE" if raw["safe"] else "PARTIAL", ticks=n, safe=raw["safe"], return_m=value, error=raw.get("error"))
    row["directional_derivative"] = ad
    if tag == "zero":
        row["ad_key"] = ref+"_ad"
        curves[row["ad_key"]] = np.array([] if ad is None else [ad], dtype=np.float64)
    curves[ref] = states[:, 0].copy()
    set_state(row, states, curves, False)
    return row


def fit_row(seed):
    folder = BASE / f"fit_01/seed_{seed}"
    path = folder/"receipt.json"
    if not path.is_file():
        return {"seed": seed, "status": "PARTIAL" if folder.exists() else "NOT_RUN", "arms": {}, "source": {}, "receipt_sha256": None}
    row = read(path)
    require(row["seed"] == seed, "wrong fit seed")
    row["receipt_sha256"] = digest(path)
    for m, arm in row["arms"].items():
        if arm["status"] == "COMPLETE":
            require(read(folder/m/"receipt.json") == arm and arm["checkpoint"]["file"] == "world.pt", "fit arm selection differs")
            checked(folder/m/"world.pt", arm["checkpoint"]["sha256"])
    return row


def verify_fit(fit, prior, plan):
    for arm, row in fit["arms"].items():
        require(arm in MODELS and row["lambda_endpoint"] == plan["arms"][arm] and
                0 <= row["updates_completed"] <= row["updates_attempted"] <= 25 and
                0 <= row["completed_model_steps"] <= row["attempted_model_steps"] <= 7200, "fit budget mismatch")
        if row["status"] != "COMPLETE":
            continue
        parent = prior["arms"]["paired"]["checkpoint"]
        ref = fit["source"]["parent_checkpoint"]
        require(all(ref[k] == v for k, v in parent.items()) and row["initial_weights_sha256"] == parent["weights_sha256"] and
                row["initial_adam_sha256"] == parent["adam_sha256"] and row["updates_completed"] == 25 and row["completed_model_steps"] == 7200 and
                row["checkpoint"]["file"] == "world.pt" and len(row["checkpoint"]["weights_sha256"]) == len(row["checkpoint"]["adam_sha256"]) == 64, "unqualified checkpoint")
        traces = fit["source"]["training_traces"]
        new = prior["source"]["new_pairs"]
        expected = [("training94", r["reset"], None, r["sha256"]) for r in prior["source"]["old_traces"]]
        expected += [("training96", p["reset"], tag, p["arms"][tag]["sha256"]) for tag in ("m1", "p1") for p in new]
        require([(t["role"], t["reset"], t.get("arm"), t["sha256"]) for t in traces] == expected and len(expected) == 18 and
                fit["batch"]["rows"] == 18 and fit["batch"]["horizon"] == TICK and fit["normalization"]["denominator_m2"] == prior["normalization"]["denominator_m2"], "training prefixes or paired loss denominator differ")
        require([e["update"] for e in row["losses"]] == list(range(1, 26)), "fit loss log misses a registered update")
        for update in row["losses"]:
            expected_loss = update["state_loss_h8"] + update["normalized_pair_loss_h8"] + row["lambda_endpoint"]*update["endpoint_loss_h16"]
            require(math.isclose(update["loss"], expected_loss, rel_tol=1e-12, abs_tol=1e-15), "reported loss differs from registered endpoint weighting")
    if fit["status"] == "COMPLETE":
        require(all(fit["arms"].get(m, {}).get("status") == "COMPLETE" for m in MODELS) and fit["policy_updates"] == fit["ppo_updates"] == 0, "unqualified complete fit")
        c, e = (fit["arms"][m]["losses"][0] for m in MODELS)
        require(all(c[k] == e[k] for k in ("state_loss_h8", "response_mse_m2", "normalized_pair_loss_h8", "endpoint_loss_h16")), "two arms did not begin at the same measured losses")


def provenance_records():
    stages, guards = {}, {}
    for stage in STAGES:
        directory = BASE/f"{stage}_01"
        path = directory/"seal.json"
        if path.is_file():
            checked_stage(stage)
            row = read(path)
            stages[stage] = {k: v for k, v in row.items() if k != "files"}
            stages[stage].update(status="SEALED", seal_sha256=digest(path), file_count=len(row["files"]))
        else:
            stages[stage] = {"status": "UNSEALED" if directory.exists() else "NOT_RUN", "seal_sha256": None}
        receipt = BASE/f"guard_{stage}_01/report.json"
        if receipt.is_file() and read(receipt)["status"] == "COMPLETE":
            guards[stage] = resource_record(stage)
        elif receipt.is_file():
            raw = read(receipt)
            guards[stage] = {"status": raw["status"], "error": raw.get("error"), "receipt_sha256": digest(receipt)}
        else:
            guards[stage] = {"status": "NO_GUARD_RECEIPT"}
    return stages, guards


def endpoint_metrics(arm, model, fit, curves):
    native, prediction = arm["native"], arm[model]
    usable = fit.get("status") == "COMPLETE" and fit["arms"].get(model, {}).get("status") == "COMPLETE" and native["h16_status"] == "OBSERVED" and prediction["h16_status"] == "ACCEPTED"
    return state_error(curves[native["h16_key"]], curves[prediction["h16_key"]], usable)


def work_counts(fits, cells):
    fit_arms = [a for f in fits for a in f["arms"].values()]
    arms = [a for c in cells for a in c["arms"].values()]
    native = [a["native"] for a in arms]
    models = [a[m] for a in arms for m in MODELS]
    total = lambda rows, k: sum(r.get(k, 0) for r in rows)
    return {**{k: total(native, k) for k in NATIVE_COUNTS}, "native_arms_planned": 45, "forecast_arms_planned": 90,
            "blind_native_episodes": sum(r["receipt_sha256"] is not None for r in native), "blind_forecasts": sum(r["receipt_sha256"] is not None for r in models),
            "training_episodes_reused": sum(len(f["source"].get("training_traces", [])) for f in fits), "new_training_native_decisions": 0,
            "world_updates": total(fit_arms, "updates_attempted"), "world_updates_completed": total(fit_arms, "updates_completed"),
            "fit_model_steps": total(fit_arms, "attempted_model_steps"), "fit_model_steps_completed": total(fit_arms, "completed_model_steps"),
            "validation_model_steps": total(models, "attempted_model_steps"), "validation_model_steps_completed": total(models, "completed_model_steps"),
            "validation_backwards": total(models, "backward_attempted"), "validation_backwards_completed": total(models, "backward_completed"),
            "model_steps_total": total(fit_arms+models, "attempted_model_steps"), "policy_updates": total(fits, "policy_updates"), "ppo_updates": total(fits, "ppo_updates")}


def provenance(report):
    return report["summary"]["candidate_supported_by_scores"] and all(f["status"] == "COMPLETE" for f in report["fits"]) and all(
        report["stages"][stage]["status"] == "SEALED" and report["guards"][stage]["status"] == "COMPLETE" for stage in STAGES)


def verify_vector(row, native, curves, used):
    key = row["curve_key"]+"_h16_state45"
    require(row["h16_key"] == key, "H16 array is assigned to a different source")
    used.add(key)
    state, x = curves[key], curves[row["curve_key"]]
    shape = (45,) if len(x) > TICK else (0,)
    require(state.shape == shape and state.dtype == (np.float64 if native else np.float32) and np.isfinite(state).all(), "invalid H16 state")
    if len(state):
        require(state[0] == x[TICK], "H16 longitudinal coordinate does not match trajectory")
    require(row["h16_status"] == h16_status(row, state, native), "wrong H16 acceptance or exceptional flag")


def verify_public(report, curves):
    plan = protocol()
    require(report["protocol_sha256"] == PLAN_SHA and report["parents_sha256"] == plan["parents_sha256"] and
            report["budget"] == plan["budget"] and report["native_flag_columns"] == list(FLAGS) and report["group_names"] == list(GROUPS),
            "protocol, group order or flag order changed")
    expected = [(s, r) for s in SEEDS for r in RESETS]
    fits = {f["seed"]: f for f in report["fits"]}
    require([f["seed"] for f in report["fits"]] == list(SEEDS) and
            [(c["training_seed"], c["reset"]) for c in report["cells"]] == expected, "incomplete or repeated heldout grid")
    previous = read(ROOT/"reports/paired_response96.json")
    for f, prior in zip(report["fits"], previous["fits"], strict=True):
        require(f["seed"] == prior["seed"], "fit identity differs from parent")
        verify_fit(f, prior, plan)
    seen, signatures, first = set(), {}, {}
    for c in report["cells"]:
        s, r, origin = c["training_seed"], c["reset"], c["origin_x_m"]
        require(set(c["arms"]) == set(ARMS), "all five planned native trajectories are required")
        for tag, arm in c["arms"].items():
            native = arm["native"]
            require(arm["alpha"] == ARMS[tag] and (native["origin_x_m"] is None or native["origin_x_m"] == origin), "wrong multiplier or physical origin")
            for data, field in ((signatures, "reset_sha256"), (first, "first_endpoint_sha256")):
                if native[field] is not None:
                    require(data.setdefault(r, native[field]) == native[field], "reset or first applied zero endpoint differs")
            verify_arm(native, curves, seen, f"native_{s}_{r}_{tag}_x", origin, True)
            verify_vector(native, True, curves, seen)
            for m in MODELS:
                a = arm[m]
                verify_arm(a, curves, seen, f"{m}_{s}_{r}_{tag}_x", origin, False, tag == "zero")
                verify_vector(a, False, curves, seen)
                if a["receipt_sha256"] is not None:
                    trained = fits[s]["arms"].get(m, {})
                    require(fits[s]["status"] == trained.get("status") == "COMPLETE" and a["checkpoint_sha256"] == trained["checkpoint"]["sha256"], "forecast refers to incomplete or different fit")
                require(arm["h16"][m] == endpoint_metrics(arm, m, fits[s], curves), "H16 normalized group error differs")
        require(c["scores"] == h32_scores(c["arms"]), "H32 cell gates differ")
    counts = work_counts(report["fits"], report["cells"])
    budget = plan["budget"]
    require(set(curves) == seen and report["counts"] == counts and all(0 <= counts[k] <= v for k, v in budget.items() if k != "updates_per_arm"), "budget, curve set or counts differ")
    require(report["summary"] == summarize(report["cells"]) and report["qualified_with_provenance"] == provenance(report), "seed medians, attribution or final decision differ")
    previous_seal = plan["local_parent_seals_sha256"]["evidence/paired_response96/fit_01/seal.json"]
    for stage in STAGES:
        ledger, guard = report["stages"][stage], report["guards"][stage]
        if ledger["status"] == "SEALED":
            require(ledger["stage"] == stage and ledger["previous_seal_sha256"] == previous_seal and ledger["sources"] == sources_for(stage)
                    and ledger["protocol_sha256"] == PLAN_SHA and len(ledger["seal_sha256"]) == 64, "stage chain or frozen producer changed")
            totals = ledger["totals"]
            if stage == "fit":
                require([totals[k] for k in FIT_COUNTS] == [counts[k] for k in ("fit_model_steps", "fit_model_steps_completed", "world_updates", "world_updates_completed")]
                        and totals["policy_updates"] == totals["ppo_updates"] == 0, "fit receipt counters differ")
            elif stage == "predictions":
                require([totals[k] for k in MODEL_COUNTS] == [counts[k] for k in ("validation_model_steps", "validation_model_steps_completed", "validation_backwards", "validation_backwards_completed")]
                        and ledger["forecasts"] == counts["blind_forecasts"] and ledger["preview_physics_substeps"] == 0, "prediction counters differ")
            else:
                require(ledger["native_episodes"] == counts["blind_native_episodes"] and all(totals[k] == counts[k] for k in NATIVE_COUNTS)
                        and all(totals[k] == 0 for k in ("world_updates", "policy_updates", "ppo_updates")), "native counters differ")
        previous_seal = ledger["seal_sha256"]
        if guard["status"] == "COMPLETE":
            limits = plan["resources"]
            require(guard["scope"].endswith(f"/d1-endpoint98-{stage}-01.service") and guard["limits_verified"] and guard["returncode"] == 0 and
                    guard["child_started"] and guard["owned_cgroup_children_empty"] and len(guard["cpu_affinity"]) == limits["cpu_affinity_count"] and
                    guard["startup_effective_bytes"] >= limits["admission_min_bytes"] and guard["runtime_min_effective_bytes"] >= limits["runtime_min_bytes"] and
                    guard["supervisor_sha256"] == limits["supervisor_sha256"] and len(guard["resources_sha256"]) == len(guard["receipt_sha256"]) == 64, "guard verification differs")


def publish():
    if REPORT.exists() or CURVES.exists():
        raise FileExistsError("preserve existing round-98 report or curves; use --check")
    plan = protocol()
    stages, guards = provenance_records()
    fits = [fit_row(s) for s in SEEDS]
    starts, signatures, first = {}, {}, {}
    for r in RESETS:
        path = BASE/f"predictions_01/start_{r}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                starts[r] = dict(data)
            signatures[r] = reset_hash({k: starts[r][k] for k in RESET_FIELDS})
    curves, cells = {}, []
    for fit in fits:
        s = fit["seed"]
        for r in RESETS:
            start = starts.get(r)
            origin = None if start is None else float(start["state45"][0, 0])
            arms = {}
            for tag, alpha in ARMS.items():
                n = native_arm(s, r, tag, signatures, first, curves)
                if origin is None:
                    origin = n["origin_x_m"]
                a = {"alpha": alpha, "native": n, **{m: model_arm(s, r, m, tag, start, curves) for m in MODELS}}
                a["h16"] = {m: endpoint_metrics(a, m, fit, curves) for m in MODELS}
                arms[tag] = a
            cells.append({"training_seed": s, "reset": r, "origin_x_m": origin, "arms": arms, "scores": h32_scores(arms)})
    report = {"schema": "d1-endpoint98-v1", "protocol_sha256": PLAN_SHA, "parents_sha256": plan["parents_sha256"],
              "budget": plan["budget"], "native_flag_columns": list(FLAGS), "validation": plan["validation"],
              "group_names": list(GROUPS), "stages": stages, "guards": guards, "fits": fits, "cells": cells,
              "counts": work_counts(fits, cells), "summary": summarize(cells)}
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
    parser.add_argument("--check", action="store_true", help="recompute public metrics without weights or local raw trajectories")
    result = check() if parser.parse_args().check else publish()
    print(json.dumps({"counts": result["counts"], "h16_all_seeds": result["summary"]["h16"]["all_three_strictly_improved"],
                      "h32_nine_cells": result["summary"]["endpoint_h32_all_nine"], "qualified_with_provenance": result["qualified_with_provenance"]}, indent=2))
