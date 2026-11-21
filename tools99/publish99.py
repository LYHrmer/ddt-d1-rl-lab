"""Publish the sealed round-99 comparison or reproduce its scores from public NumPy data."""
import argparse
import hashlib
import json
import math

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools95.publish95 import FLAGS, RESET_FIELDS, checked, require, reset_hash, same_value
from tools95.score95 import reward_m
from tools96.publish96 import FIT_COUNTS, MODEL_COUNTS, NATIVE_COUNTS, missing, verify_arm
from tools98.publish98 import endpoint_metrics, set_state, verify_vector
from tools98.score98 import GROUPS
from tools99.common99 import ARMS, BASE, HORIZON, MODELS, PLAN_SHA, RESETS, SEEDS, checked_stage, protocol, resource_record, sources_for, unit_name
from tools99.score99 import h32_scores, summarize


REPORT = ROOT / "reports/late_response99.json"
CURVES = ROOT / "reports/late_response99_curves.npz"
STAGES = ("fit", "predictions", "native")


def json_sha(record):
    return hashlib.sha256((json.dumps(record, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def training_labels(data, seed, resets):
    xs, flagged = [], []
    for tag in ("m1", "p1"):
        for r in resets:
            key = f"collection_{seed}_{r}_{tag}_x"
            x, flags = data[key], data[key+"_flags"]
            require(x.shape == (33,) and x.dtype == np.float64 and np.isfinite(x).all() and
                    flags.shape == (32, len(FLAGS)) and flags.dtype == bool and not flags[:-1].any(),
                    "unavailable round-96 paired training prefix")
            if flags[-1].any():
                flagged.append({"reset": r, "arm": tag, "tick": 32,
                                "flags": [name for name, active in zip(FLAGS, flags[-1]) if active]})
            xs.append(x)
    xs = np.stack(xs)
    require(np.array_equal(xs[:3, :2], xs[3:, :2]), "training reset or first endpoint differs")
    displacement = xs[:, 17:33] - xs[:, :1]
    return displacement[3:] - displacement[:3], flagged


def verify_fit(fit, parent, plan, training):
    seed, arms = fit["seed"], fit["arms"]
    require(seed == parent["seed"] and set(arms) <= set(MODELS), "wrong fit identity")
    if fit["receipt_sha256"] is not None:
        require(json_sha({k: v for k, v in fit.items() if k != "receipt_sha256"}) == fit["receipt_sha256"],
                "published seed receipt differs from its original bytes")
    complete = [m for m, a in arms.items() if a["status"] == "COMPLETE"]
    for m, a in arms.items():
        require(a["lambda_late"] == plan["arms"][m] and
                0 <= a["updates_completed"] <= a["updates_attempted"] <= 25 and
                0 <= a["completed_model_steps"] <= a["attempted_model_steps"] <= 9600, "fit budget or coefficient differs")
    if fit["status"] == "COMPLETE":
        require(set(complete) == set(MODELS) and fit["policy_updates"] == fit["ppo_updates"] == 0, "incomplete paired fit")
    if not complete:
        return
    source, batch, norm = fit["source"], fit["batch"], fit["normalization"]
    prior = parent["arms"]["endpoint"]
    expected = prior["checkpoint"]
    parent_key = f"evidence/endpoint98/fit_01/seed_{seed}/endpoint/world.pt"
    require(parent["status"] == prior["status"] == "COMPLETE" and source["parent_checkpoint"] == dict(expected, path=parent_key)
            and source["fit98_seal_sha256"] == plan["local_parent_seals_sha256"]["evidence/endpoint98/fit_01/seal.json"]
            and source["seed98_receipt_sha256"] == parent["receipt_sha256"] and source["endpoint98_receipt_sha256"] == json_sha(prior)
            and source["protocol_sha256"] == PLAN_SHA and all(source[k] == parent["source"][k]
            for k in ("ridge_and_stats_sha256", "policy94", "direction95_tensors_sha256")), "wrong fitted parent")
    roster = source["training_traces"]
    require(roster == parent["source"]["training_traces"] and len(roster) == 18
            and [r["role"] for r in roster[:12]] == ["training94"]*12
            and [(r["arm"], r["reset"]) for r in roster[12:]] == [(t, r) for t in ("m1", "p1") for r in plan["paired_training_resets"]]
            and not ({r["reset"] for r in roster} & set(RESETS)), "changed training partition")
    target, flagged = training_labels(training, seed, plan["paired_training_resets"])
    mse = float(np.square(target).mean())
    require(fit["flagged_endpoints"] == flagged and norm["early_denominator_m2"] == parent["normalization"]["denominator_m2"]
            and norm["floor_m2"] == 1e-10 and math.isclose(norm["late_target_mse_m2"], mse, rel_tol=1e-14)
            and norm["late_denominator_m2"] == max(norm["late_target_mse_m2"], norm["floor_m2"])
            and batch["prefix_tensors_sha256"] == source["prefix98_tensors_sha256"] == parent["batch"]["tensors_sha256"]
            and batch["late_targets_sha256"] == reset_hash({"late_targets": target})
            and [batch[k] for k in ("prefix_rows", "prefix_ticks", "suffix_rows", "suffix_ticks")] == [18, 16, 6, 16]
            and all(len(batch[k]) == 64 for k in ("late_actions_sha256", "late_targets_sha256", "tensors_sha256")),
            "training labels, fixed prefix or normalization differs")
    parts = ("state_h8", "response_h8", "state_h16", "response_h17_32")
    for m in complete:
        a = arms[m]
        require(a["updates_attempted"] == a["updates_completed"] == 25 and a["attempted_model_steps"] == a["completed_model_steps"] == 9600
                and a["initial_weights_sha256"] == expected["weights_sha256"] and a["initial_adam_sha256"] == expected["adam_sha256"]
                and a["checkpoint"]["file"] == "world.pt" and all(len(a["checkpoint"][k]) == 64 for k in ("sha256", "weights_sha256", "adam_sha256"))
                and [u["update"] for u in a["losses"]] == list(range(1, 26)), "wrong fit or incomplete checkpoint")
        for update in a["losses"]:
            require(all(math.isfinite(update[k]) and update[k] >= 0 for k in (*parts, "preclip_gradient_norm"))
                    and math.isclose(update["total"], sum(update[k] for k in parts[:3]) + a["lambda_late"]*update[parts[3]],
                                     rel_tol=1e-12, abs_tol=1e-15), "logged loss differs from registered objective")
    if len(complete) == 2:
        require(all(arms["control"]["losses"][0][k] == arms["late"]["losses"][0][k] for k in parts),
                "first loss components from the two forks differ")


def native_arm(folder, seed, reset, tag, signatures, first, curves):
    path = folder / "native.json"
    key = f"native_{seed}_{reset}_{tag}_x"
    row = missing(key, NATIVE_COUNTS, "PARTIAL" if folder.exists() else "NOT_RUN")
    row.update(flags_key=key+"_flags", origin_x_m=None, reset_sha256=None, first_endpoint_sha256=None, receipt=None)
    state, flags = np.empty((0, 45), np.float64), np.empty((0, len(FLAGS)), bool)
    if path.is_file():
        raw = read(path)
        row.update(receipt=raw, receipt_sha256=digest(path), error=raw.get("error"))
        file = raw.get("trace_file")
        if file:
            require(file in ("native/trace.npz", "native/partial_trace.npz"), "unrecognized native data path")
            exceptional = file.endswith("partial_trace.npz") or bool(row["error"])
            with np.load(checked(folder/file, raw["trace_sha256"]), allow_pickle=False) as d:
                state, applied, issued = d["state45"], d["applied16"], d["issued16"]
                n = len(applied)
                require(state.dtype == np.float64 and state.ndim == 2 and state.shape[1] == 45 and np.isfinite(state).all()
                        and (len(state) == n+1 or (exceptional and n == len(state) == 0)) and n <= HORIZON
                        and int(d["completed_transitions"]) == n and int(d["reset_seed"]) == reset and int(d["requested_ticks"]) == HORIZON
                        and int(d["explore_seed"]) == -1 and not d["exploration16"].any() and not d["command3"].any()
                        and float(d["wheel_offset_rad_s"]) == 0 and applied.shape == issued.shape == (n, 16)
                        and all(d[k].dtype == bool and d[k].shape == (n,) for k in FLAGS), "invalid native prefixes")
                flags = np.stack([d[k] for k in FLAGS], axis=1)
                if len(state):
                    signature = reset_hash({k: d[k][:1] for k in RESET_FIELDS})
                    require(signatures.setdefault(reset, signature) == signature, "native reset differs from sealed preview")
                    row.update(origin_x_m=float(state[0, 0]), reset_sha256=signature)
                if n:
                    require(not applied[0].any() and np.array_equal(applied[1:], issued[:-1]) and np.array_equal(d["raw16"], issued)
                            and np.array_equal(d["obs643"][1:, 41:57], issued), "applied action FIFO differs")
                    signature = reset_hash({"state45": state[1:2]})
                    require(first.setdefault(reset, signature) == signature, "physical first endpoint differs")
                    row["first_endpoint_sha256"] = signature
                clocks = [int(d[k]) for k in ("policy_steps", "servo_steps", "native_steps")]
                expected = [n, 5*n, 10*n]
                require(all(c >= e for c, e in zip(clocks, expected)) if exceptional else clocks == expected, "native clocks differ")
            safe = n == HORIZON and not flags.any() and not exceptional
            row.update(dict(zip(NATIVE_COUNTS, clocks)), status="PARTIAL" if exceptional else ("COMPLETE" if safe else "TERMINAL"),
                       safe=bool(safe), ticks=n, source_sha256=raw["trace_sha256"],
                       return_m=reward_m(state[:, 0], row["origin_x_m"]) if len(state) else None)
    curves[key], curves[row["flags_key"]] = state[:, 0].copy(), flags
    set_state(row, state, curves, True)
    return row


def model_arm(folder, seed, reset, model, tag, start, curves):
    path = folder/(tag+".json")
    key = f"{model}_{seed}_{reset}_{tag}_x"
    row = missing(key, MODEL_COUNTS, "PARTIAL" if (folder/(tag+".npz")).exists() else "NOT_RUN")
    row.update(directional_derivative=None, receipt=None)
    state = np.empty((0, 45), np.float32)
    if path.is_file():
        raw = read(path)
        require((raw["seed"], raw["reset"], raw["model"], raw["alpha"], raw["file"]) == (seed, reset, model, ARMS[tag], tag+".npz"),
                "mismatched blind forecast")
        with np.load(checked(folder/raw["file"], raw["state_sha256"]), allow_pickle=False) as data:
            state = data["state45"]
        require(state.ndim == 2 and state.shape[1] == 45 and state.dtype == np.float32 and np.isfinite(state).all(),
                "invalid saved forecast states")
        if len(state):
            require(start is not None and np.array_equal(state[0], start["state45"][0].astype(np.float32)), "forecast reset differs")
        row.update({k: raw[k] for k in MODEL_COUNTS}, source_sha256=raw["state_sha256"], receipt=raw, receipt_sha256=digest(path),
                   checkpoint_sha256=raw["checkpoint_sha256"], status="COMPLETE" if raw["safe"] else "PARTIAL", safe=raw["safe"],
                   ticks=max(0, len(state)-1), directional_derivative=raw["directional_derivative"], error=raw.get("error"),
                   return_m=reward_m(state[:, 0], float(start["state45"][0, 0])) if start is not None else None)
    if tag == "zero":
        ad = row["directional_derivative"]
        row["ad_key"] = key+"_ad"
        curves[row["ad_key"]] = np.array([] if ad is None else [ad], dtype=np.float64)
    curves[key] = state[:, 0].copy()
    set_state(row, state, curves, False)
    return row


def fit_record(folder, seed):
    path = folder/"receipt.json"
    if not path.is_file():
        return {"seed": seed, "status": "PARTIAL" if folder.exists() else "NOT_RUN", "arms": {}, "source": {}, "receipt_sha256": None}
    row = read(path)
    require(row["seed"] == seed, "fit seed differs")
    for m, a in row["arms"].items():
        if a["status"] == "COMPLETE":
            require(m in MODELS and read(folder/m/"receipt.json") == a and a["checkpoint"]["file"] == "world.pt", "fit arm receipt differs")
            checked(folder/m/"world.pt", a["checkpoint"]["sha256"])
    return dict(row, receipt_sha256=digest(path))


def stage_records(base):
    stages, guards = {}, {}
    for s in STAGES:
        folder = base/(s+"_01")
        seal, receipt = folder/"seal.json", base/f"guard_{s}_01/report.json"
        if seal.is_file():
            require(checked_stage(s) == folder, "unexpected sealed stage path")
            stages[s] = dict(read(seal), status="SEALED", seal_sha256=digest(seal))
        else:
            stages[s] = {"status": "UNSEALED" if folder.exists() else "NOT_RUN", "seal_sha256": None}
        if receipt.is_file() and read(receipt)["status"] == "COMPLETE":
            guards[s] = resource_record(s)
        elif receipt.is_file():
            row = read(receipt)
            guards[s] = {"status": row["status"], "error": row.get("error"), "receipt_sha256": digest(receipt)}
        else:
            guards[s] = {"status": "NO_GUARD_RECEIPT"}
    return stages, guards


def work_counts(fits, cells):
    arms = [a for c in cells for a in c["arms"].values()]
    native, models = [a["native"] for a in arms], [a[m] for a in arms for m in MODELS]
    fitted = [a for f in fits for a in f["arms"].values()]
    def total(records, key):
        return sum(r.get(key, 0) for r in records)
    counts = {k: total(native, k) for k in NATIVE_COUNTS}
    counts.update(native_arms_planned=45, forecast_arms_planned=90,
                  blind_native_episodes=sum(r["receipt_sha256"] is not None for r in native),
                  blind_forecasts=sum(r["receipt_sha256"] is not None for r in models),
                  training_episodes_reused=sum(len(f["source"].get("training_traces", [])) for f in fits),
                  new_training_native_decisions=0, policy_updates=total(fits, "policy_updates"), ppo_updates=total(fits, "ppo_updates"))
    for rows, fields, names in ((fitted, FIT_COUNTS, ("fit_model_steps", "fit_model_steps_completed", "world_updates", "world_updates_completed")),
                                (models, MODEL_COUNTS, ("validation_model_steps", "validation_model_steps_completed", "validation_backwards", "validation_backwards_completed"))):
        counts.update({name: total(rows, k) for k, name in zip(fields, names)})
    counts["model_steps_total"] = counts["fit_model_steps"] + counts["validation_model_steps"]
    return counts


def decisions(report):
    complete = (all(f["status"] == "COMPLETE" for f in report["fits"]) and
                all(report["stages"][s]["status"] == "SEALED" and report["guards"][s]["status"] == "COMPLETE" for s in STAGES))
    score = report["summary"]
    return {"sources_complete": complete, "slope_improvement_with_provenance": complete and score["slope_improvement_supported"],
            "qualified_with_provenance": complete and score["candidate_supported_by_scores"],
            "all_three_with_provenance": complete and score["all_three_criteria_passed"]}


def verify_stages(report, plan, counts):
    previous = plan["local_parent_seals_sha256"]["evidence/endpoint98/fit_01/seal.json"]
    limits = plan["resources"]
    for s in STAGES:
        ledger, guard = report["stages"][s], report["guards"][s]
        if ledger["status"] == "SEALED":
            raw = {k: v for k, v in ledger.items() if k not in ("status", "seal_sha256")}
            require(json_sha(raw) == ledger["seal_sha256"] and ledger["previous_seal_sha256"] == previous
                    and ledger["stage"] == s and ledger["unit"] == unit_name(s) and ledger["sources"] == sources_for(s)
                    and ledger["protocol_sha256"] == PLAN_SHA and ledger["direction_seal95_sha256"] ==
                    plan["local_parent_seals_sha256"]["evidence/policy_direction95/directions_01/seal.json"],
                    "stage bytes, order, producer or registered parent differs")
            totals = ledger["totals"]
            if s == "fit":
                keys = ("fit_model_steps", "fit_model_steps_completed", "world_updates", "world_updates_completed")
                roster = [{"seed": f["seed"], "status": f["status"], "arms": {m: a["status"] for m, a in f["arms"].items()}} for f in report["fits"]]
                require([totals[k] for k in FIT_COUNTS] == [counts[k] for k in keys] and ledger["seeds"] == roster
                        and totals["policy_updates"] == totals["ppo_updates"] == 0, "fit stage counter or seed roster differs")
            elif s == "predictions":
                keys = ("validation_model_steps", "validation_model_steps_completed", "validation_backwards", "validation_backwards_completed")
                require([totals[k] for k in MODEL_COUNTS] == [counts[k] for k in keys]
                        and ledger["forecasts"] == counts["blind_forecasts"] and ledger["preview_physics_substeps"] == 0,
                        "forecast counters differ")
            else:
                require(all(totals[k] == counts[k] for k in NATIVE_COUNTS) and ledger["native_episodes"] == counts["blind_native_episodes"]
                        and all(totals[k] == 0 for k in ("world_updates", "policy_updates", "ppo_updates")), "native counters differ")
        else:
            require(ledger["status"] in ("UNSEALED", "NOT_RUN") and ledger["seal_sha256"] is None, "invalid missing stage")
        previous = ledger["seal_sha256"]
        if guard["status"] == "COMPLETE":
            require(guard["scope"].endswith("/"+unit_name(s)+".service") and guard["limits_verified"] and guard["returncode"] == 0
                    and guard["child_started"] and guard["owned_cgroup_children_empty"] and len(guard["cpu_affinity"]) == limits["cpu_affinity_count"]
                    and guard["startup_effective_bytes"] >= limits["admission_min_bytes"] and guard["runtime_min_effective_bytes"] >= limits["runtime_min_bytes"]
                    and guard["supervisor_sha256"] == limits["supervisor_sha256"] and len(guard["receipt_sha256"]) == len(guard["resources_sha256"]) == 64,
                    "round-99 memory, affinity or supervisor differs")


def bind_file(stages, stage, name, checksum):
    if stages[stage]["status"] == "SEALED":
        require(stages[stage]["files"].get(name) == checksum, f"sealed file and receipt differ: {stage}/{name}")


def verify_sample(row, curves, seen, stages, origin, seed, reset, tag, model=None):
    native = model is None
    stage, label = ("native", "native") if native else ("predictions", model)
    key = f"{label}_{seed}_{reset}_{tag}_x"
    verify_arm(row, curves, seen, key, origin, native, tag == "zero")
    verify_vector(row, native, curves, seen)
    require(curves[key].dtype == (np.float64 if native else np.float32), "wrong archived coordinate precision")
    folder = f"seed_{seed}/reset_{reset}/" + (tag if native else model) + "/"
    bind_file(stages, stage, folder+("native.json" if native else tag+".json"), row["receipt_sha256"])
    raw = row["receipt"]
    if raw is None:
        require(row["receipt_sha256"] is None and row["status"] in ("NOT_RUN", "PARTIAL"), "missing raw receipt")
        return
    require(row["receipt_sha256"] == json_sha(raw) and row["error"] == raw.get("error"), "raw receipt no longer matches its recorded bytes")
    if native and not raw.get("trace_file"):
        require(row["status"] == "PARTIAL" and row["source_sha256"] is None, "absent native trace marked complete")
        return
    require(raw["ticks"] == row["ticks"] and raw["safe"] == row["safe"] and same_value(raw["return_m"], row["return_m"]),
            "raw status, H32 return or sample count differs")
    if native:
        file = raw["trace_file"]
        flags = curves[row["flags_key"]]
        exceptional = file.endswith("partial_trace.npz") or bool(raw.get("error"))
        clocks = [row[k] for k in NATIVE_COUNTS]
        expected = [row["ticks"]*v for v in (1, 5, 10)]
        require(file in ("native/partial_trace.npz", "native/trace.npz") and (row["status"] == "PARTIAL") == exceptional
                and (all(x >= y for x, y in zip(clocks, expected)) if exceptional else clocks == expected)
                and not flags[:-1].any() and all(raw[k] == bool(flags[:, i].any()) for i, k in enumerate(FLAGS))
                and raw["trace_sha256"] == row["source_sha256"], "native contact, clock or partial identity differs")
    else:
        file = raw["file"]
        require((raw["seed"], raw["reset"], raw["model"], raw["alpha"], file) == (seed, reset, model, ARMS[tag], tag+".npz")
                and raw["directional_derivative"] == row["directional_derivative"] and raw["checkpoint_sha256"] == row["checkpoint_sha256"]
                and raw["state_sha256"] == row["source_sha256"] and all(row[k] == raw[k] for k in MODEL_COUNTS)
                and row["ticks"] <= row["completed_model_steps"] <= row["attempted_model_steps"] <= HORIZON,
                "prediction, derivative or checkpoint identity differs")
    bind_file(stages, stage, folder+file, row["source_sha256"])


def verify_public(report, curves):
    plan = protocol()
    require(report["schema"] == "d1-late-response99-v1" and report["protocol_sha256"] == PLAN_SHA and report["parents_sha256"] == plan["parents_sha256"]
            and report["validation"] == plan["validation"] and report["budget"] == plan["budget"] and report["native_flag_columns"] == list(FLAGS)
            and report["group_names"] == list(GROUPS), "registered experiment, flags or state groups differ")
    fits = {f["seed"]: f for f in report["fits"]}
    require([f["seed"] for f in report["fits"]] == list(SEEDS) and
            [(c["training_seed"], c["reset"]) for c in report["cells"]] == [(s, r) for s in SEEDS for r in RESETS],
            "incomplete or different blind grid")
    parents = read(ROOT/"reports/endpoint98.json")["fits"]
    stages = report["stages"]
    with np.load(ROOT/"reports/paired_response96_curves.npz", allow_pickle=False) as training:
        for fit, parent in zip(report["fits"], parents, strict=True):
            verify_fit(fit, parent, plan, training)
            name = f"seed_{fit['seed']}/"
            bind_file(stages, "fit", name+"receipt.json", fit["receipt_sha256"])
            for m, arm in fit["arms"].items():
                if arm["status"] == "COMPLETE":
                    bind_file(stages, "fit", name+m+"/receipt.json", json_sha(arm))
                    bind_file(stages, "fit", name+m+"/world.pt", arm["checkpoint"]["sha256"])
    seen, signatures, first = set(), {}, {}
    require([v["reset"] for v in report["reset_previews"]] == list(RESETS), "wrong blind preflight roster")
    for preview in report["reset_previews"]:
        r = preview["reset"]
        bind_file(stages, "predictions", f"start_{r}.npz", preview["file_sha256"])
        if preview["file_sha256"] is not None:
            require(len(preview["observation_sha256"]) == 64 and math.isfinite(preview["origin_x_m"]), "invalid reset preview")
            signatures[r] = preview["observation_sha256"]
    for c in report["cells"]:
        s, r, origin = c["training_seed"], c["reset"], c["origin_x_m"]
        preview = report["reset_previews"][RESETS.index(r)]
        require(set(c["arms"]) == set(ARMS) and (preview["file_sha256"] is None or origin == preview["origin_x_m"]), "wrong planned arms or reset")
        for tag, a in c["arms"].items():
            n = a["native"]
            require(a["alpha"] == ARMS[tag] and (n["origin_x_m"] is None or n["origin_x_m"] == origin), "wrong multiplier or paired origin")
            verify_sample(n, curves, seen, stages, origin, s, r, tag)
            for values, field in ((signatures, "reset_sha256"), (first, "first_endpoint_sha256")):
                if n[field] is not None:
                    require(values.setdefault(r, n[field]) == n[field], "paired observations or first endpoint differ")
            for m in MODELS:
                row = a[m]
                verify_sample(row, curves, seen, stages, origin, s, r, tag, m)
                if row["receipt_sha256"] is not None:
                    fitted = fits[s]["arms"].get(m, {})
                    require(fitted.get("status") == fits[s]["status"] == "COMPLETE" and
                            row["checkpoint_sha256"] == fitted["checkpoint"]["sha256"], "forecast uses an ineligible checkpoint")
                require(a["h16"][m] == endpoint_metrics(a, m, fits[s], curves), "H16 normalized state errors differ")
        require(c["scores"] == h32_scores(c["arms"]), "frozen H32 gates differ")
    counts = work_counts(report["fits"], report["cells"])
    require(set(curves) == seen and counts == report["counts"] and
            all(0 <= counts[k] <= limit for k, limit in plan["budget"].items() if k != "updates_per_arm"), "curve set or budget differs")
    verify_stages(report, plan, counts)
    require(report["summary"] == summarize(report["cells"]) and all(report[k] == v for k, v in decisions(report).items()),
            "seed medians, slope attribution or verified promotion differs")


def publish(base=BASE, report_path=REPORT, curves_path=CURVES):
    if report_path.exists() or curves_path.exists():
        raise FileExistsError("preserve the round-99 report and curves; use --check")
    plan = protocol()
    stages, guards = stage_records(base)
    fits = [fit_record(base/f"fit_01/seed_{s}", s) for s in SEEDS]
    previews, starts, signatures, first = [], {}, {}, {}
    for r in RESETS:
        path = base/f"predictions_01/start_{r}.npz"
        preview = {"reset": r, "file_sha256": None, "origin_x_m": None, "observation_sha256": None}
        if path.is_file():
            with np.load(path, allow_pickle=False) as d:
                starts[r] = {k: d[k] for k in RESET_FIELDS}
            signatures[r] = reset_hash(starts[r])
            preview.update(file_sha256=digest(path), observation_sha256=signatures[r], origin_x_m=float(starts[r]["state45"][0, 0]))
        previews.append(preview)
    cells, curves = [], {}
    for fit in fits:
        s = fit["seed"]
        for preview in previews:
            r, origin = preview["reset"], preview["origin_x_m"]
            grid = f"seed_{s}/reset_{r}"
            arms = {}
            for tag, alpha in ARMS.items():
                native = native_arm(base/"native_01"/grid/tag, s, r, tag, signatures, first, curves)
                if origin is None:
                    origin = native["origin_x_m"]
                a = {"alpha": alpha, "native": native}
                for m in MODELS:
                    a[m] = model_arm(base/"predictions_01"/grid/m, s, r, m, tag, starts.get(r), curves)
                a["h16"] = {m: endpoint_metrics(a, m, fit, curves) for m in MODELS}
                arms[tag] = a
            cells.append({"training_seed": s, "reset": r, "origin_x_m": origin, "arms": arms, "scores": h32_scores(arms)})
    report = {"schema": "d1-late-response99-v1", "protocol_sha256": PLAN_SHA, "parents_sha256": plan["parents_sha256"],
              "budget": plan["budget"], "validation": plan["validation"], "native_flag_columns": list(FLAGS), "group_names": list(GROUPS),
              "stages": stages, "guards": guards, "fits": fits, "reset_previews": previews, "cells": cells,
              "counts": work_counts(fits, cells), "summary": summarize(cells)}
    report.update(decisions(report))
    verify_public(report, curves)
    arrays(curves_path, curves)
    report["curves_sha256"] = digest(curves_path)
    write(report_path, report)
    return report


def check(report_path=REPORT, curves_path=CURVES):
    report = read(report_path)
    with np.load(checked(curves_path, report["curves_sha256"]), allow_pickle=False) as data:
        verify_public(report, dict(data))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="recalculate using public files without raw runs or weights")
    result = check() if parser.parse_args().check else publish()
    print(json.dumps({"counts": result["counts"], **decisions(result)}, indent=2))
