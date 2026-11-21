"""Publish or verify late-trajectory replay using saved coordinates, actions and receipts."""

import argparse
import json

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools95.publish95 import checked, require
from tools96.common96 import BASE as BASE96
from tools97.common97 import BASE, PLAN_SHA, checked_stage, protocol, resource_record, sources_for, unit_name
from tools97.replay97 import CLOCKS, _f32_sha, _forecast_cut, _recorded_native
from tools97.score97 import ARMS, COMPARISONS, CUT, END, MODELS, RESETS, SEEDS, score_cell, suffix_return, summarize


REPORT = ROOT / "reports/suffix_replay97.json"
CURVES = ROOT / "reports/suffix_replay97_curves.npz"
REPORT96 = ROOT / "reports/paired_response96.json"
CURVES96 = ROOT / "reports/paired_response96_curves.npz"
STAGES = ("fixed", "anchored")
ZERO_WORK = ("new_native_decisions", "new_servo_updates", "new_physics_substeps", "new_actor_calls", "backward_passes", "world_updates", "policy_updates", "ppo_updates")


def stage_records():
    stages, guards = {}, {}
    for stage in STAGES:
        directory = BASE / f"{stage}_01"
        seal = directory / "seal.json"
        if seal.is_file():
            checked_stage(stage)
            record = read(seal)
            stages[stage] = {k: record[k] for k in ("stage", "protocol_sha256", "previous_seal_sha256", "sources", "totals")}
            stages[stage].update(status="SEALED", seal_sha256=digest(seal), files=len(record["files"]))
        else:
            failure = directory / "replay_failure.json"
            stages[stage] = {"status": "UNSEALED" if directory.exists() else "NOT_RUN", "seal_sha256": None,
                             "failure": read(failure) if failure.is_file() else None}
        guard = BASE / f"guard_{stage}_01/report.json"
        if guard.is_file() and read(guard)["status"] == "COMPLETE":
            guards[stage] = resource_record(stage)
        elif guard.is_file():
            raw = read(guard)
            guards[stage] = {"status": raw["status"], "error": raw.get("error"), "receipt_sha256": digest(guard)}
        else:
            guards[stage] = {"status": "NO_GUARD_RECEIPT"}
    return stages, guards


def parent_row(row, curves, origin, native):
    x = curves[row["curve_key"]]
    result = suffix_return(x, origin, 0)
    last = len(x)-1 if len(x) else None
    valid = result is not None and not row["error"] and row["status"] in (("COMPLETE", "TERMINAL") if native else ("COMPLETE",))
    out = {"curve_key": row["curve_key"], "source_sha256": row["source_sha256"], "status": row["status"], "safe": row["safe"],
           "ticks": max(0, (last or 0)-CUT), "last_tick": last, "return_m": result, "valid_numeric_suffix": valid}
    if native:
        flags = curves[row["flags_key"]]
        out["flag_ticks"] = {name: (np.flatnonzero(flags[:, i])+1).tolist()
                             for i, name in enumerate(("fallen", "illegal", "terminated", "truncated"))}
    return out


def parent_decision(parent, plan):
    return {"protocol_sha256": parent["protocol_sha256"], "report_sha256": plan["parents_sha256"][str(REPORT96.relative_to(ROOT))],
            "qualified_with_provenance": parent["qualified_with_provenance"], "gates": parent["summary"]["gates"],
            "worlds": {m: {k: parent["summary"]["worlds"][m][k] for k in ("cells_passed", "cells_planned", "ad_native_25pct")}
                       for m in MODELS}}


def replay_row(stage, seed, reset, model, tag, start, tape, sources, origin, curves):
    folder = BASE / f"{stage}_01/seed_{seed}/reset_{reset}/{model}"
    path, receipt = folder/f"{tag}.npz", folder/f"{tag}.json"
    ref = f"{stage}_{seed}_{reset}_{model}_{tag}_x"
    curves[ref] = np.empty(0, np.float32)
    row = {"status": "NOT_RUN", "ticks": 0, "last_tick": None, "return_m": None, "valid_numeric_suffix": False,
           "curve_key": ref, "receipt_sha256": None, "state_file_sha256": None, "error": None,
           "rejected_endpoint_tick": None, **dict.fromkeys(CLOCKS, 0)}
    if not receipt.is_file():
        if path.is_file():
            row.update(status="PARTIAL", state_file_sha256=digest(path), error="state archive has no finalized replay receipt")
        return row
    raw = read(receipt)
    require((raw["stage"], raw["seed"], raw["reset"], raw["model"], raw["tag"], raw["alpha"], raw["cut_tick"], raw["end_tick"])
            == (stage, seed, reset, model, tag, ARMS[tag], CUT, END), "wrong replay identity")
    require(all(raw[k] == v for k, v in sources.items()) and raw["start_state_sha256"] == _f32_sha(start)
            and raw["applied_tape_sha256"] == _f32_sha(tape), "replay uses different source state, tape or checkpoint")
    with np.load(checked(path, raw["state_file_sha256"]), allow_pickle=False) as data:
        states, applied = data["state45"], data["applied16"]
        require(states.dtype == applied.dtype == np.float32 and states.shape == (raw["ticks"]+1, 45) and
                np.isfinite(states).all() and np.array_equal(states[0], start), "retained 45-coordinate states or cut point differ")
        require(np.array_equal(applied, tape[:raw["attempted_model_steps"]]), "replayed controls differ from original post-FIFO controls")
        curves[ref] = states[:, 0].copy()
        row["applied_prefix_sha256"] = _f32_sha(applied)
    row.update({k: raw[k] for k in ("status", "error", "ticks", "rejected_endpoint_tick", *CLOCKS,
                                   "start_state_sha256", "applied_tape_sha256", "state_file_sha256", *sources.keys())})
    result = suffix_return(curves[ref], origin, CUT)
    row.update(last_tick=CUT+raw["ticks"], receipt_sha256=digest(receipt), return_m=result,
               valid_numeric_suffix=raw["status"] == "COMPLETE" and raw["error"] is None and result is not None)
    return row


def work_counts(cells, stages):
    result = {}
    for stage in STAGES:
        rows = [arm["models"][m][stage] for c in cells for arm in c["arms"].values() for m in MODELS]
        result[stage] = {"arms_planned": len(rows), "arms": sum(r["receipt_sha256"] is not None for r in rows),
                         "complete_arms": sum(r["status"] == "COMPLETE" for r in rows),
                         "partial_arms": sum(r["status"] == "PARTIAL" for r in rows), "not_run": sum(r["status"] == "NOT_RUN" for r in rows),
                         **{k: sum(r[k] for r in rows) for k in CLOCKS}}
    work = {k: sum(stages[s]["totals"][k] for s in STAGES) if all(stages[s]["status"] == "SEALED" for s in STAGES) else None for k in ZERO_WORK}
    arms = [arm for c in cells for arm in c["arms"].values()]
    checkpoints = {arm["models"][m][s]["checkpoint_sha256"] for arm in arms for m in MODELS for s in STAGES if arm["models"][m][s]["receipt_sha256"] is not None}
    return {"by_stage": result, "native_paths_reused": len({arm["native"]["source_sha256"] for arm in arms}),
            "closed_forecasts_reused": len({arm["models"][m]["closed"]["source_sha256"] for arm in arms for m in MODELS}),
            "checkpoints_reused": len(checkpoints), **work, "model_steps_attempted": sum(r["attempted_model_steps"] for r in result.values()),
            "model_steps_completed": sum(r["completed_model_steps"] for r in result.values()),
            "model_steps_retained": sum(r["retained_model_steps"] for r in result.values())}


def attributed(report):
    decisions = {}
    for m in MODELS:
        decisions[m] = {}
        for before, after in COMPARISONS:
            order = before+"_to_"+after
            required = STAGES[:STAGES.index(after)+1]
            valid = all(report["stages"][s]["status"] == "SEALED" and report["guards"][s]["status"] == "COMPLETE" for s in required)
            decisions[m][order] = valid and report["summary"]["models"][m]["comparisons"][order]["all_three_seeds_support"]
    return decisions


def verify_replay(row, curves, seen, ref, start, tape, expected_sources, origin):
    require(ref == row["curve_key"], "replay curve refers to different method/seed/cell/arm")
    x = curves[ref]
    seen.add(ref)
    require(x.dtype == np.float32 and x.ndim == 1 and len(x) <= END-CUT+1 and np.isfinite(x).all(), "invalid retained x prefix")
    expected_return = suffix_return(x, origin, CUT)
    require(row["return_m"] == expected_return and row["last_tick"] == (CUT+len(x)-1 if len(x) else None), "suffix clock or return differs")
    a, c, r = (row[k] for k in CLOCKS)
    require(0 <= r <= c <= a <= END-CUT and row["ticks"] == r == max(0, len(x)-1), "step counters differ from stored suffix")
    if row["receipt_sha256"] is None:
        require(row["status"] in ("PARTIAL", "NOT_RUN") and not len(x) and not row["valid_numeric_suffix"], "unreceipted branch marked valid")
        return
    require(bool(len(x)) and x[0] == start[0] and all(row[k] == v for k, v in expected_sources.items())
            and row["start_state_sha256"] == _f32_sha(start) and row["applied_tape_sha256"] == _f32_sha(tape)
            and row["applied_prefix_sha256"] == _f32_sha(tape[:a]), "cut point, applied tape or frozen parents differ")
    complete = (r == END-CUT and expected_return is not None and row["error"] is None and row["rejected_endpoint_tick"] is None)
    require(row["status"] == ("COMPLETE" if complete else "PARTIAL") and row["valid_numeric_suffix"] is complete,
            "partial or rejected prediction promoted to a complete suffix")
    if row["rejected_endpoint_tick"] is not None:
        require(row["error"] and row["rejected_endpoint_tick"] == CUT+a and a == r+1, "rejected endpoint included in scoring")


def verify_public(report, curves, parent, old_curves, plan):
    require(report["protocol_sha256"] == PLAN_SHA and report["parents_sha256"] == plan["parents_sha256"] and
            report["formal_round96"] == parent_decision(parent, plan) and report["budget"] == plan["budget"], "parent gate or protocol changed")
    old = {(c["training_seed"], c["reset"]): c for c in parent["cells"]}
    require([(c["training_seed"], c["reset"]) for c in report["cells"]] == [(s, r) for s in SEEDS for r in RESETS], "incomplete cell grid")
    fits = {f["seed"]: f for f in parent["fits"]}
    seen = set()
    for cell in report["cells"]:
        s, r = cell["training_seed"], cell["reset"]
        prior, origin = old[s, r], old[s, r]["origin_x_m"]
        require(origin == cell["origin_x_m"] and set(cell["arms"]) == set(ARMS), "registered origin or multiplier differs")
        for tag, arm in cell["arms"].items():
            native = prior["arms"][tag]["native"]
            require(arm["alpha"] == ARMS[tag] and arm["native"] == parent_row(native, old_curves, origin, True), "native source or terminal flags differ")
            prefix = f"native_{s}_{r}_{tag}"
            tkey, nkey = prefix+"_applied16", prefix+"_cut_state45"
            require(arm["tape_key"] == tkey and arm["native_cut_key"] == nkey and set(arm["models"]) == set(MODELS), "source keys or world families differ")
            tape, native_cut = curves[tkey], curves[nkey]
            require(tape.shape == (END-CUT, 16) and tape.dtype == np.float32 and np.isfinite(tape).all() and (np.abs(tape) <= 1).all(), "invalid public applied tape")
            require(native_cut.shape == (45,) and native_cut.dtype == np.float32 and np.isfinite(native_cut).all() and
                    native_cut[0] == np.float32(old_curves[native["curve_key"]][CUT]), "anchored cut differs from physical tick 16")
            seen.update((tkey, nkey))
            for m in MODELS:
                methods = arm["models"][m]
                forecast = prior["arms"][tag][m]
                require(methods["closed"] == parent_row(forecast, old_curves, origin, False) and set(methods) == {"closed", *STAGES}, "closed source differs")
                ckey = f"closed_{s}_{r}_{m}_{tag}_cut_state45"
                model_cut = curves[ckey]
                seen.add(ckey)
                require(arm["forecast_cut_keys"][m] == ckey and model_cut.shape == (45,) and model_cut.dtype == np.float32 and np.isfinite(model_cut).all()
                        and model_cut[0] == old_curves[forecast["curve_key"]][CUT], "fixed cut differs from archived model tick 16")
                sources = {"native_trace_sha256": native["source_sha256"], "forecast96_sha256": forecast["source_sha256"],
                           "checkpoint_sha256": fits[s]["arms"][m]["checkpoint"]["sha256"]}
                for stage, start in zip(STAGES, (model_cut, native_cut)):
                    verify_replay(methods[stage], curves, seen, f"{stage}_{s}_{r}_{m}_{tag}_x", start, tape, sources, origin)
        require(cell["scores"] == score_cell(cell["arms"]), f"incorrect cell scores at {s}/{r}")
    require(seen == set(curves) and report["summary"] == summarize(report["cells"]), "curve roster or seedwise attribution differs")
    counts = work_counts(report["cells"], report["stages"])
    require(report["counts"] == counts and counts["model_steps_attempted"] <= plan["budget"]["max_model_steps_total"] and
            report["attribution_with_provenance"] == attributed(report), "work totals or attribution promotion differs")
    for stage in STAGES:
        ledger, guard = report["stages"][stage], report["guards"][stage]
        expected_previous = plan["local_parent_seals_sha256"]["native"] if stage == "fixed" else report["stages"]["fixed"]["seal_sha256"]
        if ledger["status"] == "SEALED":
            require(ledger["sources"] == sources_for(stage) and ledger["protocol_sha256"] == PLAN_SHA and
                    ledger["previous_seal_sha256"] == expected_previous and len(ledger["seal_sha256"]) == 64, "stage producer or ordering differs")
            total = ledger["totals"]
            require(all(total[k] == counts["by_stage"][stage][k] for k in ("arms", "complete_arms", "partial_arms", *CLOCKS)) and
                    all(total[k] == plan["budget"][k] == 0 for k in ZERO_WORK) and total["attempted_model_steps"] <= plan["budget"]["max_model_steps_per_stage"], "stage readback counters differ")
        if guard["status"] == "COMPLETE":
            limits = plan["resources"]
            require(guard["scope"].endswith("/"+unit_name(stage)+".service") and guard["limits_verified"] and guard["returncode"] == 0 and
                    guard["child_started"] and guard["owned_cgroup_children_empty"] and len(guard["cpu_affinity"]) == limits["cpu_affinity_count"] and
                    guard["startup_effective_bytes"] >= limits["admission_min_bytes"] and guard["runtime_min_effective_bytes"] >= limits["runtime_min_bytes"] and
                    guard["supervisor_sha256"] == limits["supervisor_sha256"] and len(guard["resources_sha256"]) == len(guard["receipt_sha256"]) == 64, "resource receipt differs")


def assemble(parent, old_curves):
    curves, cells = {}, []
    fits = {f["seed"]: f for f in parent["fits"]}
    for c in parent["cells"]:
        s, r, origin = c["training_seed"], c["reset"], c["origin_x_m"]
        arms = {}
        for tag, previous in c["arms"].items():
            cell_path = f"seed_{s}/reset_{r}"
            reset_state, native_cut, tape, native_sha = _recorded_native(BASE96/"native_01"/cell_path/tag, r)
            require(native_sha == previous["native"]["source_sha256"], "native trace differs from published round 96")
            prefix = f"native_{s}_{r}_{tag}"
            tkey, nkey = prefix+"_applied16", prefix+"_cut_state45"
            curves[tkey], curves[nkey] = tape, native_cut
            arm = {"alpha": ARMS[tag], "native": parent_row(previous["native"], old_curves, origin, True), "tape_key": tkey,
                   "native_cut_key": nkey, "forecast_cut_keys": {}, "models": {}}
            for model in MODELS:
                model_cut, forecast_sha = _forecast_cut(BASE96/"predictions_01"/cell_path/model, s, r, model, tag, reset_state)
                require(forecast_sha == previous[model]["source_sha256"], "forecast differs from published round 96")
                ckey = f"closed_{s}_{r}_{model}_{tag}_cut_state45"
                curves[ckey], arm["forecast_cut_keys"][model] = model_cut, ckey
                methods = arm["models"][model] = {"closed": parent_row(previous[model], old_curves, origin, False)}
                sources = {"native_trace_sha256": native_sha, "forecast96_sha256": forecast_sha,
                           "checkpoint_sha256": fits[s]["arms"][model]["checkpoint"]["sha256"]}
                for stage, start in zip(STAGES, (model_cut, native_cut)):
                    methods[stage] = replay_row(stage, s, r, model, tag, start, tape, sources, origin, curves)
            arms[tag] = arm
        cells.append({"training_seed": s, "reset": r, "origin_x_m": origin, "arms": arms, "scores": score_cell(arms)})
    return cells, curves


def parent_data():
    plan = protocol()
    parent = read(REPORT96)
    with np.load(checked(CURVES96, parent["curves_sha256"]), allow_pickle=False) as data:
        return plan, parent, dict(data)


def publish():
    if REPORT.exists() or CURVES.exists():
        raise FileExistsError("preserve the existing suffix report or curves; use --check")
    plan, parent, old_curves = parent_data()
    stages, guards = stage_records()
    cells, curves = assemble(parent, old_curves)
    report = {"schema": "d1-suffix-replay97-v1", "protocol_sha256": PLAN_SHA, "parents_sha256": plan["parents_sha256"],
              "budget": plan["budget"], "scoring": plan["scoring"], "formal_round96": parent_decision(parent, plan),
              "stages": stages, "guards": guards, "cells": cells, "counts": work_counts(cells, stages), "summary": summarize(cells)}
    report["attribution_with_provenance"] = attributed(report)
    verify_public(report, curves, parent, old_curves, plan)
    arrays(CURVES, curves)
    report["curves_sha256"] = digest(CURVES)
    write(REPORT, report)
    return report


def check():
    plan, parent, old_curves = parent_data()
    report = read(REPORT)
    with np.load(checked(CURVES, report["curves_sha256"]), allow_pickle=False) as data:
        verify_public(report, dict(data), parent, old_curves, plan)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify published suffixes without local experiment traces")
    report = check() if parser.parse_args().check else publish()
    print(json.dumps({"counts": report["counts"], "attribution": report["attribution_with_provenance"],
                      "formal_round96_qualified": report["formal_round96"]["qualified_with_provenance"]}, indent=2))
