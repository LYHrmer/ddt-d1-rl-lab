"""Reconcile the full physical grid, model predictions and contact-conditioned errors."""
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median

import numpy as np

from tools91.score91 import curves, grade
from tools92.score92 import array_hash
from tools93.common93 import ARMS, BASE, MODELS, PARENTS, PLAN_SHA, RESETS, ROOT, checked_plan, digest, read, require, write


DIAGONAL = set(zip(PARENTS, RESETS, strict=True))
CASES = [(p, r, a) for p in PARENTS for r in RESETS for a in ARMS]
WHEELS = [9, 13, 17, 21]
HORIZON = 50


def indexed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(result) == len(rows), "duplicate case identity")
    return result


def source_path(folder, row):
    path = folder / row["trace"]
    require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            and digest(path) == row["trace_sha256"], f"source differs: {path}")
    return path


def observation_row(family, model, key, state, actions, record):
    p, r, a = key
    require(state.dtype == np.float64 and state.ndim == 2 and state.shape[1] == 45
            and 0 < len(state) <= HORIZON+1 and actions.shape[1:] == (16,) and np.isfinite(state).all()
            and np.isfinite(actions).all() and np.array_equal(actions, actions.astype(np.float32)),
            "invalid full-state history or lossy action tape")
    x, vx = curves(state[:, :23], state[:, 23:])
    return dict(family=family, model_seed=model, parent_seed=p, reset_seed=r, arm=a,
                status=record["status"], trace_sha256=record["trace_sha256"],
                first_state_sha256=array_hash(state[0], "<f8"),
                first_endpoint_sha256=array_hash(state[1], "<f8") if len(state)>1 else None,
                first_applied_sha256=array_hash(actions[0], "<f4") if len(actions) else None,
                applied_sha256=array_hash(actions, "<f4"), ticks=list(range(len(state))),
                world_x_m=x.tolist(), body_vx_m_s=vx.tolist(), world_vx_m_s=state[:,23].tolist(),
                wheel_speed_rad_s=state[1:, np.array(WHEELS)+23].tolist())


def physical_panel(plan):
    new = read(BASE / "collect_01/seal.json")
    old = read(ROOT / "evidence/wbc90/stand_01/report.json")
    pub = read(ROOT / "reports/stand_wbc90.json")
    require(old == pub["stand"] and new["protocol_sha256"] == PLAN_SHA and new["status"] == "COMPLETE",
            "physical input seal differs")
    keys = ("parent_seed", "reset_seed", "arm")
    rows = indexed(old["cases"] + new["cases"], keys)
    require(set(rows) == set(CASES), "physical grid incomplete")
    panel, paths = {}, {}
    for key in CASES:
        p, r, a = key
        archive = (p, r) in DIAGONAL
        row = rows[key]
        require(row["clean"] and row["status"] == "COMPLETE" and row["ticks"] == (100 if archive else 50)
                and row["policy_sha256"] == plan["policies_sha256"][str(p)], "unqualified physical arm")
        path = source_path(ROOT / "evidence/wbc90/stand_01" if archive else BASE / "collect_01", row)
        paths[key] = (path, row["trace_sha256"])
        with np.load(path, allow_pickle=False) as raw:
            q, v = (np.vstack((raw["initial_"+k],raw[k][:50])) for k in ("qpos","qvel"))
            actions, issued = raw["applied"][:50], raw["issued"][:50]
            require(not actions[0].any() and np.array_equal(actions[1:], issued[:-1]) and not raw["command"][:50].any()
                    and np.allclose(raw["time_s"][:50], np.arange(1,51)*.02, atol=1e-9, rtol=0)
                    and not (raw["illegal"][:50].any() or raw["fallen"][:50].any())
                    and raw["wheel_dof_indices"].tolist() == WHEELS, "physical clock, intervention or actuator mismatch")
            item = observation_row("physics", None, key, np.hstack((q,v)), actions, row)
            masks = raw["wheel_contacts_past_toe"][:50]
            require(masks.dtype == bool and masks.shape == (50,4), "invalid physical support masks")
            item.update(loaded_mask=(masks @ np.array([1,2,4,8],dtype=np.uint8)).tolist(),
                        pitch_rad=raw["pitch_rad"][:50].tolist(), initial_state_observation_sha256=row["initial_state_sha256"],
                        policy_sha256=row["policy_sha256"], torque_saturated_ticks=int(np.count_nonzero(raw["wheel_saturation_count"][:50])))
        require(abs(rollback(item, 50)-row["rollback_m"]) < 1e-12, "physical rollback differs")
        panel[key] = item
    for r in RESETS:
        arms = [panel[k] for k in CASES if k[1] == r]
        require(all(len({v[k] for v in arms}) == 1 for k in
                    ("first_state_sha256","first_endpoint_sha256","first_applied_sha256","initial_state_observation_sha256")),
                "physical starts or first endpoints differ within reset")
    require(len({v["first_state_sha256"] for v in panel.values()}) == len(RESETS), "reset states not distinct")
    return panel, paths, new


def fixed_panel(plan, physics):
    current = read(BASE / "predict_01/seal.json")
    input_file = BASE / "inputs_01/seal.json"
    inputs = read(input_file)
    require(current["protocol_sha256"] == inputs["protocol_sha256"] == PLAN_SHA and current["status"] == "COMPLETE"
            and inputs["status"] == "COMPLETE" and current["input_seal_sha256"] == digest(input_file)
            and inputs["physical_seal_sha256"] == digest(BASE/"collect_01/seal.json")
            and current["weights_sha256"] == plan["weights_sha256"]
            and all(current[k] == 0 for k in ("physics_steps","sensor_calls","actor_calls","model_updates","policy_updates")),
            "invalid fixed-action data chain")
    prepared = indexed(inputs["cases"], ("parent_seed","reset_seed","arm"))
    old_root = ROOT / "evidence/terrain_replay92/replay_01"
    old = read(old_root/"seal.json")
    require(old["status"] == "COMPLETE" and old["weights_sha256"] == plan["weights_sha256"], "unknown diagonal family")
    records = [r for r in old["cases"] if r["task"]=="stand"] + current["cases"]
    rows = indexed(records, ("model_seed","parent_seed","reset_seed","arm"))
    require(set(rows) == {(m,*k) for m in MODELS for k in CASES}, "incomplete predicted inventory")
    panel = {}
    for key, row in rows.items():
        m,p,r,a = key
        archive = (p,r) in DIAGONAL
        source = physics[p,r,a]
        if not archive:
            tape = prepared[p,r,a]
            require(tape["source_trace_sha256"] == source["trace_sha256"] and
                    tape["file_sha256"] == row["input_sha256"], "prediction not bound to physical branch")
        with np.load(source_path(old_root if archive else BASE/"predict_01", row),allow_pickle=False) as raw:
            state, actions = raw["state45"], raw["applied16"]
            require(np.array_equal(raw["endpoint_tick"], np.arange(len(state))) and
                    (len(state)-1,len(actions)) == (row["accepted_steps"],row["attempted_steps"]), "bad fixed prediction cut")
            item = observation_row("fixed", m, (p,r,a), state, actions, row)
        require(item["first_state_sha256"] == source["first_state_sha256"] and
                (not len(actions) or item["first_applied_sha256"] == source["first_applied_sha256"]),
                "fixed initial state or applied action differs")
        if row["status"] == "COMPLETE":
            require(len(state)==51 and item["applied_sha256"] == source["applied_sha256"], "fixed actions not identical")
        panel[key] = item
    for m in MODELS:
        for r in RESETS:
            anchor = panel[m, PARENTS[RESETS.index(r)], r,"original"]
            for p in PARENTS:
                for a in ARMS:
                    branch=panel[m,p,r,a]
                    require(len(branch["ticks"]) < 2 or branch["first_endpoint_sha256"] == anchor["first_endpoint_sha256"],
                            "first predicted endpoint differs from same-reset archived model")
    return panel, current


def rollback(curve, h):
    x = curve["world_x_m"]
    return max(0., x[0]-x[h]) if len(x)>h else None


def late_tags(masks):
    require(len(masks) == 50, "missing contact endpoints")
    # masks[0] is physical endpoint 1; the inclusive span at h is h-8..h.
    return {h: all(m == 15 for m in masks[h-9:h]) for h in range(33,51)}


def late_contrast(measured, predicted):
    tags = late_tags(measured["loaded_mask"])
    if predicted["status"] != "COMPLETE" or len(set(tags.values())) < 2:
        return None
    actual, est = measured["world_x_m"], predicted["world_x_m"]
    errors = {h: abs((est[h]-est[h-8])-(actual[h]-actual[h-8])) for h in tags}
    return median(errors[h] for h in tags if tags[h]) - median(errors[h] for h in tags if not tags[h])


def sign_label(errors):
    if None in errors:
        return "INCOMPLETE"
    if min(errors)>0:
        return "HIGH"
    if max(errors)<0:
        return "LOW"
    return "MIXED"


def summarize_cell(p, r, rule, physics, fixed):
    physical = [physics[p,r,a] for a in ARMS]
    truth = [rollback(v,HORIZON) for v in physical]
    models, contrasts = [], []
    errors = {h:[] for h in (16,32,50)}
    for m in MODELS:
        arms = [fixed[m,p,r,a] for a in ARMS]
        pred = [rollback(b,50) if b["status"]=="COMPLETE" else None for b in arms]
        models.append(dict(model_seed=m, grade=grade(rule,*pred,*truth)))
        for actual, est in zip(physical, arms, strict=True):
            contrasts.append(late_contrast(actual,est))
            for h in errors:
                value = rollback(est,h)
                errors[h].append(None if value is None or (h==50 and est["status"]!="COMPLETE") else value-rollback(actual,h))
    available = [v for v in contrasts if v is not None]
    return dict(parent_seed=p,reset_seed=r,source="90/92" if (p,r) in DIAGONAL else "93",
                truth=truth,true_effect_m=truth[0]-truth[1],models=models,
                bias={str(h):dict(sign=sign_label(values),errors_m=values) for h, values in errors.items()},
                late_persistent_by_arm={a:sum(late_tags(physics[p,r,a]["loaded_mask"]).values()) for a in ARMS},
                late_arm_model_contrasts=contrasts,late_contrast_m=median(available) if available else None)


def attribution(cells):
    signs = {(c["parent_seed"],c["reset_seed"]):c["bias"]["50"]["sign"] for c in cells}
    if len(signs)!=9 or "INCOMPLETE" in signs.values():
        return "INCOMPLETE"
    if set(signs.values()) == {"LOW","HIGH"}:
        if all(len({signs[p,r] for p in PARENTS})==1 for r in RESETS):
            return "RESET_PATTERN"
        if all(len({signs[p,r] for r in RESETS})==1 for p in PARENTS):
            return "PARENT_PATTERN"
    return "MIXED"


def curriculum(cells, training_late_all_four):
    eligible = [c for c in cells if c["late_contrast_m"] is not None]
    exposed = [c for c in cells if any(c["late_persistent_by_arm"].values())]
    def covers(rows):
        return ({c["parent_seed"] for c in rows} == set(PARENTS)
                and {c["reset_seed"] for c in rows} == set(RESETS))

    gap = training_late_all_four == 0 and covers(exposed)
    counts, medians = {}, {}
    for r in RESETS:
        values = [c["late_contrast_m"] for c in eligible if c["reset_seed"]==r]
        counts[str(r)] = len(values)
        medians[str(r)] = median(values) if values else None
    advise = bool(gap and len(eligible)>=6 and covers(eligible) and attribution(cells)!="INCOMPLETE"
                  and all(v is not None and v>0 for v in medians.values()))
    return dict(training_late_all_four=training_late_all_four, eligible_cells=len(eligible), eligible_by_reset=counts,
                reset_median_contrasts_m=medians,coverage_gap=gap,advise_sampling_test=advise)


def run(out, resources):
    from tools93.dataset93 import export
    out = Path(out)
    out.mkdir(exist_ok=False)
    plan = checked_plan()
    physics, paths, collected = physical_panel(plan)
    fixed, predicted = fixed_panel(plan,physics)
    corpus = export(out / "validation.npz", paths)
    rule = read(ROOT/"docs/experiments91/terrain_world_model.json")["tasks"]["stand"]
    cells = [summarize_cell(p,r,rule,physics,fixed) for p in PARENTS for r in RESETS]
    registered = read(ROOT/"reports/fixed_actions92.json")
    old_grades = {(m["seed"],p["parent_seed"],p["reset_seed"]):p["fixed"] for m in registered["models"]
                  for p in m["pairs"] if p["task"]=="stand"}
    for c in cells:
        if (c["parent_seed"],c["reset_seed"]) in DIAGONAL:
            for m in c["models"]:
                require(m["grade"]==old_grades[m["model_seed"],c["parent_seed"],c["reset_seed"]], "diagonal grade differs")
    curves_path = out/"curves.jsonl.gz"
    with curves_path.open("xb") as stream, gzip.GzipFile(fileobj=stream,filename="",mode="wb",mtime=0) as gz:
        for row in [*physics.values(),*fixed.values()]:
            gz.write((json.dumps(row,separators=(",",":"),allow_nan=False)+"\n").encode())
    counts=dict(physical_cells=9,physical_arms=len(physics),reused_physical_arms=6,new_physical_arms=len(collected["cases"]),
                new_policy_steps=collected["policy_transitions"],new_pd_updates=collected["pd_updates"],
                new_native_steps=collected["native_steps"],model_pairs=sum(len(c["models"]) for c in cells),
                fixed_arms=len(fixed),reused_fixed_arms=18,new_fixed_arms=len(predicted["cases"]),
                new_model_steps=predicted["attempted_steps"],new_model_accepted_steps=predicted["accepted_steps"],
                world_model_updates=predicted["model_updates"],policy_updates=collected["policy_updates"],physical_reset_states=3)
    require(counts["new_policy_steps"]==600 and counts["new_native_steps"]==6000 and counts["new_pd_updates"]==3000
            and counts["new_model_steps"]<=1800 and counts["model_pairs"]==27, "work differs from registration")
    coverage = BASE/"coverage_01/coverage.json"
    train_late = read(coverage)["slope_stand"]["by_last_tick"][-1]
    require(train_late["first_last_tick"]==33 and train_late["final_last_tick"]==50, "wrong training window band")
    advice = curriculum(cells,train_late["source_and_targets"]["groups"]["all_four"]["windows"])
    passed = Counter(m["model_seed"] for c in cells for m in c["models"] if m["grade"]["passed"])
    report = dict(status="COMPLETE" if predicted["all_arms_complete"] else "INCOMPLETE_MODEL_DIAGNOSTIC",
                  physical_data_complete=True, protocol_sha256=PLAN_SHA, resources=resources, counts=counts, cells=cells,
                  model_pair_passed=sum(passed.values()),per_model_passed={str(m):passed[m] for m in MODELS},
                  all_models_qualified=all(passed[m]==9 for m in MODELS),attribution=attribution(cells),curriculum=advice,
                  curves_sha256=digest(curves_path),coverage_sha256=digest(coverage),validation_dataset=corpus,
                  dataset_exporter_sha256=digest(ROOT / "tools93/dataset93.py"),
                  first_endpoint_checks=dict(physical_arms_matching_diagonal=18,
                                             fixed_arms_matching_diagonal=sum(len(v["ticks"]) >= 2 for v in fixed.values())))
    write(out/"report.json",report)
    return report
