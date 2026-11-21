"""Publish experiment 95, or check its published scores using only NumPy and the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools94.common94 import ROOT, arrays, digest, read, write
from tools95.score95 import ARMS, HORIZON, RESETS, SEEDS, reward_m, score_cell, summarize


BASE = ROOT / "evidence/policy_direction95"
PLAN = ROOT / "docs/experiments95/policy_direction.json"
REPORT = ROOT / "reports/policy_direction95.json"
CURVES = ROOT / "reports/policy_direction95_curves.npz"
STAGES = ("directions", "predictions", "native")
PRODUCERS = ("launch95.py", "run95.py", "model95.py", "score95.py")
FLAGS = ("fallen", "illegal", "terminated", "truncated")
WHEELS = (3, 7, 11, 15)
LEGS = tuple(i for i in range(16) if i not in WHEELS)
RESET_FIELDS = ("state45", "obs643", "filtered3", "ray_due", "ray_raw16")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    require(digest(path) == expected, f"checksum differs: {path}")
    return path


def same_value(recorded, measured):
    if recorded is None or measured is None:
        return recorded is measured
    return math.isclose(recorded, measured, rel_tol=1e-13, abs_tol=1e-13)


def reset_hash(values):
    # Same canonical tensor header/data format used by the archived policy fitter.
    h = hashlib.sha256()
    for name in sorted(values):
        a = np.ascontiguousarray(values[name])
        header = json.dumps({"dtype": a.dtype.str, "shape": list(a.shape)},
                            sort_keys=True, separators=(",", ":")).encode()
        h.update(name.encode() + b"\0" + header + b"\0" + a.tobytes())
    return h.hexdigest()


def stage_seal(stage, plan_hash):
    folder = BASE / f"{stage}_01"
    path = folder / "seal.json"
    if not path.is_file():
        return None, {"status": "UNSEALED" if folder.is_dir() else "NOT_RUN", "seal_sha256": None}
    seal = read(path)
    sources = {name: digest(ROOT / "tools95" / name) for name in PRODUCERS}
    listed = {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*") if p.is_file() and p != path}
    require(seal["files"] == listed and seal["protocol_sha256"] == plan_hash and
            seal["sources"] == sources, f"{stage} seal does not match its files, code or protocol")
    return seal, {"status": "SEALED", "seal_sha256": digest(path), "files": len(listed),
                  "producer_sha256": sources}


def guard(stage, policy):
    folder = BASE / f"guard_{stage}_01"
    path = folder / "report.json"
    if not path.exists():
        return {"status": "NO_GUARD_RECEIPT"}
    record = read(path)
    log = folder / "resources.jsonl"
    samples = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    starts = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    runtime = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    limits = (policy["memory_high_bytes"], policy["memory_max_bytes"], policy["swap_max_bytes"])
    match = all((s["owned"]["high_bytes"], s["owned"]["max_bytes"], s["owned"]["swap_max_bytes"]) == limits
                and s["owned"]["swap_current_bytes"] == 0 and s["owned"]["path"] == record["scope"]
                for s in samples)
    result = {k: record.get(k) for k in ("status", "error", "scope", "returncode", "child_started",
               "owned_cgroup_children_empty", "cpu_affinity", "peak_group_bytes", "elapsed_s",
               "supervisor_sha256", "final_memory_events")}
    result.update(receipt_sha256=digest(path), resources_sha256=digest(log) if log.exists() else None,
                  startup_effective_bytes=starts[-1] if starts else None,
                  runtime_min_effective_bytes=min(runtime) if runtime else None,
                  runtime_samples=len(runtime), limits_verified=bool(samples and match))
    if result["status"] == "COMPLETE":
        require(result["child_started"] and result["returncode"] == 0 and result["owned_cgroup_children_empty"]
                and len(result["cpu_affinity"]) == policy["cpu_affinity_count"] and result["limits_verified"]
                and result["scope"].endswith(f"/d1-direction95-{stage}-01.service") and
                result["supervisor_sha256"] == policy["supervisor_sha256"] and
                starts and starts[-1] >= policy["admission_min_bytes"] and
                runtime and min(runtime) >= policy["runtime_min_bytes"], f"{stage}: resource guard mismatch")
    return result


def direction_row(seed):
    folder = BASE / f"directions_01/seed_{seed}"
    path = folder / "receipt.json"
    if not path.exists():
        return {"seed": seed, "status": "PARTIAL" if folder.exists() else "NOT_RUN",
                "error": "no direction receipt", "receipt_sha256": None}
    raw = read(path)
    require(raw["seed"] == seed and all(raw[k] == 0 for k in ("world_updates", "policy_updates", "ppo_updates")),
            f"wrong seed or persisted update in {path}")
    result = {k: v for k, v in raw.items() if k != "sources"}
    result["receipt_sha256"] = digest(path)
    if "sources" in raw:
        refs = raw["sources"]
        parent = ROOT / f"evidence/cold_start94/seed{seed}_01"
        parent_record = read(checked(parent / "report.json", refs["seed_report_sha256"]))
        checked(ROOT / "reports/cold_start94.json", refs["summary94_sha256"])
        checked(ROOT / "docs/supermimic_policy_direction95_proposal.md", refs["proposal95_sha256"])
        checked(ROOT / "tools95/model95.py", refs["model95_source_sha256"])
        for name, expected in refs["sources94"].items():
            checked(ROOT / name, expected)
        require(refs["initial"] == parent_record["initial"] and
                refs["world"] == parent_record["rounds"][0]["world_checkpoint"], "wrong registered parents")
        checked(parent / "initial.pt", refs["initial"]["sha256"])
        checked(parent / "round1/world.pt", refs["world"]["sha256"])
        fingerprints = []
        roster = parent_record["rounds"][0]["fit"][:3]
        require(len(refs["starts"]) == len(roster) == 3, "wrong number of development starts")
        for s, old in zip(refs["starts"], roster, strict=True):
            require(s["tick"] == 0 and (s["reset"], s["sha256"]) == (old["reset"], old["sha256"]),
                    "development reset selection differs")
            with np.load(checked(ROOT / s["path"], s["sha256"]), allow_pickle=False) as t:
                fingerprints.append(reset_hash({k: t[k][0:1] for k in RESET_FIELDS}))
        if "reset_tensors_sha256" in raw:
            require(raw["reset_tensors_sha256"] == fingerprints, "development t0 tensors changed")
        result["parents"] = {k: refs[k] for k in ("summary94_sha256", "seed_report_sha256", "initial", "world")}
        result["development_starts"] = refs["starts"]
    if raw["status"] == "COMPLETE":
        checked(folder / "delta.pt", raw["delta_file_sha256"])
        require(len(raw["delta_tensors_sha256"]) == 64 and raw["delta_norm"] > 0 and
                raw["shadow_adam_completed"] == raw["backward_completed"] == 1, "invalid sealed displacement")
    if "partial_states_sha256" in raw:
        checked(folder / "partial_states.npz", raw["partial_states_sha256"])
    return result


def missing(kind, curve):
    row = {"status": "NOT_RUN", "termination_reason": "no arm receipt", "ticks": 0, "safe": False,
           "return_m": None, "curve_key": curve, "error": None, "receipt_sha256": None}
    if kind == "model":
        row.update(directional_derivative=None, attempted_model_steps=0, completed_model_steps=0,
                   backward_attempted=0, backward_completed=0)
    else:
        row.update(native_decisions=0, servo_updates=0, physics_substeps=0, first_issued_key=None)
    return row


def model_arm(seed, reset, key, start, curves):
    folder = BASE / f"predictions_01/seed_{seed}/reset_{reset}"
    curve = f"s{seed}_r{reset}_{key}_model_x"
    row = missing("model", curve)
    curves[curve] = np.empty(0, dtype=np.float32)
    path = folder / f"{key}.json"
    if not path.exists():
        return row
    raw = read(path)
    require((raw["seed"], raw["reset"], raw["alpha"], raw["file"]) == (seed, reset, ARMS[key], f"{key}.npz"),
            f"wrong model arm identity: {path}")
    with np.load(checked(folder / raw["file"], raw["state_sha256"]), allow_pickle=False) as data:
        states = data["state45"]
    require(states.ndim == 2 and states.shape[1] == 45 and np.isfinite(states).all(), f"invalid model states: {path}")
    x = curves[curve] = states[:, 0].copy()
    n = max(0, len(x) - 1)
    if len(x):
        require(start is not None and np.array_equal(states[0], start["state45"][0].astype(np.float32)),
                f"wrong model starting state: {path}")
    result = reward_m(x, start["state45"][0, 0]) if start is not None else None
    require(raw["ticks"] == n and same_value(raw["return_m"], result) and
            n <= raw["completed_model_steps"] <= raw["attempted_model_steps"] <= HORIZON and
            (key == "zero" or raw["directional_derivative"] is None) and
            (not raw["safe"] or (n == HORIZON and not raw.get("error"))), f"invalid model receipt: {path}")
    row.update({k: raw[k] for k in ("safe", "directional_derivative", "attempted_model_steps",
                                     "completed_model_steps", "backward_attempted", "backward_completed")})
    row.update(status="COMPLETE" if raw["safe"] else "PARTIAL", ticks=n, return_m=result,
               termination_reason="horizon" if raw["safe"] else raw.get("error", "model stopped"),
               error=raw.get("error"), receipt_sha256=digest(path), source_sha256=raw["state_sha256"])
    return row


def native_arm(seed, reset, key, start, curves, common_first):
    folder = BASE / f"native_01/seed_{seed}/reset_{reset}/{key}"
    path = folder / "native.json"
    curve = f"s{seed}_r{reset}_{key}_native_x"
    row = missing("native", curve)
    curves[curve] = np.empty(0, dtype=np.float64)
    if not path.exists():
        if folder.exists():
            row.update(status="PARTIAL", termination_reason="no finalized native receipt")
        return row
    raw = read(path)
    row.update(receipt_sha256=digest(path), error=raw.get("error"))
    if raw["trace_file"] is None:
        row.update(status="PARTIAL", termination_reason=raw.get("error") or "no recorded trace")
        return row
    require(raw["trace_file"] in ("native/trace.npz", "native/partial_trace.npz"), f"unknown trace: {path}")
    with np.load(checked(folder / raw["trace_file"], raw["trace_sha256"]), allow_pickle=False) as t:
        states, issued, applied = t["state45"], t["issued16"], t["applied16"]
        n = len(applied)
        partial = raw["trace_file"].endswith("partial_trace.npz") or raw.get("error") is not None
        require(int(t["reset_seed"]) == reset and int(t["requested_ticks"]) == HORIZON and
                int(t["explore_seed"]) == -1 and float(t["wheel_offset_rad_s"]) == 0. and
                not t["exploration16"].any() and not t["command3"].any() and n == int(t["completed_transitions"])
                and issued.shape == applied.shape == (n, 16) and states.shape[1:] == (45,) and n <= HORIZON
                and (len(states) == n + 1 or (partial and len(states) == n == 0)), f"invalid trace: {path}")
        require(np.isfinite(states).all() and np.isfinite(issued).all() and np.all(np.abs(issued) <= 1.),
                f"invalid native endpoints/actions: {path}")
        if len(states):
            require(start is not None and all(np.array_equal(t[k][0], v[0]) for k, v in start.items()),
                    f"native initial observation differs: {path}")
        if n:
            require(not applied[0].any() and np.array_equal(applied[1:], issued[:-1]) and
                    np.array_equal(t["obs643"][1:, 41:57], issued) and np.array_equal(issued, t["raw16"]),
                    f"invalid one-tick action FIFO: {path}")
            if reset in common_first:
                require(np.array_equal(states[1], common_first[reset]), "first applied zero endpoint differs")
            else:
                common_first[reset] = states[1].copy()
            action_key = f"s{seed}_r{reset}_{key}_first_issued16"
            curves[action_key] = issued[0].copy()
            row["first_issued_key"] = action_key
        flags = {k: bool(t[k].any()) for k in FLAGS}
        require(all(t[k].shape == (n,) for k in FLAGS) and
                not np.any((t["terminated"] | t["truncated"])[:-1]), f"trace includes post-terminal rows: {path}")
        counts = [int(t[k]) for k in ("policy_steps", "servo_steps", "native_steps")]
        expected = [n, 5*n, 10*n]
        require(all(a >= b for a, b in zip(counts, expected)) if partial else counts == expected,
                f"native clocks differ: {path}")
        x = curves[curve] = states[:, 0].copy()
    safe = not partial and n == HORIZON and not any(flags.values())
    result = reward_m(x, start["state45"][0, 0]) if start is not None else None
    require(raw["ticks"] == n and all(raw[k] == v for k, v in flags.items()) and raw["safe"] == safe and
            same_value(raw["return_m"], result), f"native return, flags or completion differs: {path}")
    reason = ", ".join(k for k, active in flags.items() if active)
    row.update(status="COMPLETE" if safe else ("PARTIAL" if partial else "TERMINAL"),
               termination_reason=raw.get("error") or reason or ("horizon" if safe else "short native trace"),
               flags=flags, safe=safe, ticks=n, return_m=result, native_decisions=counts[0],
               servo_updates=counts[1], physics_substeps=counts[2], source_sha256=raw["trace_sha256"])
    return row


def action_changes(arms, curves):
    base = arms["zero"]["native"]["first_issued_key"]
    result = {}
    for name, arm in arms.items():
        ref = arm["native"]["first_issued_key"]
        if ref is None or base is None:
            result[name] = None
            continue
        delta = curves[ref].astype(np.float64) - curves[base]
        legs, wheels = np.abs(delta[list(LEGS)]), np.abs(delta[list(WHEELS)])
        result[name] = {"leg_target_count_changed": int(np.count_nonzero(legs)),
                        "wheel_target_count_changed": int(np.count_nonzero(wheels)),
                        "leg_max_abs_rad": .5 * float(legs.max()),
                        "wheel_max_abs_rad_s": 5. * float(wheels.max())}
    return result


def count_work(directions, cells):
    native = [arm["native"] for c in cells for arm in c["arms"].values()]
    models = [arm["model"] for c in cells for arm in c["arms"].values()]
    def total(rows, field):
        return sum(r.get(field, 0) for r in rows)
    return {"native_episodes": sum(r["receipt_sha256"] is not None for r in native),
            "native_decisions": total(native, "native_decisions"), "servo_updates": total(native, "servo_updates"),
            "physics_substeps": total(native, "physics_substeps"),
            "development_model_steps": total(directions, "attempted_model_steps"),
            "validation_model_steps": total(models, "attempted_model_steps"),
            "model_steps_total": total(directions + models, "attempted_model_steps"),
            "development_model_steps_completed": total(directions, "completed_model_steps"),
            "validation_model_steps_completed": total(models, "completed_model_steps"),
            "development_backwards": total(directions, "backward_attempted"),
            "validation_backwards": total(models, "backward_attempted"),
            "development_backwards_completed": total(directions, "backward_completed"),
            "validation_backwards_completed": total(models, "backward_completed"),
            "temporary_adam_displacements": total(directions, "shadow_adam_attempted"),
            "temporary_adam_displacements_completed": total(directions, "shadow_adam_completed"),
            "world_updates": total(directions, "world_updates"), "persistent_policy_updates": total(directions, "policy_updates"),
            "ppo_updates": total(directions, "ppo_updates")}


def verify_public(report, curves):
    plan = read(PLAN)
    require(report["protocol_sha256"] == digest(PLAN) and report["budget"] == plan["budget"], "protocol differs")
    directions, cells = report["directions"], report["cells"]
    require([r["seed"] for r in directions] == list(SEEDS), "missing direction status")
    used = set()
    for cell in cells:
        arms = cell["arms"]
        origin_key = f"r{cell['reset']}_physical_origin_m"
        used.add(origin_key)
        origins = curves[origin_key]
        origin = float(origins[0]) if len(origins) else None
        require(origin == cell["origin_x_m"] and set(arms) == set(ARMS), "wrong origin or missing arms")
        for key, alpha in ARMS.items():
            require(arms[key]["alpha"] == alpha, "wrong multiplier")
            for kind in ("native", "model"):
                arm = arms[key][kind]
                ref = arm["curve_key"]
                require(ref == f"s{cell['training_seed']}_r{cell['reset']}_{key}_{kind}_x", "wrong curve")
                x = curves[ref]
                used.add(ref)
                require(x.ndim == 1 and np.isfinite(x).all() and len(x) <= HORIZON + 1 and
                        arm["ticks"] == max(0, len(x)-1), f"invalid public prefix: {ref}")
                if len(x):
                    x0 = origin if kind == "native" else float(np.float32(origin))
                    require(x[0] == x0, f"initial coordinate differs: {ref}")
                measured = reward_m(x, origin) if origin is not None else None
                require(measured == arm["return_m"] and (not arm["safe"] or arm["ticks"] == HORIZON),
                        f"incorrect return or complete flag: {ref}")
                if kind == "native":
                    if "flags" in arm:
                        safe = (arm["ticks"] == HORIZON and not any(arm["flags"].values()) and
                                not arm["error"] and arm["status"] != "PARTIAL")
                        require(safe == arm["safe"], "native safety flag differs")
                    action = arm["first_issued_key"]
                    if action is not None:
                        used.add(action)
                        require(action == f"s{cell['training_seed']}_r{cell['reset']}_{key}_first_issued16"
                                and curves[action].shape == (16,) and np.isfinite(curves[action]).all(),
                                "invalid first issued action")
        ad_key = f"s{cell['training_seed']}_r{cell['reset']}_ad_m"
        used.add(ad_key)
        stored_ad = curves[ad_key]
        ad = float(stored_ad[0]) if len(stored_ad) else None
        require(ad == arms["zero"]["model"]["directional_derivative"] and
                cell["first_target_changes_vs_zero"] == action_changes(arms, curves) and
                cell["score"] == score_cell(arms), "held-out derivative, first actions or score differs")
    require(used == set(curves), "unexpected or missing public curve")
    require(report["summary"] == summarize(cells), "published nine-cell summary differs")
    counts = count_work(directions, cells)
    require(report["counts"] == counts and all(0 <= counts[k] <= limit for k, limit in report["budget"].items()),
            "reported work differs or exceeded budget")
    qualified = (report["summary"]["qualified"] and
                 all(report["stages"][s]["status"] == "SEALED" and report["guards"][s]["status"] == "COMPLETE"
                     for s in STAGES) and all(row["status"] == "COMPLETE" for row in directions))
    require(report["qualified_with_provenance"] == qualified, "qualification and provenance disagree")


def publish():
    if REPORT.exists() or CURVES.exists():
        raise FileExistsError("publication exists; use --check")
    plan_hash = digest(PLAN)
    plan = read(PLAN)
    seals, stages, guards = {}, {}, {}
    for stage in STAGES:
        seals[stage], stages[stage] = stage_seal(stage, plan_hash)
        guards[stage] = guard(stage, plan["resources"])
    require(any(seals.values()), "no sealed experimental stage to publish")
    directions = [direction_row(seed) for seed in SEEDS]
    curves, starts, common_first, cells = {}, {}, {}, []
    for reset in RESETS:
        path = BASE / f"predictions_01/start_{reset}.npz"
        start = None
        if path.exists():
            with np.load(path, allow_pickle=False) as t:
                start = dict(t)
        starts[reset] = start
        curves[f"r{reset}_physical_origin_m"] = np.asarray(
            [start["state45"][0, 0]] if start is not None else [], dtype=np.float64)
    for seed in SEEDS:
        for reset, start in starts.items():
            arms = {}
            for tag, alpha in ARMS.items():
                arms[tag] = {"alpha": alpha, "model": model_arm(seed, reset, tag, start, curves),
                             "native": native_arm(seed, reset, tag, start, curves, common_first)}
            ad = arms["zero"]["model"]["directional_derivative"]
            curves[f"s{seed}_r{reset}_ad_m"] = np.array([] if ad is None else [ad], dtype=np.float64)
            cells.append({"training_seed": seed, "reset": reset, "origin_x_m": (
                          float(start["state45"][0, 0]) if start is not None else None), "arms": arms,
                          "first_target_changes_vs_zero": action_changes(arms, curves), "score": score_cell(arms)})
    counts = count_work(directions, cells)
    sealed_dirs, sealed_models, sealed_native = (seals[s] for s in STAGES)
    if sealed_dirs is not None:
        require(len(sealed_dirs["seeds"]) == len(directions) and all(
            all(d[k] == v for k, v in sealed.items()) for d, sealed in zip(directions, sealed_dirs["seeds"], strict=True)),
            "direction seal totals differ")
    if sealed_models is not None:
        require(sealed_models["preview_physics_substeps"] == 0 and sealed_models["holdout_reset_previews"] == len(RESETS)
                and sealed_models["totals"] == {"attempted_model_steps": counts["validation_model_steps"],
                    "completed_model_steps": counts["validation_model_steps_completed"],
                    "backward_attempted": counts["validation_backwards"], "backward_completed": counts["validation_backwards_completed"]},
                "model seal totals differ")
    if sealed_native is not None:
        native_counts = {k: counts[k] for k in ("native_decisions", "servo_updates", "physics_substeps", "world_updates", "ppo_updates")}
        native_counts["policy_updates"] = counts["persistent_policy_updates"]
        require(sealed_native["native_episodes"] == counts["native_episodes"] and sealed_native["totals"] == native_counts,
                "native seal totals differ")
    summary = summarize(cells)
    report = {"schema": "d1-policy-direction95-v1", "protocol_sha256": plan_hash,
              "reward": plan["reward"], "horizon_ticks": HORIZON,
              "derivative_source": "The zero-arm analytic derivative is read from the sealed neutral forward pass; the public x curves independently reproduce returns and finite differences.",
              "budget": plan["budget"], "stages": stages, "guards": guards, "directions": directions,
              "cells": cells, "summary": summary, "counts": counts,
              "qualified_with_provenance": summary["qualified"] and
                  all(stages[s]["status"] == "SEALED" and guards[s]["status"] == "COMPLETE" for s in STAGES)
                  and all(row["status"] == "COMPLETE" for row in directions)}
    verify_public(report, curves)
    arrays(CURVES, curves)
    report["curves_sha256"] = digest(CURVES)
    write(REPORT, report)
    return report


def check():
    report = read(REPORT)
    with np.load(checked(CURVES, report["curves_sha256"]), allow_pickle=False) as curves:
        verify_public(report, dict(curves))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the public data without local simulation archives")
    result = check() if parser.parse_args().check else publish()
    print(json.dumps({"summary": result["summary"], "counts": result["counts"],
                      "qualified_with_provenance": result["qualified_with_provenance"]}, indent=2))
