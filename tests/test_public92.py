"""Independent fixed-action readback using only versioned reports and compressed curves."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PLAN = ROOT / "docs/experiments92/fixed_actions.json"
PLAN_SHA = "b40fe92e8ec9bea5c44416c2ea63a666d307800fa478ffaeca84037a662509eb"
MODES = ("closed", "fixed")
SEEDS = (91011, 91023, 91037)
SIGNALS = ("body_vx_m_s", "world_vx_m_s", "world_x_m")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_hash(fingerprint):
    return isinstance(fingerprint, str) and len(fingerprint) == 64 and len(bytes.fromhex(fingerprint)) == 32


def read_json(path):
    return json.loads(path.read_bytes())


def read_curves(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def index(rows, *fields):
    indexed = {tuple(row[k] for k in fields): row for row in rows}
    assert len(indexed) == len(rows), f"duplicate identifiers for {fields}"
    return indexed


def matches(actual, expected):
    """Compare gate booleans exactly and allow only rounding error in measured scalars."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for k in expected:
            matches(actual[k], expected[k])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            matches(a, b)
    elif isinstance(expected, float):
        assert type(actual) is not bool and actual == pytest.approx(expected, rel=0, abs=1e-12)
    elif isinstance(expected, bool) or expected is None:
        assert actual is expected
    else:
        assert actual == expected


def check_fields(stored, computed):
    for field, value in computed.items():
        matches(stored[field], value)


def value(curve):
    if curve["status"] != "COMPLETE":
        return None
    if curve["task"] == "move":
        window = [v for t, v in zip(curve["ticks"], curve["body_vx_m_s"], strict=True) if 201 <= t <= 300]
        assert len(window) == 100
        return mean(window)
    assert curve["task"] == "stand" and curve["ticks"] == list(range(51))
    xs = curve["world_x_m"]
    return max(0., xs[0] - xs[-1])


def grade(task, rule, truth, predicted):
    direction = 1 if task == "move" else -1
    delta = direction * (truth[1] - truth[0])
    complete = None not in predicted
    estimate = direction * (predicted[1] - predicted[0]) if complete else None
    effect_error = abs(estimate - delta) if complete else None
    arm_errors = [abs(actual - ref) if actual is not None else None
                  for actual, ref in zip(predicted, truth, strict=True)]
    checks = dict(both_complete=complete, positive=complete and estimate > 0,
                  beats_zero_effect=complete and effect_error < abs(delta),
                  effect_tolerance=complete and effect_error <= rule["effect_tolerance"],
                  both_absolute_arms=complete and max(arm_errors) <= rule["per_arm_tolerance"])
    return dict(truth=truth, prediction=predicted, true_effect=delta, predicted_effect=estimate,
                arm_abs_errors=arm_errors, checks=checks, passed=all(checks.values()),
                zero_effect_abs_error=abs(delta), effect_abs_error=effect_error)


def prefix(task, curves, h):
    if any(len(curve["ticks"]) <= h for curve in curves):
        return None
    a, b = curves
    result = {k: b[k][h] - a[k][h] for k in ("body_vx_m_s", "world_vx_m_s")}
    motion = [curve["world_x_m"][h] - curve["world_x_m"][0] for curve in curves]
    result["forward_displacement_m"] = motion[1] - motion[0]
    if task == "stand":
        result["rollback_reduction_m"] = max(0., -motion[0]) - max(0., -motion[1])
    return result


def extrema(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(min=None, median=None, max=None)
    return dict(min=min(values), median=median(values), max=max(values))


def distribution(grades):
    return dict(
        pairs=len(grades), complete=sum(p["checks"]["both_complete"] for p in grades),
        passed=sum(p["passed"] for p in grades),
        positive=sum(p["checks"]["positive"] for p in grades),
        effect_tolerance=sum(p["checks"]["effect_tolerance"] for p in grades),
        beats_zero_effect=sum(p["checks"]["beats_zero_effect"] for p in grades),
        both_absolute_arms=sum(p["checks"]["both_absolute_arms"] for p in grades),
        below_truth=sum(p["predicted_effect"] is not None and p["predicted_effect"] < p["true_effect"] for p in grades),
        predicted_effect=extrema(p["predicted_effect"] for p in grades),
        true_effect=extrema(p["true_effect"] for p in grades),
        effect_abs_error=extrema(p["effect_abs_error"] for p in grades),
        arm_abs_error=extrema(e for p in grades for e in p["arm_abs_errors"]),
    )


def motion_decision(grades_by_model):
    pairs = [pair for model in grades_by_model for pair in model]
    if any(not pair[mode]["checks"]["both_complete"] for pair in pairs for mode in MODES):
        return "INCOMPLETE_CONDITIONAL"
    if all(p["fixed"]["passed"] for p in pairs) and any(not p["closed"]["passed"] for p in pairs):
        return "FEEDBACK_PRIORITY"
    if (all(p["fixed"]["predicted_effect"] < p["fixed"]["true_effect"] for p in pairs)
            and all(any(not p["fixed"]["checks"]["effect_tolerance"] for p in model) for model in grades_by_model)):
        return "DYNAMICS_DATA_PRIORITY"
    return "MIXED_UNRESOLVED"


def error_counts(pairs):
    counts = dict(better=0, same=0, worse=0, unavailable=0)
    for pair in pairs:
        a, b = (pair[mode]["effect_abs_error"] for mode in MODES)
        if a is None or b is None:
            counts["unavailable"] += 1
        elif b < a:
            counts["better"] += 1
        elif b > a:
            counts["worse"] += 1
        else:
            counts["same"] += 1
    return counts


def check_summary(summary, pairs):
    for mode in MODES:
        check_fields(summary[mode], distribution([p[mode] for p in pairs]))
    matches(summary["effect_error_comparison"], error_counts(pairs))


@pytest.fixture(scope="module")
def public():
    plan = read_json(PLAN)
    report = read_json(REPORTS / "fixed_actions92.json")
    path = REPORTS / "fixed_actions92_curves.jsonl.gz"
    assert sha(path) == report["curves_sha256"]
    curves = index(read_curves(path), "family", "model_seed", "task", "reset_seed", "arm")
    assert sha(PLAN) == report["protocol_sha256"] == PLAN_SHA
    return plan, report, curves


def test_independent_grade_boundaries_and_partial_prefix():
    # Binary-exact data distinguish an inclusive tolerance from the strict null comparison.
    rule = dict(effect_tolerance=.125, per_arm_tolerance=.125)
    equal = grade("move", rule, [0., .25], [.125, .25])
    assert equal["passed"] and equal["effect_abs_error"] == .125 and equal["arm_abs_errors"][0] == .125
    tied = grade("stand", rule, [.125, 0.], [.25, 0.])
    assert tied["checks"]["effect_tolerance"] and not tied["checks"]["beats_zero_effect"] and not tied["passed"]
    bad_arms = grade("move", rule, [0., .25], [.5, .75])
    assert bad_arms["effect_abs_error"] == 0 and not bad_arms["checks"]["both_absolute_arms"]
    partial = grade("stand", rule, [.25, 0.], [.25, None])
    assert partial["predicted_effect"] is None and not partial["passed"]
    q = dict(ticks=[0, 1], world_x_m=[1., .875], body_vx_m_s=[0., .25], world_vx_m_s=[0., .5])
    z = dict(ticks=[0, 1], world_x_m=[1., 1.25], body_vx_m_s=[0., -.25], world_vx_m_s=[0., .75])
    assert prefix("stand", [q, z], 1) == dict(body_vx_m_s=-.5, world_vx_m_s=.25,
                                               forward_displacement_m=.375, rollback_reduction_m=.125)
    assert prefix("stand", [q, z], 2) is None


def test_registered_task_decisions_treat_partial_and_seed_consistency_separately():
    rule = dict(effect_tolerance=.125, per_arm_tolerance=.125)

    def pair(estimate):
        return dict(closed=grade("move", rule, [0., 1.], [0., .125]),
                    fixed=grade("move", rule, [0., 1.], [0., estimate]))

    assert motion_decision([[pair(.75)]*3 for _ in SEEDS]) == "DYNAMICS_DATA_PRIORITY"
    below = [[pair(.9375) for _ in range(3)] for _ in SEEDS]
    below[0][0] = pair(.75)
    assert motion_decision(below) == "MIXED_UNRESOLVED"
    feedback = [[pair(1.) for _ in range(3)] for _ in SEEDS]
    assert motion_decision(feedback) == "FEEDBACK_PRIORITY"
    feedback[0][0] = pair(.5)
    assert motion_decision(feedback) == "MIXED_UNRESOLVED"
    feedback[2][0] = pair(None)
    assert motion_decision(feedback) == "INCOMPLETE_CONDITIONAL"


def test_figure_retains_missing_pairs_and_rejects_inconsistent_completion(tmp_path):
    from tools92.plot92 import read_effects

    plan = read_json(PLAN)
    models = []
    for seed in SEEDS:
        pairs = []
        for task, rule in plan["tasks"].items():
            truth = [0., .012] if task == "move" else [.012, 0.]
            for reset in rule["reset_seeds"]:
                pairs.append(dict(task=task, reset_seed=reset, closed=grade(task, rule, truth, truth),
                                  fixed=grade(task, rule, truth, truth)))
        models.append(dict(seed=seed, pairs=pairs))
    report = dict(protocol_sha256=sha(PLAN), models=models)
    grade_to_remove = models[0]["pairs"][0]["fixed"]
    grade_to_remove["predicted_effect"] = None
    grade_to_remove["checks"].update(both_complete=False, effect_tolerance=False)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    measured, predicted = read_effects(path)
    assert len(measured) == 6 and len(predicted) == 36
    assert sum(v is None for v in predicted.values()) == 1
    grade_to_remove["checks"]["both_complete"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completion"):
        read_effects(path)


def test_archived_h8_training_windows_and_stationary_slope_start_coverage():
    mixed = [m for m in read_json(REPORTS / "uphill_curriculum89.json")["training"]["models"] if m["arm"] == "mixed"]
    world = read_json(REPORTS / "terrain_world91.json")
    sources = {s["training_seed"]: s["trace_sha256"] for s in world["dataset"]["sources"]}
    assert {m["train_seed"]: m["train_trace_sha256"] for m in mixed} == sources and len(sources) == 3
    ended = [e for m in mixed for e in m["episodes"] if e["terminated"] or e["truncated"]]
    assert len(ended) == world["dataset"]["completed_episodes"] == 42
    standing = [e for e in ended if e["terrain"] == "up" and e["slope_start"]]
    assert len(standing) == 6
    # The archives store post-step states. A length-H transition window needs H+1 of them.
    h = 8
    total = sum(max(0, e["ticks"]-h) for e in ended)
    assert total == 11472
    assert sum(max(0, e["ticks"]-e["moving_ticks"]-h) for e in standing) == 252
    models = world["families"]["candidate"]["models"]
    assert {m["seed"] for m in models} == set(SEEDS)
    for m in models:
        fit = m["source"]["model"]
        assert fit["status"] == "COMPLETE" and fit["horizon"] == h and fit["training_windows"] == total


def test_public_curve_inventory_origins_and_parity(public):
    plan, report, curves = public
    assert tuple(plan["model_seeds"]) == SEEDS
    expected = {(mode, s, task, reset, arm)
                for task, rule in plan["tasks"].items() for reset in rule["reset_seeds"]
                for arm in ("original", rule["intervention"])
                for mode, seeds in (("physics", (None,)), ("closed", SEEDS), ("fixed", SEEDS)) for s in seeds}
    assert curves.keys() == expected and len(curves) == 84
    old = index(read_curves(REPORTS / "terrain_world91_curves.jsonl.gz"),
                "family", "model_seed", "task", "reset_seed", "arm")
    old_stand_vx = index(read_json(REPORTS / "terrain_world91_kinematics.json")["arms"],
                         "family", "model_seed", "task", "reset_seed", "arm")
    old_report = read_json(REPORTS / "terrain_world91.json")
    old_models = index(old_report["families"]["candidate"]["models"], "seed")
    physical = {"move": index(read_json(REPORTS / "uphill_diagnosis88.json")["cases"], "seed", "arm"),
                "stand": index(read_json(REPORTS / "stand_wbc90.json")["stand"]["cases"], "reset_seed", "arm")}
    statuses = {mode: Counter() for mode in ("physics", *MODES)}
    first_issued = first_endpoint = 0
    tapes = {}
    for (mode, seed, task, reset, arm), row in curves.items():
        rule = plan["tasks"][task]
        begin, h = rule["start_tick"], rule["horizon"]
        n, attempted = row["accepted_steps"], row["attempted_steps"]
        assert 0 <= n <= attempted <= h and attempted - n <= 1
        assert row["ticks"] == list(range(begin, begin+n+1))
        assert all(len(row[k]) == n+1 and all(math.isfinite(x) for x in row[k]) for k in SIGNALS)
        if row["status"] == "COMPLETE":
            assert n == h and row["reason"] is None
        else:
            assert row["reason"]
        assert all(valid_hash(row[k]) for k in ("trace_sha256", "initial_state_sha256", "applied_sha256"))
        for k, present in (("first_applied_sha256", attempted > 0), ("first_endpoint_sha256", n > 0)):
            assert valid_hash(row[k]) if present else row[k] is None
        statuses[mode][row["status"]] += 1
        if mode in ("physics", "closed"):
            old_key = ("physics" if mode == "physics" else "candidate", seed, task, reset, arm)
            archived = old[old_key]
            for k in ("status", "ticks", "world_x_m", "body_vx_m_s"):
                assert row[k] == archived[k]
            if task == "stand":
                assert row["world_vx_m_s"] == old_stand_vx[old_key]["world_vx_m_s"]
        if mode == "physics":
            record = physical[task][reset, arm]
            assert row["status"] == "COMPLETE" and record["trace_sha256"] == row["trace_sha256"]
            assert row["input_sha256"] == row["trace_sha256"]
            field = "score_vx_m_s" if task == "move" else "rollback_m"
            matches(value(row), record[field])
        elif mode == "closed":
            assert row["input_sha256"] is None
            pair = next(p for p in old_models[seed,]["pairs"] if (p["task"], p["reset_seed"]) == (task, reset))
            source_arm = pair["arms"][0 if arm == "original" else 1]
            for k in ("attempted_steps", "accepted_steps", "trace_sha256", "status"):
                assert row[k] == source_arm[k]
        else:
            peer = curves["closed", seed, task, reset, arm]
            truth = curves["physics", None, task, reset, arm]
            assert row["parent_seed"] == peer["parent_seed"] == truth["parent_seed"]
            assert valid_hash(row["input_sha256"])
            assert tapes.setdefault((task, reset, arm), row["input_sha256"]) == row["input_sha256"]
            assert all(row[k][0] == peer[k][0] == truth[k][0] for k in SIGNALS)
            assert row["initial_state_sha256"] == peer["initial_state_sha256"] == truth["initial_state_sha256"]
            if min(attempted, peer["attempted_steps"]):
                first_issued += 1
                assert row["first_applied_sha256"] == peer["first_applied_sha256"] == truth["first_applied_sha256"]
            if n >= 1 and peer["accepted_steps"] >= 1:
                first_endpoint += 1
                assert all(row[k][1] == peer[k][1] for k in SIGNALS)
                assert row["first_endpoint_sha256"] == peer["first_endpoint_sha256"]
            if attempted == truth["attempted_steps"]:
                assert row["applied_sha256"] == truth["applied_sha256"]
    for task, rule in plan["tasks"].items():
        for reset in rule["reset_seeds"]:
            a, b = (curves["physics", None, task, reset, arm] for arm in ("original", rule["intervention"]))
            assert a["initial_state_sha256"] == b["initial_state_sha256"]
            assert a["first_applied_sha256"] == b["first_applied_sha256"]
    check_fields(report["counts"], dict(physics_arms=12, closed_arms=36, fixed_arms=36, curves=84,
                                        physical_pairs=6, model_task_pairs=18, statuses=statuses))
    check_fields(report["first_step_parity"], dict(arms=36, first_applied_equal=first_issued,
                                                    first_endpoint_equal=first_endpoint))


def test_public_seals_resource_receipts_and_budgets(public):
    plan, report, curves = public
    registered = plan["source_sha256"]
    for path, fingerprint in registered.items():
        if not path.startswith("evidence/"):
            assert sha(ROOT / path) == fingerprint
    closed = read_json(REPORTS / "terrain_world91.json")
    sources = report["sources"]
    assert sources["closed_report_sha256"] == sha(REPORTS / "terrain_world91.json")
    assert sources["closed_protocol_sha256"] == closed["protocol_sha256"]
    assert sources["closed_seal_sha256"] == closed["prediction_seals"]["candidate"] == registered[
        "evidence/terrain_world91/candidate_01/seal.json"]
    matches(sources["physical_reports_sha256"], closed["truth_sha256"])
    for path, sha256 in sources["physical_reports_sha256"].items():
        assert sha256 == registered[path]
    models = index(closed["families"]["candidate"]["models"], "seed")
    for row in report["models"]:
        seed = row["seed"]
        model = models[seed,]["source"]
        assert row["weights_sha256"] == model["model"]["weights_sha256"] == plan["weights_sha256"][str(seed)]
        assert row["closed_predictions_sha256"] == model["predictions_sha256"]
    resources, budget = plan["resources"], plan["budget"]
    assert report["resources"].keys() == {"prepare", "replay"}
    for stage, kind in (("prepare", "input"), ("replay", "replay")):
        receipt = report["resources"][stage]
        assert valid_hash(receipt["guard_sha256"])
        assert receipt["receipt_sha256"] == sources[kind+"_seal_sha256"]
        assert valid_hash(receipt["receipt_sha256"])
        assert receipt["admitted_effective_bytes"] >= resources["minimum_effective_gib_at_launch"] << 30
        assert receipt["minimum_running_effective_bytes"] >= resources["minimum_effective_gib_running"] << 30
        assert 0 < receipt["peak_group_bytes"] <= resources["memory_max_gib"] << 30
        assert receipt["elapsed_s"] > 0 and len(receipt["cpu_affinity"]) == 1
        assert all(receipt["final_memory_events"][k] == 0 for k in ("oom", "oom_kill"))
    for k in ("policy_updates", "model_updates", "new_physics_steps", "policy_calls"):
        assert report[k] == budget[k] == 0
    assert report["sensor_calls"] == 0
    counts = report["counts"]
    for mode in MODES:
        for key in ("attempted_steps", "accepted_steps"):
            assert counts[mode+"_"+key] == sum(c[key] for c in curves.values() if c["family"] == mode)
    assert counts["fixed_attempted_steps"] <= budget["max_model_calls"] == 3060
    assert counts["fixed_arms"] == budget["new_model_rollouts"] == 36
    assert counts["closed_arms"] == budget["closed_rollouts_reused"] == 36
    assert counts["physics_arms"] == budget["physical_rollouts_reused"] == 12


def test_public_scores_prefixes_and_decisions(public):
    plan, report, curves = public
    models = index(report["models"], "seed")
    assert models.keys() == {(s,) for s in SEEDS}
    archived91 = read_json(REPORTS / "terrain_world91.json")["families"]["candidate"]
    accepted_by_mode = {mode: [] for mode in MODES}
    task_grades = {task: [] for task in plan["tasks"]}
    for seed in SEEDS:
        model = models[seed,]
        pairs = index(model["pairs"], "task", "reset_seed")
        archived = next(m for m in archived91["models"] if m["seed"] == seed)
        old_pairs = index(archived["pairs"], "task", "reset_seed")
        assert pairs.keys() == old_pairs.keys()
        all_grades = []
        for task, rule in plan["tasks"].items():
            per_model = []
            for reset in rule["reset_seeds"]:
                pair = pairs[task, reset]
                assert pair["parent_seed"] == old_pairs[task, reset]["parent_seed"]
                arms = ("original", rule["intervention"])
                panel = {mode: [curves[mode, None if mode == "physics" else seed, task, reset, arm] for arm in arms]
                         for mode in ("physics", *MODES)}
                for arm, old, fixed in zip(arms, panel["closed"], panel["fixed"], strict=True):
                    parity = pair["first_step_parity"][arm]
                    n_action = min(r["attempted_steps"] for r in (old, fixed))
                    n_endpoint = min(len(r["ticks"]) for r in (old, fixed))
                    assert parity["first_applied_equal"] is (n_action > 0)
                    assert parity["first_endpoint_equal"] is (n_endpoint > 1)
                    ai, si = parity["first_different_action_index"], parity["first_different_endpoint_index"]
                    assert ai is None or 1 <= ai < n_action
                    assert si is None or (ai is not None and ai < si < n_endpoint)
                    shared = n_endpoint if si is None else si
                    assert all(old[k][:shared] == fixed[k][:shared] for k in SIGNALS)
                truth = [value(c) for c in panel["physics"]]
                computed = {}
                for mode in MODES:
                    computed[mode] = grade(task, rule, truth, [value(c) for c in panel[mode]])
                    check_fields(pair[mode], computed[mode])
                check_fields(old_pairs[task, reset], computed["closed"])
                assert [p["h"] for p in pair["prefixes"]] == rule["prefix_steps"]
                for row in pair["prefixes"]:
                    by_mode = {mode: prefix(task, arms, row["h"]) for mode, arms in panel.items()}
                    for mode, measured in by_mode.items():
                        matches(row[mode], measured)
                    for mode in MODES:
                        pred, real = by_mode[mode], by_mode["physics"]
                        errors = None if pred is None else {k: abs(pred[k]-real[k]) for k in real}
                        matches(row[mode+"_abs_errors"], errors)
                per_model.append(computed)
            check_summary(model["tasks"][task], per_model)
            task_grades[task].append(per_model)
            all_grades.extend(per_model)
        matches(model["effect_error_comparison"], error_counts(all_grades))
        for mode in MODES:
            passed = all(p[mode]["passed"] for p in all_grades)
            assert model[mode+"_passed"] is passed
            accepted_by_mode[mode].append(passed)
    decisions = []
    for task, by_model in task_grades.items():
        summary = report["tasks"][task]
        pairs = [p for model in by_model for p in model]
        assert len(pairs) == 9
        check_summary(summary, pairs)
        decision = motion_decision(by_model)
        assert decision == summary["decision"]
        assert summary["qualified"] is all(p["fixed"]["passed"] and p["closed"]["checks"]["both_complete"] for p in pairs)
        assert summary["all_fixed_effects_positive"] is all(p["fixed"]["checks"]["positive"] for p in pairs)
        decisions.append(decision)
    assert report["decision"] == ("INCOMPLETE_CONDITIONAL" if "INCOMPLETE_CONDITIONAL" in decisions else "COMPLETE_DIAGNOSTIC")
    for mode in MODES:
        assert report[mode+"_family_passed"] is all(accepted_by_mode[mode])
