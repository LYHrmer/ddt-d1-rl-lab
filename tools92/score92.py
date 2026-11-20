"""Score sealed fixed-action rollouts against the original physics and closed-loop arms."""
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

from tools91 import score91
from tools92.common92 import PLAN_SHA, ROOT, checked_plan, digest, read, require, write


CLOSED_DIR = ROOT / "evidence/terrain_world91/candidate_01"
PHYSICS_DIR = {"move": ROOT / "evidence/uphill88/probe_01", "stand": ROOT / "evidence/wbc90/stand_01"}
PHYSICS_REPORT = {"move": "reports/uphill_diagnosis88.json", "stand": "reports/stand_wbc90.json"}
CLOSED_REPORT = "reports/terrain_world91.json"
CLOSED_PLAN = "docs/experiments91/terrain_world_model.json"
FAMILIES = ("physics", "closed", "fixed")


def keyed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(result) == len(rows), f"duplicate {keys}")
    return result


def verified_path(folder, relative, expected):
    path = (folder / relative).resolve()
    require(path.is_relative_to(folder.resolve()) and digest(path) == expected, f"seal differs: {relative}")
    return path


def array_hash(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


@dataclass
class Arm:
    ticks: np.ndarray
    state: np.ndarray
    applied: np.ndarray
    status: str
    reason: str | None
    trace_sha256: str
    input_sha256: str | None
    parent: int
    x: np.ndarray
    body_vx: np.ndarray
    world_vx: np.ndarray
    score: float | None

    def public(self, family, seed, key):
        task, reset, arm = key
        return dict(family=family, model_seed=seed, task=task, reset_seed=reset, parent_seed=self.parent,
                    arm=arm, status=self.status, reason=self.reason, accepted_steps=len(self.ticks) - 1,
                    attempted_steps=len(self.applied), trace_sha256=self.trace_sha256, input_sha256=self.input_sha256,
                    initial_state_sha256=array_hash(self.state[0], "<f8"),
                    first_applied_sha256=array_hash(self.applied[0], "<f4") if len(self.applied) else None,
                    first_endpoint_sha256=array_hash(self.state[1], "<f8") if len(self.state) > 1 else None,
                    applied_sha256=array_hash(self.applied, "<f4"),
                    ticks=self.ticks.tolist(), world_x_m=self.x.tolist(), body_vx_m_s=self.body_vx.tolist(),
                    world_vx_m_s=self.world_vx.tolist())


def arm_from_arrays(task, rule, ticks, state, applied, status, reason, sha, input_sha, parent):
    steps = len(ticks) - 1
    horizon = rule["horizon"]
    expected_ticks = rule["start_tick"] + np.arange(steps + 1)
    require(0 <= steps <= len(applied) <= horizon and ticks.dtype.kind in "iu"
            and np.array_equal(ticks, expected_ticks), "ticks exceed or skip a registered interval")
    require(state.shape == (steps + 1, 45) and applied.shape == (len(applied), 16)
            and np.isfinite(state).all() and np.isfinite(applied).all(), "invalid accepted states or actions")
    require(np.all(np.abs(np.linalg.norm(state[:, 3:7], axis=1) - 1.) <= 1e-4), "invalid accepted quaternion")
    require(status != "COMPLETE" or steps == horizon, "complete label on a truncated arm")
    x, body_vx = score91.curves(state[:, :23], state[:, 23:])
    value = score91.scalar(task, ticks, x, body_vx) if status == "COMPLETE" else None
    return Arm(ticks, state, applied, status, reason, sha, input_sha, parent, x, body_vx, state[:, 23], value)


def load_physics(plan, rules91, inputs_dir, input_rows):
    panel = {}
    for task, rule in plan["tasks"].items():
        report = read(ROOT / PHYSICS_REPORT[task])
        cases = (report if task == "move" else report["stand"])["cases"]
        for case in cases:
            reset = case.get("reset_seed", case.get("seed"))
            arm = case["arm"]
            key = task, reset, arm
            require(key in input_rows and key not in panel, "unexpected physical case")
            source = input_rows[key]
            path = verified_path(PHYSICS_DIR[task], f"{arm}_{reset}/trace.npz", case["trace_sha256"])
            horizon, first = rule["horizon"], rule["start_tick"]
            with np.load(path, allow_pickle=False) as raw:
                if task == "move":
                    require(case["full_clean"] and case["ticks"] == 300, "incomplete physical move reference")
                    q, v = raw["qpos"][first:first+horizon+1], raw["qvel"][first:first+horizon+1]
                    actions = raw["applied_action"][first+1:first+horizon+1]
                    logged, parent = case["score_vx_m_s"], 25
                else:
                    require(case["clean"] and case["status"] == "COMPLETE" and case["ticks"] == 100,
                            "incomplete physical stand reference")
                    q = np.vstack((raw["initial_qpos"], raw["qpos"][:horizon]))
                    v = np.vstack((raw["initial_qvel"], raw["qvel"][:horizon]))
                    actions = raw["applied"][:horizon]
                    logged, parent = case["rollback_m"], case["parent_seed"]
                    require(parent == rules91[task]["parent_seeds"][rule["reset_seeds"].index(reset)], "wrong parent")
            ticks = first + np.arange(horizon + 1)
            item = arm_from_arrays(task, rule, ticks, np.hstack((q, v)), actions, "COMPLETE", None,
                                   case["trace_sha256"], case["trace_sha256"], parent)
            require(score91.close(logged, item.score) and source["parent_seed"] == parent
                    and source["start_tick"] == first and source["horizon"] == horizon
                    and source["source_trace_sha256"] == item.trace_sha256, "physical reference or boundary differs")
            path = verified_path(inputs_dir, source["file"], source["file_sha256"])
            with np.load(path, allow_pickle=False) as prepared:
                require(set(prepared.files) == {"state45", "applied16"} and prepared["applied16"].dtype == np.float32
                        and np.array_equal(prepared["state45"], item.state[0])
                        and np.array_equal(prepared["applied16"], actions), "input differs from same-arm physical tape")
            panel[key] = item
        for reset in rule["reset_seeds"]:
            original = panel[task, reset, "original"]
            changed = panel[task, reset, rule["intervention"]]
            require(np.array_equal(original.state[0], changed.state[0])
                    and np.array_equal(original.applied[0], changed.applied[0]), "unequal physical branch starts")
    require(panel.keys() == input_rows.keys(), "physical inventory differs from input seal")
    return panel


def load_predictions(family, rows, folder, plan, truth, input_rows):
    panel = {}
    for key, row in rows.items():
        task = key[0]
        observed = truth[key]
        path = verified_path(folder, row["trace"], row["trace_sha256"])
        require(row["parent_seed"] == observed.parent, "prediction parent differs")
        is_fixed = family == "fixed"
        count = row["accepted_steps"] if is_fixed else row["steps"]
        reason = row["reason"] if is_fixed else row["error"]
        source = input_rows[key]
        input_sha = source["file_sha256"] if is_fixed else None
        with np.load(path, allow_pickle=False) as trace:
            item = arm_from_arrays(task, plan["tasks"][task], trace["endpoint_tick"], trace["state45"],
                                   trace["applied16"], row["status"], reason, row["trace_sha256"], input_sha,
                                   observed.parent)
        n = len(item.applied)
        require(len(item.ticks) == count + 1 and row["attempted_steps"] == n
                and np.array_equal(item.state[0], observed.state[0]), "prediction boundary or receipt differs")
        if is_fixed:
            require(row["input_sha256"] == input_sha and row["start_tick"] == source["start_tick"]
                    and row["horizon"] == source["horizon"]
                    and np.array_equal(item.applied, observed.applied[:n]), "fixed applied action chronology differs")
        else:
            require(row["last_endpoint"] == item.ticks[-1] and score91.close(item.score, row["score"])
                    and row["start_sha256"] == source["start_sha256"], "closed arm differs from its seal")
        panel[key] = item
    require(panel.keys() == truth.keys(), "incomplete prediction inventory")
    return panel


def first_step(closed, fixed):
    """Prove alignment at the common causal boundary before interpreting later error."""
    require(np.array_equal(closed.state[0], fixed.state[0]), "initial states differ")
    shared_actions = min(len(closed.applied), len(fixed.applied))
    shared_states = min(len(closed.state), len(fixed.state))
    if shared_actions:
        require(closed.applied[0].dtype == fixed.applied[0].dtype
                and closed.applied[0].tobytes() == fixed.applied[0].tobytes(), "first applied action differs from 91")
    if shared_states > 1:
        require(closed.state[1].dtype == fixed.state[1].dtype
                and closed.state[1].tobytes() == fixed.state[1].tobytes(), "first model endpoint differs from 91")
    differences = np.flatnonzero(np.any(closed.applied[:shared_actions] != fixed.applied[:shared_actions], axis=1))
    first_action = int(differences[0]) if len(differences) else None
    differences = np.flatnonzero(np.any(closed.state[:shared_states] != fixed.state[:shared_states], axis=1))
    first_state = int(differences[0]) if len(differences) else None
    require(first_state is None or (first_action is not None and first_state > first_action),
            "model states diverged before their applied actions")
    return dict(first_applied_equal=shared_actions > 0, first_endpoint_equal=shared_states > 1,
                first_different_action_index=first_action, first_different_endpoint_index=first_state)


def paired_prefix(task, original, changed, h):
    if h >= min(len(original.ticks), len(changed.ticks)):
        return None
    dx_a = float(original.x[h] - original.x[0])
    dx_b = float(changed.x[h] - changed.x[0])
    result = dict(body_vx_m_s=float(changed.body_vx[h] - original.body_vx[h]),
                  world_vx_m_s=float(changed.world_vx[h] - original.world_vx[h]),
                  forward_displacement_m=dx_b - dx_a)
    if task == "stand":
        result["rollback_reduction_m"] = max(0., -dx_a) - max(0., -dx_b)
    return result


def prefix_errors(estimate, physical):
    if estimate is None or physical is None:
        return None
    return {quantity: abs(estimate[quantity] - value) for quantity, value in physical.items()}


def check_archived_grade(calculated, published):
    for k, actual in calculated.items():
        expected = published[k]
        if isinstance(actual, list):
            valid = len(actual) == len(expected) and all(score91.close(a, b) for a, b in zip(actual, expected))
        elif k in ("checks", "passed"):
            valid = actual == expected
        else:
            valid = score91.close(actual, expected)
        require(valid, "reconstructed gate differs from public 91: " + k)


def score_pair(task, reset, rule, old_rule, panels, published):
    keys = [(task, reset, a) for a in ("original", rule["intervention"])]
    arms = {family: [panel[k] for k in keys] for family, panel in panels.items()}
    true_scores = [a.score for a in arms["physics"]]
    pair = dict(task=task, reset_seed=reset, parent_seed=arms["physics"][0].parent)
    for family in ("closed", "fixed"):
        pair[family] = score91.grade(old_rule, *(a.score for a in arms[family]), *true_scores)
    check_archived_grade(pair["closed"], published)
    require(published["parent_seed"] == pair["parent_seed"] and
            all(a.trace_sha256 == p["trace_sha256"] and a.status == p["status"]
                for a, p in zip(arms["closed"], published["arms"], strict=True)), "public arm receipt differs")
    pair["first_step_parity"] = {key[2]: first_step(panels["closed"][key], panels["fixed"][key]) for key in keys}
    prefixes = []
    for h in rule["prefix_steps"]:
        record = {family: paired_prefix(task, *items, h) for family, items in arms.items()}
        prefixes.append(dict(h=h, **record, closed_abs_errors=prefix_errors(record["closed"], record["physics"]),
                             fixed_abs_errors=prefix_errors(record["fixed"], record["physics"])))
    pair["prefixes"] = prefixes
    return pair


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(min=min(values), median=median(values), max=max(values)) if values else dict(min=None, median=None, max=None)


def summarize(pairs):
    summary = {}
    for family in ("closed", "fixed"):
        grades = [p[family] for p in pairs]
        checks = {key: sum(g["checks"][key] for g in grades)
                  for key in ("both_complete", "positive", "effect_tolerance", "beats_zero_effect", "both_absolute_arms")}
        summary[family] = dict(pairs=len(grades), complete=checks.pop("both_complete"), **checks,
                               passed=sum(g["passed"] for g in grades),
                               below_truth=sum(g["predicted_effect"] is not None and g["predicted_effect"] < g["true_effect"]
                                               for g in grades),
                               predicted_effect=distribution(g["predicted_effect"] for g in grades),
                               true_effect=distribution(g["true_effect"] for g in grades),
                               effect_abs_error=distribution(g["effect_abs_error"] for g in grades),
                               arm_abs_error=distribution(e for g in grades for e in g["arm_abs_errors"]))
    counts = dict(better=0, same=0, worse=0, unavailable=0)
    for p in pairs:
        before, after = p["closed"]["effect_abs_error"], p["fixed"]["effect_abs_error"]
        if before is None or after is None:
            counts["unavailable"] += 1
        elif after < before:
            counts["better"] += 1
        elif after > before:
            counts["worse"] += 1
        else:
            counts["same"] += 1
    summary["effect_error_comparison"] = counts
    return summary


def task_summary(by_model):
    result = summarize([p for group in by_model for p in group])
    fixed, closed = result["fixed"], result["closed"]
    n = fixed["pairs"]
    require(len(by_model) == 3 and all(len(group) == 3 for group in by_model), "task requires three matched 3-reset models")
    complete = fixed["complete"] == closed["complete"] == n
    result["qualified"] = complete and fixed["passed"] == n
    result["all_fixed_effects_positive"] = fixed["positive"] == n
    if not complete:
        decision = "INCOMPLETE_CONDITIONAL"
    elif result["qualified"] and closed["passed"] < n:
        decision = "FEEDBACK_PRIORITY"
    elif fixed["below_truth"] == n and all(any(not p["fixed"]["checks"]["effect_tolerance"] for p in group)
                                           for group in by_model):
        decision = "DYNAMICS_DATA_PRIORITY"
    else:
        decision = "MIXED_UNRESOLVED"
    result["decision"] = decision
    return result


def run(out, replay_dir, inputs_dir, *, resources=None):
    out, replay_dir, inputs_dir = map(Path, (out, replay_dir, inputs_dir))
    out.mkdir(parents=True, exist_ok=False)
    plan = checked_plan()
    old_plan, archive = read(ROOT / CLOSED_PLAN), read(ROOT / CLOSED_REPORT)
    seeds, rules = plan["model_seeds"], plan["tasks"]
    expected_keys = {(t, s, a) for t, r in rules.items() for s in r["reset_seeds"] for a in ("original", r["intervention"])}
    input_file, replay_file, closed_file = inputs_dir / "seal.json", replay_dir / "seal.json", CLOSED_DIR / "seal.json"
    input_seal, replay_seal, closed_seal = map(read, (input_file, replay_file, closed_file))
    input_sha, replay_sha, closed_sha = map(digest, (input_file, replay_file, closed_file))
    require(input_seal["protocol_sha256"] == replay_seal["protocol_sha256"] == PLAN_SHA
            and input_seal["status"] == replay_seal["status"] == "COMPLETE"
            and replay_seal["input_seal_sha256"] == input_sha and replay_seal["model_seeds"] == seeds
            and replay_seal["weights_sha256"] == plan["weights_sha256"], "invalid fixed input or prediction seal")
    counters = {k: replay_seal[k] for k in ("model_updates", "policy_updates", "sensor_calls")}
    counters.update(policy_calls=replay_seal["actor_calls"], new_physics_steps=replay_seal["physics_steps"])
    require(all(v == 0 for v in counters.values()), "replay used an unregistered update, sensor, actor or physics call")
    require(closed_sha == archive["prediction_seals"]["candidate"] and closed_seal["family"] == "candidate"
            and closed_seal["protocol_sha256"] == archive["protocol_sha256"] == plan["source_sha256"][CLOSED_PLAN],
            "invalid archived 91 candidate seal")
    keys = ("task", "reset_seed", "arm")
    input_rows = keyed(input_seal["cases"], keys)
    fixed_rows = keyed(replay_seal["cases"], ("model_seed", *keys))
    require(input_rows.keys() == expected_keys and fixed_rows.keys() == {(s, *k) for s in seeds for k in expected_keys},
            "fixed input or predicted inventory differs from registration")
    physical = load_physics(plan, old_plan["tasks"], inputs_dir, input_rows)
    models, curves = [], [row.public("physics", None, k) for k, row in physical.items()]
    pub_models = keyed(archive["families"]["candidate"]["models"], ("seed",))
    require([m["seed"] for m in closed_seal["models"]] == seeds and set(pub_models) == {(s,) for s in seeds},
            "unexpected archived models")
    for source in closed_seal["models"]:
        seed = source["seed"]
        public = pub_models[seed,]
        require(public["source"] == source and source["model"]["weights_sha256"] == plan["weights_sha256"][str(seed)],
                "closed models differ from registered checkpoints")
        folder = CLOSED_DIR / f"seed{seed}"
        suite = read(verified_path(folder, "predictions.json", source["predictions_sha256"]))
        require(suite["status"] == "COMPLETE", "archived prediction collection is incomplete")
        closed = load_predictions("closed", keyed(suite["cases"], keys), folder, plan, physical, input_rows)
        fixed = load_predictions("fixed", {k: fixed_rows[(seed, *k)] for k in physical},
                                 replay_dir, plan, physical, input_rows)
        for family, panel in (("closed", closed), ("fixed", fixed)):
            curves.extend(a.public(family, seed, k) for k, a in panel.items())
        panels = dict(physics=physical, closed=closed, fixed=fixed)
        pub_pairs = keyed(public["pairs"], ("task", "reset_seed"))
        pairs = [score_pair(t, s, r, old_plan["tasks"][t], panels, pub_pairs[t, s])
                 for t, r in rules.items() for s in r["reset_seeds"]]
        require(len(pub_pairs) == len(pairs) == 6 and sum(p["closed"]["passed"] for p in pairs) == public["pairs_passed"],
                "archived pair inventory differs")
        summaries = {t: summarize([p for p in pairs if p["task"] == t]) for t in rules}
        comparisons = {k: sum(s["effect_error_comparison"][k] for s in summaries.values())
                       for k in ("better", "same", "worse", "unavailable")}
        models.append(dict(seed=seed, pairs=pairs, tasks=summaries, effect_error_comparison=comparisons,
                           fixed_passed=all(p["fixed"]["passed"] for p in pairs),
                           closed_passed=all(p["closed"]["passed"] for p in pairs),
                           weights_sha256=source["model"]["weights_sha256"], closed_predictions_sha256=source["predictions_sha256"]))
    tasks = {t: task_summary([[p for p in m["pairs"] if p["task"] == t] for m in models]) for t in rules}
    statuses = {f: dict(Counter(a["status"] for a in curves if a["family"] == f)) for f in FAMILIES}
    counts = {f + "_arms": sum(statuses[f].values()) for f in FAMILIES}
    counts.update(curves=len(curves), physical_pairs=len(physical)//2, model_task_pairs=sum(len(m["pairs"]) for m in models),
                  statuses=statuses)
    for family in ("closed", "fixed"):
        for key in ("attempted_steps", "accepted_steps"):
            counts[family + "_" + key] = sum(a[key] for a in curves if a["family"] == family)
    require(counts["fixed_attempted_steps"] == replay_seal["attempted_steps"] <= plan["budget"]["max_model_calls"]
            and counts["fixed_accepted_steps"] == replay_seal["accepted_steps"], "replay counter differs")
    parity = [arm for m in models for p in m["pairs"] for arm in p["first_step_parity"].values()]
    public_curves = out / "curves.jsonl.gz"
    with public_curves.open("xb") as raw, gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as gz:
        for row in curves:
            gz.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    report = dict(protocol_sha256=PLAN_SHA, decision="INCOMPLETE_CONDITIONAL" if any(
        t["decision"] == "INCOMPLETE_CONDITIONAL" for t in tasks.values()) else "COMPLETE_DIAGNOSTIC",
        counts=counts, tasks=tasks, models=models, fixed_family_passed=all(m["fixed_passed"] for m in models),
        closed_family_passed=all(m["closed_passed"] for m in models), resources=resources or {},
        **counters,
        first_step_parity=dict(arms=len(parity), first_applied_equal=sum(a["first_applied_equal"] for a in parity),
                               first_endpoint_equal=sum(a["first_endpoint_equal"] for a in parity)),
        sources=dict(closed_report_sha256=plan["source_sha256"][CLOSED_REPORT], closed_protocol_sha256=archive["protocol_sha256"],
                     closed_seal_sha256=closed_sha, input_seal_sha256=input_sha, replay_seal_sha256=replay_sha,
                     physical_reports_sha256=old_plan["truth_sha256"]),
        curves_sha256=digest(public_curves))
    write(out / "report.json", report)
    return report
