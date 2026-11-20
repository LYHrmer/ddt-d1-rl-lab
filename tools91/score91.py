"""Read sealed predictions and independently reconstruct the registered physical scores."""
from collections import Counter
import gzip
import json

import numpy as np

from tools91.launch91 import BASE, ROOT, PLAN, PLAN_SHA, digest, ensure_guard, read, require


def curves(qpos, qvel):
    """Project world translational velocity onto the quaternion's body x axis."""
    q = qpos[:, 3:7]
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    axes = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y)], axis=1)
    return qpos[:, 0].copy(), np.sum(axes*qvel[:, :3], axis=1)


def scalar(task, ticks, xs, vxs):
    if task == "stand":
        require(np.array_equal(ticks, np.arange(51)), "wrong rollback window")
        return max(0., float(xs[0] - xs[50]))
    require(task == "move" and np.array_equal(ticks, np.arange(180, 301)), "wrong velocity window")
    return float(np.mean(vxs[ticks > 200], dtype=np.float64))


def grade(rule, original, intervention, true_original, true_intervention):
    sign = 1 if rule["effect"] == "intervention minus original" else -1
    real = sign * (true_intervention - true_original)
    effect = None if original is None or intervention is None else sign * (intervention - original)
    error = None if effect is None else abs(effect - real)
    a_error = None if original is None else abs(original - true_original)
    b_error = None if intervention is None else abs(intervention - true_intervention)
    checks = {"both_complete": effect is not None, "positive": effect is not None and effect > 0,
              "effect_tolerance": error is not None and error <= rule["effect_tolerance"],
              "beats_zero_effect": error is not None and error < abs(real),
              "both_absolute_arms": all(e is not None and e <= rule["per_arm_tolerance"] for e in (a_error, b_error))}
    return dict(truth=[true_original, true_intervention], prediction=[original, intervention], true_effect=real,
                predicted_effect=effect, effect_abs_error=error, zero_effect_abs_error=abs(real),
                arm_abs_errors=[a_error, b_error], checks=checks, passed=all(checks.values()))


def close(a, b):
    return a is b if a is None or b is None else abs(a - b) < 1e-12


def physical(plan):
    for path, expected in plan["truth_sha256"].items():
        require(digest(ROOT / path) == expected, "physical report changed")
    published = {"move": read(ROOT / "reports/uphill_diagnosis88.json"),
                 "stand": read(ROOT / "reports/stand_wbc90.json")["stand"]}
    roots = {"move": ROOT / "evidence/uphill88/probe_01", "stand": ROOT / "evidence/wbc90/stand_01"}
    truth = {}
    for task, record in published.items():
        for case in record["cases"]:
            seed, arm = case.get("reset_seed", case.get("seed")), case["arm"]
            path = roots[task] / f"{arm}_{seed}/trace.npz"
            require(digest(path) == case["trace_sha256"], "physical trace hash differs")
            with np.load(path, allow_pickle=False) as trace:
                if task == "move":
                    require(case["full_clean"] and case["ticks"] == 300, "partial physical 88")
                    q, v = trace["qpos"][180:301], trace["qvel"][180:301]
                    ticks, logged = np.arange(180, 301), case["score_vx_m_s"]
                    first_raw, first_applied = trace["raw_cpp_action"][181], trace["applied_action"][181]
                else:
                    require(case["clean"] and case["ticks"] == 100 and case["status"] == "COMPLETE", "partial physical 90")
                    q = np.vstack([trace["initial_qpos"], trace["qpos"][:50]])
                    v = np.vstack([trace["initial_qvel"], trace["qvel"][:50]])
                    ticks, logged = np.arange(51), case["rollback_m"]
                    first_raw, first_applied = trace["policy_raw"][0], trace["applied"][0]
                xs, vxs = curves(q, v)
                measured = scalar(task, ticks, xs, vxs)
                require(close(logged, measured), "public physical score differs")
            truth[task, seed, arm] = dict(ticks=ticks, xs=xs, vxs=vxs, scalar=measured,
                                          start_state=np.r_[q[0], v[0]], first_raw=first_raw, first_applied=first_applied)
    return truth


def trajectory_row(family, seed, task, reset, arm, status, ticks, xs, vxs):
    return dict(family=family, model_seed=seed, task=task, reset_seed=reset, arm=arm, status=status,
                ticks=ticks.tolist(), world_x_m=xs.tolist(), body_vx_m_s=vxs.tolist())


def read_family(stage, seal, truth, plan, series):
    groups = []
    for model in seal["models"]:
        seed = model["seed"]
        folder = BASE / f"{stage}_01/seed{seed}"
        suite = read(folder / "predictions.json")
        require(suite["status"] == "COMPLETE" and len(suite["cases"]) == 12 and len(suite["pairs"]) == 6,
                "incomplete panel data collection")
        cases = {}
        for row in suite["cases"]:
            task, reset, arm = row["task"], row["reset_seed"], row["arm"]
            k = (task, reset, arm)
            require(k in truth and k not in cases, "unknown or repeated rollout")
            observed = truth[k]
            path = folder / row["trace"]
            require(digest(path) == row["trace_sha256"], "imagined trajectory changed")
            with np.load(path, allow_pickle=False) as trace:
                ticks, state = trace["endpoint_tick"], trace["state45"]
                require(np.array_equal(ticks, observed["ticks"][:len(ticks)])
                        and len(ticks) == len(state) == row["steps"] + 1
                        and np.array_equal(state[0], observed["start_state"]), "state boundary differs")
                applied, issued, raw = trace["applied16"], trace["issued16"], trace["raw16"]
                require(len(applied) == len(raw) == len(issued) == row["attempted_steps"] and
                        np.array_equal(applied[0], observed["first_applied"]) and
                        np.array_equal(applied[1:], issued[:-1]), "imagined FIFO chronology differs")
                raw_error = float(np.max(np.abs(raw[0] - observed["first_raw"])))
                require(raw_error <= 1e-5, "initial policy action differs from archived controller")
                xs, vxs = curves(state[:, :23], state[:, 23:])
                value = scalar(task, ticks, xs, vxs) if row["status"] == "COMPLETE" else None
                require(close(row["score"], value), "sealed imagined metric differs")
                series.append(trajectory_row(stage, seed, task, reset, arm, row["status"], ticks, xs, vxs))
                cases[k] = dict(score=value, status=row["status"], accepted_steps=row["steps"],
                                attempted_steps=row["attempted_steps"], raw_action_error=raw_error,
                                trace_sha256=row["trace_sha256"], error=row["error"])
        require(cases.keys() == truth.keys(), "incomplete case inventory")
        pairs = []
        for pair in suite["pairs"]:
            task, reset = pair["task"], pair["reset_seed"]
            rule = plan["tasks"][task]
            keys = [(task, reset, arm) for arm in ("original", rule["intervention"])]
            a, b = (cases[k] for k in keys)
            result = grade(rule, a["score"], b["score"], *(truth[k]["scalar"] for k in keys))
            require(close(result["predicted_effect"], pair["predicted_effect"]), "pair sign or score differs")
            pairs.append(dict(task=task, reset_seed=reset, parent_seed=pair["parent_seed"], arms=[a, b], **result))
        expected = {(task, r) for task, rule in plan["tasks"].items() for r in rule["reset_seeds"]}
        require({(p["task"], p["reset_seed"]) for p in pairs} == expected, "incomplete pair inventory")
        groups.append(dict(seed=seed, source=model, pairs=pairs, pairs_passed=sum(p["passed"] for p in pairs),
                           passed=all(p["passed"] for p in pairs), cases_by_status=dict(Counter(c["status"] for c in cases.values())),
                           imagined_steps=sum(c["attempted_steps"] for c in cases.values())))
    return dict(models=groups, models_passed=sum(g["passed"] for g in groups), passed=all(g["passed"] for g in groups))


def score_experiment():
    plan = read(PLAN)
    stages = ("prepare", "archived", "fit", "candidate")
    guards = {}
    for stage in stages:
        guard = ensure_guard(stage)
        resource_dir = BASE / f"guard_{stage}_01"
        samples_file = resource_dir / "resources.jsonl"
        samples = [json.loads(line) for line in samples_file.read_text().splitlines()]
        guards[stage] = {k: guard[k] for k in ("elapsed_s", "cpu_affinity", "minimum_effective_bytes", "peak_group_bytes", "final_memory_events")}
        admission = [s for s in samples if s["stage"] == "startup"][-1]["effective_available_bytes"]
        runtime_min = min(s["effective_available_bytes"] for s in samples if s["stage"] == "runtime")
        require(admission >= 4 << 30 and runtime_min >= 2 << 30, "memory reserve violated")
        guards[stage].update(guard_sha256=digest(resource_dir / "report.json"), samples=len(samples),
                             admitted_effective_bytes=admission, minimum_running_effective_bytes=runtime_min)
    seals = {}
    for stage in ("archived", "candidate"):
        folder = BASE / f"{stage}_01"
        seal = read(folder / "seal.json")
        expected = plan["old_models" if stage == "archived" else "fit"]["seeds"]
        require(seal["protocol_sha256"] == PLAN_SHA and [m["seed"] for m in seal["models"]] == expected and
                all(digest(folder / f"seed{m['seed']}/predictions.json") == m["predictions_sha256"] for m in seal["models"]),
                "missing or broken prediction seal")
        seals[stage] = seal
    truth = physical(plan)
    series = [trajectory_row("physics", None, *k, "COMPLETE", v["ticks"], v["xs"], v["vxs"]) for k, v in truth.items()]
    families = {stage: read_family(stage, seal, truth, plan, series) for stage, seal in seals.items()}
    source = BASE / "prepare_01/report.json"
    dataset = read(source)
    out = BASE / "score_01/curves.jsonl.gz"
    with out.open("xb") as file, gzip.GzipFile(filename="", fileobj=file, mode="wb", mtime=0) as archive:
        for row in series:
            archive.write((json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode())
    steps = sum(m["imagined_steps"] for f in families.values() for m in f["models"])
    updates = sum(m["source"]["model"]["updates"] for m in families["candidate"]["models"])
    require(0 < steps <= plan["budget"]["max_imagined_steps_total"] and updates == 900, "execution budget mismatch")
    return dict(protocol_sha256=PLAN_SHA, registration_commit="7fb8ac4", design="retrospective", resources=guards,
                starts=dataset["starts"], dataset=dataset["dataset"], families=families, truth_sha256=plan["truth_sha256"],
                prediction_seals={s: digest(BASE / f"{s}_01/seal.json") for s in seals},
                curves_sha256=digest(out), curve_count=len(series), physical_pairs=6, policy_updates=0,
                new_physics_steps=0, model_updates=updates, imagined_steps=steps)
