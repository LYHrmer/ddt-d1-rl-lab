"""Evaluate the two learned curricula from the original approach and audit their native traces."""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from statistics import mean

import numpy as np

from tools87.readback87 import LEGS, WHEELS, check, compare
from tools87.run87 import PLAN as PLAN87, read, sha, snapshot, write
from tools87.scoring87 import score
from tools89.common89 import PLAN, check_worker, registered_plan


def matrix(plan):
    training, evaluation = plan["training"], plan["evaluation"]
    seeds, resets = training["seeds"], evaluation["reset_seeds"]
    arms = training["arms"]
    check(len(seeds) == len(set(seeds)) == len(resets) == len(set(resets)) == 3
          and arms == ["original", "mixed"] and set(evaluation["terrains"]) == {"up", "flat", "down"},
          "registered training arms, seeds or terrain matrix")
    return [(arm, seed, terrain, reset) for seed, matched in zip(seeds, resets, strict=True)
            for arm in arms for terrain in evaluation["terrains"]
            for reset in (resets if terrain == "up" else [matched])]


def checked_models(training_dir, plan):
    header = read(training_dir / "report.json")
    protocol_sha = sha(PLAN)
    check(header["status"] == "COMPLETE" and header["protocol_sha256"] == protocol_sha, "training stage")
    parents = dict(zip(plan["training"]["seeds"], plan["training"]["parent_seeds"], strict=True))
    paths = {}
    for model in header["models"]:
        arm, seed = model["arm"], model["train_seed"]
        check(model["model"] == f"{arm}_{seed}", "model directory")
        folder = training_dir / model["model"]
        report, weights = folder / "report.json", folder / "policy.zip"
        item = read(report)
        check(item["status"] == "COMPLETE" and (item["arm"], item["training_seed"]) == (arm, seed)
              and item["protocol_sha256"] == protocol_sha and item["parent_seed"] == model["parent_seed"] == parents[seed]
              and not weights.is_symlink() and sha(weights) == item["policy_sha256"] == model["policy_sha256"]
              and sha(report) == model["report_sha256"] and (arm, seed) not in paths, f"{folder} identity")
        paths[arm, seed] = (weights, model["policy_sha256"])
    check(set(paths) == {(arm, seed) for arm, seed, _, _ in matrix(plan)}, "incomplete model matrix")
    return paths


def load_actor(weights):
    from tools87.ppo87 import load_weights, predictor, single_cpu
    single_cpu()
    return predictor(load_weights(weights), "range")


def score_window(trace, interval):
    start, end = interval
    vx = np.asarray(trace["body_vx_m_s"], dtype=float)
    if len(vx) <= end:
        return None
    selected = vx[start + 1:end + 1]
    check(selected.shape == (end - start,) and np.isfinite(selected).all(), "invalid score samples")
    return float(selected.mean())


def episode(output, key, predict, identity, course, evaluation):
    from tools87.terrain87 import make_env
    from tools89.course89 import loaded_wheels_past_toe, wheel_geom_slots

    arm, seed, terrain, reset = key
    name = f"{arm}_{seed}_{terrain}_{reset}"
    folder = output / name
    folder.mkdir()
    trace, env = {}, None
    result = {"case": name, "status": "INCOMPLETE", "arm": arm, "training_seed": seed, "seed": reset,
              "terrain": terrain, "protocol_sha256": sha(PLAN), "policy_sha256": identity}
    try:
        env = make_env(terrain, reset, perception=True, max_transitions=course["ticks"])
        slots = wheel_geom_slots(env)
        obs, reset_info = env.reset(seed=reset)
        check(reset_info["delay_ticks"] == course["action_delay_ticks"] == 1
              and obs.shape == (643,) and env.data.qpos[env.free_qpos] == 0., name + " original reset")
        zero = np.zeros(16, dtype=np.float32)
        snapshot(env, trace, 0., 0., zero, zero, {}, 0.)
        trace["wheel_on_slope"] = [np.zeros(4, dtype=bool)]
        for _ in range(course["ticks"]):
            packet = env.last_raw_features.copy()
            check(np.array_equal(obs[627:], packet), name + " causal range input")
            real = predict(obs[:627], packet)
            masked = predict(obs[:627], zero)
            check(real.shape == (16,) and np.isfinite(real).all() and np.all(np.abs(real) <= 1),
                  name + " invalid actor output")
            obs, _, terminated, truncated, info = env.step(real)
            snapshot(env, trace, info["body_vx_m_s"], info["command_during_interval"][0], real,
                     packet, info, float(np.max(np.abs(real - masked))))
            trace["wheel_on_slope"].append(loaded_wheels_past_toe(env, slots))
            if terminated or truncated:
                break
        scored = score(trace, course, evaluation["course_mark_m_past_toe"])
        counts = env.counters()
        ticks = scored["ticks"]
        check((counts["policy_transitions"], counts["native_steps"], counts["pd_updates"])
              == (ticks, 10 * ticks, 5 * ticks) and (scored["full_horizon"] or scored["terminated"]),
              name + " physics accounting or termination")
        result.update(scored, score_vx_m_s=score_window(trace, evaluation["speed_interval_end_ticks"]),
                      four_wheel_slope_ticks=int(np.all(trace["wheel_on_slope"][1:], axis=1).sum()),
                      rear_wheel_slope_ticks=int(np.any(np.asarray(trace["wheel_on_slope"])[1:, 2:], axis=1).sum()),
                      pd_updates=counts["pd_updates"], native_steps=counts["native_steps"])
        result["status"] = "COMPLETE"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            result["attempted_counts"] = env.counters()
        raise
    finally:
        try:
            if trace:
                path = folder / ("trace.npz" if result["status"] == "COMPLETE" else "partial_trace.npz")
                with path.open("xb") as stream:
                    np.savez_compressed(stream, **trace)
                result["trace_sha256"] = sha(path)
            write(folder / "metrics.json", result)
        finally:
            if env is not None:
                env.close()
    print(name, result["score_vx_m_s"], result["reached_mark"], flush=True)
    return result


def summarize(cases, plan):
    evaluation = plan["evaluation"]
    by_key = {(c["arm"], c["training_seed"], c["terrain"], c["seed"]): c for c in cases}
    check(set(by_key) == set(matrix(plan)) and len(by_key) == len(cases) == evaluation["max_episodes"],
          "missing or duplicate case in registered evaluation matrix")
    resets = evaluation["reset_seeds"]
    uphill = []
    for seed in plan["training"]["seeds"]:
        selected = {arm: [by_key[arm, seed, "up", reset] for reset in resets] for arm in ("original", "mixed")}
        means = {}
        for arm, rows in selected.items():
            speeds = [r["score_vx_m_s"] for r in rows]
            means[arm] = mean(speeds) if all(v is not None for v in speeds) else None
        delta = None if None in means.values() else means["mixed"] - means["original"]
        clean = all(c["course_completed"] for rows in selected.values() for c in rows)
        uphill.append({"training_seed": seed, "original_vx_m_s": means["original"], "mixed_vx_m_s": means["mixed"],
                       "paired_delta_vx_m_s": delta, "all_six_clean": clean,
                       "passes": clean and delta is not None and delta >= evaluation["min_paired_uphill_delta_m_s"]})
    retention = {}
    for terrain in ("flat", "down"):
        pairs = []
        for seed, reset in zip(plan["training"]["seeds"], resets, strict=True):
            orig, mixed = (by_key[a, seed, terrain, reset] for a in ("original", "mixed"))
            a, b = orig["fixed_speed_rmse_m_s"], mixed["fixed_speed_rmse_m_s"]
            pairs.append({"training_seed": seed, "seed": reset, "original_rmse_m_s": a, "mixed_rmse_m_s": b,
                          "delta_rmse_m_s": b - a, "both_clean": orig["course_completed"] and mixed["course_completed"]})
        delta = mean(p["delta_rmse_m_s"] for p in pairs)
        retention[terrain] = {"paired": pairs, "mean_delta_rmse_m_s": delta,
                              "passes": all(p["both_clean"] for p in pairs)
                              and delta <= evaluation["max_mean_retention_rmse_increase_m_s"]}
    marks = {arm: sum(c["course_completed"] and c["reached_mark"] for c in cases if c["arm"] == arm
                      and c["terrain"] == "up") for arm in plan["training"]["arms"]}
    return {"uphill_paired_train_seed_means": uphill, "uphill_speed_pass": all(p["passes"] for p in uphill),
            "uphill_clean_marks": marks, "uphill_stable_pass": marks["mixed"] == len(uphill) * len(resets),
            "pace_retention": retention, "retention_pass": all(x["passes"] for x in retention.values()),
            "total_falls": sum(c["fallen"] for c in cases), "total_illegal_contacts": sum(c["illegal_contact"] for c in cases),
            "total_clean_full_horizon": sum(c["course_completed"] for c in cases)}


def evaluate(training_dir, output):
    plan, course = registered_plan(), read(PLAN87)["course"]
    output.mkdir(exist_ok=False)
    receipt = {"status": "INCOMPLETE", "stage": "evaluate", "cases": [], "protocol_sha256": sha(PLAN),
               "source_sha256": sha(__file__)}
    try:
        models = checked_models(training_dir, plan)
        receipt["training_report_sha256"] = sha(training_dir / "report.json")
        current_key, actor = None, None
        for key in matrix(plan):
            model_key = key[:2]
            weights, identity = models[model_key]
            if model_key != current_key:
                actor, current_key = load_actor(weights), model_key
            receipt["cases"].append(episode(output, key, actor, identity, course, plan["evaluation"]))
        rows = receipt["cases"]
        counts = {"policy_transitions": sum(r["ticks"] for r in rows),
                  "native_steps": sum(r["native_steps"] for r in rows), "pd_updates": sum(r["pd_updates"] for r in rows)}
        check(counts["policy_transitions"] <= plan["evaluation"]["max_transitions"], "evaluation budget")
        receipt.update(status="COMPLETE", counts=counts, verdict=summarize(rows, plan))
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(output / "report.json", receipt)


def case_readback(root, row, predict, identity, course, evaluation):
    from tools88.policy88 import observations_and_rewards, recast_packets

    name = f"{row['arm']}_{row['training_seed']}_{row['terrain']}_{row['seed']}"
    folder = root / name
    source = folder / "trace.npz"
    check(row == read(folder / "metrics.json") and row["case"] == name and row["status"] == "COMPLETE"
          and row["policy_sha256"] == identity and row["protocol_sha256"] == sha(PLAN)
          and row["trace_sha256"] == sha(source), name + " receipt/trace identity")
    with np.load(source, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    n = len(trace["time_s"]) - 1
    check(0 < n <= course["ticks"] and all(len(v) == n + 1 for v in trace.values()) and row["ticks"] == n
          and row["pd_updates"] == 5*n and row["native_steps"] == 10*n, name + " full track accounting")
    t = trace["time_s"]
    check(np.allclose(t, np.arange(n + 1)/course["policy_hz"], atol=1e-8, rtol=0)
          and trace["qpos"].shape == (n+1, 23) and trace["qvel"].shape == (n+1, 22)
          and trace["qpos"][0, 0] == 0 and np.array_equal(trace["body_x_m"], trace["qpos"][:, 0]), name + " poses")
    issued, applied = trace["issued_action"], trace["applied_action"]
    check(np.array_equal(issued[:-1], applied[1:]) and np.allclose(applied[1:, WHEELS] * 5.0,
          trace["wheel_target_rad_s"][1:], rtol=0, atol=1e-6) and np.allclose(np.tile([0, .8, -1.5], 4)
          + .5 * applied[1:, LEGS], trace["leg_target_rad"][1:], rtol=0, atol=1e-7), name + " FIFO and joint targets")
    actor_inputs, _, reward_checks = observations_and_rewards(trace, row, course["policy_hz"])
    packets, _ = recast_packets(trace, row, row["terrain"])
    check(np.array_equal(packets, trace["raw_ranges16"][1:]), name + " replayed sensor inputs")
    actions = np.stack([predict(body, packet) for body, packet in zip(actor_inputs, packets, strict=True)])
    check(np.array_equal(actions, issued[1:]), name + " issued actions differ from model on reconstructed inputs")
    start, end = evaluation["speed_interval_end_ticks"]
    recorded = trace["body_vx_m_s"][1:]
    check(np.isfinite(recorded).all(), name + " invalid body speed")
    wheel_mask = trace["wheel_on_slope"][1:]
    check(wheel_mask.shape == (n, 4) and wheel_mask.dtype == np.bool_
          and (row["terrain"] == "up" or not np.any(wheel_mask)), name + " slope support")
    cmd = course["command_forward_m_s"]
    warm = course["stand_ticks"]
    check(np.allclose(trace["command_vx_m_s"][1:], np.where(np.arange(n) >= warm, cmd, 0.),
                      rtol=0, atol=1e-7), name + " requested command")
    errors = np.full(course["ticks"] - warm, cmd)
    moving = recorded[warm:]
    errors[:len(moving)] = moving - cmd
    loaded = trace["front_contact_x_m"][1:]
    check(not np.isinf(loaded).any(), name + " invalid front contact")
    loaded = loaded[np.isfinite(loaded)]
    progress = float(loaded.max() - course["toe_x_m"]) if len(loaded) else None
    fell, illegal = bool(trace["fallen"].any()), bool(trace["illegal_body_contact"].any())
    measured = {"score_vx_m_s": float(np.mean(recorded[start:end])) if n >= end else None,
                "fixed_speed_rmse_m_s": float(np.sqrt(np.mean(errors ** 2))), "fallen": fell,
                "illegal_contact": illegal, "course_completed": n == course["ticks"] and not (fell or illegal),
                "max_loaded_front_contact_past_toe_m": progress,
                "reached_mark": progress is not None and progress >= evaluation["course_mark_m_past_toe"],
                "four_wheel_slope_ticks": int(np.all(wheel_mask, axis=1).sum()),
                "rear_wheel_slope_ticks": int(np.any(wheel_mask[:, 2:], axis=1).sum()),
                "wheel_saturated_ticks": int((trace["wheel_saturation_count"][1:] > 0).sum())}
    for k, value in measured.items():
        compare(row[k], value, name + "/" + k)
    return {k: row[k] for k in ("case", "arm", "training_seed", "terrain", "seed", "ticks",
                                 "policy_sha256", "trace_sha256")} | measured | {
        "max_replayed_vx_error_m_s": reward_checks["vx_error"], "actor_components_replayed": actions.size,
        "range_components_recast": packets.size, "wheel_target_mean_4_to_6_rad_s": (
            float(trace["wheel_target_rad_s"][start+1:end+1].mean()) if n >= end else None)}


def readback(training_dir, root):
    plan, course = registered_plan(), read(PLAN87)["course"]
    header = read(root / "report.json")
    check(header["status"] == "COMPLETE" and header["protocol_sha256"] == sha(PLAN)
          and header["source_sha256"] == sha(__file__)
          and header["training_report_sha256"] == sha(training_dir / "report.json"), "evaluation stage receipt")
    models = checked_models(training_dir, plan)
    cases = []
    current_key, actor = None, None
    for row in header["cases"]:
        key = row["arm"], row["training_seed"]
        weights, identity = models[key]
        if key != current_key:
            actor, current_key = load_actor(weights), key
        cases.append(case_readback(root, row, actor, identity, course, plan["evaluation"]))
    verdict = summarize(cases, plan)
    # Independent reductions of fixed-horizon SSE can differ by one machine ulp.
    for k in ("uphill_speed_pass", "uphill_stable_pass", "retention_pass", "uphill_clean_marks",
              "total_falls", "total_illegal_contacts", "total_clean_full_horizon"):
        check(verdict[k] == header["verdict"][k], "recomputed evaluation " + k)
    count = sum(c["ticks"] for c in cases)
    check(header["counts"] == {"policy_transitions": count, "native_steps": count*10, "pd_updates": count*5},
          "recomputed evaluation step totals")
    return {"status": "COMPLETE", "protocol_sha256": sha(PLAN), "evaluation_report_sha256": sha(root / "report.json"),
            "counts": header["counts"], "new_native_steps_in_readback": 0, "cases": cases, "verdict": verdict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "readback"))
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    check_worker("evaluate" if args.mode == "run" else "readback")
    if args.mode == "run":
        evaluate(args.training, args.root)
    elif args.output is None:
        parser.error("readback requires a fresh --output path")
    else:
        with args.output.open("x") as stream:
            try:
                result = readback(args.training, args.root)
            except BaseException as exc:
                json.dump({"status": "INCOMPLETE", "error": str(exc)}, stream)
                raise
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
