"""Independently read the D1 smoke trajectories, PPO rollouts and selected video."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "docs/experiments87/full_body_perception.json"
WHEELS = [3, 7, 11, 15]
LEGS = [i for i in range(16) if i not in WHEELS]
PAIR_FIELDS = ("raw_range", "issued", "applied", "qpos", "reward", "terrain", "terminated", "truncated")


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, label):
    if not ok:
        raise ValueError(label)


def compare(expected, actual, label):
    if isinstance(actual, float):
        check(isinstance(expected, (int, float)) and math.isclose(expected, actual, rel_tol=1e-9,
              abs_tol=2e-12), f"{label}: saved {expected}, recomputed {actual}")
    else:
        check(expected == actual, f"{label}: saved {expected}, recomputed {actual}")


def case_readback(folder, receipt, course, mark):
    metadata = load(folder / "metrics.json")
    check(metadata == receipt and receipt["status"] == "COMPLETE"
          and receipt["protocol_sha256"] == digest(PROTOCOL), f"{folder.name}: metadata")
    trace = folder / "trace.npz"
    check(digest(trace) == metadata["trace_sha256"], f"{folder.name}: trace digest")
    with np.load(trace, allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    hz, horizon, warm = course["policy_hz"], course["ticks"], course["stand_ticks"]
    toe, target = course["toe_x_m"], course["command_forward_m_s"]
    n = len(a["time_s"]) - 1
    check(0 < n <= horizon and all(len(x) == n + 1 for x in a.values()), f"{folder.name}: clock")
    check(np.allclose(a["time_s"], np.arange(n + 1)/hz, atol=1e-8, rtol=0),
          f"{folder.name}: native boundary times")
    moving = np.arange(n) >= warm
    check(np.allclose(a["command_vx_m_s"][1:], target*moving, atol=1e-7, rtol=0),
          f"{folder.name}: requested velocity")
    check(np.array_equal(a["body_x_m"], a["qpos"][:, 0]) and a["qpos"].shape == (n + 1, 23)
          and a["qvel"].shape == (n + 1, 22) and np.isfinite(a["qpos"]).all()
          and np.isfinite(a["qvel"]).all(), f"{folder.name}: native pose")
    check(np.array_equal(a["issued_action"][:-1], a["applied_action"][1:]) and
          np.all(np.abs(a["issued_action"]) <= 1), f"{folder.name}: applied 20 ms FIFO")
    applied = a["applied_action"][1:]
    legs = np.tile([0., 0.8, -1.5], 4) + .5 * applied[:, LEGS]
    check(np.allclose(legs, a["leg_target_rad"][1:], rtol=0, atol=1e-7)
          and np.allclose(5. * applied[:, WHEELS], a["wheel_target_rad_s"][1:], rtol=0, atol=1e-6),
          f"{folder.name}: twelve leg and four wheel targets")
    check(np.all(np.abs(a["last_torque_nm"]) <= np.tile([80, 80, 80, 12], 4) + 1e-9),
          f"{folder.name}: motor torque limits")
    velocity = a["body_vx_m_s"][1:][moving]
    errors = np.full(horizon - warm, target, dtype=float)
    errors[:len(velocity)] = velocity - target
    xcontact = a["front_contact_x_m"][1:]
    loaded = xcontact[np.isfinite(xcontact)]
    past = float(loaded.max() - toe) if loaded.size else None
    flags = a["raw_ranges16"][1:, 8:]
    check(np.isin(flags, (0, 1)).all() and
          np.all(a["raw_ranges16"][1:, :8][flags == 0] == 0.), f"{folder.name}: invalid rays")
    fallen = bool(np.any(a["fallen"]))
    illegal = bool(np.any(a["illegal_body_contact"]))
    result = {
        "ticks": n, "fallen": fallen, "illegal_contact": illegal, "full_horizon": n == horizon,
        "course_completed": n == horizon and not (fallen or illegal),
        "reached_mark": past is not None and past >= mark, "max_loaded_front_contact_past_toe_m": past,
        "fixed_speed_rmse_m_s": math.sqrt(float(np.mean(errors**2))),
        "observed_speed_rmse_m_s": math.sqrt(float(np.mean((velocity-target)**2))) if velocity.size else None,
        "mean_forward_speed_while_commanded_m_s": float(velocity.mean()) if velocity.size else None,
        "motion_samples": len(velocity), "missing_motion_samples": len(errors) - len(velocity),
        "peak_roll_or_pitch_rad": float(max(np.abs(a["pitch_rad"]).max(), np.abs(a["roll_rad"]).max())),
        "wheel_saturated_ticks": int((a["wheel_saturation_count"] > 0).sum()),
        "ticks_with_valid_rays": int(flags.any(axis=1).sum()), "valid_ray_fraction": float(flags.mean()),
        "max_action_change_with_terrain_masked": float(a["mask_action_max_abs"].max()),
        "end_body_x_m": float(a["qpos"][-1, 0]), "pd_updates": 5*n, "native_steps": 10*n,
    }
    for key, value in result.items():
        compare(metadata[key], value, f"{folder.name}/{key}")
    if metadata["arm"] == "baseline":
        cpp = load(folder / "cpp34/report.json")
        check(cpp["status"] == "CLOSED" and cpp["cleanup_complete"]
              and cpp["requests_completed"] == cpp["requests_attempted"] == n,
              f"{folder.name}: native C++ action count")
    result.update({k: metadata[k] for k in ("case", "seed", "arm", "training_seed", "terrain", "trace_sha256")})
    result["body_displacement_x_m"] = float(a["qpos"][-1, 0] - a["qpos"][0, 0])
    return result


def paired_training(root, report, plan):
    parameters = plan["training"]
    seeds = parameters["seeds"]
    declared = {m["model"]: m for m in report["models"]}
    check(len(report["models"]) == len(declared) == len(seeds)*2, "training matrix")
    rows = []
    for seed in seeds:
        traces = {}
        for arm in parameters["arms"]:
            name = f"{arm}_{seed}"
            folder = root / name
            saved = load(folder / "report.json")
            check(saved["seed"] == seed and saved["arm"] == arm and saved["status"] == "COMPLETE"
                  and digest(folder / "policy.zip") == saved["policy_sha256"]
                  and digest(folder / "report.json") == declared[name]["report_sha256"], f"{name}: identity")
            check(saved["warm_start"]["parent_sha256"] == plan["parent"]["checkpoint_sha256"]
                  and saved["initial_probes"]["max_abs_error_cpp34"] <= parameters["valid_initial_action_max_abs_error"]
                  and saved["initial_probes"]["initial_masked_range_bitwise_equal"], f"{name}: initialization")
            with np.load(folder / "train_trace.npz", allow_pickle=False) as archive:
                t = {k: archive[k] for k in archive.files}
            traces[arm] = t
            n = len(t["reward"])
            check(n == parameters["rollout_transitions_per_model"] and all(len(v) == n for v in t.values())
                  and t["qpos"].shape == (n, 23) and np.all(np.isfinite(t["qpos"]))
                  and np.any(np.diff(t["qpos"][:, 0]) != 0) and np.isfinite(t["reward"]).all(),
                  f"{name}: real rollouts")
            check(all(hashlib.sha256(v.tobytes()).hexdigest() == saved["trace"]["array_sha256"][k]
                      for k, v in t.items()), f"{name}: array digests")
            counts = saved["counts"]
            check((n, n*5, n*10) == (counts["policy_transitions"], counts["pd_updates"], counts["native_steps"]),
                  f"{name}: native accounting")
            resets = np.r_[True, (t["terminated"] | t["truncated"])[:-1]]
            expected = np.roll(t["issued"], 1, axis=0)
            expected[resets] = 0
            check(np.array_equal(expected, t["applied"]), f"{name}: interval actions")
            valid = int(np.any(t["raw_range"][:, 8:], axis=1).sum())
            check(valid == saved["trace"]["valid_raw_range_ticks"] and valid > 0, f"{name}: sensor")
            if arm == "range":
                check(np.array_equal(t["raw_range"], t["visible_range"]), f"{name}: delivered rays")
            else:
                check(not np.any(t["visible_range"]), f"{name}: masked rays")
            check(int(t["terminated"].sum()) == saved["trace"]["terminated_episodes"], f"{name}: terminations")
            delta = np.asarray(saved["action_head_row_delta_l2"])
            projection_grad = saved["postclip_max_gradient_norm"]["projection"]
            check(delta.shape == (16,) and np.isfinite(delta).all() and (delta >= 0).all()
                  and math.isfinite(projection_grad) and (projection_grad > 0) == (arm == "range")
                  and 0 < saved["optimizer_steps"] <= parameters["max_epochs"]*n//parameters["minibatch"],
                  f"{name}: PPO parameter update")
            with np.load(folder / "probes.npz", allow_pickle=False) as z:
                action_change = float(np.max(np.abs(z["actions"] - z["parent"])))
                terrain_change = float(np.max(np.abs(z["actions"] - z["masked"])))
                check(np.array_equal(z["actions"], z["reloaded"]) and action_change > 0
                      and (terrain_change > 0) == (arm == "range"), f"{name}: reload and terrain response")
            compare(saved["max_action_change_from_parent"], action_change, name)
            compare(saved["max_measured_vs_masked_action"], terrain_change, name)
            leg_count, wheel_count = int((delta[LEGS] > 0).sum()), int((delta[WHEELS] > 0).sum())
            check((leg_count, wheel_count) == (saved["leg_rows_changed"], saved["wheel_rows_changed"]),
                  f"{name}: changed outputs")
            rows.append({"training_seed": seed, "arm": arm, "policy_steps": n, "pd_updates": n*5,
                         "native_steps": n*10, "optimizer_steps": saved["optimizer_steps"],
                         "leg_outputs_changed": leg_count, "wheel_outputs_changed": wheel_count,
                         "range_projection_gradient_norm": projection_grad, "valid_sensor_ticks": valid,
                         "max_action_change_from_parent": action_change, "max_sensor_response": terrain_change,
                         "terminated_episodes": int(t["terminated"].sum()), "policy_sha256": saved["policy_sha256"]})
        check(all(np.array_equal(traces["masked"][k], traces["range"][k]) for k in PAIR_FIELDS)
              and report["paired_rollout_equal"][str(seed)] is True, f"seed {seed}: paired physics and rewards")
    return rows


def resources(root):
    summaries = {}
    for stage in ("preflight", "baseline", "training", "evaluate", "video"):
        folder = root / f"guard_{stage}_01"
        result = load(folder / "report.json")
        check(result["status"] == "COMPLETE" and result["returncode"] == 0 and result["child_reaped"]
              and result["owned_cgroup_children_empty"] and not result["remaining_owned_pids"]
              and len(result["cpu_affinity"]) == 1, f"{stage}: supervision and cleanup")
        low, high, last_start = None, 0, None
        samples = 0
        with (folder / "resources.jsonl").open() as log:
            for line in log:
                snap = json.loads(line)
                owned, effective = snap["owned"], snap["effective_available_bytes"]
                check((owned["high_bytes"], owned["max_bytes"], owned["swap_max_bytes"],
                       owned["swap_current_bytes"]) == (2 << 30, 3 << 30, 0, 0)
                      and not any(owned["events"][k] for k in ("oom", "oom_kill", "oom_group_kill"))
                      and not snap["errors"], f"{stage}: cgroup limits")
                low = effective if low is None else min(low, effective)
                high = max(high, owned["current_bytes"], owned["peak_bytes"] or 0)
                if snap["stage"] == "startup":
                    last_start = effective
                samples += 1
        check(last_start is not None and last_start >= 4 << 30 and low >= 2 << 30
              and (low, high) == (result["minimum_effective_bytes"], result["peak_group_bytes"]),
              f"{stage}: reserve and peaks")
        summaries[stage] = {"samples": samples, "startup_effective_bytes": last_start,
                            "minimum_effective_bytes": low, "peak_group_bytes": high,
                            "children_reaped": True, "elapsed_s": result["elapsed_s"]}
    return summaries


def select(rows):
    refs = {(c["terrain"], c["seed"]): c for c in rows if c["arm"] == "baseline"}
    pools = {}
    for candidate in (r for r in rows if r["arm"] == "range"):
        terrain = candidate["terrain"]
        base = refs[terrain, candidate["seed"]]
        qualified = all(r["course_completed"] and (terrain == "flat" or r["reached_mark"])
                        for r in (base, candidate))
        rank = (not qualified, ("up", "down", "flat").index(terrain))
        pools.setdefault(rank, []).append((base, candidate))
    rank = min(pools)
    pool = pools[rank]
    middle = median(r["fixed_speed_rmse_m_s"] for _, r in pool)
    base, chosen = min(pool, key=lambda p: (abs(p[1]["fixed_speed_rmse_m_s"] - middle), p[1]["training_seed"]))
    return {"baseline": base["case"], "range": chosen["case"], "terrain": chosen["terrain"],
            "seed": chosen["seed"], "training_seed": chosen["training_seed"],
            "qualified_pair_for_display": not rank[0], "group_median_rmse_m_s": middle}


def audit(root):
    plan = load(PROTOCOL)
    stages = {s: load(root/f"{s}_01/report.json") for s in ("baseline", "training", "evaluate")}
    protocol_sha = digest(PROTOCOL)
    for s, r in stages.items():
        check(r["stage"] == s and r["status"] == "COMPLETE" and r["protocol_sha256"] == protocol_sha, s)
    train_seeds = plan["training"]["seeds"]
    held_seeds = plan["evaluation"]["heldout_reset_seeds"]
    terrains = plan["course"]["terrains"]
    pairs = dict(zip(held_seeds, train_seeds, strict=True))
    all_rows = {}
    for stage, arms, seeds in (("baseline", ["baseline"], plan["baseline"]["reset_seeds"]),
                              ("evaluate", ["baseline", "masked", "range"], held_seeds)):
        folder = root/f"{stage}_01"
        expected = {(a, s, t, None if a == "baseline" else pairs[s]) for a in arms for s in seeds for t in terrains}
        actual = [(c["arm"], c["seed"], c["terrain"], c["training_seed"]) for c in stages[stage]["cases"]]
        check(len(actual) == len(expected) and set(actual) == expected, f"{stage}: full case matrix")
        all_rows[stage] = [case_readback(folder/c["case"], c, plan["course"], plan["evaluation"]["course_mark_m_after_toe"])
                           for c in stages[stage]["cases"]]
    held = all_rows["evaluate"]
    index = {(c["arm"], c["seed"], c["terrain"]): c for c in held}
    effects = []
    for seed in held_seeds:
        for terrain in terrains:
            b, m, r = [index[a, seed, terrain]["fixed_speed_rmse_m_s"] for a in ("baseline", "masked", "range")]
            effects.append({"training_seed": pairs[seed], "heldout_seed": seed, "terrain": terrain,
                            "original_rmse_m_s": b, "masked_rmse_m_s": m, "range_rmse_m_s": r,
                            "range_minus_masked_rmse_m_s": r-m, "range_minus_original_rmse_m_s": r-b})
    chosen = select(held)
    for k, v in chosen.items():
        compare(stages["evaluate"]["video_selection"][k], v, "video selection/"+k)
    movie = load(root / "showcase_01.json")
    video_path = root / "showcase_01.mp4"
    check(movie["status"] == "COMPLETE" and all(movie[k] == chosen[k] for k in ("terrain", "seed", "training_seed"))
          and movie["physics_steps"] == movie["policy_calls"] == 0
          and movie["frames_written"] == 150 and movie["fps"] == 25 and float(movie["stream"]["duration"]) == 6.
          and 0 < movie["bytes"] == video_path.stat().st_size <= 5_000_000 and movie["sha256"] == digest(video_path),
          "selected video and encoding")
    for a in ("baseline", "range"):
        case = root/"evaluate_01"/chosen[a]
        check(all(movie[a][f] == digest(case/f) for f in ("trace.npz", "metrics.json")), f"video source {a}")
    trained = paired_training(root/"training_01", stages["training"], plan)
    original = all_rows["baseline"]
    eval_n = sum(c["ticks"] for c in original+held)
    train_n = sum(c["policy_steps"] for c in trained)
    return {"status": "COMPLETE", "protocol_sha256": protocol_sha, "new_physics_steps_in_readback": 0,
            "counts": {"baseline_cases": len(original), "heldout_cases": len(held), "checkpoints": len(trained),
                       "training_transitions": train_n, "evaluation_transitions_including_baseline": eval_n,
                       "native_steps": (eval_n+train_n)*10, "pd_updates": (eval_n+train_n)*5,
                       "adam_steps": sum(r["optimizer_steps"] for r in trained)},
            "original": original, "heldout": held, "paired_heldout": effects, "training": trained,
            "training_rollouts_bitwise_paired": train_seeds,
            "video": {**chosen, "sha256": movie["sha256"], "bytes": movie["bytes"], "stream": movie["stream"]},
            "resources": resources(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            result = audit(args.root)
        except Exception as exc:
            json.dump({"status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
