"""Replay archived policy boundaries to measure the direct effect of the range input."""

import argparse
import hashlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np

from tools87.run87 import PLAN, evidence, read, sha
from tools87.terrain87 import (CAPTURE_HZ, MAX_RANGE_M, POLICY_HZ, RAMP_START_X_M, SENSOR_DELAY_TICKS,
                               TERRAINS, capture_due, ground_height, make_env)
from training.d1_rl.interface import ACTOR_DIM, WHEEL_INDICES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_trace(evaluation, case, case_id, ticks):
    folder = evaluation / case_id
    require(case == read(folder / "metrics.json") and case["status"] == "COMPLETE"
            and case["full_horizon"] and case["ticks"] == ticks, f"incomplete case: {case_id}")
    trace = folder / "trace.npz"
    require(sha(trace) == case["trace_sha256"], f"trace identity: {case_id}")
    with np.load(trace, allow_pickle=False) as archive:
        return {key: archive[key] for key in ("time_s", "qpos", "qvel", "issued_action", "raw_ranges16",
                "mask_action_max_abs", "command_vx_m_s", "body_vx_m_s", "front_contact_x_m")}


def training_exposure(model, report, hz):
    with np.load(model / "train_trace.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ("terrain", "qpos", "terminated", "truncated")}
    hashes = report["trace"]["array_sha256"]
    for key, array in arrays.items():
        require(hashlib.sha256(array.tobytes()).hexdigest() == hashes[key], f"training trace identity: {model.name}/{key}")
    terrain = arrays["terrain"]
    ticks = len(terrain)
    done = arrays["terminated"] | arrays["truncated"]
    require(ticks == report["trace"]["completed_recorded_ticks"] == report["budget"] and
            arrays["qpos"].shape == (ticks, 23) and done.shape == (ticks,), "training tick coverage")
    uphill = terrain == TERRAINS.index("up")
    body_x = arrays["qpos"][:, 0]  # The archived free joint stores the post-interval base origin in world x.
    past = body_x > RAMP_START_X_M
    starts = np.r_[0, np.flatnonzero(done[:-1]) + 1]
    ends = np.r_[starts[1:], ticks]
    episodes = []
    for start, end in zip(starts, ends):
        require(np.all(terrain[start:end] == terrain[start]), "terrain switched within a training episode")
        if uphill[start]:
            episodes.append({"start_tick": int(start), "end_tick_exclusive": int(end),
                             "recorded_terminal_boundary": bool(done[end - 1]),
                             "ticks_base_origin_past_toe": int(np.count_nonzero(past[start:end]))})
    rollout_steps = report["hyperparameters"]["n_steps"]
    require(ticks % rollout_steps == 0, "incomplete training rollout")
    return {"training_seed": report["seed"], "all_ticks": ticks, "uphill_ticks": int(uphill.sum()),
            "uphill_ticks_base_origin_past_toe": int(np.count_nonzero(uphill & past)),
            "uphill_max_base_origin_past_toe_m": float(body_x[uphill].max() - RAMP_START_X_M),
            "toe_x_m": RAMP_START_X_M, "policy_hz": hz, "completed_rollouts": ticks // rollout_steps,
            "optimizer_steps": report["optimizer_steps"], "uphill_episodes": episodes,
            "verified_train_array_sha256": {key: hashes[key] for key in arrays}}


def observations_and_rewards(trace, case, hz):
    """Keep the recorded state/action history. The frozen sensor code only calls mj_forward."""
    ticks = case["ticks"]
    env = make_env(case["terrain"], case["seed"], max_transitions=1)
    try:
        env.reset(seed=case["seed"])
        require(np.array_equal(trace["qpos"][0], env.data.qpos)
                and np.array_equal(trace["qvel"][0], env.data.qvel), f"reset mismatch: {case['case']}")
        require(np.allclose(trace["time_s"], np.arange(ticks + 1) / hz, rtol=0, atol=1e-9),
                f"control boundary times: {case['case']}")
        bodies = np.empty((ticks, ACTOR_DIM), dtype=np.float32)
        speed_reward = np.empty(ticks)
        height_reward = np.empty(ticks)
        max_vx_error = 0.0
        for t in range(ticks):
            # Snapshot row t+1 contains the packet and issued action at boundary t,
            # plus the post-interval pose, velocity and speed reward measurement.
            bodies[t] = env.current_observation()[:ACTOR_DIM]
            command = env._command.copy()
            require(command[0] == trace["command_vx_m_s"][t + 1], "interval command mismatch")
            previous_action = env._last_policy_action.copy()
            env._history[:-1] = env._history[1:].copy()
            env._history[-1] = env._current_obs
            env._last_policy_action[:] = trace["issued_action"][t + 1]
            env.data.qpos[:] = trace["qpos"][t + 1]
            env.data.qvel[:] = trace["qvel"][t + 1]
            env.data.time = trace["time_s"][t + 1]
            env._episode_steps = t + 1
            env._update_command()
            env._observe()
            # Only these two reward terms are used; neither depends on torque/contact.
            _, reward_terms, metrics = env._reward(command, previous_action, 0.0, False, 0)
            terms = reward_terms["contributions_per_tick"]
            speed_reward[t], height_reward[t] = terms["vx_tracking"], terms["height_error_squared"]
            max_vx_error = max(max_vx_error, abs(metrics["body_vx_m_s"] - trace["body_vx_m_s"][t + 1]))
        counts = env.counters()
        require(all(counts[key] == 0 for key in ("native_steps", "pd_updates", "policy_transitions")),
                "unexpected physics integration")
        require(max_vx_error <= 1e-9, f"reward speed reconstruction error: {max_vx_error}")
        cfg = env.config["reward"]
        dt = cfg["multiply_total_by_policy_dt_s"]
        qpos = trace["qpos"][1:]
        height = qpos[:, env.free_qpos + 2]
        road_z = ground_height(case["terrain"], qpos[:, env.free_qpos])
        relative = height - road_z - env.nominal_base_height_m
        return bodies, env.action_scales[WHEEL_INDICES].copy(), {
            "speed_gap": cfg["vx_tracking_weight"] * dt - speed_reward, "world": height_reward,
            "relative": cfg["height_error_squared_weight"] * relative ** 2 * dt,
            "road_z": road_z, "nominal_height": env.nominal_base_height_m, "vx_error": max_vx_error,
        }
    finally:
        env.close()


def action_summary(measured, masked, parent, wheel_scales, selection):
    wheel_action = measured[selection][:, WHEEL_INDICES] * wheel_scales
    wheel_delta = (measured[selection] - masked[selection])[:, WHEEL_INDICES] * wheel_scales
    result = {
        "ticks": int(selection.sum()), "measured_mean_per_wheel_rad_s": wheel_action.mean(axis=0).tolist(),
        "range_minus_masked_mean_per_wheel_rad_s": wheel_delta.mean(axis=0).tolist(),
        "range_minus_masked_four_wheel_mean_rad_s": float(wheel_delta.mean()),
        "range_minus_masked_mean_absolute_rad_s": float(np.abs(wheel_delta).mean()),
        "range_minus_masked_max_absolute_rad_s": float(np.abs(wheel_delta).max()),
        "mean_absolute_change_over_mean_absolute_target": float(np.abs(wheel_delta).mean() / np.abs(wheel_action).mean()),
    }
    if parent is not None:
        body_delta = (masked[selection] - parent[selection])[:, WHEEL_INDICES] * wheel_scales
        result["masked_minus_parent_mean_per_wheel_rad_s"] = body_delta.mean(axis=0).tolist()
        result["masked_minus_parent_mean_absolute_rad_s"] = float(np.abs(body_delta).mean())
    return result


def reward_summary(terms, selection):
    actual = terms["world"][selection]
    relative = terms["relative"][selection]
    return {
        "speed_tracking_gap_to_exact_command": float(terms["speed_gap"][selection].sum()),
        "frozen_world_height_contribution": float(actual.sum()),
        "ground_relative_height_counterfactual_contribution": float(relative.sum()),
        "ground_relative_minus_world_contribution": float((relative - actual).sum()),
        "max_abs_road_height_under_base_m": float(np.abs(terms["road_z"][selection]).max()),
    }


def recast_packets(trace, case, road):
    """Resample the original sensor on stored poses, using its native capture/hold/delivery logic."""
    import mujoco

    ticks = case["ticks"]
    env = make_env(road, case["seed"], max_transitions=1)
    try:
        env.reset(seed=case["seed"])  # The reset samples the first packet at tick zero.
        require(np.array_equal(env.data.qpos, trace["qpos"][0])
                and np.array_equal(env.data.qvel, trace["qvel"][0]), "recast reset pose mismatch")
        packets = np.empty((ticks, 16), dtype=np.float32)
        captures, deliveries = [], []
        for t in range(ticks):
            if capture_due(t):
                captures.append(t)
                if t:
                    env.data.qpos[:] = trace["qpos"][t]
                    env.data.qvel[:] = trace["qvel"][t]
                    env.data.time = trace["time_s"][t]
                    mujoco.mj_forward(env.model, env.data)
            if t:
                env._update_measurement(t)
            if env.sensor_delivery_tick == t:
                require(env.sensor_capture_tick + SENSOR_DELAY_TICKS == t, "recast sensor latency")
                deliveries.append(t)
            packets[t] = env.last_raw_features
        require(deliveries == [t + SENSOR_DELAY_TICKS for t in captures if t + SENSOR_DELAY_TICKS < ticks],
                "recast delivery schedule mismatch")
        require(all(env.counters()[k] == 0 for k in ("native_steps", "pd_updates", "policy_transitions")),
                "recast advanced physics")
        return packets, deliveries
    finally:
        env.close()


def terrain_only_effect(trace, case, predict, bodies, measured, wheel_scales, periods):
    result = {"case": case["case"], "status": "UNVERIFIED", "capture_hz": CAPTURE_HZ,
              "control_hz": POLICY_HZ, "delivery_delay_ticks": SENSOR_DELAY_TICKS}
    try:
        actual, up_deliveries = recast_packets(trace, case, "up")
        archived = trace["raw_ranges16"][1:]
        equal = bool(np.array_equal(actual, archived))
        result.update(archived_feature_values=int(archived.size), archived_features_exact=equal,
                      max_abs_archived_feature_error=float(np.abs(actual - archived).max()))
        require(equal, "on-slope recast differs from the archived policy packet")
        flat, flat_deliveries = recast_packets(trace, case, "flat")
        require(up_deliveries == flat_deliveries, "road change altered sensor clock")
        substituted = np.stack([predict(body, packet) for body, packet in zip(bodies, flat)])
        delta = (measured - substituted)[:, WHEEL_INDICES] * wheel_scales
        both_valid = (actual[:, 8:] != 0) & (flat[:, 8:] != 0)
        distances = np.abs(actual[:, :8] - flat[:, :8]) * MAX_RANGE_M
        result.update(status="VERIFIED", delivered_captures=len(up_deliveries),
                      ticks_changed_packet=int(np.any(actual != flat, axis=1).sum()),
                      ticks_changed_validity=int(np.any(actual[:, 8:] != flat[:, 8:], axis=1).sum()),
                      max_abs_distance_change_jointly_valid_m=float(distances[both_valid].max(initial=0.0)),
                      max_abs_change_normalized_all_16=float(np.abs(measured - substituted).max()),
                      periods={name: {"ticks": int(selected.sum()),
                                      "up_minus_flat_mean_per_wheel_rad_s": delta[selected].mean(axis=0).tolist(),
                                      "up_minus_flat_four_wheel_mean_rad_s": float(delta[selected].mean()),
                                      "mean_abs_up_minus_flat_rad_s": float(np.abs(delta[selected]).mean()),
                                      "max_abs_up_minus_flat_rad_s": float(np.abs(delta[selected]).max())}
                               for name, selected in periods.items()})
    except Exception as exc:
        # Keep independently verified action masking, reward and exposure results.
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit(root):
    from tools87.ppo87 import load_predictor, parent_predictor

    training, evaluation = root / "training_01", root / "evaluate_01"
    models = {m["model"]: m for m in evidence(training, "training")["models"]}
    cases = {c["case"]: c for c in evidence(evaluation, "evaluate")["cases"]}
    plan = read(PLAN)
    ticks, hz = plan["course"]["ticks"], plan["course"]["policy_hz"]
    tolerance = plan["training"]["valid_initial_action_max_abs_error"]
    train_seeds = plan["training"]["seeds"]
    test_seeds = plan["evaluation"]["heldout_reset_seeds"]
    require(len(train_seeds) == len(test_seeds) == 3, "expected three registered pairs")
    parent = parent_predictor()
    zero = np.zeros(16, dtype=np.float32)
    exposures, parent_checks, audits, terrain_effects = [], [], [], []
    for train_seed, seed in zip(train_seeds, test_seeds):
        model_name = f"range_{train_seed}"
        model = training / model_name
        require(sha(model / "report.json") == models[model_name]["report_sha256"], "training report identity")
        predict = load_predictor(model)
        model_report = read(model / "report.json")
        exposures.append(training_exposure(model, model_report, hz))
        # Validate the Python parent against C++34 on actual held-out uphill states.
        parent_id = f"baseline_0_up_{seed}"
        case = cases[parent_id]
        trace = checked_trace(evaluation, case, parent_id, ticks)
        bodies, _, _ = observations_and_rewards(trace, case, hz)
        actions = np.stack([parent(body, zero) for body in bodies])
        parent_error = float(np.max(np.abs(actions - trace["issued_action"][1:])))
        require(parent_error <= tolerance, f"parent issued action error {parent_id}: {parent_error}")
        parent_checks.append({"case": parent_id, "ticks": ticks, "max_abs_error": parent_error,
                              "trace_sha256": case["trace_sha256"]})
        for terrain in plan["course"]["terrains"]:
            case_id = f"{model_name}_{terrain}_{seed}"
            case = cases[case_id]
            trace = checked_trace(evaluation, case, case_id, ticks)
            bodies, wheel_scales, rewards = observations_and_rewards(trace, case, hz)
            measured = np.stack([predict(body, packet) for body, packet in zip(bodies, trace["raw_ranges16"][1:])])
            error = float(np.max(np.abs(measured - trace["issued_action"][1:])))
            require(error <= tolerance, f"measured issued action error {case_id}: {error}")
            masked = np.stack([predict(body, zero) for body in bodies])
            peak = np.max(np.abs(measured - masked), axis=1)
            mask_error = float(np.max(np.abs(peak - trace["mask_action_max_abs"][1:])))
            require(mask_error <= 1e-6, f"archived mask sensitivity mismatch {case_id}: {mask_error}")
            parent_actions = np.stack([parent(body, zero) for body in bodies]) if terrain == "up" else None
            periods = {"moving": trace["command_vx_m_s"][1:] > 0, "last_second": np.arange(ticks) >= ticks - hz}
            loaded = np.flatnonzero(trace["front_contact_x_m"][:-1] >= RAMP_START_X_M)
            if loaded.size:
                periods["from_first_loaded_crossing"] = np.arange(ticks) >= loaded[0]
            if terrain == "up":
                terrain_effects.append(terrain_only_effect(trace, case, predict, bodies, measured, wheel_scales, periods))
            audits.append({
                "case": case_id, "terrain": terrain, "training_seed": train_seed, "reset_seed": seed,
                "trace_sha256": case["trace_sha256"], "policy_sha256": model_report["policy_sha256"],
                "issued_action_max_abs_error": error, "per_tick_mask_max_abs_error": mask_error,
                "archived_speed_max_abs_error_m_s": rewards["vx_error"],
                "nominal_base_height_m": rewards["nominal_height"],
                "periods": {key: {"actions": action_summary(measured, masked, parent_actions, wheel_scales, select),
                                  "reward": reward_summary(rewards, select)} for key, select in periods.items()},
            })
    return {"protocol87_sha256": sha(PLAN), "audit_source_sha256": sha(__file__),
            "action_tolerance": tolerance, "mask_tolerance": 1e-6,
            "wheel_order": ["FL", "FR", "RL", "RR"], "native_steps": 0, "optimizer_steps": 0,
            "same_pose_uphill_vs_flat_sensor": terrain_effects,
            "training_exposure": exposures, "parent_uphill_cpp_checks": parent_checks, "range_cases": audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report = {"status": "COMPLETE", **audit(args.root)}
        except Exception as exc:
            json.dump({"status": "INCOMPLETE", "error": str(exc)}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
