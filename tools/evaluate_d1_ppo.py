#!/usr/bin/env python3
"""Fixed 12 s pure-D1 PPO evaluation with strict episode and physics caps.

Pilot: 3 sequences x reset seed 7001 x 0 ms delay = 3 episodes.
Formal: 3 sequences x seeds 1001/1002/1003 x delays 0/20/40/60 ms = 36.
This script writes a compact trace per episode; it is not a training script.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from d1_rl.factory03 import environment_class, environment_sources  # noqa: E402
from d1_rl.interface import ACTION_DIM, ACTION_SCALE, ACTOR_DIM, OBS_DIM  # noqa: E402
from export_d1_ppo import check_checkpoint  # noqa: E402

DEFAULT_CONFIG = ROOT / "training/configs/pilot03.json"
POLICY_DT_S = 0.02
EPISODE_TICKS = 600
FAILURE_PENALTY_J = 100.0
MISSING_INTERVAL_ERROR = 0.2

SEQUENCES = {
    "forward_stop": [
        {"start_s": 0.0, "vx": 0.0, "wz": 0.0},
        {"start_s": 2.0, "vx": 0.2, "wz": 0.0},
        {"start_s": 8.0, "vx": 0.0, "wz": 0.0},
    ],
    "reverse_stop": [
        {"start_s": 0.0, "vx": 0.0, "wz": 0.0},
        {"start_s": 2.0, "vx": -0.2, "wz": 0.0},
        {"start_s": 8.0, "vx": 0.0, "wz": 0.0},
    ],
    "turn_stop": [
        {"start_s": 0.0, "vx": 0.0, "wz": 0.0},
        {"start_s": 2.0, "vx": 0.0, "wz": 0.2},
        {"start_s": 5.0, "vx": 0.0, "wz": -0.2},
        {"start_s": 8.0, "vx": 0.0, "wz": 0.0},
    ],
}

THRESHOLDS = {
    "stable_vx_mae_m_s": 0.06,
    "stable_wz_mae_rad_s": 0.08,
    "stop_last_second_horizontal_speed_m_s": 0.02,
    "stop_last_second_abs_wz_rad_s": 0.03,
    "stop_displacement_m": 0.05,
    "peak_abs_roll_or_pitch_rad": 0.20,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def make_scenarios(phase):
    delays = [0] if phase == "pilot" else [0, 1, 2, 3]
    seeds = [7001] if phase == "pilot" else [1001, 1002, 1003]
    return [(name, delay, seed) for name in SEQUENCES for delay in delays for seed in seeds]


def make_predictor(checkpoint, onnx_path, config_path):
    if checkpoint is not None:
        protocol_path, protocol, result_path, result = check_checkpoint(checkpoint, config_path)
        model = PPO.load(str(checkpoint), device="cpu")
        if tuple(model.policy.observation_space.shape) != (ACTOR_DIM,) or (
                tuple(model.policy.action_space.shape) != (ACTION_DIM,)):
            raise ValueError("Checkpoint actor shape differs from frozen 627/16")

        def predict(actor_obs):
            actions, _ = model.predict(actor_obs, deterministic=True)
            result_action = np.asarray(actions, dtype=np.float32)
            if result_action.shape != (ACTION_DIM,) or not np.all(np.isfinite(result_action)):
                raise ValueError("Non-finite or malformed SB3 deterministic action")
            return np.clip(result_action, -1.0, 1.0)

        provenance = {"kind": "self_trained_sb3_checkpoint",
                      "checkpoint_sha256": sha256(checkpoint),
                      "training_protocol_sha256": sha256(protocol_path),
                      "training_result_sha256": sha256(result_path),
                      "training_kind": protocol["kind"], "training_seed": protocol["seed"],
                      "training_arm": protocol["arm"],
                      "trained_transitions": result["trained_transitions"],
                      "random_initialization": True, "teacher_policy_used": False,
                      "vecnormalize_used": False}
        return predict, provenance

    metadata_path = onnx_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scene = ROOT / config["scene"]
    robot = scene.parent / "robot.xml"
    schema = ROOT / config["interface_schema"]
    audited_schema = ROOT / config.get("audited_interface_schema", config["interface_schema"])
    expected = {"policy_onnx": onnx_path, "config": config_path,
                "scene_xml": scene, "robot_xml": robot, "interface_schema": schema,
                "audited_interface_schema": audited_schema,
                "golden_fixtures": onnx_path.parent / "golden_fixtures.json"}
    for name, path in expected.items():
        if sha256(path) != metadata["sha256"][name]:
            raise ValueError(f"ONNX metadata SHA mismatch for {name}")
    if sha256(audited_schema) != config.get(
            "audited_interface_schema_sha256", config["interface_schema_sha256"]):
        raise ValueError("Audited observation schema SHA256 mismatch")
    if (metadata["format"] != "d1-pure-ppo-onnx-export-v1" or
            metadata["interface_id"] != config.get("interface_id", "d1-selftrained-pilot03-v1") or
            metadata["config_version"] != config["version"] or
            metadata["action_scales"] != config.get("action_scales", ACTION_SCALE.tolist()) or
            metadata["audited_interface_schema_sha256"] != config.get(
                "audited_interface_schema_sha256", config["interface_schema_sha256"]) or
            metadata["onnx_ir_version"] != 7 or metadata["onnx_opset"] != 11 or
            metadata["history_order"] != "oldest_to_newest_previous_only" or
            metadata["policy_operation"] != "deterministic Gaussian mean, clipped to [-1,1]" or
            metadata["golden_parity"]["fixtures"] != 64 or
            metadata["golden_parity"]["torch_vs_python_ort_max_abs_error"] > 1e-5 or
            metadata["golden_parity"]["sb3_predict_vs_torch_max_abs_error"] > 1e-6 or
            metadata["teacher_policy_used"] is not False or
            metadata["random_initialization"] is not True or
            metadata["vecnormalize_used"] is not False):
        raise ValueError("ONNX metadata lacks frozen self-trained provenance")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options,
                                   providers=["CPUExecutionProvider"])
    if ([(x.name, x.shape, x.type) for x in session.get_inputs()] != [
            ("obs", [1, 57], "tensor(float)"),
            ("history", [1, 10, 57], "tensor(float)")] or
            [(x.name, x.shape, x.type) for x in session.get_outputs()] != [
                ("actions", [1, 16], "tensor(float)")]):
        raise ValueError("ONNX runtime signature must be obs/history -> actions")

    def predict(actor_obs):
        vector = np.asarray(actor_obs, dtype=np.float32)
        if vector.shape != (ACTOR_DIM,) or not np.all(np.isfinite(vector)):
            raise ValueError("Malformed actor observation")
        output = session.run(["actions"], {
            "obs": vector[:OBS_DIM].reshape(1, 57),
            "history": vector[OBS_DIM:].reshape(1, 10, 57),
        })[0]
        if output.shape != (1, ACTION_DIM) or not np.all(np.isfinite(output)):
            raise ValueError("Non-finite or malformed ONNX action")
        return np.clip(output[0], -1.0, 1.0).astype(np.float32)

    provenance = {"kind": "self_trained_onnx_python_ort",
                  "onnx_sha256": sha256(onnx_path), "metadata_sha256": sha256(metadata_path),
                  "training_kind": metadata["training_kind"],
                  "training_seed": metadata["training_seed"],
                  "training_arm": metadata["training_arm"],
                  "trained_transitions": metadata["trained_transitions"],
                  "python_ort_version": ort.__version__,
                  "cxx_ort_1_10_status": "separate_runtime_gate",
                  "random_initialization": True, "teacher_policy_used": False,
                  "vecnormalize_used": False}
    return predict, provenance


def trace_metrics(name, delay, seed, trace, positions, done_reason):
    n = len(trace["reward"])
    t = np.arange(n, dtype=np.float64) * POLICY_DT_S
    vx = np.asarray(trace["body_vx_m_s"], dtype=np.float64)
    vy = np.asarray(trace["body_vy_m_s"], dtype=np.float64)
    wz = np.asarray(trace["body_wz_rad_s"], dtype=np.float64)
    command = np.asarray(trace["command_during_interval"], dtype=np.float64).reshape(n, 3)
    speed = np.hypot(vx, vy)
    full = n == EPISODE_TICKS and done_reason == "HORIZON_12S"

    def window_mean(values, lo, hi):
        selected = (t >= lo - 1e-12) & (t < hi - 1e-12)
        return float(np.mean(values[selected])) if np.any(selected) else None

    if name in ("forward_stop", "reverse_stop"):
        stable_vx = window_mean(np.abs(vx - command[:, 0]), 4, 8)
        stable_wz = None
    else:
        # Both signs are tested separately so one good turn cannot hide the other.
        first = window_mean(np.abs(wz - command[:, 2]), 3, 5)
        second = window_mean(np.abs(wz - command[:, 2]), 6, 8)
        stable_vx = None
        stable_wz = max(first, second) if first is not None and second is not None else None
    stop_speed = window_mean(speed, 11, 12)
    stop_wz = window_mean(np.abs(wz), 11, 12)
    stop_displacement = (float(np.linalg.norm(positions[600] - positions[400]))
                         if len(positions) == EPISODE_TICKS + 1 else None)
    peak_tilt_components = (float(max(np.max(np.abs(trace["base_roll_rad"])),
                                      np.max(np.abs(trace["base_pitch_rad"])))) if n else None)
    failure = not full or bool(np.any(trace["fallen"])) or bool(np.any(trace["illegal_body_contact"]))

    # Missing intervals are explicitly charged at one normalized unit per
    # component, plus a fixed failure penalty, so early falls cannot lower J.
    vx_error = np.full(EPISODE_TICKS, MISSING_INTERVAL_ERROR, dtype=np.float64)
    wz_error = np.full(EPISODE_TICKS, MISSING_INTERVAL_ERROR, dtype=np.float64)
    stop_error = np.full(EPISODE_TICKS, MISSING_INTERVAL_ERROR, dtype=np.float64)
    vx_error[:n] = np.abs(vx - command[:, 0])
    wz_error[:n] = np.abs(wz - command[:, 2])
    stop_error[:n] = speed
    components = {
        "full_horizon_vx_mae_over_0_2": float(np.mean(vx_error) / 0.2),
        "full_horizon_wz_mae_over_0_2": float(np.mean(wz_error) / 0.2),
        "stop_8_to_12_horizontal_norm_mean_over_0_2": float(np.mean(stop_error[400:600]) / 0.2),
        "fixed_early_failure_penalty": FAILURE_PENALTY_J if failure else 0.0,
    }
    j = float(sum(components.values()))
    flags = {
        "completed_12s": full,
        "no_fall_or_illegal_support": not failure,
        "stable_vx": None if stable_vx is None else stable_vx <= THRESHOLDS["stable_vx_mae_m_s"],
        "stable_wz": None if stable_wz is None else stable_wz <= THRESHOLDS["stable_wz_mae_rad_s"],
        "stop_horizontal_speed": stop_speed is not None and stop_speed <= THRESHOLDS[
            "stop_last_second_horizontal_speed_m_s"],
        "stop_abs_wz": stop_wz is not None and stop_wz <= THRESHOLDS[
            "stop_last_second_abs_wz_rad_s"],
        "stop_displacement": stop_displacement is not None and stop_displacement <= THRESHOLDS[
            "stop_displacement_m"],
        "peak_roll_pitch": peak_tilt_components is not None and peak_tilt_components <= THRESHOLDS[
            "peak_abs_roll_or_pitch_rad"],
    }
    reference_thresholds_met = all(value for value in flags.values() if value is not None)
    return {"sequence": name, "delay_ticks": delay, "delay_ms": delay * 20,
            "reset_seed": seed, "policy_transitions": n,
            "native_steps": n * 10, "done_reason": done_reason,
            "status": "FAIL" if failure else "COMPLETE",
            "failure": failure, "J": j, "J_components": components,
            "stable_vx_mae_m_s": stable_vx, "stable_wz_mae_rad_s": stable_wz,
            "stop_last_second_horizontal_speed_m_s": stop_speed,
            "stop_last_second_abs_wz_rad_s": stop_wz,
            "stop_displacement_m": stop_displacement,
            "peak_abs_roll_or_pitch_rad": peak_tilt_components,
            "threshold_flags": flags,
            "reference_thresholds_met": reference_thresholds_met,
            "nominal_zero_delay_pass": reference_thresholds_met if delay == 0 else None}


def run_episode(env, predictor, name, delay, seed, trace_path, native_budget_remaining):
    env.eval_delay_ticks = delay
    before = env.counters()
    observation, reset_info = env.reset(seed=seed)
    positions = [np.array([env.data.qpos[env.free_qpos], env.data.qpos[env.free_qpos + 1]],
                          dtype=np.float64)]
    keys = ("reward", "command_during_interval", "body_vx_m_s", "body_vy_m_s",
            "body_wz_rad_s", "base_x_m", "base_y_m", "base_height_m",
            "base_roll_rad", "base_pitch_rad", "base_yaw_rad", "tilt_rad",
            "fallen", "illegal_body_contact", "tau_saturation_count",
            "issued_action", "applied_action")
    trace = {key: [] for key in keys}
    done_reason = "HORIZON_12S"
    for tick in range(EPISODE_TICKS):
        if native_budget_remaining < (tick + 1) * 10:
            raise RuntimeError("Evaluation native-step budget exhausted before episode horizon")
        action = predictor(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        item = {"reward": reward, "issued_action": np.asarray(action, dtype=np.float32), **info}
        for key in keys:
            trace[key].append(item[key])
        positions.append(np.array([info["base_x_m"], info["base_y_m"]], dtype=np.float64))
        if terminated or truncated:
            done_reason = "TERMINATED" if terminated else "ENV_TRUNCATED_BEFORE_12S"
            break
    after = env.counters()
    delta = {key: after[key] - before[key] for key in after}
    if delta["native_steps"] != 10 * delta["policy_transitions"] or (
            delta["pd_updates"] != 5 * delta["policy_transitions"]):
        raise RuntimeError(f"Actual native/PD/policy counters are inconsistent: {delta}")
    metrics = trace_metrics(name, delay, seed, trace, positions, done_reason)
    metrics["actual_counts"] = delta
    metrics["reset_info"] = reset_info
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **{key: np.asarray(values) for key, values in trace.items()},
                            boundary_xy_m=np.asarray(positions),
                            policy_interval_start_s=np.arange(len(trace["reward"])) * POLICY_DT_S)
    metrics["trace_file"] = trace_path.name
    metrics["trace_sha256"] = sha256(trace_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--onnx", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=["pilot", "formal"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-native-steps", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not math.isclose(config["native_dt_s"] * config["policy_native_steps"], POLICY_DT_S,
                        abs_tol=1e-12):
        parser.error("Evaluation requires 20 ms policy ticks")
    scenarios = make_scenarios(args.phase)
    cap = 3 if args.phase == "pilot" else 36
    if len(scenarios) != cap:
        raise AssertionError("Frozen evaluation scenario count changed")
    requested = cap if args.max_episodes is None else args.max_episodes
    if not 1 <= requested <= cap:
        parser.error(f"--max-episodes must be in 1..{cap} for {args.phase}")
    scenarios = scenarios[:requested]
    native_cap = cap * EPISODE_TICKS * 10
    max_native = native_cap if args.max_native_steps is None else args.max_native_steps
    if not 10 <= max_native <= native_cap:
        parser.error(f"--max-native-steps must be in 10..{native_cap}")
    if max_native < requested * EPISODE_TICKS * 10:
        parser.error("--max-native-steps must cover every requested full 12 s episode")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    onnx_path = args.onnx.resolve() if args.onnx else None
    predictor, provenance = make_predictor(checkpoint, onnx_path, config_path)
    scene = ROOT / config["scene"]
    robot = scene.parent / "robot.xml"
    schema = ROOT / config["interface_schema"]
    contract = {
        "format": "d1-fixed-evaluation-03-v1", "phase": args.phase,
        "scope": "pure D1 flat-ground policy; no terrain traversal claim",
        "provenance": provenance,
        "environment_class": environment_class(config).__name__,
        "environment_sources_sha256": {str(path): sha256(path) for path in environment_sources(config)},
        "source_sha256": {"config": sha256(config_path), "scene": sha256(scene),
                          "robot": sha256(robot), "interface_schema": sha256(schema),
                          "evaluation_script": sha256(Path(__file__).resolve())},
        "sequences": SEQUENCES, "scenarios": [{"sequence": n, "delay_ticks": d,
                                              "reset_seed": s} for n, d, s in scenarios],
        "full_phase_episodes": cap, "requested_episodes": requested,
        "episode_seconds": 12.0, "policy_tick_seconds": POLICY_DT_S,
        "policy_ticks_per_episode": EPISODE_TICKS,
        "planned_max_native_steps": requested * EPISODE_TICKS * 10,
        "hard_max_native_steps": max_native,
        "stable_windows_s": {"translation": [4, 8], "turn_positive": [3, 5],
                             "turn_negative": [6, 8], "stop_last_second": [11, 12],
                             "stop_displacement": [8, 12]},
        "thresholds_nominal_zero_delay": THRESHOLDS,
        "J_definition": "mean full-12s |vx-cmd_vx|/0.2 + mean full-12s |wz-cmd_wz|/0.2 + mean 8..12s horizontal Euclidean speed/0.2; missing intervals cost 0.2 per component plus fixed 100 on early failure",
        "early_failure_penalty_J": FAILURE_PENALTY_J,
        "missing_interval_error": MISSING_INTERVAL_ERROR,
        "stop_speed_definition": "mean Euclidean norm sqrt(body_vx^2+body_vy^2)",
        "stop_displacement_definition": "Euclidean world XY displacement from t=8.0 to t=12.0",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "evaluation_contract.json").open("x", encoding="utf-8") as stream:
        stream.write(json_text(contract))
    results = []
    actual_native = 0
    environment = None
    try:
        for name in SEQUENCES:
            members = [(n, d, s) for n, d, s in scenarios if n == name]
            if not members:
                continue
            environment = environment_class(config)(config=config_path, arm="B", seed=members[0][2], rank=0,
                                max_transitions=EPISODE_TICKS * len(members),
                                eval_delay_ticks=members[0][1],
                                eval_command_sequence=SEQUENCES[name])
            for _, delay, seed in members:
                trace_path = output / f"{name}_delay{delay}_seed{seed}.npz"
                result = run_episode(environment, predictor, name, delay, seed,
                                     trace_path, max_native - actual_native)
                results.append(result)
                actual_native += result["actual_counts"]["native_steps"]
                with (output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            environment.close()
            environment = None
    except BaseException as error:
        partial_counts = environment.counters() if environment is not None else None
        partial_native = 0
        if partial_counts is not None:
            completed_in_current = sum(item["actual_counts"]["native_steps"] for item in results
                                       if item["sequence"] == name)
            partial_native = max(0, partial_counts["native_steps"] - completed_in_current)
        failure = {"status": "INVALID_INTERRUPTED", "error": f"{type(error).__name__}: {error}",
                   "actual_native_steps": actual_native + partial_native,
                   "partial_current_environment_counts": partial_counts,
                   "completed_episode_reports": len(results),
                   "traceback": traceback.format_exc()}
        with (output / "error.json").open("x", encoding="utf-8") as stream:
            stream.write(json_text(failure))
        raise
    finally:
        if environment is not None:
            environment.close()
    grouped = defaultdict(list)
    for item in results:
        grouped[str(item["delay_ms"])].append(item)
    summary = {
        "status": "COMPLETE" if len(results) == cap else "PARTIAL",
        "phase": args.phase, "episodes_completed": len(results), "full_phase_episodes": cap,
        "actual_native_steps": actual_native,
        "actual_policy_transitions": sum(x["actual_counts"]["policy_transitions"] for x in results),
        "actual_pd_updates": sum(x["actual_counts"]["pd_updates"] for x in results),
        "early_failures": sum(x["failure"] for x in results),
        "zero_delay_nominal_pass_count": sum(x["nominal_zero_delay_pass"] is True for x in results),
        "by_delay_ms": {delay: {"episodes": len(items),
                                "failures": sum(x["failure"] for x in items),
                                "J_mean": float(np.mean([x["J"] for x in items])),
                                "J_values": [x["J"] for x in items]}
                        for delay, items in grouped.items()},
        "contract_sha256": sha256(output / "evaluation_contract.json"),
        "interpretation": "Python evaluator only; C++ ORT 1.10 and terrain capability require separate evidence",
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        stream.write(json_text(summary))
    print(json_text(summary))
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
