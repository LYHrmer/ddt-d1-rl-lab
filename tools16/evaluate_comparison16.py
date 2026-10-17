#!/usr/bin/env python3
"""Bounded pilot evaluation of completed comparison16 checkpoints only.

Imports are stdlib-only until a caller explicitly runs evaluation. The frozen
evaluator's scenario, rollout and metric functions are reused without importing
its old PPO/ONNX predictor. No experiment runs when this module is imported.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_d1_ppo.py"
TASKS = ("forward_stop", "reverse_stop", "turn_stop")
COUNTS = {"policy_transitions": 1800, "native_steps": 18000, "pd_updates": 9000}
CASE_COUNTS = {"policy_transitions": 600, "native_steps": 6000, "pd_updates": 3000}
RESULT_FORMAT = "d1-comparison16-evaluation-result-v1"
PROTOCOL_FORMAT = "d1-comparison16-evaluation-protocol-v1"
VERSION_KEYS = {"python", "torch", "sb3", "numpy", "gymnasium", "mujoco"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    result = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(result, dict), "expected a JSON object")
    return result


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contract_api():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools16 import contract16
    return contract16


def load_inputs(release_path):
    api = contract_api()
    release = api.load_release(Path(release_path), require_authorized=True)
    api.validate_sources(release)
    return release


def validate_launch(release, release_path, stage_name, output):
    from tools16.run_comparison16 import validate_stage_launch
    validate_stage_launch(release, release_path, stage_name, output, root=ROOT)


def output_root(release):
    path = Path(release["output_root"])
    return (path if path.is_absolute() else ROOT / path).resolve()


def counts_match(value, expected):
    return isinstance(value, dict) and all(type(value.get(key)) is int and value[key] == count
                                          for key, count in expected.items())


def training_identity(release_path, release, arm, checkpoint):
    """Check bytes and exact completed training provenance before loading SB3."""
    require(arm in ("control", "velocity"), "unknown comparison arm")
    checkpoint = Path(checkpoint).resolve()
    expected = output_root(release) / f"train_{arm}" / "policy.zip"
    require(checkpoint == expected and checkpoint.name == "policy.zip", "only the formal train policy.zip is eligible")
    protocol_path, result_path = checkpoint.parent / "protocol.json", checkpoint.parent / "result.json"
    protocol, result = read_json(protocol_path), read_json(result_path)
    release_sha = sha256(release_path)
    aux = release["arms"][arm]["aux_coef"]
    transitions = release["stages"]["train"]["transitions"]
    expected_counts = {"policy_transitions": transitions, "native_steps": transitions * 10,
                       "pd_updates": transitions * 5}
    common = {"release_sha256": release_sha, "arm": arm, "stage": "train",
              "envs": release["envs"], "seed": release["seed"]}
    for item in (protocol, result):
        require(all(item.get(key) == value for key, value in common.items()), "training identity mismatch")
    require(protocol.get("format") == "d1-comparison16-training-protocol-v1"
            and result.get("format") == "d1-comparison16-training-result-v1", "unexpected training schema")
    require(protocol.get("planned_transitions") == transitions
            and protocol.get("planned_native_steps") == transitions * 10
            and protocol.get("inputs_sha256") == release["sources_sha256"], "training budget or source mismatch")
    config = read_json(ROOT / release["config"])
    require(protocol.get("config") == config and protocol.get("random_initialization") is True
            and protocol.get("warm_start") is False and protocol.get("teacher_policy_used") is False,
            "training config or initialization provenance mismatch")
    hyper = protocol.get("hyperparameters", {})
    require(all(hyper.get(key) == value for key, value in {
        "aux_coef": aux, "device": "cpu", "cpu_threads": 1, "n_steps": 4096 // release["envs"],
        "batch_size": 256, "n_epochs": 4}.items()), "training hyperparameter mismatch")
    require(result.get("status") == "COMPLETE" and result.get("eligible_for_evaluation") is True
            and result.get("inputs_unchanged") is True and result.get("cleanup_complete") is True
            and result.get("error") is None and result.get("aux_coef") == aux
            and result.get("trained_transitions") == transitions
            and counts_match(result.get("actual_counts"), expected_counts), "training is incomplete or ineligible")
    workers = result.get("per_worker_counts")
    require(isinstance(workers, list) and len(workers) == release["envs"]
            and all(counts_match(worker, {key: value // release["envs"]
                                         for key, value in expected_counts.items()}) for worker in workers),
            "per-worker training counters mismatch")
    require(result.get("protocol_sha256") == sha256(protocol_path), "training protocol SHA mismatch")
    require(result.get("checkpoint_sha256") == sha256(checkpoint), "checkpoint SHA mismatch")
    versions = protocol.get("versions")
    require(isinstance(versions, dict) and set(versions) == VERSION_KEYS
            and all(isinstance(value, str) and value for value in versions.values())
            and versions["sb3"] == "2.7.0", "training runtime versions missing or unsupported")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "training_protocol_sha256": sha256(protocol_path), "training_result_sha256": sha256(result_path),
            "training_versions": versions}


def frozen_helpers(release):
    """Compile only the original pure helpers/rollout, excluding heavy imports.

    The source is trusted and SHA-bound by the release, not user-supplied code.
    Bodies and constants are taken verbatim; metric formulas are not copied.
    """
    source_name = str(EVALUATOR.relative_to(ROOT))
    source_bytes = EVALUATOR.read_bytes()
    require(release["sources_sha256"].get(source_name) == hashlib.sha256(source_bytes).hexdigest(),
            "frozen evaluator source SHA mismatch")
    import numpy as np
    constants = {"POLICY_DT_S", "EPISODE_TICKS", "FAILURE_PENALTY_J", "MISSING_INTERVAL_ERROR",
                 "SEQUENCES", "THRESHOLDS"}
    functions = {"make_scenarios", "run_episode", "trace_metrics"}
    nodes, found = [], set()
    for node in ast.parse(source_bytes, filename=str(EVALUATOR)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.FunctionDef):
            name = node.name
        else:
            continue
        if name in constants | functions:
            require(name not in found, "duplicate frozen evaluator definition")
            require(not isinstance(node, ast.FunctionDef) or not node.decorator_list,
                    "decorated frozen evaluator helper")
            nodes.append(node)
            found.add(name)
    require(found == constants | functions, "frozen evaluator helper set changed")
    namespace = {"np": np, "sha256": sha256}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(EVALUATOR), "exec"), namespace)  # noqa: S102 -- SHA-bound frozen functions only.
    return SimpleNamespace(**{key: namespace[key] for key in constants | functions})


def valid_case_completion(item):
    n, reason = item.get("policy_transitions"), item.get("done_reason")
    return (type(n) is int and 1 <= n <= 600 and item.get("native_steps") == n * 10
            and counts_match(item.get("actual_counts"), {"policy_transitions": n, "native_steps": n * 10,
                                                         "pd_updates": n * 5})
            and reason in ("HORIZON_12S", "TERMINATED", "ENV_TRUNCATED_BEFORE_12S")
            and (reason != "HORIZON_12S" or n == 600))


def verify_episode_trace(path, item, helpers):
    """Check actual trace/count/command bytes and recompute frozen metrics."""
    import numpy as np
    name = item["sequence"]
    require(name in TASKS and item.get("trace_file") == f"{name}.npz", "unexpected trace filename")
    trace_path = Path(path).parent / item["trace_file"]
    require(sha256(trace_path) == item.get("trace_sha256"), "trace SHA mismatch")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    n = len(trace["reward"])
    require(1 <= n <= 600, "trace exceeds the registered horizon or is empty")
    scalars = ("reward", "body_vx_m_s", "body_vy_m_s", "body_wz_rad_s", "base_x_m", "base_y_m",
               "base_height_m", "base_roll_rad", "base_pitch_rad", "base_yaw_rad", "tilt_rad", "tau_saturation_count")
    shapes = {key: (n,) for key in scalars}
    shapes.update(command_during_interval=(n, 3), issued_action=(n, 16), applied_action=(n, 16),
                  boundary_xy_m=(n + 1, 2), policy_interval_start_s=(n,))
    for key, shape in shapes.items():
        array = trace[key]
        require(array.shape == shape and np.issubdtype(array.dtype, np.number)
                and bool(np.isfinite(array).all()), f"invalid trace field: {key}")
    for key in ("fallen", "illegal_body_contact"):
        require(trace[key].shape == (n,) and trace[key].dtype == np.bool_, f"invalid trace flags: {key}")
    require(np.array_equal(trace["policy_interval_start_s"], np.arange(n) * .02), "trace clock differs")
    commands = []
    for start_s in trace["policy_interval_start_s"]:
        segment = helpers.SEQUENCES[name][0]
        for candidate in helpers.SEQUENCES[name]:
            if candidate["start_s"] <= start_s + 1e-12:
                segment = candidate
            else:
                break
        commands.append((segment["vx"], 0., segment["wz"]))
    require(np.array_equal(trace["command_during_interval"], np.asarray(commands, dtype=np.float64).reshape(n, 3)),
            "trace commands differ from the frozen pilot sequence")
    require(np.array_equal(trace["boundary_xy_m"][1:, 0], trace["base_x_m"])
            and np.array_equal(trace["boundary_xy_m"][1:, 1], trace["base_y_m"]), "trace boundary positions differ")
    expected_counts = {"policy_transitions": n, "native_steps": n * 10, "pd_updates": n * 5}
    require(counts_match(item.get("actual_counts"), expected_counts), "case counters differ from trace")
    require(valid_case_completion(item), "illegal episode completion/counts")
    calculated = helpers.trace_metrics(name, 0, 7001, trace, trace["boundary_xy_m"], item["done_reason"])
    require(all(item.get(key) == value for key, value in calculated.items()), "stored metrics differ from trace")
    return calculated


def evaluation_protocol(release_path, release, arm, identity, backend):
    require(release["stages"]["eval"]["transitions"] == 1800, "evaluation budget must be exactly 1800")
    return {"format": PROTOCOL_FORMAT, "release_sha256": sha256(release_path), "arm": arm,
            "seed": release["seed"], "envs": release["envs"], "aux_coef": release["arms"][arm]["aux_coef"],
            "config": release["config"], "inputs_sha256": release["sources_sha256"],
            "training": identity, "backend": backend, "phase": "pilot",
            "scenarios": [{"sequence": name, "delay_ticks": 0, "reset_seed": 7001} for name in TASKS],
            "counts": COUNTS, "case_counts": CASE_COUNTS, "episode_seconds": 12.0,
            "max_wall_s": release["stages"]["eval"]["max_wall_s"],
            "metrics_source_sha256": sha256(EVALUATOR),
            "score_definition": {"tracking": "mean of each task stable error divided by 0.06(vx) or 0.08(wz)",
                                 "parking": "mean over three tasks of mean(speed_last1s/0.02,abswz_last1s/0.03,displacement8to12s/0.05)"},
            "scope": "single training seed, three fixed zero-delay pilot cases; exploratory only"}


def numerical_predictor(model, np):
    """Supply a dummy label; only deployable actor627 controls the action."""
    def predict(observation):
        actor = np.asarray(observation)
        require(actor.shape == (627,) and actor.dtype == np.float32 and bool(np.isfinite(actor).all()),
                "expected finite float32 actor627")
        action, _ = model.predict({"actor": actor, "velocity_target": np.zeros(3, dtype=np.float32)},
                                  deterministic=True)
        action = np.asarray(action)
        require(action.shape == (16,) and bool(np.isfinite(action).all()), "invalid deterministic action")
        return np.clip(action, -1., 1.).astype(np.float32)
    return predict


def runtime(release, arm, checkpoint):
    # Called only by explicit evaluation, never by inspection/comparison.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    for directory in (ROOT / ".python-deps", ROOT / "training"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import gymnasium
    import numpy as np
    import stable_baselines3 as sb3
    import torch
    from d1_rl.factory03 import environment_class
    from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO

    import mujoco
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model = AuxiliaryVelocityPPO.load(str(checkpoint), device="cpu")
    require(model.num_timesteps == release["stages"]["train"]["transitions"]
            and model.aux_coef == release["arms"][arm]["aux_coef"] and model.n_envs == release["envs"]
            and model.seed == release["seed"], "loaded checkpoint attributes mismatch")
    config_path = ROOT / release["config"]
    cls = environment_class(config_path)
    backend = {"kind": "sb3_cpu_deterministic", "algorithm": "d1_rl.velocity_policy15.AuxiliaryVelocityPPO",
               "device": "cpu", "cpu_threads": 1, "target_for_control": False,
               "python": platform.python_version(), "gymnasium": gymnasium.__version__,
               "sb3": sb3.__version__, "torch": torch.__version__, "mujoco": mujoco.__version__, "numpy": np.__version__}
    def make_env(name, sequence):
        return cls(config=config_path, arm="B", seed=7001, rank=0, max_transitions=600,
                   eval_delay_ticks=0, eval_command_sequence=sequence)
    return SimpleNamespace(predict=numerical_predictor(model, np), make_env=make_env, backend=backend)


def run_evaluation(release_path, arm, checkpoint, output, *, runtime_factory=runtime, clock=time.monotonic):
    """Explicit bounded execution; the parent runner supplies a hard wall limit."""
    started = clock()
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    release = load_inputs(release_path)
    validate_launch(release, release_path, f"eval_{arm}", output)
    identity = training_identity(release_path, release, arm, checkpoint)
    require(output == output_root(release) / f"eval_{arm}", "evaluation output is outside its registered arm")
    require(not output.exists(), "refusing to overwrite evaluation output")
    helpers = frozen_helpers(release)
    require(helpers.make_scenarios("pilot") == [(name, 0, 7001) for name in TASKS], "pilot scenarios changed")
    output.mkdir(parents=True)
    report = {"format": RESULT_FORMAT, "status": "INCOMPLETE", "release_sha256": sha256(release_path),
              "arm": arm, "seed": release["seed"], "envs": release["envs"],
              "aux_coef": release["arms"][arm]["aux_coef"], **identity, "cases": [],
              "actual_counts": None, "inputs_unchanged": False, "cleanup_complete": True,
              "error": None, "zero_delay_nominal_pass_count": 0, "elapsed_s": None,
              "full_horizons_complete": False,
              "interpretation": "Python SB3 pilot only; no statistical, generalization or C++ qualification claim"}
    environment, current_counted = None, False
    constructing = False
    total = dict.fromkeys(COUNTS, 0)
    unknown_counts = False
    try:
        rt = runtime_factory(release, arm, Path(identity["checkpoint"]))
        report["backend"] = rt.backend
        require({key: rt.backend.get(key) for key in VERSION_KEYS} == identity["training_versions"],
                "evaluation runtime versions differ from training")
        protocol = evaluation_protocol(release_path, release, arm, identity, rt.backend)
        write_json(output / "protocol.json", protocol)
        report["protocol_sha256"] = sha256(output / "protocol.json")
        deadline = started + protocol["max_wall_s"]
        def predict(observation):
            if clock() >= deadline:
                raise TimeoutError("evaluation wall budget exceeded")
            return rt.predict(observation)
        for name in TASKS:
            require(clock() < deadline, "evaluation wall budget exceeded")
            current_counted = False
            constructing = True
            environment = rt.make_env(name, helpers.SEQUENCES[name])
            constructing = False
            item = helpers.run_episode(environment, predict, name, 0, 7001,
                                       output / f"{name}.npz", COUNTS["native_steps"] - total["native_steps"])
            require(item.get("sequence") == name, "evaluation case identity differs")
            verify_episode_trace(output / "report.json", item, helpers)
            report["cases"].append(item)
            for key in total:
                total[key] += item["actual_counts"][key]
                require(total[key] <= COUNTS[key], "evaluation counter cap exceeded")
            current_counted = True
            environment.close()
            environment = None
        contract_api().validate_sources(release)
        require(training_identity(release_path, release, arm, checkpoint) == identity, "training artifacts changed during evaluation")
        report["inputs_unchanged"] = True
        report["zero_delay_nominal_pass_count"] = sum(item["nominal_zero_delay_pass"] is True for item in report["cases"])
        complete = (len(report["cases"]) == 3 and all(valid_case_completion(item) for item in report["cases"])
                    and all(0 < total[key] <= cap for key, cap in COUNTS.items()))
        report["full_horizons_complete"] = (complete and counts_match(total, COUNTS)
            and all(item["done_reason"] == "HORIZON_12S" for item in report["cases"]))
        require(clock() < deadline, "evaluation wall budget exceeded")
        report["status"] = "COMPLETE" if complete else "INCOMPLETE"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if constructing:
            unknown_counts = True
            report["cleanup_complete"] = False
        if environment is not None and not current_counted:
            try:
                partial = environment.counters()
                for key in total:
                    require(type(partial[key]) is int and partial[key] >= 0, "unavailable partial counter")
                    total[key] += partial[key]
            except (ValueError, KeyError, TypeError, RuntimeError, OSError, AttributeError):
                unknown_counts = True
        if not isinstance(exc, Exception):
            raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except (ValueError, RuntimeError, OSError) as exc:
                report["cleanup_complete"] = False
                report["error"] = f"cleanup {type(exc).__name__}: {exc}"
                report["status"] = "INCOMPLETE"
        report["actual_counts"] = None if unknown_counts else total
        report["elapsed_s"] = clock() - started
        write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "velocity"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(args.release, args.arm, args.checkpoint, args.output)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
