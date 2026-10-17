#!/usr/bin/env python3
"""Export only the published, completed comparison16 control checkpoint.

--plan reads provenance without loading a neural checkpoint or creating output.
Export constructs the saved SB3 neural model, never an environment or robot.
Numerical parity is separate from subsequent C++ parity and formal qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "d1-completed-control-onnx-export-v17"
LIMITS = {"sb3_vs_torch": 1e-6, "torch_vs_ort": 1e-5}
VERSION_KEYS = {"python", "torch", "sb3", "numpy", "gymnasium", "mujoco"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_api():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools17 import provenance17
    return provenance17


def numerical_helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools15 import export_velocity15
    return export_velocity15


def destination(training_root, output):
    original = Path(output).absolute()
    cursor = original
    while cursor != cursor.parent:
        require(not cursor.is_symlink(), "output path cannot contain symlinks")
        cursor = cursor.parent
    output, training_root = original.resolve(), Path(training_root).resolve()
    require(not output.is_relative_to(training_root), "export must be outside the historical training bundle")
    require(not output.exists(), "refusing to overwrite existing export output")
    return training_root, output


def training_identity(training_root, release, protocol, result, checkpoint):
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint == training_root/release["output_root"]/"train_control/policy.zip",
            "only the completed formal control policy.zip is exportable")
    require(protocol["arm"] == result["arm"] == "control" and protocol["stage"] == result["stage"] == "train"
            and protocol["planned_transitions"] == result["trained_transitions"] == 1048576
            and result["status"] == "COMPLETE" and result["cleanup_complete"] is True
            and result["inputs_unchanged"] is True and result["eligible_for_evaluation"] is True
            and result["error"] is None and result["aux_coef"] == 0., "incomplete or foreign control training")
    protocol_path, result_path = checkpoint.parent/"protocol.json", checkpoint.parent/"result.json"
    require(json.loads(protocol_path.read_text()) == protocol and json.loads(result_path.read_text()) == result,
            "returned training records differ from stored records")
    require(result["checkpoint_sha256"] == sha(checkpoint)
            and result["protocol_sha256"] == sha(protocol_path), "training checkpoint/protocol SHA mismatch")
    return {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
            "training_protocol_sha256": sha(protocol_path), "training_result_sha256": sha(result_path),
            "release_sha256": protocol["release_sha256"],
            "execution_report_sha256": sha(training_root/release["output_root"]/"report.json")}


def inspect_inputs(training_root, api):
    training_root = Path(training_root).resolve()
    release, protocol, result, checkpoint = api.load_completed_control(training_root)
    identity = training_identity(training_root, release, protocol, result, checkpoint)
    proof = api.source_identity()
    api.check_sources(proof)
    return release, protocol, result, Path(checkpoint), identity, proof


def plan(training_root, output, *, api=None):
    training_root, output = destination(training_root, output)
    api = api or provenance_api()
    _, _, _, _, identity, proof = inspect_inputs(training_root, api)
    return {"format": FORMAT, "status": "PLAN_ONLY", "training_root": str(training_root),
            "output": str(output), "training_identity": identity, "source_identity": proof,
            "checkpoint_loaded_by_exporter": False, "robot_models_constructed": 0,
            "physics_steps_executed": 0, "output_created": False}


def load_model(checkpoint):
    import platform
    helper = numerical_helpers()
    lib = helper.libraries()
    import gymnasium
    import mujoco
    from d1_rl.velocity_policy15 import AuxiliaryVelocityPPO
    model = AuxiliaryVelocityPPO.load(str(checkpoint), device="cpu", env=None)
    require(type(model) is AuxiliaryVelocityPPO, "unsupported checkpoint algorithm")
    versions = {"python": platform.python_version(), "torch": lib.torch.__version__,
                "sb3": lib.sb3.__version__, "numpy": lib.np.__version__,
                "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__}
    return model, versions


def check_loaded_model(model, protocol, versions):
    require(type(model.num_timesteps) is int and model.num_timesteps == 1048576
            and type(model.aux_coef) is float and model.aux_coef == 0.
            and model.seed == protocol["seed"] and model.n_envs == protocol["envs"]
            and model.device.type == "cpu" and model.get_env() is None,
            "loaded checkpoint attributes do not match completed control training")
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm"):
        require(getattr(model, key) == protocol["hyperparameters"][key], f"loaded checkpoint hyperparameter changed: {key}")
    require(isinstance(versions, dict) and set(versions) == VERSION_KEYS
            and versions == protocol["versions"], "export runtime versions differ from training")


def export_numerical(model, output):
    """Reuse frozen15 complete-graph helpers, with trained-checkpoint provenance above."""
    helper = numerical_helpers()
    lib = helper.libraries()
    torch = lib.torch
    policy = model.policy
    helper.check_policy(policy)
    state = helper.policy_state_sha256(policy)
    previous_mode = policy.training
    try:
        policy.set_training_mode(False)
        actor = helper.actor_module(policy).eval()
        path = output/"policy.onnx"
        with torch.no_grad():
            torch.onnx.export(actor, (torch.zeros((1, 57)), torch.zeros((1, 10, 57))), str(path),
                              input_names=["obs", "history"], output_names=["actions"], opset_version=11,
                              dynamo=False, do_constant_folding=True)
        graph = lib.onnx.load(str(path))
        graph.ir_version = 7
        lib.onnx.save(graph, str(path))
        counts, signature = helper.check_onnx(path)
        rows, errors = helper.golden_parity(actor, policy, path)
        require(helper.policy_state_sha256(policy) == state, "policy state changed during export")
        return {"onnx_node_counts": counts, "onnx_signature": signature, "parity": errors,
                "policy_state_sha256": state, "rows": rows,
                "numerical_versions": {"onnx": lib.onnx.__version__, "python_ort": lib.ort.__version__},
                "fixture_seed": helper.FIXTURE_SEED, "numerical_helper_sha256": sha(Path(helper.__file__))}
    finally:
        policy.set_training_mode(previous_mode)


def cpp_fixtures(numerical):
    counts, signature, parity = (numerical[key] for key in ("onnx_node_counts", "onnx_signature", "parity"))
    expected = {"inputs": [{"name": "obs", "elem_type": 1, "shape": [1, 57]},
                            {"name": "history", "elem_type": 1, "shape": [1, 10, 57]}],
                "outputs": [{"name": "actions", "elem_type": 1, "shape": [1, 16]}]}
    require(counts.get("Gemm") == 6 and signature == expected, "export omitted the complete encoder or changed its signature")
    require(parity["fixtures"] == 64 and parity["prediction_batch_size"] == 1
            and parity["target_actor_max_abs_change"] == parity["target_critic_max_abs_change"] == 0.,
            "wrong parity coverage or target leakage")
    for key, maximum in (("sb3_predict_vs_torch_max_abs_error", 1e-6), ("torch_vs_python_ort_max_abs_error", 1e-5)):
        error = parity[key]
        require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= maximum,
                "numerical parity tolerance exceeded")

    def vector(value, length):
        require(isinstance(value, list) and len(value) == length
                and all(type(item) in (int, float) and math.isfinite(item) for item in value), "invalid golden vector")
    rows = numerical["rows"]
    require(isinstance(rows, list) and len(rows) == 64, "exactly 64 golden fixtures are required")
    converted = []
    for index, row in enumerate(rows):
        require(type(row["index"]) is int and row["index"] == index, "golden fixture index differs")
        vector(row["obs"], 57)
        history = row["history_oldest_to_newest_previous_only"]
        require(isinstance(history, list) and len(history) == 10, "invalid golden history")
        for frame in history:
            vector(frame, 57)
        for key in ("sb3_actions", "torch_actions", "python_ort_actions"):
            vector(row[key], 16)
        require(max(abs(a-b) for a, b in zip(row["sb3_actions"], row["torch_actions"])) <= 1e-6
                and max(abs(a-b) for a, b in zip(row["torch_actions"], row["python_ort_actions"])) <= 1e-5,
                "serialized golden actions exceed parity tolerance")
        # runtime03's immutable C++ parser reads this shorter key, unchanged values.
        converted.append(dict(row, history_oldest_to_newest=history))
    return {"format": "d1-onnx-two-input-golden-64-v1", "seed": numerical["fixture_seed"], "count": 64,
            "input_order": ["obs", "history"], "history_order": "oldest_to_newest_previous_only", "fixtures": converted}


def run(training_root, output, *, api=None, model_loader=None, numerical_exporter=None):
    training_root, output = destination(training_root, output)
    api = api or provenance_api()
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": FORMAT, "status": "FAIL", "training_root": str(training_root), "error": None,
              "checkpoint_sha256": None, "onnx_sha256": None, "fixtures_sha256": None,
              "source_identity": None, "completed_checkpoint_provenance_checked": False,
              "checkpoint_loaded_by_exporter": False, "training_started": False,
              "robot_models_constructed": 0, "physics_steps_executed": 0,
              "cxx_ort_1_10_status": "NOT_TESTED", "qualification_eligible": False,
              "release_ready": False, "absolute_limits": LIMITS,
              "history_order": "oldest_to_newest_previous_only",
              "full_actor_path": "obs+history -> encoder -> current57+estimated_velocity3 -> actor -> action_head -> clip"}
    try:
        release, protocol, result, checkpoint, identity, proof = inspect_inputs(training_root, api)
        report.update(training_identity=identity, source_identity=proof,
                      checkpoint_sha256=identity["checkpoint_sha256"], completed_checkpoint_provenance_checked=True)

        def unchanged():
            api.check_sources(proof)
            a, b, c, d = api.load_completed_control(training_root)
            require(training_identity(training_root, a, b, c, d) == identity,
                    "completed training artifacts changed during export")
        model, versions = (model_loader or load_model)(checkpoint)
        report["checkpoint_loaded_by_exporter"] = True
        check_loaded_model(model, protocol, versions)
        unchanged()
        numerical = (numerical_exporter or export_numerical)(model, output)
        fixtures = cpp_fixtures(numerical)
        fixture_path = output/"golden_fixtures.json"
        api.atomic_json(fixture_path, fixtures)
        unchanged()
        report.update(status="PASS", onnx_sha256=sha(output/"policy.onnx"), fixtures_sha256=sha(fixture_path),
                      onnx_ir_version=7, onnx_opset=11, versions={**versions, **numerical["numerical_versions"]},
                      **{key: numerical[key] for key in ("onnx_node_counts", "onnx_signature", "parity",
                                                       "policy_state_sha256", "numerical_helper_sha256")})
        metadata = dict(report, artifact_role="completed_control_policy", training_arm="control",
                        training_kind="comparison16_train", training_seed=protocol["seed"],
                        trained_transitions=result["trained_transitions"], random_initialization=True,
                        teacher_policy_used=False, warm_start=False, vecnormalize_used=False,
                        config=protocol["config"], policy_operation="deterministic Gaussian mean, clipped to [-1,1]",
                        sha256={"checkpoint": report["checkpoint_sha256"], "policy_onnx": report["onnx_sha256"],
                                "training_protocol": identity["training_protocol_sha256"],
                                "training_result": identity["training_result_sha256"],
                                "golden_fixtures": report["fixtures_sha256"]})
        api.atomic_json(output/"metadata.json", metadata)
        report["metadata_sha256"] = sha(output/"metadata.json")
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        api.atomic_json(output/"report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", action="store_true", help="read provenance only; no checkpoint load or output creation")
    args = parser.parse_args(argv)
    try:
        report = plan(args.training_root, args.output) if args.plan else run(args.training_root, args.output)
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "physics_steps_executed": 0}))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
