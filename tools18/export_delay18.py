#!/usr/bin/env python3
"""Export and numerically verify only a completed, newly fine-tuned arm18."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("export_zero", "export_random", "cpp_zero", "cpp_random")
VERSIONS = {"python", "torch", "sb3", "numpy", "gymnasium", "mujoco"}
ZERO = {"policy_transitions": 0, "native_steps": 0, "pd_updates": 0}
LIMITS = {"sb3_vs_torch": 1e-6, "torch_vs_ort": 1e-5}
BINARY, SCHEMA = "runtime03/build/d1_runtime03", "runtime03/schema03d.json"


def require(value, message):
    if not value:
        raise ValueError(message)


def common():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("tools18.common18")


def helpers():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (importlib.import_module("tools17.export_completed17"),
            importlib.import_module("tools17.verify_cpp17"))


def training_identity(api, plan, plan_sha, arm):
    checkpoint, result = api.completed_training(plan, arm)
    checkpoint = Path(checkpoint)
    expected = Path(api.ROOT)/plan["output_root"]/("train_"+arm)/"policy.zip"
    require(checkpoint.resolve() == expected.absolute() and not checkpoint.is_symlink(),
            "only this arm's completed final policy.zip may be exported")
    protocol_path, result_path = checkpoint.parent/"protocol.json", checkpoint.parent/"result.json"
    require(all(path.is_file() and not path.is_symlink() for path in (protocol_path, result_path)),
            "training protocol/result must be regular files")
    protocol = api.read_json(protocol_path)
    require(api.read_json(result_path) == result, "training receipt differs from stored bytes")
    mode = "zero" if arm == "zero" else "uniform_0_3"
    identity = {"plan_sha256": plan_sha, "stage": "train_"+arm, "mode": mode,
                "seed": plan["seed"], "envs": plan["envs"], "source_identity": plan["source_identity"]}
    require(all(item.get(key) == value for item in (protocol, result) for key, value in identity.items()),
            "new training protocol/result identity differs")
    require(protocol.get("format") == "d1-delay18-training-protocol-v1"
            and result.get("format") == "d1-delay18-training-result-v1"
            and protocol.get("planned_new_transitions") == 1048576
            and protocol.get("base_checkpoint") == plan["base_checkpoint"]
            and protocol.get("base_protocol") == plan["base_protocol"]
            and protocol.get("warm_start") is True and protocol.get("reset_num_timesteps") is False
            and protocol.get("teacher_policy_used") is False,
            "new training provenance differs")
    require(result.get("status") == "COMPLETE" and result.get("cleanup_complete") is True
            and result.get("inputs_unchanged") is True and result.get("error") is None
            and result.get("eligible_for_evaluation") is True and result.get("aux_coef") == 0.
            and result.get("initial_num_timesteps") == 1048576
            and result.get("new_trained_transitions") == 1048576
            and result.get("final_num_timesteps") == 2097152
            and result.get("base_checkpoint_sha256") == plan["base_checkpoint"]["sha256"]
            and result.get("checkpoint_sha256") == api.sha(checkpoint)
            and result.get("protocol_sha256") == api.sha(protocol_path), "incomplete new training arm")
    versions = protocol.get("versions")
    require(isinstance(versions, dict) and set(versions) == VERSIONS
            and all(isinstance(value, str) and value for value in versions.values())
            and versions["sb3"] == "2.7.0", "missing or unsupported training runtime versions")
    bound = {"checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": api.sha(checkpoint),
             "training_protocol_sha256": api.sha(protocol_path), "training_result_sha256": api.sha(result_path)}
    return checkpoint, protocol, result, bound


def check_model(model, protocol, versions):
    require(type(model.num_timesteps) is int and model.num_timesteps == 2097152
            and type(model.aux_coef) is float and model.aux_coef == 0.
            and model.seed == 181 and model.n_envs == 4 and str(model.device) == "cpu"
            and model.get_env() is None, "loaded model is not the completed new fine-tuned arm")
    hp = protocol["hyperparameters"]
    for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "ent_coef", "vf_coef",
                "max_grad_norm", "learning_rate"):
        require(getattr(model, key) == hp[key], "loaded PPO setting differs: "+key)
    require(model.clip_range(1.) == model.clip_range(0.) == hp["clip_range"]
            and model.normalize_advantage is True and model.clip_range_vf is None
            and model.target_kl is None and model.use_sde is False, "loaded PPO support boundary differs")
    require(versions == protocol["versions"], "export runtime versions differ from new training")


def inspect_export(api, plan, plan_sha, arm, training):
    numerical, cpp = helpers()
    directory = Path(api.ROOT)/plan["output_root"]/("export_"+arm)
    report = api.read_json(directory/"report.json")
    require(report.get("format") == "d1-delay18-onnx-export-v1" and report.get("status") == "PASS"
            and report.get("error") is None and report.get("stage") == "export_"+arm
            and report.get("arm") == arm and report.get("mode") == ("zero" if arm == "zero" else "uniform_0_3")
            and report.get("plan_sha256") == plan_sha
            and report.get("source_identity") == plan["source_identity"]
            and report.get("training_identity") == training
            and report.get("checkpoint_sha256") == training["checkpoint_sha256"]
            and report.get("actual_counts") == ZERO and report.get("cleanup_complete") is True,
            "export is not this completed arm and source")
    for name, key in (("policy.onnx", "onnx_sha256"), ("golden_fixtures.json", "fixtures_sha256")):
        path = directory/name
        require(path.is_file() and not path.is_symlink() and api.sha(path) == report.get(key),
                "export artifact changed: "+name)
    require(report.get("absolute_limits") == LIMITS and report.get("onnx_ir_version") == 7
            and report.get("onnx_opset") == 11 and report.get("onnx_node_counts", {}).get("Gemm") == 6,
            "export full graph or parity settings changed")
    # Reuse the exact numerical/serialized-action gate, without its old17 identity gate.
    fixtures = api.read_json(directory/"golden_fixtures.json")
    numerical.cpp_fixtures({"onnx_node_counts": report["onnx_node_counts"],
                           "onnx_signature": report["onnx_signature"], "parity": report["parity"],
                           "rows": fixtures["fixtures"], "fixture_seed": fixtures["seed"]})
    return directory, report, cpp.check_fixtures(fixtures)


def run(plan_path, stage, output, *, api=None, model_loader=None, numerical_exporter=None, cpp_runner=None):
    require(stage in STAGES, "unknown numerical stage")
    api = api or common()
    plan_path, output = Path(plan_path).resolve(), Path(output).absolute()
    plan = api.load_plan(plan_path)
    expected = Path(api.ROOT)/plan["output_root"]/stage
    cursor = output
    while cursor != cursor.parent:
        require(not cursor.is_symlink(), "output path contains symlink")
        cursor = cursor.parent
    require(output.resolve() == expected.resolve(), "output outside the fixed new stage")
    require(not output.exists(), "stage already exists; no retry or overwrite")
    api.validate_launch(plan, plan_path, stage, output)
    plan_sha = api.sha(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    kind, arm = stage.split("_", 1)
    report = {"format": "d1-delay18-onnx-export-v1" if kind == "export" else "d1-delay18-cpp-parity-v1",
              "status": "FAIL", "stage": stage, "arm": arm,
              "mode": "zero" if arm == "zero" else "uniform_0_3", "error": None,
              "source_identity": plan["source_identity"], "plan_sha256": plan_sha,
              "checkpoint_sha256": None, "onnx_sha256": None, "fixtures_sha256": None,
              "actual_counts": dict(ZERO), "cleanup_complete": True, "inputs_unchanged": False,
              "checkpoint_loaded_by_exporter": False, "cpp_process_attempted": False,
              "training_started": False, "physics_steps_executed": 0, "robot_models_constructed": 0,
              "qualification_eligible": False, "release_ready": False, "cxx_ort_1_10_status": "NOT_TESTED"}
    try:
        checkpoint, protocol, result, training = training_identity(api, plan, plan_sha, arm)
        report.update(checkpoint_sha256=training["checkpoint_sha256"], training_identity=training)

        def unchanged():
            require(api.sha(plan_path) == plan_sha and api.load_plan(plan_path) == plan,
                    "frozen new plan or sources changed")
            require(training_identity(api, plan, plan_sha, arm)[3] == training,
                    "completed new training bytes changed")

        numerical, cpp = helpers()
        if kind == "export":
            model, versions = (model_loader or numerical.load_model)(checkpoint)
            report["checkpoint_loaded_by_exporter"] = True
            check_model(model, protocol, versions)
            unchanged()
            values = (numerical_exporter or numerical.export_numerical)(model, output)
            fixtures = numerical.cpp_fixtures(values)
            api.atomic_json(output/"golden_fixtures.json", fixtures)
            unchanged()
            report.update(onnx_sha256=api.sha(output/"policy.onnx"),
                          fixtures_sha256=api.sha(output/"golden_fixtures.json"),
                          absolute_limits=dict(LIMITS), onnx_ir_version=7, onnx_opset=11,
                          versions={**versions, **values["numerical_versions"]},
                          history_order="oldest_to_newest_previous_only",
                          full_actor_path="obs+history -> encoder -> current57+estimated3 -> actor -> clip",
                          **{key: values[key] for key in ("onnx_node_counts", "onnx_signature", "parity",
                                                         "policy_state_sha256", "numerical_helper_sha256")})
            metadata = dict(report, status="PASS", inputs_unchanged=True,
                            artifact_role="completed_delay18_policy", warm_start=True,
                            random_initialization=False, base_checkpoint=plan["base_checkpoint"],
                            training_seed=181, initial_num_timesteps=1048576,
                            new_trained_transitions=1048576, final_num_timesteps=2097152)
            api.atomic_json(output/"metadata.json", metadata)
            report["metadata_sha256"] = api.sha(output/"metadata.json")
        else:
            export_dir, exported, fixtures = inspect_export(api, plan, plan_sha, arm, training)
            old_release = api.verify_inputs(plan)[0]
            binary, schema = Path(api.ROOT)/BINARY, Path(api.ROOT)/SCHEMA
            identities = {"policy_sha256": api.sha(export_dir/"policy.onnx"),
                          "fixture_sha256": api.sha(export_dir/"golden_fixtures.json"),
                          "runtime_binary_sha256": api.sha(binary), "schema_sha256": api.sha(schema)}
            require(all(old_release["sources_sha256"].get(name) == identities[key]
                        for name, key in ((BINARY, "runtime_binary_sha256"), (SCHEMA, "schema_sha256"))),
                    "frozen C++ binary or schema changed")
            report.update(identities=identities, onnx_sha256=identities["policy_sha256"],
                          fixtures_sha256=identities["fixture_sha256"],
                          runtime_binary_sha256=identities["runtime_binary_sha256"],
                          schema_sha256=identities["schema_sha256"], export_report_sha256=api.sha(export_dir/"report.json"))
            command = [str(binary), "--schema", str(schema), "--onnx-golden", str(export_dir/"golden_fixtures.json"),
                       "--policy", str(export_dir/"policy.onnx"), "--expected-policy-sha256", identities["policy_sha256"],
                       "--output", str(output/"cpp_native")]
            libraries = f'/opt/ros/humble/lib:{Path(api.ROOT)/"install/mujoco/lib"}'
            if os.environ.get("LD_LIBRARY_PATH"):
                libraries += ":"+os.environ["LD_LIBRARY_PATH"]
            env = dict(os.environ, LD_LIBRARY_PATH=libraries, OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            report.update(command=command, execution_timeout_s=60)
            # A started native child needs its structured zero-work proof; an
            # absent or invalid receipt must never be treated as measured zero.
            report.update(cpp_process_attempted=True, actual_counts=None, cleanup_complete=False,
                          physics_steps_executed=None, robot_models_constructed=None)
            execution = (cpp_runner or cpp.invoke_cpp)(command, cwd=api.ROOT, env=env,
                        stdout_path=output/"stdout.txt", stderr_path=output/"stderr.txt", timeout_s=60)
            report.update(cpp_execution=execution, cleanup_complete=execution.get("cleanup_complete") is True)
            require(execution.get("process_started") is True and execution.get("returncode") == 0
                    and execution.get("timed_out") is False and report["cleanup_complete"]
                    and not execution.get("cleanup_signals") and execution.get("error") is None,
                    "C++ process failed or required cleanup")
            native_path = output/"cpp_native/report.json"
            maximum = cpp.check_native(api.read_json(native_path), fixtures, identities)
            unchanged()
            current = inspect_export(api, plan, plan_sha, arm, training)[1]
            require(current == exported and api.sha(export_dir/"report.json") == report["export_report_sha256"]
                    and api.sha(binary) == identities["runtime_binary_sha256"]
                    and api.sha(schema) == identities["schema_sha256"], "C++ inputs changed during verification")
            report.update(cpp_native_report_sha256=api.sha(native_path), comparisons=64,
                          onnx_inference_calls=64, max_absolute_error=maximum,
                          absolute_tolerance=1e-5, cxx_ort_1_10_status="PASS", actual_counts=dict(ZERO),
                          physics_steps_executed=0, robot_models_constructed=0)
        unchanged()
        report.update(status="PASS", inputs_unchanged=True)
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        api.atomic_json(output/"report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        value = run(args.plan, args.stage, args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"status": "REFUSED_BEFORE_NUMERICS", "error": str(exc)}))
        return 2
    print(json.dumps(value, allow_nan=False))
    return 0 if value["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
