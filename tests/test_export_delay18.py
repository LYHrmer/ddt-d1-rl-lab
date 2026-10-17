"""Only synthetic bytes, fake neural models and fake C++ reports are used."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/"tools18/export_delay18.py"
spec = importlib.util.spec_from_file_location("tested_delay_export18", ENTRY)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+"\n")


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path/"synthetic_repository"
    root.mkdir()
    source = root/"source.py"
    source.write_text("# fake source proof\n")
    proof = {"git_commit": "a"*40, "sources_sha256": {"source.py": sha(source)}}
    plan_path = root/"evidence/local_optimization18/plan_01.json"
    plan = {"source_identity": proof, "output_root": "evidence/local_optimization18/execution_01",
            "seed": 181, "envs": 4,
            "base_checkpoint": {"path": str(tmp_path/"old_control.zip"), "sha256": "b"*64,
                                "trained_transitions": 1048576},
            "base_protocol": {"path": str(tmp_path/"old_protocol.json"), "sha256": "d"*64}}
    put(plan_path, plan)
    execution = root/plan["output_root"]
    versions = {key: "synthetic" for key in export.VERSIONS}
    versions["sb3"] = "2.7.0"
    hp = {"aux_coef": 0., "device": "cpu", "cpu_threads": 1, "learning_rate": .0003,
          "n_steps": 1024, "batch_size": 256, "n_epochs": 4, "gamma": .99,
          "gae_lambda": .95, "clip_range": .2, "ent_coef": 0., "vf_coef": .5,
          "max_grad_norm": .5, "log_std_init": -1.}
    for arm in ("zero", "random"):
        directory = execution/("train_"+arm)
        directory.mkdir(parents=True)
        (directory/"policy.zip").write_bytes(b"synthetic bytes, never neural weights: "+arm.encode())
        identity = {"plan_sha256": sha(plan_path), "source_identity": proof, "stage": "train_"+arm,
                    "mode": "zero" if arm == "zero" else "uniform_0_3", "seed": 181, "envs": 4}
        protocol = dict(identity, format="d1-delay18-training-protocol-v1", planned_new_transitions=1048576,
                        base_checkpoint=plan["base_checkpoint"], base_protocol=plan["base_protocol"],
                        warm_start=True, reset_num_timesteps=False, teacher_policy_used=False,
                        versions=versions, hyperparameters=hp)
        put(directory/"protocol.json", protocol)
        result = dict(identity, format="d1-delay18-training-result-v1", status="COMPLETE", aux_coef=0.,
                      cleanup_complete=True, inputs_unchanged=True, error=None, eligible_for_evaluation=True,
                      initial_num_timesteps=1048576, new_trained_transitions=1048576, final_num_timesteps=2097152,
                      base_checkpoint_sha256=plan["base_checkpoint"]["sha256"], checkpoint_sha256=sha(directory/"policy.zip"),
                      protocol_sha256=sha(directory/"protocol.json"))
        put(directory/"result.json", result)
    for name in (export.BINARY, export.SCHEMA):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen fake binary or schema: "+name.encode())
    old = {"sources_sha256": {name: sha(root/name) for name in (export.BINARY, export.SCHEMA)}}
    launches = []

    class Api:
        ROOT = root
        sha = staticmethod(sha)
        read_json = staticmethod(read)
        atomic_json = staticmethod(put)

        def load_plan(self, path):
            assert Path(path) == plan_path
            if sha(source) != proof["sources_sha256"]["source.py"]:
                raise ValueError("source changed")
            return read(path)

        def validate_launch(self, supplied, path, stage, output):
            assert supplied == plan and Path(path) == plan_path
            assert output == execution/stage and not output.exists()
            launches.append(stage)

        def completed_training(self, supplied, arm):
            assert supplied == plan
            directory = execution/("train_"+arm)
            return directory/"policy.zip", read(directory/"result.json")

        def verify_inputs(self, supplied):
            assert supplied == plan
            return old, {}, {}, None

    attrs = {key: hp[key] for key in ("n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
                                     "ent_coef", "vf_coef", "max_grad_norm", "learning_rate")}
    model = SimpleNamespace(num_timesteps=2097152, aux_coef=0., seed=181, n_envs=4, device="cpu",
                            get_env=lambda: None, clip_range=lambda _: .2, normalize_advantage=True,
                            clip_range_vf=None, target_kl=None, use_sde=False, **attrs)
    return SimpleNamespace(root=root, execution=execution, plan=plan, plan_path=plan_path, source=source,
                           api=Api(), model=model, versions=versions, launches=launches, cpp_calls=[])


def fake_numerical(model, output):
    (output/"policy.onnx").write_bytes(b"not an ONNX model; no inference is performed")
    return {"onnx_node_counts": {"Gemm": 6, "Concat": 2, "Clip": 1, "Tanh": 4},
            "onnx_signature": {"inputs": [{"name": "obs", "elem_type": 1, "shape": [1, 57]},
                                          {"name": "history", "elem_type": 1, "shape": [1, 10, 57]}],
                               "outputs": [{"name": "actions", "elem_type": 1, "shape": [1, 16]}]},
            "parity": {"fixtures": 64, "prediction_batch_size": 1,
                       "target_actor_max_abs_change": 0., "target_critic_max_abs_change": 0.,
                       "sb3_predict_vs_torch_max_abs_error": 0., "torch_vs_python_ort_max_abs_error": 0.},
            "rows": [{"index": i, "obs": [i/100.]*57,
                      "history_oldest_to_newest_previous_only": [[(i+j)/100.]*57 for j in range(10)],
                      "sb3_actions": [0.]*16, "torch_actions": [0.]*16, "python_ort_actions": [0.]*16}
                     for i in range(64)],
            "fixture_seed": 20261017, "policy_state_sha256": "c"*64, "numerical_helper_sha256": "e"*64,
            "numerical_versions": {"onnx": "synthetic", "python_ort": "synthetic"}}


def fake_cpp(fixture, *, native_change=None, execution_change=None, hook=None):
    def invoke(command, **kwargs):
        fixture.cpp_calls.append((command, kwargs))
        assert kwargs["timeout_s"] == 60 and kwargs["cwd"] == fixture.root
        assert kwargs["env"]["OMP_NUM_THREADS"] == kwargs["env"]["OPENBLAS_NUM_THREADS"] == '1'
        policy = Path(command[command.index("--policy")+1])
        fixtures = Path(command[command.index("--onnx-golden")+1])
        output = Path(command[command.index("--output")+1])
        native = {"status": "PASS", "policy_sha256": sha(policy), "fixture_sha256": sha(fixtures),
                  "runtime_binary_sha256": sha(fixture.root/export.BINARY),
                  "schema_sha256": sha(fixture.root/export.SCHEMA),
                  "onnxruntime_version": "1.10.0", "mujoco_version": "3.3.0",
                  "physics_steps": 0, "native_steps_attempted": 0, "native_steps_completed": 0,
                  "pd_updates": 0, "policy_updates": 0, "onnx_inference_calls": 64,
                  "onnx_signature": {"inputs": {"obs": [1, 57], "history": [1, 10, 57]},
                                     "outputs": {"actions": [1, 16]}},
                  "comparisons": [{"index": i, "actions": [0.]*16, "max_absolute_error": 0., "passed": True}
                                  for i in range(64)], "max_absolute_error": 0., "absolute_tolerance": 1e-5}
        if native_change:
            native_change(native)
        put(output/"report.json", native)
        if hook:
            hook()
        execution = {"process_started": True, "returncode": 0, "timed_out": False,
                     "cleanup_complete": True, "cleanup_signals": [], "error": None}
        if execution_change:
            execution_change(execution)
        return execution
    return invoke


def run(fixture, stage="export_zero", *, loader=None, numerical=None, cpp=None):
    return export.run(fixture.plan_path, stage, fixture.execution/stage, api=fixture.api,
                      model_loader=loader or (lambda _: (fixture.model, fixture.versions)),
                      numerical_exporter=numerical or fake_numerical,
                      cpp_runner=cpp or fake_cpp(fixture))


def test_import_does_not_load_any_neural_or_simulator_library():
    code = (f'import importlib.util,sys;s=importlib.util.spec_from_file_location("light18",{str(ENTRY)!r});'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not {"numpy","torch","mujoco","onnxruntime","onnx","stable_baselines3"}&sys.modules.keys()')
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arm", ["zero", "random"])
def test_new_completed_arm_exports_full_graph_with_correct_provenance(fixture, arm):
    report = run(fixture, "export_"+arm)
    assert report["status"] == "PASS" and report["error"] is None
    assert report["mode"] == ("zero" if arm == "zero" else "uniform_0_3")
    assert report["plan_sha256"] == sha(fixture.plan_path) and report["source_identity"] == fixture.plan["source_identity"]
    assert report["checkpoint_sha256"] == sha(fixture.execution/("train_"+arm)/"policy.zip")
    assert report["training_identity"]["checkpoint_path"].endswith("train_"+arm+"/policy.zip")
    assert report["actual_counts"] == export.ZERO and report["cleanup_complete"] is report["inputs_unchanged"] is True
    assert report["cxx_ort_1_10_status"] == "NOT_TESTED" and report["qualification_eligible"] is False
    assert report["onnx_node_counts"]["Gemm"] == 6 and report["onnx_ir_version"] == 7 and report["onnx_opset"] == 11
    directory = fixture.execution/("export_"+arm)
    metadata = read(directory/"metadata.json")
    assert metadata["status"] == "PASS" and metadata["warm_start"] is True
    assert metadata["new_trained_transitions"] == 1048576 and metadata["final_num_timesteps"] == 2097152
    assert sha(directory/"metadata.json") == report["metadata_sha256"]
    assert read(directory/"report.json") == report


@pytest.mark.parametrize("field,value", [("status", "INCOMPLETE"), ("stage", "preflight_zero"),
    ("final_num_timesteps", 1048576), ("initial_num_timesteps", 0), ("mode", "uniform_0_3"),
    ("new_trained_transitions", 32768), ("source_identity", {}), ("plan_sha256", "f"*64),
    ("eligible_for_evaluation", False), ("aux_coef", 1.), ("cleanup_complete", False)])
def test_partial_old_or_foreign_training_cannot_load_a_model(fixture, field, value):
    path = fixture.execution/"train_zero/result.json"
    result = read(path); result[field] = value; put(path, result)
    def forbidden(_):
        raise AssertionError("foreign provenance reached checkpoint loading")
    report = run(fixture, loader=forbidden)
    assert report["status"] == "FAIL" and "AssertionError" not in report["error"]
    assert not (fixture.execution/"export_zero/policy.onnx").exists()


@pytest.mark.parametrize("attr,value", [("num_timesteps", 1048576), ("aux_coef", 1.), ("seed", 101),
                                       ("n_envs", 2), ("n_steps", 2048), ("normalize_advantage", False)])
def test_actual_loaded_model_support_boundary_is_checked(fixture, attr, value):
    setattr(fixture.model, attr, value)
    report = run(fixture)
    assert report["status"] == "FAIL" and not (fixture.execution/"export_zero/policy.onnx").exists()


def test_runtime_version_difference_is_not_accepted(fixture):
    report = run(fixture, loader=lambda _: (fixture.model, dict(fixture.versions, torch="different")))
    assert report["status"] == "FAIL" and "versions differ" in report["error"]


@pytest.mark.parametrize("fault", ["source", "checkpoint", "protocol", "result", "plan"])
def test_mutation_during_export_cannot_produce_pass(fixture, fault):
    paths = {"source": fixture.source, "checkpoint": fixture.execution/"train_zero/policy.zip",
             "protocol": fixture.execution/"train_zero/protocol.json", "result": fixture.execution/"train_zero/result.json",
             "plan": fixture.plan_path}
    def mutate(model, output):
        value = fake_numerical(model, output)
        path = paths[fault]; path.write_bytes(path.read_bytes()+b" ")
        return value
    assert run(fixture, numerical=mutate)["status"] == "FAIL"


@pytest.mark.parametrize("fault", ["encoder", "signature", "fixtures", "leak", "tolerance", "serialized"])
def test_full_graph_and_saved_fixture_checks_are_not_bypassed(fixture, fault):
    def broken(model, output):
        value = fake_numerical(model, output)
        if fault == "encoder": value["onnx_node_counts"]["Gemm"] = 3
        elif fault == "signature": value["onnx_signature"]["inputs"][0]["shape"] = [1, 627]
        elif fault == "fixtures": value["rows"].pop()
        elif fault == "leak": value["parity"]["target_actor_max_abs_change"] = .1
        elif fault == "tolerance": value["parity"]["torch_vs_python_ort_max_abs_error"] = 2e-5
        else: value["rows"][0]["python_ort_actions"][0] = 2e-5
        return value
    assert run(fixture, numerical=broken)["status"] == "FAIL"


@pytest.mark.parametrize("arm", ["zero", "random"])
def test_cpp_uses_this_new_arm_and_preserves_export_not_tested_receipt(fixture, arm, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/test/path")
    exported = run(fixture, "export_"+arm)
    report = run(fixture, "cpp_"+arm)
    assert report["status"] == "PASS" and report["error"] is None
    assert report["cxx_ort_1_10_status"] == "PASS" and report["comparisons"] == report["onnx_inference_calls"] == 64
    assert report["max_absolute_error"] == 0. and report["absolute_tolerance"] == 1e-5
    assert report["checkpoint_sha256"] == exported["checkpoint_sha256"]
    assert report["training_identity"] == exported["training_identity"]
    assert report["onnx_sha256"] == exported["onnx_sha256"]
    assert report["actual_counts"] == export.ZERO and report["cleanup_complete"] is True
    assert read(fixture.execution/("export_"+arm)/"report.json")["cxx_ort_1_10_status"] == "NOT_TESTED"
    assert fixture.cpp_calls[0][1]["env"]["LD_LIBRARY_PATH"].endswith(":/existing/test/path")


@pytest.mark.parametrize("field,value", [("plan_sha256", "f"*64), ("checkpoint_sha256", "f"*64),
                                       ("source_identity", {}), ("stage", "export_random")])
def test_cpp_rejects_export_from_another_plan_or_arm_before_process(fixture, field, value):
    run(fixture)
    path = fixture.execution/"export_zero/report.json"
    report = read(path); report[field] = value; put(path, report)
    assert run(fixture, "cpp_zero")["status"] == "FAIL" and fixture.cpp_calls == []


@pytest.mark.parametrize("name", [export.BINARY, export.SCHEMA])
def test_cpp_frozen_runtime_inputs_must_match_baseline_closure(fixture, name):
    run(fixture)
    (fixture.root/name).write_bytes(b"different frozen binary/schema")
    report = run(fixture, "cpp_zero")
    assert report["status"] == "FAIL" and "frozen C++" in report["error"] and fixture.cpp_calls == []


@pytest.mark.parametrize("fault", ["duplicate", "native", "version", "arithmetic", "identity", "count"])
def test_native_cpp_report_is_independently_validated(fixture, fault):
    run(fixture)
    def change(native):
        if fault == "duplicate": native["comparisons"][-1]["index"] = 0
        elif fault == "native": native["physics_steps"] = 1
        elif fault == "version": native["onnxruntime_version"] = "new"
        elif fault == "arithmetic": native["comparisons"][0]["actions"][0] = 2e-5
        elif fault == "identity": native["policy_sha256"] = "f"*64
        else: native["onnx_inference_calls"] = 63
    report = run(fixture, "cpp_zero", cpp=fake_cpp(fixture, native_change=change))
    assert report["status"] == "FAIL" and report["actual_counts"] is None


@pytest.mark.parametrize("field,value", [("returncode", 1), ("timed_out", True), ("cleanup_complete", False),
                                       ("cleanup_signals", ["SIGINT"]), ("error", "interrupted")])
def test_cpp_timeout_interrupt_or_cleanup_failure_never_passes(fixture, field, value):
    run(fixture)
    report = run(fixture, "cpp_zero", cpp=fake_cpp(fixture, execution_change=lambda item: item.update({field: value})))
    assert report["status"] == "FAIL" and report["actual_counts"] is None
    if field == "cleanup_complete":
        assert report["cleanup_complete"] is False


def test_cpp_rechecks_inputs_after_child_exit(fixture):
    run(fixture)
    path = fixture.execution/"export_zero/policy.onnx"
    report = run(fixture, "cpp_zero", cpp=fake_cpp(fixture, hook=lambda: path.write_bytes(b"changed")))
    assert report["status"] == "FAIL" and "artifact changed" in report["error"]


def test_output_and_live_supervision_are_required_before_loading(fixture):
    output = fixture.execution/"export_zero"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        run(fixture)
    assert fixture.launches == []
    with pytest.raises(ValueError, match="fixed new stage"):
        export.run(fixture.plan_path, "export_zero", fixture.root/"elsewhere", api=fixture.api)


def test_invalid_live_launch_never_claims_directory(fixture, monkeypatch):
    def refused(*_):
        raise ValueError("not the live nonce stage")
    monkeypatch.setattr(fixture.api, "validate_launch", refused)
    with pytest.raises(ValueError, match="live nonce"):
        run(fixture)
    assert not (fixture.execution/"export_zero").exists()


def test_old_control_loader_is_never_called_by_new_pipeline(fixture, monkeypatch):
    _, cpp = export.helpers()
    monkeypatch.setattr(cpp, "training_identity", lambda *_: (_ for _ in ()).throw(AssertionError("old control gate")))
    assert run(fixture)["status"] == "PASS"
    assert run(fixture, "cpp_zero")["status"] == "PASS"
