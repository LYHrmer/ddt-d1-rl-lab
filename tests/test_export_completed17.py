"""Synthetic provenance/models only; the actual completed checkpoint is never read."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/"tools17/export_completed17.py"
spec = importlib.util.spec_from_file_location("tested_completed_export17", ENTRY)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+"\n")


@pytest.fixture
def bundle(tmp_path):
    training = tmp_path/"synthetic_training"
    output = tmp_path/"new_export"
    run = training/"evidence/execution_01"
    train = run/"train_control"
    train.mkdir(parents=True)
    checkpoint = train/"policy.zip"
    checkpoint.write_bytes(b"synthetic placeholder, not neural weights")
    source = tmp_path/"source.py"; source.write_text("# synthetic source proof\n")
    release = {"output_root": "evidence/execution_01"}
    hyper = {"n_steps": 1024, "batch_size": 256, "n_epochs": 4, "gamma": .99,
             "gae_lambda": .95, "ent_coef": 0., "vf_coef": .5, "max_grad_norm": .5}
    versions = {key: "synthetic" for key in exporter.VERSION_KEYS}
    protocol = {"arm": "control", "stage": "train", "seed": 101, "envs": 4,
                "planned_transitions": 1048576, "release_sha256": "1"*64,
                "config": {"version": "synthetic"}, "hyperparameters": hyper, "versions": versions}
    put(train/"protocol.json", protocol)
    result = {"arm": "control", "stage": "train", "trained_transitions": 1048576, "aux_coef": 0.,
              "status": "COMPLETE", "cleanup_complete": True, "inputs_unchanged": True,
              "eligible_for_evaluation": True, "error": None,
              "checkpoint_sha256": exporter.sha(checkpoint), "protocol_sha256": exporter.sha(train/"protocol.json")}
    put(train/"result.json", result)
    put(run/"report.json", {"status": "COMPLETE", "stages": {str(i): {"status": "COMPLETE"} for i in range(7)}})

    class Api:
        def load_completed_control(self, root):
            assert root == training
            p, r = (json.loads((train/name).read_text()) for name in ("protocol.json", "result.json"))
            require_complete = json.loads((run/"report.json").read_text())
            if any(item["status"] != "COMPLETE" for item in require_complete["stages"].values()):
                raise ValueError("seven completed stages are required")
            return release, p, r, checkpoint

        def source_identity(self):
            return {"git_commit": "a"*40, "sources_sha256": {"source.py": exporter.sha(source)}}

        def check_sources(self, proof):
            if exporter.sha(source) != proof["sources_sha256"]["source.py"]:
                raise ValueError("source changed")

        atomic_json = staticmethod(put)

    model = SimpleNamespace(num_timesteps=1048576, aux_coef=0., seed=101, n_envs=4,
                            device=SimpleNamespace(type="cpu"), get_env=lambda: None, **hyper)
    return SimpleNamespace(training=training, output=output, run=run, train=train, checkpoint=checkpoint,
                           source=source, release=release, protocol=protocol, result=result,
                           versions=versions, model=model, api=Api())


def numerical(model, output):
    (output/"policy.onnx").write_bytes(b"synthetic ONNX placeholder; no inference")
    return {"onnx_node_counts": {"Gemm": 6, "Concat": 2, "Tanh": 4, "Clip": 1},
            "onnx_signature": {"inputs": [{"name": "obs", "elem_type": 1, "shape": [1, 57]},
                                          {"name": "history", "elem_type": 1, "shape": [1, 10, 57]}],
                               "outputs": [{"name": "actions", "elem_type": 1, "shape": [1, 16]}]},
            "parity": {"fixtures": 64, "prediction_batch_size": 1, "target_actor_max_abs_change": 0.,
                       "target_critic_max_abs_change": 0., "sb3_predict_vs_torch_max_abs_error": 0.,
                       "torch_vs_python_ort_max_abs_error": 0.},
            "policy_state_sha256": "2"*64, "numerical_helper_sha256": "3"*64,
            "numerical_versions": {"onnx": "synthetic", "python_ort": "synthetic"}, "fixture_seed": 20261017,
            "rows": [{"index": i, "obs": [i/100.]*57,
                      "history_oldest_to_newest_previous_only": [[(i+j)/100.]*57 for j in range(10)],
                      "sb3_actions": [0.]*16, "torch_actions": [0.]*16, "python_ort_actions": [0.]*16}
                     for i in range(64)]}


def run(bundle, *, loader=None, export=None):
    return exporter.run(bundle.training, bundle.output, api=bundle.api,
                        model_loader=loader or (lambda checkpoint: (bundle.model, bundle.versions)),
                        numerical_exporter=export or numerical)


def test_import_is_stdlib_only():
    code = ('import importlib.util,sys; '
            f's=importlib.util.spec_from_file_location("light_export17",{str(ENTRY)!r}); '
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert not {"torch","onnx","onnxruntime","mujoco","stable_baselines3"}&sys.modules.keys()')
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_plan_reads_provenance_without_checkpoint_loading_or_output_creation(bundle):
    value = exporter.plan(bundle.training, bundle.output, api=bundle.api)
    assert value["status"] == "PLAN_ONLY" and value["checkpoint_loaded_by_exporter"] is False
    assert not bundle.output.exists() and value["physics_steps_executed"] == 0


def test_complete_export_binds_bytes_and_does_not_claim_cpp_or_qualification(bundle):
    report = run(bundle)
    assert report["status"] == "PASS" and report["error"] is None
    assert report["checkpoint_sha256"] == exporter.sha(bundle.checkpoint)
    assert report["onnx_sha256"] == exporter.sha(bundle.output/"policy.onnx")
    assert report["fixtures_sha256"] == exporter.sha(bundle.output/"golden_fixtures.json")
    assert report["metadata_sha256"] == exporter.sha(bundle.output/"metadata.json")
    assert report["completed_checkpoint_provenance_checked"] is True
    assert report["checkpoint_loaded_by_exporter"] is True
    assert report["qualification_eligible"] is report["release_ready"] is False
    assert report["cxx_ort_1_10_status"] == "NOT_TESTED"
    assert report["absolute_limits"] == {"sb3_vs_torch": 1e-6, "torch_vs_ort": 1e-5}
    fixtures = json.loads((bundle.output/"golden_fixtures.json").read_text())
    assert fixtures["count"] == len(fixtures["fixtures"]) == 64
    for row in fixtures["fixtures"]:
        assert row["history_oldest_to_newest"] == row["history_oldest_to_newest_previous_only"]
        assert len(row["obs"]) == 57 and len(row["history_oldest_to_newest"]) == 10
        assert row["python_ort_actions"] == row["torch_actions"]
    assert json.loads((bundle.output/"report.json").read_text()) == report


@pytest.mark.parametrize("fault", ["partial", "velocity", "steps", "cleanup", "seventh_stage"])
def test_incomplete_or_foreign_training_refused_before_model_load(bundle, fault):
    if fault == "seventh_stage":
        report = json.loads((bundle.run/"report.json").read_text()); report["stages"]["6"]["status"] = "FAIL"
        put(bundle.run/"report.json", report)
    else:
        result = json.loads((bundle.train/"result.json").read_text())
        if fault == "partial": result["status"] = "INCOMPLETE"
        if fault == "velocity": result["arm"] = "velocity"
        if fault == "steps": result["trained_transitions"] -= 1
        if fault == "cleanup": result["cleanup_complete"] = False
        put(bundle.train/"result.json", result)
    def forbidden(checkpoint): raise AssertionError("invalid provenance must not load any model")
    with pytest.raises(ValueError): run(bundle, loader=forbidden)
    report = json.loads((bundle.output/"report.json").read_text())
    assert report["status"] == "FAIL" and report["checkpoint_loaded_by_exporter"] is False


@pytest.mark.parametrize("attribute,value", [("aux_coef", 1.), ("num_timesteps", 32768),
                                           ("seed", 102), ("n_envs", 2), ("n_steps", 2048)])
def test_loaded_checkpoint_attributes_must_match_completed_control(bundle, attribute, value):
    setattr(bundle.model, attribute, value)
    with pytest.raises(ValueError, match="loaded checkpoint"):
        run(bundle)
    assert not (bundle.output/"policy.onnx").exists()


def test_different_export_runtime_version_is_rejected(bundle):
    versions = dict(bundle.versions, torch="other stack")
    with pytest.raises(ValueError, match="versions differ"):
        run(bundle, loader=lambda checkpoint: (bundle.model, versions))


@pytest.mark.parametrize("fault", ["source", "checkpoint", "protocol", "execution_report"])
def test_artifact_or_source_mutation_during_export_never_produces_pass(bundle, fault):
    def changing(model, output):
        value = numerical(model, output)
        path = {"source": bundle.source, "checkpoint": bundle.checkpoint,
                "protocol": bundle.train/"protocol.json", "execution_report": bundle.run/"report.json"}[fault]
        path.write_bytes(path.read_bytes()+b" ")
        return value
    with pytest.raises(ValueError): run(bundle, export=changing)
    assert json.loads((bundle.output/"report.json").read_text())["status"] == "FAIL"
    assert not (bundle.output/"metadata.json").exists()


@pytest.mark.parametrize("fault", ["encoder", "signature", "count", "target_leak", "sb3_tolerance", "ort_tolerance", "serialized_actions", "nonfinite"])
def test_graph_fixture_and_tolerance_mismatches_fail_closed(bundle, fault):
    def incorrect(model, output):
        value = numerical(model, output)
        if fault == "encoder": value["onnx_node_counts"]["Gemm"] = 3
        if fault == "signature": value["onnx_signature"]["inputs"][0]["shape"] = [1, 627]
        if fault == "count": value["rows"].pop()
        if fault == "target_leak": value["parity"]["target_actor_max_abs_change"] = .01
        if fault == "sb3_tolerance": value["parity"]["sb3_predict_vs_torch_max_abs_error"] = 2e-6
        if fault == "ort_tolerance": value["parity"]["torch_vs_python_ort_max_abs_error"] = 2e-5
        if fault == "serialized_actions": value["rows"][0]["python_ort_actions"][0] = 2e-5
        if fault == "nonfinite": value["rows"][0]["obs"][0] = float("nan")
        return value
    with pytest.raises(ValueError): run(bundle, export=incorrect)
    assert json.loads((bundle.output/"report.json").read_text())["status"] == "FAIL"


def test_refuses_output_overwrite_and_writes_inside_historical_bundle(bundle):
    bundle.output.mkdir(); (bundle.output/"keep").write_text("original")
    with pytest.raises(ValueError, match="overwrite"): run(bundle)
    assert (bundle.output/"keep").read_text() == "original"
    with pytest.raises(ValueError, match="outside"):
        exporter.plan(bundle.training, bundle.training/"new_export", api=bundle.api)


def test_exception_after_valid_provenance_persists_failed_report(bundle):
    def failure(checkpoint): raise RuntimeError("synthetic load failure")
    with pytest.raises(RuntimeError): run(bundle, loader=failure)
    report = json.loads((bundle.output/"report.json").read_text())
    assert report["status"] == "FAIL" and "synthetic load failure" in report["error"]
    assert report["completed_checkpoint_provenance_checked"] is True


def test_cli_plan_does_not_construct_model(bundle, monkeypatch, capsys):
    monkeypatch.setattr(exporter, "provenance_api", lambda: bundle.api)
    assert exporter.main(["--training-root", str(bundle.training), "--output", str(bundle.output), "--plan"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    assert not bundle.output.exists()


def test_numerical_random_network_full_graph_cpp_fixture_compatibility(tmp_path):
    # Optional dependency test: random NN only, not the actual completed policy.
    for path in (ROOT/".python-deps", ROOT/"training"):
        if str(path) not in sys.path: sys.path.insert(0, str(path))
    for name in ("torch", "onnx", "onnxruntime", "stable_baselines3", "gymnasium"):
        pytest.importorskip(name)
    helper = exporter.numerical_helpers()
    output = tmp_path/"synthetic_network"; output.mkdir()
    policy = helper.make_random_policy()
    before = helper.policy_state_sha256(policy)
    numerical_result = exporter.export_numerical(SimpleNamespace(policy=policy), output)
    fixtures = exporter.cpp_fixtures(numerical_result)
    assert numerical_result["onnx_node_counts"]["Gemm"] == 6
    assert fixtures["count"] == 64 and helper.policy_state_sha256(policy) == before
    assert numerical_result["parity"]["sb3_predict_vs_torch_max_abs_error"] <= 1e-6
    assert numerical_result["parity"]["torch_vs_python_ort_max_abs_error"] <= 1e-5
    assert all(row["history_oldest_to_newest"] == row["history_oldest_to_newest_previous_only"]
               for row in fixtures["fixtures"])
