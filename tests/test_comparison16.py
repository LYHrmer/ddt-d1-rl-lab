"""Synthetic checkpoint receipts/fake envs/traces only: no Torch or physics."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools16 import compare_results16 as comparison
from tools16 import evaluate_comparison16 as evaluation


def save(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False))


class FakeEnv:
    def __init__(self, name, sequence, *, factor=1., early=None, truncated=False, fail=None, tilt=0., illegal=False, clock=None):
        self.name, self.sequence = name, sequence
        self.factor, self.early, self.fail, self.tilt, self.illegal = factor, early, fail, tilt, illegal
        self.truncated = truncated
        self.tick, self.x, self.closed = 0, 0., False
        self.free_qpos = 0
        self.data = SimpleNamespace(qpos=np.zeros(2))
        self.clock = clock

    def counters(self):
        return {"policy_transitions": self.tick, "native_steps": self.tick*10, "pd_updates": self.tick*5}

    def reset(self, *, seed):
        assert seed == 7001
        return np.zeros(627, np.float32), {"seed": seed}

    def step(self, action):
        t = self.tick*.02
        self.tick += 1
        if self.clock is not None:
            self.clock[0] += 1
        if self.fail == self.tick:
            raise RuntimeError("synthetic step failure after counted transition")
        command = max((item for item in self.sequence if item["start_s"] <= t), key=lambda item: item["start_s"])
        vx, wz = command["vx"]+.04*self.factor, command["wz"]+.04*self.factor
        self.x += vx*.02
        info = {"command_during_interval": [command["vx"], 0., command["wz"]],
                "body_vx_m_s": vx, "body_vy_m_s": 0., "body_wz_rad_s": wz,
                "base_x_m": self.x, "base_y_m": 0., "base_height_m": .5,
                "base_roll_rad": self.tilt, "base_pitch_rad": 0., "base_yaw_rad": 0., "tilt_rad": self.tilt,
                "fallen": False, "illegal_body_contact": self.illegal, "tau_saturation_count": 0,
                "applied_action": action.copy()}
        ended = self.early == self.tick
        return np.zeros(627, np.float32), .1, ended and not self.truncated, ended and self.truncated, info

    def close(self):
        self.closed = True


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    release_path = tmp_path / "release.json"
    release = {"status": "AUTHORIZED_FROZEN", "output_root": str(tmp_path / "run"), "envs": 2, "seed": 101,
               "config": "training/configs/pilot03d.json", "arms": {"control": {"aux_coef": 0.}, "velocity": {"aux_coef": 1.}},
               "stages": {"train": {"transitions": 1048576}, "eval": {"transitions": 1800, "max_wall_s": 300}},
               "comparison": {"required_relative_reduction": .05},
               "sources_sha256": {name: evaluation.sha256(ROOT / name) for name in
                                  ("tools/evaluate_d1_ppo.py", "training/configs/pilot03d.json")}}
    save(release_path, release)
    versions = {"python": "synthetic", "torch": "synthetic", "sb3": "2.7.0", "numpy": np.__version__,
                "gymnasium": "synthetic", "mujoco": "synthetic"}
    def validate_sources(value):
        for name, digest in value["sources_sha256"].items():
            evaluation.require(evaluation.sha256(ROOT / name) == digest, "source changed")
    def load_release(path, require_authorized=True):
        value = evaluation.read_json(path)
        evaluation.require(require_authorized and value["status"] == "AUTHORIZED_FROZEN", "not authorized")
        return value
    api = SimpleNamespace(load_release=load_release, validate_sources=validate_sources)
    monkeypatch.setattr(evaluation, "contract_api", lambda: api)
    monkeypatch.setattr(evaluation, "validate_launch", lambda *args: None)
    for arm in ("control", "velocity"):
        folder = Path(release["output_root"]) / f"train_{arm}"
        folder.mkdir(parents=True)
        (folder / "policy.zip").write_bytes(f"fake complete {arm} checkpoint".encode())
        protocol = {"format": "d1-comparison16-training-protocol-v1", "release_sha256": evaluation.sha256(release_path),
                    "arm": arm, "stage": "train", "envs": 2, "seed": 101, "planned_transitions": 1048576,
                    "planned_native_steps": 10485760, "inputs_sha256": release["sources_sha256"],
                    "config": evaluation.read_json(ROOT / release["config"]), "versions": versions, "random_initialization": True,
                    "warm_start": False, "teacher_policy_used": False,
                    "hyperparameters": {"aux_coef": release["arms"][arm]["aux_coef"], "device": "cpu", "cpu_threads": 1,
                                        "n_steps": 2048, "batch_size": 256, "n_epochs": 4, "ent_coef": 0.}}
        save(folder / "protocol.json", protocol)
        result = {"format": "d1-comparison16-training-result-v1", "status": "COMPLETE", "release_sha256": evaluation.sha256(release_path),
                  "protocol_sha256": evaluation.sha256(folder / "protocol.json"), "arm": arm, "stage": "train", "envs": 2,
                  "seed": 101, "aux_coef": release["arms"][arm]["aux_coef"], "eligible_for_evaluation": True,
                  "trained_transitions": 1048576, "actual_counts": {key: value for key, value in
                    zip(evaluation.COUNTS, (1048576, 10485760, 5242880), strict=True)},
                  "per_worker_counts": [{"policy_transitions": 524288, "native_steps": 5242880, "pd_updates": 2621440}]*2,
                  "inputs_unchanged": True, "cleanup_complete": True, "error": None,
                  "checkpoint_sha256": evaluation.sha256(folder / "policy.zip")}
        save(folder / "result.json", result)
    backend = {"kind": "sb3_cpu_deterministic", "algorithm": "d1_rl.velocity_policy15.AuxiliaryVelocityPPO",
               "device": "cpu", "cpu_threads": 1, "target_for_control": False, **versions}
    created = []
    def factory(*, factors=None, options=None):
        factors = factors or {"control": 1., "velocity": .5}
        options = options or {}
        def make_runtime(release, arm, checkpoint):
            def make_env(name, sequence):
                env = FakeEnv(name, sequence, factor=factors[arm], **options.get((arm, name), {}))
                created.append(env)
                return env
            return SimpleNamespace(make_env=make_env, predict=lambda actor: np.zeros(16, np.float32), backend=dict(backend))
        return make_runtime
    def run(arm, **kwargs):
        return evaluation.run_evaluation(release_path, arm, Path(release["output_root"]) / f"train_{arm}/policy.zip",
                                         Path(release["output_root"]) / f"eval_{arm}",
                                         runtime_factory=factory(**kwargs), clock=lambda: 0.)
    return SimpleNamespace(path=release_path, release=release, root=Path(release["output_root"]), run=run,
                           factory=factory, created=created, api=api)


def compare(fixture):
    return comparison.compare_results(fixture.path, fixture.root / "eval_control/report.json",
                                      fixture.root / "eval_velocity/report.json", fixture.root / "comparison.json")


def test_import_and_help_never_load_numerical_libraries():
    code = ("import sys; from tools16 import evaluate_comparison16,compare_results16; "
            "assert not {'torch','mujoco','stable_baselines3','onnxruntime'} & sys.modules.keys()")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr


def test_complete_paired_evaluation_preserves_original_metrics_and_is_exploratory(prepared):
    control, velocity = prepared.run("control"), prepared.run("velocity")
    assert control["status"] == velocity["status"] == "COMPLETE"
    assert control["full_horizons_complete"] is velocity["full_horizons_complete"] is True
    assert control["actual_counts"] == velocity["actual_counts"] == evaluation.COUNTS
    assert all(env.closed for env in prepared.created)
    result = compare(prepared)
    assert result["status"] == "COMPLETE" and result["verdict"] == "BOTH_IMPROVED_EXPLORATORY"
    assert result["relative_reductions"]["tracking"] == pytest.approx(.5)
    assert result["relative_reductions"]["parking"] == pytest.approx(.5)
    assert not result["statistical_significance_claimed"] and not result["generalization_claimed"]
    for arm in ("control", "velocity"):
        assert result["evaluations"][arm]["cases"][0]["J_components"]["fixed_early_failure_penalty"] == 0.
        assert "peak_abs_roll_or_pitch_rad" in result["evaluations"][arm]["cases"][0]


@pytest.mark.parametrize("fault", ["partial", "preflight", "short_counts", "worker_counts", "checkpoint", "protocol", "sources", "teacher", "aux"])
def test_ineligible_training_rejected_before_runtime(prepared, fault):
    folder = prepared.root / "train_control"
    protocol, result = evaluation.read_json(folder / "protocol.json"), evaluation.read_json(folder / "result.json")
    if fault == "partial": result["status"] = "PARTIAL"
    if fault == "preflight": protocol["stage"] = "preflight"
    if fault == "short_counts": result["actual_counts"]["policy_transitions"] -= 1
    if fault == "worker_counts": result["per_worker_counts"] = []
    if fault == "checkpoint": (folder / "policy.zip").write_bytes(b"changed")
    if fault == "protocol": result["protocol_sha256"] = "0"*64
    if fault == "sources": protocol["inputs_sha256"] = {}
    if fault == "teacher": protocol["teacher_policy_used"] = True
    if fault == "aux": protocol["hyperparameters"]["aux_coef"] = 1.
    save(folder / "protocol.json", protocol)
    save(folder / "result.json", result)
    with pytest.raises(ValueError): prepared.run("control")
    assert not prepared.created and not (prepared.root / "eval_control").exists()


def test_partial_filename_and_prepared_release_cannot_execute(prepared):
    partial = prepared.root / "train_control/partial_policy.zip"
    partial.write_bytes(b"partial")
    with pytest.raises(ValueError, match="formal train"):
        evaluation.training_identity(prepared.path, prepared.release, "control", partial)
    release = copy.deepcopy(prepared.release); release["status"] = "PREPARED_NOT_AUTHORIZED"
    save(prepared.path, release)
    with pytest.raises(ValueError, match="not authorized"): prepared.run("control")
    assert not prepared.created


def test_model_predict_receives_actor_and_zero_dummy_label_only():
    original = np.arange(627, dtype=np.float32)
    class Predictor:
        def predict(self, value, *, deterministic):
            assert deterministic and value["actor"] is original
            np.testing.assert_array_equal(value["velocity_target"], np.zeros(3, np.float32))
            return np.full(16, 2., np.float32), None
    predict = evaluation.numerical_predictor(Predictor(), np)
    np.testing.assert_array_equal(predict(original), np.ones(16, np.float32))
    with pytest.raises(ValueError): predict(original.astype(np.float64))


def test_short_easy_episode_retains_failure_penalty_but_cannot_improve(prepared):
    prepared.run("control")
    short = prepared.run("velocity", options={("velocity", "forward_stop"): {"early": 10}})
    assert short["status"] == "COMPLETE" and short["full_horizons_complete"] is False
    assert short["actual_counts"]["policy_transitions"] == 1210
    assert short["cases"][0]["J_components"]["fixed_early_failure_penalty"] == 100.
    result = compare(prepared)
    assert result["verdict"] == "INCOMPLETE" and result["scores"] is result["relative_reductions"] is None


@pytest.mark.parametrize("truncated", [False, True])
def test_control_task_failure_keeps_three_cases_and_allows_velocity_pair_without_improvement(prepared, truncated):
    control = prepared.run("control", options={("control", "forward_stop"): {"early": 10, "truncated": truncated}})
    assert control["status"] == "COMPLETE" and control["full_horizons_complete"] is False
    assert control["error"] is None and len(control["cases"]) == 3
    assert control["actual_counts"] == {"policy_transitions": 1210, "native_steps": 12100, "pd_updates": 6050}
    first = control["cases"][0]
    assert first["policy_transitions"] == 10 and first["failure"] is True
    assert first["done_reason"] == ("ENV_TRUNCATED_BEFORE_12S" if truncated else "TERMINATED")
    assert first["J_components"]["fixed_early_failure_penalty"] == 100.
    velocity = prepared.run("velocity")
    assert velocity["status"] == "COMPLETE" and velocity["full_horizons_complete"] is True
    assert len(prepared.created) == 6 and all(env.closed for env in prepared.created)
    result = compare(prepared)
    assert result["status"] == result["verdict"] == "INCOMPLETE"
    assert result["scores"] is result["relative_reductions"] is None
    assert len(result["evaluations"]["control"]["cases"]) == len(result["evaluations"]["velocity"]["cases"]) == 3
    assert result["evaluations"]["control"]["actual_counts"]["policy_transitions"] == 1210


def test_exception_reports_real_partial_counts_and_closes_environment(prepared):
    report = prepared.run("control", options={("control", "forward_stop"): {"fail": 3}})
    assert report["status"] == "INCOMPLETE" and report["actual_counts"] == {
        "policy_transitions": 3, "native_steps": 30, "pd_updates": 15}
    assert report["error"] and all(env.closed for env in prepared.created)


def test_unavailable_partial_counters_are_null_not_zero(prepared, monkeypatch):
    monkeypatch.setattr(FakeEnv, "counters", lambda self: (_ for _ in ()).throw(RuntimeError("missing")))
    report = prepared.run("control")
    assert report["status"] == "INCOMPLETE" and report["actual_counts"] is None


def test_constructor_failure_does_not_claim_known_zero_or_complete_cleanup(prepared):
    def broken_runtime(release, arm, checkpoint):
        rt = prepared.factory()(release, arm, checkpoint)
        def fail(*args):
            raise RuntimeError("constructor failed before returning ownership")
        rt.make_env = fail
        return rt
    report = evaluation.run_evaluation(prepared.path, "control", prepared.root / "train_control/policy.zip",
        prepared.root / "eval_control", runtime_factory=broken_runtime, clock=lambda: 0.)
    assert report["status"] == "INCOMPLETE" and report["actual_counts"] is None
    assert report["cleanup_complete"] is False


def test_training_runtime_version_mismatch_stops_before_environment(prepared):
    def mismatched_runtime(release, arm, checkpoint):
        rt = prepared.factory()(release, arm, checkpoint)
        rt.backend["torch"] = "other runtime"
        return rt
    report = evaluation.run_evaluation(prepared.path, "control", prepared.root / "train_control/policy.zip",
        prepared.root / "eval_control", runtime_factory=mismatched_runtime, clock=lambda: 0.)
    assert report["status"] == "INCOMPLETE" and "versions differ" in report["error"]
    assert not prepared.created


def test_post_evaluation_source_change_invalidates_full_counts(prepared, monkeypatch):
    calls = []
    def validate(release):
        calls.append(1)
        if len(calls) > 1: raise ValueError("source changed during evaluation")
    monkeypatch.setattr(prepared.api, "validate_sources", validate)
    report = prepared.run("control")
    assert report["status"] == "INCOMPLETE" and report["actual_counts"] == evaluation.COUNTS
    assert report["inputs_unchanged"] is False and "source changed" in report["error"]


@pytest.mark.parametrize("fault", ["missing_case", "metric", "clock", "command", "trace_bytes", "backend", "budget", "counts", "qualification"])
def test_false_comparability_is_rejected_and_never_reported_as_zero_improvement(prepared, fault):
    prepared.run("control"); prepared.run("velocity")
    path = prepared.root / "eval_velocity/report.json"
    report = evaluation.read_json(path)
    if fault == "missing_case": report["cases"].pop()
    if fault == "metric": report["cases"][0]["stable_vx_mae_m_s"] = 0.
    if fault == "counts": report["actual_counts"]["native_steps"] -= 10
    if fault == "qualification": report["zero_delay_nominal_pass_count"] = 3
    if fault == "trace_bytes":
        with (path.parent / "forward_stop.npz").open("ab") as stream: stream.write(b"changed")
    if fault in ("clock", "command"):
        trace_path = path.parent / "forward_stop.npz"
        with np.load(trace_path, allow_pickle=False) as loaded: trace = {key: loaded[key] for key in loaded.files}
        if fault == "clock":
            trace["policy_interval_start_s"][10] += .001
        else:
            trace["command_during_interval"][100, 0] = -.2
            helpers = evaluation.frozen_helpers(prepared.release)
            report["cases"][0].update(helpers.trace_metrics("forward_stop", 0, 7001, trace,
                                      trace["boundary_xy_m"], report["cases"][0]["done_reason"]))
        np.savez_compressed(trace_path, **trace)
        report["cases"][0]["trace_sha256"] = evaluation.sha256(trace_path)
    if fault in ("backend", "budget"):
        protocol_path = path.parent / "protocol.json"
        protocol = evaluation.read_json(protocol_path)
        if fault == "backend":
            report["backend"]["torch"] = "different"
            protocol["backend"] = report["backend"]
        else: protocol["counts"]["policy_transitions"] = 1700
        save(protocol_path, protocol)
        report["protocol_sha256"] = evaluation.sha256(protocol_path)
    save(path, report)
    result = compare(prepared)
    assert result["verdict"] == "INCOMPLETE" and result["reasons"]
    assert result["scores"] is result["relative_reductions"] is None
    if fault == "command": assert "trace commands differ" in " ".join(result["reasons"])


def test_both_five_percent_reductions_are_required(prepared):
    prepared.run("control"); prepared.run("velocity", factors={"control": 1., "velocity": .951})
    result = compare(prepared)
    assert result["verdict"] == "NO_DEMONSTRATED_BENEFIT"
    assert all(0 < value < .05 for value in result["relative_reductions"].values())


def test_better_scores_cannot_hide_lost_qualification(prepared):
    prepared.run("control", factors={"control": .2, "velocity": .1})
    prepared.run("velocity", factors={"control": .2, "velocity": .1},
                 options={("velocity", "turn_stop"): {"tilt": .3}})
    result = compare(prepared)
    assert result["verdict"] == "TRADEOFF"
    assert result["qualification_counts"] == {"control": 3, "velocity": 2}
    assert not result["gates"]["velocity_qualification_count_not_lower"]


def test_better_scores_cannot_hide_illegal_support(prepared):
    prepared.run("control"); prepared.run("velocity", options={("velocity", "turn_stop"): {"illegal": True}})
    result = compare(prepared)
    assert result["verdict"] == "TRADEOFF" and not result["gates"]["no_fall_or_illegal_support"]


def test_zero_control_score_is_no_demonstrated_reduction_not_divide_by_zero(prepared):
    prepared.run("control", factors={"control": 0., "velocity": 0.})
    prepared.run("velocity", factors={"control": 0., "velocity": 0.})
    result = compare(prepared)
    assert result["verdict"] == "NO_DEMONSTRATED_BENEFIT"
    assert result["relative_reductions"] == {"tracking": None, "parking": None}


def test_frozen_helper_source_identity_precedes_execution(prepared):
    changed = copy.deepcopy(prepared.release)
    changed["sources_sha256"]["tools/evaluate_d1_ppo.py"] = "0"*64
    with pytest.raises(ValueError, match="source SHA"):
        evaluation.frozen_helpers(changed)


def test_outputs_are_never_overwritten(prepared):
    prepared.run("control"); prepared.run("velocity")
    before = (prepared.root / "eval_control/report.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"): prepared.run("control")
    assert (prepared.root / "eval_control/report.json").read_bytes() == before
    compare(prepared)
    with pytest.raises(ValueError, match="overwrite"): compare(prepared)
