"""Only fake workers, fake models and clocks; no robot/model/physics imports."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools18"))
import train_delay18 as trainer  # noqa: E402

GIB = 2**30
INITIAL = 1048576


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class Vec:
    def __init__(self, envs, close_error=False):
        self.envs, self.steps, self.closed, self.close_error = envs, 0, False, close_error
        self.processes = []

    def close(self):
        self.closed = True
        if self.close_error:
            raise OSError("synthetic close failure")


class Model:
    def __init__(self, save_error=False):
        self.num_timesteps, self.save_error = INITIAL, save_error

    def save(self, path):
        if self.save_error:
            raise OSError("synthetic checkpoint failure")
        Path(path).write_bytes(f"synthetic:{self.num_timesteps}".encode())


class Runtime:
    def __init__(self, clock, fault=None):
        self.clock, self.fault = clock, fault
        self.prepared = self.spawn_calls = 0
        self.vec = self.model = None

    def prepare(self):
        self.prepared += 1
        self.versions = dict.fromkeys(("python", "torch", "sb3", "numpy", "gymnasium", "mujoco"), "fake")
        if self.fault == "versions":
            self.versions["torch"] = "different runtime"
        if self.fault == "import":
            raise ImportError("synthetic import failure")
        if self.fault == "slow_import":
            self.clock.now = 1000.

    def create_vec(self, plan, transitions, output, root, mode):
        self.spawn_calls += 1
        self.mode = mode
        if self.fault == "spawn":
            raise PermissionError("synthetic forkserver failure")
        self.vec = Vec(plan["envs"], self.fault == "close")
        if self.fault == "spawn_signal":
            signal.raise_signal(signal.SIGTERM)
        return self.vec

    def create_model(self, vec, plan, hyper, output):
        self.hyper = hyper
        self.model = Model(self.fault == "save")
        self.initial_identity = {"num_timesteps": INITIAL, "policy_state_sha256": "a"*64,
                                 "optimizer_state_sha256": "b"*64, "optimizer_restored": True, "updates": 1024}
        if self.fault == "wrong_initial":
            self.initial_identity["num_timesteps"] = 0
        if self.fault == "optimizer_missing":
            self.initial_identity["optimizer_restored"] = False
        return self.model

    def policy_sha(self, model):
        return f"{model.num_timesteps:064x}"

    def counters(self, vec):
        if self.fault == "counter_rpc":
            raise OSError("synthetic counter failure")
        result = [dict(trainer.expected_counts(vec.steps//vec.envs), resets=1,
                       forward_calls=vec.steps//vec.envs+3) for _ in range(vec.envs)]
        if self.fault == "counts":
            result[0]["native_steps"] += 1
        return result

    def delay_state(self, vec):
        if self.fault == "delay_rpc":
            raise OSError("synthetic delay RPC failure")
        rows = []
        for rank in range(vec.envs):
            delay = rank if self.mode == "uniform_0_3" else 0
            rows.append({"last_selection": {"mode": self.mode, "rank": rank, "selected_delay_ticks": delay},
                         "delay_counts": {str(index): int(index == delay) for index in range(4)}})
        if self.fault == "delays":
            rows[0]["delay_counts"]["3"] = True
        return rows

    def learn(self, model, transitions, watchdog):
        if self.fault == "signal":
            signal.raise_signal(signal.SIGINT)
        if self.fault == "wall":
            self.clock.now = 1000.
        if self.fault == "changed":
            (watchdog.output.parents[2]/"base.zip").write_bytes(b"changed baseline")
        if self.fault == "plan_changed":
            plan_path = watchdog.output.parents[2]/"plan.json"
            plan_path.write_text(plan_path.read_text()+"\n")
        if not watchdog.poll(force=True):
            return
        cap = 4096 if self.fault in ("partial", "absolute_callback") else transitions
        infos = [{"training_phase": "drive_forward", "reward_terms": {"contributions_per_tick": {"vx": .1}}}
                 for _ in range(self.vec.envs)]
        for _ in range(cap//4096):
            for _ in range(4096//self.vec.envs):
                self.vec.steps += self.vec.envs
                model.num_timesteps = INITIAL+self.vec.steps
                if not watchdog.on_step(infos, [.1]*self.vec.envs, self.vec.steps):
                    return
            self.clock.now += .1
            watchdog.on_rollout_end(model.num_timesteps if self.fault == "absolute_callback" else self.vec.steps)
        if self.fault == "late_wall":
            self.clock.now = 1000.


@pytest.fixture
def case(tmp_path):
    (tmp_path/"base.zip").write_bytes(b"synthetic completed checkpoint")
    (tmp_path/"base_protocol.json").write_text(json.dumps({"hyperparameters": trainer.HYPERPARAMETERS,
        "versions": dict.fromkeys(("python", "torch", "sb3", "numpy", "gymnasium", "mujoco"), "fake")}))
    (tmp_path/"base.json").write_text('{}')
    plan = {"envs": 4, "seed": 181, "config": "base.json", "output_root": "evidence/run",
            "stages": {name: {"mode": mode, "transitions": transitions,
                               "max_wall_s": 900 if name.startswith("preflight") else 5400}
                       for name, (mode, transitions) in trainer.STAGES.items()},
            "base_checkpoint": {"path": str(tmp_path/"base.zip"), "sha256": trainer.sha(tmp_path/"base.zip"),
                                "trained_transitions": INITIAL},
            "base_protocol": {"path": str(tmp_path/"base_protocol.json"),
                              "sha256": trainer.sha(tmp_path/"base_protocol.json")},
            "resources": {"start_min_effective_bytes": 6*GIB, "reserve_bytes": 3*GIB, "poll_s": 1.},
            "source_identity": {"git_commit": "fake", "sources_sha256": {}}}
    path = tmp_path/"plan.json"
    path.write_text(json.dumps(plan))

    def loader(path):
        value = json.loads(Path(path).read_text())
        if trainer.sha(value["base_checkpoint"]["path"]) != value["base_checkpoint"]["sha256"]:
            raise ValueError("baseline SHA changed")
        return value
    return tmp_path, path, plan, loader


def execute(case, stage="preflight_zero", *, fault=None, memory=None):
    root, path, plan, loader = case
    clock = Clock()
    runtime = Runtime(clock, fault)
    output = root/plan["output_root"]/stage
    result = trainer.run(path, stage, output, root=root, runtime=runtime, loader=loader,
        launch_validator=lambda *args: None, clock=clock,
        memory_reader=memory or (lambda: {"available_ram_bytes": 8*GIB,
                                         "cgroup_remaining_bytes": None, "effective_available_bytes": 7*GIB}))
    return result, runtime, output


@pytest.mark.parametrize("stage", tuple(trainer.STAGES))
def test_completed_stage_records_only_new_transitions_and_preserves_initial_identity(case, stage):
    result, runtime, output = execute(case, stage)
    mode, transitions = trainer.STAGES[stage]
    assert result["status"] == "COMPLETE" and result["error"] is None
    assert result["initial_num_timesteps"] == INITIAL and result["final_num_timesteps"] == INITIAL+transitions
    assert result["new_trained_transitions"] == transitions
    assert result["actual_counts"]["policy_transitions"] == transitions
    assert result["initial_state"]["optimizer_restored"] is True
    assert result["initial_state"]["optimizer_state_sha256"] == "b"*64
    assert result["cleanup_complete"] and result["inputs_unchanged"] and runtime.vec.closed
    assert result["eligible_for_evaluation"] == stage.startswith("train_")
    assert runtime.hyper == trainer.HYPERPARAMETERS and runtime.mode == mode
    assert result["checkpoint_sha256"] == trainer.sha(output/"policy.zip")
    assert result["protocol_sha256"] == trainer.sha(output/"protocol.json")
    protocol = json.loads((output/"protocol.json").read_text())
    assert protocol["reset_num_timesteps"] is False and protocol["warm_start"] is True
    assert result["source_identity"] == protocol["source_identity"]
    ledger = [json.loads(line) for line in (output/"ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["transitions"] == 4096 and ledger[-1]["transitions"] == transitions
    assert all(row["phase_rewards"]["drive_forward"]["ticks"] == 4096 for row in ledger)


def test_preflight_weights_are_not_the_formal_start_and_relative_prefix_is_identical(case):
    pre, _, pre_dir = execute(case)
    # A corrupt/discarded preflight file must never become the next stage's input.
    (pre_dir/"policy.zip").write_bytes(b"discarded preflight weight")
    formal, _, formal_dir = execute(case, "train_zero")
    assert pre["initial_state"] == formal["initial_state"]
    before = [json.loads(line)["deterministic"] for line in (pre_dir/"ledger.jsonl").read_text().splitlines()]
    after = [json.loads(line)["deterministic"] for line in (formal_dir/"ledger.jsonl").read_text().splitlines()[:8]]
    assert before == after and not formal["preflight_weights_used_for_training"]


@pytest.mark.parametrize("fault", ["import", "slow_import", "partial", "counts", "counter_rpc", "close", "save",
    "spawn", "changed", "plan_changed", "wall", "late_wall", "delay_rpc", "delays", "wrong_initial",
    "optimizer_missing", "absolute_callback"])
def test_failures_never_promote_partial_weights_and_leave_structured_receipt(case, fault):
    result, runtime, output = execute(case, fault=fault)
    assert result["status"] != "COMPLETE" and result["eligible_for_evaluation"] is False
    assert (output/"result.json").exists() and not (output/"policy.zip").exists()
    if runtime.model is not None and fault != "save":
        assert (output/"partial_policy.zip").exists()
    if fault == "spawn":
        assert result["actual_counts"] is None and result["cleanup_complete"] is False
        assert result["final_num_timesteps"] is None and result["new_trained_transitions"] is None
    if fault == "counter_rpc":
        assert result["actual_counts"] is result["worker_counts"] is None
    if fault in ("changed", "plan_changed"):
        assert result["inputs_unchanged"] is False
    if fault in ("import", "slow_import"):
        assert runtime.spawn_calls == 0


def test_effective_memory_drop_is_recorded_even_with_large_host_available(case):
    calls = 0

    def memory():
        nonlocal calls
        calls += 1
        effective = 7*GIB if calls <= 3 else 2*GIB
        return {"available_ram_bytes": 8*GIB, "cgroup_remaining_bytes": effective,
                "effective_available_bytes": effective}
    result, runtime, output = execute(case, memory=memory)
    assert result["status"] == "RAM_RESERVE_BELOW_3_GIB" and runtime.vec.closed
    assert result["new_trained_transitions"] == 0
    assert result["stop_sample"]["effective_available_bytes"] == 2*GIB
    assert result["stop_sample"]["available_ram_bytes"] == 8*GIB
    assert (output/"stop_sample.json").exists()


def test_low_start_ram_is_refused_without_output_import_or_spawn(case):
    root, path, plan, loader = case
    runtime = Runtime(Clock())
    output = root/plan["output_root"]/"preflight_zero"
    with pytest.raises(ValueError, match="NOT_READY"):
        trainer.run(path, "preflight_zero", output, root=root, runtime=runtime, loader=loader,
                    launch_validator=lambda *args: None,
                    memory_reader=lambda: {"effective_available_bytes": 5*GIB})
    assert not output.exists() and runtime.prepared == runtime.spawn_calls == 0


def test_runtime_version_drift_is_recorded_and_refused_before_worker_spawn(case):
    result, runtime, output = execute(case, fault="versions")
    assert result["status"] == "ERROR" and "runtime versions differ" in result["error"]
    assert runtime.spawn_calls == 0 and result["actual_counts"] is None
    assert result["versions"]["torch"] == "different runtime"
    assert json.loads((output/"protocol.json").read_text())["versions"] == result["versions"]


@pytest.mark.parametrize("fault,name", [("spawn_signal", "SIGTERM"), ("signal", "SIGINT")])
def test_signals_restore_handlers_and_close_only_created_workers(case, fault, name):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    result, runtime, _ = execute(case, fault=fault)
    assert result["status"] == name and result["cleanup_complete"] and runtime.vec.closed
    assert previous == {sig: signal.getsignal(sig) for sig in previous}


def test_missing_parent_ticket_refuses_before_output_or_heavy_import(case):
    root, path, plan, loader = case
    runtime = Runtime(Clock())

    def denied(*args):
        raise ValueError("missing live parent stage ticket")
    output = root/plan["output_root"]/"train_zero"
    with pytest.raises(ValueError, match="ticket"):
        trainer.run(path, "train_zero", output, root=root, runtime=runtime, loader=loader, launch_validator=denied)
    assert not output.exists() and runtime.prepared == runtime.spawn_calls == 0


@pytest.mark.parametrize("change", ["hyper", "checkpoint", "wrong_output", "exists", "symlink", "scope"])
def test_changed_baseline_scope_or_unsafe_output_refuses_before_prepare(case, change):
    root, path, plan, loader = case
    output = root/plan["output_root"]/"preflight_zero"
    if change == "hyper":
        protocol_path = Path(plan["base_protocol"]["path"])
        protocol_path.write_text(json.dumps({"hyperparameters": dict(trainer.HYPERPARAMETERS, n_steps=2048)}))
        plan["base_protocol"]["sha256"] = trainer.sha(protocol_path)
    elif change == "checkpoint":
        Path(plan["base_checkpoint"]["path"]).write_bytes(b"changed")
    elif change == "wrong_output":
        output = root/"wrong"
    elif change == "exists":
        output.mkdir(parents=True)
    elif change == "symlink":
        (root/"evidence").mkdir()
        (root/"other").mkdir()
        (root/"evidence/run").symlink_to(root/"other", target_is_directory=True)
    elif change == "scope":
        plan["stages"]["preflight_zero"]["transitions"] += 4096
    path.write_text(json.dumps(plan))
    runtime = Runtime(Clock())
    with pytest.raises(ValueError):
        trainer.run(path, "preflight_zero", output, root=root, runtime=runtime, loader=loader,
                    launch_validator=lambda *args: None)
    assert runtime.prepared == runtime.spawn_calls == 0


@pytest.mark.parametrize("bad", ["num_timesteps", "aux_coef", "seed", "n_envs", "learning_rate", "clip_range"])
def test_loaded_model_settings_cannot_silently_replace_the_baseline(bad):
    settings = {key: value for key, value in trainer.HYPERPARAMETERS.items()
                if key not in ("clip_range", "cpu_threads", "log_std_init")}
    settings.update(num_timesteps=INITIAL, clip_range=lambda _: .2, seed=181, n_envs=4,
                    normalize_advantage=True, clip_range_vf=None, target_kl=None, use_sde=False)
    settings[bad] = (lambda _: .3) if bad == "clip_range" else 1. if bad == "aux_coef" else 0
    with pytest.raises(ValueError):
        trainer.validate_loaded_model(SimpleNamespace(**settings), trainer.HYPERPARAMETERS, INITIAL)


def test_restored_optimizer_hash_is_stable_and_changes_with_adam_state():
    class Optimizer:
        def __init__(self, step):
            self.step = step

        def state_dict(self):
            return {"state": {0: {"step": self.step, "exp_avg": [.1, .2]}},
                    "param_groups": [{"lr": .0003, "params": [0]}]}
    assert trainer.optimizer_sha(Optimizer(1024)) == trainer.optimizer_sha(Optimizer(1024))
    assert trainer.optimizer_sha(Optimizer(1024)) != trainer.optimizer_sha(Optimizer(0))


def test_empty_optimizer_state_is_rejected():
    with pytest.raises(ValueError, match="missing restored"):
        trainer.optimizer_sha(SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": [{}]}))


def test_real_model_loading_seam_restores_optimizer_and_never_overrides_ppo_hyperparameters(case, monkeypatch):
    _, _, plan, _ = case
    captured = {}
    settings = {key: value for key, value in trainer.HYPERPARAMETERS.items()
                if key not in ("clip_range", "cpu_threads", "log_std_init")}
    optimizer = SimpleNamespace(state_dict=lambda: {
        "state": {0: {"step": 1024, "exp_avg": [.1]}}, "param_groups": [{"lr": .0003, "params": [0]}]})
    settings.update(num_timesteps=INITIAL, clip_range=lambda _: .2, seed=181, n_envs=4,
                    normalize_advantage=True, clip_range_vf=None, target_kl=None, use_sde=False,
                    policy=SimpleNamespace(optimizer=optimizer), _n_updates=1024,
                    set_logger=lambda logger: captured.update(logger=logger))
    model = SimpleNamespace(**settings)

    def load(path, **kwargs):
        captured.update(path=path, kwargs=kwargs)
        return model
    policy = ModuleType("d1_rl.velocity_policy15")
    policy.AuxiliaryVelocityPPO = SimpleNamespace(load=load)
    logger = ModuleType("stable_baselines3.common.logger")
    logger.configure = lambda *args: "fake logger"
    monkeypatch.setitem(sys.modules, "d1_rl.velocity_policy15", policy)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.logger", logger)
    runtime = trainer.RealRuntime()
    runtime.policy_sha = lambda model: "a"*64
    vec = object()
    assert runtime.create_model(vec, plan, trainer.HYPERPARAMETERS, Path("fake")) is model
    assert captured["path"] == plan["base_checkpoint"]["path"]
    assert captured["kwargs"] == {"env": vec, "device": "cpu", "seed": 181, "force_reset": True}
    assert runtime.initial_identity["optimizer_state_sha256"] == trainer.optimizer_sha(optimizer)
    assert runtime.initial_identity["num_timesteps"] == INITIAL and captured["logger"] == "fake logger"


def test_real_learn_seam_disables_reset_and_subtracts_baseline_from_callbacks(monkeypatch):
    callbacks = ModuleType("stable_baselines3.common.callbacks")
    callbacks.BaseCallback = type("BaseCallback", (), {})
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.callbacks", callbacks)
    calls = []
    watchdog = SimpleNamespace(poll=lambda **kwargs: True,
        on_step=lambda infos, rewards, transitions: calls.append(("step", transitions)) or True,
        on_rollout_end=lambda transitions: calls.append(("rollout", transitions)))

    def learn(**kwargs):
        assert kwargs["reset_num_timesteps"] is False and kwargs["total_timesteps"] == 32768
        callback = kwargs["callback"]
        callback.num_timesteps = INITIAL+4096
        callback.locals = {"infos": [], "rewards": []}
        assert callback._on_step() is True
        callback._on_rollout_end()
    trainer.RealRuntime().learn(SimpleNamespace(num_timesteps=INITIAL, learn=learn), 32768, watchdog)
    assert calls == [("step", 4096), ("rollout", 4096)]
