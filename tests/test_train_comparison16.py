"""Pure fake processes/models/counters; no Torch, robot, or physical execution."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools16"))
import train_comparison16 as trainer  # noqa: E402

GIB = 1024**3


class Clock:
    now = 0.
    def __call__(self): return self.now


class Vec:
    def __init__(self, envs, close_error=False):
        self.envs, self.steps, self.closed, self.close_error = envs, 0, False, close_error
        self.processes = []

    def close(self):
        self.closed = True
        if self.close_error: raise RuntimeError("fake close failure")


class Model:
    num_timesteps = 0
    def __init__(self, save_error=False): self.save_error = save_error
    def save(self, path):
        if self.save_error: raise OSError("fake save failure")
        Path(path).write_bytes(f"synthetic-checkpoint:{self.num_timesteps}".encode())


class Runtime:
    def __init__(self, clock, fault=None):
        self.clock, self.fault, self.prepared, self.spawn_calls = clock, fault, 0, 0
        self.vec = self.model = self.hyper = None

    def prepare(self):
        self.prepared += 1
        self.versions = {name: "synthetic" for name in ("python", "torch", "sb3", "numpy", "gymnasium", "mujoco")}
        if self.fault == "slow_prepare": self.clock.now = 1000.
    def create_vec(self, release, transitions, output, root):
        self.spawn_calls += 1
        if self.fault == "spawn": raise RuntimeError("fake constructor interruption")
        self.vec = Vec(release["envs"], self.fault == "close")
        if self.fault == "spawn_signal": signal.raise_signal(signal.SIGTERM)
        return self.vec

    def create_model(self, vec, release, hyper, output):
        self.hyper = hyper
        self.model = Model(self.fault == "save")
        return self.model

    def counters(self, vec):
        if self.fault == "counter_rpc": raise RuntimeError("fake counter RPC failure")
        rows = [dict(trainer.expected_counts(vec.steps//vec.envs), resets=1, forward_calls=vec.steps//vec.envs+3)
                for _ in range(vec.envs)]
        if self.fault == "counts": rows[0]["native_steps"] += 1
        return rows

    def policy_sha(self, model): return f"{model.num_timesteps:064x}"

    def learn(self, model, transitions, watchdog):
        if self.fault == "signal": signal.raise_signal(signal.SIGTERM)
        if self.fault == "wall": self.clock.now = 1000.
        if self.fault == "changed":
            (watchdog.output.parents[2]/"base.json").write_text('{"changed":true}')
        if not watchdog.poll(force=True): return
        cap = min(transitions, 4096) if self.fault == "partial" else transitions
        infos = [{"training_phase": "drive_forward", "reward_terms": {"contributions_per_tick": {"vx": .1}}}
                 for _ in range(self.vec.envs)]
        for _ in range(cap//4096):
            for _ in range(4096//self.vec.envs):
                self.vec.steps += self.vec.envs
                model.num_timesteps = self.vec.steps
                if not watchdog.on_step(infos, [.1]*self.vec.envs, model.num_timesteps): return
            self.clock.now += .1
            watchdog.on_rollout_end(model.num_timesteps)
        if self.fault == "late_wall": self.clock.now = 1000.


@pytest.fixture
def case(tmp_path):
    (tmp_path/"base.json").write_text('{}')
    release = {"format": "d1-local-comparison-v16", "status": "AUTHORIZED_FROZEN", "envs": 4,
               "seed": 101, "config": "base.json", "output_root": "evidence/run",
               "arms": {"control": {"aux_coef": 0.}, "velocity": {"aux_coef": 1.}},
               "stages": {"preflight": {"transitions": 32768, "max_wall_s": 900},
                          "train": {"transitions": 1048576, "max_wall_s": 5400}},
               "resources": {"start_min_effective_bytes": 6*GIB, "reserve_bytes": 3*GIB, "poll_s": 1.},
               "sources_sha256": {"base.json": trainer.sha(tmp_path/"base.json")}}
    path = tmp_path/"release.json"
    path.write_text(json.dumps(release))

    def loader(path, require_authorized, root):
        value = json.loads(Path(path).read_text())
        assert require_authorized is True
        if value["status"] != "AUTHORIZED_FROZEN": raise ValueError("not authorized")
        return value

    def validator(release, root):
        for name, digest in release["sources_sha256"].items():
            if trainer.sha(root/name) != digest: raise ValueError("source SHA changed")
    return tmp_path, path, release, loader, validator


def execute(case, *, arm="control", stage="preflight", fault=None, memory=None):
    root, path, release, loader, validator = case
    clock = Clock(); runtime = Runtime(clock, fault)
    result = trainer.run(path, arm, stage, root/release["output_root"]/f"{stage}_{arm}", root=root,
        runtime=runtime, clock=clock, memory_reader=memory or (lambda: {
            "available_ram_bytes": 8*GIB, "cgroup_remaining_bytes": 7*GIB, "effective_available_bytes": 7*GIB}),
        loader=loader, validator=validator, ticket_validator=lambda *args, **kwargs: None)
    return result, runtime, root/release["output_root"]/f"{stage}_{arm}"


@pytest.mark.parametrize("envs", [2, 4])
@pytest.mark.parametrize("arm,coef", [("control", 0.), ("velocity", 1.)])
def test_complete_preflight_is_exact_and_records_phase_denominators(case, envs, arm, coef):
    root, path, release, _, _ = case
    release["envs"] = envs; path.write_text(json.dumps(release))
    result, runtime, output = execute(case, arm=arm)
    assert result["status"] == "COMPLETE" and result["cleanup_complete"] and result["inputs_unchanged"]
    assert result["actual_counts"]["policy_transitions"] == 32768
    assert all(row["native_steps"] == 32768//envs*10 for row in result["per_worker_counts"])
    assert not result["eligible_for_evaluation"] and (output/"policy.zip").is_file()
    assert result["checkpoint_sha256"] == trainer.sha(output/"policy.zip")
    assert runtime.hyper["n_steps"] == 4096//envs and runtime.hyper["aux_coef"] == coef
    ledger = [json.loads(line) for line in (output/"ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == 8 and ledger[-1]["transitions"] == 32768
    assert all(row["phase_rewards"]["drive_forward"]["ticks"] == 4096 for row in ledger)
    assert ledger[0]["phase_rewards"]["drive_forward"]["reward_mean_per_tick"] == pytest.approx(.1)
    assert result["protocol_sha256"] == trainer.sha(output/"protocol.json")


def test_formal_complete_is_eligible_and_its_eight_prefix_rows_equal_preflight(case):
    pre, _, pre_dir = execute(case)
    result, _, output = execute(case, stage="train")
    assert pre["status"] == result["status"] == "COMPLETE"
    assert result["eligible_for_evaluation"] and result["trained_transitions"] == 1048576
    first = [json.loads(line)["deterministic"] for line in (pre_dir/"ledger.jsonl").read_text().splitlines()]
    second = [json.loads(line)["deterministic"] for line in (output/"ledger.jsonl").read_text().splitlines()[:8]]
    assert first == second


@pytest.mark.parametrize("fault", ["partial", "counts", "counter_rpc", "close", "save", "spawn", "changed", "wall", "late_wall"])
def test_incomplete_counts_cleanup_changed_source_and_wall_never_qualify(case, fault):
    result, runtime, output = execute(case, fault=fault)
    assert result["status"] != "COMPLETE" and not result["eligible_for_evaluation"]
    assert not (output/"policy.zip").exists()
    if runtime.model is not None and fault != "save": assert (output/"partial_policy.zip").exists()
    if fault == "changed": assert not result["inputs_unchanged"]
    if fault == "spawn":
        assert result["actual_counts"] is None and not result["cleanup_complete"]
    if fault == "counter_rpc": assert result["actual_counts"] is result["per_worker_counts"] is None
    if fault == "wall": assert result["stop_sample"]["phase"] == "training"
    if fault == "late_wall": assert result["stop_sample"]["phase"] == "after_learn"


def test_effective_cgroup_reserve_stops_even_when_host_available_is_high(case):
    calls = 0
    def memory():
        nonlocal calls
        calls += 1
        effective = 7*GIB if calls <= 3 else 2*GIB
        return {"available_ram_bytes": 8*GIB, "cgroup_remaining_bytes": effective, "effective_available_bytes": effective}
    result, runtime, output = execute(case, memory=memory)
    assert result["status"] == "RAM_RESERVE_BELOW_3_GIB" and result["trained_transitions"] == 0
    assert result["stop_sample"]["available_ram_bytes"] == 8*GIB
    assert result["stop_sample"]["effective_available_bytes"] == 2*GIB
    assert (output/"partial_policy.zip").exists() and runtime.vec.closed


def test_low_start_ram_refuses_before_output_or_spawn(case):
    root, path, release, loader, validator = case
    runtime = Runtime(Clock())
    output = root/release["output_root"]/"preflight_control"
    with pytest.raises(ValueError, match="NOT_READY"):
        trainer.run(path, "control", "preflight", output, root=root, runtime=runtime,
            loader=loader, validator=validator, memory_reader=lambda: {"effective_available_bytes": 5*GIB},
            ticket_validator=lambda *args, **kwargs: None)
    assert runtime.spawn_calls == runtime.prepared == 0 and not output.exists()


def test_after_final_update_reserve_sample_can_refuse_an_exact_finished_budget(case):
    calls = 0
    def memory():
        nonlocal calls
        calls += 1
        value = 7*GIB if calls <= 12 else 2*GIB
        return {"available_ram_bytes": 8*GIB, "cgroup_remaining_bytes": value, "effective_available_bytes": value}
    result, _, output = execute(case, memory=memory)
    assert result["trained_transitions"] == 32768
    assert result["status"] == "RAM_RESERVE_BELOW_3_GIB"
    assert result["stop_sample"]["phase"] == "after_learn"
    assert not (output/"policy.zip").exists() and (output/"partial_policy.zip").exists()


def test_direct_stage_without_runner_ticket_refuses_before_heavy_prepare(case):
    root, path, release, loader, validator = case
    runtime = Runtime(Clock())
    def denied(*args, **kwargs): raise ValueError("missing STARTED launch ticket")
    output = root/release["output_root"]/"train_control"
    with pytest.raises(ValueError, match="ticket"):
        trainer.run(path, "control", "train", output, root=root, runtime=runtime,
                    loader=loader, validator=validator, ticket_validator=denied)
    assert runtime.prepared == runtime.spawn_calls == 0 and not output.exists()


def test_heavy_import_time_is_included_in_wall_limit_before_any_worker_spawn(case):
    root, path, release, loader, validator = case
    clock = Clock(); runtime = Runtime(clock, "slow_prepare")
    output = root/release["output_root"]/"preflight_control"
    with pytest.raises(ValueError, match="WALL_LIMIT before spawn"):
        trainer.run(path, "control", "preflight", output, root=root, runtime=runtime, clock=clock,
            loader=loader, validator=validator, ticket_validator=lambda *args, **kwargs: None,
            memory_reader=lambda: {"effective_available_bytes": 7*GIB})
    assert runtime.spawn_calls == 0 and not output.exists()


@pytest.mark.parametrize("fault", ["spawn_signal", "signal"])
def test_sigterm_is_structured_and_cleans_even_during_deferred_constructor(case, fault):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    result, runtime, output = execute(case, fault=fault)
    assert result["status"] == "SIGTERM" and result["cleanup_complete"] and runtime.vec.closed
    assert not result["eligible_for_evaluation"] and not (output/"policy.zip").exists()
    assert previous == {sig: signal.getsignal(sig) for sig in previous}


def test_close_error_escalates_only_own_worker_and_verifies_final_liveness():
    class Worker:
        alive = True
        def is_alive(self): return self.alive
        def terminate(self): self.terminated = True
        def join(self, timeout): pass
        def kill(self): self.alive = False
    vec = Vec(2, close_error=True)
    vec.processes = [Worker()]
    complete, error = trainer.close_workers(vec)
    assert complete and vec.processes[0].terminated and not vec.processes[0].alive
    assert "close failed" in error


def test_unapproved_release_and_existing_wrong_or_symlink_outputs_refuse_before_prepare(case):
    root, path, release, loader, validator = case
    runtime = Runtime(Clock())
    output = root/release["output_root"]/"preflight_control"
    release["status"] = "PREPARED_NOT_AUTHORIZED"; path.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="authorized"):
        trainer.run(path, "control", "preflight", output, root=root, runtime=runtime, loader=loader, validator=validator)
    release["status"] = "AUTHORIZED_FROZEN"; path.write_text(json.dumps(release))
    output.mkdir(parents=True)
    for bad in (output, root/"wrong"):
        with pytest.raises(ValueError, match="output"):
            trainer.run(path, "control", "preflight", bad, root=root, runtime=runtime, loader=loader, validator=validator)
    assert runtime.spawn_calls == runtime.prepared == 0


def test_training_phases_use_pre_step_course_and_separate_cut_from_stationary():
    course = {"segments": [(0, 50, (0., 0., 0.)), (50, 250, (.2, 0., 0.)),
                           (250, 350, (0., 0., 0.)), (350, 550, (-.2, 0., 0.)), (550, 1000, (0., 0., 0.))]}
    assert [trainer.training_phase(course, tick) for tick in (49, 50, 249, 250, 274, 275, 350, 550, 575)] == [
        "stationary_initial", "drive_forward", "drive_forward", "cut_zero_after_forward", "cut_zero_after_forward",
        "stationary_after_forward", "drive_backward", "cut_zero_after_backward", "stationary_after_backward"]
    with pytest.raises(ValueError): trainer.training_phase(course, 1000)
