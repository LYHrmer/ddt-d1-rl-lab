"""Asset-free53 collection checks using a synthetic environment; zero physics."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lib():
    for directory in (ROOT, ROOT / ".python-deps"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from tools53 import collection53
    return SimpleNamespace(np=np, torch=torch, entry=collection53)


def plan(lib):
    return {"data_seeds": copy.deepcopy(lib.entry.DATA_SEEDS), "identities": {
        "tools/evaluate_d1_ppo.py": {"sha256": hashlib.sha256(
            (ROOT / "tools/evaluate_d1_ppo.py").read_bytes()).hexdigest()}}}


def fake_environment(lib, monkeypatch, *, fail_seed=None, fail_tick=3, failure="fallen"):
    np = lib.np
    record = {"resets": [], "noise": {}}

    class Data:
        def __init__(self, model):
            self.model = model
            self.qpos, self.qvel = np.zeros(23), np.zeros(22)
            self.qpos[2:4] = [.5, 1.]
            self.qpos[7:] = [0., .8, -1.5, 0.] * 4
            self.time = 0.
            self.narena, self.nbuffer = 128, 64

        def __copy__(self):
            result = Data(self.model)
            result.qpos, result.qvel = self.qpos.copy(), self.qvel.copy()
            result.time = self.time
            return result

    class Env:
        def __init__(self):
            self.model = SimpleNamespace(nq=23, nv=22)
            self.data = Data(self.model)
            self._counts = {key: 0 for key in lib.entry.COUNTERS}
            self.eval_command_sequence = [(0., 0., 0.)]
            self.nominal_base_height_m = .5
            self._delay_fifo = deque()

        def _validate_eval_sequence(self, sequence):
            return [(row["start_s"], row["vx"], row["wz"]) for row in sequence]

        def _update(self):
            segment = self.eval_command_sequence[0]
            for candidate in self.eval_command_sequence:
                if candidate[0] <= self._episode_steps * .02 + 1e-12:
                    segment = candidate
            self._command = np.array([segment[1], 0., segment[2]])
            self._current_obs = np.zeros(57, np.float32)
            self._current_obs[5] = -1.
            self._current_obs[6:9] = self._command * [2., 2., .25]
            self._current_obs[41:] = self._last_policy_action

        def reset(self, *, seed):
            self.seed = seed
            record["resets"].append(seed)
            self._episode_steps, self._episode_done = 0, False
            self.data = Data(self.model)
            self._last_policy_action = np.zeros(16, np.float32)
            self._applied_action = np.zeros(16, np.float32)
            self._filtered_gyro = np.zeros(3)
            self._update()
            self._history = np.repeat(self._current_obs[None], 10, axis=0)
            self._counts["resets"] += 1
            self._counts["forward_calls"] += 1
            return self.current_observation(), {}

        def current_observation(self):
            return np.r_[self._current_obs, self._history.ravel()].astype(np.float32)

        def counters(self):
            return dict(self._counts)

        def _sensor(self, name):
            return {"trunk_gyro": self.data.qvel[3:6], "trunk_quat": self.data.qpos[3:7],
                    "frame_vel": self.data.qvel[:3]}[name]

        def step(self, action):
            assert not self._episode_done
            command = self._command.copy()
            self._history = np.concatenate((self._history[1:], self._current_obs[None]))
            self._last_policy_action = np.clip(action, -1, 1).astype(np.float32)
            self._applied_action = self._last_policy_action.copy()
            for key, increment in (("policy_transitions", 1), ("native_steps", 10),
                                   ("pd_updates", 5), ("forward_calls", 1)):
                self._counts[key] += increment
            self._episode_steps += 1
            self.data.time = self._episode_steps * .02
            self._update()
            failed = self.seed == fail_seed and self._episode_steps == fail_tick
            if failed and failure == "raise":
                raise RuntimeError("synthetic failure after attempted work")
            terminated, truncated = failed and failure != "truncated", failed and failure == "truncated"
            self._episode_done = bool(terminated or truncated)
            info = {"command_during_interval": command,
                    "fallen": bool(failed and failure == "fallen"),
                    "illegal_body_contact": bool(failed and failure == "illegal_body_contact"),
                    "base_height_m": .5, "tilt_rad": 0.}
            return self.current_observation(), 0., bool(terminated), bool(truncated), info

    def action(env, shaper, _actor, theta, noise=None):
        assert theta == 0.
        shaper(env.current_observation())
        item = record["noise"].setdefault(env.seed, {"calls": 0, "noise_calls": 0, "first": None})
        item["calls"] += 1
        if noise is not None:
            item["noise_calls"] += 1
            if item["first"] is None:
                item["first"] = noise.copy()
        return np.zeros(16, np.float32) if noise is None else np.clip(noise, -1, 1).astype(np.float32)

    monkeypatch.setattr(lib.entry.control, "real_action", action)
    return Env(), record


def test_helpers_bind_original_timing_and_never_mix_translation_with_turning(lib):
    registered = plan(lib)
    sequences = lib.entry.helpers(registered).SEQUENCES
    assert tuple(sequences) == ("forward_stop", "reverse_stop", "turn_stop")
    assert [(row["start_s"], row["vx"], row["wz"]) for row in sequences["turn_stop"]] == [
        (0., 0., 0.), (2., 0., .2), (5., 0., -.2), (8., 0., 0.)]
    for name, sign in (("forward_stop", 1), ("reverse_stop", -1)):
        assert [(row["start_s"], row["vx"], row["wz"]) for row in sequences[name]] == [
            (0., 0., 0.), (2., sign * .2, 0.), (8., 0., 0.)]
    assert all(row["vx"] == 0. or row["wz"] == 0. for rows in sequences.values() for row in rows)
    registered["identities"]["tools/evaluate_d1_ppo.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA mismatch"):
        lib.entry.helpers(registered)


def test_make_env_audit_callback_is_outside_snapshot_dictionary(lib, monkeypatch):
    seen, audit = [], []

    class Base:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.model = SimpleNamespace(nq=23, nv=22)
            self.data = SimpleNamespace(model=self.model, time=0.)
            self._counts = {"policy_transitions": 0}
            self.free_qpos = self.free_dof = 0
            self.qpos_ids, self.dof_ids = lib.np.arange(7, 23), lib.np.arange(6, 22)

        def step(self, action):
            self.data.time += .02
            self._counts["policy_transitions"] += 1
            return ("synthetic", action)

    monkeypatch.setitem(sys.modules, "d1_rl.factory03", SimpleNamespace(environment_class=lambda _config: Base))
    class Callback:
        def __call__(self, env, action, result):
            audit.append((env, action, result))

        def __deepcopy__(self, _memo):
            raise AssertionError("audit callback must stay outside the snapshot")

    callback = Callback()
    env = lib.entry.make_env(24000, audit_step=callback)
    snapshot = lib.entry.adapter.capture(env)
    result = env.step("issued")
    assert seen[0]["seed"] == 53000 and seen[0]["arm"] == "B" and seen[0]["eval_delay_ticks"] == 0
    assert seen[0]["max_transitions"] == 24000
    assert not any(value is callback for value in vars(env).values())
    assert audit == [(env, "issued", result)]
    lib.entry.adapter.restore(env, None, snapshot)
    assert env._counts["policy_transitions"] == 1
    env.step("after restore")
    assert len(audit) == 2 and env._counts["policy_transitions"] == 2
    assert type(lib.entry.make_env(256)) is Base


def test_complete_collection_uses_registered_roles_tasks_noise_and_points(lib, monkeypatch, tmp_path):
    env, record = fake_environment(lib, monkeypatch)
    arrays, snapshots = {}, []

    def saved_npz(path, values):
        arrays[path.name] = {key: value.copy() for key, value in values.items()}
        path.write_bytes(b"synthetic saved arrays")

    def saved_points(path, values, _env):
        snapshots.append((path.name, [(p["seed"], p["tick"], p["commands"][0].copy()) for p in values]))
        path.write_bytes(b"synthetic snapshots")

    monkeypatch.setattr(lib.entry, "save_npz", saved_npz)
    monkeypatch.setattr(lib.entry, "save_points", saved_points)
    # Failure tests below exercise real flush/write behavior.  Avoid duplicating
    # thousands of large JSON lines in this successful schedule/partition test.
    monkeypatch.setattr(lib.entry, "_event", lambda _stream, _row: None)
    datasets, points = lib.entry.collect(env, object(), tmp_path, plan(lib))
    seeds = [seed for role in lib.entry.ROLES for seed in lib.entry.DATA_SEEDS[role]]
    assert record["resets"] == seeds
    assert env.counters()["policy_transitions"] == 10800
    assert {role: len(data["s"]) for role, data in datasets.items()} == {
        "train": 7200, "development": 1800, "probe": 1800}
    assert {role: len(value) for role, value in points.items()} == {"train": 0, "development": 9, "probe": 9}
    summary = json.loads((tmp_path / "collection.json").read_text())
    assert summary["status"] == "COMPLETE"
    assert [row["task"] for row in summary["cases"][:12]] == list(lib.entry.TASKS) * 4
    assert all(row["task"] == "turn_stop" for row in summary["cases"][12:])
    assert all(row["complete_12s_trial"] and row["steps"] == 600 for row in summary["cases"])
    for role in lib.entry.ROLES:
        data = datasets[role]
        assert lib.np.flatnonzero(data["done"]).tolist() == list(range(599, len(data["done"]), 600))
        assert data["tick"].tolist() == list(range(600)) * len(lib.entry.DATA_SEEDS[role])
        for seed in lib.entry.DATA_SEEDS[role]:
            assert record["noise"][seed]["calls"] == 600
            assert record["noise"][seed]["noise_calls"] == (600 if role == "train" else 0)
            if role == "train":
                expected = lib.np.random.default_rng(seed + 900000).normal(0., .02, 16)
                lib.np.testing.assert_array_equal(record["noise"][seed]["first"], expected)
    for role in ("development", "probe"):
        assert [(p["seed"], p["tick"]) for p in points[role]] == [
            (seed, tick) for seed in lib.entry.DATA_SEEDS[role] for tick in (110, 260, 410)]
        assert [p["commands"][0, 2] for p in points[role]] == [.2, -.2, 0.] * 3
        assert any(name == f"{role}_snapshots.pkl" for name, _ in snapshots)
    for row in summary["cases"]:
        trace = arrays[row["trace_file"]]
        commands = trace["command_during_interval"]
        assert not lib.np.any((commands[:, 0] != 0) & (commands[:, 2] != 0))
        assert lib.np.all(commands[:100] == 0) and lib.np.all(commands[400:] == 0)
        if row["task"] == "turn_stop":
            assert lib.np.all(commands[100:250, 2] == .2)
            assert lib.np.all(commands[250:400, 2] == -.2)
        else:
            sign = 1 if row["task"] == "forward_stop" else -1
            assert lib.np.all(commands[100:400, 0] == sign * .2)
        for key in ("issued_action", "applied_action", "fallen", "illegal_body_contact",
                    "base_height_m", "tilt_rad", "terminated", "truncated", "native_steps", "pd_updates"):
            assert len(trace[key]) == 600


@pytest.mark.parametrize("failure", ["fallen", "illegal_body_contact", "truncated", "raise"])
def test_first_failure_preserves_case_and_stops_all_remaining_work(lib, monkeypatch, tmp_path, failure):
    env, record = fake_environment(lib, monkeypatch, fail_seed=53102, failure=failure)
    with pytest.raises(RuntimeError, match="first early termination|synthetic failure"):
        lib.entry.collect(env, object(), tmp_path, plan(lib))
    assert record["resets"] == [53101, 53102]
    summary = json.loads((tmp_path / "collection.json").read_text())
    assert summary["status"] == "INCOMPLETE"
    assert summary["cases"][0]["status"] == "COMPLETE"
    failed = summary["cases"][1]
    assert failed["status"] == ("INCOMPLETE" if failure == "raise" else "TERMINATED")
    assert all(row["status"] == "NOT_RUN" and row["reason"] for row in summary["cases"][2:])
    assert summary["actual_counts"]["policy_transitions"] == 603
    assert failed["actual_counts"]["policy_transitions"] == 3
    completed = 2 if failure == "raise" else 3
    assert failed["steps"] == completed and failed["completed_prefix_counts"]["policy_transitions"] == completed
    if failure != "raise":
        assert failure in failed["termination_reasons"]
    with lib.np.load(tmp_path / failed["trace_file"], allow_pickle=False) as trace:
        assert trace["s"].shape == (completed, 45)
        assert trace["issued_action"].shape == trace["applied_action"].shape == (completed, 16)
        if failure != "raise":
            assert bool(trace[failure][-1])
    assert (tmp_path / "train_transitions_partial.npz").exists()
    assert not (tmp_path / "train_transitions.npz").exists()
    assert not list(tmp_path.glob("collection_development_*.npz"))
    events = [json.loads(line) for line in (tmp_path / "collection_events.jsonl").read_text().splitlines()]
    begins = [row for row in events if row["event"] == "STEP_BEGIN" and row["seed"] == 53102]
    completed_rows = [row for row in events if row["event"] == "STEP_COMPLETE" and row["seed"] == 53102]
    assert len(begins) == 3 and len(completed_rows) == completed
    assert completed_rows[-1]["policy_transitions"] == 600 + completed
    assert events[-1]["event"] == "COLLECTION_FAILED"


def test_wrong_seed_roles_are_rejected_before_any_reset(lib, monkeypatch, tmp_path):
    env, record = fake_environment(lib, monkeypatch)
    registered = plan(lib)
    registered["data_seeds"]["probe"][0] = 53101
    with pytest.raises(ValueError, match="role seeds differ"):
        lib.entry.collect(env, object(), tmp_path, registered)
    assert record["resets"] == []


def test_preflight_covers_original_turn_and_exact_224_step_budget(lib, monkeypatch, tmp_path):
    env, record = fake_environment(lib, monkeypatch)
    result = lib.entry.preflight(env, object(), tmp_path, plan(lib))
    assert record["resets"] == [53000]
    assert result["status"] == "COMPLETE" and result["mapping_steps"] == 160
    assert result["same_snapshot_32_step_exact"] is True
    assert result["branch_work_not_refunded"] is True
    assert result["actual_counts"]["policy_transitions"] == env.counters()["policy_transitions"] == 224
    with lib.np.load(tmp_path / "preflight_mapping.npz", allow_pickle=False) as trace:
        assert lib.np.all(trace["command_during_interval"][:100] == 0)
        assert lib.np.all(trace["command_during_interval"][100:, 2] == .2)
    assert (tmp_path / "restore_parity.npz").exists()


def test_preflight_failure_writes_diagnostics_before_reraising(lib, monkeypatch, tmp_path):
    env, _ = fake_environment(lib, monkeypatch, fail_seed=53000, fail_tick=3)
    with pytest.raises(ValueError, match="admission trajectory terminated"):
        lib.entry.preflight(env, object(), tmp_path, plan(lib))
    report = json.loads((tmp_path / "preflight_diagnostics.json").read_text())
    assert report["status"] == "INCOMPLETE" and report["mapping_steps"] == 3
    assert "fallen" in report["termination_reasons"]
    assert report["actual_counts"]["policy_transitions"] == 3
    assert all(branch["status"] == "NOT_RUN" for branch in report["branches"])
    with lib.np.load(tmp_path / "preflight_mapping.npz", allow_pickle=False) as trace:
        assert bool(trace["fallen"][-1]) and bool(trace["terminated"][-1])
