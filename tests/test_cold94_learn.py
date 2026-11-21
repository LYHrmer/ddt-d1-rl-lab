"""Synthetic CPU checks of the causal model/actor seam. No native integration."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from tools94 import learn94 as learn


def trace(seed=0, ticks=50):
    rng = np.random.default_rng(seed)
    state = np.zeros((ticks + 1, 45), dtype=np.float64)
    state[:, 0] = 1.10 - .002 * np.arange(ticks + 1)
    state[:, 2], state[:, 3] = .55, 1.
    state[:, 7:23] = np.tile([0., .8, -1.5, 0.], 4)
    issued = rng.uniform(-.05, .05, (ticks, 16)).astype(np.float32)
    applied = np.concatenate((np.zeros((1, 16), dtype=np.float32), issued[:-1]))
    obs = np.zeros((ticks + 1, 643), dtype=np.float32)
    obs[:, 5] = -1.
    obs[1:, 41:57] = issued
    flags = np.zeros(ticks, dtype=bool)
    flags[-1] = ticks < 50
    due = np.array([learn._due(k) for k in range(ticks + 1)])
    rays = np.zeros((ticks + 1, 16), dtype=np.float32)
    rays[due] = .5
    obs[2:, 627:] = .5
    return {"state45": state, "issued16": issued, "applied16": applied, "obs643": obs,
            "filtered3": np.zeros((ticks + 1, 3)), "ray_due": due, "ray_raw16": rays,
            "reset_seed": np.array(seed), "terminated": flags, "truncated": np.zeros(ticks, dtype=bool)}


def test_fit_samples_reset_and_terminal_without_crossing_and_rejects_partial():
    traces = [trace(s, 8 + s % 4) for s in range(12)]
    arrays, windows, receipt = learn._round_arrays(traces)
    assert receipt["transitions"] == sum(8 + s % 4 for s in range(12))
    assert len(windows) == sum(1 + s % 4 for s in range(12))
    np.testing.assert_array_equal(arrays["s"][0], traces[0]["state45"][0])
    np.testing.assert_array_equal(arrays["target"][7], traces[0]["state45"][-1])
    for row in windows.numpy():
        assert len(set(arrays["episode"][row])) == 1
    traces[0]["terminated"][:] = False
    with pytest.raises(ValueError, match="unfinished"):
        learn._round_arrays(traces)


def test_world_ridge_stays_frozen_and_new_batches_mix_equally(monkeypatch):
    monkeypatch.setattr(learn, "WORLD_UPDATES", 1)
    rounds = [[trace(s, 10) for s in range(12)], [trace(s + 20, 10) for s in range(12)]]
    world, adam, first = learn.fit_world(rounds[:1], seed=3)
    model_hash = learn.mapping_sha256(world.state_dict())
    new, _, second = learn.fit_world(rounds, seed=3, previous=(world, adam))
    assert first["rows_per_group_per_update"] == 32 and second["rows_per_group_per_update"] == 16
    assert first["ridge_and_stats_sha256"] == second["ridge_and_stats_sha256"]
    assert first["updates_completed"] == second["updates_completed"] == 1
    assert first["imagined_steps"] == second["imagined_steps"] == 256
    assert new is not world and learn.mapping_sha256(world.state_dict()) == model_hash
    assert not any(p.requires_grad for p in new.parameters())
    with pytest.raises(learn.LearnFailure) as exc:
        learn.fit_world([rounds[0], rounds[0]], seed=3, previous=(world, adam))
    assert exc.value.receipt["updates_completed"] == 0 and exc.value.receipt["imagined_steps"] == 0


def test_late_cut_restores_pending_ray_and_waits_for_delivery():
    data = trace()
    data["obs643"][18, 627:] = .2
    data["ray_raw16"][17] = .7
    queue = learn._Ranges(data, 18)
    np.testing.assert_allclose(queue.visible, .2)
    np.testing.assert_allclose(queue.at(19, None), .7)
    np.testing.assert_allclose(queue.at(20, None), .7)


class Plant(nn.Module):
    def step(self, state, action, terrain):
        # The measured correction appears through the four wheel targets; legs also affect x.
        drive = .01 * action[:, list(learn.WHEELS)].mean(1) + .003 * action[:, list(learn.LEGS)].mean(1)
        x = state[:, 0] - .01 + drive
        return torch.cat((x[:, None], state[:, 1:]), dim=1)


class Zero(nn.Module):
    def forward(self, inputs):
        return inputs.new_zeros((len(inputs), 16))


def test_probe_matches_physical_scale_h32_and_single_tick_fifo(monkeypatch):
    monkeypatch.setattr(learn, "_measure_packet", lambda state: np.zeros(16, dtype=np.float32))
    data = trace()
    cut = {key: data[key][:1] for key in ("state45", "obs643", "filtered3", "ray_due", "ray_raw16")}
    baseline, gradient = learn.probe(Plant(), Zero(), cut, 0., derivative=True)
    plus, _ = learn.probe(Plant(), Zero(), cut, .1)
    minus, _ = learn.probe(Plant(), Zero(), cut, -.1)
    assert baseline.shape == (51, 45)
    assert gradient == pytest.approx(15.5 * .01 / 5., abs=2e-7)
    np.testing.assert_array_equal(plus[:2], minus[:2])
    observed = (plus[1:33, 0].astype(float) - minus[1:33, 0]).mean() / .2
    assert observed == pytest.approx(gradient, abs=2e-5)


def test_probe_preserves_predicted_fall_and_counts(monkeypatch):
    class Falling(Plant):
        def step(self, state, action, terrain):
            out = super().step(state, action, terrain)
            return torch.cat((out[:, :2], out[:, 2:3] - .2, out[:, 3:]), 1)
    monkeypatch.setattr(learn, "_measure_packet", lambda state: np.zeros(16, dtype=np.float32))
    with pytest.raises(learn.LearnFailure, match="predicted fall") as raised:
        learn.probe(Falling(), Zero(), trace(), 0.)
    assert raised.value.partial_states.shape == (3, 45)
    assert raised.value.receipt["imagined_steps"] == raised.value.receipt["imagined_steps_attempted"] == 2


class SmallActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(643, 8)
        self.hidden_layer = nn.Linear(8, 8)
        self.action_head = nn.Linear(8, 16)

    def forward(self, inputs):
        return self.action_head(self.hidden_layer(self.input_layer(inputs).tanh()).tanh()).tanh()


def test_actor_updates_real_direction_all_output_groups_and_seals_original(monkeypatch):
    monkeypatch.setattr(learn, "_measure_packet", lambda state: np.full(16, .5, dtype=np.float32))
    monkeypatch.setattr(learn, "POLICY_UPDATES", 1)
    with torch.random.fork_rng():
        torch.manual_seed(4)
        actor = SmallActor()
    original = copy.deepcopy(actor.state_dict())
    traces = [trace(s) for s in range(12)]
    candidate, optimizer, audit = learn.update_actor(Plant(), actor, traces, seed=3)
    assert audit["status"] == "COMPLETE" and audit["updates_completed"] == 1
    assert audit["direction_check"]["passed"] and audit["direction_check"]["relative_gap"] <= .25
    assert audit["optimization_work"]["imagined_steps"] == 8 * 32
    assert audit["direction_work"]["imagined_steps"] == 2 * 8 * 32
    assert all(audit["parameter_changes"][k] > 0. for k in ("leg_output", "wheel_output", "range_input_columns"))
    assert all(torch.equal(original[k], v) for k, v in actor.state_dict().items())
    assert {id(p) for p in candidate.parameters()} == {id(p) for g in optimizer.param_groups for p in g["params"]}
