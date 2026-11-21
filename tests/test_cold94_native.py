"""Cold actor and trace semantics, using a synthetic clock with no MuJoCo steps."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools94 import actor94, native94
from training.d1_rl.interface import LEG_INDICES, WHEEL_INDICES


class SyntheticPlant:
    def __init__(self, *, fail_at=None, fall_at=None):
        self.fail_at = fail_at
        self.fall_at = fall_at
        self.closed = False
        self.data = SimpleNamespace(qpos=np.zeros(23), qvel=np.zeros(22), time=0.)
        self.data.qpos[3] = 1.
        self.free_qpos = self.free_dof = 0
        self.qpos_ids, self.dof_ids = np.arange(7, 23), np.arange(6, 22)
        self.action_scales = np.tile([0.5, 0.5, 0.5, 5.], 4)
        self.counts = dict(policy_transitions=0, pd_updates=0, native_steps=0)

    def counters(self):
        return self.counts.copy()

    def observation(self, tick):
        if tick == 0 or tick * 6 // 50 != (tick-1) * 6 // 50:
            self._pending = (tick, np.full(16, (tick + 1) / 100., dtype=np.float32))
        if self._pending is not None and self._pending[0] + 2 == tick:
            self.sensor_capture_tick, self.last_raw_features = self._pending
            self.sensor_delivery_tick = tick
            self._pending = None
        obs = np.zeros(643, dtype=np.float32)
        self._filtered_gyro = np.full(3, tick / 10.)
        obs[:3] = self._filtered_gyro * .25
        obs[41:57] = self._last_policy_action
        obs[627:] = self.last_raw_features
        return obs

    def reset(self, *, seed=None):
        self._last_policy_action = np.zeros(16, dtype=np.float32)
        self.last_raw_features = np.zeros(16, dtype=np.float32)
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self._pending = None
        self.data.qpos[0] = 1.1 if seed is None else 0.
        return self.observation(0), {"episode_index": int(seed is None), "slope_start": seed is None,
                                     "start_x_m": self.data.qpos[0], "delay_ticks": 1}

    def step(self, issued):
        self.counts["policy_transitions"] += 1
        tick = self.counts["policy_transitions"]
        if tick == self.fail_at:
            self.counts["native_steps"] += 3
            self.counts["pd_updates"] += 2
            raise RuntimeError("synthetic integration failure")
        self.counts["native_steps"] += 10
        self.counts["pd_updates"] += 5
        previous = self._last_policy_action
        self._last_policy_action = issued.copy()
        self.data.time = tick * .02
        self.data.qpos[0] -= .001
        fallen = tick == self.fall_at
        info = dict(applied_action=previous, fallen=fallen, illegal_body_contact=False,
                    command_during_interval=np.zeros(3), wheel_saturation_count=0,
                    loaded_wheels_past_toe=np.full(4, True))
        return self.observation(tick), 0., fallen, False, info

    def close(self):
        self.closed = True


class ConstantActor(torch.nn.Module):
    def __init__(self, value=0.):
        super().__init__()
        self.value = value

    def forward(self, obs):
        return obs.new_full((len(obs), 16), self.value)


def test_actor_is_fresh_reproducible_and_differentiable():
    original_rng = torch.random.get_rng_state()
    actor, clone, other = (actor94.make_actor(seed) for seed in (41, 41, 42))
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for name, parameter in actor.named_parameters():
        assert torch.equal(parameter, clone.state_dict()[name])
        if name.endswith("bias"):
            assert not torch.count_nonzero(parameter)
        else:
            assert not torch.equal(parameter, other.state_dict()[name])
    limit = .05 * np.sqrt(6 / (64 + 16))
    assert 0 < actor.action_head.weight.abs().max().item() <= limit
    obs = torch.linspace(-.5, .5, 643)[None, :].requires_grad_()
    output = actor(obs)
    gradient, = torch.autograd.grad(output.sum(), obs)
    assert gradient[:, :627].abs().sum() > 0 and gradient[:, 627:].abs().sum() > 0
    np.testing.assert_array_equal(actor94.predict(actor, obs.detach()[0].numpy()), output.detach()[0].numpy())


def test_calibration_is_in_physical_units_and_sensor_queue_is_restorable(monkeypatch):
    plant = SyntheticPlant()
    monkeypatch.setattr(native94, "_open_env", lambda seed, ticks: plant)
    trace = native94.collect_episode(ConstantActor(), 71, ticks=20, wheel_offset_rad_s=.2)
    assert plant.closed and trace["reset_seed"] == 71 and trace["wheel_offset_rad_s"] == .2
    assert (trace["policy_steps"], trace["servo_steps"], trace["native_steps"]) == (20, 100, 200)
    assert trace["state45"].shape == (21, 45) and trace["obs643"].shape == (21, 643)
    assert not trace["issued16"][:, LEG_INDICES].any()
    np.testing.assert_array_equal(trace["issued16"][:, WHEEL_INDICES], np.full((20, 4), .04, np.float32))
    assert not trace["applied16"][0].any()
    np.testing.assert_array_equal(trace["applied16"][1:], trace["issued16"][:-1])
    assert trace["ray_due"].nonzero()[0].tolist() == [0, 9, 17]
    assert trace["pending_capture_tick"][18] == 17 and trace["sensor_delivery_tick"][18] == 11
    np.testing.assert_array_equal(trace["ray_raw16"][17], trace["pending_ray16"][18])
    np.testing.assert_array_equal(trace["obs643"][18, 627:], trace["ray_raw16"][9])
    np.testing.assert_array_equal(trace["obs643"][19, 627:], trace["ray_raw16"][17])
    np.testing.assert_array_equal(trace["obs643"][:, :3], (trace["filtered3"] * .25).astype(np.float32))


def test_held_exploration_is_seeded_and_clipping_follows_wheel_offset(monkeypatch):
    monkeypatch.setattr(native94, "_open_env", lambda seed, ticks: SyntheticPlant())
    sampled = native94.collect_episode(ConstantActor(), 72, ticks=11, explore_seed=20)
    replayed = native94.collect_episode(ConstantActor(), 72, ticks=11, explore_seed=20)
    noise = sampled["exploration16"]
    np.testing.assert_array_equal(noise, replayed["exploration16"])
    np.testing.assert_array_equal(noise, sampled["issued16"])
    assert (np.diff(noise, axis=0) != 0).any(axis=1).nonzero()[0].tolist() == [4, 9]
    assert np.abs(noise[:, LEG_INDICES]).max() <= .06
    assert np.abs(noise[:, WHEEL_INDICES]).max() <= .14
    assert np.ptp(noise[:, WHEEL_INDICES], axis=1).max() <= .08
    clipped = native94.collect_episode(ConstantActor(.99), 72, ticks=1, wheel_offset_rad_s=.2)
    assert (clipped["issued16"][0, WHEEL_INDICES] == 1.).all()
    np.testing.assert_array_equal(clipped["issued16"][0, LEG_INDICES], clipped["raw16"][0, LEG_INDICES])


def test_native_failure_keeps_complete_prefix_and_partial_integration_counts(monkeypatch):
    plant = SyntheticPlant(fail_at=4)
    monkeypatch.setattr(native94, "_open_env", lambda seed, ticks: plant)
    seen = []
    with pytest.raises(native94.NativeEpisodeError, match="synthetic integration failure") as caught:
        native94.collect_episode(ConstantActor(), 73, ticks=10, progress=lambda trace: seen.append(trace))
    trace = caught.value.trace
    assert plant.closed and [len(t["raw16"]) for t in seen] == [0, 1, 2, 3]
    assert trace["completed_transitions"] == 3 and trace["state45"].shape == (4, 45)
    assert (trace["policy_steps"], trace["servo_steps"], trace["native_steps"]) == (4, 17, 33)
    assert not trace["terminated"].any()


def test_early_terminal_interval_is_included_once(monkeypatch):
    monkeypatch.setattr(native94, "_open_env", lambda seed, ticks: SyntheticPlant(fall_at=3))
    trace = native94.collect_episode(ConstantActor(), 74, ticks=10)
    assert trace["terminated"].tolist() == [False, False, True]
    assert trace["fallen"].tolist() == trace["terminated"].tolist()
    assert trace["completed_transitions"] == trace["policy_steps"] == 3
