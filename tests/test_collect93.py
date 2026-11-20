"""The crossed one-second collector uses a synthetic, counted environment."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools93 import collect93 as collect


class CountedFixture:
    def __init__(self, mode):
        self.mode = mode
        self.data = SimpleNamespace(qpos=np.zeros(23), qvel=np.zeros(22), time=0.)
        self.data.qpos[[0, 2, 3]] = (1.10, .4, 1.)
        self.free_qpos = 0
        self.action_scales = np.full(16, 5.)
        self.dof_ids = np.arange(6, 22, dtype=np.int32)
        self.last_raw_features = np.zeros(16, np.float32)
        self._last_policy_action = np.zeros(16, np.float32)
        self.counts = dict.fromkeys(collect.CLOCKS, 0)
        self.resets, self.closed = [], False

    def reset(self, *, seed=None):
        self.resets.append(seed)
        index = len(self.resets)-1
        return np.zeros(643, np.float32), dict(slope_start=bool(index), episode_index=index, delay_ticks=1, start_x_m=1.10)

    def counters(self):
        return self.counts.copy()

    def step(self, issued):
        self.counts["policy_transitions"] += 1
        tick = self.counts["policy_transitions"]
        if tick == 2 and self.mode == "native_error":
            self.counts["pd_updates"] += 1
            self.counts["native_steps"] += 1
            raise FloatingPointError("synthetic native failure")
        self.counts["pd_updates"] += 5
        self.counts["native_steps"] += 10
        applied = self._last_policy_action.astype(np.float64)
        self._last_policy_action[:] = issued
        # The first applied target is zero. The fixture then obeys its own recorded feedback.
        self.data.qvel[0] = -.02 + .01*applied[3]
        self.data.qpos[0] += .02*self.data.qvel[0]
        self.data.time = tick*.02
        if self.mode == "first_endpoint" and tick == 1:
            self.data.qvel[4] = np.float64(-0.)  # numerically equal but different state bytes
        self.last_raw_features[8:] = 1
        next_obs = np.zeros(643, np.float32)
        next_obs[-16:] = self.last_raw_features
        fall = tick == 2 and self.mode == "early_fall"
        info = dict(applied_action=applied.tolist(), wheel_target_rad_s=applied[collect.WHEEL_INDICES]*5.,
                    last_torque_nm=np.zeros(16), command_during_interval=np.zeros(3),
                    body_vx_m_s=self.data.qvel[0], base_pitch_rad=0., base_roll_rad=0.,
                    loaded_wheels_past_toe=np.ones(4, np.bool_), wheel_saturation_count=0,
                    fallen=fall, illegal_body_contact=False)
        return next_obs, 0., fall, False, info

    def close(self):
        self.closed = True


def fixtures(mode):
    env = CountedFixture(mode)
    reference = dict(initial_qpos=env.data.qpos.copy(), initial_qvel=env.data.qvel.copy(),
                     initial_obs=np.zeros(643, np.float32), free_qpos_index=np.asarray(env.free_qpos),
                     wheel_action_scales_rad_s=env.action_scales[collect.WHEEL_INDICES].copy(),
                     wheel_dof_indices=env.dof_ids[collect.WHEEL_INDICES].copy(),
                     first_qpos=env.data.qpos.copy(), first_qvel=env.data.qvel.copy(), first_applied=np.zeros(16))
    reference["first_qpos"][0] += .02*(-.02)
    reference["first_qvel"][0] = -.02
    if mode == "initial":
        reference["initial_obs"][4] = 1.

    def factory(terrain, reset, **kwargs):
        assert (terrain, reset) == ("up", 90102)
        assert kwargs == dict(curriculum=True, slope_start_x_m=1.10, perception=True, max_transitions=50)
        return env

    def predict(obs, packet):
        assert obs.shape == (627,) and np.array_equal(packet, env.last_raw_features)
        return np.full(16, -.25, np.float32)

    return env, reference, factory, predict


def test_six_missing_cells_only_and_exclusive_attempt(tmp_path):
    inventory = collect.new_arms()
    assert len(inventory) == len(set(inventory)) == 12
    for p, original_reset in zip(collect.PARENTS, collect.RESETS, strict=True):
        assert {(s, a) for pp, s, a in inventory if pp == p} == {
            (s, a) for s in collect.RESETS if s != original_reset for a in collect.ARMS}
    with pytest.raises(FileExistsError):
        collect.run(tmp_path)
    with pytest.raises(ValueError, match="missing physical arms"):
        collect.episode(tmp_path, 87011, 90101, "original", "", None, None, {})


@pytest.mark.parametrize("arm", collect.ARMS)
def test_initial_to_first_target_and_full_one_second_without_continuation(tmp_path, arm):
    env, reference, factory, predict = fixtures("ok")
    case = collect.episode(tmp_path, 87011, 90102, arm, "synthetic", predict, factory, reference)
    assert env.resets == [90102, None] and env.closed
    assert case["clean"] and case["status"] == "COMPLETE" and case["reason"] is None
    assert [case[k] for k in ("ticks", "policy_calls", *collect.CLOCKS)] == [50, 50, 50, 250, 500]
    assert case["initial_state_sha256"] == collect.array_digest(reference[k] for k in collect.INITIAL)
    with np.load(tmp_path / case["trace"], allow_pickle=False) as tape:
        assert tape["time_s"].shape == (50,) and tape["time_s"][-1] == 1.
        assert tape["applied"].shape == (50, 16) and not tape["applied"][0].any()
        np.testing.assert_array_equal(tape["applied"][1:], tape["issued"][:-1])
        np.testing.assert_array_equal(tape["qpos"][0], reference["first_qpos"])
        assert np.all(tape["issued"][:, collect.WHEEL_INDICES] == (0. if arm == "common" else -.25))
    with pytest.raises(FileExistsError):
        collect.episode(tmp_path, 87011, 90102, arm, "synthetic", predict, factory, reference)
    assert env.counters()["native_steps"] == 500


@pytest.mark.parametrize("mode,exception,expected_counts", [
    ("initial", ValueError, (0, 0, 0, 0, 0)),
    ("first_endpoint", ValueError, (1, 1, 1, 5, 10)),
    ("native_error", FloatingPointError, (1, 2, 2, 6, 11)),
])
def test_mismatch_and_midinterval_error_preserve_real_rows_and_cost(tmp_path, mode, exception, expected_counts):
    env, reference, factory, predict = fixtures(mode)
    with pytest.raises(exception):
        collect.episode(tmp_path, 87011, 90102, "original", "synthetic", predict, factory, reference)
    case = collect.read(tmp_path / "parent87011/original_90102/case.json")
    assert case["status"] == "PARTIAL" and not case["clean"] and case["reason"] and env.closed
    assert tuple(case[k] for k in ("ticks", "policy_calls", *collect.CLOCKS)) == expected_counts
    assert case["trace"].endswith("partial_trace.npz")
    with np.load(tmp_path / case["trace"], allow_pickle=False) as tape:
        assert len(tape["qpos"]) == expected_counts[0] if "qpos" in tape else expected_counts[0] == 0
        assert "initial_qpos" in tape


def test_fall_keeps_actual_prefix_and_is_excluded_from_complete_cells(tmp_path):
    env, reference, factory, predict = fixtures("early_fall")
    case = collect.episode(tmp_path, 87011, 90102, "common", "synthetic", predict, factory, reference)
    assert case["status"] == "TERMINATED" and case["ticks"] == 2 and case["fallen"] and not case["clean"]
    assert "rollback_m" not in case and case["native_steps"] == 20 and env.closed
    with np.load(tmp_path / case["trace"], allow_pickle=False) as tape:
        assert tape["fallen"].tolist() == [False, True]
