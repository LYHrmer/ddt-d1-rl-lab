"""Cached-label alignment with pure arrays/fake Gym envs; zero models/physics."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"training"))
from d1_rl.velocity_labels15 import (
    VELOCITY_LABEL_CONTRACT,
    VelocityLabelWrapper,
    label_observation,
)


def actor(value=0.):
    return np.full(627, value, dtype=np.float32)


def yaw_quat(yaw):
    return np.array([math.cos(yaw/2), 0., 0., math.sin(yaw/2)])


class SensorSequenceEnv(gym.Env):
    """No simulator: each base method sets a distinct sensor-cache boundary."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(627,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., shape=(16,), dtype=np.float32)
    def __init__(self, *, end="terminated", terminal_info=True):
        self.metadata = {"render_modes": []}
        self.end, self.terminal_info = end, terminal_info
        self.episode = -1
        self.calls = []
        self.sensor_reads = []

    def _boundary(self, marker):
        self.marker = marker
        self.current_actor = actor(marker)
        self.quat = yaw_quat(math.pi/2)
        self.velocity = np.array([marker, 2*marker, -marker], dtype=np.float64)

    def _sensor(self, name):
        self.sensor_reads.append((name, self.marker))
        if name == "trunk_quat":
            return self.quat.copy()
        if name == "frame_vel":
            return self.velocity.copy()
        raise AssertionError(name)

    def _observe(self):
        raise AssertionError("The label wrapper must never refresh sensors")

    def reset(self, *, seed=None, options=None):
        self.calls.append(("reset", seed, options))
        self.episode += 1
        self._boundary(float(10+self.episode))
        self.last_info = {"episode": self.episode, "nested": {"unchanged": True}}
        return self.current_actor, self.last_info

    def step(self, action):
        self.calls.append(("step", action))
        self._boundary(float(20+self.episode))
        self.last_info = {"marker": self.marker, "nested": {"unchanged": True}}
        terminated, truncated = self.end == "terminated", self.end == "truncated"
        if self.terminal_info and (terminated or truncated):
            self.last_info["terminal_observation"] = self.current_actor.copy()
        self.last_reward = np.float64(1.25)
        return self.current_actor, self.last_reward, terminated, truncated, self.last_info


def test_nonzero_yaw_rotates_world_velocity_without_actor_scaling_or_mutation():
    observation = np.arange(627, dtype=np.float32)
    quat, world = yaw_quat(math.pi/2), np.array([1., 2., 3.])
    before = [array.copy() for array in (observation, quat, world)]
    result = label_observation(observation, quat, world)
    assert result["actor"].dtype == result["velocity_target"].dtype == np.float32
    np.testing.assert_array_equal(result["actor"], observation)
    np.testing.assert_allclose(result["velocity_target"], [2., -1., 3.], atol=1e-7)
    for old, original in zip(before, (observation, quat, world), strict=True):
        np.testing.assert_array_equal(original, old)
    result["actor"][0] = -999
    result["velocity_target"][0] = -999
    assert observation[0] == 0 and world[0] == 1
    assert VELOCITY_LABEL_CONTRACT["velocity_target"]["units"] == "m/s"
    assert VELOCITY_LABEL_CONTRACT["source_quaternion"]["order"] == "wxyz_body_to_world"


@pytest.mark.parametrize("fault", ["actor_shape", "actor_dtype", "actor_nan", "quat_shape",
                                  "quat_zero", "quat_inf", "velocity_shape", "velocity_nan", "overflow"])
def test_invalid_or_unrepresentable_labels_are_rejected(fault):
    observation, quat, world = actor(), yaw_quat(0), np.zeros(3)
    if fault == "actor_shape": observation = np.zeros(626, dtype=np.float32)
    if fault == "actor_dtype": observation = observation.astype(np.float64)
    if fault == "actor_nan": observation[0] = np.nan
    if fault == "quat_shape": quat = np.ones(3)
    if fault == "quat_zero": quat[:] = 0
    if fault == "quat_inf": quat[0] = np.inf
    if fault == "velocity_shape": world = np.zeros(2)
    if fault == "velocity_nan": world[0] = np.nan
    if fault == "overflow": world[0] = 1e100
    with pytest.raises(ValueError):
        label_observation(observation, quat, world)


def test_reset_and_nonterminal_step_read_each_current_cache_once_and_keep_base_info():
    base = SensorSequenceEnv(end="none")
    wrapper = VelocityLabelWrapper(base)
    observation, info = wrapper.reset(seed=31, options={"a": 1})
    assert info is base.last_info
    np.testing.assert_allclose(observation["velocity_target"], [20., -10., -10.])
    assert base.calls == [("reset", 31, {"a": 1})]
    action = np.zeros(16, dtype=np.float32)
    observation, reward, terminated, truncated, info = wrapper.step(action)
    assert base.calls[-1][1] is action and reward is base.last_reward
    assert (terminated, truncated) == (False, False) and info is base.last_info
    assert wrapper.observation_space.contains(observation)
    np.testing.assert_array_equal(observation["actor"], actor(20.))
    np.testing.assert_allclose(observation["velocity_target"], [40., -20., -20.])
    assert base.sensor_reads == [("trunk_quat", 10.), ("frame_vel", 10.),
                                 ("trunk_quat", 20.), ("frame_vel", 20.)]
    base.current_actor[:] = -1
    assert np.all(observation["actor"] == 20)


@pytest.mark.parametrize("end", ["terminated", "truncated"])
def test_terminal_info_is_copied_and_transformed_at_the_terminal_boundary(end):
    base = SensorSequenceEnv(end=end)
    wrapper = VelocityLabelWrapper(base)
    wrapper.reset()
    observation, reward, terminated, truncated, info = wrapper.step(np.zeros(16))
    assert reward is base.last_reward
    assert (terminated, truncated) == (end == "terminated", end == "truncated")
    assert info is not base.last_info and info["nested"] is base.last_info["nested"]
    assert isinstance(base.last_info["terminal_observation"], np.ndarray)
    for key in ("actor", "velocity_target"):
        np.testing.assert_array_equal(info["terminal_observation"][key], observation[key])
        assert not np.shares_memory(info["terminal_observation"][key], observation[key])
    info["terminal_observation"]["actor"][:] = -1
    assert np.all(base.last_info["terminal_observation"] == 20)


def test_a_different_terminal_frame_is_refused_instead_of_mislabelled():
    class WrongTerminal(SensorSequenceEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info["terminal_observation"] = actor(-4.)
            return observation, reward, terminated, truncated, info

    wrapper = VelocityLabelWrapper(WrongTerminal())
    wrapper.reset()
    with pytest.raises(ValueError, match="misaligned"):
        wrapper.step(np.zeros(16))


@pytest.mark.parametrize("end", ["terminated", "truncated"])
@pytest.mark.parametrize("terminal_info", [True, False])
def test_outer_dummy_vecenv_autoreset_keeps_distinct_terminal_and_reset_targets(end, terminal_info):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    base = SensorSequenceEnv(end=end, terminal_info=terminal_info)
    vector = DummyVecEnv([lambda: VelocityLabelWrapper(base)])
    try:
        reset = vector.reset()
        np.testing.assert_allclose(reset["velocity_target"][0], [20., -10., -10.])
        observation, rewards, dones, infos = vector.step(np.zeros((1, 16), dtype=np.float32))
        assert dones.tolist() == [True] and rewards.tolist() == [1.25]
        # The vector return is episode 1's reset; info retains episode 0's terminal.
        np.testing.assert_array_equal(observation["actor"][0], actor(11.))
        np.testing.assert_allclose(observation["velocity_target"][0], [22., -11., -11.])
        terminal = infos[0]["terminal_observation"]
        np.testing.assert_array_equal(terminal["actor"], actor(20.))
        np.testing.assert_allclose(terminal["velocity_target"], [40., -20., -20.])
        assert infos[0]["TimeLimit.truncated"] is (end == "truncated")
        assert [name for name, _ in base.sensor_reads] == ["trunk_quat", "frame_vel"]*3
    finally:
        vector.close()


@pytest.mark.parametrize("bad_space", [gym.spaces.Box(-1., 1., (57,), dtype=np.float32),
                                      gym.spaces.Box(-1., 1., (627,), dtype=np.float64)])
def test_wrapper_refuses_a_different_actor_contract_before_sensor_reads(bad_space):
    base = SensorSequenceEnv()
    base.observation_space = bad_space
    with pytest.raises(ValueError, match="observation space"):
        VelocityLabelWrapper(base)
    assert base.sensor_reads == [] and base.calls == []
