"""Versioned 1.5 m/s flat course, native whole-body actor and wheel-reference diagnostic."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from tools87.terrain87 import D1Terrain87, POLICY_HZ, TERRAIN_DIM
from training.d1_rl.delay_env18 import D1Delay18Env
from training.d1_rl.interface import ACTOR_DIM, KD, LEG_INDICES, WHEEL_INDICES
from training.d1_rl.task03d import DEFAULT_CONFIG_03D


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/speed109_proposal.md"
CHECKPOINT = ROOT / "evidence/uphill89/training_01/mixed_89011/policy.zip"
CHECKPOINT_SHA = "9663c97d27ad307fb71ca2efbadf0fa3e1fa7c9e87b8b0e40d8a05f22ab8b4c0"
DEV = (109011, 109023, 109037)
BLIND = (109101, 109103, 109107)
ARM_SCALE = {"actor5": 5., "actor24": 24., "reference24": 24.}
TICKS = 8 * POLICY_HZ
RADIUS = .087


def command_at(tick: int) -> float:
    """Command at the beginning of this policy interval on the fixed course."""
    if not 0 <= tick <= TICKS:
        raise ValueError("tick outside the 8 s course")
    return float(np.interp(tick, [0, 50, 150, 275, 350, TICKS], [0., 0., 1.5, 1.5, 0., 0.]))


def encode_forward_command(frame, command):
    if not np.isfinite(command) or abs(command) > 1.5 or frame.shape != (57,):
        raise ValueError("invalid speed109 forward command or frame")
    # build_observation clips before scaling. Preserve the rest of the original sensor frame.
    frame[6] = np.float32(2. * command)
    return frame


def issued_action(arm, nominal, command):
    """Only the diagnostic arm supplies wheel targets outside the frozen actor."""
    if arm not in ARM_SCALE or not 0. <= command <= 1.5:
        raise ValueError("unregistered control arm or speed")
    action = np.asarray(nominal, dtype=np.float32).copy()
    if action.shape != (16,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
        raise ValueError("invalid frozen actor output")
    if arm == "reference24":
        wheel_rate = 1.02 * command / RADIUS
        drag = .1 * wheel_rate + .2 * np.sign(wheel_rate)
        target = wheel_rate + drag / KD[WHEEL_INDICES]
        if np.any(np.abs(target) > ARM_SCALE[arm]):
            raise ValueError("registered wheel reference exceeds the action range")
        action[WHEEL_INDICES] = target / ARM_SCALE[arm]
    return action


class FlatCruise109(D1Delay18Env):
    """Original D1 physics on the frozen unlimited plane with the 87 onboard range clock."""

    _measure = D1Terrain87._measure
    _update_measurement = D1Terrain87._update_measurement

    def __init__(self, arm, seed):
        if arm not in ARM_SCALE:
            raise ValueError("unregistered control arm")
        super().__init__(config=DEFAULT_CONFIG_03D, seed=seed, arm="B", eval_delay_ticks=1,
                         max_transitions=TICKS)
        self.experiment_arm = arm
        self.episode_ticks = TICKS
        self.action_scales = self.action_scales.copy()
        self.action_scales[WHEEL_INDICES] = ARM_SCALE[arm]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (ACTOR_DIM + TERRAIN_DIM,), dtype=np.float32)
        self.base_body = self.model.body("base_link").id
        self.last_raw_features = np.zeros(TERRAIN_DIM, dtype=np.float32)
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self._pending = None
        wheels = self.dof_ids[WHEEL_INDICES]
        if (self.model.geom_type[self.model.geom("floor").id] != mujoco.mjtGeom.mjGEOM_PLANE
                or not np.array_equal(self.action_scales[LEG_INDICES], np.full(12, .5))
                or not np.allclose(self.model.geom_size[sorted(self.wheel_geoms), 0], RADIUS)
                or not np.allclose(self.model.dof_damping[wheels], .1)
                or not np.allclose(self.model.dof_frictionloss[wheels], .2)):
            raise ValueError("D1 wheel, leg scale or flat scene differs from the speed109 protocol")

    def _update_command(self):
        self._command[:] = (command_at(self._episode_steps), 0., 0.)

    def _observe(self):
        return encode_forward_command(super()._observe(), self._command[0])

    def set_command(self, vx, wz):
        raise ValueError("speed109 follows the registered ramp via command_at()")

    def get_initial_state(self):
        state = super().get_initial_state()
        state.update(interface_id="d1-flat-cruise109", control_arm=self.experiment_arm)
        return state

    def current_observation(self):
        return np.concatenate((super().current_observation(), self.last_raw_features))

    def reset(self, *, seed=None, options=None):
        self._pending = None
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self.last_raw_features = np.zeros(TERRAIN_DIM, dtype=np.float32)
        obs, info = super().reset(seed=seed, options=options)
        self._update_measurement(0)
        return np.concatenate((obs, self.last_raw_features)), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_measurement(self._episode_steps)
        full = np.concatenate((obs, self.last_raw_features))
        if self._episode_done:
            info["terminal_observation"] = full.copy()
        return full, reward, terminated, truncated, info
