"""Contract 03c command and reward task on the frozen fast D1 environment.

No reset, observation, action, PD, contact, or integration method changes here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .fast_checks import D1FastChecksEnv
from .interface import ACTION_DIM, DEFAULT_Q


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_03C = ROOT / "training/configs/pilot03c.json"
BASE_CONFIG_03B = ROOT / "training/configs/pilot03b.json"
BASE_CONFIG_03B_SHA256 = "3996e6e7c677be59e2852692d3f71ef7d0de8a689b6be3493f9ba466d92a589f"
WHEEL_INDICES = (3, 7, 11, 15)
LEG_INDICES = tuple(index for index in range(ACTION_DIM) if index not in WHEEL_INDICES)


def _expected_03c_config() -> dict[str, Any]:
    """Freeze every inherited field against the recorded 03b configuration."""
    if hashlib.sha256(BASE_CONFIG_03B.read_bytes()).hexdigest() != BASE_CONFIG_03B_SHA256:
        raise ValueError("Frozen pilot03b configuration SHA256 changed")
    expected = json.loads(BASE_CONFIG_03B.read_text(encoding="utf-8"))
    expected["version"] = "d1-pure-ppo-pilot-03c"
    expected["reward_contract"] = "astra-reviewed-03c-before-second-pilot"
    expected["commands"]["category_probabilities"] = {
        "stop": 0.2, "pure_vx": 0.3, "pure_wz": 0.3, "mixed": 0.2,
    }
    reward = expected["reward"]
    reward.update({
        "wz_tracking_weight": 2.0,
        "wz_tracking_error_scale_rad2_s2": 0.04,
        "gravity_xy_squared_weight": -6.0,
        "height_error_squared_weight": -20.0,
        "action_rate_mean_squared_weight": -0.1,
        "leg_pose_deviation_sum_squared_weight": -0.5,
        "standstill_tracking_weight": 1.0,
        "standstill_linear_error_scale_m2_s2": 0.0004,
        "standstill_yaw_error_scale_rad2_s2": 0.0025,
    })
    del expected["change_from_pilot03"]
    expected["change_from_pilot03b"] = (
        "Only command-category sampling and the explicitly frozen 03c reward "
        "terms/weights change; interface, action scales, timing, reset, and "
        "termination remain pilot03b."
    )
    return expected


class D1Task03cEnv(D1FastChecksEnv):
    """03c pure PPO task with the existing 03b physical and actor interface."""

    def __init__(self, config: str | Path | dict[str, Any] | None = None, **kwargs):
        super().__init__(config=DEFAULT_CONFIG_03C if config is None else config, **kwargs)

    def _validate_config(self):
        if self.config != _expected_03c_config():
            raise ValueError("pilot03c must match its frozen 03b base and 03c task changes")
        # The inherited validator checks the 03b interface and physics contract.
        # It gates on the old config version, so present only that version during
        # validation, restoring the real 03c identity before model construction.
        config03c = self.config
        config_for_interface = dict(config03c)
        config_for_interface["version"] = "d1-pure-ppo-pilot-03b"
        try:
            self.config = config_for_interface
            super()._validate_config()
        finally:
            self.config = config03c

    def _update_command(self):
        # Preserve manual/evaluation precedence and the first 1 s zero command.
        if (self._manual_command or self.eval_command_sequence is not None or
                self._episode_steps < self.first_zero_ticks):
            return super()._update_command()
        phase = (self._episode_steps - self.first_zero_ticks) // self.command_hold_ticks
        if phase == self._command_phase:
            return
        self._command_phase = phase
        command_config = self.config["commands"]
        probability = command_config["category_probabilities"]
        category_draw = self._command_rng.random()
        stop_end = probability["stop"]
        vx_end = stop_end + probability["pure_vx"]
        wz_end = vx_end + probability["pure_wz"]
        if category_draw < stop_end:
            self._command[:] = 0.0
        elif category_draw < vx_end:
            self._command[:] = (self._command_rng.uniform(*command_config["vx_uniform_m_s"]),
                                0.0, 0.0)
        elif category_draw < wz_end:
            self._command[:] = (0.0, 0.0,
                                self._command_rng.uniform(*command_config["wz_uniform_rad_s"]))
        else:
            self._command[:] = (self._command_rng.uniform(*command_config["vx_uniform_m_s"]),
                                0.0,
                                self._command_rng.uniform(*command_config["wz_uniform_rad_s"]))

    def _reward(self, interval_command, previous_action, mean_tau_squared, illegal, self_contacts):
        _, terms, metrics = super()._reward(
            interval_command, previous_action, mean_tau_squared, illegal, self_contacts)
        reward_config = self.config["reward"]
        actual_joints = self.data.qpos[self.qpos_ids]
        leg_error = actual_joints[list(LEG_INDICES)] - DEFAULT_Q[list(LEG_INDICES)]
        leg_pose_squared = float(np.dot(leg_error, leg_error))
        if interval_command[0] == 0.0 and interval_command[2] == 0.0:
            standstill = math.exp(
                -(metrics["body_vx_m_s"] ** 2 + metrics["body_vy_m_s"] ** 2) /
                reward_config["standstill_linear_error_scale_m2_s2"] -
                metrics["body_wz_rad_s"] ** 2 /
                reward_config["standstill_yaw_error_scale_rad2_s2"])
        else:
            standstill = 0.0
        terms["raw"].update({
            "leg_pose_deviation_sum_squared": leg_pose_squared,
            "standstill_tracking": standstill,
        })
        terms["weights"].update({
            "leg_pose_deviation_sum_squared": reward_config["leg_pose_deviation_sum_squared_weight"],
            "standstill_tracking": reward_config["standstill_tracking_weight"],
        })
        contributions = terms["contributions_per_second"]
        for name in ("leg_pose_deviation_sum_squared", "standstill_tracking"):
            contributions[name] = terms["raw"][name] * terms["weights"][name]
        policy_dt = terms["policy_dt_s"]
        terms["contributions_per_tick"] = {
            name: value * policy_dt for name, value in contributions.items()
        }
        reward = float(sum(contributions.values()) * policy_dt)
        if not math.isfinite(reward):
            self._episode_done = True
            raise FloatingPointError("Non-finite pilot03c reward; abort this development run")
        return reward, terms, metrics
