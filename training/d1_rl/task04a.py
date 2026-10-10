"""Contract 04a: one added reward term on the otherwise frozen 03d task.

The 03b/03c/03d reward only ever sees instantaneous velocity, while the failing
gate measures the Euclidean XY displacement accumulated over the 4 s stop
window. This task adds exactly one term - a squared/Huber penalty on the
distance travelled away from the pose where the current zero-command segment
began - and changes nothing else. Command schedule, interface, observation,
action scales, timing, reset, termination and PPO stay 03d.

Deliberately not changed here, so that a pass or a fail is attributable:
no action-magnitude penalty, no retuned tracking scales, no yaw-hold term, no
teacher, no warm start, no true-velocity observation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .task03d import D1Task03dEnv, ROOT


DEFAULT_CONFIG_04A = ROOT / "training/configs/pilot04a.json"
BASE_CONFIG_03D = ROOT / "training/configs/pilot03d.json"
BASE_CONFIG_03D_SHA256 = "9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea"

CHANGE_FROM_PILOT03D = (
    "Only one reward term is added: standstill_anchor_displacement penalises the "
    "squared/Huber distance from the pose where the current zero-command segment "
    "began. Command schedule, reward weights and scales already present in 03d, "
    "interface, observation, action scales, timing, reset, termination and PPO "
    "are unchanged."
)


def _expected_04a_config() -> dict[str, Any]:
    """Allow exactly the one added reward term on the frozen 03d config."""
    if hashlib.sha256(BASE_CONFIG_03D.read_bytes()).hexdigest() != BASE_CONFIG_03D_SHA256:
        raise ValueError("Frozen pilot03d configuration SHA256 changed")
    expected = json.loads(BASE_CONFIG_03D.read_text(encoding="utf-8"))
    expected["version"] = "d1-pure-ppo-pilot-04a"
    expected["reward"]["standstill_anchor_displacement_weight"] = -20.0
    expected["reward"]["standstill_anchor_displacement_knee_m"] = 0.05
    expected["reward"]["standstill_anchor_displacement_saturation_m"] = 0.10
    del expected["change_from_pilot03c"]
    expected["change_from_pilot03d"] = CHANGE_FROM_PILOT03D
    return expected


def anchor_penalty_raw(distance_m: float, knee_m: float, saturation_m: float) -> float:
    """Huber in the gate's own unit: quadratic inside the gate, then linear.

    Quadratic below the knee keeps the gradient finite but shrinking as the
    robot settles, linear above it keeps a constant gradient over the whole
    observed failure range, and the saturation bound stops a long drift from
    making the episode return so negative that terminating early looks better
    than standing still.
    """
    if not knee_m > 0.0 or not saturation_m > knee_m:
        raise ValueError("Anchor penalty needs 0 < knee < saturation")
    distance = min(abs(float(distance_m)), saturation_m)
    if distance <= knee_m:
        return distance * distance / knee_m
    return 2.0 * distance - knee_m


class D1Task04aEnv(D1Task03dEnv):
    """03d plus a zero-command position anchor.

    The anchor is the base XY at the first policy interval of each zero-command
    segment, so it coincides exactly with the t=8 s origin that the fixed
    evaluation measures stop displacement from.
    """

    def __init__(self, config: str | Path | dict[str, Any] | None = None, **kwargs):
        super().__init__(config=DEFAULT_CONFIG_04A if config is None else config, **kwargs)
        self._anchor_xy = None
        self._previous_interval_was_zero = False
        self._interval_start_xy = np.zeros(2, dtype=np.float64)

    def _validate_config(self):
        expected = _expected_04a_config()
        if self.config != expected:
            raise ValueError("pilot04a must match the frozen 03d base and the single added term")
        config04a = self.config
        try:
            # Run the full 03d/03c/03b validator chain against the frozen 03d base.
            self.config = json.loads(BASE_CONFIG_03D.read_text(encoding="utf-8"))
            super()._validate_config()
        finally:
            self.config = config04a
        reward = self.config["reward"]
        if reward["standstill_anchor_displacement_weight"] >= 0.0:
            raise ValueError("The anchor term must be a penalty")
        anchor_penalty_raw(0.0, reward["standstill_anchor_displacement_knee_m"],
                           reward["standstill_anchor_displacement_saturation_m"])

    def _base_xy(self):
        return np.array(self.data.qpos[self.free_qpos:self.free_qpos + 2], dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        # No interval has run yet, so the reset pose is the start of the first one.
        self._anchor_xy = None
        self._previous_interval_was_zero = False
        self._interval_start_xy = self._base_xy()
        return observation, info

    def _reward(self, interval_command, previous_action, mean_tau_squared, illegal, self_contacts):
        reward, terms, metrics = super()._reward(
            interval_command, previous_action, mean_tau_squared, illegal, self_contacts)
        reward_config = self.config["reward"]
        # Gate on exactly the same condition as the inherited standstill term.
        zero_command = interval_command[0] == 0.0 and interval_command[2] == 0.0
        if zero_command and not self._previous_interval_was_zero:
            # _interval_start_xy still holds the pose before this interval ran.
            self._anchor_xy = self._interval_start_xy.copy()
        elif not zero_command:
            self._anchor_xy = None
        self._previous_interval_was_zero = zero_command
        current_xy = self._base_xy()
        if self._anchor_xy is None:
            distance = 0.0
            anchor_raw = 0.0
        else:
            distance = float(np.linalg.norm(current_xy - self._anchor_xy))
            anchor_raw = anchor_penalty_raw(
                distance, reward_config["standstill_anchor_displacement_knee_m"],
                reward_config["standstill_anchor_displacement_saturation_m"])
        self._interval_start_xy = current_xy
        terms["raw"]["standstill_anchor_displacement"] = anchor_raw
        terms["weights"]["standstill_anchor_displacement"] = (
            reward_config["standstill_anchor_displacement_weight"])
        contributions = terms["contributions_per_second"]
        contributions["standstill_anchor_displacement"] = (
            anchor_raw * reward_config["standstill_anchor_displacement_weight"])
        policy_dt = terms["policy_dt_s"]
        terms["contributions_per_tick"] = {
            name: value * policy_dt for name, value in contributions.items()
        }
        metrics["standstill_anchor_distance_m"] = distance
        metrics["standstill_anchor_active"] = self._anchor_xy is not None
        reward = float(sum(contributions.values()) * policy_dt)
        if not np.isfinite(reward):
            self._episode_done = True
            raise FloatingPointError("Non-finite pilot04a reward; abort this development run")
        return reward, terms, metrics
