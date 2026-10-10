"""Contract 04b: one added reward term on the otherwise frozen 03d task.

04a added a penalty on net XY displacement from the pose where each
zero-command segment began. Its primary prediction held - turn_stop stop
displacement fell from 0.0604 m to 0.0349 m, the net zero-command wheel bias
fell from +0.460 to +0.115 rad/s, and the policy started braking actively for
the first time - but the actor observes neither its position nor its linear
velocity, so the only memoryless way it could hold a position was to dither the
net wheel target at 12 Hz (std 0.976 rad/s against 03d's 0.034-0.060). That
dither broke the last-second speed gate on all three tasks.

04b therefore keeps the same slot, gating and Huber shape and changes only the
measured quantity, to the one whose optimum is both observable and memoryless:
the magnitude of the four wheel actions. With kd=0.5, r=0.087 m and a 48.15 kg
robot, a wheel action of zero is a pure damper that brakes from 0.1965 m/s in
about 0.038 m and then holds still, which satisfies the 0.05 m displacement
gate and the 0.02 m/s last-second speed gate at once. The same term penalises
03d's constant bias and 04a's dither.

04b is a sibling of 04a, not a successor: both add exactly one term to the same
03d base, so comparing them isolates the choice of measured quantity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .task03d import D1Task03dEnv, ROOT


DEFAULT_CONFIG_04B = ROOT / "training/configs/pilot04b.json"
BASE_CONFIG_03D = ROOT / "training/configs/pilot03d.json"
BASE_CONFIG_03D_SHA256 = "9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea"
WHEEL_ACTION_INDEX = (3, 7, 11, 15)

CHANGE_FROM_PILOT03D = (
    "Only one reward term is added: wheel_action_mean_squared penalises the mean "
    "square of the four wheel actions while the commanded vx and wz are both zero. "
    "Command schedule, reward terms and weights already present in 03d, interface, "
    "observation, action scales, timing, reset, termination and PPO are unchanged."
)


def _expected_04b_config() -> dict[str, Any]:
    """Allow exactly the one added reward term on the frozen 03d config."""
    if hashlib.sha256(BASE_CONFIG_03D.read_bytes()).hexdigest() != BASE_CONFIG_03D_SHA256:
        raise ValueError("Frozen pilot03d configuration SHA256 changed")
    expected = json.loads(BASE_CONFIG_03D.read_text(encoding="utf-8"))
    expected["version"] = "d1-pure-ppo-pilot-04b"
    expected["reward"]["wheel_action_mean_squared_weight"] = -10.0
    expected["reward"]["wheel_action_mean_squared_saturation"] = 0.25
    del expected["change_from_pilot03c"]
    expected["change_from_pilot03d"] = CHANGE_FROM_PILOT03D
    return expected


def wheel_action_penalty_raw(wheel_actions, saturation: float) -> float:
    """Mean square of the four wheel actions, bounded.

    The bound keeps the worst case below the per-second positive reward floor,
    so a long drift can never make ending the episode early look cheaper than
    standing still. It sits above the whole measured range instead of inside
    it: the per-tick mean square is 0.055-0.065 for 03d's constant opposed
    pattern, 0.174 for 04a's 12 Hz dither and about 0.135 at initialisation
    with log_std_init=-1, so saturating at 0.25 leaves the gradient live
    everywhere the policy actually operates.
    """
    if not 0.0 < saturation <= 1.0:
        raise ValueError("Wheel action penalty saturation must lie in (0, 1]")
    values = np.asarray(wheel_actions, dtype=np.float64)
    if values.shape != (len(WHEEL_ACTION_INDEX),):
        raise ValueError("Expected exactly four wheel actions")
    return float(min(float(np.mean(np.square(values))), saturation))


class D1Task04bEnv(D1Task03dEnv):
    """03d plus a zero-command wheel action magnitude penalty."""

    def __init__(self, config: str | Path | dict[str, Any] | None = None, **kwargs):
        super().__init__(config=DEFAULT_CONFIG_04B if config is None else config, **kwargs)

    def _validate_config(self):
        expected = _expected_04b_config()
        if self.config != expected:
            raise ValueError("pilot04b must match the frozen 03d base and the single added term")
        config04b = self.config
        try:
            # Run the full 03d/03c/03b validator chain against the frozen 03d base.
            self.config = json.loads(BASE_CONFIG_03D.read_text(encoding="utf-8"))
            super()._validate_config()
        finally:
            self.config = config04b
        reward = self.config["reward"]
        if reward["wheel_action_mean_squared_weight"] >= 0.0:
            raise ValueError("The wheel action term must be a penalty")
        worst = reward["wheel_action_mean_squared_saturation"] * abs(
            reward["wheel_action_mean_squared_weight"])
        floor = (reward["vx_tracking_weight"] + reward["wz_tracking_weight"]
                 + reward["alive_weight"]) * 0.95
        if worst >= floor:
            raise ValueError("Worst-case wheel action penalty must stay below the positive floor")

    def _reward(self, interval_command, previous_action, mean_tau_squared, illegal, self_contacts):
        reward, terms, metrics = super()._reward(
            interval_command, previous_action, mean_tau_squared, illegal, self_contacts)
        reward_config = self.config["reward"]
        # Same gate as the inherited standstill term, and the same issued clipped
        # action the inherited action_rate term uses.
        zero_command = interval_command[0] == 0.0 and interval_command[2] == 0.0
        wheel_actions = self._last_policy_action[list(WHEEL_ACTION_INDEX)]
        raw = (wheel_action_penalty_raw(wheel_actions,
                                        reward_config["wheel_action_mean_squared_saturation"])
               if zero_command else 0.0)
        terms["raw"]["wheel_action_mean_squared"] = raw
        terms["weights"]["wheel_action_mean_squared"] = (
            reward_config["wheel_action_mean_squared_weight"])
        contributions = terms["contributions_per_second"]
        contributions["wheel_action_mean_squared"] = (
            raw * reward_config["wheel_action_mean_squared_weight"])
        policy_dt = terms["policy_dt_s"]
        terms["contributions_per_tick"] = {
            name: value * policy_dt for name, value in contributions.items()
        }
        # Diagnostics for the offline comparison against 03d and 04a.
        metrics["wheel_action_mean_squared"] = raw
        metrics["wheel_action_term_active"] = bool(zero_command)
        metrics["net_wheel_target_rad_s"] = float(
            np.mean(wheel_actions) * self.action_scales[WHEEL_ACTION_INDEX[0]])
        reward = float(sum(contributions.values()) * policy_dt)
        if not np.isfinite(reward):
            self._episode_done = True
            raise FloatingPointError("Non-finite pilot04b reward; abort this development run")
        return reward, terms, metrics
