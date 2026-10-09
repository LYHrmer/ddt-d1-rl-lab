"""Frozen 03d paired-command curriculum on the unchanged 03c D1 task.

Only training command generation differs. No physical, reward, observation,
action, reset, or integration method is overridden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .task03c import D1Task03cEnv, ROOT


DEFAULT_CONFIG_03D = ROOT / "training/configs/pilot03d.json"
BASE_CONFIG_03C = ROOT / "training/configs/pilot03c.json"
BASE_CONFIG_03C_SHA256 = "562bdc6b26fba7bbb4906f7045ba11cbe70848a3335bd8d5e1659540a3f2b604"


def _expected_03d_config() -> dict[str, Any]:
    """Allow exactly the command-course change to the frozen 03c config."""
    if hashlib.sha256(BASE_CONFIG_03C.read_bytes()).hexdigest() != BASE_CONFIG_03C_SHA256:
        raise ValueError("Frozen pilot03c configuration SHA256 changed")
    expected = json.loads(BASE_CONFIG_03C.read_text(encoding="utf-8"))
    expected["version"] = "d1-pure-ppo-pilot-03d"
    del expected["commands"]["category_probabilities"]
    expected["commands"]["paired_schedule"] = {
        "type": "paired_translation_turn_v1",
        "episode_ticks": 1000,
        "first_zero_ticks": 50,
        "mode_probabilities": {"translation": 0.5, "turn": 0.5},
        "positive_sign_probability": 0.5,
        "translation": {
            "amplitude_m_s": [0.1, 0.3],
            "motion_ticks_inclusive": [200, 300],
            "middle_zero_ticks_inclusive": [100, 150],
        },
        "turn": {
            "amplitude_rad_s": [0.1, 0.3],
            "motion_ticks_inclusive": [100, 150],
            "middle_zero_ticks_inclusive": [100, 150],
        },
    }
    del expected["change_from_pilot03b"]
    expected["change_from_pilot03c"] = (
        "Only the training command schedule changes to paired translation/turn "
        "with guaranteed motion-to-zero transitions; reward, interface, action "
        "scales, timing, reset, termination, and PPO remain pilot03c."
    )
    return expected


class D1Task03dEnv(D1Task03cEnv):
    """03d course: paired sign-balanced commands, sampled once per episode.

    The inherited resample_seconds, zero_probability, and uniform command
    fields remain in the config for parent-contract validation only. Training
    uses paired_schedule exclusively; it does not sample every four seconds.
    """

    def __init__(self, config: str | Path | dict[str, Any] | None = None, **kwargs):
        super().__init__(config=DEFAULT_CONFIG_03D if config is None else config, **kwargs)
        self._training_schedule = None
        self._schedule_episode_index = None
        self._schedule_rng = None

    def _validate_config(self):
        expected = _expected_03d_config()
        if self.config != expected:
            raise ValueError("pilot03d must match the frozen 03c base and 03d command course")
        config03d = self.config
        try:
            # Exercise both the 03c task and the 03b physical/interface validators.
            self.config = json.loads(BASE_CONFIG_03C.read_text(encoding="utf-8"))
            super()._validate_config()
        finally:
            self.config = config03d
        if self.episode_ticks != 1000 or self.first_zero_ticks != 50:
            raise ValueError("03d requires exactly 1000 episode ticks and 50 initial zero ticks")

    def _make_training_schedule(self):
        spec = self.config["commands"]["paired_schedule"]
        rng = self._command_rng
        mode = "translation" if rng.random() < spec["mode_probabilities"]["translation"] else "turn"
        sign = 1 if rng.random() < spec["positive_sign_probability"] else -1
        start = spec["first_zero_ticks"]
        segments = [(0, start, (0.0, 0.0, 0.0))]

        def add(duration: int, vx: float = 0.0, wz: float = 0.0):
            nonlocal start
            end = start + duration
            segments.append((start, end, (vx, 0.0, wz)))
            start = end

        if mode == "translation":
            axis = spec["translation"]
            amplitude = float(rng.uniform(*axis["amplitude_m_s"]))
            duration = int(rng.integers(axis["motion_ticks_inclusive"][0],
                                        axis["motion_ticks_inclusive"][1] + 1))
            gap = int(rng.integers(axis["middle_zero_ticks_inclusive"][0],
                                   axis["middle_zero_ticks_inclusive"][1] + 1))
            add(duration, vx=sign * amplitude)
            add(gap)
            add(duration, vx=-sign * amplitude)
            parameters = {"amplitudes": [amplitude], "motion_ticks": [duration, duration],
                          "middle_zero_ticks": gap}
        else:
            axis = spec["turn"]
            amplitude1 = float(rng.uniform(*axis["amplitude_rad_s"]))
            amplitude2 = float(rng.uniform(*axis["amplitude_rad_s"]))
            duration1 = int(rng.integers(axis["motion_ticks_inclusive"][0],
                                         axis["motion_ticks_inclusive"][1] + 1))
            duration2 = int(rng.integers(axis["motion_ticks_inclusive"][0],
                                         axis["motion_ticks_inclusive"][1] + 1))
            gap = int(rng.integers(axis["middle_zero_ticks_inclusive"][0],
                                   axis["middle_zero_ticks_inclusive"][1] + 1))
            add(duration1, wz=sign * amplitude1)
            add(duration1, wz=-sign * amplitude1)
            add(gap)
            add(duration2, wz=-sign * amplitude2)
            add(duration2, wz=sign * amplitude2)
            parameters = {"amplitudes": [amplitude1, amplitude2],
                          "motion_ticks": [duration1, duration1, duration2, duration2],
                          "middle_zero_ticks": gap}
        add(spec["episode_ticks"] - start)
        if start != spec["episode_ticks"] or segments[-1][1] != spec["episode_ticks"]:
            raise ValueError("03d command schedule must cover exactly one episode")
        return {"mode": mode, "sign": sign, "parameters": parameters,
                "segments": tuple(segments)}

    def _update_command(self):
        # These overrides retain the original precedence and never draw course RNG.
        if self._manual_command or self.eval_command_sequence is not None:
            return super()._update_command()
        if (self._schedule_episode_index != self._episode_index or
                self._schedule_rng is not self._command_rng):
            if self._episode_steps != 0:
                raise RuntimeError("03d training schedule missing after episode start")
            self._training_schedule = self._make_training_schedule()
            self._schedule_episode_index = self._episode_index
            self._schedule_rng = self._command_rng
        tick = self._episode_steps
        if tick >= self.episode_ticks:
            self._command[:] = 0.0
            return
        for begin, end, command in self._training_schedule["segments"]:
            if begin <= tick < end:
                self._command[:] = command
                return
        raise RuntimeError(f"03d command schedule has no segment for tick {tick}")
