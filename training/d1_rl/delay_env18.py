"""New delay-only training variant of the unchanged, frozen 03d task.

Only the per-reset execution delay changes. The observation, reward, action
mapping, command course, reset physics, and integration remain inherited.
The private delay stream cannot advance the command or reset random streams.
"""
from __future__ import annotations

import numpy as np

from .task03d import D1Task03dEnv


DELAY_STREAM18 = 18018
TRAINING_DELAY_MODES18 = ("zero", "uniform_0_3")


class D1Delay18Env(D1Task03dEnv):
    """Use exactly zero or episode-uniform 0/1/2/3 policy ticks in training.

    A fixed evaluation delay has precedence. It is a Python integer from zero
    through three, as in the registered formal scenario matrix. A policy tick
    remains 20 ms; selecting a delay does not change any control clock.
    """

    def __init__(self, config=None, *, training_delay_mode="zero", **kwargs):
        if type(training_delay_mode) is not str or training_delay_mode not in TRAINING_DELAY_MODES18:
            raise ValueError("training_delay_mode must be zero or uniform_0_3")
        if kwargs.get("arm", "B") != "B":
            raise ValueError("delay18 uses only the original B arm with an explicit delay mode")
        fixed = kwargs.get("eval_delay_ticks")
        if fixed is not None and (type(fixed) is not int or fixed not in (0, 1, 2, 3)):
            raise ValueError("fixed evaluation delay must be a Python int in 0, 1, 2, 3")
        self.training_delay_mode = training_delay_mode
        self._delay_counts18 = dict.fromkeys(("0", "1", "2", "3"), 0)
        self._last_delay18 = None
        super().__init__(config=config, **kwargs)

    def _pick_delay(self):
        mode = self.training_delay_mode
        if mode not in TRAINING_DELAY_MODES18:
            raise ValueError("delay18 training mode changed")
        # The inherited reset has already incremented _episode_index once.
        episode = self._episode_index - 1
        if episode < 0:
            raise RuntimeError("delay18 selection requires an inherited reset")
        fixed = self.eval_delay_ticks
        if fixed is not None:
            if type(fixed) is not int or fixed not in (0, 1, 2, 3):
                raise ValueError("fixed evaluation delay must be a Python int in 0, 1, 2, 3")
            selected, selection = fixed, "fixed_eval"
        elif mode == "zero":
            selected, selection = 0, "zero"
        else:
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.base_seed, self.rank, episode, DELAY_STREAM18]))
            selected, selection = int(rng.integers(0, 4)), "uniform_0_3"
        self._delay_counts18[str(selected)] += 1
        self._last_delay18 = {"mode": mode, "seed": self.base_seed, "rank": self.rank,
                              "episode_index": episode, "stream": DELAY_STREAM18,
                              "selected_delay_ticks": selected, "selection": selection}
        return selected

    def get_delay_state18(self):
        """Return JSON-ready sampling provenance without consuming any RNG."""
        return {"last_selection": None if self._last_delay18 is None else dict(self._last_delay18),
                "delay_counts": dict(self._delay_counts18)}
