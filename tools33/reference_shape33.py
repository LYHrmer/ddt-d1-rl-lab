"""Causal yaw reference shaping with the actor's exact past input history."""
import math

import numpy as np

GAIN = np.float32(1.5)
ALPHA = math.exp(-.02/.1)
YAW_COLUMNS = np.arange(11)*57+8


class ReferenceShaper:
    """One instance per reset; observation order is current, oldest..newest."""

    def __init__(self):
        self.state = None
        self.history = None
        self.raw_history = None

    def __call__(self, observation):
        obs = np.asarray(observation)
        if obs.shape != (627,) or obs.dtype != np.float32 or not np.isfinite(obs).all():
            raise ValueError('finite actor627 float32 required')
        raw = obs[YAW_COLUMNS]
        target = np.clip(raw*GAIN, -.25, .25)
        if self.state is None:
            if not np.all(raw == raw[0]):
                raise ValueError('reset requires constant prehistory')
            self.state = float(target[0])
            self.history = target[1:].copy()
            self.raw_history = raw[1:].copy()
        if not np.array_equal(raw[1:], self.raw_history):
            raise ValueError('raw command history is not the previous ten inputs')
        self.state = ALPHA*self.state + (1-ALPHA)*float(target[0])
        current = np.float32(self.state)
        result = obs.copy()
        result[YAW_COLUMNS[0]] = current
        result[YAW_COLUMNS[1:]] = self.history
        self.history = np.concatenate((self.history[1:], [current])).astype(np.float32)
        self.raw_history = np.concatenate((self.raw_history[1:], raw[:1]))
        return result
