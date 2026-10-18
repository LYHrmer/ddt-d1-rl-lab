"""Causal action notch: fixed 5 Hz at 50 Hz, independent state per episode.

RBJ notch coefficients, normalized by a0; see W3C Audio EQ Cookbook.
The first sample initializes a constant prehistory, avoiding an invented
zero-action transient. Internal recurrence retains the unclipped output;
only the issued action is clipped. No observations or future samples enter.
"""
from __future__ import annotations

import math
from numbers import Real

SAMPLE_HZ = 50.0
CENTER_HZ = 5.0
QUALITY = 2.0
CHANNELS = 16


def coefficients(sample_hz=SAMPLE_HZ, center_hz=CENTER_HZ, quality=QUALITY):
    values = (sample_hz, center_hz, quality)
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) for x in values):
        raise ValueError('finite real filter parameters required')
    if not (sample_hz > 0 and 0 < center_hz < sample_hz / 2 and quality > 0):
        raise ValueError('invalid notch frequency or quality')
    angle = 2 * math.pi * center_hz / sample_hz
    alpha = math.sin(angle) / (2 * quality)
    a0 = 1 + alpha
    return (1/a0, -2*math.cos(angle)/a0, 1/a0), (1.0, -2*math.cos(angle)/a0, (1-alpha)/a0)


class NotchFilter30:
    def __init__(self):
        self.b, self.a = coefficients()
        self.reset()

    def reset(self):
        self._x1 = self._x2 = self._y1 = self._y2 = None
        self.last_linear = None
        self.calls = self.clipped_components = 0

    def step(self, action):
        x = tuple(action)
        if len(x) != CHANNELS or any(isinstance(v, bool) or not isinstance(v, Real)
                                    or not math.isfinite(v) or abs(v) > 1 for v in x):
            raise ValueError('sixteen finite clipped actor outputs required')
        x = tuple(float(v) for v in x)
        if self._x1 is None:
            self._x1 = self._x2 = self._y1 = self._y2 = x
            y = x
        else:
            b0, b1, b2 = self.b
            _, a1, a2 = self.a
            y = tuple(b0*v + b1*p + b2*q - a1*r - a2*s
                      for v, p, q, r, s in zip(x, self._x1, self._x2, self._y1, self._y2))
        if not all(math.isfinite(v) for v in y):
            raise ValueError('nonfinite filter state')
        issued = tuple(max(-1., min(1., v)) for v in y)
        self.clipped_components += sum(v != w for v, w in zip(y, issued))
        self._x2, self._x1, self._y2, self._y1 = self._x1, x, self._y1, y
        self.last_linear = y
        self.calls += 1
        return issued

    def receipt(self):
        return {'sample_hz': SAMPLE_HZ, 'center_hz': CENTER_HZ, 'quality': QUALITY,
                'channels': CHANNELS, 'b': list(self.b), 'a': list(self.a),
                'calls': self.calls, 'clipped_components': self.clipped_components,
                'initialization': 'constant_first_action_prehistory',
                'state_uses': 'unclipped_float64_filter_output',
                'output': 'clipped_to_original_action_bounds_before_float32_issue'}
