"""Longitudinal speed correction for the 1.5 m/s course and its braking ramp."""
from statistics import median

from tools112.velocity import VelocityTrim


class BrakeTrim:
    def __init__(self):
        self.cruise = VelocityTrim()
        self.trim = 0.
        self.encoder_forward_m_s = 0.

    def update(self, request, last_actual, wheels):
        """Use the previous corrected actor request and the current four encoders."""
        cruise_trim = self.cruise.update(request, last_actual, wheels)
        self.encoder_forward_m_s = .087 * float(median(wheels))
        request_fraction = min(1., max(0., request / 1.5))
        limit = .18 * request_fraction + max(0., -self.encoder_forward_m_s)
        target = min(cruise_trim, limit)
        self.trim += min(.0036, max(-.0036, target - self.trim))
        return self.trim
