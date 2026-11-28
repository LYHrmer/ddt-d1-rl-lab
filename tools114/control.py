"""Remove the common wheel-speed target on a driver's stop after forward travel."""
import numpy as np


class StopProjection:
    """Remember the last driving request and project wheel targets while parked."""

    def __init__(self):
        self._last_request_forward = False
        self.active = False
        self.removed_rad_s = 0.

    def apply(self, driver_command, targets):
        command = np.asarray(driver_command, dtype=np.float64)
        issued = np.asarray(targets).copy()
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("expected a finite three-axis driver command")
        if issued.shape != (16,) or issued.dtype.kind != "f" or not np.isfinite(issued).all():
            raise ValueError("expected sixteen finite floating-point actuator targets")
        if np.any(command != 0.):
            self._last_request_forward = bool(command[0] > 0.)
            self.active = False
        else:
            self.active = self._last_request_forward
        self.removed_rad_s = 0.
        if self.active:
            wheels = issued[3::4].astype(np.float64)
            mean = float(wheels.mean())
            issued[3::4] = wheels - mean
            # The frozen vendor wheel servo in tools110.native.torque uses 5*a rad/s.
            self.removed_rad_s = 5. * mean
        return issued
