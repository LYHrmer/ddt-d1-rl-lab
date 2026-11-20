"""Opt-in MuJoCo 3.3.0 classical-acceleration correction for the WBC wheel rows."""
from __future__ import annotations

import numpy as np

from tools90.wbc90 import WBC90


class RollingBiasWBC90(WBC90):
    """Use the world-frame classical acceleration of each moving wheel material point."""

    def _contacts(self, support, normals):
        contacts = super()._contacts(support, normals)
        twist = np.empty(6)
        for wheel in np.flatnonzero(support):
            self._mj.mj_objectVelocity(self.model, self.data, self._mj.mjtObj.mjOBJ_GEOM,
                                       self._geoms[wheel], twist, 0)
            # MuJoCo 3.3.0 supplies the spatial point acceleration in mj_jacDot.
            # The classical world derivative adds angular velocity x material-point
            # velocity. The acceleration right-hand side contains the negative bias.
            material_velocity = contacts.jac[wheel] @ self.data.qvel
            transport = np.cross(twist[:3], material_velocity)
            contacts.target[wheel] -= contacts.frame[wheel] @ transport
        return contacts
