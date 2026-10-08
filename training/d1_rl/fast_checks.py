"""Optional vectorized diagnostics; inherited D1 rollout semantics stay intact.

Only the two diagnostic methods differ from D1Env. Their invocation frequency,
step/PD/history/reward/reset implementations and thresholds remain inherited.
Model geom roles are immutable under the frozen scene/config contract. Contact
arrays are deliberately read afresh on each call because their buffer/length
can change across native steps. No integration occurs in either method.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .env import D1Env


class D1FastChecksEnv(D1Env):
    """Drop-in D1Env with equivalent live-warning and contact diagnostics."""

    def _check_finite_state(self):
        cache = self.__dict__.get('_fast_finite_cache')
        if cache is None or cache[0] is not self.data:
            # These are live MjData views, not snapshots; mj_resetData clears
            # their existing storage. Rebinding data builds a fresh cache.
            cache = (self.data, self.data.qpos, self.data.qvel, self.data.qacc,
                     self.data.warning.number)
            self._fast_finite_cache = cache
        if not (np.isfinite(cache[1]).all() and np.isfinite(cache[2]).all()
                and np.isfinite(cache[3]).all()):
            self._episode_done = True
            raise FloatingPointError('Non-finite MuJoCo state; abort this development run')
        # Preserve the original >0 predicate, including negative injected
        # counters. Iterating warning wrapper objects is intentionally avoided.
        if np.any(cache[4] > 0):
            self._episode_done = True
            raise FloatingPointError('MuJoCo emitted a numerical warning; abort this development run')

    def _contact_diagnostics(self):
        count = self.data.ncon
        if count == 0:
            return False, 0
        cache = self.__dict__.get('_fast_contact_cache')
        if cache is None or cache[0] is not self.model:
            nonworld = np.asarray(self.model.geom_bodyid) != 0
            supported_world = ((np.asarray(self.model.geom_contype) != 0) &
                               np.isin(self.model.geom_type, (
                                   mujoco.mjtGeom.mjGEOM_PLANE,
                                   mujoco.mjtGeom.mjGEOM_BOX,
                                   mujoco.mjtGeom.mjGEOM_HFIELD)))
            wheel = np.zeros(len(nonworld), dtype=bool)
            wheel[list(self.wheel_geoms)] = True
            cache = (self.model, nonworld, supported_world, wheel, np.zeros(6, dtype=np.float64))
            self._fast_contact_cache = cache
        _, nonworld, supported_world, wheel, force = cache
        # Never retain these dynamic contact views across calls/reset/step.
        geom = self.data.contact.geom[:count]
        geom_a, geom_b = geom[:, 0], geom[:, 1]
        nonworld_a, nonworld_b = nonworld[geom_a], nonworld[geom_b]
        self_contacts = int(np.count_nonzero(nonworld_a & nonworld_b))
        cross = np.flatnonzero(nonworld_a ^ nonworld_b)
        if cross.size == 0:
            return False, self_contacts
        world = np.where(nonworld_a[cross], geom_b[cross], geom_a[cross])
        robot = np.where(nonworld_a[cross], geom_a[cross], geom_b[cross])
        relevant = cross[supported_world[world] & ~wheel[robot]]
        illegal = False
        threshold = self.config['termination']['illegal_support_contact_normal_force_above_n']
        for index in relevant:
            force[:] = 0
            mujoco.mj_contactForce(self.model, self.data, int(index), force)
            if force[0] > threshold:
                illegal = True
        return illegal, self_contacts
