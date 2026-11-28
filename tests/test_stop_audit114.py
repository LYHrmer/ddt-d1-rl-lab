import unittest
from unittest.mock import patch

import numpy as np

from tools114.audit import audit_seed
from tools114.common import DEV, baseline
from tools114.control import StopProjection
from tools114.results import torque_at_recorded_speed


class ArchivedProjectionAudit(unittest.TestCase):
    def test_audit_rejects_a_logged_correction_that_was_never_issued(self):
        class ReportOnly(StopProjection):
            def apply(self, driver, targets):
                super().apply(driver, targets)
                return targets.copy()

        with patch("tools114.native.StopProjection", ReportOnly):
            with self.assertRaises(AssertionError):
                audit_seed(DEV[0])

    def test_fixed_encoder_torque_uses_the_already_delivered_wheel_targets(self):
        for row in torque_at_recorded_speed():
            with self.subTest(seed=row['seed']):
                path, _ = baseline(row['seed'])
                with np.load(path, allow_pickle=False) as trace:
                    measured = trace['first_torque_nm'][575:, 3::4].mean()
                    previous_target = 5. * trace['applied'][575:, 3::4].astype(float).mean()
                self.assertLess(row['projected_peak_abs_nm'], 12.)
                self.assertAlmostEqual(row['original_mean_nm'], measured, places=8)
                self.assertAlmostEqual(row['projected_mean_nm'] - measured, -.5 * previous_target, places=7)
