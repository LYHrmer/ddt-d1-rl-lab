import unittest
from unittest.mock import patch

from tools114.audit import audit_seed
from tools114.common import DEV
from tools114.control import StopProjection


class ArchivedProjectionAudit(unittest.TestCase):
    def test_audit_rejects_a_logged_correction_that_was_never_issued(self):
        class ReportOnly(StopProjection):
            def apply(self, driver, targets):
                super().apply(driver, targets)
                return targets.copy()

        with patch("tools114.native.StopProjection", ReportOnly):
            with self.assertRaises(AssertionError):
                audit_seed(DEV[0])
