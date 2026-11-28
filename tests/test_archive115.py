import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools115 import archive


class ParkingEvidence(unittest.TestCase):
    def test_pre_dispatch_failure_survives_without_worker_or_supervisor(self):
        with TemporaryDirectory() as temporary:
            local = Path(temporary)/"local"
            public = Path(temporary)/"public"
            local.mkdir()
            receipt = dict(stage="policy", attempt=1, unit="d1-parking115-policy-01", child_started=False,
                           registration_sha256="registered", error="headroom timeout")
            (local/"admission_policy_01.json").write_text(json.dumps(receipt))
            with patch.object(archive, "BASE", local), patch.object(archive, "PUBLIC", public):
                files = archive._collect()
                entry = archive._ledger("policy_01", files["policy_01"], "registered")
                self.assertEqual(entry["status"], "PRE_ADMISSION_FAILURE")
                self.assertEqual(entry["error"], receipt["error"])
                self.assertEqual(archive._select({"policy_01": entry})["policy"]["status"], "NOT_STARTED")
                self.assertFalse((public/"policy").exists())
                with self.assertRaises(ValueError):
                    archive._ledger("policy_01", files["policy_01"], "different")

    def test_incomplete_supervisor_never_turns_a_written_report_into_a_completed_case(self):
        guard = dict(status="INCOMPLETE", child_started=True)
        self.assertEqual(archive._status(guard, True, False), "PARTIAL")
        self.assertEqual(archive._status(guard, False, False), "INTERRUPTED_BEFORE_RESERVATION")
        records = {"policy_01": dict(stage="policy", attempt=1, status="PARTIAL"),
                   "policy_02": dict(stage="policy", attempt=2, status="COMPLETE")}
        with self.assertRaises(ValueError):
            archive._select(records)

    def test_tilt_gate_uses_all_six_measured_windows_and_the_preregistered_size(self):
        def verdict(changes):
            rows = [dict(tilt=None if v is None else dict(delta_peak_deg=v, window_complete=True)) for v in changes]
            return archive.tilt_verdict(rows, .10, 6)["outcome"]
        self.assertEqual(verdict([.1]*6), "ORIGINAL_CONTRAST_REDUCED_PEAK_TILT")
        self.assertEqual(verdict([-.1]*6), "ORIGINAL_CONTRAST_INCREASED_PEAK_TILT")
        self.assertEqual(verdict([.5]*5+[.099]), "DIRECTION_OR_MAGNITUDE_DID_NOT_PASS")
        self.assertEqual(verdict([-.5]*5+[None]), "MISSING_WINDOWS")
