"""Verify the unpaired holdout scorer with archived closed loops and intentionally broken records."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools91.launch91 import ROOT, read
from tools118.score import score_trace, verify_reset


DEVELOPMENT = ROOT / "reports/compound117"


class HoldoutScore(unittest.TestCase):
    def test_all_final_development_runs_match_the_original_measurements(self):
        fingerprints = set()
        for original in read(DEVELOPMENT / "results.json")["parking_cases"]:
            seed = original["seed"]
            with self.subTest(seed=seed):
                path = DEVELOPMENT / f"parking_{seed}/trace.npz"
                row = score_trace(path, seed)
                for key in ("counts", "complete_horizon", "motion", "lidar", "stop", "stop_endpoints",
                            "gate_pass", "driving_pass", "first_failed_section", "trace_sha256"):
                    self.assertEqual(row[key], original[key])
                fingerprints.add(row["reset"]["fingerprint_sha256"])
        self.assertEqual(len(fingerprints), 3)

    def test_seed_height_and_zero_velocity_are_individually_bound(self):
        path = DEVELOPMENT / "parking_111101/trace.npz"
        with np.load(path, allow_pickle=False) as z:
            q, v = z["qpos"][0], z["qvel"][0]
            height = float(z["nominal_base_height_m"])
        with self.assertRaises(AssertionError):
            score_trace(path, 111107)
        moving = v.copy()
        moving[0] += .0001
        lifted = q.copy()
        lifted[2] += .0001
        for name, pose, velocity, nominal in (("initial velocity", q, moving, height),
                                              ("nominal height", q, v, height+.0001),
                                              ("initial height", lifted, v, height)):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                verify_reset(111101, pose, velocity, nominal)

    def test_early_safety_terminal_counts_but_interruption_cannot_score(self):
        # Use a real common prefix and mark an artificial safety terminal on its last interval.
        n, packets = 20, 3
        with np.load(DEVELOPMENT / "parking_111101/trace.npz", allow_pickle=False) as original:
            z = {}
            for key, series in original.items():
                lengths = {601: n + 1, 600: n, 72: packets}
                z[key] = series[:lengths[len(series)]] if series.ndim else series
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.npz"
            np.savez(path, **z)
            with self.assertRaisesRegex(ValueError, "unexplained early"):
                score_trace(path, 111101)
            z["illegal"][-1] = True
            np.savez(path, **z)
            row = score_trace(path, 111101)
            self.assertEqual(row["counts"]["decisions"], n)
            self.assertIn("legal_support", row["motion"]["failed"])
            self.assertFalse(row["gate_pass"] or row["driving_pass"] or row["complete_horizon"])
            self.assertIsNone(row["stop"])
            self.assertEqual(row["stop_endpoints"], 0)
            (path.parent / "worker_partial.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "partial"):
                score_trace(path, 111101)


if __name__ == "__main__":
    unittest.main()
