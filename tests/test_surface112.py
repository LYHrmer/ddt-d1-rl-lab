import subprocess
import sys
import unittest

from tools112.run_surface import paths
from tools112.surface import SEEDS, paired_scores, reference
from tools112.surface_results import verdict


class SurfaceDiagnosis(unittest.TestCase):
    def test_public_reader_works_without_native_simulation_packages(self):
        script = """
import sys
for name in ("gymnasium", "mujoco", "onnxruntime"):
    sys.modules[name] = None
from tools112.surface_results import verify
"""
        subprocess.run([sys.executable, "-B", "-c", script], check=True)

    def test_separate_speed_and_stop_evidence(self):
        plane = dict(speed_m_s=1.55,stop_tail_max_m_s=.025)
        speed_only = paired_scores(plane,dict(speed_m_s=1.44,stop_tail_max_m_s=.030))
        stop_only = paired_scores(plane,dict(speed_m_s=1.54,stop_tail_max_m_s=.095))
        incomplete = paired_scores(plane,dict(speed_m_s=None,stop_tail_max_m_s=None))
        self.assertTrue(speed_only["cruise_supported"] and not speed_only["stop_supported"])
        self.assertTrue(stop_only["stop_supported"] and not stop_only["cruise_supported"])
        self.assertIsNone(incomplete["cruise_loss_m_s"])

    def test_only_registered_surface_cases_can_start(self):
        for seed in SEEDS:
            path, episode = reference(seed)
            self.assertTrue(path.is_file() and episode["motion"]["passed"])
            self.assertEqual(paths(str(seed),1)[0].name, f"{seed}_01")
        for task in ("blind","trim","112201"):
            with self.assertRaises(ValueError):
                paths(task,1)

    def test_incomplete_pairs_cannot_confirm_either_cause(self):
        score = dict(ticks=600, cruise_supported=True, stop_supported=True)
        completed = dict(case_status="COMPLETE", comparison=score)
        for missing in (dict(case_status="PARTIAL"), dict(case_status="NOT_STARTED"),
                        dict(case_status="COMPLETE", comparison=dict(score,ticks=599))):
            result = verdict([completed,completed,missing])
            self.assertFalse(result["cruise_supported"])
            self.assertFalse(result["stop_supported"])
