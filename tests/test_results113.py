import unittest
from pathlib import Path
from unittest.mock import patch

from tools113.results import compare_metrics, metric_difference, native_case, score_change


class ProspectiveSpeedChange(unittest.TestCase):
    def test_direction_and_symmetric_error_use_the_observed_signed_change(self):
        for direction in (1., -1.):
            for predicted, observed in ((.125, .09375), (.09375, .125)):
                result = score_change(direction * predicted, direction * observed)
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(result["relative_error"], .25)
            self.assertEqual(score_change(direction * .125, direction * .09)["status"], "FAIL")
            reversal = score_change(-direction * .125, direction * .125)
            self.assertFalse(reversal["sign_correct"])
            self.assertEqual(reversal["relative_error"], 2.)
            self.assertFalse(score_change(0., direction * .125)["passed"])

    def test_unmeasurable_and_missing_changes_never_pass(self):
        for observed in (-.000999, 0., .000999):
            result = score_change(1., observed)
            self.assertEqual(result["status"], "UNIDENTIFIABLE")
            self.assertIsNone(result["sign_correct"])
        self.assertTrue(score_change(.001, .001)["passed"])
        self.assertEqual(score_change(None, .1)["status"], "UNAVAILABLE")
        self.assertFalse(score_change(.1, None)["passed"])
        with self.assertRaises(ValueError):
            score_change(float("nan"), .1)
        delta = metric_difference({"cruise_m_s": None, "stop_peak_m_s": .03125},
                                  {"cruise_m_s": 1.5, "stop_peak_m_s": .0625})
        compare_metrics(delta, {"cruise_m_s": None, "stop_peak_m_s": -.03125})

    def test_motion_gate_and_prediction_gate_are_independent(self):
        seed = 111011
        off = {"motion": {"speed_m_s": 1.4, "stop_tail_max_m_s": .125}}
        measurement = {"cruise_m_s": 1.5, "stop_peak_m_s": .0625}
        motion = {"ticks": 600, "passed": True}
        record = {"result": {"seed": seed, "motion": motion}}
        with patch("tools113.results.grid_case", return_value=(Path("unused"), off)), \
             patch("tools113.results.verify_episode", return_value=measurement):
            forecast = {"off": {}, "on": {}, "delta": {"cruise_m_s": -.1, "stop_peak_m_s": -.0625}}
            result = native_case(seed, record, forecast)
            self.assertTrue(result["control_passed"])
            self.assertFalse(result["model_passed"])
            motion["passed"] = False
            forecast["delta"]["cruise_m_s"] = .1
            result = native_case(seed, record, forecast)
            self.assertTrue(result["model_passed"])
            self.assertFalse(result["control_passed"])
