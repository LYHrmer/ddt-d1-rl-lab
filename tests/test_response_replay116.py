import unittest

import numpy as np

from tools111.control import SPEED_GAIN
from tools116.response_replay import ORIGINAL_SLEW, RELEASE_SLEW, TRIM_CAP, SaturationRelease, audit


class TestCompoundResponseReplay(unittest.TestCase):
    def test_touching_cap_only_changes_fall_when_lidar_is_reducing_speed(self):
        base = SPEED_GAIN * 1.5
        for preview, expected_fall in ((base, ORIGINAL_SLEW), (base - .045, RELEASE_SLEW)):
            with self.subTest(preview=preview):
                trim = SaturationRelease(RELEASE_SLEW)
                trim.cruise.last_request = 1.5
                trim.cruise.held_ticks = 12
                trim.cruise.trim = trim.trim = TRIM_CAP
                output = trim.update_preview(1.5, base, np.full(4, 1.6/.087), preview)
                self.assertAlmostEqual(TRIM_CAP - output, expected_fall)
                self.assertAlmostEqual(trim.cruise.trim, output)

    def test_replay_matches_all_baselines_and_preserves_non_saturated_case(self):
        result = audit()
        rows = result['cases']
        self.assertEqual([r['seed'] for r in rows], [111101,111103,111107])
        self.assertEqual([r['speed']['identical_full_command_trace'] for r in rows], [False,True,False])
        self.assertEqual(sum(r['speed']['total_decisions'] for r in rows), 1800)
        self.assertEqual(sum(r['parking']['reverse_targets'] for r in rows), 300)
        for row in rows:
            self.assertLess(row['parking']['mean_measured_first_servo_torque_nm'], 0.)
            self.assertGreater(row['parking']['mean_projected_torque_at_same_measured_speeds_nm'], 0.)
