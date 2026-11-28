import unittest
from unittest.mock import Mock

import numpy as np

from tools110.course import command
from tools116.response_replay import INDEX, RELEASE_SLEW, trim_at_recorded_states
from tools117.native import ReleaseActor, with_parking
from tools117.run import SEEDS, parking_required


class LegGeometry:
    def raised_targets(self, raw, lift):
        issued = raw.copy()
        issued[0] += lift[0]
        return issued, lift


def geometry_loop(raw, lift):
    return LegGeometry().raised_targets(raw, lift)


class NativeAdapters(unittest.TestCase):
    def test_live_actor_adapter_matches_the_registered_shadow_on_all_three_archives(self):
        for seed in SEEDS:
            with self.subTest(seed=seed), np.load(INDEX.parent / f"native_{seed}/trace.npz") as z:
                preview, wheels = z["preview_actor_vx"], z["encoder_wheel_rad_s"]
                expected = trim_at_recorded_states(wheels, preview, RELEASE_SLEW)
                samples = iter(wheels)
                actor = ReleaseActor(Mock(), lambda: next(samples))
                for vx in preview:
                    frame = np.zeros((1, 57), np.float32)
                    frame[0, 6] = 2. * vx
                    actor.set_preview(vx)
                    actor.run([], {"nn_input0": frame, "nn_input1": np.zeros((1, 10, 57), np.float32)})
                np.testing.assert_array_equal(actor.increments, expected)
                np.testing.assert_array_equal(actor.commands, preview + expected)
                self.assertEqual(np.array_equal(expected, z["velocity_trim"]), seed == 111103)

    def test_projection_sees_the_driver_and_preserves_the_lift_at_the_issue_boundary(self):
        rows = {"raw": []}
        loop = with_parking(geometry_loop, rows)
        raw = np.arange(16, dtype=np.float32)
        lift = np.array([.012, .012, 0., 0.])
        driving = False
        for k in range(600):
            issued, achieved = loop(raw, lift)
            base, _ = geometry_loop(raw, lift)
            if command(k) > 0.:
                driving = True
            if driving and command(k) == 0.:
                base[3::4] -= base[3::4].mean()
            np.testing.assert_array_equal(issued, base)
            np.testing.assert_array_equal(achieved, lift)
            rows["raw"].append(raw)

    def test_parking_gate_needs_three_full_driving_passes_and_an_outstanding_stop(self):
        rows = [dict(seed=s, parking=False, status="COMPLETE", driving_pass=True, gate_pass=True,
                     motion={"ticks": 600}) for s in SEEDS]
        self.assertFalse(parking_required(rows))
        rows[0]["gate_pass"] = False
        self.assertTrue(parking_required(rows))
        rows[-1]["driving_pass"] = False
        self.assertFalse(parking_required(rows))
        self.assertFalse(parking_required(rows[:2]))
