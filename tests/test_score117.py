"""Exercise the new score against the sealed traces without a policy session or simulator."""
import unittest

import numpy as np

from tools91.launch91 import ROOT
from tools110.course import command
from tools114.control import StopProjection
from tools117.score import _paired_prefix, _projected_wheels, score_trace


def baseline(seed):
    return ROOT / f"reports/compound116/native_{seed}/trace.npz"


class CompoundScore117(unittest.TestCase):
    def test_unchanged_start_reaches_all_driving_gates_and_fails_solely_at_stop(self):
        path = baseline(111103)
        row = score_trace(path, path, 111103)
        self.assertTrue(row["driving_pass"])
        self.assertFalse(row["gate_pass"])
        self.assertEqual(row["motion"]["failed"], ["stopped"])
        self.assertEqual(len(row["motion"]["sections_m_s"]), 3)
        self.assertEqual(row["stop_endpoints"], 26)
        self.assertEqual(row["stop"]["endpoints_above_limit"], 16)
        self.assertLess(row["stop"]["forward_displacement_mm"], 0.)
        self.assertIsNone(row["prefix"]["first_changed_input_tick"])
        self.assertEqual(row["prefix"]["equal_state_endpoints"], 601)

    def test_unchanged_records_that_required_release_cannot_pass_as_candidates(self):
        path = baseline(111101)
        with self.assertRaises(AssertionError):
            score_trace(path, path, 111101)

    def test_early_or_differential_parking_interventions_rejected(self):
        with np.load(baseline(111103), allow_pickle=False) as trace:
            z = {key: trace[key] for key in ("raw", "issued", "lift_demand")}
        projected = StopProjection()
        issued = []
        for k, target in enumerate(z["issued"]):
            # The frozen leg IK returns float32 on ticks with zero requested lift.
            dtype = target.dtype if np.any(z["lift_demand"][k]) else z["raw"].dtype
            issued.append(projected.apply((command(k), 0., 0.), target.astype(dtype)))
        z["issued"] = np.stack(issued)
        first = _projected_wheels(z, True)
        self.assertEqual(first, 525)
        for k in (524, 560):
            with self.subTest(tick=k):
                bad = z["issued"].copy()
                bad[k, [3, 7]] += [-.02, .02]
                with self.assertRaises(AssertionError):
                    _projected_wheels(dict(z, issued=bad), True)

    def test_an_unchanged_input_cannot_hide_a_changed_state(self):
        with np.load(baseline(111103), allow_pickle=False) as archive:
            old = dict(archive)
        pose = old["qpos"].copy()
        pose[580, 0] += .01
        with self.assertRaises(AssertionError):
            _paired_prefix(dict(old, qpos=pose), old, parking=False, first_trigger=None)


if __name__ == "__main__":
    unittest.main()
