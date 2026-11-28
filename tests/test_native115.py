from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from tools110.course import command


class LegGeometry:
    def __init__(self, model):
        pass

    def raised_targets(self, raw, lift):
        return raw.copy(), lift


def replay_policy_outputs(raw, rows):
    """Exercise only the parking and motor FIFO interfaces with archived policy outputs."""
    geometry = LegGeometry(None)
    previous = np.zeros(16, np.float32)
    for tick, action in enumerate(raw):
        command(tick)
        issued, _ = geometry.raised_targets(action,np.zeros(4))
        rows["issued"].append(issued)
        rows["applied"].append(previous)
        previous = issued


class TestRightFrontIntervention(unittest.TestCase):
    def test_release_keeps_mean_legs_and_relative_speeds_of_other_three(self):
        from tools115.native import release_rf_contrast

        action = np.arange(16, dtype=np.float32)
        action[3::4] = [.5, -1., .25, 1.25]
        raw = action.copy()
        out = release_rf_contrast(action)
        # m=.25, RF=-1.0, so the other three each move by -1.25/3.
        np.testing.assert_allclose(out[3::4], [1/12, .25, -1/6, 5/6], atol=3e-8, rtol=0.)
        np.testing.assert_array_equal(out.reshape(4,4)[:,:3], action.reshape(4,4)[:,:3])
        np.testing.assert_array_equal(action, raw)
        self.assertEqual(out.dtype, np.float32)

    def test_peak_excludes_pre_pulse_state_and_includes_tenth_endpoint(self):
        from tools115.native_compare import tilt_peak

        q = np.zeros((538,23))
        q[:,3] = 1.
        for k, deg in ((526,20.), (527,2.), (536,3.), (537,25.)):
            half = np.deg2rad(deg)/2.
            q[k,3:7] = [np.cos(half),0.,np.sin(half),0.]
        value,tick = tilt_peak(q)
        self.assertAlmostEqual(value,3.)
        self.assertEqual(tick,536)
        self.assertEqual(tilt_peak(q[:536]),(None,None))

    def test_scoring_and_single_pulse_on_archived_actions_without_integrating_physics(self):
        from tools115.native import with_rf_probe
        from tools115.native_compare import check_probe, reference114, score_candidate

        path, ref = reference114(111011)
        with np.load(path, allow_pickle=False) as archive:
            z = dict(archive)
        rows = defaultdict(list)
        with_rf_probe(replay_policy_outputs,rows)(z["raw"],rows)
        issued, applied = (np.stack(rows[k]) for k in ("issued","applied"))
        np.testing.assert_array_equal(np.flatnonzero((issued!=z["issued"]).any(axis=1)),[525])
        np.testing.assert_array_equal(np.flatnonzero((applied!=z["applied"]).any(axis=1)),[526])
        np.testing.assert_array_equal(rows["parking_issued_before_probe"],z["issued"])
        z.update({k:np.asarray(v) for k,v in rows.items()})
        # Synthetic actuator-only replay: the endpoints are intentionally the old
        # measurements. A correct trajectory scorer must therefore return zero tilt difference.
        z["first_torque_nm"][526,3::4] = np.clip(.5*(5.*applied[526,3::4].astype(float)
                                                     - z["encoder_wheel_rad_s"][526]),-12.,12.)
        with TemporaryDirectory() as tmp:
            candidate = Path(tmp)/"trace.npz"
            np.savez(candidate,**z)
            score = score_candidate(candidate,111011)
        self.assertEqual(score["tilt"]["delta_peak_deg"],0.)
        self.assertEqual(score["tilt"]["candidate_peak_tick"],532)
        self.assertEqual(score["reference_sha256"],ref["trace_sha256"])
        self.assertEqual(score["counts"]["decisions"],600)
        self.assertTrue(score["gate_pass"])
        self.assertEqual(score["wiring"]["paired_prefix"]["equal_state_endpoints"],527)
        z["issued"][526,7] += .25
        with self.assertRaises(AssertionError):
            check_probe(z)
