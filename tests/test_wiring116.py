from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tools111.kinematics import vendor_targets
from tools116.control import PrevTrimActor
from tools116.wiring import ArchiveGeometry, _check_leg_targets, reference_speeds


def inputs(speed):
    frame = np.arange(57,dtype=np.float32)[None,:]
    frame[0,6] = 2.*speed
    history = np.arange(10*57,dtype=np.float32).reshape(1,10,57)
    return {"nn_input0":frame,"nn_input1":history}


class CompoundWiring(unittest.TestCase):
    def test_uses_full_precision_preview_and_previous_actual_only(self):
        seen = []
        def update(request, previous, wheels):
            seen.append(previous)
            return .037

        actor = PrevTrimActor(Mock(),lambda:[1.,2.,3.,4.])
        actor.control = SimpleNamespace(update=update,encoder_forward_m_s=.2)
        preview = .876543219876543
        history = None
        for speed in (preview,.82):
            inp = inputs(speed)
            expected = inp["nn_input0"].copy()
            expected[0,6] = 2.*(speed+.037)
            history = inp["nn_input1"].copy()
            actor.set_preview(speed)
            actor.run([],inp)
            np.testing.assert_array_equal(inp["nn_input0"],expected)
            np.testing.assert_array_equal(inp["nn_input1"],history)
        self.assertEqual(seen,[0.,preview+.037])
        trace = actor.trace({})
        np.testing.assert_array_equal(trace["trim_previous_actor_vx"],seen)
        np.testing.assert_array_equal(trace["actor_previous_raw"][-1],expected[0,41:57])
        np.testing.assert_array_equal(trace["actor_history_previous_raw"][-1],history[0,:,41:57])
        np.testing.assert_array_equal(trace["actor_history_frame_command"][-1],history[0,:,6:9])

    def test_disabled_switch_preserves_every_input_bit_with_nonzero_trim(self):
        actor = PrevTrimActor(Mock(),lambda:[0.]*4,False)
        actor.control = SimpleNamespace(update=lambda *args:.17,encoder_forward_m_s=.1)
        vx = 1.123456789012345
        inp = inputs(vx)
        before = {key:arr.copy() for key,arr in inp.items()}
        actor.set_preview(vx)
        actor.run([],inp)
        for name in inp:
            np.testing.assert_array_equal(inp[name],before[name])
        row = actor.trace({})
        self.assertEqual(row["actor_vx"][0],vx)
        self.assertEqual(row["velocity_trim"][0],0.)

    def test_misclocked_preview_and_failed_input_are_observable(self):
        throw = Mock()
        throw.run.side_effect = ArithmeticError("synthetic actor failure")
        actor = PrevTrimActor(throw,lambda:[0.]*4)
        with self.assertRaises(ValueError):
            actor.run([],inputs(.9))
        actor.set_preview(.9)
        with self.assertRaises(ValueError):
            actor.set_preview(1.)
        with self.assertRaises(ValueError):
            actor.run([],inputs(.95))
        with self.assertRaises(ArithmeticError):
            actor.run([],inputs(.9))
        record = actor.trace({"raw":[]})
        self.assertEqual((len(record["raw"]),len(record["actor_previous_raw"])),(0,1))
        self.assertEqual(record["preview_actor_vx"].tolist(),[.9])

    def test_independent_reference_accumulates_from_prior_preview(self):
        n = 220
        encoders = np.repeat(np.linspace(.2,17.,n)[:,None],4,axis=1)
        preview = np.linspace(0.,1.,n)
        a,trim,_ = reference_speeds(preview,encoders,True)
        b,masked_trim,_ = reference_speeds(preview+.6,encoders,True)
        np.testing.assert_array_equal(a,preview+trim)
        np.testing.assert_array_equal(b,preview+.6+masked_trim)
        # A different previous actual biases the estimator even with identical encoder and driver samples.
        self.assertGreater(np.abs(trim-masked_trim).max(),.05)
        np.testing.assert_array_equal(reference_speeds(preview,encoders,False)[0],preview)

    def test_fk_catches_altered_stored_height_without_simulator(self):
        chain = ArchiveGeometry()
        raw = np.zeros((3,16),np.float32)
        joints = np.vstack((vendor_targets(raw),vendor_targets(raw[-1:])))
        q = np.zeros((4,23))
        q[:,3],q[:,7:] = 1.,joints
        centre = chain.forward(joints)[0]
        zero = np.zeros((3,4))
        z = dict(raw=raw,issued=raw,issued_lift_fk=zero.copy(),applied_lift_fk=zero.copy(),
                 lift_demand=zero.copy(),qpos=q,wheels_xy=centre[:,:,:2],measured_relative_z=centre[:,:,2])
        self.assertEqual(_check_leg_targets(z,chain)["max_error_logged_fk_m"],0.)
        z["issued_lift_fk"][1,0] += .001
        with self.assertRaises(AssertionError):
            _check_leg_targets(z,chain)
