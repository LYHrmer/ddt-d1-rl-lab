import unittest

import numpy as np

from tools110.course import command
from tools111.control import SPEED_GAIN
from tools112.velocity import VelocityTrim
from tools113.control import BrakeTrim
from tools113.native import TrimmedActor


class BrakeCompensation(unittest.TestCase):
    def test_forward_coast_and_reverse_need_different_zero_command_responses(self):
        control = BrakeTrim()
        nominal = VelocityTrim()
        previous = 0.
        responses = []
        for tick in range(600):
            request = command(tick)
            measured = max(.02, .85*request) if tick < 535 else -.08
            encoders = [measured/.087]*4
            uncapped = nominal.update(request, previous, encoders)
            increment = control.update(request, previous, encoders)
            previous = SPEED_GAIN*request + increment
            responses.append(increment)
            if tick == 450:
                self.assertEqual(increment, uncapped)
                self.assertGreater(increment, .1)
        self.assertEqual(responses[525], 0.)
        self.assertAlmostEqual(responses[-1], .08)
        self.assertLessEqual(np.max(np.abs(np.diff([0., *responses]))), .0036 + 1e-12)

    def test_only_forward_frame_component_changes_and_enters_actor_history(self):
        class Session:
            def run(self, outputs, inputs):
                return inputs["nn_input0"].copy(), inputs["nn_input1"].copy()

        actor = TrimmedActor(Session(), lambda: np.zeros(4))
        history = np.zeros((1,10,57), dtype=np.float32)
        for tick in range(200):
            frame = np.linspace(-.5, .5, 57, dtype=np.float32)
            frame[6] = 2.*SPEED_GAIN*command(tick)
            before = frame.copy()
            if tick == 0:
                history[:] = frame
            sent, received_history = actor.run([], {"nn_input0": frame[None,:], "nn_input1": history})
            np.testing.assert_array_equal(sent[0,:6], before[:6])
            np.testing.assert_array_equal(sent[0,7:], before[7:])
            self.assertEqual(sent[0,6], np.float32(2.*actor.commands[tick]))
            if tick > 0:
                self.assertEqual(received_history[0,-1,6], np.float32(2.*actor.commands[tick-1]))
            history[:, :-1] = history[:, 1:].copy()
            history[:, -1] = frame
        self.assertGreater(actor.increments[-1], 0.)
