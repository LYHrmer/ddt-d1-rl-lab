import unittest
from unittest.mock import patch

import numpy as np

from tools113 import forecast


class FrozenActorForecast(unittest.TestCase):
    def test_high_speed_request_and_previous_action_reach_the_delayed_servo(self):
        qpos = np.zeros(23)
        qpos[2:4] = [.35, 1.]
        qvel = np.zeros(22)
        qvel[0] = 1.5
        frames = []
        actions = np.arange(1, 49, dtype=np.float32).reshape(3, 16) / 8.

        def actor(tick, frame, history):
            frames.append(frame.copy())
            if tick:
                np.testing.assert_array_equal(history[-1], frames[tick - 1])
                np.testing.assert_array_equal(frame[41:], actions[tick - 1])
            return actions[tick]

        class ConstantSpeed:
            def step(self, tick, state, target):
                expected = np.zeros(16) if tick == 0 else actions[tick - 1]
                np.testing.assert_array_equal(target, forecast.physical_targets(expected))
                return state.copy()

        with patch.object(forecast, "TICKS", 3), patch.object(forecast, "command", return_value=1.5):
            trace = forecast.rollout(ConstantSpeed(), qpos, qvel, False, actor)
        self.assertEqual(frames[0][6], np.float32(2. * 1.5 * forecast.SPEED_GAIN))
        np.testing.assert_allclose(trace["xy"][:, 0], [0., .03, .06, .09])
