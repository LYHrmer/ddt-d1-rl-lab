import unittest

import numpy as np

from tools109.cruise109 import ARM_SCALE, RADIUS, command_at, encode_forward_command, issued_action
from tools109.score109 import score
from training.d1_rl.env import D1Env
from training.d1_rl.interface import DEFAULT_Q, LEG_INDICES, WHEEL_INDICES, build_observation


class TestSpeed109(unittest.TestCase):
    def test_high_command_reaches_actor_with_original_low_speed_encoding(self):
        for cmd in (0., .2, 1., 1.5):
            frame, _ = build_observation(np.zeros(3), np.array([1., 0., 0., 0.]), np.array([cmd, 0., 0.]),
                                         DEFAULT_Q, np.zeros(16), np.zeros(16), np.zeros(3))
            before = frame.copy()
            encode_forward_command(frame, cmd)
            self.assertEqual(frame[6], np.float32(2.*cmd))
            if cmd <= 1.:
                np.testing.assert_array_equal(frame, before)
            else:
                self.assertEqual(before[6], 2.)
                np.testing.assert_array_equal(np.delete(frame, 6), np.delete(before, 6))
        with self.assertRaises(ValueError):
            D1Env._validate_eval_sequence([{"start_s": 0., "vx": 1.5, "wz": 0.}])

    def test_wheel_reference_keeps_legs_and_stops(self):
        nominal = np.linspace(-1., 1., 16, dtype=np.float32)
        self.assertEqual([command_at(t) for t in [0, 50, 150, 275, 350, 400]], [0., 0., 1.5, 1.5, 0., 0.])
        for arm in ("actor5", "actor24"):
            np.testing.assert_array_equal(issued_action(arm, nominal, 1.5), nominal)
        candidate = issued_action("reference24", nominal, 1.5)
        np.testing.assert_array_equal(candidate[LEG_INDICES], nominal[LEG_INDICES])
        rates = candidate[WHEEL_INDICES] * ARM_SCALE["reference24"]
        self.assertTrue(np.all(rates > 1.5/RADIUS) and np.all(rates < 24.))
        np.testing.assert_array_equal(issued_action("reference24", nominal, 0.)[WHEEL_INDICES], np.zeros(4))

    def test_cruise_and_stop_gates_use_displacement_and_the_whole_stop_window(self):
        t = np.arange(401)/50.
        q = np.zeros((401, 23))
        q[:, 3] = 1.
        q[:, 0] = np.clip(t-3.5, 0., 2.) * 1.52
        dq = np.zeros((401, 22))
        dq[(t >= 3.5) & (t < 5.5), 0] = 1.52
        trace = dict(time_s=t, qpos=q, qvel=dq, wheels_actual=np.zeros((401, 4)),
                     wheel_target=np.zeros((400, 4)), wheel_saturated=np.zeros((400, 4), bool),
                     fallen=np.zeros(400, bool), illegal=np.zeros(400, bool))
        self.assertTrue(score(trace)["passed"])
        dq[380, 1] = .06
        self.assertEqual(score(trace)["failed"], ["stop_speed"])
        dq[380, 1] = 0.
        q[:, [0, 1]] = q[:, [1, 0]]
        self.assertEqual(score(trace)["failed"], ["cruise_distance"])
        q[:, 3], q[:, 6] = np.cos(np.pi/4), np.sin(np.pi/4)
        self.assertTrue(score(trace)["passed"])
        trace["illegal"][123] = True
        self.assertEqual(score(trace)["failed"], ["illegal_contact"])
        partial = {key: value[:(126 if len(value)==401 else 125)] for key, value in trace.items()}
        self.assertFalse(score(partial)["passed"])
        self.assertIsNone(score(partial)["cruise_distance_m"])


if __name__ == "__main__":
    unittest.main()
