import unittest

import numpy as np

from tools115.signals import audit, mode, rf_contrast_share


class WheelBrakeEvidence(unittest.TestCase):
    def test_power_uses_measured_rotation_and_torque(self):
        self.assertEqual(mode(.025, -2.1), "braking")
        self.assertEqual(mode(-.025, -2.1), "driving_negative")
        self.assertEqual(mode(-.025, 2.1), "braking")
        self.assertEqual(mode(.025, 2.1), "driving_positive")
        self.assertEqual(mode(0., -2.1), "zero_power")
        with self.assertRaises(ValueError):
            mode(float("nan"), -2.)

    def test_rf_against_the_other_three_is_a_normalized_contrast(self):
        self.assertAlmostEqual(rf_contrast_share([1., -5., 1., 1.]), 1.)
        self.assertAlmostEqual(rf_contrast_share([1., 0., -1., 0.]), 0.)
        self.assertIsNone(rf_contrast_share([2.]*4))
        v = np.array([.8, -5., -.7, .9])
        self.assertAlmostEqual(rf_contrast_share(v), rf_contrast_share(3. * v + 42.))

    def test_same_state_and_raw_action_have_different_delivered_targets(self):
        result = audit()
        first, old = result["round114"][0], result["round113_original"]
        self.assertEqual(first["seed"], old["seed"])
        self.assertEqual(first["raw_wheel_action"], old["raw_wheel_action"])
        self.assertEqual(first["wheel_encoder_rad_s"], old["wheel_encoder_rad_s"])
        self.assertNotEqual(first["delivered_wheel_target_rad_s"], old["delivered_wheel_target_rad_s"])
        self.assertGreater(first["first_servo_torque_nm"][1], old["first_servo_torque_nm"][1])
        self.assertEqual(result["rf_braking_first_samples"], len(result["round114"]))
