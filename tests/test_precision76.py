"""Standard-library arithmetic checks; no model, NumPy, Torch or physics."""
import json
import math
import sys
import unittest
from fractions import Fraction

from tools76.precision76 import FROZEN_FLOOR, endpoint_precision


def fraction(record):
    return Fraction(int(record['numerator']), int(record['denominator']))


class EndpointPrecisionTests(unittest.TestCase):
    def test_low_magnitude_can_have_well_separated_endpoints(self):
        result = endpoint_precision(1. + 2. ** -30, 1., 2. ** -11, 2. ** -20)
        self.assertLess(result['exact_fd']['float'], FROZEN_FLOOR)
        self.assertEqual(fraction(result['endpoint_gap_over_combined_ulp']), 2 ** 21)
        self.assertEqual(fraction(result['saved_minus_exact']), 0)
        self.assertFalse(result['one_neighbor_interval']['contains_zero'])
        self.assertFalse(result['one_neighbor_interval']['touches_or_crosses_positive_floor'])
        self.assertTrue(result['python_replay']['bitwise_equal_saved'])

    def test_equal_stored_endpoints_do_not_prove_zero_mathematical_derivative(self):
        baseline, eps = 2. ** 40, 2. ** -21
        self.assertEqual(baseline + eps, baseline - eps)
        result = endpoint_precision(baseline + eps, baseline - eps, eps, 0.)
        self.assertEqual(fraction(result['exact_fd']), 0)
        self.assertEqual(fraction(result['saved_minus_exact']), 0)
        exact_plus = Fraction.from_float(baseline) + Fraction.from_float(eps)
        exact_minus = Fraction.from_float(baseline) - Fraction.from_float(eps)
        self.assertEqual((exact_plus - exact_minus) / (2 * Fraction.from_float(eps)), 1)
        self.assertTrue(result['one_neighbor_interval']['crosses_zero'])
        self.assertFalse(result['changes_epsilon_or_qualification'])

    def test_two_rounding_operations_are_distinguished_from_exact_ratio(self):
        plus, minus, eps = 1., -2. ** -53, .15
        saved = (plus - minus) / (2 * eps)
        result = endpoint_precision(plus, minus, eps, saved)
        exact = (Fraction.from_float(plus) - Fraction.from_float(minus)) / (2 * Fraction.from_float(eps))
        self.assertEqual(fraction(result['exact_fd']), exact)
        self.assertEqual(fraction(result['python_replay']['subtraction_roundoff']), -Fraction(1, 2 ** 53))
        self.assertNotEqual(fraction(result['python_replay']['division_roundoff']), 0)
        self.assertNotEqual(fraction(result['saved_minus_exact']), 0)
        self.assertEqual(fraction(result['python_replay']['minus_saved']), 0)
        changed = endpoint_precision(plus, minus, eps, math.nextafter(saved, math.inf))
        self.assertFalse(changed['python_replay']['bitwise_equal_saved'])
        self.assertLess(fraction(changed['python_replay']['minus_saved']), 0)

    def test_subnormal_and_zero_have_exact_fraction_records(self):
        tiny = math.ulp(0.)
        for plus, minus, eps, expected in ((tiny, 0., .5, tiny), (tiny, 0., tiny, .5), (0., 0., 1., 0.)):
            with self.subTest(plus=plus, epsilon=eps):
                result = endpoint_precision(plus, minus, eps, expected)
                self.assertEqual(fraction(result['exact_fd']), Fraction.from_float(expected))
                self.assertEqual(fraction(result['saved_minus_exact']), 0)
                json.dumps(result, allow_nan=False)
        signed = endpoint_precision(-0., 0., .5, 0.)
        self.assertFalse(signed['python_replay']['bitwise_equal_saved'])
        self.assertEqual(fraction(signed['saved_minus_exact']), 0)

    def test_neighbor_interval_preserves_negative_direction_and_binade_spacing(self):
        result = endpoint_precision(-1., -1. + 2. ** -30, .5, -2. ** -30)
        interval = result['one_neighbor_interval']
        self.assertLess(fraction(interval['upper']), 0)
        expected = (Fraction.from_float(math.nextafter(-1., -math.inf))
                    - Fraction.from_float(math.nextafter(-1. + 2. ** -30, math.inf)))
        self.assertEqual(fraction(interval['lower']), expected)

    def test_original_positive_and_negative_floor_boundaries_are_descriptive(self):
        for sign, name in ((1., 'positive_floor'), (-1., 'negative_floor')):
            result = endpoint_precision(sign * FROZEN_FLOOR, 0., .5, sign * FROZEN_FLOOR)
            self.assertTrue(result['one_neighbor_interval']['crosses_' + name])
            self.assertEqual(result['frozen_magnitude_floor'], 1e-5)

    def test_overflow_results_are_null_with_reasons_and_exact_fractions_survive(self):
        largest = sys.float_info.max
        for plus, minus, eps in ((largest, -largest, math.ulp(0.)), (largest, -largest, largest), (1., 0., largest)):
            with self.subTest(epsilon=eps):
                result = endpoint_precision(plus, minus, eps, 0.)
                self.assertTrue(result['derived_unavailable_reasons'])
                json.dumps(result, allow_nan=False)
                self.assertEqual(fraction(result['exact_fd']),
                                 (Fraction.from_float(plus) - Fraction.from_float(minus))
                                 / (2 * Fraction.from_float(eps)))
        self.assertIsNone(endpoint_precision(largest, -largest, 1., 0.)['one_neighbor_interval']['crosses_zero'])

    def test_nonfinite_invalid_epsilon_and_non_scalar_inputs_are_rejected(self):
        for index in range(4):
            for bad in (math.inf, -math.inf, math.nan, True, '1', [], 10 ** 1000):
                arguments = [1., 0., .5, 1.]
                arguments[index] = bad
                with self.subTest(index=index, bad=repr(bad)), self.assertRaises((ValueError, TypeError)):
                    endpoint_precision(*arguments)
        for eps in (0., -0., -.1):
            with self.assertRaises(ValueError):
                endpoint_precision(1., 0., eps, 1.)


if __name__ == '__main__':
    unittest.main()
