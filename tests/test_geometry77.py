"""Small standard-library77 geometry checks; no simulation or model calls."""
import math
import unittest

from tools77.geometry77 import MODES, compose, preservation_receipt


def state(quaternion=(1., 0., 0., 0.)):
    values = [i / 100 for i in range(45)]
    values[3:7] = quaternion
    values[23:26] = [.7, -.3, .2]
    return values


class Geometry77Tests(unittest.TestCase):
    def test_tilted_world_replacement_changes_body_but_protected_preserves_it(self):
        control = state((math.cos(math.pi / 12), 0., math.sin(math.pi / 12), 0.))
        active = state((1., .1, .2, .3))
        active[25] = .6
        direct = preservation_receipt(control, active, compose(control, active, 'world_only'))
        self.assertAlmostEqual(direct['body_xy_abs_gap'][0], .2, places=14)
        protected = compose(control, active, 'body_protected')
        receipt = preservation_receipt(control, active, protected)
        self.assertTrue(receipt['unchanged_nonvelocity'])
        self.assertTrue(receipt['world_vz_exact'])
        self.assertLess(max(receipt['body_xy_abs_gap']), 1e-14)
        self.assertAlmostEqual(receipt['Rzz'], math.cos(math.pi / 6), places=14)
        self.assertAlmostEqual(protected[23], .7 + math.tan(math.pi / 6) * .4, places=14)

    def test_all_modes_copy_the_registered_coordinates_without_mutation(self):
        control = state(); active = [x + .1 for x in control]
        originals = [list(control), list(active)]
        for mode in MODES:
            result = compose(control, active, mode)
            self.assertIsNot(result, control)
            self.assertIsNot(result, active)
            if mode in ('control', 'active'):
                self.assertEqual(result, control if mode == 'control' else active)
            elif mode in ('world_only', 'other_only'):
                expected = list(control if mode == 'world_only' else active)
                expected[25] = active[25] if mode == 'world_only' else control[25]
                self.assertEqual(result, expected)
        self.assertEqual([control, active], originals)

    def test_equal_arms_are_bitwise_identity_in_every_mode(self):
        control = state((2., .4, .6, .2)); control[23] = -0.
        for mode in MODES:
            self.assertEqual([x.hex() for x in compose(control, control, mode)], [x.hex() for x in control])

    def test_normalization_and_quaternion_sign_do_not_change_geometry(self):
        control = state((2., .4, .6, .2)); active = state(); active[25] = -.8
        expected = compose(control, active, 'body_protected')
        for factor in (-1., 8.):
            scaled = list(control); scaled[3:7] = [factor * q for q in control[3:7]]
            result = compose(scaled, active, 'body_protected')
            for i in (23, 24, 25): self.assertAlmostEqual(result[i], expected[i], places=14)
            self.assertEqual(result[3:7], scaled[3:7])
            self.assertLess(max(preservation_receipt(scaled, active, result)['body_xy_abs_gap']), 1e-14)

    def test_horizontal_and_near_horizontal_are_rejected_without_fallback(self):
        for angle in (math.pi / 2, math.radians(80)):
            control = state((math.cos(angle / 2), 0., math.sin(angle / 2), 0.))
            with self.assertRaisesRegex(ValueError, 'Rzz'):
                compose(control, control, 'body_protected')
            self.assertEqual(compose(control, control, 'world_only'), control)

    def test_exact_boundary_and_negative_rzz_are_allowed(self):
        for quaternion in ((math.sqrt(.75), 0., .5, 0.), (0., 1., 0., 0.)):
            control = state(quaternion); active = state(); active[25] = -.7
            receipt = preservation_receipt(control, active, compose(control, active, 'body_protected'))
            self.assertGreaterEqual(abs(receipt['Rzz']), .5)
            self.assertTrue(receipt['world_vz_exact'])
            self.assertLess(max(receipt['body_xy_abs_gap']), 1e-14)

    def test_nonvelocity_and_world_exactness_include_signed_zero(self):
        control = state(); active = state(); active[25] = -0.
        result = compose(control, active, 'body_protected')
        self.assertEqual(result[25].hex(), (-0.).hex())
        result[0] = -0.
        self.assertFalse(preservation_receipt(control, active, result)['unchanged_nonvelocity'])

    def test_invalid_shape_nonfinite_quaternion_and_mode_are_rejected(self):
        bad = [[], [0.] * 44, [0.] * 46, [0.] * 45]
        for index, value in ((2, math.nan), (25, math.inf), (4, -math.inf), (30, True)):
            item = state(); item[index] = value; bad.append(item)
        for values in bad:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                compose(state(), values, 'control')
            with self.assertRaises((TypeError, ValueError)):
                preservation_receipt(state(), state(), values)
        with self.assertRaises(ValueError): compose(state(), state(), 'automatic_fallback')


if __name__ == '__main__':
    unittest.main()
