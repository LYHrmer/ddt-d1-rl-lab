"""Pure unittest fixtures from complete72 plus explicit rule boundary cases."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools76.reasons76 import explain_response, explain_selfcheck


def receipt(ad, fd, agrees, direction):
    return {'gradient': ad, 'model_central_difference': fd,
            'autograd_agrees': agrees, 'direction_identifiable': direction}


class ResponseReasons76Tests(unittest.TestCase):
    def test_archived72_low_body_vx_is_agreement_without_magnitude_qualification(self):
        # report72 $.models[0].heads.fresh.points[5].body.vx.eps[0]
        # control64001 / physical72302 tick410 / theta epsilon3.125e-5.
        ad, fd, line = 9.70496064039949e-6, 9.704960678841301e-6, 9.704960682093908e-6
        result = explain_response(fd, 1.7519780702579535e-5, True,
            {'endpoint': receipt(ad, fd, True, False), 'line': receipt(ad, line, True, False)}, False)
        self.assertEqual(result['reasons'], ['MODEL_BELOW_MAGNITUDE_FLOOR'])
        self.assertTrue(result['all_recorded_selfchecks_agree'])
        self.assertFalse(result['all_recorded_directions_identifiable'])
        self.assertTrue(result['raw_sign_same'])
        self.assertFalse(result['qualified_match'])
        self.assertFalse(result['original_valid_match'])
        self.assertAlmostEqual(result['selfchecks']['endpoint']['absolute_error'], 3.844181104082495e-14, delta=1e-25)

    def test_archived72_resolved_physical_opposite_is_not_model_selfcheck_conflict(self):
        # $.models[5].heads.old64.points[4].cost_h1
        # active64003 / physical64302 tick260 / free H1 cost.
        ad, fd = -.0003199814705968482, -.000319981470153774
        result = explain_response(ad, .0006111541339809401, True,
                                  {'cost': receipt(ad, fd, True, True)}, False)
        self.assertEqual(result['reasons'], ['QUALIFIED_MODEL_PHYSICS_OPPOSITE_SIGN'])
        self.assertTrue(result['model_qualification'])
        self.assertTrue(result['raw_sign_opposite'])
        self.assertLess(result['signed_model_to_true_gain_ratio'], 0.)
        self.assertFalse(result['qualified_match'])

    def test_archived72_unknown_physical_reference_is_not_rejudged(self):
        # $.models[0].diagnostics.old64.points[0], control64001 /64301 tick110/H32.
        ad, fd = .0013716619434725643, .0013716642385946942
        result = explain_response(ad, -.0010469808913310397, False,
                                  {'cost': receipt(ad, fd, True, True)}, None)
        self.assertEqual(result['reasons'], ['PHYSICAL_REFERENCE_UNRESOLVED'])
        self.assertTrue(result['raw_sign_opposite'])
        self.assertIsNone(result['qualified_match'])
        self.assertIsNone(result['qualified_model_physics_opposite_sign'])
        self.assertIsNone(result['signed_model_to_true_gain_ratio'])
        self.assertIsNone(result['original_valid_match'])

    def test_absolute_tolerance_can_accept_opposite_signs_and_large_relative_error(self):
        result = explain_response(-2e-5, -4e-5, True,
                                  {'endpoint': receipt(2e-5, -2e-5, True, False)}, False)
        self.assertEqual(result['reasons'], ['MODEL_SELF_SIGN_CONFLICT'])
        check = result['selfchecks']['endpoint']
        self.assertTrue(check['autograd_agrees'])
        self.assertEqual(check['relative_error_to_abs_fd'], 2.)
        self.assertTrue(check['analytic_magnitude_admitted'] and check['fd_magnitude_admitted'])
        self.assertFalse(result['model_qualification'])
        self.assertFalse(result['rules']['agreement_implies_small_relative_error'])
        self.assertFalse(result['rules']['agreement_implies_identifiable_direction'])

    def test_multiple_failures_are_retained_independently(self):
        result = explain_response(-1e-6, 1., False,
                                  {'endpoint': receipt(.1, -1e-6, False, False)}, False)
        self.assertEqual(set(result['reasons']), {'AD_FD_DISAGREEMENT', 'MODEL_BELOW_MAGNITUDE_FLOOR',
                                                'MODEL_SELF_SIGN_CONFLICT', 'PHYSICAL_REFERENCE_UNRESOLVED'})
        self.assertTrue(result['raw_sign_opposite'])
        self.assertIsNone(result['qualified_match'])

    def test_zero_has_explicit_flags_and_is_never_opposite_sign(self):
        result = explain_response(0., 2e-5, True, {'endpoint': receipt(-0., 0., True, False)}, False)
        self.assertEqual(set(result['reasons']), {'MODEL_BELOW_MAGNITUDE_FLOOR',
                                                'MODEL_SELF_ZERO_DERIVATIVE', 'MODEL_RESPONSE_ZERO'})
        self.assertIsNone(result['raw_sign_opposite'])
        self.assertIsNone(result['selfchecks']['endpoint']['opposite_nonzero_sign'])
        self.assertIsNone(result['selfchecks']['endpoint']['relative_error_to_abs_fd'])
        self.assertEqual(result['signed_model_to_true_gain_ratio'], 0.)
        self.assertFalse(result['qualified_match'])
        self.assertFalse(result['rules']['scalar_zero_implies_zero_action_direction'])

    def test_zero_physical_value_cannot_supply_direction_or_gain(self):
        result = explain_response(2e-5, 0., True, {'endpoint': receipt(2e-5, 2e-5, True, True)}, False)
        self.assertEqual(result['reasons'], ['PHYSICAL_REFERENCE_ZERO'])
        self.assertFalse(result['physical_direction_available'])
        self.assertIsNone(result['qualified_match'])
        self.assertIsNone(result['raw_sign_same'])
        self.assertIsNone(result['signed_model_to_true_gain_ratio'])
        self.assertEqual(result['gain_ratio_unavailable_reason'], 'PHYSICAL_REFERENCE_ZERO')

    def test_missing_selfchecks_never_turn_yaw_fd_into_ad_pass(self):
        for checks in (None, {}):
            with self.subTest(checks=checks):
                result = explain_response(2e-5, 4e-5, True, checks, True)
                self.assertEqual(result['reasons'], ['NO_MODEL_SELFCHECK_RECORDED'])
                self.assertEqual(result['qualification_basis'], 'selfcheck_not_recorded')
                self.assertIsNone(result['all_recorded_selfchecks_agree'])
                self.assertIsNone(result['all_recorded_directions_identifiable'])
                self.assertIsNone(result['model_qualification'])
                self.assertIsNone(result['qualified_match'])
                self.assertTrue(result['original_valid_match'])
                self.assertTrue(result['raw_sign_same'])
                self.assertEqual(result['signed_model_to_true_gain_ratio'], .5)

    def test_unrecorded_reference_is_distinct_from_unresolved(self):
        result = explain_response(2e-5)
        self.assertEqual(result['reasons'], ['NO_MODEL_SELFCHECK_RECORDED', 'PHYSICAL_REFERENCE_NOT_RECORDED'])
        self.assertIsNone(result['reference_is_zero'])
        self.assertFalse(result['reference_value_recorded'])
        with_value = explain_response(2e-5, -2e-5)
        self.assertTrue(with_value['raw_sign_opposite'])
        self.assertIsNone(with_value['signed_model_to_true_gain_ratio'])

    def test_exact_floor_is_inclusive(self):
        result = explain_response(1e-5, 2e-5, True, {'endpoint': receipt(1e-5, 1e-5, True, True)}, True)
        self.assertEqual(result['reasons'], [])
        self.assertTrue(result['qualified_match'])

    def test_frozen_booleans_are_recomputed_and_mismatch_is_rejected(self):
        for invalid in (receipt(2e-5, 2e-5, False, True), receipt(2e-5, -2e-5, True, True),
                        receipt(0., 0., True, True), receipt(.1, -.1, True, False)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                explain_selfcheck(invalid)

    def test_all_checks_must_pass_and_inputs_are_not_modified(self):
        checks = {'line': receipt(2e-5, 2e-5, True, True), 'endpoint': receipt(2e-5, 0., True, False)}
        before = copy.deepcopy(checks)
        result = explain_response(2e-5, 4e-5, True, checks, False)
        self.assertTrue(result['all_recorded_selfchecks_agree'])
        self.assertFalse(result['all_recorded_directions_identifiable'])
        self.assertFalse(result['qualified_match'])
        self.assertEqual(checks, before)
        self.assertEqual(set(result['reasons']), {'MODEL_BELOW_MAGNITUDE_FLOOR', 'MODEL_SELF_ZERO_DERIVATIVE'})

    def test_original_decision_is_retained_even_when_explanatory_coverage_differs(self):
        result = explain_response(-2e-5, 2e-5, True, original_valid_match=False)
        self.assertFalse(result['original_valid_match'])
        self.assertFalse(result['original_decision_replaced'])
        self.assertIsNone(result['qualified_match'])
        self.assertNotIn('QUALIFIED_MODEL_PHYSICS_OPPOSITE_SIGN', result['reasons'])

    def test_invalid_receipts_and_nonfinite_values_are_rejected(self):
        for value in (float('nan'), float('inf'), True, '0.1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                explain_response(value)
        for kwargs in ({'reference_gradient': float('nan')}, {'reference_resolved': 1},
                       {'reference_resolved': True}, {'checks': []}, {'original_valid_match': 1},
                       {'checks': {'endpoint': {}}}, {'checks': {'': receipt(1., 1., True, True)}},
                       {'checks': {'endpoint': receipt(1., 1., 1, True)}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                explain_response(1., **kwargs)

    def test_gain_overflow_and_difference_overflow_are_json_safe(self):
        result = explain_response(1e308, 1e-308, True,
                                  {'endpoint': receipt(1e308, -1e308, False, False)}, False)
        self.assertIsNone(result['signed_model_to_true_gain_ratio'])
        self.assertEqual(result['gain_ratio_unavailable_reason'], 'GAIN_RATIO_NOT_FINITE')
        self.assertFalse(result['selfchecks']['endpoint']['absolute_error_finite'])
        self.assertIsNone(result['selfchecks']['endpoint']['absolute_error'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
