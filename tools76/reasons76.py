"""Explain archived scalar response gates without changing their decisions.

Inputs must retain their original derivative coordinate and units. This module
does not normalize action directions, infer missing JVPs, select epsilon, or
replace an archived valid_match. In particular, an absolute-tolerance pass is
neither a small-relative-error claim nor proof of a physical direction.
"""
from __future__ import annotations

import math

ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = .1
MAGNITUDE_FLOOR = 1e-5


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _finite(value, name):
    _require(type(value) in (int, float), name + ' must be a real scalar, not a boolean')
    value = float(value)
    _require(math.isfinite(value), name + ' must be finite')
    return value


def _optional_boolean(value, name):
    _require(value is None or type(value) is bool, name + ' must be boolean or None')


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def _finite_ratio(numerator, denominator):
    if denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def explain_selfcheck(receipt):
    """Strictly verify frozen65's two booleans and expose independent reasons."""
    _require(isinstance(receipt, dict), 'selfcheck receipt must be a dict')
    required = ('gradient', 'model_central_difference', 'autograd_agrees', 'direction_identifiable')
    _require(all(key in receipt for key in required), 'incomplete selfcheck receipt')
    analytic = _finite(receipt['gradient'], 'gradient')
    numerical = _finite(receipt['model_central_difference'], 'model_central_difference')
    _require(type(receipt['autograd_agrees']) is bool and type(receipt['direction_identifiable']) is bool,
             'archived selfcheck gates must be booleans')
    gap = abs(analytic - numerical)
    tolerance = RELATIVE_TOLERANCE * abs(numerical) + ABSOLUTE_TOLERANCE
    agrees = gap <= tolerance
    analytic_floor, numerical_floor = abs(analytic) >= MAGNITUDE_FLOOR, abs(numerical) >= MAGNITUDE_FLOOR
    # Keep the exact archived arithmetic, including its strict product sign.
    identifiable = analytic_floor and numerical_floor and analytic * numerical > 0
    _require(receipt['autograd_agrees'] == agrees, 'archived autograd_agrees differs from frozen formula')
    _require(receipt['direction_identifiable'] == identifiable,
             'archived direction_identifiable differs from frozen formula')
    signs = (_sign(analytic), _sign(numerical))
    nonzero = all(signs)
    return {'original_receipt': dict(receipt), 'gradient': analytic, 'model_central_difference': numerical,
            'autograd_agrees': agrees, 'direction_identifiable': identifiable,
            'absolute_error': gap if math.isfinite(gap) else None,
            'absolute_error_finite': math.isfinite(gap), 'agreement_tolerance': tolerance,
            'relative_error_to_abs_fd': _finite_ratio(gap, abs(numerical)),
            'analytic_magnitude_admitted': analytic_floor, 'fd_magnitude_admitted': numerical_floor,
            'analytic_is_zero': analytic == 0, 'fd_is_zero': numerical == 0,
            'same_nonzero_sign': signs[0] == signs[1] if nonzero else None,
            'opposite_nonzero_sign': signs[0] != signs[1] if nonzero else None}


def explain_response(model_fd, reference_gradient=None, reference_resolved=None,
                     checks=None, original_valid_match=None):
    """Explain one archived response, keeping absent evidence explicitly absent.

    ``checks`` maps names such as endpoint/line to frozen selfcheck receipts.
    None or an empty mapping means no model selfcheck was recorded; both the
    explanatory model qualification and qualified match then remain None.
    ``reference_resolved`` distinguishes not recorded(None), unresolved(False)
    and resolved(True). A resolved reference requires a recorded finite value.
    Zero is explicit and never called an opposite sign. A scalar zero does not
    establish that the action-direction vector was zero.
    """
    model = _finite(model_fd, 'model_fd')
    reference = None if reference_gradient is None else _finite(reference_gradient, 'reference_gradient')
    _optional_boolean(reference_resolved, 'reference_resolved')
    _optional_boolean(original_valid_match, 'original_valid_match')
    _require(reference_resolved is not True or reference is not None, 'resolved physical reference requires its value')
    _require(checks is None or isinstance(checks, dict), 'checks must be a named mapping or None')
    checks = {} if checks is None else checks
    _require(all(isinstance(name, str) and name for name in checks), 'selfcheck names must be nonempty strings')
    explained = {name: explain_selfcheck(receipt) for name, receipt in checks.items()}
    recorded = bool(explained)
    all_agree = all(c['autograd_agrees'] for c in explained.values()) if recorded else None
    all_directions = all(c['direction_identifiable'] for c in explained.values()) if recorded else None
    all_magnitudes = all(c['analytic_magnitude_admitted'] and c['fd_magnitude_admitted']
                         for c in explained.values()) if recorded else None
    model_floor = abs(model) >= MAGNITUDE_FLOOR
    model_qualified = bool(model_floor and all_agree and all_directions) if recorded else None
    reference_nonzero = reference is not None and reference != 0
    raw_nonzero = reference_nonzero and model != 0
    same = _sign(model) == _sign(reference) if raw_nonzero else None
    opposite = not same if raw_nonzero else None
    physical_direction_available = reference_resolved is True and reference_nonzero
    can_classify = model_qualified is not None and physical_direction_available
    qualified_match = bool(model_qualified and same is True) if can_classify else None
    qualified_opposite = bool(model_qualified and opposite is True) if can_classify else None
    if reference_resolved is not True:
        gain, gain_missing = None, 'PHYSICAL_REFERENCE_NOT_RECORDED' if reference_resolved is None else 'PHYSICAL_REFERENCE_UNRESOLVED'
    elif reference == 0:
        gain, gain_missing = None, 'PHYSICAL_REFERENCE_ZERO'
    else:
        gain = _finite_ratio(model, reference)
        gain_missing = None if gain is not None else 'GAIN_RATIO_NOT_FINITE'
    conditions = {
        'NO_MODEL_SELFCHECK_RECORDED': not recorded,
        'AD_FD_DISAGREEMENT': any(not c['autograd_agrees'] for c in explained.values()),
        'MODEL_BELOW_MAGNITUDE_FLOOR': not model_floor or all_magnitudes is False,
        'MODEL_SELF_SIGN_CONFLICT': any(c['opposite_nonzero_sign'] is True for c in explained.values()),
        'MODEL_SELF_ZERO_DERIVATIVE': any(c['analytic_is_zero'] or c['fd_is_zero'] for c in explained.values()),
        'MODEL_RESPONSE_ZERO': model == 0,
        'PHYSICAL_REFERENCE_NOT_RECORDED': reference_resolved is None,
        'PHYSICAL_REFERENCE_UNRESOLVED': reference_resolved is False,
        'PHYSICAL_REFERENCE_ZERO': reference is not None and reference == 0,
        'QUALIFIED_MODEL_PHYSICS_OPPOSITE_SIGN': qualified_opposite is True,
    }
    return {'schema': 'd1-response-reasons76-v1', 'model_fd': model, 'reference_gradient': reference,
            'reference_resolved': reference_resolved, 'reference_value_recorded': reference is not None,
            'physical_direction_available': physical_direction_available,
            'original_valid_match': original_valid_match, 'original_decision_replaced': False,
            'selfchecks': explained, 'selfcheck_count': len(explained), 'selfchecks_recorded': recorded,
            'all_recorded_selfchecks_agree': all_agree, 'all_recorded_directions_identifiable': all_directions,
            'all_recorded_selfcheck_magnitudes_admitted': all_magnitudes,
            'model_response_magnitude_admitted': model_floor,
            'qualification_basis': 'recorded_selfchecks_and_response_magnitude' if recorded else 'selfcheck_not_recorded',
            'model_qualification': model_qualified, 'qualified_match': qualified_match,
            'qualified_model_physics_opposite_sign': qualified_opposite,
            'model_is_zero': model == 0, 'reference_is_zero': reference == 0 if reference is not None else None,
            'raw_sign_same': same, 'raw_sign_opposite': opposite,
            'signed_model_to_true_gain_ratio': gain, 'gain_ratio_unavailable_reason': gain_missing,
            'reason_flags': conditions, 'reasons': [reason for reason, present in conditions.items() if present],
            'rules': {'absolute_tolerance': ABSOLUTE_TOLERANCE, 'relative_tolerance': RELATIVE_TOLERANCE,
                      'magnitude_floor': MAGNITUDE_FLOOR, 'magnitude_comparison': 'inclusive',
                      'agreement_implies_small_relative_error': False,
                      'agreement_implies_identifiable_direction': False,
                      'scalar_zero_implies_zero_action_direction': False,
                      'new_epsilon_or_threshold_selected': False}}
