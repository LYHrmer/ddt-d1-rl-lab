"""Synthetic receipt tests only: no numerical library or model execution."""
from __future__ import annotations

import math

import pytest

from tools72.compare72 import (
    ARMS,
    COST_ROLES,
    EPSILONS,
    FIRST_ROLES,
    LABEL_EPSILONS,
    LABEL_SEEDS,
    MODEL_SEEDS,
    ROLE_SEEDS,
    ROLES,
    STATE_GROUPS,
    TICKS,
    comparisons,
)

ELIGIBLE = {'pairing_complete': True, 'control_reproduced69_active': True}


def _points(role):
    return [{'seed': seed, 'tick': tick} for seed in ROLE_SEEDS[role] for tick in TICKS]


def _ledger(points):
    return {'status': 'COMPLETE', 'completed_points': 9, 'points': points}


def _evidence():
    models = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            diagnostics = {role: _ledger([p | {'gradient': -1., 'autograd_agrees': True,
                'direction_identifiable': True} for p in _points(role)]) | {'mean_state_group_loss': .25,
                'mean_state_group_losses': dict.fromkeys(STATE_GROUPS, .25)} for role in ROLES}
            firsts = {role: _ledger([p | {'status': 'COMPLETE', 'eps': [
                {'epsilon': e, 'model_fd': -.8 if arm == 'control' else -.9} for e in EPSILONS]}
                for p in _points(role)]) for role in FIRST_ROLES}
            heads = {role: _ledger([p | {'status': 'COMPLETE', 'body': {channel: {'eps': [
                {'epsilon': e, 'model_fd': -.9, 'numeric_valid': True} for e in EPSILONS]}
                for channel in ('vx', 'vy')}, 'cost_h1': {'gradient': -1., 'autograd_agrees': True,
                    'direction_identifiable': True}} for p in _points(role)]) for role in FIRST_ROLES}
            sources = {source: {'status': 'COMPLETE', 'label_rows': 48, 'independent_states': 12,
                'rows': [{'seed': site, 'tick': tick, 'epsilon': eps,
                         'normalized_error': .2 if arm == 'control' else .1,
                         'predicted_yaw_jvp': .3, 'target_float64': .4, 'target_training_float32': .4}
                        for site in LABEL_SEEDS for tick in TICKS for eps in LABEL_EPSILONS]}
                for source in ('58', '62')}
            sources['60'] = {'status': 'INCOMPLETE', 'ignored_descriptive_source': True}
            models.append({'seed': seed, 'arm': arm, 'status': 'COMPLETE',
                'fit': {'status': 'COMPLETE', 'updates': 1000}, 'diagnostics': diagnostics,
                'first_step': firsts, 'heads': heads,
                'final_training_label_diagnostics': {'status': 'COMPLETE', 'sources': sources}})
    costs = {role: [p | {'gradient': -1., 'resolved': True} for p in _points(role)] for role in COST_ROLES}
    firsts = {role: [p | {'resolved': True, 'eps': [{'epsilon': e, 'central64': -1.} for e in EPSILONS]}
                    for p in _points(role)] for role in FIRST_ROLES}
    heads = {role: [p | {'body': {channel: {'resolved': True, 'eps': [
        {'epsilon': e, 'central64': -1.} for e in EPSILONS]} for channel in ('vx', 'vy')},
        'cost_h1': {'resolved': True, 'gradient': -1.}} for p in _points(role)] for role in FIRST_ROLES}
    return models, costs, firsts, heads


def _model(models, arm='active', seed=64001):
    return next(row for row in models if row['seed'] == seed and row['arm'] == arm)


def _compare(evidence):
    return comparisons(*evidence, eligibility=ELIGIBLE)


def _state(model, role, value):
    model['diagnostics'][role].update(mean_state_group_loss=value,
        mean_state_group_losses=dict.fromkeys(STATE_GROUPS, value))


def test_all_registered_denominators_and_exact_endpoint_rmse():
    result = _compare(_evidence())
    assert result['candidate_accepted'] and result['complete_repair_validated']
    assert result['training_improved'] and result['generalization_target_passed']
    assert (result['cost_points'], result['yaw_first_points'], result['body_channels'], result['cost_h1_points']) == (54, 36, 72, 36)
    for row in result['models']:
        assert (row['cost_valid_matches'], row['first_step_valid_matches'], row['body_valid_matches'],
                row['cost_h1_valid_matches'], row['body_error_scalar_count']) == (54, 36, 72, 36, 288)
        assert row['fresh_yaw_error']['scalar_count'] == 36
    assert all(p['fresh_yaw_rmse_ratio_active_over_control'] == pytest.approx(.5) for p in result['paired'])


def test_training_reduces_all96_rows_and_requires_every_seed_strictly_down():
    evidence = _evidence()
    active = _model(evidence[0], seed=64003)
    for source, error in (('58', 0.), ('62', .4)):
        for row in active['final_training_label_diagnostics']['sources'][source]['rows']:
            row['normalized_error'] = error
    result = _compare(evidence)
    pair = result['paired'][2]
    assert pair['active_training_yaw_rmse'] == pytest.approx(math.sqrt(.08))
    assert not result['training_improved'] and not result['candidate_accepted']
    assert result['generalization_target_passed']


def test_equality_is_not_training_improvement_and_no_seed_can_compensate():
    evidence = _evidence()
    for source in ('58', '62'):
        for row in _model(evidence[0])['final_training_label_diagnostics']['sources'][source]['rows']:
            row['normalized_error'] = .2
    assert not _compare(evidence)['training_improved']


def test_all_four_scales_are_used_for_fresh_error():
    evidence = _evidence()
    model = _model(evidence[0])
    for point in model['first_step']['fresh']['points']:
        point['eps'][3]['model_fd'] = -.3
    result = _compare(evidence)
    assert not result['raw_fresh_yaw_rmse_gain'] and not result['generalization_target_passed']
    assert result['no_regressions']  # Same signs cannot rescue a failed amplitude-error target.


@pytest.mark.parametrize('kind', ('h32', 'h1', 'yaw', 'body'))
def test_old69_is_a_formal_regression_and_new_failures_cannot_be_paid_for_elsewhere(kind):
    evidence = _evidence()
    model = _model(evidence[0])
    if kind == 'h32':
        model['diagnostics']['old69']['points'][0]['gradient'] = 1.
    elif kind == 'h1':
        model['heads']['old69']['points'][0]['cost_h1']['gradient'] = 1.
    elif kind == 'yaw':
        model['first_step']['old69']['points'][0]['eps'][0]['model_fd'] = .9
    else:
        model['heads']['old69']['points'][0]['body']['vx']['eps'][0]['model_fd'] = .9
    result = _compare(evidence)
    assert not result['no_regressions'] and not result['candidate_accepted']
    assert result['generalization_target_passed']


@pytest.mark.parametrize('kind', ('yaw', 'body'))
def test_per_epsilon_regression_is_detected_when_point_counts_are_unchanged(kind):
    evidence = _evidence()
    control, active = [_model(evidence[0], arm) for arm in ARMS]
    if kind == 'yaw':
        left, right = [m['first_step']['old64']['points'][1]['eps'] for m in (control, active)]
    else:
        left, right = [m['heads']['old64']['points'][1]['body']['vx']['eps'] for m in (control, active)]
    left[0]['model_fd'] = .9
    right[1]['model_fd'] = .9
    result = _compare(evidence)
    failures = result['paired'][0]['new_' + kind + '_qualification_failures']
    assert failures == [{'role': 'old64', 'seed': 64301, 'tick': 260, 'channel': 'yaw' if kind == 'yaw' else 'vx',
                         'epsilon': EPSILONS[1], 'lost_qualifications': ['combined_qualification']}]
    assert not result['no_regressions']


@pytest.mark.parametrize('kind', ('yaw', 'body'))
def test_magnitude_regression_is_known_even_when_baseline_direction_is_wrong(kind):
    evidence = _evidence()
    control, active = [_model(evidence[0], arm) for arm in ARMS]
    if kind == 'yaw':
        left, right = [m['first_step']['old69']['points'][0]['eps'][0] for m in (control, active)]
    else:
        left, right = [m['heads']['old69']['points'][0]['body']['vx']['eps'][0] for m in (control, active)]
    left['model_fd'], right['model_fd'] = .9, 1e-6
    result = _compare(evidence)
    failures = result['paired'][0]['new_' + kind + '_qualification_failures']
    assert failures[0]['lost_qualifications'] == ['magnitude']
    assert not result['no_regressions']


def test_numeric_regression_is_known_even_for_unresolved_body_reference():
    evidence = _evidence()
    evidence[3]['old64'][0]['body']['vx']['resolved'] = False
    _model(evidence[0])['heads']['old64']['points'][0]['body']['vx']['eps'][0]['numeric_valid'] = False
    result = _compare(evidence)
    assert result['paired'][0]['new_body_qualification_failures'][0]['lost_qualifications'] == ['numeric']
    assert result['no_regressions_status'] == result['candidate_status'] == 'FAILED'


def test_body_rmse_increase_blocks_candidate_without_direction_failure():
    evidence = _evidence()
    for role in FIRST_ROLES:
        for point in _model(evidence[0])['heads'][role]['points']:
            point['body']['vy']['eps'][0]['model_fd'] = -.8
    result = _compare(evidence)
    assert not result['paired'][0]['body_rmse_nonincreasing']
    assert not result['candidate_accepted'] and result['complete_repair_validated']


@pytest.mark.parametrize('role', ROLES)
def test_state_guard_covers_each_validation_role_including_development(role):
    evidence = _evidence()
    _state(_model(evidence[0]), role, .276)
    result = _compare(evidence)
    assert not result['paired'][0]['state_loss'][role]['passed_le1p1']
    assert not result['no_regressions'] and not result['candidate_accepted']


def test_zero_state_baseline_has_explicit_nondividing_rule():
    evidence = _evidence()
    for arm in ARMS:
        _state(_model(evidence[0], arm), 'fresh', 0.)
    assert _compare(evidence)['paired'][0]['state_loss']['fresh']['passed_le1p1']
    _state(_model(evidence[0]), 'fresh', 1e-12)
    assert not _compare(evidence)['paired'][0]['state_loss']['fresh']['passed_le1p1']


def test_prior_unknown_preserves_fresh_target_but_blocks_final_acceptance():
    evidence = _evidence()
    evidence[1]['old64'][0]['resolved'] = False
    result = _compare(evidence)
    assert result['generalization_target_passed'] and result['training_improved']
    assert result['candidate_status'] == result['complete_repair_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['candidate_accepted'] and not result['complete_repair_validated']
    assert result['prior_unresolved_references'] == [{'role': 'old64', 'kind': 'cost_h32', 'seed': 64301, 'tick': 110}]


def test_unknown_never_hides_a_known_cost_or_target_failure():
    evidence = _evidence()
    evidence[1]['old64'][0]['resolved'] = False
    _model(evidence[0])['diagnostics']['old69']['points'][0]['gradient'] = 1.
    result = _compare(evidence)
    assert result['candidate_status'] == result['complete_repair_status'] == 'FAILED'
    evidence[2]['fresh'][0]['resolved'] = False
    for point in _model(evidence[0])['first_step']['fresh']['points']:
        for eps in point['eps']:
            eps['model_fd'] = -.7
    _model(evidence[0])['first_step']['fresh']['points'][1]['eps'][0]['model_fd'] = .7
    result = _compare(evidence)
    assert result['generalization_target_status'] == 'FAILED'
    assert result['unresolved_references']


def test_unresolved_fresh_raw_rmse_is_not_itself_a_known_direction_or_target_failure():
    evidence = _evidence()
    evidence[2]['fresh'][0]['resolved'] = False
    for point in _model(evidence[0])['first_step']['fresh']['points']:
        for eps in point['eps']:
            eps['model_fd'] = -.7
    result = _compare(evidence)
    assert not result['raw_fresh_yaw_rmse_gain']
    assert result['generalization_target_status'] == result['candidate_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['generalization_target_known_failures']


def test_unresolved_body_rmse_retains_raw_comparison_without_fabricating_failure():
    evidence = _evidence()
    evidence[3]['old64'][0]['body']['vx']['resolved'] = False
    _model(evidence[0])['heads']['old64']['points'][0]['body']['vx']['eps'][0]['model_fd'] = -.1
    result = _compare(evidence)
    assert not result['paired'][0]['body_rmse_nonincreasing']
    assert result['no_regressions_status'] == result['candidate_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['paired'][0]['regression_known_failures']


@pytest.mark.parametrize('field', ('autograd_agrees', 'direction_identifiable'))
def test_baseline_numeric_failure_blocks_pair_but_not_active_absolute_direction(field):
    evidence = _evidence()
    _model(evidence[0], 'control')['diagnostics']['development']['points'][0][field] = False
    result = _compare(evidence)
    assert result['complete_repair_validated'] and result['generalization_target_passed']
    assert not result['all_six_numeric_valid'] and not result['candidate_accepted']


@pytest.mark.parametrize('arm', ARMS)
def test_fresh_magnitude_failure_blocks_target_even_when_rmse_improves(arm):
    evidence = _evidence()
    for ref in evidence[2]['fresh']:
        for eps in ref['eps']:
            eps['central64'] = -1.1e-5
    for model in evidence[0]:
        for point in model['first_step']['fresh']['points']:
            for eps in point['eps']:
                eps['model_fd'] = -.1 if model['arm'] == 'control' else -1.2e-5
    _model(evidence[0], arm)['first_step']['fresh']['points'][0]['eps'][0]['model_fd'] = -9e-6
    result = _compare(evidence)
    assert result['raw_fresh_yaw_rmse_gain']
    assert not result['generalization_target_passed']


@pytest.mark.parametrize('eligibility', (None, {'pairing_complete': True, 'control_reproduced69_active': False}))
def test_no_candidate_without_explicit_reproduction_and_pairing(eligibility):
    result = comparisons(*_evidence(), eligibility=eligibility)
    assert not result['candidate_accepted'] and not result['pairing_and_reproduction_eligible']
    assert result['generalization_target_passed']


@pytest.mark.parametrize('bad', ('partial', 'missing_role', 'duplicate_point', 'nonfinite', 'target_changed', 'missing_epsilon'))
def test_malformed_or_partial_evidence_is_rejected(bad):
    evidence = _evidence()
    model = _model(evidence[0])
    if bad == 'partial':
        model['fit']['updates'] = 999
    elif bad == 'missing_role':
        del evidence[1]['old69']
    elif bad == 'duplicate_point':
        model['diagnostics']['fresh']['points'][1].update(seed=72301, tick=110)
    elif bad == 'nonfinite':
        model['heads']['fresh']['points'][0]['body']['vx']['eps'][0]['model_fd'] = float('nan')
    elif bad == 'target_changed':
        model['final_training_label_diagnostics']['sources']['62']['rows'][0]['target_float64'] = .9
    else:
        model['first_step']['old69']['points'][0]['eps'].pop()
    with pytest.raises(ValueError):
        _compare(evidence)
