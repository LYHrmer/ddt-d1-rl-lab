"""Pure receipt tests for75 target isolation and preserved regression gates."""
from __future__ import annotations

import math

import pytest

from tools75.compare75 import (
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
                {'epsilon': e, 'model_fd': -.9} for e in EPSILONS]} for p in _points(role)]) for role in FIRST_ROLES}
            heads = {role: _ledger([p | {'status': 'COMPLETE', 'body': {channel: {'eps': [
                {'epsilon': e, 'model_fd': -.9, 'numeric_valid': True} for e in EPSILONS]} for channel in ('vx', 'vy')},
                'cost_h1': {'gradient': -1., 'autograd_agrees': True, 'direction_identifiable': True}}
                for p in _points(role)]) for role in FIRST_ROLES}
            world = {role: _ledger([p | {'status': 'COMPLETE', 'eps': [
                {'epsilon': e, 'model_fd': -.8 if arm == 'control' else -.9, 'numeric_valid': True} for e in EPSILONS]}
                for p in _points(role)]) for role in FIRST_ROLES}
            training = {'status': 'COMPLETE', 'label_rows': 96, 'rows': [
                {'source': source, 'seed': site, 'tick': tick, 'epsilon': e,
                 'predicted_world_vz_jvp': .3, 'target_float64': .4, 'target_training_float32': .4,
                 'normalized_error': .2 if arm == 'control' else .1}
                for source in ('58', '62') for site in LABEL_SEEDS for tick in TICKS for e in LABEL_EPSILONS]}
            models.append({'seed': seed, 'arm': arm, 'status': 'COMPLETE', 'fit': {'status': 'COMPLETE', 'updates': 1000},
                'diagnostics': diagnostics, 'first_step': firsts, 'heads': heads, 'world_vz': world,
                'final_training_world_vz_diagnostics': training,
                'final_training_label_diagnostics': {'ignored_yaw_diagnostic': True}})
    costs = {role: [p | {'gradient': -1., 'resolved': True} for p in _points(role)] for role in COST_ROLES}
    firsts = {role: [p | {'resolved': True, 'eps': [{'epsilon': e, 'central64': -1.} for e in EPSILONS]}
                    for p in _points(role)] for role in FIRST_ROLES}
    heads = {role: [p | {'body': {channel: {'resolved': True, 'eps': [
        {'epsilon': e, 'central64': -1.} for e in EPSILONS]} for channel in ('vx', 'vy')},
        'cost_h1': {'resolved': True, 'gradient': -1.}} for p in _points(role)] for role in FIRST_ROLES}
    world = {role: [p | {'resolved': True, 'eps': [{'epsilon': e, 'central64': -1.} for e in EPSILONS]}
                   for p in _points(role)] for role in FIRST_ROLES}
    return models, costs, firsts, heads, world


def _model(models, arm='active', seed=64001):
    return next(row for row in models if row['seed'] == seed and row['arm'] == arm)


def _compare(evidence):
    return comparisons(*evidence, eligibility=ELIGIBLE)


def test_denominators_body_weighting_and_world_target_do_not_require_yaw_training_gain():
    result = _compare(_evidence())
    assert result['candidate_accepted'] and result['complete_repair_validated']
    assert (result['cost_points'], result['yaw_first_points'], result['body_channels'],
            result['cost_h1_points'], result['world_vz_points']) == (63, 45, 90, 45, 45)
    for row in result['models']:
        assert (row['cost_valid_matches'], row['first_step_valid_matches'], row['body_valid_matches'],
                row['cost_h1_valid_matches'], row['world_vz_valid_matches']) == (63, 45, 90, 45, 45)
        assert row['body_error_scalar_count'] == 360
        assert row['body_normalized_mse'] == pytest.approx((.1 / .30000001192092896) ** 2)
    assert all(p['fresh_world_vz_rmse_ratio_active_over_control'] == pytest.approx(.5) for p in result['paired'])


def test_training_requires_all96_world_rows_and_every_seed():
    evidence = _evidence()
    for row in _model(evidence[0], seed=64003)['final_training_world_vz_diagnostics']['rows']:
        row['normalized_error'] = 0. if row['source'] == '58' else .4
    result = _compare(evidence)
    assert result['paired'][2]['active_training_world_vz_rmse'] == pytest.approx(math.sqrt(.08))
    assert not result['training_improved'] and not result['candidate_accepted']
    assert result['generalization_target_passed']


def test_training_equality_is_not_improvement():
    evidence = _evidence()
    for row in _model(evidence[0])['final_training_world_vz_diagnostics']['rows']:
        row['normalized_error'] = .2
    assert not _compare(evidence)['training_improved']


def test_fresh_rmse_uses_all_four_scales_for_each_seed():
    evidence = _evidence()
    for point in _model(evidence[0], seed=64003)['world_vz']['fresh']['points']:
        point['eps'][3]['model_fd'] = -.3
    result = _compare(evidence)
    assert not result['raw_fresh_world_vz_rmse_gain']
    assert result['generalization_target_status'] == 'FAILED'
    assert result['no_regressions']


@pytest.mark.parametrize('kind', ('h32', 'h1', 'yaw', 'body', 'world_vz'))
def test_old72_is_formal_and_no_new_failure_can_be_paid_for_by_other_gains(kind):
    evidence = _evidence()
    model = _model(evidence[0])
    if kind == 'h32':
        model['diagnostics']['old72']['points'][0]['gradient'] = 1.
    elif kind == 'h1':
        model['heads']['old72']['points'][0]['cost_h1']['gradient'] = 1.
    elif kind == 'body':
        model['heads']['old72']['points'][0]['body']['vx']['eps'][0]['model_fd'] = .9
    else:
        key = 'first_step' if kind == 'yaw' else 'world_vz'
        model[key]['old72']['points'][0]['eps'][0]['model_fd'] = .9
    result = _compare(evidence)
    assert result['generalization_target_passed']
    assert not result['no_regressions'] and not result['candidate_accepted']
    assert not result['complete_repair_validated']


def test_world_per_epsilon_regression_is_detected_even_when_point_count_is_unchanged():
    evidence = _evidence()
    control, active = [_model(evidence[0], arm)['world_vz']['old69']['points'][0]['eps'] for arm in ARMS]
    control[0]['model_fd'], active[1]['model_fd'] = .8, .9
    result = _compare(evidence)
    assert result['paired'][0]['new_world_vz_qualification_failures'] == [
        {'role': 'old69', 'seed': 69301, 'tick': 110, 'channel': 'world_vz', 'epsilon': EPSILONS[1],
         'lost_qualifications': ['combined_qualification']}]


def test_world_magnitude_regression_is_known_even_if_baseline_was_already_wrong():
    evidence = _evidence()
    left, right = [_model(evidence[0], arm)['world_vz']['old72']['points'][0]['eps'][0] for arm in ARMS]
    left['model_fd'], right['model_fd'] = .8, 1e-6
    result = _compare(evidence)
    assert result['paired'][0]['new_world_vz_qualification_failures'][0]['lost_qualifications'] == ['magnitude']
    assert result['no_regressions_status'] == 'FAILED'


@pytest.mark.parametrize('role', ROLES)
def test_state_guard_still_applies_to_all_eight_roles(role):
    evidence = _evidence()
    _model(evidence[0])['diagnostics'][role].update(mean_state_group_loss=.276,
        mean_state_group_losses=dict.fromkeys(STATE_GROUPS, .276))
    result = _compare(evidence)
    assert not result['paired'][0]['state_loss'][role]['passed_le1p1']
    assert not result['no_regressions']


def test_five_role_body_rmse_guard_is_retained_without_a_world_global_rmse_gate():
    evidence = _evidence()
    model = _model(evidence[0])
    for point in model['world_vz']['old72']['points']:
        for e in point['eps']:
            e['model_fd'] = -.5
    assert _compare(evidence)['candidate_accepted']  # World regression signs still pass; no new global RMSE gate.
    model['heads']['old72']['points'][0]['body']['vy']['eps'][0]['model_fd'] = -.8
    result = _compare(evidence)
    assert not result['paired'][0]['body_rmse_nonincreasing'] and not result['candidate_accepted']


def test_prior_unknown_does_not_erase_fresh_target_or_known_regression():
    evidence = _evidence()
    evidence[1]['old64'][0]['resolved'] = False
    result = _compare(evidence)
    assert result['generalization_target_passed']
    assert result['candidate_status'] == result['complete_repair_status'] == 'REFERENCE_UNRESOLVED'
    _model(evidence[0])['diagnostics']['old72']['points'][0]['gradient'] = 1.
    result = _compare(evidence)
    assert result['candidate_status'] == result['complete_repair_status'] == 'FAILED'


def test_fresh_unknown_keeps_raw_error_but_numeric_failure_still_takes_precedence():
    evidence = _evidence()
    evidence[4]['fresh'][0]['resolved'] = False
    for point in _model(evidence[0])['world_vz']['fresh']['points']:
        for e in point['eps']:
            e['model_fd'] = -.7
    result = _compare(evidence)
    assert not result['raw_fresh_world_vz_rmse_gain']
    assert result['generalization_target_status'] == 'REFERENCE_UNRESOLVED'
    _model(evidence[0])['world_vz']['fresh']['points'][1]['eps'][0]['numeric_valid'] = False
    assert _compare(evidence)['generalization_target_status'] == 'FAILED'


@pytest.mark.parametrize('kind', ('old_h32', 'old_h1', 'old_body', 'old_yaw_magnitude', 'world_vz'))
def test_fresh_existing_numeric_and_new_world_checks_all_remain_required(kind):
    evidence = _evidence()
    model = _model(evidence[0], 'control')
    if kind == 'old_h32':
        model['diagnostics']['fresh']['points'][0]['autograd_agrees'] = False
    elif kind == 'old_h1':
        model['heads']['fresh']['points'][0]['cost_h1']['direction_identifiable'] = False
    elif kind == 'old_body':
        model['heads']['fresh']['points'][0]['body']['vx']['eps'][0]['numeric_valid'] = False
    elif kind == 'old_yaw_magnitude':
        model['first_step']['fresh']['points'][0]['eps'][0]['model_fd'] = -1e-6
    else:
        model['world_vz']['fresh']['points'][0]['eps'][0]['numeric_valid'] = False
    result = _compare(evidence)
    assert result['raw_fresh_world_vz_rmse_gain'] and not result['generalization_target_passed']
    assert result['complete_repair_validated']  # Active absolute checks are independent of control's failure.


@pytest.mark.parametrize('problem', ('partial_world', 'missing_old72', 'training_target', 'missing_label', 'nonfinite'))
def test_partial_missing_or_changed_evidence_is_not_a_complete_candidate(problem):
    evidence = _evidence()
    model = _model(evidence[0])
    if problem == 'partial_world':
        model['world_vz']['fresh']['status'] = 'INCOMPLETE'
    elif problem == 'missing_old72':
        del evidence[4]['old72']
    elif problem == 'training_target':
        model['final_training_world_vz_diagnostics']['rows'][0]['target_float64'] = .7
    elif problem == 'missing_label':
        model['final_training_world_vz_diagnostics']['rows'].pop()
    else:
        model['world_vz']['fresh']['points'][0]['eps'][0]['model_fd'] = float('nan')
    with pytest.raises(ValueError):
        _compare(evidence)


def test_pairing_and_reproduction_still_refer_to69_active():
    result = comparisons(*_evidence(), eligibility={'pairing_complete': True, 'control_reproduced69_active': False})
    assert not result['candidate_accepted'] and not result['pairing_and_reproduction_eligible']
