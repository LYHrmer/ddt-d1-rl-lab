"""Pure receipt tests: no model, training, NumPy, Torch or physical execution."""
from __future__ import annotations

import pytest

from tools69.run69 import (
    ARMS,
    COST_ROLES,
    EPSILONS,
    FIRST_ROLES,
    FIXED_BETA,
    FIXED_LAMBDA,
    MODEL_SEEDS,
    PAIR_FIELDS,
    ROLES,
    check_pairing,
    comparisons,
)

ROLE_SEEDS = {'development': (53201, 53202, 53203), 'old_probe': (53301, 53302, 53303),
              'old55_local': (55301, 55302, 55303), 'old56': (56301, 56302, 56303),
              'old64': (64301, 64302, 64303), 'fresh': (69301, 69302, 69303)}


def _points(role):
    return [{'seed': seed, 'tick': tick} for seed in ROLE_SEEDS[role] for tick in (110, 260, 410)]


def _evidence():
    models = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            diagnostics = {role: {'points': [p | {'gradient': -.5, 'autograd_agrees': True,
                'direction_identifiable': True} for p in _points(role)], 'mean_state_group_loss': .25,
                'mean_state_group_losses': {'synthetic': .25}} for role in ROLES}
            first = {role: {'points': [p | {'eps': [{'epsilon': e, 'model_fd': -.5} for e in EPSILONS]}
                                     for p in _points(role)]} for role in FIRST_ROLES}
            heads = {role: {'points': [p | {'status': 'COMPLETE',
                'body': {channel: {'eps': [{'epsilon': e, 'model_fd': -.8 if arm == 'control' else -.9,
                                            'numeric_valid': True} for e in EPSILONS]} for channel in ('vx', 'vy')},
                'cost_h1': {'gradient': -.5, 'autograd_agrees': True, 'direction_identifiable': True}}
                for p in _points(role)]} for role in FIRST_ROLES}
            models.append({'seed': seed, 'arm': arm, 'diagnostics': diagnostics, 'first_step': first, 'heads': heads})
    costs = {role: [p | {'gradient': -.5, 'resolved': True} for p in _points(role)] for role in COST_ROLES}
    firsts = {role: [p | {'resolved': True, 'eps': [{'epsilon': e, 'central64': -.5} for e in EPSILONS]}
                    for p in _points(role)] for role in FIRST_ROLES}
    heads = {role: [p | {'body': {channel: {'resolved': True, 'eps': [
        {'epsilon': e, 'central64': -1.} for e in EPSILONS]} for channel in ('vx', 'vy')},
        'cost_h1': {'resolved': True, 'gradient': -.5}} for p in _points(role)] for role in FIRST_ROLES}
    return models, costs, firsts, heads


def _model(rows, arm='active', seed=MODEL_SEEDS[0]):
    return next(row for row in rows if row['seed'] == seed and row['arm'] == arm)


def test_registered_denominators_and_body_gain_do_not_fabricate_direction_gain():
    result = comparisons(*_evidence())
    assert result['active_repair_validated'] and result['body_target_improved']
    assert not result['paired_improved'] and not result['paired_any_strict_improvement']
    assert all((r['cost_valid_matches'], r['first_step_valid_matches'], r['body_valid_matches'],
                r['cost_h1_valid_matches']) == (45, 27, 54, 27) for r in result['models'])
    assert all(p['body_rmse_ratio_active_over_control'] == pytest.approx(.5) for p in result['paired'])


def test_old64301_unknown_remains_in_formal_denominator_without_counting_a_sign_error():
    models, costs, firsts, heads = _evidence()
    costs['old64'][0]['resolved'] = False
    _model(models)['diagnostics']['old64']['points'][0]['gradient'] = .5
    result = comparisons(models, costs, firsts, heads)
    assert result['active_status'] == result['paired_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['active_repair_validated'] and not result['paired_improved']
    point = _model(result['models'])['cost']['old64']['points'][0]
    assert point['valid_match'] is point['same_sign'] is None
    assert _model(result['models'])['cost_valid_matches'] == 44


@pytest.mark.parametrize('kind', ('cost', 'yaw', 'body', 'h1'))
@pytest.mark.parametrize('role', FIRST_ROLES)
def test_each_new_and_historical_output_is_required(kind, role):
    models, costs, firsts, heads = _evidence()
    row = _model(models)
    if kind == 'cost':
        row['diagnostics'][role]['points'][0]['gradient'] = .5
    elif kind == 'yaw':
        row['first_step'][role]['points'][0]['eps'][2]['model_fd'] = .5
    elif kind == 'body':
        row['heads'][role]['points'][0]['body']['vy']['eps'][2]['model_fd'] = .9
    else:
        row['heads'][role]['points'][0]['cost_h1']['gradient'] = .5
    result = comparisons(models, costs, firsts, heads)
    assert not result['active_repair_validated']


@pytest.mark.parametrize('field', ('autograd_agrees', 'direction_identifiable'))
@pytest.mark.parametrize('arm', ARMS)
def test_control_numeric_failure_does_not_revoke_active_absolute_success(field, arm):
    evidence = _evidence()
    _model(evidence[0], arm)['diagnostics']['development']['points'][0][field] = False
    result = comparisons(*evidence)
    assert result['active_repair_validated'] is (arm == 'control')
    assert result['paired_status'] == result['target_status'] == 'NUMERIC_UNRESOLVED'


@pytest.mark.parametrize('scale', range(4))
def test_body_needs_every_scale_and_numerical_qualification(scale):
    evidence = _evidence()
    p = _model(evidence[0])['heads']['fresh']['points'][0]['body']['vx']['eps'][scale]
    p.update(model_fd=-1e-6, numeric_valid=False)
    result = comparisons(*evidence)
    assert not result['active_repair_validated'] and not result['body_target_improved']
    assert _model(result['models'])['body_valid_matches'] == 53


@pytest.mark.parametrize('kind', ('body', 'h1'))
def test_new_output_unknown_is_not_packaged_as_false_direction(kind):
    models, costs, firsts, heads = _evidence()
    ref = heads['old64'][0]['body']['vx'] if kind == 'body' else heads['old64'][0]['cost_h1']
    ref['resolved'] = False
    result = comparisons(models, costs, firsts, heads)
    assert result['active_status'] == result['paired_status'] == 'REFERENCE_UNRESOLVED'
    point = _model(result['models'])['heads']['old64']['points'][0]
    if kind == 'body':
        assert point['body']['vx']['passed'] is None
        assert result['target_status'] == 'REFERENCE_UNRESOLVED'
    else:
        assert point['cost_h1']['valid_match'] is None


def test_paired_gain_requires_a_real_count_gain_in_addition_to_body_rmse():
    evidence = _evidence()
    _model(evidence[0], 'control')['diagnostics']['fresh']['points'][0]['gradient'] = .5
    result = comparisons(*evidence)
    assert result['paired_improved'] and result['body_target_improved']


@pytest.mark.parametrize('horizon', (1, 32))
def test_net_count_gain_cannot_hide_a_new_resolved_cost_failure(horizon):
    evidence = _evidence()
    for index in (0, 1):
        _model(evidence[0], 'control')['diagnostics']['fresh']['points'][index]['gradient'] = .5
    active = _model(evidence[0])
    if horizon == 32:
        active['diagnostics']['old64']['points'][4]['gradient'] = .5
    else:
        active['heads']['old64']['points'][4]['cost_h1']['gradient'] = .5
    result = comparisons(*evidence)
    assert result['paired_non_decreasing'] and result['paired_any_strict_improvement']
    assert result['raw_body_rmse_gain'] and not result['body_target_improved']
    assert not result['paired_improved'] and not result['no_new_resolved_H1_H32_failure']
    assert result['paired'][0]['new_resolved_cost_failures'] == [
        {'role': 'old64', 'horizon': horizon, 'seed': 64302, 'tick': 260}]


def test_yaw_count_regression_blocks_target_even_with_better_body_and_cost():
    evidence = _evidence()
    _model(evidence[0], 'control')['diagnostics']['fresh']['points'][0]['gradient'] = .5
    _model(evidence[0])['first_step']['old64']['points'][0]['eps'][0]['model_fd'] = .5
    result = comparisons(*evidence)
    assert result['raw_body_rmse_gain'] and not result['body_target_improved']
    assert not result['paired_improved']


@pytest.mark.parametrize('horizon', (1, 32))
def test_lost_numeric_qualification_is_also_a_new_cost_failure(horizon):
    evidence = _evidence()
    model = _model(evidence[0])
    point = (model['diagnostics']['old64']['points'][4] if horizon == 32
             else model['heads']['old64']['points'][4]['cost_h1'])
    point['direction_identifiable'] = False
    result = comparisons(*evidence)
    assert not result['no_new_resolved_H1_H32_failure']
    assert result['paired'][0]['new_resolved_cost_failures'] == [
        {'role': 'old64', 'horizon': horizon, 'seed': 64302, 'tick': 260}]


def test_unknown_old64_group_cannot_be_dropped_from_denominator():
    models, costs, firsts, heads = _evidence()
    del costs['old64']
    with pytest.raises(ValueError, match='registered45/27'):
        comparisons(models, costs, firsts, heads)


def test_rmse_improvement_cannot_be_averaged_across_model_initializations():
    evidence = _evidence()
    _model(evidence[0], 'control')['diagnostics']['fresh']['points'][0]['gradient'] = .5
    for role in FIRST_ROLES:
        for p in _model(evidence[0], seed=MODEL_SEEDS[2])['heads'][role]['points']:
            for channel in ('vx', 'vy'):
                for e in p['body'][channel]['eps']:
                    e['model_fd'] = -.81
    result = comparisons(*evidence)
    assert result['paired'][2]['body_rmse_ratio_active_over_control'] == pytest.approx(.95)
    assert not result['body_target_improved'] and not result['paired_improved']


def _fits():
    shared = {key: 'same-' + key for key in PAIR_FIELDS if key != 'initial_model_state_sha256'}
    expected = {'common': shared.copy(), 'by_arm': {
        arm: {'fixed_beta': beta, 'body_coefficient': beta, 'beta_sha256': 'beta-' + arm}
        for arm, beta in (('control', 0.), ('active', FIXED_BETA))},
        'control_final_sha256': {str(seed): f'old64-active-{seed}' for seed in MODEL_SEEDS}}
    rows = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            fit = shared | expected['by_arm'][arm] | {'status': 'COMPLETE', 'eligible_for_evaluation': True,
                'updates': 1000, 'updates_attempted': 1000, 'seed': seed, 'arm': arm, 'horizon': 8,
                'training_transitions': 7200, 'training_windows': 6828, 'training_episodes': list(range(12)),
                'initial_model_state_sha256': f'init-{seed}', 'fixed_lambda': FIXED_LAMBDA,
                'effective_lambda': FIXED_LAMBDA, 'calibration_performed': False, 'second_label_source': '62',
                'final_model_state_sha256': f'old64-active-{seed}' if arm == 'control' else f'new-{seed}'}
            rows.append({'seed': seed, 'arm': arm, 'fit': fit})
    return rows, expected


def test_control_must_reproduce_each_old64_active_mapping_exactly():
    rows, expected = _fits()
    assert check_pairing(rows, expected)['status'] == 'COMPLETE'
    _model(rows, 'control', MODEL_SEEDS[1])['fit']['final_model_state_sha256'] = 'different'
    with pytest.raises(ValueError, match='exactly reproduce64 active'):
        check_pairing(rows, expected)


@pytest.mark.parametrize('identity', PAIR_FIELDS)
def test_pair_identity_changes_are_rejected(identity):
    rows, expected = _fits()
    _model(rows)['fit'][identity] = 'changed'
    with pytest.raises(ValueError):
        check_pairing(rows, expected)


@pytest.mark.parametrize(('key', 'value'), [('updates', 999), ('fixed_lambda', 0.),
                                         ('second_label_source', '60'), ('fixed_beta', 0.),
                                         ('calibration_performed', True)])
def test_partial_or_second_intervention_is_not_an_eligible_pair(key, value):
    rows, expected = _fits()
    _model(rows)['fit'][key] = value
    with pytest.raises(ValueError):
        check_pairing(rows, expected)
