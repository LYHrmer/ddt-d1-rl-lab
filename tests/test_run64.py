"""Synthetic decision receipts for64; no model, fitting or physics is invoked."""
from __future__ import annotations

import pytest

from tools64.run64 import (
    ARMS,
    COST_ROLES,
    EPSILONS,
    FIRST_ROLES,
    FIXED_LAMBDA,
    MODEL_SEEDS,
    PAIR_FIELDS,
    ROLES,
    check_pairing,
    comparisons,
)

ROLE_SEEDS = {
    'development': (53201, 53202, 53203),
    'old_probe': (53301, 53302, 53303),
    'old55_local': (55301, 55302, 55303),
    'old56': (56301, 56302, 56303),
    'fresh': (64301, 64302, 64303),
}


def _points(role):
    return [{'seed': seed, 'tick': tick} for seed in ROLE_SEEDS[role] for tick in (110, 260, 410)]


def _evidence():
    models = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            diagnostics = {role: {
                'points': [point | {'gradient': -.5, 'autograd_agrees': True,
                                    'direction_identifiable': True} for point in _points(role)],
                'mean_state_group_loss': .25, 'mean_state_group_losses': {'synthetic': .25},
            } for role in ROLES}
            first = {role: {'points': [point | {'eps': [
                {'epsilon': epsilon, 'model_fd': -.5} for epsilon in EPSILONS
            ]} for point in _points(role)]} for role in FIRST_ROLES}
            models.append({'seed': seed, 'arm': arm, 'diagnostics': diagnostics, 'first_step': first})
    costs = {role: [point | {'gradient': -.5, 'resolved': True} for point in _points(role)]
             for role in COST_ROLES}
    firsts = {role: [point | {'resolved': True, 'eps': [
        {'epsilon': epsilon, 'central64': -.5} for epsilon in EPSILONS
    ]} for point in _points(role)] for role in FIRST_ROLES}
    return models, costs, firsts


def _model(models, arm, seed=MODEL_SEEDS[0]):
    return next(row for row in models if row['arm'] == arm and row['seed'] == seed)


def _wrong(models, arm, kind, *, seed=MODEL_SEEDS[0], role='fresh', index=0):
    model = _model(models, arm, seed)
    if kind == 'cost':
        model['diagnostics'][role]['points'][index]['gradient'] = .5
    else:
        model['first_step'][role]['points'][index]['eps'][0]['model_fd'] = .5


def test_all_pass_is_absolute_repair_without_claiming_paired_benefit():
    result = comparisons(*_evidence())
    assert result['active_repair_validated'] and result['active_status'] == 'PASSED'
    assert result['paired_status'] == 'COMPLETE' and not result['paired_improved']
    assert result['selected_seed'] is result['selected_control_parameter'] is None
    assert all(row['cost_valid_matches'] == 36 and row['first_step_valid_matches'] == 18
               for row in result['models'])


@pytest.mark.parametrize('field', ('autograd_agrees', 'direction_identifiable'))
@pytest.mark.parametrize('arm', ARMS)
def test_numerical_qualification_is_separate_from_active_absolute_gate(field, arm):
    models, costs, firsts = _evidence()
    # Development has no physical reference, but its numeric admission still counts.
    point = _model(models, arm)['diagnostics']['development']['points'][0]
    point[field] = False
    if field == 'direction_identifiable':
        point['gradient'] = -1e-6
    result = comparisons(models, costs, firsts)
    assert result['active_repair_validated'] is (arm == ARMS[0])
    assert result['active_status'] == ('PASSED' if arm == ARMS[0] else 'FAILED')
    assert result['paired_status'] == 'NUMERIC_UNRESOLVED' and not result['paired_improved']
    invalid = _model(result['models'], arm)
    assert invalid['all_selfchecks'] is (field != 'autograd_agrees')
    assert invalid['all_model_directions_identifiable'] is (field != 'direction_identifiable')


@pytest.mark.parametrize('role', FIRST_ROLES)
@pytest.mark.parametrize('scale', range(4))
def test_every_first_step_scale_must_match_even_when_all_cost_directions_match(role, scale):
    models, costs, firsts = _evidence()
    _model(models, ARMS[1])['first_step'][role]['points'][4]['eps'][scale]['model_fd'] = .5
    result = comparisons(models, costs, firsts)
    active = _model(result['models'], ARMS[1])
    assert active['full_direction_gate'] and not active['first_step_gate']
    assert active['cost_valid_matches'] == 36 and active['first_step_valid_matches'] == 17
    assert not result['active_repair_validated']


def test_tiny_same_sign_first_step_response_is_not_an_admitted_match():
    models, costs, firsts = _evidence()
    _model(models, ARMS[1])['first_step']['fresh']['points'][0]['eps'][0]['model_fd'] = -1e-6
    result = comparisons(models, costs, firsts)
    active = _model(result['models'], ARMS[1])
    point = active['first_step']['fresh']['points'][0]['eps'][0]
    assert point['same_sign'] and not point['model_magnitude_admitted'] and not point['valid_match']
    assert active['all_selfchecks'] and not result['active_repair_validated']


@pytest.mark.parametrize(('kind', 'role'), [*(('cost', r) for r in COST_ROLES),
                                          *(('first_step', r) for r in FIRST_ROLES)])
def test_unresolved_reference_is_never_counted_as_a_model_sign_error(kind, role):
    models, costs, firsts = _evidence()
    references = costs if kind == 'cost' else firsts
    references[role][4]['resolved'] = False
    for arm in ARMS:
        _wrong(models, arm, kind, role=role, index=4)
    result = comparisons(models, costs, firsts)
    assert result['active_status'] == result['paired_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['active_repair_validated'] and not result['paired_improved']
    detail = _model(result['models'], ARMS[1])[kind][role]
    assert detail['resolved_count'] == detail['valid_matches'] == 8
    point = detail['points'][4]
    if kind == 'cost':
        assert point['same_sign'] is point['valid_match'] is point['normalized_error'] is None
        assert detail['resolved_sign_mismatches'] == 0
    else:
        assert point['passed'] is None and detail['resolved_mismatches'] == 0
        assert all(eps['same_sign'] is eps['valid_match'] is None for eps in point['eps'])


def test_paired_gain_can_come_from_either_count_without_requiring_absolute_repair():
    models, costs, firsts = _evidence()
    _wrong(models, ARMS[0], 'cost')
    _wrong(models, ARMS[0], 'first_step', seed=MODEL_SEEDS[1])
    # A shared historical error means the intervention still has not repaired all points.
    for arm in ARMS:
        _wrong(models, arm, 'cost', seed=MODEL_SEEDS[2], role='old_probe')
    result = comparisons(models, costs, firsts)
    assert result['paired_improved'] and result['paired_status'] == 'COMPLETE'
    assert not result['active_repair_validated']
    assert [(p['cost_delta'], p['first_step_delta']) for p in result['paired']] == [(1, 0), (0, 1), (0, 0)]


def test_cost_gain_cannot_pay_for_first_step_regression_in_the_same_seed():
    models, costs, firsts = _evidence()
    _wrong(models, ARMS[0], 'cost')
    _wrong(models, ARMS[1], 'first_step')
    result = comparisons(models, costs, firsts)
    assert result['paired_any_strict_improvement'] and not result['paired_non_decreasing']
    assert not result['paired_improved']


@pytest.mark.parametrize('kind', ('cost', 'first_step'))
def test_net_gain_cannot_hide_a_regression_in_another_paired_seed(kind):
    models, costs, firsts = _evidence()
    _wrong(models, ARMS[0], kind, index=0)
    _wrong(models, ARMS[0], kind, index=1)
    _wrong(models, ARMS[1], kind, seed=MODEL_SEEDS[1])
    result = comparisons(models, costs, firsts)
    field = 'cost_delta' if kind == 'cost' else 'first_step_delta'
    assert [pair[field] for pair in result['paired']] == [2, -1, 0]
    assert not result['paired_improved']


def _fits():
    shared = {key: 'same-' + key for key in PAIR_FIELDS if key != 'initial_model_state_sha256'}
    expected = {'common': shared.copy(), 'by_arm': {
        arm: {'labels_sha256': 'labels-' + arm, 'second_labels_sha256': 'second-' + source,
              'second_label_source': source} for arm, source in zip(ARMS, ('60', '62'), strict=True)}}
    rows = []
    for seed in MODEL_SEEDS:
        for arm in ARMS:
            fit = shared | expected['by_arm'][arm] | {
                'status': 'COMPLETE', 'eligible_for_evaluation': True, 'updates': 1000,
                'updates_attempted': 1000, 'seed': seed, 'arm': arm, 'horizon': 8,
                'training_transitions': 7200, 'training_windows': 6828,
                'training_episodes': list(range(12)), 'initial_model_state_sha256': f'init-{seed}',
                'fixed_lambda': FIXED_LAMBDA, 'effective_lambda': FIXED_LAMBDA}
            rows.append({'seed': seed, 'arm': arm, 'fit': fit})
    return rows, expected


def test_pairing_allows_exactly_the_registered_second_source_difference():
    rows, expected = _fits()
    assert check_pairing(rows, expected)['status'] == 'COMPLETE'
    assert not check_pairing(rows, expected)['recalibrated']


@pytest.mark.parametrize('identity', PAIR_FIELDS)
def test_any_shared_pair_identity_mismatch_rejects_comparison(identity):
    rows, expected = _fits()
    _model(rows, ARMS[1])['fit'][identity] = 'changed'
    with pytest.raises(ValueError):
        check_pairing(rows, expected)


@pytest.mark.parametrize('field', ('labels_sha256', 'second_labels_sha256', 'second_label_source'))
def test_registered_arm_source_is_required(field):
    rows, expected = _fits()
    _model(rows, ARMS[1])['fit'][field] = _model(rows, ARMS[0])['fit'][field]
    with pytest.raises(ValueError, match='registered label source'):
        check_pairing(rows, expected)


@pytest.mark.parametrize('field', ('fixed_lambda', 'effective_lambda'))
def test_recalibrating_or_disabling_auxiliary_loss_is_rejected(field):
    rows, expected = _fits()
    _model(rows, ARMS[0])['fit'][field] = 0.0
    with pytest.raises(ValueError, match='fixed59 lambda'):
        check_pairing(rows, expected)


def test_partial_fit_never_qualifies_even_with_matching_hashes():
    rows, expected = _fits()
    _model(rows, ARMS[1])['fit']['updates'] = 999
    with pytest.raises(ValueError, match='incomplete'):
        check_pairing(rows, expected)


@pytest.mark.parametrize('field', ('initial_common58_predictions_sha256', 'initial_common58_yaw_jvp_sha256',
                                    'ridge_and_normalization_sha256'))
def test_initial_function_must_also_match_across_paired_seeds(field):
    rows, expected = _fits()
    del expected['common'][field]
    for arm in ARMS:
        _model(rows, arm)['fit'][field] = 'changed-but-equal-within-this-pair'
    with pytest.raises(ValueError, match='across seeds'):
        check_pairing(rows, expected)


