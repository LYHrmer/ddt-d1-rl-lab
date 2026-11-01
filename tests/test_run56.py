"""Synthetic evidence checks for the paired56 decision, without fitting or physics."""
from __future__ import annotations

import pytest

from tools56.run56 import (
    MODEL_SEEDS,
    PAIR_FIELDS,
    ROLES,
    check_pairing,
    comparisons,
    local_references,
)

EPSILONS = (.00003125, .0000625, .000125, .00025)
TICKS = (110, 260, 410)
ROLE_SEEDS = {
    'development': (53201, 53202, 53203),
    'old_probe': (53301, 53302, 53303),
    'old55_local': (55301, 55302, 55303),
    'fresh': (56301, 56302, 56303),
}


def _points(role):
    return [
        {'seed': seed, 'tick': tick, 'gradient': -.5,
         'model_central_difference': -.5, 'autograd_agrees': True,
         'direction_identifiable': True}
        for seed in ROLE_SEEDS[role] for tick in TICKS
    ]


def _models():
    models = []
    for seed in MODEL_SEEDS:
        for horizon in (8, 32):
            diagnostics = {
                role: {'points': _points(role), 'all_selfchecks': True,
                       'all_model_directions_identifiable': True,
                       'mean_state_group_loss': .25,
                       'mean_state_group_losses': {'synthetic_group': .25}}
                for role in ROLES
            }
            models.append({'seed': seed, 'horizon': horizon, 'diagnostics': diagnostics})
    return models


def _references():
    return {
        role: [{'seed': point['seed'], 'tick': point['tick'],
                'gradient': -.5, 'resolved': True} for point in _points(role)]
        for role in ('old_probe', 'old55_local', 'fresh')
    }


def _change_prediction(models, horizon, role, index=0, **changes):
    model = next(row for row in models if row['seed'] == MODEL_SEEDS[0]
                 and row['horizon'] == horizon)
    diagnostics = model['diagnostics'][role]
    diagnostics['points'][index].update(changes)
    diagnostics['all_selfchecks'] = all(p['autograd_agrees'] for p in diagnostics['points'])
    diagnostics['all_model_directions_identifiable'] = all(
        p['direction_identifiable'] for p in diagnostics['points'])


def _fresh_physical_rows():
    """All four finite-difference scales agree until a test changes one of them."""
    return [
        {'seed': point['seed'], 'tick': point['tick'], 'status': 'COMPLETE',
         'all_four_wheels_resolved': True,
         'branches': [{'status': 'COMPLETE', 'steps': 32} for _ in range(10)],
         'gradients': [{'epsilon': epsilon, 'central64': -.5,
                        'forward64': -.5, 'backward64': -.5} for epsilon in EPSILONS]}
        for point in _points('fresh')
    ]


@pytest.mark.parametrize('point_index', range(9))
def test_any_unstable_fresh_reference_is_unresolved_not_a_model_sign_error(point_index):
    physical = _fresh_physical_rows()
    physical[point_index]['gradients'][0].update(
        central64=.5, forward64=.5, backward64=.5)
    references = _references()
    references['fresh'] = local_references(physical)
    models = _models()
    # Make the prediction oppose the still-negative four-scale mean. The
    # unresolved reference must nevertheless produce no sign-error judgment.
    for model in models:
        model['diagnostics']['fresh']['points'][point_index].update(
            gradient=.5, model_central_difference=.5)

    result = comparisons(models, references)

    assert result['primary_status'] == 'REFERENCE_UNRESOLVED'
    assert not result['fresh_reference_admitted']
    assert result['fresh_reference_resolved_count'] == 8
    assert not result['primary_passed']
    assert not result['direction_improved']
    for model in result['models']:
        fresh = model['fresh']
        point = fresh['points'][point_index]
        assert point['same_sign'] is None
        assert point['direction_match'] is None
        assert point['normalized_gradient_error'] is None
        assert fresh['resolved_sign_mismatches'] == 0
        assert fresh['direction_matches'] == 8
        assert fresh['mean_normalized_gradient_error'] is None


def test_both_arms_all_nine_is_a_direction_pass_but_not_horizon_improvement():
    result = comparisons(_models(), _references())

    assert result['primary_status'] == 'PASSED'
    assert result['h8_direction_gate'] and result['h32_direction_gate']
    assert result['primary_passed']
    assert not result['direction_improved']
    assert result['selected_horizon'] is None
    assert all(row['fresh']['direction_matches'] == 9 for row in result['models'])


def test_paired_improvement_does_not_require_h8_to_pass_its_direction_gate():
    models = _models()
    _change_prediction(models, 8, 'fresh', gradient=.5, model_central_difference=.5)

    result = comparisons(models, _references())

    assert not result['h8_direction_gate']
    assert result['h32_direction_gate']
    assert result['primary_passed']
    assert result['direction_improved']
    assert result['selected_horizon'] is None


@pytest.mark.parametrize('horizon', (8, 32))
def test_failed_model_selfcheck_cannot_be_reported_as_horizon_improvement(horizon):
    models = _models()
    _change_prediction(models, 8, 'fresh', gradient=.5, model_central_difference=.5)
    _change_prediction(models, horizon, 'fresh', autograd_agrees=False)

    result = comparisons(models, _references())

    assert not result['direction_improved']
    failed = next(row for row in result['models'] if row['seed'] == MODEL_SEEDS[0]
                  and row['horizon'] == horizon)
    assert not failed['all_selfchecks']
    assert failed['fresh']['points'][0]['direction_match'] is False


@pytest.mark.parametrize('horizon', (8, 32))
def test_unidentifiable_small_gradient_is_not_a_numerical_failure_or_a_gain(horizon):
    models = _models()
    _change_prediction(models, 8, 'fresh', gradient=.5, model_central_difference=.5)
    small = 1e-6 if horizon == 8 else -1e-6
    _change_prediction(models, horizon, 'fresh', gradient=small,
                       model_central_difference=small, direction_identifiable=False)

    result = comparisons(models, _references())

    assert not result['direction_improved']
    failed = next(row for row in result['models'] if row['seed'] == MODEL_SEEDS[0]
                  and row['horizon'] == horizon)
    assert failed['all_selfchecks']
    assert not failed['all_model_directions_identifiable']
    assert failed['fresh']['points'][0]['direction_match'] is False


def test_old55_correct_h8_point_cannot_regress_despite_fresh_improvement():
    models = _models()
    _change_prediction(models, 8, 'fresh', gradient=.5, model_central_difference=.5)
    _change_prediction(models, 32, 'old55_local', gradient=.5, model_central_difference=.5)

    result = comparisons(models, _references())

    assert not result['h32_old55_no_regression']
    assert not result['primary_passed']
    assert result['primary_status'] == 'FAILED'
    assert not result['direction_improved']
    pair = next(row for row in result['paired'] if row['seed'] == MODEL_SEEDS[0])
    assert pair['old55_correct_h8_regressions'] == [{'seed': 55301, 'tick': 110}]


@pytest.mark.parametrize('role', ('old_probe', 'old55_local'))
def test_a_shared_old_error_is_not_excused_by_no_paired_regression(role):
    models = _models()
    for horizon in (8, 32):
        _change_prediction(models, horizon, role, gradient=.5, model_central_difference=.5)

    result = comparisons(models, _references())

    assert result['h32_old55_no_regression']
    assert not result['h8_direction_gate']
    assert not result['h32_direction_gate']
    assert not result['primary_passed']
    assert result['primary_status'] == 'FAILED'


def _paired_fits():
    shared = {name: 'shared-' + name for name in PAIR_FIELDS
              if name != 'initial_model_state_sha256'}
    models = []
    for seed in MODEL_SEEDS:
        for horizon in (8, 32):
            fitted = {**shared, 'initial_model_state_sha256': f'initial-{seed}',
                      'status': 'COMPLETE', 'eligible_for_evaluation': True,
                      'updates': 1000, 'updates_attempted': 1000,
                      'seed': seed, 'horizon': horizon, 'training_transitions': 7200,
                      'training_windows': 6828, 'training_episodes': list(range(12))}
            models.append({'seed': seed, 'horizon': horizon, 'fit': fitted})
    return models, shared


def test_pairing_preserves_each_seed_initialization_without_requiring_equal_seeds():
    models, admitted = _paired_fits()

    result = check_pairing(models, admitted)

    assert result['status'] == 'COMPLETE'
    assert [pair['seed'] for pair in result['pairs']] == list(MODEL_SEEDS)
    assert all(all(pair['equal_by_field'].values()) for pair in result['pairs'])


@pytest.mark.parametrize('identity', PAIR_FIELDS)
def test_any_pairing_identity_mismatch_rejects_evaluation(identity):
    models, admitted = _paired_fits()
    models[1]['fit'][identity] = 'different-' + identity

    with pytest.raises(ValueError):
        check_pairing(models, admitted)


@pytest.mark.parametrize(('field', 'value'), [
    ('status', 'INCOMPLETE'), ('eligible_for_evaluation', False),
    ('updates', 999), ('updates_attempted', 1001),
])
def test_partial_or_nonfinal_fit_cannot_enter_the_paired_comparison(field, value):
    models, admitted = _paired_fits()
    models[1]['fit'][field] = value

    with pytest.raises(ValueError, match='incomplete/nonpaired fit'):
        check_pairing(models, admitted)
