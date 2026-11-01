"""75 pairing and plan contracts; synthetic metadata only, no robot/models."""
from __future__ import annotations

import copy

import pytest

from tools75 import run75 as run


def fits():
    shared = {key: 'same-' + key for key in run.PAIR_FIELDS if key != 'initial_model_state_sha256'}
    expected = {'common': shared.copy(), 'by_arm': {
        arm: {'fixed_lambda': weight, 'effective_lambda': weight, 'lambda_sha256': shared['lambda_sha256'],
              'fixed_beta': run.FIXED_BETA, 'body_coefficient': run.FIXED_BETA,
              'beta_sha256': shared['beta_sha256'],
              'gamma_sha256': 'gamma-' + arm, 'fixed_gamma': run.WORLD_COEFFICIENT[arm],
              'world_vz_coefficient': run.WORLD_COEFFICIENT[arm]} for arm, weight in run.YAW_COEFFICIENT.items()},
        'control_final_sha256': {str(seed): f'old69-active-{seed}' for seed in run.MODEL_SEEDS}}
    rows = []
    for seed in run.MODEL_SEEDS:
        for arm in run.ARMS:
            fit = shared | expected['by_arm'][arm] | {'status': 'COMPLETE', 'eligible_for_evaluation': True,
                'updates': 1000, 'updates_attempted': 1000, 'seed': seed, 'arm': arm, 'horizon': 8,
                'training_transitions': 7200, 'training_windows': 6828, 'training_episodes': list(range(12)),
                'initial_model_state_sha256': f'init-{seed}', 'reference_lambda': run.FIXED_LAMBDA,
                'calibration_performed': False, 'second_label_source': '62',
                'final_model_state_sha256': f'old69-active-{seed}' if arm == 'control' else f'new-{seed}'}
            rows.append({'seed': seed, 'arm': arm, 'fit': fit})
    return rows, expected


def test_exact69_active_control_with_shared_lambda_beta_and_added_gamma():
    rows, expected = fits()
    result = run.check_pairing(rows, expected)
    assert result['status'] == 'COMPLETE'
    assert all(p['control_reproduces69_active'] for p in result['pairs'])
    assert 'lambda_sha256' in run.PAIR_FIELDS and 'beta_sha256' in run.PAIR_FIELDS
    assert 'gamma_sha256' not in run.PAIR_FIELDS
    assert result['yaw_coefficients'] == dict.fromkeys(run.ARMS, run.FIXED_LAMBDA)
    assert result['world_vz_coefficients'] == {'control': 0., 'active': run.FIXED_LAMBDA}
    assert result['body_coefficients'] == dict.fromkeys(run.ARMS, run.FIXED_BETA)


@pytest.mark.parametrize('seed', run.MODEL_SEEDS)
def test_each_control_must_reproduce_69_not64(seed):
    rows, expected = fits()
    next(row for row in rows if row['seed'] == seed and row['arm'] == 'control')['fit']['final_model_state_sha256'] = 'old64'
    with pytest.raises(ValueError, match='exactly reproduce69'):
        run.check_pairing(rows, expected)


@pytest.mark.parametrize('identity', run.PAIR_FIELDS)
def test_single_identity_drift_rejected(identity):
    rows, expected = fits()
    rows[-1]['fit'][identity] = 'drift'
    with pytest.raises(ValueError):
        run.check_pairing(rows, expected)


@pytest.mark.parametrize('field,value', [('updates', 999), ('fixed_beta', 0.), ('reference_lambda', 0.),
                                       ('effective_lambda', 2 * run.FIXED_LAMBDA), ('fixed_gamma', 0.), ('second_label_source', '60'),
                                       ('calibration_performed', True), ('eligible_for_evaluation', False)])
def test_partial_or_extra_variable_rejected(field, value):
    rows, expected = fits()
    rows[-1]['fit'][field] = value
    with pytest.raises(ValueError):
        run.check_pairing(rows, expected)


def plan():
    return {'schema': 'd1-world-vz75-v1', 'model_seeds': list(run.MODEL_SEEDS),
            'probe_seeds': list(run.PROBE_SEEDS), 'arms': list(run.ARMS), 'updates_per_model': 1000,
            'expected_policy_steps': 4680, 'hard_policy_cap': 5000, 'cpu_seconds': 600, 'wall_seconds': 900,
            'fixed_lambda': run.FIXED_LAMBDA, 'fixed_beta': run.FIXED_BETA,
            'yaw_coefficients': run.YAW_COEFFICIENT.copy(), 'body_coefficients': run.BODY_COEFFICIENT.copy(),
            'world_vz_coefficients': run.WORLD_COEFFICIENT.copy(),
            'label_rows': 96, 'decision_spec': copy.deepcopy(run.DECISION_SPEC), 'source_sha256': {}}


@pytest.mark.parametrize('field,value', [('cpu_seconds', 601), ('wall_seconds', 901), ('updates_per_model', 1001),
                                       ('expected_policy_steps', 4681), ('fixed_beta', 0.), ('label_rows', 48)])
def test_work_or_coefficient_budget_cannot_drift(field, value):
    changed = plan() | {field: value}
    with pytest.raises(ValueError, match='budget/coefficient'):
        run.verify75(changed)


def test_cannot_drop_old72_or_development_numeric_scope():
    for key in ('numeric_roles', 'cost_h32_roles', 'first_and_head_roles'):
        changed = plan()
        changed['decision_spec'][key].remove('old72')
        with pytest.raises(ValueError, match='decision rules'):
            run.verify75(changed)
    assert run.DECISION_SPEC['numeric_roles'][0] == 'development'
    assert (run.DECISION_SPEC['cost_h32_points'], run.DECISION_SPEC['yaw_first_points'],
            run.DECISION_SPEC['body_state_channels'], run.DECISION_SPEC['cost_h1_points']) == (63, 45, 90, 45)


def test_all_old69_grids_and_control_weights_bound():
    paths = run.required_inputs75()
    assert sum(str(run.OLD69 / 'grid_') in p for p in paths) == 90
    assert all(str(run.OLD69 / f'active_seed{seed}/final_state_dict.pt') in paths for seed in run.MODEL_SEEDS)
    with pytest.raises(ValueError, match='required inputs unbound'):
        run.verify75(plan())


def test_final_audit_failure_withholds_all_nested_successes_but_retains_raw_evidence():
    provisional = {'status': 'PROVISIONAL', 'candidate_accepted': True, 'candidate_status': 'PASSED',
                   'generalization_target_status': 'PASSED', 'complete_repair_status': 'PASSED',
                   'paired': [{'target_passed': True}], 'models': [{'seed': 64001}]}
    report = {'comparison': copy.deepcopy(provisional)}
    run.withhold_final_eligibility(report)
    assert report['comparison_before_final_eligibility'] == provisional
    assert 'paired' not in report['comparison']
    assert not report['comparison']['candidate_accepted']
    assert all(report['comparison'][key] == 'INCOMPLETE' for key in
               ('status', 'candidate_status', 'generalization_target_status', 'complete_repair_status'))


def test_all_old72_failed_case_inputs_bound_and_fresh_seeds_disjoint():
    paths = run.required_inputs75()
    assert sum(str(run.OLD72 / 'grid_') in p for p in paths) == 90
    assert str(run.OLD72 / 'report.json') in paths
    assert str(run.LABEL75 / 'world_vz_labels96.npz') in paths
    assert run.PROBE_SEEDS == (75301, 75302, 75303)
    assert run.DECISION_SPEC['world_vz_state_index'] == 25
    assert run.DECISION_SPEC['fresh_world_vz_error_scalar_count'] == 36
    assert run.DECISION_SPEC['world_vz_points'] == 45
