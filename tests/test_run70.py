"""Archive accounting and complete-grid safeguards, without robot/model calls."""
from __future__ import annotations

import copy

import pytest

from tools70 import run70

np = pytest.importorskip('numpy')


def _coordinates(base, direction, epsilon):
    center = np.zeros(22, dtype=np.float64)
    slope = np.zeros_like(center)
    center[0], slope[0] = base, direction
    return center, center + epsilon * slope, center - epsilon * slope


def _fixture():
    from tools70 import decompose70 as algebra
    mz, mp, mm = _coordinates(2., 3., .001)
    tz, tp, tm = _coordinates(1., 2., .00025)
    model = {}
    for name, coordinates in (('theta0', mz), ('plus', mp), ('minus', mm)):
        parts = algebra.component_costs(coordinates)
        model[name] = {'coordinates': coordinates, 'components': parts, 'cost': float(parts.sum())}
    values = np.stack((tz, tz, tp, tm))
    parts = algebra.component_costs(values)
    physical = {'coordinates': values, 'components': parts, 'costs': parts.sum(axis=-1)}
    return model, physical


def test_original_total_AD_gap_is_kept_separate_and_all_coordinates_are_saved():
    model, physical = _fixture()
    model_fd = (model['plus']['cost'] - model['minus']['cost']) / .002
    ad = model_fd + 2e-8
    row, arrays = run70.interval(model, physical, ad, 0, .00025)
    assert row['status'] == 'COMPLETE'
    assert row['epsilon_model'] == .001 and row['epsilon_true'] == .00025
    assert row['model_AD_minus_model_FD'] == pytest.approx(2e-8, abs=1e-14)
    assert row['midpoint_contribution'][0] == pytest.approx(5., abs=1e-10)
    assert row['response_contribution'][0] == pytest.approx(3., abs=1e-10)
    assert row['model_AD_minus_true_FD'] == pytest.approx(8. + 2e-8, abs=1e-10)
    assert all(check['passed'] for check in row['checks'].values())
    assert arrays['scalar22_model_response'].shape == (22,)


def test_failed_identity_retains_signed_values_checks_and_arrays(monkeypatch):
    from tools70 import decompose70 as algebra
    original = algebra.decompose

    def corrupted(*args, **kwargs):
        result = original(*args, **kwargs)
        result['response_contribution'][0] += .1
        return result

    monkeypatch.setattr(algebra, 'decompose', corrupted)
    model, physical = _fixture()
    receipt, arrays = {}, {}
    with pytest.raises(ValueError, match='component_identity'):
        run70.interval(model, physical, 12., 0, .00025, receipt=receipt, arrays=arrays)
    assert receipt['status'] == 'INCOMPLETE'
    assert receipt['checks']['component_identity']['passed'] is False
    assert receipt['response_contribution'][0] == pytest.approx(3.1, abs=1e-10)
    assert 'scalar22_model_response' in arrays


def test_absolute_mass_keeps_cancelling_coordinates_inside_smoothness():
    from tools70 import decompose70 as algebra
    mz = np.zeros(22, dtype=np.float64)
    tz = mz.copy()
    mz[6:8], tz[6:8] = (2., 0.), (1., 1.)
    slope = np.zeros_like(mz)
    slope[6:8] = 3.
    model = {}
    for name, coordinates in (('theta0', mz), ('plus', mz + .001*slope), ('minus', mz - .001*slope)):
        parts = algebra.component_costs(coordinates)
        model[name] = {'coordinates': coordinates, 'components': parts, 'cost': float(parts.sum())}
    values = np.stack((tz, tz, tz + .00025*slope, tz - .00025*slope))
    parts = algebra.component_costs(values)
    physical = {'coordinates': values, 'components': parts, 'costs': parts.sum(axis=-1)}
    ad = (model['plus']['cost'] - model['minus']['cost'])/.002
    row, _ = run70.interval(model, physical, ad, 0, .00025)
    assert abs(row['midpoint_contribution'][5]) < 1e-10
    assert row['coordinate_absolute_mass']['midpoint'][5] == pytest.approx(.075, abs=1e-10)


def _complete_points():
    rows = []
    for seed in run70.MODEL_SEEDS:
        for arm in run70.ARMS:
            for role in run70.ROLES:
                for reset in run70.RESET_SEEDS[role]:
                    for tick in run70.TICKS:
                        error = 2. if arm == 'control' else 1.
                        if seed == 64003 and arm == 'active':
                            error = 2.
                        eps = [{'status': 'COMPLETE', 'epsilon_true': epsilon, 'model_saved_component_fd': [error] * 6,
                                'true_frozen_component_fd': [0.] * 6,
                                'midpoint_contribution': [10.] * 6,
                                'zero_bias_contribution': [9.] * 6,
                                'midpoint_offset_contribution': [1.] * 6,
                                'response_contribution': [-10. + error] * 6,
                                'coordinate_absolute_mass': {'midpoint': [10.] * 6,
                                    'zero_bias': [9.] * 6, 'midpoint_offset': [1.] * 6,
                                    'response': [abs(-10. + error)] * 6}}
                               for epsilon in run70.EPSILONS]
                        rows.append({'status': 'COMPLETE', 'model_seed': seed, 'arm': arm, 'role': role,
                                     'seed': reset, 'tick': tick, 'eps': eps})
    return rows


def test_complete_grid_preserves_ties_and_does_not_turn_absolute_mass_into_gain():
    result = run70.summaries(_complete_points(), ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness'))
    assert len(result['metric_rows']) == 3 * 2 * 4 * 6
    assert len(result['paired_component_FD_MAE']) == 4 * 6
    assert all(row['classification'] == 'MIXED_OR_TIED' for row in result['paired_component_FD_MAE'])
    first = result['paired_component_FD_MAE'][0]
    assert [row['active_minus_control'] for row in first['pairs']] == [-1., -1., 0.]
    assert result['automatic_new_candidate'] is None
    assert result['absolute_mass_is_not_causal_or_additive_fraction']
    assert not result['source69_rejudged']


@pytest.mark.parametrize('failure', ('missing_point', 'partial_point', 'missing_epsilon', 'partial_epsilon',
                                    'wrong_role', 'duplicate_point', 'wrong_epsilon'))
def test_incomplete_or_wrong_grid_cannot_yield_summary(failure):
    rows = _complete_points()
    if failure == 'missing_point':
        rows.pop()
    elif failure == 'partial_point':
        rows[0]['status'] = 'INCOMPLETE'
    elif failure == 'missing_epsilon':
        rows[0]['eps'].pop()
    elif failure == 'partial_epsilon':
        rows[0]['eps'][0]['status'] = 'INCOMPLETE'
    elif failure == 'wrong_role':
        rows[0]['role'] = 'cherry_picked'
    elif failure == 'duplicate_point':
        rows[0] = copy.deepcopy(rows[1])
    else:
        rows[0]['eps'][0]['epsilon_true'] = .1
    with pytest.raises(ValueError):
        run70.summaries(rows, ('vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness'))


@pytest.mark.parametrize('mutation', ('decision', 'cpu', 'wall', 'missing_binding', 'changed_binding', 'inherited_binding'))
def test_preregistration_requires_exact_rules_and_inputs(monkeypatch, mutation):
    monkeypatch.setattr(run70, 'required_inputs', lambda: {'fixture.npz'})
    monkeypatch.setattr(run70, 'inherited_bindings', lambda: {'frozen.py': 'a'})
    monkeypatch.setattr(run70, 'sha', lambda _path: 'a')
    plan = {'schema': 'd1-cost-decomposition70-v1', 'decision_spec': copy.deepcopy(run70.DECISION_SPEC),
            'cpu_seconds': 30, 'wall_seconds': 60, 'source_sha256': {'fixture.npz': 'a', 'frozen.py': 'a'}}
    run70.verify70(plan)
    if mutation == 'decision':
        plan['decision_spec']['model_points'] = 161
    elif mutation == 'cpu':
        plan['cpu_seconds'] = 31
    elif mutation == 'wall':
        plan['wall_seconds'] = 61
    elif mutation == 'missing_binding':
        plan['source_sha256'] = {}
    elif mutation == 'changed_binding':
        plan['source_sha256']['fixture.npz'] = 'b'
    else:
        plan['source_sha256']['frozen.py'] = 'b'
    with pytest.raises(ValueError):
        run70.verify70(plan)
