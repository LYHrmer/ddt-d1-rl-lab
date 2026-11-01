"""Pure63 pairing and witness contracts; no model or simulator is invoked."""
from __future__ import annotations

import pytest

from tools58.collect_response58 import EPSILONS, TICKS, TRAIN_SEEDS
from tools63.phase_swap63 import phase_hypothesis, phase_partner

EXPECTED_PARTNERS = {53103: 53106, 53106: 53103, 53109: 53112, 53112: 53109}


def _points():
    deltas = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
    return [{
        'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'arms': {
            arm: {'status': 'COMPLETE', 'label_admitted': True,
                  'branches': [{'status': 'COMPLETE', 'steps': 1, 'delta': delta} for delta in deltas],
                  'yaw28': [{'epsilon': epsilon, 'central': .02 if arm == 'original' else .01,
                            'forward': .02 if arm == 'original' else .01,
                            'backward': .02 if arm == 'original' else .01} for epsilon in EPSILONS]}
            for arm in ('original', 'transplanted')
        },
    } for seed in TRAIN_SEEDS for tick in TICKS]


def _witness(point):
    for response in point['arms']['transplanted']['yaw28']:
        response.update(central=-.01, forward=-.01, backward=-.01)


@pytest.mark.parametrize('seed', TRAIN_SEEDS)
def test_partner_is_the_registered_reciprocal_seed_at_the_same_tick(seed):
    for tick in TICKS:
        partner = phase_partner(seed, tick)
        assert partner == (EXPECTED_PARTNERS[seed], tick)
        assert phase_partner(*partner) == (seed, tick)


@pytest.mark.parametrize(('seed', 'tick'), ((53101, 260), (53103, 259), (53103, 411)))
def test_unregistered_seed_or_tick_cannot_supply_a_donor_phase(seed, tick):
    with pytest.raises(ValueError):
        phase_partner(seed, tick)


def test_all_stable_without_reversal_means_no_witness_not_a_validated_mask():
    result = phase_hypothesis(_points())
    assert result['status'] == 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
    assert not result['witness_found'] and result['witness_count'] == 0
    assert len(result['all_point_diagnostics']) == 12
    assert all(row['reference_resolved'] and not row['witness']
               for row in result['all_point_diagnostics'])


@pytest.mark.parametrize('witness_index', range(12))
def test_any_stable_witness_survives_other_complete_but_unresolved_points(witness_index):
    points = _points()
    for index, point in enumerate(points):
        if index == witness_index:
            _witness(point)
        else:
            point['arms']['transplanted']['label_admitted'] = False
    result = phase_hypothesis(points)
    assert result['status'] == 'WITNESS_FOUND' and result['witness_found']
    assert result['witness_count'] == 1
    rows = result['all_point_diagnostics']
    assert rows[witness_index]['witness']
    assert sum(row['status'] == 'REFERENCE_UNRESOLVED' for row in rows) == 11


def test_an_apparent_reversal_at_an_unresolved_point_does_not_establish_a_witness():
    points = _points()
    _witness(points[0])
    points[0]['arms']['original']['label_admitted'] = False
    result = phase_hypothesis(points)
    assert result['status'] == 'REFERENCE_UNRESOLVED' and not result['witness_found']
    assert result['all_point_diagnostics'][0]['opposite_sign_all_eps'] is None


@pytest.mark.parametrize('scale', range(4))
def test_reversed_mean_cannot_replace_reversal_at_every_registered_scale(scale):
    points = _points()
    _witness(points[0])
    points[0]['arms']['transplanted']['yaw28'][scale].update(central=.001, forward=.001, backward=.001)
    result = phase_hypothesis(points)
    assert result['status'] == 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
    assert not result['witness_found']
    assert result['all_point_diagnostics'][0]['opposite_sign_all_eps'] is False


@pytest.mark.parametrize('partial', ('point', 'original', 'transplanted', 'original_branch', 'transplanted_branch'))
def test_partial_work_prevents_a_formal_witness_even_if_another_point_reverses(partial):
    points = _points()
    _witness(points[0])
    point = points[-1]
    if partial == 'point':
        record = point
    elif partial.endswith('_branch'):
        record = point['arms'][partial.removesuffix('_branch')]['branches'][-1]
    else:
        record = point['arms'][partial]
    record['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError):
        phase_hypothesis(points)


@pytest.mark.parametrize('missing', ('point', 'branch', 'scale', 'duplicate_point'))
def test_incomplete_or_duplicate_registered_grid_cannot_qualify(missing):
    points = _points()
    _witness(points[0])
    if missing == 'point':
        points.pop()
    elif missing == 'branch':
        points[-1]['arms']['original']['branches'].pop()
    elif missing == 'scale':
        points[-1]['arms']['transplanted']['yaw28'].pop()
    else:
        points[-1] = points[0]
    with pytest.raises(ValueError):
        phase_hypothesis(points)
