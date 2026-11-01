"""Pure61 witness decisions, without models, numeric arrays or simulation."""
from __future__ import annotations

import pytest

from tools58.collect_response58 import EPSILONS, TICKS, TRAIN_SEEDS
from tools61.warmstart61 import warmstart_hypothesis


def _points():
    deltas = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
    return [
        {'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'arms': {
            arm: {'status': 'COMPLETE', 'label_admitted': True,
                  'branches': [{'status': 'COMPLETE', 'steps': 1, 'delta': delta} for delta in deltas],
                  'yaw28': [{'epsilon': epsilon, 'central': .02 if arm == 'preserved' else .01,
                            'forward': .02 if arm == 'preserved' else .01,
                            'backward': .02 if arm == 'preserved' else .01} for epsilon in EPSILONS]}
            for arm in ('preserved', 'zeroed')
        }}
        for seed in TRAIN_SEEDS for tick in TICKS
    ]


def _witness(point):
    for response in point['arms']['zeroed']['yaw28']:
        response.update(central=-.01, forward=-.01, backward=-.01)


def test_all_stable_without_reversal_means_only_no_witness_in_this_intervention():
    result = warmstart_hypothesis(_points())
    assert result['status'] == 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
    assert not result['witness_found'] and result['witness_count'] == 0
    assert len(result['all_point_diagnostics']) == 12
    assert all(row['reference_resolved'] and not row['witness']
               for row in result['all_point_diagnostics'])


@pytest.mark.parametrize('witness_index', range(12))
def test_any_stable_registered_witness_survives_other_unresolved_references(witness_index):
    points = _points()
    for index, point in enumerate(points):
        if index == witness_index:
            _witness(point)
        else:
            point['arms']['zeroed']['label_admitted'] = False
    result = warmstart_hypothesis(points)
    assert result['status'] == 'WITNESS_FOUND' and result['witness_found']
    assert result['witness_count'] == 1
    rows = result['all_point_diagnostics']
    assert rows[witness_index]['witness']
    assert sum(row['status'] == 'REFERENCE_UNRESOLVED' for row in rows) == 11


def test_no_witness_with_an_unresolved_reference_remains_unresolved():
    points = _points()
    _witness(points[0])
    points[0]['arms']['preserved']['label_admitted'] = False
    result = warmstart_hypothesis(points)
    assert result['status'] == 'REFERENCE_UNRESOLVED'
    assert not result['witness_found'] and result['witness_count'] == 0
    assert result['all_point_diagnostics'][0]['opposite_sign_all_eps'] is None


@pytest.mark.parametrize('scale', range(4))
def test_reversed_mean_is_insufficient_when_one_scale_has_the_original_sign(scale):
    points = _points()
    _witness(points[0])
    # The four-scale mean is still negative. Every corresponding scale must reverse.
    points[0]['arms']['zeroed']['yaw28'][scale].update(central=.001, forward=.001, backward=.001)
    result = warmstart_hypothesis(points)
    assert result['status'] == 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
    assert not result['witness_found']
    assert result['all_point_diagnostics'][0]['opposite_sign_all_eps'] is False


@pytest.mark.parametrize('partial', ('point', 'preserved', 'zeroed', 'preserved_branch', 'zeroed_branch'))
def test_partial_work_invalidates_formal_decision_even_if_another_point_is_a_witness(partial):
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
        warmstart_hypothesis(points)


@pytest.mark.parametrize('missing', ('point', 'branch', 'scale', 'duplicate_point'))
def test_missing_or_duplicated_registered_evidence_cannot_establish_a_witness(missing):
    points = _points()
    _witness(points[0])
    if missing == 'point':
        points.pop()
    elif missing == 'branch':
        points[-1]['arms']['preserved']['branches'].pop()
    elif missing == 'scale':
        points[-1]['arms']['zeroed']['yaw28'].pop()
    else:
        points[-1] = points[0]
    with pytest.raises(ValueError):
        warmstart_hypothesis(points)
