"""Synthetic62 admission and coverage decisions; no models or physics run."""
from __future__ import annotations

import pytest

from tools58.collect_response58 import EPSILONS, TICKS, TRAIN_SEEDS
from tools62.collect_clean62 import clean_history_hypothesis


def _evidence():
    cases = [{'seed': seed, 'status': 'COMPLETE', 'steps': 600} for seed in TRAIN_SEEDS]
    deltas = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
    points = [{
        'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'label_admitted': True,
        'branches': [{'status': 'COMPLETE', 'steps': 1, 'delta': delta} for delta in deltas],
        'yaw28': [{'epsilon': epsilon, 'central': -.02, 'forward': -.02, 'backward': -.02}
                  for epsilon in EPSILONS],
    } for seed in TRAIN_SEEDS for tick in TICKS]
    return cases, points


def _point(points, seed, tick=260):
    return next(point for point in points if (point['seed'], point['tick']) == (seed, tick))


def _positive(point):
    for row in point['yaw28']:
        row.update(central=.02, forward=.02, backward=.02)


def test_complete_stable_data_and_four_negative_primary_points_support_next_step():
    result = clean_history_hypothesis(*_evidence())
    assert result['data_admitted'] and result['supported']
    assert result['status'] == result['coverage']['status'] == 'SUPPORTED'
    assert [(point['seed'], point['tick']) for point in result['coverage']['primary_points']] == [
        (seed, 260) for seed in TRAIN_SEEDS]


@pytest.mark.parametrize('seed', TRAIN_SEEDS)
def test_one_positive_primary_point_rejects_coverage_despite_eleven_negative_points(seed):
    cases, points = _evidence()
    _positive(_point(points, seed))
    result = clean_history_hypothesis(cases, points)
    assert result['data_admitted']
    assert result['status'] == result['coverage']['status'] == 'REJECTED'
    assert not result['supported'] and not result['coverage']['supported']


def test_secondary_signs_do_not_replace_the_fixed_tick260_hypothesis():
    cases, points = _evidence()
    for point in points:
        if point['tick'] != 260:
            _positive(point)
    result = clean_history_hypothesis(cases, points)
    assert result['data_admitted'] and result['coverage']['supported'] and result['supported']


def test_unknown_secondary_point_withholds_data_even_when_primary_coverage_is_supported():
    cases, points = _evidence()
    _point(points, TRAIN_SEEDS[0], tick=110)['label_admitted'] = False
    result = clean_history_hypothesis(cases, points)
    assert not result['data_admitted'] and not result['supported']
    assert result['status'] == 'REFERENCE_UNRESOLVED'
    assert result['coverage']['status'] == 'SUPPORTED' and result['coverage']['supported']


@pytest.mark.parametrize('seed', TRAIN_SEEDS)
def test_unknown_primary_reference_is_not_relabelled_as_a_negative_result(seed):
    cases, points = _evidence()
    point = _point(points, seed)
    _positive(point)
    point['label_admitted'] = False
    result = clean_history_hypothesis(cases, points)
    assert result['status'] == result['coverage']['status'] == 'REFERENCE_UNRESOLVED'
    assert not result['data_admitted'] and not result['supported']


@pytest.mark.parametrize('scale', range(4))
def test_negative_average_cannot_hide_a_positive_primary_scale(scale):
    cases, points = _evidence()
    _point(points, TRAIN_SEEDS[0])['yaw28'][scale].update(central=.001, forward=.001, backward=.001)
    result = clean_history_hypothesis(cases, points)
    assert result['coverage']['status'] == 'REJECTED' and not result['supported']


@pytest.mark.parametrize('partial', ('case', 'short_case', 'point', 'branch'))
def test_partial_work_cannot_produce_complete_data_admission(partial):
    cases, points = _evidence()
    if partial == 'case':
        cases[-1]['status'] = 'INCOMPLETE'
    elif partial == 'short_case':
        cases[-1]['steps'] = 599
    elif partial == 'point':
        points[-1]['status'] = 'INCOMPLETE'
    else:
        points[-1]['branches'][-1]['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError):
        clean_history_hypothesis(cases, points)


@pytest.mark.parametrize('missing', ('case', 'point', 'branch', 'scale', 'duplicate_point'))
def test_missing_or_duplicated_registered_evidence_cannot_qualify(missing):
    cases, points = _evidence()
    if missing == 'case':
        cases.pop()
    elif missing == 'point':
        points.pop()
    elif missing == 'branch':
        points[-1]['branches'].pop()
    elif missing == 'scale':
        points[-1]['yaw28'].pop()
    else:
        points[-1] = points[0]
    with pytest.raises(ValueError):
        clean_history_hypothesis(cases, points)
