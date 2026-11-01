"""Synthetic60 hypothesis decisions; no arrays, models or physics are loaded."""
from __future__ import annotations

import pytest

from tools60.collect_noise60 import (
    CENTERS,
    EPSILONS,
    TICKS,
    TRAIN_SEEDS,
    paired_hypothesis,
)


def _points():
    return [
        {'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'centers': {
            center: {'status': 'COMPLETE', 'label_admitted': True, 'yaw28': [
                {'epsilon': epsilon, 'central': -.02 if center == 'clean' else .02}
                for epsilon in EPSILONS
            ]} for center in CENTERS
        }}
        for seed in TRAIN_SEEDS for tick in TICKS
    ]


def _point(points, seed, tick=260):
    return next(point for point in points if (point['seed'], point['tick']) == (seed, tick))


def _set_response(point, center, value):
    for row in point['centers'][center]['yaw28']:
        row['central'] = value


def test_all_four_primary_reversals_support_hypothesis_without_selecting_other_ticks():
    points = _points()
    # All eight secondary points disagree; they cannot replace the fixed primary set.
    for point in points:
        if point['tick'] != 260:
            _set_response(point, 'clean', .02)
    result = paired_hypothesis(points)
    assert result['status'] == 'SUPPORTED' and result['supported']
    assert [(row['seed'], row['tick']) for row in result['primary_cases']] == [
        (seed, 260) for seed in TRAIN_SEEDS]
    assert len(result['all_point_diagnostics']) == 12
    assert sum(row['status'] == 'REJECTED' for row in result['all_point_diagnostics']) == 8
    assert result['selected_points_or_eps'] is False


@pytest.mark.parametrize('seed', TRAIN_SEEDS)
def test_one_resolved_primary_counterexample_rejects_despite_eleven_supporting_points(seed):
    points = _points()
    _set_response(_point(points, seed), 'clean', .02)
    result = paired_hypothesis(points)
    assert result['status'] == 'REJECTED' and not result['supported']
    assert sum(row['noisy_positive_clean_negative'] is True
               for row in result['all_point_diagnostics']) == 11
    assert sum(row['status'] == 'REJECTED' for row in result['primary_cases']) == 1


@pytest.mark.parametrize('seed', TRAIN_SEEDS)
@pytest.mark.parametrize('center', CENTERS)
def test_unresolved_primary_center_is_unknown_instead_of_a_false_hypothesis(seed, center):
    points = _points()
    point = _point(points, seed)
    point['centers'][center]['label_admitted'] = False
    _set_response(point, 'clean', .02)  # The apparent counterexample is inadmissible.
    result = paired_hypothesis(points)
    assert result['status'] == 'REFERENCE_UNRESOLVED' and not result['supported']
    unknown = next(row for row in result['primary_cases'] if row['seed'] == seed)
    assert unknown['reference_resolved'] is False
    assert unknown['noisy_positive_clean_negative'] is None
    assert unknown['status'] == 'REFERENCE_UNRESOLVED'
    assert all(row['status'] != 'REJECTED' for row in result['primary_cases'])


def test_clean_negative_alone_is_insufficient_when_noisy_is_also_negative():
    points = _points()
    _set_response(_point(points, TRAIN_SEEDS[0]), 'noisy', -.02)
    result = paired_hypothesis(points)
    assert result['status'] == 'REJECTED' and not result['supported']
    assert result['primary_cases'][0]['noisy_positive_clean_negative'] is False


def test_unresolved_secondary_tick_cannot_change_the_fixed_primary_judgment():
    points = _points()
    point = _point(points, TRAIN_SEEDS[0], tick=110)
    point['centers']['clean']['label_admitted'] = False
    result = paired_hypothesis(points)
    assert result['status'] == 'SUPPORTED'
    assert result['all_point_diagnostics'][0]['status'] == 'REFERENCE_UNRESOLVED'


@pytest.mark.parametrize('incomplete', ('point', 'clean', 'noisy'))
def test_incomplete_work_is_never_accepted_as_a_resolved_primary_point(incomplete):
    points = _points()
    point = _point(points, TRAIN_SEEDS[0])
    record = point if incomplete == 'point' else point['centers'][incomplete]
    record['status'] = 'INCOMPLETE'
    result = paired_hypothesis(points)
    assert result['status'] == 'REFERENCE_UNRESOLVED' and not result['supported']


def test_omitting_or_replacing_a_registered_point_is_rejected():
    points = _points()
    points[-1] = points[0]
    with pytest.raises(ValueError, match='point grid differs'):
        paired_hypothesis(points)
