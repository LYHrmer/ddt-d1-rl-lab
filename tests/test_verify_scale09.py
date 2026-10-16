"""Tests for tools09/verify_scale09.py, written before any contract 09 result. No physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools08', 'tools09'):
    sys.path.insert(0, str(ROOT/sub))

import verify_scale09 as verify  # noqa: E402

PASS = {'equivalence': 'PASS'}
REPLAY08 = ROOT/'evidence/career_contract_08/scale08_training_replay_01/report.json'
EVAL08 = ROOT/'evidence/career_contract_08/scale08_python_eval_01'
LAUNCH08 = ROOT/'evidence/career_contract_08/scale08_train_01_launcher/report.json'


def replay(stds, forward_ends):
    return {'ppo_timeline': [{'train/std': s} for s in stds],
            'timeline': [{'early_ends_by_segment': {'drive_forward': f, 'turn_left': 9}} for f in forward_ends]}


def task(passed=False, early=False):
    return {'passed': passed, 'early_failure': early}


def tasks(n_passed=1, forward_early=False):
    names = ('forward_stop', 'reverse_stop', 'turn_stop')
    return {t: task(i < n_passed, forward_early and t == 'forward_stop') for i, t in enumerate(names)}


def test_last_quarter_std_is_the_mean_of_the_last_four_of_sixteen_slices():
    r = replay([.3]*12 + [.4, .5, .6, .7], [0]*16)
    assert verify.last_quarter_std(r) == pytest.approx(.55)


def test_a_replay_with_the_wrong_number_of_slices_is_refused():
    with pytest.raises(SystemExit, match='16 slices'):
        verify.last_quarter_std(replay([.3]*15, [0]*15))


def test_a_last_quarter_slice_without_std_is_refused():
    with pytest.raises(SystemExit, match='no std'):
        verify.last_quarter_std(replay([.3]*15 + [None], [0]*16))


def test_forward_ends_count_only_forward_drive_segments_of_the_last_quarter():
    assert verify.last_quarter_forward_ends(replay([.3]*16, [100]*12 + [1, 2, 3, 4])) == 10


@pytest.mark.parametrize('std, expected', [(.4499, 'MET'), (.45, 'NOT_MET'), (.676, 'NOT_MET')])
def test_q1_is_strict_below_the_frozen_threshold(std, expected):
    assert verify.judge(std, 0, tasks(), PASS)['Q1'] == expected


@pytest.mark.parametrize('ends, expected', [(15, 'MET'), (16, 'NOT_MET'), (43, 'NOT_MET'), (0, 'MET')])
def test_q2_allows_up_to_the_frozen_count(ends, expected):
    assert verify.judge(.3, ends, tasks(), PASS)['Q2'] == expected


def test_q3_needs_two_tasks_and_q4_needs_forward_to_finish():
    assert verify.judge(.3, 0, tasks(2), PASS)['Q3'] == 'MET' and verify.judge(.3, 0, tasks(1), PASS)['Q3'] == 'NOT_MET'
    assert verify.judge(.3, 0, tasks(forward_early=True), PASS)['Q4'] == 'NOT_MET'
    assert verify.judge(.3, 0, tasks(), PASS)['Q4'] == 'MET'


def test_a_failed_equivalence_gate_makes_every_prediction_not_evaluable():
    out = verify.judge(.3, 0, tasks(3), {'equivalence': 'FAIL'})
    assert out['E1'] == 'NOT_MET' and all(out[q] == 'NOT_EVALUABLE' for q in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'))


def test_the_interpretation_follows_the_frozen_table():
    t = verify.release.INTERPRETATION
    assert verify.interpretation({'Q1': 'MET', 'Q2': 'MET'}) == t['Q1 met, Q2 met']
    assert verify.interpretation({'Q1': 'MET', 'Q2': 'NOT_MET'}) == t['Q1 met, Q2 not met']
    assert verify.interpretation({'Q1': 'NOT_MET', 'Q2': 'MET'}) == t['Q1 not met']
    assert 'not evaluable' in verify.interpretation({'Q1': 'NOT_EVALUABLE'})


@needs_workstation_evidence
def test_judging_contract_08_by_contract_09_rules_reproduces_its_baseline():
    """Contract 08 is the baseline the thresholds were set against: it must fail Q1 to Q4."""
    for path in (REPLAY08, EVAL08/'episodes.jsonl', LAUNCH08):
        if not path.is_file():
            pytest.skip('contract 08 receipts not on this machine')
    r = json.loads(REPLAY08.read_text())
    std, ends = verify.last_quarter_std(r), verify.last_quarter_forward_ends(r)
    assert std == pytest.approx(verify.release.BASELINE_08['last_quarter_mean_std'], abs=5e-4)
    assert ends == verify.release.BASELINE_08['forward_drive_early_ends_last_quarter']
    out = verify.judge(std, ends, verify.gates.task_results(verify.gates.read_episodes(EVAL08)), json.loads(LAUNCH08.read_text()))
    assert (out['Q1'], out['Q2'], out['Q3'], out['Q4']) == ('NOT_MET', 'NOT_MET', 'NOT_MET', 'NOT_MET')
    speeds = verify.drive_speeds(EVAL08)
    assert speeds['forward_stop'] == pytest.approx(.062, abs=5e-4) and speeds['reverse_stop'] == pytest.approx(.281, abs=5e-4)
