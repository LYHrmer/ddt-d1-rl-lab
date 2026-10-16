"""Tests for tools08/verify_scale08.py, the contract 08 judge. No model, no physics."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import verify_scale08 as verify  # noqa: E402

BASE_03D = ROOT/'evidence/career_contract_03/pilot03d_python_eval_01'
PASS = {'equivalence': 'PASS'}


def task(passed, displacement=.03, wz=.05):
    return {'passed': passed, 'stop_displacement_m': displacement, 'stable_wz_mae_rad_s': wz, 'early_failure': False}


def tasks(fwd=False, rev=True, turn=False, fwd_disp=.081, turn_wz=.103):
    return {'forward_stop': task(fwd, fwd_disp, None), 'reverse_stop': task(rev, .016, None),
            'turn_stop': task(turn, .06, turn_wz)}


def test_p1_needs_at_least_two_tasks():
    late = {'relative_change': .02}
    assert verify.judge(tasks(), tasks(), late, PASS)['P1'] == 'NOT_MET'
    assert verify.judge(tasks(fwd=True), tasks(), late, PASS)['P1'] == 'MET'


def test_p2_and_p3_are_strict_below():
    late = {'relative_change': .02}
    assert verify.judge(tasks(fwd_disp=.0499), tasks(), late, PASS)['P2'] == 'MET'
    assert verify.judge(tasks(fwd_disp=.05), tasks(), late, PASS)['P2'] == 'NOT_MET'
    assert verify.judge(tasks(turn_wz=.0799), tasks(), late, PASS)['P3'] == 'MET'
    assert verify.judge(tasks(turn_wz=.08), tasks(), late, PASS)['P3'] == 'NOT_MET'
    assert verify.judge(tasks(turn_wz=None), tasks(), late, PASS)['P3'] == 'NOT_EVALUABLE'


def test_p5_is_a_regression_check_on_reverse():
    late = {'relative_change': .02}
    assert verify.judge(tasks(rev=True), tasks(rev=True), late, PASS)['P5'] == 'MET'
    assert verify.judge(tasks(rev=False), tasks(rev=True), late, PASS)['P5'] == 'NOT_MET'
    assert verify.judge(tasks(rev=False), tasks(rev=False), late, PASS)['P5'] == 'NOT_EVALUABLE'


def test_a_failed_equivalence_makes_every_prediction_not_evaluable():
    out = verify.judge(tasks(fwd=True, turn=True), tasks(), {'relative_change': 0.}, {'equivalence': 'FAIL'})
    assert out['E1'] == 'NOT_MET' and all(out[p] == 'NOT_EVALUABLE' for p in ('P1', 'P2', 'P3', 'P4', 'P5'))
    out = verify.judge(tasks(), tasks(), None, {'equivalence': 'INCOMPLETE'})
    assert out['E1'] == 'NOT_MET'


def test_p4_is_reported_not_judged():
    assert verify.judge(tasks(), tasks(), {'relative_change': .3}, PASS)['P4'] == 'REPORTED'
    assert verify.judge(tasks(), tasks(), None, PASS)['P4'] == 'NOT_EVALUABLE'


def test_late_return_change_uses_the_last_quarter(tmp_path):
    rows = ['rollout/ep_rew_mean,x'] + [f'{v},0' for v in [0.]*192 + [10.]*20 + [10.]*24 + [15.]*20]
    path = tmp_path/'progress.csv'; path.write_text('\n'.join(rows)+'\n')
    out = verify.late_return_change(path)
    assert out['iterations'] == 256 and out['first_mean'] == pytest.approx(10.) and out['last_mean'] == pytest.approx(15.)
    assert out['relative_change'] == pytest.approx(.5)


def test_late_return_change_skips_empty_cells(tmp_path):
    """Two columns, so an empty cell is a real row: csv drops a blank single-column line entirely."""
    rows = ['rollout/ep_rew_mean,x'] + [',0']*10 + ['4.,0']*10
    path = tmp_path/'p.csv'; path.write_text('\n'.join(rows)+'\n')
    assert verify.late_return_change(path, fraction=1., edge=20)['first_mean'] == pytest.approx(4.)


@needs_workstation_evidence
def test_judging_03d_against_itself_reproduces_the_published_table():
    """The README gate table says 03d passed only reverse_stop; the judge must agree."""
    episodes = verify.read_episodes(BASE_03D)
    results = verify.task_results(episodes)
    assert {t: r['passed'] for t, r in results.items()} == {'forward_stop': False, 'reverse_stop': True, 'turn_stop': False}
    assert results['forward_stop']['stop_displacement_m'] == pytest.approx(.0806, abs=5e-4)
    assert results['turn_stop']['stable_wz_mae_rad_s'] == pytest.approx(.1029, abs=5e-4)
    out = verify.judge(results, results, {'relative_change': .1}, PASS)
    assert (out['P1'], out['P2'], out['P3'], out['P5']) == ('NOT_MET', 'NOT_MET', 'NOT_MET', 'MET')


@needs_workstation_evidence
def test_a_flag_that_disagrees_with_its_metric_is_refused():
    episodes = copy.deepcopy(verify.read_episodes(BASE_03D))
    episodes['forward_stop']['threshold_flags']['stop_displacement'] = True
    with pytest.raises(ValueError, match='disagrees'):
        verify.task_results(episodes)


def test_an_evaluation_missing_a_task_is_refused(tmp_path):
    (tmp_path/'episodes.jsonl').write_text(json.dumps({'sequence': 'forward_stop', 'delay_ticks': 0, 'reset_seed': 7001})+'\n')
    with pytest.raises(SystemExit, match='lacks tasks'):
        verify.read_episodes(tmp_path)


def test_the_tool_imports_no_simulator():
    source = Path(verify.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


@needs_workstation_evidence
def test_the_evaluator_flag_is_cross_checked_on_its_own():
    """Flip only reference_thresholds_met: the gate layer accepts it, this cross-check must not."""
    episodes = copy.deepcopy(verify.read_episodes(BASE_03D))
    episodes['forward_stop']['reference_thresholds_met'] = True
    with pytest.raises(ValueError, match='evaluator flag'):
        verify.task_results(episodes)


def test_only_the_zero_delay_seed_7001_episode_of_each_task_is_read(tmp_path):
    rows = []
    for t in verify.TASKS:
        rows.append({'sequence': t, 'delay_ticks': 0, 'reset_seed': 7001, 'tag': 'the one'})
        rows.append({'sequence': t, 'delay_ticks': 2, 'reset_seed': 7001, 'tag': 'delayed'})
        rows.append({'sequence': t, 'delay_ticks': 0, 'reset_seed': 1001, 'tag': 'other seed'})
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = verify.read_episodes(tmp_path)
    assert {r['tag'] for r in out.values()} == {'the one'}
