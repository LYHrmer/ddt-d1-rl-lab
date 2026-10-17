"""Check matrix coverage and qualified selection, with no model or physics."""
from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools18 import common18 as common  # noqa: E402
from tools18 import evaluate_delay18 as evaluator  # noqa: E402


def test_probe_is_the_actual_known_failure_and_heldout_seeds_are_disjoint():
    assert evaluator.matrix('probe_base') == [('turn_stop', 3, 1001)]
    formal = evaluator.matrix('eval_random')
    heldout = evaluator.matrix('holdout')
    assert len(set(formal)) == len(set(heldout)) == 36
    assert {row[2] for row in formal} == {1001, 1002, 1003}
    assert {row[2] for row in heldout} == {2001, 2002, 2003}
    assert {row[2] for row in formal}.isdisjoint({row[2] for row in heldout})
    assert {row[:2] for row in formal} == {row[:2] for row in heldout}


def records(tmp_path, monkeypatch, *, zero_pass=False, random_pass=False, zero_j=1., random_j=1.):
    monkeypatch.setattr(evaluator, 'ROOT', tmp_path)
    plan = {'output_root': 'execution', 'source_identity': {'fake': 'source'}}
    for mode, passed, score in (('zero', zero_pass, zero_j), ('random', random_pass, random_j)):
        path = tmp_path/'execution'/('eval_'+mode)
        path.mkdir(parents=True)
        report = {'status': 'COMPLETE', 'stage': 'eval_'+mode, 'source_identity': plan['source_identity'],
                  'cleanup_complete': True, 'inputs_unchanged': True,
                  'actual_counts': common.expected_counts(21600),
                  'all_36_reference_pass': passed,
                  'cases': [{'J': score, 'reference_thresholds_met': passed, 'sequence': task,
                             'delay_ticks': delay, 'reset_seed': seed, 'done_reason': 'HORIZON_12S',
                             'failure': False, 'actual_counts': common.expected_counts(600)}
                            for task, delay, seed in evaluator.matrix('eval_'+mode)]}
        common.atomic_json(path/'report.json', report)
    return plan


@pytest.mark.parametrize('zero,random,expected', [(False, False, None), (True, False, 'zero'),
                                                (False, True, 'random'), (True, True, 'zero')])
def test_only_all36_pass_can_enter_holdout(tmp_path, monkeypatch, zero, random, expected):
    plan = records(tmp_path, monkeypatch, zero_pass=zero, random_pass=random)
    assert evaluator.selection(plan) == expected


def test_selection_uses_predeclared_score_after_qualification(tmp_path, monkeypatch):
    plan = records(tmp_path, monkeypatch, zero_pass=True, random_pass=True, random_j=.5)
    assert evaluator.selection(plan) == 'random'


@pytest.mark.parametrize('change', ['partial', 'missing_case', 'failed_cleanup', 'false_aggregate',
                                   'duplicate_case', 'foreign_source', 'short_horizon', 'nonfinite_score', 'wrong_count'])
def test_incomplete_or_contradictory_candidate_is_rejected(tmp_path, monkeypatch, change):
    plan = records(tmp_path, monkeypatch, random_pass=True)
    path = tmp_path/'execution/eval_random/report.json'
    report = common.read_json(path)
    if change == 'partial':
        report['status'] = 'INCOMPLETE'
    elif change == 'missing_case':
        report['cases'].pop()
    elif change == 'failed_cleanup':
        report['cleanup_complete'] = False
    elif change == 'false_aggregate':
        report['cases'][0]['reference_thresholds_met'] = False
    elif change == 'duplicate_case':
        report['cases'][0] = dict(report['cases'][1])
    elif change == 'foreign_source':
        report['source_identity'] = {'foreign': True}
    elif change == 'short_horizon':
        report['cases'][0]['done_reason'] = 'TERMINATED'
    elif change == 'nonfinite_score':
        report['cases'][0]['J'] = float('nan')
    else:
        report['actual_counts']['native_steps'] -= 1
    if change == 'nonfinite_score':
        path.write_text(json.dumps(report, allow_nan=True))
    else:
        common.atomic_json(path, report)
    with pytest.raises(ValueError):
        evaluator.selection(plan)


def test_partial_counter_budget_does_not_fill_early_termination():
    zero = common.expected_counts(0)
    short = common.expected_counts(270)
    assert evaluator.add_counts(zero, short, common.expected_counts(600)) == short
    with pytest.raises(ValueError, match='cap exceeded'):
        evaluator.add_counts(short, common.expected_counts(400), common.expected_counts(600))
