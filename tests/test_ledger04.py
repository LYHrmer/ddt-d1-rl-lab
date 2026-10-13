"""Regression tests for the contract 04 ledger's gate counting and verdicts.

These pin the correction that motivated them. The first ledger counted 04b's
reverse_stop as 7 of 8 gates because a null (not applicable) flag was summed
as a pass, and it wrote 'overall': 'FAIL' and the blocking-gate sentence as
literal strings. The true figure is 6 of 7 applicable gates. Everything here
works on synthetic inputs, plus the real receipts only to confirm that the
derived numbers still agree with them; nothing is written under evidence/.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))

import build_final_ledger04 as ledger  # noqa: E402

GATES = {'stable_vx_mae_m_s': .06, 'stable_wz_mae_rad_s_each_sign': .08,
         'stop_last_second_horizontal_speed_m_s': .02, 'stop_last_second_abs_wz_rad_s': .03,
         'stop_displacement_m': .05, 'peak_abs_roll_or_pitch_rad': .20,
         'completed_12s': True, 'no_fall_or_illegal_support': True,
         'tasks_required_to_pass': 3}

REVERSE_FLAGS = {'completed_12s': True, 'no_fall_or_illegal_support': True,
                 'stable_vx': False, 'stable_wz': None, 'stop_horizontal_speed': True,
                 'stop_abs_wz': True, 'stop_displacement': True, 'peak_roll_pitch': True}


def reverse_metrics(**overrides):
    """A reverse_stop record shaped like 04b's: one failing gate, stable_wz null."""
    base = {'sequence': 'reverse_stop', 'policy_transitions': 600, 'done_reason': 'HORIZON_12S',
            'failure': False, 'stable_vx_mae_m_s': .0695, 'stable_wz_mae_rad_s': None,
            'stop_last_second_horizontal_speed_m_s': .0026,
            'stop_last_second_abs_wz_rad_s': .0008, 'stop_displacement_m': .0278,
            'peak_abs_roll_or_pitch_rad': .0841, 'threshold_flags': dict(REVERSE_FLAGS)}
    base.update(overrides)
    return base


# --- gate_summary: true, false and null are three different things ------------

def test_null_is_not_applicable_and_is_never_counted_as_a_pass():
    out = ledger.gate_summary(REVERSE_FLAGS)
    assert (out['gates_passed'], out['gates_applicable'], out['gates_total']) == (6, 7, 8)
    assert out['not_applicable'] == ['stable_wz'] and out['failed_gates'] == ['stable_vx']


def test_the_original_seven_of_eight_error_is_what_a_null_as_pass_would_produce():
    """Documents the bug: summing the null as a pass gives the figure that was wrong."""
    wrong = {**REVERSE_FLAGS, 'stable_wz': True}
    out = ledger.gate_summary(wrong)
    assert (out['gates_passed'], out['gates_applicable']) == (7, 8)
    assert out['not_applicable'] == []   # the tell: nothing is left not applicable


def test_all_true_is_eight_of_eight_with_no_failures():
    out = ledger.gate_summary({k: True for k in REVERSE_FLAGS})
    assert (out['gates_passed'], out['gates_applicable']) == (8, 8) and out['failed_gates'] == []


@pytest.mark.parametrize('bad_value', [1, 0, 'true', 'false', 1.0, [], {}])
def test_only_true_false_or_null_are_accepted(bad_value):
    with pytest.raises(ValueError, match='true, false or null'):
        ledger.gate_summary({**REVERSE_FLAGS, 'stop_displacement': bad_value})


@pytest.mark.parametrize('flags', [None, [], 'x', {}, {k: True for k in list(REVERSE_FLAGS)[:-1]},
                                   {**REVERSE_FLAGS, 'extra': True}])
def test_exactly_the_eight_gates_are_required(flags):
    with pytest.raises(ValueError, match='exactly the eight'):
        ledger.gate_summary(flags)


@pytest.mark.parametrize('gate', ['completed_12s', 'no_fall_or_illegal_support'])
def test_completion_gates_apply_to_every_task_and_cannot_be_null(gate):
    with pytest.raises(ValueError, match='cannot be null'):
        ledger.gate_summary({**REVERSE_FLAGS, gate: None})


def test_an_all_null_record_is_rejected_by_the_completion_gate_rule_not_summarised_as_0_of_0():
    """The rule that actually stops it. A separate all-null check used to exist; it was
    unreachable, because this rule already forbids null on both completion gates."""
    with pytest.raises(ValueError, match='cannot be null'):
        ledger.gate_summary({k: None for k in REVERSE_FLAGS})


def test_every_accepted_shape_has_at_least_two_applicable_gates():
    """Why no all-null check is needed: the completion gates are always applicable."""
    import itertools
    names = list(ledger.NUMERIC_GATES)
    for completion in itertools.product([True, False], repeat=2):
        for rest in itertools.product([True, False, None], repeat=len(names)):
            flags = dict(zip(names, rest))
            flags.update(dict(zip(ledger.BOOLEAN_GATES, completion)))
            assert ledger.gate_summary(flags)['gates_applicable'] >= 2


def test_gate_summary_alone_cannot_catch_a_null_turned_into_true():
    """The division of labour, pinned: this layer cannot know which null is legitimate."""
    assert ledger.gate_summary({**REVERSE_FLAGS, 'stable_wz': True})['gates_applicable'] == 8


# --- task_gate_summary: explanations come from raw metrics and frozen gates --

def test_blocking_gate_is_derived_from_the_raw_metric_and_the_frozen_threshold():
    out = ledger.task_gate_summary(reverse_metrics(), GATES)
    assert out['failed_gates'] == ['stable_vx']
    assert out['blocking_gate'] == 'stable_vx_mae_m_s 0.0695 > 0.06'
    detail = out['failed_gate_details'][0]
    assert (detail['value'], detail['threshold'], detail['operator']) == (.0695, .06, '<=')


def test_the_explanation_follows_the_numbers_rather_than_a_stored_sentence():
    """Change the metric and the threshold and the sentence changes with them."""
    out = ledger.task_gate_summary(reverse_metrics(stable_vx_mae_m_s=.0712), GATES)
    assert out['blocking_gate'] == 'stable_vx_mae_m_s 0.0712 > 0.06'
    looser = dict(GATES, stable_vx_mae_m_s=.07)
    out = ledger.task_gate_summary(reverse_metrics(stable_vx_mae_m_s=.0712), looser)
    assert out['blocking_gate'] == 'stable_vx_mae_m_s 0.0712 > 0.07'


def test_a_task_with_no_failing_gate_has_no_blocking_gate():
    flags = {**REVERSE_FLAGS, 'stable_vx': True}
    out = ledger.task_gate_summary(reverse_metrics(stable_vx_mae_m_s=.05, threshold_flags=flags),
                                   GATES)
    assert out['blocking_gate'] is None and out['failed_gate_details'] == []
    assert (out['gates_passed'], out['gates_applicable']) == (7, 7)


def test_a_flag_that_disagrees_with_the_raw_metric_is_rejected():
    lying = {**REVERSE_FLAGS, 'stable_vx': True}       # metric .0695 > .06 says False
    with pytest.raises(ValueError, match='flag disagrees'):
        ledger.task_gate_summary(reverse_metrics(threshold_flags=lying), GATES)


def test_a_failing_displacement_is_caught_even_when_the_flag_says_pass():
    bad = reverse_metrics(stop_displacement_m=.06)
    with pytest.raises(ValueError, match='flag disagrees'):
        ledger.task_gate_summary(bad, GATES)


def test_forward_and_reverse_must_have_stable_wz_not_applicable():
    wrong = {**REVERSE_FLAGS, 'stable_wz': True}
    with pytest.raises(ValueError, match='exactly one inapplicable gate: stable_wz'):
        ledger.task_gate_summary(
            reverse_metrics(threshold_flags=wrong, stable_wz_mae_rad_s=.05), GATES)


def test_turn_must_have_stable_vx_not_applicable():
    flags = {**REVERSE_FLAGS, 'stable_vx': None, 'stable_wz': False}
    out = ledger.task_gate_summary(
        reverse_metrics(sequence='turn_stop', threshold_flags=flags, stable_vx_mae_m_s=None,
                        stable_wz_mae_rad_s=.11), GATES)
    assert out['not_applicable'] == ['stable_vx'] and out['failed_gates'] == ['stable_wz']


def test_an_inapplicable_metric_must_actually_be_null():
    with pytest.raises(ValueError, match='inapplicable metric must be null'):
        ledger.task_gate_summary(reverse_metrics(stable_wz_mae_rad_s=.01), GATES)


@pytest.mark.parametrize('field,value', [
    ('stop_displacement_m', None), ('stop_displacement_m', -1.), ('stop_displacement_m', 'x'),
    ('stop_displacement_m', float('nan')), ('stop_displacement_m', float('inf')),
    ('stop_displacement_m', True),
])
def test_numeric_metrics_must_be_finite_nonnegative_numbers(field, value):
    with pytest.raises(ValueError):
        ledger.task_gate_summary(reverse_metrics(**{field: value}), GATES)


@pytest.mark.parametrize('threshold_key,value', [
    ('stable_vx_mae_m_s', None), ('stable_vx_mae_m_s', -.1), ('stable_vx_mae_m_s', 'x'),
    ('stable_vx_mae_m_s', float('nan')),
])
def test_a_missing_or_invalid_frozen_threshold_is_rejected(threshold_key, value):
    with pytest.raises(ValueError, match='frozen threshold'):
        ledger.task_gate_summary(reverse_metrics(), dict(GATES, **{threshold_key: value}))


def test_a_missing_frozen_threshold_key_is_rejected():
    gates = {k: v for k, v in GATES.items() if k != 'stop_displacement_m'}
    with pytest.raises(ValueError, match='frozen threshold'):
        ledger.task_gate_summary(reverse_metrics(), gates)


def test_an_unknown_sequence_is_rejected():
    with pytest.raises(ValueError, match='unknown or missing'):
        ledger.task_gate_summary(reverse_metrics(sequence='sideways'), GATES)


@pytest.mark.parametrize('metrics,gates', [(None, GATES), (reverse_metrics(), None),
                                           ([], GATES), (reverse_metrics(), [])])
def test_non_object_inputs_are_rejected(metrics, gates):
    with pytest.raises(ValueError):
        ledger.task_gate_summary(metrics, gates)


def test_a_fall_is_reported_as_a_failed_completion_gate_not_hidden():
    """A run that ended early has no stop-phase metrics; the receipt records that."""
    flags = {**REVERSE_FLAGS, 'completed_12s': False, 'no_fall_or_illegal_support': False}
    fell = reverse_metrics(policy_transitions=558, done_reason='TERMINATED', failure=True,
                           threshold_flags=flags)
    out = ledger.task_gate_summary(fell, GATES)
    assert {'completed_12s', 'no_fall_or_illegal_support'} <= set(out['failed_gates'])


def test_a_completion_flag_that_disagrees_with_the_run_is_rejected():
    fell_but_flagged_ok = reverse_metrics(policy_transitions=558, done_reason='TERMINATED',
                                          failure=True)
    with pytest.raises(ValueError, match='completion/failure'):
        ledger.task_gate_summary(fell_but_flagged_ok, GATES)


def test_the_frozen_boolean_requirements_must_be_true():
    with pytest.raises(ValueError, match='frozen boolean requirement'):
        ledger.task_gate_summary(reverse_metrics(), dict(GATES, completed_12s=False))


# --- qualification_verdict: a partial result is never a qualification --------

@pytest.mark.parametrize('passed,total,early,required,expected', [
    (3, 3, 0, 3, 'PASS'),
    (2, 3, 0, 3, 'FAIL'),     # the best partial result still fails
    (1, 3, 0, 3, 'FAIL'),     # 03d
    (0, 3, 1, 3, 'FAIL'),
    (3, 3, 1, 3, 'FAIL'),     # all passed but a run ended early
    (2, 2, 0, 3, 'FAIL'),     # every evaluated task passed, but a task was never evaluated
    (3, 4, 0, 3, 'FAIL'),     # extra tasks change what was evaluated
])
def test_overall_is_derived_and_a_partial_result_is_never_a_pass(
        passed, total, early, required, expected):
    assert ledger.qualification_verdict(passed, total, early, required) == expected


@pytest.mark.parametrize('args', [(4, 3, 0, 3), (-1, 3, 0, 3), (0, 3, -1, 3), (0, 3, 0, 0),
                                  (True, 3, 0, 3), (3, 3, False, 3), (3.0, 3, 0, 3),
                                  ('3', 3, 0, 3), (None, 3, 0, 3)])
def test_verdict_rejects_impossible_or_ill_typed_counts(args):
    with pytest.raises(ValueError):
        ledger.qualification_verdict(*args)


def test_a_true_three_of_three_would_turn_the_verdict_to_pass():
    """The literal 'FAIL' this replaced could never have said so."""
    assert ledger.qualification_verdict(3, 3, 0, 3) == 'PASS'


# --- the real receipts agree with the derived numbers ------------------------

from _workstation import needs_workstation_evidence as needs_evidence  # noqa: E402

EVIDENCE = ROOT/'evidence/career_contract_04'


@needs_evidence
def test_the_real_04b_reverse_stop_is_six_of_seven_with_stable_vx_blocking():
    release = json.loads((EVIDENCE/'pilot04b_release_01.json').read_text())
    report = json.loads(
        (EVIDENCE/'pilot04b_deployment_plan_01/score02_reverse_stop/report.json').read_text())
    out = ledger.task_gate_summary(report['metrics'], release['qualification_gates_unchanged'])
    assert (out['gates_passed'], out['gates_applicable'], out['gates_total']) == (6, 7, 8)
    assert out['not_applicable'] == ['stable_wz'] and out['failed_gates'] == ['stable_vx']
    assert out['blocking_gate'].startswith('stable_vx_mae_m_s 0.0694') and out['blocking_gate'].endswith('> 0.06')


@needs_evidence
def test_every_real_python_episode_passes_the_task_level_cross_check():
    release = json.loads((EVIDENCE/'pilot04b_release_01.json').read_text())
    gates = release['qualification_gates_unchanged']
    lines = (EVIDENCE/'pilot04b_python_eval_01/episodes.jsonl').read_text().strip().split('\n')
    reverse = next(json.loads(x) for x in lines if json.loads(x)['sequence'] == 'reverse_stop')
    assert ledger.task_gate_summary(reverse, gates)['gates_applicable'] == 7


@needs_evidence
def test_the_historical_ledger_is_preserved_and_the_corrected_one_agrees_with_the_receipts():
    old = json.loads((EVIDENCE/'final_delivery_01.json').read_text())
    new = json.loads((EVIDENCE/'final_delivery_02.json').read_text())
    assert (old['control_qualification']['best_single_task']['gates_applicable'] == 8
            and new['control_qualification']['best_single_task']['gates_applicable'] == 7)
    for field in ('contract04_transitions_total', 'contract04_native_total_all_pools'):
        assert old[field] == new[field]


@needs_evidence
def test_versions_passing_covers_contract_04_only_not_the_whole_project():
    new = json.loads((EVIDENCE/'final_delivery_02.json').read_text())
    versions = new['control_qualification']['versions']
    assert sorted(versions) == ['04a', '04b']     # 03b, 03c and 03d are not in this ledger
    assert all(v['tasks_passed'] < 3 for v in versions.values())


def test_ledger_functions_do_not_mutate_their_inputs():
    metrics = reverse_metrics()
    gates = dict(GATES)
    before = (copy.deepcopy(metrics), copy.deepcopy(gates))
    ledger.task_gate_summary(metrics, gates)
    assert (metrics, gates) == before
