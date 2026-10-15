"""Tests for tools04/compare_training_ledgers04.py.

The comparison argues from training ledgers about what a version was rewarded and
penalised for. The places it could mislead are exact: a term only one version has
must not be read as zero, the cost of a new term is measured against the tracking
it gave up, and only the last rollouts count. Nothing here loads a model or runs
physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))

import compare_training_ledgers04 as cmp  # noqa: E402

RECEIPT = ROOT/'evidence/career_contract_04/training_ledger_comparison_01/report.json'


def rows(values_by_term, n=30):
    """n rollouts; each term takes the matching entry of its list (or its constant)."""
    out = []
    for k in range(n):
        out.append({'reward_contribution_mean_per_tick': {
            t: (v[k] if isinstance(v, list) else v) for t, v in values_by_term.items()}})
    return out


# --- per-second means ------------------------------------------------------------
def test_per_tick_contributions_become_per_second_by_the_policy_rate():
    out = cmp.per_second_means(rows({'a': .02}))
    assert out['a'] == pytest.approx(1.0)                  # 0.02 per tick * 50 ticks per second


def test_only_the_last_rollouts_count():
    early_bad_late_good = [-1.]*10 + [.04]*20
    out = cmp.per_second_means(rows({'a': early_bad_late_good}), last=20)
    assert out['a'] == pytest.approx(.04*50)


def test_the_default_window_is_the_last_twenty_rollouts_the_results_document_names():
    """Constant data would give the same mean for any window, so the earlier ten differ from the later ten."""
    values = [0.]*10 + [1.]*10 + [3.]*10                  # last 20 average 2, last 10 average 3
    out = cmp.per_second_means(rows({'a': values}))
    assert cmp.LAST_ROLLOUTS == 20
    assert out['a'] == pytest.approx(2.*50)


def test_the_window_is_exactly_the_requested_number_of_rollouts():
    values = [0.]*9 + [1.]*21                              # last 21 are ones
    assert cmp.per_second_means(rows({'a': values}), last=21)['a'] == pytest.approx(50.)
    assert cmp.per_second_means(rows({'a': values}), last=22)['a'] < 50.


def test_a_term_missing_from_some_rollouts_is_nan_not_zero():
    data = rows({'a': .01}); del data[-1]['reward_contribution_mean_per_tick']['a']
    assert np.isnan(cmp.per_second_means(data, last=5)['a'])


def test_terms_are_the_union_over_all_rollouts():
    data = rows({'a': .01}); data[0]['reward_contribution_mean_per_tick']['b'] = .5
    assert set(cmp.per_second_means(data)) == {'a', 'b'}


# --- difference ------------------------------------------------------------------
def test_difference_is_second_minus_first():
    assert cmp.difference({'a': 1.}, {'a': 1.5})['a'] == pytest.approx(.5)
    assert cmp.difference({'a': 1.}, {'a': .25})['a'] == pytest.approx(-.75)


def test_a_term_only_one_version_has_is_none_never_a_difference_from_zero():
    out = cmp.difference({'a': 1.}, {'a': 1., 'new': -.3})
    assert out['new'] is None
    out = cmp.difference({'a': 1., 'old': .2}, {'a': 1.})
    assert out['old'] is None


def test_a_term_that_is_nan_on_one_side_is_none():
    assert cmp.difference({'a': float('nan')}, {'a': 1.})['a'] is None


# --- cost of a new term against the tracking given up -----------------------------
def test_ratio_is_the_new_terms_cost_over_the_tracking_shortfall():
    assert cmp.tracking_shortfall_ratio(-.305, -.052) == pytest.approx(.305/.052)


def test_ratio_is_none_when_tracking_did_not_fall():
    assert cmp.tracking_shortfall_ratio(-.3, 0.) is None
    assert cmp.tracking_shortfall_ratio(-.3, .05) is None
    assert cmp.tracking_shortfall_ratio(-.3, None) is None


def test_ratio_uses_magnitudes_so_a_negative_cost_is_not_a_negative_ratio():
    assert cmp.tracking_shortfall_ratio(-.4, -.1) == pytest.approx(4.)
    assert cmp.tracking_shortfall_ratio(.4, -.1) == pytest.approx(4.)


# --- the late trend of the return ----------------------------------------------------
def test_a_rising_tail_is_reported_as_rising_and_a_flat_one_as_flat():
    rising = np.concatenate([np.linspace(0., 50., 192), np.linspace(50., 60., 64)])
    flat = np.concatenate([np.linspace(0., 50., 192), np.full(64, 50.)])
    assert cmp.late_trend(rising)['relative_change'] > .05
    assert cmp.late_trend(flat)['relative_change'] == pytest.approx(0., abs=1e-9)


def test_a_falling_tail_has_a_negative_change():
    falling = np.concatenate([np.linspace(0., 50., 192), np.linspace(50., 40., 64)])
    assert cmp.late_trend(falling)['relative_change'] < 0.


def test_the_tail_is_the_last_quarter_of_the_iterations():
    out = cmp.late_trend(np.arange(256, dtype=float))
    assert out['iterations_in_tail'] == 64


def test_relative_change_is_relative_to_the_start_of_the_tail():
    y = np.concatenate([np.zeros(192), np.full(20, 10.), np.full(24, 10.), np.full(20, 15.)])
    out = cmp.late_trend(y, edge=20)
    assert out['first_mean'] == pytest.approx(10.) and out['last_mean'] == pytest.approx(15.)
    assert out['relative_change'] == pytest.approx(.5)


def test_progress_at_a_fraction_averages_the_few_iterations_before_it():
    y = np.arange(256, dtype=float)
    assert cmp.progress_at(y, .5, width=8) == pytest.approx(np.mean(y[120:129]))


def test_progress_reading_turns_empty_cells_into_nan(tmp_path):
    path = tmp_path/'progress.csv'
    path.write_text('rollout/ep_rew_mean,x\n1.5,a\n,b\n2.5,c\n')
    out = cmp.read_progress(path)
    assert out[0] == 1.5 and np.isnan(out[1]) and out[2] == 2.5


def test_ledger_reading_skips_blank_lines(tmp_path):
    path = tmp_path/'ledger.jsonl'
    path.write_text(json.dumps({'a': 1})+'\n\n'+json.dumps({'a': 2})+'\n')
    assert cmp.read_ledger(path) == [{'a': 1}, {'a': 2}]


def test_the_tool_imports_no_simulator():
    source = Path(cmp.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


# --- the numbers quoted in the results document come from the receipt -------------
@needs_workstation_evidence
def test_numbers_in_the_results_document_agree_with_the_receipt():
    if not RECEIPT.is_file():
        pytest.skip('training ledger comparison receipt not on this machine')
    report = json.loads(RECEIPT.read_text())
    text = (ROOT/'docs/career_execution_results_04.md').read_text()
    diff = report['difference_from_baseline']['04b']
    means = report['per_second_by_term']
    assert f"{means['04b']['wheel_action_mean_squared']:.3f}" in text
    assert f"{diff['vx_tracking']:+.3f}" in text
    ratio = report['notes']['04b']['new_term_cost_over_tracking_shortfall']['wheel_action_mean_squared']
    assert f'{ratio:.1f}' in text
    assert f"{100*report['late_trend']['04b']['relative_change']:.1f}" in text
    assert f"{100*report['late_trend']['03d']['relative_change']:.1f}" in text
