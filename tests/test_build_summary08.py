"""Tests for tools08/build_summary08.py, the published summary of contract 08. No physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import build_summary08 as summary  # noqa: E402

PUBLISHED = ROOT/'reports/contract08_summary.json'


def launcher(native, transitions):
    return {'trainer_result': {'actual_counts': {'native_steps': native}, 'trained_transitions': transitions}}


CAP = {'native': 84559440, 'transitions': 8454144}


def test_budget_adds_the_three_charged_steps():
    out = summary.budget(launcher(327680, 32768), launcher(83886080, 8388608), {'actual_native_steps': 16850}, CAP)
    assert out['used_native'] == 327680+83886080+16850 and out['used_transitions'] == 32768+8388608


def test_budget_over_the_frozen_cap_is_refused():
    with pytest.raises(SystemExit, match='over its cap'):
        summary.budget(launcher(327680, 32768), launcher(83886080, 8388608), {'actual_native_steps': CAP['native']}, CAP)
    with pytest.raises(SystemExit, match='over its cap'):
        summary.budget(launcher(0, 32768), launcher(0, CAP['transitions']), {'actual_native_steps': 0}, CAP)


def test_cumulative_adds_what_was_used_to_the_previous_total():
    out = summary.cumulative({'transitions': 5242880, 'native': 52932140}, 84230610, 8421376)
    assert out['native'] == 137162750 and out['transitions'] == 13664256


@needs_workstation_evidence
def test_the_published_summary_is_what_the_receipts_produce_today():
    if not PUBLISHED.is_file() or not (summary.EV/'scale08_verdict_01/verdict.json').is_file():
        pytest.skip('contract 08 receipts not on this machine')
    a, b = json.loads(PUBLISHED.read_text()), summary.build()
    a.pop('prepared_utc'); b.pop('prepared_utc')
    assert a == b


def test_the_published_summary_is_self_consistent_without_receipts():
    if not PUBLISHED.is_file():
        pytest.skip('summary not generated yet')
    s = json.loads(PUBLISHED.read_text())
    assert s['simulation_only'] is True and s['export_gate']['status'] == 'FAIL'
    assert sum(s['budget']['by_step'].values()) == s['budget']['used_native'] <= s['budget']['cap']['native']
    pc = s['project_cumulative']
    assert pc['native'] == pc['before_contract_08']['native'] + s['budget']['used_native']
    assert s['qualification']['overall'] == 'FAIL' and s['pre_registered_verdict']['P1'] == 'NOT_MET'
