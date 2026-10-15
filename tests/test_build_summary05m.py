"""Tests for tools05/build_summary05m.py, the published summary of contract 05M.

The summary is the only place a reader of the repository sees 05M, so the parts that
could make it wrong are pinned: the budget must add up and stay within its cap, every
run must have completed, the fidelity gate must have passed, and the project total
must be the previous total plus what 05M used. No model, no physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools05'))

import build_summary05m as summary  # noqa: E402

PUBLISHED = ROOT/'reports/contract05m_summary.json'


def run(native, status='COMPLETE'):
    return {'actual_counts': {'native_steps': native}, 'status': status}


def test_budget_adds_up_and_reports_what_is_left():
    out = summary.budget({'smoke': run(20), 'A': run(6000), 'B': run(6000), 'C': run(6000)}, 18020)
    assert out['used_native_total'] == 18020 and out['unused_native'] == 0 and out['transitions'] == 0
    assert out['used_native_by_run'] == {'smoke': 20, 'A': 6000, 'B': 6000, 'C': 6000}


def test_budget_over_the_cap_is_refused():
    with pytest.raises(SystemExit, match='more than'):
        summary.budget({'A': run(6000), 'B': run(6000), 'C': run(6021)}, 18020)


def test_budget_exactly_at_the_cap_is_allowed():
    assert summary.budget({'A': run(18020)}, 18020)['unused_native'] == 0


def test_a_run_that_did_not_complete_is_refused():
    with pytest.raises(SystemExit, match='did not complete'):
        summary.budget({'A': run(6000), 'B': run(4000, 'CRASHED')}, 18020)


def test_project_cumulative_is_the_previous_total_plus_what_was_used():
    out = summary.project_cumulative({'transitions': 5242880, 'native': 52914120}, 18020)
    assert out['native'] == 52932140 and out['transitions'] == 5242880
    assert out['before_contract_05m'] == {'transitions': 5242880, 'native': 52914120}
    assert out['added_by_contract_05m'] == {'transitions': 0, 'native': 18020}


def test_readings_take_each_field_from_the_matching_analysis_key():
    analysis = {'entry_speed_m_s': .1, 'decay_time_constant_s': {'ticks_10_to_50': .6},
                'final_euclidean_displacement_m': .05, 'ticks_until_speed_below_0p02_m_s': 23,
                'first_order_stop': {'measured_path_m': .055, 'predicted_over_path': 1.6}}
    assert summary.readings(analysis) == {'entry_speed_m_s': .1, 'tau_s_ticks_10_to_50': .6,
                                          'displacement_200_ticks_euclidean_m': .05, 'path_200_ticks_m': .055,
                                          'first_order_predicted_over_path': 1.6, 'ticks_until_speed_below_0p02': 23}


def test_the_corrections_name_each_overturned_published_claim():
    text = json.dumps(summary.CORRECTIONS)
    assert 'already stopped' in text and 'entry speed times decay constant' in text and 'rad per m/s' in text
    assert all(c['status'] in ('overturned', 'demoted to an empirical fit on the policy arm') for c in summary.CORRECTIONS)


def test_the_tool_imports_no_simulator():
    source = Path(summary.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


@needs_workstation_evidence
def test_a_failed_fidelity_gate_is_refused(tmp_path, monkeypatch):
    """Copy the real receipts, break only run A's fidelity, and the build must stop."""
    import shutil
    if not (summary.EV/'probe05_runA_01/report.json').is_file():
        pytest.skip('05M receipts not on this machine')
    ev = tmp_path/'ev'; shutil.copytree(summary.EV, ev, ignore=shutil.ignore_patterns('*.npz'))
    report = json.loads((ev/'probe05_runA_01/report.json').read_text()); report['fidelity']['passed'] = False
    (ev/'probe05_runA_01/report.json').write_text(json.dumps(report))
    with pytest.raises(SystemExit, match='fidelity'):
        summary.build(ev=ev)


@needs_workstation_evidence
def test_the_published_summary_is_what_the_receipts_produce_today():
    if not PUBLISHED.is_file() or not (summary.EV/'probe05_runA_01/report.json').is_file():
        pytest.skip('summary or receipts not on this machine')
    published, fresh = json.loads(PUBLISHED.read_text()), summary.build()
    for key in ('prepared_utc',):
        published.pop(key); fresh.pop(key)
    assert published == fresh


def test_the_published_summary_is_self_consistent_without_receipts():
    """Runs in a clean clone: the file in reports/ is published and must add up on its own."""
    if not PUBLISHED.is_file():
        pytest.skip('summary not generated yet')
    s = json.loads(PUBLISHED.read_text())
    assert s['simulation_only'] is True and s['real_robot_validation'] is False
    assert sum(s['budget']['used_native_by_run'].values()) == s['budget']['used_native_total'] <= s['budget']['cap_native']
    pc = s['project_cumulative']
    assert pc['native'] == pc['before_contract_05m']['native'] + s['budget']['used_native_total']
    assert s['pre_registered_verdict']['P2'] == 'REFUTED' and 'passive damping' in s['interpretation_limit']
