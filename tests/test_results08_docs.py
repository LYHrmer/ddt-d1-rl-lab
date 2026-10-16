"""The numbers the contract 08 documents quote must be the numbers in the receipts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/'reports/contract08_summary.json'


@pytest.fixture
def summary():
    if not SUMMARY.is_file():
        pytest.skip('contract 08 summary not generated')
    return json.loads(SUMMARY.read_text())


def test_the_front_page_quotes_the_published_summary(summary):
    text = (ROOT/'README.md').read_text()
    pc = summary['project_cumulative']
    assert f"{pc['native']:,}" in text and f"{pc['transitions']:,}" in text
    assert f"{100*summary['training']['late_return_change']['relative_change']:+.1f} %" in text
    turn = summary['post_hoc_not_pre_registered']['what_each_task_did']['scale_8x']['turn_stop']['after_cut']
    assert f"{turn['path_over_euclidean']:.2f}" in text
    assert '六次自训都没有通过' in text and '训练量不是瓶颈' in text
    replay = ROOT/'evidence/career_contract_08/scale08_training_replay_01/report.json'
    if replay.is_file():
        r = json.loads(replay.read_text())
        assert f"{r['ppo_timeline'][-1]['train/std']:.2f}" in text and f"{r['episodes']:,}" in text


def test_the_results_document_quotes_the_published_summary(summary):
    text = (ROOT/'docs/career_execution_results_08.md').read_text()
    for key, value in summary['pre_registered_verdict'].items():
        assert f'**{value}**' in text or value == 'REPORTED', key
    b = summary['budget']
    assert f"{b['used_native']:,}" in text and f"{b['cap']['native']:,}" in text
    diag = summary['export_gate']['diagnosis']
    assert f"{diag['scale_8x']['difference_over_eps_times_peak']:.2f}" in text
    assert f"{diag['03d_1x']['difference_over_eps_times_peak']:.2f}" in text
    assert '事后修补' in text and '只可能让结论更差' in text


def test_the_summary_says_the_deployment_chain_is_not_verified(summary):
    assert summary['export_gate']['status'] == 'FAIL'
    assert 'not verified' in summary['export_gate']['deployment_chain']
    assert 'deployment chain not verified' in summary['limits']


def test_the_training_replay_section_quotes_its_receipt():
    receipt = ROOT/'evidence/career_contract_08/scale08_training_replay_01/report.json'
    if not receipt.is_file():
        pytest.skip('training replay receipt not on this machine')
    r = json.loads(receipt.read_text())
    text = (ROOT/'docs/career_execution_results_08.md').read_text()
    assert r['replay_verified_against_environment']['passed'] is True
    assert f"{r['replay_verified_against_environment']['episodes_checked']} 个回合**全部一致**" in text
    assert f"{r['ppo_timeline'][-1]['train/std']:.2f}" in text
    late = sum(b['early_ends_by_segment'].get('drive_forward', 0) for b in r['timeline'][12:])
    assert f'向前行驶段内提前终止 {late} 次' in text
    assert all(p['identical_prefix'] for p in r['baseline_prefix_check'].values())
