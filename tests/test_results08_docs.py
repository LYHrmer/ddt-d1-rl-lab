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
