"""Tests for tools08/release_scale08.py: it must refuse to freeze an unchecked contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import release_scale08 as release  # noqa: E402
import run_scale08 as launcher  # noqa: E402


def static(tmp_path, **overrides):
    report = {'status': 'PASS', 'native_integration_steps_executed': 0, 'training_started': False,
              'new_entry_sha256': release.sha(ROOT/launcher.ENTRY), 'launcher_sha256': release.sha(launcher.__file__),
              'checks': [{}]*7}
    report.update(overrides)
    path = tmp_path/'static.json'; path.write_text(json.dumps(report))
    return path


@pytest.mark.parametrize('field, value, message', [
    ('status', 'FAIL', 'not PASS'), ('native_integration_steps_executed', 10, 'ran physics'),
    ('training_started', True, 'ran physics'), ('new_entry_sha256', '0'*64, 'different new entry'),
    ('launcher_sha256', '0'*64, 'different launcher')])
def test_each_refusal(tmp_path, field, value, message):
    with pytest.raises(SystemExit, match=message):
        release.build(static(tmp_path, **{field: value}))


def test_a_missing_frozen_input_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='missing'):
        release.build(static(tmp_path), root=tmp_path)


def test_the_predictions_and_budget_match_the_contract():
    by_id = {p['id']: p for p in release.PREDICTIONS}
    assert by_id['P1']['threshold'] == 2 and by_id['P1']['baseline_03d_1x'] == 1
    assert by_id['P2']['threshold'] == .05 and by_id['P3']['threshold'] == .08
    b = release.BUDGET
    assert b['scale_training'] == {'transitions': 8388608, 'native': 83886080, 'max_wall_s': 9000}
    assert b['cap']['native'] == 327680*2 + 83886080 + 18000 and b['cap']['transitions'] == 32768*2 + 8388608
    assert b['retries_allowed'] is False and b['checkpoint_selection'] == 'final_only'


def test_the_launcher_frozen_set_includes_the_new_entry_and_the_frozen_ledger():
    assert launcher.ENTRY in release.FROZEN_INPUTS and launcher.FROZEN_LEDGER in release.FROZEN_INPUTS
    assert 'tools/train_d1_ppo.py' in release.FROZEN_INPUTS


def test_the_contract_text_states_the_same_numbers():
    text = (ROOT/release.CONTRACT).read_text()
    for fragment in ('8,388,608', '83,886,080', '84,559,440', '8,454,144', '9,000 s', '1e-9', '≥ 2', '< 0.05 m', '< 0.08 rad/s'):
        assert fragment in text, fragment
