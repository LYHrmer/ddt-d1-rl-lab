"""Tests for contract 09: its wrapper, its five-line entry and its release. No physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools08', 'tools09'):
    sys.path.insert(0, str(ROOT/sub))

import check_scale08_static as static08  # noqa: E402
import check_scale09_static as static09  # noqa: E402
import release_scale09 as release  # noqa: E402
import run_scale08 as base  # noqa: E402
import run_scale09 as launcher  # noqa: E402


# --- the wrapper -----------------------------------------------------------------
def test_configure_points_the_contract_08_launcher_at_contract_09_and_can_be_undone():
    before_entry, before_plans = base.ENTRY, base.PLANS
    previous = launcher.configure()
    try:
        assert base.ENTRY == launcher.ENTRY and base.PLANS is launcher.PLANS
        cmd = base.command('scale', Path('/out'), 'python3')
        assert cmd[2].endswith('tools09/train_d1_ppo09.py') and cmd[cmd.index('--steps')+1] == '8388608'
    finally:
        base.ENTRY, base.PLANS = previous['ENTRY'], previous['PLANS']
    assert (base.ENTRY, base.PLANS) == (before_entry, before_plans)


def test_only_the_first_rollout_is_compared_and_outputs_are_contract_09():
    assert all(p['compare_rollouts'] == 1 for p in launcher.PLANS.values())
    assert all(p['output'].startswith('evidence/career_contract_09/') for p in launcher.PLANS.values())
    assert launcher.PLANS['scale']['steps'] == 8388608 and launcher.PLANS['scale']['max_wall'] == 9000


# --- the entry differs from the frozen trainer in exactly five lines ---------------
def test_the_new_entry_is_the_frozen_trainer_with_five_lines_changed():
    changes = static08.line_changes((ROOT/'tools/train_d1_ppo.py').read_text(), (ROOT/launcher.ENTRY).read_text())
    assert changes == static09.EXPECTED_CHANGES and len(changes) == 5


def test_the_two_extra_changes_are_exactly_the_entropy_coefficient():
    extra = {k: v for k, v in static09.EXPECTED_CHANGES.items() if k not in static08.EXPECTED_CHANGES}
    for old, new in extra.values():
        assert old.replace('.005', '0.0') == new and 'ent_coef' in old


# --- the release ---------------------------------------------------------------------
def static(tmp_path, **overrides):
    report = {'status': 'PASS', 'native_integration_steps_executed': 0, 'training_started': False,
              'new_entry_sha256': release.sha(ROOT/launcher.ENTRY), 'wrapper_sha256': release.sha(ROOT/'tools09/run_scale09.py'),
              'launcher_sha256': release.sha(ROOT/'tools08/run_scale08.py'), 'checks': [{}]*6}
    report.update(overrides)
    path = tmp_path/'s.json'; path.write_text(json.dumps(report))
    return path


@pytest.mark.parametrize('field, value, message', [
    ('status', 'FAIL', 'not PASS'), ('native_integration_steps_executed', 5, 'ran physics'),
    ('training_started', True, 'ran physics'), ('new_entry_sha256', '0'*64, 'different tools09/train_d1_ppo09.py'),
    ('wrapper_sha256', '0'*64, 'different tools09/run_scale09.py'), ('launcher_sha256', '0'*64, 'different tools08/run_scale08.py')])
def test_each_refusal(tmp_path, monkeypatch, field, value, message):
    monkeypatch.setattr(release, 'FROZEN_INPUTS', tuple(p for p in release.FROZEN_INPUTS if not p.startswith('evidence/')))
    with pytest.raises(SystemExit, match=message):
        release.build(static(tmp_path, **{field: value}))


def test_a_missing_frozen_input_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='missing'):
        release.build(static(tmp_path), root=tmp_path)


def test_predictions_budget_and_interpretation_match_the_contract():
    by_id = {p['id']: p for p in release.PREDICTIONS}
    assert by_id['Q1']['threshold'] == .45 and by_id['Q1']['baseline_08'] == .676
    assert by_id['Q2']['threshold'] == 15 and by_id['Q2']['baseline_08'] == 43
    assert by_id['Q3']['threshold'] == 2
    b = release.BUDGET
    assert b['cap']['native'] == b['preflight']['native'] + b['scale_training']['native'] + b['python_qualification_native']
    assert b['cap']['transitions'] == b['preflight']['transitions'] + b['scale_training']['transitions']
    assert set(release.INTERPRETATION) == {'Q1 met, Q2 met', 'Q1 met, Q2 not met', 'Q1 not met'}
    text = (ROOT/release.CONTRACT).read_text()
    for fragment in ('0.45', '≤ 15 次', '0.676', '43', '84,231,760', '8,421,376', '5 行', '第 1 个 rollout'):
        assert fragment in text, fragment


def test_the_frozen_set_includes_the_reused_launcher_its_tests_and_this_file():
    for path in ('tools08/run_scale08.py', 'tests/test_run_scale08.py', 'tests/test_scale09.py', launcher.ENTRY):
        assert path in release.FROZEN_INPUTS


@needs_workstation_evidence
def test_the_reused_launcher_is_still_the_one_frozen_into_contract_08():
    receipt = ROOT/'evidence/career_contract_08/scale08_release_01.json'
    if not receipt.is_file():
        pytest.skip('contract 08 release not on this machine')
    assert json.loads(receipt.read_text())['launcher_sha256'] == release.sha(ROOT/'tools08/run_scale08.py')
