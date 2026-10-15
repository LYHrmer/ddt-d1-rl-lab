"""Tests for tools05/release_probe05.py, the receipt that freezes contract 05M.

The refusals are the point: a release must not be written from a static report
that failed, that ran physics, or that checked a different probe than the one
that will run. Those paths read nothing under evidence/ and run anywhere. Only the
full build against the real frozen inputs needs the workstation. No physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools05'))

import probe_stop_factorisation05 as probe  # noqa: E402
import release_probe05 as release  # noqa: E402


def static_report(tmp_path, **overrides):
    report = {'status': 'PASS', 'native_integration_steps_executed': 0,
              'probe_sha256': probe.sha(probe.__file__), 'checks': [{}]*14}
    report.update(overrides)
    path = tmp_path/'static.json'
    path.write_text(json.dumps(report))
    return path


# --- the refusals --------------------------------------------------------------
def test_a_failed_static_report_cannot_be_frozen(tmp_path):
    with pytest.raises(SystemExit, match='not PASS'):
        release.build_receipt(static_report(tmp_path, status='FAIL'))


def test_a_static_report_that_ran_physics_cannot_be_frozen(tmp_path):
    with pytest.raises(SystemExit, match='ran physics'):
        release.build_receipt(static_report(tmp_path, native_integration_steps_executed=10))


def test_a_static_report_without_the_physics_counter_cannot_be_frozen(tmp_path):
    path = static_report(tmp_path)
    report = json.loads(path.read_text()); del report['native_integration_steps_executed']
    path.write_text(json.dumps(report))
    with pytest.raises(SystemExit, match='ran physics'):
        release.build_receipt(path)


def test_a_static_report_for_a_different_probe_cannot_be_frozen(tmp_path):
    with pytest.raises(SystemExit, match='different probe'):
        release.build_receipt(static_report(tmp_path, probe_sha256='0'*64))


def test_a_status_other_than_pass_is_refused_even_if_it_looks_close(tmp_path):
    for status in ('PASSED', 'pass', '', None):
        with pytest.raises(SystemExit):
            release.build_receipt(static_report(tmp_path, status=status))


# --- what gets frozen ----------------------------------------------------------
def test_predictions_are_taken_from_the_probe_constants_not_retyped():
    by_id = {p['id']: p for p in release.predictions()}
    assert by_id['P1']['max_abs_difference'] == probe.P1_TOLERANCE
    assert (by_id['P2']['supported_at_or_below_s'], by_id['P2']['refuted_at_or_above_s']) == (.29, .41)
    assert by_id['P2']['baseline_policy_arm_s'] == probe.TAU_POLICY_ARM_S
    assert by_id['P3']['threshold_m'] == .05
    assert tuple(by_id['P4']['range']) == probe.P4_RATIO_RANGE
    assert tuple(by_id['P5']['range']) == probe.P5_RATIO_RANGE
    assert list(by_id) == ['P1', 'P2', 'P3', 'P4', 'P5']


def test_the_p4_precondition_is_frozen_into_the_predictions():
    """Revision 1 exists to freeze this; a receipt that dropped it would freeze nothing."""
    p4 = {p['id']: p for p in release.predictions()}['P4']
    assert tuple(p4['valid_only_if_entry_speed_ratio_B_over_A_is_outside']) == probe.P4_ENTRY_RATIO_EXCLUDED
    assert p4['otherwise'] == 'NOT_EVALUABLE'
    assert tuple(p4['range']) == probe.P4_RATIO_RANGE


def test_release_refuses_to_overwrite_an_existing_receipt(tmp_path, monkeypatch, capsys):
    target = tmp_path/'release.json'; target.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['release', '--static-report', str(tmp_path/'s.json'),
                                      '--output', str(target)])
    with pytest.raises(SystemExit, match='Refusing to overwrite'):
        release.main()
    assert target.read_text() == '{}'


@needs_workstation_evidence
def test_the_full_receipt_freezes_every_input_the_probe_will_demand(tmp_path):
    reports = sorted((ROOT/'evidence/career_contract_05').glob('probe05_static_*/report.json'))
    if not reports:
        pytest.skip('static report not on this machine')
    path = reports[-1]      # the newest; a stale one must fail here, as the release itself would refuse it
    receipt = release.build_receipt(path)
    assert receipt['status'] == 'FROZEN_BEFORE_MEASUREMENT'
    assert receipt['frozen_inputs'] == probe.FROZEN_INPUTS
    assert receipt['tool_sha256'] == probe.sha(probe.__file__)
    assert receipt['contract_sha256'] == probe.sha(release.CONTRACT)
    assert receipt['budget']['cap_native'] == 18020 and receipt['budget']['transitions'] == 0
    assert receipt['budget']['retries_allowed'] is False
    assert sum(r['planned_native'] for r in receipt['runs'].values()) == 18020
    assert receipt['physics_steps_executed_by_this_receipt'] == 0
    # the receipt must satisfy the probe's own gate
    out = tmp_path/'release.json'; out.write_text(json.dumps(receipt))
    assert probe.verify_release(out)['status'] == 'FROZEN_BEFORE_MEASUREMENT'


def test_no_outcome_authorises_training():
    source = Path(release.__file__).read_text()
    assert "'not_authorised_by_any_outcome': ['training'" in source


def test_the_frozen_inputs_are_actually_checked_when_building_the_receipt(tmp_path):
    """Pointing the root at an empty directory must fail on the first missing frozen input.

    Against the real tree every input matches, so deleting the check would change nothing
    there; an empty root is the only place where checking and not checking differ.
    """
    contract = tmp_path/'contract.md'; contract.write_text('x')
    with pytest.raises(ValueError, match='missing'):
        release.build_receipt(static_report(tmp_path), contract=contract, root=tmp_path)


def test_the_receipt_file_is_opened_exclusively_even_if_the_exists_check_is_skipped(tmp_path, monkeypatch):
    """The exists() test is the first guard and open('x') the second; this exercises the second."""
    target = tmp_path/'release.json'
    monkeypatch.setattr(release, 'build_receipt', lambda *a, **k: {'status': 'S', 'frozen_inputs': {},
                                                                  'physics_steps_executed_by_this_receipt': 0})
    monkeypatch.setattr(sys, 'argv', ['release', '--static-report', 'unused', '--output', str(target)])
    original = Path.exists
    monkeypatch.setattr(Path, 'exists', lambda self: False if self == target else original(self))
    target.write_text('original')
    with pytest.raises(FileExistsError):
        release.main()
    assert target.read_text() == 'original'


def test_the_receipt_declares_that_retries_and_seed_changes_are_not_allowed():
    """Declarations, not behaviour: nothing reads these flags. What binds is the probe's SEED constant."""
    source = Path(release.__file__).read_text()
    assert "'retries_allowed': False" in source and "'seed_change_allowed': False" in source
    assert "'seed': probe.SEED" in source
    assert probe.SEED == 7001


def test_superseding_a_receipt_that_does_not_exist_is_refused(tmp_path, monkeypatch):
    """The frozen-input check runs first and needs evidence/, which a clean clone lacks.

    It is switched off here so this test isolates the one thing it names, and runs anywhere.
    """
    monkeypatch.setattr(probe, 'check_frozen_inputs', lambda *args, **kwargs: None)
    with pytest.raises(SystemExit, match='does not exist'):
        release.build_receipt(static_report(tmp_path), supersedes=tmp_path/'absent.json')


def test_superseding_records_the_old_receipt_and_its_digest(tmp_path):
    """Uses an empty root so the build stops at the frozen-input check, after nothing is written."""
    old = tmp_path/'old.json'; old.write_text('{}')
    contract = tmp_path/'contract.md'; contract.write_text('x')
    with pytest.raises(ValueError, match='missing'):
        release.build_receipt(static_report(tmp_path), contract=contract, root=tmp_path, supersedes=old)


@needs_workstation_evidence
def test_the_real_receipt_names_what_it_supersedes(tmp_path):
    reports = sorted((ROOT/'evidence/career_contract_05').glob('probe05_static_*/report.json'))
    old = ROOT/'evidence/career_contract_05/probe05_release_01.json'
    if not reports or not old.is_file():
        pytest.skip('receipts not on this machine')
    receipt = release.build_receipt(reports[-1], supersedes=old)
    assert receipt['supersedes']['path'].endswith('probe05_release_01.json')
    assert receipt['supersedes']['sha256'] == probe.sha(old)
    assert receipt['supersedes']['status_of_old_receipt'] == 'SUPERSEDED_BEFORE_ANY_PHYSICS_RAN'
    assert release.build_receipt(reports[-1])['supersedes'] is None
