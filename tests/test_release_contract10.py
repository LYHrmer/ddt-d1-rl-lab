"""Release refusals use only fake receipts and byte copies, never a training process."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools12'))
import release_contract10 as gate  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def setup_plan(root):
    original = json.loads((ROOT/'training/configs/contract10_supervisor_draft.json').read_text())
    plan = copy.deepcopy(original)
    plan['status'] = 'AUTHORISED_PENDING_FREEZE'
    plan['proposed_budget']['authorised'] = True
    names = set(original['frozen_inputs']) | set(gate.REQUIRED_INPUTS)
    for name in names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT/name
        path.write_bytes(source.read_bytes() if source.is_file() else b'fake frozen source\n')
    plan['frozen_inputs'] = {name: gate.sha(root/name) for name in sorted(names)}
    plan['supervisor_sha256'] = gate.sha(ROOT/'tools10/supervise_scale10.py')
    plan_path = write_json(root/'plan.json', plan)
    auth = {'format': gate.AUTHORISATION_FORMAT, 'contract_id': 10, 'authorised': True,
            'cap': gate.CAP, 'qualification_native': 18000,
            'start_min_available_ram_bytes': 7*1024**3,
            'retries_allowed': False, 'resume_allowed': False, 'checkpoint_selection': 'final_only'}
    auth_path = write_json(root/'evidence/career_contract_10/budget_authorisation_02.json', auth)
    static = {'status': 'PASS', 'physics_steps_executed': 0, 'training_started': False,
              'plan_sha256': gate.sha(plan_path), 'budget_authorisation_sha256': gate.sha(auth_path),
              'tool_sha256': plan['frozen_inputs'][gate.STATIC_TOOL],
              'frozen_inputs': plan['frozen_inputs']}
    static_path = write_json(root/'evidence/career_contract_10/static_fake/report.json', static)
    return plan_path, auth_path, static_path


def test_release_is_a_new_frozen_receipt_not_a_training_invocation(tmp_path):
    plan, auth, static = setup_plan(tmp_path)
    receipt = gate.build(plan, auth, static, root=tmp_path)
    assert receipt['status'] == 'FROZEN_BEFORE_TRAINING'
    assert receipt['physics_steps_executed_by_this_receipt'] == 0
    assert receipt['budget_authorisation']['sha256'] == gate.sha(auth)
    assert set(gate.REQUIRED_INPUTS) <= set(receipt['frozen_inputs'])
    assert receipt['plan_sha256'] == gate.sha(plan)
    assert not (tmp_path/receipt['plans']['scale']['output']).exists()


@pytest.mark.parametrize('field,value', [('authorised', False), ('contract_id', 9),
                                       ('cap', {'transitions': 8421377, 'native': 84231760}),
                                       ('qualification_native', 18001),
                                       ('start_min_available_ram_bytes', 6*1024**3),
                                       ('retries_allowed', True), ('resume_allowed', True),
                                       ('checkpoint_selection', 'best')])
def test_unapproved_or_changed_budget_cannot_be_frozen(tmp_path, field, value):
    paths = setup_plan(tmp_path)
    auth = json.loads(paths[1].read_text()); auth[field] = value
    write_json(paths[1], auth)
    with pytest.raises(SystemExit, match='authorisation'):
        gate.build(*paths, root=tmp_path)


@pytest.mark.parametrize('change', ['draft', 'unapproved', 'ram', 'steps', 'seed', 'new_output',
                                   'missing_gate', 'changed_source', 'bad_static', 'physics',
                                   'old_plan_sha', 'static_inputs', 'old_auth_sha', 'old_checker_sha'])
def test_ineligible_plan_or_stale_static_report_is_rejected(tmp_path, change):
    paths = setup_plan(tmp_path)
    plan = json.loads(paths[0].read_text()); static = json.loads(paths[2].read_text())
    if change == 'draft': plan['status'] = 'DRAFT_NOT_AUTHORISED'
    elif change == 'unapproved': plan['proposed_budget']['authorised'] = False
    elif change == 'ram': plan['resources']['start_min_available_ram_bytes'] -= 1
    elif change == 'steps': plan['plans']['scale']['steps'] -= 4096
    elif change == 'seed': plan['seed'] = 102
    elif change == 'new_output': plan['plans']['scale']['output'] += '_retry'
    elif change == 'missing_gate': del plan['frozen_inputs']['tools12/verify_contract10.py']
    elif change == 'changed_source': (tmp_path/plan['entry']).write_text('changed')
    elif change == 'bad_static': static['status'] = 'FAIL'
    elif change == 'physics': static['physics_steps_executed'] = 1
    elif change == 'old_plan_sha': static['plan_sha256'] = '0'*64
    elif change == 'static_inputs': static['frozen_inputs'] = {}
    elif change == 'old_auth_sha': static['budget_authorisation_sha256'] = '0'*64
    elif change == 'old_checker_sha': static['tool_sha256'] = '0'*64
    if plan != json.loads(paths[0].read_text()):
        write_json(paths[0], plan); static['plan_sha256'] = gate.sha(paths[0])
        static['frozen_inputs'] = plan['frozen_inputs']
    write_json(paths[2], static)
    with pytest.raises(SystemExit):
        gate.build(*paths, root=tmp_path)


@pytest.mark.parametrize('claimed', ['scale10_preflight_01', 'scale10_train_01',
                                   'scale10_preflight_01_supervisor', 'scale10_train_01_supervisor'])
def test_a_previous_attempt_blocks_a_new_release(tmp_path, claimed):
    paths = setup_plan(tmp_path)
    (tmp_path/'evidence/career_contract_10'/claimed).mkdir()
    with pytest.raises(SystemExit, match='attempt'):
        gate.build(*paths, root=tmp_path)


def test_write_is_exclusive_and_confined_to_the_single_release_path(tmp_path):
    paths = setup_plan(tmp_path)
    output = tmp_path/gate.RELEASE_PATH
    gate.freeze(*paths, output, root=tmp_path)
    frozen_bytes = output.read_bytes()
    with pytest.raises(SystemExit, match='already exists'):
        gate.freeze(*paths, output, root=tmp_path)
    assert output.read_bytes() == frozen_bytes
    with pytest.raises(SystemExit, match='release path'):
        gate.freeze(*paths, tmp_path/'retry.json', root=tmp_path)


def test_traversal_in_frozen_inputs_cannot_read_outside_root(tmp_path):
    paths = setup_plan(tmp_path)
    plan = json.loads(paths[0].read_text())
    plan['frozen_inputs']['../elsewhere'] = '0'*64
    write_json(paths[0], plan)
    with pytest.raises(SystemExit, match='frozen input'):
        gate.build(*paths, root=tmp_path)
