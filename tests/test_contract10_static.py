"""Authorisation and plan checks use disposable text fixtures, never training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools12'))
import check_contract10_static as checker  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    return path


def fixture(root):
    original = {'format': 'd1-contract10-budget-authorisation-v1',
                'created_utc': '2026-10-16T08:16:00.862547+00:00',
                'authorised_by': 'repository owner in this conversation',
                'approval_verbatim': '批准合同 10 新预算，内存达标后执行',
                'contract': 'docs/career_execution_contract_10.md',
                'cap': {'transitions': 8421376, 'native': 84231760},
                'start_min_available_ram_bytes': 7516192768, 'single_attempt': True,
                'resume_allowed': False, 'source': 'request_user_input_async affirmative reply'}
    original_path = write(root/'evidence/career_contract_10/budget_authorisation_01.json', original)
    auth = {'format': 'd1-contract10-budget-authorisation-v1', 'contract_id': 10, 'authorised': True,
            'contract': 'docs/career_execution_contract_10.md',
            'cap': {'transitions': 8421376, 'native': 84231760}, 'qualification_native': 18000,
            'start_min_available_ram_bytes': 7516192768, 'single_attempt': True,
            'retries_allowed': False, 'resume_allowed': False, 'checkpoint_selection': 'final_only',
            'approval_verbatim': original['approval_verbatim'],
            'source_record': {'path': 'evidence/career_contract_10/budget_authorisation_01.json',
                              'sha256': sha(original_path)}}
    auth_path = write(root/'evidence/career_contract_10/budget_authorisation_02.json', auth)
    plans = {
        'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': 8,
                      'frozen_ledger': 'evidence/career_contract_09/scale09_preflight_01/ledger.jsonl',
                      'output': 'evidence/career_contract_10/scale10_preflight_01'},
        'scale': {'steps': 8388608, 'max_wall': 9000, 'compare_rollouts': 1173,
                  'frozen_ledger': 'evidence/career_contract_09/scale09_train_01/ledger.jsonl',
                  'output': 'evidence/career_contract_10/scale10_train_01'}}
    names = ['tools09/train_d1_ppo09.py', 'training/configs/pilot03d.json', 'tools08/run_scale08.py',
             'tools10/supervise_scale10.py', 'tests/test_supervise_scale10.py',
             'tools12/check_contract10_static.py', 'tests/test_contract10_static.py',
             'tools12/release_contract10.py', 'tests/test_release_contract10.py',
             'tools12/verify_contract10.py', 'tests/test_verify_contract10.py',
             'tools10/export_single_sample10.py', 'tests/test_export_single_sample10.py',
             'tools/evaluate_d1_ppo.py', 'tools08/replay_training_episodes08.py',
             'tools08/verify_scale08.py', 'tools09/verify_scale09.py',
             'docs/career_execution_contract_10.md', *(p['frozen_ledger'] for p in plans.values())]
    names = sorted(set(names) | set(checker.REQUIRED_INPUTS))
    for name in names:
        if (root/name).exists():
            continue
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT/name
        target.write_bytes(source.read_bytes() if source.is_file() and not name.startswith('evidence/')
                           else b'disposable text fixture, never executed\n')
    plan = {'format': 'd1-contract10-supervisor-release-v1', 'contract_id': 10,
            'status': 'AUTHORISED_PENDING_FREEZE', 'entry': 'tools09/train_d1_ppo09.py',
            'config': 'training/configs/pilot03d.json', 'contract': 'docs/career_execution_contract_10.md',
            'seed': 101, 'envs': 8, 'plans': plans,
            'project_cumulative_before': {'transitions': 18504192, 'native': 185562110},
            'resources': {'start_min_available_ram_bytes': 7516192768, 'poll_s': .5,
                          'interrupt_grace_s': 20., 'terminate_grace_s': 5., 'kill_grace_s': 5.,
                          'process_wall_grace_s': 180.},
            'proposed_budget': {'transitions': 8421376, 'native': 84231760,
                                'qualification_native': 18000, 'authorised': True},
            'frozen_inputs': {name: sha(root/name) for name in names},
            'supervisor_sha256': sha(root/'tools10/supervise_scale10.py')}
    plan_path = write(root/'training/configs/contract10_supervisor_draft.json', plan)
    return plan_path, auth_path, plan, auth


def test_authorised_fixed_plan_emits_exact_bound_zero_physics_receipt(tmp_path):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    out = tmp_path/'static'
    receipt = checker.check(plan_path, auth_path, out, root=tmp_path)
    assert receipt['status'] == 'PASS'
    assert receipt['physics_steps_executed'] == 0 and receipt['training_started'] is False
    assert receipt['plan_sha256'] == sha(plan_path)
    assert receipt['budget_authorisation_sha256'] == sha(auth_path)
    assert receipt['frozen_inputs'] == plan['frozen_inputs']
    assert receipt['tool_sha256'] == sha(ROOT/'tools12/check_contract10_static.py')
    assert json.loads((out/'report.json').read_text()) == receipt
    assert not (tmp_path/plan['plans']['scale']['output']).exists()


def test_an_unapproved_budget_gets_a_failed_zero_physics_receipt(tmp_path):
    plan_path, auth_path, _, auth = fixture(tmp_path)
    auth['authorised'] = False
    write(auth_path, auth)
    receipt = checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)
    assert receipt['status'] == 'FAIL'
    assert receipt['physics_steps_executed'] == 0 and receipt['training_started'] is False


def test_another_seed_cannot_pass_the_fixed_scientific_plan(tmp_path):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    plan['seed'] = 102
    write(plan_path, plan)
    assert checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)['status'] == 'FAIL'


def test_a_traversal_cannot_be_hidden_in_the_frozen_input_map(tmp_path):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    outside = write(tmp_path.parent/'outside-static-fixture.json', {'private': 'must not be read'})
    plan['frozen_inputs']['../outside-static-fixture.json'] = sha(outside)
    write(plan_path, plan)
    assert checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)['status'] == 'FAIL'


@pytest.mark.parametrize('field,value', [
    ('format', 'unapproved-format'), ('contract_id', 9), ('contract_id', 10.0),
    ('contract', 'docs/career_execution_contract_09.md'),
    ('cap', {'transitions': 8421377, 'native': 84231760}),
    ('cap', {'transitions': 8421376.0, 'native': 84231760}),
    ('qualification_native', 18001), ('start_min_available_ram_bytes', 6*1024**3),
    ('single_attempt', False), ('retries_allowed', True), ('resume_allowed', True),
    ('checkpoint_selection', 'best'), ('approval_verbatim', 'approve anything'),
    ('source_record', {'path': '../approval.json', 'sha256': '0'*64}),
    ('source_record', {'path': 'evidence/career_contract_10/budget_authorisation_01.json', 'sha256': '0'*64}),
    ('source_record', []), ('authorised_by', 'someone else'), ('source', 'invented approval'),
])
def test_changed_or_untraceable_authorisation_is_rejected(tmp_path, field, value):
    plan_path, auth_path, _, auth = fixture(tmp_path)
    auth[field] = value; write(auth_path, auth)
    receipt = checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)
    assert receipt['status'] == 'FAIL' and receipt['error']


def test_rewriting_both_original_approval_and_its_claimed_hash_is_rejected(tmp_path):
    plan_path, auth_path, _, auth = fixture(tmp_path)
    original_path = tmp_path/auth['source_record']['path']
    original = json.loads(original_path.read_text()); original['approval_verbatim'] = 'rewritten approval'
    write(original_path, original)
    auth['approval_verbatim'] = original['approval_verbatim']; auth['source_record']['sha256'] = sha(original_path)
    write(auth_path, auth)
    assert checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)['status'] == 'FAIL'


@pytest.mark.parametrize('path,value', [
    ('format', 'other'), ('contract_id', 9), ('status', 'DRAFT_NOT_AUTHORISED'),
    ('entry', 'tools/train_d1_ppo.py'), ('config', 'training/configs/pilot03c.json'),
    ('contract', 'docs/career_execution_contract_09.md'), ('seed', 101.0), ('envs', 4),
    ('plans.preflight.steps', 4096), ('plans.preflight.max_wall', 601),
    ('plans.preflight.compare_rollouts', 1), ('plans.scale.steps', 4807168),
    ('plans.scale.max_wall', 9001), ('plans.scale.compare_rollouts', 1174),
    ('plans.scale.output', 'evidence/career_contract_10/scale10_train_02'),
    ('plans.scale.frozen_ledger', 'evidence/career_contract_09/scale09_preflight_01/ledger.jsonl'),
    ('proposed_budget.authorised', False), ('proposed_budget.transitions', 8421377),
    ('proposed_budget.transitions', 8421376.0), ('proposed_budget.native', 84231761),
    ('proposed_budget.qualification_native', 18001),
    ('project_cumulative_before.transitions', 18504193), ('project_cumulative_before.native', 185562111),
    ('resources.start_min_available_ram_bytes', 7*1024**3-1), ('resources.poll_s', 0),
    ('resources.poll_s', 3), ('resources.interrupt_grace_s', -1),
    ('resources.terminate_grace_s', float('inf')), ('resources.kill_grace_s', 0),
    ('resources.process_wall_grace_s', 181), ('supervisor_sha256', '0'*64),
])
def test_scientific_budget_resource_and_single_attempt_constraints_are_fixed(tmp_path, path, value):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    target = plan; parts = path.split('.')
    for key in parts[:-1]: target = target[key]
    target[parts[-1]] = value; write(plan_path, plan)
    receipt = checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)
    assert receipt['status'] == 'FAIL' and receipt['error']


@pytest.mark.parametrize('name', ['tools12/check_contract10_static.py', 'tests/test_contract10_static.py',
                                 'tools12/run_contract10.py', 'tests/test_run_contract10.py',
                                 'tests/test_run_contract10_integration.py', 'tools11/preflight_deployment11.py',
                                 'tools10/summarize_export10.py', 'reports/contract08_summary.json',
                                 'reports/contract09_summary.json', 'runtime03/build/d1_runtime03',
                                 'runtime03/schema03d.json', 'tools04/build_final_ledger04.py',
                                 'tools04/release_pilot04b.py', 'tools08/release_scale08.py'])
def test_required_gate_and_deployment_dependencies_cannot_be_omitted(tmp_path, name):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    del plan['frozen_inputs'][name]; write(plan_path, plan)
    assert checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)['status'] == 'FAIL'


@pytest.mark.parametrize('name', ['tools09/train_d1_ppo09.py', 'training/configs/pilot03d.json',
                                 'tools/evaluate_d1_ppo.py', 'tools08/replay_training_episodes08.py'])
def test_updating_a_plan_hash_cannot_rebaseline_old_scientific_code(tmp_path, name):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    (tmp_path/name).write_text('changed historical source')
    plan['frozen_inputs'][name] = sha(tmp_path/name); write(plan_path, plan)
    receipt = checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)
    assert receipt['status'] == 'FAIL' and 'rebaselined' in receipt['error']


@pytest.mark.parametrize('change', ['bad_hash', 'missing_file', 'absolute_path', 'external_symlink'])
def test_changed_missing_or_outside_frozen_files_are_rejected(tmp_path, change):
    plan_path, auth_path, plan, _ = fixture(tmp_path)
    if change == 'bad_hash': plan['frozen_inputs']['docs/career_execution_contract_10.md'] = '0'*64
    elif change == 'missing_file': (tmp_path/'docs/career_execution_contract_10.md').unlink()
    else:
        outside = write(tmp_path.parent/'outside-static-fixture.json', {'outside': True})
        name = str(outside)
        if change == 'external_symlink':
            alias = tmp_path/'outside-alias.json'; alias.symlink_to(outside); name = alias.name
        plan['frozen_inputs'][name] = sha(outside)
    write(plan_path, plan)
    assert checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)['status'] == 'FAIL'


def test_checker_reports_the_actual_pinned_supervisor_reserve_without_importing_it(tmp_path):
    plan_path, auth_path, _, _ = fixture(tmp_path)
    receipt = checker.check(plan_path, auth_path, tmp_path/'static', root=tmp_path)
    assert receipt['status'] == 'PASS' and receipt['supervisor_ram_reserve_bytes'] == 3221225472


def test_an_existing_static_receipt_is_never_overwritten(tmp_path):
    plan_path, auth_path, _, _ = fixture(tmp_path)
    out = tmp_path/'static'
    checker.check(plan_path, auth_path, out, root=tmp_path)
    before = (out/'report.json').read_bytes()
    with pytest.raises(FileExistsError): checker.check(plan_path, auth_path, out, root=tmp_path)
    assert (out/'report.json').read_bytes() == before


def test_cli_checks_only_disposable_text_and_emits_a_bound_receipt(tmp_path):
    plan_path, auth_path, _, _ = fixture(tmp_path)
    output = tmp_path/'cli-static'
    process = subprocess.run([sys.executable, '-B', str(tmp_path/'tools12/check_contract10_static.py'),
                              '--plan', str(plan_path), '--budget-authorisation', str(auth_path),
                              '--output', str(output)], capture_output=True, text=True, timeout=5)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['status'] == 'PASS'
    assert json.loads((output/'report.json').read_text())['frozen_inputs'] == json.loads(plan_path.read_text())['frozen_inputs']


def test_import_path_loads_no_training_simulation_or_inference_module():
    code = ('import sys,json; sys.path.insert(0,'+repr(str(ROOT/'tools12'))+'); '
            'import check_contract10_static; print(json.dumps([n for n in '
            '("torch","mujoco","gymnasium","stable_baselines3","onnxruntime","numpy") if n in sys.modules]))')
    process = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=5)
    assert process.returncode == 0 and json.loads(process.stdout) == []
