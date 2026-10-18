"""Independent25 contract admission tests; fake files and no runtime libraries."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools25 import common25 as c  # noqa: E402


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'runtime_dependencies', lambda root: {'fake_dependency.py': 'e'*64})
    old = {'config': 'training/configs/fake.json', 'base_checkpoint': {'path': '/fake/policy.zip',
           'sha256': 'a'*64, 'trained_transitions': c.INITIAL},
           'base_protocol': {'path': '/fake/protocol.json', 'sha256': 'b'*64}}
    result = {'status': 'COMPLETE', 'stage': 'train_zero', 'mode': 'zero', 'seed': 181, 'envs': 4,
              'base_checkpoint_sha256': 'a'*64, 'initial_num_timesteps': c.INITIAL,
              'initial_state': copy.deepcopy(c.BASE_INITIAL_STATE)}
    initial = tmp_path/c.BASE_INITIAL_RESULT
    initial.parent.mkdir(parents=True)
    initial.write_text(json.dumps(result))
    monkeypatch.setattr(c, 'BASE_INITIAL_RESULT_SHA256', c.sha(initial))
    old_plan = tmp_path/c.OLD_PLAN
    old_plan.parent.mkdir(parents=True, exist_ok=True)
    old_plan.write_text('{}')
    monkeypatch.setattr(c, 'base_inputs', lambda: (old, {}, {}))
    proof = {'git_commit': 'c'*40, 'sources_sha256': {'fake': 'd'*64}}
    monkeypatch.setattr(c, 'source_identity', lambda **kwargs: copy.deepcopy(proof))
    monkeypatch.setattr(c, 'check_sources', lambda *args, **kwargs: None)
    return old, result, initial


def write_plan(inputs):
    old, _, _ = inputs
    plan = {**c.fixed(old, root=c.ROOT), 'source_identity': c.source_identity()}
    path = c.ROOT/c.PLAN
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan))
    return plan, path


def test_two_arm_budget_modes_order_and_selection_are_independent(inputs):
    plan, _ = write_plan(inputs)
    assert plan['budget'] == {'policy_transitions': 610688, 'native_steps': 6106880, 'pd_updates': 3053440}
    assert sum(stage['transitions'] for stage in c.STAGES.values()) == 610688
    assert plan['stage_order'] == list(c.STAGE_ORDER)
    assert c.STAGES['train_random']['mode'] == 'uniform_0_3'
    assert c.STAGES['train_zero']['mode'] == 'zero'
    assert plan['hyperparameters']['target_kl'] == .005
    assert plan['resources']['required_nice'] == 10
    assert plan['resources']['interstage_admission_wait_s'] == 300
    assert plan['resources']['admission_poll_s'] == 5.
    assert plan['selection']['exact_tie'] == 'zero'
    assert plan['selection']['eligibility'] == 'own_original36_and_holdout36_all_pass'
    assert plan['checkpoint_selection'] == 'final_only'
    assert not plan['automatic_model_replacement'] and not plan['partial_checkpoint_eligible']
    assert not plan['retry_or_resume_allowed'] and plan['base_initial_state'] == c.BASE_INITIAL_STATE
    assert plan['base_initial_result']['sha256'] == c.sha(inputs[2])


@pytest.mark.parametrize('field,value', [('num_timesteps', 0), ('updates', 1),
                                      ('optimizer_restored', 1), ('policy_state_sha256', 'f'*64)])
def test_known_initial_policy_and_adam_state_are_not_replaceable(inputs, monkeypatch, field, value):
    old, result, path = inputs
    result['initial_state'][field] = value
    path.write_text(json.dumps(result))
    monkeypatch.setattr(c, 'BASE_INITIAL_RESULT_SHA256', c.sha(path))
    with pytest.raises(ValueError, match='initial state'):
        c.fixed(old, root=c.ROOT)


def test_initial_receipt_sha_change_rejected_without_reading_outer_attempt(inputs):
    old, _, path = inputs
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='receipt changed'):
        c.fixed(old, root=c.ROOT)


@pytest.mark.parametrize('key', c.STAGE_ORDER)
def test_arm_stage_paths_and_commands_do_not_cross_arm(inputs, key):
    plan, path = write_plan(inputs)
    arm, kind = c.split_stage(key)
    output = c.stage_path(plan, key, root=c.ROOT)
    assert output == c.ROOT/c.OUTPUT/arm/kind
    command = c.command(path, key, output)
    assert command[command.index('--arm')+1] == arm
    assert command[command.index('--output')+1] == str(output)
    assert command[2].endswith('train_delay25.py' if kind == 'train' else 'run_delay_comparison25.py')
    assert ('--stage' in command) is (kind != 'train')


@pytest.mark.parametrize('mutation', ['target_kl', 'mode', 'budget', 'selection', 'extra', 'source',
                                     'typed_flag', 'typed_poll'])
def test_load_plan_rejects_scope_or_source_drift(inputs, monkeypatch, mutation):
    plan, path = write_plan(inputs)
    if mutation == 'target_kl':
        plan['hyperparameters']['target_kl'] = None
    elif mutation == 'mode':
        plan['stages']['train_random']['mode'] = 'zero'
    elif mutation == 'budget':
        plan['budget']['native_steps'] -= 1
    elif mutation == 'selection':
        plan['selection']['exact_tie'] = 'random'
    elif mutation == 'extra':
        plan['resume'] = True
    elif mutation == 'typed_flag':
        plan['retry_or_resume_allowed'] = 0
    elif mutation == 'typed_poll':
        plan['resources']['poll_s'] = True
    else:
        monkeypatch.setattr(c, 'check_sources', lambda *args, **kwargs: c.require(False, 'source changed'))
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        c.load_plan(path)


@pytest.fixture
def launch(inputs, monkeypatch):
    plan, path = write_plan(inputs)
    monkeypatch.setenv('D1_STAGE25_NONCE', 'active25')
    def make(key):
        output = c.stage_path(plan, key, root=c.ROOT)
        records = {stage: {'status': 'COMPLETE'} for stage in c.STAGE_ORDER[:c.STAGE_ORDER.index(key)]}
        records[key] = {'status': 'STARTED', 'nonce': 'active25', 'command': c.command(path, key, output)}
        report = {'format': c.EXECUTION_FORMAT, 'status': 'RUNNING', 'active_stage': key,
                  'supervisor_pid': c.os.getppid(), 'plan_sha256': c.sha(path),
                  'source_identity': plan['source_identity'], 'stages': records}
        parent = c.ROOT/c.OUTPUT
        parent.mkdir(parents=True, exist_ok=True)
        return output, report
    return plan, path, make


@pytest.mark.parametrize('mutation', ['none', 'pid', 'nonce', 'command', 'sha', 'status', 'source', 'cross_arm'])
def test_worker_needs_current_parent_and_exact_arm_command(launch, mutation):
    plan, path, make = launch
    output, report = make('train_zero')
    if mutation == 'pid':
        report['supervisor_pid'] += 1
    elif mutation == 'nonce':
        report['stages']['train_zero']['nonce'] = 'stale'
    elif mutation == 'command':
        report['stages']['train_zero']['command'].append('--resume')
    elif mutation == 'sha':
        report['plan_sha256'] = '0'*64
    elif mutation == 'status':
        report['status'] = 'COMPLETE'
    elif mutation == 'source':
        report['source_identity'] = {}
    elif mutation == 'cross_arm':
        output = c.stage_path(plan, 'train_random', root=c.ROOT)
    (c.ROOT/c.OUTPUT/'report.json').write_text(json.dumps(report))
    if mutation == 'none':
        c.validate_launch(plan, path, 'train_zero', output)
    else:
        with pytest.raises(ValueError):
            c.validate_launch(plan, path, 'train_zero', output)


@pytest.mark.parametrize('zero_pass,zero_status,random_pass,valid', [
    (False, 'SKIPPED_NO_QUALIFIED_CANDIDATE', True, True),
    (True, 'SKIPPED_NO_QUALIFIED_CANDIDATE', True, False),
    (False, 'INCOMPLETE', True, False), (False, 'SKIPPED_NO_QUALIFIED_CANDIDATE', False, False)])
def test_holdout_skip_must_be_own_arm_ineligibility_and_other_arm_still_can_run(
        launch, zero_pass, zero_status, random_pass, valid):
    plan, path, make = launch
    output, report = make('holdout_random')
    report['stages']['holdout_zero']['status'] = zero_status
    for arm, passed in [('zero', zero_pass), ('random', random_pass)]:
        directory = c.stage_path(plan, 'formal_'+arm, root=c.ROOT)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'report.json').write_text(json.dumps({'all_36_reference_pass': passed}))
    (c.ROOT/c.OUTPUT/'report.json').write_text(json.dumps(report))
    if valid:
        c.validate_launch(plan, path, 'holdout_random', output)
    else:
        with pytest.raises(ValueError):
            c.validate_launch(plan, path, 'holdout_random', output)


def test_non_holdout_skip_never_satisfies_prior_stage(launch):
    plan, path, make = launch
    output, report = make('export_zero')
    report['stages']['train_zero']['status'] = 'SKIPPED_NO_QUALIFIED_CANDIDATE'
    (c.ROOT/c.OUTPUT/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='illegal skip'):
        c.validate_launch(plan, path, 'export_zero', output)


def test_prepare_is_exclusive_and_load_roundtrips_only_new_path(inputs):
    path = c.ROOT/c.PLAN
    plan = c.prepare(path)
    assert c.load_plan(path) == plan
    with pytest.raises(ValueError, match='new fixed25'):
        c.prepare(path)
    with pytest.raises(ValueError, match='fixed25'):
        c.load_plan(c.ROOT/'foreign.json')


def test_plan_cli_is_cold_and_does_not_prepare(monkeypatch, capsys):
    monkeypatch.setattr(c, 'prepare', lambda: pytest.fail('plan prepared files'))
    monkeypatch.setattr(c, 'base_inputs', lambda: pytest.fail('plan read old evidence'))
    monkeypatch.setattr(c, 'load_plan', lambda: pytest.fail('plan entered gate'))
    assert c.main(['--plan']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'PLAN_ONLY'


def test_base_inputs_delegates_only_original_source_reader(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(c.previous, 'base_inputs', lambda root: seen.append(root) or ('old', 'release', 'protocol'))
    assert c.base_inputs(tmp_path) == ('old', 'release', 'protocol')
    assert seen == [tmp_path]


def test_source_checker_binds_exact_closure_to_committed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'SOURCES', ('source.py',))
    path = tmp_path/'source.py'
    path.write_text('value = 1\n')
    proof = {'git_commit': 'a'*40, 'sources_sha256': {'source.py': c.sha(path)}}
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *args, **kwargs: path.read_bytes())
    c.check_sources(proof, root=tmp_path)
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *args, **kwargs: b'changed\n')
    with pytest.raises(ValueError, match='commit25'):
        c.check_sources(proof, root=tmp_path)
    proof['sources_sha256']['foreign.py'] = 'f'*64
    with pytest.raises(ValueError, match='closure'):
        c.check_sources(proof, root=tmp_path)


def test_one_env_fixed4096_rollout_not_identical_four_env_sampling(inputs):
    plan, _ = write_plan(inputs)
    assert plan['envs'] == 1 and plan['hyperparameters']['n_steps'] == 4096
    assert plan['vector_layout'] == c.VECTOR_LAYOUT
    assert not plan['vector_layout']['comparable_to24_as_single_variable']
    assert plan['runtime_dependencies_sha256'] == {'fake_dependency.py': 'e'*64}


def test_runtime_dependency_bytes_are_bound_without_pretending_git_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'RUNTIME_DEPENDENCIES', ('deps/base.py',))
    path = tmp_path/'deps/base.py'
    path.parent.mkdir()
    path.write_text('load = 1\n')
    before = c.runtime_dependencies(tmp_path)
    path.write_text('load = 2\n')
    assert before != c.runtime_dependencies(tmp_path)


@pytest.mark.parametrize('change', ['envs', 'n_steps', 'dependencies', 'layout'])
def test_25_rejects_four_env_or_runtime_dependency_drift(inputs, change):
    plan, path = write_plan(inputs)
    if change == 'envs': plan['envs'] = 4
    elif change == 'n_steps': plan['hyperparameters']['n_steps'] = 1024
    elif change == 'dependencies': plan['runtime_dependencies_sha256']['fake_dependency.py'] = 'f'*64
    else: plan['vector_layout']['interprocess_rpc'] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='scope25'):
        c.load_plan(path)
