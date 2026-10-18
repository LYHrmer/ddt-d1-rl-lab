"""Cold tests for bounded KL comparison, immutable plans and complete matrices."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools29 import common29 as c


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    base = {'config': 'training/configs/pilot03d.json',
            'base_checkpoint': {'path': 'complete25.zip', 'sha256': 'a' * 64,
                                'trained_transitions': c.INITIAL},
            'base_protocol': {'path': 'protocol25.json', 'sha256': 'b' * 64},
            'baseline_provenance': {'serialized_entries_sha256': dict(c.BASE_SERIALIZED)},
            'runtime_dependencies_sha256': {'base_class.py': 'c' * 64}}
    proof = {'git_commit': 'd' * 40, 'sources_sha256': {'fake.py': 'e' * 64}}
    monkeypatch.setattr(c, 'base_inputs', lambda: (copy.deepcopy(base), {}, {}))
    monkeypatch.setattr(c, 'source_identity', lambda: copy.deepcopy(proof))
    monkeypatch.setattr(c, 'check_sources', lambda *a, **kw: None)
    plan = {**c.fixed(base), 'source_identity': proof}
    path = tmp_path / c.PLAN
    path.parent.mkdir(parents=True)
    c.atomic_json(path, plan)
    return plan, path


def test_budget_covers_both_arms_and_both_full_development_matrices(inputs):
    plan, _ = inputs
    assert plan['budget'] == {'policy_transitions': 108736, 'native_steps': 1087360, 'pd_updates': 543680}
    assert sum(stage['transitions'] for stage in c.STAGES.values()) == 108736
    assert len(c.STAGES) == 8 and all('holdout' not in key for key in c.STAGES)
    assert all(stage['mode'] == 'uniform_0_3' for stage in c.STAGES.values())
    assert plan['arms'] == {'control': 0.0, 'anchor': 1.0}
    assert plan['anchor']['reward_yaw_weight'] == 2.0 and not plan['anchor']['reward_changed']
    assert plan['anchor']['teacher_actor_samples_max'] == 32768
    assert plan['anchor']['teacher_batch_calls_max'] == 8
    assert plan['selection']['maximum_holdout_arms'] == 0
    assert not plan['partial_checkpoint_eligible'] and not plan['automatic_model_replacement']
    assert plan['resources']['start_min_effective_bytes'] == 4 << 30
    assert plan['resources']['reserve_bytes'] == 2 << 30


@pytest.mark.parametrize('fault', ['coef', 'seed', 'reward', 'teacher', 'cap', 'reserve', 'typed', 'holdout', 'resume'])
def test_plan_rejects_changes_to_intervention_or_budget(inputs, fault):
    plan, path = inputs
    if fault == 'coef':
        plan['arms']['anchor'] = 2.0
    elif fault == 'seed':
        plan['seed'] = 182
    elif fault == 'reward':
        plan['anchor']['reward_yaw_weight'] = 3.0
    elif fault == 'teacher':
        plan['anchor']['teacher_requires_grad'] = True
    elif fault == 'cap':
        plan['budget']['policy_transitions'] += 1
    elif fault == 'reserve':
        plan['resources']['reserve_bytes'] = 0
    elif fault == 'typed':
        plan['envs'] = True
    elif fault == 'holdout':
        plan['stage_order'].append('holdout_anchor')
    else:
        plan['retry_or_resume_allowed'] = True
    c.atomic_json(path, plan)
    with pytest.raises(ValueError, match='scope29'):
        c.load_plan(path)


def formal(passed=True, cost=1.0):
    return {'status': 'COMPLETE', 'error': None, 'cleanup_complete': True,
            'all_36_reference_pass': passed,
            'cases': [{'sequence': s, 'delay_ticks': d, 'reset_seed': z, 'J': cost,
                       'reference_thresholds_met': passed, 'failure': not passed,
                       'done_reason': 'HORIZON_12S', 'policy_transitions': 600}
                      for s in ('forward_stop', 'reverse_stop', 'turn_stop')
                      for d in range(4) for z in (1001, 1002, 1003)]}


def test_low_cost_cannot_qualify_failed_task_or_trigger_holdout():
    result = c.select_candidate({'control': formal(), 'anchor': formal(False, .01)})
    assert result['eligible_arms'] == ['control']
    assert result['selected_arm'] is None and not result['uses_holdout']
    assert not result['qualification_granted']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'partial', 'nonfinite', 'short', 'unknown'])
def test_partial_or_unverified_matrix_cannot_be_called_complete(fault):
    reports = {name: formal() for name in c.ARMS}
    value = reports['anchor']
    if fault == 'missing':
        value['cases'].pop()
    elif fault == 'duplicate':
        value['cases'][0] = copy.deepcopy(value['cases'][1])
    elif fault == 'partial':
        value['status'] = 'INCOMPLETE'
    elif fault == 'nonfinite':
        value['cases'][0]['J'] = float('nan')
    elif fault == 'short':
        value['cases'][0]['policy_transitions'] = 599
    else:
        value['cases'][0]['reference_thresholds_met'] = None
    with pytest.raises(ValueError):
        c.select_candidate(reports)


@pytest.mark.parametrize('fault', [None, 'nonce', 'parent', 'prior', 'command', 'source', 'cross_arm'])
def test_child_requires_actual_current_parent_stage(inputs, monkeypatch, fault):
    plan, path = inputs
    key = 'train_anchor'
    output = c.stage_path(plan, key)
    monkeypatch.setenv('D1_STAGE29_NONCE', 'nonce29')
    journal = {'format': c.EXECUTION_FORMAT, 'status': 'RUNNING', 'active_stage': key,
               'supervisor_pid': c.os.getppid(), 'plan_sha256': c.sha(path),
               'source_identity': plan['source_identity'],
               'stages': {'train_control': {'status': 'COMPLETE'},
                          key: {'status': 'STARTED', 'nonce': 'nonce29', 'command': c.command(path, key, output)}}}
    if fault == 'nonce':
        journal['stages'][key]['nonce'] = 'stale'
    elif fault == 'parent':
        journal['supervisor_pid'] += 1
    elif fault == 'prior':
        journal['stages']['train_control']['status'] = 'INCOMPLETE'
    elif fault == 'command':
        journal['stages'][key]['command'].append('--resume')
    elif fault == 'source':
        journal['source_identity'] = {}
    elif fault == 'cross_arm':
        output = c.stage_path(plan, 'train_control')
    target = c.ROOT / c.OUTPUT / 'report.json'
    target.parent.mkdir(parents=True)
    c.atomic_json(target, journal)
    if fault:
        with pytest.raises(ValueError):
            c.validate_launch(plan, path, key, output)
    else:
        c.validate_launch(plan, path, key, output)


def test_prepare_never_overwrites_or_reuses_executed_path(inputs):
    plan, path = inputs
    with pytest.raises(ValueError):
        c.prepare(path)
    path.unlink()
    (c.ROOT / c.OUTPUT).mkdir()
    with pytest.raises(ValueError, match='execution already exists'):
        c.prepare(path)


def test_source_proof_binds_committed_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'SOURCES', ('one.py',))
    path = tmp_path / 'one.py'
    path.write_text('x = 1\n')
    proof = {'git_commit': 'a' * 40, 'sources_sha256': {'one.py': c.sha(path)}}
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *a, **kw: b'x = 1\n')
    c.check_sources(proof, tmp_path)
    monkeypatch.setattr(c.subprocess, 'check_output', lambda *a, **kw: b'x = 2\n')
    with pytest.raises(ValueError, match='source first'):
        c.check_sources(proof, tmp_path)


@pytest.fixture
def complete_fake(tmp_path, monkeypatch):
    path = Path(__file__).with_name('test_train_anchor29.py')
    spec = importlib.util.spec_from_file_location('common29_training_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, plan_path, plan = module.case.__wrapped__(tmp_path)
    plan['output_root'] = c.OUTPUT
    c.atomic_json(plan_path, plan)
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'base_inputs', lambda: ({}, {'sources_sha256': {}}, {'versions': module.VERSIONS}))
    return module, (root, plan_path, plan)


@pytest.mark.parametrize('arm', ['control', 'anchor'])
def test_actual_trainer_fake_artifacts_pass_independent_completion_reader(complete_fake, arm):
    module, case = complete_fake
    result, _, directory = module.run_fake(case, arm=arm)
    assert result['status'] == 'COMPLETE'
    bundle = c.check_completed_training(case[2], case[1], arm)
    assert bundle['checkpoint'] == directory / 'policy.zip'
    assert bundle['result'] == result
    assert bundle['identity']['role'] == 'completed_anchor29_' + arm + '_final_only'


@pytest.mark.parametrize('fault', ['partial', 'counts', 'teacher', 'cache', 'updates', 'reward', 'first', 'checkpoint'])
def test_completed_reader_rejects_false_training_or_teacher_evidence(complete_fake, fault):
    module, case = complete_fake
    result, _, directory = module.run_fake(case, arm='anchor')
    if fault == 'partial':
        result['status'] = 'PARTIAL'
    elif fault == 'counts':
        result['actual_counts']['resets'] = True
    elif fault == 'teacher':
        result['anchor29']['frozen_actor_sha256_after_training'] = '0' * 64
    elif fault == 'cache':
        result['anchor29']['cache_rollouts'][0]['generation'] += 4096
    elif fault == 'updates':
        result['anchor29']['optimizer_updates'] = 0
    elif fault == 'reward':
        result['reward_state29']['extra_reward_sum'] = 1.0
    elif fault == 'first':
        result['first_rollout_nonreward']['sha256'] = '0' * 64
    else:
        (directory / 'policy.zip').write_bytes(b'foreign checkpoint')
    c.atomic_json(directory / 'result.json', result)
    with pytest.raises(ValueError):
        c.check_completed_training(case[2], case[1], 'anchor')
