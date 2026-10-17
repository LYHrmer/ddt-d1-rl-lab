"""Check new scope and live-parent admission without models or physics."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools22 import common22 as c  # noqa: E402


def test_budget_is_independent_and_final_only():
    assert c.TRANSITIONS == 64*4096 == 262144
    assert c.OUTPUT != 'evidence/local_optimization18/execution_01'
    assert c.HYPERPARAMETERS == {**c.BASE_HYPERPARAMETERS, 'learning_rate': 1e-4}
    assert c.RESOURCES['reserve_bytes'] == 4 << 30
    assert c.RESOURCES['start_min_effective_bytes'] == 6 << 30
    assert c.STAGES['train']['mode'] == 'zero'
    assert sum(v['transitions'] for v in c.STAGES.values()) == 305344


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    plan_path = tmp_path/'plan.json'
    plan_path.write_text('{}')
    output = tmp_path/c.OUTPUT
    output.mkdir(parents=True)
    record = {'status': 'STARTED', 'nonce': 'new22',
              'command': c.command(plan_path, 'train', output/'train')}
    report = {'status': 'RUNNING', 'active_stage': 'train', 'supervisor_pid': c.os.getppid(),
              'plan_sha256': c.sha(plan_path), 'stages': {'train': record}}
    monkeypatch.setenv('D1_STAGE22_NONCE', 'new22')
    return plan_path, output, report


@pytest.mark.parametrize('mutation', ['none', 'pid', 'nonce', 'stage', 'command', 'plan_sha', 'status'])
def test_child_requires_actual_current_parent_and_exact_command(launch, mutation):
    path, output, report = launch
    if mutation == 'pid':
        report['supervisor_pid'] += 1
    elif mutation == 'nonce':
        report['stages']['train']['nonce'] = 'stale'
    elif mutation == 'stage':
        report['active_stage'] = 'export'
    elif mutation == 'command':
        report['stages']['train']['command'].append('--resume')
    elif mutation == 'plan_sha':
        report['plan_sha256'] = '0'*64
    elif mutation == 'status':
        report['status'] = 'COMPLETE'
    (output/'report.json').write_text(json.dumps(report))
    if mutation == 'none':
        c.validate_launch({}, path, 'train', output/'train')
    else:
        with pytest.raises(ValueError, match='parent'):
            c.validate_launch({}, path, 'train', output/'train')


def test_holdout_cannot_run_after_failed_original36(launch):
    path, output, report = launch
    report['active_stage'] = 'holdout'
    report['stages'] = {stage: {'status': 'COMPLETE'} for stage in ('train','export','cpp','formal')}
    report['stages']['holdout'] = {'status': 'STARTED', 'nonce': 'new22',
                                  'command': c.command(path, 'holdout', output/'holdout')}
    (output/'report.json').write_text(json.dumps(report))
    (output/'formal').mkdir()
    (output/'formal/report.json').write_text(json.dumps({'all_36_reference_pass': False}))
    with pytest.raises(ValueError, match='original36'):
        c.validate_launch({}, path, 'holdout', output/'holdout')


def test_fixed_scope_rejects_nonfinite_json(tmp_path):
    path = tmp_path/'bad.json'
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match='nonfinite'):
        c.read_json(path)


def test_lr_change_does_not_silently_change_other_parameters():
    before = copy.deepcopy(c.BASE_HYPERPARAMETERS)
    assert [key for key in before if before[key] != c.HYPERPARAMETERS[key]] == ['learning_rate']
