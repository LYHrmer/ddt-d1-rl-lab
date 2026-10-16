"""Orchestration must never turn a restart into a second physical attempt."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools12'))
import run_contract10 as runner  # noqa: E402


def test_stage_is_bound_to_artifacts_and_can_only_be_read_on_resume(tmp_path):
    output = tmp_path/'output.json'
    calls = []
    def execute():
        calls.append(True)
        output.write_text('{"status":"PASS"}')
        return {'returncode': 0, 'cleanup_complete': True}
    journal = runner.Journal(tmp_path/'journal', 'release-sha')
    first = journal.stage('evaluation', execute, lambda: [output])
    assert first['status'] == 'FINISHED'
    assert journal.stage('evaluation', execute, lambda: [output]) == first
    assert calls == [True]
    output.write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match='changed'):
        journal.stage('evaluation', execute, lambda: [output])
    assert calls == [True]


def test_started_or_failed_stage_cannot_be_retried(tmp_path):
    journal = runner.Journal(tmp_path/'journal', 'release-sha')
    def crash():
        raise RuntimeError('killed before final receipt')
    with pytest.raises(RuntimeError):
        journal.stage('scale', crash, lambda: [])
    with pytest.raises(ValueError, match='already attempted'):
        journal.stage('scale', lambda: pytest.fail('must not retry'), lambda: [])


def test_resource_refusal_does_not_consume_attempt(tmp_path):
    journal = runner.Journal(tmp_path/'journal', 'release-sha')
    result = journal.stage('preflight', lambda: {'returncode': 2, 'not_ready': True}, lambda: [])
    assert result['status'] == 'NOT_READY'
    calls = []
    journal.stage('preflight', lambda: calls.append(True) or {'returncode': 0}, lambda: [])
    assert calls == [True]


def test_resume_rejects_changed_release_and_runner(tmp_path, monkeypatch):
    runner.Journal(tmp_path/'journal', 'one')
    with pytest.raises(ValueError, match='identity'):
        runner.Journal(tmp_path/'journal', 'two')
    monkeypatch.setattr(runner, 'sha', lambda p: 'different')
    with pytest.raises(ValueError, match='identity'):
        runner.Journal(tmp_path/'journal', 'one')


def test_plan_has_only_one_checkpoint_evaluation_and_zero_step_replay(tmp_path):
    release = {'entry': 'trainer.py', 'config': 'config.json',
               'plans': {'preflight': {'output': 'preflight'}, 'scale': {'output': 'train'}}}
    commands = runner.commands(release, tmp_path/'release.json', tmp_path)
    evaluation = commands['evaluation']
    assert '--checkpoint' in evaluation and '--onnx' not in evaluation
    assert evaluation[evaluation.index('--max-native-steps')+1] == '18000'
    replay = commands['replay']
    assert replay[replay.index('--verify-episodes')+1] == '0'


def test_partial_accounting_does_not_invent_missing_counts(tmp_path):
    plan = {'proposed_budget': {'transitions': 8421376, 'native': 84231760},
            'project_cumulative_before': {'transitions': 18504192, 'native': 185562110},
            'plans': {'preflight': {'output': 'preflight', 'steps': 32768},
                      'scale': {'output': 'train', 'steps': 8388608}}}
    (tmp_path/'preflight').mkdir()
    (tmp_path/'preflight/result.json').write_text(json.dumps({
        'status': 'COMPLETE', 'actual_counts': {'policy_transitions': 32768, 'native_steps': 327680}}))
    report = runner.accounting(plan, {'preflight': {'status': 'FINISHED',
        'result': {'cleanup_complete': True}}, 'scale': {}}, tmp_path)
    assert report['known_actual'] == {'transitions': 32768, 'native': 327680}
    assert report['unknown_actual_pools'] == ['scale']
    assert report['actual_complete'] is False


@pytest.mark.parametrize('value', [True, -1, 1.5, None])
def test_bad_counts_remain_unknown(tmp_path, value):
    plan = {'proposed_budget': {}, 'project_cumulative_before': {},
            'plans': {'preflight': {'output': 'preflight', 'steps': 32768},
                      'scale': {'output': 'train', 'steps': 8388608}}}
    (tmp_path/'preflight').mkdir()
    (tmp_path/'preflight/result.json').write_text(json.dumps({
        'actual_counts': {'policy_transitions': value, 'native_steps': 10}}))
    report = runner.accounting(plan, {'preflight': {}}, tmp_path)
    assert report['unknown_actual_pools'] == ['preflight']
    assert report['known_actual']['transitions'] == 0


def test_interrupted_evaluation_preserves_reported_lower_bound(tmp_path):
    plan = {'proposed_budget': {}, 'project_cumulative_before': {}, 'plans': {}}
    path = tmp_path/runner.OUTPUTS['evaluation']/'error.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'INVALID_INTERRUPTED', 'actual_native_steps': 3200}))
    report = runner.accounting(plan, {'evaluation': {'status': 'FINISHED',
        'result': {'cleanup_complete': True}}}, tmp_path)
    assert report['known_actual'] == {'transitions': 0, 'native': 3200}
    assert report['unknown_actual_pools'] == ['evaluation']
    assert report['pools']['evaluation']['receipt_sha256'] == runner.sha(path)
