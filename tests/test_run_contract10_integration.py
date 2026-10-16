"""Independent fake-process sequence tests: no training, model or integration steps."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools12'))
import run_contract10 as runner  # noqa: E402
import verify_contract10 as verdict_tools  # noqa: E402
from test_verify_contract10 import setup_completed  # noqa: E402
from test_release_contract10 import write_json  # noqa: E402


def fake_sequence(root, monkeypatch, *, low_first=False, crash=False, eval_interrupt=False,
                  foreign_report=False):
    rel, _, replay_template, eval_template, frozen = setup_completed(root)
    fixtures = root/'fixtures'
    fixtures.mkdir()
    for kind, plan in frozen['plans'].items():
        output = root/plan['output']
        shutil.move(output, fixtures/kind)
        shutil.move(runner.supervisor.receipt_dir(output), fixtures/(kind+'_supervisor'))
    calls = []

    def train(kind, *_args, **_kwargs):
        calls.append(kind)
        if low_first and calls == ['preflight']:
            return 2
        output = root/frozen['plans'][kind]['output']
        shutil.copytree(fixtures/kind, output)
        shutil.copytree(fixtures/(kind+'_supervisor'), runner.supervisor.receipt_dir(output))
        if crash:
            raise RuntimeError('fake crash after attempt was claimed')
        if foreign_report and kind == 'scale':
            path = runner.supervisor.receipt_dir(output)/'report.json'
            value = json.loads(path.read_text()); value['release_sha256'] = '0'*64
            write_json(path, value)
        return 0

    def child(argv, directory, _timeout, _env):
        name = Path(directory).name
        calls.append(name)
        output = Path(argv[argv.index('--output')+1])
        if name == 'export':
            write_json(output/'failure.json', {'status': 'FAIL', 'reason': 'fake frozen tolerance refusal'})
            return {'returncode': 1, 'cleanup_complete': True}
        if name == 'evaluation':
            assert '--checkpoint' in argv and '--onnx' not in argv
            if eval_interrupt:
                write_json(output/'error.json', {'status': 'INVALID_INTERRUPTED', 'actual_native_steps': 3200})
                return {'returncode': 1, 'cleanup_complete': True}
            shutil.copytree(eval_template, output)
        elif name == 'replay':
            shutil.copytree(replay_template.parent, output)
        elif name == 'verdict':
            verdict_tools.write_verdict(rel,
                runner.supervisor.receipt_dir(root/frozen['plans']['scale']['output'])/'report.json',
                root/runner.OUTPUTS['replay']/'report.json' if '--replay' in argv else None,
                root/runner.OUTPUTS['evaluation'] if '--eval-dir' in argv else None,
                output, root=root)
        else:
            pytest.fail(f'unexpected fake stage {name}')
        return {'returncode': 0, 'cleanup_complete': True}

    monkeypatch.setattr(runner.supervisor, 'run', train)
    monkeypatch.setattr(runner, 'run_checked', child)
    return rel, calls


def journal(root):
    return json.loads((root/runner.BASE/'execution_01/journal.json').read_text())


def test_complete_fake_sequence_retains_export_failure_and_evaluates_only_final_checkpoint(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch)
    assert runner.execute(rel, root=tmp_path) == 0
    assert calls == ['preflight', 'scale', 'export', 'evaluation', 'replay', 'verdict']
    value = journal(tmp_path)
    assert value['status'] == 'COMPLETE'
    assert value['stages']['export']['result']['returncode'] == 1
    assert 'cpp' not in value['stages']
    assert value['accounting']['actual_complete'] is True
    assert value['accounting']['known_actual'] == {'transitions': 8421376, 'native': 84225760}
    assert runner.execute(rel, root=tmp_path) == 0
    assert len(calls) == 6


def test_readiness_refusal_can_be_admitted_again_but_never_runs_downstream(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch, low_first=True)
    assert runner.execute(rel, root=tmp_path) == 2
    assert calls == ['preflight']
    assert journal(tmp_path)['status'] == 'WAITING_RAM'
    assert journal(tmp_path)['stages']['preflight']['status'] == 'NOT_READY'
    assert runner.execute(rel, root=tmp_path) == 0
    assert calls == ['preflight', 'preflight', 'scale', 'export', 'evaluation', 'replay', 'verdict']


def test_crash_after_claim_cannot_start_training_a_second_time(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch, crash=True)
    with pytest.raises(RuntimeError, match='fake crash'):
        runner.execute(rel, root=tmp_path)
    assert journal(tmp_path)['stages']['preflight']['status'] == 'STARTED'
    assert journal(tmp_path)['accounting']['unknown_actual_pools'] == ['preflight']
    with pytest.raises(ValueError, match='already attempted'):
        runner.execute(rel, root=tmp_path)
    assert calls == ['preflight']


def test_changed_finished_artifact_blocks_resume_without_second_physical_attempt(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch)
    assert runner.execute(rel, root=tmp_path) == 0
    (tmp_path/runner.OUTPUTS['evaluation']/'episodes.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='artifact changed'):
        runner.execute(rel, root=tmp_path)
    assert len(calls) == 6


def test_interrupted_eval_has_unknown_budget_and_no_replay_or_retry(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch, eval_interrupt=True)
    assert runner.execute(rel, root=tmp_path) == 1
    assert calls == ['preflight', 'scale', 'export', 'evaluation']
    value = journal(tmp_path)
    assert value['status'] == 'EVALUATION_FAILED'
    assert value['accounting']['actual_complete'] is False
    assert value['accounting']['unknown_actual_pools'] == ['evaluation']
    assert runner.execute(rel, root=tmp_path) == 1
    assert len(calls) == 4


def test_main_default_only_prints_plan_and_never_claims_a_journal(tmp_path, monkeypatch, capsys):
    frozen = {'config': 'config.json', 'plans': {'preflight': {'output': 'preflight'},
                                               'scale': {'output': 'train'}}}
    monkeypatch.setattr(runner, 'load_release', lambda *_args: frozen)
    monkeypatch.setattr(runner, 'execute', lambda *_args: pytest.fail('default must not execute'))
    monkeypatch.setattr(sys, 'argv', ['run_contract10.py', '--release', str(tmp_path/'release.json')])
    assert runner.main() == 0
    value = json.loads(capsys.readouterr().out)
    assert value['mode'] == 'PLAN_ONLY' and value['physics_steps_executed'] == 0
    assert not (tmp_path/runner.BASE/'execution_01').exists()


def test_foreign_supervisor_report_is_rejected_before_export_or_evaluation(tmp_path, monkeypatch):
    rel, calls = fake_sequence(tmp_path, monkeypatch, foreign_report=True)
    with pytest.raises((SystemExit, ValueError)):
        runner.execute(rel, root=tmp_path)
    assert calls == ['preflight', 'scale']


def test_malformed_accounting_receipt_is_unknown_instead_of_destroying_closeout(tmp_path):
    release = {'proposed_budget': {}, 'project_cumulative_before': {},
               'plans': {'preflight': {'output': 'preflight', 'steps': 32768},
                         'scale': {'output': 'scale', 'steps': 8388608}}}
    (tmp_path/'preflight').mkdir()
    (tmp_path/'preflight/result.json').write_text('{"unfinished":')
    result = runner.accounting(release, {'preflight': {'status': 'STARTED'}}, root=tmp_path)
    assert result['unknown_actual_pools'] == ['preflight']
    assert result['actual_complete'] is False
