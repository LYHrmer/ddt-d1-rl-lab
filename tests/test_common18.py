"""No-model regressions for the new optimizer's process and identity boundary."""
from pathlib import Path
from types import SimpleNamespace
import copy
import signal
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools18 import common18 as common  # noqa: E402
from tools18 import run_optimization18 as runner  # noqa: E402


@pytest.mark.parametrize('counts,complete,expected', [
    ({'policy_transitions': 4, 'native_steps': 40, 'pd_updates': 20}, True, True),
    ({'policy_transitions': 4, 'native_steps': 0, 'pd_updates': 0}, False, True),
    ({'policy_transitions': 4, 'native_steps': 0, 'pd_updates': 0}, True, False),
    ({'policy_transitions': 4, 'native_steps': 31, 'pd_updates': 16}, False, True),
    ({'policy_transitions': 4, 'native_steps': 31, 'pd_updates': 14}, False, False),
    ({'policy_transitions': 4, 'native_steps': 41, 'pd_updates': 20}, False, False),
    ({'policy_transitions': True, 'native_steps': 10, 'pd_updates': 5}, False, False),
    ({'policy_transitions': 10, 'native_steps': 0, 'pd_updates': 0}, False, False),
    ({'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000}, True, True),
    ({'policy_transitions': 601, 'native_steps': 6010, 'pd_updates': 3005}, True, False),
])
def test_actual_physical_feasibility(counts, complete, expected):
    assert runner.valid_counts(counts, common.expected_counts(600), complete=complete, envs=4) is expected


def fake_attempt(tmp_path, monkeypatch, *, late=False, foreign=False, spawn_signal=None,
                 cleanup_error=False, changed_final_plan=False, accounting_signal=None):
    plan_path = tmp_path/'plan.json'
    plan_path.write_text('{}')
    plan = {'output_root': 'execution', 'stage_order': ['probe_base'],
            'stages': {'probe_base': copy.deepcopy(common.STAGES['probe_base'])},
            'source_identity': {'sources': 'fake'}, 'resources': common.RESOURCES,
            'envs': 4, 'budget': common.expected_counts(600)}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.common, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'unchanged_stages', lambda report: None)
    now = [0.]
    cleanup_calls = []
    if accounting_signal is not None:
        original_atomic = common.atomic_json
        captured = {}
        class InterruptedCounts(dict):
            interrupted = False
            def __getitem__(self, key):
                value = super().__getitem__(key)
                if key == 'native_steps' and not self.interrupted:
                    self.interrupted = True
                    signal.raise_signal(accounting_signal)
                return value
        def atomic(path, value):
            if 'known_actual_counts' in value:
                captured['report'] = value
            return original_atomic(path, value)
        def archive(directory):
            record = captured['report']['stages']['probe_base']
            record['actual_counts'] = InterruptedCounts(record['actual_counts'])
            return {}
        monkeypatch.setattr(common, 'atomic_json', atomic)
        monkeypatch.setattr(runner, 'artifacts', archive)

    def popen(argv, **kwargs):
        output = Path(argv[-1])
        output.mkdir()
        protocol = {'plan_sha256': 'foreign' if foreign else common.sha(plan_path),
                    'source_identity': plan['source_identity'], 'stage': 'probe_base'}
        common.atomic_json(output/'protocol.json', protocol)
        child = {'status': 'COMPLETE', 'stage': 'probe_base', 'mode': 'baseline',
                 'source_identity': plan['source_identity'], 'plan_sha256': common.sha(plan_path),
                 'actual_counts': common.expected_counts(270), 'cleanup_complete': True,
                 'inputs_unchanged': True, 'error': None,
                 'protocol_sha256': common.sha(output/'protocol.json'),
                 'cases': [{'sequence': 'turn_stop', 'delay_ticks': 3, 'reset_seed': 1001}]}
        common.atomic_json(output/'report.json', child)
        if late:
            now[0] = 200.
        if spawn_signal is not None:
            signal.raise_signal(spawn_signal)
        return SimpleNamespace(pid=99999999, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    def cleanup(*args, **kwargs):
        cleanup_calls.append(True)
        if cleanup_error:
            raise RuntimeError('synthetic cleanup failure')
        return [], True
    monkeypatch.setattr(runner, 'cleanup', cleanup)
    loads = [0]
    def loader(path):
        loads[0] += 1
        if changed_final_plan and loads[0] == 3:
            path.write_text(path.read_text()+'\n')
        return plan
    report = runner.execute(plan_path, loader=loader,
                            memory_reader=lambda: {'effective_available_bytes': 7*2**30}, clock=lambda: now[0])
    return report, cleanup_calls, plan_path, plan


def test_terminal_child_cleanup_contract_and_real_counts(tmp_path, monkeypatch):
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch)
    assert report['status'] == 'COMPLETE'
    assert report['known_actual_counts'] == common.expected_counts(270)
    assert report['cleanup_complete'] is True and calls == [True]
    assert report['stages']['probe_base']['cleanup_signals'] == []


def test_exited_process_cannot_hide_wall_violation(tmp_path, monkeypatch):
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch, late=True)
    assert report['status'] == 'STOPPED' and 'deadline' in report['error']
    assert calls == [True] and report['actual_counts'] == common.expected_counts(270)


def test_foreign_partial_protocol_remains_unknown(tmp_path, monkeypatch):
    report, _, _, _ = fake_attempt(tmp_path, monkeypatch, late=True, foreign=True)
    assert report['status'] == 'STOPPED' and report['actual_counts'] is None
    assert report['unknown_count_stages'] == ['probe_base']
    assert report['known_actual_counts'] == common.expected_counts(0)


def test_existing_attempt_is_never_restarted(tmp_path, monkeypatch):
    _, _, path, plan = fake_attempt(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='no resume or retry'):
        runner.execute(path, loader=lambda _: plan)


def test_low_memory_admission_creates_no_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = {'output_root': 'execution', 'resources': common.RESOURCES}
    path = tmp_path/'plan.json'
    path.write_text('{}')
    report = runner.execute(path, loader=lambda _: plan, memory_reader=lambda: {'effective_available_bytes': 5*2**30})
    assert report['status'] == 'NOT_READY' and not report['execution_started']
    assert not (tmp_path/'execution').exists()


def test_frozen_source_requires_commit_and_unchanged_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(common, 'SOURCE_NAMES', ('source.py',))
    path = tmp_path/'source.py'
    path.write_text('original')
    proof = {'git_commit': 'a'*40, 'sources_sha256': {'source.py': common.sha(path)}}
    monkeypatch.setattr(common.subprocess, 'check_output', lambda *args, **kwargs: b'original')
    common.check_sources(proof)
    path.write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        common.check_sources(proof)
    path.write_text('original')
    monkeypatch.setattr(common.subprocess, 'check_output', lambda *args, **kwargs: b'other commit blob')
    with pytest.raises(ValueError, match='uncommitted'):
        common.check_sources(proof)


def test_total_budget_includes_preflights_probe_and_holdout():
    assert common.budget() == common.expected_counts(2228088)
    assert sum(item['transitions'] for name, item in common.STAGES.items() if name.startswith(('preflight_', 'train_'))) == 2162688


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_signal_during_popen_return_still_cleans_the_owned_child(tmp_path, monkeypatch, signum):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch, spawn_signal=signum)
    assert report['status'] == 'STOPPED'
    assert calls == [True], 'the process created inside Popen must reach owned cleanup'
    assert previous == {sig: signal.getsignal(sig) for sig in previous}


def test_cleanup_exception_is_recorded_as_unconfirmed_cleanup(tmp_path, monkeypatch):
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch, cleanup_error=True)
    assert report['status'] == 'STOPPED' and calls == [True]
    assert report['cleanup_complete'] is False
    assert report['stages']['probe_base']['cleanup_complete'] is False


def test_last_stage_plan_change_cannot_receive_complete(tmp_path, monkeypatch):
    report, _, path, _ = fake_attempt(tmp_path, monkeypatch, changed_final_plan=True)
    assert report['status'] == 'STOPPED'
    assert report['plan_sha256'] != common.sha(path)


def test_execution_uses_frozen_stage_order_instead_of_json_object_key_order(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(common, 'verify_inputs', lambda plan: None)
    plan = {'format': common.FORMAT, 'owner_statement': common.OWNER, 'status': 'FROZEN_NEW_SCOPE',
            'config': 'training/configs/pilot03d.json', 'seed': 181, 'envs': 4,
            'rollout_transitions': 4096, 'resources': common.RESOURCES,
            'stages': {'train_zero': common.STAGES['train_zero'],
                       **{key: value for key, value in common.STAGES.items() if key != 'train_zero'}},
            'stage_order': list(common.STAGES), 'budget': common.budget(), 'output_root': common.OUTPUT,
            'one_attempt_per_stage': True, 'retry_or_resume_allowed': False, 'warm_start': True,
            'preflight_weights_discarded': True,
            'selection': 'All36 original reference flags true; lowest mean J among eligible arms, zero tie first.',
            'holdout_reset_seeds': [2001, 2002, 2003],
            'source_identity': {'sources_sha256': dict.fromkeys(common.SOURCE_NAMES, 'fake')}}
    path = tmp_path/'evidence/local_optimization18/plan.json'
    path.parent.mkdir(parents=True)
    common.atomic_json(path, plan)
    assert common.load_plan(path)['stage_order'][0] == 'probe_base'
    spawned = []
    def stop_at_spawn(argv, **kwargs):
        spawned.append(argv[argv.index('--stage')+1])
        raise OSError('synthetic first launch stop')
    monkeypatch.setattr(runner.subprocess, 'Popen', stop_at_spawn)
    report = runner.execute(path, memory_reader=lambda: {'effective_available_bytes': 7*2**30})
    assert report['status'] == 'STOPPED'
    assert spawned == ['probe_base']


def test_failure_after_success_accounting_does_not_count_the_receipt_twice(tmp_path, monkeypatch):
    def disconnected_stdout(*args, **kwargs):
        raise BrokenPipeError('synthetic progress pipe closed')
    monkeypatch.setattr('builtins.print', disconnected_stdout)
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch)
    assert calls == [True] and report['status'] == 'STOPPED'
    assert report['known_actual_counts'] == common.expected_counts(270)


def test_signal_while_reading_stage_counts_retains_one_known_subtotal(tmp_path, monkeypatch):
    report, calls, _, _ = fake_attempt(tmp_path, monkeypatch, accounting_signal=signal.SIGTERM)
    assert calls == [True] and report['status'] == 'STOPPED'
    assert report['known_actual_counts'] == common.expected_counts(270)
