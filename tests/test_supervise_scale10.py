"""Exercise actual supervision with tiny Python children; no model or physics imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools10'))
import supervise_scale10 as supervisor  # noqa: E402

GIB = 1024**3
FAKE = '''
import json, pathlib, signal, subprocess, sys, time
out = pathlib.Path(sys.argv[sys.argv.index('--output')+1]); out.mkdir(parents=True)
steps = int(sys.argv[sys.argv.index('--steps')+1]); envs = int(sys.argv[sys.argv.index('--envs')+1])
mode = MODE
if mode in ('hang', 'orphan', 'graceful'):
    signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child_code = 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
    if mode == 'graceful': child_code = 'import signal,time; signal.signal(signal.SIGINT,signal.SIG_DFL); signal.signal(signal.SIGTERM,signal.SIG_DFL); time.sleep(30)'
    child = subprocess.Popen([sys.executable, '-c', child_code])
    (out/'child_pid.txt').write_text(str(child.pid))
row = {'rollout': 1, 'transitions': 4096, 'actual_counts': {'native_steps': 40960},
       'reward_contribution_mean_per_tick': {'alive': .005}}
if mode in ('mismatch', 'late_mismatch'): row['transitions'] = 7
(out/'ledger.jsonl').write_text(json.dumps(row)+'\\n')
worker = {'policy_transitions': steps//envs, 'native_steps': steps//envs*10,
          'pd_updates': steps//envs*5, 'resets': 1, 'forward_calls': steps//envs*10+2}
counts = {key: value*envs for key, value in worker.items()}
result = {'status': 'COMPLETE', 'inputs_unchanged': True, 'error': None,
          'trained_transitions': steps, 'actual_counts': counts, 'per_worker_counts': [worker]*envs}
if mode == 'partial': result['status'] = 'RAM_RESERVE_BELOW_3_GIB'
if mode == 'bad_counts': result['actual_counts']['native_steps'] -= 1
if mode == 'changed_inputs': result['inputs_unchanged'] = False
if mode == 'missing_counts': result['actual_counts'] = None; result['per_worker_counts'] = None
if mode == 'tamper': pathlib.Path(sys.argv[sys.argv.index('--config')+1]).write_text('changed during run')
if mode == 'delete_input': pathlib.Path(sys.argv[sys.argv.index('--config')+1]).unlink()
(out/'result.json').write_text(json.dumps(result))
if mode == 'graceful':
    def stop(*_):
        time.sleep(.03)
        result['worker_alive_for_counter_collection'] = child.poll() is None
        child.terminate(); child.wait(timeout=1)
        result['status'] = 'PARTIAL'
        (out/'result.json').write_text(json.dumps(result)); sys.exit(1)
    signal.signal(signal.SIGINT, stop)
if mode in ('hang', 'mismatch', 'graceful'): time.sleep(30)
if mode == 'exit7': sys.exit(7)
'''


def memory():
    return {'available_ram_bytes': 6*GIB, 'cgroup_remaining_bytes': None}


def setup_release(tmp_path, mode='complete', max_wall=2.):
    (tmp_path/'fake.py').write_text(FAKE.replace('MODE', repr(mode)))
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'docs').mkdir()
    (tmp_path/'docs/career_execution_contract_10.md').write_text('New dummy-only contract, no physics.')
    (tmp_path/'tools08').mkdir()
    (tmp_path/supervisor.COMPARATOR).write_bytes((ROOT/supervisor.COMPARATOR).read_bytes())
    row = {'rollout': 1, 'transitions': 4096, 'actual_counts': {'native_steps': 40960},
           'reward_contribution_mean_per_tick': {'alive': .005}}
    (tmp_path/'frozen.jsonl').write_text(json.dumps(row)+'\n')
    inputs = ['fake.py', 'config.json', 'frozen.jsonl', 'docs/career_execution_contract_10.md',
              supervisor.COMPARATOR]
    release = {'format': supervisor.RELEASE_FORMAT, 'contract_id': 10,
               'status': 'FROZEN_BEFORE_TRAINING', 'supervisor_sha256': supervisor.sha(supervisor.__file__),
               'frozen_inputs': {name: supervisor.sha(tmp_path/name) for name in inputs},
               'entry': 'fake.py', 'config': 'config.json',
               'contract': 'docs/career_execution_contract_10.md', 'seed': 101, 'envs': 8,
               'plans': {kind: {'steps': 4096, 'max_wall': max_wall, 'compare_rollouts': 1,
                                'frozen_ledger': 'frozen.jsonl',
                                'output': f'evidence/career_contract_10/{kind}_01'}
                         for kind in ('preflight', 'scale')},
               'resources': {'start_min_available_ram_bytes': 5*GIB, 'poll_s': .01,
                             'interrupt_grace_s': .05, 'terminate_grace_s': .05,
                             'kill_grace_s': .3, 'process_wall_grace_s': 0.}}
    path = tmp_path/'release.json'; path.write_text(json.dumps(release))
    return path, release


def read_report(tmp_path, kind='preflight'):
    path = tmp_path/f'evidence/career_contract_10/{kind}_01_supervisor/report.json'
    return json.loads(path.read_text())


def test_success_requires_full_training_and_per_worker_accounting(tmp_path):
    path, _ = setup_release(tmp_path)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 0
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
    report = read_report(tmp_path)
    assert report['status'] == 'PASS' and report['trainer_returncode'] == 0
    assert report['equivalence'] == 'PASS' and report['cleanup_complete'] is True
    assert report['resource_reservation_guaranteed'] is False
    assert report['minimum_observed_available_ram_bytes'] == 6*GIB


@pytest.mark.parametrize('mode', ['exit7', 'partial', 'bad_counts', 'changed_inputs', 'missing_counts', 'tamper', 'delete_input'])
def test_failure_cannot_be_hidden_by_complete_label_or_equivalence(tmp_path, mode):
    path, _ = setup_release(tmp_path, mode)
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
    report = read_report(tmp_path)
    assert report['equivalence'] == 'PASS' and report['status'] == 'FAIL'
    if mode == 'exit7':
        assert report['trainer_result']['status'] == 'COMPLETE' and report['trainer_returncode'] == 7
        assert report['stop_reason'] == 'TRAINER_EXIT_NONZERO'
    if mode == 'missing_counts':
        assert report['trainer_result']['actual_counts'] is None
    if mode in ('tamper', 'delete_input'):
        assert report['stop_reason'] == 'FROZEN_INPUTS_CHANGED' and report['inputs_unchanged'] is False
        assert report['trainer_result']['actual_counts']['policy_transitions'] == 4096


def test_scale_requires_its_own_successful_preflight(tmp_path):
    path, _ = setup_release(tmp_path)
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 0
    receipt = tmp_path/'evidence/career_contract_10/preflight_01_supervisor/report.json'
    assert supervisor.run('scale', path, receipt, root=tmp_path, memory_reader=memory) == 0


@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('trainer_returncode', 7),
                                         ('inputs_unchanged', False), ('cleanup_complete', False),
                                         ('release_sha256', '0'*64), ('kind', 'scale'),
                                         ('supervisor_sha256', '0'*64), ('rollouts_compared', 0),
                                         ('stop_reason', 'RAM_RESERVE_BELOW_3_GIB')])
def test_failed_or_unrelated_preflight_never_spawns_scale(tmp_path, monkeypatch, field, value):
    path, _ = setup_release(tmp_path)
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 0
    receipt = tmp_path/'evidence/career_contract_10/preflight_01_supervisor/report.json'
    report = json.loads(receipt.read_text()); report[field] = value
    receipt.write_text(json.dumps(report))
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('scale must not spawn'))
    with pytest.raises(SystemExit, match='preflight'):
        supervisor.run('scale', path, receipt, root=tmp_path, memory_reader=memory)
    assert not (tmp_path/'evidence/career_contract_10/scale_01_supervisor').exists()


def test_a_changed_raw_preflight_result_is_rejected(tmp_path):
    path, _ = setup_release(tmp_path)
    supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)
    result_path = tmp_path/'evidence/career_contract_10/preflight_01/result.json'
    result = json.loads(result_path.read_text()); result['status'] = 'RAM_RESERVE_BELOW_3_GIB'
    result_path.write_text(json.dumps(result))
    receipt = tmp_path/'evidence/career_contract_10/preflight_01_supervisor/report.json'
    with pytest.raises(SystemExit, match='preflight'):
        supervisor.run('scale', path, receipt, root=tmp_path, memory_reader=memory)


def test_changed_frozen_source_or_old_output_is_rejected_before_spawn(tmp_path, monkeypatch):
    path, release = setup_release(tmp_path)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
    original = (tmp_path/'fake.py').read_bytes(); (tmp_path/'fake.py').write_text('changed')
    with pytest.raises(SystemExit, match='frozen input'):
        supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)
    (tmp_path/'fake.py').write_bytes(original)
    release['plans']['preflight']['output'] = 'evidence/career_contract_09/retry'
    path.write_text(json.dumps(release))
    with pytest.raises(SystemExit, match='contract 10'):
        supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)


def test_existing_output_or_attempt_is_never_reused(tmp_path):
    path, release = setup_release(tmp_path)
    supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)
    with pytest.raises(SystemExit, match='already exists'):
        supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)
    output = tmp_path/release['plans']['preflight']['output']
    for item in output.iterdir(): item.unlink()
    output.rmdir()
    with pytest.raises(SystemExit, match='spawn attempt already exists'):
        supervisor.run('preflight', path, root=tmp_path, memory_reader=memory)


def test_not_ready_is_recorded_without_consuming_the_spawn_attempt(tmp_path, monkeypatch):
    path, _ = setup_release(tmp_path)
    low = lambda: {'available_ram_bytes': 4*GIB, 'cgroup_remaining_bytes': None}
    with monkeypatch.context() as patch:
        patch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
        assert supervisor.run('preflight', path, root=tmp_path, memory_reader=low) == 2
    assert not (tmp_path/'evidence/career_contract_10/preflight_01_supervisor').exists()
    receipts = list((tmp_path/'evidence/career_contract_10/readiness_checks').glob('*.json'))
    report = json.loads(receipts[0].read_text())
    assert report['status'] == 'NOT_READY' and report['training_started'] is False
    assert report['minimum_observed_available_ram_bytes'] == 4*GIB
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 0


def test_runtime_low_memory_sample_and_minimum_are_persisted(tmp_path):
    path, _ = setup_release(tmp_path, 'hang')
    values = iter([6*GIB, 4*GIB, supervisor.RESERVE-1])
    def changing_memory():
        return {'available_ram_bytes': next(values, 6*GIB), 'cgroup_remaining_bytes': None}
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=changing_memory) == 1
    report = read_report(tmp_path)
    assert report['stop_reason'] == 'RAM_RESERVE_BELOW_3_GIB'
    assert report['stop_sample']['available_ram_bytes'] == supervisor.RESERVE-1
    assert report['minimum_observed_available_ram_bytes'] == supervisor.RESERVE-1
    assert report['cleanup_complete'] is True
    samples = (tmp_path/'evidence/career_contract_10/preflight_01_supervisor/resources.jsonl').read_text()
    assert str(supervisor.RESERVE-1) in samples


def test_cgroup_limit_can_refuse_spawn_even_with_plenty_of_host_ram(tmp_path, monkeypatch):
    path, _ = setup_release(tmp_path)
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not spawn'))
    limited = lambda: {'available_ram_bytes': 6*GIB, 'cgroup_remaining_bytes': 2*GIB}
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=limited) == 2
    receipt = next((tmp_path/'evidence/career_contract_10/readiness_checks').glob('*.json'))
    report = json.loads(receipt.read_text())
    assert report['stop_reason'] == 'START_RAM_HEADROOM' and report['trainer_returncode'] is None
    assert report['stop_sample']['effective_available_bytes'] == 2*GIB


def test_ignored_interrupt_and_terminate_are_bounded_and_only_own_group_is_killed(tmp_path):
    path, _ = setup_release(tmp_path, 'hang', max_wall=.25)
    unrelated = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
                                 start_new_session=True)
    try:
        assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
        report = read_report(tmp_path)
        assert report['stop_reason'] == 'PROCESS_WALL_LIMIT'
        assert report['cleanup_signals'] == ['SIGINT', 'SIGTERM', 'SIGKILL']
        assert report['cleanup_complete'] is True and report['wall_s'] < 3
        assert unrelated.poll() is None
        child = int((tmp_path/'evidence/career_contract_10/preflight_01/child_pid.txt').read_text())
        if Path(f'/proc/{child}/stat').exists():
            assert Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    finally:
        unrelated.terminate(); unrelated.wait(timeout=2)


@pytest.mark.parametrize('mode', ['mismatch', 'late_mismatch'])
def test_equivalence_mismatch_stops_or_rejects_the_child(tmp_path, mode):
    path, _ = setup_release(tmp_path, mode)
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
    report = read_report(tmp_path)
    assert report['stop_reason'] == 'EQUIVALENCE_MISMATCH' and report['equivalence'] == 'FAIL'
    assert report['first_mismatch']['rollout'] == 1 and report['cleanup_complete'] is True


def test_resource_reader_failure_still_records_failure_and_cleans_up(tmp_path):
    path, _ = setup_release(tmp_path, 'hang')
    calls = 0
    def broken_reader():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError('injected memory read failure')
        return memory()
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=broken_reader) == 1
    report = read_report(tmp_path)
    assert report['stop_reason'] == 'SUPERVISOR_ERROR' and 'memory read failure' in report['error']
    assert report['cleanup_complete'] is True


def test_an_orphaned_worker_cannot_turn_a_zero_exit_into_success(tmp_path):
    path, _ = setup_release(tmp_path, 'orphan')
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
    report = read_report(tmp_path)
    assert report['trainer_returncode'] == 0
    assert report['stop_reason'] == 'RESIDUAL_PROCESS_GROUP' and report['cleanup_complete'] is True


def test_failed_initial_resource_read_is_unknown_and_does_not_consume_spawn(tmp_path):
    path, _ = setup_release(tmp_path)
    def failed_read():
        raise OSError('unavailable MemAvailable')
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=failed_read) == 2
    receipt = next((tmp_path/'evidence/career_contract_10/readiness_checks').glob('*.json'))
    report = json.loads(receipt.read_text())
    assert report['stop_reason'] == 'RESOURCE_READ_FAILED'
    assert report['minimum_observed_available_ram_bytes'] is None and report['stop_sample'] is None
    assert not (tmp_path/'evidence/career_contract_10/preflight_01_supervisor').exists()


def test_graceful_interrupt_leaves_workers_alive_for_final_counter_collection(tmp_path):
    path, release = setup_release(tmp_path, 'graceful', max_wall=.25)
    release['resources']['interrupt_grace_s'] = .3
    path.write_text(json.dumps(release))
    assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
    report = read_report(tmp_path)
    assert report['stop_reason'] == 'PROCESS_WALL_LIMIT'
    assert report['trainer_result']['worker_alive_for_counter_collection'] is True
    assert report['cleanup_signals'] == ['SIGINT'] and report['cleanup_complete'] is True


def test_external_sigterm_cleans_own_training_group_and_writes_failed_receipt(tmp_path):
    path, _ = setup_release(tmp_path, 'hang', max_wall=10)
    code = ('import sys; sys.path.insert(0, '+repr(str(ROOT/'tools10'))+'); '
            'import supervise_scale10 as s; '
            'raise SystemExit(s.run("preflight", '+repr(str(path))+', root='+repr(str(tmp_path))+', '
            'memory_reader=lambda: {"available_ram_bytes":6*1024**3,"cgroup_remaining_bytes":None}))')
    monitor = subprocess.Popen([sys.executable, '-B', '-c', code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    group = None
    try:
        marker = tmp_path/'evidence/career_contract_10/preflight_01/child_pid.txt'
        deadline = time.monotonic()+3
        while not marker.exists() and time.monotonic() < deadline and monitor.poll() is None:
            time.sleep(.01)
        assert marker.exists(), 'dummy training did not start'
        child = int(marker.read_text())
        group = int(Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()[2])
        monitor.terminate()
        stdout, stderr = monitor.communicate(timeout=3)
        assert monitor.returncode == 1, (stdout, stderr)
        report = read_report(tmp_path)
        assert report['stop_reason'] == 'SUPERVISOR_INTERRUPTED' and 'SIGTERM' in report['error']
        assert report['cleanup_complete'] is True and supervisor.live_group_members(group) == []
    finally:
        if group is not None and supervisor.live_group_members(group):
            os.killpg(group, signal.SIGKILL)
        if monitor.poll() is None:
            monitor.kill(); monitor.wait(timeout=2)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_spawn_does_not_lose_the_new_process_handle(tmp_path, monkeypatch, signum):
    path, _ = setup_release(tmp_path, 'hang')
    original_popen, spawned = supervisor.subprocess.Popen, []
    def signal_during_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        os.kill(os.getpid(), signum)
        return process
    monkeypatch.setattr(supervisor.subprocess, 'Popen', signal_during_popen)
    try:
        assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
        report = read_report(tmp_path)
        assert report['stop_reason'] == 'SUPERVISOR_INTERRUPTED'
        assert report['cleanup_complete'] is True
        assert supervisor.live_group_members(spawned[0].pid) == []
    finally:
        for process in spawned:
            if supervisor.live_group_members(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_during_bounded_cleanup_cannot_abandon_the_child(tmp_path, monkeypatch, signum):
    path, _ = setup_release(tmp_path, 'hang', max_wall=.15)
    original_stop, children = supervisor.stop_group, []
    def interrupt_cleanup(process, resources):
        children.append(process)
        os.kill(os.getpid(), signum)
        return original_stop(process, resources)
    monkeypatch.setattr(supervisor, 'stop_group', interrupt_cleanup)
    try:
        assert supervisor.run('preflight', path, root=tmp_path, memory_reader=memory) == 1
        assert read_report(tmp_path)['cleanup_complete'] is True
    finally:
        for process in children:
            if supervisor.live_group_members(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
