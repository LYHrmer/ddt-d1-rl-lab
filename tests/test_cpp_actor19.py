"""Fake transport/process checks only: never start C++, ORT, a model or physics."""
from __future__ import annotations

import ast
import io
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools19'))
import cpp_actor19 as adapter  # noqa: E402


class Process:
    pid = 99999999
    returncode = None

    def __init__(self):
        self.stdin, self.stdout = io.BytesIO(), io.BytesIO()

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0
        return 0


class Transport:
    def __init__(self, actor, fault=None):
        self.actor, self.fault, self.started, self.ended = actor, fault, False, False
        self.requests = []

    def write(self, data, timeout):
        assert 0 < timeout <= 10
        if self.fault == 'write_timeout':
            raise TimeoutError('synthetic write timeout')
        self.requests.append(json.loads(data))

    def read(self, timeout):
        assert timeout > 0
        if not self.started:
            self.started = True
            value = {'format': 'd1-cpp-actor19-ready-v1', 'status': 'READY',
                     **{key: self.actor.identities[key] for key in
                        ('binary_sha256', 'model_sha256', 'source_sha256', 'ort_library_sha256')},
                     'signature': adapter.SIGNATURE, 'onnxruntime_version': '1.10.0',
                     'cpu_threads': 1, 'max_requests': self.actor.max_requests,
                     'inference_attempts': 0, 'inference_returns': 0,
                     'robot_models_constructed': 0, 'physics_steps_executed': 0}
            if self.fault == 'startup_identity':
                value['model_sha256'] = '0'*64
            elif self.fault == 'startup_signature':
                value['signature'] = {'wrong': True}
            elif self.fault == 'startup_counts':
                value['physics_steps_executed'] = True
            elif self.fault == 'startup_timeout':
                raise TimeoutError('synthetic startup timeout')
        elif self.ended:
            value = {'format': 'd1-cpp-actor19-closed-v1', 'status': 'CLOSED',
                     'requests': len(self.requests), 'inference_attempts': len(self.requests),
                     'inference_returns': len(self.requests), 'robot_models_constructed': 0,
                     'physics_steps_executed': 0}
            if self.fault == 'bad_eof':
                value['inference_returns'] += 1
        else:
            if self.fault == 'read_timeout':
                raise TimeoutError('synthetic response timeout')
            if self.fault == 'early_eof':
                raise EOFError('synthetic missing response')
            if self.fault == 'bad_json':
                return b'not a protocol line'
            if self.fault == 'duplicate_key':
                return b'{"status":"OK","status":"FAIL"}'
            request = self.requests[-1]
            count = len(self.requests)
            value = {'format': 'd1-cpp-actor19-response-v1', 'status': 'OK', 'seq': request['seq'],
                     'actions': [0.]*16, 'inference_attempts': count, 'inference_returns': count}
            if self.fault == 'sequence':
                value['seq'] += 1
            elif self.fault == 'counter':
                value['inference_returns'] -= 1
            elif self.fault == 'extra_field':
                value['unrequested'] = 1
            elif self.fault == 'action_shape':
                value['actions'].pop()
            elif self.fault == 'action_range':
                value['actions'][0] = 1.01
            elif self.fault == 'action_bool':
                value['actions'][0] = True
            elif self.fault == 'action_nan':
                return json.dumps(dict(value, actions=[float('nan')]*16)).encode()
            elif self.fault == 'run_failure':
                value = {'format': 'd1-cpp-actor19-failure-v1', 'status': 'FAIL',
                         'error': 'synthetic Run error', 'accepted_requests': count-1,
                         'inference_attempts': count, 'inference_returns': count-1}
        return json.dumps(value).encode()

    def close_input(self):
        self.ended = True


@pytest.fixture
def np():
    return pytest.importorskip('numpy')


@pytest.fixture
def make_actor(tmp_path):
    def make(fault=None, *, maximum=3, spawn_signal=None):
        paths = {}
        for name in ('binary', 'model', 'source', 'library'):
            paths[name] = tmp_path/name
            paths[name].write_bytes(('synthetic-'+name).encode())
        process, calls = Process(), {'spawn': 0, 'cleanup': 0}
        def factory(command, **kwargs):
            calls['spawn'] += 1
            calls['command'], calls['kwargs'] = command, kwargs
            if spawn_signal is not None:
                signal.raise_signal(spawn_signal)
            return process
        def cleanup(owned):
            assert owned is process
            calls['cleanup'] += 1
            if fault == 'cleanup_exception':
                raise RuntimeError('synthetic cleanup failure')
            if owned.returncode is None:
                owned.returncode = -15
            return ['SIGTERM'] if owned.returncode else [], fault != 'cleanup_incomplete'
        actor = adapter.CppActor19(paths['binary'], paths['model'], output=tmp_path/'output',
            expected_binary_sha256=adapter.sha(paths['binary']), expected_model_sha256=adapter.sha(paths['model']),
            expected_source_sha256=adapter.sha(paths['source']), source=paths['source'], ort_library=paths['library'],
            max_requests=maximum, process_factory=factory, transport_factory=lambda _: transport,
            cleanup_function=cleanup)
        transport = Transport(actor, fault)
        return actor, transport, process, calls, paths
    return make


def test_constructor_is_cold_and_start_uses_only_its_new_owned_session(make_actor):
    actor, _, _, calls, _ = make_actor()
    assert calls['spawn'] == 0 and not actor.output.exists()
    actor.start()
    assert calls['spawn'] == 1 and calls['kwargs']['start_new_session'] is True
    assert actor.stats['status'] == 'READY'
    assert actor.close()['status'] == 'CLOSED' and calls['cleanup'] == 1
    assert actor.stats['eof_receipt']['requests'] == 0


def test_actual_observation_and_previous_history_are_transmitted_without_hidden_state(make_actor, np):
    actor, transport, _, calls, _ = make_actor()
    actor.start()
    first = np.arange(627, dtype=np.float32)/627
    second = first.copy()
    second[57:] *= -1
    before = second.copy()
    for item in (first, second):
        action = actor.predict(item)
        assert action.shape == (16,) and action.dtype == np.float32
    assert transport.requests[0]['actor'] == first.tolist()
    assert transport.requests[1]['actor'] == second.tolist()
    assert transport.requests[0]['seq'] == 0 and transport.requests[1]['seq'] == 1
    assert np.array_equal(before, second)
    report = actor.close()
    assert report['status'] == 'CLOSED' and report['requests_completed'] == 2
    assert report['inference_attempts'] == report['inference_returns'] == 2
    assert report['cleanup_complete'] and calls['cleanup'] == 1
    assert (actor.output/'stdout.jsonl').read_text().count('\n') == 4


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'batch', 'nan', 'list'])
def test_invalid_observation_is_terminal_before_any_request(make_actor, np, bad):
    actor, transport, _, calls, _ = make_actor()
    actor.start()
    value = np.zeros(627, dtype=np.float32)
    if bad == 'dtype':
        value = value.astype(np.float64)
    elif bad == 'shape':
        value = value[:-1]
    elif bad == 'batch':
        value = value.reshape(1, 627)
    elif bad == 'nan':
        value[0] = np.nan
    else:
        value = value.tolist()
    with pytest.raises(ValueError, match='finite float32'):
        actor.predict(value)
    assert not transport.requests and calls['cleanup'] == 1
    assert actor.stats['status'] == 'FAIL' and actor.stats['requests_attempted'] == 0
    with pytest.raises(ValueError, match='not ready'):
        actor.predict(np.zeros(627, dtype=np.float32))


@pytest.mark.parametrize('fault', ['startup_identity', 'startup_signature', 'startup_counts', 'startup_timeout'])
def test_startup_must_prove_actual_identity_signature_and_zero_counts(make_actor, fault):
    actor, _, _, calls, _ = make_actor(fault)
    with pytest.raises((ValueError, TimeoutError)):
        actor.start()
    assert actor.stats['status'] == 'FAIL' and calls['cleanup'] == 1
    with pytest.raises(ValueError, match='restarted'):
        actor.start()
    assert json.loads((actor.output/'report.json').read_text())['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['write_timeout', 'read_timeout', 'early_eof', 'bad_json', 'duplicate_key',
    'sequence', 'counter', 'extra_field', 'action_shape', 'action_range', 'action_bool', 'action_nan', 'run_failure'])
def test_per_request_failure_is_terminal_and_unknown_run_counts_are_not_zero(make_actor, np, fault):
    actor, _, _, calls, _ = make_actor(fault)
    actor.start()
    with pytest.raises((ValueError, TimeoutError, EOFError)):
        actor.predict(np.zeros(627, dtype=np.float32))
    report = actor.close()
    assert report['status'] == 'FAIL' and report['requests_attempted'] == 1
    assert report['requests_completed'] == 0 and calls['cleanup'] == 1
    if fault in ('read_timeout', 'write_timeout', 'early_eof'):
        assert report['inference_attempts'] is report['inference_returns'] is None
    if fault == 'run_failure':
        assert report['inference_attempts'] == 1 and report['inference_returns'] == 0


def test_request_cap_refuses_a_fourth_request_without_restarting(make_actor, np):
    actor, transport, _, calls, _ = make_actor(maximum=3)
    actor.start()
    for _ in range(3):
        actor.predict(np.zeros(627, dtype=np.float32))
    with pytest.raises(ValueError, match='cap exhausted'):
        actor.predict(np.zeros(627, dtype=np.float32))
    assert len(transport.requests) == 3 and calls['spawn'] == calls['cleanup'] == 1


@pytest.mark.parametrize('fault', ['bad_eof', 'cleanup_exception', 'cleanup_incomplete'])
def test_final_eof_and_cleanup_both_gate_success(make_actor, fault):
    actor, _, _, calls, _ = make_actor(fault)
    actor.start()
    report = actor.close()
    assert report['status'] == 'FAIL' and calls['cleanup'] == 1
    if fault.startswith('cleanup'):
        assert report['cleanup_complete'] is False


@pytest.mark.parametrize('name', ['model', 'binary', 'source', 'library'])
def test_changed_inputs_are_detected_at_final_close(make_actor, name):
    actor, _, _, _, paths = make_actor()
    actor.start()
    paths[name].write_bytes(b'changed')
    assert actor.close()['status'] == 'FAIL'


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_signal_during_popen_handle_return_is_deferred_then_cleans(make_actor, signum):
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    actor, _, _, calls, _ = make_actor(spawn_signal=signum)
    with pytest.raises(ValueError, match='termination during'):
        actor.start()
    assert calls['spawn'] == calls['cleanup'] == 1 and actor.stats['status'] == 'FAIL'
    assert old == {sig: signal.getsignal(sig) for sig in old}


def test_existing_output_is_not_overwritten_or_started(make_actor):
    actor, _, _, calls, _ = make_actor()
    actor.output.mkdir()
    sentinel = actor.output/'keep'
    sentinel.write_text('keep')
    with pytest.raises(ValueError, match='no overwrite'):
        actor.start()
    assert sentinel.read_text() == 'keep' and calls['spawn'] == 0


@pytest.fixture
def fake_build_inputs(tmp_path, monkeypatch):
    paths = [tmp_path/name for name in ('compiler', 'ort_library', 'header')]
    for path in paths:
        path.write_bytes(b'synthetic compile-only input')
    monkeypatch.setattr(adapter, 'ORT_LIBRARY', paths[1])
    monkeypatch.setattr(adapter, 'HEADERS', (paths[2],))
    return str(paths[0])


def test_compile_entry_builds_only_and_binds_command_and_actual_binary(tmp_path, fake_build_inputs):
    seen = []
    def runner(command, **kwargs):
        seen.append(command)
        Path(command[-1]).write_bytes(b'synthetic compiled file')
        assert kwargs['timeout'] == 120
        return SimpleNamespace(returncode=0)
    report = adapter.build_server(tmp_path/'build', compiler=fake_build_inputs, runner=runner)
    assert report['status'] == 'PASS' and report['server_started'] is False
    assert report['onnx_sessions_created'] == report['physics_steps_executed'] == 0
    assert '-O1' in seen[0] and '-lonnxruntime' in seen[0] and '-lcrypto' in seen[0]
    assert not any('mujoco' in item or 'rclcpp' in item for item in seen[0])
    assert report['binary_sha256'] == adapter.sha(tmp_path/'build/actor_server19')
    with pytest.raises(ValueError, match='new'):
        adapter.build_server(tmp_path/'build', compiler=fake_build_inputs, runner=runner)


def test_compile_failure_still_has_a_receipt(tmp_path, fake_build_inputs):
    report = adapter.build_server(tmp_path/'build', compiler=fake_build_inputs,
                                  runner=lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert report['status'] == 'FAIL' and report['binary_sha256'] is None
    assert (tmp_path/'build/build.json').is_file()


def test_transport_module_has_no_heavy_imports_at_module_scope():
    tree = ast.parse((ROOT/'tools19/cpp_actor19.py').read_text())
    top_imports = {alias.name.split('.')[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert not top_imports & {'torch', 'mujoco', 'numpy', 'onnxruntime', 'stable_baselines3', 'rclpy'}


def test_real_pipe_transport_reads_separate_lines_and_refuses_missing_final_line():
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    process = SimpleNamespace(stdin=os.fdopen(input_write, 'wb', buffering=0),
                              stdout=os.fdopen(output_read, 'rb', buffering=0))
    try:
        transport = adapter.PipeTransport(process)
        os.write(output_write, b'{"first":1}\n{"second":2}\npartial')
        assert transport.read(.1) == b'{"first":1}'
        assert transport.read(.1) == b'{"second":2}'
        os.close(output_write)
        output_write = None
        with pytest.raises(EOFError):
            transport.read(.1)
    finally:
        process.stdin.close()
        process.stdout.close()
        os.close(input_read)
        if output_write is not None:
            os.close(output_write)


def test_real_pipe_transport_bounds_an_empty_read_and_a_blocked_write():
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    process = SimpleNamespace(stdin=os.fdopen(input_write, 'wb', buffering=0),
                              stdout=os.fdopen(output_read, 'rb', buffering=0))
    try:
        transport = adapter.PipeTransport(process)
        with pytest.raises(TimeoutError, match='response timed out'):
            transport.read(.005)
        with pytest.raises(TimeoutError, match='write timed out'):
            transport.write(b'x'*(1 << 20), .005)
    finally:
        process.stdin.close()
        process.stdout.close()
        os.close(input_read)
        os.close(output_write)
