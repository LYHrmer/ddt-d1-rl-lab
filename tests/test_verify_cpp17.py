"""Fake C++ runner only; no native executable, model or physics is run."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'tools17/verify_cpp17.py'
SPEC = importlib.util.spec_from_file_location('verify_cpp17_test_entry', ENTRY)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


class API:
    def __init__(self, root):
        self.ROOT = root
        self.source = root / 'source.txt'
        self.source.write_text('synthetic committed source')
        self.proof = {'git_commit': 'a' * 40, 'sources_sha256': {'source.txt': self.sha256(self.source)}}
        for name in (verify.BINARY, verify.SCHEMA):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic frozen ' + name)
        self.release = {'output_root': 'evidence/local_comparison16/execution_01',
                        'sources_sha256': {name: self.sha256(root / name) for name in (verify.BINARY, verify.SCHEMA)}}
        self.training = root / 'completed-training'
        base = self.training / 'evidence/local_comparison16'
        out = self.training / self.release['output_root'] / 'train_control'
        out.mkdir(parents=True)
        self.checkpoint = out / 'policy.zip'
        self.checkpoint.write_bytes(b'published completed aux0 control weights; fake fixture')
        self.protocol = {'random_initialization': True, 'warm_start': False, 'teacher_policy_used': False}
        self.result = {'status': 'COMPLETE', 'arm': 'control', 'aux_coef': 0.,
                       'eligible_for_evaluation': True, 'checkpoint_sha256': self.sha256(self.checkpoint)}
        write(base / 'release_01.json', self.release)
        write(out / 'protocol.json', self.protocol)
        write(out / 'result.json', self.result)
        write(out.parent / 'report.json', {'status': 'COMPLETE'})
    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def source_identity(self):
        self.check_sources(self.proof)
        return copy.deepcopy(self.proof)
    def check_sources(self, proof):
        if proof != self.proof or self.sha256(self.source) != self.proof['sources_sha256']['source.txt']:
            raise ValueError('source changed')
    def load_completed_control(self, training_root):
        if Path(training_root) != self.training or self.sha256(self.checkpoint) != self.result['checkpoint_sha256']:
            raise ValueError('not the published completed control checkpoint')
        return self.release, self.protocol, self.result, self.checkpoint
    @staticmethod
    def atomic_json(path, value):
        write(path, value)


class FakeRunner:
    def __init__(self, api):
        self.api, self.calls = api, []
        self.change = None
        self.execution = {'process_started': True, 'returncode': 0, 'timed_out': False,
                          'cleanup_complete': True, 'cleanup_signals': [], 'error': None}
    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        native_dir = Path(command[command.index('--output') + 1])
        native_dir.mkdir()
        policy = Path(command[command.index('--policy') + 1])
        fixture = Path(command[command.index('--onnx-golden') + 1])
        schema = Path(command[command.index('--schema') + 1])
        rows = verify.read_json(fixture)['fixtures']
        result = {'status': 'PASS', 'policy_sha256': self.api.sha256(policy),
                  'fixture_sha256': self.api.sha256(fixture), 'runtime_binary_sha256': self.api.sha256(command[0]),
                  'schema_sha256': self.api.sha256(schema), 'mujoco_version': '3.3.0', 'onnxruntime_version': '1.10.0',
                  'physics_steps': 0, 'native_steps_attempted': 0, 'native_steps_completed': 0,
                  'pd_updates': 0, 'policy_updates': 0, 'onnx_signature': verify.SIGNATURE,
                  'onnx_inference_calls': 64, 'absolute_tolerance': 1e-5, 'max_absolute_error': 0.,
                  'comparisons': [{'index': row['index'], 'actions': row['torch_actions'],
                                   'max_absolute_error': 0., 'passed': True} for row in rows]}
        if self.change:
            self.change(result)
        # Intentionally permit malicious nonfinite JSON bytes for validator tests.
        (native_dir / 'report.json').write_text(json.dumps(result, allow_nan=True))
        Path(kwargs['stdout_path']).write_text('fake native output\n')
        Path(kwargs['stderr_path']).write_text('')
        return dict(self.execution)


@pytest.fixture
def data(tmp_path):
    api = API(tmp_path)
    export = tmp_path / 'export'
    export.mkdir()
    (export / 'policy.onnx').write_bytes(b'fake ONNX bytes; never loaded')
    history = [[0.] * 57 for _ in range(10)]
    fixtures = {'format': 'd1-onnx-two-input-golden-64-v1', 'count': 64,
                'fixtures': [{'index': i, 'obs': [0.] * 57, 'history_oldest_to_newest': history,
                              'history_oldest_to_newest_previous_only': history, 'torch_actions': [0.] * 16}
                             for i in range(64)]}
    write(export / 'golden_fixtures.json', fixtures)
    execution = api.training / api.release['output_root']
    identity = {'checkpoint_path': str(api.checkpoint), 'checkpoint_sha256': api.sha256(api.checkpoint),
                'training_protocol_sha256': api.sha256(api.checkpoint.parent / 'protocol.json'),
                'training_result_sha256': api.sha256(api.checkpoint.parent / 'result.json'),
                'release_sha256': api.sha256(api.training / 'evidence/local_comparison16/release_01.json'),
                'execution_report_sha256': api.sha256(execution / 'report.json')}
    report = {'format': 'd1-completed-control-onnx-export-v17', 'status': 'PASS', 'error': None,
              'checkpoint_sha256': api.sha256(api.checkpoint), 'training_identity': identity,
              'training_root': str(api.training), 'source_identity': api.source_identity(),
              'onnx_sha256': api.sha256(export / 'policy.onnx'),
              'fixtures_sha256': api.sha256(export / 'golden_fixtures.json'),
              'completed_checkpoint_provenance_checked': True, 'physics_steps_executed': 0,
              'robot_models_constructed': 0, 'absolute_limits': {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5},
              'parity': {'sb3_predict_vs_torch_max_abs_error': 0., 'torch_vs_python_ort_max_abs_error': 0.,
                         'fixtures': 64, 'prediction_batch_size': 1}}
    write(export / 'report.json', report)
    return SimpleNamespace(api=api, export=export, output=tmp_path / 'cpp', runner=FakeRunner(api),
                           export_report=report, fixtures=fixtures)


def run(data):
    return verify.run(data.export, data.output, api=data.api, runner=data.runner)


def test_success_binds_frozen_binary_real_control_and_zero_physics(data):
    result = run(data)
    assert result['status'] == 'PASS' and result['physics_steps_executed'] == 0
    assert result['robot_models_constructed'] == 0 and result['comparisons'] == result['onnx_inference_calls'] == 64
    assert result['checkpoint_sha256'] == data.api.sha256(data.api.checkpoint)
    assert result['onnx_sha256'] == data.export_report['onnx_sha256']
    assert result['fixtures_sha256'] == data.export_report['fixtures_sha256']
    assert result['runtime_binary_sha256'] == data.api.sha256(data.api.ROOT / verify.BINARY)
    assert result['schema_sha256'] == data.api.sha256(data.api.ROOT / verify.SCHEMA)
    assert result['cxx_ort_1_10_status'] == 'PASS' and result['training_started'] is False
    command, kwargs = data.runner.calls[0]
    assert command[0] == str(data.api.ROOT / verify.BINARY)
    assert '--model' not in command and '--validate-only' not in command and '--mode' not in command
    assert command[-1] == str(data.output / 'cpp_native')
    assert command[command.index('--schema') + 1] == str(data.api.ROOT / verify.SCHEMA)
    assert command[command.index('--expected-policy-sha256') + 1] == data.api.sha256(data.export / 'policy.onnx')
    assert kwargs['timeout_s'] <= 90 and kwargs['env']['OMP_NUM_THREADS'] == '1'
    assert kwargs['env']['LD_LIBRARY_PATH'] == f'/opt/ros/humble/lib:{data.api.ROOT}/install/mujoco/lib'
    assert verify.read_json(data.output / 'report.json') == result


def test_existing_output_cannot_be_overwritten(data):
    data.output.mkdir()
    (data.output / 'sentinel').write_text('old attempt')
    with pytest.raises(FileExistsError):
        run(data)
    assert (data.output / 'sentinel').read_text() == 'old attempt'
    assert not data.runner.calls and not (data.output / 'report.json').exists()


@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('format', 'fake'), ('checkpoint_sha256', 'foreign'),
    ('completed_checkpoint_provenance_checked', False), ('training_root', 'foreign'), ('onnx_sha256', 'bad'),
    ('fixtures_sha256', 'bad'), ('physics_steps_executed', 1), ('robot_models_constructed', 1)])
def test_bad_export_refuses_before_cpp_and_writes_receipt(data, field, value):
    data.export_report[field] = value
    write(data.export / 'report.json', data.export_report)
    result = run(data)
    assert result['status'] == 'FAIL' and result['error']
    assert result['cpp_process_attempted'] is False and not data.runner.calls
    assert result['physics_steps_executed'] == 0
    assert all((data.output / name).exists() for name in ('stdout.txt', 'stderr.txt', 'report.json'))


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'obs_shape', 'obs_bool', 'history_alias', 'alias_bool',
                                   'history_shape', 'actions_shape', 'unclipped', 'nonfinite'])
def test_fixture_signature_indices_and_history_aliases_are_strict(data, fault):
    value = data.fixtures
    if fault == 'duplicate': value['fixtures'][1]['index'] = 0
    if fault == 'missing': value['fixtures'].pop()
    if fault == 'obs_shape': value['fixtures'][0]['obs'].pop()
    if fault == 'obs_bool': value['fixtures'][0]['obs'][0] = True
    if fault == 'history_alias': value['fixtures'][0]['history_oldest_to_newest_previous_only'] = [[1.] * 57] * 10
    if fault == 'alias_bool': value['fixtures'][0]['history_oldest_to_newest_previous_only'] = [[False] * 57] * 10
    if fault == 'history_shape': value['fixtures'][0]['history_oldest_to_newest'] = []
    if fault == 'actions_shape': value['fixtures'][0]['torch_actions'].pop()
    if fault == 'unclipped': value['fixtures'][0]['torch_actions'][0] = 2.
    if fault == 'nonfinite':
        (data.export / 'golden_fixtures.json').write_text('{"fixtures": [NaN]}')
    else:
        write(data.export / 'golden_fixtures.json', value)
    data.export_report['fixtures_sha256'] = data.api.sha256(data.export / 'golden_fixtures.json')
    write(data.export / 'report.json', data.export_report)
    result = run(data)
    assert result['status'] == 'FAIL' and not data.runner.calls


@pytest.mark.parametrize('fault', ['status', 'error', 'policy_sha256', 'fixture_sha256', 'runtime_binary_sha256',
    'schema_sha256', 'onnxruntime_version', 'physics_steps', 'native_steps_attempted', 'pd_updates',
    'policy_updates', 'onnx_inference_calls', 'signature', 'duplicate', 'missing', 'nonfinite',
    'wrong_actions', 'declared_error', 'tolerance', 'passed'])
def test_native_failure_or_forged_success_is_rejected(data, fault):
    def change(value):
        if fault in ('policy_sha256', 'fixture_sha256', 'runtime_binary_sha256', 'schema_sha256'):
            value[fault] = 'wrong'
        if fault == 'status': value['status'] = 'FAIL'
        if fault == 'error': value['error'] = 'native error'
        if fault == 'onnxruntime_version': value[fault] = '1.23.2'
        if fault in ('physics_steps', 'native_steps_attempted', 'pd_updates', 'policy_updates'): value[fault] = 1
        if fault == 'onnx_inference_calls': value[fault] = 63
        if fault == 'signature': value['onnx_signature'] = {}
        if fault == 'duplicate': value['comparisons'][1]['index'] = 0
        if fault == 'missing': value['comparisons'].pop()
        if fault == 'nonfinite': value['comparisons'][0]['actions'][0] = float('inf')
        if fault == 'wrong_actions': value['comparisons'][0]['actions'] = [.001] * 16
        if fault == 'declared_error': value['max_absolute_error'] = 1e-6
        if fault == 'tolerance': value['absolute_tolerance'] = 1e-3
        if fault == 'passed': value['comparisons'][0]['passed'] = False
    data.runner.change = change
    result = run(data)
    assert result['status'] == 'FAIL' and result['cpp_process_attempted'] is True
    assert result['physics_steps_executed'] is None and result['error']


@pytest.mark.parametrize('fault', ['returncode', 'timed_out', 'cleanup_complete', 'cleanup_signals', 'error'])
def test_process_result_overrides_native_pass(data, fault):
    data.runner.execution[fault] = {'returncode': 7, 'timed_out': True, 'cleanup_complete': False,
                                    'cleanup_signals': ['SIGTERM'], 'error': 'failed'}[fault]
    assert run(data)['status'] == 'FAIL'


@pytest.mark.parametrize('fault', ['source', 'onnx', 'checkpoint', 'export_receipt', 'binary'])
def test_input_mutation_during_cpp_cannot_pass(data, fault):
    def change(value):
        path = {'source': data.api.source, 'onnx': data.export / 'policy.onnx',
                'checkpoint': data.api.checkpoint, 'export_receipt': data.export / 'report.json',
                'binary': data.api.ROOT / verify.BINARY}[fault]
        path.write_bytes(path.read_bytes() + b' ')
    data.runner.change = change
    result = run(data)
    assert result['status'] == 'FAIL' and result['error']


def test_import_failure_after_claim_keeps_a_receipt(data, monkeypatch):
    monkeypatch.setattr(verify, 'provenance', lambda: (_ for _ in ()).throw(ImportError('dependency unavailable')))
    result = verify.run(data.export, data.output)
    assert result['status'] == 'FAIL' and 'ImportError' in result['error']
    assert verify.read_json(data.output / 'report.json') == result


def test_cli_uses_real_run_seam_with_fake_runner(data, monkeypatch, capsys):
    monkeypatch.setattr(verify, 'provenance', lambda: data.api)
    monkeypatch.setattr(verify, 'invoke_cpp', data.runner)
    assert verify.main(['--export-dir', str(data.export), '--output', str(data.output)]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'PASS'


def test_process_start_failure_still_restores_signals_and_keeps_logs(data, monkeypatch):
    @contextmanager
    def guard():
        yield {'defer': False, 'received': None}
    monkeypatch.setattr(verify, 'lifecycle', lambda: SimpleNamespace(termination_guard=guard))
    monkeypatch.setattr(verify.subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(OSError('cannot start')))
    result = verify.invoke_cpp(['fake'], cwd=data.api.ROOT, env={},
                               stdout_path=data.api.ROOT / 'stdout', stderr_path=data.api.ROOT / 'stderr')
    assert result['process_started'] is False and result['cleanup_complete'] is True
    assert result['returncode'] is None and 'cannot start' in result['error']
    assert (data.api.ROOT / 'stdout').exists() and (data.api.ROOT / 'stderr').exists()


def test_signal_arriving_inside_popen_still_cleans_its_owned_process(data, monkeypatch):
    control = {'defer': False, 'received': None}
    class Interrupted(Exception):
        pass
    @contextmanager
    def guard():
        yield control
    process = SimpleNamespace(pid=987654, returncode=None)
    def popen(*args, **kwargs):
        assert control['defer'] is True
        control['received'] = 'SIGTERM'
        return process
    def cleanup(child, resources, natural_exit):
        assert child is process and natural_exit is False
        process.returncode = -15
        return ['SIGTERM'], True
    monkeypatch.setattr(verify, 'lifecycle', lambda: SimpleNamespace(
        termination_guard=guard, Interrupted=Interrupted, cleanup=cleanup))
    monkeypatch.setattr(verify.subprocess, 'Popen', popen)
    result = verify.invoke_cpp(['fake'], cwd=data.api.ROOT, env={},
                               stdout_path=data.api.ROOT / 'stdout', stderr_path=data.api.ROOT / 'stderr')
    assert result['process_started'] is True and result['cleanup_complete'] is True
    assert result['cleanup_signals'] == ['SIGTERM'] and 'SIGTERM' in result['error']


@pytest.mark.parametrize('timeout', [False, True])
def test_default_process_wrapper_is_bounded_new_session_and_logs(data, monkeypatch, timeout):
    calls = {}
    class Process:
        pid = 987654
        returncode = None
        def wait(self, **kwargs):
            calls['timeout'] = kwargs['timeout']
            if timeout:
                raise subprocess.TimeoutExpired('fake cpp', kwargs['timeout'])
            self.returncode = 0
    process = Process()
    def popen(command, **kwargs):
        calls['popen'] = kwargs
        kwargs['stdout'].write('fake stdout\n')
        kwargs['stderr'].write('fake stderr\n')
        return process
    @contextmanager
    def guard():
        yield {'defer': False, 'received': None}
    def cleanup(p, resources, natural_exit):
        assert p is process and resources == {'interrupt_grace_s': 5, 'terminate_grace_s': 3, 'kill_grace_s': 2}
        calls['natural_exit'] = natural_exit
        process.returncode = -15 if timeout else 0
        return (['SIGTERM'] if timeout else []), True
    monkeypatch.setattr(verify.subprocess, 'Popen', popen)
    monkeypatch.setattr(verify, 'lifecycle', lambda: SimpleNamespace(termination_guard=guard, cleanup=cleanup))
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = verify.invoke_cpp(['fake binary'], cwd=data.api.ROOT, env={},
                               stdout_path=data.api.ROOT / 'stdout', stderr_path=data.api.ROOT / 'stderr')
    assert calls['popen']['start_new_session'] is True and calls['timeout'] == 60
    assert result['timed_out'] is timeout and result['cleanup_complete'] is True
    assert calls['natural_exit'] is not timeout
    assert (data.api.ROOT / 'stdout').read_text() == 'fake stdout\n'
    assert all(signal.getsignal(sig) == old for sig, old in before.items())
