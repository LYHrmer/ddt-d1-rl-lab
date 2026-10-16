"""Fake exports and tiny children test preparation; never load an ONNX/XML model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools11'))
import preflight_deployment11 as preflight  # noqa: E402


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    root, export = tmp_path/'workspace', tmp_path/'export'
    for name in (*preflight.PINNED_SOURCES, 'tools/export_d1_ppo.py', 'tools10/export_single_sample10.py'):
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    binary = root/'runtime03/build/d1_runtime03'; binary.parent.mkdir(parents=True)
    binary.write_bytes(b'fake binary, never executed')
    export.mkdir(); (export/'policy.onnx').write_bytes(b'fake ONNX, never loaded')
    write(export/'golden_fixtures.json', {'cases': 'synthetic receipt fixture'})
    metadata = {'training_kind': 'scale', 'training_seed': 101, 'trained_transitions': 8388608,
                'config_version': 'd1-pure-ppo-pilot-03d',
                'sha256': {'policy_onnx': preflight.sha(export/'policy.onnx'), 'checkpoint': 'c'*64},
                'golden_parity': {'sb3_predict_vs_torch_max_abs_error': 0.,
                                  'torch_vs_python_ort_max_abs_error': 1e-6,
                                  'fixtures': 64, 'python_ort_version': 'test'}}
    write(export/'metadata.json', metadata)
    sidecar = {'status': 'PASS', 'upstream_metadata_sha256': preflight.sha(export/'metadata.json'),
               'checkpoint_sha256': 'c'*64, 'entry_sha256': preflight.sha(root/'tools10/export_single_sample10.py'),
               'upstream_exporter_sha256': preflight.sha(root/'tools/export_d1_ppo.py'),
               'absolute_limits_unchanged': {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5},
               'prediction_path': {'batch_size_used_for_gate': 1}, 'physics_steps_executed': 0}
    write(export/'single_sample_export.json', sidecar)
    cpp = root/'cpp.json'
    write(cpp, {'status': 'PASS', 'policy_sha256': preflight.sha(export/'policy.onnx'),
                'fixture_sha256': preflight.sha(export/'golden_fixtures.json'),
                'max_absolute_error': 1e-6, 'absolute_tolerance': 1e-5, 'onnx_inference_calls': 64,
                'comparisons': [{'passed': True}]*64, 'physics_steps': 0,
                'native_steps_attempted': 0, 'native_steps_completed': 0,
                'runtime_binary_sha256': preflight.sha(binary), 'onnxruntime_version': '1.10.0'})
    write(root/'reports/contract08_summary.json', {'export_gate': {'status': 'FAIL'},
                                                 'qualification': {'overall': 'FAIL', 'tasks_passed': 1}})
    return root, export, cpp, metadata


def fake_runner(root, export, metadata, mode='pass'):
    calls = []
    def run(command, prefix, timeout_s, env=None):
        calls.append(command)
        stdout, stderr = prefix.with_suffix('.stdout.txt'), prefix.with_suffix('.stderr.txt')
        result = {'status': 'PASS', 'returncode': 0, 'error': None, 'cleanup_complete': True,
                  'stdout_path': str(stdout), 'stderr_path': str(stderr)}
        stderr.write_text('')
        if '--print-command' in command:
            native_output = command[command.index('--output')+1]
            info = {key: metadata[key] for key in ('training_kind', 'training_seed', 'trained_transitions', 'config_version')}
            info['native_cap'] = 0
            argv = ['rtk', 'proxy', str(root/'runtime03/build/d1_runtime03'), '--policy', str(export/'policy.onnx'),
                    '--expected-policy-sha256', preflight.sha(export/'policy.onnx'), '--output', native_output,
                    '--mode', 'replay', '--seconds', '12', '--max-native', '6000']
            if mode == 'bad_command': argv[2] = '/unrelated/binary'
            text = json.dumps(info)+'\n'+json.dumps(argv)
            if mode == 'bad_print_json': text = 'not JSON'
            if mode == 'extra_print_json': text += '\n{}'
            stdout.write_text(text)
            if mode == 'print_exit7': result.update(status='EXIT_NONZERO', returncode=7)
            if mode == 'print_timeout': result.update(status='TIMEOUT', returncode=-9, error='injected timeout')
        else:
            assert command[:2] == ['bash', str(root/'runtime04/launch_env04.sh')]
            assert str(root/'runtime04/preflight_ros_imports04.py') in command
            assert env['ROS_DOMAIN_ID'] == '84' and env['ROS_LOCALHOST_ONLY'] == '1'
            child = Path(command[command.index('--output')+1])
            report = {'format': 'd1-contract04-ros-import-preflight-v1', 'status': 'PASS',
                      'physics_steps_executed': 0, 'transitions_consumed': 0, 'ros_nodes_created': 0,
                      'ros_messages_published': 0, 'ros_messages_received': 0,
                      'watchdog_validated': False, 'control_qualification_claimed': False,
                      'checks': [{'check': name, 'status': 'PASS', 'detail': {'default_context_ok': False}}
                                 for name in preflight.ROS_CHECKS],
                      'entry_sha256': {name: preflight.sha(root/'runtime04'/name) for name in
                                       ('preflight_ros_imports04.py', 'ros_env04.sh', 'launch_env04.sh')}}
            if mode == 'import_fail':
                report['status'] = 'FAIL'
                report['checks'].append({'check': 'import:rclpy', 'status': 'FAIL',
                                         'detail': {'error': 'ModuleNotFoundError', 'message': 'missing rpyutils'}})
                result.update(status='EXIT_NONZERO', returncode=1)
                stderr.write_text('injected missing rpyutils')
            if mode == 'ros_timeout': result.update(status='TIMEOUT', returncode=-9, error='injected timeout')
            if mode == 'context_live':
                for row in report['checks']:
                    if row['check'] == 'rclpy_context_not_initialised': row['detail']['default_context_ok'] = True
            if mode == 'nodes_created': report['ros_nodes_created'] = 1
            if mode == 'missing_check': report['checks'].pop()
            write(child/'report.json', report)
            stdout.write_text(json.dumps({'status': report['status']}))
            if mode == 'mutate_binary': (root/'runtime03/build/d1_runtime03').write_bytes(b'changed during preflight')
        return result
    return run, calls


def test_scale_export_prepares_argv_without_running_native_validate_or_live(tmp_path):
    root, export, cpp, metadata = fixture(tmp_path)
    run, calls = fake_runner(root, export, metadata)
    out = tmp_path/'receipt'
    assert preflight.preflight(export, cpp, out, root=root, runner=run) == 0
    report = preflight.read(out/'report.json')
    assert report['status'] == 'PASS' and report['trial_authorised'] is False
    assert report['physics_steps_executed'] == report['ros_nodes_created'] == 0
    assert report['native_commands_executed'] == 0 and report['watchdog_validated'] is False
    assert len(calls) == 2 and '--print-command' in calls[0]
    assert all('--validate-only' not in cmd and '--execute-live-publisher' not in cmd for cmd in calls)
    assert report['commands']['validate_only_argv'][-1] == '--validate-only'
    assert not (out/'unexecuted_native_trial').exists()
    assert report['artifacts']['historical_contract08_export_gate'] == 'FAIL'


@pytest.mark.parametrize('mode,stage', [('bad_print_json', 'print_parse'), ('extra_print_json', 'print_parse'),
                                       ('bad_command', 'print_parse'), ('print_exit7', 'print_command'),
                                       ('print_timeout', 'print_command'), ('import_fail', 'ros_imports'),
                                       ('ros_timeout', 'ros_imports'), ('context_live', 'ros_imports'),
                                       ('nodes_created', 'ros_imports'), ('missing_check', 'ros_imports'),
                                       ('mutate_binary', 'final_identity')])
def test_every_preparation_failure_has_a_structured_receipt(tmp_path, mode, stage):
    root, export, cpp, metadata = fixture(tmp_path)
    run, _ = fake_runner(root, export, metadata, mode)
    out = tmp_path/'receipt'
    assert preflight.preflight(export, cpp, out, root=root, runner=run) == 1
    report = preflight.read(out/'report.json')
    assert report['status'] == 'FAIL' and report['failure_stage'] == stage
    assert report['error'] and report['failure_code']
    assert report['trial_authorised'] is False and report['native_commands_executed'] == 0
    if mode == 'import_fail':
        assert report['ros_import_receipt']['status'] == 'FAIL'
        assert 'missing rpyutils' in (out/'ros.stderr.txt').read_text()


@pytest.mark.parametrize('target', ['binary', 'metadata', 'policy', 'sidecar', 'source'])
def test_changed_artifacts_fail_before_any_preparation_child(tmp_path, target):
    root, export, cpp, _ = fixture(tmp_path)
    paths = {'binary': root/'runtime03/build/d1_runtime03', 'metadata': export/'metadata.json',
             'policy': export/'policy.onnx', 'sidecar': export/'single_sample_export.json',
             'source': root/'runtime04/preflight_ros_imports04.py'}
    paths[target].write_bytes(b'changed')
    def forbidden(*args, **kwargs): pytest.fail('bad identities must not start a child')
    out = tmp_path/'receipt'
    assert preflight.preflight(export, cpp, out, root=root, runner=forbidden) == 1
    assert preflight.read(out/'report.json')['failure_stage'] == 'artifacts'


def test_exclusive_output_is_not_overwritten(tmp_path):
    root, export, cpp, metadata = fixture(tmp_path)
    run, _ = fake_runner(root, export, metadata)
    out = tmp_path/'receipt'
    preflight.preflight(export, cpp, out, root=root, runner=run)
    digest = preflight.sha(out/'report.json')
    with pytest.raises(FileExistsError): preflight.preflight(export, cpp, out, root=root, runner=run)
    assert preflight.sha(out/'report.json') == digest


def test_real_runner_bounds_ignored_signals_and_keeps_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, 'CLEANUP_GRACES', {'interrupt_grace_s': .03, 'terminate_grace_s': .03, 'kill_grace_s': .2})
    code = ('import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); '
            'signal.signal(signal.SIGTERM,signal.SIG_IGN); print("started",flush=True); time.sleep(30)')
    result = preflight.run_checked([sys.executable, '-B', '-c', code], tmp_path/'hang', .15)
    assert result['status'] == 'TIMEOUT' and result['returncode'] == -signal.SIGKILL
    assert result['cleanup_complete'] is True and result['wall_s'] < 3
    assert 'started' in Path(result['stdout_path']).read_text()


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_real_runner_handles_signal_during_spawn_without_orphaning(tmp_path, monkeypatch, signum):
    original, children = preflight.subprocess.Popen, []
    def interrupt(*args, **kwargs):
        child = original(*args, **kwargs); children.append(child)
        os.kill(os.getpid(), signum)
        return child
    monkeypatch.setattr(preflight.subprocess, 'Popen', interrupt)
    result = preflight.run_checked([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], tmp_path/'signal', 1)
    assert result['status'] == 'ERROR' and result['cleanup_complete'] is True
    assert children[0].poll() is not None


def test_failed_real_runner_start_still_produces_preflight_failure_receipt(tmp_path, monkeypatch):
    root, export, cpp, _ = fixture(tmp_path)
    def failed_start(*args, **kwargs): raise FileNotFoundError('injected missing interpreter')
    monkeypatch.setattr(preflight.subprocess, 'Popen', failed_start)
    out = tmp_path/'receipt'
    assert preflight.preflight(export, cpp, out, root=root) == 1
    report = preflight.read(out/'report.json')
    assert report['failure_stage'] == 'print_command' and 'missing interpreter' in report['error']
    assert report['stages']['print_command']['returncode'] is None
