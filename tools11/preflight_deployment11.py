#!/usr/bin/env python3
"""Check engineering exports and ROS imports; prepare commands without running them.

Only the frozen launcher's --print-command and the zero-node ROS import probe are
executed. Native, --validate-only and live commands are never executed here. A PASS
is deployment preparation, not physical validation or permission to run a trial.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools10'))
from summarize_export10 import build as export_summary  # noqa: E402
from supervise_scale10 import atomic_json, handle_termination, sha, stop_group  # noqa: E402

PINNED_SOURCES = {
    'tools/run_selftrained03.py': '1f4bb6f269fa7d80df108a25cc7f9c9916a523810ecbf44e19da46cccc673c67',
    'runtime03/deployment_variants.py': '5688e133480ff07efb64e2d862763358628f8c2fd6c79495fc442a4e575f9a39',
    'runtime04/launch_env04.sh': 'fc529b7693078c296c64147b0b2a7f4d90b5f075326b938b4d6717c8cfbf884b',
    'runtime04/ros_env04.sh': '60cb747da3f36e7035ee914a44c02c233d50c82db3fd62db4a80df3ede7d1d46',
    'runtime04/preflight_ros_imports04.py': '69a9d88f7de0edb7f03f7e9712230f7b3b6a9f788611134cec4b3c4fe95bd0f7',
    'tools10/summarize_export10.py': '235d7d4ac0b078ee645287a906b63ce824bff9ef77733fbe6a895ccd85f02d55',
    'tools10/supervise_scale10.py': 'f82e3d0255ad8ff7931ad8ddcfe3a6c235b5f50295b6c42e346b1a2a4a599a0b',
}
CLEANUP_GRACES = {'interrupt_grace_s': 1., 'terminate_grace_s': 1., 'kill_grace_s': 1.}
ROS_CHECKS = {'ros_distro_is_humble', 'ament_prefix_path_has_humble',
              'pythonpath_has_rpyutils_site_packages', 'ros_localhost_only', 'ros_domain_id_is_84',
              'import:rpyutils', 'import:rclpy', 'import:geometry_msgs.msg', 'import:std_msgs.msg',
              'no_simulation_or_inference_module_imported', 'rclpy_context_not_initialised'}


class PreflightFailure(Exception):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def read(path):
    return json.loads(Path(path).read_text())


def check_artifacts(export, cpp_report, root):
    for name, digest in PINNED_SOURCES.items():
        if sha(root/name) != digest:
            raise ValueError(f'frozen preparation source changed: {name}')
    summary = export_summary(export, cpp_report, root/'reports/contract08_summary.json')
    if (summary['cpp_binary_sha256'] != sha(root/'runtime03/build/d1_runtime03')
            or summary['entry_sha256'] != sha(root/'tools10/export_single_sample10.py')
            or summary['upstream_exporter_sha256'] != sha(root/'tools/export_d1_ppo.py')):
        raise ValueError('current runtime/exporter files differ from the numerical receipts')
    metadata = read(export/'metadata.json')
    if metadata['training_kind'] not in ('pilot', 'scale'):
        raise ValueError('this preparation entry accepts only pilot or scale engineering exports')
    return summary, metadata


def run_checked(command, log_prefix, timeout_s, env=None):
    """Bounded child with durable logs and cleanup of its own new process group."""
    process = None
    status, error, signals, clean = 'PASS', None, [], True
    started = time.monotonic()
    stdout_path, stderr_path = log_prefix.with_suffix('.stdout.txt'), log_prefix.with_suffix('.stderr.txt')
    with handle_termination() as control, stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        try:
            control['defer'] = True
            try:
                process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                           start_new_session=True)
            finally:
                control['defer'] = False
            if control['received']:
                raise InterruptedError(control['received'])
            process.wait(timeout=timeout_s)
            if process.returncode != 0:
                status = 'EXIT_NONZERO'
        except subprocess.TimeoutExpired:
            status, error = 'TIMEOUT', f'process exceeded {timeout_s} seconds'
        except BaseException as exc:
            status, error = 'ERROR', f'{type(exc).__name__}: {exc}'
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                signals, clean = stop_group(process, CLEANUP_GRACES)
                if not clean or (signals and status == 'PASS'):
                    status = 'CLEANUP_FAILED' if not clean else 'RESIDUAL_PROCESS_GROUP'
    return {'status': status, 'command': command,
            'returncode': process.returncode if process is not None else None,
            'stdout_path': str(stdout_path), 'stderr_path': str(stderr_path),
            'error': error, 'wall_s': time.monotonic()-started,
            'cleanup_signals': signals, 'cleanup_complete': clean}


def parse_printed_command(text, metadata, export, native_output, root):
    decoder = json.JSONDecoder()
    info, end = decoder.raw_decode(text.lstrip())
    remainder = text.lstrip()[end:].lstrip()
    command, end = decoder.raw_decode(remainder)
    if remainder[end:].strip() or not isinstance(info, dict) or not isinstance(command, list):
        raise ValueError('expected exactly the identity object and argv from the frozen launcher')
    if (not all(isinstance(arg, str) for arg in command)
            or command[:3] != ['rtk', 'proxy', str(root/'runtime03/build/d1_runtime03')]
            or info.get('native_cap') != 0
            or any(info.get(key) != metadata[key] for key in
                   ('training_kind', 'training_seed', 'trained_transitions', 'config_version'))):
        raise ValueError('printed launcher identity or command differs from the requested export')
    expected = {'--policy': str(export/'policy.onnx'), '--expected-policy-sha256': sha(export/'policy.onnx'),
                '--output': str(native_output), '--mode': 'replay', '--seconds': '12', '--max-native': '6000'}
    if any(command.count(flag) != 1 or command[command.index(flag)+1] != value for flag, value in expected.items()):
        raise ValueError('printed command has unexpected policy, mode, output or native allowance')
    if '--validate-only' in command:
        raise ValueError('unexpected validate-only flag in the ordinary prepared command')
    return info, {'native_replay_argv': command, 'validate_only_argv': [*command, '--validate-only']}


def check_ros_receipt(path, root):
    report = read(path)
    if (report.get('format') != 'd1-contract04-ros-import-preflight-v1' or report.get('status') != 'PASS'
            or any(report.get(field) != 0 for field in
                   ('physics_steps_executed', 'transitions_consumed', 'ros_nodes_created',
                    'ros_messages_published', 'ros_messages_received'))
            or report.get('watchdog_validated') is not False
            or report.get('control_qualification_claimed') is not False
            or not report.get('checks') or not all(row['status'] == 'PASS' for row in report['checks'])
            or {row['check'] for row in report['checks']} != ROS_CHECKS
            or not any(row['check'] == 'rclpy_context_not_initialised' and row['detail'].get('default_context_ok') is False
                       for row in report['checks'])):
        raise ValueError('ROS receipt does not establish imports with an uninitialised context and zero nodes')
    for name in ('preflight_ros_imports04.py', 'ros_env04.sh', 'launch_env04.sh'):
        if report['entry_sha256'].get(name) != sha(root/'runtime04'/name):
            raise ValueError(f'ROS import receipt source differs: {name}')
    return report


def preflight(export, cpp_report, output, timeout_s=20., root=ROOT, runner=run_checked):
    export, cpp_report, output, root = map(lambda p: Path(p).resolve(), (export, cpp_report, output, root))
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-deployment-preflight11-v1', 'status': 'FAIL', 'stages': {},
              'failure_stage': None, 'failure_code': None, 'error': None, 'commands': None,
              'physics_steps_executed': 0, 'ros_nodes_created': 0, 'native_commands_executed': 0,
              'watchdog_validated': False, 'control_qualification_claimed': False,
              'trial_authorised': False, 'scope': 'artifact identity, print-only preparation and ROS imports',
              'entry_sha256': sha(__file__), 'process_timeout_s': timeout_s}
    stage = 'artifacts'
    with handle_termination():
        try:
            if not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
                raise ValueError('process timeout must be finite and in (0, 60] seconds')
            summary, metadata = check_artifacts(export, cpp_report, root)
            inputs = {str(path): sha(path) for path in
                      (export/'metadata.json', export/'policy.onnx', export/'golden_fixtures.json',
                       export/'single_sample_export.json', cpp_report, root/'runtime03/build/d1_runtime03',
                       *(root/name for name in PINNED_SOURCES))}
            report['artifacts'] = summary
            report['input_sha256'] = inputs
            native_output = output/'unexecuted_native_trial'
            stage = 'print_command'
            command = [sys.executable, '-B', str(root/'tools/run_selftrained03.py'), '--export', str(export),
                       '--output', str(native_output), '--print-command']
            result = runner(command, output/'launcher', timeout_s)
            report['stages'][stage] = result
            if result['status'] != 'PASS' or result['returncode'] != 0 or result['cleanup_complete'] is not True:
                raise PreflightFailure('PRINT_COMMAND_'+result['status'], result.get('error') or 'launcher print-only failed')
            stage = 'print_parse'
            info, commands = parse_printed_command(Path(result['stdout_path']).read_text(), metadata,
                                                   export, native_output, root)
            report['launcher_identity'], report['commands'] = info, commands
            stage = 'ros_imports'
            child_output = output/'ros_imports'
            command = ['bash', str(root/'runtime04/launch_env04.sh'), '/usr/bin/python3', '-B',
                       str(root/'runtime04/preflight_ros_imports04.py'), '--output', str(child_output)]
            env = dict(os.environ, ROS_DOMAIN_ID='84', ROS_LOCALHOST_ONLY='1', PYTHONDONTWRITEBYTECODE='1')
            result = runner(command, output/'ros', timeout_s, env=env)
            report['stages'][stage] = result
            if (child_output/'report.json').is_file():
                report['ros_import_receipt'] = read(child_output/'report.json')
                report['ros_import_receipt_sha256'] = sha(child_output/'report.json')
            if result['status'] != 'PASS' or result['returncode'] != 0 or result['cleanup_complete'] is not True:
                raise PreflightFailure('ROS_IMPORT_'+result['status'], result.get('error') or 'ROS import preflight failed')
            report['ros_import_receipt'] = check_ros_receipt(child_output/'report.json', root)
            stage = 'final_identity'
            if not all(sha(Path(path)) == digest for path, digest in inputs.items()):
                raise ValueError('a checked input changed during preparation')
            report['status'] = 'PASS'
        except BaseException as exc:
            report.update(failure_stage=stage, failure_code=getattr(exc, 'code', stage.upper()+'_ERROR'),
                          error=f'{type(exc).__name__}: {exc}')
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            atomic_json(output/'report.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'failure_stage', 'failure_code', 'physics_steps_executed')}))
    return 0 if report['status'] == 'PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--cpp-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new, exclusive receipt directory')
    parser.add_argument('--max-wall-seconds', type=float, default=20., help='per preparation subprocess, at most 60')
    args = parser.parse_args()
    return preflight(args.export, args.cpp_report, args.output, args.max_wall_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
