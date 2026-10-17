#!/usr/bin/env python3
"""Verify 64 real control-weight fixtures with the frozen C++ ORT binary, zero physics."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1e-5
TIMEOUT_S = 60
BINARY = 'runtime03/build/d1_runtime03'
SCHEMA = 'runtime03/schema03d.json'
SIGNATURE = {'inputs': {'obs': [1, 57], 'history': [1, 10, 57]}, 'outputs': {'actions': [1, 16]}}


def provenance():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module('tools17.provenance17')


def lifecycle():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module('tools16.run_comparison16')


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(value, dict), 'expected a JSON object')
    return value


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def vector(value, size, label):
    require(isinstance(value, list) and len(value) == size
            and all(finite_number(item) for item in value), 'invalid finite ' + label)
    return value


def check_fixtures(value):
    require(value.get('format') == 'd1-onnx-two-input-golden-64-v1', 'unexpected fixture format')
    rows = value.get('fixtures')
    require(isinstance(rows, list) and len(rows) == 64, 'exactly 64 fixtures are required')
    if 'count' in value:
        require(type(value['count']) is int and value['count'] == 64, 'fixture count differs')
    indexed = {}
    for item in rows:
        require(isinstance(item, dict), 'fixture must be an object')
        index = item.get('index')
        require(type(index) is int and 0 <= index < 64 and index not in indexed, 'duplicate or invalid fixture index')
        vector(item.get('obs'), 57, 'obs57')
        history = item.get('history_oldest_to_newest')
        require(isinstance(history, list) and len(history) == 10, 'history must contain ten previous frames')
        for row in history:
            vector(row, 57, 'history57')
        if 'history_oldest_to_newest_previous_only' in item:
            alias = item['history_oldest_to_newest_previous_only']
            require(isinstance(alias, list) and len(alias) == 10, 'invalid fixture history alias')
            for row in alias:
                vector(row, 57, 'history alias57')
            require(alias == history, 'fixture history aliases differ')
        vector(item.get('torch_actions'), 16, 'Torch actions16')
        require(all(-1 <= action <= 1 for action in item['torch_actions']), 'Torch actions are not clipped')
        indexed[index] = item
    return indexed


def training_identity(api, export):
    release, protocol, result, checkpoint = api.load_completed_control(Path(export['training_root']))
    checkpoint = Path(checkpoint).resolve()
    training_root = Path(export['training_root']).resolve()
    execution = training_root / release['output_root']
    identity = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': api.sha256(checkpoint),
                'training_protocol_sha256': api.sha256(checkpoint.parent / 'protocol.json'),
                'training_result_sha256': api.sha256(checkpoint.parent / 'result.json'),
                'release_sha256': api.sha256(training_root / 'evidence/local_comparison16/release_01.json'),
                'execution_report_sha256': api.sha256(execution / 'report.json')}
    require(export.get('checkpoint_sha256') == identity['checkpoint_sha256']
            and export.get('training_identity') == identity, 'export does not bind the published completed control checkpoint')
    require(result.get('checkpoint_sha256') == identity['checkpoint_sha256']
            and result.get('status') == 'COMPLETE' and result.get('arm') == 'control'
            and result.get('aux_coef') == 0 and result.get('eligible_for_evaluation') is True
            and protocol.get('random_initialization') is True and protocol.get('warm_start') is False
            and protocol.get('teacher_policy_used') is False, 'ineligible control checkpoint')
    return release, identity


def inspect_export(export_dir, api):
    export_dir = Path(export_dir)
    report = read_json(export_dir / 'report.json')
    require(report.get('format') == 'd1-completed-control-onnx-export-v17'
            and report.get('status') == 'PASS' and report.get('error') is None
            and report.get('completed_checkpoint_provenance_checked') is True
            and type(report.get('physics_steps_executed')) is int and report['physics_steps_executed'] == 0
            and type(report.get('robot_models_constructed')) is int and report['robot_models_constructed'] == 0,
            'export did not pass the completed-checkpoint numerical gate')
    api.check_sources(report.get('source_identity'))
    release, identity = training_identity(api, report)
    for name, field in (('policy.onnx', 'onnx_sha256'), ('golden_fixtures.json', 'fixtures_sha256')):
        path = export_dir / name
        require(path.is_file() and not path.is_symlink() and api.sha256(path) == report.get(field),
                'export artifact SHA differs: ' + name)
    limits, parity = report.get('absolute_limits', {}), report.get('parity', {})
    require(limits == {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5}, 'export parity limits changed')
    for field, maximum in (('sb3_predict_vs_torch_max_abs_error', 1e-6),
                           ('torch_vs_python_ort_max_abs_error', 1e-5)):
        value = parity.get(field)
        require(finite_number(value) and 0 <= value <= maximum, 'export numerical parity failed: ' + field)
    require(parity.get('fixtures') == 64 and parity.get('prediction_batch_size') == 1, 'export fixture/batch count differs')
    fixtures = check_fixtures(read_json(export_dir / 'golden_fixtures.json'))
    return report, release, identity, fixtures


def invoke_cpp(command, *, cwd, env, stdout_path, stderr_path, timeout_s=TIMEOUT_S):
    """Bounded new-session child; cleanup functions are the unchanged contract16 ones."""
    require(0 < timeout_s <= 60, 'C++ timeout must not exceed 60 seconds')
    tools = lifecycle()
    result = {'process_started': False, 'returncode': None, 'timed_out': False,
              'cleanup_complete': True, 'cleanup_signals': [], 'error': None}
    process = None
    with tools.termination_guard() as control:
        try:
            with Path(stdout_path).open('w') as stdout, Path(stderr_path).open('w') as stderr:
                control['defer'] = True
                try:
                    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                               start_new_session=True)
                    result['process_started'] = True
                finally:
                    control['defer'] = False
                if control['received']:
                    raise tools.Interrupted(control['received'])
                process.wait(timeout=timeout_s)
        except BaseException as exc:
            result['timed_out'] = isinstance(exc, subprocess.TimeoutExpired)
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                for sig in previous:
                    signal.signal(sig, signal.SIG_IGN)
                if process is not None:
                    try:
                        sent, complete = tools.cleanup(process, {'interrupt_grace_s': 5,
                            'terminate_grace_s': 3, 'kill_grace_s': 2}, natural_exit=result['error'] is None)
                        result.update(cleanup_signals=sent, cleanup_complete=complete)
                    except BaseException as exc:
                        result.update(cleanup_complete=False, error=f'cleanup {type(exc).__name__}: {exc}')
                    result['returncode'] = process.returncode
            finally:
                for sig, previous_handler in previous.items():
                    signal.signal(sig, previous_handler)
    return result


def check_native(native, fixtures, identities):
    for field in ('policy_sha256', 'fixture_sha256', 'runtime_binary_sha256', 'schema_sha256'):
        require(native.get(field) == identities[field], 'C++ identity differs: ' + field)
    require(native.get('status') == 'PASS' and native.get('error') is None,
            'C++ report did not pass cleanly')
    require(native.get('onnxruntime_version') == '1.10.0' and native.get('mujoco_version') == '3.3.0',
            'C++ runtime library versions differ')
    for field in ('physics_steps', 'native_steps_attempted', 'native_steps_completed', 'pd_updates', 'policy_updates'):
        require(type(native.get(field)) is int and native[field] == 0, 'C++ did not report zero ' + field)
    require(native.get('onnx_signature') == SIGNATURE, 'C++ ONNX signature differs')
    require(type(native.get('onnx_inference_calls')) is int and native['onnx_inference_calls'] == 64,
            'C++ must execute exactly 64 inferences')
    comparisons = native.get('comparisons')
    require(isinstance(comparisons, list) and len(comparisons) == 64, 'C++ must report 64 comparisons')
    seen, errors = set(), []
    for row in comparisons:
        require(isinstance(row, dict), 'malformed C++ comparison')
        index = row.get('index')
        require(type(index) is int and index in fixtures and index not in seen, 'C++ duplicate or unknown fixture index')
        seen.add(index)
        actual = vector(row.get('actions'), 16, 'C++ actions16')
        calculated = max(abs(a - b) for a, b in zip(actual, fixtures[index]['torch_actions']))
        declared = row.get('max_absolute_error')
        require(finite_number(declared) and 0 <= declared <= LIMIT and row.get('passed') is True
                and calculated <= LIMIT and abs(calculated - declared) <= 1e-12, 'C++ comparison error or arithmetic differs')
        errors.append(calculated)
    require(seen == set(range(64)), 'C++ fixture indices are incomplete')
    maximum = native.get('max_absolute_error')
    require(finite_number(maximum) and 0 <= maximum <= LIMIT
            and abs(maximum - max(errors)) <= 1e-12 and native.get('absolute_tolerance') == LIMIT,
            'C++ maximum error or tolerance differs')
    return max(errors)


def run(export_dir, output, *, api=None, runner=None):
    output, export_dir = Path(output).resolve(), Path(export_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'format': 'd1-completed-control-cpp-parity-v17', 'status': 'FAIL', 'error': None,
              'export_dir': str(export_dir), 'source_identity': None, 'training_identity': None,
              'completed_checkpoint_provenance_checked': False, 'training_started': False,
              'physics_steps_executed': 0, 'robot_models_constructed': 0,
              'cpp_process_attempted': False, 'cpp_execution': None, 'command': None,
              'execution_timeout_s': TIMEOUT_S, 'absolute_tolerance': LIMIT,
              'scope': '64 real control-weight fixtures; numerical inference only; no deployment or control qualification'}
    try:
        for name in ('stdout.txt', 'stderr.txt'):
            (output / name).touch(exist_ok=False)
        api = api or provenance()
        proof = api.source_identity()
        report['source_identity'] = proof
        exported, release, identity, fixtures = inspect_export(export_dir, api)
        require(exported['source_identity']['sources_sha256'] == proof['sources_sha256'], 'export and verifier source sets differ')
        root = Path(api.ROOT).resolve()
        binary, schema = root / BINARY, root / SCHEMA
        identities = {'policy_sha256': api.sha256(export_dir / 'policy.onnx'),
                      'fixture_sha256': api.sha256(export_dir / 'golden_fixtures.json'),
                      'runtime_binary_sha256': api.sha256(binary), 'schema_sha256': api.sha256(schema)}
        require(all(release['sources_sha256'].get(name) == identities[field]
                    for name, field in ((BINARY, 'runtime_binary_sha256'), (SCHEMA, 'schema_sha256'))),
                'frozen C++ binary or schema identity changed')
        report.update(training_identity=identity, completed_checkpoint_provenance_checked=True,
                      checkpoint_sha256=identity['checkpoint_sha256'], identities=identities,
                      onnx_sha256=identities['policy_sha256'], fixtures_sha256=identities['fixture_sha256'],
                      runtime_binary_sha256=identities['runtime_binary_sha256'], schema_sha256=identities['schema_sha256'],
                      export_report_sha256=api.sha256(export_dir / 'report.json'))
        command = [str(binary), '--schema', str(schema), '--onnx-golden', str(export_dir / 'golden_fixtures.json'),
                   '--policy', str(export_dir / 'policy.onnx'), '--expected-policy-sha256', identities['policy_sha256'],
                   '--output', str(output / 'cpp_native')]
        report['command'] = command
        env = dict(os.environ, LD_LIBRARY_PATH=f'/opt/ros/humble/lib:{root / "install/mujoco/lib"}',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        report.update(cpp_process_attempted=True, physics_steps_executed=None, robot_models_constructed=None)
        execution = (runner or invoke_cpp)(command, cwd=root, env=env, stdout_path=output / 'stdout.txt',
                                          stderr_path=output / 'stderr.txt', timeout_s=TIMEOUT_S)
        report['cpp_execution'] = execution
        native_path = output / 'cpp_native/report.json'
        native = read_json(native_path) if native_path.is_file() else None
        if native is not None:
            report.update(cpp_native_report_sha256=api.sha256(native_path), cpp_native_status=native.get('status'),
                          cpp_native_error=native.get('error'))
        require(execution.get('process_started') is True and execution.get('returncode') == 0
                and execution.get('timed_out') is False and execution.get('cleanup_complete') is True
                and not execution.get('cleanup_signals') and execution.get('error') is None,
                'C++ process failed, timed out, was interrupted or required cleanup')
        require(native is not None, 'C++ returned no structured report')
        maximum = check_native(native, fixtures, identities)
        api.check_sources(proof)
        current, _, current_identity, _ = inspect_export(export_dir, api)
        require(current == exported and current_identity == identity
                and api.sha256(export_dir / 'report.json') == report['export_report_sha256']
                and api.sha256(binary) == identities['runtime_binary_sha256']
                and api.sha256(schema) == identities['schema_sha256'], 'inputs changed during C++ verification')
        report.update(status='PASS', comparisons=64, onnx_inference_calls=64, max_absolute_error=maximum,
                      physics_steps_executed=0, robot_models_constructed=0, cxx_ort_1_10_status='PASS')
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['wall_s'] = time.monotonic() - started
        if api is not None:
            api.atomic_json(output / 'report.json', report)
        else:
            (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.export_dir, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
