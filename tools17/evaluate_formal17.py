#!/usr/bin/env python3
"""One bounded formal 36-case evaluation of the completed control's exported ONNX."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools17 import provenance17 as provenance  # noqa: E402
from tools17.run_closeout17 import BUDGET, load_contract, validate_launch  # noqa: E402
from tools16.contract16 import read_memory  # noqa: E402
from tools16.evaluate_comparison16 import frozen_helpers  # noqa: E402
from tools16.train_comparison16 import termination_scope  # noqa: E402


def scenarios(helpers):
    actual = helpers.make_scenarios('formal')
    expected = [(name, delay, seed) for name in ('forward_stop', 'reverse_stop', 'turn_stop')
                for delay in (0, 1, 2, 3) for seed in (1001, 1002, 1003)]
    provenance.require(actual == expected and len(actual) == 36, 'frozen formal scenario matrix changed')
    return actual


def add_counts(total, attempted):
    provenance.require(isinstance(attempted, dict) and all(type(attempted.get(key)) is int
                       and attempted[key] >= 0 for key in BUDGET), 'invalid actual physical counters')
    result = {key: total[key] + attempted[key] for key in BUDGET}
    provenance.require(all(result[key] <= cap for key, cap in BUDGET.items()), 'formal counter cap exceeded')
    return result


def verify_trace(output, item, helpers):
    """Recompute the unchanged evaluator's metrics from the saved trace bytes."""
    import numpy as np
    name, delay, seed = item['sequence'], item['delay_ticks'], item['reset_seed']
    expected_name = f'{name}_delay{delay}_seed{seed}.npz'
    provenance.require(item.get('trace_file') == expected_name, 'case trace filename differs')
    path = Path(output) / expected_name
    provenance.require(provenance.sha256(path) == item.get('trace_sha256'), 'case trace SHA differs')
    with np.load(path, allow_pickle=False) as handle:
        trace = {key: handle[key] for key in handle.files}
    n = len(trace['reward'])
    provenance.require(1 <= n <= 600, 'invalid trace horizon')
    scalar = ('reward', 'body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'base_x_m', 'base_y_m',
              'base_height_m', 'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad', 'tilt_rad', 'tau_saturation_count')
    shapes = {key: (n,) for key in scalar}
    shapes.update(command_during_interval=(n, 3), issued_action=(n, 16), applied_action=(n, 16),
                  boundary_xy_m=(n + 1, 2), policy_interval_start_s=(n,))
    for key, shape in shapes.items():
        value = trace[key]
        provenance.require(value.shape == shape and np.issubdtype(value.dtype, np.number)
                           and bool(np.isfinite(value).all()), 'invalid trace array: ' + key)
    provenance.require(all(trace[key].shape == (n,) and trace[key].dtype == np.bool_
                           for key in ('fallen', 'illegal_body_contact')), 'invalid trace flags')
    provenance.require(np.array_equal(trace['policy_interval_start_s'], np.arange(n) * .02), 'trace clock differs')
    expected_commands = []
    for tick in range(n):
        segment = helpers.SEQUENCES[name][0]
        for candidate in helpers.SEQUENCES[name]:
            if candidate['start_s'] <= tick * .02 + 1e-12:
                segment = candidate
        expected_commands.append((segment['vx'], 0., segment['wz']))
    provenance.require(np.array_equal(trace['command_during_interval'], np.asarray(expected_commands)), 'trace commands differ')
    provenance.require(np.array_equal(trace['boundary_xy_m'][1:, 0], trace['base_x_m'])
                       and np.array_equal(trace['boundary_xy_m'][1:, 1], trace['base_y_m']), 'trace boundaries differ')
    expected_counts = {'policy_transitions': n, 'native_steps': n * 10, 'pd_updates': n * 5}
    provenance.require(all(type(item.get('actual_counts', {}).get(key)) is int
                           and item['actual_counts'][key] == count for key, count in expected_counts.items()),
                       'trace and actual counters differ')
    provenance.require(item.get('done_reason') in ('HORIZON_12S', 'TERMINATED', 'ENV_TRUNCATED_BEFORE_12S')
                       and (item['done_reason'] != 'HORIZON_12S' or n == 600), 'invalid case completion')
    calculated = helpers.trace_metrics(name, delay, seed, trace, trace['boundary_xy_m'], item['done_reason'])
    provenance.require(all(item.get(key) == value for key, value in calculated.items()), 'trace metrics differ')
    return calculated


def check_numeric_inputs(export_dir, cpp_report_path, release):
    from tools17.verify_cpp17 import check_fixtures, check_native
    export_dir, cpp_report_path = Path(export_dir).resolve(), Path(cpp_report_path).resolve()
    execution = ROOT / release['output_root']
    provenance.require(export_dir == execution / 'export' and cpp_report_path == execution / 'cpp/report.json',
                       'numeric inputs must be the same claimed closeout attempt')
    exported = provenance.read_json(export_dir / 'report.json')
    cpp = provenance.read_json(cpp_report_path)
    for report in (exported, cpp):
        provenance.require(report.get('status') == 'PASS'
                           and report.get('checkpoint_sha256') == release['checkpoint_sha256']
                           and report.get('source_identity') == release['source_identity'],
                           'numeric input does not prove the same complete checkpoint/source')
    onnx_path = export_dir / 'policy.onnx'
    provenance.require(exported.get('onnx_sha256') == provenance.sha256(onnx_path), 'exported ONNX changed')
    provenance.require(exported.get('fixtures_sha256') == provenance.sha256(export_dir / 'golden_fixtures.json'),
                       'exported fixtures changed')
    provenance.require(cpp.get('onnx_sha256') == exported['onnx_sha256'], 'C++ checked a different ONNX')
    provenance.require(cpp.get('export_report_sha256') == provenance.sha256(export_dir / 'report.json')
                       and cpp.get('fixtures_sha256') == exported['fixtures_sha256'], 'C++ export or fixture identity differs')
    native_path = cpp_report_path.parent / 'cpp_native/report.json'
    provenance.require(cpp.get('cpp_native_report_sha256') == provenance.sha256(native_path), 'C++ native report changed')
    fixtures = check_fixtures(provenance.read_json(export_dir / 'golden_fixtures.json'))
    identities = cpp['identities']
    provenance.require(identities.get('policy_sha256') == exported['onnx_sha256']
                       and identities.get('fixture_sha256') == exported['fixtures_sha256'], 'C++ artifact identities differ')
    check_native(provenance.read_json(native_path), fixtures, identities)
    journal = provenance.read_json(execution / 'report.json')
    for name, path in (('export', export_dir / 'report.json'), ('cpp', cpp_report_path)):
        stage = journal['stages'].get(name, {})
        provenance.require(stage.get('status') == 'COMPLETE'
                           and stage.get('receipt_sha256') == provenance.sha256(path), 'completed numeric stage receipt changed')
    return onnx_path, {'export_report_sha256': provenance.sha256(export_dir / 'report.json'),
                       'cpp_report_sha256': provenance.sha256(cpp_report_path), 'onnx_sha256': provenance.sha256(onnx_path)}


def runtime(release, onnx_path, training_protocol):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    for path in (ROOT / '.python-deps', ROOT / 'training'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import gymnasium
    import mujoco
    import numpy as np
    import onnxruntime as ort
    from d1_rl.factory03 import environment_class
    versions = {'python': platform.python_version(), 'gymnasium': gymnasium.__version__,
                'mujoco': mujoco.__version__, 'numpy': np.__version__}
    provenance.require(all(training_protocol['versions'][key] == value for key, value in versions.items()),
                       'physical/runtime versions differ from training')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    provenance.require([(x.name, x.shape, x.type) for x in session.get_inputs()] == [
        ('obs', [1, 57], 'tensor(float)'), ('history', [1, 10, 57], 'tensor(float)')]
        and [(x.name, x.shape, x.type) for x in session.get_outputs()] == [('actions', [1, 16], 'tensor(float)')],
        'ORT runtime signature differs')
    def predict(observation):
        value = np.asarray(observation)
        provenance.require(value.shape == (627,) and value.dtype == np.float32
                           and bool(np.isfinite(value).all()), 'invalid actor observation')
        result = session.run(['actions'], {'obs': value[:57].reshape(1, 57),
                                          'history': value[57:].reshape(1, 10, 57)})[0]
        provenance.require(result.shape == (1, 16) and bool(np.isfinite(result).all())
                           and bool((np.abs(result) <= 1.).all()), 'invalid exported action')
        return result[0].astype(np.float32)
    cls = environment_class(ROOT / release['config'])
    def make_env(name, sequence, delay, seed):
        return cls(config=ROOT / release['config'], arm='B', seed=seed, rank=0, max_transitions=600,
                   eval_delay_ticks=delay, eval_command_sequence=sequence)
    return SimpleNamespace(predict=predict, make_env=make_env,
                           backend={'kind': 'python_ort_cpu_complete_export', **versions,
                                    'python_ort': ort.__version__, 'cpu_threads': 1})


def run(release_path, export_dir, cpp_report_path, output, *, runtime_factory=runtime,
        loader=load_contract, launch_validator=validate_launch, memory_reader=read_memory,
        clock=time.monotonic, bundle_loader=provenance.load_completed_control):
    started = clock()
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    release = loader(release_path)
    launch_validator(release, release_path, output)
    provenance.require(not output.exists(), 'formal evaluation refuses an existing output')
    old_release, training_protocol, training_result, checkpoint = bundle_loader(release['training_root'])
    onnx_path, identity = check_numeric_inputs(export_dir, cpp_report_path, release)
    helpers = frozen_helpers(old_release)
    matrix = scenarios(helpers)
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-closeout17-formal-result-v1', 'status': 'INCOMPLETE',
              'release_sha256': provenance.sha256(release_path), 'source_identity': release['source_identity'],
              'checkpoint_sha256': training_result['checkpoint_sha256'], **identity,
              'cases': [], 'planned_cases': 36, 'actual_counts': None, 'known_actual_counts': dict.fromkeys(BUDGET, 0),
              'inputs_unchanged': False, 'cleanup_complete': True, 'full_horizons_complete': False,
              'all_36_reference_pass': False, 'nominal_zero_delay_pass_count': 0,
              'error': None, 'training_transitions': 0, 'minimum_effective_bytes': None,
              'scope': 'one trained seed101; formal 3 tasks x 3 reset seeds x 4 delays; Python ORT CPU only'}
    environment, constructing, accounted, unknown = None, False, False, False
    total = dict.fromkeys(BUDGET, 0)
    last_sample_s = -float('inf')
    def poll(force=False):
        nonlocal last_sample_s
        now = clock()
        provenance.require(now - started < release['stage_limits_s']['formal'], 'formal wall limit')
        if force or now - last_sample_s >= 1.:
            sample = memory_reader()
            value = sample['effective_available_bytes']
            report['minimum_effective_bytes'] = value if report['minimum_effective_bytes'] is None else min(value, report['minimum_effective_bytes'])
            provenance.require(value >= release['resources']['reserve_bytes'], 'formal RAM below 3 GiB')
            last_sample_s = now
    with termination_scope():
        try:
            poll(force=True)
            provenance.require(memory_reader()['effective_available_bytes'] >= release['resources']['start_min_effective_bytes'],
                               'formal start RAM below 6 GiB')
            rt = runtime_factory(old_release, onnx_path, training_protocol)
            report['backend'] = rt.backend
            protocol = {'format': 'd1-closeout17-formal-protocol-v1', 'release_sha256': report['release_sha256'],
                        'source_identity': release['source_identity'], 'checkpoint_sha256': report['checkpoint_sha256'],
                        **identity, 'backend': rt.backend, 'budget_max': BUDGET,
                        'scenarios': [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed} for name, delay, seed in matrix],
                        'thresholds': helpers.THRESHOLDS, 'episode_ticks': 600,
                        'max_wall_s': release['stage_limits_s']['formal'], 'training_checkpoint': str(checkpoint)}
            provenance.exclusive_json(output / 'protocol.json', protocol)
            report['protocol_sha256'] = provenance.sha256(output / 'protocol.json')
            def predict(observation):
                poll()
                return rt.predict(observation)
            with termination_scope():
                for name, delay, seed in matrix:
                    poll(force=True)
                    accounted, constructing = False, True
                    environment = rt.make_env(name, helpers.SEQUENCES[name], delay, seed)
                    constructing = False
                    item = helpers.run_episode(environment, predict, name, delay, seed,
                                               output / f'{name}_delay{delay}_seed{seed}.npz',
                                               BUDGET['native_steps'] - total['native_steps'])
                    provenance.require((item['sequence'], item['delay_ticks'], item['reset_seed']) == (name, delay, seed),
                                       'actual case differs from planned matrix')
                    verify_trace(output, item, helpers)
                    report['cases'].append(item)
                    total = add_counts(total, item['actual_counts'])
                    accounted = True
                    environment.close()
                    environment = None
                    report['known_actual_counts'] = dict(total)
                    provenance.atomic_json(output / 'report.json', report)
                poll(force=True)
                provenance.check_sources(release['source_identity'])
                bundle_loader(release['training_root'])
                _, current_identity = check_numeric_inputs(export_dir, cpp_report_path, release)
                provenance.require(current_identity == identity, 'numeric input changed during formal evaluation')
                report['inputs_unchanged'] = True
                report['full_horizons_complete'] = (total == BUDGET and all(case['done_reason'] == 'HORIZON_12S' for case in report['cases']))
                report['all_36_reference_pass'] = report['full_horizons_complete'] and all(case['reference_thresholds_met'] for case in report['cases'])
                report['nominal_zero_delay_pass_count'] = sum(case['nominal_zero_delay_pass'] is True for case in report['cases'])
                report['by_delay_ms'] = {str(delay * 20): {
                    'cases': len(items), 'full_horizons': sum(case['done_reason'] == 'HORIZON_12S' for case in items),
                    'failures': sum(case['failure'] for case in items),
                    'reference_pass_count': sum(case['reference_thresholds_met'] for case in items)}
                    for delay in (0, 1, 2, 3)
                    for items in [[case for case in report['cases'] if case['delay_ticks'] == delay]]}
                report['status'] = 'COMPLETE' if len(report['cases']) == 36 else 'INCOMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output / 'error.txt').write_text(traceback.format_exc())
            if constructing:
                unknown = True
                report['cleanup_complete'] = False
            if environment is not None and not accounted:
                try:
                    total = add_counts(total, environment.counters())
                except Exception:
                    unknown = True
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if environment is not None:
                try:
                    environment.close()
                except Exception as exc:
                    report.update(cleanup_complete=False, error=f'cleanup {type(exc).__name__}: {exc}', status='INCOMPLETE')
            report['known_actual_counts'] = total
            report['actual_counts'] = None if unknown else total
            report['elapsed_s'] = clock() - started
            provenance.atomic_json(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--cpp-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.release, args.export_dir, args.cpp_report, args.output)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
