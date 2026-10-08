#!/usr/bin/env python3
"""Read existing Python/C++ trajectories. Never load MuJoCo or execute dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path('/home/lyh/ddt_ros2_ws')
TOLERANCE = 1e-5


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def error(a, b, shape):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(f'Nonfinite or wrong shape: {a.shape}, {b.shape}; expected {shape}')
    return float(np.max(np.abs(a - b))), (a - b).tolist()


def compare(py, cpp):
    pr, cr = load(py/'report.json'), load(cpp/'report.json')
    protocol = load(py/'protocol.json')
    pi, ci = load(py/'initial_state.json'), load(cpp/'initial_state.json')
    p, c = rows(py/'native_trace.jsonl'), rows(cpp/'trace.jsonl')
    checks = {}
    checks['nonempty_equal_row_counts'] = len(p) == len(c) and len(p) > 0
    checks['python_native_count_contiguous'] = [r['native_step'] for r in p] == list(range(1, len(p)+1))
    checks['cpp_native_count_contiguous'] = [r['native_count'] for r in c] == list(range(1, len(c)+1))
    checks['python_counts_match_rows'] = pr['native_attempts'] == len(p) == pr['actual_counts']['native_steps']
    checks['cpp_counts_match_rows'] = cr['native_steps_attempted'] == len(c) == cr['native_steps_completed']
    count_bytes = (cpp/'native_step_counter.bin').read_bytes()
    checks['cpp_durable_native_counter_matches'] = len(count_bytes) == 16 and struct.unpack('=QQ', count_bytes) == (len(c), len(c))
    checks['python_input_identities_unchanged'] = pr['inputs_unchanged'] is True
    checks['python_initial_file_hash_matches_cpp_import'] = sha(py/'initial_state.json') == cr['initial_state_sha256']
    checks['model_hashes_match'] = pi['scene_xml_sha256'] == ci['model_sha256'] == cr['model_sha256']
    checks['cpp_schema_hashes_match'] = ci['schema_sha256'] == cr['schema_sha256']
    checks['mujoco_versions_match_3_3_0'] = protocol['mujoco_version'] == cr['mujoco_version'] == '3.3.0'
    checks['robot_hashes_match'] = pi['robot_xml_sha256'] == cr['robot_xml_sha256']
    schema = cr['schema']
    behavior_hash = schema.get('training_interface_schema_sha256', schema['audited_interface_schema_sha256'])
    checks['behavior_interface_hashes_match'] = pi['interface_schema_sha256'] == behavior_hash
    training_config = schema['training_config']
    checks['config_hashes_match'] = protocol['inputs_sha256'].get(training_config) == schema['training_config_sha256']
    policy_mode = 'policy_sha256' in protocol
    if policy_mode:
        checks['policy_hashes_match'] = protocol['policy_sha256'] == cr['policy_sha256']
        checks['policy_deployment_uses_actual_inference'] = cr['policy_inference_attempts'] >= cr['policy_updates']
    else:
        checks['action_fixture_hash_matches'] = cr['action_fixture_sha256'] in protocol['inputs_sha256'].values()
    checks['delay_ticks_match'] = protocol['delay_ticks'] == pi['delay_ticks'] == ci['delay_ticks'] == cr['delay_ticks']
    checks['fresh_reset_boundary'] = pi['episode_policy_tick'] == 0 and pi['simulation_time_s'] == 0
    checks['warmstart_explicitly_restored'] = cr.get('initial_qacc_warmstart_restored') is True
    initial_errors = {}
    for key, shape in [('qpos', (23,)), ('qvel', (22,)), ('qacc_warmstart', (22,)),
                       ('current_observation_57', (57,)), ('filtered_gyro_body', (3,)),
                       ('last_clipped_policy_action', (16,)), ('executed_action', (16,))]:
        initial_errors[key], _ = error(pi[key], ci[key], shape)
    initial_errors['history'], _ = error(np.asarray(pi['history_oldest_to_newest_10x57']).reshape(570),
                                         ci['history_oldest_to_newest_flat570'], (570,))
    checks['initial_arrays_within_tolerance'] = max(initial_errors.values()) <= TOLERANCE
    max_errors = {key: 0. for key in ['time', 'qpos', 'qvel', 'torque']}
    if policy_mode:
        max_errors['issued_action'] = 0.
    details = []
    for pyrow, crow in zip(p, c):
        index = pyrow['native_step']
        qfrc = np.asarray(pyrow['qfrc_applied'], dtype=np.float64)
        if qfrc.shape != (22,) or not np.isfinite(qfrc).all():
            raise ValueError(f'Invalid Python applied force at native {index}')
        item = {'native_step': index, 'cpp_native_step': crow['native_count'], 'errors': {}, 'differences_python_minus_cpp': {}}
        pairs = [
            ('time', pyrow['time_s'], crow['sim_time'], ()),
            ('qpos', pyrow['qpos'], crow['qpos'], (23,)),
            ('qvel', pyrow['qvel'], crow['qvel'], (22,)),
            ('torque', qfrc[6:22], crow['torque'], (16,)),
        ]
        if policy_mode:
            pairs.append(('issued_action', pyrow['issued_action'], crow['issued_action'], (16,)))
        for key, a, b, shape in pairs:
            difference, vector = error(a, b, shape)
            item['errors'][key] = difference
            item['differences_python_minus_cpp'][key] = vector
            max_errors[key] = max(max_errors[key], difference)
        checks.setdefault('python_freebase_applied_force_zero', True)
        checks['python_freebase_applied_force_zero'] &= bool(np.all(qfrc[:6] == 0.))
        checks.setdefault('native_time_matches_count', True)
        checks['native_time_matches_count'] &= abs(pyrow['time_s']-index*.002) <= 1e-10 and abs(crow['sim_time']-crow['native_count']*.002) <= 1e-10
        checks.setdefault('cpp_control_counts_match_schedule', True)
        checks['cpp_control_counts_match_schedule'] &= crow['pd_count'] == (index+1)//2 and crow['policy_count'] == (index+9)//10
        item['within_absolute_tolerance'] = max(item['errors'].values()) <= TOLERANCE
        details.append(item)
    checks['all_native_arrays_within_tolerance'] = max(max_errors.values()) <= TOLERANCE
    checks['python_budget_respected'] = len(p) <= protocol['max_native']
    checks['cpp_budget_respected'] = len(c) <= cr['planned_native_cap']
    checks['cpp_reported_native_pd_policy_periods'] = (cr['native_dt'] == .002 and cr['pd_dt'] == .004 and cr['policy_dt'] == .02)
    checks['policy_and_pd_final_counts_match'] = (pr['actual_counts']['policy_transitions'] == cr['policy_updates'] and
                                                 pr['actual_counts']['pd_updates'] == cr['pd_updates'])
    terminal = cr['terminal_reason']
    metrics = pr.get('last_metrics') or {}
    if pr['status'] == 'TERMINATED':
        expected = 'ILLEGAL_BODY_SUPPORT' if metrics.get('illegal_body_contact') else 'FALL'
        checks['matching_terminal_outcome'] = terminal == expected and len(p) == len(c)
    else:
        checks['matching_terminal_outcome'] = (pr['status'] in ('TRUNCATED', 'COMPLETED_FIXED_HORIZON') and
            terminal in ('TIME_LIMIT', 'NATIVE_BUDGET_LIMIT', 'ACTION_FIXTURE_END') and len(p) == len(c))
    checks['no_python_error'] = pr.get('error') is None
    report = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'physics_steps_executed_by_comparator': 0,
              'comparison_mode': 'closed_loop_onnx' if policy_mode else 'fixed_action',
              'absolute_tolerance': TOLERANCE, 'python_native_rows': len(p), 'cpp_native_rows': len(c),
              'checks': checks, 'initial_max_absolute_errors': initial_errors,
              'max_absolute_errors': max_errors, 'python_status': pr['status'], 'cpp_terminal_reason': terminal,
              'first_bad_native_row': next((r['native_step'] for r in details if not r['within_absolute_tolerance']), None),
              'evidence_sha256': {str(path): sha(path) for path in
                  [py/'report.json', py/'protocol.json', py/'initial_state.json', py/'native_trace.jsonl',
                   cpp/'report.json', cpp/'initial_state.json', cpp/'trace.jsonl', cpp/'native_step_counter.bin']}}
    return report, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-run', type=Path, required=True)
    parser.add_argument('--cpp-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report, details = compare(args.python_run.resolve(), args.cpp_run.resolve())
    except Exception as exc:
        report, details = {'status': 'FAIL', 'physics_steps_executed_by_comparator': 0,
                           'error': f'{type(exc).__name__}: {exc}', 'absolute_tolerance': TOLERANCE}, []
    with (args.output/'native_differences.jsonl').open('x') as stream:
        for row in details:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
