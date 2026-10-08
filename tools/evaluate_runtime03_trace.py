#!/usr/bin/env python3
"""Score existing C++ traces with the existing Python evaluator's exact function.

This program does not import MuJoCo, Torch, ROS or ONNX and cannot run physics.
The AST loader executes only trace_metrics and reads its literal constants.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_SOURCE = ROOT/'tools/evaluate_d1_ppo.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics_contract():
    tree = ast.parse(METRICS_SOURCE.read_text(), filename=str(METRICS_SOURCE))
    names = {'POLICY_DT_S', 'EPISODE_TICKS', 'FAILURE_PENALTY_J', 'MISSING_INTERVAL_ERROR',
             'SEQUENCES', 'THRESHOLDS'}
    namespace = {'np': np}
    functions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                namespace[name] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == 'trace_metrics':
            functions.append(node)
    if len(functions) != 1 or not names <= namespace.keys():
        raise ValueError('Expected literal evaluation constants and one trace_metrics function')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(METRICS_SOURCE), 'exec'), namespace)
    if namespace['POLICY_DT_S'] != .02 or namespace['EPISODE_TICKS'] != 600:
        raise ValueError('Evaluation horizon is no longer the frozen 12 s / 600 policy ticks')
    return namespace


def command_at(points, time_s):
    selected = points[0]
    for point in points:
        if time_s + 1e-10 >= point['start_s']:
            selected = point
    return np.array([selected['vx'], 0., selected['wz']], dtype=np.float64)


def exact_runtime_sequence(points):
    return {'segments': [
        {'start': point['start_s'],
         'end': points[i+1]['start_s'] if i+1 < len(points) else 12.,
         'command': [point['vx'], 0., point['wz']]}
        for i, point in enumerate(points)]}


def evaluate(args):
    contract = metrics_contract()
    report = read_json(args.run/'report.json')
    initial = read_json(args.run/'initial_state.json')
    schema = read_json(args.schema)
    checks = {
        'model_sha256': report['model_sha256'] == args.expected_model_sha256,
        'policy_sha256': report['policy_sha256'] == args.expected_policy_sha256,
        'schema_sha256': report['schema_sha256'] == args.expected_schema_sha256 == sha(args.schema),
        'embedded_schema_matches': report['schema'] == schema,
        'runtime_binary_sha256': report['runtime_binary_sha256'] == args.expected_runtime_sha256,
        'model_and_schema_initial_identities': initial['model_sha256'] == report['model_sha256'] and initial['schema_sha256'] == report['schema_sha256'],
        'exact_libraries': report['mujoco_version'] == '3.3.0' and report['onnxruntime_version'] == '1.10.0',
        'replay_mode': report['mode'] == 'replay',
        'native_pd_policy_periods': report['native_dt'] == .002 and report['pd_dt'] == .004 and report['policy_dt'] == .02,
        'requested_horizon_12s': report['seconds_requested'] == 12.,
        'native_budget_frozen': report['planned_native_cap'] == 6000 == args.max_native,
        'schema03b': schema['interface_id'] == 'd1-selftrained-pilot03b-v2',
        'declared_sequence_matches_full_python_horizon': read_json(args.run/'sequence.json') == exact_runtime_sequence(contract['SEQUENCES'][args.sequence]),
    }
    native = []
    with (args.run/'trace.jsonl').open() as stream:
        for line in stream:
            if line.strip():
                native.append(json.loads(line))
                if len(native) > args.max_native:
                    raise ValueError('Trace exceeds frozen 6000-native deployment budget')
    n_native = len(native)
    checks['native_counts_complete'] = report['native_steps_attempted'] == n_native == report['native_steps_completed']
    checks['native_counter_file_matches'] = (args.run/'native_step_counter.bin').read_bytes() == struct.pack('=QQ', n_native, n_native)
    checks['native_counts_contiguous'] = [row['native_count'] for row in native] == list(range(1, n_native+1))
    checks['report_pd_policy_counts'] = report['pd_updates'] == (n_native+1)//2 and report['policy_updates'] == (n_native+9)//10
    checks['native_time_matches_actual_count'] = all(abs(row['sim_time']-row['native_count']*.002) <= 1e-10 for row in native)
    checks['native_pd_policy_count_schedule'] = all(row['pd_count'] == (row['native_count']+1)//2 and row['policy_count'] == (row['native_count']+9)//10 for row in native)
    expected_policy_times = np.arange(report['policy_updates'], dtype=np.float64)*.02
    expected_pd_times = np.arange(report['pd_updates'], dtype=np.float64)*.004
    checks['reported_policy_update_times'] = np.allclose(report['policy_update_sim_times'], expected_policy_times, rtol=0, atol=1e-12)
    checks['reported_pd_update_times'] = np.allclose(report['pd_update_sim_times'], expected_pd_times, rtol=0, atol=1e-12)
    checks['commands_match_python_intervals'] = all(np.allclose(row['command'], command_at(contract['SEQUENCES'][args.sequence], ((row['native_count']-1)//10)*.02), rtol=0, atol=1e-12) for row in native)
    checks['complete_policy_intervals_only'] = n_native % 10 == 0
    checks['no_runtime_error'] = 'error' not in report
    trace = {name: [] for name in ('reward', 'body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s',
        'command_during_interval', 'base_roll_rad', 'base_pitch_rad', 'fallen', 'illegal_body_contact')}
    positions = [np.asarray(initial['qpos'][:2], dtype=np.float64)]
    sampled = []
    for row in native[9::10]:
        vector = np.asarray([row['body_vx'], row['body_velocity'][1], row['body_wz'],
                             row['roll'], row['pitch'], row['tilt'], row['base_z'], *row['position']], dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError('Nonfinite truth sample')
        fallen = row['tilt'] > .6 or row['base_z'] < report['h_nom']-.18
        illegal = bool(row['illegal_body_support_since_policy'])
        values = {'reward': 0., 'body_vx_m_s': row['body_vx'], 'body_vy_m_s': row['body_velocity'][1],
                  'body_wz_rad_s': row['body_wz'], 'command_during_interval': row['command'],
                  'base_roll_rad': row['roll'], 'base_pitch_rad': row['pitch'],
                  'fallen': fallen, 'illegal_body_contact': illegal}
        for key, value in values.items():
            trace[key].append(value)
        positions.append(np.asarray(row['position'][:2], dtype=np.float64))
        sampled.append({'policy_interval_start_s': (row['native_count']-10)*.002,
                        'truth_sample_time_s': row['sim_time'], 'native_count': row['native_count'],
                        **{key: value for key, value in values.items() if key != 'reward'}})
    terminal = report['terminal_reason']
    # Contact force can cross the threshold only in the final boundary forward;
    # preserve that actual runtime terminal evidence, not merely the last native log.
    if sampled and terminal in ('FALL', 'ILLEGAL_BODY_SUPPORT'):
        key = 'fallen' if terminal == 'FALL' else 'illegal_body_contact'
        trace[key][-1] = True
        sampled[-1][key] = True
    completed = n_native == 6000 and terminal == 'TIME_LIMIT' and report['status'] == 'COMPLETED'
    done_reason = 'HORIZON_12S' if completed else terminal
    scores = contract['trace_metrics'](args.sequence, report['delay_ticks'], report['seed'], trace, positions, done_reason)
    valid = bool(all(checks.values()))
    return {
        'status': 'INVALID' if not valid else ('PASS' if scores['reference_thresholds_met'] else 'FAIL'),
        'physics_steps_executed_by_evaluator': 0, 'evidence_valid': valid,
        'checks': {key: bool(value) for key, value in checks.items()}, 'metrics': scores,
        'actual_native_steps': report['native_steps_completed'],
        'actual_policy_updates': report['policy_updates'], 'actual_pd_updates': report['pd_updates'],
        'metric_function_source': str(METRICS_SOURCE), 'metric_function_source_sha256': sha(METRICS_SOURCE),
        'reuse_method': 'AST-extracted original trace_metrics function with its original literal constants; no rewritten scoring formula and no Torch/MuJoCo/ROS import',
        'reward_note': 'C++ does not log reward. Zero placeholders provide only the length expected by trace_metrics; reward is not scored or reported as measured.',
        'sampling': 'One truth sample at the end of each complete 10-native interval; metric time is the interval start, exactly like the Python evaluator',
        'thresholds': contract['THRESHOLDS'], 'inference_latency': report['inference_latency'],
        'runtime_binary_sha256': report['runtime_binary_sha256'],
        'input_sha256': {str(path): sha(path) for path in [args.run/'report.json', args.run/'trace.jsonl',
            args.run/'sequence.json', args.run/'initial_state.json', args.run/'native_step_counter.bin', args.schema]},
        'scope': 'One C++ replay deployment trial only; not a 3-seed result, live watchdog check or terrain traversal claim',
    }, sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--sequence', choices=['forward_stop', 'reverse_stop', 'turn_stop'], required=True)
    parser.add_argument('--schema', type=Path, required=True)
    for name in ('model', 'policy', 'schema', 'runtime'):
        parser.add_argument(f'--expected-{name}-sha256', required=True)
    parser.add_argument('--max-native', type=int, default=6000, choices=[6000])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result, sampled = evaluate(args)
    except Exception as exc:
        result, sampled = {'status': 'INVALID', 'physics_steps_executed_by_evaluator': 0,
                           'error': f'{type(exc).__name__}: {exc}'}, []
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (args.output/'policy_truth.jsonl').open('x') as stream:
        for row in sampled:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
