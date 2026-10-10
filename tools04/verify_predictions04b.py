#!/usr/bin/env python3
"""Score the pre-registered 04b predictions and the unchanged qualification gates.

Reads only the frozen evaluation receipt and traces the evaluator already wrote,
plus the pre-training release receipt that fixes the thresholds. Runs no physics.

Handles episodes that terminated early: a task that fell has no stop-phase
metrics, which counts against every prediction that needs them rather than
being silently skipped.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
GATE_FIELD = {'stable_vx_mae_m_s': 'stable_vx_mae_m_s',
              'stable_wz_mae_rad_s_each_sign': 'stable_wz_mae_rad_s',
              'stop_last_second_horizontal_speed_m_s': 'stop_last_second_horizontal_speed_m_s',
              'stop_last_second_abs_wz_rad_s': 'stop_last_second_abs_wz_rad_s',
              'stop_displacement_m': 'stop_displacement_m',
              'peak_abs_roll_or_pitch_rad': 'peak_abs_roll_or_pitch_rad'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def window_metrics(npz_path, lo, hi):
    with np.load(npz_path) as handle:
        time_s = np.asarray(handle['policy_interval_start_s'], dtype=float)
        mask = (time_s >= lo) & (time_s < hi)
        if mask.sum() < 2:
            return None
        issued = np.asarray(handle['issued_action'], dtype=float)[mask][:, list(WHEEL_ACTION_INDEX)]
        net = (issued*5.).mean(axis=1)
        x = np.asarray(handle['base_x_m'], dtype=float)[mask]
        y = np.asarray(handle['base_y_m'], dtype=float)[mask]
        vx = np.asarray(handle['body_vx_m_s'], dtype=float)[mask]
        vy = np.asarray(handle['body_vy_m_s'], dtype=float)[mask]
        return {'window_s': [lo, hi], 'policy_ticks': int(mask.sum()),
                'mean_wheel_action_squared_per_tick': float(np.mean(np.square(issued))),
                'net_wheel_target_mean_rad_s': float(net.mean()),
                'net_wheel_target_std_rad_s': float(net.std()),
                'per_wheel_abs_mean_rad_s': float(np.abs(issued*5.).mean()),
                'negative_net_fraction': float(np.mean(net < 0.)),
                'euclidean_displacement_m': float(np.hypot(x[-1]-x[0], y[-1]-y[0])),
                'path_length_m': float(np.hypot(np.diff(x), np.diff(y)).sum()),
                'horizontal_speed_mean_m_s': float(np.hypot(vx, vy).mean()),
                'body_vx_mean_m_s': float(vx.mean()),
                'body_vx_std_m_s': float(vx.std())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--release', type=Path,
                        default=ROOT/'evidence/career_contract_04/pilot04b_release_01.json')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    if release['status'] != 'FROZEN_BEFORE_TRAINING':
        raise SystemExit('The release receipt must be the pre-training frozen one')
    gates = release['qualification_gates_unchanged']

    episodes = [json.loads(line) for line in
                (args.eval_dir/'episodes.jsonl').read_text().strip().split('\n')]
    by_sequence = {entry['sequence']: entry for entry in episodes}

    qualification = {}
    for sequence, entry in by_sequence.items():
        checks = {'completed_12s': bool(entry['threshold_flags']['completed_12s']),
                  'no_fall_or_illegal_support':
                      bool(entry['threshold_flags']['no_fall_or_illegal_support'])}
        for gate_name, field in GATE_FIELD.items():
            value = entry[field]
            checks[gate_name] = None if value is None else bool(value <= gates[gate_name])
        qualification[sequence] = {
            'checks': checks,
            'measured': {field: entry[field] for field in GATE_FIELD.values()},
            'native_steps': entry['native_steps'], 'status': entry['status'],
            'done_reason': entry['done_reason'], 'early_failure': bool(entry['failure']),
            'pass': bool(entry['reference_thresholds_met'])}

    windows = {}
    for sequence in by_sequence:
        npz = args.eval_dir/f'{sequence}_delay0_seed7001.npz'
        if not npz.is_file():
            windows[sequence] = None
            continue
        windows[sequence] = {name: window_metrics(npz, lo, hi) for name, (lo, hi) in
                             (('8_to_9s', (8., 9.)), ('9_to_10s', (9., 10.)),
                              ('10_to_11s', (10., 11.)), ('11_to_12s', (11., 12.)),
                              ('8_to_12s', (8., 12.0001)))}

    def finished(sequence):
        return qualification[sequence]['checks']['completed_12s']

    verdicts = []
    for prediction in release['predictions_preregistered']:
        pid = prediction['id']
        if prediction.get('not_predicted'):
            measured = by_sequence['turn_stop']['stable_wz_mae_rad_s']
            verdicts.append({**prediction, 'measured': measured, 'verdict': 'NOT_A_PREDICTION',
                             'met': None if measured is None else bool(measured <= prediction['threshold']),
                             'note': 'recorded for completeness; this contract never predicted it'})
            continue
        if pid == 'P1':
            per_task = {}
            for sequence in ('forward_stop', 'reverse_stop', 'turn_stop'):
                entry = by_sequence[sequence]
                displacement = entry['stop_displacement_m']
                speed = entry['stop_last_second_horizontal_speed_m_s']
                ok = (finished(sequence) and displacement is not None and speed is not None
                      and displacement < prediction['displacement_threshold']
                      and speed < prediction['speed_threshold'])
                per_task[sequence] = {'completed_12s': finished(sequence),
                                      'stop_displacement_m': displacement,
                                      'stop_last_second_horizontal_speed_m_s': speed,
                                      'both_met': bool(ok)}
            measured, met = per_task, all(v['both_met'] for v in per_task.values())
        elif pid == 'P2':
            measured = {s: (windows[s]['11_to_12s']['mean_wheel_action_squared_per_tick']
                            if windows[s] and windows[s]['11_to_12s'] else None)
                        for s in windows}
            met = all(finished(s) and v is not None and v < prediction['threshold']
                      for s, v in measured.items())
        elif pid == 'P3':
            measured = {s: (windows[s]['11_to_12s']['net_wheel_target_std_rad_s']
                            if windows[s] and windows[s]['11_to_12s'] else None)
                        for s in windows}
            met = all(finished(s) and v is not None and v < prediction['threshold']
                      for s, v in measured.items())
        elif pid == 'P4':
            entry = windows['forward_stop']['8_to_9s'] if windows['forward_stop'] else None
            measured = entry['euclidean_displacement_m'] if entry else None
            met = measured is not None and measured < prediction['threshold']
        else:
            raise SystemExit(f'Unhandled pre-registered prediction {pid}')
        verdicts.append({**prediction, 'measured': measured, 'met': bool(met),
                         'verdict': 'MET' if met else 'NOT_MET',
                         'per_completed_task_only': None})

    # P2/P3 restricted to the tasks that actually finished, reported separately so
    # a fall does not hide whether the wheel term worked where it could be observed.
    completed = [s for s in by_sequence if finished(s)]
    for verdict in verdicts:
        if verdict['id'] in ('P2', 'P3') and isinstance(verdict['measured'], dict):
            threshold = verdict['threshold']
            verdict['per_completed_task_only'] = {
                'tasks': completed,
                'all_met_on_completed_tasks': all(
                    verdict['measured'][s] is not None and verdict['measured'][s] < threshold
                    for s in completed)}

    primary = next(v for v in verdicts if v.get('primary'))
    passed_count = sum(1 for v in qualification.values() if v['pass'])
    report = {
        'format': 'd1-contract04b-prediction-verdict-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'physics_steps_executed_by_this_tool': 0,
        'release_receipt': {'path': str(args.release.relative_to(ROOT)), 'sha256': sha(args.release)},
        'evaluation_dir': str(args.eval_dir),
        'evaluation_summary_sha256': sha(args.eval_dir/'summary.json'),
        'hypothesis_verdict': {
            'primary_prediction': primary['id'], 'primary_metric': primary['metric'],
            'primary_measured': primary['measured'],
            'hypothesis': 'SUPPORTED' if primary['met'] else 'REJECTED',
            'falsification_rule': release['falsification_rule']},
        'qualification_verdict': {
            'tasks_passed': passed_count, 'tasks_total': len(qualification),
            'overall': 'PASS' if passed_count == gates['tasks_required_to_pass'] else 'FAIL',
            'early_failures': sum(1 for v in qualification.values() if v['early_failure']),
            'gates_unchanged_from_contract': gates, 'per_task': qualification},
        'stop_phase_windows': windows,
        'predictions': verdicts,
        'separation_note': ('Prediction verdicts test the single reward change. They do not '
                            'grant control qualification, which requires all three tasks to '
                            'pass every unchanged gate plus the deployment gates.'),
        'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'hypothesis': report['hypothesis_verdict']['hypothesis'],
                      'qualification': report['qualification_verdict']['overall'],
                      'tasks_passed': passed_count,
                      'early_failures': report['qualification_verdict']['early_failures'],
                      'predictions': {v['id']: v['verdict'] for v in verdicts},
                      'output': str(args.output/'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
