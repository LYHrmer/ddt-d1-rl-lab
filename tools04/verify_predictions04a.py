#!/usr/bin/env python3
"""Score the pre-registered 04a predictions and the unchanged qualification gates.

Reads only the frozen evaluation receipt and traces that the evaluator already
wrote. It runs no physics and cannot change a gate: the thresholds come from
evidence/career_contract_04/pilot04a_release_02.json, which was written before
training, and the receipt is rejected if that file is missing.

Prediction verdicts and qualification verdicts are reported separately, because
meeting the predictions does not qualify the policy and failing a prediction is
not the same as a control failure.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GATE_FIELD = {'stable_vx_mae_m_s': 'stable_vx_mae_m_s',
              'stable_wz_mae_rad_s_each_sign': 'stable_wz_mae_rad_s',
              'stop_last_second_horizontal_speed_m_s': 'stop_last_second_horizontal_speed_m_s',
              'stop_last_second_abs_wz_rad_s': 'stop_last_second_abs_wz_rad_s',
              'stop_displacement_m': 'stop_displacement_m',
              'peak_abs_roll_or_pitch_rad': 'peak_abs_roll_or_pitch_rad'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def creep_last_second(npz_path):
    """Mean body vx over 11..12 s, the residual the anchor term targets."""
    with np.load(npz_path) as handle:
        time_s = np.asarray(handle['policy_interval_start_s'], dtype=float)
        mask = (time_s >= 11.) & (time_s < 12.)
        vx = np.asarray(handle['body_vx_m_s'], dtype=float)[mask]
        vy = np.asarray(handle['body_vy_m_s'], dtype=float)[mask]
        return {'mean_body_vx_m_s': float(vx.mean()),
                'abs_mean_body_vx_m_s': float(abs(vx.mean())),
                'mean_speed_m_s': float(np.hypot(vx, vy).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-dir', type=Path, required=True,
                        help='the 04a Python evaluation directory written by the frozen evaluator')
    parser.add_argument('--release', type=Path,
                        default=ROOT/'evidence/career_contract_04/pilot04a_release_02.json')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    if release['status'] != 'FROZEN_BEFORE_TRAINING':
        raise SystemExit('The release receipt must be the pre-training frozen one')

    episodes = [json.loads(line) for line in
                (args.eval_dir/'episodes.jsonl').read_text().strip().split('\n')]
    by_sequence = {entry['sequence']: entry for entry in episodes}
    gates = release['qualification_gates_unchanged']

    qualification = {}
    for sequence, entry in by_sequence.items():
        checks = {'completed_12s': entry['threshold_flags']['completed_12s'],
                  'no_fall_or_illegal_support': entry['threshold_flags']['no_fall_or_illegal_support']}
        for gate_name, field in GATE_FIELD.items():
            value = entry[field]
            if value is None:
                # wz is not scored on the translation tasks, as in the 03 receipts.
                checks[gate_name] = None
                continue
            checks[gate_name] = bool(value <= gates[gate_name])
        passed = all(v is not False for v in checks.values()) and all(
            v is not None or k == 'stable_wz_mae_rad_s_each_sign' or k == 'stable_vx_mae_m_s'
            for k, v in checks.items())
        qualification[sequence] = {
            'checks': checks,
            'measured': {field: entry[field] for field in GATE_FIELD.values()},
            'native_steps': entry['native_steps'],
            'done_reason': entry['done_reason'],
            'evaluator_reference_thresholds_met': entry['reference_thresholds_met'],
            'pass': bool(entry['reference_thresholds_met']),
            'pass_recomputed_here': bool(passed)}

    creep = {}
    for sequence in by_sequence:
        npz = args.eval_dir/f'{sequence}_delay0_seed7001.npz'
        creep[sequence] = creep_last_second(npz) if npz.is_file() else None

    def displacement(sequence):
        return by_sequence[sequence]['stop_displacement_m']

    verdicts = []
    for prediction in release['predictions_preregistered']:
        pid = prediction['id']
        if prediction.get('not_predicted'):
            measured = by_sequence['turn_stop']['stable_wz_mae_rad_s']
            verdicts.append({**prediction, 'measured': measured,
                             'verdict': 'NOT_A_PREDICTION',
                             'met': None if measured is None else bool(measured <= prediction['threshold']),
                             'note': 'recorded for completeness; this contract never predicted it'})
            continue
        if pid == 'P1':
            measured, met = displacement('turn_stop'), displacement('turn_stop') < prediction['threshold']
        elif pid == 'P2':
            measured, met = displacement('forward_stop'), displacement('forward_stop') < prediction['threshold']
        elif pid == 'P3':
            measured = {s: creep[s]['abs_mean_body_vx_m_s'] for s in creep}
            met = all(v < prediction['threshold'] for v in measured.values())
        elif pid == 'P4':
            measured, met = displacement('reverse_stop'), displacement('reverse_stop') < prediction['threshold']
        else:
            raise SystemExit(f'Unhandled pre-registered prediction {pid}')
        verdicts.append({**prediction, 'measured': measured,
                         'met': bool(met), 'verdict': 'MET' if met else 'NOT_MET'})

    primary = next(v for v in verdicts if v.get('primary'))
    passed_count = sum(1 for v in qualification.values() if v['pass'])
    report = {
        'format': 'd1-contract04a-prediction-verdict-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'physics_steps_executed_by_this_tool': 0,
        'release_receipt': {'path': str(args.release.relative_to(ROOT)), 'sha256': sha(args.release)},
        'evaluation_dir': str(args.eval_dir),
        'evaluation_summary_sha256': sha(args.eval_dir/'summary.json'),
        'hypothesis_verdict': {
            'primary_prediction': primary['id'],
            'primary_metric': primary['metric'],
            'primary_measured': primary['measured'],
            'primary_threshold': primary['threshold'],
            'baseline_03d': primary['baseline_03d'],
            'hypothesis': 'SUPPORTED' if primary['met'] else 'REJECTED',
            'falsification_rule': release['falsification_rule']},
        'qualification_verdict': {
            'tasks_passed': passed_count,
            'tasks_total': len(qualification),
            'overall': 'PASS' if passed_count == gates['tasks_required_to_pass'] else 'FAIL',
            'gates_unchanged_from_contract': gates,
            'per_task': qualification},
        'residual_creep_11_to_12s': creep,
        'predictions': verdicts,
        'separation_note': ('Prediction verdicts test the single reward change. They do not '
                            'grant control qualification, which requires all three tasks to '
                            'pass every unchanged gate plus the deployment gates.'),
        'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'hypothesis': report['hypothesis_verdict']['hypothesis'],
                      'primary': {primary['id']: primary['measured'],
                                  'threshold': primary['threshold']},
                      'qualification': report['qualification_verdict']['overall'],
                      'tasks_passed': passed_count,
                      'output': str(args.output/'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
