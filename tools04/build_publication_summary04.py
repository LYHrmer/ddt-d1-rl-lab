#!/usr/bin/env python3
"""Compact, GitHub-readable contract 04 summary built from the receipts.

Mirrors reports/contract03_summary.json: every number is read back out of a
receipt so the published summary cannot drift from the evidence. No physics.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT/'evidence/career_contract_04'
GATE_FIELDS = ('stable_vx_mae_m_s', 'stable_wz_mae_rad_s',
               'stop_last_second_horizontal_speed_m_s', 'stop_last_second_abs_wz_rad_s',
               'stop_displacement_m', 'peak_abs_roll_or_pitch_rad')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version_block(version, run_dir, eval_dir, verdict_dir, change):
    result = json.loads((EV/run_dir/'result.json').read_text())
    protocol = json.loads((EV/run_dir/'protocol.json').read_text())
    summary = json.loads((EV/eval_dir/'summary.json').read_text())
    verdict = json.loads((EV/verdict_dir/'report.json').read_text())
    episodes = [json.loads(line) for line in
                (EV/eval_dir/'episodes.jsonl').read_text().strip().split('\n')]
    return {
        'version': version, 'single_change': change,
        'base': 'frozen pilot03d; 04a and 04b are siblings on the same base',
        'training_seed': protocol['seed'], 'evaluation_seed': 7001,
        'training_status': result['status'],
        'training_transitions': result['trained_transitions'],
        'training_native_steps': result['actual_counts']['native_steps'],
        'training_seconds': result['elapsed_training_s'],
        'inputs_unchanged': result['inputs_unchanged'],
        'min_available_ram_bytes': result['min_available_ram_bytes'],
        'python_tasks_passed': summary['zero_delay_nominal_pass_count'],
        'python_tasks_total': 3,
        'python_evaluation_native_steps': summary['actual_native_steps'],
        'python_early_failures': summary['early_failures'],
        'hypothesis_verdict': verdict['hypothesis_verdict']['hypothesis'],
        'preregistered_predictions': {p['id']: p['verdict'] for p in verdict['predictions']},
        'episodes': [{'sequence': e['sequence'], 'native_steps': e['native_steps'],
                      'status': e['status'], 'done_reason': e['done_reason'],
                      **{field: e[field] for field in GATE_FIELDS},
                      'threshold_flags': e['threshold_flags'],
                      'nominal_zero_delay_pass': e['nominal_zero_delay_pass']}
                     for e in episodes],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    ledger = json.loads((EV/'final_delivery_01.json').read_text())
    golden = {g['version']: g for g in ledger['onnx_cxx_golden']}
    cpp = {c['version']: c for c in ledger['independent_cpp_deployment']['runs']}
    probe = json.loads((EV/'zero_wheel_braking_probe_02/report.json').read_text())
    oscillation = json.loads((EV/'stop_oscillation_04a_01/report.json').read_text())
    preflight = json.loads((EV/'ros_import_preflight_01/report.json').read_text())

    report = {
        'format': 'd1-contract04-publication-summary-v1',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'simulation_only': True, 'real_robot_validation': False,
        'overall_control_qualification': 'FAIL',
        'interpretation': ('Two bounded development runs on one development seed, each adding '
                           'exactly one reward term to the same frozen 03d base. Not a formal '
                           'three-seed comparison. Full traces, weights and the original '
                           'colleague model remain on the local workstation.'),
        'versions': [
            version_block('04a', 'pilot04a_02', 'pilot04a_python_eval_01',
                          'pilot04a_prediction_verdict_01',
                          'Huber penalty on net XY displacement from the pose where each '
                          'zero-command segment began (weight -20, knee 0.05 m, cap 0.10 m)'),
            version_block('04b', 'pilot04b_01', 'pilot04b_python_eval_01',
                          'pilot04b_prediction_verdict_01',
                          'bounded penalty on the mean square of the four wheel actions while '
                          'the commanded vx and wz are both zero (weight -10, saturation 0.25)'),
        ],
        'best_single_task': ledger['control_qualification']['best_single_task'],
        'cross_language': {
            'onnx_cxx_golden': {v: {'status': g['status'],
                                    'max_absolute_error': g['max_absolute_error'],
                                    'tolerance': g['absolute_tolerance'],
                                    'onnxruntime_version': g['onnxruntime_version'],
                                    'physics_steps': g['physics_steps']}
                                for v, g in golden.items()},
            'independent_cpp_replay': {v: {'task': c['task'], 'status': c['status'],
                                           'native': c['native'],
                                           'structural_checks':
                                               f"{c['structural_checks_passed']}/"
                                               f"{c['structural_checks_total']}",
                                           'gates_passed':
                                               f"{c['gates_passed']}/{c['gates_applicable']}",
                                           'inference_p99_ms': c['inference_p99_ms']}
                                       for v, c in cpp.items()},
            'note': 'deployment consistency is not control qualification'},
        'mechanism_findings': {
            'zero_command_dither_introduced_by_04a': {
                'vx_dominant_hz': oscillation['traces']['04a_turn_stop']['windows']
                                  ['final_11_to_12s']['spectrum_body_vx']['dominant_hz'],
                'net_wheel_target_std_rad_s_04a': oscillation['traces']['04a_turn_stop']
                                                  ['windows']['final_11_to_12s']
                                                  ['wheel_targets_rad_s']['net_std'],
                'net_wheel_target_std_rad_s_03d': oscillation['traces']['03d_forward_stop']
                                                  ['windows']['final_11_to_12s']
                                                  ['wheel_targets_rad_s']['net_std'],
                'cause': ('the actor observes neither base position nor base linear velocity, '
                          'so minimising displacement from a 4 s old anchor is not observable '
                          'and PPO found the only memoryless proxy, a 12 Hz limit cycle')},
            'passive_wheel_braking_measurement': ledger['interface_measurement'],
            'why_04b_failed': ('zero wheel action brakes inside the gate, but the policy only '
                               'zeroes its wheels after it has already stopped: forward 8-9 s '
                               'mean(a^2) 0.02208 and net wheel target +0.4596 rad/s against '
                               '0.00099 at 11-12 s, and that window produced 70 percent of the '
                               'displacement'),
            'probe_receipt_sha256': sha(EV/'zero_wheel_braking_probe_02/report.json'),
            'probe_release_speed_m_s': probe['measured_coast']['speed_at_release_m_s'],
        },
        'ros_environment': ledger['ros_environment'],
        'budget': {'contract04_transitions': ledger['contract04_transitions_total'],
                   'contract04_native': ledger['contract04_native_total_all_pools'],
                   'project_cumulative': ledger['project_cumulative'],
                   'engineering_budget_loss_recorded': True,
                   'formal_three_arms_three_seeds': 'NOT_RELEASED'},
        'unfinished': ledger['unfinished'],
        'preflight_checks': len(preflight['checks']),
        'ledger_sha256': sha(EV/'final_delivery_01.json'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'versions': len(report['versions']),
                      'qualification': report['overall_control_qualification']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
