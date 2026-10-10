#!/usr/bin/env python3
"""Pre-register the contract 04b identity before its single training run.

Writes the frozen SHA-256 of every input the run depends on, the pre-registered
predictions and the budget caps, so none of it can be renegotiated after the
result is known. Executes no physics and starts no training.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))

# docs/career_execution_contract_04b.md section 7; edited only before training.
PREDICTIONS = [
    {'id': 'P1', 'primary': True,
     'metric': 'all three tasks satisfy stop_displacement_m < 0.05 AND '
               'stop_last_second_horizontal_speed_m_s < 0.02 simultaneously',
     'displacement_threshold': .05, 'speed_threshold': .02,
     'baseline_03d': {'displacement': [.080584, .016340, .060764],
                      'last_second_speed': [.004507, .002254, .014936]},
     'baseline_04a': {'displacement': [.075552, .077812, .034852],
                      'last_second_speed': [.025420, .029180, .022621]},
     'why': ('03d met the speed gate but failed displacement on two tasks; 04a met '
             'the turn displacement gate but failed speed on all three. Neither has '
             'met both at once, so this is the cleanest test of the zero-wheel-action '
             'mechanism.')},
    {'id': 'P2', 'primary': False,
     'metric': 'zero-command mean(wheel action^2) per tick over 11..12 s',
     'threshold': .02, 'direction': 'below',
     'baseline_03d': [.06516, .05481], 'baseline_04a': [.17363]},
    {'id': 'P3', 'primary': False,
     'metric': 'zero-command net wheel target standard deviation over 11..12 s, rad/s',
     'threshold': .20, 'direction': 'below',
     'baseline_03d': [.0600, .0336], 'baseline_04a': [.9756]},
    {'id': 'P4', 'primary': False,
     'metric': 'forward_stop Euclidean displacement over 8..9 s',
     'threshold': .045, 'direction': 'below',
     'baseline_03d': .04981, 'baseline_04a': .08535,
     'why': 'pure wheel damping predicts 0.0383-0.0431 m from 0.1965-0.221 m/s'},
    {'id': 'P5', 'primary': False, 'not_predicted': True,
     'metric': 'turn_stop stable wz MAE rad/s', 'threshold': .08,
     'baseline_03d': .102882, 'baseline_04a': .111786,
     'why': 'the wheel magnitude term is inactive while a nonzero turn command is held'},
]
BUDGET = {'training_transitions': 1048576, 'training_native': 10485760, 'training_runs': 1,
          'engineering_native': 2000, 'python_eval_native': 18000,
          'cpp_deployment_native': 18000, 'cpp_deployment_trials': 3,
          'total_new_native_cap': 10523760,
          'formal_three_arms_three_seeds': 'NOT_RELEASED',
          'contract03_pools_borrowed': 'NONE', 'contract04a_pools_borrowed': 'NONE',
          'historical_native_before_this_contract': 42401110}
QUALIFICATION_GATES = {'completed_12s': True, 'no_fall_or_illegal_support': True,
                       'stable_vx_mae_m_s': .06, 'stable_wz_mae_rad_s_each_sign': .08,
                       'stop_last_second_horizontal_speed_m_s': .02,
                       'stop_last_second_abs_wz_rad_s': .03,
                       'stop_displacement_m': .05, 'peak_abs_roll_or_pitch_rad': .20,
                       'tasks_required_to_pass': 3}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite an existing release receipt: {args.output}')

    from d1_rl.factory04b import environment_sources
    config = ROOT/'training/configs/pilot04b.json'
    static = ROOT/'evidence/career_contract_04/task04b_static_03/report.json'
    preflight = ROOT/'evidence/career_contract_04/ros_import_preflight_01/report.json'
    evidence_04a = {
        'prediction_verdict': ROOT/'evidence/career_contract_04/pilot04a_prediction_verdict_01/report.json',
        'offline_trace_diagnosis': ROOT/'evidence/career_contract_04/offline_trace_diagnosis_04a_01/report.json',
        'stop_oscillation': ROOT/'evidence/career_contract_04/stop_oscillation_04a_01/report.json',
        'cxx_score': ROOT/'evidence/career_contract_04/pilot04a_deployment_plan_01/score03_turn_stop/report.json',
    }

    inputs = [config, ROOT/'docs/career_execution_contract_04b.md',
              ROOT/'tools04/run_frozen_entry.py', ROOT/'tools04/check_task04b_static.py',
              ROOT/'tools04/check_task04a_static.py',
              ROOT/'tools04/analyze_pilot_traces04.py',
              ROOT/'tools04/characterize_stop_oscillation04.py',
              ROOT/'tools/train_d1_ppo.py', ROOT/'tools/export_d1_ppo.py',
              ROOT/'tools/evaluate_d1_ppo.py',
              ROOT/'training/d1_rl/forkserver_bootstrap.py',
              *(Path(p) for p in environment_sources(config))]
    seen = {}
    for path in inputs:
        seen[str(Path(path).resolve().relative_to(ROOT))] = sha(path)

    static_report = json.loads(static.read_text())
    if static_report['status'] != 'PASS':
        raise SystemExit('Static acceptance is not PASS; training is not authorised')
    if json.loads(preflight.read_text())['status'] != 'PASS':
        raise SystemExit('ROS import preflight is not PASS')
    if 'task04a.py' in {Path(p).name for p in environment_sources(config)}:
        raise SystemExit('04b must not inherit the 04a task')

    receipt = {
        'format': 'd1-contract04b-release-v1',
        'status': 'FROZEN_BEFORE_TRAINING',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'contract': 'docs/career_execution_contract_04b.md',
        'relation_to_04a': ('sibling, not successor: 04a and 04b each add exactly one term '
                            'to the same 03d base, so comparing them isolates the choice of '
                            'measured quantity. 04b does not inherit the 04a anchor term.'),
        'single_change': ('one added reward term: wheel_action_mean_squared, a bounded penalty '
                          'on the mean square of the four wheel actions while the commanded vx '
                          'and wz are both zero'),
        'term': {'weight': -10.0, 'saturation': 0.25,
                 'gate': 'interval_command vx == 0 and wz == 0',
                 'worst_case_per_second': 2.5,
                 'approximate_positive_floor_per_second': 4.0375,
                 'gradient_live_over_measured_range': True},
        'mechanism_claim': ('with wheel kp=0, kd=0.5 N m/(rad/s), r=0.087 m and 51.5 kg '
                            'effective mass, a wheel action of zero is a pure damper that '
                            'brakes from 0.1965 m/s in about 0.0383 m and then holds still, '
                            'satisfying the 0.05 m displacement gate and the 0.02 m/s '
                            'last-second speed gate at once. Active braking is a bonus, '
                            'not a requirement.'),
        'inherits_unchanged_from_03d': ['command schedule', 'all 03c/03d reward terms and weights',
                                        'actor interface and history', 'observation', 'action scales',
                                        'action clip', 'timing', 'reset', 'termination', 'PPO'],
        'excluded_by_contract': ['the 04a anchor displacement term', 'teacher policy',
                                 'behaviour cloning', 'warm start',
                                 'true linear velocity actor input', 'PID assistance',
                                 'action post-processing', 'retuned tracking scales'],
        'training_plan': {'runs': 1, 'kind': 'pilot', 'arm': 'B', 'seed': 101, 'envs': 8,
                          'transitions': 1048576, 'random_initialization': True,
                          'max_wall_s': 5400, 'min_available_ram_bytes': 3*1024**3,
                          'resume_or_retry_allowed': False, 'checkpoint_selection': 'final_only'},
        'budget': BUDGET,
        'qualification_gates_unchanged': QUALIFICATION_GATES,
        'predictions_preregistered': PREDICTIONS,
        'falsification_rule': ('P1 not met rejects the hypothesis. If P2 and P3 are met while '
                              'P1 is not, the interface arithmetic itself is wrong and the '
                              'braking distance must be remeasured rather than the reward '
                              'retuned. No weight retune and retry under this contract.'),
        'expected_outcome_statement': ('3/3 qualification is NOT expected. turn_stop is also '
                                       'blocked by its wz MAE (03d 0.1029, 04a 0.1118 against '
                                       'a 0.08 gate), which this term does not address. Best '
                                       'expected outcome is 2/3.'),
        'physics_steps_executed_by_this_receipt': 0,
        'evidence_before_training': {
            'static_checks': {'path': str(static.relative_to(ROOT)), 'sha256': sha(static),
                              'status': static_report['status'],
                              'checks': len(static_report['checks']),
                              'native_integration_steps_executed':
                                  static_report['native_integration_steps_executed']},
            'static_check_failures_preserved': [
                'evidence/career_contract_04/task04b_static_01',
                'evidence/career_contract_04/task04b_static_02'],
            'ros_import_preflight': {'path': str(preflight.relative_to(ROOT)),
                                     'sha256': sha(preflight), 'watchdog_validated': False},
            'contract_04a_results': {name: {'path': str(path.relative_to(ROOT)),
                                            'sha256': sha(path)}
                                     for name, path in evidence_04a.items()},
        },
        'inputs_sha256': dict(sorted(seen.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'inputs': len(seen),
                      'status': receipt['status'],
                      'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
