#!/usr/bin/env python3
"""Pre-register the contract 04a identity before the single training run.

Writes the frozen SHA-256 of every input the run depends on, the pre-registered
predictions, and the budget caps, so that nothing can be renegotiated after the
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

# docs/career_execution_contract_04a.md section 7; edited only before training.
PREDICTIONS = [
    {'id': 'P1', 'primary': True, 'metric': 'turn_stop stop_displacement_m',
     'baseline_03d': .060764, 'threshold': .05, 'direction': 'below',
     'why': 'turn_stop has vx command identically zero, so its displacement failure is pure creep'},
    {'id': 'P2', 'primary': False, 'metric': 'forward_stop stop_displacement_m',
     'baseline_03d': .080584, 'threshold': .05, 'direction': 'below'},
    {'id': 'P3', 'primary': False, 'metric': 'all three tasks |mean body_vx| over 11..12 s',
     'baseline_03d': [.00449, .00224, .01453], 'threshold': .005, 'direction': 'below'},
    {'id': 'P4', 'primary': False, 'metric': 'reverse_stop stop_displacement_m',
     'baseline_03d': .016340, 'threshold': .05, 'direction': 'below',
     'why': 'may rise once the Euclidean cancellation of the forward bias disappears'},
    {'id': 'P5', 'primary': False, 'not_predicted': True,
     'metric': 'turn_stop stable wz MAE rad/s', 'baseline_03d': .102882, 'threshold': .08,
     'why': 'the anchor term is inactive while a nonzero turn command is held'},
]
BUDGET = {'training_transitions': 1048576, 'training_native': 10485760, 'training_runs': 1,
          'engineering_native': 2000, 'python_eval_native': 18000,
          'cpp_deployment_native': 18000, 'cpp_deployment_trials': 3,
          'total_new_native_cap': 10523760,
          'formal_three_arms_three_seeds': 'NOT_RELEASED',
          'contract03_pools_borrowed': 'NONE'}
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

    from d1_rl.factory04 import environment_sources
    config = ROOT/'training/configs/pilot04a.json'
    static = ROOT/'evidence/career_contract_04/task04a_static_02/report.json'
    preflight = ROOT/'evidence/career_contract_04/ros_import_preflight_01/report.json'
    diagnosis = ROOT/'evidence/career_contract_04/offline_trace_diagnosis_01/report.json'
    probe = ROOT/'evidence/career_contract_04/reward_signal_probe_01/report.json'
    cxx_audit = ROOT/'evidence/career_contract_04/cxx_trace_audit_01/report.json'

    inputs = [config, ROOT/'docs/career_execution_contract_04a.md',
              ROOT/'tools04/train_d1_ppo04.py', ROOT/'tools04/check_task04a_static.py',
              ROOT/'tools04/analyze_pilot_traces04.py', ROOT/'tools04/reward_signal_probe04.py',
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

    receipt = {
        'format': 'd1-contract04a-release-v1',
        'status': 'FROZEN_BEFORE_TRAINING',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'contract': 'docs/career_execution_contract_04a.md',
        'single_change': ('one added reward term: standstill_anchor_displacement, '
                          'a Huber penalty on XY distance from the pose where the current '
                          'zero-command segment began'),
        'inherits_unchanged_from_03d': ['command schedule', 'all 03c/03d reward terms and weights',
                                        'actor interface and history', 'observation', 'action scales',
                                        'action clip', 'timing', 'reset', 'termination', 'PPO'],
        'excluded_by_contract': ['teacher policy', 'behaviour cloning', 'warm start',
                                 'true linear velocity actor input', 'PID assistance',
                                 'action post-processing', 'retuned tracking scales',
                                 'wheel action magnitude penalty', 'yaw hold term'],
        'training_plan': {'runs': 1, 'kind': 'pilot', 'arm': 'B', 'seed': 101, 'envs': 8,
                          'transitions': 1048576, 'random_initialization': True,
                          'max_wall_s': 5400, 'min_available_ram_bytes': 3*1024**3,
                          'resume_or_retry_allowed': False, 'checkpoint_selection': 'final_only'},
        'budget': BUDGET,
        'qualification_gates_unchanged': QUALIFICATION_GATES,
        'predictions_preregistered': PREDICTIONS,
        'falsification_rule': ('P1 not met means the hypothesis is rejected: the creep is not '
                              'driven by the missing displacement gradient. No weight retune '
                              'and retry under this contract.'),
        'expected_outcome_statement': ('3/3 qualification is NOT expected from this single '
                                       'change; turn_stop is additionally blocked by its wz MAE, '
                                       'which this term does not address.'),
        'physics_steps_executed_by_this_receipt': 0,
        'evidence_before_training': {
            'static_checks': {'path': str(static.relative_to(ROOT)), 'sha256': sha(static),
                              'status': static_report['status'],
                              'checks': len(static_report['checks']),
                              'native_integration_steps_executed':
                                  static_report['native_integration_steps_executed']},
            'ros_import_preflight': {'path': str(preflight.relative_to(ROOT)),
                                     'sha256': sha(preflight), 'watchdog_validated': False},
            'offline_trace_diagnosis': {'path': str(diagnosis.relative_to(ROOT)),
                                        'sha256': sha(diagnosis)},
            'reward_signal_probe': {'path': str(probe.relative_to(ROOT)), 'sha256': sha(probe)},
            'independent_cxx_trace_audit': {'path': str(cxx_audit.relative_to(ROOT)),
                                            'sha256': sha(cxx_audit)},
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
