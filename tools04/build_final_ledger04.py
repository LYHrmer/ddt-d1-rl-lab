#!/usr/bin/env python3
"""Build the contract 04 final ledger from the receipts, not from prose.

Every number is read back out of the receipt it came from and the totals are
recomputed here, so a mistake in the results document cannot silently become
the ledger. Executes no physics.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT/'evidence/career_contract_04'
# docs/career_execution_results_03.md and evidence/.../final_delivery_01.json
CONTRACT03 = {'transitions': 3145728, 'native': 31891350}
NUMERIC_GATES = {
    'stable_vx': ('stable_vx_mae_m_s', 'stable_vx_mae_m_s'),
    'stable_wz': ('stable_wz_mae_rad_s', 'stable_wz_mae_rad_s_each_sign'),
    'stop_horizontal_speed': ('stop_last_second_horizontal_speed_m_s',
                              'stop_last_second_horizontal_speed_m_s'),
    'stop_abs_wz': ('stop_last_second_abs_wz_rad_s', 'stop_last_second_abs_wz_rad_s'),
    'stop_displacement': ('stop_displacement_m', 'stop_displacement_m'),
    'peak_roll_pitch': ('peak_abs_roll_or_pitch_rad', 'peak_abs_roll_or_pitch_rad'),
}
BOOLEAN_GATES = ('completed_12s', 'no_fall_or_illegal_support')


def gate_summary(flags):
    """Count the eight receipt fields without treating null or 1 as true.

    What this layer can and cannot see. It knows nothing about the task, so it
    cannot tell which numeric gate is legitimately null for it; a stable_wz of
    null turned into true still sums to 7/8 here. That check belongs to
    task_gate_summary, which knows forward and reverse have no stable_wz and
    turn has no stable_vx. This layer rejects only what is wrong for every task:
    a wrong key set, a non-boolean value, and a null on either of the two
    completion gates. The last rule also guarantees at least two applicable gates,
    so an all-null record can never be summarised as 0/0; no separate check for
    that is needed, and an earlier one was removed as unreachable (exhaustively
    verified over the 2,916 shapes that pass the completion-gate rule).
    """
    if not isinstance(flags, dict) or set(flags) != set(NUMERIC_GATES) | set(BOOLEAN_GATES):
        raise ValueError('threshold_flags must contain exactly the eight qualification gates')
    if any(value is not None and type(value) is not bool for value in flags.values()):
        raise ValueError('threshold_flags values must be true, false or null')
    if any(flags[gate] is None for gate in BOOLEAN_GATES):
        raise ValueError('completion gates apply to every task and cannot be null')
    return {'gates_passed': sum(value is True for value in flags.values()),
            'gates_applicable': sum(value is not None for value in flags.values()),
            'gates_total': len(flags),
            'not_applicable': [key for key, value in flags.items() if value is None],
            'failed_gates': [key for key, value in flags.items() if value is False]}


def task_gate_summary(metrics, qualification_gates):
    """Explain one task using raw metrics and the release's frozen thresholds.

    This describes gates only: it cannot award overall control qualification.
    Reject incomplete or inconsistent evidence instead of inventing a result.
    """
    if not isinstance(metrics, dict) or not isinstance(qualification_gates, dict):
        raise ValueError('metrics and qualification_gates must be objects')
    summary = gate_summary(metrics.get('threshold_flags'))
    flags = metrics['threshold_flags']
    sequence = metrics.get('sequence')
    if sequence not in ('forward_stop', 'reverse_stop', 'turn_stop'):
        raise ValueError('unknown or missing qualification sequence')
    expected_na = 'stable_vx' if sequence == 'turn_stop' else 'stable_wz'
    if summary['not_applicable'] != [expected_na]:
        raise ValueError(f'{sequence} requires exactly one inapplicable gate: {expected_na}')

    details = []
    for gate, (metric, threshold_key) in NUMERIC_GATES.items():
        threshold = qualification_gates.get(threshold_key)
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError(f'missing or invalid frozen threshold: {threshold_key}')
        if metric not in metrics:
            raise ValueError(f'missing metric: {metric}')
        value = metrics[metric]
        if gate == expected_na:
            if value is not None:
                raise ValueError(f'inapplicable metric must be null: {metric}')
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid numeric metric: {metric}')
        if flags[gate] is not (value <= threshold):
            raise ValueError(f'flag disagrees with raw metric and frozen threshold: {gate}')
        if flags[gate] is False:
            details.append({'gate': gate, 'metric': metric, 'value': value,
                            'operator': '<=', 'threshold_key': threshold_key,
                            'threshold': threshold,
                            'explanation': f'{metric} {value:.8g} > {threshold:.8g}'})

    transitions, reason, failure = (metrics.get('policy_transitions'),
                                     metrics.get('done_reason'), metrics.get('failure'))
    if (type(transitions) is not int or not 0 <= transitions <= 600
            or not isinstance(reason, str) or not reason or type(failure) is not bool):
        raise ValueError('missing or invalid completion/failure metrics')
    boolean_values = {'completed_12s': transitions == 600 and reason == 'HORIZON_12S',
                      'no_fall_or_illegal_support': not failure}
    for gate, value in boolean_values.items():
        if qualification_gates.get(gate) is not True:
            raise ValueError(f'missing or invalid frozen boolean requirement: {gate}')
        if flags[gate] is not value:
            raise ValueError(f'flag disagrees with completion/failure metrics: {gate}')
        if value is False:
            details.append({'gate': gate, 'value': value, 'operator': '==',
                            'threshold_key': gate, 'threshold': True,
                            'explanation': f'{gate} false != true'})
    summary['failed_gate_details'] = details
    summary['blocking_gate'] = '; '.join(item['explanation'] for item in details) or None
    return summary


def qualification_verdict(tasks_passed, tasks_total, early_failures, tasks_required):
    """Derive one version's overall verdict from counts instead of writing it down.

    PASS needs all three: every required task passed, every required task was
    actually evaluated, and nothing ended early. A partial result such as 2/3, or
    3/3 over a truncated evaluation, is FAIL. This function can only report a
    verdict; it never upgrades a task-level improvement into a qualification.
    """
    values = {'tasks_passed': tasks_passed, 'tasks_total': tasks_total,
              'early_failures': early_failures, 'tasks_required': tasks_required}
    for name, value in values.items():
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer, got {value!r}')
    if tasks_required == 0:
        raise ValueError('tasks_required must be positive')
    if tasks_passed > tasks_total:
        raise ValueError('tasks_passed cannot exceed tasks_total')
    passed = (tasks_passed == tasks_required and tasks_total == tasks_required
              and early_failures == 0)
    return 'PASS' if passed else 'FAIL'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(relative):
    path = EV/relative
    return json.loads(path.read_text()), str(path.relative_to(ROOT)), sha(path)


def training_entry(version, run_dir):
    result, path, digest = load(f'{run_dir}/result.json')
    protocol, _, _ = load(f'{run_dir}/protocol.json')
    return {'version': version, 'status': result['status'],
            'environment_class': protocol['environment_class'],
            'config_version': protocol['config']['version'],
            'transitions': result['trained_transitions'],
            'native': result['actual_counts']['native_steps'],
            'inputs_unchanged': result['inputs_unchanged'],
            'elapsed_training_s': result['elapsed_training_s'],
            'min_available_ram_bytes': result['min_available_ram_bytes'],
            'resets': result['actual_counts']['resets'],
            'native_equals_ten_times_transitions':
                result['actual_counts']['native_steps'] == result['trained_transitions']*10,
            'result': path, 'result_sha256': digest}


def eval_entry(version, eval_dir, verdict_dir):
    summary, summary_path, summary_sha = load(f'{eval_dir}/summary.json')
    verdict, verdict_path, verdict_sha = load(f'{verdict_dir}/report.json')
    return {'version': version, 'status': summary['status'],
            'native': summary['actual_native_steps'],
            'episodes_completed': summary['episodes_completed'],
            'early_failures': summary['early_failures'],
            'pass_count': summary['zero_delay_nominal_pass_count'],
            'tasks': 3,
            'hypothesis_verdict': verdict['hypothesis_verdict']['hypothesis'],
            'predictions': {p['id']: p['verdict'] for p in verdict['predictions']},
            'summary': summary_path, 'summary_sha256': summary_sha,
            'verdict': verdict_path, 'verdict_sha256': verdict_sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')

    training = [training_entry('04a', 'pilot04a_02'), training_entry('04b', 'pilot04b_01')]
    evaluation = [eval_entry('04a', 'pilot04a_python_eval_01', 'pilot04a_prediction_verdict_01'),
                  eval_entry('04b', 'pilot04b_python_eval_01', 'pilot04b_prediction_verdict_01')]

    cpp = []
    for version, score in (('04a', 'pilot04a_deployment_plan_01/score03_turn_stop'),
                           ('04b', 'pilot04b_deployment_plan_01/score02_reverse_stop')):
        report, path, digest = load(f'{score}/report.json')
        gates = gate_summary(report['metrics']['threshold_flags'])
        cpp.append({'version': version, 'task': report['metrics']['sequence'],
                    'status': report['status'],
                    'native': report['actual_native_steps'],
                    'structural_checks_passed': sum(report['checks'].values()),
                    'structural_checks_total': len(report['checks']),
                    'all_structural_checks_passed': all(report['checks'].values()),
                    'gates_passed': gates['gates_passed'],
                    'gates_applicable': gates['gates_applicable'],
                    'inference_p99_ms': report['inference_latency']['p99'],
                    'trials_used': 1, 'trials_allowed': 3,
                    'native_cap': 18000, 'native_unused': 18000-report['actual_native_steps'],
                    'score': path, 'score_sha256': digest})

    golden = []
    for version, directory in (('04a', 'pilot04a_cxx_golden_01'), ('04b', 'pilot04b_cxx_golden_01')):
        report, path, digest = load(f'{directory}/report.json')
        golden.append({'version': version, 'status': report['status'],
                       'onnx_inference_calls': report['onnx_inference_calls'],
                       'max_absolute_error': report['max_absolute_error'],
                       'absolute_tolerance': report['absolute_tolerance'],
                       'onnxruntime_version': report['onnxruntime_version'],
                       'physics_steps': report['physics_steps'],
                       'report': path, 'report_sha256': digest})

    static = []
    for version, directory in (('04a', 'task04a_static_02'), ('04b', 'task04b_static_03')):
        report, path, digest = load(f'{directory}/report.json')
        static.append({'version': version, 'status': report['status'],
                       'checks': len(report['checks']),
                       'native_integration_steps_executed':
                           report['native_integration_steps_executed'],
                       'report': path, 'report_sha256': digest})

    open_loop, open_path, open_sha = load('passive_braking_probe_01/report.json')
    zero_wheel, zero_path, zero_sha = load('zero_wheel_braking_probe_02/report.json')
    smoke, smoke_path, smoke_sha = load('zero_wheel_braking_smoke_01/report.json')
    preflight, pre_path, pre_sha = load('ros_import_preflight_01/report.json')
    launch_fail, lf_path, lf_sha = load('pilot04a_launch_failure_01.json')

    engineering = {
        '04a_pool_cap': 2000, '04a_pool_used': 0,
        '04b_pool_cap': 2000,
        '04b_pool_used': 2000,
        '04b_pool_breakdown': [
            {'item': 'open loop fixed-action probe', 'native':
             open_loop['actual_counts']['native_steps'], 'receipt': open_path},
            {'item': 'policy-legs zero-wheel probe, first attempt',
             'native_charged': 1650, 'native_actual': 'UNKNOWN',
             'receipt': None,
             'loss_reason': ('the physics loop and env.close() completed, then the receipt '
                             'construction raised ValueError in Path.relative_to before the '
                             'output directory was created, so the counters were lost with '
                             'the process. Charged at the planned cap; this is a tool defect, '
                             'not a measurement, and supports no physical conclusion.'),
             'fix': 'tools04/probe_zero_wheel_braking04.py now falls back to an absolute path'}],
        'addendum_a_cap': 1670,
        'addendum_a_used': (smoke['actual_counts']['native_steps']
                            + zero_wheel['actual_counts']['native_steps']),
        'addendum_a_breakdown': [
            {'item': 'receipt-write smoke test', 'native': smoke['actual_counts']['native_steps'],
             'receipt': smoke_path},
            {'item': 'zero-wheel braking measurement',
             'native': zero_wheel['actual_counts']['native_steps'], 'receipt': zero_path}],
    }

    native_total = (sum(e['native'] for e in training)
                    + sum(e['native'] for e in evaluation)
                    + sum(e['native'] for e in cpp)
                    + engineering['04a_pool_used'] + engineering['04b_pool_used']
                    + engineering['addendum_a_used'])
    transitions_total = sum(e['transitions'] for e in training)
    best = next(e for e in cpp if e['version'] == '04b' and e['task'] == 'reverse_stop')
    best_report = json.loads((ROOT/best['score']).read_text())
    release, release_path, release_sha = load('pilot04b_release_01.json')
    best_gates = task_gate_summary(best_report['metrics'], release['qualification_gates_unchanged'])
    tasks_required = release['qualification_gates_unchanged']['tasks_required_to_pass']
    versions_passing = [e['version'] for e in evaluation
                        if qualification_verdict(e['pass_count'], e['tasks'],
                                                 e['early_failures'], tasks_required) == 'PASS']

    ledger = {
        'format': 'd1-contract04-final-delivery-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'BOUNDED_WORK_CLOSED_OVERALL_QUALIFICATION_FAILED',
        'simulation_only': True,
        'real_robot_validation': False,
        'contracts': ['docs/career_execution_contract_04a.md',
                      'docs/career_execution_contract_04b.md',
                      'docs/career_execution_contract_04b_addendum_a.md'],
        'relation_between_versions': ('04a and 04b are siblings: each adds exactly one reward '
                                      'term to the same frozen 03d base, so comparing them '
                                      'isolates the choice of measured quantity. 04b does not '
                                      'inherit the 04a term.'),
        'training': {'pilots_consumed': len(training), 'maximum_pilots_per_contract': 1,
                     'runs': training, 'transitions_total': transitions_total,
                     'native_total': sum(e['native'] for e in training),
                     'launch_failures_with_zero_integration': [
                         {'version': '04a', 'status': launch_fail['status'],
                          'native': launch_fail['native_steps_executed'],
                          'training_budget_charged': launch_fail['training_budget_charged'],
                          'receipt': lf_path, 'receipt_sha256': lf_sha}]},
        'zero_integration_static_checks': static,
        'onnx_cxx_golden': golden,
        'python_qualification_evaluation': {
            'runs': evaluation, 'native_total': sum(e['native'] for e in evaluation)},
        'independent_cpp_deployment': {
            'runs': cpp, 'native_total': sum(e['native'] for e in cpp)},
        'engineering_and_probes': engineering,
        'ros_environment': {
            'import_preflight_status': preflight['status'],
            'checks': len(preflight['checks']),
            'physics_steps_executed': preflight['physics_steps_executed'],
            'ros_nodes_created': preflight['ros_nodes_created'],
            'watchdog_validated': preflight['watchdog_validated'],
            'receipt': pre_path, 'receipt_sha256': pre_sha,
            'note': 'import success is not watchdog validation'},
        'interface_measurement': {
            'question': 'does a zero wheel action brake inside the 0.05 m displacement gate',
            'answer': zero_wheel['g6_verdict']['zero_wheel_action_alone_meets_gate'],
            'release_speed_m_s': zero_wheel['measured_coast']['speed_at_release_m_s'],
            'distance_until_under_0p02_m':
                zero_wheel['measured_coast']['distance_until_under_0p02_m'],
            'arithmetic_predicted_distance_m':
                zero_wheel['g6_verdict']['predicted_distance_from_release_speed_m'],
            'gate_m': 0.05,
            'limitations': [
                'release speed is 45 percent of the 0.1965 m/s the gate implies; the '
                'extrapolated 0.016 m at gate speed is an extrapolation, not a measurement',
                'after braking the speed rebounded to 0.0296 m/s because forcing the wheels '
                'to zero while the legs follow a policy that expected nonzero wheel commands '
                'is an inconsistent action pair; the fitted 0.322 s is contaminated by that '
                'segment and is not a damping time constant',
                'an earlier offline estimate of 0.84 to 1.63 s was wrong for the same reason '
                'and is superseded'],
            'receipts': [open_path, zero_path]},
        'control_qualification': {
            'versions': {entry['version']: {'tasks_passed': entry['pass_count'],
                                            'tasks_total': entry['tasks'],
                                            'early_failures': entry['early_failures'],
                                            'tasks_required': tasks_required,
                                            'overall': qualification_verdict(
                                                entry['pass_count'], entry['tasks'],
                                                entry['early_failures'], tasks_required)}
                         for entry in evaluation},
            'best_single_task': {'version': '04b', 'task': 'reverse_stop',
                                 **best_gates,
                                 'score': best['score'],
                                 'score_sha256': best['score_sha256'],
                                 'qualification_release': release_path,
                                 'qualification_release_sha256': release_sha,
                                 'note': ('single-task result verified independently in C++; '
                                          'null flags are not applicable, not passes; '
                                          'not an overall qualification pass')},
            'versions_passing': versions_passing,
            'overall': 'PASS' if versions_passing else 'FAIL'},
        'formal_training_runs': 0,
        'formal_evaluation_runs': 0,
        'terrain_policy_trials': 0,
        'contract04_transitions_total': transitions_total,
        'contract04_native_total_all_pools': native_total,
        'historical_contract03': CONTRACT03,
        'project_cumulative': {
            'transitions': CONTRACT03['transitions']+transitions_total,
            'native': CONTRACT03['native']+native_total},
        'unused_budget_not_a_future_licence': {
            '04a_cpp_native_unused': cpp[0]['native_unused'],
            '04b_cpp_native_unused': cpp[1]['native_unused'],
            '04a_engineering_native_unused': 2000,
            'note': ('unused allowance expires with its contract and is not available to a '
                     'later experiment')},
        'unfinished': [
            'overall flat low speed qualification, three tasks',
            'turn gain asymmetry, unmet across 03c/03d/04a/04b',
            'drive phase vx tracking and turn stability, the two 04b regressions',
            'watchdog physical stop after real ROS command timeout',
            'own policy terrain crossing',
            'real 15mm step crossing',
            'continuous lateral motion',
            '1.5 m per s',
            'formal three arms three seeds',
            'physical self recovery, a reset is not self recovery',
            'real robot validation'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output),
                      'contract04_transitions': transitions_total,
                      'contract04_native': native_total,
                      'project_cumulative_native': ledger['project_cumulative']['native'],
                      'qualification': 'FAIL'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
