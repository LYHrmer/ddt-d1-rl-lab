"""Strict stdlib79 episode comparison, independent of model qualification.

Rows wrap the complete frozen evaluate_d1_ppo.run_episode item in ``metrics``.
Outer task/reset identities must match it. This module admits only the full
registered evaluation grid; it never fills, drops or ranks missing episodes.
Mean reductions use math.fsum/n, not a claimed bitwise NumPy reproduction.
"""
from __future__ import annotations

import math

MODEL_SEEDS = (64001, 64002, 64003)
RESET_SEEDS = (79401, 79402, 79403)
ARMS = ('frozen', 'refresh')
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
SCENARIOS = tuple((task, seed) for task in TASKS for seed in RESET_SEEDS)
COUNTS = {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000}
J_COMPONENTS = ('full_horizon_vx_mae_over_0_2', 'full_horizon_wz_mae_over_0_2',
                'stop_8_to_12_horizontal_norm_mean_over_0_2', 'fixed_early_failure_penalty')
THRESHOLDS = {'stable_vx_mae_m_s': .06, 'stable_wz_mae_rad_s': .08,
              'stop_last_second_horizontal_speed_m_s': .02, 'stop_last_second_abs_wz_rad_s': .03,
              'stop_displacement_m': .05, 'peak_abs_roll_or_pitch_rad': .20}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float), name + ' must be a real scalar')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(name + ' is outside finite binary64') from exc
    _require(math.isfinite(value) and value >= 0, name + ' must be finite and nonnegative')
    return value


def _finite_tree(value):
    if isinstance(value, float):
        _require(math.isfinite(value), 'nonfinite original metric or metadata')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)


def _metrics(row):
    item = row['metrics']
    required = {'sequence', 'reset_seed', 'delay_ticks', 'delay_ms', 'policy_transitions', 'native_steps',
                'done_reason', 'status', 'failure', 'J', 'J_components', 'actual_counts', 'threshold_flags',
                'reference_thresholds_met', 'nominal_zero_delay_pass', *THRESHOLDS}
    _require(isinstance(item, dict) and required <= item.keys(), 'complete original run_episode metrics required')
    _finite_tree(item)
    _require(item['sequence'] == row['task'] and type(item['reset_seed']) is int and item['reset_seed'] == row['seed'],
             'outer task/reset differs from original metrics')
    for key, expected in (('delay_ticks', 0), ('delay_ms', 0), ('policy_transitions', 600), ('native_steps', 6000)):
        _require(type(item[key]) is int and item[key] == expected, 'incomplete horizon, count or delay: ' + key)
    _require(item['done_reason'] == 'HORIZON_12S', 'terminated/truncated episode is not a complete candidate')
    _require(type(item['failure']) is bool and item['status'] == ('FAIL' if item['failure'] else 'COMPLETE'),
             'original status/failure mismatch')
    counts = item['actual_counts']
    _require(isinstance(counts, dict) and all(type(counts.get(k)) is int and counts[k] == v for k, v in COUNTS.items()),
             'actual policy/native/PD counters must be600/6000/3000')
    components = item['J_components']
    _require(isinstance(components, dict) and set(components) == set(J_COMPONENTS), 'original four J components required')
    values = [_number(components[key], key) for key in J_COMPONENTS]
    _require(values[-1] == (100. if item['failure'] else 0.), 'original failure penalty differs')
    _require(_number(item['J'], 'J') == sum(values), 'J differs from original ordered component sum')
    turn = row['task'] == 'turn_stop'
    stable_key, unused_key = ('stable_wz_mae_rad_s', 'stable_vx_mae_m_s') if turn else ('stable_vx_mae_m_s', 'stable_wz_mae_rad_s')
    _require(item[unused_key] is None, 'task-inapplicable stable metric must remain None')
    measured = {key: _number(item[key], key) for key in THRESHOLDS if key != unused_key}
    expected_flags = {'completed_12s': True, 'no_fall_or_illegal_support': not item['failure'],
        'stable_vx': None if turn else measured[stable_key] <= THRESHOLDS[stable_key],
        'stable_wz': measured[stable_key] <= THRESHOLDS[stable_key] if turn else None,
        'stop_horizontal_speed': measured['stop_last_second_horizontal_speed_m_s'] <= .02,
        'stop_abs_wz': measured['stop_last_second_abs_wz_rad_s'] <= .03,
        'stop_displacement': measured['stop_displacement_m'] <= .05,
        'peak_roll_pitch': measured['peak_abs_roll_or_pitch_rad'] <= .20}
    flags = item['threshold_flags']
    _require(isinstance(flags, dict) and set(flags) == set(expected_flags) and
             all(flags[k] is v for k, v in expected_flags.items()), 'original threshold flags missing, unknown or inconsistent')
    passed = all(value for value in expected_flags.values() if value is not None)
    _require(item['reference_thresholds_met'] is passed and item['nominal_zero_delay_pass'] is passed,
             'original reference threshold decision differs')
    return item


def _index(rows, candidate):
    _require(isinstance(rows, (list, tuple)) and len(rows) == (54 if candidate else 9), 'exact registered episode count required')
    indexed = {}
    for row in rows:
        required = {'task', 'seed', 'metrics'} | ({'model_seed', 'arm'} if candidate else set())
        _require(isinstance(row, dict) and required <= row.keys(), 'episode wrapper fields missing')
        _require(row['task'] in TASKS and type(row['seed']) is int and row['seed'] in RESET_SEEDS, 'unexpected evaluation task/reset')
        if 'status' in row:
            _require(row['status'] == 'COMPLETE', 'wrapper marks partial episode')
        if candidate:
            _require(type(row['model_seed']) is int and row['model_seed'] in MODEL_SEEDS and row['arm'] in ARMS,
                     'unexpected model seed/arm')
        else:
            _require('model_seed' not in row and 'arm' not in row, 'shared original baseline is not a model-specific policy')
        key = (row['model_seed'], row['arm'], row['task'], row['seed']) if candidate else (row['task'], row['seed'])
        _require(key not in indexed, 'duplicate registered episode')
        indexed[key] = _metrics(row)
    expected = {(m, a, t, s) for m in MODEL_SEEDS for a in ARMS for t, s in SCENARIOS} if candidate else set(SCENARIOS)
    _require(set(indexed) == expected, 'registered episode grid incomplete')
    return indexed


def _mean(values):
    try:
        value = math.fsum(values) / len(values)
    except OverflowError as exc:
        raise ValueError('episode mean exceeds finite arithmetic') from exc
    _require(math.isfinite(value), 'nonfinite episode mean')
    return value


def _ratio(top, bottom, minimum=0.):
    if bottom <= minimum:
        return None
    value = top / bottom
    return value if math.isfinite(value) else None


def _key(task, seed):
    return f'{task}/seed{seed}'


def compare(baseline_rows, candidate_rows):
    """Compare exactly9 original baselines and54 candidate episodes; no I/O.

    Missing/partial/nonfinite/inconsistent inputs raise ValueError. A complete
    600-step original-threshold failure remains a valid failed observation.
    Global model/deployment acceptance is never granted by this comparison.
    """
    baseline, candidates = _index(baseline_rows, False), _index(candidate_rows, True)
    observations, paired = [], []
    for model_seed in MODEL_SEEDS:
        rows = []
        for task, seed in SCENARIOS:
            frozen, refresh = [candidates[(model_seed, arm, task, seed)] for arm in ARMS]
            row = {'model_seed': model_seed, 'task': task, 'seed': seed,
                   'baseline_key': _key(task, seed), 'frozen_J': frozen['J'], 'refresh_J': refresh['J'],
                   'J_ratio_refresh_over_frozen': _ratio(refresh['J'], frozen['J']),
                   'J_pair_le1p05': refresh['J'] <= 1.05 * frozen['J'],
                   'frozen_reference_thresholds_met': frozen['reference_thresholds_met'],
                   'refresh_reference_thresholds_met': refresh['reference_thresholds_met'],
                   'new_original_task_failure': frozen['reference_thresholds_met'] and not refresh['reference_thresholds_met']}
            rows.append(row)
        fj, rj = [_mean([row[key] for row in rows]) for key in ('frozen_J', 'refresh_J')]
        paired.append({'model_seed': model_seed, 'paired_scenarios': 9, 'frozen_mean_J': fj, 'refresh_mean_J': rj,
                       'J_ratio_refresh_over_frozen': _ratio(rj, fj), 'strict_mean_J_improvement': rj < fj})
        observations.extend(rows)
    overall_frozen, overall_refresh = [_mean([p[key] for p in paired]) for key in ('frozen_mean_J', 'refresh_mean_J')]
    overall_ratio = _ratio(overall_refresh, overall_frozen)
    feedback_gates = {'all_three_seed_means_strictly_improve': all(p['strict_mean_J_improvement'] for p in paired),
        'equal_seed_overall_ratio_le0p99': overall_ratio is not None and overall_ratio <= .99,
        'all27_individual_J_pairs_le1p05': all(p['J_pair_le1p05'] for p in observations),
        'no_new_original_task_failure': not any(p['new_original_task_failure'] for p in observations)}
    controls = []
    for model_seed in MODEL_SEEDS:
        for arm in ARMS:
            pairs, turns = [], []
            for task, seed in SCENARIOS:
                base, candidate = baseline[(task, seed)], candidates[(model_seed, arm, task, seed)]
                pairs.append({'task': task, 'seed': seed, 'baseline_key': _key(task, seed),
                    'baseline_J': base['J'], 'candidate_J': candidate['J'],
                    'J_ratio': _ratio(candidate['J'], base['J'], 1e-12), 'J_pair_le1p05': candidate['J'] <= 1.05 * base['J'],
                    'baseline_reference_thresholds_met': base['reference_thresholds_met'],
                    'candidate_reference_thresholds_met': candidate['reference_thresholds_met']})
                if task == 'turn_stop':
                    turns.append({'seed': seed, 'baseline_wz_mae': base['stable_wz_mae_rad_s'],
                        'candidate_wz_mae': candidate['stable_wz_mae_rad_s'],
                        'strict_improvement': candidate['stable_wz_mae_rad_s'] < base['stable_wz_mae_rad_s']})
            j_ratio = _ratio(_mean([p['candidate_J'] for p in pairs]), _mean([p['baseline_J'] for p in pairs]), 1e-12)
            yaw_ratio = _ratio(_mean([p['candidate_wz_mae'] for p in turns]), _mean([p['baseline_wz_mae'] for p in turns]), 1e-12)
            gates = {'all_baseline_and_candidate_original_thresholds': all(p[k] for p in pairs for k in
                        ('baseline_reference_thresholds_met', 'candidate_reference_thresholds_met')),
                     'all_three_turn_resets_strictly_improve': all(p['strict_improvement'] for p in turns),
                     'turn_mean_ratio_le0p95': yaw_ratio is not None and yaw_ratio <= .95,
                     'J_mean_ratio_le1': j_ratio is not None and j_ratio <= 1.,
                     'all_nine_J_pairs_le1p05': all(p['J_pair_le1p05'] for p in pairs)}
            controls.append({'model_seed': model_seed, 'arm': arm, 'passed': all(gates.values()), 'gates': gates,
                'reasons': [key for key, passed in gates.items() if not passed], 'J_ratio': j_ratio, 'yaw_mae_ratio': yaw_ratio,
                'pairs': pairs, 'turn_pairs': turns, 'baseline_keys': [_key(t, s) for t, s in SCENARIOS],
                'ratio_denominator_must_exceed': 1e-12})
    counts = {label: {key: sum(m['actual_counts'][key] for m in rows.values()) for key in COUNTS}
              for label, rows in (('baseline', baseline), ('candidate', candidates))}
    counts['total'] = {key: counts['baseline'][key] + counts['candidate'][key] for key in COUNTS}
    return {'schema': 'd1-feedback79-comparison-v1', 'status': 'COMPLETE',
        'feedback': {'passed': all(feedback_gates.values()), 'gates': feedback_gates,
            'reasons': [key for key, passed in feedback_gates.items() if not passed], 'by_model_seed': paired,
            'overall_frozen_mean_J': overall_frozen, 'overall_refresh_mean_J': overall_refresh,
            'overall_J_ratio': overall_ratio, 'overall_ratio_definition': 'ratio of equal-weight three learner means'},
        'paired_observations': observations, 'control_gates': controls, 'evaluation_counts': counts,
        'sampling_unit': {'paired_model_seed_units': 3, 'paired_observations': 27, 'scenarios_per_policy': 9,
            'candidate_policies': 6, 'candidate_episodes': 54, 'unique_baseline_episodes': 9,
            'baseline_episode_references_in_control_comparisons': 54, 'baseline_policy_steps_counted_once': 5400,
            '27_or54_observations_are_not_independent_model_seeds': True,
            'model_seeds_are_world_model_learners_not_from_scratch_PPO_seeds': True},
        'reduction': 'binary64 math.fsum/n; each complete episode and model seed has its registered equal weight',
        'undefined_ratio_is_not_a_pass': True, 'complete_model_eligibility': 'NOT_REQUALIFIED',
        'global_accepted': False, 'candidate_accepted': False, 'selected_seed': None,
        'scope': 'complete evaluation metrics only; runner must separately verify sources, updates and experiment completion'}
