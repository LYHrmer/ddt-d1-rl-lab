"""Registered81 decisions from complete saved metrics, using only stdlib.

The four-scale observations never choose a candidate. The fixed candidate is
scale_index=0/sign=+1. Source, parameter and numerical admission remain the
runner's responsibility; no result here requalifies a frozen world model.
"""
from __future__ import annotations

from tools79.compare79 import (
    COUNTS,
    MODEL_SEEDS,
    TASKS,
    _finite_tree,
    _mean,
    _metrics,
    _number,
    _ratio,
    _require,
)
from tools80.compare80 import TICK_ROLES

ARMS = ('C6', 'Q6')
SCALES = (.009, .0045, .00225, .001125)
RESET_SEEDS = (81401, 81402, 81403)
SCENARIOS = tuple((task, seed) for task in TASKS for seed in RESET_SEEDS)
SCREEN_SEEDS = (81207, 81208, 81209)
SCREEN_TICKS = (110, 175, 260, 325, 410)
STEADY_TICKS = (175, 325)
SCREEN_COUNTS = {'steps': 32, 'native_steps': 320, 'pd_updates': 160}


def branch_specs(tick):
    """Return (model_seed, arm, repeat, scale_index, sign): 50 or8 specs."""
    _require(type(tick) is int and tick in SCREEN_TICKS, 'unregistered screen tick')
    probes = ((i, sign) for i in range(4) for sign in (1, -1)) if tick in STEADY_TICKS else ((0, 1),)
    probes = tuple(probes)
    return ((None, 'anchor', 0, None, 0), (None, 'anchor', 1, None, 0)) + tuple(
        (model, arm, 0, index, sign) for model in MODEL_SEEDS for arm in ARMS for index, sign in probes)


def _screen_index(rows):
    _require(isinstance(rows, (list, tuple)) and len(rows) == 372, 'exact372 registered branches required')
    required = {'model_seed', 'arm', 'repeat', 'scale_index', 'sign', 'seed', 'tick', 'status',
                'steps', 'native_steps', 'pd_updates', 'fallen', 'illegal_body_contact',
                'cost', 'yaw_mae', 'trace_identity_sha256'}
    indexed = {}
    for row in rows:
        _require(isinstance(row, dict) and required <= row.keys(), 'short branch fields missing')
        _finite_tree(row)
        _require(type(row['seed']) is int and row['seed'] in SCREEN_SEEDS and
                 type(row['tick']) is int and row['tick'] in SCREEN_TICKS, 'unexpected short site')
        _require(type(row['repeat']) is int and type(row['sign']) is int, 'repeat/sign must be integers')
        if row['arm'] == 'anchor':
            _require(row['model_seed'] is None and row['repeat'] in (0, 1) and
                     row['scale_index'] is None and row['sign'] == 0, 'unexpected anchor identity')
        else:
            _require(type(row['model_seed']) is int and row['model_seed'] in MODEL_SEEDS and
                     row['arm'] in ARMS and row['repeat'] == 0 and type(row['scale_index']) is int and
                     row['scale_index'] in range(4) and row['sign'] in (-1, 1), 'unexpected probe identity')
        spec = tuple(row[k] for k in ('model_seed', 'arm', 'repeat', 'scale_index', 'sign'))
        _require(spec in branch_specs(row['tick']), 'probe outside registered tick/scale/sign grid')
        _require(row['status'] == 'COMPLETE' and
                 all(type(row[k]) is int and row[k] == v for k, v in SCREEN_COUNTS.items()),
                 'short branch must be COMPLETE with32/320/160 counts')
        if 'actual_counts' in row:
            expected = {'policy_transitions': 32, 'native_steps': 320, 'pd_updates': 160}
            _require(isinstance(row['actual_counts'], dict) and all(
                type(row['actual_counts'].get(k)) is int and row['actual_counts'][k] == v
                for k, v in expected.items()), 'short actual counters differ')
        for key in ('fallen', 'illegal_body_contact', 'terminated', 'truncated'):
            if key in row:
                _require(type(row[key]) is bool, 'explicit boolean safety/termination flags required')
        _number(row['cost'], 'short cost')
        _number(row['yaw_mae'], 'short yaw MAE')
        sha = row['trace_identity_sha256']
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
                 'lowercase64-digit trace SHA256 required')
        key = (row['seed'], row['tick'], *spec)
        _require(key not in indexed, 'duplicate short branch identity')
        indexed[key] = row
    expected = {(seed, tick, *spec) for seed in SCREEN_SEEDS for tick in SCREEN_TICKS for spec in branch_specs(tick)}
    _require(set(indexed) == expected, 'registered short grid incomplete')
    return indexed


def _q(indexed, seed, model, arm, scale_index=0, sign=1):
    if arm == 'anchor':
        model, scale_index, sign = None, None, 0
    values = [indexed[(seed, tick, model, arm, 0, scale_index, sign)]['yaw_mae'] for tick in STEADY_TICKS]
    return {'positive_turn_fragment_mae': values[0], 'negative_turn_fragment_mae': values[1],
            'max_fragment_mae': max(values)}


def _raw_order(qplus, qzero, qminus):
    return {'Q_plus': qplus, 'Q_zero': qzero, 'Q_minus': qminus,
            'plus_minus_zero_change': qplus - qzero, 'zero_minus_minus_change': qzero - qminus,
            'raw_strict_descent': qplus < qzero < qminus,
            'raw_strict_ascent': qminus < qzero < qplus,
            'plus_equals_zero': qplus == qzero, 'minus_equals_zero': qminus == qzero}


def _diagnostics(indexed):
    observations, aggregated = [], []
    for model in MODEL_SEEDS:
        for arm in ARMS:
            for scale_index, scale in enumerate(SCALES):
                rows = []
                for seed in SCREEN_SEEDS:
                    qs = [_q(indexed, seed, model, arm, scale_index, sign)['max_fragment_mae'] for sign in (1, -1)]
                    zero = _q(indexed, seed, None, 'anchor')['max_fragment_mae']
                    row = {'model_seed': model, 'arm': arm, 'seed': seed, 'scale_index': scale_index,
                           'nominal_scale': scale, **_raw_order(qs[0], zero, qs[1])}
                    rows.append(row)
                    observations.append(row)
                means = [_mean([r[k] for r in rows]) for k in ('Q_plus', 'Q_zero', 'Q_minus')]
                aggregated.append({'model_seed': model, 'arm': arm, 'scale_index': scale_index,
                    'nominal_scale': scale, **_raw_order(*means),
                    'all_three_resets_raw_strict_descent': all(r['raw_strict_descent'] for r in rows)})
    return {'by_model_arm_reset_scale': observations, 'by_model_arm_scale': aggregated,
        'all_Q6_model_reset_scale_raw_strict_descent': all(r['raw_strict_descent'] for r in observations if r['arm'] == 'Q6'),
        'all_C6_model_reset_scale_raw_strict_descent': all(r['raw_strict_descent'] for r in observations if r['arm'] == 'C6'),
        'used_for_candidate_selection': False, 'action_resolution_checked_by_this_module': False,
        'numerical_derivative_qualification': None, 'original_qualification_changed': False,
        'scope': 'raw finite-scale order only; no derivative floor, AD/FD gate or confidence interval'}


def diagnostics(rows):
    """Report all72 reset-wise and24 equal-reset means; never choose a scale."""
    return _diagnostics(_screen_index(rows))


def screen_compare(rows):
    """Validate all372 branches; screen only the six fixed positive endpoints.

Missing/partial records raise ValueError. Complete safety/replay failures are
reported failures. Q is mean_reset(max(MAE175,MAE325)), with three resets.
"""
    indexed = _screen_index(rows)
    points, replay_failures = [], []
    for seed in SCREEN_SEEDS:
        for tick in SCREEN_TICKS:
            anchors = [indexed[(seed, tick, None, 'anchor', r, None, 0)] for r in (0, 1)]
            keys = ('trace_identity_sha256', 'cost', 'yaw_mae', 'fallen', 'illegal_body_contact', 'terminated', 'truncated')
            replay = all(anchors[0].get(k) == anchors[1].get(k) for k in keys)
            if not replay:
                replay_failures.append({'seed': seed, 'tick': tick})
            points.append({'seed': seed, 'tick': tick, 'fragment': TICK_ROLES[tick], 'anchor_replay_exact': replay,
                'branches': [dict(indexed[(seed, tick, *spec)]) for spec in branch_specs(tick)]})
    paired = []
    for model in MODEL_SEEDS:
        resets = [{'seed': seed, **{arm: _q(indexed, seed, model, arm) for arm in ('anchor', *ARMS)}}
                  for seed in SCREEN_SEEDS]
        qs = {arm: _mean([r[arm]['max_fragment_mae'] for r in resets]) for arm in ('anchor', *ARMS)}
        paired.append({'model_seed': model, 'Q_anchor': qs['anchor'], 'Q_C6': qs['C6'], 'Q_Q6': qs['Q6'],
            'Q6_lt_C6': qs['Q6'] < qs['C6'], 'Q6_lt_anchor': qs['Q6'] < qs['anchor'],
            'passed': qs['Q6'] < qs['C6'] and qs['Q6'] < qs['anchor'], 'by_reset': resets})
    unsafe = [{k: r[k] for k in ('seed', 'tick', 'model_seed', 'arm', 'repeat', 'scale_index', 'sign')}
              for r in indexed.values() if any(r.get(k, False) for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated'))]
    gates = {'all372_branches_complete': True, 'all372_safe': not unsafe,
             'all15_anchor_replays_exact': not replay_failures, 'all_three_seed_Q_improve': all(p['passed'] for p in paired)}
    passed = all(gates.values())
    return {'schema': 'd1-objective81-screen-v1', 'status': 'COMPLETE', 'passed': passed,
        'enter_full_evaluation': passed, 'gates': gates, 'reasons': [k for k, v in gates.items() if not v],
        'by_model_seed': paired, 'points': points, 'unsafe_branches': unsafe, 'anchor_replay_failures': replay_failures,
        'counts': {'sites': 15, 'branches': 372, 'policy_transitions': 11904, 'native_steps': 119040, 'pd_updates': 59520,
                   'steady_branches': 300, 'other_branches': 72, 'fixed_candidate_and_anchor_subset_branches': 120},
        'fixed_candidate': {'scale_index': 0, 'sign': 1, 'nominal_scale': SCALES[0]},
        'diagnostics': _diagnostics(indexed), 'Q_definition': 'mean_reset(max(MAE175,MAE325))',
        'full_evaluation_if_failed': 'NOT_RUN_FOR_ALL_SIX_POLICIES_AND_SHARED_BASELINE',
        'selected_seed': None, 'complete_model_eligibility': 'NOT_REQUALIFIED', 'global_candidate_accepted': False,
        'scope': 'independent validation screen; not full-window control or derivative qualification'}


def _index(rows, candidate):
    _require(isinstance(rows, (list, tuple)) and len(rows) == (54 if candidate else 9), 'exact registered episode count required')
    indexed = {}
    for row in rows:
        required = {'task', 'seed', 'metrics'} | ({'model_seed', 'arm'} if candidate else set())
        _require(isinstance(row, dict) and required <= row.keys(), 'episode wrapper fields missing')
        _finite_tree(row)
        _require(row['task'] in TASKS and type(row['seed']) is int and row['seed'] in RESET_SEEDS,
                 'unexpected evaluation task/reset')
        if 'status' in row:
            _require(row['status'] == 'COMPLETE', 'wrapper marks partial episode')
        if candidate:
            _require(type(row['model_seed']) is int and row['model_seed'] in MODEL_SEEDS and row['arm'] in ARMS,
                     'unexpected model seed/arm')
        else:
            _require('model_seed' not in row and 'arm' not in row, 'baseline must be shared original actor')
        key = (row['model_seed'], row['arm'], row['task'], row['seed']) if candidate else (row['task'], row['seed'])
        _require(key not in indexed, 'duplicate registered episode')
        indexed[key] = _metrics(row)
    expected = {(m, a, t, s) for m in MODEL_SEEDS for a in ARMS for t, s in SCENARIOS} if candidate else set(SCENARIOS)
    _require(set(indexed) == expected, 'registered episode grid incomplete')
    return indexed


def _control_gate(model, arm, baseline, candidates):
    pairs, turns = [], []
    for task, seed in SCENARIOS:
        base, candidate = baseline[(task, seed)], candidates[(model, arm, task, seed)]
        pairs.append({'task': task, 'seed': seed, 'baseline_key': f'{task}/seed{seed}',
            'baseline_J': base['J'], 'candidate_J': candidate['J'], 'J_ratio': _ratio(candidate['J'], base['J'], 1e-12),
            'J_pair_le1p05': candidate['J'] <= 1.05 * base['J'],
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
    return {'model_seed': model, 'arm': arm, 'passed': all(gates.values()), 'gates': gates,
        'reasons': [k for k, v in gates.items() if not v], 'J_ratio': j_ratio, 'yaw_mae_ratio': yaw_ratio,
        'pairs': pairs, 'turn_pairs': turns, 'baseline_keys': [f'{t}/seed{s}' for t, s in SCENARIOS],
        'ratio_denominator_must_exceed': 1e-12}


def compare(baseline_rows, candidate_rows):
    """Judge all63 H episodes; C6 relative improvement is not a required gate."""
    baseline, candidates = _index(baseline_rows, False), _index(candidate_rows, True)
    controls = [_control_gate(m, a, baseline, candidates) for m in MODEL_SEEDS for a in ARMS]
    paired, observations = [], []
    for model in MODEL_SEEDS:
        turns = []
        for task, seed in SCENARIOS:
            control, active = [candidates[(model, arm, task, seed)] for arm in ARMS]
            row = {'model_seed': model, 'task': task, 'seed': seed, 'baseline_key': f'{task}/seed{seed}',
                'C6_J': control['J'], 'Q6_J': active['J'], 'J_ratio_Q6_over_C6': _ratio(active['J'], control['J']),
                'C6_reference_thresholds_met': control['reference_thresholds_met'],
                'Q6_reference_thresholds_met': active['reference_thresholds_met']}
            if task == 'turn_stop':
                row.update(C6_wz_mae=control['stable_wz_mae_rad_s'], Q6_wz_mae=active['stable_wz_mae_rad_s'])
                turns.append(row)
            observations.append(row)
        cmean, qmean = [_mean([r[k] for r in turns]) for k in ('C6_wz_mae', 'Q6_wz_mae')]
        paired.append({'model_seed': model, 'turn_reset_count': 3, 'C6_mean_stable_wz_mae': cmean,
            'Q6_mean_stable_wz_mae': qmean, 'stable_wz_mae_ratio_Q6_over_C6': _ratio(qmean, cmean),
            'strict_stable_wz_mae_improvement': qmean < cmean, 'turn_pairs': turns})
    gates = {'all_three_Q6_original_control_gates': all(c['passed'] for c in controls if c['arm'] == 'Q6'),
        'all_three_paired_stable_yaw_means_strictly_improve': all(p['strict_stable_wz_mae_improvement'] for p in paired),
        'all63_original_task_stop_safety_thresholds': all(m['reference_thresholds_met'] for m in
                                                        (*baseline.values(), *candidates.values()))}
    counts = {label: {key: sum(m['actual_counts'][key] for m in rows.values()) for key in COUNTS}
              for label, rows in (('baseline', baseline), ('candidate', candidates))}
    counts['total'] = {key: counts['baseline'][key] + counts['candidate'][key] for key in COUNTS}
    passed = all(gates.values())
    return {'schema': 'd1-objective81-comparison-v1', 'status': 'COMPLETE', 'objective_passed': passed,
        'objective': {'passed': passed, 'gates': gates, 'reasons': [k for k, v in gates.items() if not v],
            'primary_metric': 'stable_wz_mae_rad_s', 'by_model_seed': paired},
        'paired_observations': observations, 'control_gates': controls, 'evaluation_counts': counts,
        'sampling_unit': {'paired_model_seed_units': 3, 'paired_observations': 27, 'scenarios_per_policy': 9,
            'candidate_policies': 6, 'candidate_episodes': 54, 'unique_baseline_episodes': 9,
            'baseline_episode_references_in_control_comparisons': 54, 'baseline_policy_steps_counted_once': 5400,
            '27_or54_observations_are_not_independent_model_seeds': True,
            'model_seeds_are_world_model_learners_not_from_scratch_PPO_seeds': True},
        'reduction': 'binary64 math.fsum/n; stable yaw primary; original J and safety gates unchanged',
        'undefined_ratio_is_not_a_pass': True, 'complete_model_eligibility': 'NOT_REQUALIFIED',
        'global_candidate_accepted': False, 'global_accepted': False, 'candidate_accepted': False, 'selected_seed': None,
        'scope': 'complete H metrics only; runner separately verifies source, update and V admission'}
