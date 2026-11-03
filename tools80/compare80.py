"""Registered80 decisions from complete saved metrics; stdlib only.

The original79 metric validator is reused unchanged. Sampling is judged by
stable turn yaw MAE; the separate original control and model gates are retained.
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

ARMS = ('legacy', 'steady')
RESET_SEEDS = (80401, 80402, 80403)
SCENARIOS = tuple((task, seed) for task in TASKS for seed in RESET_SEEDS)
SCREEN_SEEDS = (80107, 80108, 80109)
SCREEN_TICKS = (110, 175, 260, 325, 410)
SCREEN_COUNTS = {'steps': 32, 'native_steps': 320, 'pd_updates': 160}
SCREEN_POLICIES = ((None, 'anchor', 0), (None, 'anchor', 1)) + tuple(
    (model_seed, arm, 0) for model_seed in MODEL_SEEDS for arm in ARMS)
TICK_ROLES = {110: 'positive_turn_transition', 175: 'positive_turn_stable_fragment',
              260: 'negative_turn_transition', 325: 'negative_turn_stable_fragment',
              410: 'stopping_fragment'}


def _index(rows, candidate):
    _require(isinstance(rows, (list, tuple)) and len(rows) == (54 if candidate else 9),
             'exact registered episode count required')
    indexed = {}
    for row in rows:
        required = {'task', 'seed', 'metrics'} | ({'model_seed', 'arm'} if candidate else set())
        _require(isinstance(row, dict) and required <= row.keys(), 'episode wrapper fields missing')
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


def _control_gate(model_seed, arm, baseline, candidates):
    pairs, turns = [], []
    for task, seed in SCENARIOS:
        base, candidate = baseline[(task, seed)], candidates[(model_seed, arm, task, seed)]
        pairs.append({'task': task, 'seed': seed, 'baseline_key': f'{task}/seed{seed}',
            'baseline_J': base['J'], 'candidate_J': candidate['J'],
            'J_ratio': _ratio(candidate['J'], base['J'], 1e-12),
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
    return {'model_seed': model_seed, 'arm': arm, 'passed': all(gates.values()), 'gates': gates,
        'reasons': [key for key, passed in gates.items() if not passed], 'J_ratio': j_ratio,
        'yaw_mae_ratio': yaw_ratio, 'pairs': pairs, 'turn_pairs': turns,
        'baseline_keys': [f'{t}/seed{s}' for t, s in SCENARIOS], 'ratio_denominator_must_exceed': 1e-12}


def compare(baseline_rows, candidate_rows):
    """Compare all9 shared baselines and all54 candidates, never select a seed.

Missing, partial or inconsistent records raise ValueError. Complete original
threshold failures remain failed observations. Execution/source eligibility is
the runner's responsibility; this function never requalifies the world model.
"""
    baseline, candidates = _index(baseline_rows, False), _index(candidate_rows, True)
    controls = [_control_gate(m, a, baseline, candidates) for m in MODEL_SEEDS for a in ARMS]
    paired, observations = [], []
    for model_seed in MODEL_SEEDS:
        turns = []
        for task, seed in SCENARIOS:
            legacy, steady = [candidates[(model_seed, arm, task, seed)] for arm in ARMS]
            row = {'model_seed': model_seed, 'task': task, 'seed': seed,
                'baseline_key': f'{task}/seed{seed}', 'legacy_J': legacy['J'], 'steady_J': steady['J'],
                'J_ratio_steady_over_legacy': _ratio(steady['J'], legacy['J']),
                'legacy_reference_thresholds_met': legacy['reference_thresholds_met'],
                'steady_reference_thresholds_met': steady['reference_thresholds_met']}
            if task == 'turn_stop':
                row.update(legacy_wz_mae=legacy['stable_wz_mae_rad_s'], steady_wz_mae=steady['stable_wz_mae_rad_s'])
                turns.append(row)
            observations.append(row)
        legacy_mean, steady_mean = [_mean([r[key] for r in turns]) for key in ('legacy_wz_mae', 'steady_wz_mae')]
        paired.append({'model_seed': model_seed, 'turn_reset_count': 3,
            'legacy_mean_stable_wz_mae': legacy_mean, 'steady_mean_stable_wz_mae': steady_mean,
            'stable_wz_mae_ratio_steady_over_legacy': _ratio(steady_mean, legacy_mean),
            'strict_stable_wz_mae_improvement': steady_mean < legacy_mean, 'turn_pairs': turns})
    gates = {'all_three_steady_original_control_gates': all(c['passed'] for c in controls if c['arm'] == 'steady'),
        'all_three_paired_stable_yaw_means_strictly_improve': all(p['strict_stable_wz_mae_improvement'] for p in paired),
        'all63_original_task_stop_safety_thresholds': all(m['reference_thresholds_met'] for m in
                                                        (*baseline.values(), *candidates.values()))}
    counts = {label: {key: sum(m['actual_counts'][key] for m in rows.values()) for key in COUNTS}
              for label, rows in (('baseline', baseline), ('candidate', candidates))}
    counts['total'] = {key: counts['baseline'][key] + counts['candidate'][key] for key in COUNTS}
    passed = all(gates.values())
    return {'schema': 'd1-steady-sampling80-comparison-v1', 'status': 'COMPLETE', 'sampling_passed': passed,
        'sampling': {'passed': passed, 'gates': gates, 'reasons': [k for k, v in gates.items() if not v],
            'primary_metric': 'stable_wz_mae_rad_s', 'by_model_seed': paired},
        'paired_observations': observations, 'control_gates': controls, 'evaluation_counts': counts,
        'sampling_unit': {'paired_model_seed_units': 3, 'paired_observations': 27, 'scenarios_per_policy': 9,
            'candidate_policies': 6, 'candidate_episodes': 54, 'unique_baseline_episodes': 9,
            'baseline_episode_references_in_control_comparisons': 54, 'baseline_policy_steps_counted_once': 5400,
            '27_or54_observations_are_not_independent_model_seeds': True,
            'model_seeds_are_world_model_learners_not_from_scratch_PPO_seeds': True},
        'reduction': 'binary64 math.fsum/n; registered equal weights; paired primary metric is stable yaw, not J',
        'undefined_ratio_is_not_a_pass': True, 'complete_model_eligibility': 'NOT_REQUALIFIED',
        'global_candidate_accepted': False, 'global_accepted': False, 'candidate_accepted': False, 'selected_seed': None,
        'scope': 'complete physical metrics only; sources, updates, screen admission and execution verified separately'}


def _screen_index(rows):
    _require(isinstance(rows, (list, tuple)) and len(rows) == 120, 'exact120 registered short branches required')
    indexed = {}
    required = {'model_seed', 'arm', 'repeat', 'seed', 'tick', 'status', 'steps', 'native_steps', 'pd_updates',
                'fallen', 'illegal_body_contact', 'cost', 'yaw_mae', 'trace_identity_sha256'}
    for row in rows:
        _require(isinstance(row, dict) and required <= row.keys(), 'short branch fields missing')
        _finite_tree(row)
        _require(type(row['seed']) is int and row['seed'] in SCREEN_SEEDS and
                 type(row['tick']) is int and row['tick'] in SCREEN_TICKS, 'unexpected short site')
        _require(type(row['repeat']) is int, 'repeat must be an integer')
        if row['arm'] == 'anchor':
            _require(row['model_seed'] is None and row['repeat'] in (0, 1), 'anchor identity must be shared with two repeats')
        else:
            _require(row['arm'] in ARMS and type(row['model_seed']) is int and row['model_seed'] in MODEL_SEEDS and
                     row['repeat'] == 0, 'unexpected short candidate identity')
        _require(row['status'] == 'COMPLETE' and all(type(row[k]) is int and row[k] == n for k, n in SCREEN_COUNTS.items()),
                 'short branch must be COMPLETE with32/320/160 counts')
        _require(type(row['fallen']) is bool and type(row['illegal_body_contact']) is bool, 'explicit boolean safety flags required')
        _number(row['cost'], 'short cost')
        _number(row['yaw_mae'], 'short yaw MAE')
        sha = row['trace_identity_sha256']
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
                 'lowercase64-digit trace SHA256 required')
        key = (row['seed'], row['tick'], row['model_seed'], row['arm'], row['repeat'])
        _require(key not in indexed, 'duplicate short branch identity')
        indexed[key] = row
    expected = {(s, t, m, a, r) for s in SCREEN_SEEDS for t in SCREEN_TICKS for m, a, r in SCREEN_POLICIES}
    _require(set(indexed) == expected, 'registered short branch grid incomplete')
    return indexed


def screen_compare(rows):
    """Admit all three learners together using15 sites and120 complete branches.

Replay/safety failures are complete failed observations. Missing/partial rows
raise ValueError. Q is mean over resets of max(two steady-window fragments),
not the original full-window metric or a physical derivative qualification.
"""
    indexed = _screen_index(rows)
    points, replay_failures = [], []
    for seed in SCREEN_SEEDS:
        for tick in SCREEN_TICKS:
            anchors = [indexed[(seed, tick, None, 'anchor', r)] for r in (0, 1)]
            replay = all(anchors[0][k] == anchors[1][k] for k in ('trace_identity_sha256', 'cost', 'yaw_mae',
                                                               'fallen', 'illegal_body_contact'))
            if not replay:
                replay_failures.append({'seed': seed, 'tick': tick})
            points.append({'seed': seed, 'tick': tick, 'fragment': TICK_ROLES[tick], 'anchor_replay_exact': replay,
                'branches': [dict(indexed[(seed, tick, m, a, r)]) for m, a, r in SCREEN_POLICIES]})
    paired = []
    for model_seed in MODEL_SEEDS:
        resets = []
        for seed in SCREEN_SEEDS:
            row = {'seed': seed}
            for arm in ('anchor', *ARMS):
                model = None if arm == 'anchor' else model_seed
                pos, neg = [indexed[(seed, tick, model, arm, 0)]['yaw_mae'] for tick in (175, 325)]
                row[arm] = {'positive_turn_fragment_mae': pos, 'negative_turn_fragment_mae': neg,
                            'max_fragment_mae': max(pos, neg)}
            resets.append(row)
        qs = {arm: _mean([r[arm]['max_fragment_mae'] for r in resets]) for arm in ('anchor', *ARMS)}
        passed = qs['steady'] < qs['legacy'] and qs['steady'] < qs['anchor']
        paired.append({'model_seed': model_seed, 'Q_anchor': qs['anchor'], 'Q_legacy': qs['legacy'],
            'Q_steady': qs['steady'], 'steady_lt_legacy': qs['steady'] < qs['legacy'],
            'steady_lt_anchor': qs['steady'] < qs['anchor'], 'passed': passed, 'by_reset': resets})
    unsafe = [{'seed': r['seed'], 'tick': r['tick'], 'model_seed': r['model_seed'], 'arm': r['arm'], 'repeat': r['repeat']}
              for r in indexed.values() if r['fallen'] or r['illegal_body_contact']]
    gates = {'all120_branches_complete': True, 'all120_safe': not unsafe,
             'all15_anchor_replays_exact': not replay_failures, 'all_three_seed_Q_improve': all(p['passed'] for p in paired)}
    passed = all(gates.values())
    return {'schema': 'd1-steady-sampling80-screen-v1', 'status': 'COMPLETE', 'passed': passed,
        'enter_full_evaluation': passed, 'gates': gates, 'reasons': [k for k, v in gates.items() if not v],
        'by_model_seed': paired, 'points': points, 'unsafe_branches': unsafe, 'anchor_replay_failures': replay_failures,
        'counts': {'sites': 15, 'branches': 120, 'policy_transitions': 3840, 'native_steps': 38400, 'pd_updates': 19200},
        'Q_definition': 'mean over three resets of max(MAE of175 branch, MAE of325 branch)',
        'full_evaluation_if_failed': 'NOT_RUN_FOR_ALL_SIX_POLICIES_AND_SHARED_BASELINE',
        'selected_seed': None, 'complete_model_eligibility': 'NOT_REQUALIFIED', 'global_candidate_accepted': False,
        'scope': 'development screen only; not full stable-window control evaluation or true derivative qualification'}
