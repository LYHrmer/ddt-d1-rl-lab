"""Synthetic81 comparator cases; no numerical libraries or physical work."""
import copy
import json
import math
import unittest

from tools81.compare81 import (
    ARMS,
    MODEL_SEEDS,
    RESET_SEEDS,
    SCALES,
    SCREEN_SEEDS,
    SCREEN_TICKS,
    STEADY_TICKS,
    TASKS,
    branch_specs,
    compare,
    diagnostics,
    screen_compare,
)


def episode(task, seed, j=1., yaw=.04):
    turn = task == 'turn_stop'
    return {'sequence': task, 'reset_seed': seed, 'delay_ticks': 0, 'delay_ms': 0,
        'policy_transitions': 600, 'native_steps': 6000, 'done_reason': 'HORIZON_12S', 'status': 'COMPLETE',
        'failure': False, 'J': j, 'J_components': {'full_horizon_vx_mae_over_0_2': j,
            'full_horizon_wz_mae_over_0_2': 0., 'stop_8_to_12_horizontal_norm_mean_over_0_2': 0.,
            'fixed_early_failure_penalty': 0.},
        'actual_counts': {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000},
        'stable_vx_mae_m_s': None if turn else .03, 'stable_wz_mae_rad_s': yaw if turn else None,
        'stop_last_second_horizontal_speed_m_s': .01, 'stop_last_second_abs_wz_rad_s': .02,
        'stop_displacement_m': .02, 'peak_abs_roll_or_pitch_rad': .1,
        'threshold_flags': {'completed_12s': True, 'no_fall_or_illegal_support': True,
            'stable_vx': None if turn else True, 'stable_wz': True if turn else None,
            'stop_horizontal_speed': True, 'stop_abs_wz': True, 'stop_displacement': True, 'peak_roll_pitch': True},
        'reference_thresholds_met': True, 'nominal_zero_delay_pass': True}


def formal_fixture():
    baseline = [{'task': t, 'seed': s, 'metrics': episode(t, s)} for t in TASKS for s in RESET_SEEDS]
    candidates = [{'model_seed': m, 'arm': a, 'task': t, 'seed': s,
                   'metrics': episode(t, s, .98, .04 if a == 'C6' else .036)}
                  for m in MODEL_SEEDS for a in ARMS for t in TASKS for s in RESET_SEEDS]
    return baseline, candidates


def screen_fixture():
    rows = []
    for seed in SCREEN_SEEDS:
        for tick in SCREEN_TICKS:
            for model, arm, repeat, index, sign in branch_specs(tick):
                yaw = .06 if arm == 'anchor' else .06 - sign * (.01 if arm == 'C6' else .02) / (2 ** index)
                rows.append({'model_seed': model, 'arm': arm, 'repeat': repeat, 'scale_index': index,
                    'sign': sign, 'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'steps': 32,
                    'native_steps': 320, 'pd_updates': 160, 'fallen': False, 'illegal_body_contact': False,
                    'cost': .1, 'yaw_mae': yaw, 'trace_identity_sha256': f'{seed * 1000 + tick:064x}'})
    return rows


def set_j(row, value):
    row['metrics']['J'] = value
    row['metrics']['J_components']['full_horizon_vx_mae_over_0_2'] = value


def fail_threshold(row):
    item = row['metrics']
    item['stop_last_second_horizontal_speed_m_s'] = .021
    item['threshold_flags']['stop_horizontal_speed'] = False
    item['reference_thresholds_met'] = False
    item['nominal_zero_delay_pass'] = False


class ScreenCompare81Tests(unittest.TestCase):
    def test_complete_registered_grid_counts_and_immutability(self):
        rows = screen_fixture()
        before = copy.deepcopy(rows)
        result = screen_compare(rows)
        self.assertTrue(result['enter_full_evaluation'])
        self.assertEqual(len(rows), 372)
        self.assertEqual([len(branch_specs(t)) for t in SCREEN_TICKS], [8, 50, 8, 50, 8])
        self.assertEqual(len(result['points']), 15)
        self.assertEqual(sum(len(p['branches']) for p in result['points']), 372)
        self.assertEqual(result['counts']['policy_transitions'], 11904)
        self.assertEqual(result['counts']['native_steps'], 119040)
        self.assertEqual(result['counts']['pd_updates'], 59520)
        self.assertEqual(result['counts']['fixed_candidate_and_anchor_subset_branches'], 120)
        self.assertEqual(result['fixed_candidate'], {'scale_index': 0, 'sign': 1, 'nominal_scale': .009})
        self.assertFalse(result['global_candidate_accepted'])
        self.assertEqual(rows, before)
        json.dumps(result, allow_nan=False)

    def test_diagnostics_all_scales_and_resets_are_not_more_independent_seeds(self):
        result = diagnostics(screen_fixture())
        self.assertEqual(len(result['by_model_arm_reset_scale']), 72)
        self.assertEqual(len(result['by_model_arm_scale']), 24)
        self.assertEqual({r['nominal_scale'] for r in result['by_model_arm_reset_scale']}, set(SCALES))
        self.assertTrue(result['all_Q6_model_reset_scale_raw_strict_descent'])
        self.assertTrue(result['all_C6_model_reset_scale_raw_strict_descent'])
        self.assertFalse(result['used_for_candidate_selection'])
        self.assertIsNone(result['numerical_derivative_qualification'])
        self.assertFalse(result['action_resolution_checked_by_this_module'])

    def test_nonmonotone_diagnostic_does_not_veto_fixed_candidate(self):
        rows = screen_fixture()
        for row in rows:
            if (row['arm'] == 'Q6' and row['tick'] in STEADY_TICKS and
                    (row['sign'] == -1 or row['scale_index'] == 3)):
                row['yaw_mae'] = .06
        result = screen_compare(rows)
        self.assertTrue(result['passed'])
        self.assertFalse(result['diagnostics']['all_Q6_model_reset_scale_raw_strict_descent'])
        unknown = [r for r in result['diagnostics']['by_model_arm_reset_scale'] if r['arm'] == 'Q6' and r['scale_index'] == 3]
        self.assertTrue(all(r['plus_equals_zero'] and r['minus_equals_zero'] for r in unknown))

    def test_cannot_rescue_main_candidate_by_selecting_smaller_scale(self):
        rows = screen_fixture()
        for row in rows:
            if row['arm'] == 'Q6' and row['tick'] in STEADY_TICKS and row['sign'] == 1:
                row['yaw_mae'] = .07 if row['scale_index'] == 0 else .001
        result = screen_compare(rows)
        self.assertFalse(result['passed'])
        self.assertTrue(all(p['Q_Q6'] == .07 for p in result['by_model_seed']))
        self.assertEqual(result['fixed_candidate']['scale_index'], 0)

    def test_max_is_within_reset_before_mean_and_other_ticks_do_not_score(self):
        rows = screen_fixture()
        for row in rows:
            if row['arm'] == 'Q6':
                if row['tick'] not in STEADY_TICKS:
                    row['yaw_mae'] = 999.
                else:
                    low_tick = 175 if row['seed'] == SCREEN_SEEDS[0] else 325
                    row['yaw_mae'] = 0. if row['tick'] == low_tick else .06
        result = screen_compare(rows)
        self.assertFalse(result['passed'])
        self.assertTrue(all(p['Q_Q6'] == .06 for p in result['by_model_seed']))
        self.assertTrue(any(b['yaw_mae'] == 999. for p in result['points'] for b in p['branches']))

    def test_no_seed_compensation_or_equality_pass(self):
        for against in ('C6', 'anchor'):
            rows = screen_fixture()
            for row in rows:
                if against == 'anchor' and row['arm'] == 'C6':
                    row['yaw_mae'] = .07
                if row['model_seed'] == MODEL_SEEDS[0] and row['arm'] == 'Q6':
                    row['yaw_mae'] = .05 if against == 'C6' else .06
            with self.subTest(against=against):
                result = screen_compare(rows)
                self.assertFalse(result['passed'])
                self.assertEqual([p['passed'] for p in result['by_model_seed']], [False, True, True])

    def test_safety_in_reverse_diagnostic_or_parking_blocks(self):
        for tick, sign, index in ((175, -1, 3), (410, 1, 0)):
            rows = screen_fixture()
            target = next(r for r in rows if r['arm'] == 'C6' and r['tick'] == tick and
                          r['sign'] == sign and r['scale_index'] == index)
            target['illegal_body_contact'] = True
            result = screen_compare(rows)
            with self.subTest(tick=tick):
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertTrue(result['gates']['all_three_seed_Q_improve'])
                self.assertFalse(result['gates']['all372_safe'])
                self.assertFalse(result['passed'])

    def test_anchor_mismatch_retains_complete_failed_observation(self):
        for key, value in (('trace_identity_sha256', 'f' * 64), ('yaw_mae', .061), ('cost', .2)):
            rows = screen_fixture()
            next(r for r in rows if r['arm'] == 'anchor' and r['repeat'] == 1)[key] = value
            with self.subTest(key=key):
                result = screen_compare(rows)
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertFalse(result['gates']['all15_anchor_replays_exact'])
                self.assertEqual(len(result['anchor_replay_failures']), 1)
                self.assertFalse(result['passed'])

    def test_incomplete_duplicate_malformed_or_nonfinite_grid_rejected(self):
        mutations = [('missing', None), ('duplicate', None), ('seed', 80107), ('tick', 176),
            ('status', 'PARTIAL'), ('steps', 31), ('native_steps', 319), ('pd_updates', 159),
            ('model_seed', MODEL_SEEDS[0]), ('repeat', True), ('scale_index', 0), ('sign', True),
            ('trace_identity_sha256', 'bad'), ('fallen', None), ('yaw_mae', math.nan), ('cost', math.inf)]
        for key, value in mutations:
            rows = screen_fixture()
            if key == 'missing':
                rows.pop()
            elif key == 'duplicate':
                rows[-1] = copy.deepcopy(rows[0])
            else:
                rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen_compare(rows)

    def test_nonsteady_extra_probes_and_candidate_identity_rejected(self):
        for key, value in (('scale_index', 1), ('sign', -1), ('repeat', 1), ('arm', 'steady')):
            rows = screen_fixture()
            target = next(r for r in rows if r['arm'] == 'Q6' and r['tick'] == 110)
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen_compare(rows)
        rows = screen_fixture()
        rows[0]['actual_counts'] = {'policy_transitions': 32, 'native_steps': 320, 'pd_updates': 159}
        with self.assertRaises(ValueError):
            screen_compare(rows)


class FormalCompare81Tests(unittest.TestCase):
    def test_Q6_success_does_not_require_C6_relative_improvement(self):
        b, c = formal_fixture()
        before = copy.deepcopy((b, c))
        result = compare(b, c)
        self.assertTrue(result['objective_passed'])
        self.assertTrue(all(r['passed'] for r in result['control_gates'] if r['arm'] == 'Q6'))
        self.assertTrue(all(not r['passed'] for r in result['control_gates'] if r['arm'] == 'C6'))
        self.assertEqual(result['evaluation_counts']['total'],
                         {'policy_transitions': 37800, 'native_steps': 378000, 'pd_updates': 189000})
        self.assertEqual(result['evaluation_counts']['baseline']['policy_transitions'], 5400)
        self.assertEqual(len(result['paired_observations']), 27)
        self.assertEqual(result['sampling_unit']['paired_model_seed_units'], 3)
        self.assertEqual(result['sampling_unit']['unique_baseline_episodes'], 9)
        self.assertEqual(result['complete_model_eligibility'], 'NOT_REQUALIFIED')
        self.assertFalse(result['global_candidate_accepted'])
        self.assertEqual((b, c), before)
        json.dumps(result, allow_nan=False)

    def test_C6_absolute_failure_still_blocks(self):
        b, c = formal_fixture()
        fail_threshold(next(r for r in c if r['arm'] == 'C6'))
        result = compare(b, c)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertTrue(result['objective']['gates']['all_three_Q6_original_control_gates'])
        self.assertFalse(result['objective']['gates']['all63_original_task_stop_safety_thresholds'])
        self.assertFalse(result['objective_passed'])

    def test_Q6_pair_improvement_required_for_every_model_even_when_control_passes(self):
        b, c = formal_fixture()
        for row in c:
            if row['arm'] == 'C6' and row['task'] == 'turn_stop' and row['model_seed'] == MODEL_SEEDS[0]:
                row['metrics']['stable_wz_mae_rad_s'] = .036
        result = compare(b, c)
        self.assertTrue(result['objective']['gates']['all_three_Q6_original_control_gates'])
        self.assertFalse(result['objective_passed'])
        self.assertEqual([p['strict_stable_wz_mae_improvement'] for p in result['objective']['by_model_seed']],
                         [False, True, True])

    def test_original_five_percent_and_each_turn_reset_rules(self):
        for value, expected in ((.038, True), (.0380001, False)):
            b, c = formal_fixture()
            for row in c:
                if row['arm'] == 'Q6' and row['task'] == 'turn_stop':
                    row['metrics']['stable_wz_mae_rad_s'] = value
            with self.subTest(value=value):
                self.assertEqual(compare(b, c)['objective_passed'], expected)
        b, c = formal_fixture()
        selected = [r for r in c if r['arm'] == 'Q6' and r['model_seed'] == MODEL_SEEDS[0] and r['task'] == 'turn_stop']
        for row in selected:
            row['metrics']['stable_wz_mae_rad_s'] = .01
        selected[0]['metrics']['stable_wz_mae_rad_s'] = .04
        result = compare(b, c)
        gate = next(r for r in result['control_gates'] if r['arm'] == 'Q6')
        self.assertTrue(gate['gates']['turn_mean_ratio_le0p95'])
        self.assertFalse(gate['gates']['all_three_turn_resets_strictly_improve'])
        self.assertFalse(result['objective_passed'])

    def test_original_J_mean_pair_limits_and_primary_not_J(self):
        b, c = formal_fixture()
        selected = [r for r in c if r['arm'] == 'Q6' and r['model_seed'] == MODEL_SEEDS[0]]
        for row in selected:
            set_j(row, .9)
        set_j(selected[0], 1.05)
        self.assertTrue(compare(b, c)['objective_passed'])
        set_j(selected[0], math.nextafter(1.05, math.inf))
        self.assertFalse(compare(b, c)['objective_passed'])
        for row in selected:
            set_j(row, 1.001)
        gate = next(r for r in compare(b, c)['control_gates'] if r['arm'] == 'Q6')
        self.assertFalse(gate['gates']['J_mean_ratio_le1'])
        self.assertTrue(gate['gates']['all_nine_J_pairs_le1p05'])

    def test_zero_denominator_is_not_a_control_pass(self):
        b, c = formal_fixture()
        for row in b + c:
            set_j(row, 0.)
        result = compare(b, c)
        self.assertFalse(result['objective_passed'])
        self.assertTrue(all(r['J_ratio'] is None for r in result['control_gates']))

    def test_missing_duplicate_old_seed_partial_count_or_unknown_rejected(self):
        for mutation in ('missing', 'duplicate', 'old_seed', 'partial', 'count', 'wrong_arm', 'wrapper_identity', 'nan', 'unknown'):
            b, c = formal_fixture()
            if mutation == 'missing':
                c.pop()
            elif mutation == 'duplicate':
                c[-1] = copy.deepcopy(c[0])
            elif mutation == 'old_seed':
                c[0]['seed'] = 80401
            elif mutation == 'partial':
                c[0]['status'] = 'PARTIAL'
            elif mutation == 'count':
                c[0]['metrics']['actual_counts']['pd_updates'] -= 1
            elif mutation == 'wrong_arm':
                c[0]['arm'] = 'steady'
            elif mutation == 'wrapper_identity':
                c[0]['metrics']['reset_seed'] = RESET_SEEDS[1]
            elif mutation == 'nan':
                c[0]['metrics']['J'] = math.nan
            else:
                c[0]['metrics']['reference_thresholds_met'] = None
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(b, c)


if __name__ == '__main__':
    unittest.main()
