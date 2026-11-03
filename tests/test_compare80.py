"""Synthetic decisions only: no actor, model, numerical library or physics."""
import copy
import json
import math
import unittest

from tools80.compare80 import (
    ARMS,
    MODEL_SEEDS,
    RESET_SEEDS,
    SCREEN_POLICIES,
    SCREEN_SEEDS,
    SCREEN_TICKS,
    TASKS,
    compare,
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
                   'metrics': episode(t, s, .98, .037 if a == 'legacy' else .036)}
                  for m in MODEL_SEEDS for a in ARMS for t in TASKS for s in RESET_SEEDS]
    return baseline, candidates


def screen_fixture():
    rows = []
    for seed in SCREEN_SEEDS:
        for tick in SCREEN_TICKS:
            for model, arm, repeat in SCREEN_POLICIES:
                rows.append({'model_seed': model, 'arm': arm, 'repeat': repeat, 'seed': seed, 'tick': tick,
                    'status': 'COMPLETE', 'steps': 32, 'native_steps': 320, 'pd_updates': 160,
                    'fallen': False, 'illegal_body_contact': False, 'cost': .1,
                    'yaw_mae': {'anchor': .06, 'legacy': .05, 'steady': .04}[arm],
                    'trace_identity_sha256': f'{seed * 1000 + tick:064x}'})
    return rows


def set_j(row, value):
    row['metrics']['J'] = value
    row['metrics']['J_components']['full_horizon_vx_mae_over_0_2'] = value


def fail_threshold(row):
    row['metrics']['stop_last_second_horizontal_speed_m_s'] = .021
    row['metrics']['threshold_flags']['stop_horizontal_speed'] = False
    row['metrics']['reference_thresholds_met'] = False
    row['metrics']['nominal_zero_delay_pass'] = False


class FormalCompare80Tests(unittest.TestCase):
    def test_complete_sampling_success_keeps_model_ineligible_and_shared_baseline_once(self):
        b, c = formal_fixture()
        before = copy.deepcopy((b, c))
        result = compare(b, c)
        self.assertTrue(result['sampling_passed'])
        self.assertTrue(all(r['passed'] for r in result['control_gates']))
        self.assertEqual(result['sampling']['primary_metric'], 'stable_wz_mae_rad_s')
        self.assertFalse(result['global_candidate_accepted'])
        self.assertEqual(result['complete_model_eligibility'], 'NOT_REQUALIFIED')
        self.assertEqual(result['evaluation_counts']['total'],
                         {'policy_transitions': 37800, 'native_steps': 378000, 'pd_updates': 189000})
        self.assertEqual(result['evaluation_counts']['baseline']['policy_transitions'], 5400)
        self.assertEqual(len(result['paired_observations']), 27)
        self.assertEqual(result['sampling_unit']['paired_model_seed_units'], 3)
        self.assertEqual((b, c), before)
        json.dumps(result, allow_nan=False)

    def test_primary_is_stable_yaw_not_J_or_old_feedback_one_percent(self):
        b, c = formal_fixture()
        for row in c:
            set_j(row, .99 if row['arm'] == 'steady' else .90)
        self.assertTrue(compare(b, c)['sampling_passed'])
        for row in c:
            if row['arm'] == 'steady':
                set_j(row, .5)
                if row['task'] == 'turn_stop':
                    row['metrics']['stable_wz_mae_rad_s'] = .037
        result = compare(b, c)
        self.assertTrue(all(r['passed'] for r in result['control_gates']))
        self.assertFalse(result['sampling_passed'])

    def test_no_seed_compensation_even_with_good_overall_yaw(self):
        b, c = formal_fixture()
        for row in c:
            if row['arm'] == 'steady' and row['task'] == 'turn_stop':
                row['metrics']['stable_wz_mae_rad_s'] = .037 if row['model_seed'] == MODEL_SEEDS[0] else .01
        result = compare(b, c)
        self.assertFalse(result['sampling_passed'])
        self.assertEqual([p['strict_stable_wz_mae_improvement'] for p in result['sampling']['by_model_seed']],
                         [False, True, True])

    def test_legacy_need_not_pass_relative_control_but_all63_absolute_thresholds_must(self):
        b, c = formal_fixture()
        for row in c:
            if row['arm'] == 'legacy' and row['task'] == 'turn_stop':
                row['metrics']['stable_wz_mae_rad_s'] = .04
        self.assertTrue(compare(b, c)['sampling_passed'])
        fail_threshold(next(r for r in c if r['arm'] == 'legacy'))
        result = compare(b, c)
        self.assertTrue(result['sampling']['gates']['all_three_steady_original_control_gates'])
        self.assertFalse(result['sampling']['gates']['all63_original_task_stop_safety_thresholds'])
        self.assertFalse(result['sampling_passed'])

    def test_original_five_percent_yaw_mean_and_each_reset_gate(self):
        for value, passed in ((.038, True), (.0380001, False)):
            b, c = formal_fixture()
            for row in c:
                if row['task'] == 'turn_stop':
                    row['metrics']['stable_wz_mae_rad_s'] = .039 if row['arm'] == 'legacy' else value
            with self.subTest(value=value):
                self.assertEqual(compare(b, c)['sampling_passed'], passed)
        b, c = formal_fixture()
        selected = [r for r in c if r['arm'] == 'steady' and r['model_seed'] == MODEL_SEEDS[0] and r['task'] == 'turn_stop']
        for row in selected:
            row['metrics']['stable_wz_mae_rad_s'] = .01
        selected[0]['metrics']['stable_wz_mae_rad_s'] = .04
        gate = next(r for r in compare(b, c)['control_gates'] if r['arm'] == 'steady')
        self.assertTrue(gate['gates']['turn_mean_ratio_le0p95'])
        self.assertFalse(gate['gates']['all_three_turn_resets_strictly_improve'])

    def test_original_J_mean_and_individual_cap_are_retained(self):
        b, c = formal_fixture()
        selected = [r for r in c if r['arm'] == 'steady' and r['model_seed'] == MODEL_SEEDS[0]]
        for row in selected:
            set_j(row, .9)
        set_j(selected[0], 1.05)
        self.assertTrue(compare(b, c)['sampling_passed'])
        set_j(selected[0], math.nextafter(1.05, math.inf))
        self.assertFalse(compare(b, c)['sampling_passed'])
        for row in selected:
            set_j(row, 1.001)
        result = compare(b, c)
        gate = next(r for r in result['control_gates'] if r['arm'] == 'steady')
        self.assertFalse(gate['gates']['J_mean_ratio_le1'])
        self.assertTrue(gate['gates']['all_nine_J_pairs_le1p05'])

    def test_partial_duplicate_wrong_reset_or_count_rejected(self):
        for mutation in ('missing', 'duplicate', 'old_seed', 'partial', 'count', 'wrong_arm', 'wrapper_identity'):
            b, c = formal_fixture()
            if mutation == 'missing':
                c.pop()
            elif mutation == 'duplicate':
                c[-1] = copy.deepcopy(c[0])
            elif mutation == 'old_seed':
                c[0]['seed'] = 79401
            elif mutation == 'partial':
                c[0]['status'] = 'PARTIAL'
            elif mutation == 'count':
                c[0]['metrics']['actual_counts']['pd_updates'] -= 1
            elif mutation == 'wrong_arm':
                c[0]['arm'] = 'refresh'
            else:
                c[0]['metrics']['reset_seed'] = RESET_SEEDS[1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare(b, c)

    def test_nonfinite_or_inconsistent_original_flags_rejected(self):
        b, c = formal_fixture()
        c[0]['metrics']['J'] = math.nan
        with self.assertRaises(ValueError):
            compare(b, c)
        b, c = formal_fixture()
        c[0]['metrics']['reference_thresholds_met'] = None
        with self.assertRaises(ValueError):
            compare(b, c)

    def test_zero_baseline_ratio_not_passed(self):
        b, c = formal_fixture()
        for row in b + c:
            set_j(row, 0.)
        result = compare(b, c)
        self.assertFalse(result['sampling_passed'])
        self.assertTrue(all(r['J_ratio'] is None for r in result['control_gates']))


class ScreenCompare80Tests(unittest.TestCase):
    def test_full_grid_counts_details_and_input_immutability(self):
        rows = screen_fixture()
        before = copy.deepcopy(rows)
        result = screen_compare(rows)
        self.assertTrue(result['enter_full_evaluation'])
        self.assertEqual(len(result['points']), 15)
        self.assertEqual(sum(len(p['branches']) for p in result['points']), 120)
        self.assertEqual(result['counts']['policy_transitions'], 3840)
        self.assertEqual(result['counts']['native_steps'], 38400)
        self.assertEqual(result['counts']['pd_updates'], 19200)
        self.assertEqual(rows, before)
        self.assertFalse(result['global_candidate_accepted'])
        json.dumps(result, allow_nan=False)

    def test_max_within_reset_before_mean_and_other_fragments_do_not_enter_Q(self):
        rows = screen_fixture()
        for row in rows:
            if row['arm'] == 'steady':
                row['yaw_mae'] = (0. if row['tick'] == 175 else .08) if row['tick'] in (175, 325) else 999.
        result = screen_compare(rows)
        self.assertFalse(result['passed'])
        self.assertTrue(all(p['Q_steady'] == .08 for p in result['by_model_seed']))
        self.assertTrue(any(b['yaw_mae'] == 999. for p in result['points'] for b in p['branches']))

    def test_no_best_seed_selection_and_strict_equalities(self):
        for arm, value in (('legacy', .05), ('anchor', .06)):
            rows = screen_fixture()
            if arm == 'anchor':
                for row in rows:
                    if row['arm'] == 'legacy':
                        row['yaw_mae'] = .07
            for row in rows:
                if row['model_seed'] == MODEL_SEEDS[0] and row['arm'] == 'steady':
                    row['yaw_mae'] = value
            result = screen_compare(rows)
            with self.subTest(arm=arm):
                self.assertFalse(result['enter_full_evaluation'])
                self.assertEqual([p['passed'] for p in result['by_model_seed']], [False, True, True])
                self.assertIsNone(result['selected_seed'])

    def test_anchor_replay_mismatch_is_failed_complete_screen(self):
        for key, value in (('trace_identity_sha256', 'f' * 64), ('yaw_mae', .061), ('cost', .2)):
            rows = screen_fixture()
            target = next(r for r in rows if r['arm'] == 'anchor' and r['repeat'] == 1)
            target[key] = value
            result = screen_compare(rows)
            with self.subTest(key=key):
                self.assertEqual(result['status'], 'COMPLETE')
                self.assertFalse(result['gates']['all15_anchor_replays_exact'])
                self.assertFalse(result['passed'])
                self.assertEqual(len(result['anchor_replay_failures']), 1)

    def test_safety_at_nonscoring_stop_fragment_still_blocks(self):
        rows = screen_fixture()
        target = next(r for r in rows if r['tick'] == 410 and r['arm'] == 'legacy')
        target['illegal_body_contact'] = True
        result = screen_compare(rows)
        self.assertTrue(result['gates']['all_three_seed_Q_improve'])
        self.assertFalse(result['gates']['all120_safe'])
        self.assertFalse(result['enter_full_evaluation'])

    def test_missing_duplicate_wrong_identity_or_partial_rejected(self):
        mutations = [('missing', None), ('duplicate', None), ('seed', 80101), ('tick', 176),
                     ('status', 'PARTIAL'), ('steps', 31), ('native_steps', 319), ('pd_updates', 159),
                     ('model_seed', MODEL_SEEDS[0]), ('repeat', True), ('trace_identity_sha256', 'bad')]
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

    def test_candidate_repeat_nonfinite_and_unknown_safety_rejected(self):
        for key, value in (('repeat', 1), ('yaw_mae', math.nan), ('cost', math.inf),
                           ('fallen', None), ('yaw_mae', -1.)):
            rows = screen_fixture()
            target = next(r for r in rows if r['arm'] == 'steady')
            target[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen_compare(rows)


if __name__ == '__main__':
    unittest.main()
