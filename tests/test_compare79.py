"""Synthetic stdlib79 decision tests, with no actor, model or environment."""
import copy
import json
import math
import unittest

from tools79.compare79 import ARMS, MODEL_SEEDS, RESET_SEEDS, TASKS, compare


def episode(task, seed, j=1., wz=.04):
    turn = task == 'turn_stop'
    return {'sequence': task, 'reset_seed': seed, 'delay_ticks': 0, 'delay_ms': 0,
        'policy_transitions': 600, 'native_steps': 6000, 'done_reason': 'HORIZON_12S', 'status': 'COMPLETE',
        'failure': False, 'J': j, 'J_components': {'full_horizon_vx_mae_over_0_2': j,
            'full_horizon_wz_mae_over_0_2': 0., 'stop_8_to_12_horizontal_norm_mean_over_0_2': 0., 'fixed_early_failure_penalty': 0.},
        'actual_counts': {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000},
        'stable_vx_mae_m_s': None if turn else .03, 'stable_wz_mae_rad_s': wz if turn else None,
        'stop_last_second_horizontal_speed_m_s': .01, 'stop_last_second_abs_wz_rad_s': .02,
        'stop_displacement_m': .02, 'peak_abs_roll_or_pitch_rad': .1,
        'threshold_flags': {'completed_12s': True, 'no_fall_or_illegal_support': True, 'stable_vx': None if turn else True,
            'stable_wz': True if turn else None, 'stop_horizontal_speed': True, 'stop_abs_wz': True,
            'stop_displacement': True, 'peak_roll_pitch': True},
        'reference_thresholds_met': True, 'nominal_zero_delay_pass': True}


def fixture():
    baseline = [{'task': t, 'seed': s, 'metrics': episode(t, s)} for t in TASKS for s in RESET_SEEDS]
    candidates = [{'model_seed': m, 'arm': a, 'task': t, 'seed': s,
                   'metrics': episode(t, s, 1. if a == 'frozen' else .98, .0375 if a == 'frozen' else .036)}
                  for m in MODEL_SEEDS for a in ARMS for t in TASKS for s in RESET_SEEDS]
    return baseline, candidates


def set_j(row, value):
    row['metrics']['J'] = value
    row['metrics']['J_components']['full_horizon_vx_mae_over_0_2'] = value


def fail_threshold(row):
    row['metrics']['stop_last_second_horizontal_speed_m_s'] = .021
    row['metrics']['threshold_flags']['stop_horizontal_speed'] = False
    row['metrics']['reference_thresholds_met'] = False
    row['metrics']['nominal_zero_delay_pass'] = False


class Compare79Tests(unittest.TestCase):
    def test_complete_feedback_and_control_are_separate_from_global_acceptance(self):
        baseline, candidates = fixture(); before = copy.deepcopy((baseline, candidates))
        result = compare(baseline, candidates)
        self.assertTrue(result['feedback']['passed'])
        self.assertTrue(all(x['passed'] for x in result['control_gates']))
        self.assertFalse(result['global_accepted']); self.assertFalse(result['candidate_accepted'])
        self.assertEqual(result['complete_model_eligibility'], 'NOT_REQUALIFIED')
        self.assertEqual((baseline, candidates), before)
        json.dumps(result, allow_nan=False)

    def test_shared_baselines_are_counted_once_and_not_independent_learners(self):
        result = compare(*fixture()); unit = result['sampling_unit']
        self.assertEqual((unit['paired_model_seed_units'], unit['paired_observations'], unit['unique_baseline_episodes']), (3, 27, 9))
        self.assertEqual(len(result['paired_observations']), 27)
        self.assertEqual(len({k for p in result['control_gates'] for k in p['baseline_keys']}), 9)
        self.assertEqual(result['evaluation_counts']['baseline']['policy_transitions'], 5400)
        self.assertEqual(result['evaluation_counts']['candidate']['policy_transitions'], 32400)
        self.assertEqual(result['evaluation_counts']['total'], {'policy_transitions': 37800, 'native_steps': 378000, 'pd_updates': 189000})

    def test_incomplete_duplicate_or_wrong_seed_grids_are_rejected(self):
        for change in ('missing_baseline', 'missing_candidate', 'duplicate_baseline', 'duplicate_candidate', 'wrong_seed', 'baseline_repeated_six_times'):
            with self.subTest(change=change):
                b, c = fixture()
                if change == 'missing_baseline': b.pop()
                elif change == 'missing_candidate': c.pop()
                elif change == 'duplicate_baseline': b[-1] = copy.deepcopy(b[0])
                elif change == 'duplicate_candidate': c[-1] = copy.deepcopy(c[0])
                elif change == 'wrong_seed': c[0]['model_seed'] = 64004
                else: b *= 6
                with self.assertRaises(ValueError): compare(b, c)

    def test_partial_counts_identity_unknown_and_nonfinite_are_rejected(self):
        mutations = [('policy_transitions', 599), ('native_steps', 5999), ('done_reason', 'TERMINATED'),
                     ('reset_seed', 79404), ('delay_ticks', 1), ('J', math.nan), ('reference_thresholds_met', None),
                     ('stable_vx_mae_m_s', None)]
        for key, value in mutations:
            b, c = fixture(); c[0]['metrics'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): compare(b, c)
        b, c = fixture(); c[0]['metrics']['actual_counts']['pd_updates'] = 2999
        with self.assertRaises(ValueError): compare(b, c)
        b, c = fixture(); c[0]['metrics']['reset_info'] = {'nested': [math.inf]}
        with self.assertRaises(ValueError): compare(b, c)

    def test_complete_threshold_failure_is_failed_observation_not_partial(self):
        b, c = fixture(); row = next(x for x in c if x['arm'] == 'refresh'); fail_threshold(row)
        result = compare(b, c)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertFalse(result['feedback']['gates']['no_new_original_task_failure'])
        self.assertFalse(next(x for x in result['control_gates'] if x['model_seed'] == row['model_seed'] and x['arm'] == 'refresh')['passed'])

    def test_baseline_failure_does_not_change_feedback_question(self):
        b, c = fixture(); fail_threshold(b[0]); result = compare(b, c)
        self.assertTrue(result['feedback']['passed'])
        self.assertTrue(all(not x['passed'] for x in result['control_gates']))

    def test_each_model_seed_must_strictly_improve_even_if_overall_does(self):
        b, c = fixture()
        for row in c:
            if row['model_seed'] == MODEL_SEEDS[0] and row['arm'] == 'refresh': set_j(row, 1.)
        result = compare(b, c)
        self.assertLess(result['feedback']['overall_J_ratio'], .99)
        self.assertFalse(result['feedback']['gates']['all_three_seed_means_strictly_improve'])

    def test_feedback_overall_ratio_boundary(self):
        for value, passed in ((.99, True), (.990001, False)):
            b, c = fixture()
            for row in c:
                if row['arm'] == 'refresh': set_j(row, value)
            with self.subTest(value=value): self.assertEqual(compare(b, c)['feedback']['passed'], passed)

    def test_feedback_pair_cap_boundary_is_independent_of_good_means(self):
        for value, passed in ((1.05, True), (math.nextafter(1.05, math.inf), False)):
            b, c = fixture()
            selected = [x for x in c if x['model_seed'] == MODEL_SEEDS[0] and x['arm'] == 'refresh']
            for row in selected: set_j(row, .95)
            set_j(selected[0], value)
            with self.subTest(value=value):
                result = compare(b, c)
                self.assertTrue(result['feedback']['gates']['equal_seed_overall_ratio_le0p99'])
                self.assertEqual(result['feedback']['gates']['all27_individual_J_pairs_le1p05'], passed)

    def test_original_control_J_mean_and_pair_boundaries(self):
        b, c = fixture(); selected = [x for x in c if x['model_seed'] == MODEL_SEEDS[0] and x['arm'] == 'frozen']
        set_j(selected[0], 1.05); set_j(selected[1], .95)
        gate = compare(b, c)['control_gates'][0]
        self.assertTrue(gate['gates']['J_mean_ratio_le1']); self.assertTrue(gate['gates']['all_nine_J_pairs_le1p05'])
        set_j(selected[0], math.nextafter(1.05, math.inf))
        self.assertFalse(compare(b, c)['control_gates'][0]['gates']['all_nine_J_pairs_le1p05'])
        for row in selected: set_j(row, 1.001)
        self.assertFalse(compare(b, c)['control_gates'][0]['gates']['J_mean_ratio_le1'])

    def test_turn_mean_boundary_and_individual_strict_requirement(self):
        for value, passed in ((.038, True), (.0380001, False)):
            b, c = fixture()
            selected = [x for x in c if x['model_seed'] == MODEL_SEEDS[0] and x['arm'] == 'frozen' and x['task'] == 'turn_stop']
            for row in selected: row['metrics']['stable_wz_mae_rad_s'] = value
            with self.subTest(value=value): self.assertEqual(compare(b, c)['control_gates'][0]['gates']['turn_mean_ratio_le0p95'], passed)
        b, c = fixture(); selected = [x for x in c if x['model_seed'] == MODEL_SEEDS[0] and x['arm'] == 'frozen' and x['task'] == 'turn_stop']
        for row in selected: row['metrics']['stable_wz_mae_rad_s'] = .03
        selected[0]['metrics']['stable_wz_mae_rad_s'] = .04
        gate = compare(b, c)['control_gates'][0]
        self.assertTrue(gate['gates']['turn_mean_ratio_le0p95']); self.assertFalse(gate['gates']['all_three_turn_resets_strictly_improve'])

    def test_zero_denominators_are_unknown_ratios_and_never_pass(self):
        b, c = fixture()
        for row in b + c: set_j(row, 0.)
        result = compare(b, c)
        self.assertIsNone(result['feedback']['overall_J_ratio']); self.assertFalse(result['feedback']['passed'])
        self.assertTrue(all(x['J_ratio'] is None and not x['passed'] for x in result['control_gates']))
        b, c = fixture()
        for row in b + c:
            if row['task'] == 'turn_stop': row['metrics']['stable_wz_mae_rad_s'] = 0.
        self.assertTrue(all(x['yaw_mae_ratio'] is None and not x['passed'] for x in compare(b, c)['control_gates']))

    def test_feedback_uses_ratio_of_equal_seed_means_not_mean_of_ratios(self):
        b, c = fixture()
        for row in c:
            seed = row['model_seed']; frozen = 1000. if seed == MODEL_SEEDS[0] else 1.
            set_j(row, frozen * (1. if row['arm'] == 'frozen' else .995 if seed == MODEL_SEEDS[0] else .5))
        result = compare(b, c)['feedback']
        self.assertGreater(result['overall_J_ratio'], .99)
        self.assertLess(sum(x['J_ratio_refresh_over_frozen'] for x in result['by_model_seed']) / 3, .99)
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
