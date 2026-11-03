"""Standard-library tests of complete78 receipts and unchanged qualification."""
from __future__ import annotations

import copy
import math
import unittest

from tools75 import compare75 as old
from tools78.compare78 import compare


def evidence():
    def points(role):
        return [{'seed': s, 'tick': t} for s in old.ROLE_SEEDS[role] for t in old.TICKS]

    def ledger(rows):
        return {'status': 'COMPLETE', 'completed_points': 9, 'points': rows}

    def eps(value, *, numeric=False):
        return [{'epsilon': e, 'model_fd': value, **({'numeric_valid': True} if numeric else {})} for e in old.EPSILONS]

    def ref():
        return {'resolved': True, 'eps': [{'epsilon': e, 'central64': -1.} for e in old.EPSILONS]}

    models = []
    for seed in old.MODEL_SEEDS:
        for arm in old.ARMS:
            training = {'status': 'COMPLETE', 'label_rows': 96, 'rows': [
                {'source': source, 'seed': s, 'tick': t, 'epsilon': e, 'predicted_world_vz_jvp': .3,
                 'target_float64': .4, 'target_training_float32': .4, 'normalized_error': .2 if arm == 'control' else .1}
                for source in ('58', '62') for s in old.LABEL_SEEDS for t in old.TICKS for e in old.LABEL_EPSILONS]}
            models.append({'seed': seed, 'arm': arm, 'status': 'COMPLETE',
                'fit': {'status': 'COMPLETE', 'updates': 1000}, 'final_training_world_vz_diagnostics': training,
                'diagnostics': {r: ledger([p | {'status': 'COMPLETE', 'gradient': -1., 'autograd_agrees': True,
                    'direction_identifiable': True} for p in points(r)]) | {'mean_state_group_loss': .25,
                    'mean_state_group_losses': dict.fromkeys(old.STATE_GROUPS, .25)} for r in old.ROLES},
                'first_step': {r: ledger([p | {'status': 'COMPLETE', 'eps': eps(-.9)} for p in points(r)]) for r in old.FIRST_ROLES},
                'world_vz': {r: ledger([p | {'status': 'COMPLETE', 'eps': eps(-.8 if arm == 'control' else -.9, numeric=True)}
                    for p in points(r)]) for r in old.FIRST_ROLES},
                'heads': {r: ledger([p | {'status': 'COMPLETE', 'body': {c: {'eps': eps(-.9, numeric=True)} for c in ('vx', 'vy')},
                    'cost_h1': {'gradient': -1., 'autograd_agrees': True, 'direction_identifiable': True}}
                    for p in points(r)]) for r in old.FIRST_ROLES}})
    costs = {r: [p | {'gradient': -1., 'resolved': True} for p in points(r)] for r in old.COST_ROLES}
    firsts = {r: [p | ref() for p in points(r)] for r in old.FIRST_ROLES}
    heads = {r: [p | {'body': {c: ref() for c in ('vx', 'vy')}, 'cost_h1': {'gradient': -1., 'resolved': True}}
                 for p in points(r)] for r in old.FIRST_ROLES}
    world = copy.deepcopy(firsts)
    summary = old.comparisons(models, costs, firsts, heads, world,
                             eligibility={'pairing_complete': True, 'control_reproduced69_active': True})
    summary['status'] = 'COMPLETE'
    source = {'status': 'COMPLETE', 'verification_after': 'COMPLETE', 'comparison': summary, 'models': models,
              'cost_references': costs, 'first_step_references': firsts, 'head_references': heads, 'world_vz_references': world}
    fused = [{k: copy.deepcopy(v) for k, v in m.items() if k != 'fit'} for m in models if m['arm'] == 'active']
    return fused, source


class ComparisonTests(unittest.TestCase):
    def test_complete_without_new_fit_and_control_cache_unchanged(self):
        models, source = evidence()
        before = copy.deepcopy(source)
        report = compare(models, source)
        self.assertEqual(report['status'], 'COMPLETE')
        self.assertTrue(report['candidate_accepted'])
        self.assertTrue(report['complete_repair_validated'])
        self.assertEqual((report['cost_points'], report['yaw_first_points'], report['body_channels']), (63, 45, 90))
        self.assertEqual((report['new_fit_count'], report['model_updates']), (0, 0))
        self.assertEqual(source, before)

    def test_unknowns_retained_but_do_not_hide_known_cost_failure(self):
        models, source = evidence()
        source['cost_references']['old64'][0]['resolved'] = False
        source['cost_references']['fresh'][8]['resolved'] = False
        source['comparison'] = old.comparisons(source['models'], *(source[key] for key in (
            'cost_references', 'first_step_references', 'head_references', 'world_vz_references')),
            eligibility={'pairing_complete': True, 'control_reproduced69_active': True})
        source['comparison']['status'] = 'COMPLETE'
        report = compare(models, source)
        self.assertEqual(report['candidate_status'], 'REFERENCE_UNRESOLVED')
        self.assertEqual(report['complete_repair_status'], 'REFERENCE_UNRESOLVED')
        self.assertEqual(len(report['unresolved_references']), 2)
        models[0]['diagnostics']['old72']['points'][0]['gradient'] = 1.
        report = compare(models, source)
        self.assertEqual(report['candidate_status'], 'FAILED')
        self.assertEqual(report['complete_repair_status'], 'FAILED')
        self.assertEqual(len(report['paired'][0]['new_resolved_cost_failures']), 1)

    def test_same_sign_below_floor_is_failure_even_when_selfcheck_passes(self):
        models, source = evidence()
        models[0]['first_step']['fresh']['points'][0]['eps'][0]['model_fd'] = -1e-6
        report = compare(models, source)
        self.assertFalse(report['all_six_numeric_valid'])
        self.assertEqual(report['generalization_target_status'], 'FAILED')
        failure = report['paired'][0]['new_yaw_qualification_failures'][0]
        self.assertEqual(failure['lost_qualifications'], ['combined_qualification', 'magnitude'])

    def test_strict_body_guard_has_no_roundoff_allowance(self):
        models, source = evidence()
        for role in old.FIRST_ROLES:
            for point in models[0]['heads'][role]['points']:
                for channel in ('vx', 'vy'):
                    for row in point['body'][channel]['eps']:
                        row['model_fd'] += 1e-12
        pair = compare(models, source)['paired'][0]
        self.assertGreater(pair['body_rmse_ratio_active_over_control'], 1.)
        self.assertLess(pair['body_rmse_ratio_active_over_control'], 1. + 1e-9)
        self.assertFalse(pair['body_rmse_nonincreasing'])
        self.assertIn('body_RMSE_increased', pair['regression_known_failures'])

    def test_control_numeric_failure_blocks_candidate_but_not_active_repair(self):
        models, source = evidence()
        cached = source['comparison']['models'][0]
        cached['numerics_valid'] = cached['fresh_numerics_valid'] = False
        report = compare(models, source)
        self.assertFalse(report['candidate_accepted'])
        self.assertTrue(report['complete_repair_validated'])

    def test_each_role_state_guard_remains_registered(self):
        for role in old.ROLES:
            with self.subTest(role=role):
                models, source = evidence()
                models[0]['diagnostics'][role].update(mean_state_group_loss=.276,
                    mean_state_group_losses=dict.fromkeys(old.STATE_GROUPS, .276))
                self.assertFalse(compare(models, source)['paired'][0]['state_loss'][role]['passed_le1p1'])

    def test_incomplete_changed_or_nonfinite_evidence_is_rejected(self):
        for defect in ('seed', 'point', 'epsilon', 'nan', 'partial', 'fit', 'new_updates', 'partial_cost', 'label', 'identity'):
            with self.subTest(defect=defect):
                models, source = evidence()
                if defect == 'seed':
                    models.pop()
                elif defect == 'point':
                    models[0]['world_vz']['old72']['points'].pop()
                elif defect == 'epsilon':
                    models[0]['heads']['fresh']['points'][0]['body']['vx']['eps'].pop()
                elif defect == 'nan':
                    models[0]['heads']['fresh']['points'][0]['body']['vx']['eps'][0]['model_fd'] = math.nan
                elif defect == 'partial':
                    models[0]['status'] = 'INCOMPLETE'
                elif defect == 'fit':
                    models[0]['fit'] = {'status': 'COMPLETE', 'updates': 1000}
                elif defect == 'new_updates':
                    models[0]['new_updates'] = 1
                elif defect == 'partial_cost':
                    models[0]['diagnostics']['old64']['points'][0]['status'] = 'INCOMPLETE'
                elif defect == 'label':
                    models[0]['final_training_world_vz_diagnostics']['rows'][0]['target_float64'] = .5
                else:
                    models[0]['diagnostics']['fresh']['points'][0]['tick'] = 111
                with self.assertRaises((ValueError, KeyError)):
                    compare(models, source)


if __name__ == '__main__':
    unittest.main()
