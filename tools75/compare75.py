"""Pure receipt comparisons for the registered paired world-vz75 candidate.

No numerical library, model, optimizer or environment is used here. Training
RMSE reduces all 96 saved F32 normalized errors in F64; heldout world-vz RMSE uses
actual-action endpoint responses at all four registered theta perturbations.
Unknown references remain visible and never erase a known failed condition.
"""
from __future__ import annotations

import hashlib
import json
import math

from tools55.probe_local55 import EPSILONS
from tools59.run59 import cost_comparison, first_step_comparison
from tools69.outputs69 import head_comparison
from tools75.outputs75 import world_vz_comparison

MODEL_SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
COST_ROLES = ('old_probe', 'old55_local', 'old56', 'old64', 'old69', 'old72', 'fresh')
ROLES = ('development', *COST_ROLES)
FIRST_ROLES = ('old56', 'old64', 'old69', 'old72', 'fresh')
ROLE_SEEDS = {'development': (53201, 53202, 53203), 'old_probe': (53301, 53302, 53303),
              'old55_local': (55301, 55302, 55303), 'old56': (56301, 56302, 56303),
              'old64': (64301, 64302, 64303), 'old69': (69301, 69302, 69303),
              'old72': (72301, 72302, 72303), 'fresh': (75301, 75302, 75303)}
TICKS = (110, 260, 410)
LABEL_SEEDS = (53103, 53106, 53109, 53112)
LABEL_EPSILONS = (3.125e-6, 6.25e-6, 1.25e-5, 2.5e-5)
STATE_GROUPS = ('position', 'quaternion', 'joint_position', 'base_linear_velocity',
                'base_angular_velocity', 'leg_velocity', 'wheel_velocity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, name, *, nonnegative=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), 'finite numeric ' + name + ' required')
    require(not nonnegative or value >= 0, 'nonnegative ' + name + ' required')
    return float(value)


def ratio(active, control):
    """A zero baseline admits only another exact zero, without dividing by zero."""
    return active / control if control > 0 else 1.0 if active == 0 else None


def _identities(points, role):
    require([(p['seed'], p['tick']) for p in points] ==
            [(seed, tick) for seed in ROLE_SEEDS[role] for tick in TICKS],
            'complete ordered nine-point grid required: ' + role)


def _ledger(ledger, role):
    require(ledger['status'] == 'COMPLETE' and ledger['completed_points'] == 9,
            'complete prediction ledger required: ' + role)
    _identities(ledger['points'], role)


def _boolean(value):
    require(type(value) is bool, 'boolean qualification required')


def _eps(rows):
    require([row['epsilon'] for row in rows] == list(EPSILONS), 'all four theta eps required')


def training_world_vz_rmse(model):
    """All96 world-vz rows from58+62; no yaw-training improvement gate."""
    diagnostics = model['final_training_world_vz_diagnostics']
    require(diagnostics['status'] == 'COMPLETE' and diagnostics['label_rows'] == len(diagnostics['rows']) == 96,
            'complete96-row world-vz training diagnostics required')
    require(all(str(row['source']) in ('58', '62') for row in diagnostics['rows']), 'unexpected training source')
    errors, targets, sources = [], [], {}
    for source in ('58', '62'):
        rows = [row for row in diagnostics['rows'] if str(row['source']) == source]
        require([(r['seed'], r['tick'], r['epsilon']) for r in rows] ==
                [(seed, tick, eps) for seed in LABEL_SEEDS for tick in TICKS for eps in LABEL_EPSILONS],
                'training source must retain all12 centers and4 scales')
        values = [finite(row['normalized_error'], 'training world-vz normalized error') for row in rows]
        for row in rows:
            finite(row['predicted_world_vz_jvp'], 'training world-vz prediction')
            targets.append([source, row['seed'], row['tick'], row['epsilon'],
                            finite(row['target_float64'], 'training F64 target'),
                            finite(row['target_training_float32'], 'training F32 target')])
        mse = math.fsum(value * value for value in values) / 48
        finite(mse, 'training MSE', nonnegative=True)
        sources[source] = {'rows': 48, 'normalized_mse': mse, 'normalized_rmse': math.sqrt(mse)}
        errors.extend(values)
    mse = math.fsum(value * value for value in errors) / 96
    return {'rows': 96, 'independent_states': 24, 'normalized_mse': mse,
            'normalized_rmse': math.sqrt(mse), 'by_source': sources,
            'reduction': 'all96 saved F32 normalized errors, squared and reduced in F64',
            'coordinate': 'absolute recorded training action perturbation; not heldout policy theta',
            'target_identity_sha256': hashlib.sha256(json.dumps(targets, separators=(',', ':')).encode()).hexdigest()}


def _validate_model(model):
    require(model['status'] == model['fit']['status'] == 'COMPLETE'
            and model['fit']['updates'] == 1000, 'complete1000-update model required')
    require(set(model['diagnostics']) == set(ROLES)
            and set(model['first_step']) == set(model['heads']) == set(model['world_vz']) == set(FIRST_ROLES),
            'all registered model roles required')
    for role in ROLES:
        diagnostic = model['diagnostics'][role]
        _ledger(diagnostic, role)
        groups = diagnostic['mean_state_group_losses']
        require(set(groups) == set(STATE_GROUPS), 'all seven state groups required')
        values = [finite(groups[k], 'state group', nonnegative=True) for k in STATE_GROUPS]
        mean = finite(diagnostic['mean_state_group_loss'], 'state group mean', nonnegative=True)
        require(math.isclose(mean, math.fsum(values) / 7, rel_tol=1e-10, abs_tol=1e-12),
                'state mean must equally weight all seven groups')
        for point in diagnostic['points']:
            finite(point['gradient'], 'cost gradient')
            _boolean(point['autograd_agrees'])
            _boolean(point['direction_identifiable'])
    for role in FIRST_ROLES:
        _ledger(model['first_step'][role], role)
        _ledger(model['heads'][role], role)
        _ledger(model['world_vz'][role], role)
        for point in model['world_vz'][role]['points']:
            require(point['status'] == 'COMPLETE', 'complete world-vz point required')
            _eps(point['eps'])
            for e in point['eps']:
                finite(e['model_fd'], 'world-vz FD')
                _boolean(e['numeric_valid'])
        for point in model['first_step'][role]['points']:
            require(point['status'] == 'COMPLETE', 'complete first-step point required')
            _eps(point['eps'])
            for e in point['eps']:
                finite(e['model_fd'], 'first-step yaw FD')
        for point in model['heads'][role]['points']:
            require(point['status'] == 'COMPLETE', 'complete head point required')
            finite(point['cost_h1']['gradient'], 'H1 cost gradient')
            _boolean(point['cost_h1']['autograd_agrees'])
            _boolean(point['cost_h1']['direction_identifiable'])
            for channel in ('vx', 'vy'):
                _eps(point['body'][channel]['eps'])
                for e in point['body'][channel]['eps']:
                    finite(e['model_fd'], 'body FD')
                    _boolean(e['numeric_valid'])


def _references(costs, firsts, heads, world_vz):
    require(set(costs) == set(COST_ROLES) and set(firsts) == set(heads) == set(world_vz) == set(FIRST_ROLES),
            'registered63/45/90/45 reference roles required')
    unknown = []
    for role in COST_ROLES:
        _identities(costs[role], role)
        for point in costs[role]:
            finite(point['gradient'], 'physical cost gradient')
            _boolean(point['resolved'])
            if not point['resolved']:
                unknown.append({'role': role, 'kind': 'cost_h32', 'seed': point['seed'], 'tick': point['tick']})
    for role in FIRST_ROLES:
        _identities(firsts[role], role)
        _identities(heads[role], role)
        _identities(world_vz[role], role)
        refs = [(p, 'yaw_first', p) for p in firsts[role]]
        refs.extend((p, 'world_vz', p) for p in world_vz[role])
        for point in heads[role]:
            refs.append((point['cost_h1'], 'cost_h1', point))
            refs.extend((point['body'][c], 'body_' + c, point) for c in ('vx', 'vy'))
        for ref, kind, point in refs:
            _boolean(ref['resolved'])
            if kind == 'cost_h1':
                finite(ref['gradient'], 'physical H1 gradient')
            else:
                _eps(ref['eps'])
                for e in ref['eps']:
                    finite(e['central64'], 'physical endpoint FD')
            if not ref['resolved']:
                unknown.append({'role': role, 'kind': kind, 'seed': point['seed'], 'tick': point['tick']})
    return unknown


def _new_cost_failures(control, active):
    failures = []
    for role in COST_ROLES:
        for left, right in zip(control['cost'][role]['points'], active['cost'][role]['points'], strict=True):
            if left['valid_match'] is True and right['valid_match'] is not True:
                failures.append({'role': role, 'horizon': 32, 'seed': left['seed'], 'tick': left['tick']})
    for role in FIRST_ROLES:
        for left, right in zip(control['heads'][role]['points'], active['heads'][role]['points'], strict=True):
            if left['cost_h1']['valid_match'] is True and right['cost_h1']['valid_match'] is not True:
                failures.append({'role': role, 'horizon': 1, 'seed': left['seed'], 'tick': left['tick']})
    return failures


def _new_response_failures(control, active, kind):
    failures = []
    for role in FIRST_ROLES:
        key = 'first_step' if kind == 'yaw' else 'world_vz' if kind == 'world_vz' else 'heads'
        for left, right in zip(control[key][role]['points'], active[key][role]['points'], strict=True):
            channels = ((kind, left, right),) if kind in ('yaw', 'world_vz') else tuple(
                (c, left['body'][c], right['body'][c]) for c in ('vx', 'vy'))
            for channel, lrow, rrow in channels:
                for le, re in zip(lrow['eps'], rrow['eps'], strict=True):
                    lost = []
                    if le['valid_match'] is True and re['valid_match'] is not True:
                        lost.append('combined_qualification')
                    if abs(le['model_fd']) >= 1e-5 and abs(re['model_fd']) < 1e-5:
                        lost.append('magnitude')
                    if kind in ('body', 'world_vz') and le['numeric_valid'] and not re['numeric_valid']:
                        lost.append('numeric')
                    if lost:
                        failures.append({'role': role, 'seed': left['seed'], 'tick': left['tick'],
                                         'channel': channel, 'epsilon': le['epsilon'], 'lost_qualifications': lost})
    return failures


def _decision(known_failures, unknown):
    return 'FAILED' if known_failures else 'REFERENCE_UNRESOLVED' if unknown else 'PASSED'


def comparisons(models, cost_references, first_refs, head_refs, world_vz_refs, *, eligibility=None):
    """Compare complete paired receipts; final runner audit can still revoke them."""
    require(len(models) == 6 and {(m['seed'], m['arm']) for m in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'all six paired models required')
    unknown = _references(cost_references, first_refs, head_refs, world_vz_refs)
    result = {'models': [], 'paired': [], 'selected_seed': None, 'selected_control_parameter': None,
              'cost_points': 63, 'yaw_first_points': 45, 'body_channels': 90, 'cost_h1_points': 45, 'world_vz_points': 45,
              'unresolved_references': unknown, 'prior_unresolved_references': [p for p in unknown if p['role'] != 'fresh'],
              'all_physical_references_resolved': not unknown,
              'prior_reference_status': 'REFERENCE_UNRESOLVED' if any(p['role'] != 'fresh' for p in unknown) else 'RESOLVED',
              'old53_reference_scope': 'historical wide-epsilon regression; not an infinitesimal derivative'}
    by_key, raw = {}, {(m['seed'], m['arm']): m for m in models}
    for model in models:
        _validate_model(model)
        checks = all(p['autograd_agrees'] for role in ROLES for p in model['diagnostics'][role]['points'])
        directions = all(p['direction_identifiable'] for role in ROLES for p in model['diagnostics'][role]['points'])
        heads = {role: head_comparison(model['heads'][role]['points'], head_refs[role]) for role in FIRST_ROLES}
        firsts = {role: first_step_comparison(model['first_step'][role]['points'], first_refs[role]) for role in FIRST_ROLES}
        world = {role: world_vz_comparison(model['world_vz'][role]['points'], world_vz_refs[role]) for role in FIRST_ROLES}
        yaw_magnitude = all(e['model_magnitude_admitted'] for value in firsts.values()
                            for p in value['points'] for e in p['eps'])
        body_magnitude = {role: all(abs(e['model_fd']) >= 1e-5 for p in heads[role]['points']
                                   for c in ('vx', 'vy') for e in p['body'][c]['eps']) for role in FIRST_ROLES}
        row = {'seed': model['seed'], 'arm': model['arm'], 'all_selfchecks': checks,
               'all_model_directions_identifiable': directions, 'all_yaw_endpoint_magnitudes_admitted': yaw_magnitude,
               'numerics_valid': checks and directions and yaw_magnitude and all(body_magnitude.values())
                   and all(v['numeric_valid'] for v in heads.values()) and all(v['numeric_valid'] for v in world.values()),
               'cost': {role: cost_comparison(model['diagnostics'][role]['points'], cost_references[role]) for role in COST_ROLES},
               'first_step': firsts, 'heads': heads, 'world_vz': world, 'training_world_vz': training_world_vz_rmse(model)}
        row['cost_valid_matches'] = sum(v['valid_matches'] for v in row['cost'].values())
        row['first_step_valid_matches'] = sum(v['valid_matches'] for v in firsts.values())
        row['body_valid_matches'] = sum(v['body_valid_matches'] for v in heads.values())
        row['cost_h1_valid_matches'] = sum(v['cost_h1_valid_matches'] for v in heads.values())
        row['world_vz_valid_matches'] = sum(v['valid_matches'] for v in world.values())
        row['body_normalized_mse'] = math.fsum(v['body_normalized_mse'] for v in heads.values()) / 5
        row['body_normalized_rmse'] = math.sqrt(row['body_normalized_mse'])
        row['body_error_scalar_count'] = 360
        row['fresh_world_vz_error'] = {key: world['fresh'][key] for key in ('normalized_mse', 'normalized_rmse', 'error_scalar_count', 'scale')}
        row['fresh_numerics_valid'] = heads['fresh']['numeric_valid'] and body_magnitude['fresh'] and all(
            p['autograd_agrees'] and p['direction_identifiable'] for p in model['diagnostics']['fresh']['points']) and all(
            e['model_magnitude_admitted'] for p in firsts['fresh']['points'] for e in p['eps']) and world['fresh']['numeric_valid']
        row['repair_gate'] = row['numerics_valid'] and all(v['passed_9_of_9'] for v in row['cost'].values()) and all(
            v['passed_9_of_9'] for v in firsts.values()) and all(
            v['body_passed_18_of18'] and v['cost_h1_passed_9_of9'] for v in heads.values()) and all(
            v['passed_9_of_9'] for v in world.values())
        row['known_repair_failures'] = []
        if not row['numerics_valid']:
            row['known_repair_failures'].append('model_numeric_or_magnitude_qualification')
        for role in COST_ROLES:
            if any(p['valid_match'] is False for p in row['cost'][role]['points']):
                row['known_repair_failures'].append('cost_h32:' + role)
        for role in FIRST_ROLES:
            if any(p['passed'] is False for p in firsts[role]['points']):
                row['known_repair_failures'].append('yaw_first:' + role)
            if any(p['cost_h1']['valid_match'] is False for p in heads[role]['points']):
                row['known_repair_failures'].append('cost_h1:' + role)
            if any(p['body'][c]['passed'] is False for p in heads[role]['points'] for c in ('vx', 'vy')):
                row['known_repair_failures'].append('body:' + role)
            if any(p['passed'] is False for p in world[role]['points']):
                row['known_repair_failures'].append('world_vz:' + role)
        by_key[(row['seed'], row['arm'])] = row
        result['models'].append(row)
    for seed in MODEL_SEEDS:
        left, right = [by_key[(seed, arm)] for arm in ARMS]
        require(left['training_world_vz']['target_identity_sha256'] == right['training_world_vz']['target_identity_sha256'],
                'paired training targets differ')
        training = right['training_world_vz']['normalized_rmse'] < left['training_world_vz']['normalized_rmse']
        vz_control, vz_active = [row['fresh_world_vz_error']['normalized_rmse'] for row in (left, right)]
        gain = vz_active <= .8 * vz_control
        fresh_unknown = [p for p in unknown if p['role'] == 'fresh' and p['kind'] == 'world_vz']
        target_failures = []
        if not gain and not fresh_unknown:
            target_failures.append('fresh_world_vz_RMSE_reduction_below20pct')
        if not (left['fresh_numerics_valid'] and right['fresh_numerics_valid']):
            target_failures.append('fresh_numeric_or_magnitude_qualification')
        if any(p['passed'] is False for p in right['world_vz']['fresh']['points']):
            target_failures.append('fresh_active_world_vz_direction')
        target_status = _decision(target_failures, fresh_unknown)
        cost_failures = _new_cost_failures(left, right)
        yaw_failures = _new_response_failures(left, right, 'yaw')
        body_failures = _new_response_failures(left, right, 'body')
        world_failures = _new_response_failures(left, right, 'world_vz')
        body_nonincrease = right['body_normalized_rmse'] <= left['body_normalized_rmse']
        body_resolved = all(v['body_resolved_count'] == 18 for v in right['heads'].values())
        states = {}
        for role in ROLES:
            ld, rd = [raw[(seed, arm)]['diagnostics'][role] for arm in ARMS]
            lv, rv = ld['mean_state_group_loss'], rd['mean_state_group_loss']
            states[role] = {'control': lv, 'active': rv, 'ratio_active_over_control': ratio(rv, lv),
                            'passed_le1p1': rv <= 1.1 * lv, 'control_groups': ld['mean_state_group_losses'],
                            'active_groups': rd['mean_state_group_losses']}
        regression_failures = [name for name, failed in (
            ('new_resolved_H1_H32_failure', bool(cost_failures)), ('new_yaw_qualification_failure', bool(yaw_failures)),
            ('new_body_qualification_failure', bool(body_failures)), ('body_RMSE_increased', body_resolved and not body_nonincrease),
            ('new_world_vz_qualification_failure', bool(world_failures)),
            ('state_error_guard', not all(item['passed_le1p1'] for item in states.values()))) if failed]
        regression_status = _decision(regression_failures, not body_resolved)
        result['paired'].append({'seed': seed, 'training_improved': training,
            'control_training_world_vz_rmse': left['training_world_vz']['normalized_rmse'],
            'active_training_world_vz_rmse': right['training_world_vz']['normalized_rmse'],
            'raw_fresh_world_vz_rmse_gain': gain, 'control_fresh_world_vz_rmse': vz_control, 'active_fresh_world_vz_rmse': vz_active,
            'fresh_world_vz_rmse_ratio_active_over_control': ratio(vz_active, vz_control),
            'generalization_target_status': target_status, 'generalization_target_known_failures': target_failures,
            'generalization_target_passed': target_status == 'PASSED', 'state_loss': states,
            'new_resolved_cost_failures': cost_failures, 'new_yaw_qualification_failures': yaw_failures,
            'new_body_qualification_failures': body_failures, 'control_body_normalized_rmse': left['body_normalized_rmse'],
            'new_world_vz_qualification_failures': world_failures,
            'active_body_normalized_rmse': right['body_normalized_rmse'], 'body_rmse_nonincreasing': body_nonincrease,
            'body_rmse_references_resolved': body_resolved,
            'body_rmse_ratio_active_over_control': ratio(right['body_normalized_rmse'], left['body_normalized_rmse']),
            'regression_known_failures': regression_failures, 'no_regressions_status': regression_status,
            'no_regressions': regression_status == 'PASSED'})
    supplied = eligibility or {}
    require(set(supplied) <= {'pairing_complete', 'control_reproduced69_active'}, 'unknown eligibility field')
    for value in supplied.values():
        _boolean(value)
    eligible = all(supplied.get(k) is True for k in ('pairing_complete', 'control_reproduced69_active'))
    all_numeric = all(row['numerics_valid'] for row in result['models'])
    training = all(pair['training_improved'] for pair in result['paired'])
    target_failures = [{'seed': p['seed'], 'reason': reason} for p in result['paired']
                       for reason in p['generalization_target_known_failures']]
    target_unknown = any(p['generalization_target_status'] == 'REFERENCE_UNRESOLVED' for p in result['paired'])
    target_status = _decision(target_failures, target_unknown)
    regression_free = all(pair['no_regressions'] for pair in result['paired'])
    regression_failed = any(pair['regression_known_failures'] for pair in result['paired'])
    failures = ([name for name, passed in (('training_world_vz_RMSE_not_strictly_down_all3', training),
                 ('regression_or_state_guard', not regression_failed), ('all_six_numeric_or_magnitude_qualification', all_numeric),
                 ('pairing_or_control_reproduction', eligible)) if not passed]
                + (['fresh_generalization_target'] if target_failures else []))
    candidate_status = _decision(failures, unknown or target_unknown)
    active_failures = [{'seed': row['seed'], 'reason': reason} for row in result['models'] if row['arm'] == 'active'
                       for reason in row['known_repair_failures']]
    repair_status = _decision(active_failures, unknown)
    result.update(eligibility=supplied, pairing_and_reproduction_eligible=eligible,
        training_improved=training, raw_fresh_world_vz_rmse_gain=all(p['raw_fresh_world_vz_rmse_gain'] for p in result['paired']),
        generalization_target_status=target_status, generalization_target_known_failures=target_failures,
        generalization_target_passed=target_status == 'PASSED', no_regressions=regression_free,
        no_regressions_status=_decision(regression_failed, not regression_free),
        all_six_numeric_valid=all_numeric, candidate_known_failures=failures,
        candidate_status=candidate_status, candidate_accepted=candidate_status == 'PASSED',
        complete_repair_status=repair_status, complete_repair_known_failures=active_failures,
        complete_repair_validated=repair_status == 'PASSED')
    return result
