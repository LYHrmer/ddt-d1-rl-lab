"""Receipt-only78 comparison against cached75 controls; no fits are invented."""
from __future__ import annotations

import math

from tools75 import compare75 as old

MODEL_SEEDS, ROLES, FIRST_ROLES = old.MODEL_SEEDS, old.ROLES, old.FIRST_ROLES


def _finite_tree(value):
    if isinstance(value, float):
        old.require(math.isfinite(value), 'nonfinite receipt value')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_tree(item)


def _states(model):
    old.require(set(model['diagnostics']) == set(ROLES), 'all eight diagnostic roles required')
    for role in ROLES:
        row = model['diagnostics'][role]
        old._ledger(row, role)
        groups = row['mean_state_group_losses']
        old.require(set(groups) == set(old.STATE_GROUPS), 'all seven state groups required')
        values = [old.finite(groups[g], 'state group', nonnegative=True) for g in old.STATE_GROUPS]
        mean = old.finite(row['mean_state_group_loss'], 'state mean', nonnegative=True)
        old.require(math.isclose(mean, math.fsum(values) / 7, rel_tol=1e-10, abs_tol=1e-12), 'state mean differs')
        for point in row['points']:
            old.require(point['status'] == 'COMPLETE', 'partial diagnostic point')
            old.finite(point['gradient'], 'cost gradient')
            old._boolean(point['autograd_agrees'])
            old._boolean(point['direction_identifiable'])


def _summary(model, costs, firsts, heads, world):
    _finite_tree(model)
    old.require(model['status'] == 'COMPLETE' and model.get('arm', 'active') == 'active', 'complete fused model required')
    old.require('fit' not in model and model.get('updates', 0) == model.get('new_updates', 0) == 0,
                '78 must not invent a fit or updates')
    _states(model)
    for key in ('first_step', 'heads', 'world_vz'):
        old.require(set(model[key]) == set(FIRST_ROLES), 'all five first-step roles required')
        for role in FIRST_ROLES:
            old._ledger(model[key][role], role)
            for point in model[key][role]['points']:
                old.require(point['status'] == 'COMPLETE', 'partial response point')
                channels = [point['body'][c] for c in ('vx', 'vy')] if key == 'heads' else [point]
                for channel in channels:
                    old._eps(channel['eps'])
                    for epsilon in channel['eps']:
                        old.finite(epsilon['model_fd'], 'response FD')
                        if key != 'first_step':
                            old._boolean(epsilon['numeric_valid'])
                if key == 'heads':
                    old.finite(point['cost_h1']['gradient'], 'H1 gradient')
                    old._boolean(point['cost_h1']['autograd_agrees'])
                    old._boolean(point['cost_h1']['direction_identifiable'])
    hs = {r: old.head_comparison(model['heads'][r]['points'], heads[r]) for r in FIRST_ROLES}
    fs = {r: old.first_step_comparison(model['first_step'][r]['points'], firsts[r]) for r in FIRST_ROLES}
    ws = {r: old.world_vz_comparison(model['world_vz'][r]['points'], world[r]) for r in FIRST_ROLES}
    cs = {r: old.cost_comparison(model['diagnostics'][r]['points'], costs[r]) for r in old.COST_ROLES}
    checks = all(p['autograd_agrees'] for r in ROLES for p in model['diagnostics'][r]['points'])
    directions = all(p['direction_identifiable'] for r in ROLES for p in model['diagnostics'][r]['points'])
    yaw = {r: all(e['model_magnitude_admitted'] for p in fs[r]['points'] for e in p['eps']) for r in FIRST_ROLES}
    body = {r: all(abs(e['model_fd']) >= 1e-5 for p in hs[r]['points'] for c in ('vx', 'vy')
                   for e in p['body'][c]['eps']) for r in FIRST_ROLES}
    numeric = checks and directions and all(yaw.values()) and all(body.values()) and all(
        hs[r]['numeric_valid'] and ws[r]['numeric_valid'] for r in FIRST_ROLES)
    failures = [] if numeric else ['model_numeric_or_magnitude_qualification']
    failures.extend('cost_h32:' + r for r in old.COST_ROLES if any(p['valid_match'] is False for p in cs[r]['points']))
    for role in FIRST_ROLES:
        for name, failed in (('yaw_first', any(p['passed'] is False for p in fs[role]['points'])),
                ('world_vz', any(p['passed'] is False for p in ws[role]['points'])),
                ('cost_h1', any(p['cost_h1']['valid_match'] is False for p in hs[role]['points'])),
                ('body', any(p['body'][c]['passed'] is False for p in hs[role]['points'] for c in ('vx', 'vy')))):
            if failed:
                failures.append(name + ':' + role)
    mse = math.fsum(hs[r]['body_normalized_mse'] for r in FIRST_ROLES) / 5
    return {'seed': model['seed'], 'arm': 'active', 'cost': cs, 'heads': hs, 'first_step': fs, 'world_vz': ws,
        'all_selfchecks': checks, 'all_model_directions_identifiable': directions,
        'all_yaw_endpoint_magnitudes_admitted': all(yaw.values()), 'numerics_valid': numeric,
        'training_world_vz': old.training_world_vz_rmse(model), 'body_normalized_mse': mse,
        'body_normalized_rmse': math.sqrt(mse), 'body_error_scalar_count': 360,
        'cost_valid_matches': sum(v['valid_matches'] for v in cs.values()),
        'first_step_valid_matches': sum(v['valid_matches'] for v in fs.values()),
        'body_valid_matches': sum(v['body_valid_matches'] for v in hs.values()),
        'cost_h1_valid_matches': sum(v['cost_h1_valid_matches'] for v in hs.values()),
        'world_vz_valid_matches': sum(v['valid_matches'] for v in ws.values()),
        'fresh_world_vz_error': {k: ws['fresh'][k] for k in ('normalized_mse', 'normalized_rmse', 'error_scalar_count', 'scale')},
        'fresh_numerics_valid': hs['fresh']['numeric_valid'] and body['fresh'] and yaw['fresh']
            and ws['fresh']['numeric_valid'] and all(p['autograd_agrees'] and p['direction_identifiable']
            for p in model['diagnostics']['fresh']['points']), 'known_repair_failures': failures,
        'repair_gate': numeric and all(v['passed_9_of_9'] for v in (*cs.values(), *fs.values(), *ws.values()))
            and all(v['body_passed_18_of18'] and v['cost_h1_passed_9_of9'] for v in hs.values())}


def compare(models, source75):
    """Return COMPLETE only for all three complete fused ledgers; errors propagate."""
    old.require(source75['status'] == source75['verification_after'] == source75['comparison']['status'] == 'COMPLETE',
                'complete verified75 source required')
    old.require(source75['comparison']['pairing_and_reproduction_eligible'] is True, '75 pairing identity required')
    old.require(len(models) == 3 and {m['seed'] for m in models} == set(MODEL_SEEDS), 'all three fused seeds required')
    refs = [source75[k] for k in ('cost_references', 'first_step_references', 'head_references', 'world_vz_references')]
    unknown = old._references(*refs)
    controls = [m for m in source75['comparison']['models'] if m['arm'] == 'control']
    raw_controls = [m for m in source75['models'] if m['arm'] == 'control']
    old.require(len(controls) == len(raw_controls) == 3 and {m['seed'] for m in controls} ==
                {m['seed'] for m in raw_controls} == set(MODEL_SEEDS), 'cached controls incomplete')
    _finite_tree(controls)
    cached, raw = {m['seed']: m for m in controls}, {m['seed']: m for m in raw_controls}
    result = {'models': [], 'paired': [], 'selected_seed': None, 'selected_control_parameter': None,
        'cost_points': 63, 'yaw_first_points': 45, 'body_channels': 90, 'cost_h1_points': 45, 'world_vz_points': 45,
        'unresolved_references': unknown, 'prior_unresolved_references': [p for p in unknown if p['role'] != 'fresh'],
        'all_physical_references_resolved': not unknown, 'new_fit_count': 0, 'model_updates': 0,
        'scope': 'retrospective inference fusion; cached75 controls; no new training or physical evidence'}
    for model in sorted(models, key=lambda m: m['seed']):
        seed = model['seed']
        left, right = cached[seed], _summary(model, *refs)
        _states(raw[seed])
        for key, roles in (('cost', old.COST_ROLES), ('first_step', FIRST_ROLES), ('heads', FIRST_ROLES), ('world_vz', FIRST_ROLES)):
            old.require(set(left[key]) == set(roles), 'cached control roles incomplete')
            for role in roles:
                old._identities(left[key][role]['points'], role)
        old.require(left['training_world_vz']['target_identity_sha256'] == right['training_world_vz']['target_identity_sha256'],
                    'training diagnostic targets differ')
        result['models'].extend((left, right))
        train = right['training_world_vz']['normalized_rmse'] < left['training_world_vz']['normalized_rmse']
        lc, rc = (v['fresh_world_vz_error']['normalized_rmse'] for v in (left, right))
        gain, fresh_unknown = rc <= .8 * lc, [p for p in unknown if p['role'] == 'fresh' and p['kind'] == 'world_vz']
        target_fail = [name for name, failed in (
            ('fresh_world_vz_RMSE_reduction_below20pct', not gain and not fresh_unknown),
            ('fresh_numeric_or_magnitude_qualification', not (left['fresh_numerics_valid'] and right['fresh_numerics_valid'])),
            ('fresh_active_world_vz_direction', any(p['passed'] is False for p in right['world_vz']['fresh']['points']))) if failed]
        target_status = old._decision(target_fail, fresh_unknown)
        failures = {'new_resolved_cost_failures': old._new_cost_failures(left, right)} | {
            'new_' + k + '_qualification_failures': old._new_response_failures(left, right, k) for k in ('yaw', 'body', 'world_vz')}
        states = {}
        for role in ROLES:
            ld, rd = raw[seed]['diagnostics'][role], model['diagnostics'][role]
            lv, rv = ld['mean_state_group_loss'], rd['mean_state_group_loss']
            states[role] = {'control': lv, 'active': rv, 'ratio_active_over_control': old.ratio(rv, lv),
                'passed_le1p1': rv <= 1.1 * lv, 'control_groups': ld['mean_state_group_losses'], 'active_groups': rd['mean_state_group_losses']}
        body_resolved = all(v['body_resolved_count'] == 18 for v in right['heads'].values())
        body_ok = right['body_normalized_rmse'] <= left['body_normalized_rmse']
        regression_fail = [k for k, v in failures.items() if v] + [name for name, failed in (
            ('body_RMSE_increased', body_resolved and not body_ok),
            ('state_error_guard', not all(v['passed_le1p1'] for v in states.values()))) if failed]
        regression_status = old._decision(regression_fail, not body_resolved)
        result['paired'].append({'seed': seed, 'training_improved': train,
            'control_training_world_vz_rmse': left['training_world_vz']['normalized_rmse'],
            'active_training_world_vz_rmse': right['training_world_vz']['normalized_rmse'],
            'raw_fresh_world_vz_rmse_gain': gain, 'control_fresh_world_vz_rmse': lc, 'active_fresh_world_vz_rmse': rc,
            'fresh_world_vz_rmse_ratio_active_over_control': old.ratio(rc, lc),
            'generalization_target_status': target_status, 'generalization_target_known_failures': target_fail,
            'generalization_target_passed': target_status == 'PASSED', **failures, 'state_loss': states,
            'body_rmse_nonincreasing': body_ok, 'body_rmse_references_resolved': body_resolved,
            'control_body_normalized_rmse': left['body_normalized_rmse'], 'active_body_normalized_rmse': right['body_normalized_rmse'],
            'body_rmse_ratio_active_over_control': old.ratio(right['body_normalized_rmse'], left['body_normalized_rmse']),
            'regression_known_failures': regression_fail, 'no_regressions_status': regression_status, 'no_regressions': regression_status == 'PASSED'})
    pairs = result['paired']
    target_fail = [{'seed': p['seed'], 'reason': e} for p in pairs for e in p['generalization_target_known_failures']]
    target_unknown = any(p['generalization_target_status'] == 'REFERENCE_UNRESOLVED' for p in pairs)
    numeric, train = all(m['numerics_valid'] for m in result['models']), all(p['training_improved'] for p in pairs)
    regress = any(p['regression_known_failures'] for p in pairs)
    failures = [name for name, failed in (('training_world_vz_RMSE_not_strictly_down_all3', not train),
        ('regression_or_state_guard', regress), ('all_six_numeric_or_magnitude_qualification', not numeric),
        ('fresh_generalization_target', bool(target_fail))) if failed]
    repairs = [{'seed': m['seed'], 'reason': e} for m in result['models'] if m['arm'] == 'active' for e in m['known_repair_failures']]
    candidate, repair = old._decision(failures, unknown or target_unknown), old._decision(repairs, unknown)
    result.update(status='COMPLETE', training_improved=train, raw_fresh_world_vz_rmse_gain=all(p['raw_fresh_world_vz_rmse_gain'] for p in pairs),
        generalization_target_status=old._decision(target_fail, target_unknown), generalization_target_known_failures=target_fail,
        generalization_target_passed=old._decision(target_fail, target_unknown) == 'PASSED', all_six_numeric_valid=numeric,
        no_regressions=all(p['no_regressions'] for p in pairs), no_regressions_status=old._decision(regress, any(not p['no_regressions'] for p in pairs)),
        candidate_known_failures=failures, candidate_status=candidate, candidate_accepted=candidate == 'PASSED',
        complete_repair_known_failures=repairs, complete_repair_status=repair, complete_repair_validated=repair == 'PASSED')
    return result
