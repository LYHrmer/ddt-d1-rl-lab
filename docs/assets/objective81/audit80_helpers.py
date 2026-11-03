"""Independent stdlib80 readback; run only after formal completion is announced.

No simulator, numerical library, pickle, or frozen experiment helper is loaded.
The separate stdlib79 auditor supplies only its NPY reader, scalar metric
reducer, SHA utility and structural assertions. Its file is also SHA-bound.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import time
import zipfile
from pathlib import Path

HELPER = Path('/home/lyh/model_feedback79_work_20261103/audit79.py')
spec = importlib.util.spec_from_file_location('independent79_scalar_helpers', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
array, require, sha = helper.array, helper.require, helper.sha
mean, ratio, equal, finite = helper.mean, helper.ratio, helper.nested_equal, helper.finite
ROOT = Path('/home/lyh/ddt_ros2_ws')
SEEDS, ARMS = (64001, 64002, 64003), ('legacy', 'steady')
TASKS, RESETS = ('forward_stop', 'reverse_stop', 'turn_stop'), (80401, 80402, 80403)
DEV = {task: tuple(range(80101 + 3 * i, 80104 + 3 * i)) for i, task in enumerate(TASKS)}
TURN = (110, 175, 260, 325, 410)
COUNTERS = ('policy_transitions', 'native_steps', 'pd_updates')
SCENARIOS = tuple((t, s) for t in TASKS for s in RESETS)
POLICIES = ((None, 'anchor', 0), (None, 'anchor', 1)) + tuple((s, a, 0) for s in SEEDS for a in ARMS)


def inverse_rotate(q, v):
    length = math.sqrt(sum(x * x for x in q))
    require(length > 0, 'zero quaternion')
    w, x, y, z = [item / length for item in q]
    cross = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    twice = (y * cross[2] - z * cross[1], z * cross[0] - x * cross[2], x * cross[1] - y * cross[0])
    return tuple(v[i] - 2 * w * cross[i] + 2 * twice[i] for i in range(3))


def cost_parts(state, command, action, previous, nominal):
    velocity = inverse_rotate(state[3:7], state[23:26])
    gravity = inverse_rotate(state[3:7], (0., 0., -1.))
    return (((velocity[0] - command[0]) / .2) ** 2,
            ((state[28] - command[2]) / .2) ** 2,
            (velocity[1] / .2) ** 2,
            6 * (gravity[0] ** 2 + gravity[1] ** 2),
            20 * (state[2] - nominal) ** 2,
            .1 * mean([(a - p) ** 2 for a, p in zip(action, previous, strict=True)]))


def short_metrics(arrays, point, nominal, model=False):
    states, actions = arrays['states'], arrays['actions' if model else 'issued_actions']
    previous = point['last']
    costs, components, errors = [], [], []
    for tick in range(32):
        state, action, command = states[tick * 45:(tick + 1) * 45], actions[tick * 16:(tick + 1) * 16], point['commands'][tick * 3:(tick + 1) * 3]
        parts = cost_parts(state, command, action, previous, nominal)
        components.append(parts)
        costs.append(sum(parts))
        yaw = state[28] if model else arrays['body_wz'][tick]
        errors.append(abs(yaw - command[2]))
        previous = action
    return {'cost': mean(costs), 'yaw_mae': mean(errors), 'per_step_cost64': costs,
            'independent_component_means': [mean([row[i] for row in components]) for i in range(6)]}


def raw_identity(archive):
    """Recreate array_identity80 from stored dtype/shape/payload, no arrays."""
    import ast
    digest = hashlib.sha256()
    dtype_names = {'<f8': 'float64', '<f4': 'float32', '|b1': 'bool', '<i8': 'int64', '<i4': 'int32'}
    for member in sorted(archive.namelist()):
        raw = archive.read(member)
        require(member.endswith('.npy') and raw[:6] == b'\x93NUMPY', 'invalid identity member')
        width = 2 if raw[6] == 1 else 4
        size = int.from_bytes(raw[8:8 + width], 'little')
        header = ast.literal_eval(raw[8 + width:8 + width + size].decode('latin1'))
        require(header['fortran_order'] is False, 'Fortran trace identity')
        digest.update(json.dumps([member[:-4], dtype_names[header['descr']], list(header['shape'])], separators=(',', ':')).encode())
        digest.update(raw[8 + width + size:])
    return digest.hexdigest()


def screen_decision(rows):
    expected = {(seed, tick, m, a, r) for seed in DEV['turn_stop'] for tick in TURN for m, a, r in POLICIES}
    require(set(rows) == expected, 'short branch grid incomplete')
    pairs = []
    for model_seed in SEEDS:
        resets = []
        for seed in DEV['turn_stop']:
            row = {'seed': seed}
            for arm in ('anchor', *ARMS):
                model = None if arm == 'anchor' else model_seed
                pos, neg = [rows[(seed, tick, model, arm, 0)]['yaw_mae'] for tick in (175, 325)]
                row[arm] = {'positive_turn_fragment_mae': pos, 'negative_turn_fragment_mae': neg,
                            'max_fragment_mae': max(pos, neg)}
            resets.append(row)
        qs = {a: mean([r[a]['max_fragment_mae'] for r in resets]) for a in ('anchor', *ARMS)}
        pairs.append({'model_seed': model_seed, 'Q_anchor': qs['anchor'], 'Q_legacy': qs['legacy'], 'Q_steady': qs['steady'],
            'steady_lt_legacy': qs['steady'] < qs['legacy'], 'steady_lt_anchor': qs['steady'] < qs['anchor'],
            'passed': qs['steady'] < qs['legacy'] and qs['steady'] < qs['anchor'], 'by_reset': resets})
    gates = {'all120_branches_complete': True, 'all120_safe': True, 'all15_anchor_replays_exact': True,
             'all_three_seed_Q_improve': all(p['passed'] for p in pairs)}
    return {'status': 'COMPLETE', 'passed': all(gates.values()), 'enter_full_evaluation': all(gates.values()),
            'gates': gates, 'by_model_seed': pairs}


def evaluation_decision(baseline, candidates):
    require(set(baseline) == set(SCENARIOS), 'incomplete original evaluation grid')
    require(set(candidates) == {(m, a, t, s) for m in SEEDS for a in ARMS for t, s in SCENARIOS}, 'incomplete candidate evaluation grid')
    controls, paired = [], []
    for model_seed in SEEDS:
        for arm in ARMS:
            bj = [baseline[(t, s)]['J'] for t, s in SCENARIOS]
            cj = [candidates[(model_seed, arm, t, s)]['J'] for t, s in SCENARIOS]
            by = [baseline[('turn_stop', s)]['stable_wz_mae_rad_s'] for s in RESETS]
            cy = [candidates[(model_seed, arm, 'turn_stop', s)]['stable_wz_mae_rad_s'] for s in RESETS]
            jr, yr = ratio(mean(cj), mean(bj), 1e-12), ratio(mean(cy), mean(by), 1e-12)
            gates = {'all_baseline_and_candidate_original_thresholds': all(baseline[(t, s)]['reference_thresholds_met'] and
                candidates[(model_seed, arm, t, s)]['reference_thresholds_met'] for t, s in SCENARIOS),
                'all_three_turn_resets_strictly_improve': all(c < b for c, b in zip(cy, by, strict=True)),
                'turn_mean_ratio_le0p95': yr is not None and yr <= .95, 'J_mean_ratio_le1': jr is not None and jr <= 1.,
                'all_nine_J_pairs_le1p05': all(c <= 1.05 * b for c, b in zip(cj, bj, strict=True))}
            controls.append({'model_seed': model_seed, 'arm': arm, 'passed': all(gates.values()), 'gates': gates,
                             'J_ratio': jr, 'yaw_mae_ratio': yr})
        lm, sm = [mean([candidates[(model_seed, a, 'turn_stop', s)]['stable_wz_mae_rad_s'] for s in RESETS]) for a in ARMS]
        paired.append({'model_seed': model_seed, 'turn_reset_count': 3, 'legacy_mean_stable_wz_mae': lm,
            'steady_mean_stable_wz_mae': sm, 'stable_wz_mae_ratio_steady_over_legacy': ratio(sm, lm),
            'strict_stable_wz_mae_improvement': sm < lm})
    gates = {'all_three_steady_original_control_gates': all(c['passed'] for c in controls if c['arm'] == 'steady'),
        'all_three_paired_stable_yaw_means_strictly_improve': all(p['strict_stable_wz_mae_improvement'] for p in paired),
        'all63_original_task_stop_safety_thresholds': all(r['reference_thresholds_met'] for r in (*baseline.values(), *candidates.values()))}
    return {'status': 'COMPLETE', 'sampling_passed': all(gates.values()),
            'sampling': {'passed': all(gates.values()), 'gates': gates, 'by_model_seed': paired}, 'control_gates': controls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'evidence/steady_sampling80/execution_01')
    parser.add_argument('--plan', type=Path, default=ROOT / 'docs/experiments80/steady_sampling80_preregistered.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/steady_sampling80_independent_audit_20261103.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive independent audit output required')
    output, started, files = args.input.resolve(), time.process_time(), {}
    audit = {'schema': 'd1-steady-sampling80-independent-audit-v1', 'status': 'INCOMPLETE', 'failure': None,
        'new_model_calls': 0, 'new_actor_calls': 0, 'new_physics_steps': 0, 'new_updates': 0,
        'global_candidate_accepted': False, 'original_qualification_changed': False,
        'arithmetic_comparison': {'absolute': 1e-12, 'relative': 1e-12,
            'scope': 'scalar stdlib-versus-Torch/NumPy readback only; all exact decision gates unchanged'},
        'limitations': ['No pickle archive is executed or reconstructed as live MjData.',
            'Parameter/mapping changes are checked through receipts and file hashes; Torch tensors are not loaded.',
            'All Python/XML/config source bindings are independently checked; the full inherited payload closure is not rehashed.']}

    def bind(path):
        path = path.resolve()
        receipt = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        if str(path) in files:
            require(files[str(path)] == receipt, 'file changed between reads: ' + str(path))
        files[str(path)] = receipt
        return receipt['sha256']

    def read(path):
        bind(path)
        result = json.loads(path.read_text())
        finite(result)
        return result

    def journal(path):
        if not path.exists():
            return []
        bind(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        finite(rows)
        return rows

    try:
        bind(Path(__file__))
        bind(HELPER)
        report, plan = read(output / 'report.json'), read(args.plan)
        complete = report['status'] == report['verification_after'] == 'COMPLETE'
        audit.update(formal_status=report['status'], formal_complete=complete, formal_outcome=report.get('outcome'),
                     formal_failure=report.get('failure'), formal_plan_sha256=report['plan_sha256'])
        require(bind(args.plan) == report['plan_sha256'], 'plan SHA differs')
        require(plan['schema'] == 'd1-steady-sampling80-preregistered-v1', 'plan schema differs')
        require(report['global_candidate_accepted'] is False and report['complete_model_eligibility'] == 'NOT_REQUALIFIED', 'qualification boundary')
        source_count = 0
        for rel, digest in plan['source_sha256'].items():
            if rel.endswith(('.py', '.xml')) or rel.startswith('training/configs/'):
                require(bind(ROOT / rel) == digest, 'bound frozen source: ' + rel)
                source_count += 1
        audit['independently_verified_source_bindings'] = source_count
        for rel, digest in report.get('raw_artifacts_sha256', {}).items():
            require(bind(output / rel) == digest, 'formal raw artifact hash: ' + rel)
        audit['formal_raw_artifacts_verified'] = len(report.get('raw_artifacts_sha256', {}))
        old79_path = ROOT / 'evidence/feedback79/execution_01/report.json'
        old79 = read(old79_path)
        oldplan_path = ROOT / 'docs/experiments79/feedback79_preregistered.json'
        oldplan = read(oldplan_path)
        require(old79['status'] == old79['verification_after'] == 'COMPLETE' and old79['plan_sha256'] == bind(oldplan_path), 'nominal height source incomplete')
        heights = [r['metrics']['reset_info']['nominal_base_height_m'] for r in old79['baseline'] + old79['candidates']]
        require(len(heights) == 63 and len(set(heights)) == 1, 'nonconstant nominal height provenance')
        for rel, digest in oldplan['source_sha256'].items():
            require(plan['source_sha256'].get(rel) == digest, 'inherited model/config source identity differs')
        nominal = heights[0]
        audit['nominal_height_provenance'] = {'value_m': nominal, 'source': str(old79_path.relative_to(ROOT)),
            'sha256': bind(old79_path), 'agreeing_reset_info_values': len(heights),
            'same_frozen_environment_source_closure': True, 'new_environment_measurement_performed': False}

        audit['development'] = []
        points = {}
        dev_seal_path = output / 'development_sealed_before_training.json'
        if dev_seal_path.exists():
            seal = read(dev_seal_path)
            require(bind(dev_seal_path) == report['development_seal_sha256'] and seal['episodes'] == report['development'], 'development seal mismatch')
            for rel, digest in seal['artifact_sha256'].items():
                require(bind(output / rel) == digest, 'development raw binding')
        for row in report['development']:
            task, seed = row['task'], row['seed']
            require(task in DEV and seed in DEV[task], 'development identity differs')
            folder = output / 'development' / f'{task}_{seed}'
            require(read(folder / 'receipt.json') == row, 'development receipt differs')
            trace, context = folder / 'full_or_partial_trace.npz', folder / 'point_arrays.npz'
            require(bind(trace) == row['trace_sha256'], 'development trace hash')
            bind(context)
            bind(folder / 'integration_context_archive.pkl')
            ticks = TURN if task == 'turn_stop' else (110, 260, 410)
            if row['status'] != 'COMPLETE':
                audit['development'].append({'task': task, 'seed': seed, 'status': row['status'], 'steps': row['steps']})
                continue
            require(row['steps'] == 600 and row['points'] == [{'seed': seed, 'tick': t, 'task': task} for t in ticks], 'development complete grid')
            equal(dict(zip(COUNTERS, (600, 6000, 3000), strict=True)), row['actual_counts'], 'development counts')
            with zipfile.ZipFile(trace) as raw, zipfile.ZipFile(context) as ctx:
                states = array(raw, 's', (600, 45))[0]
                actions = array(raw, 'a', (600, 16))[0]
                commands = array(raw, 'command_during_interval', (600, 3))[0]
                require(actions == array(raw, 'issued_action', (600, 16))[0] == array(raw, 'applied_action', (600, 16))[0], 'development zero-delay actions')
                require(array(raw, 'tick', (600,))[0] == tuple(range(600)), 'development tick ordering')
                require(array(raw, 'done', (600,))[0] == (False,) * 599 + (True,), 'development boundaries')
                for key in ('fallen', 'illegal_body_contact', 'terminated', 'truncated'):
                    require(not any(array(raw, key, (600,))[0]), 'development safety')
                for index, tick in enumerate(ticks):
                    point = {name: array(ctx, f'{index}_{name}', dims)[0] for name, dims in
                             (('state', (45,)), ('last', (16,)), ('commands', (33, 3)))}
                    require(point['state'] == states[tick * 45:(tick + 1) * 45] and
                            point['last'] == actions[(tick - 1) * 16:tick * 16] and
                            point['commands'] == commands[tick * 3:(tick + 33) * 3], 'point before-action alignment')
                    points[(seed, tick)] = point
            audit['development'].append({'task': task, 'seed': seed, 'status': 'COMPLETE', 'steps': 600, 'point_ticks': ticks})
        expected_points = {(seed, tick) for task, seeds in DEV.items() for seed in seeds for tick in (TURN if task == 'turn_stop' else (110, 260, 410))}
        if complete:
            require(set(points) == expected_points and len(report['development']) == 9, '33 unique development point requirement')
        selected = {arm: [[task, seed, tick] for task, seeds in DEV.items() for seed in seeds for tick in
                    ((175, 325, 410) if arm == 'steady' and task == 'turn_stop' else (110, 260, 410))] for arm in ARMS}
        if 'sampling_point_ids' in report:
            require(report['sampling_point_ids'] == selected, 'registered27-point sets differ')
        audit['sampling_point_ids'] = selected
        audit['unique_development_points'] = len(points)

        audit['policy_updates'] = []
        source75_path = ROOT / 'evidence/world_vz75/execution_01/report.json'
        source75 = read(source75_path)
        require(source75['status'] == source75['verification_after'] == 'COMPLETE', '75 source incomplete')
        require(bind(source75_path) == plan['source_sha256'][str(source75_path.relative_to(ROOT))], '75 source report SHA')
        updates = 0
        for seed in SEEDS:
            old = next(m for m in source75['models'] if m['seed'] == seed and m['arm'] == 'control')
            source_weight = source75_path.parent / old['directory'] / 'final_state_dict.pt'
            require(bind(source_weight) == source75['sealed_artifacts_sha256'][old['directory'] + '/final_state_dict.pt']
                    == plan['source_sha256'][str(source_weight.relative_to(ROOT))], 'source75 weights SHA')
            for arm in ARMS:
                expected_actor = report.get('anchor_mapping_sha256')
                for step in (1, 2):
                    folder = output / f'seed{seed}' / f'actor_{arm}_{step}'
                    entry = {'model_seed': seed, 'arm': arm, 'step': step, 'status': 'NOT_RUN', 'updates_completed': 0}
                    if (folder / 'receipt.json').exists():
                        r = read(folder / 'receipt.json')
                        entry.update(status=r['status'], updates_completed=r['updates_completed'], updates_attempted=r['updates_attempted'],
                            failure=r.get('failure'), final_row_l1=r.get('final_row_l1'), selfcheck=r.get('selfcheck'))
                        require(0 <= r['updates_completed'] <= r['updates_attempted'] <= 1, 'policy update count')
                        updates += r['updates_completed']
                        events = journal(folder / 'events.jsonl')
                        require(sum(x['stage'] == 'sgd_completed' for x in events) == r['updates_completed'], 'update journal differs')
                        if r['status'] == 'COMPLETE':
                            require(r['updates_completed'] == 1 and r['verification_after'] == 'COMPLETE' and r['point_count'] == 27,
                                    'complete actor update differs')
                            require(r['actor_before_mapping_sha256'] == expected_actor and r['anchor_mapping_sha256'] == report['anchor_mapping_sha256']
                                    and r['model_before_mapping_sha256'] == old['fit']['final_model_state_sha256'], 'actor/model start chain')
                            require([[p['seed'], p['tick']] for p in r['points']] == [p[1:] for p in selected[arm]], 'actor update27 starts differ')
                            require(r['protected_before_mapping_sha256'] == r['protected_after_mapping_sha256'] and
                                    len(r['final_row_l1']) == 4 and all(0 <= x <= .01 for x in r['final_row_l1']), 'cumulative anchor parameter bound')
                            check = r['selfcheck']; ad, fd = check['gradient'], check['model_central_difference']
                            agreement = abs(ad - fd) <= 1e-4 + .1 * abs(fd)
                            identifiable = abs(ad) >= 1e-5 and abs(fd) >= 1e-5 and ad * fd > 0
                            require(check['autograd_agrees'] is agreement and check['direction_identifiable'] is identifiable
                                    and check['passed'] is (agreement and identifiable) and check['passed'], 'actor AD/FD qualification')
                            require(r['allowed_parameter_count'] == 516 and r['learning_rate'] == .001 and r['horizon'] == 32,
                                    'fixed policy update settings')
                            expected_actor = r['actor_after_mapping_sha256']
                        for name, digest in (('weights_file', 'weights_file_sha256'), ('head_step_file', 'head_step_file_sha256'),
                                             ('partial_actor_file', 'partial_actor_file_sha256')):
                            if name in r:
                                require(bind(folder / r[name]) == r[digest], 'policy artifact hash')
                    audit['policy_updates'].append(entry)
        require(updates == report['policy_network_updates'] and report['model_updates'] == 0, 'formal update accounting')
        if complete:
            require(updates == 12 and all(r['status'] == 'COMPLETE' for r in audit['policy_updates']), 'incomplete12-update schedule')

        seal_path = output / 'policies_sealed_before_screen.json'
        if seal_path.exists():
            seal = read(seal_path)
            require(bind(seal_path) == report['sealed_policies_sha256'] and len(seal['policies']) == 6, 'policy seal differs')
            require({(r['model_seed'], r['arm']) for r in seal['policies']} == {(s, a) for s in SEEDS for a in ARMS}, 'six-policy grid')
            for row in seal['policies']:
                require(bind(output / row['policy_file']) == row['file_sha256'], 'final policy bytes')
                receipt = read(output / f"seed{row['model_seed']}/actor_{row['arm']}_2/receipt.json")
                require(row['mapping_sha256'] == receipt['actor_after_mapping_sha256'], 'sealed final policy mapping')
            audit['sealed_final_policies'] = seal

        audit['predictions'] = []
        prediction_seal = output / 'predictions_sealed_before_physics.json'
        if prediction_seal.exists():
            require(read(prediction_seal) == report['predictions'] and bind(prediction_seal) == report['prediction_seal_sha256'], 'prediction seal differs')
        prediction_keys = set()
        for row in report['predictions']:
            key = (row['model_seed'], row['arm'], row['seed'], row['tick'])
            require(key not in prediction_keys, 'duplicate model prediction')
            prediction_keys.add(key)
            path = output / row['trace_path']
            require(bind(path) == row['trace_sha256'], 'prediction trace SHA')
            with zipfile.ZipFile(path) as raw:
                arrays = {name: array(raw, name, dims)[0] for name, dims in
                    (('states', (32, 45)), ('actions', (32, 16)), ('costs', (32,)), ('cost_components', (32, 6)))}
            metrics = short_metrics(arrays, points[(row['seed'], row['tick'])], nominal, model=True)
            equal({'cost': metrics['cost'], 'yaw_mae': metrics['yaw_mae']}, row, 'predicted trace scalar reduction')
            equal(metrics['per_step_cost64'], arrays['costs'], 'model saved costs')
            audit['predictions'].append({'identity': key, 'cost': metrics['cost'], 'yaw_mae': metrics['yaw_mae'],
                'max_cost_abs_difference': max(abs(x - y) for x, y in zip(metrics['per_step_cost64'], arrays['costs'], strict=True))})
        if complete:
            require(prediction_keys == {(s, a, d, t) for s in SEEDS for a in ('anchor', *ARMS) for d in DEV['turn_stop'] for t in TURN},
                    '135 model prediction grid differs')

        audit['short_branches'], audit['anchor_replays'] = [], []
        short, anchor_members = {}, {}
        for row in report['short_branches']:
            key = (row['seed'], row['tick'], row['model_seed'], row['arm'], row['repeat'])
            require(key not in short, 'duplicate physical branch')
            path = output / row.get('trace_path', f"screen/{row['seed']}_{row['tick']}_{row['model_seed']}_{row['arm']}_{row['repeat']}.npz")
            require(bind(path) == row['trace_sha256'], 'short trace SHA')
            saved = read(path.with_suffix('.json'))
            require(saved == {k: v for k, v in row.items() if k != 'trace_path'}, 'short branch JSON differs')
            if row['status'] != 'COMPLETE':
                audit['short_branches'].append({'identity': key, 'status': row['status'], 'steps': row['steps']})
                continue
            require(row['steps'] == 32 and row['native_steps'] == 320 and row['pd_updates'] == 160, 'short work count')
            equal(dict(zip(COUNTERS, (32, 320, 160), strict=True)), row['actual_counts'], 'short actual counters')
            with zipfile.ZipFile(path) as raw:
                arrays = {name: array(raw, name, dims)[0] for name, dims in
                    (('states', (32, 45)), ('observations', (32, 627)), ('rewards', (32,)), ('commands', (32, 3)),
                     ('body_wz', (32,)), ('issued_actions', (32, 16)), ('applied_actions', (32, 16)),
                     ('fallen', (32,)), ('illegal_body_contact', (32,)), ('terminated', (32,)), ('truncated', (32,)))}
                require(raw_identity(raw) == row['trace_identity_sha256'], 'raw array identity hash differs')
                if row['arm'] == 'anchor':
                    members = {name: raw.read(name) for name in raw.namelist()}
                    site = key[:2]
                    if row['repeat'] == 0:
                        anchor_members[site] = members
                    else:
                        require(anchor_members[site] == members, 'anchor duplicate not byte exact')
                        audit['anchor_replays'].append({'seed': site[0], 'tick': site[1], 'all_npy_members_bit_exact': True})
            point = points[key[:2]]
            require(arrays['commands'] == point['commands'][:96], 'short current-command alignment')
            require(arrays['issued_actions'] == arrays['applied_actions'], 'short zero-delay action')
            require(all(arrays['observations'][t * 627 + 41:t * 627 + 57] == arrays['issued_actions'][t * 16:t * 16 + 16]
                        for t in range(32)), 'actual action observation seam')
            require(not any(arrays[k][t] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated') for t in range(32)), 'unsafe short branch')
            metrics = short_metrics(arrays, point, nominal)
            equal({k: metrics[k] for k in ('cost', 'yaw_mae', 'per_step_cost64')}, row, 'short independently reduced metrics')
            short[key] = metrics
            audit['short_branches'].append({'identity': key, 'status': 'COMPLETE', **metrics,
                'max_cost_abs_difference': max(abs(a - b) for a, b in zip(metrics['per_step_cost64'], row['per_step_cost64'], strict=True)),
                'yaw_mae_abs_difference': abs(metrics['yaw_mae'] - row['yaw_mae']),
                'raw_gyro_vs_state28_max_gap': max(abs(arrays['body_wz'][t] - arrays['states'][t * 45 + 28]) for t in range(32))})
        if complete:
            screen = screen_decision(short)
            equal(screen, report['screen'], 'independent screen Q/gates')
            require(len(audit['anchor_replays']) == 15, '15 anchor parity sites')
            audit['independent_screen'] = screen
        else:
            audit['independent_screen'] = {'status': 'NOT_EVALUATED_INCOMPLETE', 'passed': False}

        baseline, candidates = {}, {}
        audit['evaluation_episodes'] = []
        for kind, indexed in (('baseline', baseline), ('candidates', candidates)):
            for row in report[kind]:
                if row['status'] != 'COMPLETE':
                    audit['evaluation_episodes'].append({'status': row['status'], 'record': row})
                    continue
                path = output / row['trace_path']
                require(bind(path) == row['trace_sha256'] == row['metrics']['trace_sha256'], 'holdout trace SHA')
                require(read(path.with_suffix('.json')) == row, 'holdout JSON differs')
                metrics = helper.recompute_trace(path, row['task'], row['seed'], row['metrics']['done_reason'])
                equal(metrics, row['metrics'], 'holdout independent metrics')
                key = (row['task'], row['seed']) if kind == 'baseline' else (row['model_seed'], row['arm'], row['task'], row['seed'])
                require(key not in indexed, 'duplicate holdout')
                indexed[key] = metrics
                audit['evaluation_episodes'].append({'identity': key, 'kind': kind, 'status': 'COMPLETE', 'metrics': metrics})
        if complete and audit['independent_screen']['enter_full_evaluation']:
            comparison = evaluation_decision(baseline, candidates)
            equal(comparison, report['comparison'], 'whole-control comparison')
            require(report['evaluation_status'] == 'COMPLETE' and report['outcome'] == 'FULL_EVALUATION_COMPLETE', 'holdout status contradiction')
            audit['independent_comparison'] = comparison
        elif complete:
            require(not report['baseline'] and not report['candidates'] and report['evaluation_status'] == 'NOT_RUN'
                    and report['comparison']['status'] == 'NOT_RUN' and report['outcome'] == 'SCREEN_REJECTED', 'holdout touched after rejection')
            require(not (output / 'evaluation').exists(), 'unregistered holdout files exist after rejection')
            audit['independent_comparison'] = {'status': 'NOT_RUN', 'not_run_original_baselines': 9,
                'not_run_candidates': 54, 'candidate_accepted': False}

        physics = journal(output / 'all_physics_steps.jsonl')
        phases, unsafe = collections.Counter(), []
        for i, row in enumerate(physics, 1):
            equal(dict(zip(COUNTERS, (i, i * 10, i * 5), strict=True)), row['counts'], 'monotonic physical counters')
            phases[row['phase']] += 1
            require(row['issued_action'] == row['applied_action'], 'journal zero-delay actual actions')
            if any(row[k] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated')):
                unsafe.append(row)
        last = {k: physics[-1]['counts'][k] if physics else 0 for k in COUNTERS}
        equal(last, report['actual_counts'], 'reported physical counts')
        totals = {name: sum(n for phase, n in phases.items() if phase.startswith(prefix)) for name, prefix in
                  (('development', 'development/'), ('screen', 'screen/'), ('evaluation', 'evaluate/'))}
        require(sum(totals.values()) == len(physics), 'unregistered physical phase')
        if complete:
            expected = 47040 if audit['independent_screen']['enter_full_evaluation'] else 9240
            require(len(physics) == expected and totals == {'development': 5400, 'screen': 3840, 'evaluation': expected - 9240}
                    and not unsafe, 'formal budget/safety differs')
            require(report['model_calls'] == 12 * 27 * 3 * 32 + 135 * 32 == 35424, 'model forward formula differs')
            require(report['actor_calls'] == report['model_calls'] + len(physics) + 192, 'actor hook count formula differs')
        admission = report['admission']
        if admission.get('status') == 'COMPLETE':
            path = Path(admission['path']); preflight = read(path)
            require(bind(path) == admission['sha256'] and preflight['status'] == preflight['verification_after'] == 'COMPLETE'
                    and preflight['plan_sha256'] == report['plan_sha256'], 'preflight same-plan admission')
            equal(dict(zip(COUNTERS, (224, 2240, 1120), strict=True)), preflight['actual_counts'], 'preflight counters')
        audit['physical_accounting'] = {'formal_counts': last, 'policy_steps_by_stage': totals,
            'policy_steps_by_phase': dict(phases), 'unsafe_rows': unsafe,
            'including224_preflight': {k: v + x for (k, v), x in zip(last.items(), (224, 2240, 1120), strict=True)}
                if admission.get('status') == 'COMPLETE' else None}
        progress = journal(output / 'progress.jsonl')
        sealed_policy = [i for i, r in enumerate(progress) if r['event'] == 'ALL_SIX_POLICIES_SEALED']
        sealed_prediction = [i for i, r in enumerate(progress) if r['event'] == 'ALL_PREDICTIONS_SEALED']
        first_screen = [i for i, r in enumerate(progress) if r['phase'].startswith('screen/')]
        if first_screen:
            require(len(sealed_policy) == len(sealed_prediction) == 1 and sealed_policy[0] < sealed_prediction[0] < min(first_screen),
                    'physics preceded policy/prediction seals')
        eval_events = [i for i, r in enumerate(progress) if r['phase'].startswith('evaluate/')]
        passed_events = [i for i, r in enumerate(progress) if r['event'] == 'WHOLE_SCREEN_PASSED']
        require(not eval_events or len(passed_events) == 1 and passed_events[0] < min(eval_events), 'holdout preceded whole screen admission')
        audit['failure_files'] = []
        for path in sorted(output.rglob('failure.txt')):
            bind(path)
            audit['failure_files'].append({'path': str(path.relative_to(output)), 'text': path.read_text()})
        audit['resource_receipt'] = {k: report.get(k) for k in ('cpu_seconds', 'wall_seconds', 'model_calls', 'actor_calls',
            'policy_network_updates', 'model_updates', 'raw_bytes_before_report', 'cleanup_errors')}
        audit['sampling_unit'] = {'model_seed_units': 3, 'development_reset_units': 9, 'unique_turn_sites': 15,
            'short_branches_are_not120_independent_seeds': True}
        audit['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve independent audit failure without touching formal results.
        audit['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        changed = [path for path, receipt in files.items() if not Path(path).is_file() or sha(Path(path)) != receipt['sha256']]
        audit['verification_after'] = 'INCOMPLETE' if changed else 'COMPLETE'
        if changed:
            audit.update(status='INCOMPLETE', changed_inputs=changed)
        audit.update(inputs=files, audited_input_files=len(files), cpu_seconds=time.process_time() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(audit, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: audit.get(k) for k in ('status', 'formal_complete', 'formal_outcome', 'failure', 'cpu_seconds', 'audited_input_files')}))
    return int(audit['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
