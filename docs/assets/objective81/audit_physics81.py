"""Independent stdlib81 readback. Execute only after the root's explicit release.

No frozen comparator, NumPy, Torch, actor, simulator or pickle is executed.
The independent79/80 auditors supply a standard-library NPY reader and scalar
cost equations. Stored files are SHA-bound before and after this readback.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import time
import zipfile
from fractions import Fraction
from pathlib import Path

HELPER80 = Path('/home/lyh/steady_sampling80_work_20261103/audit80.py')
spec = importlib.util.spec_from_file_location('independent80_helpers', HELPER80)
helper80 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper80)
helper = helper80.helper
array, require, sha = helper.array, helper.require, helper.sha
mean, ratio, equal, finite = helper.mean, helper.ratio, helper.nested_equal, helper.finite
ROOT = Path('/home/lyh/ddt_ros2_ws')
MODELS, ARMS = (64001, 64002, 64003), ('C6', 'Q6')
D_SEEDS, V_SEEDS, H_SEEDS = (81107, 81108, 81109), (81207, 81208, 81209), (81401, 81402, 81403)
TICKS, STEADY, SCALES, PREFIXES = (110, 175, 260, 325, 410), (175, 325), (.009, .0045, .00225, .001125), (1, 8, 16, 32)
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
SCENARIOS = tuple((t, s) for t in TASKS for s in H_SEEDS)
COUNTERS = ('policy_transitions', 'native_steps', 'pd_updates')


def specs(tick):
    probes = tuple((k, sign) for k in range(4) for sign in (1, -1)) if tick in STEADY else ((0, 1),)
    return ((None, 'anchor', 0, None, 0), (None, 'anchor', 1, None, 0)) + tuple(
        (m, a, 0, k, sign) for m in MODELS for a in ARMS for k, sign in probes)


def derived_trace(archive, point, model):
    states = array(archive, 'states', (32, 45))[0]
    actions = array(archive, 'actions' if model else 'issued_actions', (32, 16))[0]
    applied = array(archive, 'applied_actions', (32, 16))[0]
    commands = array(archive, 'commands', (32, 3))[0]
    require(commands == point['commands'][:96], 'trace command/start clock differs')
    require(actions == applied and all(abs(a) <= 1 for a in actions), 'zero-delay action mismatch')
    if model:
        yaw = tuple(states[t * 45 + 28] for t in range(32))
    else:
        yaw = array(archive, 'body_wz', (32,))[0]
        obs = array(archive, 'observations', (32, 627))[0]
        require(all(actions[t * 16:(t + 1) * 16] == obs[t * 627 + 41:t * 627 + 57] for t in range(32)),
                'physical issued/current observation action seam differs')
        for name in ('fallen', 'illegal_body_contact', 'terminated', 'truncated'):
            require(not any(array(archive, name, (32,))[0]), 'unsafe physical branch: ' + name)
        array(archive, 'first_actor_input', (627,))
    previous, parts = point['last'], []
    for tick in range(32):
        action = actions[tick * 16:(tick + 1) * 16]
        parts.append(helper80.cost_parts(states[tick * 45:(tick + 1) * 45],
            commands[tick * 3:(tick + 1) * 3], action, previous, point['nominal_height']))
        previous = action
    costs = [sum(p) for p in parts]
    signed = [yaw[t] - commands[t * 3 + 2] for t in range(32)]
    errors = [abs(x) for x in signed]
    saved_costs = array(archive, 'costs', (32,))[0]
    saved_parts = array(archive, 'cost_components', (32, 6))[0]
    equal(costs, saved_costs, 'independent32 costs')
    equal([v for row in parts for v in row], saved_parts, 'independent32x6 costs')
    prefixes = {str(h): {'cost': mean(costs[:h]), 'yaw_mae': mean(errors[:h]),
        'cost_components': [mean([p[k] for p in parts[:h]]) for k in range(6)],
        'zero_yaw_error_count': sum(x == 0 for x in signed[:h]),
        'minimum_abs_yaw_error': min(errors[:h])} for h in PREFIXES}
    metrics = {'cost': mean(costs), 'yaw_mae': mean(errors), 'prefix_metrics': prefixes,
               'action_at_bound_coordinates': sum(abs(a) >= 1 for a in actions)}
    if not model:
        metrics['per_step_cost64'] = costs
    errors = {'max_cost64_gap': max(abs(a - b) for a, b in zip(costs, saved_costs, strict=True)),
        'max_cost_component_gap': max(abs(a - b) for a, b in zip((v for p in parts for v in p), saved_parts, strict=True)),
        'raw_gyro_vs_state28_max_gap': max(abs(yaw[t] - states[t * 45 + 28]) for t in range(32))}
    return metrics, errors


def q_screen(rows):
    expected = {(s, t, *p) for s in V_SEEDS for t in TICKS for p in specs(t)}
    require(set(rows) == expected, '372 physical branch identities differ')
    paired = []
    for model in MODELS:
        resets = []
        for seed in V_SEEDS:
            r = {'seed': seed}
            for arm in ('anchor', *ARMS):
                m, k, sign = (None, None, 0) if arm == 'anchor' else (model, 0, 1)
                a, b = [rows[(seed, t, m, arm, 0, k, sign)]['yaw_mae'] for t in STEADY]
                r[arm] = {'positive_turn_fragment_mae': a, 'negative_turn_fragment_mae': b, 'max_fragment_mae': max(a, b)}
            resets.append(r)
        qs = {a: mean([r[a]['max_fragment_mae'] for r in resets]) for a in ('anchor', *ARMS)}
        paired.append({'model_seed': model, 'Q_anchor': qs['anchor'], 'Q_C6': qs['C6'], 'Q_Q6': qs['Q6'],
            'Q6_lt_C6': qs['Q6'] < qs['C6'], 'Q6_lt_anchor': qs['Q6'] < qs['anchor'],
            'passed': qs['Q6'] < qs['C6'] and qs['Q6'] < qs['anchor'], 'by_reset': resets})
    gates = {'all372_branches_complete': True, 'all372_safe': True, 'all15_anchor_replays_exact': True,
             'all_three_seed_Q_improve': all(p['passed'] for p in paired)}
    return {'status': 'COMPLETE', 'passed': all(gates.values()), 'enter_full_evaluation': all(gates.values()),
            'gates': gates, 'by_model_seed': paired}


def full_decision(baseline, candidates):
    require(set(baseline) == set(SCENARIOS), 'nine baseline grid')
    require(set(candidates) == {(m, a, t, s) for m in MODELS for a in ARMS for t, s in SCENARIOS}, '54 candidate grid')
    controls, paired = [], []
    for model in MODELS:
        for arm in ARMS:
            bj = [baseline[(t, s)]['J'] for t, s in SCENARIOS]
            cj = [candidates[(model, arm, t, s)]['J'] for t, s in SCENARIOS]
            by = [baseline[('turn_stop', s)]['stable_wz_mae_rad_s'] for s in H_SEEDS]
            cy = [candidates[(model, arm, 'turn_stop', s)]['stable_wz_mae_rad_s'] for s in H_SEEDS]
            jr, yr = ratio(mean(cj), mean(bj), 1e-12), ratio(mean(cy), mean(by), 1e-12)
            gates = {'all_baseline_and_candidate_original_thresholds': all(baseline[(t, s)]['reference_thresholds_met'] and
                candidates[(model, arm, t, s)]['reference_thresholds_met'] for t, s in SCENARIOS),
                'all_three_turn_resets_strictly_improve': all(c < b for c, b in zip(cy, by, strict=True)),
                'turn_mean_ratio_le0p95': yr is not None and yr <= .95, 'J_mean_ratio_le1': jr is not None and jr <= 1.,
                'all_nine_J_pairs_le1p05': all(c <= 1.05 * b for c, b in zip(cj, bj, strict=True))}
            controls.append({'model_seed': model, 'arm': arm, 'passed': all(gates.values()), 'gates': gates,
                             'J_ratio': jr, 'yaw_mae_ratio': yr})
        cm, qm = [mean([candidates[(model, arm, 'turn_stop', s)]['stable_wz_mae_rad_s'] for s in H_SEEDS]) for arm in ARMS]
        paired.append({'model_seed': model, 'turn_reset_count': 3, 'C6_mean_stable_wz_mae': cm,
            'Q6_mean_stable_wz_mae': qm, 'stable_wz_mae_ratio_Q6_over_C6': ratio(qm, cm),
            'strict_stable_wz_mae_improvement': qm < cm})
    gates = {'all_three_Q6_original_control_gates': all(c['passed'] for c in controls if c['arm'] == 'Q6'),
        'all_three_paired_stable_yaw_means_strictly_improve': all(p['strict_stable_wz_mae_improvement'] for p in paired),
        'all63_original_task_stop_safety_thresholds': all(r['reference_thresholds_met'] for r in (*baseline.values(), *candidates.values()))}
    return {'status': 'COMPLETE', 'objective_passed': all(gates.values()), 'control_gates': controls,
            'objective': {'passed': all(gates.values()), 'gates': gates, 'by_model_seed': paired}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'evidence/objective81/execution_01')
    parser.add_argument('--plan', type=Path, default=ROOT / 'docs/experiments81/objective81_preregistered.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/objective81_physics_independent_audit_20261103.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'exclusive audit output required')
    output, started, files = args.input.resolve(), time.process_time(), {}
    audit = {'schema': 'd1-objective81-independent-physics-audit-v1', 'status': 'INCOMPLETE', 'failure': None,
        'new_model_calls': 0, 'new_actor_calls': 0, 'new_physics_steps': 0, 'new_updates': 0,
        'global_candidate_accepted': False, 'original_qualification_changed': False,
        'arithmetic_readback_tolerance': {'absolute': 1e-12, 'relative': 1e-12,
            'scope': 'cross-implementation scalar reduction only; original decision inequalities unchanged'},
        'limitations': ['No stored pickle is executed and no live MjData is reconstructed.',
            'Weights/gradients are not loaded; policy parameter claims are bound to sealed receipts/file SHA.',
            'Python/XML/config sources are rehashed, not the entire inherited historical payload closure.']}

    def bind(path):
        path = path.resolve()
        current = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        if str(path) in files:
            require(current == files[str(path)], 'file changed between reads: ' + str(path))
        files[str(path)] = current
        return current['sha256']

    def read(path):
        bind(path)
        value = json.loads(path.read_text())
        finite(value)
        return value

    def journal(path):
        bind(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        finite(rows)
        return rows

    try:
        for path in (Path(__file__), HELPER80, helper80.HELPER):
            bind(path)
        report, plan = read(output / 'report.json'), read(args.plan)
        audit.update(formal_status=report['status'], formal_outcome=report.get('outcome'), formal_failure=report.get('failure'))
        require(report['status'] == report['verification_after'] == 'COMPLETE', 'formal run incomplete; no candidate evaluation')
        require(report['plan_sha256'] == bind(args.plan) and plan['schema'] == 'd1-objective81-preregistered-v2', '81 plan identity')
        require(report['global_candidate_accepted'] is False and report['complete_model_eligibility'] == 'NOT_REQUALIFIED', 'qualification boundary')
        n_sources = 0
        for rel, digest in plan['source_sha256'].items():
            if rel.endswith(('.py', '.xml')) or rel.startswith('training/configs/'):
                require(bind(ROOT / rel) == digest, 'frozen source changed: ' + rel)
                n_sources += 1
        audit['verified_source_files'] = n_sources
        for rel, digest in report['raw_artifacts_sha256'].items():
            require(bind(output / rel) == digest, 'raw artifact identity: ' + rel)
        audit['verified_raw_artifacts'] = len(report['raw_artifacts_sha256'])
        audit['seals'] = []
        for name, digest in report['seals'].items():
            require(bind(output / name) == digest, 'original seal identity: ' + name)
            members = read(output / name)['artifact_sha256']
            for member, expected in members.items():
                require(bind(output / member) == expected, 'sealed member changed: ' + member)
            audit['seals'].append({'name': name, 'sha256': digest, 'members': len(members)})

        points, audit['collections'] = {}, []
        for role, seeds, ticks in (('development', D_SEEDS, STEADY), ('validation', V_SEEDS, TICKS)):
            require(len(report[role]) == 3 and {r['seed'] for r in report[role]} == set(seeds), 'collection role grid')
            for row in report[role]:
                seed = row['seed']
                require(row['status'] == 'COMPLETE' and row['task'] == 'turn_stop' and row['steps'] == 600, 'full collection required')
                folder = output / role / f'turn_stop_{seed}'
                require(read(folder / 'receipt.json') == row, 'collection receipt differs')
                require(bind(folder / 'full_or_partial_trace.npz') == row['trace_sha256'], 'collection trace SHA')
                bind(folder / 'point_arrays.npz')
                bind(folder / 'integration_context_archive.pkl')
                equal(dict(zip(COUNTERS, (600, 6000, 3000), strict=True)), row['actual_counts'], 'collection work')
                require(row['points'] == [{'seed': seed, 'tick': t, 'task': 'turn_stop'} for t in ticks], 'collection point order')
                with zipfile.ZipFile(folder / 'full_or_partial_trace.npz') as raw, zipfile.ZipFile(folder / 'point_arrays.npz') as ctx:
                    states = array(raw, 's', (600, 45))[0]
                    actions = array(raw, 'issued_action', (600, 16))[0]
                    commands = array(raw, 'command_during_interval', (600, 3))[0]
                    require(actions == array(raw, 'a', (600, 16))[0] == array(raw, 'applied_action', (600, 16))[0], 'collection action seam')
                    require(array(raw, 'tick', (600,))[0] == tuple(range(600)), 'collection ticks')
                    require(array(raw, 'done', (600,))[0] == (False,) * 599 + (True,), 'collection episode end')
                    for flag in ('fallen', 'illegal_body_contact', 'terminated', 'truncated'):
                        require(not any(array(raw, flag, (600,))[0]), 'collection unsafe')
                    for tick in range(600):
                        wz = (.2 if tick < 250 else -.2) if 100 <= tick < 400 else 0.
                        require(commands[tick * 3:(tick + 1) * 3] == (0., 0., wz), 'original turn command schedule')
                    for index, tick in enumerate(ticks):
                        point = {key: array(ctx, f'{index}_{key}', dims)[0] for key, dims in
                                 (('state', (45,)), ('last', (16,)), ('commands', (33, 3)))}
                        point['nominal_height'] = array(ctx, f'{index}_nominal_height', ())[0][0]
                        require(point['state'] == states[tick * 45:(tick + 1) * 45] and
                                point['last'] == actions[(tick - 1) * 16:tick * 16] and
                                point['commands'] == commands[tick * 3:(tick + 33) * 3], 'snapshot pre-action alignment')
                        points[(seed, tick)] = point
                audit['collections'].append({'role': role, 'seed': seed, 'steps': 600, 'point_ticks': ticks})
        require(len(points) == 21, 'six D plus15 V numerical contexts required')

        branches, predictions = {}, {}
        audit['physical_traces'], audit['prediction_traces'], audit['anchor_replays'] = [], [], []
        for row in report['short_branches']:
            identity = tuple(row[k] for k in ('seed', 'tick', 'model_seed', 'arm', 'repeat', 'scale_index', 'sign'))
            require(identity not in branches and row['status'] == 'COMPLETE', 'partial/duplicate physical branch')
            equal({'steps': 32, 'native_steps': 320, 'pd_updates': 160,
                   'actual_counts': dict(zip(COUNTERS, (32, 320, 160), strict=True))}, row, 'branch counts')
            path = output / row['trace_path']
            require(bind(path) == row['trace_sha256'] and read(path.with_suffix('.json')) == row, 'branch bindings')
            with zipfile.ZipFile(path) as archive:
                metrics, gaps = derived_trace(archive, points[identity[:2]], False)
                first_input = array(archive, 'first_actor_input', (627,))[0]
                require(helper80.raw_identity(archive) == row['trace_identity_sha256'], 'physical array identity')
                if row['arm'] == 'anchor' and row['repeat'] == 1:
                    other = branches[(row['seed'], row['tick'], None, 'anchor', 0, None, 0)]['source']
                    with zipfile.ZipFile(output / other['trace_path']) as first:
                        require(set(first.namelist()) == set(archive.namelist()) and
                                all(first.read(k) == archive.read(k) for k in first.namelist()), 'anchor member-byte parity')
                    audit['anchor_replays'].append({'seed': row['seed'], 'tick': row['tick'], 'all_npy_members_bit_exact': True})
            equal(metrics, row, 'independent short metrics')
            branches[identity] = {**metrics, 'source': row, 'first_input': first_input}
            audit['physical_traces'].append({'identity': identity, **gaps, 'cost': metrics['cost'], 'yaw_mae': metrics['yaw_mae']})
        require(len(audit['anchor_replays']) == 15, '15 duplicate anchor sites')
        for row in report['predictions']:
            identity = tuple(row[k] for k in ('model_seed', 'arm', 'scale_index', 'sign', 'seed', 'tick'))
            require(identity not in predictions, 'duplicate prediction')
            path = output / row['trace_path']
            require(bind(path) == row['trace_sha256'], 'prediction trace SHA')
            with zipfile.ZipFile(path) as archive:
                metrics, gaps = derived_trace(archive, points[(row['seed'], row['tick'])], True)
                first_input = array(archive, 'actor_inputs', (32, 627))[0][:627]
            equal(metrics, row, 'independent prediction scalars')
            predictions[identity] = {**metrics, 'source': row, 'first_input': first_input}
            audit['prediction_traces'].append({'identity': identity, **gaps, 'cost': metrics['cost'], 'yaw_mae': metrics['yaw_mae']})
        expected_predictions = {(m, a, k, sign, s, t) for m in MODELS for s in V_SEEDS for t in TICKS
            for a, k, sign in [('anchor', None, 0)] + [(a, k, sign) for a in ARMS
                for k in (range(4) if t in STEADY else (0,)) for sign in ((1, -1) if t in STEADY else (1,))]}
        require(set(predictions) == expected_predictions and len(predictions) == 387, '387 prediction grid')
        screen = q_screen(branches)
        equal(screen, report['screen'], 'raw independent screen')
        audit['independent_screen'] = screen
        audit['physical_reduction_max_gaps'] = {key: max(r[key] for r in audit['physical_traces']) for key in
            ('max_cost64_gap', 'max_cost_component_gap', 'raw_gyro_vs_state28_max_gap')}
        audit['prediction_reduction_max_gaps'] = {key: max(r[key] for r in audit['prediction_traces']) for key in
            ('max_cost64_gap', 'max_cost_component_gap')}

        # FD values are exact Fraction reductions of saved, independently checked
        # scalar prefix means; this does not pretend to bound their full chain error.
        cross = read(output / 'cross_readback.json')
        require(cross == report['cross_readback'] and len(cross['responses']) == 288, 'cross-prefix response grid')
        seen = set()
        for row in cross['responses']:
            model, arm, seed, k, h = (row[n] for n in ('model_seed', 'arm', 'seed', 'scale_index', 'horizon'))
            key = (model, arm, seed, k, h)
            require(key not in seen and model in MODELS and arm in ARMS and seed in V_SEEDS and k in range(4) and h in PREFIXES,
                    'cross response identity')
            seen.add(key)
            require(row['h'] == SCALES[k], 'cross nominal scale')
            for label in ('model', 'physical'):
                values = []
                for selected_arm, selected_k, sign in (('anchor', None, 0), (arm, k, 1), (arm, k, -1)):
                    rr = [predictions[(model, selected_arm, selected_k, sign, seed, tick)]['source'] if label == 'model'
                          else branches[(seed, tick, None if selected_arm == 'anchor' else model,
                                         selected_arm, 0, selected_k, sign)]['source'] for tick in STEADY]
                    prefix = [r['prefix_metrics'][str(h)] for r in rr]
                    values.append({'C': sum(p['cost'] for p in prefix) / 2, 'Q': max(p['yaw_mae'] for p in prefix)})
                for metric in ('C', 'Q'):
                    zero, plus, minus = [v[metric] for v in values]
                    z, p, m, step = map(Fraction.from_float, (zero, plus, minus, SCALES[k]))
                    expected = {'zero': zero, 'plus': plus, 'minus': minus, 'plus_delta': float(p - z),
                        'minus_delta': float(m - z), 'central': float((p - m) / (2 * step)),
                        'forward': float((p - z) / step), 'backward': float((z - m) / step),
                        'strong_monotone': plus < zero < minus, 'stored_plus_equals_zero': plus == zero,
                        'stored_minus_equals_zero': minus == zero}
                    require(all(row[label][metric][n] == v for n, v in expected.items()), 'cross stored scalar FD differs')
            require(all(v is None for v in row['gain_ratio'].values()), 'unqualified gain ratio reported')
        require(len(cross['actual_action_response']) == 342, 'actual nonanchor first-action count')
        input_sites = set()
        for row in cross['actual_action_response']:
            model, arm, k, sign, seed, tick = (row[n] for n in ('model_seed', 'arm', 'scale_index', 'sign', 'seed', 'tick'))
            key = (seed, tick, model, arm, 0, k, sign)
            require(key not in input_sites, 'duplicate first-input diagnostic')
            input_sites.add(key)
            zero = branches[(seed, tick, None, 'anchor', 0, None, 0)]['first_input']
            actual = branches[key]['first_input']
            predicted = predictions[(model, arm, k, sign, seed, tick)]['first_input']
            require(actual == zero and row['physical_first_actor_input_bit_exact'] is True, 'different physical first actor input')
            gap = max(abs(a - b) for a, b in zip(predicted, actual, strict=True))
            require(gap == row['model_physical_first_actor_input_max_abs'], 'model/physical first-input gap differs')
        require(len(input_sites) == 342, 'all actual probe first inputs accounted')
        audit['same_physical_first_inputs_verified'] = len(input_sites)
        audit['cross_prefix_Fraction_reductions_verified'] = len(seen) * 2 * 2

        baseline, candidates = {}, {}
        audit['evaluation_episodes'] = []
        for role, indexed in (('baseline', baseline), ('candidates', candidates)):
            for row in report[role]:
                require(row['status'] == 'COMPLETE', 'partial H episode')
                path = output / row['trace_path']
                require(bind(path) == row['trace_sha256'] == row['metrics']['trace_sha256'], 'H trace binding')
                require(read(path.with_suffix('.json')) == row, 'H JSON mismatch')
                metrics = helper.recompute_trace(path, row['task'], row['seed'], row['metrics']['done_reason'])
                equal(metrics, row['metrics'], 'H independent original metrics')
                key = (row['task'], row['seed']) if role == 'baseline' else (row['model_seed'], row['arm'], row['task'], row['seed'])
                require(key not in indexed, 'duplicate H episode')
                indexed[key] = metrics
                audit['evaluation_episodes'].append({'identity': key, 'role': role, 'metrics': metrics})
        if screen['passed']:
            comparison = full_decision(baseline, candidates)
            equal(comparison, report['comparison'], 'independent full H decision')
            require(report['outcome'] == 'FULL_EVALUATION_COMPLETE' and report['evaluation_status'] == 'COMPLETE', 'H outcome')
            audit['independent_comparison'] = comparison
        else:
            require(not baseline and not candidates and report['evaluation_status'] == report['comparison']['status'] == 'NOT_RUN'
                    and report['outcome'] == 'SCREEN_REJECTED' and not (output / 'evaluation').exists(), 'H touched after rejection')
            audit['independent_comparison'] = {'status': 'NOT_RUN', 'baseline_episodes_not_run': 9,
                'candidate_episodes_not_run': 54, 'control_gate_evaluated': False}
        require(report['objective_passed'] is (screen['passed'] and audit['independent_comparison']['objective_passed']),
                'overall objective flag differs')

        physics = journal(output / 'all_physics_steps.jsonl')
        phases = collections.Counter()
        for i, row in enumerate(physics, 1):
            equal(dict(zip(COUNTERS, (i, i * 10, i * 5), strict=True)), row['counts'], 'monotonic physical counts')
            require(row['issued_action'] == row['applied_action'], 'journal delay0 seam')
            require(not any(row[k] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated')), 'unsafe journal row')
            phases[row['phase']] += 1
        expected_steps = 53304 if screen['passed'] else 15504
        require(len(physics) == expected_steps, 'formal policy budget')
        totals = {name: sum(n for phase, n in phases.items() if phase.startswith(prefix)) for name, prefix in
            (('development', 'development/'), ('validation', 'validation/'), ('screen', 'screen/'), ('evaluation', 'evaluate/'))}
        require(totals == {'development': 1800, 'validation': 1800, 'screen': 11904, 'evaluation': expected_steps - 15504}, 'phase counts')
        equal(dict(zip(COUNTERS, (expected_steps, expected_steps * 10, expected_steps * 5), strict=True)), report['actual_counts'], 'formal counts')
        require(report['model_calls'] == 6 * 9 * 6 * 32 + 3 * 6 * 32 + 387 * 32 == 23328, 'model forward count')
        require(report['actor_calls'] == report['model_calls'] + expected_steps + 192, 'actor hook count')
        require(report['policy_network_updates'] == 6 and report['model_updates'] == 0, 'update accounting')
        updates, direction_calls = [], 0
        for model in MODELS:
            for arm in ARMS:
                folder = output / f'seed{model}/actor_{arm}'
                receipt = read(folder / 'receipt.json')
                require(receipt['status'] == receipt['verification_after'] == 'COMPLETE'
                        and receipt['updates_completed'] == receipt['updates_attempted'] == 1, 'complete direction receipt')
                require(receipt['anchor_mapping_sha256'] == report['anchor_mapping_sha256'] and receipt['point_count'] == 6,
                        'common original policy/six D points')
                require(receipt['point_identities'] == [{'seed': s, 'tick': t} for s in D_SEEDS for t in STEADY], 'direction data leakage')
                require(receipt['gradient_points_completed'] == 6 and receipt['fd_points_completed'] == 48 and
                        receipt['model_calls'] == receipt['model_calls_attempted'] == 9 * 6 * 32, 'direction forward ledger')
                direction_calls += receipt['model_calls']
                require(len(receipt['probes']) == 8, 'all four scales retained')
                for probe in receipt['probes']:
                    require(probe['status'] == 'COMPLETE' and all(v <= .01 for v in probe['final_row_l1']), 'probe execution bound')
                    for kind in ('actual', 'ideal'):
                        require(bind(folder / probe[kind + '_file']) == probe[kind + '_file_sha256'], 'probe weight file SHA')
                require(bind(folder / receipt['weights_file']) == receipt['weights_file_sha256'], 'candidate weight SHA')
                events = journal(folder / 'events.jsonl')
                require(sum(e['stage'] == 'main_candidate_constructed' for e in events) == 1, 'main update journal')
                updates.append({'model_seed': model, 'arm': arm, 'updates': 1,
                    'mapping_sha256': receipt['actor_after_mapping_sha256'], 'weights_sha256': receipt['weights_file_sha256']})
        audit['policy_receipts'] = updates
        require(direction_calls + len(report['predictions']) * 32 + 3 * 6 * 32 == report['model_calls'], 'forward ledgers/trace count sum')
        admission = read(Path(report['admission']['path']))
        require(bind(Path(report['admission']['path'])) == report['admission']['sha256'] and
                admission['status'] == admission['verification_after'] == 'COMPLETE' and
                admission['plan_sha256'] == report['plan_sha256'], 'same-plan preflight')
        equal(dict(zip(COUNTERS, (224, 2240, 1120), strict=True)), admission['actual_counts'], 'preflight counts')
        audit['physical_accounting'] = {'formal_policy_steps': expected_steps, 'stages': totals,
            'including_preflight': dict(zip(COUNTERS, (expected_steps + 224, (expected_steps + 224) * 10,
                                                      (expected_steps + 224) * 5), strict=True)),
            'phase_counts': dict(phases), 'model_calls': report['model_calls'], 'actor_calls': report['actor_calls']}
        progress = journal(output / 'progress.jsonl')
        def event_index(name):
            indices = [i for i, r in enumerate(progress) if r['event'] == name]
            require(len(indices) == 1, 'single event required: ' + name)
            return indices[0]
        policy_seal = event_index('ALL_SIX_CANDIDATES_AND_PROBES_SEALED_BEFORE_V')
        prediction_seal = event_index('ALL_PREDICTIONS_SEALED')
        first_v = min(i for i, r in enumerate(progress) if r['phase'].startswith('validation/'))
        first_screen = min(i for i, r in enumerate(progress) if r['phase'].startswith('screen/'))
        require(policy_seal < first_v < prediction_seal < first_screen, 'sealing order')
        if screen['passed']:
            first_h = min(i for i, r in enumerate(progress) if r['phase'].startswith('evaluate/'))
            require(event_index('WHOLE_SCREEN_PASSED') < first_h, 'H before V admission')
        audit['seal_event_indices'] = {'all_policies': policy_seal, 'first_validation': first_v,
                                      'all_predictions': prediction_seal, 'first_screen': first_screen}
        audit['resource_receipt'] = {k: report.get(k) for k in ('cpu_seconds', 'wall_seconds', 'raw_bytes_before_report', 'cleanup_errors')}
        audit['sampling_units'] = {'model_initializations': 3, 'D_environment_resets': 3, 'V_environment_resets': 3,
            'V_sites': 15, 'V_branch_count': 372, 'V_branches_are_not_independent_resets': True,
            'H_environment_resets_if_run': 3, 'shared_H_baseline_episodes_if_run': 9}
        audit['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 -- preserve this audit's failure, never mutate formal data.
        audit['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        changed = [path for path, receipt in files.items() if not Path(path).is_file() or sha(Path(path)) != receipt['sha256']]
        audit['verification_after'] = 'INCOMPLETE' if changed else 'COMPLETE'
        if changed:
            audit.update(status='INCOMPLETE', changed_inputs=changed)
        audit.update(inputs=files, audited_input_files=len(files), cpu_seconds=time.process_time() - started)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(audit, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: audit.get(k) for k in ('status', 'formal_status', 'formal_outcome', 'failure', 'cpu_seconds', 'audited_input_files')}))
    return int(audit['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
