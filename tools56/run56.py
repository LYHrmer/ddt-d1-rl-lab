"""Paired56 horizon8/32 fits and free-model gradient validation, final-only.

Only complete53 training transitions fit the models. Actual future states are
used for diagnostic loss after each free rollout, never to correct its forward
path or its derivative. This runner performs no policy/parameter update and
does not rejudge53/55 or select a best seed, checkpoint, epsilon or horizon.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import pickle
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import make_step_recorder, require, sha, verify, write_json
from tools54.reproduce54 import load_points
from tools55.probe_local55 import EPSILONS, action_delta_ulps, cost64, local_stability
from tools55.run55 import audit_work, collect_probes

MODEL_SEEDS = (56001, 56002, 56003)
PROBE_SEEDS = (56301, 56302, 56303)
HORIZONS = (8, 32)
TICKS = (110, 260, 410)
ROLES = ('development', 'old_probe', 'old55_local', 'fresh')
PHASES = ('admission', 'training', 'pairing', 'historical_numeric', 'fresh_collection',
          'fresh_numeric', 'seal_predictions', 'physical_grid', 'comparison')
ORIGINAL_PLAN = Path('docs/experiments53/model_gradient53_preregistered.json')
SUMMARY53 = Path('reports/model_gradient53_results_20261101.json')
EVIDENCE53 = Path('evidence/model_gradient53')
OLD55 = Path('evidence/observed_gradient55/execution_01')
REFERENCE55 = Path('evidence/observed_gradient55/local_01/report.json')
HARD_CAP = 5000
EXPECTED_STEPS = 4680
PAIR_FIELDS = ('training_data_sha256', 'common_windows_sha256', 'planned_sampling_rows_sha256',
               'completed_sampling_rows_sha256', 'ridge_and_normalization_sha256', 'initial_model_state_sha256')


def verify56(plan):
    require(plan['schema'] == 'd1-paired-horizon56-v1', 'wrong56 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['horizons'] == list(HORIZONS), 'registered arms/seeds changed')
    require(plan['updates_per_model'] == 1000 and plan['expected_policy_steps'] == EXPECTED_STEPS
            and plan['hard_policy_cap'] == HARD_CAP, 'registered work changed')
    require(plan['cpu_seconds'] == 600 and plan['wall_seconds'] == 900, 'time budget changed')
    required = {str(ORIGINAL_PLAN), str(SUMMARY53), str(REFERENCE55), str(OLD55 / 'probe_snapshots.pkl'),
                'tools56/run56.py', 'tools56/fit56.py', 'tools54/numerics54.py', 'tools54/reproduce54.py',
                'tools55/probe_local55.py', 'tools55/run55.py', 'tools53/run_pilot53.py', 'tools53/collection53.py'}
    required.update(str(OLD55 / f'probe_{index:02d}.npz') for index in range(9))
    require(required <= set(plan['source_sha256']), 'required source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '56 source/input changed: ' + name)


def bound53(relative, manifest, consumed):
    """Admit only an original53 raw-manifest entry; rechecked after the run."""
    identity = manifest[relative]
    path = ROOT / EVIDENCE53 / relative
    require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
            'original53 raw identity changed: ' + relative)
    consumed[str(EVIDENCE53 / relative)] = dict(identity)
    return path


def check_points(points, seeds):
    require([(point['seed'], point['tick']) for point in points] ==
            [(seed, tick) for seed in seeds for tick in TICKS], 'point ordering/identity changed')


def reference53(rows):
    """Keep the original53 physical epsilon/identifiability rules unchanged."""
    return [{'seed': row['seed'], 'tick': row['tick'], 'gradient': row['gradient'],
             'resolved': bool(row['identifiable'] and row['repeated_baseline_exact']
                              and all(branch['status'] == 'COMPLETE' for branch in row['branches'])),
             'source': 'original53 epsilon0.005; original0.01 stability check; no rejudging53'} for row in rows]


def local_references(rows):
    """Use every fixed55C central value, with no selected epsilon."""
    references = []
    for row in rows:
        diagnostic = local_stability(row['gradients'])
        resolved = bool(row['status'] == 'COMPLETE' and diagnostic['locally_stable']
                        and row['all_four_wheels_resolved']
                        and all(branch['status'] == 'COMPLETE' for branch in row['branches']))
        references.append({'seed': row['seed'], 'tick': row['tick'],
                           'gradient': math.fsum(g['central64'] for g in row['gradients']) / len(EPSILONS),
                           'central64_all_eps': [g['central64'] for g in row['gradients']],
                           'resolved': resolved, 'source': 'arithmetic mean of all four fixed55C central64 values'})
    return references


def direction_comparison(predictions, references):
    """Unresolved references never count as model direction errors."""
    require(len(predictions) == len(references) == 9, 'nine paired gradients required')
    rows = []
    for predicted, reference in zip(predictions, references, strict=True):
        require((predicted['seed'], predicted['tick']) == (reference['seed'], reference['tick']),
                'gradient/reference identities differ')
        resolved = reference['resolved']
        same_sign = predicted['gradient'] * reference['gradient'] > 0 if resolved else None
        rows.append({'seed': predicted['seed'], 'tick': predicted['tick'],
                     'predicted_gradient': predicted['gradient'], 'reference_gradient': reference['gradient'],
                     'reference_resolved': resolved, 'same_sign': same_sign,
                     'direction_match': bool(same_sign and predicted['direction_identifiable']
                                             and predicted['autograd_agrees']) if resolved else None,
                     'normalized_gradient_error': abs(predicted['gradient'] - reference['gradient'])
                     / max(abs(reference['gradient']), 1e-5) if resolved else None})
    resolved_all = all(row['reference_resolved'] for row in rows)
    return {'points': rows, 'reference_resolved_count': sum(row['reference_resolved'] for row in rows),
            'sign_matches': sum(row['same_sign'] is True for row in rows),
            'direction_matches': sum(row['direction_match'] is True for row in rows),
            'resolved_sign_mismatches': sum(row['same_sign'] is False for row in rows),
            'mean_normalized_gradient_error': math.fsum(row['normalized_gradient_error'] for row in rows) / 9
            if resolved_all else None, 'normalization_floor': 1e-5,
            'all_selfchecks': all(row['autograd_agrees'] for row in predictions),
            'all_model_directions_identifiable': all(row['direction_identifiable'] for row in predictions),
            'passed_9_of_9': resolved_all and all(row['direction_match'] is True for row in rows)}


def check_pairing(models, expected):
    """Assert6 final-only paired fits; horizon is the sole changed fit setting."""
    require(len(models) == 6, 'six complete fits required before evaluation')
    for row in models:
        fitted = row['fit']
        require(fitted['status'] == 'COMPLETE' and fitted['eligible_for_evaluation']
                and fitted['updates'] == fitted['updates_attempted'] == 1000
                and fitted['seed'] == row['seed'] and fitted['horizon'] == row['horizon']
                and fitted['training_transitions'] == 7200 and fitted['training_windows'] == 6828
                and fitted['training_episodes'] == list(range(12)), 'incomplete/nonpaired fit')
        for key, value in expected.items():
            require(fitted[key] == value, 'fit differs from admitted data/sampling: ' + key)
    paired = []
    for seed in MODEL_SEEDS:
        left, right = [next(row for row in models if row['seed'] == seed and row['horizon'] == horizon)
                       for horizon in HORIZONS]
        equal = {key: left['fit'][key] == right['fit'][key] for key in PAIR_FIELDS}
        require(all(equal.values()), 'horizon arms not exactly paired')
        paired.append({'seed': seed, 'equal_by_field': equal})
    for key in PAIR_FIELDS[:-1]:
        require(len({row['fit'][key] for row in models}) == 1, 'shared identity differs: ' + key)
    return {'status': 'COMPLETE', 'pairs': paired, 'common32_windows': 6828, 'only_variable': 'training_horizon'}


def evaluate_points(model, actor, points, targets, output, ledger, deadline):
    """Free32-step model gradients and theta0 traces; truth enters loss only."""
    import numpy as np
    import torch

    from tools52 import dynamics52 as dynamics
    from tools52.run_pilot52 import save_npz
    from tools54 import numerics54 as numeric
    output.mkdir(parents=True, exist_ok=False)
    ledger.update(status='INCOMPLETE', completed_points=0, points=[])
    try:
        require(len(points) == len(targets) == 9, 'nine evaluation points required')
        for index, (point, target) in enumerate(zip(points, targets, strict=True)):
            require(time.process_time() < deadline, 'CPU deadline before numeric point')
            gradient, = numeric.gradients(model, actor, [point])
            with torch.no_grad():
                value, trace = numeric.imaginary(model, actor, point, torch.tensor(0., dtype=torch.float64),
                                                 return_trace=True)
                actual = torch.tensor(np.asarray(target), dtype=torch.float64)
                require(actual.shape == trace['states'].shape == (32, 45)
                        and bool(torch.isfinite(actual).all()), 'complete finite32-step target required')
                error = (trace['states'] - actual) / torch.tensor(dynamics.PHYSICAL_SCALE, dtype=torch.float64)
                groups = {name: float(error[:, list(indices)].square().mean())
                          for name, indices in dynamics.GROUPS.items()}
                group_loss = float(dynamics.group_loss(trace['states'], actual))
            path = output / f'point_{index:02d}.npz'
            save_npz(path, {key: tensor.detach().numpy() for key, tensor in trace.items()}
                     | {'target_states': actual.numpy()})
            row = {**gradient, 'status': 'COMPLETE', 'trace_file': path.name,
                   'trace_sha256': sha(path), 'trace_cost': float(value),
                   'state_group_loss': groups, 'state_group_loss_mean': group_loss,
                   'observed_forward_correction': False, 'training_oracle': False}
            write_json(output / f'point_{index:02d}.json', row)
            ledger['points'].append(row)
            ledger['completed_points'] += 1
        ledger.update(status='COMPLETE', all_selfchecks=all(row['autograd_agrees'] for row in ledger['points']),
                      all_model_directions_identifiable=all(row['direction_identifiable'] for row in ledger['points']),
                      mean_state_group_loss=math.fsum(row['state_group_loss_mean'] for row in ledger['points']) / 9,
                      mean_state_group_losses={name: math.fsum(row['state_group_loss'][name]
                          for row in ledger['points']) / 9 for name in dynamics.GROUPS})
    except BaseException as exc:
        ledger['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(output / 'result.json', ledger)


def physical_grid(env, actor, points, baselines, output, rows, progress, buffer, deadline):
    """Save every branch and any interrupted prefix before checking its gates."""
    import numpy as np

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52 as control
    from tools52.run_pilot52 import save_npz
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    for index, point in enumerate(points):
        row, first_zero = rows[index], None
        row['status'] = 'INCOMPLETE'
        for order, theta in enumerate(sequence):
            require(time.process_time() < deadline, 'CPU deadline before physical branch')
            progress.update(point=index, branch=order, theta=theta)
            buffer['rows'] = []
            before = env.counters()
            stem = output / f'grid_{index:02d}_{order:02d}'
            branch = {'status': 'INCOMPLETE', 'theta': theta, 'steps': 0}
            arrays = None
            try:
                branch, states, observations, rewards = control.real_branch(
                    env, ReferenceShaper(), actor, point, theta)
                arrays = {'states': np.asarray(states), 'observations': np.asarray(observations),
                          'rewards': np.asarray(rewards)}
                save_npz(stem.with_suffix('.npz'), arrays)
                branch.update(counts_before=before, counts_after=env.counters(),
                              actual_float32_theta=float(np.float32(theta)),
                              unsafe_steps=[entry[3] for entry in buffer['rows'] if any(entry[3].values())])
                if branch['status'] == 'COMPLETE':
                    value, costs = cost64(point, states, observations)
                    branch.update(cost64=value, per_step_cost64=costs)
                    if order == 0:
                        first_zero = arrays
                        branch['collected_baseline_exact'] = bool(np.array_equal(states, baselines[index]))
                    elif order == 1:
                        branch['repeated_zero_exact'] = all(np.array_equal(value, first_zero[key])
                                                           for key, value in arrays.items())
                    else:
                        branch['first_step_action_delta'] = action_delta_ulps(
                            first_zero['observations'][0, 41:57], arrays['observations'][0, 41:57])
            except BaseException as exc:
                branch.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}',
                              counts_before=before, counts_after=env.counters())
                if arrays is None:
                    branch['steps'] = len(buffer['rows'])
                    save_npz(stem.with_suffix('.npz'), {
                        'states': np.asarray([entry[0] for entry in buffer['rows']]).reshape(-1, 45),
                        'observations': np.asarray([entry[1] for entry in buffer['rows']]).reshape(-1, 627),
                        'rewards': np.asarray([entry[2] for entry in buffer['rows']])})
                raise
            finally:
                row['branches'][order] = branch
                write_json(stem.with_suffix('.json'), branch)
                buffer['rows'] = None
            require(branch['status'] == 'COMPLETE' and not branch['unsafe_steps'], 'incomplete/unsafe physical branch')
            if order == 0:
                require(branch['collected_baseline_exact'], 'zero branch differs from fresh collected baseline')
            elif order == 1:
                require(branch['repeated_zero_exact'], 'repeated zero branch differs')
        zero = row['branches'][0]['cost64']
        for index_epsilon, epsilon in enumerate(EPSILONS):
            plus, minus = row['branches'][2 + 2 * index_epsilon:4 + 2 * index_epsilon]
            row['gradients'].append({'epsilon': epsilon,
                'central64': (plus['cost64'] - minus['cost64']) / (2 * epsilon),
                'central32': (plus['cost'] - minus['cost']) / (2 * epsilon),
                'forward64': (plus['cost64'] - zero) / epsilon,
                'backward64': (zero - minus['cost64']) / epsilon})
        row['local_reference_diagnostic'] = local_stability(row['gradients'])
        row['all_four_wheels_resolved'] = all(branch['first_step_action_delta']['all_four_wheels_nonzero']
                                              for branch in row['branches'][2:])
        row['status'] = 'COMPLETE'
        write_json(output / f'physical_point_{index:02d}.json', row)


def comparisons(models, references):
    """Separate complete execution, reference admission, strict gates and pairing."""
    result = {'models': [], 'paired': [], 'no_policy_update': True, 'selected_horizon': None}
    by_key = {}
    for model in models:
        row = {'seed': model['seed'], 'horizon': model['horizon'],
               'all_selfchecks': all(model['diagnostics'][role]['all_selfchecks'] for role in ROLES),
               'all_model_directions_identifiable': all(model['diagnostics'][role]['all_model_directions_identifiable']
                                                        for role in ROLES)}
        for role, reference in references.items():
            row[role] = direction_comparison(model['diagnostics'][role]['points'], reference)
        row['direction_gate'] = row['all_selfchecks'] and row['all_model_directions_identifiable'] \
            and row['old_probe']['passed_9_of_9'] \
            and row['old55_local']['passed_9_of_9'] and row['fresh']['passed_9_of_9']
        result['models'].append(row)
        by_key[(row['horizon'], row['seed'])] = row
    for seed in MODEL_SEEDS:
        left, right = (by_key[(horizon, seed)] for horizon in HORIZONS)
        pair = {'seed': seed, 'by_role': {}, 'old55_correct_h8_regressions': []}
        for role in references:
            a, b = left[role], right[role]
            pair['by_role'][role] = {
                'h8_sign_matches': a['sign_matches'], 'h32_sign_matches': b['sign_matches'],
                'sign_matches_delta_h32_minus_h8': b['sign_matches'] - a['sign_matches'],
                'h8_direction_matches': a['direction_matches'], 'h32_direction_matches': b['direction_matches'],
                'direction_matches_delta_h32_minus_h8': b['direction_matches'] - a['direction_matches'],
                'h8_mean_normalized_gradient_error': a['mean_normalized_gradient_error'],
                'h32_mean_normalized_gradient_error': b['mean_normalized_gradient_error'],
                'normalized_gradient_error_ratio_h32_over_h8':
                    b['mean_normalized_gradient_error'] / a['mean_normalized_gradient_error']
                    if a['mean_normalized_gradient_error'] not in (None, 0.)
                    and b['mean_normalized_gradient_error'] is not None else None}
        for a, b in zip(left['old55_local']['points'], right['old55_local']['points'], strict=True):
            if a['direction_match'] is True and b['direction_match'] is not True:
                pair['old55_correct_h8_regressions'].append({'seed': a['seed'], 'tick': a['tick']})
        pair['state_group_loss'] = {}
        for role in ROLES:
            diagnostics = [next(row for row in models if row['horizon'] == horizon and row['seed'] == seed)
                           ['diagnostics'][role] for horizon in HORIZONS]
            a, b = diagnostics
            pair['state_group_loss'][role] = {
                'h8': a['mean_state_group_loss'], 'h32': b['mean_state_group_loss'],
                'ratio_h32_over_h8': b['mean_state_group_loss'] / a['mean_state_group_loss']
                    if a['mean_state_group_loss'] else None,
                'h8_groups': a['mean_state_group_losses'], 'h32_groups': b['mean_state_group_losses']}
        result['paired'].append(pair)
    reference_admitted = all(row['resolved'] for row in references['fresh'])
    h8 = all(row['direction_gate'] for row in result['models'] if row['horizon'] == 8)
    h32 = all(row['direction_gate'] for row in result['models'] if row['horizon'] == 32)
    no_regression = all(not row['old55_correct_h8_regressions'] for row in result['paired'])
    deltas = [row['by_role']['fresh']['direction_matches_delta_h32_minus_h8'] for row in result['paired']]
    result.update(fresh_reference_resolved_count=sum(row['resolved'] for row in references['fresh']),
                  fresh_reference_admitted=reference_admitted, h8_direction_gate=h8,
                  h32_direction_gate=h32, h32_old55_no_regression=no_regression,
                  primary_passed=reference_admitted and h32 and no_regression,
                  primary_status='REFERENCE_UNRESOLVED' if not reference_admitted
                  else 'PASSED' if h32 and no_regression else 'FAILED',
                  direction_improved=reference_admitted and no_regression
                  and all(row['all_selfchecks'] and row['all_model_directions_identifiable'] for row in result['models'])
                  and all(delta >= 0 for delta in deltas) and any(delta > 0 for delta in deltas),
                  direction_improvement_scope='paired fresh valid direction matches; no best-seed selection')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-paired-horizon56-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'policy_updates': 0, 'control_parameter_updates': 0, 'model_updates': 0,
              'model_updates_attempted': 0, 'completed_model_fits': 0, 'counts': None,
              'original53_decision_changed': False, 'original55_decision_changed': False,
              'training_physics_steps': 0, 'retries_or_resume': False, 'consumed53_inputs': {},
              'phases': {name: {'status': 'NOT_RUN'} for name in PHASES},
              'models': [{'seed': seed, 'horizon': horizon, 'status': 'NOT_RUN',
                          'directory': f'h{horizon}_seed{seed}',
                          'fit': {'status': 'NOT_RUN', 'updates': 0, 'updates_attempted': 0},
                          'diagnostics': {role: {'status': 'NOT_RUN', 'completed_points': 0, 'points': []}
                                          for role in ROLES}}
                         for seed in MODEL_SEEDS for horizon in HORIZONS],
              'collection': {'status': 'NOT_RUN'}, 'pairing': {'status': 'NOT_RUN'},
              'comparison': {'status': 'NOT_RUN'}, 'physical_probes': [
                  {'seed': seed, 'tick': tick, 'status': 'NOT_RUN', 'gradients': [],
                   'branches': [{'theta': theta, 'status': 'NOT_RUN'} for theta in sequence]}
                  for seed in PROBE_SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress, buffer = {'phase': 'admission'}, {'rows': None}
    sealed_artifacts, trained = {}, {}
    events = (args.output / 'phase_events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, 'phase': progress['phase'],
                                'cpu_seconds': time.process_time() - cpu,
                                'model_updates': report['model_updates'],
                                'completed_model_fits': report['completed_model_fits'],
                                'counts': None if env is None else env.counters()}, allow_nan=False) + '\n')
        events.flush()

    def begin(name):
        require(time.process_time() < 598., 'CPU deadline before phase ' + name)
        progress['phase'] = name
        report['phases'][name] = {'status': 'INCOMPLETE'}
        event('PHASE_BEGIN')

    def complete():
        report['phases'][progress['phase']]['status'] = 'COMPLETE'
        event('PHASE_COMPLETE')

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (599, 600))
        begin('admission')
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify56(plan)
        original = json.loads((ROOT / ORIGINAL_PLAN).read_text())
        verify(original)
        summary = json.loads((ROOT / SUMMARY53).read_text())
        require(summary['formal']['status'] == 'COMPLETE'
                and summary['plan_sha256'] == sha(ROOT / ORIGINAL_PLAN), 'complete bound53 evidence required')
        manifest, consumed = summary['raw_manifest'], report['consumed53_inputs']
        collection_path = bound53('execution_01/collection.json', manifest, consumed)
        collected = json.loads(collection_path.read_text())
        require(collected == summary['formal']['collection'] and collected['status'] == 'COMPLETE',
                'original53 collection evidence differs/incomplete')
        for role in ('train', 'development', 'probe'):
            cases = [case for case in collected['cases'] if case['role'] == role]
            require([case['seed'] for case in cases] == original['data_seeds'][role]
                    and all(case['status'] == 'COMPLETE' and case['steps'] == 600
                            and case['complete_12s_trial'] and not case['termination_reasons'] for case in cases),
                    'incomplete original53 role: ' + role)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'], 'registered package versions differ')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        from tools56 import fit56
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report['torch_runtime_version'] = torch.__version__
        require(torch.__version__ == summary['formal']['torch_runtime_version'], 'torch runtime differs')
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        actor64 = copy.deepcopy(actor32).double().eval().requires_grad_(False)
        train_path = bound53('execution_01/train_transitions.npz', manifest, consumed)
        with np.load(train_path, allow_pickle=False) as raw:
            train = dynamics.validate_data({key: raw[key].copy() for key in fit56.DATA_KEYS})
        require(len(train['s']) == 7200 and np.array_equal(np.unique(train['episode']), np.arange(12))
                and all(np.count_nonzero(train['episode'] == episode) == 600 for episode in range(12)),
                'training must contain exactly12 complete original600-step episodes')
        common = fit56.common_window_indices(train).astype('<i8', copy=False)
        require(common.shape == (6828, 32), 'common32 window count differs')
        sampling_sha = fit56.array_sha256(fit56.sampled_window_rows(len(common)).numpy().astype('<i8', copy=False))
        expected = {'training_data_sha256': fit56.mapping_sha256(train),
                    'common_windows_sha256': fit56.array_sha256(common),
                    'planned_sampling_rows_sha256': sampling_sha, 'completed_sampling_rows_sha256': sampling_sha}
        report['training_admission'] = {**expected, 'transitions': 7200, 'episodes': 12,
                                        'common32_windows': 6828, 'source': str(EVIDENCE53 / 'execution_01/train_transitions.npz')}
        complete()
        begin('training')
        for row in report['models']:
            row['status'] = 'INCOMPLETE'
            row['fit']['status'] = 'INCOMPLETE'
            model, row['fit'] = fit56.fit_model(train, row['seed'], row['horizon'],
                                               args.output / row['directory'], 598.)
            trained[(row['horizon'], row['seed'])] = model
            weight = args.output / row['directory'] / 'final_state_dict.pt'
            sealed_artifacts[str(weight.relative_to(args.output))] = sha(weight)
            report['model_updates'] += row['fit']['updates']
            report['model_updates_attempted'] += row['fit']['updates_attempted']
            report['completed_model_fits'] += 1
            event('FIT_COMPLETE')
        require(report['model_updates'] == 6000 and report['completed_model_fits'] == 6, 'all6 fits must complete first')
        complete()
        begin('pairing')
        report['pairing'] = check_pairing(report['models'], expected)
        write_json(args.output / 'pairing.json', report['pairing'])
        complete()
        begin('historical_numeric')
        historical_points, targets = {}, {}
        for role, original_role in (('development', 'development'), ('old_probe', 'probe')):
            bound53(f'execution_01/{original_role}_snapshots.pkl', manifest, consumed)
            points = load_points(original_role, manifest)
            check_points(points, original['data_seeds'][original_role])
            historical_points[role], targets[role] = points, []
            for point in points:
                path = bound53(f'execution_01/collection_{original_role}_{point["seed"]}.npz', manifest, consumed)
                with np.load(path, allow_pickle=False) as data:
                    require(np.array_equal(point['state'], data['s'][point['tick']]), 'old snapshot and actual state differ')
                    targets[role].append(data['target'][point['tick']:point['tick'] + 32].copy())
        # This is a locally generated, explicitly SHA-bound artifact, never an
        # untrusted pickle or an incomplete substitute for environment restore.
        with (ROOT / OLD55 / 'probe_snapshots.pkl').open('rb') as stream:
            stored = pickle.load(stream)
        historical_points['old55_local'] = [point | {'shaper': SimpleNamespace(**point['python_state']['shaper'])}
                                             for point in stored]
        check_points(historical_points['old55_local'], (55301, 55302, 55303))
        targets['old55_local'] = []
        for index in range(9):
            with np.load(ROOT / OLD55 / f'probe_{index:02d}.npz', allow_pickle=False) as data:
                targets['old55_local'].append(data['states_0'].copy())
        local55 = json.loads((ROOT / REFERENCE55).read_text())
        require(local55['status'] == 'COMPLETE', 'old55C diagnostic incomplete')
        references = {'old_probe': reference53(summary['formal']['physical_probes']),
                      'old55_local': local_references(local55['points'])}
        require(len(references['old55_local']) == 9 and all(row['resolved'] for row in references['old55_local']),
                'old55C references must retain their admitted whole-grid result')
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['horizon'], row['seed'])]).double().eval().requires_grad_(False)
            for role in ROLES[:-1]:
                evaluate_points(model64, actor64, historical_points[role], targets[role],
                                args.output / row['directory'] / role, row['diagnostics'][role], 598.)
        complete()
        begin('fresh_collection')
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        base_recorder = make_step_recorder(audit, progress)

        def recorder(current, action, result):
            if buffer['rows'] is not None:
                obs, reward, terminated, truncated, info = result
                buffer['rows'].append((control.physical(current), np.asarray(obs).copy(), float(reward),
                    {'terminated': bool(terminated), 'truncated': bool(truncated),
                     'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact'])}))
            base_recorder(current, action, result)
            audit.flush()

        env = collection.make_env(HARD_CAP, audit_step=recorder)
        report['collection'] = {'status': 'INCOMPLETE'}
        fresh_points, fresh_targets, report['collection'] = collect_probes(env, actor32, args.output, plan, original)
        check_points(fresh_points, PROBE_SEEDS)
        require(env.counters()['policy_transitions'] == 1800, 'fresh baseline work differs')
        complete()
        begin('fresh_numeric')
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['horizon'], row['seed'])]).double().eval().requires_grad_(False)
            evaluate_points(model64, actor64, fresh_points, fresh_targets, args.output / row['directory'] / 'fresh',
                            row['diagnostics']['fresh'], 598.)
            row['status'] = 'COMPLETE'
        complete()
        begin('seal_predictions')
        prediction_path = args.output / 'predictions_before_fresh_fd.json'
        write_json(prediction_path, {'models': report['models'], 'pairing': report['pairing'],
                                    'fresh_fd_results_seen': False, 'observed_forward_correction': False,
                                    'policy_updates': 0, 'checkpoint_selection': 'final_only'})
        report['predictions_sha256'] = sha(prediction_path)
        sealed_artifacts[prediction_path.name] = report['predictions_sha256']
        complete()
        begin('physical_grid')
        physical_grid(env, actor32, fresh_points, fresh_targets, args.output, report['physical_probes'], progress, buffer, 598.)
        require(env.counters()['policy_transitions'] == EXPECTED_STEPS, 'final physical work differs')
        complete()
        begin('comparison')
        references['fresh'] = local_references(report['physical_probes'])
        report['references'] = references
        report['comparison'] = {'status': 'COMPLETE', **comparisons(report['models'], references)}
        write_json(args.output / 'comparison.json', report['comparison'])
        complete()
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - every partial remains ineligible and cannot resume.
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['phases'][progress['phase']]['failure'] = report['failure']
        event('INCOMPLETE')
    finally:
        cleanup = []
        report['last_phase'] = dict(progress)
        # Read durable receipts, including interrupted fits, instead of treating
        # a missing return from fit_model as zero completed updates.
        for row in report['models']:
            directory = args.output / row['directory']
            for filename in ('partial_receipt.json', 'fit_receipt.json'):
                path = directory / filename
                if path.exists():
                    try:
                        row['fit'] = json.loads(path.read_text())
                    except (OSError, ValueError) as exc:
                        cleanup.append(f'fit receipt {row["directory"]}/{filename}: {exc}')
                    break
            journal = directory / 'updates.jsonl'
            completed_updates = 0
            if journal.exists():
                try:
                    with journal.open() as stream:
                        for line in stream:
                            entry = json.loads(line)
                            require(entry['update'] == completed_updates + 1, 'nonsequential completed update journal')
                            completed_updates += 1
                    require(completed_updates == row['fit'].get('updates', 0), 'receipt/update journal differ')
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    cleanup.append(f'fit update journal {row["directory"]}: {exc}')
            row['journaled_completed_updates'] = completed_updates
            # A missing/truncated receipt must not hide a flushed completed
            # update prefix. It remains partial, regardless of weight files.
            if row['fit'].get('updates', 0) < completed_updates:
                row['fit'].update(status='INCOMPLETE', updates=completed_updates,
                                  updates_attempted=max(row['fit'].get('updates_attempted', 0), completed_updates))
        report['model_updates'] = sum(row['fit'].get('updates', 0) for row in report['models'])
        report['model_updates_attempted'] = sum(row['fit'].get('updates_attempted', 0) for row in report['models'])
        report['completed_model_fits'] = sum(row['fit']['status'] == 'COMPLETE' for row in report['models'])
        report['update_count_semantics'] = 'completed optimizer steps from durable receipts; interrupted work is a lower bound'
        if (args.output / 'collection.json').exists():
            try:
                report['collection'] = json.loads((args.output / 'collection.json').read_text())
            except (OSError, ValueError) as exc:
                cleanup.append(f'collection receipt: {exc}')
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup errors as incomplete.
                cleanup.append(f'env: {exc}')
        if audit is not None:
            try:
                audit.close()
                report['work_journal'] = audit_work(args.output / 'all_physics_steps.jsonl')
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'], 'work journal differs')
                if report['status'] == 'COMPLETE':
                    require(report['work_journal']['by_phase'] == {'fresh_collection': 1800, 'physical_grid': 2880},
                            'phase work differs')
            except Exception as exc:  # noqa: BLE001 - counts and journal prefixes both remain available.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify56(plan)
            if original is not None:
                verify(original)
            for name, identity in report['consumed53_inputs'].items():
                require((ROOT / name).stat().st_size == identity['bytes'] and sha(ROOT / name) == identity['sha256'],
                        'consumed input changed: ' + name)
            for name, digest in sealed_artifacts.items():
                require(sha(args.output / name) == digest, 'sealed output changed: ' + name)
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - keep the original failure and verification failure.
            report['verification_after'] = 'FAILED'
            cleanup.append(f'verification: {exc}')
        if cleanup:
            report.update(status='INCOMPLETE', cleanup_errors=cleanup)
        report['sealed_artifacts_sha256'] = sealed_artifacts
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'model_updates', 'completed_model_fits',
                                                      'counts', 'cpu_seconds')}, indent=2), flush=True)
    # A complete negative result is an eligible result, never a failed command.
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
