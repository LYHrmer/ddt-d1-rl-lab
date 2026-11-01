"""Paired64 mixed-sign supervision sources with unchanged physical gates.

The sole fit intervention is the source of the second48 auxiliary rows. Both
arms retain all58 rows, fixed59 lambda and original53 state training. All models
finish before heldout evaluation; old59 failures remain a descriptive regression
appendix.62 stays rejected under its original coverage hypothesis.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import make_step_recorder, require, sha, verify, write_json
from tools55.probe_local55 import EPSILONS
from tools55.run55 import audit_work, collect_probes
from tools56.run56 import (
    bound53,
    check_points,
    evaluate_points,
    local_references,
    physical_grid,
)
from tools59.run59 import (
    cost_comparison,
    evaluate_first_steps,
    final_training_label_diagnostics,
    first_references,
    first_step_comparison,
    historical_inputs,
    load_grid,
    load_own_points,
    refresh_fit_counts,
)

MODEL_SEEDS = (64001, 64002, 64003)
PROBE_SEEDS = (64301, 64302, 64303)
ARMS = ('control', 'active')
FIXED_LAMBDA = 0.08980082414438752
ROLES = ('development', 'old_probe', 'old55_local', 'old56', 'fresh')
COST_ROLES = ROLES[1:]
FIRST_ROLES = ('old56', 'fresh')
TICKS = (110, 260, 410)
PLAN53 = Path('docs/experiments53/model_gradient53_preregistered.json')
SUMMARY53 = Path('reports/model_gradient53_results_20261101.json')
OLD55 = Path('evidence/observed_gradient55/execution_01')
LOCAL55 = Path('evidence/observed_gradient55/local_01/report.json')
ADMISSION55 = Path('evidence/observed_gradient55/admission_01/report.json')
OLD56 = Path('evidence/paired_horizon56/execution_01')
LABEL58 = Path('evidence/train_action_response58/collection_01')
OLD59 = Path('evidence/yaw_jvp59/execution_01')
LABEL60 = Path('evidence/noise_center60/execution_01')
LABEL62 = Path('evidence/clean_history62/execution_01')
PHASES = ('admission', 'training', 'pairing', 'final_training_label_diagnostics', 'historical_numeric',
          'regression59', 'fresh_collection', 'fresh_numeric', 'seal_predictions', 'physical_grid',
          'fresh_first_step', 'comparison')
PAIR_FIELDS = ('training_data_sha256', 'common58_labels_sha256', 'common_windows_sha256',
               'planned_sampling_rows_sha256', 'completed_sampling_rows_sha256',
               'initial_model_state_sha256', 'ridge_and_normalization_sha256', 'lambda_sha256',
               'initial_common58_predictions_sha256', 'initial_common58_yaw_jvp_sha256')


def verify64(plan):
    require(plan['schema'] == 'd1-mixed-sign64-v1', 'wrong64 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS), '64 arms/seeds changed')
    require(plan['updates_per_model'] == 1000 and plan['expected_policy_steps'] == 4680
            and plan['hard_policy_cap'] == 5000 and plan['cpu_seconds'] == 600 and plan['wall_seconds'] == 900
            and plan['fixed_lambda'] == FIXED_LAMBDA and plan['label_rows'] == 96,
            '64 work/weight budget changed')
    required = {str(PLAN53), str(SUMMARY53), str(LOCAL55), str(ADMISSION55), str(OLD55 / 'probe_snapshots.pkl'),
                str(OLD56 / 'report.json'), str(OLD56 / 'probe_snapshots.pkl'),
                str(OLD59 / 'report.json'), str(OLD59 / 'probe_snapshots.pkl'),
                str(LABEL58 / 'report.json'), str(LABEL60 / 'report.json'), str(LABEL62 / 'report.json'),
                'tools64/run64.py', 'tools64/fit64.py', 'tools59/run59.py', 'tools59/fit59.py',
                'tools56/run56.py', 'tools56/fit56.py', 'tools54/numerics54.py', 'tools54/reproduce54.py',
                'tools55/run55.py', 'tools55/probe_local55.py', 'tools53/collection53.py', 'tools53/run_pilot53.py'}
    required.update(str(OLD55 / f'probe_{index:02d}.npz') for index in range(9))
    for directory in (OLD56, OLD59):
        required.update(str(directory / f'grid_{point:02d}_{branch:02d}.npz')
                        for point in range(9) for branch in range(10))
    for directory, suffix in ((LABEL58, ''), (LABEL60, '_clean'), (LABEL62, '')):
        required.update(str(directory / f'branch_{point:02d}{suffix}_{branch:02d}.npz')
                        for point in range(12) for branch in range(10))
        required.update(str(directory / f'response_{point:02d}{suffix}.npz') for point in range(12))
    require(required <= set(plan['source_sha256']), 'required64 sources/inputs not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '64 source/input changed: ' + name)


def check_pairing(models, expected):
    require(len(models) == 6 and {(r['seed'], r['arm']) for r in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'six registered paired fits required')
    for row in models:
        fit = row['fit']
        require(fit['status'] == 'COMPLETE' and fit['eligible_for_evaluation']
                and fit['updates'] == fit['updates_attempted'] == 1000 and fit['seed'] == row['seed']
                and fit['arm'] == row['arm'] and fit['horizon'] == 8
                and fit['training_transitions'] == 7200 and fit['training_windows'] == 6828
                and fit['training_episodes'] == list(range(12)), 'incomplete/nonpaired64 fit')
        for name, value in expected['common'].items():
            require(fit[name] == value, 'fit differs from common admitted data/sampling: ' + name)
        for name, value in expected['by_arm'][row['arm']].items():
            require(fit[name] == value, 'fit differs from registered label source: ' + name)
        require(fit['fixed_lambda'] == fit['effective_lambda'] == FIXED_LAMBDA,
                'fixed59 lambda must be identical in both arms without recalibration')
    pairs = []
    for seed in MODEL_SEEDS:
        left, right = [next(r['fit'] for r in models if r['seed'] == seed and r['arm'] == arm) for arm in ARMS]
        require(all(left[key] == right[key] for key in PAIR_FIELDS), 'paired data/init/sampling/weight identity differs')
        require(left['labels_sha256'] != right['labels_sha256'], 'the registered label source must differ')
        pairs.append({'seed': seed, 'status': 'COMPLETE', 'identical_fields': list(PAIR_FIELDS),
                      'only_fit_variable': 'second48 source:60 versus62',
                      'control_label_hash': left['labels_sha256'], 'active_label_hash': right['labels_sha256']})
    for key in ('lambda_sha256', 'ridge_and_normalization_sha256',
                'initial_common58_predictions_sha256', 'initial_common58_yaw_jvp_sha256'):
        require(len({r['fit'][key] for r in models}) == 1, 'initial function/weight identity differs across seeds: ' + key)
    return {'status': 'COMPLETE', 'pairs': pairs, 'fixed_lambda': FIXED_LAMBDA, 'recalibrated': False}


def comparisons(models, cost_references, first_references):
    require(len(models) == 6 and {(row['seed'], row['arm']) for row in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'six64 models required')
    result = {'models': [], 'paired': [], 'selected_seed': None, 'selected_control_parameter': None,
              'old53_reference_scope': 'historical wide-epsilon regression; not a true infinitesimal derivative'}
    by_key = {}
    for model in models:
        all_checks = all(point['autograd_agrees'] for role in ROLES for point in model['diagnostics'][role]['points'])
        all_directions = all(point['direction_identifiable'] for role in ROLES for point in model['diagnostics'][role]['points'])
        row = {'seed': model['seed'], 'arm': model['arm'], 'all_selfchecks': all_checks,
               'all_model_directions_identifiable': all_directions, 'numerics_valid': all_checks and all_directions,
               'cost': {role: cost_comparison(model['diagnostics'][role]['points'], cost_references[role]) for role in COST_ROLES},
               'first_step': {role: first_step_comparison(model['first_step'][role]['points'], first_references[role])
                              for role in FIRST_ROLES}}
        row['full_direction_gate'] = row['numerics_valid'] and all(value['passed_9_of_9'] for value in row['cost'].values())
        row['first_step_gate'] = all(value['passed_9_of_9'] for value in row['first_step'].values())
        row['repair_gate'] = row['full_direction_gate'] and row['first_step_gate']
        row['cost_valid_matches'] = sum(value['valid_matches'] for value in row['cost'].values())
        row['first_step_valid_matches'] = sum(value['valid_matches'] for value in row['first_step'].values())
        result['models'].append(row)
        by_key[(row['seed'], row['arm'])] = row
    for seed in MODEL_SEEDS:
        control, active = [by_key[(seed, arm)] for arm in ARMS]
        pair = {'seed': seed, 'control_cost_valid_matches_of36': control['cost_valid_matches'],
                'active_cost_valid_matches_of36': active['cost_valid_matches'],
                'cost_delta': active['cost_valid_matches'] - control['cost_valid_matches'],
                'control_first_step_valid_matches_of18': control['first_step_valid_matches'],
                'active_first_step_valid_matches_of18': active['first_step_valid_matches'],
                'first_step_delta': active['first_step_valid_matches'] - control['first_step_valid_matches'], 'state_loss': {}}
        for role in ROLES:
            diagnostics = [next(model for model in models if model['seed'] == seed and model['arm'] == arm)
                           ['diagnostics'][role] for arm in ARMS]
            left, right = diagnostics
            pair['state_loss'][role] = {'control': left['mean_state_group_loss'], 'active': right['mean_state_group_loss'],
                'ratio_active_over_control': right['mean_state_group_loss'] / left['mean_state_group_loss']
                if left['mean_state_group_loss'] else None,
                'control_groups': left['mean_state_group_losses'], 'active_groups': right['mean_state_group_losses']}
        result['paired'].append(pair)
    resolved = all(row['resolved'] for refs in (*cost_references.values(), *first_references.values()) for row in refs)
    active_pass = all(row['repair_gate'] for row in result['models'] if row['arm'] == ARMS[1])
    numeric_comparison = all(row['numerics_valid'] for row in result['models'])
    nondecreasing = all(row['cost_delta'] >= 0 and row['first_step_delta'] >= 0 for row in result['paired'])
    strictly_better = any(row['cost_delta'] > 0 or row['first_step_delta'] > 0 for row in result['paired'])
    result.update(all_physical_references_resolved=resolved,
                  active_repair_validated=resolved and active_pass,
                  active_status='REFERENCE_UNRESOLVED' if not resolved else 'PASSED' if active_pass else 'FAILED',
                  paired_status='REFERENCE_UNRESOLVED' if not resolved else 'NUMERIC_UNRESOLVED' if not numeric_comparison else 'COMPLETE',
                  paired_improved=resolved and numeric_comparison and nondecreasing and strictly_better,
                  paired_non_decreasing=nondecreasing, paired_any_strict_improvement=strictly_better)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-mixed-sign64-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'model_updates_attempted': 0, 'completed_model_fits': 0,
              'policy_updates': 0, 'control_parameter_updates': 0, 'training_physics_steps': 0,
              'original_results_rejudged': False, 'counts': None, 'consumed53_inputs': {},
              'phases': {name: {'status': 'NOT_RUN'} for name in PHASES},
              'collection': {'status': 'NOT_RUN'}, 'pairing': {'status': 'NOT_RUN'}, 'comparison': {'status': 'NOT_RUN'},
              'models': [{'seed': seed, 'arm': arm, 'status': 'NOT_RUN', 'directory': f'{arm}_seed{seed}',
                          'fit': {'status': 'NOT_RUN', 'updates': 0, 'updates_attempted': 0},
                          'final_training_label_diagnostics': {'status': 'NOT_RUN'},
                          'diagnostics': {role: {'status': 'NOT_RUN', 'completed_points': 0, 'points': []} for role in ROLES},
                          'first_step': {role: {'status': 'NOT_RUN', 'completed_points': 0, 'points': []} for role in FIRST_ROLES}}
                         for seed in MODEL_SEEDS for arm in ARMS],
              'physical_probes': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN', 'gradients': [],
                  'branches': [{'theta': theta, 'status': 'NOT_RUN'} for theta in sequence]} for seed in PROBE_SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress, buffer = {'phase': 'admission'}, {'rows': None}
    sealed, trained = {}, {}
    events = (args.output / 'phase_events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress, 'model_updates': report['model_updates'],
                                'completed_model_fits': report['completed_model_fits'],
                                'counts': None if env is None else env.counters(),
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
        events.flush()

    def begin(name):
        require(time.process_time() < 598., 'CPU deadline before ' + name)
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
        require(all(os.environ.get(name) == '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify64(plan)
        original = json.loads((ROOT / PLAN53).read_text())
        verify(original)
        summary = json.loads((ROOT / SUMMARY53).read_text())
        require(summary['formal']['status'] == 'COMPLETE' and summary['plan_sha256'] == sha(ROOT / PLAN53),
                'complete bound53 evidence required')
        consumed, manifest = report['consumed53_inputs'], summary['raw_manifest']
        collection53 = json.loads(bound53('execution_01/collection.json', manifest, consumed).read_text())
        require(collection53 == summary['formal']['collection'] and collection53['status'] == 'COMPLETE',
                'complete original53 collection required')
        for role in ('train', 'development', 'probe'):
            cases = [case for case in collection53['cases'] if case['role'] == role]
            require([case['seed'] for case in cases] == original['data_seeds'][role]
                    and all(case['status'] == 'COMPLETE' and case['steps'] == 600
                            and case['complete_12s_trial'] and not case['termination_reasons'] for case in cases),
                    'incomplete original role: ' + role)
        admission = json.loads((ROOT / ADMISSION55).read_text())
        require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                and admission['counts']['policy_transitions'] == 224, 'complete55 interface admission required')
        report['interface_admission_source'] = str(ADMISSION55)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'], 'package versions differ')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        from tools64 import fit64
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report['torch_runtime_version'] = torch.__version__
        require(torch.__version__ == summary['formal']['torch_runtime_version'], 'torch runtime differs')
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        actor64 = copy.deepcopy(actor32).double().eval().requires_grad_(False)
        path = bound53('execution_01/train_transitions.npz', manifest, consumed)
        with np.load(path, allow_pickle=False) as raw:
            train = dynamics.validate_data({name: raw[name].copy() for name in fit64.paired.DATA_KEYS})
        require(len(train['s']) == 7200 and np.array_equal(np.unique(train['episode']), np.arange(12))
                and all(np.count_nonzero(train['episode'] == episode) == 600 for episode in range(12)), 'exact7200 train rows required')
        sources = fit64.load_label_sources(ROOT / LABEL58, ROOT / LABEL60, ROOT / LABEL62)
        labels = sources['58']
        labels_by_arm = {arm: fit64.compose_labels(sources, arm) for arm in ARMS}
        for index in range(48):
            training_index = (int(labels['seed'][index]) - 53101) * 600 + int(labels['tick'][index])
            require(np.array_equal(labels['s'][index], train['s'][training_index])
                    and np.array_equal(labels['a0'][index], train['a'][training_index]), 'label base state/action is not original train row')
        common = fit64.paired.common_window_indices(train).astype('<i8', copy=False)
        require(common.shape == (6828, 32), 'common32 window count differs')
        sampling_hash = fit64.paired.array_sha256(fit64.paired.sampled_window_rows(6828).numpy().astype('<i8', copy=False))
        expected = {'common': {
            'training_data_sha256': fit64.paired.mapping_sha256(train),
            'common58_labels_sha256': fit64.paired.mapping_sha256(sources['58']),
            'common_windows_sha256': fit64.paired.array_sha256(common),
            'planned_sampling_rows_sha256': sampling_hash, 'completed_sampling_rows_sha256': sampling_hash},
            'by_arm': {arm: {'labels_sha256': fit64.paired.mapping_sha256(labels_by_arm[arm]),
                            'second_labels_sha256': fit64.paired.mapping_sha256(sources[source]),
                            'second_label_source': source}
                       for arm, source in zip(ARMS, ('60', '62'), strict=True)}}
        report['training_admission'] = {'expected_identities': expected, 'transitions': 7200, 'complete_episodes': 12,
            'common32_windows': 6828, 'label_rows_per_arm': 96, 'distinct_states': {'control': 12, 'active': 24},
            'heldout_used_for_fit_or_calibration': False, 'fixed_lambda': FIXED_LAMBDA,
            'prior62_coverage_status': 'REJECTED', 'prior62_rejudged': False}
        complete()
        begin('training')
        for row in report['models']:
            row['status'], row['fit']['status'] = 'INCOMPLETE', 'INCOMPLETE'
            model, row['fit'] = fit64.fit_model(train, labels_by_arm[row['arm']], row['seed'], row['arm'], args.output / row['directory'], 598.)
            trained[(row['seed'], row['arm'])] = model
            weight_name = row['directory'] + '/final_state_dict.pt'
            sealed[weight_name] = sha(args.output / weight_name)
            report['model_updates'] += row['fit']['updates']
            report['model_updates_attempted'] += row['fit']['updates_attempted']
            report['completed_model_fits'] += 1
            event('FIT_COMPLETE')
        require(report['model_updates'] == 6000 and report['completed_model_fits'] == 6, 'all six final fits required before heldout')
        complete()
        begin('pairing')
        report['pairing'] = check_pairing(report['models'], expected)
        write_json(args.output / 'pairing.json', report['pairing'])
        complete()
        begin('final_training_label_diagnostics')
        for row in report['models']:
            require(time.process_time() < 598., 'CPU budget before final training labels')
            row['final_training_label_diagnostics'] = {'status': 'INCOMPLETE', 'sources': {}}
            for source in ('58', '60', '62'):
                folder = args.output / row['directory'] / f'label_diagnostic_{source}'
                folder.mkdir(exist_ok=False)
                row['final_training_label_diagnostics']['sources'][source] = final_training_label_diagnostics(
                    trained[(row['seed'], row['arm'])], sources[source], folder, row['fit']['final_model_state_sha256'])
            row['final_training_label_diagnostics']['status'] = 'COMPLETE'
        complete()
        begin('historical_numeric')
        points, targets, cost_references, old56_grids = historical_inputs(original, summary, consumed)
        first_refs = {'old56': first_references(points['old56'], old56_grids)}
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            for role in ROLES[:-1]:
                evaluate_points(model64, actor64, points[role], targets[role], args.output / row['directory'] / role,
                                row['diagnostics'][role], 598.)
            evaluate_first_steps(model64, points['old56'], old56_grids, args.output / row['directory'] / 'first_step_old56',
                                 row['first_step']['old56'], 598.)
        complete()
        begin('regression59')
        previous = json.loads((ROOT / OLD59 / 'report.json').read_text())
        require(previous['status'] == 'COMPLETE', 'complete59 heldout archive required')
        prior_points = load_own_points(ROOT / OLD59 / 'probe_snapshots.pkl')
        check_points(prior_points, (59301, 59302, 59303))
        prior_grids = load_grid(ROOT / OLD59, previous['physical_probes'])
        prior_targets = [grid[0]['states'].copy() for grid in prior_grids]
        prior_cost = local_references(previous['physical_probes'])
        prior_first = first_references(prior_points, prior_grids)
        report['regression59_references'] = {'cost': prior_cost, 'first_step': prior_first,
            'used_for_fit_or_gate_or_selection': False, 'new_physics_steps': 0}
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            regression = {'cost_diagnostics': {}, 'first_diagnostics': {}, 'used_for_gate_or_selection': False}
            row['regression59'] = regression
            evaluate_points(model64, actor64, prior_points, prior_targets,
                            args.output / row['directory'] / 'old59', regression['cost_diagnostics'], 598.)
            evaluate_first_steps(model64, prior_points, prior_grids,
                                 args.output / row['directory'] / 'first_step_old59', regression['first_diagnostics'], 598.)
            regression['cost'] = cost_comparison(regression['cost_diagnostics']['points'], prior_cost)
            regression['first_step'] = first_step_comparison(regression['first_diagnostics']['points'], prior_first)
        complete()
        begin('fresh_collection')
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        original_recorder = make_step_recorder(audit, progress)

        def recorder(current, action, result):
            if buffer['rows'] is not None:
                obs, reward, terminated, truncated, info = result
                buffer['rows'].append((control.physical(current), np.asarray(obs).copy(), float(reward),
                    {'terminated': bool(terminated), 'truncated': bool(truncated),
                     'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact'])}))
            original_recorder(current, action, result)
            audit.flush()

        env = collection.make_env(5000, audit_step=recorder)
        report['collection'] = {'status': 'INCOMPLETE'}
        points['fresh'], targets['fresh'], report['collection'] = collect_probes(env, actor32, args.output, plan, original)
        check_points(points['fresh'], PROBE_SEEDS)
        require(env.counters()['policy_transitions'] == 1800, 'fresh baseline work differs')
        complete()
        begin('fresh_numeric')
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            evaluate_points(model64, actor64, points['fresh'], targets['fresh'], args.output / row['directory'] / 'fresh',
                            row['diagnostics']['fresh'], 598.)
        complete()
        begin('seal_predictions')
        prediction = args.output / 'predictions_before_fresh_fd.json'
        write_json(prediction, {'models': report['models'], 'pairing': report['pairing'],
                               'fresh_physical_fd_results_seen': False, 'observed_forward_correction': False,
                               'policy_updates': 0, 'control_parameter_updates': 0,
                               'first_step_readback_requires_recorded_actions_after_physics': True})
        sealed[prediction.name] = sha(prediction)
        report['predictions_sha256'] = sealed[prediction.name]
        complete()
        begin('physical_grid')
        physical_grid(env, actor32, points['fresh'], targets['fresh'], args.output, report['physical_probes'], progress, buffer, 598.)
        require(env.counters()['policy_transitions'] == 4680, 'final physical work differs')
        cost_references['fresh'] = local_references(report['physical_probes'])
        complete()
        begin('fresh_first_step')
        fresh_grids = load_grid(args.output, report['physical_probes'])
        first_refs['fresh'] = first_references(points['fresh'], fresh_grids)
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            evaluate_first_steps(model64, points['fresh'], fresh_grids, args.output / row['directory'] / 'first_step_fresh',
                                 row['first_step']['fresh'], 598.)
            row['status'] = 'COMPLETE'
        complete()
        begin('comparison')
        report.update(cost_references=cost_references, first_step_references=first_refs)
        report['comparison'] = {'status': 'PROVISIONAL', 'eligibility_pending_final_audit': True,
                                **comparisons(report['models'], cost_references, first_refs)}
        write_json(args.output / 'comparison.json', report['comparison'])
        complete()
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - all incomplete work remains ineligible without retry/resume.
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['phases'][progress['phase']]['failure'] = report['failure']
        event('INCOMPLETE')
    finally:
        errors = []
        report['last_phase'] = dict(progress)
        refresh_fit_counts(args.output, report, errors)
        report['update_count_semantics'] = 'durable completed optimizer steps; interrupted work is a lower bound'
        if (args.output / 'collection.json').exists():
            try:
                report['collection'] = json.loads((args.output / 'collection.json').read_text())
            except (OSError, ValueError) as exc:
                errors.append(f'collection receipt: {exc}')
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - retain cleanup failure.
                errors.append(f'env: {exc}')
        if audit is not None:
            try:
                audit.close()
                report['work_journal'] = audit_work(args.output / 'all_physics_steps.jsonl')
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'], 'work journal differs')
                if report['status'] == 'COMPLETE':
                    require(report['work_journal']['by_phase'] == {'fresh_collection': 1800, 'physical_grid': 2880}, 'phase work differs')
            except Exception as exc:  # noqa: BLE001 - preserve attempted counters and flushed prefixes.
                errors.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify64(plan)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'versions changed')
            if 'torch_runtime_version' in report:
                require(sys.modules['torch'].__version__ == report['torch_runtime_version'], 'torch runtime changed')
            for name, identity in report['consumed53_inputs'].items():
                require((ROOT / name).stat().st_size == identity['bytes'] and sha(ROOT / name) == identity['sha256'], 'consumed53 input changed')
            for name, digest in sealed.items():
                require(sha(args.output / name) == digest, 'sealed output changed: ' + name)
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - retain both verification and original failure.
            report['verification_after'] = 'FAILED'
            errors.append(f'verification: {exc}')
        if errors:
            report.update(status='INCOMPLETE', cleanup_errors=errors)
        if report['comparison'].get('status') == 'PROVISIONAL':
            if report['status'] == 'COMPLETE':
                report['comparison'].update(status='COMPLETE', eligibility_pending_final_audit=False)
            else:
                report['comparison_before_final_eligibility'] = copy.deepcopy(report['comparison'])
                report['comparison'].update(status='INCOMPLETE', eligibility_pending_final_audit=False,
                    active_repair_validated=False, paired_improved=False, active_status='INCOMPLETE', paired_status='INCOMPLETE',
                    final_eligibility_withheld=True)
        report['sealed_artifacts_sha256'] = sealed
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'model_updates', 'completed_model_fits',
                                                      'counts', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
