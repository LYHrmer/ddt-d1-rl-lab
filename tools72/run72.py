"""72 paired yaw-weight candidate, preserving69 active baseline and all old gates.

Only lambda differs; six complete fits precede evaluation. All old references,
including unknowns, are retained; predictions precede fresh perturbed truth.
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
from tools69.run69 import required_inputs69
from tools72.compare72 import comparisons

MODEL_SEEDS = (64001, 64002, 64003)
PROBE_SEEDS = (72301, 72302, 72303)
ARMS = ('control', 'active')
FIXED_LAMBDA = 0.08980082414438752
FIXED_BETA = FIXED_LAMBDA
YAW_COEFFICIENT = {'control': FIXED_LAMBDA, 'active': 2 * FIXED_LAMBDA}
BODY_COEFFICIENT = dict.fromkeys(ARMS, FIXED_BETA)
ROLES = ('development', 'old_probe', 'old55_local', 'old56', 'old64', 'old69', 'fresh')
COST_ROLES = ROLES[1:]
FIRST_ROLES = ('old56', 'old64', 'old69', 'fresh')
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
OLD64 = Path('evidence/mixed_sign64/execution_01')
OLD69 = Path('evidence/body_jvp69/execution_01')
PLAN69 = Path('docs/experiments69/body_jvp69_preregistered.json')
LABEL68 = Path('evidence/body_response68/admission_01')
PHASES = ('admission', 'training', 'pairing', 'final_training_label_diagnostics', 'historical_numeric',
          'head_historical', 'regression59', 'fresh_collection', 'fresh_numeric', 'seal_predictions', 'physical_grid',
          'fresh_first_step', 'head_fresh', 'comparison')
PAIR_FIELDS = ('training_data_sha256', 'common58_labels_sha256', 'common_windows_sha256',
               'planned_sampling_rows_sha256', 'completed_sampling_rows_sha256',
               'initial_model_state_sha256', 'ridge_and_normalization_sha256', 'beta_sha256',
               'initial_common58_predictions_sha256', 'initial_common58_yaw_jvp_sha256',
               'labels_sha256', 'body_labels_sha256', 'combined_labels_sha256',
               'second_labels_sha256', 'initial_body_jvp_sha256')
DECISION_SPEC = {
    'numeric_roles': list(ROLES), 'cost_h32_roles': list(COST_ROLES),
    'first_and_head_roles': list(FIRST_ROLES), 'cost_h32_points': 54, 'yaw_first_points': 36,
    'body_state_channels': 72, 'cost_h1_points': 36, 'body_error_scalar_count': 288,
    'fresh_yaw_error_scalar_count': 36, 'points_per_role': 9, 'ticks': list(TICKS),
    'physical_theta_epsilons': list(EPSILONS), 'model_selfcheck_epsilon': .001,
    'selfcheck_atol': 1e-4, 'selfcheck_rtol': .1, 'direction_magnitude_floor': 1e-5,
    'physical_central_spread_max': .25, 'physical_one_sided_gap_max': .25,
    'all_four_eps_same_sign_and_all_four_wheels_resolved': True,
    'body_and_yaw_scale': .30000001192092896,
    'fresh_yaw_rmse_ratio_max_each_seed': .8, 'training_yaw_rmse_strictly_down_all3': True,
    'fresh_target_requires_all_physical_yaw_refs_and_active_directions': True,
    'fresh_target_both_arms_original_numeric_eligibility': True,
    'body_rmse_ratio_max_each_seed_all_registered_points': 1.,
    'no_new_yaw_or_body_failure_per_epsilon': True, 'no_new_resolved_H1_H32_failure': True,
    'state_mean_seven_group_ratio_max_each_role_including_development': 1.1,
    'all_six_global_numeric_gate_including_development_preserved': True,
    'global_candidate_and_repair_require_all_formal_references_resolved': True,
    'fresh_local_target_is_not_global_candidate_acceptance_or_repair': True,
    'old53_reference_scope': 'unchanged historical wide-epsilon regression',
    'old64_unknown_retained': True, 'old69_references_exact': True,
    'control_exact69active_mapping_all3_required': True,
    'all_six_fits_before_heldout': True, 'fresh_free_H1_H32_sealed_before_physical_FD': True,
    'yaw_and_body_condition_on_actual_F32_first_actions': True,
    'training_sources': ['58', '62'], 'training_centers': 24, 'training_label_rows': 96,
    'regression59_and_source60_descriptive_only': True,
    'no_seed_or_epsilon_or_weight_search': True, 'updates_policy_or_controller': 0,
}


def required_inputs72():
    required = required_inputs69() | {str(PLAN69), str(OLD69 / 'report.json'),
        str(OLD69 / 'probe_snapshots.pkl'), 'tools72/run72.py', 'tools72/admit72.py',
        'tools72/fit72.py', 'tools72/compare72.py', 'tests/test_run72.py',
        'tests/test_fit72.py', 'tests/test_compare72.py', 'docs/yaw_weight72_plan_20261101.md',
        '.github/workflows/yaw-weight72.yml'}
    required.update(str(OLD69 / f'grid_{point:02d}_{branch:02d}.npz')
                    for point in range(9) for branch in range(10))
    required.update(str(OLD69 / f'active_seed{seed}/final_state_dict.pt') for seed in MODEL_SEEDS)
    return required


def verify72(plan):
    require(plan['schema'] == 'd1-yaw-weight72-v1', 'wrong72 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS), '72 arms/seeds changed')
    require(plan['updates_per_model'] == 1000 and plan['expected_policy_steps'] == 4680
            and plan['hard_policy_cap'] == 5000 and plan['cpu_seconds'] == 600 and plan['wall_seconds'] == 900
            and plan['fixed_lambda'] == FIXED_LAMBDA and plan['fixed_beta'] == FIXED_BETA
            and plan['yaw_coefficients'] == YAW_COEFFICIENT and plan['body_coefficients'] == BODY_COEFFICIENT
            and plan['label_rows'] == 96, '72 budget/coefficient changed')
    require(plan['decision_spec'] == DECISION_SPEC, '72 decision rules changed')
    require(required_inputs72() <= set(plan['source_sha256']), '72 required inputs unbound')
    inherited = json.loads((ROOT / PLAN69).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(name) == digest for name, digest in inherited.items()),
            '72 must retain all69 bindings unchanged')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '72 source/input changed: ' + name)


def prior69():
    previous = json.loads((ROOT / OLD69 / 'report.json').read_text())
    require(previous['status'] == previous['verification_after'] == 'COMPLETE'
            and previous['plan_sha256'] == sha(ROOT / PLAN69) and previous['completed_model_fits'] == 6
            and previous['model_updates'] == 6000, 'complete69 baseline required')
    controls = [row for row in previous['models'] if row['arm'] == 'active']
    require([row['seed'] for row in controls] == list(MODEL_SEEDS)
            and all(row['fit']['status'] == 'COMPLETE' and row['fit']['eligible_for_evaluation']
                    and row['fit']['updates'] == row['fit']['updates_attempted'] == 1000 for row in controls),
            'three complete69 active models required')
    for row in controls:
        path = row['directory'] + '/final_state_dict.pt'
        require(sha(ROOT / OLD69 / path) == previous['sealed_artifacts_sha256'][path], '69 sealed baseline changed')
    return previous, {str(row['seed']): row['fit']['final_model_state_sha256'] for row in controls}


def check_pairing(models, expected):
    require(len(models) == 6 and {(r['seed'], r['arm']) for r in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'six72 paired fits required')
    for row in models:
        fit, arm = row['fit'], row['arm']
        require(fit['status'] == 'COMPLETE' and fit['eligible_for_evaluation']
                and fit['updates'] == fit['updates_attempted'] == 1000 and fit['seed'] == row['seed']
                and fit['arm'] == arm and fit['horizon'] == 8 and fit['training_transitions'] == 7200
                and fit['training_windows'] == 6828 and fit['training_episodes'] == list(range(12)),
                'incomplete72 paired fit')
        for name, value in expected['common'].items():
            require(fit[name] == value, 'common fit identity differs: ' + name)
        for name, value in expected['by_arm'][arm].items():
            require(fit[name] == value, 'registered coefficient differs: ' + name)
        require(fit['fixed_lambda'] == fit['effective_lambda'] == YAW_COEFFICIENT[arm]
                and fit['reference_lambda'] == FIXED_LAMBDA and fit['second_label_source'] == '62'
                and not fit['calibration_performed'] and fit['fixed_beta'] == fit['body_coefficient'] == FIXED_BETA,
                'only yaw coefficient may change')
    pairs = []
    for seed in MODEL_SEEDS:
        left, right = [next(row['fit'] for row in models if row['seed'] == seed and row['arm'] == arm) for arm in ARMS]
        require(all(left[key] == right[key] for key in PAIR_FIELDS), 'paired data/init/beta/sampling differs')
        digest = expected['control_final_sha256'][str(seed)]
        require(left['final_model_state_sha256'] == digest, 'control must exactly reproduce69 active')
        pairs.append({'seed': seed, 'status': 'COMPLETE', 'identical_fields': list(PAIR_FIELDS),
                      'only_fit_variable': 'yaw JVP coefficient lambda',
                      'control_reproduces69_active': True, 'control_final_sha256': digest})
    for key in ('beta_sha256', 'ridge_and_normalization_sha256', 'initial_common58_predictions_sha256',
                'initial_common58_yaw_jvp_sha256', 'initial_body_jvp_sha256'):
        require(len({row['fit'][key] for row in models}) == 1, 'initial function/beta differs across seeds: ' + key)
    return {'status': 'COMPLETE', 'pairs': pairs, 'yaw_coefficients': YAW_COEFFICIENT,
            'body_coefficients': BODY_COEFFICIENT, 'recalibrated': False}


def historical72(original, summary, consumed):
    """Read all old data; no model call/actor/environment or physical integration."""
    from tools69 import outputs69
    points, targets, costs, old56_grids = historical_inputs(original, summary, consumed)
    grids = {'old56': old56_grids}
    first = {'old56': first_references(points['old56'], old56_grids)}
    heads = {'old56': outputs69.head_references(points['old56'], old56_grids)}
    for role, directory, seeds in (('old64', OLD64, (64301, 64302, 64303)),
                                    ('old69', OLD69, (69301, 69302, 69303))):
        previous = json.loads((ROOT / directory / 'report.json').read_text())
        require(previous['status'] == previous['verification_after'] == 'COMPLETE', 'complete old reference required')
        points[role] = load_own_points(ROOT / directory / 'probe_snapshots.pkl')
        check_points(points[role], seeds)
        grids[role] = load_grid(ROOT / directory, previous['physical_probes'])
        targets[role] = [grid[0]['states'].copy() for grid in grids[role]]
        costs[role] = local_references(previous['physical_probes'])
        first[role] = first_references(points[role], grids[role])
        heads[role] = outputs69.head_references(points[role], grids[role])
        require(costs[role] == previous['cost_references']['fresh']
                and first[role] == previous['first_step_references']['fresh'], 'old cost/yaw reference changed')
        if role == 'old69':
            require(heads[role] == previous['head_references']['fresh'], 'old69 body/H1 reference changed')
    return points, targets, costs, grids, first, heads


def withhold_final_eligibility(report):
    """Retain provisional detail, but publish no accepted sub-result after a failed final audit."""
    report['comparison_before_final_eligibility'] = copy.deepcopy(report['comparison'])
    report['comparison'] = {
        'status': 'INCOMPLETE', 'eligibility_pending_final_audit': False,
        'active_repair_validated': False, 'complete_repair_validated': False,
        'candidate_accepted': False, 'generalization_target_passed': False, 'training_improved': False,
        'candidate_status': 'INCOMPLETE', 'generalization_target_status': 'INCOMPLETE',
        'complete_repair_status': 'INCOMPLETE', 'final_eligibility_withheld': True,
        'raw_detail_retained_in': 'comparison_before_final_eligibility',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-yaw-weight72-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'model_updates_attempted': 0, 'completed_model_fits': 0,
              'policy_updates': 0, 'control_parameter_updates': 0, 'training_physics_steps': 0,
              'original_results_rejudged': False, 'counts': None, 'consumed53_inputs': {},
              'phases': {name: {'status': 'NOT_RUN'} for name in PHASES},
              'collection': {'status': 'NOT_RUN'}, 'pairing': {'status': 'NOT_RUN'}, 'comparison': {'status': 'NOT_RUN'},
              'models': [{'seed': seed, 'arm': arm, 'status': 'NOT_RUN', 'directory': f'{arm}_seed{seed}',
                          'fit': {'status': 'NOT_RUN', 'updates': 0, 'updates_attempted': 0},
                          'final_training_label_diagnostics': {'status': 'NOT_RUN'},
                          'diagnostics': {role: {'status': 'NOT_RUN', 'completed_points': 0, 'points': []} for role in ROLES},
                          'first_step': {role: {'status': 'NOT_RUN', 'completed_points': 0, 'points': []} for role in FIRST_ROLES},
                          'cost_h1': {role: {'status': 'NOT_RUN', 'points': []} for role in FIRST_ROLES},
                          'heads': {role: {'status': 'NOT_RUN', 'points': []} for role in FIRST_ROLES}}
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
        verify72(plan)
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
        from tools69 import outputs69
        from tools72 import fit72
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
        labels = fit72.load_labels68(ROOT / LABEL68)
        legacy_labels = fit64.compose_labels(sources, 'active')
        require(all(np.array_equal(labels[name], values) for name, values in legacy_labels.items()),
                '68 state/yaw rows must exactly reproduce64 active data')
        for index in range(48):
            training_index = (int(labels['seed'][index]) - 53101) * 600 + int(labels['tick'][index])
            require(np.array_equal(labels['s'][index], train['s'][training_index])
                    and np.array_equal(labels['a0'][index], train['a'][training_index]), '58 label not original train row')
        common = fit64.paired.common_window_indices(train).astype('<i8', copy=False)
        require(common.shape == (6828, 32), 'common32 window count differs')
        sampling_hash = fit64.paired.array_sha256(fit64.paired.sampled_window_rows(6828).numpy().astype('<i8', copy=False))
        previous69, controls69 = prior69()
        require(report['versions'] == previous69['versions']
                and report['torch_runtime_version'] == previous69['torch_runtime_version'], '69 runtime differs')
        expected = {'common': {
            'training_data_sha256': fit64.paired.mapping_sha256(train), **fit72.label_hashes(labels),
            'common_windows_sha256': fit64.paired.array_sha256(common),
            'planned_sampling_rows_sha256': sampling_hash, 'completed_sampling_rows_sha256': sampling_hash},
            'by_arm': {arm: {'fixed_lambda': weight, 'effective_lambda': weight,
                'lambda_sha256': fit64.paired.array_sha256(np.asarray([weight], dtype=np.float64)),
                'fixed_beta': FIXED_BETA, 'body_coefficient': FIXED_BETA,
                'beta_sha256': fit64.paired.array_sha256(np.asarray([FIXED_BETA], dtype=np.float64))}
                for arm, weight in YAW_COEFFICIENT.items()}, 'control_final_sha256': controls69}
        for old_model in previous69['models']:
            if old_model['arm'] == 'active':
                require(all(old_model['fit'][key] == value for key, value in expected['common'].items()),
                        'baseline69 training/sampling identities differ')
        report['training_admission'] = {'expected_identities': expected, 'transitions': 7200, 'complete_episodes': 12,
            'common32_windows': 6828, 'label_rows_per_arm': 96, 'distinct_states_per_arm': 24,
            'heldout_used_for_fit_or_calibration': False, 'fixed_lambda': FIXED_LAMBDA,
            'yaw_coefficients': YAW_COEFFICIENT, 'body_coefficients': BODY_COEFFICIENT,
            'prior62_coverage_status': 'REJECTED', 'prior62_rejudged': False}
        complete()
        begin('training')
        for row in report['models']:
            row['status'], row['fit']['status'] = 'INCOMPLETE', 'INCOMPLETE'
            model, row['fit'] = fit72.fit_model(train, labels, row['seed'], row['arm'], args.output / row['directory'], 598.)
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
        report['pairing'] = {'status': 'INCOMPLETE'}
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
            model = trained[(row['seed'], row['arm'])]
            responses = fit72.body_jvp(model, labels, create_graph=False).detach().numpy().astype(np.float64)
            body_scale = float(dynamics.PHYSICAL_SCALE[23])
            residual = (responses - labels['target_body']) / body_scale
            body_rows = [{'source': int(labels['source_id'][i]), 'seed': int(labels['seed'][i]),
                         'tick': int(labels['tick'][i]), 'epsilon': float(labels['epsilon'][i]),
                         'prediction': responses[i].tolist(), 'target': labels['target_body'][i].tolist(),
                         'same_sign': (responses[i] * labels['target_body'][i] > 0).tolist()}
                        for i in range(96)]
            body_report = {'status': 'COMPLETE', 'rows': body_rows, 'normalized_mse': float(np.mean(residual ** 2)),
                           'physical_scale': body_scale, 'rows_count': 96, 'state_count': 24,
                           'used_for_selection_or_gate': False}
            write_json(args.output / row['directory'] / 'final_body_training_diagnostic.json', body_report)
            row['final_training_label_diagnostics']['body'] = body_report
            require(fit72.paired.mapping_sha256(model.state_dict()) == row['fit']['final_model_state_sha256'],
                    'training diagnostic changed final weights')
            row['final_training_label_diagnostics']['status'] = 'COMPLETE'
        complete()
        begin('historical_numeric')
        points, targets, cost_references, historical_grids, first_refs, head_refs = historical72(
            original, summary, consumed)
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            for role in ROLES[:-1]:
                evaluate_points(model64, actor64, points[role], targets[role], args.output / row['directory'] / role,
                                row['diagnostics'][role], 598.)
            for role, grid in historical_grids.items():
                evaluate_first_steps(model64, points[role], grid, args.output / row['directory'] / ('first_step_' + role),
                                     row['first_step'][role], 598.)
                outputs69.evaluate_h1_costs(model64, actor64, points[role], args.output / row['directory'] / ('cost_h1_' + role),
                                            row['cost_h1'][role], 598.)
        complete()
        begin('head_historical')
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            for role, grid in historical_grids.items():
                outputs69.evaluate_heads(model64, points[role], grid, row['cost_h1'][role]['points'],
                    args.output / row['directory'] / ('heads_' + role), row['heads'][role], 598.)
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
            outputs69.evaluate_h1_costs(model64, actor64, points['fresh'], args.output / row['directory'] / 'cost_h1_fresh',
                                        row['cost_h1']['fresh'], 598.)
        complete()
        begin('seal_predictions')
        prediction = args.output / 'predictions_before_fresh_fd.json'
        write_json(prediction, {'models': report['models'], 'pairing': report['pairing'],
                               'fresh_physical_fd_results_seen': False, 'observed_forward_correction': False,
                               'policy_updates': 0, 'control_parameter_updates': 0,
                               'first_step_readback_requires_recorded_actions_after_physics': True})
        for row in report['models']:
            for role in ROLES:
                for path in (args.output / row['directory'] / role).rglob('*'):
                    if path.is_file():
                        sealed[str(path.relative_to(args.output))] = sha(path)
            for role in FIRST_ROLES:
                for path in (args.output / row['directory'] / ('cost_h1_' + role)).rglob('*'):
                    if path.is_file():
                        sealed[str(path.relative_to(args.output))] = sha(path)
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
        complete()
        begin('head_fresh')
        head_refs['fresh'] = outputs69.head_references(points['fresh'], fresh_grids)
        for row in report['models']:
            model64 = copy.deepcopy(trained[(row['seed'], row['arm'])]).double().eval().requires_grad_(False)
            outputs69.evaluate_heads(model64, points['fresh'], fresh_grids, row['cost_h1']['fresh']['points'],
                args.output / row['directory'] / 'heads_fresh', row['heads']['fresh'], 598.)
            row['status'] = 'COMPLETE'
        complete()
        begin('comparison')
        report.update(cost_references=cost_references, first_step_references=first_refs, head_references=head_refs)
        report['comparison'] = {'status': 'PROVISIONAL', 'eligibility_pending_final_audit': True,
                                **comparisons(report['models'], cost_references, first_refs, head_refs, eligibility={
                                    'pairing_complete': report['pairing']['status'] == 'COMPLETE',
                                    'control_reproduced69_active': all(p['control_reproduces69_active'] for p in report['pairing']['pairs'])})}
        write_json(args.output / 'comparison.json', report['comparison'])
        complete()
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - all incomplete work remains ineligible without retry/resume.
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['phases'][progress['phase']]['failure'] = report['failure']
        if progress['phase'] == 'pairing':
            report['pairing'].update(status='INVALID', failure=report['failure'])
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
                verify72(plan)
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
                withhold_final_eligibility(report)
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
