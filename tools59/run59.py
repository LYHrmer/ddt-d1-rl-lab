"""Paired59 yaw-JVP loss trial with unchanged free-rollout and physical gates.

Only complete53 train transitions and all58 training labels enter fitting or
lambda calibration. True future states enter diagnostic loss only. Six complete
fits precede all heldout evaluation; new free predictions and weights are sealed
before physical perturbations. No actor/control update, oracle rollout, best seed
or changed finite-difference reference is used.
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
from tools55.probe_local55 import EPSILONS, action_delta_ulps, local_stability
from tools55.run55 import audit_work, collect_probes
from tools56.run56 import (
    bound53,
    check_points,
    evaluate_points,
    local_references,
    physical_grid,
    reference53,
)

MODEL_SEEDS = (59001, 59002, 59003)
PROBE_SEEDS = (59301, 59302, 59303)
ARMS = ('state_only', 'state_plus_yaw_jvp')
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
PHASES = ('admission', 'training', 'pairing', 'final_training_label_diagnostics', 'historical_numeric', 'fresh_collection',
          'fresh_numeric', 'seal_predictions', 'physical_grid', 'fresh_first_step', 'comparison')
PAIR_FIELDS = ('training_data_sha256', 'labels_sha256', 'common_windows_sha256',
               'planned_sampling_rows_sha256', 'completed_sampling_rows_sha256',
               'initial_model_state_sha256', 'ridge_and_normalization_sha256', 'lambda_sha256')
CALIBRATION_FIELDS = ('S0', 'R0', 'lambda', 'lambda_sha256', 'initial_state_predictions_sha256', 'initial_yaw_jvp_sha256')


def verify59(plan):
    require(plan['schema'] == 'd1-yaw-jvp59-v1', 'wrong59 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS), '59 arms/seeds changed')
    require(plan['updates_per_model'] == 1000 and plan['expected_policy_steps'] == 4680
            and plan['hard_policy_cap'] == 5000 and plan['cpu_seconds'] == 600 and plan['wall_seconds'] == 900,
            '59 work/time budget changed')
    required = {str(PLAN53), str(SUMMARY53), str(LOCAL55), str(ADMISSION55), str(OLD55 / 'probe_snapshots.pkl'),
                str(OLD56 / 'report.json'), str(OLD56 / 'probe_snapshots.pkl'), str(LABEL58 / 'report.json'),
                'tools59/run59.py', 'tools59/fit59.py', 'tools56/run56.py', 'tools56/fit56.py',
                'tools54/numerics54.py', 'tools54/reproduce54.py', 'tools55/run55.py',
                'tools55/probe_local55.py', 'tools53/collection53.py', 'tools53/run_pilot53.py'}
    required.update(str(OLD55 / f'probe_{index:02d}.npz') for index in range(9))
    required.update(str(OLD56 / f'grid_{point:02d}_{branch:02d}.npz') for point in range(9) for branch in range(10))
    required.update(str(LABEL58 / f'branch_{point:02d}_{branch:02d}.npz') for point in range(12) for branch in range(10))
    required.update(str(LABEL58 / f'response_{point:02d}.npz') for point in range(12))
    require(required <= set(plan['source_sha256']), 'required59 sources/inputs not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '59 source/input changed: ' + name)


def check_pairing(models, expected):
    require(len(models) == 6 and {(row['seed'], row['arm']) for row in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'six registered paired fits required')
    for row in models:
        fit = row['fit']
        require(fit['status'] == 'COMPLETE' and fit['eligible_for_evaluation']
                and fit['updates'] == fit['updates_attempted'] == 1000 and fit['seed'] == row['seed']
                and fit['arm'] == row['arm'] and fit['horizon'] == 8
                and fit['training_transitions'] == 7200 and fit['training_windows'] == 6828
                and fit['training_episodes'] == list(range(12)), 'incomplete/nonpaired59 fit')
        for name, value in expected.items():
            require(fit[name] == value, 'fit differs from admitted data/sampling: ' + name)
        cal = fit['calibration']
        require(all(math.isfinite(cal[key]) and cal[key] > 0 for key in ('S0', 'R0', 'lambda'))
                and cal['lambda'] == cal['S0'] / cal['R0'] and cal['lambda_sha256'] == fit['lambda_sha256'],
                'invalid train-only lambda calibration')
        require(fit['effective_lambda'] == (0. if row['arm'] == ARMS[0] else cal['lambda']), 'effective arm weight differs')
    pairs = []
    for seed in MODEL_SEEDS:
        control, active = [next(row['fit'] for row in models if row['seed'] == seed and row['arm'] == arm) for arm in ARMS]
        equal = {name: control[name] == active[name] for name in PAIR_FIELDS}
        equal['calibration'] = control['calibration'] == active['calibration']
        require(all(equal.values()), '59 arms are not exactly paired')
        pairs.append({'seed': seed, 'equal_by_field': equal})
    for name in PAIR_FIELDS:
        if name != 'initial_model_state_sha256':
            require(len({row['fit'][name] for row in models}) == 1, 'shared identity differs: ' + name)
    calibration = {}
    for name in CALIBRATION_FIELDS:
        values = [row['fit']['calibration'][name] for row in models]
        require(all(value == values[0] for value in values), 'initial function/calibration differs across seeds: ' + name)
        calibration[name] = values[0]
    return {'status': 'COMPLETE', 'pairs': pairs, 'shared_initial_function_calibration': calibration,
            'only_variable': 'effective auxiliary loss weight',
            'seed_specific_initial_weight_hashes': {str(seed): next(row['fit']['initial_model_state_sha256']
                for row in models if row['seed'] == seed) for seed in MODEL_SEEDS}}


def cost_comparison(predictions, references):
    require(len(predictions) == len(references) == 9, 'nine cost points required')
    rows = []
    for predicted, actual in zip(predictions, references, strict=True):
        require((predicted['seed'], predicted['tick']) == (actual['seed'], actual['tick']), 'cost identities differ')
        resolved = actual['resolved']
        same_sign = predicted['gradient'] * actual['gradient'] > 0 if resolved else None
        rows.append({'seed': actual['seed'], 'tick': actual['tick'], 'reference_resolved': resolved,
                     'prediction': predicted['gradient'], 'reference': actual['gradient'], 'same_sign': same_sign,
                     'valid_match': bool(same_sign and predicted['autograd_agrees'] and predicted['direction_identifiable'])
                     if resolved else None,
                     'normalized_error': abs(predicted['gradient'] - actual['gradient']) / max(abs(actual['gradient']), 1e-5)
                     if resolved else None})
    resolved = all(row['reference_resolved'] for row in rows)
    return {'points': rows, 'resolved_count': sum(row['reference_resolved'] for row in rows),
            'valid_matches': sum(row['valid_match'] is True for row in rows),
            'resolved_sign_mismatches': sum(row['same_sign'] is False for row in rows),
            'mean_normalized_error': math.fsum(row['normalized_error'] for row in rows) / 9 if resolved else None,
            'passed_9_of_9': resolved and all(row['valid_match'] is True for row in rows)}


def first_step_comparison(predictions, references):
    require(len(predictions) == len(references) == 9, 'nine first-step points required')
    rows = []
    for predicted, actual in zip(predictions, references, strict=True):
        require((predicted['seed'], predicted['tick']) == (actual['seed'], actual['tick']), 'first-step identities differ')
        require([row['epsilon'] for row in predicted['eps']] == [row['epsilon'] for row in actual['eps']] == list(EPSILONS),
                'all four original theta eps required')
        eps = []
        for pred, ref in zip(predicted['eps'], actual['eps'], strict=True):
            value = pred['model_fd']
            require(math.isfinite(value), 'nonfinite first-step model FD')
            same_sign = value * ref['central64'] > 0 if actual['resolved'] else None
            eps.append({'epsilon': pred['epsilon'], 'model_fd': value, 'true_fd': ref['central64'],
                        'model_magnitude_admitted': abs(value) >= 1e-5, 'same_sign': same_sign,
                        'valid_match': bool(same_sign and abs(value) >= 1e-5) if actual['resolved'] else None})
        rows.append({'seed': actual['seed'], 'tick': actual['tick'], 'reference_resolved': actual['resolved'], 'eps': eps,
                     'passed': all(row['valid_match'] for row in eps) if actual['resolved'] else None})
    return {'points': rows, 'resolved_count': sum(row['reference_resolved'] for row in rows),
            'valid_matches': sum(row['passed'] is True for row in rows),
            'resolved_mismatches': sum(any(item['same_sign'] is False for item in row['eps']) for row in rows),
            'resolved_direction_failures': sum(row['passed'] is False for row in rows),
            'passed_9_of_9': all(row['passed'] is True for row in rows)}


def comparisons(models, cost_references, first_references):
    require(len(models) == 6 and {(row['seed'], row['arm']) for row in models} ==
            {(seed, arm) for seed in MODEL_SEEDS for arm in ARMS}, 'six59 models required')
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


def load_own_points(path):
    """Only call after admitting this own-generated local pickle by exact SHA."""
    with path.open('rb') as stream:
        points = pickle.load(stream)
    return [point | {'shaper': SimpleNamespace(**point['python_state']['shaper'])} for point in points]


def load_grid(directory, rows):
    import numpy as np
    grids = []
    for index, row in enumerate(rows):
        require(row['status'] == 'COMPLETE' and len(row['branches']) == 10
                and all(branch['status'] == 'COMPLETE' and branch['steps'] == 32 for branch in row['branches']), 'incomplete grid')
        require([branch['theta'] for branch in row['branches']] ==
                [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))], 'physical theta grid differs')
        grid = []
        for order in range(10):
            with np.load(directory / f'grid_{index:02d}_{order:02d}.npz', allow_pickle=False) as raw:
                branch = {name: raw[name].copy() for name in ('states', 'observations', 'rewards')}
            require(branch['states'].shape == (32, 45) and branch['observations'].shape == (32, 627)
                    and branch['rewards'].shape == (32,) and all(np.isfinite(array).all() for array in branch.values()), 'bad grid arrays')
            action = branch['observations'][:, 41:57]
            require(np.array_equal(action, action.astype(np.float32)) and bool((np.abs(action) <= 1).all()), 'actual action seam changed')
            grid.append(branch)
        require(all(np.array_equal(grid[0][name], grid[1][name]) for name in grid[0]), 'zero branches differ')
        grids.append(grid)
    return grids


def first_references(points, grids):
    result = []
    for start, grid in zip(points, grids, strict=True):
        zero = float(grid[0]['states'][0, 28])
        eps = []
        actions_resolved = True
        for index, epsilon in enumerate(EPSILONS):
            plus, minus = grid[2 + 2 * index:4 + 2 * index]
            a, b = float(plus['states'][0, 28]), float(minus['states'][0, 28])
            eps.append({'epsilon': epsilon, 'central64': (a - b) / (2 * epsilon),
                        'forward64': (a - zero) / epsilon, 'backward64': (zero - b) / epsilon})
            for branch in (plus, minus):
                actions_resolved &= action_delta_ulps(grid[0]['observations'][0, 41:57],
                                                     branch['observations'][0, 41:57])['all_four_wheels_nonzero']
        stability = local_stability(eps)
        result.append({'seed': start['seed'], 'tick': start['tick'], 'eps': eps,
                       'stability': stability, 'all_four_wheels_resolved': actions_resolved,
                       'resolved': bool(stability['locally_stable'] and actions_resolved),
                       'coordinate': 'original policy theta, actual float32 first actions; state index28'})
    return result


def evaluate_first_steps(model, points, grids, output, ledger, deadline):
    import numpy as np
    import torch

    from tools52.run_pilot52 import save_npz
    output.mkdir(parents=True, exist_ok=False)
    ledger.update(status='INCOMPLETE', points=[], completed_points=0)
    try:
        for index, (start, grid) in enumerate(zip(points, grids, strict=True)):
            require(time.process_time() < deadline, 'CPU budget before first-step diagnostic')
            row = {'seed': start['seed'], 'tick': start['tick'], 'status': 'INCOMPLETE', 'eps': []}
            ledger['points'].append(row)
            arrays = {}
            try:
                require(np.asarray(start['fifo']).shape == (0, 16), 'first-step issued/applied seam requires zero delay')
                state = torch.tensor(start['state'][None], dtype=torch.float64)
                with torch.no_grad():
                    for order, epsilon in enumerate(EPSILONS):
                        plus, minus = grid[2 + 2 * order:4 + 2 * order]
                        action_plus = torch.tensor(plus['observations'][0:1, 41:57], dtype=torch.float64)
                        action_minus = torch.tensor(minus['observations'][0:1, 41:57], dtype=torch.float64)
                        next_plus, next_minus = model(state, action_plus), model(state, action_minus)
                        fd = (next_plus - next_minus)[0] / (2 * epsilon)
                        require(bool(torch.isfinite(fd).all()), 'nonfinite first-step model FD')
                        row['eps'].append({'epsilon': epsilon, 'model_fd': float(fd[28])})
                        arrays.update({f'epsilon_{order}_{key}': value for key, value in {
                            'model_state45_fd': fd.numpy(), 'model_next_plus': next_plus.numpy(),
                            'model_next_minus': next_minus.numpy(), 'actual_action_plus': action_plus.numpy(),
                            'actual_action_minus': action_minus.numpy(), 'actual_next_plus': plus['states'][0],
                            'actual_next_minus': minus['states'][0], 'initial_state': start['state']}.items()})
                row.update(status='COMPLETE', actor_recomputed=False, observed_forward_correction=False)
                ledger['completed_points'] += 1
            except BaseException as exc:
                row['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                save_npz(output / f'point_{index:02d}.npz', arrays)
                write_json(output / f'point_{index:02d}.json', row)
        require(ledger['completed_points'] == 9, 'nine first-step diagnostics required')
        ledger['status'] = 'COMPLETE'
    finally:
        write_json(output / 'result.json', ledger)


def historical_inputs(original, summary, consumed):
    """Read heldout snapshots/targets only after all six train-only fits finish."""
    import numpy as np
    points, targets = {}, {}
    manifest = summary['raw_manifest']
    for role, old_role in (('development', 'development'), ('old_probe', 'probe')):
        bound53(f'execution_01/{old_role}_snapshots.pkl', manifest, consumed)
        points[role] = load_points(old_role, manifest)
        check_points(points[role], original['data_seeds'][old_role])
        targets[role] = []
        for point in points[role]:
            path = bound53(f'execution_01/collection_{old_role}_{point["seed"]}.npz', manifest, consumed)
            with np.load(path, allow_pickle=False) as data:
                require(np.array_equal(point['state'], data['s'][point['tick']]), 'old53 snapshot/state differs')
                targets[role].append(data['target'][point['tick']:point['tick'] + 32].copy())
    points['old55_local'] = load_own_points(ROOT / OLD55 / 'probe_snapshots.pkl')
    check_points(points['old55_local'], (55301, 55302, 55303))
    targets['old55_local'] = []
    for index in range(9):
        with np.load(ROOT / OLD55 / f'probe_{index:02d}.npz', allow_pickle=False) as raw:
            targets['old55_local'].append(raw['states_0'].copy())
    local55 = json.loads((ROOT / LOCAL55).read_text())
    old56 = json.loads((ROOT / OLD56 / 'report.json').read_text())
    require(local55['status'] == old56['status'] == 'COMPLETE', 'complete historical evidence required')
    points['old56'] = load_own_points(ROOT / OLD56 / 'probe_snapshots.pkl')
    check_points(points['old56'], (56301, 56302, 56303))
    old56_grids = load_grid(ROOT / OLD56, old56['physical_probes'])
    require(len(old56_grids) == 9, 'nine old56 physical grids required')
    targets['old56'] = [grid[0]['states'].copy() for grid in old56_grids]
    references = {'old_probe': reference53(summary['formal']['physical_probes']),
                  'old55_local': local_references(local55['points']), 'old56': local_references(old56['physical_probes'])}
    return points, targets, references, old56_grids


def refresh_fit_counts(output, report, errors):
    """Retain durable completed prefixes, including a fit that never returned."""
    for row in report['models']:
        directory = output / row['directory']
        for name in ('partial_receipt.json', 'fit_receipt.json'):
            path = directory / name
            if path.exists():
                try:
                    row['fit'] = json.loads(path.read_text())
                except (OSError, ValueError) as exc:
                    errors.append(f'fit receipt {row["directory"]}: {exc}')
                break
        completed = 0
        journal = directory / 'updates.jsonl'
        if journal.exists():
            try:
                with journal.open() as stream:
                    for line in stream:
                        item = json.loads(line)
                        require(item['update'] == completed + 1, 'nonsequential update journal')
                        completed += 1
                require(completed == row['fit'].get('updates', 0), 'receipt/update journal differs')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(f'fit update journal {row["directory"]}: {exc}')
        row['journaled_completed_updates'] = completed
        if completed > row['fit'].get('updates', 0):
            row['fit'].update(status='INCOMPLETE', updates=completed,
                              updates_attempted=max(completed, row['fit'].get('updates_attempted', 0)))
    report['model_updates'] = sum(row['fit'].get('updates', 0) for row in report['models'])
    report['model_updates_attempted'] = sum(row['fit'].get('updates_attempted', 0) for row in report['models'])
    report['completed_model_fits'] = sum(row['fit']['status'] == 'COMPLETE' for row in report['models'])


def final_training_label_diagnostics(model, labels, output, expected_model_hash):
    """Observe final CPU F32 JVP fit on all48 training rows, without a gate."""
    import torch

    from tools52.run_pilot52 import save_npz
    from tools59 import fit59
    response = fit59.yaw_jvp(model, labels, create_graph=False).detach()
    target = torch.as_tensor(labels['target'], dtype=torch.float32)
    normalized = (response - target) / float(fit59.frozen.PHYSICAL_SCALE[28])
    require(response.shape == (48,) and bool(torch.isfinite(normalized).all()), 'nonfinite final training response')
    rows = []
    for index in range(48):
        predicted, truth = float(response[index]), float(target[index])
        same_sign = predicted * truth > 0
        magnitude_admitted = abs(predicted) >= 1e-5 and abs(truth) >= 1e-5
        rows.append({'seed': int(labels['seed'][index]), 'tick': int(labels['tick'][index]),
                     'epsilon': float(labels['epsilon'][index]), 'predicted_yaw_jvp': predicted,
                     'target_float64': float(labels['target'][index]), 'target_training_float32': truth,
                     'same_sign': same_sign, 'model_abs_ge_1e_5': abs(predicted) >= 1e-5,
                     'target_abs_ge_1e_5': abs(truth) >= 1e-5, 'direction_and_magnitude_match': same_sign and magnitude_admitted,
                     'magnitude_ratio': abs(predicted) / abs(truth) if truth else None,
                     'normalized_error': float(normalized[index])})
    value = {'status': 'COMPLETE', 'dtype': 'float32', 'device': 'cpu', 'rows': rows,
             'label_rows': 48, 'independent_states': 12, 'normalized_rmse': float(normalized.square().mean().sqrt()),
             'auxiliary_loss': float(normalized.square().mean()),
             'sign_matches': sum(row['same_sign'] for row in rows),
             'direction_and_magnitude_matches': sum(row['direction_and_magnitude_match'] for row in rows),
             'used_for_gate_or_selection': False, 'parameter_updates': 0}
    save_npz(output / 'final_train_labels.npz', {'predicted_yaw_jvp': response.numpy(),
        'target_training_float32': target.numpy(), 'normalized_error': normalized.numpy(),
        **{key: labels[key] for key in ('target', 'seed', 'tick', 'epsilon')}})
    write_json(output / 'final_train_labels.json', value)
    require(fit59.paired.mapping_sha256(model.state_dict()) == expected_model_hash, 'training diagnostic changed final model')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-yaw-jvp59-result-v1', 'status': 'INCOMPLETE', 'failure': None,
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
        verify59(plan)
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
        from tools59 import fit59
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
            train = dynamics.validate_data({name: raw[name].copy() for name in fit59.paired.DATA_KEYS})
        require(len(train['s']) == 7200 and np.array_equal(np.unique(train['episode']), np.arange(12))
                and all(np.count_nonzero(train['episode'] == episode) == 600 for episode in range(12)), 'exact7200 train rows required')
        labels = fit59.load_labels58(ROOT / LABEL58)
        for index in range(48):
            training_index = (int(labels['seed'][index]) - 53101) * 600 + int(labels['tick'][index])
            require(np.array_equal(labels['s'][index], train['s'][training_index])
                    and np.array_equal(labels['a0'][index], train['a'][training_index]), 'label base state/action is not original train row')
        common = fit59.paired.common_window_indices(train).astype('<i8', copy=False)
        require(common.shape == (6828, 32), 'common32 window count differs')
        sampling_hash = fit59.paired.array_sha256(fit59.paired.sampled_window_rows(6828).numpy().astype('<i8', copy=False))
        expected = {'training_data_sha256': fit59.paired.mapping_sha256(train), 'labels_sha256': fit59.paired.mapping_sha256(labels),
                    'common_windows_sha256': fit59.paired.array_sha256(common), 'planned_sampling_rows_sha256': sampling_hash,
                    'completed_sampling_rows_sha256': sampling_hash}
        report['training_admission'] = {**expected, 'transitions': 7200, 'complete_episodes': 12,
                                        'common32_windows': 6828, 'label_rows': 48, 'independent_label_states': 12,
                                        'heldout_used_for_fit_or_calibration': False}
        complete()
        begin('training')
        for row in report['models']:
            row['status'], row['fit']['status'] = 'INCOMPLETE', 'INCOMPLETE'
            model, row['fit'] = fit59.fit_model(train, labels, row['seed'], row['arm'], args.output / row['directory'], 598.)
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
            row['final_training_label_diagnostics'] = {'status': 'INCOMPLETE'}
            row['final_training_label_diagnostics'] = final_training_label_diagnostics(
                trained[(row['seed'], row['arm'])], labels, args.output / row['directory'], row['fit']['final_model_state_sha256'])
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
                verify59(plan)
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
