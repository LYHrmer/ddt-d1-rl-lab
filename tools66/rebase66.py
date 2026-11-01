"""66: cost-only oracle relocation on all frozen64 fresh model trajectories.

The free model/actor/history path is unchanged. Observed theta0 states/actions
only translate the inputs of the cost function after the complete free rollout.
Offsets are detached once, shared by theta0, JVP and both fixed model-FD branches.
This offline diagnostic never qualifies a controller or changes64/65 decisions.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
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
from tools53.run_pilot53 import require, sha, verify, write_json
from tools65 import diagnose65 as previous

SOURCE64 = previous.SOURCE64
SOURCE65 = Path('evidence/cost_prefix65/diagnostic_01')
PLAN65 = Path('docs/experiments65/cost_prefix65_preregistered.json')
MODEL_SEEDS = previous.MODEL_SEEDS
PROBE_SEEDS = previous.PROBE_SEEDS
ARMS = previous.ARMS
TICKS = previous.TICKS
HORIZONS = previous.HORIZONS
COMPONENTS = previous.COMPONENTS


def required_inputs():
    paths = previous.required_inputs() | {str(PLAN65), str(SOURCE65 / 'report.json'),
                                         'tools66/rebase66.py', 'tools65/diagnose65.py'}
    paths.update(str(SOURCE65 / f'physical/point_{point:02d}.{suffix}')
                 for point in range(9) for suffix in ('npz', 'json'))
    paths.update(str(SOURCE65 / f'{arm}_seed{seed}/point_{point:02d}.{suffix}')
                 for seed in MODEL_SEEDS for arm in ARMS for point in range(9) for suffix in ('npz', 'json'))
    return paths


def verify66(plan):
    require(plan['schema'] == 'd1-cost-rebase66-diagnostic-v1', 'wrong66 protocol')
    require(plan['model_seeds'] == list(MODEL_SEEDS) and plan['probe_seeds'] == list(PROBE_SEEDS)
            and plan['arms'] == list(ARMS) and plan['horizons'] == list(HORIZONS)
            and plan['eps'] == list(previous.EPSILONS), 'registered66 grid changed')
    require(plan['cpu_seconds'] == 180 and plan['wall_seconds'] == 300
            and plan['expected_model_points'] == 54 and plan['new_physics_steps'] == 0
            and plan['model_updates'] == 0 and plan['policy_updates'] == 0, '66 budget changed')
    require(plan['parity_atol'] == previous.ATOL and plan['parity_rtol'] == previous.RTOL
            and plan['model_fd_epsilon'] == previous.MODEL_EPS, '66 numeric tolerances changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required66 sources/inputs not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '66 source/input changed: ' + name)


def make_rebased_rollout(free_rollout, start, baseline_states, actual_actions):
    """Return (closure, fixed_tensors) for one complete32-step free rollout.

    free_rollout(theta)->(scalar mean,trace) never accepts observed future data.
    The closure retains the free trace states/actions, replacing only costs and
    cost_components and adding cost_eval_* arrays plus free_previous_actions.
    target + (free(theta) - detached_free0) is essential: theta0 cancels exactly
    and perturbations still move cost inputs. Never detach the current trace.
    """
    import torch

    from tools54 import numerics54 as numeric
    target = torch.as_tensor(baseline_states, dtype=torch.float64).detach().clone()
    actions = torch.as_tensor(actual_actions, dtype=torch.float64).detach().clone()
    commands = torch.as_tensor(start['commands'][:32], dtype=torch.float64).detach().clone()
    first_previous = torch.as_tensor(start['last'][None], dtype=torch.float64).detach().clone()
    require(target.shape == (32, 45) and actions.shape == (32, 16) and commands.shape == (32, 3)
            and first_previous.shape == (1, 16), 'complete F64 baseline inputs required')
    require(all(value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                for value in (target, actions, commands, first_previous)), 'finite CPU baseline required')
    with torch.no_grad():
        _, origin = free_rollout(torch.tensor(0., dtype=torch.float64))
    fixed = {'origin_states': origin['states'].detach().clone(),
             'origin_actions': origin['actions'].detach().clone(),
             'baseline_states': target, 'baseline_actions': actions,
             'commands': commands, 'first_previous': first_previous}
    require(fixed['origin_states'].shape == target.shape and fixed['origin_actions'].shape == actions.shape,
            'free origin shapes differ')

    def rebased(theta):
        _, free = free_rollout(theta)
        state = target + (free['states'] - fixed['origin_states'])
        action = actions + (free['actions'] - fixed['origin_actions'])
        prior = torch.cat((first_previous, action[:-1]))
        free_prior = torch.cat((first_previous, free['actions'][:-1]))
        parts = numeric.cost_components(state, commands, action, prior, start['nominal_height'])
        costs = numeric.cost(state, commands, action, prior, start['nominal_height'])
        trace = dict(free)
        trace.update(costs=costs, cost_components=parts, cost_eval_states=state,
                     cost_eval_actions=action, cost_eval_previous_actions=prior,
                     free_previous_actions=free_prior,
                     free_costs=free['costs'], free_cost_components=free['cost_components'])
        return costs.mean(), trace

    return rebased, fixed


def sensitivity_admission(rebased, fixed, *, arrays=None):
    """One tuple JVP plus fixed±.001 FD of free versus translated cost inputs.

    All state/action/previous entries are retained. This tests the entire free
    closed-loop directional sensitivity, without claiming an A/B attribution.
    """
    import numpy as np
    import torch
    arrays = {} if arrays is None else arrays
    names = ('states', 'cost_eval_states', 'actions', 'cost_eval_actions',
             'free_previous_actions', 'cost_eval_previous_actions')

    def outputs(theta):
        _, trace = rebased(theta)
        return tuple(trace[name] for name in names)

    values, tangents = torch.autograd.functional.jvp(
        outputs, torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64),
        create_graph=False, strict=False)
    parity = {}
    for name, value, tangent in zip(names, values, tangents, strict=True):
        arrays['sensitivity_theta0_' + name] = value.numpy()
        arrays['sensitivity_jvp_' + name] = tangent.numpy()
    require(torch.equal(values[0], fixed['origin_states']) and torch.equal(values[2], fixed['origin_actions']),
            'free origin changed during rebase admission')
    require(torch.equal(values[1], fixed['baseline_states']) and torch.equal(values[3], fixed['baseline_actions']),
            'theta0 cost state/action does not exactly restore the observed baseline')
    expected_prior = torch.cat((fixed['first_previous'], fixed['baseline_actions'][:-1]))
    require(torch.equal(values[5], expected_prior), 'theta0 previous action differs from observed baseline')
    for index, label in ((0, 'states'), (2, 'actions'), (4, 'previous_actions')):
        parity[label + '_jvp'] = previous.close_receipt(tangents[index + 1].numpy(), tangents[index].numpy())
        require(parity[label + '_jvp']['passed'], 'cost translation changed free ' + label + ' JVP')
    require(bool((tangents[4][0] == 0).all()) and bool((tangents[5][0] == 0).all()),
            'first previous action must be constant')
    endpoint = []
    with torch.no_grad():
        for sign, label in ((1, 'plus'), (-1, 'minus')):
            current = outputs(torch.tensor(sign * previous.MODEL_EPS, dtype=torch.float64))
            endpoint.append(current)
            arrays.update({'sensitivity_' + label + '_' + name: value.numpy()
                           for name, value in zip(names, current, strict=True)})
    for index, name in enumerate(names):
        arrays['sensitivity_fd_' + name] = ((endpoint[0][index] - endpoint[1][index])
                                           / (2 * previous.MODEL_EPS)).numpy()
    for index, label in ((0, 'states'), (2, 'actions'), (4, 'previous_actions')):
        parity[label + '_fd'] = previous.close_receipt(arrays['sensitivity_fd_' + names[index + 1]],
                                                      arrays['sensitivity_fd_' + names[index]])
        require(parity[label + '_fd']['passed'], 'cost translation changed free ' + label + ' FD')
    require(all(np.isfinite(array).all() for array in arrays.values()), 'nonfinite sensitivity evidence')
    return {'status': 'COMPLETE', 'parity': parity, 'theta0_cost_inputs_exact': True,
            'first_previous_action_jvp_zero': True, 'offset_origin_theta': 0.,
            'offset_recomputed_per_branch': False, 'physical_perturbed_labels_used': False}


def annotate_references(result, physical):
    """Reuse65 total/component reference eligibility verbatim; unknown stays None."""
    require(physical['status'] == 'COMPLETE' and len(result['prefixes']) == len(physical['prefixes']) == 32,
            'all32 complete prefixes required')
    for prefix, reference in zip(result['prefixes'], physical['prefixes'], strict=True):
        require(prefix['horizon'] == reference['horizon'], 'prefix alignment differs')

        def annotate(row, ref):
            row.update(physical_reference_resolved=ref['resolved'],
                physical_reference_gradient=ref['reference_gradient'],
                sign_matches_reference=row['gradient'] * ref['reference_gradient'] > 0 if ref['resolved'] else None,
                valid_direction_match=bool(row['autograd_agrees'] and row['direction_identifiable']
                    and row['gradient'] * ref['reference_gradient'] > 0) if ref['resolved'] else None)

        annotate(prefix, reference)
        for name, component in prefix['components'].items():
            annotate(component, physical['component_prefixes'][name][prefix['horizon'] - 1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-cost-rebase66-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'new_physics_steps': 0, 'model_updates': 0, 'policy_updates': 0,
              'source_decisions_changed': False, 'diagnostics_complete': False,
              'offline_oracle_diagnostic': True, 'is_controller_repair': False,
              'selected_horizon': None, 'selected_seed': None, 'completed_model_points': 0,
              'components': list(COMPONENTS), 'summary_horizons': list(HORIZONS),
              'models': [{'seed': seed, 'arm': arm, 'status': 'NOT_RUN', 'points': [
                  {'seed': reset, 'tick': tick, 'status': 'NOT_RUN'} for reset in PROBE_SEEDS for tick in TICKS]}
                  for seed in MODEL_SEEDS for arm in ARMS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    plan, original, plan65 = None, None, None
    progress = {'phase': 'admission', 'model': None, 'point': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress, 'completed_model_points': report['completed_model_points'],
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
        events.flush()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (179, 180))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify66(plan)
        original = json.loads((ROOT / previous.PLAN53).read_text())
        verify(original)
        plan65 = json.loads((ROOT / PLAN65).read_text())
        previous.verify65(plan65)
        source64 = json.loads((ROOT / SOURCE64 / 'report.json').read_text())
        source65 = json.loads((ROOT / SOURCE65 / 'report.json').read_text())
        require(source64['status'] == source65['status'] == 'COMPLETE'
                and source64['verification_after'] == source65['verification_after'] == 'COMPLETE'
                and source64['completed_model_fits'] == 6 and source64['model_updates'] == 6000
                and source65['completed_model_points'] == 54 and source65['diagnostics_complete']
                and source64['plan_sha256'] == sha(ROOT / previous.PLAN64)
                and source65['plan_sha256'] == sha(ROOT / PLAN65), 'complete64/65 sources required')
        report['source64_comparison'] = source64['comparison']
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == source64['versions'] == source65['versions'],
                'package versions differ')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools54 import numerics54 as numeric
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == source64['torch_runtime_version'] == source65['torch_runtime_version'],
                'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        actor = copy.deepcopy(actor32).double().eval().requires_grad_(False)
        # Only SHA-bound own-generated local snapshots are unpickled.
        with (ROOT / SOURCE64 / 'probe_snapshots.pkl').open('rb') as stream:
            stored = pickle.load(stream)
        points = [point | {'shaper': SimpleNamespace(**point['python_state']['shaper'])} for point in stored]
        require([(point['seed'], point['tick']) for point in points] ==
                [(seed, tick) for seed in PROBE_SEEDS for tick in TICKS], '64 point identities differ')
        physical = []
        for index, start in enumerate(points):
            row = source65['physical'][index]
            require(json.loads((ROOT / SOURCE65 / f'physical/point_{index:02d}.json').read_text()) == row
                    and row['status'] == 'COMPLETE' and (row['seed'], row['tick']) == (start['seed'], start['tick'])
                    and row['source64_reference'] == source64['cost_references']['fresh'][index]
                    and row['prefixes'][-1]['resolved'] == row['source64_reference']['resolved'],
                    '65 physical reference identity differs')
            require(np.asarray(start['fifo']).shape == (0, 16), 'actual action requires original zero delay')
            with np.load(ROOT / SOURCE64 / f'grid_{index:02d}_00.npz', allow_pickle=False) as raw:
                target, action = raw['states'].copy(), raw['observations'][:, 41:57].copy()
            with np.load(ROOT / SOURCE65 / f'physical/point_{index:02d}.npz', allow_pickle=False) as raw:
                require(np.array_equal(target, raw['branch_0_states'])
                        and np.array_equal(action, raw['branch_actual_actions'][0]), '65 baseline arrays differ from64')
                cost, parts = raw['branch_costs'][0].copy(), raw['branch_components'][0].copy()
            old = source64['physical_probes'][index]['branches'][0]
            require(old == json.loads((ROOT / SOURCE64 / f'grid_{index:02d}_00.json').read_text())
                    and old['status'] == 'COMPLETE' and old['steps'] == 32
                    and np.array_equal(cost, old['per_step_cost64']), 'complete64 baseline costs required')
            physical.append({'target': target, 'action': action, 'cost': cost, 'parts': parts,
                             'mean_cost': old['cost64'], 'reference': row})
        report['physical_reference_source'] = str(SOURCE65 / 'report.json')
        event('ADMITTED')
        for model_row in report['models']:
            require(time.process_time() < 178., 'CPU deadline before model')
            model_row['status'] = 'INCOMPLETE'
            directory = f'{model_row["arm"]}_seed{model_row["seed"]}'
            output = args.output / directory
            output.mkdir(exist_ok=False)
            progress.update(phase='model', model=directory, point=None)
            old64 = next(row for row in source64['models'] if (row['arm'], row['seed']) == (model_row['arm'], model_row['seed']))
            old65 = next(row for row in source65['models'] if (row['arm'], row['seed']) == (model_row['arm'], model_row['seed']))
            require(old64['status'] == old65['status'] == 'COMPLETE' and old64['fit']['status'] == 'COMPLETE'
                    and old64['fit']['updates'] == 1000, 'complete64/65 model evidence required')
            weights = directory + '/final_state_dict.pt'
            require(sha(ROOT / SOURCE64 / weights) == source64['sealed_artifacts_sha256'][weights], '64 final weight differs')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(ROOT / SOURCE64 / weights, map_location='cpu', weights_only=True))
            model = model.double().eval().requires_grad_(False)
            for index, start in enumerate(points):
                require(time.process_time() < 178., 'CPU deadline before numeric point')
                row = model_row['points'][index]
                row['status'] = 'INCOMPLETE'
                free_arrays, rebased_arrays, evidence = {}, {}, {}
                progress.update(phase='model_point', point=index)
                event('POINT_BEGIN')
                try:
                    saved64 = old64['diagnostics']['fresh']['points'][index]
                    saved65 = old65['points'][index]
                    path64 = ROOT / SOURCE64 / directory / f'fresh/point_{index:02d}'
                    path65 = ROOT / SOURCE65 / directory / f'point_{index:02d}'
                    require(json.loads(path64.with_suffix('.json').read_text()) == saved64
                            and sha(path64.with_suffix('.npz')) == saved64['trace_sha256']
                            and json.loads(path65.with_suffix('.json').read_text()) == saved65
                            and saved64['status'] == saved65['status'] == 'COMPLETE', 'saved model point differs')
                    with np.load(path64.with_suffix('.npz'), allow_pickle=False) as raw:
                        trace64 = {key: raw[key].copy() for key in raw.files}

                    def free_rollout(theta, model=model, start=start):
                        return numeric.imaginary(model, actor, start, theta, return_trace=True)

                    free, _ = previous.decompose_model(free_rollout, expected_trace=trace64,
                        expected_receipt=saved64, arrays=free_arrays)
                    with np.load(path65.with_suffix('.npz'), allow_pickle=False) as raw:
                        require(set(free_arrays) == set(raw.files), '65 decomposition array fields differ')
                        row['free65_array_parity'] = {key: previous.close_receipt(value, raw[key])
                                                      for key, value in free_arrays.items()}
                    require(all(check['passed'] for check in row['free65_array_parity'].values()), 'free65 arrays did not reproduce')
                    row['free65_scalar_parity'] = {key: previous.close_receipt(free[key], saved65[key])
                                                  for key in ('cost', 'gradient', 'model_central_difference')}
                    require(all(check['passed'] for check in row['free65_scalar_parity'].values()), 'free65 scalar did not reproduce')
                    baseline = physical[index]
                    rebased, fixed = make_rebased_rollout(free_rollout, start, baseline['target'], baseline['action'])
                    evidence.update({'fixed_' + key: value.numpy().copy() for key, value in fixed.items()})
                    fixed_before = {key: value.clone() for key, value in fixed.items()}
                    row['sensitivity_admission'] = sensitivity_admission(rebased, fixed, arrays=evidence)
                    relocated, _ = previous.decompose_model(rebased, arrays=rebased_arrays)
                    row['rebased_theta0_physical_cost_exact'] = bool(np.array_equal(
                        rebased_arrays['theta0_costs'], baseline['cost']) and relocated['cost'] == baseline['mean_cost'])
                    row['rebased_theta0_physical_components_exact'] = bool(np.array_equal(
                        rebased_arrays['theta0_cost_components'], baseline['parts']))
                    require(row['rebased_theta0_physical_cost_exact'] and row['rebased_theta0_physical_components_exact'],
                            'rebased theta0 cost does not exactly restore the physical baseline')
                    require(all(torch.equal(fixed[key], old) for key, old in fixed_before.items()), 'fixed offset inputs mutated')
                    row.update(free=annotate_references(free, baseline['reference']),
                               cost_rebased=annotate_references(relocated, baseline['reference']),
                               source64_reference=baseline['reference']['source64_reference'],
                               offset_inputs_unchanged=True, status='COMPLETE')
                    report['completed_model_points'] += 1
                except BaseException as exc:
                    row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    arrays = {'free_' + key: value for key, value in free_arrays.items()}
                    arrays.update({'rebased_' + key: value for key, value in rebased_arrays.items()})
                    arrays.update(evidence)
                    shared.save_npz(output / f'point_{index:02d}.npz', arrays)
                    write_json(output / f'point_{index:02d}.json', row)
                    event('POINT_SAVED')
            model_row['status'] = 'COMPLETE'
            model_row['summary_by_horizon'] = {str(h): {name: {
                'resolved_references': sum(point[name]['prefixes'][h - 1]['physical_reference_resolved'] for point in model_row['points']),
                'valid_direction_matches': sum(point[name]['prefixes'][h - 1]['valid_direction_match'] is True for point in model_row['points']),
                'total_selfchecks': sum(point[name]['prefixes'][h - 1]['autograd_agrees'] for point in model_row['points']),
                'component_selfchecks': sum(point[name]['prefixes'][h - 1]['all_component_selfchecks'] for point in model_row['points'])}
                for name in ('free', 'cost_rebased')} for h in HORIZONS}
            write_json(output / 'result.json', model_row)
        require(report['completed_model_points'] == 54, 'all6x9 points required')
        report.update(status='COMPLETE', diagnostics_complete=True)
    except BaseException as exc:  # noqa: BLE001 - partial diagnostics are preserved, never resumed.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], '66 plan changed')
                verify66(plan)
            if plan65 is not None:
                previous.verify65(plan65)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            report['verification_after'] = 'COMPLETE' if original is not None and plan65 is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - retain both execution and final verification failures.
            report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=str(exc))
        if report['status'] != 'COMPLETE':
            report['diagnostics_complete'] = False
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'completed_model_points', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
