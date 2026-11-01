"""57: read-only56 action-response and oracle-linearization diagnostics.

No environment is constructed, no physical step is taken, and no parameter is
updated. Teacher-forced comparisons use each stored physical branch's own
pre-state and actual float32-issued action promoted to double. Only their first
step isolates action response at a shared state; later steps also contain the
true perturbed-state contribution. Oracle rebasings diagnose the existing free
gradient and are never substituted for its56 qualification result.
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
from tools53.run_pilot53 import require, sha, verify, write_json

SOURCE56 = Path('evidence/paired_horizon56/execution_01')
ORIGINAL_PLAN = Path('docs/experiments53/model_gradient53_preregistered.json')
MODEL_SEEDS = (56001, 56002, 56003)
HORIZONS = (8, 32)
PROBE_SEEDS = (56301, 56302, 56303)
TICKS = (110, 260, 410)
EPSILONS = (.00003125, .0000625, .000125, .00025)
MODEL_FD_EPS = .001


def verify57(plan):
    require(plan['schema'] == 'd1-action-response57-diagnostic-v1', 'wrong57 protocol')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420, 'time budget changed')
    required = {'tools57/diagnose57.py', 'tools54/numerics54.py', 'tools54/center54.py',
                'tools55/observed55.py', 'tools52/adapter52.py', 'tools52/dynamics52.py',
                'tools53/run_pilot53.py', str(ORIGINAL_PLAN),
                str(SOURCE56 / 'report.json'), str(SOURCE56 / 'probe_snapshots.pkl')}
    required.update(str(SOURCE56 / f'grid_{point:02d}_{branch:02d}.npz')
                    for point in range(9) for branch in range(10))
    required.update(str(SOURCE56 / f'h{horizon}_seed{seed}/final_state_dict.pt')
                    for horizon in HORIZONS for seed in MODEL_SEEDS)
    require(required <= set(plan['source_sha256']), 'required source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '57 source/input changed: ' + name)


def gradient_receipt(start, value, analytic, numerical):
    require(all(math.isfinite(number) for number in (value, analytic, numerical)), 'nonfinite gradient diagnostic')
    return {'seed': start['seed'], 'tick': start['tick'], 'cost': value, 'gradient': analytic,
            'model_central_difference': numerical, 'selfcheck_epsilon': MODEL_FD_EPS,
            'direction_identifiable': abs(analytic) >= 1e-5 and abs(numerical) >= 1e-5 and analytic * numerical > 0,
            'autograd_agrees': abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4}


def cost_only_rebase(model, actor, start, baseline_states, actual_actions):
    """54 cost-only rebase: retain the free dynamics/actor linearization path.

    At theta0 shift cost states/actions to observations, with a fixed detached
    offset shared by theta0 and ±.001. This changes cost partial derivatives,
    not the free rollout's composite state/action sensitivity. No calibration
    from physical perturbed trajectories or physical FD labels enters here.
    """
    import torch

    from tools54 import numerics54 as numeric
    target = torch.as_tensor(baseline_states, dtype=torch.float64).detach().clone()
    action = torch.as_tensor(actual_actions, dtype=torch.float64).detach().clone()
    require(target.shape == (32, 45) and action.shape == (32, 16), 'complete baseline/action shapes required')
    command = torch.tensor(start['commands'][:32], dtype=torch.float64)
    previous = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), action[:-1]))
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    _, trace = numeric.imaginary(model, actor, start, theta, return_trace=True)
    origin = {key: trace[key].detach().clone() for key in ('states', 'actions')}

    def rebased(current):
        states = target + (current['states'] - origin['states'])
        actions = action + (current['actions'] - origin['actions'])
        prior = previous + torch.cat((torch.zeros((1, 16), dtype=torch.float64),
                                      current['actions'][:-1] - origin['actions'][:-1]))
        return numeric.cost(states, command, actions, prior, start['nominal_height']).mean()

    value = rebased(trace)
    derivative, = torch.autograd.grad(value, theta)
    with torch.no_grad():
        _, plus = numeric.imaginary(model, actor, start, torch.tensor(MODEL_FD_EPS, dtype=torch.float64), return_trace=True)
        _, minus = numeric.imaginary(model, actor, start, torch.tensor(-MODEL_FD_EPS, dtype=torch.float64), return_trace=True)
        numerical = float((rebased(plus) - rebased(minus)) / (2 * MODEL_FD_EPS))
    receipt = gradient_receipt(start, float(value.detach()), float(derivative), numerical)
    receipt.update(scope='offline cost-only oracle diagnostic; unchanged free linearization path',
                   requires_observed_theta0_baseline=True, physical_perturbed_labels_used=False)
    return receipt, {key: tensor.detach() for key, tensor in trace.items()}


def group_rms(error):
    """Seven physical groups; normalized RMS has units of response per theta."""
    import torch

    from tools52 import dynamics52 as dynamics
    require(error.ndim == 2 and error.shape[-1] == 45, 'state response must be [steps,45]')
    normalized = error / torch.tensor(dynamics.PHYSICAL_SCALE, dtype=torch.float64)
    return {name: float(normalized[:, list(indices)].square().mean().sqrt())
            for name, indices in dynamics.GROUPS.items()}


def response_diagnostics(model, start, branches):
    """Four fixed epsilons, using actual pre-states/actions on each ± branch.

    Returned arrays retain all32 steps. At t0, pre-state+ equals pre-state−;
    later they intentionally differ. There is no actor evaluation in this
    function, so actor32/64 differences cannot become model-response error.
    """
    import torch
    rows, arrays = [], {}
    initial = torch.tensor(start['state'][None], dtype=torch.float64)
    with torch.no_grad():
        for index, epsilon in enumerate(EPSILONS):
            plus, minus = branches[2 + 2 * index:4 + 2 * index]
            physical, pre_states, actions, predicted = {}, {}, {}, {}
            for sign, branch in (('plus', plus), ('minus', minus)):
                physical[sign] = torch.tensor(branch['states'], dtype=torch.float64)
                actions[sign] = torch.tensor(branch['observations'][:, 41:57], dtype=torch.float64)
                pre_states[sign] = torch.cat((initial, physical[sign][:-1]))
                # Preserve the frozen batch1 arithmetic; no rollout uses the
                # previous model output as the next input in this diagnostic.
                predicted[sign] = torch.cat([model(pre_states[sign][tick:tick + 1], actions[sign][tick:tick + 1])
                                              for tick in range(32)])
                require(bool(torch.isfinite(predicted[sign]).all()), 'nonfinite teacher-forced model response')
            actual_fd = (physical['plus'] - physical['minus']) / (2 * epsilon)
            model_fd = (predicted['plus'] - predicted['minus']) / (2 * epsilon)
            action_fd = (actions['plus'] - actions['minus']) / (2 * epsilon)
            error = model_fd - actual_fd
            error_rms, actual_rms = group_rms(error), group_rms(actual_fd)
            rows.append({'epsilon': epsilon, 'normalized_rms_error_by_group': error_rms,
                         'normalized_rms_actual_response_by_group': actual_rms,
                         'normalized_rms_model_response_by_group': group_rms(model_fd),
                         'relative_rms_error_by_group': {name: error_rms[name] / max(actual_rms[name], 1e-12)
                                                         for name in error_rms},
                         'first_step_normalized_rms_error_by_group': group_rms(error[:1]),
                         'first_step_yaw28': {'actual_fd': float(actual_fd[0, 28]),
                                              'model_fd': float(model_fd[0, 28]), 'error': float(error[0, 28])},
                         'first_step_actual_action_fd': action_fd[0].tolist(),
                         'first_step_actual_actions': {sign: actions[sign][0].tolist() for sign in ('plus', 'minus')},
                         'first_step_prestate_exact': bool(torch.equal(pre_states['plus'][0], pre_states['minus'][0])),
                         'actor_evaluated': False, 'fd_denominator': 2 * epsilon})
            prefix = f'epsilon_{index}_'
            arrays.update({prefix + key: value.numpy() for key, value in {
                'actual_state_fd': actual_fd, 'model_state_fd': model_fd, 'state_fd_error': error,
                'actual_action_fd': action_fd,
                **{sign + '_pre_states': pre_states[sign] for sign in ('plus', 'minus')},
                **{sign + '_actual_actions': actions[sign] for sign in ('plus', 'minus')},
                **{sign + '_actual_next_states': physical[sign] for sign in ('plus', 'minus')},
                **{sign + '_model_next_states': predicted[sign] for sign in ('plus', 'minus')},
            }.items()})
    return rows, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-action-response57-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'new_physics_steps': 0, 'model_updates': 0, 'policy_updates': 0, 'rejudges56': False,
              'oracle_diagnostic_is_repair': False, 'completed_points': 0, 'eps': list(EPSILONS),
              'models': [{'horizon': horizon, 'seed': seed, 'status': 'NOT_RUN', 'points': [
                  {'seed': reset, 'tick': tick, 'status': 'NOT_RUN'} for reset in PROBE_SEEDS for tick in TICKS]}
                  for seed in MODEL_SEEDS for horizon in HORIZONS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    plan, original = None, None
    progress = {'phase': 'admission', 'model': None, 'point': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress, 'completed_points': report['completed_points'],
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
        events.flush()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (299, 300))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify57(plan)
        original = json.loads((ROOT / ORIGINAL_PLAN).read_text())
        verify(original)
        source = json.loads((ROOT / SOURCE56 / 'report.json').read_text())
        require(source['status'] == 'COMPLETE' and source['model_updates'] == 6000
                and source['completed_model_fits'] == 6 and source['counts']['policy_transitions'] == 4680,
                'complete56 evidence required')
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == source['versions'], 'package versions differ')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools54 import numerics54 as numeric
        from tools55.observed55 import predict_gradients
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == source['torch_runtime_version'], 'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        actor = copy.deepcopy(actor32).double().eval().requires_grad_(False)
        # Only own-generated snapshots whose exact bytes were admitted above.
        with (ROOT / SOURCE56 / 'probe_snapshots.pkl').open('rb') as stream:
            stored = pickle.load(stream)
        points = [point | {'shaper': SimpleNamespace(**point['python_state']['shaper'])} for point in stored]
        require([(point['seed'], point['tick']) for point in points] ==
                [(seed, tick) for seed in PROBE_SEEDS for tick in TICKS], '56 point identities differ')
        physical = []
        sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
        for index, point in enumerate(points):
            require(np.asarray(point['fifo']).shape == (0, 16), 'issued action requires original zero delay')
            old = source['physical_probes'][index]
            require((old['seed'], old['tick']) == (point['seed'], point['tick']) and old['status'] == 'COMPLETE'
                    and [row['theta'] for row in old['branches']] == sequence, '56 physical branch grid differs')
            branches = []
            for order in range(10):
                require(old['branches'][order]['status'] == 'COMPLETE'
                        and old['branches'][order]['steps'] == 32, 'partial physical branch is ineligible')
                with np.load(ROOT / SOURCE56 / f'grid_{index:02d}_{order:02d}.npz', allow_pickle=False) as raw:
                    branch = {key: raw[key].copy() for key in ('states', 'observations', 'rewards')}
                require(branch['states'].shape == (32, 45) and branch['observations'].shape == (32, 627)
                        and branch['rewards'].shape == (32,) and all(np.isfinite(value).all() for value in branch.values()),
                        'nonfinite/incomplete physical branch arrays')
                issued = branch['observations'][:, 41:57]
                require(np.array_equal(issued, issued.astype(np.float32).astype(issued.dtype))
                        and bool((np.abs(issued) <= 1).all()), 'stored actions must retain actual clipped float32 values')
                branches.append(branch)
            require(all(np.array_equal(branches[0][key], branches[1][key]) for key in branches[0]),
                    'original repeated zero branch differs')
            physical.append(branches)
        event('ADMITTED')
        for model_row in report['models']:
            require(time.process_time() < 298., 'CPU budget before model')
            model_row['status'] = 'INCOMPLETE'
            directory = f'h{model_row["horizon"]}_seed{model_row["seed"]}'
            progress.update(phase='model', model=directory, point=None)
            output = args.output / directory
            output.mkdir(exist_ok=False)
            old_model = next(row for row in source['models']
                             if (row['horizon'], row['seed']) == (model_row['horizon'], model_row['seed']))
            require(old_model['status'] == 'COMPLETE' and old_model['fit']['status'] == 'COMPLETE'
                    and old_model['fit']['updates'] == 1000, 'incomplete source model')
            weight_name = directory + '/final_state_dict.pt'
            require(sha(ROOT / SOURCE56 / weight_name) == source['sealed_artifacts_sha256'][weight_name],
                    '56 final weight identity differs')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(ROOT / SOURCE56 / weight_name, map_location='cpu', weights_only=True))
            model = model.double().eval().requires_grad_(False)
            for index, (start, branches) in enumerate(zip(points, physical, strict=True)):
                require(time.process_time() < 298., 'CPU budget before point')
                row = model_row['points'][index]
                row['status'] = 'INCOMPLETE'
                progress.update(phase='point', point=index)
                event('POINT_BEGIN')
                arrays = {}
                try:
                    free, = numeric.gradients(model, actor, [start])
                    saved = old_model['diagnostics']['fresh']['points'][index]
                    parity = {key: value == saved[key] for key, value in free.items()}
                    row.update(free=free, free_exact_by_field=parity, free_reproduces56_exactly=all(parity.values()))
                    require(row['free_reproduces56_exactly'], 'free gradient did not reproduce56 exactly')
                    row['responses'], arrays = response_diagnostics(model, start, branches)
                    target, actual_actions = branches[0]['states'], branches[0]['observations'][:, 41:57]
                    rebased, free_trace = cost_only_rebase(model, actor, start, target, actual_actions)
                    row['cost_only_rebase'] = rebased
                    row['cost_only_baseline_exact'] = rebased['cost'] == source['physical_probes'][index]['branches'][0]['cost64']
                    require(row['cost_only_baseline_exact'], 'cost rebase did not reproduce recorded double baseline cost')
                    observed, traces = predict_gradients(model, actor, start, target, return_trace=True)
                    row['observed_trajectory_rebase'] = observed
                    row['teacher64_vs_physical32_action_max_gap'] = float(
                        (traces['teacher']['actions'] - torch.tensor(actual_actions, dtype=torch.float64)).abs().max())
                    row['free64_vs_physical32_first_action_max_gap'] = float(
                        (free_trace['actions'][0] - torch.tensor(actual_actions[0], dtype=torch.float64)).abs().max())
                    for name, trace in (('free_theta0', free_trace), ('observed_teacher', traces['teacher']),
                                        ('observed_theta0', traces['centered'])):
                        arrays.update({name + '_' + key: value.detach().numpy() for key, value in trace.items()})
                    reference = source['references']['fresh'][index]
                    require((reference['seed'], reference['tick']) == (start['seed'], start['tick']), 'reference identity differs')
                    row.update(reference=reference, diagnostic_sign_agreement={name: result['gradient'] * reference['gradient'] > 0
                        if reference['resolved'] else None for name, result in
                        (('free', free), ('cost_only_rebase', rebased), ('observed_trajectory_rebase', observed))},
                        source56_decision_changed=False, status='COMPLETE')
                    row['diagnostic_valid_agreement'] = {name: bool(reference['resolved']
                        and result['autograd_agrees'] and result['direction_identifiable']
                        and result['gradient'] * reference['gradient'] > 0)
                        for name, result in (('free', free), ('cost_only_rebase', rebased),
                                             ('observed_trajectory_rebase', observed))}
                    report['completed_points'] += 1
                except BaseException as exc:
                    row['failure'] = f'{type(exc).__name__}: {exc}'
                    if hasattr(exc, 'admission'):
                        row['failed_oracle_admission'] = exc.admission
                    raise
                finally:
                    shared.save_npz(output / f'point_{index:02d}.npz', arrays)
                    write_json(output / f'point_{index:02d}.json', row)
                    event('POINT_SAVED')
            model_row['status'] = 'COMPLETE'
            model_row['diagnostic_sign_counts'] = {name: sum(row['diagnostic_sign_agreement'][name] is True
                for row in model_row['points']) for name in ('free', 'cost_only_rebase', 'observed_trajectory_rebase')}
            model_row['diagnostic_valid_counts'] = {name: sum(row['diagnostic_valid_agreement'][name]
                for row in model_row['points']) for name in ('free', 'cost_only_rebase', 'observed_trajectory_rebase')}
            write_json(output / 'result.json', model_row)
        require(report['completed_points'] == 54, 'all6 models × all9 points required')
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - partial diagnostics are never resumed or substituted for56.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify57(plan)
            if original is not None:
                verify(original)
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - keep both the original and verification failure.
            report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=str(exc))
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'completed_points', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
