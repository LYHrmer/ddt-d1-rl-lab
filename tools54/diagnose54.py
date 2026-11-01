"""Precision and horizon diagnostics on sealed53 evidence; no physics or fitting."""
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
from tools53.run_pilot53 import require, sha, verify, write_json
from tools54.reproduce54 import load_points

HORIZONS = (1, 2, 4, 8, 16, 32)
EPSILONS = (.0005, .001, .002, .005, .01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'failure': None, 'new_physics_steps': 0,
              'model_updates': 0, 'policy_updates': 0, 'models': [], 'physical_readback': [],
              'model_rollouts_completed': 0, 'autograd_calls_completed': 0,
              'cost_component_names': ['vx', 'yaw', 'lateral', 'tilt', 'height', 'action_smoothness'],
              'response_comparison_epsilon': .005, 'original_selfcheck_epsilon': .001}
    write_json(args.output / 'initial_incomplete.json', report)
    started = time.process_time()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        own_plan = json.loads(args.plan.read_text())
        for name, digest in own_plan['source_sha256'].items():
            require(sha(ROOT / name) == digest, '54 source changed: ' + name)
        report['plan_sha256'] = sha(args.plan)
        plan = json.loads((ROOT / 'docs/experiments53/model_gradient53_preregistered.json').read_text())
        verify(plan)
        report['versions'] = {name: importlib.metadata.version(name) for name in plan['versions']}
        require(report['versions'] == plan['versions'], 'runtime package versions differ')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'thread/GPU limits differ')
        summary = json.loads((ROOT / 'reports/model_gradient53_results_20261101.json').read_text())
        manifest, formal = summary['raw_manifest'], summary['formal']
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import dynamics52 as dynamics
        from tools54 import numerics54 as numeric
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        actor32 = adapter.load_actor(ROOT / plan['checkpoint'])
        actor64 = copy.deepcopy(actor32).double()
        points = {role: load_points(role, manifest) for role in ('development', 'probe')}

        def rollout(model, actor, start, theta):
            value, trace = numeric.imaginary(model, actor, start, theta, return_trace=True)
            report['model_rollouts_completed'] += 1
            return value, trace

        def prefix(values):
            return {str(h): float(values[:h].mean()) for h in HORIZONS}

        true_traces = []
        for index, start in enumerate(points['probe']):
            name = f'execution_01/probe_{index:02d}.npz'
            path = ROOT / 'evidence/model_gradient53' / name
            require(sha(path) == manifest[name]['sha256'], 'physical trace identity differs')
            saved = np.load(path, allow_pickle=False)
            branches = {}
            for order in range(6):
                state = torch.tensor(saved[f'states_{order}'], dtype=torch.float64)
                action = torch.tensor(saved[f'observations_{order}'][:, 41:57], dtype=torch.float64)
                previous = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), action[:-1]))
                command = torch.tensor(start['commands'][:32], dtype=torch.float64)
                with torch.no_grad():
                    costs32 = numeric.cost(state.float(), command.float(), action.float(), previous.float(), start['nominal_height'])
                    costs64 = numeric.cost(state, command, action, previous, start['nominal_height'])
                    components = numeric.cost_components(state, command, action, previous, start['nominal_height'])
                # Frozen physical code computes a Python mean of scalar float32 values.
                original = formal['physical_probes'][index]['branches'][order]['cost']
                require(float(costs32.double().mean()) == original, 'physical cost reconstruction differs')
                branches[order] = {'state': state, 'action': action, 'costs': costs64, 'components': components}
            small = {k: (branches[2][k] - branches[3][k]) / .01 for k in branches[2]}
            large = {k: (branches[4][k] - branches[5][k]) / .02 for k in branches[2]}
            true_traces.append(small)
            report['physical_readback'].append({'seed': start['seed'], 'tick': start['tick'],
                'original_float32_cost_gradient': formal['physical_probes'][index]['gradient'],
                'float64_cost_gradient': float(small['costs'].mean()),
                'float64_double_epsilon_gradient': float(large['costs'].mean()),
                'prefix_gradient': prefix(small['costs']),
                'double_epsilon_prefix_gradient': prefix(large['costs']),
                'cost_component_gradients': small['components'].mean(0).tolist(),
                'yaw_state_derivative_per_step': small['state'][:, 28].tolist(),
                'yaw_baseline_per_step': branches[0]['state'][:, 28].tolist(),
                'yaw_command_per_step': start['commands'][:32, 2].tolist(),
                'cost_derivative_per_step': small['costs'].tolist(),
                'first_step_state_derivative45': small['state'][0].tolist(),
                'first_step_action_derivative16': small['action'][0].tolist()})

        for old_model in formal['seeds']:
            seed = old_model['seed']
            path = ROOT / f'evidence/model_gradient53/execution_01/model_{seed}/final_state_dict.pt'
            require(sha(path) == manifest[f'execution_01/model_{seed}/final_state_dict.pt']['sha256'], 'weight identity differs')
            model32 = dynamics.Dynamics52()
            model32.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model32.eval().requires_grad_(False)
            model64 = copy.deepcopy(model32).double()
            rows = []
            for role, starts in points.items():
                field = 'development_gradients' if role == 'development' else 'test_gradients'
                for index, (start, old) in enumerate(zip(starts, old_model[field], strict=True)):
                    theta32 = torch.tensor(0., requires_grad=True)
                    value32, _ = rollout(model32, actor32, start, theta32)
                    derivative32, = torch.autograd.grad(value32, theta32)
                    report['autograd_calls_completed'] += 1
                    with torch.no_grad():
                        plus32, _ = rollout(model32, actor32, start, torch.tensor(.001))
                        minus32, _ = rollout(model32, actor32, start, torch.tensor(-.001))
                    require(float(value32.detach()) == old['cost'] and float(derivative32) == old['gradient']
                            and float((plus32 - minus32) / .002) == old['model_central_difference'],
                            'new float32 path does not reproduce original cost/gradient/FD exactly')
                    theta64 = torch.tensor(0., dtype=torch.float64, requires_grad=True)
                    value64, trace = rollout(model64, actor64, start, theta64)
                    derivative64, = torch.autograd.grad(value64, theta64)
                    report['autograd_calls_completed'] += 1
                    checks = []
                    traces = None
                    larger_prefix = None
                    clamp_rows = []
                    for eps in EPSILONS:
                        with torch.no_grad():
                            plus, plus_trace = rollout(model64, actor64, start, torch.tensor(eps, dtype=torch.float64))
                            minus, minus_trace = rollout(model64, actor64, start, torch.tensor(-eps, dtype=torch.float64))
                        fd = float((plus - minus) / (2 * eps))
                        checks.append({'epsilon': eps, 'central_difference': fd,
                            'absolute_gap': abs(float(derivative64) - fd),
                            'original_tolerance_passes': abs(float(derivative64) - fd) <= .1 * abs(fd) + 1e-4})
                        if eps == .005:
                            commands = torch.tensor(start['commands'][:32, 2], dtype=torch.float64)
                            for label, branch in (('zero', trace), ('plus', plus_trace), ('minus', minus_trace)):
                                clamp_rows.append({'branch': label,
                                    'issued_action_min_margin': 1. - float(branch['actions'].detach().abs().max()),
                                    'residual_error_min_margin': 1. - float(((commands - branch['frames'].detach()[:, 2] / .25) / .3).abs().max())})
                        if eps == .005:
                            traces = {k: (plus_trace[k].detach() - minus_trace[k].detach()) / (2 * eps)
                                      for k in ('costs', 'states', 'actions', 'cost_components')}
                        if eps == .01:
                            larger_prefix = prefix((plus_trace['costs'] - minus_trace['costs']) / (2 * eps))
                    row = {'role': role, 'seed': start['seed'], 'tick': start['tick'],
                        'float32_exact_parity': True, 'cost64': float(value64.detach()),
                        'gradient32': old['gradient'], 'gradient64': float(derivative64), 'checks': checks,
                        'action_residual_clamp_margins': clamp_rows,
                        'prefix_gradient': prefix(traces['costs']),
                        'double_epsilon_prefix_gradient': larger_prefix,
                        'cost_component_gradients': traces['cost_components'].mean(0).tolist(),
                        'yaw_state_derivative_per_step': traces['states'][:, 28].tolist(),
                        'yaw_baseline_per_step': trace['states'].detach()[:, 28].tolist(),
                        'cost_derivative_per_step': traces['costs'].tolist(),
                        'first_step_state_derivative45': traces['states'][0].tolist(),
                        'first_step_action_derivative16': traces['actions'][0].tolist()}
                    if role == 'probe':
                        true_gradient = float(true_traces[index]['costs'].mean())
                        row['true_gradient'] = true_gradient
                        row['same_sign_as_physics'] = float(derivative64) * true_gradient > 0
                        row['first_step_action_derivative_max_gap'] = float(
                            (traces['actions'][0] - true_traces[index]['action'][0]).abs().max())
                    rows.append(row)
            item = {'seed': seed, 'points': rows, 'float32_parity_count': len(rows),
                'float64_original_epsilon_passes': sum(next(c for c in r['checks'] if c['epsilon'] == .001)['original_tolerance_passes'] for r in rows),
                'float64_probe_sign_matches': sum(r['same_sign_as_physics'] for r in rows if r['role'] == 'probe')}
            report['models'].append(item)
            write_json(args.output / f'model_{seed}.json', item)
        verify(plan)
        for name, digest in own_plan['source_sha256'].items():
            require(sha(ROOT / name) == digest, '54 source changed: ' + name)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - all partial evidence remains in a new directory.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['cpu_seconds'] = time.process_time() - started
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'cpu_seconds', 'new_physics_steps',
                        'model_rollouts_completed', 'autograd_calls_completed')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
