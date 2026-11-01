"""Oracle trajectory ablations for diagnosis only; never a deployable model."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'failure': None, 'new_physics_steps': 0,
              'model_updates': 0, 'policy_updates': 0, 'models': [],
              'oracle_diagnostic_only': True, 'quaternion_reprojection': False}
    write_json(args.output / 'initial_incomplete.json', report)
    started = time.process_time()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (180, 180))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        own_plan = json.loads(args.plan.read_text())
        for path, digest in own_plan['source_sha256'].items():
            require(sha(ROOT / path) == digest, 'source changed: ' + path)
        report['plan_sha256'] = sha(args.plan)
        plan = json.loads((ROOT / 'docs/experiments53/model_gradient53_preregistered.json').read_text())
        verify(plan)
        require({k: importlib.metadata.version(k) for k in plan['versions']} == plan['versions'], 'versions differ')
        summary = json.loads((ROOT / 'reports/model_gradient53_results_20261101.json').read_text())
        diagnostic = json.loads((ROOT / 'evidence/model_gradient54/diagnostic_01/report.json').read_text())
        require(diagnostic['status'] == 'COMPLETE', 'complete precision diagnosis required')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import dynamics52 as dynamics
        from tools54 import numerics54 as numeric
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        actor = adapter.load_actor(ROOT / plan['checkpoint']).double()
        points = load_points('probe', summary['raw_manifest'])

        class Teacher(torch.nn.Module):
            def __init__(self, targets):
                super().__init__()
                self.register_buffer('input_mean', torch.zeros(61, dtype=torch.float64))
                self.register_buffer('targets', targets)
                self.tick = 0

            def forward(self, _state, _action):
                value = self.targets[self.tick:self.tick + 1]
                self.tick += 1
                return value

        class Centered(torch.nn.Module):
            def __init__(self, base, initial, targets, actions):
                super().__init__()
                self.base = base
                self.register_buffer('input_mean', base.input_mean.detach().clone())
                states = torch.cat((initial, targets[:-1]))
                with torch.no_grad():
                    corrections = [targets[t:t + 1] - base(states[t:t + 1], actions[t:t + 1]) for t in range(32)]
                self.register_buffer('correction', torch.cat(corrections).detach())
                self.tick = 0

            def forward(self, state, action):
                value = self.base(state, action) + self.correction[self.tick:self.tick + 1]
                self.tick += 1
                return value

        def run(model, start, theta):
            if hasattr(model, 'tick'):
                model.tick = 0
            return numeric.imaginary(model, actor, start, theta, return_trace=True)

        refs = []
        for i, start in enumerate(points):
            relative = f'execution_01/probe_{i:02d}.npz'
            path = ROOT / 'evidence/model_gradient53' / relative
            require(sha(path) == summary['raw_manifest'][relative]['sha256'], 'raw trace changed')
            with np.load(path, allow_pickle=False) as raw:
                targets = torch.tensor(raw['states_0'], dtype=torch.float64)
                actual_actions = torch.tensor(raw['observations_0'][:, 41:57], dtype=torch.float64)
            teacher = Teacher(targets)
            with torch.no_grad():
                teacher_cost, teacher_trace = run(teacher, start, torch.tensor(0., dtype=torch.float64))
            refs.append((targets, actual_actions, teacher_trace, float(teacher_cost)))

        for old in diagnostic['models']:
            seed = old['seed']
            relative = f'execution_01/model_{seed}/final_state_dict.pt'
            path = ROOT / 'evidence/model_gradient53' / relative
            require(sha(path) == summary['raw_manifest'][relative]['sha256'], 'weights changed')
            base = dynamics.Dynamics52()
            base.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            base = copy.deepcopy(base).double().eval().requires_grad_(False)
            rows = []
            for index, start in enumerate(points):
                targets, actual_actions, teacher_trace, teacher_cost = refs[index]
                commands = torch.tensor(start['commands'][:32], dtype=torch.float64)
                previous_real = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), actual_actions[:-1]))
                theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
                original_cost, original_trace = run(base, start, theta)
                original_gradient, = torch.autograd.grad(original_cost, theta, retain_graph=True)
                bound = next(p for p in old['points'] if p['role'] == 'probe' and (p['seed'], p['tick']) == (start['seed'], start['tick']))
                require(float(original_gradient) == bound['gradient64'], 'original double gradient changed')
                origin = {k: v.detach().clone() for k, v in original_trace.items()}

                def rebased_cost(trace, targets=targets, origin=origin, actual_actions=actual_actions,
                                 previous_real=previous_real, commands=commands,
                                 nominal_height=start['nominal_height']):
                    states = targets + (trace['states'] - origin['states'])
                    actions = actual_actions + (trace['actions'] - origin['actions'])
                    previous = previous_real + torch.cat((torch.zeros((1, 16), dtype=torch.float64),
                                                          trace['actions'][:-1] - origin['actions'][:-1]))
                    return numeric.cost(states, commands, actions, previous, nominal_height).mean()

                rebased = rebased_cost(original_trace)
                rebased_gradient, = torch.autograd.grad(rebased, theta)
                centered = Centered(base, torch.tensor(start['state'][None], dtype=torch.float64),
                                    targets, teacher_trace['applied_actions'].detach())
                center_theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
                centered_cost, centered_trace = run(centered, start, center_theta)
                centered_gradient, = torch.autograd.grad(centered_cost, center_theta)
                baseline_gaps = {k: float((centered_trace[k].detach() - teacher_trace[k]).abs().max())
                                 for k in ('states', 'actions', 'applied_actions', 'frames', 'filtered', 'shaper_state', 'actor_inputs')}
                require(max(baseline_gaps.values()) < 1e-10, 'oracle baseline does not equal teacher chain')
                checks, quat_deviation = [], 0.
                for epsilon in (.001, .005, .01):
                    with torch.no_grad():
                        _, plus_trace = run(base, start, torch.tensor(epsilon, dtype=torch.float64))
                        _, minus_trace = run(base, start, torch.tensor(-epsilon, dtype=torch.float64))
                        rebased_fd = float((rebased_cost(plus_trace) - rebased_cost(minus_trace)) / (2 * epsilon))
                        plus, cplus = run(centered, start, torch.tensor(epsilon, dtype=torch.float64))
                        minus, cminus = run(centered, start, torch.tensor(-epsilon, dtype=torch.float64))
                        centered_fd = float((plus - minus) / (2 * epsilon))
                        quat_deviation = max(quat_deviation, *(float((torch.linalg.vector_norm(t['states'][:, 3:7], dim=-1) - 1).abs().max()) for t in (cplus, cminus)))
                    checks.append({'epsilon': epsilon, 'cost_only_rebase_fd': rebased_fd,
                        'recentered_fd': centered_fd,
                        'cost_only_original_tolerance_passes': abs(float(rebased_gradient) - rebased_fd) <= .1 * abs(rebased_fd) + 1e-4,
                        'recentered_original_tolerance_passes': abs(float(centered_gradient) - centered_fd) <= .1 * abs(centered_fd) + 1e-4})
                true_gradient = diagnostic['physical_readback'][index]['float64_cost_gradient']
                rows.append({'seed': start['seed'], 'tick': start['tick'], 'original_gradient': float(original_gradient),
                    'cost_only_rebase_gradient': float(rebased_gradient), 'recentered_gradient': float(centered_gradient),
                    'true_gradient': true_gradient, 'checks': checks, 'teacher_baseline_max_gaps': baseline_gaps,
                    'teacher64_vs_physical32_action_max_gap': float((teacher_trace['actions'] - actual_actions).abs().max()),
                    'recentered_max_quaternion_norm_deviation': quat_deviation,
                    'original_cost': float(original_cost.detach()), 'recentered_cost': float(centered_cost.detach()),
                    'teacher_cost': teacher_cost})
            row = {'seed': seed, 'points': rows, 'sign_matches': {key: sum(p[key] * p['true_gradient'] > 0 for p in rows)
                for key in ('original_gradient', 'cost_only_rebase_gradient', 'recentered_gradient')}}
            report['models'].append(row)
            write_json(args.output / f'model_{seed}.json', row)
        verify(plan)
        for path, digest in own_plan['source_sha256'].items():
            require(sha(ROOT / path) == digest, 'source changed: ' + path)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve every incomplete diagnostic.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['cpu_seconds'] = time.process_time() - started
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'cpu_seconds', 'new_physics_steps')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
