"""Post-hoc epsilon sensitivity of saved models; never changes the53 decision.

No environment is constructed, no optimizer is created, and no policy is
selected.  The fixed epsilon grid includes the original numerical self-check.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from tools53.run_pilot53 import require, sha, verify, write_json

EPSILONS = (.0005, .001, .002, .005, .01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'post_hoc': True, 'new_physics_steps': 0,
              'model_updates': 0, 'policy_updates': 0, 'original_decision_changed': False,
              'epsilon_grid': EPSILONS, 'models': [], 'failure': None,
              'completed_rollouts': 0, 'completed_autograd_calls': 0}
    write_json(args.output / 'initial_incomplete.json', report)
    started, cpu = time.monotonic(), time.process_time()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        plan = json.loads(args.plan.read_text())
        verify(plan)
        original = json.loads((args.input / 'report.json').read_text())
        require(original['status'] == 'COMPLETE' and original['plan_sha256'] == sha(args.plan),
                'completed bound result required')
        report['original_report_sha256'] = sha(args.input / 'report.json')
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        actor = adapter.load_actor(ROOT / plan['checkpoint'])
        points = {}
        for role in ('development', 'probe'):
            path = args.input / f'{role}_snapshots.pkl'
            report[f'{role}_snapshot_sha256'] = sha(path)
            # Only locally generated53 evidence; never accept arbitrary pickle input.
            require(path.resolve().parent == (ROOT / 'evidence/model_gradient53/execution_01').resolve(),
                    'only this local sealed execution is supported')
            with path.open('rb') as stream:
                saved = pickle.load(stream)
            points[role] = [p | {'shaper': SimpleNamespace(**p['python_state']['shaper'])} for p in saved]
        for seed_result in original['seeds']:
            seed = seed_result['seed']
            path = args.input / f'model_{seed}/final_state_dict.pt'
            identity = sha(path)
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval().requires_grad_(False)
            rows = []
            for role, starts in points.items():
                recorded = seed_result['development_gradients' if role == 'development' else 'test_gradients']
                for start, old in zip(starts, recorded, strict=True):
                    require((start['seed'], start['tick']) == (old['seed'], old['tick']), 'point identity differs')
                    theta = torch.tensor(0., requires_grad=True)
                    value = control.imaginary(model, actor, start, theta)
                    report['completed_rollouts'] += 1
                    derivative, = torch.autograd.grad(value, theta)
                    report['completed_autograd_calls'] += 1
                    analytic = float(derivative)
                    checks = []
                    for epsilon in EPSILONS:
                        with torch.no_grad():
                            plus = control.imaginary(model, actor, start, torch.tensor(epsilon))
                            report['completed_rollouts'] += 1
                            minus = control.imaginary(model, actor, start, torch.tensor(-epsilon))
                            report['completed_rollouts'] += 1
                            numerical = float((plus - minus) / (2 * epsilon))
                        checks.append({'epsilon': epsilon, 'plus_cost': float(plus), 'minus_cost': float(minus),
                            'central_difference': numerical, 'absolute_gap': abs(analytic - numerical),
                            'original_tolerance_formula_passes': abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4})
                    central = next(c for c in checks if c['epsilon'] == .001)
                    require(analytic == old['gradient'] and central['central_difference'] == old['model_central_difference'],
                            'saved-model rerender did not exactly reproduce original derivatives')
                    rows.append({'role': role, 'seed': start['seed'], 'tick': start['tick'],
                                 'analytic_gradient': analytic, 'original_autograd_agrees': old['autograd_agrees'],
                                 'checks': checks})
            require(sha(path) == identity, 'saved model changed during diagnostic')
            item = {'seed': seed, 'weights_sha256': identity, 'points': rows}
            report['models'].append(item)
            write_json(args.output / f'model_{seed}.json', item)
        require(sha(args.input / 'report.json') == report['original_report_sha256'], 'original report changed')
        verify(plan)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - retain the diagnostic as incomplete, never retry it.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['wall_seconds'], report['cpu_seconds'] = time.monotonic() - started, time.process_time() - cpu
        report['completed_model_forward_steps'] = report['completed_rollouts'] * 32
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'new_physics_steps',
                         'model_updates', 'completed_rollouts', 'cpu_seconds')}), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
