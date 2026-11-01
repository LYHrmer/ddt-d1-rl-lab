"""Replay the four observed53 failures, without physics, fitting or rejudging53."""
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
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import require, sha, verify, write_json


def load_points(role, manifest):
    path = ROOT / f'evidence/model_gradient53/execution_01/{role}_snapshots.pkl'
    require(sha(path) == manifest[f'execution_01/{role}_snapshots.pkl']['sha256'], 'snapshot identity differs')
    # Only the hash-bound, locally generated experiment53 pickle is loaded.
    with path.open('rb') as stream:
        values = pickle.load(stream)
    return [p | {'shaper': SimpleNamespace(**p['python_state']['shaper'])} for p in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assert-resolved', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'failure': None, 'new_physics_steps': 0,
              'model_updates': 0, 'policy_updates': 0, 'cases': []}
    write_json(args.output / 'initial_incomplete.json', report)
    started = time.process_time()
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        plan = json.loads((ROOT / 'docs/experiments53/model_gradient53_preregistered.json').read_text())
        verify(plan)
        summary = json.loads((ROOT / 'reports/model_gradient53_results_20261101.json').read_text())
        manifest, formal = summary['raw_manifest'], summary['formal']
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        actor = adapter.load_actor(ROOT / plan['checkpoint'])
        points = {role: load_points(role, manifest) for role in ('development', 'probe')}
        cases = ((53002, 'development', 53202, 110), (53002, 'probe', 53301, 110),
                 (53002, 'probe', 53303, 110), (53003, 'probe', 53303, 110))
        for model_seed, role, reset, tick in cases:
            path = ROOT / f'evidence/model_gradient53/execution_01/model_{model_seed}/final_state_dict.pt'
            require(sha(path) == manifest[f'execution_01/model_{model_seed}/final_state_dict.pt']['sha256'],
                    'weight identity differs')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval().requires_grad_(False)
            start = next(p for p in points[role] if (p['seed'], p['tick']) == (reset, tick))
            result, = control.gradients(model, actor, [start])
            old_model = next(m for m in formal['seeds'] if m['seed'] == model_seed)
            field = 'development_gradients' if role == 'development' else 'test_gradients'
            old = next(p for p in old_model[field] if (p['seed'], p['tick']) == (reset, tick))
            require(result == old, 'original failure did not exactly reproduce')
            actual = next((p for p in formal['physical_probes'] if (p['seed'], p['tick']) == (reset, tick)), None)
            same_sign = None if actual is None else result['gradient'] * actual['gradient'] > 0
            report['cases'].append({'model_seed': model_seed, 'role': role, **result,
                'true_gradient': None if actual is None else actual['gradient'],
                'true_sign_agrees': same_sign, 'reproduces_original_exactly': True})
        report['resolved'] = all(c['autograd_agrees'] and c['true_sign_agrees'] is not False
                                 for c in report['cases'])
        report['status'] = 'COMPLETE'
        verify(plan)
    except BaseException as exc:  # noqa: BLE001 - preserve partial diagnostic evidence.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['cpu_seconds'] = time.process_time() - started
        write_json(args.output / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE' or (args.assert_resolved and not report['resolved']))


if __name__ == '__main__':
    raise SystemExit(main())
