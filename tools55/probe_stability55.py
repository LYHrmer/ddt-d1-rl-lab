"""Independent physical FD convergence audit; never resumes or rejudges55."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pickle
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import require, sha, write_json
from tools55.run55 import audit_work, collect_probes, verify55

EPSILONS = (.00025, .0005, .001, .002, .005, .01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'failure': None, 'model_updates': 0, 'policy_updates': 0,
              'readback': [], 'points': [], 'counts': None, 'original55_decision_changed': False}
    write_json(args.output / 'initial_incomplete.json', report)
    started = time.process_time()
    env, audit = None, None
    phase = {'name': 'readback', 'point': None, 'theta': None}

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        plan = json.loads(args.plan.read_text())
        for path, digest in plan['source_sha256'].items():
            require(sha(ROOT / path) == digest, 'input/source changed: ' + path)
        report['plan_sha256'] = sha(args.plan)
        prior = json.loads((ROOT / 'docs/experiments55/observed55_preregistered.json').read_text())
        original = verify55(prior)
        require({k: importlib.metadata.version(k) for k in original['versions']} == original['versions'], 'versions differ')
        import numpy as np
        import torch

        from tools33.reference_shape33 import ReferenceShaper
        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        from tools54 import numerics54 as numeric
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        previous = ROOT / 'evidence/observed_gradient55/execution_01'
        old = json.loads((previous / 'report.json').read_text())
        require(old['status'] == 'COMPLETE', 'completed55 evidence required')
        # Exact local path and SHA are bound before loading this own-generated pickle.
        with (previous / 'probe_snapshots.pkl').open('rb') as stream:
            saved_points = pickle.load(stream)

        def cost64(start, states, observations):
            state = torch.tensor(np.asarray(states), dtype=torch.float64)
            actions = torch.tensor(np.asarray(observations)[:, 41:57], dtype=torch.float64)
            previous_actions = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), actions[:-1]))
            command = torch.tensor(start['commands'][:len(state)], dtype=torch.float64)
            with torch.no_grad():
                costs = numeric.cost(state, command, actions, previous_actions, start['nominal_height'])
            return float(costs.mean()), costs.tolist()

        def difference_record(plus, minus, epsilon):
            return (plus - minus) / (2 * epsilon)

        for index, start in enumerate(saved_points):
            with np.load(previous / f'probe_{index:02d}.npz', allow_pickle=False) as raw:
                costs = [cost64(start, raw[f'states_{i}'], raw[f'observations_{i}'])[0] for i in range(6)]
            small, large = difference_record(costs[2], costs[3], .005), difference_record(costs[4], costs[5], .01)
            identifiable = (abs(small) >= 1e-5 and abs(large) >= 1e-5 and small * large > 0
                            and abs(small - large) <= .25 * max(abs(small), abs(large)))
            report['readback'].append({'seed': start['seed'], 'tick': start['tick'],
                'float32_gradients': [old['physical_probes'][index]['gradient'], old['physical_probes'][index]['gradient_double_epsilon']],
                'float64_gradients': [small, large], 'identifiable64': identifiable})
        write_json(args.output / 'readback_before_new_physics.json', report['readback'])
        if all(p['identifiable64'] for p in report['readback']):
            report['grid_status'] = 'NOT_RUN: all references stable after double scoring'
        else:
            actor = adapter.load_actor(ROOT / original['checkpoint'])
            report['actor_admission'] = shared.actor_admission(actor)
            audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)

            def record(current, action, result):
                _obs, reward, terminated, truncated, info = result
                contacts = [[int(c.geom1), int(c.geom2), float(c.dist)] for c in current.data.contact[:current.data.ncon]]
                row = {'phase': phase['name'], 'point': phase['point'], 'theta': phase['theta'],
                    'counts': current.counters(), 'episode_tick': current._episode_steps,
                    'state45': control.physical(current).tolist(), 'action': np.asarray(action).tolist(),
                    'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated),
                    'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact']),
                    'contacts': contacts, 'constraint_count': int(current.data.nefc)}
                audit.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')

            env = collection.make_env(6000, audit_step=record)
            phase['name'] = 'baseline_replay'
            points, baselines, report['collection'] = collect_probes(env, actor, args.output, prior, original)
            for index, point in enumerate(points):
                with np.load(previous / f'probe_{index:02d}.npz', allow_pickle=False) as old_raw:
                    require(np.array_equal(baselines[index], old_raw['states_0']), 'fresh baseline did not reproduce55 exactly')
            phase['name'] = 'fd_grid'
            for index, point in enumerate(points):
                phase['point'] = index
                row = {'seed': point['seed'], 'tick': point['tick'], 'status': 'INCOMPLETE', 'branches': [], 'gradients': []}
                report['points'].append(row)
                sequence = [0., 0., *(sign * eps for eps in EPSILONS for sign in (1, -1))]
                first_zero = None
                for order, theta in enumerate(sequence):
                    phase['theta'] = theta
                    branch, states, observations, rewards = control.real_branch(env, ReferenceShaper(), actor, point, theta)
                    collection.save_npz(args.output / f'grid_{index:02d}_{order:02d}.npz',
                        {'states': states, 'observations': observations, 'rewards': rewards})
                    if branch['status'] != 'COMPLETE':
                        row['branches'].append(branch)
                        write_json(args.output / f'grid_{index:02d}_{order:02d}.json', branch)
                    require(branch['status'] == 'COMPLETE', 'grid branch incomplete')
                    value, costs = cost64(point, states, observations)
                    row['branches'].append({**branch, 'cost64': value, 'per_step_cost64': costs,
                        'actual_float32_theta': float(torch.tensor(theta, dtype=torch.float32))})
                    if order == 0:
                        first_zero = (np.asarray(states), np.asarray(observations), np.asarray(rewards))
                        require(np.array_equal(states, baselines[index]), 'new zero branch differs from collected baseline')
                    elif order == 1:
                        require(all(np.array_equal(a, b) for a, b in zip(first_zero, (states, observations, rewards), strict=True)),
                                'repeated zero branch differs')
                    write_json(args.output / f'grid_{index:02d}_{order:02d}.json', row['branches'][-1])
                for j, epsilon in enumerate(EPSILONS):
                    plus, minus = row['branches'][2 + 2 * j:4 + 2 * j]
                    zero = row['branches'][0]['cost64']
                    row['gradients'].append({'epsilon': epsilon,
                        'central64': difference_record(plus['cost64'], minus['cost64'], epsilon),
                        'central32': difference_record(plus['cost'], minus['cost'], epsilon),
                        'forward64': (plus['cost64'] - zero) / epsilon,
                        'backward64': (zero - minus['cost64']) / epsilon})
                row['status'] = 'COMPLETE'
                write_json(args.output / f'point_{index:02d}.json', row)
            report['grid_status'] = 'COMPLETE'
            require(env.counters()['policy_transitions'] == 5832, 'grid work count differs')
        for path, digest in plan['source_sha256'].items():
            require(sha(ROOT / path) == digest, 'input/source changed: ' + path)
        verify55(prior)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - never overwrite or resume a failed audit.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        cleanup = []
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - retain cleanup failure.
                cleanup.append(f'env: {exc}')
        if audit is not None:
            try:
                audit.close()
                report['work_journal'] = audit_work(args.output / 'all_physics_steps.jsonl')
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'], 'work audit mismatch')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                cleanup.append(f'audit: {exc}')
        if cleanup:
            report.update(status='INCOMPLETE', cleanup_errors=cleanup)
        report['cpu_seconds'] = time.process_time() - started
        report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if report['artifact_bytes_before_report'] > (128 << 20) - (2 << 20):
            report.update(status='INCOMPLETE', failure='artifact reserve exceeded')
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'counts', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
