"""55C: fixed small-epsilon physical FD diagnostics, without rejudging55.

All nine points and four preregistered epsilons are retained. The original
float32 actor/action path and simulator are unchanged; only cost readback uses
float64. This entry point neither trains a model nor selects a control update.
"""
from __future__ import annotations

import argparse
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
from tools53.run_pilot53 import require, sha, write_json
from tools55.run55 import audit_work, collect_probes, verify55

EPSILONS = (.00003125, .0000625, .000125, .00025)
SEEDS = (55301, 55302, 55303)
TICKS = (110, 260, 410)
PREVIOUS = Path('evidence/observed_gradient55/execution_01')
PRIOR_PLAN = Path('docs/experiments55/observed55_preregistered.json')
HARD_CAP = 5000
EXPECTED_STEPS = 4680


def verify_local(plan):
    """Bind the fixed diagnostic and every original array read during replay."""
    require(plan['schema'] == 'd1-physical-fd55c-diagnostic-v1', 'wrong55C protocol')
    require(plan['probe_seeds'] == list(SEEDS) and plan['eps'] == list(EPSILONS), 'grid changed')
    require(plan['expected_policy_steps'] == EXPECTED_STEPS and plan['hard_policy_cap'] == HARD_CAP,
            'physical budget changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420, 'time budget changed')
    require(plan['does_not_rejudge55'] is True, 'diagnostic must not rejudge55')
    required = {str(PRIOR_PLAN), 'tools55/probe_local55.py', 'tools55/run55.py',
                str(PREVIOUS / 'report.json')}
    required.update(str(PREVIOUS / f'collection_{seed}.npz') for seed in SEEDS)
    required.update(str(PREVIOUS / f'probe_{index:02d}.npz') for index in range(9))
    require(required <= set(plan['source_sha256']), 'original inputs/source not SHA-bound')
    for path, digest in plan['source_sha256'].items():
        require(sha(ROOT / path) == digest, 'input/source changed: ' + path)


def cost64(start, states, observations):
    """Score returned post-step states and actual actions using frozen54 cost."""
    import numpy as np
    import torch

    from tools54 import numerics54 as numeric
    state = torch.tensor(np.asarray(states), dtype=torch.float64)
    action = torch.tensor(np.asarray(observations)[:, 41:57], dtype=torch.float64)
    previous = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), action[:-1]))
    command = torch.tensor(start['commands'][:len(state)], dtype=torch.float64)
    with torch.no_grad():
        costs = numeric.cost(state, command, action, previous, start['nominal_height'])
    require(bool(torch.isfinite(costs).all()), 'nonfinite double readback')
    return float(costs.mean()), costs.tolist()


def action_delta_ulps(reference, perturbed):
    """Compare actual first-step actions; ULP is spacing at the reference value.

    Both inputs are float32 action16 recovered from returned observation41:57.
    Differences are computed after promotion to float64, so no subtraction or
    ratio is rounded back to float32. Four wheel indices are3,7,11,15.
    """
    import numpy as np
    reference = np.asarray(reference, dtype=np.float32)
    perturbed = np.asarray(perturbed, dtype=np.float32)
    require(reference.shape == perturbed.shape == (16,), 'action shape differs')
    require(np.isfinite(reference).all() and np.isfinite(perturbed).all(), 'nonfinite action')
    delta = perturbed.astype(np.float64) - reference.astype(np.float64)
    ulp = np.spacing(np.abs(reference)).astype(np.float64)
    ratios = np.abs(delta) / ulp
    nonzero = delta != 0
    wheels = np.array([3, 7, 11, 15])
    return {'reference': reference.tolist(), 'perturbed': perturbed.tolist(),
            'delta': delta.tolist(), 'reference_float32_ulp': ulp.tolist(), 'delta_ulps': ratios.tolist(),
            'nonzero_count': int(nonzero.sum()), 'wheel_nonzero_count': int(nonzero[wheels].sum()),
            'all_four_wheels_nonzero': bool(nonzero[wheels].all()),
            'min_nonzero_delta_ulps': float(ratios[nonzero].min()) if nonzero.any() else None,
            'maxabs': float(np.abs(delta).max())}


def local_stability(gradients):
    """Apply the whole-grid55C diagnostic rule without choosing an epsilon."""
    require([row['epsilon'] for row in gradients] == list(EPSILONS), 'incomplete epsilon grid')
    central = [row['central64'] for row in gradients]
    scale = max(abs(value) for value in central)
    spread = (max(central) - min(central)) / scale if scale else None
    same_sign = all(value > 0 for value in central) or all(value < 0 for value in central)
    one_sided = []
    for row in gradients:
        forward, backward = row['forward64'], row['backward64']
        magnitude = max(abs(forward), abs(backward))
        gap = abs(forward - backward) / magnitude if magnitude else None
        one_sided.append({'epsilon': row['epsilon'], 'same_sign': forward * backward > 0,
                          'relative_gap': gap,
                          'stable': forward * backward > 0 and gap is not None and gap <= .25})
    above_floor = all(abs(value) >= 1e-5 for value in central)
    return {'central_same_sign': same_sign, 'all_central_abs_ge_1e_5': above_floor,
            'central_whole_grid_relative_spread': spread, 'one_sided': one_sided,
            'locally_stable': same_sign and above_floor and spread is not None and spread <= .25
            and all(row['stable'] for row in one_sided), 'rejudges55': False, 'selected_epsilon': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-physical-fd55c-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'original55_decision_changed': False,
              'eps': list(EPSILONS), 'counts': None, 'collection': {'status': 'NOT_RUN'},
              'baseline_parity': [], 'points': [
                  {'seed': seed, 'tick': tick, 'status': 'NOT_RUN', 'gradients': [],
                   'branches': [{'theta': theta, 'status': 'NOT_RUN'} for theta in sequence]}
                  for seed in SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, prior = None, None, None, None
    phase = {'name': 'admission', 'point': None, 'branch': None, 'theta': None, 'buffer': None}

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        # Reserve the last CPU second for a durable partial report on SIGXCPU.
        resource.setrlimit(resource.RLIMIT_CPU, (299, 300))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'thread/GPU limits differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify_local(plan)
        prior = json.loads((ROOT / PRIOR_PLAN).read_text())
        original = verify55(prior)
        report['versions'] = {key: importlib.metadata.version(key) for key in original['versions']}
        require(report['versions'] == original['versions'], 'versions differ')
        import numpy as np
        import torch

        from tools33.reference_shape33 import ReferenceShaper
        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        old = json.loads((ROOT / PREVIOUS / 'report.json').read_text())
        require(old['status'] == 'COMPLETE' and len(old['physical_probes']) == 9,
                'complete original55 evidence required')
        require(all(row['branches'][0]['status'] == 'COMPLETE' for row in old['physical_probes']),
                'original zero branches incomplete')
        actor = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor)
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)

        def record(current, action, result):
            obs, reward, terminated, truncated, info = result
            state = control.physical(current)
            row = {'phase': phase['name'], 'point': phase['point'], 'branch': phase['branch'],
                   'theta': phase['theta'], 'counts': current.counters(), 'episode_tick': current._episode_steps,
                   'state45': state.tolist(), 'action': np.asarray(action).tolist(),
                   'applied_action': current._applied_action.tolist(),
                   'command_during_interval': np.asarray(info['command_during_interval']).tolist(),
                   'reward': float(reward), 'terminated': bool(terminated), 'truncated': bool(truncated),
                   'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact']),
                   'base_height_m': float(info['base_height_m']), 'tilt_rad': float(info['tilt_rad']),
                   'contacts': [[int(c.geom1), int(c.geom2), float(c.dist)]
                                for c in current.data.contact[:current.data.ncon]],
                   'constraint_count': int(current.data.nefc)}
            solver_niter = getattr(current.data, 'solver_niter', None)
            if solver_niter is not None:
                row['solver_niter'] = np.asarray(solver_niter).tolist()
            if phase['buffer'] is not None:
                phase['buffer'].append((state, np.asarray(obs).copy(), float(reward), row))
            audit.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            audit.flush()

        env = collection.make_env(HARD_CAP, audit_step=record)
        report['solver_options'] = {key: int(getattr(env.model.opt, key))
                                    for key in ('solver', 'iterations')}
        report['solver_options'].update({key: float(getattr(env.model.opt, key))
                                         for key in ('tolerance', 'impratio')})
        phase['name'] = 'baseline_replay'
        report['collection'] = {'status': 'INCOMPLETE'}
        points, baselines, report['collection'] = collect_probes(env, actor, args.output, prior, original)
        require([(point['seed'], point['tick']) for point in points] ==
                [(seed, tick) for seed in SEEDS for tick in TICKS], 'capture points changed')
        for seed in SEEDS:
            name = f'collection_{seed}.npz'
            with np.load(args.output / name, allow_pickle=False) as fresh, \
                    np.load(ROOT / PREVIOUS / name, allow_pickle=False) as previous:
                same_keys = set(fresh.files) == set(previous.files)
                exact = {key: bool(np.array_equal(fresh[key], previous[key]))
                         for key in fresh.files if key in previous.files}
                physical = {key: value for key, value in exact.items() if key not in collection.COUNTERS}
            row = {'seed': seed, 'same_keys': same_keys, 'exact_by_field': exact,
                   'all_physical_fields_exact': bool(same_keys and all(physical.values()))}
            report['baseline_parity'].append(row)
        write_json(args.output / 'baseline_parity.json', report['baseline_parity'])
        require(all(row['all_physical_fields_exact'] for row in report['baseline_parity']),
                'fresh full baseline did not reproduce55 exactly')
        phase['name'] = 'fd_grid'
        for index, point in enumerate(points):
            phase['point'] = index
            row = report['points'][index]
            row['status'] = 'INCOMPLETE'
            first_zero = None
            for order, theta in enumerate(sequence):
                phase.update(branch=order, theta=theta, buffer=[])
                stem = args.output / f'grid_{index:02d}_{order:02d}'
                before = env.counters()
                branch = {'status': 'INCOMPLETE', 'theta': theta, 'steps': 0}
                arrays = None
                try:
                    branch, states, observations, rewards = control.real_branch(
                        env, ReferenceShaper(), actor, point, theta)
                    arrays = {'states': np.asarray(states), 'observations': np.asarray(observations),
                              'rewards': np.asarray(rewards)}
                    collection.save_npz(stem.with_suffix('.npz'), arrays)
                    branch.update(counts_before=before, counts_after=env.counters(),
                                  actual_float32_theta=float(np.float32(theta)))
                    branch['unsafe_steps'] = [entry[3] for entry in phase['buffer']
                                             if any(entry[3][key] for key in
                                                    ('terminated', 'truncated', 'fallen', 'illegal_body_contact'))]
                    if branch['status'] == 'COMPLETE':
                        value, costs = cost64(point, states, observations)
                        branch.update(cost64=value, per_step_cost64=costs)
                        if order == 0:
                            first_zero = arrays
                            with np.load(ROOT / PREVIOUS / f'probe_{index:02d}.npz', allow_pickle=False) as saved:
                                branch['original_zero_exact_by_field'] = {
                                    key: bool(np.array_equal(value, saved[key + '_0'])) for key, value in arrays.items()}
                            branch['collected_baseline_exact'] = bool(np.array_equal(states, baselines[index]))
                        elif order == 1:
                            branch['repeated_zero_exact'] = all(np.array_equal(value, first_zero[key])
                                                               for key, value in arrays.items())
                        else:
                            branch['first_step_action_delta'] = action_delta_ulps(
                                first_zero['observations'][0, 41:57], arrays['observations'][0, 41:57])
                except BaseException as exc:
                    branch.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}',
                                  counts_before=before, counts_after=env.counters())
                    if arrays is None:
                        branch['steps'] = len(phase['buffer'])
                        collection.save_npz(stem.with_suffix('.npz'), {
                            'states': np.asarray([entry[0] for entry in phase['buffer']]).reshape(-1, 45),
                            'observations': np.asarray([entry[1] for entry in phase['buffer']]).reshape(-1, 627),
                            'rewards': np.asarray([entry[2] for entry in phase['buffer']])})
                    raise
                finally:
                    row['branches'][order] = branch
                    write_json(stem.with_suffix('.json'), branch)
                    phase['buffer'] = None
                require(branch['status'] == 'COMPLETE' and not branch['unsafe_steps'], 'grid branch incomplete/unsafe')
                if order == 0:
                    require(branch['collected_baseline_exact'] and all(branch['original_zero_exact_by_field'].values()),
                            'new zero branch differs from original/collected baseline')
                elif order == 1:
                    require(branch['repeated_zero_exact'], 'repeated zero branch differs')
            for j, epsilon in enumerate(EPSILONS):
                plus, minus = row['branches'][2 + 2 * j:4 + 2 * j]
                zero = row['branches'][0]['cost64']
                row['gradients'].append({'epsilon': epsilon,
                    'central64': (plus['cost64'] - minus['cost64']) / (2 * epsilon),
                    'central32': (plus['cost'] - minus['cost']) / (2 * epsilon),
                    'forward64': (plus['cost64'] - zero) / epsilon,
                    'backward64': (zero - minus['cost64']) / epsilon,
                    'first_step_action_delta': {'plus_vs_zero': plus['first_step_action_delta'],
                                                'minus_vs_zero': minus['first_step_action_delta']}})
            row['all_four_wheels_resolved'] = all(
                branch['first_step_action_delta']['all_four_wheels_nonzero'] for branch in row['branches'][2:])
            row['local_reference_diagnostic'] = local_stability(row['gradients'])
            row['status'] = 'COMPLETE'
            write_json(args.output / f'point_{index:02d}.json', row)
        report['all_four_wheels_resolved'] = all(row['all_four_wheels_resolved'] for row in report['points'])
        report['locally_stable_points'] = sum(row['local_reference_diagnostic']['locally_stable']
                                             for row in report['points'])
        require(env.counters()['policy_transitions'] == EXPECTED_STEPS, 'work count differs')
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - partial is retained, never resumed.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        errors = []
        report['last_phase'] = {key: value for key, value in phase.items() if key != 'buffer'}
        if (args.output / 'collection.json').exists():
            report['collection'] = json.loads((args.output / 'collection.json').read_text())
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
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'],
                        'work audit mismatch')
                if report['status'] == 'COMPLETE':
                    require(report['work_journal']['by_phase'] == {'baseline_replay': 1800, 'fd_grid': 2880},
                            'phase work differs')
            except Exception as exc:  # noqa: BLE001 - preserve a partial journal and counts.
                errors.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify_local(plan)
            if prior is not None:
                verify55(prior)
            report['verification_after'] = 'COMPLETE' if prior is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - original failure is retained too.
            report['verification_after'] = 'FAILED'
            errors.append(f'verification: {exc}')
        if errors:
            report.update(status='INCOMPLETE', cleanup_errors=errors)
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if report['artifact_bytes_before_report'] > (128 << 20) - (2 << 20):
            report.update(status='INCOMPLETE', artifact_limit_exceeded=True)
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'counts', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
