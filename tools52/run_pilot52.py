"""One bounded D1 model-gradient experiment, with a separate admission run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np
import torch

from tools52 import adapter52 as adapter
from tools52 import control52 as control
from tools52 import dynamics52 as dynamics

require, write_json = control.require, control.write_json


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value.hexdigest()


def verify(plan):
    require(len(set(plan['model_seeds'])) == len(plan['model_seeds']) == 3,
            'exactly three distinct model initializations required')
    split = plan['data_seeds']
    require({k: len(v) for k, v in split.items()} == {'train': 12, 'development': 3, 'probe': 3}
            and len(plan['evaluation_seeds']) == 3, 'registered split sizes differ')
    seeds = [s for k in ('train', 'development', 'probe') for s in split[k]] + plan['evaluation_seeds']
    require(len(set(seeds)) == len(seeds), 'data/evaluation reset seeds overlap or repeat')
    for name, identity in plan['identities'].items():
        path = ROOT / name
        require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
                'sealed identity changed: ' + name)


def save_npz(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def make_env(cap):
    from d1_rl.factory03 import environment_class
    cls = environment_class(ROOT / 'training/configs/pilot03d.json')
    env = cls(config=ROOT / 'training/configs/pilot03d.json', arm='B', seed=52001,
              rank=0, max_transitions=cap, eval_delay_ticks=0,
              eval_command_sequence=control.sequence(0))
    require(env.model.nq == 23 and env.model.nv == 22 and env.free_qpos == 0,
            'unexpected physical dimensions')
    require(np.array_equal(env.qpos_ids, np.arange(7, 23))
            and np.array_equal(env.dof_ids, np.arange(6, 22)), 'joint mapping differs')
    return env


def actor_admission(actor):
    rows = json.loads((ROOT / 'evidence/low_memory_delay25/execution_01/random/export/golden_fixtures.json').read_text())['fixtures']
    maximum = 0.
    with torch.no_grad():
        for row in rows:
            obs = np.r_[row['obs'], np.asarray(row['history_oldest_to_newest_previous_only']).ravel()]
            actual = actor(control.tensor(obs[None]))[0].numpy()
            maximum = max(maximum, float(np.max(np.abs(actual - row['python_ort_actions']))))
    require(len(rows) == 64 and maximum <= 1e-5, 'complete actor parity failed')
    return {'fixtures': len(rows), 'max_abs_difference': maximum, 'limit': 1e-5,
            'all_actor_weights_frozen': all(not p.requires_grad for p in actor.parameters())}


def preflight(env, actor, output):
    from tools33.reference_shape33 import ReferenceShaper
    spec = control.sequence(0)
    env.eval_command_sequence = env._validate_eval_sequence(spec)
    env.reset(seed=52001)
    shaper = ReferenceShaper()
    observation_error = sensor_error = reference_error = 0.
    for _ in range(64):
        raw = env.current_observation()
        with torch.no_grad():
            predicted, _ = adapter.reference_step(control.tensor(raw[:57][None]),
                control.tensor(raw[57:].reshape(1, 10, 57)), control.cache_tensor(shaper))
        import copy
        oracle_shaper = copy.deepcopy(shaper)
        reference_error = max(reference_error, float(np.max(np.abs(
            predicted[0].numpy() - oracle_shaper(raw)))))
        previous_filter = env._filtered_gyro.copy()
        action = control.real_action(env, shaper, actor, 0.)
        obs, _, terminated, truncated, _info = env.step(action)
        require(not terminated and not truncated, 'admission trajectory terminated')
        s = control.physical(env)
        with torch.no_grad():
            frame, filtered = adapter.build_observation(control.tensor(s[None], double=True),
                control.tensor(env._command[None], double=True), control.tensor(action[None]),
                control.tensor(previous_filter[None], double=True))
        observation_error = max(observation_error, float(np.max(np.abs(frame[0].numpy() - obs[:57]))),
                                float(np.max(np.abs(filtered[0].numpy() - env._filtered_gyro))))
        sensor_error = max(sensor_error, float(np.max(np.abs(s[26:29] - env._sensor('trunk_gyro')))),
                           float(np.max(np.abs(s[3:7] - env._sensor('trunk_quat')))),
                           float(np.max(np.abs(s[23:26] - env._sensor('frame_vel')))))
    write_json(output / 'mapping_errors.json', {'observation': observation_error,
               'sensors': sensor_error, 'reference': reference_error})
    require(observation_error <= 1e-6 and sensor_error <= 1e-8 and reference_error <= 1e-7,
            'state/observation/reference mapping failed')
    start = control.point(env, shaper, spec, 52001)
    before = env.counters()['policy_transitions']
    first = control.real_branch(env, shaper, actor, start, 0.)
    second = control.real_branch(env, shaper, actor, start, 0.)
    exact = first[0] == second[0] and all(np.array_equal(a, b) for a, b in zip(first[1:], second[1:], strict=True))
    require(exact and first[0]['status'] == 'COMPLETE', 'counterfactual restore is not exact')
    require(env.counters()['policy_transitions'] - before == 64, 'restore refunded work')
    save_npz(output / 'restore_parity.npz', {'first_states': first[1], 'second_states': second[1],
                                          'first_observations': first[2], 'second_observations': second[2]})
    return {'observation_max_abs_difference': observation_error, 'sensor_max_abs_difference': sensor_error,
            'reference_max_abs_difference': reference_error, 'same_snapshot_32_step_exact': exact,
            'branch_work_not_refunded': True,
            'mjdata_narena_bytes': env.data.narena, 'mjdata_nbuffer_bytes': env.data.nbuffer,
            'formal_snapshot_count': 18}


def save_points(path, points, env):
    import mujoco
    stored = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    for p in points:
        state = np.empty(mujoco.mj_stateSize(env.model, spec))
        mujoco.mj_getState(env.model, p['snapshot'].data, state, spec)
        stored.append({k: v for k, v in p.items() if k not in ('snapshot', 'shaper')} | {
            'python_state': p['snapshot'].python_state, 'integration_state': state,
            'integration_spec': int(spec)})
    with path.open('xb') as stream:
        pickle.dump(stored, stream, protocol=5)


def collect(env, actor, output, plan):
    from tools33.reference_shape33 import ReferenceShaper
    datasets, points, episodes = {}, {}, []
    for role in ('train', 'development', 'probe'):
        rows = {k: [] for k in ('s', 'a', 'target', 'episode', 'tick', 'done')}
        points[role] = []
        for index, seed in enumerate(plan['data_seeds'][role]):
            spec = control.sequence(index)
            env.eval_command_sequence = env._validate_eval_sequence(spec)
            env.reset(seed=seed)
            shaper = ReferenceShaper()
            noise_rng = np.random.default_rng(seed + 900000)
            start_count = env.counters()['policy_transitions']
            for tick in range(400):
                if role != 'train' and tick in control.POINT_TICKS:
                    points[role].append(control.point(env, shaper, spec, seed))
                s = control.physical(env)
                noise = noise_rng.normal(0., .02, 16) if role == 'train' else None
                issued = control.real_action(env, shaper, actor, 0., noise)
                _, _, terminated, truncated, _ = env.step(issued)
                for key, value in {'s': s, 'a': env._applied_action.copy(),
                    'target': control.physical(env), 'episode': index, 'tick': tick,
                    'done': bool(terminated or truncated or tick == 399)}.items():
                    rows[key].append(value)
                if terminated or truncated:
                    break
            n = env.counters()['policy_transitions'] - start_count
            episodes.append({'role': role, 'seed': seed, 'steps': n,
                             'terminated': bool(terminated), 'truncated': bool(truncated),
                             'complete_8s_trial': n == 400 and not terminated and not truncated})
            write_json(output / f'collection_{role}_{seed}.json', episodes[-1])
            save_npz(output / f'collection_{role}_{seed}.npz',
                     {k: np.asarray(v[-n:]) for k, v in rows.items()})
        data = {k: np.asarray(v) for k, v in rows.items()}
        save_npz(output / f'{role}_transitions.npz', data)
        datasets[role] = data
    write_json(output / 'collection.json', episodes)
    save_points(output / 'development_snapshots.pkl', points['development'], env)
    save_points(output / 'probe_snapshots.pkl', points['probe'], env)
    require(all(e['complete_8s_trial'] for e in episodes), 'data collection contains a failed short trial')
    require(len(points['development']) == len(points['probe']) == 9, 'missing registered branch point')
    return datasets, points


def physical_probes(env, actor, points, output):
    from tools33.reference_shape33 import ReferenceShaper
    result = []
    for index, p in enumerate(points):
        branches, arrays = [], {}
        for order, theta in enumerate((0., 0., .005, -.005, .01, -.01)):
            row, states, observations, rewards = control.real_branch(env, ReferenceShaper(), actor, p, theta)
            branches.append(row)
            arrays[f'states_{order}'] = np.asarray(states)
            arrays[f'observations_{order}'] = np.asarray(observations)
            arrays[f'rewards_{order}'] = np.asarray(rewards)
        save_npz(output / f'probe_{index:02d}.npz', arrays)
        complete = all(b['status'] == 'COMPLETE' for b in branches)
        exact = branches[0] == branches[1] and all(np.array_equal(arrays[f'{k}_0'], arrays[f'{k}_1'])
                                                  for k in ('states', 'observations', 'rewards'))
        small = (branches[2]['cost'] - branches[3]['cost']) / .01 if complete else None
        large = (branches[4]['cost'] - branches[5]['cost']) / .02 if complete else None
        identifiable = bool(complete and exact and abs(small) >= 1e-5 and abs(large) >= 1e-5
                            and small * large > 0 and abs(small - large) <= .25 * max(abs(small), abs(large)))
        row = {'seed': p['seed'], 'tick': p['tick'], 'branches': branches,
               'repeated_baseline_exact': exact, 'gradient': small, 'gradient_double_epsilon': large,
               'identifiable': identifiable}
        write_json(output / f'probe_{index:02d}.json', row)
        result.append(row)
    return result


def paired_control(env, actor, theta, output, seeds, plan):
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools33.reference_shape33 import ReferenceShaper
    helpers = frozen_helpers({'sources_sha256': {k: v['sha256'] for k, v in plan['identities'].items()}})
    rows = []
    for name, spec in helpers.SEQUENCES.items():
        for seed in seeds:
            pair = {'task': name, 'seed': seed}
            for label, value in (('baseline', 0.), ('candidate', theta)):
                env.eval_command_sequence = env._validate_eval_sequence(spec)
                shaper = ReferenceShaper()
                def predict(_observation, shaper=shaper, value=value):
                    return control.real_action(env, shaper, actor, value)
                item = helpers.run_episode(env, predict, name, 0, seed,
                                   output / f'{name}_{seed}_{label}.npz',
                                   (env.max_transitions - env.counters()['policy_transitions']) * 10)
                pair[label] = item
            rows.append(pair)
            write_json(output / f'{name}_{seed}.json', pair)
    turn = [p for p in rows if p['task'] == 'turn_stop']
    def ratio(numerator, denominator):
        if any(x is None for x in numerator + denominator):
            return None
        top, bottom = float(np.mean(numerator)), float(np.mean(denominator))
        return top / bottom if np.isfinite(top) and np.isfinite(bottom) and bottom > 1e-12 else None

    turn_ratio = ratio([p['candidate']['stable_wz_mae_rad_s'] for p in turn],
                       [p['baseline']['stable_wz_mae_rad_s'] for p in turn])
    reasons = []
    if not all(p[a]['reference_thresholds_met'] for p in rows for a in ('baseline', 'candidate')):
        reasons.append('original thresholds not passed by every baseline/candidate episode')
    if turn_ratio is None or not np.isfinite(turn_ratio) or turn_ratio > .95:
        reasons.append('paired yaw MAE improvement below 5 percent')
    if not all(p['candidate']['stable_wz_mae_rad_s'] is not None and
               p['baseline']['stable_wz_mae_rad_s'] is not None and
               p['candidate']['stable_wz_mae_rad_s'] < p['baseline']['stable_wz_mae_rad_s'] for p in turn):
        reasons.append('yaw MAE did not improve on all three reset seeds')
    j_ratio = ratio([p['candidate']['J'] for p in rows], [p['baseline']['J'] for p in rows])
    if j_ratio is None or j_ratio > 1. or any(p['candidate']['J'] > 1.05 * p['baseline']['J'] for p in rows):
        reasons.append('aggregate J or individual-pair regression gate failed')
    return {'pairs': rows, 'yaw_mae_ratio': turn_ratio, 'J_ratio': j_ratio,
            'passed': not reasons, 'reasons': reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--admission', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    report = {'status': 'INCOMPLETE', 'failure': None, 'stage': 'preflight' if args.preflight else 'experiment',
              'plan_sha256': sha(args.plan), 'policy_network_updates': 0, 'control_parameter_updates': 0,
              'model_updates': 0, 'seeds': [], 'candidate_accepted': False, 'counts': None}
    started, cpu = time.monotonic(), time.process_time()
    env = None
    write_json(args.output / 'initial_incomplete.json', report | {'counts': 'unknown until final report'})

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restriction differs')
        verify(plan)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'mujoco', 'gymnasium')}
        require(report['versions'] == plan['versions'], 'registered package versions differ')
        actor = adapter.load_actor(ROOT / plan['checkpoint'])
        report['actor_admission'] = actor_admission(actor)
        if not args.preflight:
            require(args.admission is not None, 'preflight receipt required')
            admission = json.loads(args.admission.read_text())
            require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                    and admission['plan_sha256'] == report['plan_sha256'], 'preflight is incomplete or unbound')
            report['admission_sha256'] = sha(args.admission)
        env = make_env(256 if args.preflight else 21000)
        if args.preflight:
            report['interface_admission'] = preflight(env, actor, args.output)
        else:
            datasets, points = collect(env, actor, args.output, plan)
            models = []
            for seed in plan['model_seeds']:
                row = {'seed': seed, 'status': 'INCOMPLETE', 'failure': None}
                try:
                    model, fitted = dynamics.fit_model(datasets['train'], seed, args.output / f'model_{seed}',
                        min(cpu + 3500, time.process_time() + 600))
                    report['model_updates'] += fitted['updates']
                    row.update(fit=fitted, diagnostics={name: model.evaluate(datasets[name])
                        for name in ('development', 'probe')},
                        development_gradients=control.gradients(model, actor, points['development']),
                        test_gradients=control.gradients(model, actor, points['probe']))
                    row['status'] = 'COMPLETE'; models.append(model)
                except Exception as exc:  # noqa: BLE001 - retain each failed fit; never retry.
                    row['failure'] = f'{type(exc).__name__}: {exc}'
                    (args.output / f'model_{seed}_failure.txt').write_text(traceback.format_exc())
                report['seeds'].append(row)
                write_json(args.output / f'model_{seed}_result.json', row)
                print(json.dumps({'seed': seed, 'status': row['status'], 'failure': row['failure']}), flush=True)
            require(len(models) == 3, 'not all three final models and gradients completed')
            means = [float(np.mean([g['gradient'] for g in s['development_gradients']])) for s in report['seeds']]
            agreement = min(means) > 1e-6 or max(means) < -1e-6
            gradient_valid = all(g['autograd_agrees'] and g['direction_identifiable']
                                 for s in report['seeds'] for g in s['development_gradients'])
            theta = -.01 * float(np.sign(np.mean(means))) if agreement and gradient_valid else None
            sealed = {'theta': theta, 'development_mean_gradients': means,
                      'test_gradients': [s['test_gradients'] for s in report['seeds']],
                      'physical_results_seen': False, 'selection': 'equal mean; all three directions must agree; fixed step 0.01'}
            write_json(args.output / 'proposal_before_physics.json', sealed)
            report['sealed_proposal_sha256'] = sha(args.output / 'proposal_before_physics.json')
            report['control_parameter_updates'] = int(theta is not None)
            probes = physical_probes(env, actor, points['probe'], args.output)
            report['physical_probes'] = probes
            report['direction_gate'] = control.compare_directions(sealed, probes)
            report['theta'] = theta
            if report['direction_gate']['passed']:
                folder = args.output / 'paired_control'; folder.mkdir()
                report['paired_control'] = paired_control(env, actor, theta, folder, plan['evaluation_seeds'], plan)
                report['candidate_accepted'] = report['paired_control']['passed']
            else:
                report['paired_control'] = {'status': 'NOT_RUN', 'reason': 'predeclared model-gradient gate failed'}
        verify(plan)
        report.update(status='COMPLETE', frozen_inputs_unchanged=True)
    except BaseException as exc:  # noqa: BLE001 - preserve interruption evidence and partial counts.
        report['failure'] = f'{type(exc).__name__}: {exc}'
        report['candidate_accepted'] = False
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        journal_counts = {}
        for journal in args.output.glob('model_*/updates.jsonl'):
            with journal.open() as stream:
                journal_counts[journal.parent.name] = sum(1 for line in stream if line.endswith('\n'))
        report['journaled_model_updates'] = journal_counts
        report['model_updates'] = sum(journal_counts.values())
        report['update_count_semantics'] = 'completed updates journaled; interrupted update may be unrecorded'
        if env is not None:
            report['counts'] = env.counters()
            env.close()
        report['wall_seconds'], report['cpu_seconds'] = time.monotonic() - started, time.process_time() - cpu
        report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if report['artifact_bytes_before_report'] > 128 << 20:
            report.update(status='INCOMPLETE', failure='artifact cap exceeded', candidate_accepted=False)
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('seeds', 'physical_probes', 'paired_control')}), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
