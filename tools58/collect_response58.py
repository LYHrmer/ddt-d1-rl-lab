"""58: replay complete53 turn training cases, then measure one-step action FD.

Snapshots are live, full captures after_actor_before_env_step. Only a readback
is serialized; it is not an exact restore artifact. Absolute action deltas do
not include yaw error and do not inherit the55C theta stability result. All
labels and six-model comparisons are diagnostics, with no parameter updates or
claim that admitted points qualify a training dataset.
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
from tools53.run_pilot53 import make_step_recorder, require, sha, verify, write_json
from tools55.probe_local55 import action_delta_ulps
from tools55.run55 import audit_work

TRAIN_SEEDS = (53103, 53106, 53109, 53112)
TICKS = (110, 260, 410)
EPSILONS = (.000003125, .00000625, .0000125, .000025)
MODEL_SEEDS = (56001, 56002, 56003)
HORIZONS = (8, 32)
PLAN53 = Path('docs/experiments53/model_gradient53_preregistered.json')
SUMMARY53 = Path('reports/model_gradient53_results_20261101.json')
SOURCE53 = Path('evidence/model_gradient53/execution_01')
SOURCE56 = Path('evidence/paired_horizon56/execution_01')
ADMISSION55 = Path('evidence/observed_gradient55/admission_01/report.json')
CAPTURE_PHASE = 'after_actor_before_env_step'


def verify58(plan):
    require(plan['schema'] == 'd1-train-action-response58-v1', 'wrong58 protocol')
    require(plan['train_seeds'] == list(TRAIN_SEEDS) and plan['eps'] == list(EPSILONS), '58 seeds/action eps changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['expected_policy_steps'] == 2520 and plan['hard_policy_cap'] == 3000, '58 budget changed')
    required = {str(PLAN53), str(SUMMARY53), str(ADMISSION55), str(SOURCE56 / 'report.json'),
                'tools58/collect_response58.py', 'tools52/adapter52.py', 'tools52/control52.py',
                'tools52/dynamics52.py', 'tools53/collection53.py', 'tools53/run_pilot53.py',
                'tools55/probe_local55.py', 'tools55/run55.py'}
    required.update(str(SOURCE53 / f'collection_train_{seed}.{suffix}')
                    for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    required.update(str(SOURCE56 / f'h{horizon}_seed{seed}/final_state_dict.pt')
                    for seed in MODEL_SEEDS for horizon in HORIZONS)
    require(required <= set(plan['source_sha256']), 'required58 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '58 source/input changed: ' + name)


def yaw_stability(rows):
    """Apply the declared scalar rule to absolute-action eps, never to theta."""
    require([row['epsilon'] for row in rows] == list(EPSILONS), 'incomplete action-epsilon grid')
    central = [row['central'] for row in rows]
    scale = max(abs(value) for value in central)
    spread = (max(central) - min(central)) / scale if scale else None
    same_sign = all(value > 0 for value in central) or all(value < 0 for value in central)
    sides = []
    for row in rows:
        forward, backward = row['forward'], row['backward']
        magnitude = max(abs(forward), abs(backward))
        gap = abs(forward - backward) / magnitude if magnitude else None
        sides.append({'epsilon': row['epsilon'], 'same_sign': forward * backward > 0, 'relative_gap': gap,
                      'stable': forward * backward > 0 and gap is not None and gap <= .25})
    floor = all(abs(value) >= 1e-5 for value in central)
    return {'central_same_sign': same_sign, 'all_central_abs_ge_1e_5': floor,
            'central_whole_grid_relative_spread': spread, 'one_sided': sides,
            'stable': same_sign and floor and spread is not None and spread <= .25
            and all(row['stable'] for row in sides), 'selected_epsilon': None,
            'coordinate': 'absolute action delta along fixed wheel direction; no yaw_error multiplier'}


def state_response(branches):
    """Full45-state central/one-sided arrays plus descriptive seven-group checks."""
    import numpy as np

    from tools52 import dynamics52 as dynamics
    zero = branches[0]['target']
    central, forward, backward = [], [], []
    for index, epsilon in enumerate(EPSILONS):
        plus, minus = branches[2 + 2 * index:4 + 2 * index]
        central.append((plus['target'] - minus['target']) / (2 * epsilon))
        forward.append((plus['target'] - zero) / epsilon)
        backward.append((zero - minus['target']) / epsilon)
    central, forward, backward = map(np.asarray, (central, forward, backward))
    require(all(value.shape == (4, 45) and np.isfinite(value).all() for value in (central, forward, backward)),
            'invalid state response arrays')
    scale = dynamics.PHYSICAL_SCALE.astype(np.float64)
    mean = central.mean(axis=0)

    def rms(value):
        return float(np.sqrt(np.mean(np.square(value))))

    groups = {}
    for name, indices in dynamics.GROUPS.items():
        indices = list(indices)
        mean_rms = rms((mean / scale)[indices])
        rows = []
        for index, epsilon in enumerate(EPSILONS):
            deviation = rms(((central[index] - mean) / scale)[indices])
            side_difference = rms(((forward[index] - backward[index]) / scale)[indices])
            denominator = max(rms((forward[index] / scale)[indices]), rms((backward[index] / scale)[indices]), 1e-8)
            rows.append({'epsilon': epsilon, 'normalized_rms_deviation_from_four_eps_mean': deviation,
                         'relative_rms_deviation_from_four_eps_mean': deviation / max(mean_rms, 1e-8),
                         'normalized_rms_forward_backward_difference': side_difference,
                         'relative_rms_forward_backward_difference': side_difference / denominator})
        groups[name] = {'normalized_rms_four_eps_mean': mean_rms, 'eps': rows,
                        'denominator_floor': 1e-8, 'admission_gate': False}
    yaw = [{'epsilon': epsilon, 'central': float(central[index, 28]),
            'forward': float(forward[index, 28]), 'backward': float(backward[index, 28])}
           for index, epsilon in enumerate(EPSILONS)]
    return {'central_state45_fd': central, 'forward_state45_fd': forward, 'backward_state45_fd': backward}, \
        {'yaw28': yaw, 'yaw_stability': yaw_stability(yaw), 'group_diagnostics': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deltas = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-train-action-response58-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'control_parameter_updates': 0, 'training_qualification_claimed': False,
              'capture_phase': CAPTURE_PHASE, 'counts': None, 'completed_branches': 0,
              'completed_model_fd_comparisons': 0, 'dynamics_forward_calls': 0,
              'cases': [{'seed': seed, 'episode': seed - 53101, 'status': 'NOT_RUN', 'steps': 0} for seed in TRAIN_SEEDS],
              'points': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN', 'label_admitted': False,
                          'branches': [{'delta': delta, 'status': 'NOT_RUN'} for delta in deltas]}
                         for seed in TRAIN_SEEDS for tick in TICKS],
              'models': [{'seed': seed, 'horizon': horizon, 'status': 'NOT_RUN', 'points': []}
                         for seed in MODEL_SEEDS for horizon in HORIZONS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress = {'phase': 'admission', 'seed': None, 'tick': None, 'branch': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress,
            'counts': None if env is None else env.counters(), 'completed_branches': report['completed_branches'],
            'completed_model_fd_comparisons': report['completed_model_fd_comparisons']}, allow_nan=False) + '\n')
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
        verify58(plan)
        original = json.loads((ROOT / PLAN53).read_text())
        verify(original)
        summary = json.loads((ROOT / SUMMARY53).read_text())
        require(summary['formal']['status'] == 'COMPLETE' and summary['plan_sha256'] == sha(ROOT / PLAN53),
                'complete bound53 evidence required')
        admission = json.loads((ROOT / ADMISSION55).read_text())
        require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                and admission['counts']['policy_transitions'] == 224, 'complete55 interface admission required')
        source56 = json.loads((ROOT / SOURCE56 / 'report.json').read_text())
        require(source56['status'] == 'COMPLETE' and source56['completed_model_fits'] == 6
                and source56['model_updates'] == 6000, 'six complete56 models required')
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == source56['versions'], 'package versions differ')
        import numpy as np
        import torch

        from tools33.reference_shape33 import ReferenceShaper
        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == source56['torch_runtime_version'], 'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        actor = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor)
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        base_recorder = make_step_recorder(audit, progress)

        def recorder(current, action, result):
            base_recorder(current, action, result)
            audit.flush()

        env = collection.make_env(3000, audit_step=recorder)
        spec = collection.helpers(original).SEQUENCES['turn_stop']
        sites = []
        progress['phase'] = 'baseline_replay'
        for case in report['cases']:
            require(time.process_time() < 298., 'CPU deadline before baseline')
            seed, rows = case['seed'], []
            progress.update(seed=seed, tick=None)
            case['status'] = 'INCOMPLETE'
            for suffix in ('npz', 'json'):
                name = f'execution_01/collection_train_{seed}.{suffix}'
                identity = summary['raw_manifest'][name]
                path = ROOT / 'evidence/model_gradient53' / name
                require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'], 'original train input differs')
            old_case = json.loads((ROOT / SOURCE53 / f'collection_train_{seed}.json').read_text())
            require(old_case['status'] == 'COMPLETE' and old_case['steps'] == 600
                    and old_case['seed'] == seed and old_case['episode'] == case['episode']
                    and old_case['role'] == 'train' and old_case['task'] == 'turn_stop', 'incomplete/wrong original train case')
            try:
                env.eval_command_sequence = env._validate_eval_sequence(spec)
                env.reset(seed=seed)
                shaper = ReferenceShaper()
                noise_rng = np.random.default_rng(seed + 900000)
                require(env._delay_ticks == 0 and len(env._delay_fifo) == 0, 'zero-delay action seam required')
                for tick in range(600):
                    progress['tick'] = tick
                    state = control.physical(env)
                    noise = noise_rng.normal(0., .02, 16)
                    action = control.real_action(env, shaper, actor, 0., noise)
                    site = None
                    if tick in TICKS:
                        site = {'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state.copy(),
                                'base_action': action.copy(), 'snapshot': adapter.capture(env, shaper)}
                        readback = {'capture_phase': CAPTURE_PHASE, 'exact_restore_artifact': False,
                                    'seed': seed, 'tick': tick, 'episode': case['episode'],
                                    'base_action': action, 'noise': noise, 'noise_rng_seed': seed + 900000,
                                    'environment': env.get_initial_state(), 'shaper': vars(shaper), 'counts': env.counters()}
                        write_json(args.output / f'snapshot_readback_{seed}_{tick}.json', collection._plain(readback))
                    event('BASELINE_STEP_BEGIN')
                    result = env.step(action)
                    row = collection._record(env, state, action, result, case['episode'], tick, final_tick=tick == 599)
                    rows.append(row)
                    case['steps'] = len(rows)
                    event('BASELINE_STEP_COMPLETE')
                    if site is not None:
                        site.update(baseline=row, baseline_observation=result[0].copy())
                        sites.append(site)
                    require(not any(row[key] for key in ('terminated', 'truncated', 'fallen', 'illegal_body_contact')),
                            'baseline terminated/unsafe')
                current = collection._arrays(rows)
                with np.load(ROOT / SOURCE53 / f'collection_train_{seed}.npz', allow_pickle=False) as old:
                    same_keys = set(current) == set(old.files)
                    exact = {key: bool(np.array_equal(value, old[key]))
                             for key, value in current.items() if key in old.files and key not in collection.COUNTERS}
                case.update(same_keys=same_keys, physical_exact_by_field=exact,
                            all_physical_fields_exact=bool(same_keys and all(exact.values())))
                require(case['all_physical_fields_exact'], 'full noisy training replay did not reproduce53 exactly')
                case['status'] = 'COMPLETE'
            except BaseException as exc:
                case['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                collection.save_npz(args.output / f'baseline_{seed}.npz', collection._arrays(rows))
                write_json(args.output / f'baseline_{seed}.json', case)
        require(env.counters()['policy_transitions'] == 2400 and len(sites) == 12
                and [(site['seed'], site['tick']) for site in sites] == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS],
                'baseline/capture work differs')
        report['all_four_complete_baselines_exact'] = True
        report['live_snapshot_count'] = len(sites)
        report['snapshot_data_buffer_bytes_each'] = int(env.data.narena + env.data.nbuffer)
        event('ALL_BASELINES_ADMITTED')
        direction = np.zeros(16, dtype=np.float64)
        direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
        report['action_direction'] = {'vector': direction.tolist(), 'linf': 1., 'l2': 2., 'yaw_error_multiplier': False}
        collected_branches = []
        progress['phase'] = 'one_step_branches'
        for index, site in enumerate(sites):
            point = report['points'][index]
            point['status'] = 'INCOMPLETE'
            progress.update(seed=site['seed'], tick=site['tick'])
            point_rows = []
            for order, delta in enumerate(deltas):
                require(time.process_time() < 298., 'CPU deadline before branch')
                progress['branch'] = order
                branch = {'status': 'INCOMPLETE', 'delta': delta, 'steps': 0, 'capture_phase': CAPTURE_PHASE,
                          'counts_before': env.counters()}
                record = None
                arrays = {}
                try:
                    adapter.restore(env, shaper, site['snapshot'])
                    require(np.array_equal(control.physical(env), site['state']), 'snapshot physical state changed')
                    action = np.clip(site['base_action'].astype(np.float64) + delta * direction, -1., 1.).astype(np.float32)
                    event('BRANCH_STEP_BEGIN')
                    result = env.step(action)
                    record = collection._record(env, site['state'], action, result, site['episode'], site['tick'], final_tick=False)
                    arrays = collection._arrays([record]) | {'observations': result[0][None].copy()}
                    branch.update(status='COMPLETE' if not any(record[key] for key in
                                  ('terminated', 'truncated', 'fallen', 'illegal_body_contact')) else 'TERMINATED', steps=1,
                                  record=collection._plain(record), counts_after=env.counters(),
                                  issued_equals_applied=bool(np.array_equal(action, record['applied_action'])))
                    if order < 2:
                        keys = ('s', 'a', 'target', 'command_during_interval', 'next_command', 'issued_action',
                                'applied_action', 'reward', 'fallen', 'illegal_body_contact', 'base_height_m',
                                'tilt_rad', 'terminated', 'truncated')
                        branch['baseline_exact_by_field'] = {key: bool(np.array_equal(record[key], site['baseline'][key])) for key in keys}
                        branch['baseline_observation_exact'] = bool(np.array_equal(result[0], site['baseline_observation']))
                        branch['zero_exact'] = all(branch['baseline_exact_by_field'].values()) and branch['baseline_observation_exact']
                    else:
                        branch['actual_action_delta'] = action_delta_ulps(site['base_action'], record['applied_action'])
                    point_rows.append(record)
                except BaseException as exc:
                    branch.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}', counts_after=env.counters())
                    if record is None:
                        arrays = {'attempted_post_state': control.physical(env), 'base_action': site['base_action']}
                    raise
                finally:
                    collection.save_npz(args.output / f'branch_{index:02d}_{order:02d}.npz', arrays)
                    point['branches'][order] = branch
                    write_json(args.output / f'branch_{index:02d}_{order:02d}.json', branch)
                    event('BRANCH_SAVED')
                require(branch['status'] == 'COMPLETE' and branch['issued_equals_applied'], 'branch incomplete/unsafe/delayed')
                if order < 2:
                    require(branch['zero_exact'], 'zero branch differs from original reproduced baseline')
                report['completed_branches'] += 1
            arrays, diagnostics = state_response(point_rows)
            point.update(diagnostics)
            point['all_four_wheels_nonzero'] = all(branch['actual_action_delta']['all_four_wheels_nonzero']
                                                  for branch in point['branches'][2:])
            point.update(status='COMPLETE', label_admitted=point['all_four_wheels_nonzero']
                         and point['yaw_stability']['stable'] and all(branch['zero_exact'] for branch in point['branches'][:2]))
            collection.save_npz(args.output / f'response_{index:02d}.npz', arrays)
            write_json(args.output / f'response_{index:02d}.json', point)
            collected_branches.append((point_rows, arrays))
        require(report['completed_branches'] == 120 and env.counters()['policy_transitions'] == 2520, 'physical work differs')
        report['labels_admitted'] = sum(point['label_admitted'] for point in report['points'])
        progress.update(phase='six_model_diagnostics', branch=None)
        scale = torch.tensor(dynamics.PHYSICAL_SCALE, dtype=torch.float64)
        for model_row in report['models']:
            model_row['status'] = 'INCOMPLETE'
            directory = f'h{model_row["horizon"]}_seed{model_row["seed"]}'
            output = args.output / directory
            output.mkdir(exist_ok=False)
            weight_name = directory + '/final_state_dict.pt'
            require(sha(ROOT / SOURCE56 / weight_name) == source56['sealed_artifacts_sha256'][weight_name], '56 final weights differ')
            source_model = next(row for row in source56['models']
                                if row['seed'] == model_row['seed'] and row['horizon'] == model_row['horizon'])
            require(source_model['status'] == 'COMPLETE' and source_model['fit']['updates'] == 1000, 'partial56 model ineligible')
            model = dynamics.Dynamics52()
            model.load_state_dict(torch.load(ROOT / SOURCE56 / weight_name, map_location='cpu', weights_only=True))
            model = model.double().eval().requires_grad_(False)
            for index, (site, (branches, physical_arrays)) in enumerate(zip(sites, collected_branches, strict=True)):
                require(time.process_time() < 298., 'CPU deadline before model point')
                progress.update(seed=site['seed'], tick=site['tick'])
                point = {'seed': site['seed'], 'tick': site['tick'], 'status': 'INCOMPLETE', 'eps': []}
                model_row['points'].append(point)
                values = []
                try:
                    with torch.no_grad():
                        state = torch.tensor(site['state'][None], dtype=torch.float64)
                        for order, epsilon in enumerate(EPSILONS):
                            plus, minus = branches[2 + 2 * order:4 + 2 * order]
                            a_plus = torch.tensor(plus['applied_action'][None], dtype=torch.float64)
                            a_minus = torch.tensor(minus['applied_action'][None], dtype=torch.float64)
                            predicted_plus = model(state, a_plus)
                            report['dynamics_forward_calls'] += 1
                            predicted_minus = model(state, a_minus)
                            report['dynamics_forward_calls'] += 1
                            predicted = (predicted_plus - predicted_minus)[0] / (2 * epsilon)
                            actual = torch.tensor(physical_arrays['central_state45_fd'][order], dtype=torch.float64)
                            require(bool(torch.isfinite(predicted).all()), 'nonfinite model response')
                            error = (predicted - actual) / scale
                            point['eps'].append({'epsilon': epsilon, 'yaw28_actual_fd': float(actual[28]),
                                'yaw28_model_fd': float(predicted[28]), 'yaw28_same_sign': bool(actual[28] * predicted[28] > 0),
                                'normalized_rms_error_by_group': {name: float(error[list(indices)].square().mean().sqrt())
                                                                for name, indices in dynamics.GROUPS.items()}})
                            values.append(predicted.numpy())
                            report['completed_model_fd_comparisons'] += 1
                    point.update(status='COMPLETE', training_labels_used_for_fit=False,
                                 physical_label_admitted=report['points'][index]['label_admitted'])
                except BaseException as exc:
                    point['failure'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    collection.save_npz(output / f'point_{index:02d}.npz', {
                        'model_state45_fd': np.asarray(values).reshape(-1, 45), **physical_arrays})
                    write_json(output / f'point_{index:02d}.json', point)
            model_row['status'] = 'COMPLETE'
            write_json(output / 'result.json', model_row)
        require(report['completed_model_fd_comparisons'] == 288 and report['dynamics_forward_calls'] == 576,
                'complete6×12×4 diagnostic required')
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - every failed or partial stage remains available without resuming.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        cleanup = []
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure.
                cleanup.append(f'env: {exc}')
        if audit is not None:
            try:
                audit.close()
                report['work_journal'] = audit_work(args.output / 'all_physics_steps.jsonl')
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'], 'work journal differs')
                if report['status'] == 'COMPLETE':
                    require(report['work_journal']['by_phase'] == {'baseline_replay': 2400, 'one_step_branches': 120},
                            'physical phase work differs')
            except Exception as exc:  # noqa: BLE001 - retain partial journal and attempted counters.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify58(plan)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            if 'torch_runtime_version' in report:
                require(sys.modules['torch'].__version__ == report['torch_runtime_version'], 'torch runtime changed')
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - retain the original failure too.
            report['verification_after'] = 'FAILED'
            cleanup.append(f'verification: {exc}')
        if cleanup:
            report.update(status='INCOMPLETE', cleanup_errors=cleanup)
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'counts', 'completed_branches',
                                                      'completed_model_fd_comparisons', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
