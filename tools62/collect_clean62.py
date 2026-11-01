"""62: complete clean-action histories and registered one-step response labels.

The four original53 turn seeds and reset states are retained, but exploration
noise is absent for the whole600-step history. This deliberately changes the
visited states, actor/shaper history and current actions together. Comparison
with60's clean-center/noisy-history observations is descriptive, not a pure
state intervention. No dynamics model is loaded and no parameters are updated.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from tools53.run_pilot53 import make_step_recorder, require, sha, verify, write_json
from tools55.probe_local55 import action_delta_ulps
from tools55.run55 import audit_work
from tools58.collect_response58 import (
    ADMISSION55,
    CAPTURE_PHASE,
    EPSILONS,
    PLAN53,
    SOURCE53,
    SUMMARY53,
    TICKS,
    TRAIN_SEEDS,
    state_response,
)
from tools60.collect_noise60 import PLAN58, SOURCE58

PLAN60 = Path('docs/experiments60/noise_center60_preregistered.json')
SOURCE60 = Path('evidence/noise_center60/execution_01')


def required_inputs():
    """SHA-bound62 dependencies; original53 verify also checks actor/scene identities."""
    paths = {str(PLAN53), str(SUMMARY53), str(PLAN58), str(PLAN60), str(ADMISSION55),
             str(SOURCE58 / 'report.json'), str(SOURCE60 / 'report.json'),
             'tools62/collect_clean62.py', 'tools60/collect_noise60.py', 'tools58/collect_response58.py',
             'tools52/adapter52.py', 'tools52/control52.py', 'tools52/dynamics52.py',
             'tools52/run_pilot52.py', 'tools53/collection53.py', 'tools53/run_pilot53.py',
             'tools55/probe_local55.py', 'tools55/run55.py'}
    paths.update(str(SOURCE53 / f'collection_train_{seed}.{suffix}')
                 for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    return paths


def verify62(plan):
    require(plan['schema'] == 'd1-clean-history-response62-v1', 'wrong62 protocol')
    require(plan['train_seeds'] == list(TRAIN_SEEDS) and plan['eps'] == list(EPSILONS)
            and plan['exploration_noise_sigma'] == 0, '62 history/seeds/eps changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['expected_policy_steps'] == 2520 and plan['hard_policy_cap'] == 3000,
            '62 budget changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required62 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '62 source/input changed: ' + name)


def clean_history_hypothesis(cases, points):
    """Separate full-data admission from the fixed four tick260 coverage question.

Any partial physical collection raises ValueError. An unknown secondary point
blocks data admission but does not erase separately reported coverage at260.
The combined supported flag requires both, and never selects seeds or eps.
"""
    require([case.get('seed') for case in cases] == list(TRAIN_SEEDS)
            and all(case.get('status') == 'COMPLETE' and case.get('steps') == 600 for case in cases),
            'four complete62 cases required')
    require([(point.get('seed'), point.get('tick')) for point in points]
            == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], '62 point grid differs')
    rows = []
    for point in points:
        require(point.get('status') == 'COMPLETE' and len(point.get('branches', [])) == 10
                and all(branch.get('status') == 'COMPLETE' for branch in point['branches']),
                'incomplete62 physical branch')
        require([row.get('epsilon') for row in point.get('yaw28', [])] == list(EPSILONS)
                and all(math.isfinite(row['central']) for row in point['yaw28']), 'invalid62 FD grid')
        resolved = point.get('label_admitted') is True
        negative = all(row['central'] < 0 for row in point['yaw28']) if resolved else None
        rows.append({'seed': point['seed'], 'tick': point['tick'], 'reference_resolved': resolved,
                     'all_four_eps_negative': negative})
    primary = [row for row in rows if row['tick'] == 260]
    resolved = all(row['reference_resolved'] for row in primary)
    covered = resolved and all(row['all_four_eps_negative'] for row in primary)
    data_admitted = all(row['reference_resolved'] for row in rows)
    coverage = {'status': 'SUPPORTED' if covered else 'REJECTED' if resolved else 'REFERENCE_UNRESOLVED',
                'supported': covered, 'fixed_primary_tick': 260, 'primary_points': primary}
    return {'status': coverage['status'] if data_admitted else 'REFERENCE_UNRESOLVED',
            'supported': bool(data_admitted and covered), 'data_admitted': data_admitted,
            'admitted_labels': sum(row['reference_resolved'] for row in rows),
            'coverage': coverage, 'all_point_diagnostics': rows, 'selected_points_or_eps': False,
            'scope': 'negative local-response coverage of a clean history; no control-performance or pure-state claim'}


def compare_history(point, reference):
    """Compare clean-history62 with the SHA-bound clean-center arm of60."""
    require((point['seed'], point['tick']) == (reference['seed'], reference['tick']), 'history point identity differs')
    old = reference['centers']['clean']
    require(reference['status'] == old['status'] == 'COMPLETE' and old['label_admitted']
            and len(old['branches']) == 10 and all(branch['status'] == 'COMPLETE' for branch in old['branches']),
            'complete stable60 clean-center reference required')
    current_record, old_record = point['branches'][0]['record'], old['branches'][0]['record']
    changes = {}
    for field, width in (('s', 45), ('issued_action', 16)):
        current, previous = current_record[field], old_record[field]
        require(len(current) == len(previous) == width, 'history comparison width differs')
        delta = [float(x) - float(y) for x, y in zip(current, previous, strict=True)]
        require(all(math.isfinite(value) for value in delta), 'nonfinite history comparison')
        changes[field] = {'clean_history_minus_noisy_history': delta,
                          'max_absolute': max(abs(value) for value in delta), 'exactly_equal': current == previous}
    require([row['epsilon'] for row in old['yaw28']] == list(EPSILONS)
            and [row['epsilon'] for row in point['yaw28']] == list(EPSILONS), 'history FD grid differs')
    return {'reference': str(SOURCE60 / 'report.json'), 'reference_action_center': 'clean',
            'current_history_exploration_noise_sigma': 0., 'reference_history_exploration_noise_sigma': .02,
            'state45': changes['s'], 'issued_action16': changes['issued_action'],
            'both_labels_admitted': bool(point['label_admitted'] and old['label_admitted']),
            'eps': [{'epsilon': current['epsilon'], 'clean_history_yaw28': current['central'],
                     'noisy_history_yaw28': previous['central'],
                     'same_sign': current['central'] * previous['central'] > 0}
                    for current, previous in zip(point['yaw28'], old['yaw28'], strict=True)],
            'interpretation': 'history changes state, action and policy history together; not a pure-state effect'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deltas = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    report = {'schema': 'd1-clean-history-response62-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'control_parameter_updates': 0, 'training_qualification_claimed': False,
              'capture_phase': CAPTURE_PHASE, 'counts': None, 'completed_branches': 0,
              'dynamics_models_loaded': 0, 'exploration_noise_sigma': 0.,
              'cases': [{'seed': seed, 'episode': seed - 53101, 'status': 'NOT_RUN', 'steps': 0} for seed in TRAIN_SEEDS],
              'points': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN', 'label_admitted': False,
                          'branches': [{'delta': delta, 'status': 'NOT_RUN'} for delta in deltas]}
                         for seed in TRAIN_SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress = {'phase': 'admission', 'seed': None, 'tick': None, 'branch': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress,
            'counts': None if env is None else env.counters(),
            'completed_branches': report['completed_branches']}, allow_nan=False) + '\n')
        events.flush()

    def deadline():
        require(time.process_time() < 298. and time.monotonic() - wall < 419., '62 CPU/wall deadline')

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
        verify62(plan)
        original = json.loads((ROOT / PLAN53).read_text())
        verify(original)
        summary = json.loads((ROOT / SUMMARY53).read_text())
        require(summary['formal']['status'] == 'COMPLETE' and summary['plan_sha256'] == sha(ROOT / PLAN53),
                'complete bound53 evidence required')
        admission = json.loads((ROOT / ADMISSION55).read_text())
        require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                and admission['counts']['policy_transitions'] == 224, 'complete55 interface admission required')
        source58 = json.loads((ROOT / SOURCE58 / 'report.json').read_text())
        require(source58['status'] == 'COMPLETE' and source58['plan_sha256'] == sha(ROOT / PLAN58)
                and source58['completed_branches'] == 120, 'complete bound58 evidence required')
        old_plan = json.loads((ROOT / PLAN60).read_text())
        source60 = json.loads((ROOT / SOURCE60 / 'report.json').read_text())
        require(old_plan['schema'] == 'd1-noise-center-response60-v1' and source60['status'] == 'COMPLETE'
                and source60['plan_sha256'] == sha(ROOT / PLAN60) and source60['completed_branches'] == 240
                and source60['counts']['policy_transitions'] == 2640, 'complete bound60 evidence required')
        for name in ('tools60/collect_noise60.py', 'tools58/collect_response58.py', str(PLAN53), str(SUMMARY53),
                     str(PLAN58), str(SOURCE58 / 'report.json')):
            require(sha(ROOT / name) == old_plan['source_sha256'][name], '60 lineage changed: ' + name)
        require([(point['seed'], point['tick']) for point in source60['points']]
                == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], '60 reference grid changed')
        require(all(point['status'] == 'COMPLETE' and point['centers']['clean']['status'] == 'COMPLETE'
                    and point['centers']['clean']['label_admitted'] for point in source60['points']),
                'all12 original60 clean-center references must be stable')
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == source60['versions'], 'package versions differ')
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
        require(torch.__version__ == source60['torch_runtime_version'], 'torch runtime differs')
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
        progress['phase'] = 'clean_baseline'
        for case in report['cases']:
            deadline()
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
                initial = control.physical(env)
                with np.load(ROOT / SOURCE53 / f'collection_train_{seed}.npz', allow_pickle=False) as old:
                    case['initial_state45_exact'] = bool(np.array_equal(initial, old['s'][0]))
                    case['initial_state45'] = initial.tolist()
                require(case['initial_state45_exact'], 'clean history reset differs from original53 state45')
                shaper = ReferenceShaper()
                require(env._delay_ticks == 0 and len(env._delay_fifo) == 0, 'zero-delay action seam required')
                for tick in range(600):
                    deadline()
                    progress['tick'] = tick
                    state = control.physical(env)
                    action = control.real_action(env, shaper, actor, 0., noise=None)
                    site = None
                    if tick in TICKS:
                        site = {'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state.copy(),
                                'base_action': action.copy(), 'snapshot': adapter.capture(env, shaper)}
                        readback = {'capture_phase': CAPTURE_PHASE, 'exact_restore_artifact': False,
                                    'seed': seed, 'tick': tick, 'episode': case['episode'],
                                    'base_action': action, 'exploration_noise_sigma': 0., 'state': state,
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
                require(len(rows) == 600 and case['initial_state45_exact'], 'incomplete clean history')
                case.update(status='COMPLETE', exploration_noise_sigma=0.,
                            full_baseline53_equality_required=False,
                            intervention='clean actions throughout; state/action/history may diverge after reset')
            except BaseException as exc:
                case['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                collection.save_npz(args.output / f'baseline_{seed}.npz', collection._arrays(rows))
                write_json(args.output / f'baseline_{seed}.json', case)
        require(env.counters()['policy_transitions'] == 2400 and len(sites) == 12
                and [(site['seed'], site['tick']) for site in sites] == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS],
                'baseline/capture work differs')
        report['all_four_complete_initial_states_exact'] = True
        report['live_snapshot_count'] = len(sites)
        report['snapshot_data_buffer_bytes_each'] = int(env.data.narena + env.data.nbuffer)
        event('ALL_BASELINES_ADMITTED')
        direction = np.zeros(16, dtype=np.float64)
        direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
        report['action_direction'] = {'vector': direction.tolist(), 'linf': 1., 'l2': 2., 'yaw_error_multiplier': False}
        progress['phase'] = 'one_step_branches'
        for index, site in enumerate(sites):
            point = report['points'][index]
            point['status'] = 'INCOMPLETE'
            progress.update(seed=site['seed'], tick=site['tick'])
            point_rows = []
            for order, delta in enumerate(deltas):
                deadline()
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
                    require(branch['zero_exact'], 'zero branch differs from this clean baseline')
                report['completed_branches'] += 1
            arrays, diagnostics = state_response(point_rows)
            point.update(diagnostics)
            point['all_four_wheels_nonzero'] = all(branch['actual_action_delta']['all_four_wheels_nonzero']
                                                  for branch in point['branches'][2:])
            point.update(status='COMPLETE', label_admitted=point['all_four_wheels_nonzero']
                         and point['yaw_stability']['stable'] and all(branch['zero_exact'] for branch in point['branches'][:2]))
            point['history_comparison'] = compare_history(point, source60['points'][index])
            collection.save_npz(args.output / f'response_{index:02d}.npz', arrays)
            write_json(args.output / f'response_{index:02d}.json', point)
        require(report['completed_branches'] == 120 and env.counters()['policy_transitions'] == 2520
                and env.counters()['native_steps'] == 25200 and env.counters()['pd_updates'] == 12600, 'physical work differs')
        report['labels_admitted'] = sum(point['label_admitted'] for point in report['points'])
        report['hypothesis'] = clean_history_hypothesis(report['cases'], report['points'])
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
                    require(report['work_journal']['by_phase'] == {'clean_baseline': 2400, 'one_step_branches': 120},
                            'physical phase work differs')
            except Exception as exc:  # noqa: BLE001 - retain partial journal and attempted counters.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify62(plan)
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
        if report['status'] != 'COMPLETE' and 'hypothesis' in report:
            report['hypothesis_before_final_eligibility'] = report['hypothesis']
            report['hypothesis'] = {'status': 'INCOMPLETE', 'supported': False, 'data_admitted': False}
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        report['artifact_bytes_before_report'] = sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())
        event('FINAL_REPORT')
        events.close()
        try:
            report['sealed_artifacts_sha256'] = {str(path.relative_to(args.output)): sha(path)
                for path in sorted(args.output.rglob('*')) if path.is_file()}
        except Exception as exc:  # noqa: BLE001 - sealing failure invalidates eligibility.
            report.update(status='INCOMPLETE', sealing_failure=f'{type(exc).__name__}: {exc}')
            if 'hypothesis' in report:
                report['hypothesis_before_final_eligibility'] = report['hypothesis']
                report['hypothesis'] = {'status': 'INCOMPLETE', 'supported': False, 'data_admitted': False}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'report_sha256.json', {'report.json': sha(args.output / 'report.json')})
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'counts', 'completed_branches',
                                                      'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
