"""61: test whether solver warmstart aliases the45-state action-response input.

The bounded60 replay/archival skeleton is retained in this independent entry
point. Only qacc_warmstart is intervened on after restoring a full live capture:
preserved versus zeroed. Both arms use the same clean action, physical state,
policy history, wheel phase and solver options. No extra forward/actor call is
allowed between restoration and step. This is a numerical-state sufficiency
diagnostic, not a model fit or a control-performance experiment.
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
from tools60.collect_noise60 import PLAN58, SOURCE58, exact_physical

PLAN60 = Path('docs/experiments60/noise_center60_preregistered.json')
SOURCE60 = Path('evidence/noise_center60/execution_01')
ARMS = ('preserved', 'zeroed')
DELTAS = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
UNSAFE = ('terminated', 'truncated', 'fallen', 'illegal_body_contact')


def required_inputs():
    """61 dependencies; caller binds all paths before execution, including actor53."""
    paths = {str(PLAN53), str(SUMMARY53), str(PLAN58), str(PLAN60), str(ADMISSION55),
             str(SOURCE58 / 'report.json'), str(SOURCE60 / 'report.json'),
             'tools61/warmstart61.py', 'tools60/collect_noise60.py', 'tools58/collect_response58.py',
             'tools52/adapter52.py', 'tools52/control52.py', 'tools52/dynamics52.py',
             'tools52/run_pilot52.py', 'tools53/collection53.py', 'tools53/run_pilot53.py',
             'tools55/probe_local55.py', 'tools55/run55.py'}
    paths.update(str(SOURCE53 / f'collection_train_{seed}.{suffix}')
                 for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    paths.update(str(SOURCE60 / f'branch_{point:02d}_clean_{branch:02d}.{suffix}')
                 for point in range(12) for branch in range(10) for suffix in ('npz', 'json'))
    return paths


def verify61(plan):
    require(plan['schema'] == 'd1-warmstart-response61-v1', 'wrong61 protocol')
    require(plan['train_seeds'] == list(TRAIN_SEEDS) and plan['eps'] == list(EPSILONS)
            and plan['arms'] == list(ARMS) and plan['action_center'] == 'clean', '61 intervention grid changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['expected_policy_steps'] == 2640 and plan['hard_policy_cap'] == 3000,
            '61 budget changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required61 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '61 source/input changed: ' + name)


def warmstart_hypothesis(points):
    """Return registered witnesses; require all240 branches complete first.

Complete but unidentifiable reference points remain unknown; they do not erase
an established witness at another predetermined point. Every four-epsilon
pair must reverse sign, rather than merely its mean. Missing/partial execution
raises ValueError and cannot yield an eligible result.
"""
    require([(point.get('seed'), point.get('tick')) for point in points]
            == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], '61 point grid differs')
    rows = []
    for point in points:
        require(point.get('status') == 'COMPLETE' and set(point.get('arms', {})) == set(ARMS),
                'incomplete61 point/arm grid')
        arms = point['arms']
        for arm in arms.values():
            require(arm.get('status') == 'COMPLETE' and len(arm.get('branches', [])) == 10
                    and all(branch.get('status') == 'COMPLETE' for branch in arm['branches']),
                    'incomplete61 physical branch')
            require([row.get('epsilon') for row in arm.get('yaw28', [])] == list(EPSILONS)
                    and all(math.isfinite(row['central']) for row in arm['yaw28']), 'invalid61 FD grid')
        resolved = all(arms[name].get('label_admitted') is True for name in ARMS)
        opposite = all(left['central'] * right['central'] < 0 for left, right in
                       zip(arms['preserved']['yaw28'], arms['zeroed']['yaw28'], strict=True)) if resolved else None
        witness = resolved and opposite
        rows.append({'seed': point['seed'], 'tick': point['tick'], 'reference_resolved': resolved,
                     'opposite_sign_all_eps': opposite, 'witness': witness,
                     'central_fd_ranges': {name: {'min': min(row['central'] for row in arms[name]['yaw28']),
                                                 'max': max(row['central'] for row in arms[name]['yaw28'])}
                                           for name in ARMS},
                     'status': 'WITNESS_FOUND' if witness else 'NO_WITNESS_AT_POINT' if resolved else 'REFERENCE_UNRESOLVED'})
    count = sum(row['witness'] for row in rows)
    resolved = all(row['reference_resolved'] for row in rows)
    return {'status': 'WITNESS_FOUND' if count else 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
            if resolved else 'REFERENCE_UNRESOLVED', 'witness_found': bool(count), 'witness_count': count,
            'all_point_diagnostics': rows, 'selected_points_or_eps': False,
            'scope': 'numerical executor input sufficiency at fixed state45/action; no physical-controller or performance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-warmstart-response61-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'control_parameter_updates': 0,
              'dynamics_models_loaded': 0, 'training_qualification_claimed': False,
              'capture_phase': CAPTURE_PHASE, 'action_center': 'clean', 'counts': None, 'completed_branches': 0,
              'completed_preserved60_comparisons': 0,
              'cases': [{'seed': seed, 'episode': seed - 53101, 'status': 'NOT_RUN', 'steps': 0}
                        for seed in TRAIN_SEEDS],
              'points': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN',
                          'arms': {arm: {'status': 'NOT_RUN', 'label_admitted': False,
                                      'branches': [{'delta': delta, 'status': 'NOT_RUN'} for delta in DELTAS]}
                                      for arm in ARMS}} for seed in TRAIN_SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress = {'phase': 'admission', 'seed': None, 'tick': None, 'arm': None, 'branch': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress,
            'counts': None if env is None else env.counters(),
            'completed_branches': report['completed_branches'],
            'completed_preserved60_comparisons': report['completed_preserved60_comparisons']}, allow_nan=False) + '\n')
        events.flush()

    def deadline():
        require(time.process_time() < 298. and time.monotonic() - wall < 419., '61 CPU/wall deadline')

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
        verify61(plan)
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
        old_report = json.loads((ROOT / SOURCE60 / 'report.json').read_text())
        require(old_plan['schema'] == 'd1-noise-center-response60-v1'
                and old_report['status'] == 'COMPLETE' and old_report['plan_sha256'] == sha(ROOT / PLAN60)
                and old_report['completed_branches'] == 240
                and old_report['counts']['policy_transitions'] == 2640, 'complete bound60 evidence required')
        for name in ('tools60/collect_noise60.py', 'tools58/collect_response58.py',
                     str(PLAN53), str(SUMMARY53), str(PLAN58), str(SOURCE58 / 'report.json')):
            require(sha(ROOT / name) == old_plan['source_sha256'][name], '60 lineage changed: ' + name)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == old_report['versions'], 'package versions differ')
        import numpy as np
        import torch

        import mujoco
        from tools33.reference_shape33 import ReferenceShaper
        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        require(torch.__version__ == old_report['torch_runtime_version'], 'torch runtime differs')
        report['torch_runtime_version'] = torch.__version__
        actor = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor)
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        base_recorder = make_step_recorder(audit, progress)

        def recorder(current, action, result):
            base_recorder(current, action, result)
            audit.flush()

        env = collection.make_env(3000, audit_step=recorder)

        def solver_options():
            return {key: int(getattr(env.model.opt, key)) for key in
                    ('solver', 'iterations', 'disableflags', 'enableflags', 'integrator', 'cone')} | {
                    key: float(getattr(env.model.opt, key)) for key in ('tolerance', 'impratio', 'timestep')}

        report['solver_options'] = solver_options()
        integration_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION) & ~int(mujoco.mjtState.mjSTATE_WARMSTART)
        report['integration_spec_excluding_warmstart'] = integration_spec
        spec = collection.helpers(original).SEQUENCES['turn_stop']
        sites = []
        progress['phase'] = 'baseline_replay'
        for case in report['cases']:
            deadline()
            seed, rows = case['seed'], []
            progress.update(seed=seed, tick=None)
            case['status'] = 'INCOMPLETE'
            try:
                for suffix in ('npz', 'json'):
                    name = f'execution_01/collection_train_{seed}.{suffix}'
                    identity = summary['raw_manifest'][name]
                    path = ROOT / 'evidence/model_gradient53' / name
                    require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
                            'original train input differs')
                old_case = json.loads((ROOT / SOURCE53 / f'collection_train_{seed}.json').read_text())
                require(old_case['status'] == 'COMPLETE' and old_case['steps'] == 600
                        and old_case['seed'] == seed and old_case['episode'] == case['episode']
                        and old_case['role'] == 'train' and old_case['task'] == 'turn_stop', 'wrong original train case')
                env.eval_command_sequence = env._validate_eval_sequence(spec)
                env.reset(seed=seed)
                shaper = ReferenceShaper()
                noise_rng = np.random.default_rng(seed + 900000)
                require(env._delay_ticks == 0 and len(env._delay_fifo) == 0, 'zero-delay action seam required')
                for tick in range(600):
                    deadline()
                    progress['tick'] = tick
                    state = control.physical(env)
                    noise = noise_rng.normal(0., .02, 16)
                    clean = control.real_action(env, shaper, actor, 0., noise=None)
                    noisy = np.clip(clean + noise, -1., 1.).astype(np.float32)
                    site = None
                    if tick in TICKS:
                        site = {'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state.copy(),
                                'clean': clean.copy(), 'noisy': noisy.copy(), 'noise': noise.copy(),
                                'snapshot': adapter.capture(env, shaper)}
                        readback = {'capture_phase': CAPTURE_PHASE, 'exact_restore_artifact': False,
                                    'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state,
                                    'clean': clean, 'noisy': noisy, 'noise': noise, 'noise_rng_seed': seed + 900000,
                                    'environment': env.get_initial_state(), 'shaper': vars(shaper), 'counts': env.counters()}
                        write_json(args.output / f'snapshot_readback_{seed}_{tick}.json', collection._plain(readback))
                    event('BASELINE_STEP_BEGIN')
                    result = env.step(noisy)
                    row = collection._record(env, state, noisy, result, case['episode'], tick, final_tick=tick == 599)
                    rows.append(row)
                    case['steps'] = len(rows)
                    if site is not None:
                        site['baseline_arrays'] = collection._arrays([row]) | {'observations': result[0][None].copy()}
                        sites.append(site)
                    event('BASELINE_STEP_COMPLETE')
                    require(not any(row[key] for key in UNSAFE), 'baseline terminated/unsafe')
                current = collection._arrays(rows)
                with np.load(ROOT / SOURCE53 / f'collection_train_{seed}.npz', allow_pickle=False) as old:
                    case.update(exact_physical(current, old, collection.COUNTERS))
                require(case['all_physical_fields_exact'], 'noisy replay did not reproduce53 exactly')
                case['status'] = 'COMPLETE'
            except BaseException as exc:
                case['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                collection.save_npz(args.output / f'baseline_{seed}.npz', collection._arrays(rows))
                write_json(args.output / f'baseline_{seed}.json', case)
        require(env.counters()['policy_transitions'] == 2400
                and [(site['seed'], site['tick']) for site in sites]
                == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], 'baseline/capture work differs')
        report.update(all_four_complete_baselines_exact=True, live_snapshot_count=len(sites),
                      snapshot_data_buffer_bytes_each=int(env.data.narena + env.data.nbuffer))
        event('ALL_BASELINES_ADMITTED')
        direction = np.zeros(16, dtype=np.float64)
        direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
        report['action_direction'] = {'vector': direction.tolist(), 'linf': 1., 'l2': 2., 'yaw_error_multiplier': False}
        progress['phase'] = 'warmstart_branches'
        for index, site in enumerate(sites):
            point = report['points'][index]
            point['status'] = 'INCOMPLETE'
            progress.update(seed=site['seed'], tick=site['tick'])
            zero_targets = {}
            for arm_name in ARMS:
                arm = point['arms'][arm_name]
                arm['status'] = 'INCOMPLETE'
                progress['arm'] = arm_name
                point_rows, zero_arrays = [], None
                for order, delta in enumerate(DELTAS):
                    deadline()
                    progress['branch'] = order
                    branch = {'status': 'INCOMPLETE', 'arm': arm_name, 'action_center': 'clean', 'delta': delta, 'steps': 0,
                              'capture_phase': CAPTURE_PHASE, 'counts_before': env.counters()}
                    record, arrays = None, {}
                    try:
                        adapter.restore(env, shaper, site['snapshot'])
                        require(np.array_equal(control.physical(env), site['state']), 'snapshot physical state changed')
                        action = np.clip(site['clean'].astype(np.float64) + delta * direction, -1., 1.).astype(np.float32)
                        action_before = action.copy()
                        warmstart_before = env.data.qacc_warmstart.copy()
                        other_before = np.empty(mujoco.mj_stateSize(env.model, integration_spec), dtype=np.float64)
                        mujoco.mj_getState(env.model, env.data, other_before, integration_spec)
                        # This is the only intervention: no forward, actor, phase or solver change.
                        if arm_name == 'zeroed':
                            env.data.qacc_warmstart[:] = 0.
                        warmstart_after = env.data.qacc_warmstart.copy()
                        other_after = np.empty_like(other_before)
                        mujoco.mj_getState(env.model, env.data, other_after, integration_spec)
                        branch.update(other_integration_inputs_exact=bool(np.array_equal(other_before, other_after)),
                                      state45_exact=bool(np.array_equal(control.physical(env), site['state'])),
                                      action_exact=bool(np.array_equal(action, action_before)),
                                      warmstart_before=warmstart_before.tolist(),
                                      warmstart_after_intervention=warmstart_after.tolist())
                        arrays = {'warmstart_before': warmstart_before, 'warmstart_after_intervention': warmstart_after}
                        require(branch['other_integration_inputs_exact'] and branch['state45_exact']
                                and branch['action_exact'], 'warmstart intervention changed other inputs')
                        require(np.array_equal(warmstart_after, np.zeros_like(warmstart_before)
                                if arm_name == 'zeroed' else warmstart_before), 'warmstart intervention differs')
                        require(solver_options() == report['solver_options'], 'solver options changed')
                        event('BRANCH_STEP_BEGIN')
                        result = env.step(action)
                        arrays['warmstart_after_step'] = env.data.qacc_warmstart.copy()
                        branch['warmstart_after_step'] = arrays['warmstart_after_step'].tolist()
                        record = collection._record(env, site['state'], action, result, site['episode'], site['tick'], final_tick=False)
                        physical_arrays = collection._arrays([record]) | {'observations': result[0][None].copy()}
                        arrays.update(physical_arrays)
                        branch.update(status='TERMINATED' if any(record[key] for key in UNSAFE) else 'COMPLETE', steps=1,
                                      record=collection._plain(record), counts_after=env.counters(),
                                      issued_equals_applied=bool(np.array_equal(action, record['applied_action'])))
                        if order == 0:
                            zero_arrays = physical_arrays
                            zero_targets[arm_name] = record['target'].copy()
                        if order < 2:
                            branch['center_zero_comparison'] = exact_physical(physical_arrays, zero_arrays, collection.COUNTERS)
                            branch['zero_exact'] = branch['center_zero_comparison']['all_physical_fields_exact']
                        else:
                            branch['actual_action_delta'] = action_delta_ulps(site['clean'], record['applied_action'])
                        if arm_name == 'preserved':
                            old_prefix = ROOT / SOURCE60 / f'branch_{index:02d}_clean_{order:02d}'
                            for suffix in ('.npz', '.json'):
                                name = old_prefix.with_suffix(suffix).name
                                require(sha(old_prefix.with_suffix(suffix)) == old_report['sealed_artifacts_sha256'][name],
                                        'sealed60 clean branch changed')
                            old_branch = json.loads(old_prefix.with_suffix('.json').read_text())
                            require(old_branch['status'] == 'COMPLETE' and old_branch['delta'] == delta
                                    and old_branch['steps'] == 1 and old_branch['capture_phase'] == CAPTURE_PHASE,
                                    'wrong/incomplete60 clean branch')
                            with np.load(old_prefix.with_suffix('.npz'), allow_pickle=False) as old:
                                comparison = exact_physical(physical_arrays, old, collection.COUNTERS)
                            comparison['json_record'] = exact_physical(record, old_branch['record'], collection.COUNTERS)
                            comparison['all_physical_fields_exact'] = comparison['all_physical_fields_exact'] and comparison['json_record']['all_physical_fields_exact']
                            branch['original60_comparison'] = comparison
                        point_rows.append(record)
                    except BaseException as exc:
                        branch.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}', counts_after=env.counters())
                        if record is None:
                            arrays.update(attempted_post_state=control.physical(env), action_center=site['clean'],
                                          warmstart_after_step=env.data.qacc_warmstart.copy())
                        raise
                    finally:
                        prefix = args.output / f'branch_{index:02d}_{arm_name}_{order:02d}'
                        collection.save_npz(prefix.with_suffix('.npz'), arrays)
                        arm['branches'][order] = branch
                        write_json(prefix.with_suffix('.json'), branch)
                        event('BRANCH_SAVED')
                    require(branch['status'] == 'COMPLETE' and branch['issued_equals_applied'], 'branch incomplete/unsafe/delayed')
                    if order < 2:
                        require(branch['zero_exact'], 'arm zero branches differ')
                    if arm_name == 'preserved':
                        require(branch['original60_comparison']['all_physical_fields_exact'], 'preserved warmstart branch differs from60 clean')
                        report['completed_preserved60_comparisons'] += 1
                    report['completed_branches'] += 1
                arrays, diagnostics = state_response(point_rows)
                arm.update(diagnostics)
                arm['all_four_wheels_nonzero'] = all(branch['actual_action_delta']['all_four_wheels_nonzero']
                                                      for branch in arm['branches'][2:])
                arm.update(status='COMPLETE', label_admitted=arm['all_four_wheels_nonzero']
                              and arm['yaw_stability']['stable'] and all(branch['zero_exact'] for branch in arm['branches'][:2]))
                collection.save_npz(args.output / f'response_{index:02d}_{arm_name}.npz', arrays)
                write_json(args.output / f'response_{index:02d}_{arm_name}.json', arm)
            difference = zero_targets['zeroed'] - zero_targets['preserved']
            point['zero_next_state_difference'] = {'zeroed_minus_preserved_state45': difference.tolist(),
                'exactly_equal': bool(np.array_equal(zero_targets['zeroed'], zero_targets['preserved'])),
                'max_absolute': float(np.max(np.abs(difference))), 'yaw28': float(difference[28]),
                'descriptive_only': True, 'threshold': None}
            point['status'] = 'COMPLETE'
        require(report['completed_branches'] == 240 and report['completed_preserved60_comparisons'] == 120
                and env.counters()['policy_transitions'] == 2640 and env.counters()['native_steps'] == 26400
                and env.counters()['pd_updates'] == 13200, 'complete61 work differs')
        report['labels_admitted_by_arm'] = {name: sum(point['arms'][name]['label_admitted'] for point in report['points'])
                                               for name in ARMS}
        report['hypothesis'] = warmstart_hypothesis(report['points'])
        report['solver_options_after'] = solver_options()
        require(report['solver_options_after'] == report['solver_options'], 'solver options changed')
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - every failure/partial remains available; no resume.
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
                    require(report['work_journal']['by_phase'] == {'baseline_replay': 2400, 'warmstart_branches': 240},
                            'physical phase work differs')
            except Exception as exc:  # noqa: BLE001 - keep partial journal and counters.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify61(plan)
            if original is not None:
                verify(original)
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            if 'torch_runtime_version' in report:
                require(sys.modules['torch'].__version__ == report['torch_runtime_version'], 'torch runtime changed')
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - retain original failure too.
            report['verification_after'] = 'FAILED'
            cleanup.append(f'verification: {exc}')
        if cleanup:
            report.update(status='INCOMPLETE', cleanup_errors=cleanup)
        if report['status'] != 'COMPLETE' and 'hypothesis' in report:
            report['hypothesis_before_final_eligibility'] = report['hypothesis']
            report['hypothesis'] = {'status': 'INCOMPLETE', 'witness_found': False}
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        event('FINAL_REPORT')
        events.close()
        try:
            report['sealed_artifacts_sha256'] = {str(path.relative_to(args.output)): sha(path)
                for path in sorted(args.output.rglob('*')) if path.is_file()}
        except Exception as exc:  # noqa: BLE001 - sealing failure invalidates eligibility.
            report.update(status='INCOMPLETE', sealing_failure=f'{type(exc).__name__}: {exc}')
            if 'hypothesis' in report:
                report['hypothesis_before_final_eligibility'] = report['hypothesis']
                report['hypothesis'] = {'status': 'INCOMPLETE', 'witness_found': False}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'report_sha256.json', {'report.json': sha(args.output / 'report.json')})
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'counts', 'completed_branches',
                                                      'completed_preserved60_comparisons', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
