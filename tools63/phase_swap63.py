"""63: fixed neighboring-seed wheel-phase transplants at clean62 live states.

Only four wheel qpos entries change. State45 intentionally changes, while the
Dynamics52 masked input, actual action, warmstart, other integration inputs and
first algebraic PD torque remain identical. A stable response reversal witnesses
an insufficient phase-mask equivalence class, not non-Markov state45. No model
is instantiated and no parameters or old conclusions are changed.
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

PLAN62 = Path('docs/experiments62/clean_history62_preregistered.json')
SOURCE62 = Path('evidence/clean_history62/execution_01')
ARMS = ('original', 'transplanted')
PARTNERS = {53103: 53106, 53106: 53103, 53109: 53112, 53112: 53109}
WHEEL_QPOS = (10, 14, 18, 22)
DELTAS = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
UNSAFE = ('terminated', 'truncated', 'fallen', 'illegal_body_contact')


def phase_partner(seed, tick):
    """Return the preregistered donor identity; never select a response by sign."""
    require(seed in PARTNERS and tick in TICKS, 'unregistered63 phase donor request')
    return PARTNERS[seed], tick


def required_inputs():
    """63 dependencies, with original53 actor/environment identities checked too."""
    paths = {str(PLAN53), str(SUMMARY53), str(PLAN58), str(PLAN62), str(ADMISSION55),
             str(SOURCE58 / 'report.json'), str(SOURCE62 / 'report.json'),
             'tools63/phase_swap63.py', 'tools62/collect_clean62.py', 'tools60/collect_noise60.py',
             'tools58/collect_response58.py', 'tools52/adapter52.py', 'tools52/control52.py',
             'tools52/dynamics52.py', 'tools52/run_pilot52.py', 'tools53/collection53.py',
             'tools53/run_pilot53.py', 'tools55/probe_local55.py', 'tools55/run55.py',
             'training/d1_rl/interface.py'}
    paths.update(str(SOURCE53 / f'collection_train_{seed}.{suffix}')
                 for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    paths.update(str(SOURCE62 / f'baseline_{seed}.{suffix}')
                 for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    paths.update(str(SOURCE62 / f'branch_{point:02d}_{branch:02d}.{suffix}')
                 for point in range(12) for branch in range(10) for suffix in ('npz', 'json'))
    return paths


def verify63(plan):
    require(plan['schema'] == 'd1-phase-swap63-v1', 'wrong63 protocol')
    require(plan['train_seeds'] == list(TRAIN_SEEDS) and plan['eps'] == list(EPSILONS)
            and plan['arms'] == list(ARMS) and plan['action_center'] == 'clean'
            and plan['partners'] == {str(seed): partner for seed, partner in PARTNERS.items()}
            and plan['wheel_qpos_indices'] == list(WHEEL_QPOS), '63 intervention grid changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['expected_policy_steps'] == 2640 and plan['hard_policy_cap'] == 3000,
            '63 budget changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required63 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '63 source/input changed: ' + name)


def phase_hypothesis(points):
    """Require240 complete branches, then report all registered mask witnesses.

Complete but unresolved references cannot erase a witness elsewhere. Every
corresponding epsilon must reverse sign; a mean reversal is insufficient. No
witness in this intervention does not establish that ignoring phase is valid.
"""
    require([(point.get('seed'), point.get('tick')) for point in points]
            == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], '63 point grid differs')
    rows = []
    for point in points:
        require(point.get('status') == 'COMPLETE' and set(point.get('arms', {})) == set(ARMS),
                'incomplete63 point/arm grid')
        arms = point['arms']
        for arm in arms.values():
            require(arm.get('status') == 'COMPLETE' and len(arm.get('branches', [])) == 10
                    and all(branch.get('status') == 'COMPLETE' for branch in arm['branches']),
                    'incomplete63 physical branch')
            require([row.get('epsilon') for row in arm.get('yaw28', [])] == list(EPSILONS)
                    and all(math.isfinite(row['central']) for row in arm['yaw28']), 'invalid63 FD grid')
        resolved = all(arms[name].get('label_admitted') is True for name in ARMS)
        opposite = all(left['central'] * right['central'] < 0 for left, right in
                       zip(arms['original']['yaw28'], arms['transplanted']['yaw28'], strict=True)) if resolved else None
        witness = resolved and opposite
        rows.append({'seed': point['seed'], 'tick': point['tick'],
                     'partner_seed': phase_partner(point['seed'], point['tick'])[0],
                     'reference_resolved': resolved, 'opposite_sign_all_eps': opposite, 'witness': witness,
                     'central_fd_ranges': {name: {'min': min(row['central'] for row in arms[name]['yaw28']),
                                                 'max': max(row['central'] for row in arms[name]['yaw28'])}
                                           for name in ARMS},
                     'status': 'WITNESS_FOUND' if witness else 'NO_WITNESS_AT_POINT' if resolved else 'REFERENCE_UNRESOLVED'})
    count = sum(row['witness'] for row in rows)
    resolved = all(row['reference_resolved'] for row in rows)
    return {'status': 'WITNESS_FOUND' if count else 'NO_WITNESS_IN_REGISTERED_INTERVENTION'
            if resolved else 'REFERENCE_UNRESOLVED', 'witness_found': bool(count), 'witness_count': count,
            'all_point_diagnostics': rows, 'selected_points_or_eps': False,
            'scope': 'phase-mask input equivalence only; not state45 non-Markov, mask validity, or performance',
            'rejudges62': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-phase-swap63-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'control_parameter_updates': 0,
              'dynamics_models_loaded': 0, 'training_qualification_claimed': False,
              'capture_phase': CAPTURE_PHASE, 'action_center': 'clean', 'counts': None, 'completed_branches': 0,
              'completed_original62_comparisons': 0,
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
            'completed_original62_comparisons': report['completed_original62_comparisons']}, allow_nan=False) + '\n')
        events.flush()

    def deadline():
        require(time.process_time() < 298. and time.monotonic() - wall < 419., '63 CPU/wall deadline')

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
        verify63(plan)
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
        old_plan = json.loads((ROOT / PLAN62).read_text())
        old_report = json.loads((ROOT / SOURCE62 / 'report.json').read_text())
        require(old_plan['schema'] == 'd1-clean-history-response62-v1'
                and old_report['status'] == 'COMPLETE' and old_report['plan_sha256'] == sha(ROOT / PLAN62)
                and old_report['completed_branches'] == 120
                and old_report['counts']['policy_transitions'] == 2520, 'complete bound62 evidence required')
        for name in ('tools62/collect_clean62.py', 'tools60/collect_noise60.py', 'tools58/collect_response58.py',
                     str(PLAN53), str(SUMMARY53), str(PLAN58), str(SOURCE58 / 'report.json')):
            require(sha(ROOT / name) == old_plan['source_sha256'][name], '62 lineage changed: ' + name)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == old_report['versions'], 'package versions differ')
        import numpy as np
        import torch
        from d1_rl.interface import action_to_torque

        import mujoco
        from tools33.reference_shape33 import ReferenceShaper
        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import run_pilot52 as shared
        from tools52.dynamics52 import input_features
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
        integration_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION) & ~int(mujoco.mjtState.mjSTATE_QPOS)
        report['integration_spec_excluding_qpos'] = integration_spec
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
                for suffix in ('.npz', '.json'):
                    name = f'baseline_{seed}{suffix}'
                    require(sha(ROOT / SOURCE62 / name) == old_report['sealed_artifacts_sha256'][name],
                            'sealed62 baseline changed')
                old_clean_case = json.loads((ROOT / SOURCE62 / f'baseline_{seed}.json').read_text())
                require(old_clean_case['status'] == 'COMPLETE' and old_clean_case['steps'] == 600
                        and old_clean_case['seed'] == seed, 'complete62 clean baseline required')
                env.reset(seed=seed)
                initial = control.physical(env)
                with np.load(ROOT / SOURCE53 / f'collection_train_{seed}.npz', allow_pickle=False) as old:
                    case['initial_state53_exact'] = bool(np.array_equal(initial, old['s'][0]))
                require(case['initial_state53_exact'], 'reset differs from original53 state45')
                shaper = ReferenceShaper()
                require(env._delay_ticks == 0 and len(env._delay_fifo) == 0, 'zero-delay action seam required')
                for tick in range(600):
                    deadline()
                    progress['tick'] = tick
                    state = control.physical(env)
                    clean = control.real_action(env, shaper, actor, 0., noise=None)
                    site = None
                    if tick in TICKS:
                        site = {'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state.copy(),
                                'clean': clean.copy(),
                                'snapshot': adapter.capture(env, shaper)}
                        readback = {'capture_phase': CAPTURE_PHASE, 'exact_restore_artifact': False,
                                    'seed': seed, 'tick': tick, 'episode': case['episode'], 'state': state,
                                    'clean': clean, 'exploration_noise_sigma': 0.,
                                    'environment': env.get_initial_state(), 'shaper': vars(shaper), 'counts': env.counters()}
                        write_json(args.output / f'snapshot_readback_{seed}_{tick}.json', collection._plain(readback))
                    event('BASELINE_STEP_BEGIN')
                    result = env.step(clean)
                    row = collection._record(env, state, clean, result, case['episode'], tick, final_tick=tick == 599)
                    rows.append(row)
                    case['steps'] = len(rows)
                    if site is not None:
                        site['baseline_arrays'] = collection._arrays([row]) | {'observations': result[0][None].copy()}
                        sites.append(site)
                    event('BASELINE_STEP_COMPLETE')
                    require(not any(row[key] for key in UNSAFE), 'baseline terminated/unsafe')
                current = collection._arrays(rows)
                with np.load(ROOT / SOURCE62 / f'baseline_{seed}.npz', allow_pickle=False) as old:
                    case.update(exact_physical(current, old, collection.COUNTERS))
                require(case['all_physical_fields_exact'], 'clean replay did not reproduce62 exactly')
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
        report.update(all_four_complete_baselines62_exact=True, live_snapshot_count=len(sites),
                      snapshot_data_buffer_bytes_each=int(env.data.narena + env.data.nbuffer))
        event('ALL_BASELINES_ADMITTED')
        direction = np.zeros(16, dtype=np.float64)
        direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
        report['action_direction'] = {'vector': direction.tolist(), 'linf': 1., 'l2': 2., 'yaw_error_multiplier': False}
        sites_by_key = {(site['seed'], site['tick']): site for site in sites}
        qpos_others = [index for index in range(23) if index not in WHEEL_QPOS]
        progress['phase'] = 'phase_branches'
        for index, site in enumerate(sites):
            point = report['points'][index]
            point['status'] = 'INCOMPLETE'
            progress.update(seed=site['seed'], tick=site['tick'])
            donor_key = phase_partner(site['seed'], site['tick'])
            donor_phases = sites_by_key[donor_key]['state'][list(WHEEL_QPOS)].copy()
            point['phase_partner'] = {'seed': donor_key[0], 'tick': donor_key[1],
                                      'provenance': str(SOURCE62 / f'baseline_{donor_key[0]}.npz'),
                                      'source_replayed_exactly': True, 'wheel_qpos_indices': list(WHEEL_QPOS),
                                      'original_phases': site['state'][list(WHEEL_QPOS)].tolist(),
                                      'donor_phases': donor_phases.tolist()}
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
                        before_state = control.physical(env)
                        warmstart_before = env.data.qacc_warmstart.copy()
                        other_before = np.empty(mujoco.mj_stateSize(env.model, integration_spec), dtype=np.float64)
                        mujoco.mj_getState(env.model, env.data, other_before, integration_spec)
                        features_before = input_features(before_state[None], action[None])
                        torque_before, saturation_before = action_to_torque(
                            action, before_state[7:23], before_state[29:45], scale=env.action_scales)
                        pre_contacts = {'ncon': int(env.data.ncon), 'nefc': int(env.data.nefc)}
                        # Only these four qpos values change. No forward, actor or solver-option update.
                        if arm_name == 'transplanted':
                            env.data.qpos[list(WHEEL_QPOS)] = donor_phases
                        intervened_state = control.physical(env)
                        warmstart_after = env.data.qacc_warmstart.copy()
                        other_after = np.empty_like(other_before)
                        mujoco.mj_getState(env.model, env.data, other_after, integration_spec)
                        features_after = input_features(intervened_state[None], action[None])
                        torque_after, saturation_after = action_to_torque(
                            action, intervened_state[7:23], intervened_state[29:45], scale=env.action_scales)
                        branch.update(other_integration_inputs_exact=bool(np.array_equal(other_before, other_after)),
                                      other_qpos_exact=bool(np.array_equal(before_state[qpos_others], intervened_state[qpos_others])),
                                      warmstart_exact=bool(np.array_equal(warmstart_before, warmstart_after)),
                                      action_exact=bool(np.array_equal(action, action_before)),
                                      masked_input_exact=bool(np.array_equal(features_before, features_after)),
                                      first_pd_torque_exact=bool(np.array_equal(torque_before, torque_after)),
                                      first_pd_saturation_exact=bool(np.array_equal(saturation_before, saturation_after)),
                                      phase_partner=point['phase_partner'],
                                      state45_before=before_state.tolist(), state45_after_intervention=intervened_state.tolist(),
                                      contact_cache_before_intervention=pre_contacts,
                                      contact_cache_after_intervention_recomputed=False,
                                      warmstart_before=warmstart_before.tolist(),
                                      warmstart_after_intervention=warmstart_after.tolist())
                        arrays = {'state45_before_intervention': before_state, 'state45_after_intervention': intervened_state,
                                  'phase_before': before_state[list(WHEEL_QPOS)],
                                  'phase_after': intervened_state[list(WHEEL_QPOS)], 'donor_phases': donor_phases,
                                  'masked_input_before': features_before, 'masked_input_after': features_after,
                                  'first_pd_torque_before': torque_before, 'first_pd_torque_after': torque_after,
                                  'first_pd_saturation_before': saturation_before, 'first_pd_saturation_after': saturation_after,
                                  'warmstart_before': warmstart_before, 'warmstart_after_intervention': warmstart_after}
                        require(all(branch[key] for key in ('other_integration_inputs_exact', 'other_qpos_exact',
                                'warmstart_exact', 'action_exact', 'masked_input_exact',
                                'first_pd_torque_exact', 'first_pd_saturation_exact')),
                                'phase intervention violated input-equivalence invariant')
                        require(np.array_equal(intervened_state[list(WHEEL_QPOS)], donor_phases
                                if arm_name == 'transplanted' else before_state[list(WHEEL_QPOS)]), 'phase intervention differs')
                        require(solver_options() == report['solver_options'], 'solver options changed')
                        event('BRANCH_STEP_BEGIN')
                        result = env.step(action)
                        arrays['warmstart_after_step'] = env.data.qacc_warmstart.copy()
                        branch['warmstart_after_step'] = arrays['warmstart_after_step'].tolist()
                        record = collection._record(env, intervened_state, action, result, site['episode'], site['tick'], final_tick=False)
                        branch['post_step_contacts'] = {'ncon': int(env.data.ncon), 'nefc': int(env.data.nefc),
                            'sampling': 'after the normal env.step observation refresh; descriptive only',
                            'contacts': [{'geom1': int(contact.geom1), 'geom2': int(contact.geom2), 'dist': float(contact.dist)}
                                         for contact in env.data.contact[:env.data.ncon]]}
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
                        if arm_name == 'original':
                            old_prefix = ROOT / SOURCE62 / f'branch_{index:02d}_{order:02d}'
                            for suffix in ('.npz', '.json'):
                                name = old_prefix.with_suffix(suffix).name
                                require(sha(old_prefix.with_suffix(suffix)) == old_report['sealed_artifacts_sha256'][name],
                                        'sealed62 branch changed')
                            old_branch = json.loads(old_prefix.with_suffix('.json').read_text())
                            require(old_branch['status'] == 'COMPLETE' and old_branch['delta'] == delta
                                    and old_branch['steps'] == 1 and old_branch['capture_phase'] == CAPTURE_PHASE,
                                    'wrong/incomplete62 branch')
                            with np.load(old_prefix.with_suffix('.npz'), allow_pickle=False) as old:
                                comparison = exact_physical(physical_arrays, old, collection.COUNTERS)
                            comparison['json_record'] = exact_physical(record, old_branch['record'], collection.COUNTERS)
                            comparison['all_physical_fields_exact'] = comparison['all_physical_fields_exact'] and comparison['json_record']['all_physical_fields_exact']
                            branch['original62_comparison'] = comparison
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
                    if arm_name == 'original':
                        require(branch['original62_comparison']['all_physical_fields_exact'], 'original phase branch differs from62')
                        report['completed_original62_comparisons'] += 1
                    report['completed_branches'] += 1
                arrays, diagnostics = state_response(point_rows)
                arm.update(diagnostics)
                arm['all_four_wheels_nonzero'] = all(branch['actual_action_delta']['all_four_wheels_nonzero']
                                                      for branch in arm['branches'][2:])
                arm.update(status='COMPLETE', label_admitted=arm['all_four_wheels_nonzero']
                              and arm['yaw_stability']['stable'] and all(branch['zero_exact'] for branch in arm['branches'][:2]))
                collection.save_npz(args.output / f'response_{index:02d}_{arm_name}.npz', arrays)
                write_json(args.output / f'response_{index:02d}_{arm_name}.json', arm)
            difference = zero_targets['transplanted'] - zero_targets['original']
            point['zero_next_state_difference'] = {'transplanted_minus_original_state45': difference.tolist(),
                'exactly_equal': bool(np.array_equal(zero_targets['transplanted'], zero_targets['original'])),
                'max_absolute': float(np.max(np.abs(difference))), 'yaw28': float(difference[28]),
                'descriptive_only': True, 'threshold': None}
            point['status'] = 'COMPLETE'
        require(report['completed_branches'] == 240 and report['completed_original62_comparisons'] == 120
                and env.counters()['policy_transitions'] == 2640 and env.counters()['native_steps'] == 26400
                and env.counters()['pd_updates'] == 13200, 'complete63 work differs')
        report['labels_admitted_by_arm'] = {name: sum(point['arms'][name]['label_admitted'] for point in report['points'])
                                               for name in ARMS}
        report['hypothesis'] = phase_hypothesis(report['points'])
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
                    require(report['work_journal']['by_phase'] == {'baseline_replay': 2400, 'phase_branches': 240},
                            'physical phase work differs')
            except Exception as exc:  # noqa: BLE001 - keep partial journal and counters.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify63(plan)
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
                                                      'completed_original62_comparisons', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
