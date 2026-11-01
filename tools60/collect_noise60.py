"""60: paired clean/noisy action centers at the same live training snapshot.

Replay the four original complete53 turn cases exactly, calling the shaper only
once per tick. Each counterfactual restores the same full after-actor snapshot;
only the action center changes. Frozen58 supplies the absolute-action response
diagnostics. No dynamics model is loaded and no parameters are updated. Readback
JSON is descriptive, not a substitute for the live full-MjData snapshot.
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

PLAN58 = Path('docs/experiments58/train_action_response58_preregistered.json')
SOURCE58 = Path('evidence/train_action_response58/collection_01')
CENTERS = ('clean', 'noisy')
DELTAS = (0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1)))
UNSAFE = ('terminated', 'truncated', 'fallen', 'illegal_body_contact')


def required_inputs():
    """Return paths that60 must SHA-bind; original53 identities are checked too."""
    paths = {str(PLAN53), str(SUMMARY53), str(PLAN58), str(ADMISSION55),
             str(SOURCE58 / 'report.json'), 'tools60/collect_noise60.py',
             'tools58/collect_response58.py', 'tools52/adapter52.py', 'tools52/control52.py',
             'tools52/dynamics52.py', 'tools52/run_pilot52.py', 'tools53/collection53.py',
             'tools53/run_pilot53.py', 'tools55/probe_local55.py', 'tools55/run55.py'}
    paths.update(str(SOURCE53 / f'collection_train_{seed}.{suffix}')
                 for seed in TRAIN_SEEDS for suffix in ('npz', 'json'))
    paths.update(str(SOURCE58 / f'branch_{point:02d}_{branch:02d}.{suffix}')
                 for point in range(12) for branch in range(10) for suffix in ('npz', 'json'))
    return paths


def verify60(plan):
    require(plan['schema'] == 'd1-noise-center-response60-v1', 'wrong60 protocol')
    require(plan['train_seeds'] == list(TRAIN_SEEDS) and plan['eps'] == list(EPSILONS)
            and plan['centers'] == list(CENTERS), '60 cases/action centers/eps differ')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['expected_policy_steps'] == 2640 and plan['hard_policy_cap'] == 3000,
            '60 budget changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required60 source/input not SHA-bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '60 source/input changed: ' + name)


def paired_hypothesis(points):
    """Diagnose every point; the primary hypothesis requires all four tick260s."""
    require([(point['seed'], point['tick']) for point in points]
            == [(seed, tick) for seed in TRAIN_SEEDS for tick in TICKS], 'paired point grid differs')
    rows = []
    for point in points:
        centers = point['centers']
        resolved = point['status'] == 'COMPLETE' and all(
            centers[name]['status'] == 'COMPLETE' and centers[name]['label_admitted'] for name in CENTERS)
        reversal = (all(row['central'] > 0 for row in centers['noisy']['yaw28'])
                    and all(row['central'] < 0 for row in centers['clean']['yaw28'])) if resolved else None
        rows.append({'seed': point['seed'], 'tick': point['tick'], 'reference_resolved': resolved,
                     'noisy_positive_clean_negative': reversal,
                     'status': 'SUPPORTED' if reversal else 'REJECTED' if resolved else 'REFERENCE_UNRESOLVED'})
    primary = [row for row in rows if row['tick'] == 260]
    resolved = all(row['reference_resolved'] for row in primary)
    supported = resolved and all(row['noisy_positive_clean_negative'] for row in primary)
    return {'status': 'SUPPORTED' if supported else 'REJECTED' if resolved else 'REFERENCE_UNRESOLVED',
            'supported': supported, 'fixed_primary_tick': 260, 'primary_cases': primary,
            'all_point_diagnostics': rows, 'selected_points_or_eps': False,
            'scope': 'same live state/history; changing current action center only; no control-performance claim'}


def exact_physical(current, reference, counters):
    """Compare every available non-counter array, including observations."""
    import numpy as np

    same_keys = set(current) == set(reference)
    fields = {key: bool(np.array_equal(value, reference[key]))
              for key, value in current.items() if key in reference and key not in counters}
    return {'same_keys': same_keys, 'exact_by_field': fields,
            'all_physical_fields_exact': bool(same_keys and all(fields.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-noise-center-response60-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_updates': 0, 'policy_updates': 0, 'control_parameter_updates': 0,
              'dynamics_models_loaded': 0, 'training_qualification_claimed': False,
              'capture_phase': CAPTURE_PHASE, 'counts': None, 'completed_branches': 0,
              'completed_noisy58_comparisons': 0,
              'cases': [{'seed': seed, 'episode': seed - 53101, 'status': 'NOT_RUN', 'steps': 0}
                        for seed in TRAIN_SEEDS],
              'points': [{'seed': seed, 'tick': tick, 'status': 'NOT_RUN',
                          'centers': {center: {'status': 'NOT_RUN', 'label_admitted': False,
                                      'branches': [{'delta': delta, 'status': 'NOT_RUN'} for delta in DELTAS]}
                                      for center in CENTERS}} for seed in TRAIN_SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    env, audit, plan, original = None, None, None, None
    progress = {'phase': 'admission', 'seed': None, 'tick': None, 'center': None, 'branch': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress,
            'counts': None if env is None else env.counters(),
            'completed_branches': report['completed_branches'],
            'completed_noisy58_comparisons': report['completed_noisy58_comparisons']}, allow_nan=False) + '\n')
        events.flush()

    def deadline():
        require(time.process_time() < 298. and time.monotonic() - wall < 419., '60 CPU/wall deadline')

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
        verify60(plan)
        original = json.loads((ROOT / PLAN53).read_text())
        verify(original)
        summary = json.loads((ROOT / SUMMARY53).read_text())
        require(summary['formal']['status'] == 'COMPLETE' and summary['plan_sha256'] == sha(ROOT / PLAN53),
                'complete bound53 evidence required')
        admission = json.loads((ROOT / ADMISSION55).read_text())
        require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                and admission['counts']['policy_transitions'] == 224, 'complete55 interface admission required')
        old_plan = json.loads((ROOT / PLAN58).read_text())
        old_report = json.loads((ROOT / SOURCE58 / 'report.json').read_text())
        require(old_plan['schema'] == 'd1-train-action-response58-v1'
                and old_report['status'] == 'COMPLETE' and old_report['plan_sha256'] == sha(ROOT / PLAN58)
                and old_report['completed_branches'] == 120
                and old_report['counts']['policy_transitions'] == 2520, 'complete bound58 evidence required')
        for name in ('tools58/collect_response58.py', str(PLAN53), str(SUMMARY53)):
            require(sha(ROOT / name) == old_plan['source_sha256'][name], '58 lineage changed: ' + name)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'] == old_report['versions'], 'package versions differ')
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
        progress['phase'] = 'paired_center_branches'
        for index, site in enumerate(sites):
            point = report['points'][index]
            point['status'] = 'INCOMPLETE'
            progress.update(seed=site['seed'], tick=site['tick'])
            for center_name in CENTERS:
                center = point['centers'][center_name]
                center['status'] = 'INCOMPLETE'
                progress['center'] = center_name
                point_rows, zero_arrays = [], None
                for order, delta in enumerate(DELTAS):
                    deadline()
                    progress['branch'] = order
                    branch = {'status': 'INCOMPLETE', 'center': center_name, 'delta': delta, 'steps': 0,
                              'capture_phase': CAPTURE_PHASE, 'counts_before': env.counters()}
                    record, arrays = None, {}
                    try:
                        adapter.restore(env, shaper, site['snapshot'])
                        require(np.array_equal(control.physical(env), site['state']), 'snapshot physical state changed')
                        action = np.clip(site[center_name].astype(np.float64) + delta * direction, -1., 1.).astype(np.float32)
                        event('BRANCH_STEP_BEGIN')
                        result = env.step(action)
                        record = collection._record(env, site['state'], action, result, site['episode'], site['tick'], final_tick=False)
                        arrays = collection._arrays([record]) | {'observations': result[0][None].copy()}
                        branch.update(status='TERMINATED' if any(record[key] for key in UNSAFE) else 'COMPLETE', steps=1,
                                      record=collection._plain(record), counts_after=env.counters(),
                                      issued_equals_applied=bool(np.array_equal(action, record['applied_action'])))
                        if order == 0:
                            zero_arrays = arrays
                        if order < 2:
                            branch['center_zero_comparison'] = exact_physical(arrays, zero_arrays, collection.COUNTERS)
                            branch['zero_exact'] = branch['center_zero_comparison']['all_physical_fields_exact']
                            if center_name == 'noisy':
                                branch['original_baseline_comparison'] = exact_physical(arrays, site['baseline_arrays'], collection.COUNTERS)
                                branch['zero_exact'] = branch['zero_exact'] and branch['original_baseline_comparison']['all_physical_fields_exact']
                        else:
                            branch['actual_action_delta'] = action_delta_ulps(site[center_name], record['applied_action'])
                        if center_name == 'noisy':
                            old_prefix = ROOT / SOURCE58 / f'branch_{index:02d}_{order:02d}'
                            old_branch = json.loads(old_prefix.with_suffix('.json').read_text())
                            require(old_branch['status'] == 'COMPLETE' and old_branch['delta'] == delta
                                    and old_branch['steps'] == 1 and old_branch['capture_phase'] == CAPTURE_PHASE,
                                    'wrong/incomplete58 branch')
                            with np.load(old_prefix.with_suffix('.npz'), allow_pickle=False) as old:
                                comparison = exact_physical(arrays, old, collection.COUNTERS)
                            comparison['json_record'] = exact_physical(record, old_branch['record'], collection.COUNTERS)
                            comparison['all_physical_fields_exact'] = comparison['all_physical_fields_exact'] and comparison['json_record']['all_physical_fields_exact']
                            branch['original58_comparison'] = comparison
                        point_rows.append(record)
                    except BaseException as exc:
                        branch.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}', counts_after=env.counters())
                        if record is None:
                            arrays = {'attempted_post_state': control.physical(env), 'action_center': site[center_name]}
                        raise
                    finally:
                        prefix = args.output / f'branch_{index:02d}_{center_name}_{order:02d}'
                        collection.save_npz(prefix.with_suffix('.npz'), arrays)
                        center['branches'][order] = branch
                        write_json(prefix.with_suffix('.json'), branch)
                        event('BRANCH_SAVED')
                    require(branch['status'] == 'COMPLETE' and branch['issued_equals_applied'], 'branch incomplete/unsafe/delayed')
                    if order < 2:
                        require(branch['zero_exact'], 'center zero branches differ')
                    if center_name == 'noisy':
                        require(branch['original58_comparison']['all_physical_fields_exact'], 'noisy branch differs from58')
                        report['completed_noisy58_comparisons'] += 1
                    report['completed_branches'] += 1
                arrays, diagnostics = state_response(point_rows)
                center.update(diagnostics)
                center['all_four_wheels_nonzero'] = all(branch['actual_action_delta']['all_four_wheels_nonzero']
                                                      for branch in center['branches'][2:])
                center.update(status='COMPLETE', label_admitted=center['all_four_wheels_nonzero']
                              and center['yaw_stability']['stable'] and all(branch['zero_exact'] for branch in center['branches'][:2]))
                collection.save_npz(args.output / f'response_{index:02d}_{center_name}.npz', arrays)
                write_json(args.output / f'response_{index:02d}_{center_name}.json', center)
            point['status'] = 'COMPLETE'
        require(report['completed_branches'] == 240 and report['completed_noisy58_comparisons'] == 120
                and env.counters()['policy_transitions'] == 2640 and env.counters()['native_steps'] == 26400
                and env.counters()['pd_updates'] == 13200, 'complete60 work differs')
        report['labels_admitted_by_center'] = {name: sum(point['centers'][name]['label_admitted'] for point in report['points'])
                                               for name in CENTERS}
        report['hypothesis'] = paired_hypothesis(report['points'])
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
                    require(report['work_journal']['by_phase'] == {'baseline_replay': 2400, 'paired_center_branches': 240},
                            'physical phase work differs')
            except Exception as exc:  # noqa: BLE001 - keep partial journal and counters.
                cleanup.append(f'audit: {exc}')
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], 'plan changed')
                verify60(plan)
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
            report['hypothesis'] = {'status': 'INCOMPLETE', 'supported': False}
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
                report['hypothesis'] = {'status': 'INCOMPLETE', 'supported': False}
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'report_sha256.json', {'report.json': sha(args.output / 'report.json')})
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'counts', 'completed_branches',
                                                      'completed_noisy58_comparisons', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
