"""Fresh validation of observed-trajectory gradients, with fixed models and policy."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(folder))
from tools53.run_pilot53 import (
    choose_proposal,
    make_step_recorder,
    require,
    sha,
    verify,
    write_json,
)
from tools54.reproduce54 import load_points


def verify55(plan):
    require(plan['schema'] == 'd1-observed-gradient55-v1', 'wrong55 protocol')
    require(plan['model_seeds'] == [53001, 53002, 53003], 'model seeds changed')
    require(plan['probe_seeds'] == [55301, 55302, 55303], 'probe seeds changed')
    require(plan['evaluation_seeds'] == [55401, 55402, 55403], 'evaluation seeds changed')
    for path, digest in plan['source_sha256'].items():
        require(sha(ROOT / path) == digest, '55 source changed: ' + path)
    original = json.loads((ROOT / 'docs/experiments53/model_gradient53_preregistered.json').read_text())
    verify(original)
    return original


def seal_predictions(proposal, rows):
    """Probe predictions are attached after selection; they never select theta."""
    require(len(rows) == 3, 'three models required')
    return {**proposal, 'test_gradients': [row['test_gradients'] for row in rows],
            'probe_perturbation_results_seen': False}


def audit_work(path):
    phases, total = {}, 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            require(row['counts']['policy_transitions'] == total
                    and row['counts']['native_steps'] == 10 * total
                    and row['counts']['pd_updates'] == 5 * total, 'nonmonotonic physical work journal')
            phases[row['phase']] = phases.get(row['phase'], 0) + 1
    return {'policy_steps': total, 'by_phase': phases}


def collect_probes(env, actor, output, plan, original):
    import numpy as np

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52 as control
    from tools53 import collection53 as collection
    spec = collection.helpers(original).SEQUENCES['turn_stop']
    points, baselines = [], []
    report = {'status': 'INCOMPLETE', 'failure': None,
              'cases': [{'seed': s, 'status': 'NOT_RUN', 'steps': 0} for s in plan['probe_seeds']]}
    write_json(output / 'collection_initial.json', report)
    with (output / 'collection_events.jsonl').open('x') as events:
        try:
            for index, case in enumerate(report['cases']):
                rows, local_points = [], []
                case['status'] = 'INCOMPLETE'
                try:
                    env.eval_command_sequence = env._validate_eval_sequence(spec)
                    env.reset(seed=case['seed'])
                    shaper = ReferenceShaper()
                    for tick in range(600):
                        if tick in (110, 260, 410):
                            point = control.point(env, shaper, spec, case['seed'])
                            local_points.append(point)
                            collection.save_points(output / f'snapshot_{case["seed"]}_{tick}.pkl', [point], env)
                        state = control.physical(env)
                        action = control.real_action(env, shaper, actor, 0.)
                        collection._event(events, {'event': 'STEP_BEGIN', 'seed': case['seed'],
                            'tick': tick, 'counts': env.counters(), 'state': state, 'action': action})
                        result = env.step(action)
                        row = collection._record(env, state, action, result, index, tick, final_tick=tick == 599)
                        rows.append(row)
                        case['steps'] = len(rows)
                        collection._event(events, {'event': 'STEP_COMPLETE', 'seed': case['seed'], **row})
                        require(not any(row[k] for k in ('terminated', 'truncated', 'fallen', 'illegal_body_contact')),
                                'fresh baseline terminated or contacted illegally')
                    require(len(rows) == 600 and len(local_points) == 3, 'incomplete baseline')
                    data = collection._arrays(rows)
                    for point in local_points:
                        tick = point['tick']
                        require(np.array_equal(point['state'], data['s'][tick]), 'snapshot and baseline start differ')
                        points.append(point)
                        baselines.append(data['target'][tick:tick + 32].copy())
                    case['status'] = 'COMPLETE'
                finally:
                    collection.save_npz(output / f'collection_{case["seed"]}.npz', collection._arrays(rows))
                    write_json(output / f'collection_{case["seed"]}.json', case)
            collection.save_points(output / 'probe_snapshots.pkl', points, env)
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['failure'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            write_json(output / 'collection.json', report)
    return points, baselines, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--admission', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    report = {'schema': 'd1-observed-gradient55-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'stage': 'preflight' if args.preflight else 'experiment', 'plan_sha256': sha(args.plan),
              'model_updates': 0, 'policy_network_updates': 0, 'control_parameter_updates': 0,
              'counts': None, 'candidate_accepted': False, 'observed_direction_validated': False,
              'collection': {'status': 'NOT_RUN'}, 'physical_probes': {'status': 'NOT_RUN'},
              'paired_control': {'status': 'NOT_RUN'}, 'seeds': [], 'free_seeds': []}
    write_json(args.output / 'initial_incomplete.json', report)
    started, cpu = time.monotonic(), time.process_time()
    env, audit = None, None
    progress = {'phase': 'admission'}

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'bounded single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'thread/GPU limits differ')
        original = verify55(plan)
        report['versions'] = {k: importlib.metadata.version(k) for k in original['versions']}
        require(report['versions'] == original['versions'], 'versions changed')
        import numpy as np
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import dynamics52 as dynamics
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        from tools54 import numerics54 as numeric
        from tools55.observed55 import predict_gradients
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        actor32 = adapter.load_actor(ROOT / original['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor32)
        if not args.preflight:
            admission = json.loads(args.admission.read_text())
            require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                    and admission['plan_sha256'] == report['plan_sha256']
                    and admission['counts']['policy_transitions'] == 224, 'unbound/failed admission')
            report['admission_sha256'] = sha(args.admission)
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        env = collection.make_env(256 if args.preflight else 15000,
                                 audit_step=make_step_recorder(audit, progress))
        if args.preflight:
            report['interface_admission'] = collection.preflight(env, actor32, args.output, original)
            require(env.counters()['policy_transitions'] == 224, 'preflight count differs')
        else:
            progress['phase'] = 'development_predictions'
            actor64 = copy.deepcopy(actor32).double()
            summary = json.loads((ROOT / 'reports/model_gradient53_results_20261101.json').read_text())
            development = load_points('development', summary['raw_manifest'])
            dev_baselines = []
            for point in development:
                relative = f'execution_01/collection_development_{point["seed"]}.npz'
                path = ROOT / 'evidence/model_gradient53' / relative
                require(sha(path) == summary['raw_manifest'][relative]['sha256'], 'development data changed')
                with np.load(path, allow_pickle=False) as data:
                    require(np.array_equal(point['state'], data['s'][point['tick']]), 'development state differs')
                    dev_baselines.append(data['target'][point['tick']:point['tick'] + 32].copy())
            models = []
            for seed in plan['model_seeds']:
                relative = f'execution_01/model_{seed}/final_state_dict.pt'
                path = ROOT / 'evidence/model_gradient53' / relative
                require(sha(path) == summary['raw_manifest'][relative]['sha256'], 'model weight changed')
                model = dynamics.Dynamics52()
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                model = model.double().eval().requires_grad_(False)
                models.append(model)
                free = numeric.gradients(model, actor64, development)
                observed = [predict_gradients(model, actor64, point, target, extra_eps=True)
                            for point, target in zip(development, dev_baselines, strict=True)]
                report['free_seeds'].append({'seed': seed, 'status': 'COMPLETE', 'development_gradients': free, 'test_gradients': []})
                report['seeds'].append({'seed': seed, 'status': 'COMPLETE', 'development_gradients': observed, 'test_gradients': []})
                write_json(args.output / f'development_{seed}.json', {'free': free, 'observed': observed})
            proposal = choose_proposal(report['seeds'])
            free_proposal = choose_proposal(report['free_seeds'])
            write_json(args.output / 'proposal_before_fresh_physics.json', {'observed': proposal, 'free': free_proposal})
            report['proposal_sha256'] = sha(args.output / 'proposal_before_fresh_physics.json')
            report['theta'] = proposal['theta']
            report['control_parameter_updates'] = int(proposal['theta'] is not None)
            progress['phase'] = 'fresh_baseline_collection'
            report['collection'] = {'status': 'INCOMPLETE'}
            points, baselines, report['collection'] = collect_probes(env, actor32, args.output, plan, original)
            progress['phase'] = 'fresh_gradient_predictions'
            for model, row, free_row in zip(models, report['seeds'], report['free_seeds'], strict=True):
                row['probe_status'] = free_row['probe_status'] = 'INCOMPLETE'
                row['test_gradients'] = [predict_gradients(model, actor64, point, target, extra_eps=True)
                                         for point, target in zip(points, baselines, strict=True)]
                free_row['test_gradients'] = numeric.gradients(model, actor64, points)
                row['probe_status'] = free_row['probe_status'] = 'COMPLETE'
                write_json(args.output / f'predictions_{row["seed"]}.json', {'observed': row, 'free': free_row})
            sealed, free_sealed = seal_predictions(proposal, report['seeds']), seal_predictions(free_proposal, report['free_seeds'])
            write_json(args.output / 'predictions_before_probe_branches.json', {'observed': sealed, 'free': free_sealed})
            report['predictions_sha256'] = sha(args.output / 'predictions_before_probe_branches.json')
            progress['phase'] = 'physical_probes'
            report['physical_probes'] = {'status': 'INCOMPLETE'}
            probes = shared.physical_probes(env, actor32, points, args.output)
            report['physical_probes'] = probes
            for index, targets in enumerate(baselines):
                with np.load(args.output / f'probe_{index:02d}.npz', allow_pickle=False) as branches:
                    require(np.array_equal(targets, branches['states_0']), 'fresh zero branch differs from collected baseline')
            report['observed_direction_gate'] = control.compare_directions(sealed, probes)
            report['free_direction_gate'] = control.compare_directions(free_sealed, probes)
            report['observed_direction_validated'] = (report['observed_direction_gate']['passed']
                and all(m['matches'] == 9 for m in report['observed_direction_gate']['models'])
                and all(p['identifiable'] for p in probes))
            if report['observed_direction_validated']:
                progress['phase'] = 'paired_control'
                folder = args.output / 'paired_control'
                folder.mkdir()
                report['paired_control'] = {'status': 'INCOMPLETE'}
                report['paired_control'] = shared.paired_control(env, actor32, proposal['theta'], folder,
                                                                plan['evaluation_seeds'], original)
                report['candidate_accepted'] = report['paired_control']['passed']
            else:
                report['paired_control']['reason'] = 'fresh all-model 9/9 direction requirement not met'
            require(sha(args.output / 'proposal_before_fresh_physics.json') == report['proposal_sha256']
                    and sha(args.output / 'predictions_before_probe_branches.json') == report['predictions_sha256'],
                    'selection/prediction seal changed')
        verify55(plan)
        report['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - preserve partial work; never retry or resume it.
        report.update(failure=f'{type(exc).__name__}: {exc}', failure_phase=progress['phase'],
                      candidate_accepted=False, observed_direction_validated=False)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        cleanup_errors = []
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - retain cleanup failure.
                cleanup_errors.append(f'env.close: {type(exc).__name__}: {exc}')
        if audit is not None:
            try:
                audit.close()
            except Exception as exc:  # noqa: BLE001 - retain log closure failure.
                cleanup_errors.append(f'audit.close: {type(exc).__name__}: {exc}')
        if (args.output / 'collection.json').exists():
            try:
                report['collection'] = json.loads((args.output / 'collection.json').read_text())
            except (OSError, ValueError) as exc:
                cleanup_errors.append(f'collection receipt: {type(exc).__name__}: {exc}')
        if audit is not None:
            try:
                report['work_journal'] = audit_work(args.output / 'all_physics_steps.jsonl')
                require(report['work_journal']['policy_steps'] == report['counts']['policy_transitions'],
                        'physical work journal differs from actual counters')
                if report['status'] == 'COMPLETE' and not args.preflight:
                    planned = {'fresh_baseline_collection': sum(c['steps'] for c in report['collection']['cases']),
                               'physical_probes': sum(b['steps'] for p in report['physical_probes'] for b in p['branches'])}
                    if 'pairs' in report['paired_control']:
                        planned['paired_control'] = sum(p[a]['actual_counts']['policy_transitions']
                            for p in report['paired_control']['pairs'] for a in ('baseline', 'candidate'))
                    require(report['work_journal']['by_phase'] == planned, 'per-phase receipt and work journal differ')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                cleanup_errors.append(f'work journal: {type(exc).__name__}: {exc}')
        if cleanup_errors:
            report.update(status='INCOMPLETE', cleanup_errors=cleanup_errors, candidate_accepted=False,
                          observed_direction_validated=False)
        report['wall_seconds'], report['cpu_seconds'] = time.monotonic() - started, time.process_time() - cpu
        report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if report['artifact_bytes_before_report'] > (128 << 20) - (2 << 20):
            report.update(status='INCOMPLETE', failure='artifact reserve exceeded', candidate_accepted=False,
                          observed_direction_validated=False)
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'stage', 'counts', 'theta',
            'candidate_accepted', 'observed_direction_validated', 'cpu_seconds') if k in report}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
