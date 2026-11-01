"""One fresh, bounded model-gradient pilot on the original qualification tasks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(plan):
    require(plan['schema'] == 'd1-model-gradient53-preregistration-v1', 'wrong protocol')
    require(plan['model_seeds'] == [53001, 53002, 53003], 'registered model seeds differ')
    require(plan['data_seeds'] == {'train': list(range(53101, 53113)),
            'development': [53201, 53202, 53203], 'probe': [53301, 53302, 53303]},
            'registered data seeds differ')
    require(plan['evaluation_seeds'] == [53401, 53402, 53403], 'evaluation seeds differ')
    require(plan['budget']['preflight_cap_policy'] == 256
            and plan['budget']['formal_cap_policy'] == 24000, 'budget differs')
    for name, identity in plan['identities'].items():
        path = ROOT / name
        require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
                'sealed identity changed: ' + name)


def choose_proposal(rows):
    """Only development gradients select theta; probe values are merely sealed."""
    require(len(rows) == 3 and all(r['status'] == 'COMPLETE' for r in rows),
            'three complete models required')
    development = [r['development_gradients'] for r in rows]
    require(all(len(points) == 9 for points in development), 'nine development points required')
    means = [math.fsum(g['gradient'] for g in points) / len(points) for points in development]
    require(all(math.isfinite(g) for g in means), 'nonfinite development gradient')
    agreement = min(means) > 1e-6 or max(means) < -1e-6
    valid = all(g['autograd_agrees'] and g['direction_identifiable']
                for points in development for g in points)
    theta = (-.01 if math.fsum(means) > 0 else .01) if agreement and valid else None
    return {'theta': theta, 'development_mean_gradients': means,
            'test_gradients': [r['test_gradients'] for r in rows],
            'physical_probe_results_seen': False,
            'selection': 'equal development mean; all three directions agree; fixed step 0.01'}


def make_step_recorder(stream, progress):
    """Callback lives outside env state so full snapshots never copy its handle."""
    def record(env, action, result):
        _obs, reward, terminated, truncated, info = result
        row = {'phase': progress['phase'], 'counts': env.counters(),
               'episode_tick': int(env._episode_steps), 'reward': float(reward),
               'command_during_interval': [float(x) for x in info['command_during_interval']],
               'issued_action': [float(x) for x in action],
               'applied_action': [float(x) for x in info['applied_action']],
               'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact']),
               'base_height_m': float(info['base_height_m']), 'tilt_rad': float(info['tilt_rad']),
               'terminated': bool(terminated), 'truncated': bool(truncated)}
        stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    return record


def run_models(datasets, points, actor, output, report, cpu_started):
    from tools52 import control52 as control
    from tools52 import dynamics52 as dynamics
    for row in report['seeds']:
        row['status'] = 'INCOMPLETE'
        seed = row['seed']
        try:
            model, fitted = dynamics.fit_model(datasets['train'], seed, output / f'model_{seed}',
                                               min(cpu_started + 3500, time.process_time() + 600))
            row.update(fit=fitted, diagnostics={name: model.evaluate(datasets[name])
                       for name in ('development', 'probe')},
                       development_gradients=control.gradients(model, actor, points['development']),
                       test_gradients=control.gradients(model, actor, points['probe']))
            row['status'] = 'COMPLETE'
        except BaseException as exc:
            row['failure'] = f'{type(exc).__name__}: {exc}'
            write_json(output / f'model_{seed}_result.json', row)
            raise
        write_json(output / f'model_{seed}_result.json', row)
        print(json.dumps({'seed': seed, 'status': row['status'],
                          'updates': fitted['updates'], 'cpu_seconds': fitted['cpu_seconds']}), flush=True)


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
    report = {'schema': 'd1-model-gradient53-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'stage': 'preflight' if args.preflight else 'experiment', 'plan_sha256': sha(args.plan),
              'policy_network_updates': 0, 'control_parameter_updates': 0, 'model_updates': 0,
              'seeds': [{'seed': s, 'status': 'NOT_RUN'} for s in plan['model_seeds']],
              'candidate_accepted': False, 'counts': None, 'collection': {'status': 'NOT_RUN'},
              'interface_admission': {'status': 'NOT_RUN'},
              'physical_probes': {'status': 'NOT_RUN'}, 'paired_control': {'status': 'NOT_RUN'}}
    write_json(args.output / 'initial_incomplete.json', report | {'counts': 'unknown until final report'})
    started, cpu = time.monotonic(), time.process_time()
    env, audit = None, None
    progress = {'phase': 'admission'}

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        verify(plan)
        report['versions'] = {name: importlib.metadata.version(name) for name in plan['versions']}
        require(report['versions'] == plan['versions'], 'registered package versions differ')
        import torch

        from tools52 import adapter52 as adapter
        from tools52 import control52 as control
        from tools52 import run_pilot52 as shared
        from tools53 import collection53 as collection
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['torch_runtime_version'] = torch.__version__
        actor = adapter.load_actor(ROOT / plan['checkpoint'])
        report['actor_admission'] = shared.actor_admission(actor)
        if not args.preflight:
            admission = json.loads(args.admission.read_text())
            require(admission['status'] == 'COMPLETE' and admission['stage'] == 'preflight'
                    and admission['plan_sha256'] == report['plan_sha256'], 'unbound or failed preflight')
            require(admission['counts']['policy_transitions'] == 224, 'preflight count differs')
            report['admission_sha256'] = sha(args.admission)
        audit = (args.output / 'all_physics_steps.jsonl').open('x', buffering=1)
        env = collection.make_env(256 if args.preflight else 24000,
                                  audit_step=make_step_recorder(audit, progress))
        if args.preflight:
            report['interface_admission'] = {'status': 'INCOMPLETE'}
            report['interface_admission'] = collection.preflight(env, actor, args.output, plan)
            require(env.counters()['policy_transitions'] == 224, 'preflight work differs')
        else:
            progress['phase'] = 'collection'
            report['collection'] = {'status': 'INCOMPLETE', 'counts_before': env.counters()}
            datasets, points = collection.collect(env, actor, args.output, plan)
            report['collection'] = json.loads((args.output / 'collection.json').read_text())
            progress['phase'] = 'models'
            run_models(datasets, points, actor, args.output, report, cpu)
            proposal = choose_proposal(report['seeds'])
            shared.write_json(args.output / 'proposal_before_physics.json', proposal)
            report['sealed_proposal_sha256'] = sha(args.output / 'proposal_before_physics.json')
            report['theta'] = proposal['theta']
            report['control_parameter_updates'] = int(proposal['theta'] is not None)
            progress['phase'] = 'physical_probes'
            report['physical_probes'] = {'status': 'INCOMPLETE', 'counts_before': env.counters()}
            probes = shared.physical_probes(env, actor, points['probe'], args.output)
            report['physical_probes'] = probes
            report['direction_gate'] = control.compare_directions(proposal, probes)
            if report['direction_gate']['passed']:
                progress['phase'] = 'paired_control'
                report['paired_control'] = {'status': 'INCOMPLETE', 'counts_before': env.counters()}
                folder = args.output / 'paired_control'
                folder.mkdir()
                report['paired_control'] = shared.paired_control(env, actor, proposal['theta'], folder,
                                                                plan['evaluation_seeds'], plan)
                report['candidate_accepted'] = report['paired_control']['passed']
            else:
                report['paired_control']['reason'] = 'predeclared model-gradient gate failed'
        verify(plan)
        report.update(status='COMPLETE', frozen_inputs_unchanged=True)
    except BaseException as exc:  # noqa: BLE001 - retain interrupted/partial evidence without retry.
        report.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}', failure_phase=progress['phase'],
                      candidate_accepted=False)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    finally:
        cleanup_errors = []
        if env is not None:
            report['counts'] = env.counters()
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - retain cleanup failure without losing the experiment receipt.
                cleanup_errors.append(f'env.close: {type(exc).__name__}: {exc}')
        if audit is not None:
            try:
                audit.close()
            except Exception as exc:  # noqa: BLE001 - retain failed log flush alongside original failure.
                cleanup_errors.append(f'audit.close: {type(exc).__name__}: {exc}')
        if (args.output / 'collection.json').exists():
            try:
                report['collection'] = json.loads((args.output / 'collection.json').read_text())
            except (OSError, ValueError) as exc:
                cleanup_errors.append(f'collection receipt: {type(exc).__name__}: {exc}')
        if cleanup_errors:
            report.update(status='INCOMPLETE', candidate_accepted=False, cleanup_errors=cleanup_errors)
        journals = {}
        for path in args.output.glob('model_*/updates.jsonl'):
            with path.open() as stream:
                journals[path.parent.name] = sum(line.endswith('\n') for line in stream)
        report['journaled_model_updates'] = journals
        report['model_updates'] = sum(journals.values())
        report['update_count_semantics'] = 'completed journaled updates; interrupted update may be unrecorded'
        report['wall_seconds'], report['cpu_seconds'] = time.monotonic() - started, time.process_time() - cpu
        report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if report['artifact_bytes_before_report'] > (128 << 20) - (1 << 20):
            report.update(status='INCOMPLETE', failure='artifact cap reserve exceeded', candidate_accepted=False)
        write_json(args.output / 'report.json', report)
        print(json.dumps({k: report[k] for k in ('status', 'failure', 'stage', 'counts', 'model_updates',
                         'control_parameter_updates', 'candidate_accepted', 'wall_seconds', 'cpu_seconds')}), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
