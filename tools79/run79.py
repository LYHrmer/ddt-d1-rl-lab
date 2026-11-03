"""One authorized paired feedback experiment; no old protocol is mutated."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(directory))

from tools53.run_pilot53 import require, sha, write_json
from tools74.runtime74 import run_preserving

SEEDS = (64001, 64002, 64003)
ONLINE_SEEDS = (79101, 79102, 79103)
EVAL_SEEDS = (79401, 79402, 79403)
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
ARMS = ('frozen', 'refresh')
PLAN53 = 'docs/experiments53/model_gradient53_preregistered.json'
PLAN78 = 'docs/experiments78/protected_response78_preregistered.json'
SOURCE75 = 'evidence/world_vz75/execution_01'
SUMMARY53 = 'reports/model_gradient53_results_20261101.json'
INPUTS53 = ('collection.json', 'train_transitions.npz', 'development_snapshots.pkl')


def verify(plan):
    require(plan['schema'] == 'd1-feedback79-preregistered-v1', 'wrong79 schema')
    require(plan['model_seeds'] == list(SEEDS) and plan['online_seeds'] == list(ONLINE_SEEDS)
            and plan['evaluation_seeds'] == list(EVAL_SEEDS) and plan['arms'] == list(ARMS), '79 identities differ')
    require(plan['expected_formal_policy_steps'] == 39600 and plan['preflight_policy_steps'] == 224
            and plan['formal_policy_cap'] == 40000 and plan['model_updates'] == 300
            and plan['policy_updates'] == 9 and plan['cpu_seconds'] == 900
            and plan['wall_seconds'] == 1200 and plan['artifact_cap_bytes'] == 128 << 20,
            '79 registered budget differs')
    require(plan['user_confirmation'] == 'confirmed 2026-11-03 before implementation', '79 authorization absent')
    inherited = json.loads((ROOT / PLAN78).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(k) == v for k, v in inherited.items()), '78 source closure changed')
    required = {PLAN53, PLAN78, SUMMARY53, SOURCE75 + '/report.json',
                'docs/feedback79_proposal_20261103.md', 'docs/feedback79_execution_20261103.md',
                'reports/feedback79_source_feasibility_20261103.json',
                '.github/workflows/feedback79.yml',
                *(f'tools79/{name}.py' for name in ('run79', 'policy79', 'refit79', 'compare79')),
                *(f'tests/test_{name}79.py' for name in ('run', 'policy', 'refit', 'compare')),
                *(f'evidence/model_gradient53/execution_01/{name}' for name in INPUTS53)}
    require(required <= plan['source_sha256'].keys(), '79 required source/input unbound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '79 source/input changed: ' + name)


def validate_counts(counts, expected=None):
    keys = ('policy_transitions', 'native_steps', 'pd_updates')
    require(all(type(counts.get(k)) is int and counts[k] >= 0 for k in keys), 'invalid physical counters')
    n = counts['policy_transitions']
    require(counts['native_steps'] == n * 10 and counts['pd_updates'] == n * 5,
            'physical native/PD work differs')
    if expected is not None:
        require(n == expected, 'physical policy work differs')


def recover_updates(output):
    """Count durable completion receipts, including failed update stages."""
    rows, policies, models = [], 0, 0
    for path in sorted(Path(output).glob('seed*/**/receipt.json')):
        value = json.loads(path.read_text())
        n, attempted = value['updates_completed'], value['updates_attempted']
        require(type(n) is int and type(attempted) is int and 0 <= n <= attempted, 'invalid update receipt')
        if path.parent.name.startswith('actor_'):
            require(n <= attempted <= 1, 'actor stage exceeds one update')
            policies += n
        elif path.parent.name == 'refit_refresh':
            require(n <= attempted <= 100, 'feedback stage exceeds100 updates')
            models += n
        else:
            raise ValueError('unregistered update receipt: ' + str(path))
        rows.append({'path': str(path.relative_to(output)), 'status': value['status'],
                     'updates_completed': n, 'updates_attempted': attempted, 'sha256': sha(path)})
    require(policies <= 9 and models <= 300, '79 update cap exceeded')
    return {'policy_network_updates': policies, 'model_updates': models, 'receipts': rows,
            'semantics': 'durable completed receipts; hard-kill journals may establish a larger lower bound'}


def revoke_comparison(report, reason):
    """Retain provisional numbers without letting an incomplete run pass."""
    report['global_candidate_accepted'] = False
    comparison = report.get('comparison', {})
    if comparison.get('status') == 'COMPLETE':
        report['provisional_comparison_before_revocation'] = copy.deepcopy(comparison)
        comparison['status'] = 'INCOMPLETE'
        comparison['revocation_reason'] = reason
        comparison.get('feedback', {})['passed'] = False
        for gate in comparison.get('control_gates', []):
            gate['passed'] = False


def verify_final(report, plan, plan_path, output):
    try:
        require(sha(plan_path) == plan['_sha256'], '79 plan bytes changed during execution')
        verify(plan)
        if 'versions' in report:
            require({k: importlib.metadata.version(k) for k in report['versions']} == report['versions'],
                    '79 numeric versions changed during execution')
        for name, digest in report.get('raw_artifacts_sha256', {}).items():
            require(sha(output / name) == digest, '79 generated artifact changed: ' + name)
        if 'sealed_policies_sha256' in report:
            path = output / 'policies_sealed_before_evaluation.json'
            require(sha(path) == report['sealed_policies_sha256'], '79 policy seal changed')
            for policy in json.loads(path.read_text())['policies']:
                require(sha(output / policy['policy_file']) == policy['file_sha256'], '79 sealed actor changed')
    except BaseException as exc:
        revoke_comparison(report, f'final verification failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        if report['status'] != 'COMPLETE':
            revoke_comparison(report, 'execution incomplete')


def calculate(report, plan, output, preflight, admission):
    report.update(schema='d1-feedback79-result-v1', stage='preflight' if preflight else 'experiment',
                  plan_sha256=plan['_sha256'], user_confirmation=plan['user_confirmation'],
                  global_candidate_accepted=False, complete_model_eligibility='NOT_REQUALIFIED',
                  inherited75_and78_decisions_changed=False, no_critic=True, terminal_value=0,
                  model_seed_semantics='three paired world-model initializations; not independent PPO retraining',
                  models=[], online=[], baseline=[], candidates=[], comparison={'status': 'NOT_RUN'},
                  policy_network_updates=0, model_updates=0, actual_counts=None,
                  admission={'status': 'NOT_RUN'}, consumed53_inputs={})
    verify(plan)
    resource.setrlimit(resource.RLIMIT_CPU, (119, 120) if preflight else (899, 900))
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
            'Nice10/single CPU required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'single-thread CPU/no GPU required')
    started = time.monotonic()
    cpu_deadline = time.process_time() + (110 if preflight else 880)
    original = json.loads((ROOT / PLAN53).read_text())
    summary = json.loads((ROOT / SUMMARY53).read_text())
    source = json.loads((ROOT / SOURCE75 / 'report.json').read_text())
    require(source['status'] == source['verification_after'] == 'COMPLETE'
            and source['completed_model_fits'] == 6, 'complete75 source required')
    require(summary['experiment_status'] == 'COMPLETE', 'complete53 source required')
    report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
    require(report['versions'] == original['versions'], 'numeric packages changed')
    import numpy as np
    import torch

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import adapter52, control52, dynamics52
    from tools52.run_pilot52 import actor_admission, save_npz, save_points
    from tools53 import collection53
    from tools54.reproduce54 import load_points
    from tools56.run56 import bound53, check_points
    from tools75 import fit75
    from tools79.compare79 import compare
    from tools79.policy79 import update_actor
    from tools79.refit79 import refit_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report['torch_runtime_version'] = torch.__version__
    require(torch.__version__ == source['torch_runtime_version'], 'Torch runtime changed')
    anchor = adapter52.load_actor(ROOT / original['checkpoint'])

    def hook(key):
        def count(_module, _inputs, _output):
            report[key] += 1
        return count

    anchor.action_head.register_forward_hook(hook('actor_calls'))
    report['actor_call_semantics'] = 'action_head forwards across physical/inferred/copy actors'
    report['actor_admission'] = actor_admission(anchor)
    fixtures = json.loads((ROOT / 'evidence/low_memory_delay25/execution_01/random/export/golden_fixtures.json').read_text())['fixtures']
    clone = copy.deepcopy(anchor)
    with torch.no_grad():
        for row in fixtures:
            obs = np.r_[row['obs'], np.asarray(row['history_oldest_to_newest_previous_only']).ravel()]
            x = control52.tensor(obs[None])
            require(torch.equal(anchor(x), clone(x)), 'zero-update actor clone differs')
    report['zero_update_clone_bit_exact_fixtures'] = len(fixtures)
    del clone
    helpers = collection53.helpers(original)
    require(tuple(helpers.SEQUENCES) == TASKS, 'original tasks changed')
    env = None
    physics = (output / 'all_physics_steps.jsonl').open('x', buffering=1)
    progress = (output / 'progress.jsonl').open('x', buffering=1)
    phase = 'admission'
    failed_body = False

    def limits():
        require(time.process_time() < cpu_deadline, '79 CPU deadline reached')
        require(time.monotonic() - started < (170 if preflight else 1190), '79 wall deadline reached')
        used = sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
        require(used < plan['artifact_cap_bytes'] - (8 << 20), '79 evidence cap reserve exhausted')

    def event(name, **kwargs):
        limits()
        progress.write(json.dumps({'event': name, 'phase': phase, **kwargs}, allow_nan=False) + '\n')
        progress.flush()

    def record(current, action, result):
        _obs, reward, terminated, truncated, info = result
        counts = current.counters()
        validate_counts(counts)
        report['actual_counts'] = counts
        report['physics_steps'] = counts['policy_transitions']
        physics.write(json.dumps({'phase': phase, 'counts': counts,
            'episode_tick': int(current._episode_steps), 'reward': float(reward),
            'command_during_interval': np.asarray(info['command_during_interval']).tolist(),
            'issued_action': np.asarray(action).tolist(), 'applied_action': np.asarray(info['applied_action']).tolist(),
            'fallen': bool(info['fallen']), 'illegal_body_contact': bool(info['illegal_body_contact']),
            'terminated': bool(terminated), 'truncated': bool(truncated)}, separators=(',', ':'), allow_nan=False) + '\n')
        if counts['policy_transitions'] % 100 == 0:
            limits()

    try:
        if not preflight:
            accepted = json.loads(admission.read_text())
            require(accepted['status'] == accepted['verification_after'] == 'COMPLETE'
                    and accepted['stage'] == 'preflight' and accepted['plan_sha256'] == report['plan_sha256'],
                    '79 preflight incomplete or unbound')
            validate_counts(accepted['actual_counts'], 224)
            report['admission'] = {'status': 'COMPLETE', 'path': str(admission), 'sha256': sha(admission)}
        env = collection53.make_env(256 if preflight else 40000, audit_step=record)
        if preflight:
            report['admission'] = collection53.preflight(env, anchor, output, original)
            validate_counts(env.counters(), 224)
            return
        manifest = summary['raw_manifest']
        for name in INPUTS53:
            bound53('execution_01/' + name, manifest, report['consumed53_inputs'])
        collected = json.loads((ROOT / 'evidence/model_gradient53/execution_01/collection.json').read_text())
        require(collected == summary['formal']['collection'] and collected['status'] == 'COMPLETE', '53 collection changed')
        with np.load(ROOT / 'evidence/model_gradient53/execution_01/train_transitions.npz', allow_pickle=False) as raw:
            train = dynamics52.validate_data({k: raw[k].copy() for k in ('s', 'a', 'target', 'episode', 'tick', 'done')})
        points = load_points('development', manifest)
        check_points(points, (53201, 53202, 53203))
        labels = fit75.load_labels75(ROOT / 'evidence/world_vz75/labels_01')
        anchor_sha = fit75.paired.mapping_sha256(anchor.state_dict())
        report['anchor_mapping_sha256'] = anchor_sha
        actors = {}
        for seed, online_seed in zip(SEEDS, ONLINE_SEEDS, strict=True):
            folder = output / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            row = {'seed': seed, 'status': 'INCOMPLETE', 'first_policy': {'status': 'NOT_RUN'},
                   'feedback_fit': {'status': 'NOT_RUN'}, 'second_policies': {a: {'status': 'NOT_RUN'} for a in ARMS}}
            report['models'].append(row)
            old = next(m for m in source['models'] if m['seed'] == seed and m['arm'] == 'control')
            require(old['status'] == old['fit']['status'] == 'COMPLETE'
                    and old['fit']['updates'] == old['fit']['updates_attempted'] == 1000, 'partial source model')
            rel = old['directory'] + '/final_state_dict.pt'
            weights = ROOT / SOURCE75 / rel
            require(sha(weights) == source['sealed_artifacts_sha256'][rel], '75 source weight file changed')
            model = dynamics52.Dynamics52()
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            model.eval().requires_grad_(False)
            model_sha = fit75.paired.mapping_sha256(model.state_dict())
            require(model_sha == old['fit']['final_model_state_sha256'], '75 weight mapping changed')
            model.register_forward_hook(hook('model_calls'))
            row['source_model_sha256'] = model_sha
            phase = f'seed{seed}/actor_first'
            event('BEGIN')
            first, row['first_policy'] = update_actor(model, anchor, anchor, points, folder / 'actor_first', cpu_deadline=cpu_deadline)
            event('COMPLETE', stage_updates=row['first_policy']['updates_completed'])
            phase = f'seed{seed}/online'
            online_folder = folder / 'online'
            online_folder.mkdir(exist_ok=False)
            online_row = {'model_seed': seed, 'seed': online_seed, 'task': 'turn_stop', 'status': 'INCOMPLETE', 'steps': 0}
            report['online'].append(online_row)
            online, new_points = collect_online(env, first, helpers.SEQUENCES['turn_stop'], online_seed,
                                               online_folder, online_row, event)
            online_path = online_folder / 'transitions.npz'
            save_npz(online_path, online)
            save_points(online_folder / 'snapshots.pkl', new_points, env)
            online_row.update(status='COMPLETE', transitions_sha256=sha(online_path),
                              snapshots_sha256=sha(online_folder / 'snapshots.pkl'))
            write_json(online_folder / 'collection.json', online_row)
            phase = f'seed{seed}/refit_refresh'
            event('BEGIN')
            refreshed, row['feedback_fit'] = refit_model(model, train, online, labels, folder / 'refit_refresh', cpu_deadline=cpu_deadline)
            require(fit75.paired.mapping_sha256(model.state_dict()) == model_sha, 'frozen arm model changed')
            row['paired_first_actor_sha256'] = fit75.paired.mapping_sha256(first.state_dict())
            row['paired_online_data_sha256'] = fit75.paired.mapping_sha256(online)
            row['paired_second_point_ids'] = [(p['seed'], p['tick']) for p in new_points]
            for arm, learner in (('frozen', model), ('refresh', refreshed)):
                phase = f'seed{seed}/actor_{arm}'
                event('BEGIN')
                candidate, receipt = update_actor(learner, first, anchor, new_points, folder / f'actor_{arm}', cpu_deadline=cpu_deadline)
                row['second_policies'][arm] = receipt
                actors[(seed, arm)] = candidate
                require(fit75.paired.mapping_sha256(first.state_dict()) == row['paired_first_actor_sha256'], 'shared first actor mutated')
                event('COMPLETE', stage_updates=receipt['updates_completed'])
            row['status'] = 'COMPLETE'
            write_json(folder / 'seed_result.json', row)
        report['update_accounting'] = recover_updates(output)
        require(report['update_accounting']['policy_network_updates'] == 9
                and report['update_accounting']['model_updates'] == 300, '79 incomplete learner schedule')
        sealed = [{'model_seed': seed, 'arm': arm, 'mapping_sha256': fit75.paired.mapping_sha256(actors[(seed, arm)].state_dict()),
                   'policy_file': f'seed{seed}/actor_{arm}/actor_state_dict.pt',
                   'file_sha256': sha(output / f'seed{seed}/actor_{arm}/actor_state_dict.pt')}
                  for seed in SEEDS for arm in ARMS]
        require(all(p['mapping_sha256'] != anchor_sha for p in sealed), 'an actor never changed')
        write_json(output / 'policies_sealed_before_evaluation.json', {'policies': sealed, 'baseline_mapping_sha256': anchor_sha})
        report['sealed_policies_sha256'] = sha(output / 'policies_sealed_before_evaluation.json')
        phase = 'final_evaluation'
        event('ALL_SIX_POLICIES_SEALED', policy_files=sealed)
        evaluation = output / 'evaluation'
        evaluation.mkdir(exist_ok=False)
        jobs = [(None, 'baseline', anchor)] + [(s, a, actors[(s, a)]) for s in SEEDS for a in ARMS]
        for model_seed, arm, actor in jobs:
            policy_sha = fit75.paired.mapping_sha256(actor.state_dict())
            for task in TASKS:
                for reset in EVAL_SEEDS:
                    phase = f'evaluate/{model_seed}/{arm}/{task}/{reset}'
                    event('BEGIN')
                    path = evaluation / f'{model_seed}_{arm}_{task}_{reset}.npz'
                    env.eval_command_sequence = env._validate_eval_sequence(helpers.SEQUENCES[task])
                    shaper = ReferenceShaper()
                    def predict(_obs, actor=actor, shaper=shaper):
                        return control52.real_action(env, shaper, actor, 0.)
                    metrics = helpers.run_episode(env, predict, task, 0, reset, path,
                                                 (40000 - env.counters()['policy_transitions']) * 10)
                    item = {'task': task, 'seed': reset, 'metrics': metrics, 'policy_mapping_sha256': policy_sha,
                            'trace_path': str(path.relative_to(output)), 'trace_sha256': sha(path)}
                    if model_seed is not None:
                        item.update(model_seed=model_seed, arm=arm)
                    target = report['baseline'] if arm == 'baseline' else report['candidates']
                    target.append(item)
                    write_json(path.with_suffix('.json'), item)
                    event('EPISODE_SAVED', metrics=metrics)
                    require(metrics['status'] == 'COMPLETE' and not metrics['failure']
                            and metrics['done_reason'] == 'HORIZON_12S', '79 evaluation terminated/failed')
                    validate_counts(metrics['actual_counts'], 600)
                    require(fit75.paired.mapping_sha256(actor.state_dict()) == policy_sha, 'evaluation changed policy')
        validate_counts(env.counters(), 39600)
        report['comparison'] = compare(report['baseline'], report['candidates'])
        report['raw_artifacts_sha256'] = {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('progress.jsonl', 'all_physics_steps.jsonl')}
        event('COMPLETE')
    except BaseException:
        failed_body = True
        with (output / 'failure.txt').open('x') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        errors = []
        if env is not None:
            report['actual_counts'] = env.counters()
            report['physics_steps'] = env.counters()['policy_transitions']
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - preserve cleanup evidence after any stage failure.
                errors.append(f'env.close: {type(exc).__name__}: {exc}')
        for stream in (physics, progress):
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - attempt both journal closes and retain errors.
                errors.append(f'journal.close: {type(exc).__name__}: {exc}')
        try:
            accounting = recover_updates(output)
            report.update(update_accounting=accounting, policy_network_updates=accounting['policy_network_updates'],
                          model_updates=accounting['model_updates'])
        except Exception as exc:  # noqa: BLE001 - never discard physical work if receipt readback fails.
            errors.append(f'update accounting: {type(exc).__name__}: {exc}')
        report['raw_bytes_before_report'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        report['cleanup_errors'] = errors
        if errors and not failed_body:
            raise RuntimeError('; '.join(errors))


def collect_online(env, actor, spec, seed, folder, receipt, event):
    """Keep complete prefix records even when an online trajectory fails."""
    import numpy as np

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52
    from tools52.run_pilot52 import save_npz
    from tools53 import collection53

    rows, points = [], []
    before = env.counters()
    try:
        env.eval_command_sequence = env._validate_eval_sequence(spec)
        env.reset(seed=seed)
        shaper = ReferenceShaper()
        with (folder / 'steps.jsonl').open('x', buffering=1) as journal:
            for tick in range(600):
                if tick in (110, 260, 410):
                    points.append(control52.point(env, shaper, spec, seed))
                state = control52.physical(env)
                issued = control52.real_action(env, shaper, actor, 0.)
                result = env.step(issued)
                row = collection53._record(env, state, issued, result, 0, tick, final_tick=tick == 599)
                rows.append(row)
                receipt['steps'] = len(rows)
                journal.write(json.dumps(collection53._plain(row), separators=(',', ':'), allow_nan=False) + '\n')
                require(not row['terminated'] and not row['truncated'] and not row['fallen']
                        and not row['illegal_body_contact'], '79 online trajectory terminated/failed')
        counts = {k: env.counters()[k] - before[k] for k in before}
        validate_counts(counts, 600)
        receipt['actual_counts'] = counts
        require([(p['seed'], p['tick']) for p in points] == [(seed, t) for t in (110, 260, 410)], 'online points differ')
        event('ONLINE_COMPLETE', seed=seed, steps=len(rows))
        arrays = collection53._arrays(rows)
        return {k: np.asarray(arrays[k]).copy() for k in ('s', 'a', 'target', 'episode', 'tick', 'done')}, points
    except BaseException as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        receipt['actual_counts'] = {k: env.counters()[k] - before[k] for k in before}
        save_npz(folder / 'full_or_partial_trace.npz', collection53._arrays(rows))
        write_json(folder / 'collection_prefix.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--admission', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan['_sha256'] = sha(args.plan)
    context = {}
    def body(report):
        context['report'] = report
        calculate(report, plan, args.output, args.preflight, args.admission)
    report = run_preserving(args.output, body,
                            lambda: verify_final(context['report'], plan, args.plan, args.output))
    print(json.dumps({k: report.get(k) for k in ('status', 'stage', 'failure', 'actual_counts',
                     'policy_network_updates', 'model_updates', 'cpu_seconds', 'wall_seconds')}, allow_nan=False), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
