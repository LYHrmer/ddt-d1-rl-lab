"""Approved81 v2: presealed C6/Q6 candidates, independent V and conditional H."""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from tools79.run79 import validate_counts

SEEDS = (64001, 64002, 64003)
ARMS = ('C6', 'Q6')
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
D_SEEDS = (81107, 81108, 81109)
V_SEEDS = (81207, 81208, 81209)
EVAL_SEEDS = (81401, 81402, 81403)
STEADY_TICKS = (175, 325)
TURN_TICKS = (110, 175, 260, 325, 410)
SCALES = (.009, .0045, .00225, .001125)
PLAN53 = 'docs/experiments53/model_gradient53_preregistered.json'
PLAN80 = 'docs/experiments80/steady_sampling80_preregistered.json'
SOURCE75 = 'evidence/world_vz75/execution_01'
FULL_STEPS, SCREEN_STEPS = 53304, 15504
MODEL_CALLS = 23328


def verify(plan):
    require(plan['schema'] == 'd1-objective81-preregistered-v2', 'wrong81 schema')
    expected = {'model_seeds': list(SEEDS), 'arms': list(ARMS),
        'direction_seeds': list(D_SEEDS), 'validation_seeds': list(V_SEEDS),
        'evaluation_seeds': list(EVAL_SEEDS), 'steady_ticks': list(STEADY_TICKS),
        'validation_ticks': list(TURN_TICKS), 'scales': list(SCALES), 'primary_scale_index': 0,
        'primary_sign': 1, 'starts_per_objective': 6, 'policy_updates': 6, 'model_updates': 0,
        'preflight_policy_steps': 224, 'direction_policy_steps': 1800,
        'validation_collection_policy_steps': 1800, 'screen_policy_steps': 11904,
        'evaluation_policy_steps': 37800, 'formal_policy_cap': FULL_STEPS,
        'model_forward_cap': 30000, 'expected_model_forwards': MODEL_CALLS,
        'cpu_seconds': 900, 'wall_seconds': 1200, 'artifact_cap_bytes': 160 << 20,
        'user_confirmation': 'execute approved 2026-11-03 after v2 proposal'}
    require(all(plan.get(k) == v for k, v in expected.items()), '81 authorized settings differ')
    inherited = json.loads((ROOT / PLAN80).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(k) == v for k, v in inherited.items()), '80 frozen closure changed')
    required = {PLAN53, PLAN80, SOURCE75 + '/report.json',
        'docs/objective81_proposal_20261103.md', 'docs/objective81_literature_20261103.md',
        'docs/objective81_execution_20261103.md', 'reports/objective81_source_feasibility_20261103.json',
        '.github/workflows/objective81.yml',
        *(f'tools81/{n}81.py' for n in ('run', 'policy', 'compare', 'diagnostics')),
        *(f'tests/test_{n}81.py' for n in ('run', 'policy', 'compare', 'diagnostics'))}
    require(required <= plan['source_sha256'].keys(), '81 sources not bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '81 source/input changed: ' + name)


def validate_points(points, seeds, ticks):
    ids = [(p['seed'], p['tick']) for p in points]
    require(len(ids) == len(set(ids)) == len(seeds) * len(ticks)
            and set(ids) == {(s, t) for s in seeds for t in ticks}, '81 point grid incomplete/duplicate')
    require(all(p['task'] == 'turn_stop' for p in points), '81 unexpected task')
    return sorted(points, key=lambda p: (p['seed'], p['tick']))


def recover_updates(output):
    receipts, completed = [], 0
    for path in sorted(Path(output).glob('seed*/actor_*/receipt.json')):
        require(path.parent.parent.name in {f'seed{s}' for s in SEEDS}
                and path.parent.name in {f'actor_{a}' for a in ARMS}, 'unregistered81 update')
        value = json.loads(path.read_text())
        n, attempted = value['updates_completed'], value['updates_attempted']
        require(type(n) is int and type(attempted) is int and 0 <= n <= attempted <= 1,
                'invalid81 update accounting')
        completed += n
        receipts.append({'path': str(path.relative_to(output)), 'status': value['status'],
            'updates_completed': n, 'updates_attempted': attempted, 'sha256': sha(path)})
    require(completed <= 6, '81 update cap exceeded')
    return {'policy_network_updates': completed, 'model_updates': 0, 'receipts': receipts,
            'semantics': 'durable receipts; hard-kill journals may establish a larger lower bound'}


def revoke(report, reason):
    report['global_candidate_accepted'] = report['objective_passed'] = False
    for key in ('screen', 'comparison'):
        previous = report.get(key, {})
        if previous.get('status') == 'COMPLETE':
            report['provisional_' + key + '_before_revocation'] = copy.deepcopy(previous)
            previous.update(status='INCOMPLETE', revocation_reason=reason, passed=False,
                            enter_full_evaluation=False, objective_passed=False)
            previous.get('objective', {})['passed'] = False
            for gate in previous.get('control_gates', []):
                gate['passed'] = False


def conditional_evaluation(report, callback):
    require(report['screen'].get('status') == 'COMPLETE', 'screen is incomplete')
    if report['screen'].get('enter_full_evaluation') is not True:
        require(not report['baseline'] and not report['candidates'], 'holdout touched before rejection')
        report.update(outcome='SCREEN_REJECTED', evaluation_status='NOT_RUN', objective_passed=False,
                      evaluation_not_run_reason='whole-three-model V failed; no seed/scale selection')
        return SCREEN_STEPS
    callback()
    require(report['comparison'].get('status') == 'COMPLETE', 'full comparison incomplete')
    report.update(outcome='FULL_EVALUATION_COMPLETE', evaluation_status='COMPLETE',
                  objective_passed=report['comparison']['objective_passed'])
    return FULL_STEPS


def seal(output, name, paths, report):
    members = {str(p.relative_to(output)): sha(p) for p in sorted(paths) if p.is_file()}
    path = output / name
    write_json(path, {'artifact_sha256': members})
    report.setdefault('seals', {})[name] = sha(path)


def verify_final(report, plan, plan_path, output):
    try:
        require(sha(plan_path) == plan['_sha256'], '81 plan changed during run')
        verify(plan)
        if 'versions' in report:
            require({k: importlib.metadata.version(k) for k in report['versions']} == report['versions'],
                    '81 numeric versions changed')
        for name, digest in report.get('seals', {}).items():
            path = output / name
            require(sha(path) == digest, '81 original seal changed')
            for member, member_sha in json.loads(path.read_text())['artifact_sha256'].items():
                require(sha(output / member) == member_sha, '81 sealed member changed: ' + member)
        for name, digest in report.get('raw_artifacts_sha256', {}).items():
            require(sha(output / name) == digest, '81 generated artifact changed: ' + name)
        accepted = report.get('admission', {})
        if 'path' in accepted and 'sha256' in accepted:
            require(sha(Path(accepted['path'])) == accepted['sha256'], '81 admission changed')
        require(sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file()) < plan['artifact_cap_bytes'],
                '81 final artifact cap exceeded')
    except BaseException as exc:
        revoke(report, f'final verification failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        if report['status'] != 'COMPLETE':
            revoke(report, 'execution incomplete')

def array_identity(arrays):
    """Hash complete scalar traces, excluding monotonic global work counters."""
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def collect_episode(env, actor, task, seed, spec, folder, receipt, *, ticks):
    import numpy as np

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52
    from tools52.run_pilot52 import save_npz, save_points
    from tools53 import collection53

    folder.mkdir(exist_ok=False)
    rows, points, before = [], [], env.counters()
    try:
        env.eval_command_sequence = env._validate_eval_sequence(spec)
        env.reset(seed=seed)
        shaper = ReferenceShaper()
        for tick in range(600):
            if tick in ticks:
                points.append(control52.point(env, shaper, spec, seed) | {'task': task})
            state = control52.physical(env)
            issued = control52.real_action(env, shaper, actor, 0.)
            result = env.step(issued)
            row = collection53._record(env, state, issued, result, 0, tick, final_tick=tick == 599)
            rows.append(row)
            receipt['steps'] = len(rows)
            require(not any(row[k] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated')),
                    '81 development trajectory unsafe/incomplete')
        require([(p['seed'], p['tick']) for p in points] == [(seed, tick) for tick in ticks], '81 point capture differs')
        validate_counts({k: env.counters()[k] - before[k] for k in before}, 600)
        receipt['status'] = 'COMPLETE'
        return points
    except BaseException as exc:
        receipt.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        path = folder / 'full_or_partial_trace.npz'
        save_npz(path, collection53._arrays(rows))
        save_points(folder / 'integration_context_archive.pkl', points, env)
        context = {}
        for index, point in enumerate(points):
            context.update({f'{index}_{name}': np.asarray(point[name]) for name in
                ('state', 'frame', 'history', 'filtered', 'last', 'fifo', 'commands')})
            context[f'{index}_nominal_height'] = np.asarray(point['nominal_height'])
        save_npz(folder / 'point_arrays.npz', context)
        receipt.update(actual_counts={k: env.counters()[k] - before[k] for k in before},
            trace_sha256=sha(path), points=[{'seed': p['seed'], 'tick': p['tick'], 'task': task} for p in points],
            archive_semantics='integration/Python context only; physical branches use retained full live Snapshot')
        write_json(folder / 'receipt.json', receipt)


def physical_branch(env, actor, point, path, receipt):
    import numpy as np

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import adapter52, control52
    from tools52.run_pilot52 import save_npz
    from tools55.probe_local55 import cost64

    before = env.counters()
    data = {k: [] for k in ('states', 'observations', 'rewards', 'commands', 'body_wz',
                           'issued_actions', 'applied_actions', 'fallen', 'illegal_body_contact',
                           'terminated', 'truncated')}
    captured, extra = [], {}
    handle = actor.encoder.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().cpu().numpy()[0].copy()))
    try:
        shaper = ReferenceShaper()
        adapter52.restore(env, shaper, point['snapshot'])
        for tick in range(32):
            require(np.array_equal(env._command, point['commands'][tick]), '81 branch command clock differs')
            captured.clear()
            action = control52.real_action(env, shaper, actor, 0.)
            require(len(captured) == 1, 'physical actor input capture differs')
            if tick == 0:
                extra['first_actor_input'] = captured[0]
            obs, reward, terminated, truncated, info = env.step(action)
            row = {'states': control52.physical(env), 'observations': obs.copy(), 'rewards': reward,
                'commands': np.asarray(info['command_during_interval']).copy(), 'body_wz': info['body_wz_rad_s'],
                'issued_actions': np.asarray(action).copy(), 'applied_actions': np.asarray(info['applied_action']).copy(),
                'fallen': info['fallen'], 'illegal_body_contact': info['illegal_body_contact'],
                'terminated': terminated, 'truncated': truncated}
            for key, values in data.items():
                values.append(row[key])
            receipt['steps'] = tick + 1
            require(not any(row[k] for k in ('fallen', 'illegal_body_contact', 'terminated', 'truncated')),
                    '81 short branch unsafe/incomplete')
        arrays = {k: np.asarray(v) for k, v in data.items()} | extra
        require(all(np.isfinite(v).all() for v in arrays.values()), 'nonfinite81 physical trace')
        cost, costs = cost64(point, arrays['states'], arrays['observations'])
        from tools81.diagnostics81 import physical_cost_parts, trace_metrics
        extra.update(costs=np.asarray(costs), cost_components=physical_cost_parts(point, arrays))
        arrays.update(extra)
        receipt.update(trace_metrics(arrays))
        require(np.array_equal(arrays['commands'], point['commands'][:32]), '81 applied command trace differs')
        receipt.update(status='COMPLETE', cost=cost, per_step_cost64=costs,
            yaw_mae=float(np.mean(np.abs(arrays['body_wz'] - arrays['commands'][:, 2]))),
            trace_identity_sha256=array_identity(arrays), fallen=False, illegal_body_contact=False)
        validate_counts({k: env.counters()[k] - before[k] for k in before}, 32)
    except BaseException as exc:
        receipt.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        handle.remove()
        arrays = {k: np.asarray(v) for k, v in data.items()} | extra
        save_npz(path, arrays)
        counts = {k: env.counters()[k] - before[k] for k in before}
        receipt.update(actual_counts=counts, native_steps=counts['native_steps'], pd_updates=counts['pd_updates'],
                       trace_sha256=sha(path), trace_file=path.name)
        write_json(path.with_suffix('.json'), receipt)


def evaluate(env, anchor, actors, helpers, output, report, event, set_phase, save_prefix):
    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52
    from tools56.fit56 import mapping_sha256

    evaluation = output / 'evaluation'
    evaluation.mkdir(exist_ok=False)
    jobs = [(None, 'baseline', anchor)] + [(s, a, actors[(s, a)]) for s in SEEDS for a in ARMS]
    for model_seed, arm, actor in jobs:
        policy_sha = mapping_sha256(actor.state_dict())
        for task in TASKS:
            for seed in EVAL_SEEDS:
                set_phase(f'evaluate/{model_seed}/{arm}/{task}/{seed}')
                event('BEGIN')
                path = evaluation / f'{model_seed}_{arm}_{task}_{seed}.npz'
                env.eval_command_sequence = env._validate_eval_sequence(helpers.SEQUENCES[task])
                shaper = ReferenceShaper()
                initial_xy = None
                def predict(_obs, actor=actor, shaper=shaper):
                    nonlocal initial_xy
                    if initial_xy is None:
                        initial_xy = [float(env.data.qpos[env.free_qpos]), float(env.data.qpos[env.free_qpos + 1])]
                    return control52.real_action(env, shaper, actor, 0.)
                before = env.counters()
                item = {'task': task, 'seed': seed, 'status': 'INCOMPLETE', 'policy_mapping_sha256': policy_sha}
                if model_seed is not None:
                    item.update(model_seed=model_seed, arm=arm)
                report['baseline' if arm == 'baseline' else 'candidates'].append(item)
                write_json(path.with_suffix('.pending.json'), item)
                try:
                    metrics = helpers.run_episode(env, predict, task, 0, seed, path,
                                                 (FULL_STEPS - env.counters()['policy_transitions']) * 10)
                    item.update(metrics=metrics, trace_path=str(path.relative_to(output)), trace_sha256=sha(path))
                    require(metrics['status'] == 'COMPLETE' and not metrics['failure']
                            and metrics['done_reason'] == 'HORIZON_12S', '81 evaluation incomplete/unsafe')
                    validate_counts(metrics['actual_counts'], 600)
                    require(mapping_sha256(actor.state_dict()) == policy_sha, '81 evaluation changed actor')
                    item['status'] = 'COMPLETE'
                    event('EPISODE_SAVED')
                except BaseException as exc:
                    item.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                    partial = path.with_name(path.stem + '_partial.npz')
                    item['partial_trace_rows'] = save_prefix(partial, initial_xy)
                    item.update(partial_trace_path=str(partial.relative_to(output)), partial_trace_sha256=sha(partial))
                    raise
                finally:
                    item['actual_counts_observed'] = {k: env.counters()[k] - before[k] for k in before}
                    write_json(path.with_suffix('.json'), item)


def calculate(report, plan, output, preflight, admission):
    report.update(schema='d1-objective81-result-v2', stage='preflight' if preflight else 'experiment',
        plan_sha256=plan['_sha256'], user_confirmation=plan['user_confirmation'],
        global_candidate_accepted=False, complete_model_eligibility='NOT_REQUALIFIED',
        inherited75_and78_decisions_changed=False, no_critic=True, terminal_value=0,
        model_seed_semantics='three paired world-model initializations; not independent PPO retraining',
        models=[], development=[], validation=[], predictions=[], short_branches=[], baseline=[], candidates=[],
        comparison={'status': 'NOT_RUN'}, screen={'status': 'NOT_RUN'}, objective_passed=False,
        evaluation_status='NOT_RUN', policy_network_updates=0, model_updates=0, actual_counts=None,
        admission={'status': 'NOT_RUN'})
    verify(plan)
    resource.setrlimit(resource.RLIMIT_CPU, (119, 120) if preflight else (899, 900))
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
            '81 Nice10/single CPU required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        '81 single-thread CPU/no GPU required')
    started, cpu_deadline = time.monotonic(), time.process_time() + (110 if preflight else 880)
    original = json.loads((ROOT / PLAN53).read_text())
    source = json.loads((ROOT / SOURCE75 / 'report.json').read_text())
    require(source['status'] == source['verification_after'] == 'COMPLETE' and source['completed_model_fits'] == 6,
            'complete75 source required')
    report['versions'] = {k: importlib.metadata.version(k) for k in original['versions']}
    require(report['versions'] == original['versions'], '81 numeric versions changed')
    import numpy as np
    import torch

    from tools52 import adapter52, control52, dynamics52
    from tools52.run_pilot52 import actor_admission, save_npz
    from tools53 import collection53
    from tools54 import numerics54
    from tools56.fit56 import mapping_sha256
    from tools81.compare81 import branch_specs, compare, diagnostics, screen_compare
    from tools81.diagnostics81 import cross_readback, trace_metrics, validation_ad
    from tools81.policy81 import objective_values, update_actor

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report['torch_runtime_version'] = torch.__version__
    require(torch.__version__ == source['torch_runtime_version'], 'Torch runtime changed')
    anchor = adapter52.load_actor(ROOT / original['checkpoint'])

    def hook(key):
        def count(_module, _inputs, _output):
            report[key] += 1
            if key == 'model_calls':
                require(report[key] <= plan['model_forward_cap'], '81 model forward cap exceeded')
        return count

    def before_model_forward(_module, _inputs):
        require(report['model_calls'] < plan['model_forward_cap'], '81 model cap reached before forward')

    anchor.action_head.register_forward_hook(hook('actor_calls'))
    report['actor_call_semantics'] = 'action_head forwards across physical/inferred/copy actors'
    report['actor_admission'] = actor_admission(anchor)
    fixtures = json.loads((ROOT / 'evidence/low_memory_delay25/execution_01/random/export/golden_fixtures.json').read_text())['fixtures']
    clone = copy.deepcopy(anchor)
    with torch.no_grad():
        for row in fixtures:
            obs = np.r_[row['obs'], np.asarray(row['history_oldest_to_newest_previous_only']).ravel()]
            x = control52.tensor(obs[None])
            require(torch.equal(anchor(x), clone(x)), '81 zero-update clone differs')
    report['zero_update_clone_bit_exact_fixtures'] = len(fixtures)
    del clone
    helpers = collection53.helpers(original)
    require(tuple(helpers.SEQUENCES) == TASKS, 'original tasks changed')
    anchor_sha = mapping_sha256(anchor.state_dict())
    report['anchor_mapping_sha256'] = anchor_sha
    env, models, actors, probes, directions = None, {}, {}, {}, {}
    physics = (output / 'all_physics_steps.jsonl').open('x', buffering=1)
    progress = (output / 'progress.jsonl').open('x', buffering=1)
    phase, failed_body = 'admission', False
    evaluation_rows = []

    def set_phase(value):
        nonlocal phase
        phase = value
        if phase.startswith('evaluate/'):
            evaluation_rows.clear()

    def save_evaluation_prefix(path, initial_xy):
        keys = ('reward', 'command_during_interval', 'body_vx_m_s', 'body_vy_m_s',
                'body_wz_rad_s', 'base_x_m', 'base_y_m', 'base_height_m',
                'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad', 'tilt_rad',
                'fallen', 'illegal_body_contact', 'tau_saturation_count', 'issued_action', 'applied_action')
        arrays = {k: np.asarray([row[k] for row in evaluation_rows]) for k in keys}
        positions = ([] if initial_xy is None else [initial_xy]) + [
            [row['base_x_m'], row['base_y_m']] for row in evaluation_rows]
        arrays['boundary_xy_m'] = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        arrays['policy_interval_start_s'] = np.arange(len(evaluation_rows)) * .02
        save_npz(path, arrays)
        return len(evaluation_rows)

    def limits():
        require(time.process_time() < cpu_deadline, '81 CPU deadline reached')
        require(time.monotonic() - started < (170 if preflight else 1190), '81 wall deadline reached')
        used = sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
        require(used < plan['artifact_cap_bytes'] - (8 << 20), '81 evidence cap reserve exhausted')

    def event(name, **kwargs):
        limits()
        progress.write(json.dumps({'event': name, 'phase': phase, **kwargs}, allow_nan=False) + '\n')
        progress.flush()

    def record(current, action, result):
        _obs, reward, terminated, truncated, info = result
        counts = current.counters()
        if phase.startswith('evaluate/'):
            evaluation_rows.append(copy.deepcopy(dict(info, reward=float(reward), issued_action=np.asarray(action))))
        validate_counts(counts)
        report['actual_counts'], report['physics_steps'] = counts, counts['policy_transitions']
        physics.write(json.dumps({'phase': phase, 'counts': counts, 'episode_tick': int(current._episode_steps),
            'reward': float(reward), 'command_during_interval': np.asarray(info['command_during_interval']).tolist(),
            'issued_action': np.asarray(action).tolist(), 'applied_action': np.asarray(info['applied_action']).tolist(),
            'body_wz_rad_s': float(info['body_wz_rad_s']), 'fallen': bool(info['fallen']),
            'illegal_body_contact': bool(info['illegal_body_contact']), 'terminated': bool(terminated),
            'truncated': bool(truncated)}, separators=(',', ':'), allow_nan=False) + '\n')
        if counts['policy_transitions'] % 100 == 0:
            limits()

    try:
        if not preflight:
            accepted = json.loads(admission.read_text())
            require(accepted['status'] == accepted['verification_after'] == 'COMPLETE'
                    and accepted['stage'] == 'preflight' and accepted['plan_sha256'] == plan['_sha256'],
                    '81 preflight incomplete/unbound')
            validate_counts(accepted['actual_counts'], 224)
            report['admission'] = {'status': 'COMPLETE', 'path': str(admission), 'sha256': sha(admission)}
        env = collection53.make_env(256 if preflight else FULL_STEPS, audit_step=record)
        if preflight:
            report['admission'] = collection53.preflight(env, anchor, output, original)
            report['planned_validation_live_snapshot_count81'] = 15
            report['planned_direction_points81'] = 6
            validate_counts(env.counters(), 224)
            return

        development = output / 'development'
        development.mkdir(exist_ok=False)
        d_points = []
        for seed in D_SEEDS:
            phase = f'development/turn_stop/{seed}'
            event('BEGIN')
            row = {'task': 'turn_stop', 'seed': seed, 'status': 'INCOMPLETE', 'steps': 0}
            report['development'].append(row)
            d_points.extend(collect_episode(env, anchor, 'turn_stop', seed, helpers.SEQUENCES['turn_stop'],
                development / f'turn_stop_{seed}', row, ticks=STEADY_TICKS))
            event('D_EPISODE_COMPLETE')
        d_points = validate_points(d_points, D_SEEDS, STEADY_TICKS)
        validate_counts(env.counters(), 1800)
        seal(output, 'development_sealed_before_updates.json', development.rglob('*'), report)
        # Only numerical point inputs are used by D; physical V uses its own live snapshots.
        for point in d_points:
            del point['snapshot']
        for seed in SEEDS:
            folder = output / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            row = {'seed': seed, 'status': 'INCOMPLETE', 'arms': {}}
            report['models'].append(row)
            old = next(m for m in source['models'] if m['seed'] == seed and m['arm'] == 'control')
            require(old['status'] == old['fit']['status'] == 'COMPLETE'
                    and old['fit']['updates'] == old['fit']['updates_attempted'] == 1000, 'partial75 source')
            rel = old['directory'] + '/final_state_dict.pt'
            weights = ROOT / SOURCE75 / rel
            require(sha(weights) == source['sealed_artifacts_sha256'][rel], '75 source weights changed')
            model = dynamics52.Dynamics52()
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
            model.eval().requires_grad_(False)
            model_sha = mapping_sha256(model.state_dict())
            require(model_sha == old['fit']['final_model_state_sha256'], '75 model mapping changed')
            model.register_forward_pre_hook(before_model_forward)
            model.register_forward_hook(hook('model_calls'))
            models[seed] = model
            row['source_model_sha256'] = model_sha
            for arm in ARMS:
                phase = f'seed{seed}/actor_{arm}'
                event('BEGIN')
                arm_probes, receipt = update_actor(model, anchor, d_points, arm,
                    folder / f'actor_{arm}', cpu_deadline=cpu_deadline)
                row['arms'][arm] = receipt
                require(receipt['status'] == receipt['verification_after'] == 'COMPLETE'
                        and receipt['updates_completed'] == 1, '81 candidate generation incomplete')
                require(set(arm_probes) == {(k, sign) for k in range(4) for sign in (1, -1)},
                        '81 probe grid incomplete')
                for (k, sign), actor in arm_probes.items():
                    probes[(seed, arm, k, sign)] = actor
                actors[(seed, arm)] = arm_probes[(0, 1)]
                step = torch.load(folder / f'actor_{arm}/head_step.pt', weights_only=True, map_location='cpu')
                directions[(seed, arm)] = (step['direction_weight64'], step['direction_bias64'])
                event('CANDIDATE_COMPLETE')
            require(mapping_sha256(model.state_dict()) == model_sha, '81 frozen source model changed')
            row['status'] = 'COMPLETE'
            write_json(folder / 'seed_result.json', row)
        report['update_accounting'] = recover_updates(output)
        require(report['update_accounting']['policy_network_updates'] == 6, '81 candidate schedule incomplete')
        sealed = [{'model_seed': s, 'arm': a, 'scale_index': k, 'sign': sign,
                   'mapping_sha256': mapping_sha256(actor.state_dict())}
                  for (s, a, k, sign), actor in sorted(probes.items())]
        report['unchanged_probe_count'] = sum(p['mapping_sha256'] == anchor_sha for p in sealed)
        write_json(output / 'probe_identities.json', {'probes': sealed, 'baseline_mapping_sha256': anchor_sha,
                   'primary_scale_index': 0, 'primary_sign': 1})
        seal(output, 'policies_sealed_before_validation.json',
             [output / 'probe_identities.json', *output.glob('seed*/**/*')], report)
        event('ALL_SIX_CANDIDATES_AND_PROBES_SEALED_BEFORE_V')
        validation = output / 'validation'
        validation.mkdir(exist_ok=False)
        v_points = []
        for seed in V_SEEDS:
            phase = f'validation/turn_stop/{seed}'
            event('BEGIN')
            row = {'task': 'turn_stop', 'seed': seed, 'status': 'INCOMPLETE', 'steps': 0}
            report['validation'].append(row)
            v_points.extend(collect_episode(env, anchor, 'turn_stop', seed, helpers.SEQUENCES['turn_stop'],
                validation / f'turn_stop_{seed}', row, ticks=TURN_TICKS))
            event('V_EPISODE_COMPLETE')
        v_points = validate_points(v_points, V_SEEDS, TURN_TICKS)
        validate_counts(env.counters(), 3600)
        seal(output, 'validation_sealed_before_predictions.json', validation.rglob('*'), report)
        prediction_dir = output / 'predictions'
        prediction_dir.mkdir(exist_ok=False)
        report['validation_ad'] = []
        for seed in SEEDS:
            model64 = copy.deepcopy(models[seed]).double().eval().requires_grad_(False)
            phase = f'validation_ad/{seed}'
            event('BEGIN')
            report['validation_ad'].extend(validation_ad(model64, anchor,
                [p for p in v_points if p['tick'] in STEADY_TICKS],
                {a: directions[(seed, a)] for a in ARMS}, seed, objective_values, cpu_deadline))
            for point in v_points:
                jobs = [('anchor', None, 0, anchor)] + [
                    (a, k, sign, probes[(seed, a, k, sign)]) for a in ARMS
                    for k in (range(4) if point['tick'] in STEADY_TICKS else (0,))
                    for sign in ((1, -1) if point['tick'] in STEADY_TICKS else (1,))]
                for arm, k, sign, actor in jobs:
                    phase = f'prediction/{seed}/{arm}/{k}/{sign}/{point["seed"]}/{point["tick"]}'
                    limits()
                    actor64 = copy.deepcopy(actor).double().eval().requires_grad_(False)
                    with torch.no_grad():
                        value, trace = numerics54.imaginary(model64, actor64, point,
                            torch.zeros((), dtype=torch.float64), return_trace=True, horizon=32)
                    arrays = {key: v.numpy() for key, v in trace.items()}
                    arrays['commands'] = point['commands'][:32].copy()
                    path = prediction_dir / f'{seed}_{arm}_{k}_{sign}_{point["seed"]}_{point["tick"]}.npz'
                    save_npz(path, arrays)
                    row = {'model_seed': seed, 'arm': arm, 'scale_index': k, 'sign': sign,
                        'seed': point['seed'], 'tick': point['tick'], 'cost': float(value),
                        'trace_path': str(path.relative_to(output)), 'trace_sha256': sha(path),
                        'precision': 'F64 model; exact F32 probe weights promoted to F64; no physical AD',
                        **trace_metrics(arrays)}
                    report['predictions'].append(row)
        write_json(prediction_dir / 'validation_ad.json', report['validation_ad'])
        write_json(prediction_dir / 'predictions.json', report['predictions'])
        require(len(report['predictions']) == 387, '81 prediction grid differs')
        seal(output, 'predictions_sealed_before_physics.json', prediction_dir.rglob('*'), report)
        event('ALL_PREDICTIONS_SEALED')
        screen_dir = output / 'screen'
        screen_dir.mkdir(exist_ok=False)
        for point in v_points:
            for model_seed, arm, repeat, k, sign in branch_specs(point['tick']):
                actor = anchor if arm == 'anchor' else probes[(model_seed, arm, k, sign)]
                phase = f'screen/{point["seed"]}/{point["tick"]}/{model_seed}/{arm}/{k}/{sign}/{repeat}'
                event('BEGIN')
                path = screen_dir / f'{point["seed"]}_{point["tick"]}_{model_seed}_{arm}_{k}_{sign}_{repeat}.npz'
                row = {'model_seed': model_seed, 'arm': arm, 'repeat': repeat, 'scale_index': k, 'sign': sign,
                       'seed': point['seed'], 'tick': point['tick'], 'status': 'INCOMPLETE', 'steps': 0,
                       'trace_path': str(path.relative_to(output))}
                report['short_branches'].append(row)
                physical_branch(env, actor, point, path, row)
                event('SHORT_BRANCH_COMPLETE')
        validate_counts(env.counters(), SCREEN_STEPS)
        report['screen'] = screen_compare(report['short_branches'])
        write_json(output / 'screen_comparison.json', report['screen'])
        report['multiscale_diagnostics'] = diagnostics(report['short_branches'])
        write_json(output / 'multiscale_diagnostics.json', report['multiscale_diagnostics'])
        report['cross_readback'] = cross_readback(report['predictions'], report['short_branches'],
                                                 report['validation_ad'], output)
        write_json(output / 'cross_readback.json', report['cross_readback'])

        def run_full():
            event('WHOLE_SCREEN_PASSED')
            evaluate(env, anchor, actors, helpers, output, report, event, set_phase, save_evaluation_prefix)
            validate_counts(env.counters(), FULL_STEPS)
            report['comparison'] = compare(report['baseline'], report['candidates'])

        expected_steps = conditional_evaluation(report, run_full)
        validate_counts(env.counters(), expected_steps)
        if report['evaluation_status'] == 'NOT_RUN':
            event('HOLDOUT_NOT_RUN')
        require(report['model_calls'] == MODEL_CALLS, '81 model forward count differs')
        require(mapping_sha256(anchor.state_dict()) == anchor_sha, '81 original actor changed')
        for row in report['models']:
            require(mapping_sha256(models[row['seed']].state_dict()) == row['source_model_sha256'], '81 source model changed')
        for row in sealed:
            actor = probes[(row['model_seed'], row['arm'], row['scale_index'], row['sign'])]
            require(mapping_sha256(actor.state_dict()) == row['mapping_sha256'], '81 sealed probe mutated')
        event('EXECUTION_COMPLETE')

    except BaseException:
        failed_body = True
        with (output / 'failure.txt').open('x') as f:
            f.write(traceback.format_exc())
        raise
    finally:
        errors = []
        if env is not None:
            report['actual_counts'], report['physics_steps'] = env.counters(), env.counters()['policy_transitions']
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must preserve the prefix.
                errors.append(f'env.close: {exc}')
        for stream in (physics, progress):
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001 - attempt all cleanup operations.
                errors.append(f'journal.close: {exc}')
        try:
            accounting = recover_updates(output)
            report.update(update_accounting=accounting, policy_network_updates=accounting['policy_network_updates'])
        except Exception as exc:  # noqa: BLE001 - never discard physics accounting.
            errors.append(f'update accounting: {exc}')
        report['raw_bytes_before_report'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        report['raw_artifacts_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
        report['cleanup_errors'] = errors
        if errors and not failed_body:
            raise RuntimeError('; '.join(errors))

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
    print(json.dumps({k: report.get(k) for k in ('status', 'stage', 'outcome', 'failure', 'actual_counts',
        'policy_network_updates', 'model_updates', 'cpu_seconds', 'wall_seconds')}, allow_nan=False), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
