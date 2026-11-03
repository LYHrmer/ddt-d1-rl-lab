"""Approved fixed-data sampling comparison, with a conditional physical screen."""
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
ARMS = ('legacy', 'steady')
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
DEV_SEEDS = {task: tuple(range(80101 + 3 * i, 80104 + 3 * i)) for i, task in enumerate(TASKS)}
EVAL_SEEDS = (80401, 80402, 80403)
OLD_TICKS, STEADY_TICKS = (110, 260, 410), (175, 325, 410)
TURN_TICKS = (110, 175, 260, 325, 410)
PLAN53 = 'docs/experiments53/model_gradient53_preregistered.json'
PLAN79 = 'docs/experiments79/feedback79_preregistered.json'
SOURCE75 = 'evidence/world_vz75/execution_01'
FULL_STEPS, SCREEN_STEPS = 47040, 9240


def verify(plan):
    require(plan['schema'] == 'd1-steady-sampling80-preregistered-v1', 'wrong80 schema')
    expected = {'model_seeds': list(SEEDS), 'arms': list(ARMS),
        'development_seeds': {k: list(v) for k, v in DEV_SEEDS.items()},
        'evaluation_seeds': list(EVAL_SEEDS), 'legacy_ticks': list(OLD_TICKS),
        'steady_turn_ticks': list(STEADY_TICKS), 'starts_per_update': 27,
        'updates_per_arm': 2, 'policy_updates': 12, 'model_updates': 0,
        'preflight_policy_steps': 224, 'development_policy_steps': 5400,
        'screen_policy_steps': 3840, 'evaluation_policy_steps': 37800,
        'formal_policy_cap': FULL_STEPS, 'cpu_seconds': 900, 'wall_seconds': 1200,
        'artifact_cap_bytes': 128 << 20,
        'user_confirmation': 'approved 2026-11-03 before implementation'}
    require(all(plan.get(k) == v for k, v in expected.items()), '80 authorized settings differ')
    inherited = json.loads((ROOT / PLAN79).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(k) == v for k, v in inherited.items()), '79 source closure changed')
    required = {PLAN53, PLAN79, SOURCE75 + '/report.json',
        'docs/steady_sampling80_proposal_20261103.md', 'docs/steady_sampling80_execution_20261103.md',
        'reports/steady_sampling80_source_feasibility_20261103.json', '.github/workflows/steady-sampling80.yml',
        *(f'tools80/{n}80.py' for n in ('run', 'policy', 'compare')),
        *(f'tests/test_{n}80.py' for n in ('run', 'policy', 'compare'))}
    require(required <= plan['source_sha256'].keys(), '80 sources not bound')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '80 source/input changed: ' + name)


def select_points(points, arm):
    require(arm in ARMS, 'unknown sampling arm')
    expected = {(task, seed, tick) for task, seeds in DEV_SEEDS.items() for seed in seeds
                for tick in (TURN_TICKS if task == 'turn_stop' else OLD_TICKS)}
    identities = [(p['task'], p['seed'], p['tick']) for p in points]
    require(len(identities) == len(set(identities)) == 33 and set(identities) == expected,
            'incomplete/duplicate80 live snapshot grid')
    result = [p for p in points if p['tick'] in
              (STEADY_TICKS if arm == 'steady' and p['task'] == 'turn_stop' else OLD_TICKS)]
    require(len(result) == 27, 'sampling arm must contain27 points')
    return sorted(result, key=lambda p: (TASKS.index(p['task']), p['seed'], p['tick']))


def array_identity(arrays):
    """Hash complete scalar traces, excluding monotonic global work counters."""
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def recover_updates(output):
    receipts, completed = [], 0
    for path in sorted(Path(output).glob('seed*/actor_*/receipt.json')):
        require(path.parent.parent.name in {f'seed{s}' for s in SEEDS}
                and path.parent.name in {f'actor_{a}_{i}' for a in ARMS for i in (1, 2)},
                'unregistered80 update stage')
        value = json.loads(path.read_text())
        n, attempted = value['updates_completed'], value['updates_attempted']
        require(type(n) is int and type(attempted) is int and 0 <= n <= attempted <= 1,
                'invalid80 update accounting')
        completed += n
        receipts.append({'path': str(path.relative_to(output)), 'status': value['status'],
            'updates_completed': n, 'updates_attempted': attempted, 'sha256': sha(path)})
    require(completed <= 12, '80 update cap exceeded')
    return {'policy_network_updates': completed, 'model_updates': 0, 'receipts': receipts,
            'semantics': 'durable receipts; hard-kill journals may establish a larger lower bound'}


def revoke(report, reason):
    report['global_candidate_accepted'] = False
    report['sampling_passed'] = False
    for key in ('screen', 'comparison'):
        previous = report.get(key, {})
        if previous.get('status') == 'COMPLETE':
            report['provisional_' + key + '_before_revocation'] = copy.deepcopy(previous)
            previous.update(status='INCOMPLETE', revocation_reason=reason, passed=False,
                            enter_full_evaluation=False, sampling_passed=False)
            previous.get('sampling', {})['passed'] = False
            for gate in previous.get('control_gates', []):
                gate['passed'] = False


def conditional_evaluation(report, callback):
    """All three learners enter the untouched holdout together, or none do."""
    require(report['screen'].get('status') == 'COMPLETE', 'screen is incomplete')
    if report['screen'].get('enter_full_evaluation') is not True:
        report.update(outcome='SCREEN_REJECTED', evaluation_status='NOT_RUN', sampling_passed=False,
                      evaluation_not_run_reason='registered whole-three-model screen failed; no seed selection')
        require(not report['baseline'] and not report['candidates'], 'holdout was touched before screen rejection')
        return SCREEN_STEPS
    callback()
    require(report['comparison'].get('status') == 'COMPLETE', 'full comparison incomplete')
    report.update(outcome='FULL_EVALUATION_COMPLETE', evaluation_status='COMPLETE',
                  sampling_passed=report['comparison']['sampling_passed'])
    return FULL_STEPS


def verify_final(report, plan, plan_path, output):
    try:
        require(sha(plan_path) == plan['_sha256'], '80 plan changed during run')
        verify(plan)
        if 'versions' in report:
            require({k: importlib.metadata.version(k) for k in report['versions']} == report['versions'],
                    '80 numeric versions changed')
        for name, digest in report.get('raw_artifacts_sha256', {}).items():
            require(sha(output / name) == digest, '80 generated artifact changed: ' + name)
        if 'sealed_policies_sha256' in report:
            path = output / 'policies_sealed_before_screen.json'
            require(sha(path) == report['sealed_policies_sha256'], '80 policy seal changed')
            for row in json.loads(path.read_text())['policies']:
                require(sha(output / row['policy_file']) == row['file_sha256'], 'sealed80 policy changed')
        for key, name in [('development_seal_sha256', 'development_sealed_before_training.json'),
                          ('prediction_seal_sha256', 'predictions_sealed_before_physics.json')]:
            if key in report:
                path = output / name
                require(sha(path) == report[key], '80 original seal changed: ' + name)
                payload = json.loads(path.read_text())
                members = payload['artifact_sha256'] if key == 'development_seal_sha256' else {
                    row['trace_path']: row['trace_sha256'] for row in payload}
                for member, digest in members.items():
                    require(sha(output / member) == digest, '80 sealed member changed: ' + member)
        accepted = report.get('admission', {})
        if 'path' in accepted and 'sha256' in accepted:
            require(sha(Path(accepted['path'])) == accepted['sha256'], '80 admission receipt changed')
        require(sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file()) < plan['artifact_cap_bytes'],
                '80 final artifact cap exceeded')
    except BaseException as exc:
        revoke(report, f'final verification failed: {type(exc).__name__}: {exc}')
        raise
    finally:
        if report['status'] != 'COMPLETE':
            revoke(report, 'execution incomplete')


def collect_episode(env, actor, task, seed, spec, folder, receipt):
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
        ticks = TURN_TICKS if task == 'turn_stop' else OLD_TICKS
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
                    '80 development trajectory unsafe/incomplete')
        require([(p['seed'], p['tick']) for p in points] == [(seed, tick) for tick in ticks], '80 point capture differs')
        validate_counts({k: env.counters()[k] - before[k] for k in before}, 600)
        receipt['status'] = 'COMPLETE'
        return points
    except BaseException as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        path = folder / 'full_or_partial_trace.npz'
        save_npz(path, collection53._arrays(rows))
        save_points(folder / 'integration_context_archive.pkl', points, env)
        context = {}
        for index, point in enumerate(points):
            context.update({f'{index}_{name}': np.asarray(point[name]) for name in
                ('state', 'frame', 'history', 'filtered', 'last', 'fifo', 'commands')})
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
    try:
        shaper = ReferenceShaper()
        adapter52.restore(env, shaper, point['snapshot'])
        for tick in range(32):
            require(np.array_equal(env._command, point['commands'][tick]), '80 branch command clock differs')
            action = control52.real_action(env, shaper, actor, 0.)
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
                    '80 short branch unsafe/incomplete')
        arrays = {k: np.asarray(v) for k, v in data.items()}
        require(all(np.isfinite(v).all() for v in arrays.values()), 'nonfinite80 physical trace')
        cost, costs = cost64(point, arrays['states'], arrays['observations'])
        require(np.array_equal(arrays['commands'], point['commands'][:32]), '80 applied command trace differs')
        receipt.update(status='COMPLETE', cost=cost, per_step_cost64=costs,
            yaw_mae=float(np.mean(np.abs(arrays['body_wz'] - arrays['commands'][:, 2]))),
            trace_identity_sha256=array_identity(arrays), fallen=False, illegal_body_contact=False)
        validate_counts({k: env.counters()[k] - before[k] for k in before}, 32)
    except BaseException as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        arrays = {k: np.asarray(v) for k, v in data.items()}
        save_npz(path, arrays)
        counts = {k: env.counters()[k] - before[k] for k in before}
        receipt.update(actual_counts=counts, native_steps=counts['native_steps'], pd_updates=counts['pd_updates'],
                       trace_sha256=sha(path), trace_file=path.name)
        write_json(path.with_suffix('.json'), receipt)


def calculate(report, plan, output, preflight, admission):
    report.update(schema='d1-steady-sampling80-result-v1', stage='preflight' if preflight else 'experiment',
        plan_sha256=plan['_sha256'], user_confirmation=plan['user_confirmation'],
        global_candidate_accepted=False, complete_model_eligibility='NOT_REQUALIFIED',
        inherited75_and78_decisions_changed=False, no_critic=True, terminal_value=0,
        model_seed_semantics='three paired world-model initializations; not independent PPO retraining',
        models=[], development=[], predictions=[], short_branches=[], baseline=[], candidates=[],
        comparison={'status': 'NOT_RUN'}, screen={'status': 'NOT_RUN'}, sampling_passed=False,
        evaluation_status='NOT_RUN', policy_network_updates=0, model_updates=0, actual_counts=None,
        admission={'status': 'NOT_RUN'})
    verify(plan)
    resource.setrlimit(resource.RLIMIT_CPU, (119, 120) if preflight else (899, 900))
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
            '80 Nice10/single CPU required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
        '80 single-thread CPU/no GPU required')
    started, cpu_deadline = time.monotonic(), time.process_time() + (110 if preflight else 880)
    original = json.loads((ROOT / PLAN53).read_text())
    source = json.loads((ROOT / SOURCE75 / 'report.json').read_text())
    require(source['status'] == source['verification_after'] == 'COMPLETE' and source['completed_model_fits'] == 6,
            'complete75 source required')
    report['versions'] = {k: importlib.metadata.version(k) for k in original['versions']}
    require(report['versions'] == original['versions'], '80 numeric versions changed')
    import numpy as np
    import torch

    from tools52 import adapter52, control52, dynamics52
    from tools52.run_pilot52 import actor_admission, save_npz
    from tools53 import collection53
    from tools54 import numerics54
    from tools80.compare80 import compare, screen_compare
    from tools80.policy80 import mapping_sha256, update_actor

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
            require(torch.equal(anchor(x), clone(x)), '80 zero-update clone differs')
    report['zero_update_clone_bit_exact_fixtures'] = len(fixtures)
    del clone
    helpers = collection53.helpers(original)
    require(tuple(helpers.SEQUENCES) == TASKS, 'original tasks changed')
    anchor_sha = mapping_sha256(anchor.state_dict())
    report['anchor_mapping_sha256'] = anchor_sha
    env, models, actors = None, {}, {}
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
        require(time.process_time() < cpu_deadline, '80 CPU deadline reached')
        require(time.monotonic() - started < (170 if preflight else 1190), '80 wall deadline reached')
        used = sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
        require(used < plan['artifact_cap_bytes'] - (8 << 20), '80 evidence cap reserve exhausted')

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
                    '80 preflight incomplete/unbound')
            validate_counts(accepted['actual_counts'], 224)
            report['admission'] = {'status': 'COMPLETE', 'path': str(admission), 'sha256': sha(admission)}
        env = collection53.make_env(256 if preflight else FULL_STEPS, audit_step=record)
        if preflight:
            report['admission'] = collection53.preflight(env, anchor, output, original)
            report['planned_live_snapshot_count80'] = 33
            validate_counts(env.counters(), 224)
            return
        development = output / 'development'
        development.mkdir(exist_ok=False)
        points = []
        for task, seeds in DEV_SEEDS.items():
            for seed in seeds:
                phase = f'development/{task}/{seed}'
                event('BEGIN')
                row = {'task': task, 'seed': seed, 'status': 'INCOMPLETE', 'steps': 0}
                report['development'].append(row)
                points.extend(collect_episode(env, anchor, task, seed, helpers.SEQUENCES[task],
                    development / f'{task}_{seed}', row))
                event('DEVELOPMENT_EPISODE_COMPLETE')
        validate_counts(env.counters(), 5400)
        selected = {arm: select_points(points, arm) for arm in ARMS}
        report['sampling_point_ids'] = {a: [(p['task'], p['seed'], p['tick']) for p in ps] for a, ps in selected.items()}
        write_json(output / 'development_sealed_before_training.json', {'episodes': report['development'],
            'point_ids': report['sampling_point_ids'], 'actor_mapping_sha256': anchor_sha,
            'artifact_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(development.rglob('*')) if p.is_file()}})
        report['development_seal_sha256'] = sha(output / 'development_sealed_before_training.json')
        for seed in SEEDS:
            folder = output / f'seed{seed}'
            folder.mkdir(exist_ok=False)
            row = {'seed': seed, 'status': 'INCOMPLETE', 'arms': {a: [] for a in ARMS}}
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
            model.register_forward_hook(hook('model_calls'))
            models[seed] = model
            row['source_model_sha256'] = model_sha
            for arm in ARMS:
                actor = anchor
                for step in (1, 2):
                    phase = f'seed{seed}/actor_{arm}_{step}'
                    event('BEGIN')
                    actor, receipt = update_actor(model, actor, anchor, selected[arm],
                        folder / f'actor_{arm}_{step}', cpu_deadline=cpu_deadline)
                    row['arms'][arm].append(receipt)
                    require(receipt['status'] == receipt['verification_after'] == 'COMPLETE'
                            and receipt['point_count'] == 27 and receipt['updates_completed'] == 1,
                            '80 update incomplete or point count differs')
                    event('POLICY_STEP_COMPLETE')
                actors[(seed, arm)] = actor
            require(mapping_sha256(model.state_dict()) == model_sha, '80 frozen source model changed')
            row['status'] = 'COMPLETE'
            write_json(folder / 'seed_result.json', row)
        report['update_accounting'] = recover_updates(output)
        require(report['update_accounting']['policy_network_updates'] == 12, '80 learner schedule incomplete')
        sealed = [{'model_seed': s, 'arm': a, 'mapping_sha256': mapping_sha256(actors[(s, a)].state_dict()),
            'policy_file': f'seed{s}/actor_{a}_2/actor_state_dict.pt',
            'file_sha256': sha(output / f'seed{s}/actor_{a}_2/actor_state_dict.pt')} for s in SEEDS for a in ARMS]
        require(all(p['mapping_sha256'] != anchor_sha for p in sealed), '80 unchanged actor')
        write_json(output / 'policies_sealed_before_screen.json', {'policies': sealed, 'baseline_mapping_sha256': anchor_sha})
        report['sealed_policies_sha256'] = sha(output / 'policies_sealed_before_screen.json')
        event('ALL_SIX_POLICIES_SEALED')
        turn_points = [p for p in points if p['task'] == 'turn_stop']
        prediction_dir = output / 'predictions'
        prediction_dir.mkdir(exist_ok=False)
        for seed in SEEDS:
            model64 = copy.deepcopy(models[seed]).double().eval().requires_grad_(False)
            for arm, actor in [('anchor', anchor)] + [(a, actors[(seed, a)]) for a in ARMS]:
                actor64 = copy.deepcopy(actor).double().eval().requires_grad_(False)
                for point in turn_points:
                    phase = f'prediction/{seed}/{arm}/{point["seed"]}/{point["tick"]}'
                    limits()
                    with torch.no_grad():
                        value, trace = numerics54.imaginary(model64, actor64, point,
                            torch.zeros((), dtype=torch.float64), return_trace=True, horizon=32)
                    arrays = {k: v.numpy() for k, v in trace.items()}
                    path = prediction_dir / f'{seed}_{arm}_{point["seed"]}_{point["tick"]}.npz'
                    save_npz(path, arrays)
                    row = {'model_seed': seed, 'arm': arm, 'seed': point['seed'], 'tick': point['tick'],
                        'cost': float(value), 'yaw_mae': float(np.mean(np.abs(arrays['states'][:, 28] - point['commands'][:32, 2]))),
                        'trace_path': str(path.relative_to(output)), 'trace_sha256': sha(path),
                        'precision': 'F64 model and exact F32 actor weights promoted to F64; not a physical derivative'}
                    report['predictions'].append(row)
        write_json(output / 'predictions_sealed_before_physics.json', report['predictions'])
        report['prediction_seal_sha256'] = sha(output / 'predictions_sealed_before_physics.json')
        event('ALL_PREDICTIONS_SEALED')
        screen_dir = output / 'screen'
        screen_dir.mkdir(exist_ok=False)
        jobs = [(None, 'anchor', repeat, anchor) for repeat in (0, 1)] + [(s, a, 0, actors[(s, a)]) for s in SEEDS for a in ARMS]
        for point in turn_points:
            for model_seed, arm, repeat, actor in jobs:
                phase = f'screen/{point["seed"]}/{point["tick"]}/{model_seed}/{arm}/{repeat}'
                event('BEGIN')
                row = {'model_seed': model_seed, 'arm': arm, 'repeat': repeat, 'seed': point['seed'],
                       'tick': point['tick'], 'status': 'INCOMPLETE', 'steps': 0}
                report['short_branches'].append(row)
                path = screen_dir / f'{point["seed"]}_{point["tick"]}_{model_seed}_{arm}_{repeat}.npz'
                physical_branch(env, actor, point, path, row)
                row['trace_path'] = str(path.relative_to(output))
                event('SHORT_BRANCH_COMPLETE')
        validate_counts(env.counters(), SCREEN_STEPS)
        report['screen'] = screen_compare(report['short_branches'])
        write_json(output / 'screen_comparison.json', report['screen'])

        def run_full():
            event('WHOLE_SCREEN_PASSED')
            evaluate(env, anchor, actors, helpers, output, report, event, set_phase, save_evaluation_prefix)
            validate_counts(env.counters(), FULL_STEPS)
            report['comparison'] = compare(report['baseline'], report['candidates'])

        expected_steps = conditional_evaluation(report, run_full)
        validate_counts(env.counters(), expected_steps)
        if report['evaluation_status'] == 'NOT_RUN':
            event('HOLDOUT_NOT_RUN')
        require(report['model_calls'] == 35424, '80 model forward count differs')
        require(mapping_sha256(anchor.state_dict()) == anchor_sha, '80 original actor changed')
        for row in report['models']:
            require(mapping_sha256(models[row['seed']].state_dict()) == row['source_model_sha256'], '80 source model changed')
        for row in sealed:
            require(mapping_sha256(actors[(row['model_seed'], row['arm'])].state_dict()) == row['mapping_sha256'], '80 sealed actor mutated')
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

def evaluate(env, anchor, actors, helpers, output, report, event, set_phase, save_prefix):
    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import control52
    from tools80.policy80 import mapping_sha256

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
                            and metrics['done_reason'] == 'HORIZON_12S', '80 evaluation incomplete/unsafe')
                    validate_counts(metrics['actual_counts'], 600)
                    require(mapping_sha256(actor.state_dict()) == policy_sha, '80 evaluation changed actor')
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
