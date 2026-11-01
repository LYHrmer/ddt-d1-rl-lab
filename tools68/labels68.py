"""68: train-only admission of body-vx/vy responses from all58+62 raw branches.

No checkpoint, heldout trajectory, dynamics forward, actor, fit or physics is
used. Body velocity is transformed in float64 before differencing. Original yaw
qualification admits source integrity only; every body channel has its own four
scale test. Negative and unresolved rows remain in the complete output dataset.
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
from tools53.run_pilot53 import require, sha, write_json

SOURCES = {'58': Path('evidence/train_action_response58/collection_01'),
           '62': Path('evidence/clean_history62/execution_01')}
SEEDS = (53103, 53106, 53109, 53112)
TICKS = (110, 260, 410)
EPSILONS = (.000003125, .00000625, .0000125, .000025)
CHANNELS = ('vx', 'vy')
PLAN53 = Path('docs/experiments53/model_gradient53_preregistered.json')
PLAN64 = Path('docs/experiments64/mixed_sign64_preregistered.json')


def required_inputs():
    paths = {str(PLAN53), str(PLAN64), 'tools68/labels68.py', 'tools64/fit64.py', 'tools59/fit59.py',
             'tools56/fit56.py', 'tools52/dynamics52.py', 'tools54/numerics54.py',
             'tools58/collect_response58.py', 'tools53/run_pilot53.py', 'tools55/run55.py',
             'tools55/probe_local55.py', 'tools53/collection53.py', 'tools52/control52.py',
             'tools52/adapter52.py', 'tools52/run_pilot52.py', 'tools33/reference_shape33.py'}
    for directory in SOURCES.values():
        paths.add(str(directory / 'report.json'))
        paths.update(str(directory / f'branch_{point:02d}_{branch:02d}.npz') for point in range(12) for branch in range(10))
        paths.update(str(directory / f'response_{point:02d}.npz') for point in range(12))
    return paths


def verify68(plan):
    """Check selected old64 bindings, without hashing any old model/heldout data."""
    require(plan['schema'] == 'd1-body-response68-admission-v1', 'wrong68 protocol')
    require(plan['sources'] == list(SOURCES) and plan['train_seeds'] == list(SEEDS) and plan['ticks'] == list(TICKS)
            and plan['eps'] == list(EPSILONS) and plan['channels'] == list(CHANNELS), '68 source/grid changed')
    require(plan['cpu_seconds'] == 60 and plan['wall_seconds'] == 120 and plan['expected_label_rows'] == 96
            and plan['expected_states'] == 24 and plan['expected_state_channels'] == 48
            and plan['new_physics_steps'] == plan['model_updates'] == plan['policy_updates'] == 0, '68 budget changed')
    require(required_inputs() <= set(plan['source_sha256']), 'required68 sources/raw inputs not SHA-bound')
    raw_roots = tuple(str(directory) + '/' for directory in SOURCES.values())
    for name, digest in plan['source_sha256'].items():
        require(Path(name).suffix not in ('.pt', '.pth', '.zip', '.pkl'), '68 must not read checkpoints or snapshots')
        require(not name.startswith('evidence/') or name.startswith(raw_roots), '68 must not read heldout evidence')
        require(sha(ROOT / name) == digest, '68 source/input changed: ' + name)
    old64 = json.loads((ROOT / PLAN64).read_text())
    original = json.loads((ROOT / PLAN53).read_text())
    require(old64['schema'] == 'd1-mixed-sign64-v1'
            and original['schema'] == 'd1-model-gradient53-preregistration-v1', 'old plan identity differs')
    for name in required_inputs():
        if name.startswith(raw_roots):
            require(name in old64['source_sha256'], 'training raw input lacks original64 binding: ' + name)
        if name in old64['source_sha256']:
            require(plan['source_sha256'][name] == old64['source_sha256'][name], 'bound64 source/raw identity changed: ' + name)
        if name in original['identities']:
            identity = original['identities'][name]
            require(plan['source_sha256'][name] == identity['sha256'] and (ROOT / name).stat().st_size == identity['bytes'],
                    'original53 source identity changed: ' + name)
    return original


def body_velocity(states):
    """Raw float64 state[...,45] -> float64 body linear velocity[...,2], m/s."""
    import numpy as np
    import torch

    from tools54 import numerics54 as numeric
    state = np.asarray(states)
    require(state.dtype == np.float64 and state.shape[-1] == 45 and np.isfinite(state).all(),
            'raw finite float64 state45 required before body transform')
    require(bool((np.linalg.norm(state[..., 3:7], axis=-1) > 1e-9).all()), 'nonzero quaternion required')
    with torch.no_grad():
        value = numeric._rotate_inverse(torch.as_tensor(state[..., 3:7], dtype=torch.float64),
                                        torch.as_tensor(state[..., 23:26], dtype=torch.float64))[..., :2]
    return value.numpy()


def body_response(post_states):
    """All10 poststates -> all four body FDs and independent vx/vy qualification."""
    import numpy as np

    from tools58.collect_response58 import yaw_stability
    state = np.asarray(post_states)
    require(state.shape == (10, 45) and state.dtype == np.float64, 'all10 float64 poststates required')
    require(np.array_equal(state[0], state[1]), 'repeated zero poststate differs')
    velocity = body_velocity(state)
    central, forward, backward = [], [], []
    for ep, epsilon in enumerate(EPSILONS):
        plus, minus = velocity[2 + 2 * ep:4 + 2 * ep]
        central.append((plus - minus) / (2 * epsilon))
        forward.append((plus - velocity[0]) / epsilon)
        backward.append((velocity[0] - minus) / epsilon)
    arrays = {'post_state45': state.copy(), 'post_body_vxy': velocity,
              'central_body_vxy_fd': np.asarray(central), 'forward_body_vxy_fd': np.asarray(forward),
              'backward_body_vxy_fd': np.asarray(backward)}
    rows = {}
    for column, channel in enumerate(CHANNELS):
        fd = [{'epsilon': epsilon, 'central': float(arrays['central_body_vxy_fd'][ep, column]),
               'forward': float(arrays['forward_body_vxy_fd'][ep, column]),
               'backward': float(arrays['backward_body_vxy_fd'][ep, column])} for ep, epsilon in enumerate(EPSILONS)]
        rows[channel] = {'all_four_eps': fd, 'stability': yaw_stability(fd), 'units': '(m/s) per absolute action delta'}
    return rows, arrays


def qualification(points):
    """Require all24 complete states and all48 channel qualifications; never filter."""
    expected = [(source, seed, tick) for source in SOURCES for seed in SEEDS for tick in TICKS]
    require([(row['source'], row['seed'], row['tick']) for row in points] == expected
            and all(row['status'] == 'COMPLETE' for row in points), 'all24 complete ordered points required')
    stable = sum(bool(row['channels'][channel]['stability']['stable']) for row in points for channel in CHANNELS)
    return {'status': 'ADMITTED' if stable == 48 else 'REJECTED_INCOMPLETE_BODY_RESPONSE_QUALIFICATION',
            'admitted': stable == 48, 'qualified_state_channels': stable, 'required_state_channels': 48,
            'states': 24, 'label_rows': 96, 'rows_dropped': 0, 'selected_epsilon': None,
            'old_yaw_qualification_is_body_qualification': False, 'fits_authorized_by_this_result': 0}


def assemble_labels(yaw_sources, point_rows, point_arrays):
    """Keep all96 rows, raw F64 body targets and per-channel unknown flags."""
    import numpy as np
    qualification(point_rows)  # Complete rows required; a rejected body result is retained.
    require(set(yaw_sources) == set(SOURCES) and len(point_arrays) == 24, 'both complete label sources required')
    keys = ('s', 'a0', 'a_plus', 'a_minus', 'v', 'seed', 'tick', 'epsilon')
    labels = {key: np.concatenate([yaw_sources[source][key] for source in SOURCES]) for key in keys}
    labels['target_yaw'] = np.concatenate([yaw_sources[source]['target'] for source in SOURCES])
    labels['target_body'] = np.concatenate([row['central_body_vxy_fd'] for row in point_arrays])
    labels['source_id'] = np.repeat(np.array([58, 62], dtype=np.int64), 48)
    labels['source_point_index'] = np.tile(np.repeat(np.arange(12), 4), 2)
    labels['body_channel_qualified'] = np.repeat(np.array([
        [row['channels'][channel]['stability']['stable'] for channel in CHANNELS] for row in point_rows], dtype=np.bool_), 4, axis=0)
    require(labels['target_body'].shape == (96, 2) and labels['target_body'].dtype == np.float64
            and labels['s'].shape == (96, 45) and len(np.unique(labels['s'], axis=0)) == 24,
            'complete96 raw labels from24 distinct states required')
    require(all(len(value) == 96 and np.isfinite(value).all() for value in labels.values()), 'invalid assembled labels')
    return labels


def read_point(directory, source, index, yaw_labels, source_point, *, arrays=None):
    """Read all10 branch states and exactly recover original yaw FD before body FD."""
    import numpy as np
    arrays = {} if arrays is None else arrays
    branches = []
    for order in range(10):
        with np.load(directory / f'branch_{index:02d}_{order:02d}.npz', allow_pickle=False) as raw:
            branch = {key: raw[key].copy() for key in ('s', 'applied_action', 'target')}
        require(branch['target'].dtype == np.float64 and branch['target'].shape == (1, 45), 'original F64 poststate required')
        arrays[f'branch_{order}_post_state45'] = branch['target']
        branches.append(branch)
    with np.load(directory / f'response_{index:02d}.npz', allow_pickle=False) as raw:
        original = {key: raw[key].copy() for key in ('central_state45_fd', 'forward_state45_fd', 'backward_state45_fd')}
    zero = branches[0]['target'][0]
    for ep, epsilon in enumerate(EPSILONS):
        plus, minus = [branch['target'][0] for branch in branches[2 + 2 * ep:4 + 2 * ep]]
        for name, value in (('central_state45_fd', (plus - minus) / (2 * epsilon)),
                            ('forward_state45_fd', (plus - zero) / epsilon), ('backward_state45_fd', (zero - minus) / epsilon)):
            require(np.array_equal(value, original[name][ep]), 'original raw45 response differs: ' + name)
        require(float(original['central_state45_fd'][ep, 28]) == float(yaw_labels['target'][4 * index + ep]),
                'old yaw training target differs')
        raw_yaw = source_point['yaw28'][ep]
        require(raw_yaw['epsilon'] == epsilon and all(float(original[name][ep, 28]) == raw_yaw[key]
                for name, key in (('central_state45_fd', 'central'), ('forward_state45_fd', 'forward'),
                                   ('backward_state45_fd', 'backward'))), 'original yaw JSON differs')
    channels, body = body_response(np.stack([branch['target'][0] for branch in branches]))
    arrays.update(body)
    arrays.update({'original_' + key: value for key, value in original.items()})
    return {'source': source, 'seed': source_point['seed'], 'tick': source_point['tick'], 'status': 'COMPLETE',
            'channels': channels, 'old_yaw_response_exact': True, 'old_yaw_qualification_changed': False,
            'old_yaw_qualification_used_for_body': False}, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-body-response68-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'new_physics_steps': 0, 'model_updates': 0, 'policy_updates': 0, 'fits': 0,
              'model_inferences': 0, 'checkpoints_read': 0, 'heldout_rows_read': 0, 'completed_points': 0,
              'admission': {'status': 'INCOMPLETE', 'admitted': False}, 'source_decisions_changed': False,
              'points': [{'source': source, 'seed': seed, 'tick': tick, 'status': 'NOT_RUN'}
                         for source in SOURCES for seed in SEEDS for tick in TICKS]}
    write_json(args.output / 'initial_incomplete.json', report)
    cpu, wall = time.process_time(), time.monotonic()
    plan, original = None, None
    progress = {'phase': 'admission', 'source': None, 'point': None}
    events = (args.output / 'events.jsonl').open('x', buffering=1)

    def event(name):
        events.write(json.dumps({'event': name, **progress, 'completed_points': report['completed_points'],
                                'cpu_seconds': time.process_time() - cpu}, allow_nan=False) + '\n')
        events.flush()

    def interrupted(signum, _frame):
        raise SystemExit(f'interrupted by signal {signum}')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
        signal.signal(signum, interrupted)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (59, 60))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1,
                'Nice10/single CPU required')
        require(all(os.environ.get(key) == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
                and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'thread/GPU restrictions differ')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        original = verify68(plan)
        report['versions'] = {name: importlib.metadata.version(name) for name in original['versions']}
        require(report['versions'] == original['versions'], 'original53 package versions differ')
        import numpy as np
        import torch

        from tools64 import fit64
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        report['torch_runtime_version'] = torch.__version__
        sources, point_arrays = {}, []
        for source, directory in SOURCES.items():
            progress.update(phase='load_source', source=source, point=None)
            require(time.process_time() < 58., 'CPU deadline before source')
            old = json.loads((ROOT / directory / 'report.json').read_text())
            require(old['status'] == old['verification_after'] == 'COMPLETE'
                    and old['versions'] == report['versions'] and old['torch_runtime_version'] == torch.__version__,
                    'complete version-matched source required')
            sources[source] = (fit64.load_labels58 if source == '58' else fit64.load_labels62)(ROOT / directory)
            output = args.output / f'source{source}'
            output.mkdir(exist_ok=False)
            for index in range(12):
                require(time.process_time() < 58., 'CPU deadline before point')
                progress.update(phase='body_response', point=index)
                row = report['points'][report['completed_points']]
                row['status'] = 'INCOMPLETE'
                arrays = {}
                event('POINT_BEGIN')
                try:
                    result, _ = read_point(ROOT / directory, source, index, sources[source], old['points'][index], arrays=arrays)
                    row.update(result)
                    point_arrays.append(arrays)
                    report['completed_points'] += 1
                except BaseException as exc:
                    row.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    with (output / f'point_{index:02d}.npz').open('xb') as stream:
                        np.savez_compressed(stream, **arrays)
                    write_json(output / f'point_{index:02d}.json', row)
                    event('POINT_SAVED')
        labels = assemble_labels(sources, report['points'], point_arrays)
        with (args.output / 'body_labels96.npz').open('xb') as stream:
            np.savez_compressed(stream, **labels)
        report.update(admission=qualification(report['points']), label_rows=96, states=24,
                      state_count_semantics='distinct exact states, not independent episodes',
                      label_file_sha256=sha(args.output / 'body_labels96.npz'), status='COMPLETE')
    except BaseException as exc:  # noqa: BLE001 - incomplete and rejected outcomes remain distinct and durable.
        report['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['last_phase'] = dict(progress)
        try:
            if plan is not None:
                require(sha(args.plan) == report['plan_sha256'], '68 plan changed')
                verify68(plan)
            if original is not None:
                report['versions_after'] = {name: importlib.metadata.version(name) for name in original['versions']}
                require(report['versions_after'] == original['versions'], 'package versions changed')
            report['verification_after'] = 'COMPLETE' if original is not None else 'NOT_RUN'
        except Exception as exc:  # noqa: BLE001 - keep original and final verification failures.
            report.update(status='INCOMPLETE', verification_after='FAILED', verification_failure=str(exc))
        if report['status'] != 'COMPLETE':
            report['admission'] = {'status': 'INCOMPLETE', 'admitted': False}
        report['cpu_seconds'] = time.process_time() - cpu
        report['wall_seconds'] = time.monotonic() - wall
        event('FINAL_REPORT')
        events.close()
        write_json(args.output / 'report.json', report)
        print(json.dumps({key: report[key] for key in ('status', 'failure', 'completed_points', 'admission', 'cpu_seconds')}, indent=2), flush=True)
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
