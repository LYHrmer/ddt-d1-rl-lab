#!/usr/bin/env python3
"""Judge contract 10 only after its own supervisor proves full final-policy training.

This reads receipts and arrays without constructing a trainer, model or simulator. A failed
or partial training never causes evaluation or replay receipts to be opened.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools12', 'tools10', 'tools08'):
    sys.path.insert(0, str(ROOT/sub))

import release_contract10 as release_tools  # noqa: E402
import supervise_scale10 as supervisor  # noqa: E402
import replay_training_episodes08 as replay_tools  # noqa: E402
import verify_scale08 as gates  # noqa: E402
import describe_eval08 as describe  # noqa: E402

sha, require = release_tools.sha, release_tools.require
ENVIRONMENT_SOURCES = (
    'training/d1_rl/factory03.py', 'training/d1_rl/__init__.py', 'training/d1_rl/env.py',
    'training/d1_rl/interface.py', 'training/d1_rl/fast_checks.py', 'training/d1_rl/task03c.py',
    'training/configs/pilot03b.json', 'training/d1_rl/task03d.py', 'training/configs/pilot03c.json',
)


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluator_thresholds(root):
    """Read the frozen evaluator's literal without importing its model dependencies."""
    tree = ast.parse((root/'tools/evaluate_d1_ppo.py').read_text())
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'THRESHOLDS' for target in node.targets)]
    require(len(values) == 1 and isinstance(values[0], dict), 'frozen evaluator has no unique threshold literal')
    return values[0]


def load_release(path, root):
    require(Path(path).resolve() == root/release_tools.RELEASE_PATH, 'wrong contract 10 release path')
    release = supervisor.load_release(path, root)
    require(release.get('plans') == release_tools.PLANS and release.get('seed') == 101
            and release.get('envs') == 8
            and release['resources']['start_min_available_ram_bytes'] == 7*1024**3,
            'release differs from the approved contract 10 plan')
    release_tools.check_frozen_inputs(release, root)
    require(release['frozen_inputs'].get('tools12/verify_contract10.py') == sha(__file__),
            'judge identity differs from the release')
    for key in ('budget_authorisation', 'static_report'):
        evidence = release[key]
        path = (root/evidence['path']).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path) == evidence['sha256'],
                f'changed release evidence: {key}')
    release_tools.check_authorisation(read_json(root/release['budget_authorisation']['path']))
    return release


def equivalence_complete(report, plan):
    return (report.get('equivalence') == 'PASS' and report.get('first_mismatch') is None
            and type(report.get('rollouts_compared')) is int
            and type(report.get('rollouts_required')) is int
            and report['rollouts_compared'] == report['rollouts_required'] == plan['compare_rollouts'])


def training_complete(report, release, actual_result):
    """Supervisor PASS, exact E1, raw result and all worker counts must agree."""
    return (report.get('status') == 'PASS' and report.get('training_started') is True
            and type(report.get('trainer_returncode')) is int and report['trainer_returncode'] == 0
            and report.get('inputs_unchanged') is True and report.get('cleanup_complete') is True
            and report.get('stop_reason') is None and report.get('error') is None
            and equivalence_complete(report, release['plans']['scale'])
            and report.get('trainer_result') == actual_result
            and supervisor.result_complete(actual_result, release['plans']['scale'], release['envs']))


def source_identities(identities, release, root):
    require(isinstance(identities, dict), 'source identities are missing')
    names = set()
    for name, digest in identities.items():
        path = Path(name).resolve()
        require(path.is_relative_to(root), 'source provenance is outside this repository')
        relative = str(path.relative_to(root))
        require(release['frozen_inputs'].get(relative) == digest and sha(path) == digest,
                f'source provenance differs from frozen input: {relative}')
        names.add(relative)
    return names


def final_protocol(train, release, root):
    protocol = read_json(train/'protocol.json')
    hyper = protocol.get('hyperparameters', {})
    require(protocol.get('kind') == 'scale' and protocol.get('arm') == 'B'
            and protocol.get('seed') == release['seed'] and protocol.get('envs') == release['envs']
            and protocol.get('planned_transitions') == release['plans']['scale']['steps']
            and protocol.get('planned_native_steps') == release['plans']['scale']['steps']*10
            and protocol.get('config') == read_json(root/release['config'])
            and hyper.get('ent_coef') == 0.0 and hyper.get('random_initialization') is True
            and hyper.get('teacher_policy_used') is False and hyper.get('warm_start') is False,
            'final training protocol differs from this release')
    names = source_identities(protocol.get('inputs_sha256'), release, root)
    require({release['entry'], release['config']} <= names, 'training protocol lacks frozen entry/config provenance')
    require((train/'policy.zip').is_file() and not any('vecnormalize' in p.name.lower() for p in train.iterdir()),
            'a final policy.zip with no normalization wrapper is required')
    return protocol


def validate_replay(path, train, release, protocol):
    replay = read_json(path)
    envs = release['envs']
    expected_inputs = {f'env_{rank:02d}.monitor.csv': sha(train/f'env_{rank:02d}.monitor.csv')
                       for rank in range(envs)}
    require(replay.get('format') == 'd1-contract08-training-episode-replay-v1'
            and replay.get('physics_steps_executed') == 0 and replay.get('seed') == protocol['seed']
            and replay.get('envs') == envs
            and replay.get('steps_per_env') == release['plans']['scale']['steps']//envs
            and replay.get('inputs') == expected_inputs
            and replay.get('tool_sha256') == release['frozen_inputs']['tools08/replay_training_episodes08.py'],
            'replay provenance differs from this training run')
    episodes_path = Path(path).parent/'episodes.jsonl'
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    require(replay.get('episodes') == len(episodes), 'replay episode count differs from its source rows')
    require(all(type(e.get('rank')) is int and 0 <= e['rank'] < envs for e in episodes),
            'invalid replay worker rank')
    for rank in range(envs):
        actual = [e for e in episodes if e['rank'] == rank]
        monitors = replay_tools.read_monitor(train/f'env_{rank:02d}.monitor.csv')
        require(all(0 < monitor['l'] <= replay_tools.HORIZON for monitor in monitors)
                and sum(monitor['l'] for monitor in monitors) <= replay['steps_per_env'],
                'monitor exceeds the frozen per-worker budget or episode horizon')
        # join calls only the frozen environment's course method on a SimpleNamespace.
        # It builds neither an environment nor a model and advances no simulated time.
        expected = replay_tools.join(protocol['config'], protocol['seed']+rank, rank,
                                     monitors, replay['steps_per_env'])
        for episode in expected:
            episode['rank'] = rank
        require(actual == expected, 'replay episodes differ from their monitor source and frozen command courses')
    require(replay.get('timeline') == replay_tools.fine_timeline(episodes)
            and replay.get('ppo_timeline') == replay_tools.ppo_timeline(train/'progress.csv'),
            'replay timeline differs from monitor episodes or training progress')
    require(len(replay['timeline']) == len(replay['ppo_timeline']) == 16, 'replay must have 16 training slices')
    stds = [row.get('train/std') for row in replay['ppo_timeline'][12:]]
    require(all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)
                and v >= 0 for v in stds), 'last-quarter std is missing or invalid')
    forward_ends = sum(row['early_ends_by_segment'].get('drive_forward', 0) for row in replay['timeline'][12:])
    return sum(stds)/4, forward_ends, {
        'replay_sha256': sha(path), 'replay_episodes_sha256': sha(episodes_path),
        'progress_sha256': sha(train/'progress.csv'), 'monitor_sha256': expected_inputs,
    }


def validate_eval(directory, train, release, protocol, root):
    directory = Path(directory)
    contract = read_json(directory/'evaluation_contract.json')
    summary = read_json(directory/'summary.json')
    expected_provenance = {'kind': 'self_trained_sb3_checkpoint',
                           'checkpoint_sha256': sha(train/'policy.zip'),
                           'training_protocol_sha256': sha(train/'protocol.json'),
                           'training_result_sha256': sha(train/'result.json'),
                           'training_kind': 'scale', 'training_seed': protocol['seed'], 'training_arm': 'B',
                           'trained_transitions': release['plans']['scale']['steps'],
                           'random_initialization': True, 'teacher_policy_used': False, 'vecnormalize_used': False}
    expected_scenarios = [{'sequence': task, 'delay_ticks': 0, 'reset_seed': 7001} for task in gates.TASKS]
    config = protocol['config']
    expected_sources = {'config': release['config'], 'evaluation_script': 'tools/evaluate_d1_ppo.py',
                        'scene': config['scene'], 'robot': str(Path(config['scene']).parent/'robot.xml'),
                        'interface_schema': config['interface_schema']}
    require(contract.get('format') == 'd1-fixed-evaluation-03-v1' and contract.get('phase') == 'pilot'
            and contract.get('provenance') == expected_provenance
            and contract.get('scenarios') == expected_scenarios
            and contract.get('source_sha256') == {key: release['frozen_inputs'][name]
                                                  for key, name in expected_sources.items()}
            and contract.get('full_phase_episodes') == contract.get('requested_episodes') == 3
            and contract.get('planned_max_native_steps') == contract.get('hard_max_native_steps') == 18000
            and contract.get('episode_seconds') == 12.0 and contract.get('policy_tick_seconds') == .02
            and contract.get('policy_ticks_per_episode') == 600,
            'evaluation contract does not bind this frozen final checkpoint and three-task protocol')
    require(contract.get('thresholds_nominal_zero_delay') == evaluator_thresholds(root),
            'evaluation thresholds differ from the frozen evaluator')
    require(source_identities(contract.get('environment_sources_sha256'), release, root) == set(ENVIRONMENT_SOURCES),
            'evaluation environment provenance is incomplete')
    rows = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == 3 and {row.get('sequence') for row in rows} == set(gates.TASKS)
            and all(type(row.get('delay_ticks')) is int and row['delay_ticks'] == 0
                    and type(row.get('reset_seed')) is int and row['reset_seed'] == 7001 for row in rows),
            'evaluation must contain each frozen task exactly once; duplicate or foreign tasks are invalid')
    totals = dict(policy_transitions=0, native_steps=0, pd_updates=0)
    speeds, trace_hashes = {}, {}
    for row in rows:
        n = row.get('policy_transitions')
        counts = row.get('actual_counts', {})
        require(type(n) is int and 0 < n <= 600 and type(row.get('failure')) is bool
                and type(row.get('reference_thresholds_met')) is bool
                and (row.get('done_reason') != 'HORIZON_12S' or n == 600)
                and all(type(counts.get(key)) is int and counts[key] == n*factor
                        for key, factor in (('policy_transitions', 1), ('native_steps', 10), ('pd_updates', 5))),
                'evaluation episode has incomplete or inconsistent counters')
        for key in totals:
            totals[key] += counts[key]
        task = row['sequence']
        trace_path = directory/f'{task}_delay0_seed7001.npz'
        require(row.get('trace_file') == trace_path.name and row.get('trace_sha256') == sha(trace_path),
                'evaluation episode is not bound to its current trace')
        with np.load(trace_path, allow_pickle=False) as trace:
            vx, pitch, xy = (trace[key] for key in ('body_vx_m_s', 'base_pitch_rad', 'boundary_xy_m'))
            require(vx.shape == pitch.shape == (n,) and xy.shape == (n+1, 2)
                    and all(np.all(np.isfinite(array)) for array in (vx, pitch, xy)),
                    'evaluation trace is incomplete or non-finite')
            if task != 'turn_stop':
                direction = 1. if task == 'forward_stop' else -1.
                speeds[task] = (describe.describe({'body_vx_m_s': vx, 'base_pitch_rad': pitch,
                                                   'boundary_xy_m': xy}, direction)
                                ['drive_last_second_speed_along_command_m_s'] if n >= describe.CUT else None)
        trace_hashes[task] = sha(trace_path)
    require(summary.get('status') == 'COMPLETE' and summary.get('phase') == 'pilot'
            and summary.get('episodes_completed') == summary.get('full_phase_episodes') == 3
            and summary.get('contract_sha256') == sha(directory/'evaluation_contract.json')
            and all(type(summary.get(field)) is int and summary[field] == totals[key]
                    for field, key in (('actual_native_steps', 'native_steps'),
                                       ('actual_policy_transitions', 'policy_transitions'),
                                       ('actual_pd_updates', 'pd_updates')))
            and totals['native_steps'] <= 18000,
            'evaluation summary is partial or inconsistent with the three completed reports')
    tasks = gates.task_results({row['sequence']: row for row in rows})
    return tasks, speeds, {'evaluation_contract_sha256': sha(directory/'evaluation_contract.json'),
                           'evaluation_summary_sha256': sha(directory/'summary.json'),
                           'episodes_sha256': sha(directory/'episodes.jsonl'), 'trace_sha256': trace_hashes}


def interpretation(verdicts):
    if verdicts['TRAINING'] != 'MET':
        return 'not evaluable: training did not complete the frozen budget; no final-policy qualification'
    if verdicts['Q1'] == 'NOT_MET':
        return 'without the entropy bonus the std still rises: the bonus is not the main driver (one seed)'
    if verdicts['Q2'] == 'NOT_MET':
        return 'lower noise was insufficient to improve forward behaviour in this seed'
    return ('removing the entropy bonus coincided with lower noise and improved forward behaviour in this seed; '
            'this does not independently prove that noise mediates the causal effect')


def evaluate(release_path, report_path, replay_path=None, eval_dir=None, root=ROOT):
    root = Path(root).resolve()
    release = load_release(release_path, root)
    train = root/release['plans']['scale']['output']
    require(Path(report_path).resolve() == supervisor.receipt_dir(train)/'report.json',
            'supervisor report is not from this release training directory')
    report = read_json(report_path)
    require(report.get('format') == supervisor.REPORT_FORMAT and report.get('kind') == 'scale'
            and report.get('release_sha256') == sha(release_path)
            and report.get('supervisor_sha256') == release['supervisor_sha256']
            and report.get('plan') == release['plans']['scale']
            and report.get('resources') == release['resources'],
            'supervisor report belongs to a different release or training plan')
    try:
        actual_result = read_json(train/'result.json')
    except (OSError, ValueError):
        actual_result = None
    complete = training_complete(report, release, actual_result)
    verdicts = {'E1': 'MET' if equivalence_complete(report, release['plans']['scale']) else 'NOT_MET',
                'TRAINING': 'MET' if complete else 'NOT_MET'}
    verdicts.update({f'Q{i}': 'NOT_EVALUABLE' for i in range(1, 6)})
    tasks, std, forward_ends, speeds = {}, None, None, None
    qualification = {'tasks_passed': None, 'tasks_total': 3, 'early_failures': None, 'overall': 'NOT_EVALUATED'}
    inputs = {'release_sha256': sha(release_path), 'supervisor_report_sha256': sha(report_path)}
    if complete:
        require(replay_path is not None and eval_dir is not None, 'completed training requires --replay and --eval-dir')
        preflight = supervisor.receipt_dir(root/release['plans']['preflight']['output'])/'report.json'
        supervisor.require_preflight(preflight, release, sha(release_path), root)
        protocol = final_protocol(train, release, root)
        std, forward_ends, replay_inputs = validate_replay(replay_path, train, release, protocol)
        tasks, speeds, eval_inputs = validate_eval(eval_dir, train, release, protocol, root)
        passed, early = sum(t['passed'] for t in tasks.values()), sum(t['early_failure'] for t in tasks.values())
        qualification = {'tasks_passed': passed, 'tasks_total': 3, 'early_failures': early,
                         'overall': gates.qualification_verdict(passed, 3, early, 3)}
        verdicts.update(Q1='MET' if std < .45 else 'NOT_MET', Q2='MET' if forward_ends <= 15 else 'NOT_MET',
                        Q3='MET' if passed >= 2 else 'NOT_MET',
                        Q4='NOT_MET' if tasks['forward_stop']['early_failure'] else 'MET', Q5='REPORTED')
        inputs.update(replay_inputs, **eval_inputs, checkpoint_sha256=sha(train/'policy.zip'),
                      protocol_sha256=sha(train/'protocol.json'), result_sha256=sha(train/'result.json'),
                      preflight_report_sha256=sha(preflight))
    return {'format': 'd1-contract10-verdict-v1', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'physics_steps_executed': 0, 'predictions': verdicts, 'interpretation': interpretation(verdicts),
            'readings': {'last_quarter_mean_std': std, 'last_quarter_forward_drive_early_ends': forward_ends,
                         'drive_last_second_speed_m_s': speeds},
            'training_status': actual_result.get('status') if isinstance(actual_result, dict) else None,
            'qualification': qualification,
            'tasks': tasks, 'inputs': inputs, 'tool_sha256': sha(__file__)}


def write_verdict(release, supervisor_report, replay, eval_dir, output, root=ROOT):
    output = Path(output)
    require(not output.exists(), 'verdict output already exists; refusing to overwrite')
    report = evaluate(release, supervisor_report, replay, eval_dir, root)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'verdict.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--supervisor-report', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--eval-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = write_verdict(args.release, args.supervisor_report, args.replay, args.eval_dir, args.output)
    print(json.dumps({'predictions': report['predictions'], 'qualification': report['qualification'],
                      'interpretation': report['interpretation']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
