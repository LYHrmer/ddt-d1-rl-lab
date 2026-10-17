#!/usr/bin/env python3
"""Read completed 03d/08 logs and fixed evaluation traces; never load a policy.

Reuse the existing episode replay labels, Monitor reader, PPO timeline and qualification
gates. Training logs do not contain per-tick phase rewards or phase exposure: those stay
NOT_OBSERVED. Fixed qualification traces permit strict, coverage-aware phase windows.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for name in ('tools08', 'tools10'):
    sys.path.insert(0, str(ROOT/name))
import replay_training_episodes08 as replay_tools  # noqa: E402
import verify_scale08 as gates  # noqa: E402
from supervise_scale10 import result_complete  # noqa: E402

DT = .02
HORIZON = 600
TRANSITION_TICKS = 25
END_SEGMENTS = {'initial_zero', 'middle_zero', 'final_zero', 'drive_forward', 'drive_backward',
                'turn_left', 'turn_right'}
TRACE_FIELDS = ('reward', 'body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'base_pitch_rad', 'tau_saturation_count')
# This diagnostic consumes the already completed 08 replay, never regenerates its
# RNG schedule. Monitor totals alone cannot authenticate course/direction labels.
TRUSTED_REPLAY_SHA256 = {
    'report.json': 'a529aaa674567295ef45bee8f8fd537b01fb60e8ecc0c9a7ca56d4a161b9fb25',
    'episodes.jsonl': '5f97b4766ec77cb5fd1e387c370d71899e51465b7b68f04b48a6e223407a2b01',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluator_literal(name):
    tree = ast.parse((ROOT/'tools/evaluate_d1_ppo.py').read_text())
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, f'frozen evaluator has no unique {name} literal')
    return values[0]


def validate_replay_identity(report_path):
    report_path = Path(report_path)
    require(sha(report_path) == TRUSTED_REPLAY_SHA256['report.json']
            and sha(report_path.parent/'episodes.jsonl') == TRUSTED_REPLAY_SHA256['episodes.jsonl'],
            'existing completed 08 replay identity changed')


def direction(vx, wz):
    require(not (vx and wz), 'simultaneous translation/turn is outside the fixed diagnostic scope')
    return ('forward' if vx > 0 else 'backward') if vx else ('turn_left' if wz > 0 else 'turn_right')


def phase_windows(sequence):
    """Half-open policy intervals; 0.5 s after an actual nonzero-to-zero command cut."""
    starts = [round(item['start_s']/DT) for item in sequence]
    require(starts and starts[0] == 0 and starts == sorted(set(starts)) and starts[-1] < HORIZON
            and all(abs(item['start_s']/DT-start) < 1e-9 for item, start in zip(sequence, starts)),
            'command boundaries must be unique ordered policy ticks starting at zero')
    windows, previous = [], None
    for index, (item, begin, end) in enumerate(zip(sequence, starts, starts[1:]+[HORIZON])):
        vx, wz = item['vx'], item['wz']
        require(math.isfinite(vx) and math.isfinite(wz), 'non-finite command')
        if vx or wz:
            previous = direction(vx, wz)
            parts = [('drive_'+previous, begin, end)]
        elif previous is None:
            parts = [('stationary_initial', begin, end)]
        elif sequence[index-1]['vx'] or sequence[index-1]['wz']:
            cut_end = min(end, begin+TRANSITION_TICKS)
            parts = [('cut_zero_after_'+previous, begin, cut_end),
                     ('stationary_after_'+previous, cut_end, end)]
        else:
            parts = [('stationary_after_'+previous, begin, end)]
        windows.extend({'phase': label, 'start_tick': lo, 'end_tick': hi,
                        'command': [vx, 0., wz]} for label, lo, hi in parts if hi > lo)
    return windows


def window_result(trace, window):
    n = len(trace['reward'])
    require(all(np.shape(trace[key]) == (n,) and np.all(np.isfinite(trace[key])) for key in TRACE_FIELDS)
            and np.shape(trace['command_during_interval']) == (n, 3)
            and np.all(np.isfinite(trace['command_during_interval']))
            and np.all(np.asarray(trace['tau_saturation_count']) >= 0), 'corrupt or non-finite trace arrays')
    begin, end = window['start_tick'], window['end_tick']
    observed = max(0, min(n, end)-begin)
    expected = end-begin
    result = dict(window, expected_ticks=expected, observed_ticks=observed, missing_ticks=expected-observed,
                  coverage='OBSERVED' if observed == expected else 'PARTIAL' if observed else 'NOT_OBSERVED',
                  metrics=None)
    if observed != expected:
        return result
    sl = slice(begin, end)
    vx, vy, wz = (np.asarray(trace[key])[sl] for key in ('body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s'))
    commands = np.asarray(trace['command_during_interval'])[sl]
    require(np.allclose(commands, window['command'], atol=1e-7, rtol=0), 'trace command differs from its fixed window')
    reward = np.asarray(trace['reward'])[sl]
    result['metrics'] = {'mean_abs_vx_error_m_s': float(np.abs(vx-commands[:, 0]).mean()),
                         'mean_abs_wz_error_rad_s': float(np.abs(wz-commands[:, 2]).mean()),
                         'mean_horizontal_speed_m_s': float(np.hypot(vx, vy).mean()),
                         'mean_abs_lateral_speed_m_s': float(np.abs(vy).mean()),
                         'peak_abs_pitch_rad': float(np.abs(np.asarray(trace['base_pitch_rad'])[sl]).max()),
                         'reward_sum': float(reward.sum()), 'mean_reward_per_policy_tick': float(reward.mean()),
                         'mean_saturated_joint_count_per_policy_tick': float(np.asarray(trace['tau_saturation_count'])[sl].sum()/expected)}
    return result


def episode_stats(rows):
    ends = {}
    for row in rows:
        if row['ended_early']:
            ends[row['ended_in']] = ends.get(row['ended_in'], 0)+1
    n, early = len(rows), sum(row['ended_early'] for row in rows)
    return {'finished_episodes': n, 'early_ended_episodes': early,
            'early_end_fraction': early/n if n else None,
            'mean_finished_episode_return': sum(row['return'] for row in rows)/n if n else None,
            'early_end_segment_counts': ends,
            'denominator': 'finished Monitor episodes in this group, NOT phase entries or phase ticks'}


def training_result(directory, replay_rows, replay_report):
    directory = Path(directory)
    protocol, result = read_json(directory/'protocol.json'), read_json(directory/'result.json')
    steps, envs = protocol['planned_transitions'], protocol['envs']
    require(result_complete(result, {'steps': steps}, envs), 'only complete training with exact worker counts is admissible')
    require(protocol['seed'] == replay_report['seed'] and envs == replay_report['envs']
            and steps//envs <= replay_report['steps_per_env'], 'replay seed, worker count or length differs')
    episodes, tails, total_logged_ticks = [], {}, 0
    for rank in range(envs):
        monitor = replay_tools.read_monitor(directory/f'env_{rank:02d}.monitor.csv')
        source = [row for row in replay_rows if row['rank'] == rank]
        require(len(source) >= len(monitor), 'replay is shorter than the Monitor source')
        done = 0
        for index, (row, label) in enumerate(zip(monitor, source)):
            require(label['episode'] == index and label['length'] == row['l'] and label['return'] == row['r']
                    and label['mode'] in ('translation', 'turn') and label['first_sign'] in (-1, 1)
                    and 0 < row['l'] <= replay_tools.HORIZON and math.isfinite(row['r'])
                    and label['ended_early'] is (row['l'] < replay_tools.HORIZON)
                    and (label['ended_in'] in END_SEGMENTS if label['ended_early'] else label['ended_in'] is None)
                    and label['training_fraction'] == done/replay_report['steps_per_env'],
                    'replay labels do not match this run Monitor prefix')
            episodes.append(dict(label, training_fraction=done/(steps//envs)))
            done += row['l']
        tails[str(rank)] = steps//envs-done
        require(0 <= tails[str(rank)] < replay_tools.HORIZON, 'Monitor source has missing episodes or exceeds training counts')
        total_logged_ticks += done
    ledger = json_rows(directory/'ledger.jsonl')
    require(len(ledger) >= 4 and len(ledger) % 4 == 0 and len(ledger)*4096 == steps
            and all(row['rollout'] == i+1 and row['transitions'] == (i+1)*4096
            for i, row in enumerate(ledger)), 'rollout ledger is incomplete, duplicated or out of order')
    ppo = replay_tools.ppo_timeline(directory/'progress.csv', bins=4)
    with (directory/'progress.csv').open() as stream:
        ppo_rows = list(csv.DictReader(stream))
    quarters = []
    for index in range(4):
        lo, hi = index/4, (index+1)/4
        selected = [row for row in episodes if lo <= row['training_fraction'] < hi]
        quarter = dict(episode_stats(selected), from_fraction=lo, to_fraction=hi,
                       assignment='episode START fraction, recomputed using this run per-worker budget', by_course={})
        for name in ('forward_first', 'backward_first', 'turn'):
            members = [row for row in selected if ('turn' if row['mode'] == 'turn' else
                        'forward_first' if row['first_sign'] > 0 else 'backward_first') == name]
            quarter['by_course'][name] = episode_stats(members)
        rollouts = ledger[int(len(ledger)*lo):int(len(ledger)*hi)]
        terms = sorted({key for row in rollouts for key in row['reward_contribution_mean_per_tick']})
        require(all(set(row['reward_contribution_mean_per_tick']) == set(terms) and
                    all(math.isfinite(v) for v in row['reward_contribution_mean_per_tick'].values()) for row in rollouts),
                'inconsistent rollout reward terms')
        quarter['rollout_mixed_phase_reward_mean_per_tick'] = {
            key: sum(row['reward_contribution_mean_per_tick'][key] for row in rollouts)/len(rollouts) for key in terms}
        quarter['mixed_reward_denominator'] = {'full_rollouts': len(rollouts), 'policy_transitions': len(rollouts)*4096,
                                                'phases_separable': False}
        quarter['ppo_diagnostics_mixed_phase'] = ppo[index]
        ppo_chunk = ppo_rows[int(len(ppo_rows)*lo):int(len(ppo_rows)*hi)]
        quarter['ppo_available_rows_by_metric'] = {
            key: sum(row.get(key) not in ('', None) for row in ppo_chunk) for key in replay_tools.PPO_COLUMNS}
        quarters.append(quarter)
    return {'status': 'COMPLETE', 'trained_transitions': steps, 'seed': protocol['seed'], 'envs': envs,
            'finished_episodes': len(episodes), 'quarters': quarters,
            'monitor_censoring': {'finished_episode_ticks': total_logged_ticks, 'unlogged_tail_ticks': sum(tails.values()),
                                  'unlogged_tail_ticks_by_worker': tails, 'tail_phase_and_return': 'NOT_OBSERVED'},
            'phase_exposure_ticks': {'status': 'NOT_OBSERVED', 'reason': 'existing replay rows record end labels, not all segment boundaries'},
            'phase_reward_components': {'status': 'NOT_OBSERVED', 'reason': 'ledger terms mix all phases; Monitor has episode totals only'},
            'cut_zero_vs_stationary_training_split': {'status': 'NOT_OBSERVED', 'reason': 'zero-segment end labels do not locate the end relative to the cut'}}


def evaluation_result(directory, train_directory, metadata=None):
    directory, train_directory = Path(directory), Path(train_directory)
    contract, summary = read_json(directory/'evaluation_contract.json'), read_json(directory/'summary.json')
    protocol, result = read_json(train_directory/'protocol.json'), read_json(train_directory/'result.json')
    require(result_complete(result, {'steps': protocol['planned_transitions']}, protocol['envs']),
            'qualification analysis requires complete training')
    provenance = contract['provenance']
    expected = [{'sequence': task, 'delay_ticks': 0, 'reset_seed': 7001} for task in gates.TASKS]
    require(contract['format'] == 'd1-fixed-evaluation-03-v1' and contract['phase'] == 'pilot'
            and contract['scenarios'] == expected and contract['full_phase_episodes'] == contract['requested_episodes'] == 3
            and contract['policy_tick_seconds'] == DT and contract['policy_ticks_per_episode'] == HORIZON
            and summary['status'] == 'COMPLETE' and summary['episodes_completed'] == 3
            and summary['contract_sha256'] == sha(directory/'evaluation_contract.json'), 'not the complete fixed qualification protocol')
    require(contract['sequences'] == evaluator_literal('SEQUENCES')
            and contract['thresholds_nominal_zero_delay'] == evaluator_literal('THRESHOLDS')
            and contract['source_sha256']['evaluation_script'] == sha(ROOT/'tools/evaluate_d1_ppo.py'),
            'qualification sequence, thresholds or evaluator identity changed')
    require(provenance['training_kind'] == protocol['kind'] and provenance['training_seed'] == protocol['seed']
            and provenance['training_arm'] == protocol['arm'] and provenance['trained_transitions'] == result['trained_transitions']
            and provenance['random_initialization'] is True and provenance['teacher_policy_used'] is False
            and provenance['vecnormalize_used'] is False, 'qualification belongs to different training')
    if provenance['kind'] == 'self_trained_sb3_checkpoint':
        require(provenance['training_protocol_sha256'] == sha(train_directory/'protocol.json')
                and provenance['training_result_sha256'] == sha(train_directory/'result.json'), 'checkpoint provenance mismatch')
    elif provenance['kind'] == 'self_trained_onnx_python_ort':
        require(metadata is not None and sha(metadata) == provenance['metadata_sha256'], 'ONNX evaluation needs its matching metadata')
        identities = read_json(metadata)['sha256']
        require(identities['training_protocol'] == sha(train_directory/'protocol.json')
                and identities['training_result'] == sha(train_directory/'result.json')
                and identities['policy_onnx'] == provenance['onnx_sha256'], 'ONNX metadata belongs to different training')
    else:
        raise ValueError('unsupported qualification provenance')
    rows = json_rows(directory/'episodes.jsonl')
    require(len(rows) == 3 and {row['sequence'] for row in rows} == set(gates.TASKS)
            and all(row['delay_ticks'] == 0 and row['reset_seed'] == 7001 for row in rows), 'duplicate, missing or foreign qualification task')
    by_task = {row['sequence']: row for row in rows}
    qualification = gates.task_results(by_task)
    tasks, totals = {}, {'policy_transitions': 0, 'native_steps': 0, 'pd_updates': 0}
    for task in gates.TASKS:
        row = by_task[task]
        path = directory/f'{task}_delay0_seed7001.npz'
        require(row['trace_file'] == path.name and row['trace_sha256'] == sha(path), 'trace identity differs from episode receipt')
        with np.load(path, allow_pickle=False) as handle:
            trace = {key: handle[key] for key in (*TRACE_FIELDS, 'command_during_interval')}
        n = len(trace['reward'])
        require(n == row['policy_transitions'] and 0 < n <= HORIZON
                and (row['done_reason'] != 'HORIZON_12S' or n == HORIZON), 'trace duration differs from episode report')
        windows = phase_windows(contract['sequences'][task])
        for window in windows:
            lo, hi = window['start_tick'], min(n, window['end_tick'])
            require(hi <= lo or np.allclose(trace['command_during_interval'][lo:hi], window['command'], atol=1e-7, rtol=0),
                    'observed command differs from frozen task sequence')
        for field, factor in (('policy_transitions', 1), ('native_steps', 10), ('pd_updates', 5)):
            require(type(row['actual_counts'][field]) is int and row['actual_counts'][field] == n*factor, 'inconsistent evaluation counters')
            totals[field] += row['actual_counts'][field]
        tasks[task] = {'observed_ticks': n, 'planned_ticks': HORIZON, 'ended_early': n < HORIZON or row['failure'],
                       'original_qualification': qualification[task],
                       'windows': [window_result(trace, window) for window in windows],
                       'phase_reward_components': {'status': 'NOT_OBSERVED', 'reason': 'trace contains total scalar reward only'}}
    require(all(summary[field] == totals[key] for field, key in (
        ('actual_policy_transitions', 'policy_transitions'), ('actual_native_steps', 'native_steps'), ('actual_pd_updates', 'pd_updates'))),
        'qualification summary counters differ from its traces')
    return {'predictor_backend': provenance['kind'], 'provenance': provenance,
            'tasks': tasks, 'original_tasks_passed': sum(task['passed'] for task in qualification.values())}


def build_report(training, evaluations, replay_path, metadata=None):
    require(set(training) == set(evaluations) and bool(training), 'training/evaluation labels must match')
    replay_path, metadata = Path(replay_path), metadata or {}
    validate_replay_identity(replay_path)
    replay_report = read_json(replay_path)
    rows = json_rows(replay_path.parent/'episodes.jsonl')
    require(replay_report['format'] == 'd1-contract08-training-episode-replay-v1'
            and replay_report['physics_steps_executed'] == 0 and replay_report['episodes'] == len(rows)
            and replay_report['tool_sha256'] == sha(Path(replay_tools.__file__)), 'replay identity is not the existing frozen reader')
    require(len({(row['rank'], row['episode']) for row in rows}) == len(rows), 'duplicate replay episode identities')
    sources = {replay_path, replay_path.parent/'episodes.jsonl', Path(__file__), Path(replay_tools.__file__),
               Path(gates.__file__), ROOT/'tools04/build_final_ledger04.py', ROOT/'tools04/release_pilot04b.py',
               ROOT/'tools08/release_scale08.py', ROOT/'tools08/run_scale08.py', ROOT/'tools10/supervise_scale10.py',
               ROOT/'tools/evaluate_d1_ppo.py'}
    result, configurations, replay_source_checked = {}, [], False
    for label, directory in training.items():
        directory, evaluation = Path(directory), Path(evaluations[label])
        train = training_result(directory, rows, replay_report)
        configurations.append(read_json(directory/'protocol.json')['config'])
        result[label] = {'training': train, 'qualification': evaluation_result(evaluation, directory, metadata.get(label))}
        sources.update(directory/name for name in ('protocol.json', 'result.json', 'progress.csv', 'ledger.jsonl'))
        sources.update(directory/f'env_{rank:02d}.monitor.csv' for rank in range(train['envs']))
        if train['trained_transitions']//train['envs'] == replay_report['steps_per_env']:
            require(replay_report['inputs'] == {f'env_{rank:02d}.monitor.csv': sha(directory/f'env_{rank:02d}.monitor.csv')
                                                for rank in range(train['envs'])}, 'replay source Monitor identity mismatch')
            replay_source_checked = True
        sources.update(evaluation/name for name in ('evaluation_contract.json', 'episodes.jsonl', 'summary.json'))
        sources.update(evaluation/f'{task}_delay0_seed7001.npz' for task in gates.TASKS)
        if label in metadata:
            sources.add(Path(metadata[label]))
    require(all(config == configurations[0] for config in configurations), 'runs use different task configurations')
    require(replay_source_checked, 'include the completed source training run that generated this replay')
    return {'format': 'd1-phase-learning-diagnostic14-v1', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'physics_steps_executed': 0, 'models_loaded': 0, 'policy_weight_files_read': False,
            'semantics': {'policy_tick_s': DT, 'cut_zero_window_s': TRANSITION_TICKS*DT,
                          'stationary_window': 'remaining zero-command interval after the first 0.5 s',
                          'stationary_does_not_imply_robot_stopped': True,
                          'missing_window_policy': 'whole-window metrics are null unless every planned tick is observed',
                          'training_episode_denominator': 'finished Monitor episodes by course and episode-start training quarter',
                          'causal_interpretation': 'descriptive only; neither phase totals nor backend comparison identify a cause'},
            'runs': result, 'inputs_sha256': {str(path.resolve()): sha(path) for path in sorted(sources, key=str)}}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def mappings(values):
    pairs = [item.split('=', 1) for item in values]
    require(all(len(pair) == 2 and pair[0] for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs), 'labels must be unique LABEL=PATH pairs')
    return {name: Path(path) for name, path in pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='append', required=True, metavar='LABEL=DIR')
    parser.add_argument('--eval', action='append', required=True, metavar='LABEL=DIR')
    parser.add_argument('--eval-metadata', action='append', default=[], metavar='LABEL=METADATA_JSON')
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not exist')
    args = parser.parse_args()
    require(not args.output.exists(), 'output already exists; refusing to overwrite')
    report = build_report(mappings(args.train), mappings(args.eval), args.replay, mappings(args.eval_metadata))
    write_report(args.output, report)
    print(json.dumps({'output': str(args.output), 'runs': list(report['runs']), 'models_loaded': 0, 'physics_steps_executed': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
