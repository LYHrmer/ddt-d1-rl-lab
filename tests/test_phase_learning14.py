"""Stage coverage counterexamples use synthetic arrays and Monitor rows only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools14'))
import analyze_phase_learning14 as analyzer  # noqa: E402

SEQUENCE = [{'start_s': 0., 'vx': 0., 'wz': 0.}, {'start_s': 2., 'vx': .2, 'wz': 0.},
            {'start_s': 8., 'vx': 0., 'wz': 0.}]


def trace(n=600):
    command = np.zeros((n, 3)); command[100:min(n, 400), 0] = .2
    return {'reward': np.ones(n), 'command_during_interval': command,
            'body_vx_m_s': np.zeros(n), 'body_vy_m_s': np.zeros(n), 'body_wz_rad_s': np.zeros(n),
            'base_pitch_rad': np.zeros(n), 'tau_saturation_count': np.ones(n)*2}


def test_a_fall_cannot_look_like_a_good_stationary_window():
    window = analyzer.phase_windows(SEQUENCE)[-1]
    full = analyzer.window_result(trace(), window)
    partial = analyzer.window_result(trace(485), window)
    assert full['phase'] == 'stationary_after_forward'
    assert full['expected_ticks'] == 175 and full['observed_ticks'] == 175
    assert full['metrics']['mean_horizontal_speed_m_s'] == 0.
    assert partial['coverage'] == 'PARTIAL' and partial['observed_ticks'] == 60
    assert partial['missing_ticks'] == 115 and partial['metrics'] is None


def test_no_observed_ticks_is_explicit_not_observed_not_zero():
    result = analyzer.window_result(trace(100), analyzer.phase_windows(SEQUENCE)[-1])
    assert result['coverage'] == 'NOT_OBSERVED' and result['observed_ticks'] == 0
    assert result['metrics'] is None


def test_windows_cover_every_planned_tick_once_and_keep_zero_transition_separate():
    windows = analyzer.phase_windows(SEQUENCE)
    assert [(w['phase'], w['start_tick'], w['end_tick']) for w in windows] == [
        ('stationary_initial', 0, 100), ('drive_forward', 100, 400),
        ('cut_zero_after_forward', 400, 425), ('stationary_after_forward', 425, 600)]
    assert sum(w['end_tick']-w['start_tick'] for w in windows) == 600


def test_turn_reversal_is_two_driving_directions_not_a_zero_command():
    seq = [{'start_s': 0., 'vx': 0., 'wz': 0.}, {'start_s': 2., 'vx': 0., 'wz': .2},
           {'start_s': 5., 'vx': 0., 'wz': -.2}, {'start_s': 8., 'vx': 0., 'wz': 0.}]
    assert [w['phase'] for w in analyzer.phase_windows(seq)] == [
        'stationary_initial', 'drive_turn_left', 'drive_turn_right',
        'cut_zero_after_turn_right', 'stationary_after_turn_right']


def test_saturation_is_events_per_tick_not_a_fraction_of_ticks():
    result = analyzer.window_result(trace(), analyzer.phase_windows(SEQUENCE)[1])
    assert result['metrics']['saturation_events_per_policy_tick'] == 2.
    assert result['metrics']['mean_abs_vx_error_m_s'] == pytest.approx(.2)
    assert result['metrics']['reward_sum'] == 300.


@pytest.mark.parametrize('change', ['nan', 'wrong_shape', 'negative_saturation'])
def test_corrupt_trace_never_produces_a_plausible_window(change):
    sample = trace()
    if change == 'nan': sample['body_vx_m_s'][101] = np.nan
    if change == 'wrong_shape': sample['command_during_interval'] = np.zeros((600, 2))
    if change == 'negative_saturation': sample['tau_saturation_count'][101] = -1
    with pytest.raises(ValueError):
        analyzer.window_result(sample, analyzer.phase_windows(SEQUENCE)[1])


def training_fixture(root, *, status='COMPLETE'):
    root.mkdir(parents=True, exist_ok=True)
    protocol = {'kind': 'pilot', 'arm': 'B', 'seed': 101, 'envs': 1, 'planned_transitions': 16384,
                'planned_native_steps': 163840, 'config': {'version': 'd1-pure-ppo-pilot-03d'},
                'inputs_sha256': {}}
    worker = {'policy_transitions': 16384, 'native_steps': 163840, 'pd_updates': 81920,
              'resets': 17, 'forward_calls': 16419}
    result = {'status': status, 'inputs_unchanged': True, 'error': None, 'trained_transitions': 16384,
              'actual_counts': worker, 'per_worker_counts': [worker]}
    for name, value in [('protocol', protocol), ('result', result)]:
        (root/f'{name}.json').write_text(json.dumps(value))
    rows = [{'r': float(i+1), 'l': 1000} for i in range(16)]
    (root/'env_00.monitor.csv').write_text('#{}\nr,l,t\n'+''.join(f"{r['r']},{r['l']},0\n" for r in rows))
    ledger = [{'rollout': i+1, 'transitions': (i+1)*4096, 'reward_contribution_mean_per_tick': {'alive': .1}}
              for i in range(4)]
    (root/'ledger.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    (root/'progress.csv').write_text('train/std\n.5\n.5\n.5\n.5\n')
    replay = [{'rank': 0, 'episode': i, 'length': row['l'], 'return': row['r'], 'mode': 'translation',
               'first_sign': 1, 'ended_early': False, 'ended_in': None,
               'training_fraction': i*1000/32768} for i, row in enumerate(rows)]
    report = {'seed': 101, 'envs': 1, 'steps_per_env': 32768}
    return replay, report


def test_training_bins_use_this_run_fraction_not_the_longer_replay_fraction(tmp_path):
    replay, report = training_fixture(tmp_path/'training')
    result = analyzer.training_result(tmp_path/'training', replay, report)
    assert [q['finished_episodes'] for q in result['quarters']] == [5, 4, 4, 3]
    assert result['monitor_censoring']['unlogged_tail_ticks'] == 384
    assert result['phase_exposure_ticks']['status'] == 'NOT_OBSERVED'
    assert result['phase_reward_components']['status'] == 'NOT_OBSERVED'
    assert all(q['mixed_reward_denominator']['full_rollouts'] == 1 for q in result['quarters'])


def test_phase_end_count_denominator_is_finished_episodes_not_segment_exposure(tmp_path):
    replay, report = training_fixture(tmp_path/'training')
    replay[0]['ended_early'] = True; replay[0]['ended_in'] = 'drive_forward'; replay[0]['length'] = 500
    done = 0
    for row in replay:
        row['training_fraction'] = done/report['steps_per_env']; done += row['length']
    (tmp_path/'training/env_00.monitor.csv').write_text('#{}\nr,l,t\n'+''.join(
        f"{row['return']},{row['length']},0\n" for row in replay))
    result = analyzer.training_result(tmp_path/'training', replay, report)
    group = result['quarters'][0]['by_course']['forward_first']
    assert group['finished_episodes'] == 5 and group['early_end_fraction'] == .2
    assert group['early_end_segment_counts'] == {'drive_forward': 1}


def test_incomplete_training_and_borrowed_replay_rows_are_rejected(tmp_path):
    replay, report = training_fixture(tmp_path/'training', status='RAM_RESERVE_BELOW_3_GIB')
    with pytest.raises(ValueError, match='complete'):
        analyzer.training_result(tmp_path/'training', replay, report)
    data = json.loads((tmp_path/'training/result.json').read_text()); data['status'] = 'COMPLETE'
    (tmp_path/'training/result.json').write_text(json.dumps(data))
    replay[0]['return'] = 123.
    with pytest.raises(ValueError, match='replay'):
        analyzer.training_result(tmp_path/'training', replay, report)


def test_writing_never_overwrites_an_existing_summary(tmp_path):
    path = tmp_path/'summary.json'; path.write_text('original')
    with pytest.raises(FileExistsError): analyzer.write_report(path, {})
    assert path.read_text() == 'original'


def test_consecutive_zero_segments_do_not_invent_another_command_cut():
    sequence = SEQUENCE+[{'start_s': 9., 'vx': 0., 'wz': 0.}]
    assert sum(w['phase'].startswith('cut_zero_') for w in analyzer.phase_windows(sequence)) == 1


def evaluation_fixture(root):
    training_fixture(root/'train')
    train, directory = root/'train', root/'eval'
    directory.mkdir()
    contract = {'format': 'd1-fixed-evaluation-03-v1', 'phase': 'pilot', 'full_phase_episodes': 3,
                'requested_episodes': 3, 'policy_tick_seconds': .02, 'policy_ticks_per_episode': 600,
                'sequences': analyzer.evaluator_literal('SEQUENCES'),
                'thresholds_nominal_zero_delay': analyzer.evaluator_literal('THRESHOLDS'),
                'source_sha256': {'evaluation_script': analyzer.sha(ROOT/'tools/evaluate_d1_ppo.py')},
                'scenarios': [{'sequence': task, 'delay_ticks': 0, 'reset_seed': 7001} for task in analyzer.gates.TASKS],
                'provenance': {'kind': 'self_trained_sb3_checkpoint', 'training_kind': 'pilot', 'training_seed': 101,
                               'training_arm': 'B', 'trained_transitions': 16384, 'random_initialization': True,
                               'teacher_policy_used': False, 'vecnormalize_used': False,
                               'training_protocol_sha256': analyzer.sha(train/'protocol.json'),
                               'training_result_sha256': analyzer.sha(train/'result.json')}}
    (directory/'evaluation_contract.json').write_text(json.dumps(contract))
    rows = []
    for task in analyzer.gates.TASKS:
        sample = trace(485)
        sample['command_during_interval'][:] = 0.
        for window in analyzer.phase_windows(contract['sequences'][task]):
            sample['command_during_interval'][window['start_tick']:min(485, window['end_tick'])] = window['command']
        path = directory/f'{task}_delay0_seed7001.npz'
        np.savez(path, **sample)
        rows.append({'sequence': task, 'delay_ticks': 0, 'reset_seed': 7001, 'policy_transitions': 485,
                     'failure': True, 'done_reason': 'TERMINATED', 'reference_thresholds_met': False,
                     'stop_displacement_m': None, 'stable_wz_mae_rad_s': None, 'peak_abs_roll_or_pitch_rad': .8,
                     'trace_file': path.name, 'trace_sha256': analyzer.sha(path),
                     'actual_counts': {'policy_transitions': 485, 'native_steps': 4850, 'pd_updates': 2425}})
    (directory/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary = {'status': 'COMPLETE', 'episodes_completed': 3, 'contract_sha256': analyzer.sha(directory/'evaluation_contract.json'),
               'actual_policy_transitions': 1455, 'actual_native_steps': 14550, 'actual_pd_updates': 7275}
    (directory/'summary.json').write_text(json.dumps(summary))
    return directory, train


def test_evaluation_uses_original_failed_gates_and_never_scores_missing_tail(tmp_path):
    directory, train = evaluation_fixture(tmp_path)
    result = analyzer.evaluation_result(directory, train)
    assert result['original_tasks_passed'] == 0
    tail = result['tasks']['forward_stop']['windows'][-1]
    assert tail['coverage'] == 'PARTIAL' and tail['missing_ticks'] == 115 and tail['metrics'] is None


@pytest.mark.parametrize('change', ['duplicate', 'trace_sha', 'threshold', 'summary_counts', 'partial_training'])
def test_borrowed_partial_or_changed_evaluation_is_rejected(tmp_path, change):
    directory, train = evaluation_fixture(tmp_path)
    if change in ('duplicate', 'trace_sha'):
        path = directory/'episodes.jsonl'; rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == 'duplicate': rows.append(rows[0])
        else: rows[0]['trace_sha256'] = '0'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change == 'threshold':
        path = directory/'evaluation_contract.json'; value = json.loads(path.read_text())
        value['thresholds_nominal_zero_delay']['stop_displacement_m'] = 99.
        path.write_text(json.dumps(value))
        path = directory/'summary.json'; value = json.loads(path.read_text())
        value['contract_sha256'] = analyzer.sha(directory/'evaluation_contract.json'); path.write_text(json.dumps(value))
    elif change == 'summary_counts':
        path = directory/'summary.json'; value = json.loads(path.read_text()); value['actual_native_steps'] = 1
        path.write_text(json.dumps(value))
    else:
        path = train/'result.json'; value = json.loads(path.read_text()); value['status'] = 'PARTIAL'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError): analyzer.evaluation_result(directory, train)


@pytest.mark.parametrize('change', ['mode', 'first_sign', 'report'])
def test_valid_but_tampered_replay_labels_cannot_change_course_statistics(tmp_path, monkeypatch, change):
    report = tmp_path/'report.json'; report.write_text('{"existing":true}\n')
    episodes = tmp_path/'episodes.jsonl'
    row = {'mode': 'translation', 'first_sign': 1, 'length': 1000, 'return': 1.}
    episodes.write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(analyzer, 'TRUSTED_REPLAY_SHA256', {
        'report.json': analyzer.sha(report), 'episodes.jsonl': analyzer.sha(episodes)})
    analyzer.validate_replay_identity(report)
    if change == 'report':
        report.write_text('{"existing":false}\n')
    else:
        row[change] = 'turn' if change == 'mode' else -1
        episodes.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='replay identity'):
        analyzer.validate_replay_identity(report)


def test_inexact_rollout_quarters_are_rejected(tmp_path):
    replay, report = training_fixture(tmp_path/'training')
    root = tmp_path/'training'
    protocol = json.loads((root/'protocol.json').read_text())
    protocol.update(planned_transitions=20480, planned_native_steps=204800)
    (root/'protocol.json').write_text(json.dumps(protocol))
    result = json.loads((root/'result.json').read_text())
    result['trained_transitions'] = 20480
    for counts in [result['actual_counts'], *result['per_worker_counts']]:
        counts.update(policy_transitions=20480, native_steps=204800, pd_updates=102400,
                      resets=21, forward_calls=20523)
    (root/'result.json').write_text(json.dumps(result))
    for index in range(16, 20):
        replay.append({**replay[-1], 'episode': index, 'return': float(index+1),
                       'training_fraction': index*1000/report['steps_per_env']})
    (root/'env_00.monitor.csv').write_text('#{}\nr,l,t\n'+''.join(
        f"{row['return']},{row['length']},0\n" for row in replay))
    path = root/'ledger.jsonl'
    ledger = [json.loads(line) for line in path.read_text().splitlines()]
    ledger.append(dict(ledger[-1], rollout=5, transitions=20480))
    path.write_text(''.join(json.dumps(row)+'\n' for row in ledger))
    assert analyzer.result_complete(result, {'steps': 20480}, 1)
    with pytest.raises(ValueError, match='rollout ledger'):
        analyzer.training_result(root, replay, report)
