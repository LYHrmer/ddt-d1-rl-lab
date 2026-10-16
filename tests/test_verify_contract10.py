"""Adversarial receipt checks with synthetic final weights and traces; zero physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools12'))
import release_contract10 as release  # noqa: E402
import verify_contract10 as gate  # noqa: E402
from test_release_contract10 import setup_plan, write_json  # noqa: E402


def result_for(plan):
    n = plan['steps']//8
    worker = {'policy_transitions': n, 'native_steps': n*10, 'pd_updates': n*5,
              'resets': 1, 'forward_calls': n*10+2}
    return {'status': 'COMPLETE', 'error': None, 'inputs_unchanged': True,
            'trained_transitions': plan['steps'], 'per_worker_counts': [worker]*8,
            'actual_counts': {key: value*8 for key, value in worker.items()}}


def setup_completed(root):
    paths = setup_plan(root)
    frozen = release.build(*paths, root=root)
    release_path = write_json(root/release.RELEASE_PATH, frozen)
    reports = {}
    for kind, plan in frozen['plans'].items():
        result = result_for(plan)
        write_json(root/plan['output']/'result.json', result)
        report = {'format': gate.supervisor.REPORT_FORMAT, 'kind': kind, 'status': 'PASS',
                  'release_sha256': release.sha(release_path), 'supervisor_sha256': frozen['supervisor_sha256'],
                  'plan': plan, 'resources': frozen['resources'], 'training_started': True,
                  'trainer_returncode': 0, 'trainer_result': result, 'stop_reason': None, 'error': None,
                  'inputs_unchanged': True, 'cleanup_complete': True, 'first_mismatch': None,
                  'equivalence': 'PASS', 'rollouts_compared': plan['compare_rollouts'],
                  'rollouts_required': plan['compare_rollouts']}
        reports[kind] = write_json(gate.supervisor.receipt_dir(root/plan['output'])/'report.json', report)
    train = root/frozen['plans']['scale']['output']
    (train/'policy.zip').write_bytes(b'fake final checkpoint, never loaded')
    protocol = {'kind': 'scale', 'arm': 'B', 'seed': 101, 'envs': 8,
                'planned_transitions': 8388608, 'planned_native_steps': 83886080,
                'config': json.loads((root/frozen['config']).read_text()),
                'inputs_sha256': {str(root/name): frozen['frozen_inputs'][name]
                                  for name in (frozen['entry'], frozen['config'])},
                'hyperparameters': {'ent_coef': 0.0, 'random_initialization': True,
                                    'teacher_policy_used': False, 'warm_start': False}}
    write_json(train/'protocol.json', protocol)
    for rank in range(8):
        (train/f'env_{rank:02d}.monitor.csv').write_text('#{}\nr,l,t\n1.0,400,1.0\n')
    (train/'progress.csv').write_text('train/std\n'+'.4\n'*16)
    events = []
    for rank in range(8):
        for event in gate.replay_tools.join(protocol['config'], 101+rank, rank,
                                            [{'r': 1., 'l': 400}], 1048576):
            event['rank'] = rank
            events.append(event)
    replay = {'format': 'd1-contract08-training-episode-replay-v1', 'physics_steps_executed': 0,
              'seed': 101, 'envs': 8, 'steps_per_env': 1048576, 'episodes': len(events),
              'inputs': {f'env_{r:02d}.monitor.csv': release.sha(train/f'env_{r:02d}.monitor.csv')
                         for r in range(8)},
              'tool_sha256': frozen['frozen_inputs']['tools08/replay_training_episodes08.py'],
              'timeline': gate.replay_tools.fine_timeline(events),
              'ppo_timeline': gate.replay_tools.ppo_timeline(train/'progress.csv')}
    replay_path = write_json(root/'evidence/career_contract_10/replay_fake/report.json', replay)
    (replay_path.parent/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    eval_dir = root/'evidence/career_contract_10/eval_fake'
    contract = {'format': 'd1-fixed-evaluation-03-v1', 'phase': 'pilot',
                'provenance': {'kind': 'self_trained_sb3_checkpoint',
                               'checkpoint_sha256': release.sha(train/'policy.zip'),
                               'training_protocol_sha256': release.sha(train/'protocol.json'),
                               'training_result_sha256': release.sha(train/'result.json'),
                               'training_kind': 'scale', 'training_seed': 101, 'training_arm': 'B',
                               'trained_transitions': 8388608, 'random_initialization': True,
                               'teacher_policy_used': False, 'vecnormalize_used': False},
                'source_sha256': {'config': frozen['frozen_inputs'][frozen['config']],
                                  'evaluation_script': frozen['frozen_inputs']['tools/evaluate_d1_ppo.py'],
                                  'scene': frozen['frozen_inputs'][protocol['config']['scene']],
                                  'robot': frozen['frozen_inputs'][str(Path(protocol['config']['scene']).parent/'robot.xml')],
                                  'interface_schema': frozen['frozen_inputs'][protocol['config']['interface_schema']]},
                'environment_sources_sha256': {str(root/name): frozen['frozen_inputs'][name]
                                               for name in gate.ENVIRONMENT_SOURCES},
                'full_phase_episodes': 3, 'requested_episodes': 3,
                'planned_max_native_steps': 18000, 'hard_max_native_steps': 18000,
                'episode_seconds': 12.0, 'policy_tick_seconds': .02, 'policy_ticks_per_episode': 600,
                'thresholds_nominal_zero_delay': {'stable_vx_mae_m_s': .06, 'stable_wz_mae_rad_s': .08,
                                                 'stop_last_second_horizontal_speed_m_s': .02,
                                                 'stop_last_second_abs_wz_rad_s': .03,
                                                 'stop_displacement_m': .05, 'peak_abs_roll_or_pitch_rad': .20},
                'scenarios': [{'sequence': task, 'delay_ticks': 0, 'reset_seed': 7001}
                              for task in gate.gates.TASKS]}
    write_json(eval_dir/'evaluation_contract.json', contract)
    rows = []
    for task in gate.gates.TASKS:
        row = {'sequence': task, 'delay_ticks': 0, 'delay_ms': 0, 'reset_seed': 7001,
                     'policy_transitions': 400, 'native_steps': 4000, 'done_reason': 'TERMINATED',
                     'failure': True, 'reference_thresholds_met': False, 'stop_displacement_m': None,
                     'stable_wz_mae_rad_s': None, 'peak_abs_roll_or_pitch_rad': .8,
                     'actual_counts': {'policy_transitions': 400, 'native_steps': 4000, 'pd_updates': 2000}}
        np.savez(eval_dir/f'{task}_delay0_seed7001.npz', body_vx_m_s=np.ones(400)*.1,
                 base_pitch_rad=np.zeros(400), boundary_xy_m=np.zeros((401, 2)))
        row['trace_file'] = f'{task}_delay0_seed7001.npz'
        row['trace_sha256'] = release.sha(eval_dir/row['trace_file'])
        rows.append(row)
    (eval_dir/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write_json(eval_dir/'summary.json', {'status': 'COMPLETE', 'phase': 'pilot', 'episodes_completed': 3,
                                       'full_phase_episodes': 3, 'actual_native_steps': 12000,
                                       'actual_policy_transitions': 1200, 'actual_pd_updates': 6000,
                                       'contract_sha256': release.sha(eval_dir/'evaluation_contract.json')})
    return release_path, reports['scale'], replay_path, eval_dir, frozen


def test_complete_supervisor_schema_is_accepted_without_old_launcher_interrupt_field(tmp_path):
    rel, report, replay, evaluation, _ = setup_completed(tmp_path)
    assert 'interrupted_by_launcher' not in json.loads(report.read_text())
    verdict = gate.evaluate(rel, report, replay, evaluation, root=tmp_path)
    assert verdict['predictions'] == {'E1': 'MET', 'TRAINING': 'MET', 'Q1': 'MET', 'Q2': 'MET',
                                      'Q3': 'NOT_MET', 'Q4': 'NOT_MET', 'Q5': 'REPORTED'}
    assert 'does not independently prove' in verdict['interpretation']
    assert verdict['qualification']['tasks_passed'] == 0
    assert verdict['physics_steps_executed'] == 0


@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('inputs_unchanged', False),
                                       ('cleanup_complete', False), ('trainer_returncode', 7),
                                       ('training_started', False), ('equivalence', 'INCOMPLETE'),
                                       ('rollouts_compared', 1172), ('rollouts_required', 1172),
                                       ('stop_reason', 'RAM_RESERVE_BELOW_3_GIB'),
                                       ('first_mismatch', {'rollout': 2}), ('error', 'error')])
def test_incomplete_training_never_reads_eval_or_replay(tmp_path, field, value):
    rel, path, _, _, _ = setup_completed(tmp_path)
    report = json.loads(path.read_text()); report[field] = value; write_json(path, report)
    verdict = gate.evaluate(rel, path, tmp_path/'NONEXISTENT_REPLAY', tmp_path/'NONEXISTENT_EVAL', root=tmp_path)
    assert verdict['predictions']['TRAINING'] == 'NOT_MET'
    assert all(verdict['predictions'][f'Q{i}'] == 'NOT_EVALUABLE' for i in range(1, 6))
    assert verdict['qualification']['overall'] == 'NOT_EVALUATED'


@pytest.mark.parametrize('change', ['raw_result', 'worker', 'counts', 'missing_result', 'null_result', 'list_result'])
def test_complete_label_cannot_hide_changed_or_incomplete_counts(tmp_path, change):
    rel, path, _, _, frozen = setup_completed(tmp_path)
    report = json.loads(path.read_text())
    result_path = tmp_path/frozen['plans']['scale']['output']/'result.json'
    result = json.loads(result_path.read_text())
    if change == 'missing_result': result_path.unlink()
    elif change in ('null_result', 'list_result'):
        write_json(result_path, None if change == 'null_result' else ['COMPLETE'])
    else:
        if change == 'raw_result': result['status'] = 'PARTIAL'
        if change == 'worker': result['per_worker_counts'][0]['policy_transitions'] -= 1
        if change == 'counts': result['actual_counts']['native_steps'] -= 1
        write_json(result_path, result)
        if change != 'raw_result':
            report['trainer_result'] = result; write_json(path, report)
    verdict = gate.evaluate(rel, path, root=tmp_path)
    assert verdict['predictions']['TRAINING'] == 'NOT_MET'
    assert verdict['qualification']['overall'] == 'NOT_EVALUATED'


@pytest.mark.parametrize('change', ['release', 'report_path', 'checkpoint', 'eval_contract', 'duplicate',
                                   'partial_eval', 'eval_counts', 'monitor', 'progress', 'timeline',
                                   'replay_source', 'replay_episodes', 'protocol', 'preflight',
                                   'replay_relabel', 'trace_hash', 'thresholds'])
def test_borrowed_tampered_or_partial_receipts_are_rejected(tmp_path, change):
    rel, report, replay, evaluation, frozen = setup_completed(tmp_path)
    train = tmp_path/frozen['plans']['scale']['output']
    if change == 'release':
        value = json.loads(report.read_text()); value['release_sha256'] = '0'*64; write_json(report, value)
    elif change == 'report_path': report = write_json(tmp_path/'foreign.json', json.loads(report.read_text()))
    elif change == 'checkpoint': (train/'policy.zip').write_bytes(b'other model')
    elif change == 'eval_contract':
        path = evaluation/'evaluation_contract.json'; value = json.loads(path.read_text())
        value['source_sha256']['evaluation_script'] = '0'*64; write_json(path, value)
    elif change == 'duplicate':
        path = evaluation/'episodes.jsonl'; path.write_text(path.read_text()+path.read_text().splitlines()[0]+'\n')
    elif change in ('partial_eval', 'eval_counts'):
        path = evaluation/'summary.json'; value = json.loads(path.read_text())
        value['status' if change == 'partial_eval' else 'actual_native_steps'] = 'PARTIAL' if change == 'partial_eval' else 1
        write_json(path, value)
    elif change == 'monitor': (train/'env_00.monitor.csv').write_text('changed')
    elif change == 'progress': (train/'progress.csv').write_text('train/std\n.9\n')
    elif change in ('timeline', 'replay_source'):
        value = json.loads(replay.read_text())
        if change == 'timeline': value['timeline'][12]['early_ends_by_segment']['drive_forward'] = 14
        else: value['inputs']['env_00.monitor.csv'] = '0'*64
        write_json(replay, value)
    elif change == 'replay_episodes': (replay.parent/'episodes.jsonl').write_text('')
    elif change == 'replay_relabel':
        path = replay.parent/'episodes.jsonl'
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[0]['ended_in'] = 'forged'
        path.write_text(''.join(json.dumps(event)+'\n' for event in events))
        value = json.loads(replay.read_text()); value['timeline'] = gate.replay_tools.fine_timeline(events)
        write_json(replay, value)
    elif change == 'trace_hash':
        path = evaluation/'episodes.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['trace_sha256'] = '0'*64
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif change == 'thresholds':
        path = evaluation/'evaluation_contract.json'; value = json.loads(path.read_text())
        value['thresholds_nominal_zero_delay']['stop_displacement_m'] = 100.
        write_json(path, value)
        path = evaluation/'summary.json'; value = json.loads(path.read_text())
        value['contract_sha256'] = release.sha(evaluation/'evaluation_contract.json')
        write_json(path, value)
    elif change == 'protocol':
        path = train/'protocol.json'; value = json.loads(path.read_text()); value['seed'] = 102; write_json(path, value)
    elif change == 'preflight':
        path = gate.supervisor.receipt_dir(tmp_path/frozen['plans']['preflight']['output'])/'report.json'
        value = json.loads(path.read_text()); value['cleanup_complete'] = False; write_json(path, value)
    with pytest.raises(SystemExit):
        gate.evaluate(rel, report, replay, evaluation, root=tmp_path)


def test_completed_training_requires_final_policy_evidence(tmp_path):
    rel, report, _, _, _ = setup_completed(tmp_path)
    with pytest.raises(SystemExit, match='requires --replay and --eval-dir'):
        gate.evaluate(rel, report, root=tmp_path)


def test_verdict_output_cannot_be_overwritten(tmp_path):
    rel, report, replay, evaluation, _ = setup_completed(tmp_path)
    output = tmp_path/'verdict'
    gate.write_verdict(rel, report, replay, evaluation, output, root=tmp_path)
    original = (output/'verdict.json').read_bytes()
    with pytest.raises(SystemExit, match='already exists'):
        gate.write_verdict(rel, report, replay, evaluation, output, root=tmp_path)
    assert (output/'verdict.json').read_bytes() == original
