"""Formal closeout guard checks use saved or synthetic arrays, never a simulator."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools17 import evaluate_formal17 as formal
from tools17 import provenance17 as provenance
from tools17 import run_closeout17 as runner
from tools16.evaluate_comparison16 import frozen_helpers


def helper():
    # Bind the real, unchanged metric source; compiling pure helpers does not load MuJoCo.
    path = Path(formal.ROOT / 'tools/evaluate_d1_ppo.py')
    return frozen_helpers({'sources_sha256': {'tools/evaluate_d1_ppo.py': provenance.sha256(path)}})


def test_formal_matrix_is_exact_and_ordered():
    matrix = formal.scenarios(helper())
    assert len(matrix) == len(set(matrix)) == 36
    assert {seed for _, _, seed in matrix} == {1001, 1002, 1003}
    assert {delay for _, delay, _ in matrix} == {0, 1, 2, 3}
    assert matrix[0] == ('forward_stop', 0, 1001)
    assert matrix[-1] == ('turn_stop', 3, 1003)


@pytest.mark.parametrize('change', ['pilot', 'missing', 'duplicate', 'wrong_seed', 'wrong_delay'])
def test_scenario_drift_is_rejected(change):
    h = helper()
    matrix = h.make_scenarios('formal')
    if change == 'pilot':
        matrix = h.make_scenarios('pilot')
    elif change == 'missing':
        matrix.pop()
    elif change == 'duplicate':
        matrix[-1] = matrix[0]
    elif change == 'wrong_seed':
        matrix[0] = ('forward_stop', 0, 7001)
    else:
        matrix[0] = ('forward_stop', 4, 1001)
    with pytest.raises(ValueError, match='scenario matrix'):
        formal.scenarios(SimpleNamespace(make_scenarios=lambda _: matrix))


def trace_fixture(tmp_path, *, delay=3, seed=1002, n=600, reason='HORIZON_12S'):
    h = helper()
    name = 'forward_stop'
    zeros = np.zeros(n)
    commands = []
    for tick in range(n):
        selected = h.SEQUENCES[name][0]
        for candidate in h.SEQUENCES[name]:
            if candidate['start_s'] <= tick * .02 + 1e-12:
                selected = candidate
        commands.append((selected['vx'], 0., selected['wz']))
    trace = {key: zeros.copy() for key in ('reward', 'body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s',
             'base_x_m', 'base_y_m', 'base_height_m', 'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad',
             'tilt_rad', 'tau_saturation_count')}
    trace.update(command_during_interval=np.asarray(commands), issued_action=np.zeros((n, 16), dtype=np.float32),
                 applied_action=np.zeros((n, 16)), boundary_xy_m=np.zeros((n + 1, 2)),
                 policy_interval_start_s=np.arange(n) * .02,
                 fallen=np.zeros(n, dtype=np.bool_), illegal_body_contact=np.zeros(n, dtype=np.bool_))
    item = h.trace_metrics(name, delay, seed, trace, trace['boundary_xy_m'], reason)
    item['actual_counts'] = {'policy_transitions': n, 'native_steps': n * 10, 'pd_updates': n * 5,
                             'resets': 1, 'forward_calls': n + 2}
    item['trace_file'] = f'{name}_delay{delay}_seed{seed}.npz'
    path = tmp_path / item['trace_file']
    np.savez_compressed(path, **trace)
    item['trace_sha256'] = provenance.sha256(path)
    return h, trace, item


def test_trace_uses_actual_formal_delay_and_seed(tmp_path):
    h, _, item = trace_fixture(tmp_path)
    assert formal.verify_trace(tmp_path, item, h)['nominal_zero_delay_pass'] is None
    assert item['delay_ticks'] == 3 and item['reset_seed'] == 1002


def test_short_case_is_valid_observation_but_not_full_horizon(tmp_path):
    h, _, item = trace_fixture(tmp_path, n=40, reason='TERMINATED')
    result = formal.verify_trace(tmp_path, item, h)
    assert result['failure'] is True
    assert result['threshold_flags']['completed_12s'] is False


@pytest.mark.parametrize('change', ['clock', 'commands', 'boundary', 'nan', 'action_shape', 'flag_dtype',
                                   'metric', 'sha', 'counter', 'filename', 'false_horizon'])
def test_corrupt_trace_or_receipt_cannot_pass(tmp_path, change):
    h, trace, item = trace_fixture(tmp_path)
    if change == 'clock':
        trace['policy_interval_start_s'][3] += .001
    elif change == 'commands':
        trace['command_during_interval'][100, 0] += .1
    elif change == 'boundary':
        trace['boundary_xy_m'][10, 0] += .1
    elif change == 'nan':
        trace['body_vx_m_s'][0] = np.nan
    elif change == 'action_shape':
        trace['issued_action'] = np.zeros((600, 15))
    elif change == 'flag_dtype':
        trace['fallen'] = trace['fallen'].astype(int)
    elif change == 'metric':
        item['stop_displacement_m'] += .001
    elif change == 'sha':
        item['trace_sha256'] = '0' * 64
    elif change == 'counter':
        item['actual_counts']['native_steps'] -= 1
    elif change == 'filename':
        item['trace_file'] = '../foreign.npz'
    else:
        trace['reward'] = np.zeros(599)
    if change in ('clock', 'commands', 'boundary', 'nan', 'action_shape', 'flag_dtype', 'false_horizon'):
        path = tmp_path / item['trace_file']
        np.savez_compressed(path, **trace)
        item['trace_sha256'] = provenance.sha256(path)
    with pytest.raises((ValueError, KeyError)):
        formal.verify_trace(tmp_path, item, h)


@pytest.mark.parametrize('bad', [-1, 1.5, True, None])
def test_invalid_partial_counter_is_rejected_without_changing_total(bad):
    total = dict.fromkeys(formal.BUDGET, 0)
    before = copy.deepcopy(total)
    attempted = {'policy_transitions': 1, 'native_steps': bad, 'pd_updates': 2}
    with pytest.raises(ValueError):
        formal.add_counts(total, attempted)
    assert total == before


def test_middle_interval_attempts_are_retained_and_cap_is_atomic():
    total = dict.fromkeys(formal.BUDGET, 0)
    assert formal.add_counts(total, {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}) == {
        'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    with pytest.raises(ValueError):
        formal.add_counts(total, {'policy_transitions': 21601, 'native_steps': 1, 'pd_updates': 1})
    assert all(value == 0 for value in total.values())


@pytest.mark.parametrize('bad', [True, -1, 21601, 1.25])
def test_supervisor_rejects_untrusted_counts(bad):
    assert not runner.valid_counts({'policy_transitions': bad, 'native_steps': 0, 'pd_updates': 0})


def test_stage_commands_only_export_cpp_then_formal_and_never_train(tmp_path):
    release = {'training_root': str(tmp_path), 'output_root': runner.OUTPUT_ROOT}
    commands = runner.commands(release, tmp_path / 'release.json')
    assert [row['name'] for row in commands] == ['export', 'cpp', 'formal']
    assert not any('train_comparison' in word or '--learn' in word for row in commands for word in row['argv'])
    assert '--cpp-report' in commands[-1]['argv']


def test_standalone_formal_cannot_skip_supervisor(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    release = {'training_root': str(tmp_path), 'output_root': runner.OUTPUT_ROOT}
    output = tmp_path / runner.OUTPUT_ROOT / 'formal'
    with pytest.raises(FileNotFoundError):
        runner.validate_launch(release, tmp_path / 'release.json', output)


class ArrayEnvironment:
    """The actual frozen rollout helper consumes this array-only environment."""
    def __init__(self, sequence, *, short=False, interrupt=False, close_error=False):
        self.sequence, self.short, self.interrupt, self.close_error = sequence, short, interrupt, close_error
        self.data = SimpleNamespace(qpos=np.zeros(7))
        self.free_qpos, self.eval_delay_ticks, self.tick, self.closed = 0, 0, 0, False
        self.count = {'policy_transitions': 0, 'native_steps': 0, 'pd_updates': 0, 'resets': 0, 'forward_calls': 0}

    def counters(self):
        return dict(self.count)

    def reset(self, seed):
        self.count['resets'] += 1
        self.count['forward_calls'] += 1
        return np.zeros(627, dtype=np.float32), {'seed': seed}

    def step(self, action):
        self.count['policy_transitions'] += 1
        self.count['native_steps'] += 3 if self.interrupt else 10
        self.count['pd_updates'] += 2 if self.interrupt else 5
        if self.interrupt:
            raise RuntimeError('synthetic native interval interrupted')
        segment = self.sequence[0]
        for candidate in self.sequence:
            if candidate['start_s'] <= self.tick * .02 + 1e-12:
                segment = candidate
        self.tick += 1
        self.count['forward_calls'] += 1
        info = {key: 0. for key in ('body_vx_m_s', 'body_vy_m_s', 'body_wz_rad_s', 'base_x_m', 'base_y_m',
                'base_height_m', 'base_roll_rad', 'base_pitch_rad', 'base_yaw_rad', 'tilt_rad')}
        info.update(command_during_interval=(segment['vx'], 0., segment['wz']), fallen=self.short,
                    illegal_body_contact=False, tau_saturation_count=0, applied_action=np.asarray(action))
        return np.zeros(627, dtype=np.float32), 0., self.short and self.tick == 40, False, info

    def close(self):
        self.closed = True
        if self.close_error:
            raise RuntimeError('synthetic close failure')


def fake_execution(tmp_path, monkeypatch, *, short=False, interrupt=False, construct_error=False, close_error=False):
    release_path = tmp_path / 'release.json'
    release_path.write_text('{}')
    release = {'training_root': str(tmp_path), 'stage_limits_s': {'formal': 600},
               'source_identity': {'fake': True}, 'resources': {
                   'reserve_bytes': 3 * 2**30, 'start_min_effective_bytes': 6 * 2**30}}
    h = helper()
    environments = []
    def runtime_factory(*_):
        def make_env(_, sequence, delay, seed):
            if construct_error:
                raise RuntimeError('synthetic constructor failed before counters available')
            env = ArrayEnvironment(sequence, short=short, interrupt=interrupt, close_error=close_error)
            environments.append(env)
            return env
        return SimpleNamespace(make_env=make_env, predict=lambda _: np.zeros(16, dtype=np.float32),
                               backend={'kind': 'array_only_no_model'})
    monkeypatch.setattr(formal, 'frozen_helpers', lambda _: h)
    monkeypatch.setattr(formal, 'check_numeric_inputs', lambda *_: (tmp_path / 'policy.onnx', {'onnx_sha256': 'a' * 64}))
    monkeypatch.setattr(provenance, 'check_sources', lambda _: None)
    result = formal.run(release_path, tmp_path, tmp_path / 'cpp.json', tmp_path / 'formal',
                        runtime_factory=runtime_factory, loader=lambda _: release,
                        launch_validator=lambda *_: None,
                        memory_reader=lambda: {'effective_available_bytes': 8 * 2**30},
                        bundle_loader=lambda _: ({}, {}, {'checkpoint_sha256': 'c' * 64}, tmp_path / 'policy.zip'))
    return result, environments


def test_actual_frozen_rollout_collects_all_36_without_physics(tmp_path, monkeypatch):
    result, envs = fake_execution(tmp_path, monkeypatch)
    assert result['status'] == 'COMPLETE'
    assert result['actual_counts'] == formal.BUDGET
    assert len(result['cases']) == len(envs) == 36
    assert result['full_horizons_complete'] is True
    assert result['all_36_reference_pass'] is False  # A zero-speed actor does not track commands.
    assert result['nominal_zero_delay_pass_count'] == 0
    assert all(env.closed for env in envs)
    assert all(group['cases'] == 9 for group in result['by_delay_ms'].values())


def test_task_failure_collects_the_whole_registered_matrix(tmp_path, monkeypatch):
    result, envs = fake_execution(tmp_path, monkeypatch, short=True)
    assert result['status'] == 'COMPLETE' and len(result['cases']) == 36
    assert result['full_horizons_complete'] is False and result['all_36_reference_pass'] is False
    assert result['actual_counts']['policy_transitions'] == 36 * 40
    assert all(env.closed for env in envs)


def test_mid_interval_failure_stops_and_retains_attempted_physics(tmp_path, monkeypatch):
    result, envs = fake_execution(tmp_path, monkeypatch, interrupt=True)
    assert result['status'] == 'INCOMPLETE' and result['cases'] == []
    assert len(envs) == 1 and envs[0].closed
    assert result['actual_counts'] == {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    assert 'native interval interrupted' in result['error']


def test_failed_constructor_keeps_unknown_counter_and_cleanup(tmp_path, monkeypatch):
    result, envs = fake_execution(tmp_path, monkeypatch, construct_error=True)
    assert result['status'] == 'INCOMPLETE' and result['actual_counts'] is None
    assert result['cleanup_complete'] is False and envs == []


def test_failed_close_does_not_double_count_the_completed_case(tmp_path, monkeypatch):
    result, envs = fake_execution(tmp_path, monkeypatch, close_error=True)
    assert result['status'] == 'INCOMPLETE' and result['cleanup_complete'] is False
    assert len(envs) == 1
    assert result['actual_counts'] == {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000}
