"""Supervisor tests use synthetic receipts/processes only: no workers, models or physics."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ENTRY = Path(__file__).resolve().parents[1] / 'tools16/run_comparison16.py'
SPEC = importlib.util.spec_from_file_location('comparison16_runner_test', ENTRY)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
GIB = 1024**3


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


class Clock:
    now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def memory(host=8*GIB, cap=None):
    return {'available_ram_bytes': host, 'cgroup_remaining_bytes': cap,
            'effective_available_bytes': min(host, cap) if cap is not None else host}


class FakeAPI:
    def __init__(self, root):
        self.ROOT = root
        self.samples = []
        self.source = root/'source.txt'
        self.source.write_text('synthetic frozen source')
        self.source_sha = runner.sha(self.source)
    def load_release(self, path, require_authorized=True):
        result = runner.read_json(path)
        if require_authorized and result['status'] != 'AUTHORIZED_FROZEN':
            raise ValueError('prepared plan is not authorized to execute')
        self.validate_sources(result)
        return result
    def validate_sources(self, release):
        if runner.sha(self.source) != self.source_sha:
            raise ValueError('source changed')
    def read_memory(self):
        result = self.samples.pop(0) if self.samples else memory()
        if isinstance(result, BaseException):
            raise result
        return result


class FakeProcess:
    def __init__(self, factory, name, pid, duration, code):
        self.factory, self.name, self.pid = factory, name, pid
        self.returncode, self.end, self.code = None, factory.clock.now+duration, code
        self.group_alive = True
    def poll(self):
        if self.returncode is None and self.factory.clock.now >= self.end:
            self.returncode = self.code
            self.group_alive = self.name in self.factory.residual
        return self.returncode
    def send_signal(self, sig):
        self.factory.signals.append((self.pid, sig))
        if sig not in self.factory.ignore:
            if self.returncode is None:
                self.returncode = -sig
            self.group_alive = False


class Processes:
    def __init__(self, api, release, path, clock):
        self.api, self.release, self.path, self.clock = api, release, path, clock
        self.started, self.processes, self.signals = [], {}, []
        self.duration, self.codes, self.mutate = {}, {}, {}
        self.ignore, self.residual = set(), set()
        self.during_spawn = None
    def members(self, pid):
        process = self.processes.get(pid)
        if process:
            process.poll()
        return [pid] if process and process.group_alive else []
    def killpg(self, pid, sig):
        assert pid in self.processes, 'must never signal a foreign process group'
        self.processes[pid].send_signal(sig)
    def __call__(self, argv, **kwargs):
        assert not any(item.poll() is None or item.group_alive for item in self.processes.values())
        assert kwargs['start_new_session'] is True and kwargs['cwd'] == self.api.ROOT
        name = kwargs['env']['D1_COMPARISON16_STAGE']
        journal = runner.read_json(self.api.ROOT/self.release['output_root']/'report.json')
        assert journal['active_stage'] == name
        assert journal['stages'][name]['launch_token'] == kwargs['env']['D1_COMPARISON16_LAUNCH_TOKEN']
        assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
        self.started.append(name)
        self.make_artifacts(name)
        process = FakeProcess(self, name, 500000+len(self.started), self.duration.get(name, 2.), self.codes.get(name, 0))
        self.processes[process.pid] = process
        if self.during_spawn:
            self.during_spawn(name)
        return process
    def make_artifacts(self, name):
        release, execution = self.release, self.api.ROOT/self.release['output_root']
        release_sha = runner.sha(self.path)
        if name == 'compare':
            value = {'format': 'd1-comparison16-comparison-v1', 'status': 'COMPLETE',
                     'release_sha256': release_sha, 'verdict': 'NO_DEMONSTRATED_BENEFIT',
                     'inputs_sha256': {arm: runner.sha(execution/f'eval_{arm}/report.json')
                                       for arm in ('control', 'velocity')}}
            path = execution/'comparison.json'
        else:
            stage, arm = name.split('_', 1)
            output = execution/name
            protocol = {'release_sha256': release_sha, 'stage': stage, 'arm': arm,
                        'inputs_sha256': release['sources_sha256']}
            write(output/'protocol.json', protocol)
            transitions = release['stages'][stage]['transitions']
            value = {'status': 'COMPLETE', 'release_sha256': release_sha, 'arm': arm,
                     'seed': release['seed'], 'envs': release['envs'], 'aux_coef': release['arms'][arm]['aux_coef'],
                     'protocol_sha256': runner.sha(output/'protocol.json'), 'inputs_unchanged': True,
                     'cleanup_complete': True, 'error': None, 'actual_counts': runner.expected_counts(transitions)}
            if stage == 'eval':
                train = execution/f'train_{arm}'
                value.update(format='d1-comparison16-evaluation-result-v1',
                             checkpoint_sha256=runner.sha(train/'policy.zip'),
                             training_protocol_sha256=runner.sha(train/'protocol.json'),
                             training_result_sha256=runner.sha(train/'result.json'),
                             full_horizons_complete=True,
                             cases=[{'sequence': task, 'policy_transitions': 600, 'native_steps': 6000,
                                     'done_reason': 'HORIZON_12S', 'actual_counts': runner.expected_counts(600)}
                                    for task in ('forward_stop', 'reverse_stop', 'turn_stop')])
                path = output/'report.json'
            else:
                (output/'policy.zip').write_bytes(b'fake policy '+arm.encode())
                workers = [runner.expected_counts(transitions//release['envs']) for _ in range(release['envs'])]
                value.update(format='d1-comparison16-training-result-v1', stage=stage,
                             trained_transitions=transitions, checkpoint_sha256=runner.sha(output/'policy.zip'),
                             per_worker_counts=workers, eligible_for_evaluation=stage == 'train')
                rows = []
                for index in range(1, transitions//4096+1):
                    deterministic = {'rollout': index, 'transitions': index*4096,
                                     'actual_counts': runner.expected_counts(index*4096),
                                     'per_worker_counts': [runner.expected_counts(index*4096//release['envs'])
                                                           for _ in range(release['envs'])],
                                     'phase_rewards': {'drive': {'ticks': 4096, 'reward_sum': 12.}},
                                     'policy_state_sha256': 'same initial weights' if index == 1 else f'{arm}-{index}'}
                    rows.append({'rollout': index, 'transitions': index*4096, 'deterministic': deterministic,
                                 'elapsed_training_s': 99 if stage == 'train' else 1, 'resource_sample': memory()})
                (output/'ledger.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
                path = output/'result.json'
        write(path, value)
        if name in self.mutate:
            self.mutate[name](path)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    api = FakeAPI(tmp_path)
    release = {'format': 'd1-local-comparison-v16', 'status': 'AUTHORIZED_FROZEN', 'envs': 4, 'seed': 101,
               'arms': {'control': {'aux_coef': 0.}, 'velocity': {'aux_coef': 1.}},
               'stages': {'preflight': {'transitions': 32768, 'max_wall_s': 900},
                          'train': {'transitions': 1048576, 'max_wall_s': 5400},
                          'eval': {'transitions': 1800, 'max_wall_s': 300},
                          'compare': {'transitions': 0, 'max_wall_s': 60}},
               'resources': {'start_min_effective_bytes': 6*GIB, 'reserve_bytes': 3*GIB, 'poll_s': 1.,
                             'interrupt_grace_s': 10, 'terminate_grace_s': 5, 'kill_grace_s': 2,
                             'process_wall_grace_s': 30},
               'sources_sha256': {'source.txt': api.source_sha},
               'output_root': 'evidence/local_comparison16/execution_01'}
    path = tmp_path/'release.json'
    write(path, release)
    clock = Clock()
    processes = Processes(api, release, path, clock)
    monkeypatch.setattr(runner.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(runner.time, 'sleep', clock.sleep)
    monkeypatch.setattr(runner.subprocess, 'Popen', processes)
    monkeypatch.setattr(runner, 'live_group_members', processes.members)
    monkeypatch.setattr(runner.os, 'killpg', processes.killpg)
    monkeypatch.setattr(runner, 'contract_api', lambda: api)
    return SimpleNamespace(api=api, release=release, path=path, clock=clock, processes=processes,
                           output=tmp_path/release['output_root'])


def report(setup):
    return runner.read_json(setup.output/'report.json')


def mutate_receipt(change):
    def perform(path):
        value = runner.read_json(path)
        change(value)
        write(path, value)
    return perform


def test_default_cli_plan_never_reads_memory_or_spawns(setup, monkeypatch, capsys):
    setup.release['status'] = 'PREPARED_NOT_AUTHORIZED'
    write(setup.path, setup.release)
    monkeypatch.setattr(setup.api, 'read_memory', lambda: pytest.fail('plan read memory'))
    assert runner.main(['--release', str(setup.path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PLAN_ONLY' and result['execution_authorized'] is False
    assert [row['name'] for row in result['stages']] == list(runner.ORDER)
    assert result['stages'][-1]['process_wall_limit_s'] == 90
    assert setup.processes.started == [] and not setup.output.exists()


def test_prepared_execute_rejected_before_claim(setup):
    setup.release['status'] = 'PREPARED_NOT_AUTHORIZED'
    write(setup.path, setup.release)
    with pytest.raises(ValueError, match='not authorized'):
        runner.execute(setup.path)
    assert not setup.output.exists() and not setup.processes.started


@pytest.mark.parametrize('sample', [memory(5*GIB), memory(8*GIB, 5*GIB), OSError('missing cgroup'),
                                   {'available_ram_bytes': 8*GIB, 'effective_available_bytes': None}])
def test_readiness_refusal_does_not_consume_attempt(setup, sample):
    setup.api.samples = [sample]
    assert runner.execute(setup.path) == 2
    assert not setup.output.exists() and not setup.processes.started
    files = list((setup.output.parent/'readiness_checks').glob('*.json'))
    assert len(files) == 1
    value = runner.read_json(files[0])
    assert value['status'] == 'NOT_READY' and value['execution_claimed'] is False
    assert runner.execute(setup.path) == 0  # A refusal did not consume the single spawn opportunity.


@pytest.mark.parametrize('envs', [2, 4])
def test_all_seven_stages_are_sequential_bound_and_counted(setup, envs):
    setup.release['envs'] = envs
    write(setup.path, setup.release)
    assert runner.execute(setup.path) == 0
    value = report(setup)
    assert value['status'] == 'COMPLETE' and value['child_processes_started'] == 7
    assert setup.processes.started == list(runner.ORDER)
    assert all(row['cleanup_complete'] is True and not row['cleanup_signals'] for row in value['stages'].values())
    assert value['actual_counts'] == runner.expected_counts(2166288)
    assert value['unknown_count_stages'] == [] and value['active_stage'] is None
    assert value['stages']['train_control']['rollouts_compared'] == 8
    commands = [row['command'] for row in value['stages'].values()]
    assert str(setup.output/'train_control/policy.zip') in commands[4]
    assert str(setup.output/'train_velocity/policy.zip') in commands[5]
    before = (setup.output/'report.json').read_bytes()
    with pytest.raises(FileExistsError, match='no retry'):
        runner.execute(setup.path)
    assert (setup.output/'report.json').read_bytes() == before


def test_nonzero_exit_with_complete_label_stops_round(setup):
    setup.processes.codes['preflight_control'] = 7
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert value['failed_stage'] == 'preflight_control'
    assert value['stages']['preflight_control']['stop_reason'] == 'CHILD_EXIT_NONZERO'
    assert setup.processes.started == ['preflight_control']


@pytest.mark.parametrize('fault', ['incomplete', 'counts', 'workers', 'checkpoint', 'source', 'protocol', 'missing'])
def test_exit_zero_does_not_override_bad_completion(setup, fault):
    def mutate(path):
        if fault == 'missing':
            path.unlink()
            return
        value = runner.read_json(path)
        if fault == 'incomplete': value['status'] = 'INCOMPLETE'
        if fault == 'counts': value['actual_counts']['native_steps'] -= 1
        if fault == 'workers': value['per_worker_counts'] = []
        if fault == 'checkpoint': value['checkpoint_sha256'] = 'wrong'
        if fault == 'source': value['inputs_unchanged'] = False
        if fault == 'protocol': value['protocol_sha256'] = 'wrong'
        write(path, value)
    setup.processes.mutate['preflight_control'] = mutate
    assert runner.execute(setup.path) == 1
    value = report(setup)['stages']['preflight_control']
    assert value['stop_reason'] == 'CHILD_COMPLETION_INVALID'
    assert setup.processes.started == ['preflight_control']
    if fault == 'missing':
        assert report(setup)['actual_counts'] is None
        assert report(setup)['unknown_count_stages'] == ['preflight_control']


def test_resource_stop_persists_real_below_reserve_sample(setup):
    setup.api.samples = [memory(), memory(8*GIB, 3*GIB-1)]
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['stop_reason'] == 'RAM_RESERVE_BELOW_3_GIB'
    assert row['stop_sample']['effective_available_bytes'] == 3*GIB-1
    assert row['cleanup_signals'] == ['SIGINT'] and row['cleanup_complete'] is True
    assert report(setup)['minimum_observed_effective_bytes'] == 3*GIB-1
    assert setup.processes.started == ['preflight_control']


@pytest.mark.parametrize('bad', [OSError('counter disappeared'), memory(-1),
                                {'available_ram_bytes': 8*GIB, 'cgroup_remaining_bytes': None,
                                 'effective_available_bytes': float('nan')}])
def test_failed_running_resource_sample_is_unknown_not_zero(setup, bad):
    setup.api.samples = [memory(), bad]
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['stop_reason'] == 'RESOURCE_READ_FAILED'
    assert row['stop_sample']['sample_valid'] is False and row['stop_sample']['memory'] is None
    samples = [json.loads(line) for line in (setup.output/'resources.jsonl').read_text().splitlines()]
    assert samples[1] == row['stop_sample']
    assert report(setup)['minimum_observed_effective_bytes'] == 8*GIB


def test_later_stage_headroom_refusal_stops_without_spawning_it(setup):
    # admission, first stage two running samples, after_exit, then next admission
    setup.api.samples = [memory(), memory(), memory(), memory(), memory(5*GIB)]
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert value['failed_stage'] == 'preflight_velocity'
    assert value['stages']['preflight_velocity']['stop_reason'] == 'STAGE_START_RAM_HEADROOM'
    assert setup.processes.started == ['preflight_control']


def test_ignored_int_and_term_get_bounded_kill_only_own_group(setup):
    setup.api.samples = [memory(), memory(2*GIB)]
    setup.processes.ignore = {signal.SIGINT, signal.SIGTERM}
    setup.processes.duration['preflight_control'] = 1000
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['cleanup_signals'] == ['SIGINT', 'SIGTERM', 'SIGKILL']
    assert row['cleanup_complete'] is True
    assert 15 <= setup.clock.now < 18
    assert {pid for pid, _ in setup.processes.signals} == {500001}


def test_partial_interval_counts_are_kept_without_rounding_to_full_ratio(setup):
    counts = {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    setup.processes.codes['preflight_control'] = 2
    setup.processes.mutate['preflight_control'] = mutate_receipt(
        lambda value: value.update(status='INCOMPLETE', actual_counts=counts))
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert value['actual_counts'] == counts and value['known_actual_counts'] == counts
    assert value['unknown_count_stages'] == []


@pytest.mark.parametrize('partial', [False, True])
def test_real_monitor_five_key_counts_preserve_core_subtotal(setup, partial):
    core = ({'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2} if partial
            else runner.expected_counts(32768))
    counts = dict(core, resets=4, forward_calls=8)
    setup.processes.codes['preflight_control'] = 2 if partial else 7
    setup.processes.mutate['preflight_control'] = mutate_receipt(
        lambda value: value.update(status='INCOMPLETE' if partial else 'COMPLETE', actual_counts=counts))
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert value['actual_counts'] == core and value['known_actual_counts'] == core
    assert value['unknown_count_stages'] == []


@pytest.mark.parametrize('extra', [-1, None, True, 1.5])
def test_invalid_monitor_extra_counter_is_unknown(setup, extra):
    assert runner.partial_counts(dict(runner.expected_counts(1), resets=extra), 32768) is False


def test_fractional_worker_extra_cannot_pass_by_integer_total(setup):
    def change(value):
        value['actual_counts']['resets'] = 6
        for worker in value['per_worker_counts']:
            worker['resets'] = 1.5
    setup.processes.mutate['preflight_control'] = mutate_receipt(change)
    assert runner.execute(setup.path) == 1
    assert report(setup)['stages']['preflight_control']['stop_reason'] == 'CHILD_COMPLETION_INVALID'
    assert setup.processes.started == ['preflight_control']


def test_complete_real_five_key_monitor_receipts_pass_and_total_only_core(setup):
    def change(value):
        value['actual_counts'].update(resets=4, forward_calls=8)
        for worker in value['per_worker_counts']:
            worker.update(resets=1, forward_calls=2)
    for stage in runner.ORDER[:4]:
        setup.processes.mutate[stage] = mutate_receipt(change)
    assert runner.execute(setup.path) == 0
    assert report(setup)['actual_counts'] == runner.expected_counts(2166288)
    assert report(setup)['unknown_count_stages'] == []


@pytest.mark.parametrize('field', ['format', 'release_sha256', 'arm', 'stage', 'seed', 'envs', 'aux_coef', 'protocol_sha256'])
def test_foreign_partial_receipt_is_unknown_in_subtotal(setup, field):
    def change(value):
        value.update(status='INCOMPLETE', inputs_unchanged=False,
                     actual_counts={'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2})
        value[field] = 'foreign'
    setup.processes.codes['preflight_control'] = 2
    setup.processes.mutate['preflight_control'] = mutate_receipt(change)
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert value['actual_counts'] is None and value['known_actual_counts'] == runner.expected_counts(0)
    assert value['unknown_count_stages'] == ['preflight_control']


def test_bound_failed_partial_still_counts_when_inputs_changed(setup):
    counts = {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    setup.processes.codes['preflight_control'] = 2
    setup.processes.mutate['preflight_control'] = mutate_receipt(
        lambda value: value.update(status='INCOMPLETE', inputs_unchanged=False, cleanup_complete=False,
                                   error='physical interval interrupted', actual_counts=counts))
    assert runner.execute(setup.path) == 1
    assert report(setup)['actual_counts'] == counts and report(setup)['unknown_count_stages'] == []


@pytest.mark.parametrize('reason', ['TERMINATED', 'ENV_TRUNCATED_BEFORE_12S'])
def test_legal_short_control_case_keeps_paired_velocity_evaluation(setup, reason):
    def short(value):
        case = value['cases'][0]
        case.update(policy_transitions=100, native_steps=1000, done_reason=reason,
                    actual_counts=runner.expected_counts(100))
        value.update(full_horizons_complete=False, actual_counts=runner.expected_counts(1300))
    setup.processes.mutate['eval_control'] = mutate_receipt(short)
    setup.processes.mutate['compare'] = mutate_receipt(
        lambda value: value.update(status='INCOMPLETE', verdict='INCOMPLETE', scores=None))
    setup.processes.codes['compare'] = 2
    assert runner.execute(setup.path) == 1
    value = report(setup)
    assert setup.processes.started == list(runner.ORDER)
    assert value['stages']['eval_control']['status'] == value['stages']['eval_velocity']['status'] == 'COMPLETE'
    assert value['failed_stage'] == 'compare' and value['stages']['compare']['child_receipt']['scores'] is None
    assert value['actual_counts'] == runner.expected_counts(2166288-500)


@pytest.mark.parametrize('fault', ['false_full', 'false_short', 'case_sum', 'short_horizon', 'bad_reason', 'duplicate'])
def test_malformed_short_case_or_full_flag_stops_engineering_round(setup, fault):
    def change(value):
        if fault == 'false_full':
            value['cases'][0].update(policy_transitions=100, native_steps=1000, done_reason='TERMINATED',
                                     actual_counts=runner.expected_counts(100))
            value['actual_counts'] = runner.expected_counts(1300)
        if fault == 'false_short': value['full_horizons_complete'] = False
        if fault == 'case_sum': value['actual_counts'] = runner.expected_counts(1300)
        if fault == 'short_horizon':
            value['cases'][0].update(policy_transitions=100, native_steps=1000, actual_counts=runner.expected_counts(100))
            value.update(actual_counts=runner.expected_counts(1300), full_horizons_complete=False)
        if fault == 'bad_reason': value['cases'][0]['done_reason'] = 'ENGINEERING_EXCEPTION'
        if fault == 'duplicate': value['cases'][0]['sequence'] = 'turn_stop'
    setup.processes.mutate['eval_control'] = mutate_receipt(change)
    assert runner.execute(setup.path) == 1
    assert setup.processes.started == list(runner.ORDER[:5])
    assert report(setup)['stages']['eval_control']['stop_reason'] == 'CHILD_COMPLETION_INVALID'


def test_process_wall_bound_includes_compare(setup):
    setup.processes.duration['compare'] = 1000
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['compare']
    assert row['stop_reason'] == 'PROCESS_WALL_LIMIT' and 90 <= row['wall_s'] < 91
    assert row['cleanup_complete'] is True


def test_residual_helpers_are_cleaned_and_fail_closed(setup):
    setup.processes.residual.add('preflight_control')
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['returncode'] == 0 and row['stop_reason'] == 'RESIDUAL_PROCESS_GROUP'
    assert row['cleanup_complete'] is True and row['cleanup_signals'] == ['SIGINT']


def test_cleanup_failure_still_leaves_failed_receipt(setup, monkeypatch):
    monkeypatch.setattr(runner, 'cleanup', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('signal failed')))
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['cleanup_complete'] is False and row['stop_reason'] == 'PROCESS_GROUP_CLEANUP_INCOMPLETE'


def change_ledger(path, index, key, value):
    ledger = path.parent/'ledger.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    rows[index]['deterministic'][key] = value
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_cross_arm_initial_rollout_must_match_before_aux_update(setup):
    setup.processes.mutate['preflight_velocity'] = lambda path: change_ledger(path, 0, 'policy_state_sha256', 'different seed')
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_velocity']
    assert row['stop_reason'] == 'CHILD_COMPLETION_INVALID' and 'first PPO update' in row['error']
    assert setup.processes.started == ['preflight_control', 'preflight_velocity']


def test_formal_mismatch_stops_on_first_observed_rollout(setup):
    setup.processes.mutate['train_control'] = lambda path: change_ledger(path, 2, 'policy_state_sha256', 'changed')
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['train_control']
    assert row['stop_reason'] == 'PREFLIGHT_PREFIX_MISMATCH'
    assert row['rollouts_compared'] == 2
    assert row['first_mismatch'] == {'rollout': 3, 'differing_fields': ['policy_state_sha256']}
    assert setup.processes.started == list(runner.ORDER[:3])


def test_ledger_partial_last_line_waits_and_nonfinite_fails(tmp_path):
    path = tmp_path/'ledger.jsonl'
    path.write_text('{"deterministic": {"rollout": 1}}\n{"deterministic":')
    assert len(runner.ledger_rows(path)) == 1
    path.write_text('{"deterministic": {"reward": NaN}}\n')
    with pytest.raises(ValueError, match='nonfinite'):
        runner.ledger_rows(path)


@pytest.mark.parametrize('fault', ['source', 'artifact', 'release'])
def test_between_stage_mutation_stops_round(setup, fault):
    original = setup.processes.mutate
    def change(path):
        if fault == 'source': setup.api.source.write_text('modified')
        if fault == 'release': setup.path.write_text(setup.path.read_text()+' ')
        if fault == 'artifact':
            # Mutate only after the original COMPLETE receipt's SHA was recorded.
            pass
    original['preflight_control'] = change
    if fault == 'artifact':
        original['preflight_velocity'] = lambda path: (setup.output/'preflight_control/policy.zip').write_bytes(b'changed')
    assert runner.execute(setup.path) == 1
    assert report(setup)['status'] == 'FAIL' and 'changed' in report(setup)['error']
    assert len(setup.processes.started) == (2 if fault == 'artifact' else 1)


def test_sigterm_during_popen_is_deferred_until_group_owned(setup):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    setup.processes.during_spawn = lambda name: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    assert runner.execute(setup.path) == 1
    row = report(setup)['stages']['preflight_control']
    assert row['process_started'] is True and row['stop_reason'] == 'SUPERVISOR_INTERRUPTED'
    assert row['cleanup_complete'] is True and row['cleanup_signals'] == ['SIGINT']
    assert all(signal.getsignal(sig) == old for sig, old in before.items())


def launch_journal(setup, stage, monkeypatch):
    spec = {item['name']: item for item in runner.commands(setup.release, setup.path, setup.api.ROOT)}[stage]
    value = {'format': 'd1-comparison16-execution-v1', 'status': 'RUNNING',
             'release_sha256': runner.sha(setup.path), 'runner_sha256': runner.sha(ENTRY),
             'supervisor_pid': os.getppid(), 'active_stage': stage,
             'stages': {name: {'status': 'COMPLETE'} for name in runner.ORDER[:runner.ORDER.index(stage)]}}
    value['stages'][stage] = {'status': 'STARTED', 'command': spec['argv'], 'launch_token': 'synthetic token'}
    write(setup.output/'report.json', value)
    monkeypatch.setenv('D1_COMPARISON16_STAGE', stage)
    monkeypatch.setenv('D1_COMPARISON16_LAUNCH_TOKEN', 'synthetic token')
    return value, spec


@pytest.mark.parametrize('stage', runner.ORDER)
def test_active_stage_ticket_allows_only_registered_child(setup, stage, monkeypatch):
    _, spec = launch_journal(setup, stage, monkeypatch)
    runner.validate_stage_launch(setup.release, setup.path, stage, spec['output'], root=setup.api.ROOT)


@pytest.mark.parametrize('fault', ['no_journal', 'no_token', 'token', 'parent', 'prior', 'argv', 'active', 'release', 'runner', 'status', 'output'])
def test_standalone_or_stale_stage_cannot_bypass_order(setup, fault, monkeypatch):
    stage = 'train_control'
    value, spec = launch_journal(setup, stage, monkeypatch)
    path = setup.output/'report.json'
    if fault == 'no_token': monkeypatch.delenv('D1_COMPARISON16_LAUNCH_TOKEN')
    if fault == 'token': monkeypatch.setenv('D1_COMPARISON16_LAUNCH_TOKEN', 'old token')
    if fault == 'parent': value['supervisor_pid'] += 1
    if fault == 'prior': value['stages']['preflight_velocity']['status'] = 'FAIL'
    if fault == 'argv': value['stages'][stage]['command'] += ['--resume']
    if fault == 'active': value['active_stage'] = 'train_velocity'
    if fault in ('release', 'runner'): value[f'{fault}_sha256'] = 'wrong'
    if fault == 'status': value['status'] = 'COMPLETE'
    write(path, value)
    if fault == 'no_journal': path.unlink()
    with pytest.raises((ValueError, OSError)):
        runner.validate_stage_launch(setup.release, setup.path, stage,
                                     str(setup.output/'elsewhere') if fault == 'output' else spec['output'], root=setup.api.ROOT)


def test_import_is_stdlib_only_in_fresh_process():
    script = ('import importlib.util,sys; '
              f's=importlib.util.spec_from_file_location("runner",{str(ENTRY)!r}); '
              'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
              'assert not {"torch","numpy","mujoco","stable_baselines3","onnxruntime"}&sys.modules.keys()')
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
