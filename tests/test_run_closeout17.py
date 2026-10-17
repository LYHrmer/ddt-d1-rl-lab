"""The closeout supervisor is tested with synthetic receipts and fake processes.

No child command, checkpoint loader, neural library, or simulator is executed.
"""
import copy
import json
from pathlib import Path
import signal

import pytest

from tools17 import run_closeout17 as runner


GIB = 2**30
ZERO = dict.fromkeys(runner.BUDGET, 0)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


def physical_counts(ticks):
    return {'policy_transitions': ticks, 'native_steps': ticks * 10, 'pd_updates': ticks * 5}


class FakeProcess:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.pid = 123456789  # Never used for an operating-system signal.
        self.returncode = None
        self.polls = 0

    def poll(self):
        self.polls += 1
        if self.returncode is None and self.name not in self.owner.running:
            self.returncode = self.owner.returncodes.get(self.name, 0)
        return self.returncode


class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / 'repository'
        self.root.mkdir()
        self.release_path = self.root / 'evidence/closeout17/release_01.json'
        self.release = {
            'output_root': 'evidence/closeout17/execution_01',
            'budget': dict(runner.BUDGET),
            'source_identity': {'git_commit': 'a' * 40, 'sources_sha256': {'fake.py': 'b' * 64}},
            'checkpoint_sha256': 'c' * 64,
            'training_root': str(tmp_path / 'historical_training'),
        }
        write_json(self.release_path, self.release)
        self.execution = self.root / self.release['output_root']
        self.now = 100.
        self.memory_values = []
        self.memory_calls = 0
        self.spawned, self.cleaned, self.children = [], [], {}
        self.running, self.returncodes = set(), {}
        self.modifiers, self.spawn_hooks = {}, {}
        self.spawn_errors, self.cleanup_errors = {}, {}
        self.cleanup_values = {}
        self.contract_calls = 0
        self.contract_hook = None
        self.short_ticks = None
        monkeypatch.setattr(runner, 'ROOT', self.root)
        monkeypatch.setattr(runner, 'load_contract', self.load_contract)
        monkeypatch.setattr(runner, 'read_memory', self.memory)
        monkeypatch.setattr(runner.time, 'monotonic', lambda: self.now)
        monkeypatch.setattr(runner.time, 'sleep', self.sleep)
        monkeypatch.setattr(runner.subprocess, 'Popen', self.popen)
        monkeypatch.setattr(runner, 'cleanup', self.cleanup)

    def load_contract(self, path):
        assert Path(path) == self.release_path
        self.contract_calls += 1
        if self.contract_hook:
            self.contract_hook(self.contract_calls)
        return copy.deepcopy(self.release)

    def memory(self):
        self.memory_calls += 1
        value = self.memory_values.pop(0) if self.memory_values else 8 * GIB
        if isinstance(value, BaseException):
            raise value
        return {'available_ram_bytes': value, 'cgroup_remaining_bytes': None,
                'effective_available_bytes': value}

    def sleep(self, seconds):
        self.now += seconds

    def make_receipt(self, name):
        child = {'status': 'PASS', 'source_identity': copy.deepcopy(self.release['source_identity']),
                 'checkpoint_sha256': self.release['checkpoint_sha256'], 'error': None}
        if name != 'formal':
            return child
        output = self.execution / name
        protocol = {'format': 'd1-closeout17-formal-protocol-v1',
                    'release_sha256': runner.provenance.sha256(self.release_path),
                    'checkpoint_sha256': self.release['checkpoint_sha256'],
                    'source_identity': copy.deepcopy(self.release['source_identity']),
                    'budget_max': dict(runner.BUDGET), 'episode_ticks': 600,
                    'scenarios': [{'sequence': task, 'delay_ticks': delay, 'reset_seed': seed}
                                  for task in ('forward_stop', 'reverse_stop', 'turn_stop')
                                  for delay in (0, 1, 2, 3) for seed in (1001, 1002, 1003)]}
        write_json(output / 'protocol.json', protocol)
        ticks = self.short_ticks or 600
        cases = [{'sequence': task, 'delay_ticks': delay, 'reset_seed': seed,
                  'done_reason': 'TERMINATED' if self.short_ticks else 'HORIZON_12S',
                  'actual_counts': physical_counts(ticks), 'reference_thresholds_met': False}
                 for task in ('forward_stop', 'reverse_stop', 'turn_stop')
                 for delay in (0, 1, 2, 3) for seed in (1001, 1002, 1003)]
        child.update(format='d1-closeout17-formal-result-v1', status='COMPLETE',
                     release_sha256=runner.provenance.sha256(self.release_path),
                     protocol_sha256=runner.provenance.sha256(output / 'protocol.json'),
                     inputs_unchanged=True, cleanup_complete=True, cases=cases,
                     actual_counts=physical_counts(ticks * 36),
                     full_horizons_complete=not bool(self.short_ticks), all_36_reference_pass=False)
        return child

    def popen(self, argv, **kwargs):
        name = kwargs['env']['D1_CLOSEOUT17_STAGE']
        self.spawned.append(name)
        assert name == runner.ORDER[len(self.spawned) - 1]
        assert kwargs['cwd'] == self.root and kwargs['start_new_session'] is True
        assert kwargs['env']['OMP_NUM_THREADS'] == kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
        journal = self.report()
        stage = journal['stages'][name]
        assert journal['active_stage'] == name and stage['status'] == 'STARTED'
        assert stage['command'] == argv
        assert stage['launch_token'] == kwargs['env']['D1_CLOSEOUT17_LAUNCH_TOKEN']
        assert len(stage['launch_token']) == 32
        assert all(journal['stages'][prior]['status'] == 'COMPLETE'
                   for prior in runner.ORDER[:runner.ORDER.index(name)])
        if name in self.spawn_errors:
            raise self.spawn_errors[name]
        child = self.make_receipt(name)
        if name in self.modifiers:
            self.modifiers[name](child)
        if child is not None:
            write_json(self.execution / name / 'report.json', child)
        process = FakeProcess(self, name)
        self.children[name] = process
        if name in self.spawn_hooks:
            self.spawn_hooks[name](process)
        return process

    def cleanup(self, process, resources, *, natural_exit):
        assert resources == runner.RESOURCES
        assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in (signal.SIGINT, signal.SIGTERM))
        self.cleaned.append((process.name, natural_exit))
        if process.name in self.cleanup_errors:
            raise self.cleanup_errors[process.name]
        if process.returncode is None:
            process.returncode = -signal.SIGINT
        return self.cleanup_values.get(process.name, ([], True))

    def report(self):
        return json.loads((self.execution / 'report.json').read_text())

    def execute(self):
        return runner.execute(self.release_path)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def test_complete_closeout_obeys_stage_order_and_byte_manifests(harness):
    assert harness.execute() == 0
    report = harness.report()
    assert harness.spawned == list(runner.ORDER)
    assert harness.cleaned == [(name, True) for name in runner.ORDER]
    assert report['status'] == 'COMPLETE' and report['active_stage'] is None
    assert report['actual_counts'] == report['known_actual_counts'] == runner.BUDGET
    assert report['unknown_count_stages'] == [] and report['training_transitions'] == 0
    assert len({stage['launch_token'] for stage in report['stages'].values()}) == 3
    for name, stage in report['stages'].items():
        assert stage['status'] == 'COMPLETE' and stage['cleanup_complete'] is True
        assert stage['receipt_sha256'] == runner.provenance.sha256(harness.execution / name / 'report.json')
        assert stage['artifacts_sha256']
        for path, digest in stage['artifacts_sha256'].items():
            assert runner.provenance.sha256(path) == digest


def test_valid_short_task_cases_complete_engineering_without_passing_task(harness):
    harness.short_ticks = 40
    assert harness.execute() == 0
    report = harness.report()
    assert report['status'] == 'COMPLETE' and report['actual_counts'] == physical_counts(36 * 40)
    child = report['stages']['formal']['receipt']
    assert child['full_horizons_complete'] is False and child['all_36_reference_pass'] is False


@pytest.mark.parametrize('stage', runner.ORDER)
def test_engineering_child_nonzero_stops_later_stages(harness, stage):
    harness.returncodes[stage] = 1
    assert harness.execute() == 1
    report = harness.report()
    assert harness.spawned == list(runner.ORDER[:runner.ORDER.index(stage) + 1])
    assert report['status'] == 'FAIL' and report['failed_stage'] == stage
    assert report['stages'][stage]['stop_reason'] == 'STAGE_FAILED'
    assert report['actual_counts'] == (runner.BUDGET if stage == 'formal' else ZERO)


@pytest.mark.parametrize('stage', ('export', 'cpp'))
@pytest.mark.parametrize('field,value', [('status', 'FAIL'), ('checkpoint_sha256', 'f' * 64),
                                       ('source_identity', {'git_commit': 'f' * 40})])
def test_invalid_numeric_receipt_blocks_formal(harness, stage, field, value):
    harness.modifiers[stage] = lambda child: child.update({field: value})
    assert harness.execute() == 1
    report = harness.report()
    assert stage == report['failed_stage'] and 'formal' not in harness.spawned
    assert report['actual_counts'] == ZERO and report['unknown_count_stages'] == []
    assert report['stages'][stage]['stop_reason'] == 'CHILD_COMPLETION_INVALID'


def partial_receipt(child):
    child.update(status='INCOMPLETE', error='interrupted inside one native interval', cases=[],
                 actual_counts={'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2},
                 inputs_unchanged=False)


def test_true_partial_preserves_noninteger_interval_physical_counts(harness):
    harness.modifiers['formal'] = partial_receipt
    harness.returncodes['formal'] = 1
    assert harness.execute() == 1
    report = harness.report()
    assert report['actual_counts'] == report['known_actual_counts'] == {
        'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    assert report['unknown_count_stages'] == [] and report['status'] == 'FAIL'


@pytest.mark.parametrize('foreign', ['checkpoint', 'proof', 'protocol_sha', 'release', 'format'])
def test_foreign_partial_is_unknown_instead_of_known_zero(harness, foreign):
    def alter(child):
        partial_receipt(child)
        key, value = {'checkpoint': ('checkpoint_sha256', 'f' * 64),
                      'proof': ('source_identity', {'git_commit': 'f' * 40}),
                      'protocol_sha': ('protocol_sha256', 'f' * 64),
                      'release': ('release_sha256', 'f' * 64),
                      'format': ('format', 'some-other-receipt')}[foreign]
        child[key] = value
    harness.modifiers['formal'] = alter
    harness.returncodes['formal'] = 1
    assert harness.execute() == 1
    report = harness.report()
    assert report['actual_counts'] is None and report['known_actual_counts'] == ZERO
    assert report['unknown_count_stages'] == ['formal']


@pytest.mark.parametrize('key,value', [('checkpoint_sha256', 'f' * 64),
                                     ('release_sha256', 'f' * 64),
                                     ('source_identity', {'git_commit': 'f' * 40}),
                                     ('format', 'foreign-protocol')])
def test_foreign_protocol_content_cannot_supply_known_partial(harness, key, value):
    def alter(child):
        partial_receipt(child)
        path = harness.execution / 'formal/protocol.json'
        protocol = json.loads(path.read_text())
        protocol[key] = value
        write_json(path, protocol)
        child['protocol_sha256'] = runner.provenance.sha256(path)
    harness.modifiers['formal'] = alter
    harness.returncodes['formal'] = 1
    assert harness.execute() == 1
    assert harness.report()['actual_counts'] is None
    assert harness.report()['unknown_count_stages'] == ['formal']


@pytest.mark.parametrize('counts', [None, {'policy_transitions': True, 'native_steps': 10, 'pd_updates': 5},
    {'policy_transitions': 1, 'native_steps': -1, 'pd_updates': 2},
    {'policy_transitions': 1, 'native_steps': 216001, 'pd_updates': 2},
    {'policy_transitions': 1, 'native_steps': 11, 'pd_updates': 2},
    {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 6}])
def test_invalid_or_impossible_partial_counters_remain_unknown(harness, counts):
    def alter(child):
        partial_receipt(child)
        child['actual_counts'] = counts
    harness.modifiers['formal'] = alter
    harness.returncodes['formal'] = 1
    assert harness.execute() == 1
    assert harness.report()['actual_counts'] is None
    assert harness.report()['unknown_count_stages'] == ['formal']


@pytest.mark.parametrize('memory', [6 * GIB - 1, OSError('memory unreadable')])
def test_admission_refusal_does_not_claim_attempt(harness, memory):
    harness.memory_values = [memory]
    assert harness.execute() == 2
    assert not harness.execution.exists() and harness.spawned == []
    paths = list((harness.execution.parent / 'readiness_checks').glob('*.json'))
    assert len(paths) == 1
    receipt = json.loads(paths[0].read_text())
    assert receipt['status'] == 'NOT_READY' and receipt['execution_claimed'] is False


def test_readiness_refusal_can_later_admit_without_retrying_claimed_attempt(harness):
    harness.memory_values = [5 * GIB]
    assert harness.execute() == 2
    assert harness.execute() == 0 and harness.spawned == list(runner.ORDER)
    with pytest.raises(ValueError, match='already claimed'):
        harness.execute()
    assert harness.spawned == list(runner.ORDER)


@pytest.mark.parametrize('failure', ['reserve', 'read_error', 'timeout'])
def test_running_child_is_cleaned_on_memory_or_time_failure(harness, failure):
    harness.running.add('export')
    if failure == 'reserve':
        harness.memory_values = [8 * GIB, 8 * GIB, 3 * GIB - 1]
    elif failure == 'read_error':
        harness.memory_values = [8 * GIB, 8 * GIB, OSError('cgroup read failed')]
    else:
        harness.spawn_hooks['export'] = lambda _: setattr(harness, 'now', harness.now + 211.)
    assert harness.execute() == 1
    report = harness.report()
    assert harness.spawned == ['export'] and harness.cleaned == [('export', False)]
    assert report['status'] == 'FAIL' and report['actual_counts'] == ZERO
    assert report['stages']['export']['cleanup_complete'] is True
    message = report['stages']['export']['error']
    assert {'reserve': 'below 3 GiB', 'read_error': 'cgroup read failed', 'timeout': 'wall limit'}[failure] in message
    samples = [json.loads(line) for line in (harness.execution / 'resources.jsonl').read_text().splitlines()]
    assert all(sample['stage'] == 'export' for sample in samples)
    if failure == 'read_error':
        assert samples[-1]['sample_valid'] is False


def test_child_already_exited_after_wall_limit_cannot_escape_deadline(harness):
    harness.spawn_hooks['export'] = lambda _: setattr(harness, 'now', harness.now + 211.)
    assert harness.execute() == 1
    assert harness.spawned == ['export']
    assert 'wall limit' in harness.report()['stages']['export']['error']


def test_missing_formal_receipt_after_spawn_keeps_unknown_counts(harness):
    harness.spawn_hooks['formal'] = lambda _: (harness.execution / 'formal/report.json').unlink()
    assert harness.execute() == 1
    report = harness.report()
    assert report['actual_counts'] is None and report['unknown_count_stages'] == ['formal']


@pytest.mark.parametrize('field,value', [('cleanup_complete', False), ('inputs_unchanged', False),
                                       ('error', 'case failure'), ('cases', [{}] * 35)])
def test_formal_receipt_requires_engineering_completion_fields(harness, field, value):
    harness.modifiers['formal'] = lambda child: child.update({field: value})
    assert harness.execute() == 1
    assert harness.report()['stages']['formal']['stop_reason'] == 'CHILD_COMPLETION_INVALID'


def test_stage_start_requires_six_gib_again(harness):
    harness.memory_values = [8 * GIB, 8 * GIB, 6 * GIB - 1]
    assert harness.execute() == 1
    assert harness.spawned == ['export'] and harness.report()['failed_stage'] == 'export'
    assert 'stage start effective RAM below 6 GiB' in harness.report()['error']


@pytest.mark.parametrize('stage', runner.ORDER)
def test_spawn_failure_records_unstarted_process_and_no_later_spawn(harness, stage):
    harness.spawn_errors[stage] = OSError('synthetic spawn failure')
    assert harness.execute() == 1
    record = harness.report()['stages'][stage]
    assert record['process_started'] is False and record['cleanup_complete'] is True
    assert record['returncode'] is None and 'spawn failure' in record['error']
    assert stage not in [name for name, _ in harness.cleaned]
    assert harness.report()['actual_counts'] == ZERO


@pytest.mark.parametrize('cleanup_failure', ['unclean', 'residual', 'raises'])
def test_cleanup_failure_blocks_next_stage_and_restores_signal_handlers(harness, cleanup_failure):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    if cleanup_failure == 'unclean':
        harness.cleanup_values['export'] = (['SIGINT', 'SIGTERM', 'SIGKILL'], False)
    elif cleanup_failure == 'residual':
        harness.cleanup_values['export'] = (['SIGINT'], True)
    else:
        harness.cleanup_errors['export'] = RuntimeError('synthetic cleanup failure')
    assert harness.execute() == 1
    report = harness.report()
    assert harness.spawned == ['export'] and report['status'] == 'FAIL'
    assert report['stages']['export']['stop_reason'] == 'CLEANUP_INCOMPLETE_OR_RESIDUAL'
    assert all(signal.getsignal(sig) == old for sig, old in previous.items())


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_signal_during_popen_is_deferred_until_handle_owned_and_cleaned(harness, sig):
    previous = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(_):
        handler = signal.getsignal(sig)
        assert callable(handler)
        handler(sig, None)  # Same registered handler, without signalling this test process.
    harness.spawn_hooks['export'] = interrupt
    assert harness.execute() == 1
    assert harness.cleaned == [('export', False)]
    record = harness.report()['stages']['export']
    assert record['process_started'] is True and 'interrupted during spawn' in record['error']
    assert all(signal.getsignal(item) == old for item, old in previous.items())


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_cleanup_ignores_repeated_signals_and_restores_handlers(harness, monkeypatch, sig):
    original = harness.cleanup
    def guarded(process, resources, *, natural_exit):
        assert signal.getsignal(sig) == signal.SIG_IGN
        return original(process, resources, natural_exit=natural_exit)
    monkeypatch.setattr(runner, 'cleanup', guarded)
    assert harness.execute() == 0


@pytest.mark.parametrize('sig', [signal.SIGINT, signal.SIGTERM])
def test_running_signal_cleans_owned_process_and_restores_handlers(harness, sig):
    previous = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}
    def arm(process):
        def interrupted_poll():
            handler = signal.getsignal(sig)
            assert callable(handler)
            handler(sig, None)
        process.poll = interrupted_poll
    harness.spawn_hooks['export'] = arm
    assert harness.execute() == 1
    assert harness.cleaned == [('export', False)] and harness.spawned == ['export']
    assert signal.Signals(sig).name in harness.report()['stages']['export']['error']
    assert all(signal.getsignal(item) == old for item, old in previous.items())


def test_completed_artifact_mutation_blocks_formal(harness):
    def mutate(_):
        path = harness.execution / 'export/report.json'
        path.write_text(path.read_text() + ' ')
    harness.spawn_hooks['cpp'] = mutate
    assert harness.execute() == 1
    report = harness.report()
    assert harness.spawned == ['export', 'cpp']
    assert 'completed stage artifact changed' in report['error']


def test_release_mutation_blocks_later_spawn(harness):
    harness.spawn_hooks['export'] = lambda _: harness.release_path.write_text('{}\n')
    assert harness.execute() == 1
    assert harness.spawned == ['export']
    assert 'frozen release changed' in harness.report()['error']


def test_source_revalidation_failure_prevents_second_stage(harness):
    def check(call):
        if call == 3:  # initial load, export pre-spawn, then cpp pre-spawn
            raise ValueError('synthetic source changed')
    harness.contract_hook = check
    assert harness.execute() == 1
    assert harness.spawned == ['export'] and 'source changed' in harness.report()['error']


def test_plan_prints_commands_without_claim_or_spawn(harness, capsys):
    assert runner.main(['plan', '--release', str(harness.release_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'PLAN_ONLY' and result['execution_started'] is False
    assert [stage['name'] for stage in result['commands']] == list(runner.ORDER)
    assert harness.spawned == [] and not harness.execution.exists()
