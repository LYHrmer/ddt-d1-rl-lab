"""Synthetic policies/counters only. No ORT, robot, service, or physical step."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools19 import run_cpp_closedloop19 as runner  # noqa: E402


@pytest.fixture
def np():
    return pytest.importorskip('numpy')


class Clock:
    now = 0.
    def __call__(self):
        return self.now


@pytest.fixture
def worker_case(tmp_path, np):
    selected = {'selected_arm': 'zero', 'onnx_sha256': 'a'*64, 'binary_sha256': 'b'*64}
    proof = {'git_commit': 'c'*40, 'sources_sha256': {'runtime19/actor_server19.cpp': 'd'*64}}
    release = {'budget': runner.BUDGET, 'limits': runner.LIMITS, 'source_identity': proof,
               'old18_qualification': selected}
    path = tmp_path/'release.json'
    runner.atomic_json(path, release)
    fixtures = {index: {'obs': [0.]*57, 'history_oldest_to_newest_previous_only': [[0.]*57 for _ in range(10)],
                        'torch_actions': [0.]*16, 'python_ort_actions': [0.]*16} for index in range(64)}
    bundle = {'identity': selected, 'fixtures': fixtures, 'binary': tmp_path/'binary', 'model': tmp_path/'model',
              'old_release': {}, 'training_protocol': {}}
    return path, tmp_path/'worker', release, bundle


def run_fake_worker(worker_case, np, *, fault=None, memory=None):
    path, output, release, bundle = worker_case
    clock = Clock()
    facts = {'environments': [], 'cpp_calls': 0, 'oracle_inputs': [], 'actor_closed': 0}

    class Actor:
        def __init__(self, *args, **kwargs):
            assert kwargs['max_requests'] == 21664
        def start(self):
            return self
        def predict(self, obs):
            assert obs.shape == (627,) and obs.dtype == np.float32
            facts['cpp_calls'] += 1
            if facts['cpp_calls'] <= 64:
                return np.zeros(16, dtype=np.float32)
            if fault == 'timeout':
                raise TimeoutError('synthetic C++ timeout')
            return np.full(16, 1e-3 if fault == 'diverge' else 5e-7, dtype=np.float32)
        def close(self):
            facts['actor_closed'] += 1
            return {'status': 'FAIL' if fault == 'bad_eof' else 'CLOSED',
                    'cleanup_complete': fault != 'actor_cleanup', 'requests_completed': facts['cpp_calls'],
                    'inference_attempts': facts['cpp_calls'], 'inference_returns': facts['cpp_calls']}

    class Env:
        def __init__(self):
            self.counts = dict.fromkeys(runner.BUDGET, 0)
            self.actions = []
            facts['environments'].append(self)
        def step(self, action):
            self.actions.append(action.copy())
            self.counts['policy_transitions'] += 1
            self.counts['native_steps'] += 3 if fault == 'partial_native' else 10
            self.counts['pd_updates'] += 2 if fault == 'partial_native' else 5
            if fault == 'partial_native':
                raise RuntimeError('synthetic interrupted native interval')
            if fault == 'signal':
                signal.raise_signal(signal.SIGTERM)
        def counters(self):
            if fault == 'counter_unknown':
                raise OSError('synthetic counter unavailable')
            return dict(self.counts)
        def close(self):
            if fault == 'env_cleanup':
                raise OSError('synthetic environment close failure')

    def make_env(*args):
        if fault == 'construct':
            raise RuntimeError('synthetic constructor failure')
        return Env()
    def oracle(obs):
        if len(facts['oracle_inputs']) < 66:
            facts['oracle_inputs'].append(obs.copy())
        return np.zeros(16, dtype=np.float32)
    def episode(env, predict, name, delay, seed, trace, remaining):
        n = 12 if fault == 'short' and len(facts['environments']) == 1 else 600
        assert remaining >= n*10
        for tick in range(n):
            obs = np.full(627, tick, dtype=np.float32)
            env.step(predict(obs))
        with trace.open('xb') as stream:
            np.savez_compressed(stream, issued_action=np.asarray(env.actions, dtype=np.float32))
        if fault == 'late_wall' and len(facts['environments']) == 36:
            clock.now = 1000.
        return {'sequence': name, 'delay_ticks': delay, 'reset_seed': seed, 'actual_counts': env.counters(),
                'reference_thresholds_met': n == 600, 'failure': n < 600,
                'done_reason': 'HORIZON_12S' if n == 600 else 'TERMINATED', 'trace_file': trace.name,
                'trace_sha256': runner.sha(trace)}
    helper = SimpleNamespace(make_scenarios=lambda phase: runner.matrix(), SEQUENCES={name: [] for name, _, _ in runner.matrix()},
                             THRESHOLDS={'original': True}, run_episode=episode)
    loads = [0]
    def loader():
        loads[0] += 1
        if fault == 'changed' and loads[0] > 1:
            return dict(bundle, identity={'changed': True})
        return bundle
    result = runner.run_worker(path, output, bundle_loader=loader, source_checker=lambda proof: None,
        launch_validator=lambda *args: None, runtime_factory=lambda *args: SimpleNamespace(
            predict=oracle, make_env=make_env, backend={'synthetic': True}),
        actor_factory=Actor, helper_loader=lambda old: helper,
        verifier=lambda *args: None,
        memory_reader=memory or (lambda: {'effective_available_bytes': 8 << 30}), clock=clock)
    return result, facts, output


def test_cpp_actions_alone_drive_all_36_cases_with_same_input_oracle(worker_case, np):
    report, facts, output = run_fake_worker(worker_case, np)
    assert report['status'] == 'COMPLETE' and report['error'] is None
    assert report['actual_counts'] == runner.BUDGET and report['all_36_reference_pass']
    assert len(report['golden']) == 64 and len(report['cases']) == 36
    assert facts['cpp_calls'] == report['oracle_calls'] == 21664
    assert facts['actor_closed'] == 1
    assert all(np.all(action == np.float32(5e-7)) for env in facts['environments'] for action in env.actions)
    with np.load(output/report['cases'][0]['inference_file'], allow_pickle=False) as data:
        assert data['observation'].shape == (600, 627) and data['cpp_actions'].shape == (600, 16)
        assert not data['python_ort_actions'].any()
        assert np.array_equal(data['observation'][1], facts['oracle_inputs'][65])


def test_legitimate_task_failure_keeps_all_36_cases_and_actual_denominator(worker_case, np):
    report, facts, _ = run_fake_worker(worker_case, np, fault='short')
    assert report['status'] == 'COMPLETE' and not report['all_36_reference_pass']
    assert len(report['cases']) == len(facts['environments']) == 36
    assert report['actual_counts'] == {'policy_transitions': 21012, 'native_steps': 210120, 'pd_updates': 105060}


@pytest.mark.parametrize('fault', ['diverge', 'timeout', 'construct', 'partial_native', 'env_cleanup',
                                  'actor_cleanup', 'bad_eof', 'changed', 'late_wall', 'signal'])
def test_engineering_faults_never_become_complete_or_retry(worker_case, np, fault):
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    report, facts, output = run_fake_worker(worker_case, np, fault=fault)
    assert report['status'] == 'INCOMPLETE' and (output/'report.json').exists()
    assert facts['actor_closed'] == 1
    assert old == {sig: signal.getsignal(sig) for sig in old}
    if fault in ('diverge', 'timeout'):
        assert report['actual_counts'] == dict.fromkeys(runner.BUDGET, 0)
        assert len(facts['environments']) == 1 and not facts['environments'][0].actions
    if fault == 'construct':
        assert report['actual_counts'] is None and not report['cleanup_complete']
    if fault == 'partial_native':
        assert report['actual_counts'] == {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}
    if fault == 'late_wall':
        assert report['actual_counts'] == runner.BUDGET


def test_worker_memory_guard_stops_before_any_environment_creation(worker_case, np):
    calls = [0]
    def memory():
        calls[0] += 1
        return {'effective_available_bytes': (8 << 30) if calls[0] <= 2 else (7 << 29)}
    report, facts, _ = run_fake_worker(worker_case, np, memory=memory)
    assert report['status'] == 'INCOMPLETE' and not facts['environments']
    assert facts['cpp_calls'] == 64 and report['actual_counts'] == dict.fromkeys(runner.BUDGET, 0)


@pytest.mark.parametrize('values,valid', [
    ({'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}, True),
    ({'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 1}, False),
    ({'policy_transitions': True, 'native_steps': 10, 'pd_updates': 5}, False),
    ({'policy_transitions': 21601, 'native_steps': 216010, 'pd_updates': 108005}, False),
])
def test_partial_count_bounds_keep_real_mid_interval_values(values, valid):
    assert runner.valid_counts(values, partial=True) is valid


def test_default_plan_never_checks_eligibility_or_spawns(capsys, monkeypatch):
    monkeypatch.setattr(runner, 'load_eligible', lambda: pytest.fail('plan loaded runtime'))
    assert runner.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PLAN_ONLY' and report['physics_steps_executed'] == 0
    assert report['budget'] == runner.BUDGET and report['limits']['max_cpp_inferences'] == 21664
    assert report['limits']['start_effective_bytes'] == 6 << 30
    assert report['limits']['reserve_bytes'] == 4 << 30


def test_outer_low_ram_does_not_claim_attempt(tmp_path):
    result = runner.execute(root=tmp_path, bundle_loader=lambda: {'identity': {}}, source_loader=lambda: {},
                            memory_reader=lambda: {'effective_available_bytes': 5 << 30})
    assert result['status'] == 'NOT_READY' and not (tmp_path/runner.OUTPUT).exists()


def test_outer_reserve_rejects_3_point_5_gib_after_admission(tmp_path):
    samples = iter((6 << 30, 7 << 29))
    cleanups = []
    report = runner.execute(root=tmp_path, bundle_loader=lambda: {'identity': {}}, source_loader=lambda: {},
        memory_reader=lambda: {'effective_available_bytes': next(samples)},
        process_factory=lambda *args, **kwargs: pytest.fail('spawned below 4 GiB reserve'),
        adopter=lambda: None, cleanup=lambda process: cleanups.append(process) or {'remaining': {}})
    assert report['status'] == 'INCOMPLETE' and 'below 4 GiB' in report['error']
    assert report['minimum_effective_bytes'] == 7 << 29 and cleanups == [None]


def test_existing_attempt_refuses_without_loading_or_spawning(tmp_path):
    (tmp_path/runner.OUTPUT).mkdir(parents=True)
    with pytest.raises(ValueError, match='no retry'):
        runner.execute(root=tmp_path, bundle_loader=lambda: pytest.fail('reloaded old attempt'))


@pytest.fixture
def saved_worker(worker_case, np):
    """Small complete matrix with legitimate one-tick task failures; no simulator."""
    path, output, release, bundle = worker_case
    output.mkdir()
    protocol = {'format': 'd1-cpp-closedloop19-protocol-v1', 'release_sha256': runner.sha(path),
                'source_identity': release['source_identity'], 'old18_qualification': bundle['identity'],
                'budget': runner.BUDGET, 'limits': runner.LIMITS, 'action_driver': 'cpp_ort1.10_only',
                'scenarios': runner.matrix()}
    runner.atomic_json(output/'protocol.json', protocol)
    rows = []
    for task, delay, seed in runner.matrix():
        stem = f'{task}_delay{delay}_seed{seed}'
        with (output/(stem+'.npz')).open('xb') as stream:
            np.savez_compressed(stream, issued_action=np.zeros((1, 16), dtype=np.float32))
        with (output/(stem+'_inference.npz')).open('xb') as stream:
            np.savez_compressed(stream, observation=np.zeros((1, 627), dtype=np.float32),
                cpp_actions=np.zeros((1, 16), dtype=np.float32), python_ort_actions=np.zeros((1, 16), dtype=np.float32),
                max_absolute_error=np.zeros(1, dtype=np.float64))
        rows.append({'sequence': task, 'delay_ticks': delay, 'reset_seed': seed, 'reference_thresholds_met': False,
            'actual_counts': {'policy_transitions': 1, 'native_steps': 10, 'pd_updates': 5}, 'observations': 1,
            'trace_file': stem+'.npz', 'trace_sha256': runner.sha(output/(stem+'.npz')),
            'inference_file': stem+'_inference.npz', 'inference_sha256': runner.sha(output/(stem+'_inference.npz'))})
    counts = {'policy_transitions': 36, 'native_steps': 360, 'pd_updates': 180}
    child = {'format': 'd1-cpp-closedloop19-worker-v1', 'release_sha256': runner.sha(path),
        'source_identity': release['source_identity'], 'old18_qualification': bundle['identity'],
        'status': 'COMPLETE', 'cleanup_complete': True, 'inputs_unchanged': True, 'error': None,
        'protocol_sha256': runner.sha(output/'protocol.json'), 'cases': rows, 'actual_counts': counts,
        'known_actual_counts': counts, 'maximum_same_input_error': 0., 'all_36_reference_pass': False,
        'golden': [{'index': i, 'actions': [0.]*16, 'python_ort_actions': [0.]*16, 'max_absolute_error': 0.} for i in range(64)],
        'oracle_calls': 100, 'cpp_actor': {'status': 'CLOSED', 'cleanup_complete': True,
            'requests_completed': 100, 'inference_attempts': 100, 'inference_returns': 100}}
    return output, child, release, bundle


def test_saved_worker_independently_rechecks_matrix_arrays_and_actual_driver(saved_worker):
    output, child, release, bundle = saved_worker
    verified = []
    counts = runner.verify_completed_worker(output, child, release, bundle, helpers=object(),
        verifier=lambda *args: verified.append(args[1]['sequence']))
    assert counts == {'policy_transitions': 36, 'native_steps': 360, 'pd_updates': 180} and len(verified) == 36


@pytest.mark.parametrize('fault', ['driver', 'dtype', 'error', 'oracle_nan', 'actor_eof', 'actor_bool',
                                  'golden_error', 'aggregate', 'qualified', 'wrong_protocol'])
def test_readback_rejects_forged_success_even_with_rehashed_arrays(saved_worker, np, fault):
    output, child, release, bundle = saved_worker
    item = child['cases'][0]
    if fault == 'driver':
        trace = output/item['trace_file']
        with trace.open('wb') as stream:
            np.savez_compressed(stream, issued_action=np.full((1, 16), 1e-6, dtype=np.float32))
        item['trace_sha256'] = runner.sha(trace)
    elif fault in ('dtype', 'error', 'oracle_nan'):
        path = output/item['inference_file']
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if fault == 'dtype':
            arrays['observation'] = arrays['observation'].astype(np.float64)
        elif fault == 'error':
            arrays['max_absolute_error'][0] = 1e-6
        else:
            arrays['python_ort_actions'][0, 0] = np.nan
        with path.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        item['inference_sha256'] = runner.sha(path)
    elif fault == 'actor_eof':
        child['cpp_actor']['status'] = 'FAIL'
    elif fault == 'actor_bool':
        child['cpp_actor']['inference_returns'] = True
    elif fault == 'golden_error':
        child['golden'][0]['max_absolute_error'] = 1e-3
    elif fault == 'aggregate':
        child['actual_counts'] = dict(runner.BUDGET)
    elif fault == 'qualified':
        child['all_36_reference_pass'] = True
    else:
        protocol = runner.read_json(output/'protocol.json')
        protocol['old18_qualification'] = {'another_model': True}
        runner.atomic_json(output/'protocol.json', protocol)
        child['protocol_sha256'] = runner.sha(output/'protocol.json')
    with pytest.raises(ValueError):
        runner.verify_completed_worker(output, child, release, bundle, helpers=object(), verifier=lambda *args: None)


def fake_parent(tmp_path, *, fault=None):
    clock, cleanups, sources = Clock(), [], {'source': 'synthetic'}
    bundle = {'identity': {'checkpoint': 'synthetic'}}
    calls = [0]
    def loader():
        calls[0] += 1
        if fault == 'final_wall' and calls[0] > 1:
            clock.now = 901.
        return bundle
    def factory(command, **kwargs):
        if fault == 'spawn_error':
            raise InterruptedError('fake factory failed after a possible spawn')
        output = tmp_path/runner.OUTPUT
        child_dir = output/'worker'
        child_dir.mkdir()
        release = runner.read_json(output/'release.json')
        protocol = {'format': 'd1-cpp-closedloop19-protocol-v1', 'release_sha256': runner.sha(output/'release.json'),
                    'source_identity': sources, 'old18_qualification': bundle['identity'],
                    'budget': runner.BUDGET, 'limits': runner.LIMITS}
        if fault == 'foreign_partial':
            protocol['old18_qualification'] = {'another_model': True}
        runner.atomic_json(child_dir/'protocol.json', protocol)
        counts = {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2} if fault in ('partial', 'foreign_partial') else dict(runner.BUDGET)
        child = {'format': 'd1-cpp-closedloop19-worker-v1', 'release_sha256': runner.sha(output/'release.json'),
                 'source_identity': sources, 'old18_qualification': bundle['identity'], 'status': 'COMPLETE',
                 'error': None, 'cleanup_complete': True, 'inputs_unchanged': True, 'actual_counts': counts,
                 'known_actual_counts': counts, 'protocol_sha256': runner.sha(child_dir/'protocol.json'),
                 'cases': [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed} for name, delay, seed in runner.matrix()],
                 'golden': [{'index': i} for i in range(64)], 'all_36_reference_pass': True}
        runner.atomic_json(child_dir/'report.json', child)
        if fault == 'spawn_signal':
            signal.raise_signal(signal.SIGTERM)
        class Process:
            pid = 98765
            returncode = 2 if fault in ('partial', 'foreign_partial') else 0
            def poll(self):
                if fault == 'late_exit':
                    clock.now = 881.
                return self.returncode
        return Process()
    def cleanup(process):
        cleanups.append(process)
        if fault == 'cleanup_exception':
            raise OSError('synthetic cleanup failed')
        return {'remaining': {123: 1} if fault == 'residual' else {}}
    result = runner.execute(root=tmp_path, bundle_loader=loader, source_loader=lambda: sources,
        memory_reader=lambda: {'effective_available_bytes': 8 << 30}, process_factory=factory,
        adopter=lambda: None, cleanup=cleanup, proof_checker=lambda *args: None, clock=clock, sleeper=lambda seconds: None)
    return result, cleanups


def test_parent_claims_once_checks_proof_and_cleans_fake_child(tmp_path):
    report, cleanups = fake_parent(tmp_path)
    assert report['status'] == 'COMPLETE' and report['actual_counts'] == runner.BUDGET
    assert len(cleanups) == 1 and report['cleanup_complete']
    assert (tmp_path/runner.OUTPUT/'release.json').exists()


@pytest.mark.parametrize('fault', ['spawn_error', 'spawn_signal', 'late_exit', 'final_wall', 'cleanup_exception', 'residual'])
def test_parent_failure_and_final_deadline_never_hide_behind_exit_zero(tmp_path, fault):
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    report, cleanups = fake_parent(tmp_path, fault=fault)
    assert report['status'] == 'INCOMPLETE' and len(cleanups) == 1
    assert handlers == {sig: signal.getsignal(sig) for sig in handlers}
    if fault == 'spawn_error':
        assert cleanups == [None] and report['actual_counts'] is None
    if fault in ('cleanup_exception', 'residual'):
        assert report['cleanup_complete'] is False
    if fault == 'final_wall':
        assert report['actual_counts'] == runner.BUDGET and report['elapsed_s'] == 901.


@pytest.mark.parametrize('foreign', [False, True])
def test_failed_child_partial_counts_require_model_bound_protocol(tmp_path, foreign):
    report, _ = fake_parent(tmp_path, fault='foreign_partial' if foreign else 'partial')
    assert report['status'] == 'INCOMPLETE'
    if foreign:
        assert report['actual_counts'] is None and report['known_actual_counts'] == dict.fromkeys(runner.BUDGET, 0)
    else:
        assert report['actual_counts'] == {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}


def test_no_handle_cleanup_still_scans_owned_tree(monkeypatch):
    monkeypatch.setattr(runner, 'descendants', lambda pid: {})
    monkeypatch.setattr(runner.os, 'waitpid', lambda *args: (0, 0))
    assert runner.cleanup_owned_tree(None)['remaining'] == {}


def test_old_full_qualification_cannot_hide_an_early_case():
    rows = [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed, 'reference_thresholds_met': True,
             'failure': False, 'done_reason': 'HORIZON_12S', 'actual_counts': {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000}}
            for name, delay, seed in runner.matrix()]
    rows[0]['actual_counts'] = {'policy_transitions': 1, 'native_steps': 10, 'pd_updates': 5}
    report = {'status': 'COMPLETE', 'cleanup_complete': True, 'inputs_unchanged': True, 'error': None,
              'all_36_reference_pass': True, 'actual_counts': runner.BUDGET, 'cases': rows}
    with pytest.raises(ValueError, match='contradicts'):
        runner.complete_reference(report, runner.matrix(), lambda *args: None, Path('.'), object())


@pytest.mark.parametrize('field,value', [('status', 'RUNNING'), ('full_delay_qualification', False),
                                         ('selected_arm', 'random'), ('plan_sha256', 'another-plan')])
def test_eligibility_requires_closed_18_and_rederived_winner(tmp_path, field, value):
    path = tmp_path/runner.PLAN18
    plan = {'output_root': 'old18/execution_01', 'stage_order': [], 'stages': {}, 'source_identity': {'old_source': 'proof'}}
    path.parent.mkdir(parents=True)
    runner.atomic_json(path, plan)
    execution = tmp_path/plan['output_root']
    execution.mkdir(parents=True)
    outer = {'status': 'COMPLETE', 'cleanup_complete': True, 'error': None, 'full_delay_qualification': True,
             'plan_sha256': runner.sha(path), 'source_identity': plan['source_identity'], 'stages': {}, 'selected_arm': 'zero'}
    outer[field] = value
    runner.atomic_json(execution/'report.json', outer)
    rows = [{'sequence': task, 'delay_ticks': delay, 'reset_seed': seed} for task, delay, seed in runner.matrix()]
    for arm in ('zero', 'random'):
        (execution/('eval_'+arm)).mkdir()
        runner.atomic_json(execution/('eval_'+arm)/'report.json', {'status': 'COMPLETE', 'cleanup_complete': True,
            'inputs_unchanged': True, 'error': None, 'cases': rows, 'all_36_reference_pass': False})
    called = []
    api = SimpleNamespace(load_plan=lambda p: plan, verify_inputs=lambda p: ({},))
    evaluator = SimpleNamespace(selection=lambda p: called.append('selection') or 'zero')
    with pytest.raises(ValueError, match='not completely qualified|winner differs'):
        runner.load_eligible(tmp_path, common=api, evaluator=evaluator, numerical=object(),
            verifier=lambda *args: None, helper_loader=lambda old: object())
    assert called == (['selection'] if field == 'selected_arm' else [])
