"""Synthetic policies/counters only. No ORT, robot, service, or physical step."""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools23 import run_cpp_diagnostic23 as runner


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
    proof = {'git_commit': 'c'*40, 'sources_sha256': {'runtime19/actor_server19.cpp': 'd'*64, 'tools19/cpp_actor19.py': 'e'*64}}
    selected['actor_identities'] = {'model_sha256': 'a'*64, 'binary_sha256': 'b'*64,
        'source_sha256': 'd'*64, 'adapter_sha256': 'e'*64, 'ort_library_sha256': 'f'*64}
    release = {'format': 'd1-cpp-diagnostic23-release-v1', 'one_attempt': True, 'training_allowed': False,
               'retry_or_resume_allowed': False, **runner.SCOPE, 'budget': runner.BUDGET, 'limits': runner.LIMITS, 'source_identity': proof,
               'baseline17_provenance': selected}
    path = tmp_path/'release.json'
    runner.atomic_json(path, release)
    fixtures = {index: {'obs': [0.]*57, 'history_oldest_to_newest_previous_only': [[0.]*57 for _ in range(10)],
                        'torch_actions': [0.]*16, 'python_ort_actions': [0.]*16} for index in range(64)}
    bundle = {'identity': selected, 'fixtures': fixtures, 'binary': tmp_path/'binary', 'model': tmp_path/'model',
              'old_release': {}, 'training_protocol': {}}
    return path, tmp_path/'worker', release, bundle


def run_fake_worker(worker_case, np, *, fault=None, memory=None):
    path, output, _release, bundle = worker_case
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
    assert all(report[k] == v for k, v in runner.SCOPE.items())
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
    monkeypatch.setattr(runner, 'load_baseline17', lambda: pytest.fail('plan loaded runtime'))
    assert runner.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PLAN_ONLY' and report['physics_steps_executed'] == 0
    assert report['budget'] == runner.BUDGET and report['limits']['max_cpp_inferences'] == 21664
    assert report['limits']['start_effective_bytes'] == 6 << 30
    assert report['limits']['reserve_bytes'] == 4 << 30


def test_outer_low_ram_does_not_claim_attempt(tmp_path):
    result = runner.execute(root=tmp_path, bundle_loader=lambda: {'identity': {}}, source_loader=dict,
                            memory_reader=lambda: {'effective_available_bytes': 5 << 30})
    assert result['status'] == 'NOT_READY' and not (tmp_path/runner.OUTPUT).exists()


def test_outer_reserve_rejects_3_point_5_gib_after_admission(tmp_path):
    samples = iter((6 << 30, 7 << 29))
    cleanups = []
    report = runner.execute(root=tmp_path, bundle_loader=lambda: {'identity': {}}, source_loader=dict,
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
    protocol = {'format': 'd1-cpp-diagnostic23-protocol-v1', 'release_sha256': runner.sha(path),
                'source_identity': release['source_identity'], 'baseline17_provenance': bundle['identity'],
                'budget': runner.BUDGET, 'limits': runner.LIMITS, 'action_driver': 'cpp_ort1.10_only',
                'scenarios': runner.matrix(), **runner.SCOPE}
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
            'inference_file': stem+'_inference.npz', 'inference_sha256': runner.sha(output/(stem+'_inference.npz')),
            'failure': True, 'done_reason': 'TERMINATED'})
    counts = {'policy_transitions': 36, 'native_steps': 360, 'pd_updates': 180}
    child = {'format': 'd1-cpp-diagnostic23-worker-v1', 'release_sha256': runner.sha(path),
        'source_identity': release['source_identity'], 'baseline17_provenance': bundle['identity'],
        'status': 'COMPLETE', 'cleanup_complete': True, 'inputs_unchanged': True, 'error': None,
        'protocol_sha256': runner.sha(output/'protocol.json'), 'cases': rows, 'actual_counts': counts,
        'known_actual_counts': counts, 'maximum_same_input_error': 0., 'all_36_reference_pass': False,
        'golden': [{'index': i, 'actions': [0.]*16, 'python_ort_actions': [0.]*16, 'max_absolute_error': 0.} for i in range(64)],
        'by_delay_ms': runner.summaries(rows), **runner.SCOPE, 'oracle_calls': 100, 'cpp_actor': {'status': 'CLOSED', 'cleanup_complete': True,
            'requests_completed': 100, 'inference_attempts': 100, 'inference_returns': 100}}
    child['cpp_actor'] = complete_actor_receipt(bundle, 100)
    (output/'cpp_actor').mkdir()
    runner.atomic_json(output/'cpp_actor/report.json', child['cpp_actor'])
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
        protocol['baseline17_provenance'] = {'another_model': True}
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
        protocol = {'format': 'd1-cpp-diagnostic23-protocol-v1', 'release_sha256': runner.sha(output/'release.json'),
                    'source_identity': sources, 'baseline17_provenance': bundle['identity'],
                    'budget': runner.BUDGET, 'limits': runner.LIMITS, **runner.SCOPE}
        if fault == 'foreign_partial':
            protocol['baseline17_provenance'] = {'another_model': True}
        runner.atomic_json(child_dir/'protocol.json', protocol)
        counts = {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2} if fault in ('partial', 'foreign_partial') else dict(runner.BUDGET)
        child = {'format': 'd1-cpp-diagnostic23-worker-v1', 'release_sha256': runner.sha(output/'release.json'),
                 'source_identity': sources, 'baseline17_provenance': bundle['identity'], 'status': 'COMPLETE',
                 'error': None, 'cleanup_complete': True, 'inputs_unchanged': True, 'actual_counts': counts,
                 'known_actual_counts': counts, 'protocol_sha256': runner.sha(child_dir/'protocol.json'),
                 'cases': [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed} for name, delay, seed in runner.matrix()],
                 'golden': [{'index': i} for i in range(64)], 'all_36_reference_pass': True, 'by_delay_ms': {}, **runner.SCOPE}
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



@pytest.mark.parametrize('field,value', [('diagnostic_only', False), ('qualification_eligible', True),
    ('qualification_eligible', 0), ('deployment_qualified', True), ('old19_eligibility_granted', True),
    ('training_allowed', True), ('retry_or_resume_allowed', True), ('format', 'd1-cpp-closedloop19-release-v1')])
def test_new_release_never_grants_old_or_deployment_qualification(worker_case, field, value):
    _, _, release, _ = worker_case
    release[field] = value
    with pytest.raises(ValueError, match='scope/budget'):
        runner.validate_release(release)


def test_default_plan_does_not_load_any_old19_gate(monkeypatch, capsys):
    monkeypatch.setattr(runner.engine, 'load_eligible', lambda *a, **k: pytest.fail('old19 eligibility invoked'))
    monkeypatch.setattr(runner, 'source_identity', lambda: pytest.fail('plan claimed a source freeze'))
    assert runner.main(['--plan-only']) == 0
    value = json.loads(capsys.readouterr().out)
    assert runner.valid_scope(value) and value['onnx_sessions_created'] == value['robot_models_constructed'] == 0


def test_uncommitted_source_is_rejected_before_bundle_or_attempt(tmp_path, monkeypatch):
    calls = []
    def source():
        calls.append('source')
        raise ValueError('uncommitted new23 source')
    with pytest.raises(ValueError, match='uncommitted'):
        runner.execute(root=tmp_path, source_loader=source, bundle_loader=lambda: pytest.fail('bundle before source'))
    assert calls == ['source'] and not (tmp_path/runner.OUTPUT).exists()


def test_source_commit_gate_checks_actual_git_blob(tmp_path, monkeypatch):
    import subprocess
    names = ('new.py',)
    (tmp_path/'new.py').write_text('approved source')
    proof = {'git_commit': 'a'*40, 'sources_sha256': {'new.py': runner.sha(tmp_path/'new.py')}}
    monkeypatch.setattr(runner, 'source_names', lambda root: names)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: b'approved source')
    runner.check_sources(proof, root=tmp_path)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k: b'other committed bytes')
    with pytest.raises(ValueError, match='commit new23 sources'):
        runner.check_sources(proof, root=tmp_path)
    (tmp_path/'new.py').write_text('dirty source')
    with pytest.raises(ValueError, match='source changed'):
        runner.check_sources(proof, root=tmp_path)


def test_worker_nonce_requires_this_live_parent_and_fixed_paths(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    output = tmp_path/runner.OUTPUT
    output.mkdir(parents=True)
    release = output/'release.json'
    release.write_text('{}')
    journal = {'format': 'd1-cpp-diagnostic23-execution-v1', 'status': 'RUNNING',
               'release_sha256': runner.sha(release), 'supervisor_pid': os.getppid(), 'nonce': 'unique'}
    runner.atomic_json(output/'report.json', journal)
    monkeypatch.setenv('D1_CPP23_NONCE', 'unique')
    runner.validate_launch(release, output/'worker')
    monkeypatch.setenv('D1_CPP23_NONCE', 'foreign')
    with pytest.raises(ValueError, match='active bounded'):
        runner.validate_launch(release, output/'worker')
    monkeypatch.setenv('D1_CPP23_NONCE', 'unique')
    journal['supervisor_pid'] = 0
    runner.atomic_json(output/'report.json', journal)
    with pytest.raises(ValueError, match='active bounded'):
        runner.validate_launch(release, output/'worker')


@pytest.fixture
def baseline_files(tmp_path, monkeypatch):
    """CI-clean synthetic old17 receipts; no local models/assets/dependencies."""
    from tools17 import verify_cpp17
    checkpoint = tmp_path/'completed16/policy.zip'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'synthetic completed policy, never loaded')
    for name in ('protocol.json', 'result.json'):
        runner.atomic_json(checkpoint.parent/name, {})
    checkpoint_sha = runner.sha(checkpoint)
    monkeypatch.setattr(runner, 'CHECKPOINT', checkpoint_sha)
    monkeypatch.setattr(runner.provenance, 'check_sources', lambda *a, **k: None)
    monkeypatch.setattr(verify_cpp17, 'check_fixtures', lambda obj: {})
    execution = tmp_path/runner.BASELINE/'execution_01'
    proof = {'git_commit': 'c'*40, 'sources_sha256': {}}
    old = {'status': 'AUTHORIZED_FROZEN', 'checkpoint_sha256': checkpoint_sha,
           'source_identity': proof, 'output_root': str(execution.relative_to(tmp_path))}
    (tmp_path/runner.BASELINE).mkdir(parents=True)
    runner.atomic_json(tmp_path/runner.BASELINE/'release_01.json', old)
    for name in ('export', 'cpp', 'formal'):
        (execution/name).mkdir(parents=True)
    model = execution/'export/policy.onnx'
    model.write_bytes(b'not an ONNX model; pure fixture')
    runner.atomic_json(execution/'export/golden_fixtures.json', {})
    fixtures_sha = runner.sha(execution/'export/golden_fixtures.json')
    runner.atomic_json(execution/'export/report.json', {'checkpoint_sha256': checkpoint_sha, 'fixtures_sha256': fixtures_sha})
    runner.atomic_json(execution/'cpp/report.json', {})
    rows = [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed, 'done_reason': 'TERMINATED',
             'failure': True, 'reference_thresholds_met': False,
             'actual_counts': {'policy_transitions': 1, 'native_steps': 10, 'pd_updates': 5}}
            for name, delay, seed in runner.matrix()]
    formal = {'status': 'COMPLETE', 'cleanup_complete': True, 'inputs_unchanged': True, 'error': None,
              'all_36_reference_pass': False, 'checkpoint_sha256': checkpoint_sha, 'onnx_sha256': runner.sha(model),
              'cases': rows, 'by_delay_ms': runner.summaries(rows),
              'actual_counts': {'policy_transitions': 36, 'native_steps': 360, 'pd_updates': 180}}
    runner.atomic_json(execution/'formal/report.json', formal)
    stages = {}
    for name in ('export', 'cpp', 'formal'):
        receipt = execution/name/'report.json'
        stages[name] = {'status': 'COMPLETE', 'cleanup_complete': True, 'returncode': 0,
                       'receipt_sha256': runner.sha(receipt), 'artifacts_sha256': {str(receipt): runner.sha(receipt)}}
    runner.atomic_json(execution/'report.json', {'status': 'COMPLETE', 'error': None, 'stages': stages})
    summary = {'engineering_validation_status': 'COMPLETE', 'full_delay_reference_qualification': 'NOT_PASSED',
               'checkpoint_sha256': checkpoint_sha, 'source_identity': proof,
               'onnx': {'sha256': runner.sha(model), 'fixture_sha256': fixtures_sha}}
    (tmp_path/'reports').mkdir()
    runner.atomic_json(tmp_path/'reports/closeout17_20261018.json', summary)
    build_dir = tmp_path/'runtime19/build'
    build_dir.mkdir(parents=True)
    library = tmp_path/'ort_library.so'
    monkeypatch.setattr(runner.actor_api, 'ORT_LIBRARY', library)
    files = [tmp_path/'runtime19/actor_server19.cpp', tmp_path/'tools19/cpp_actor19.py', library]
    (tmp_path/'tools19').mkdir()
    for file in files:
        file.write_bytes(b'only source fixture')
    binary = build_dir/'actor_server19'
    binary.write_bytes(b'not executable')
    runner.atomic_json(build_dir/'build.json', {'status': 'PASS', 'server_started': False,
        'onnx_sessions_created': 0, 'robot_models_constructed': 0, 'physics_steps_executed': 0,
        'binary_sha256': runner.sha(binary), 'inputs_sha256': {str(f): runner.sha(f) for f in files}})
    evaluator = SimpleNamespace(check_numeric_inputs=lambda *a: (model, {'onnx_sha256': runner.sha(model)}),
                                verify_trace=lambda *a: None)
    training_release = {'sources_sha256': {'original_evaluator': 'a'*64}, 'config': 'training/configs/pilot03d.json'}
    def helper_loader(release):
        assert release is training_release, 'closeout17 release passed instead of original training release'
        return object()
    def load():
        return runner.load_baseline17(tmp_path, loader=lambda *a, **k: (training_release, {}, {'status': 'COMPLETE'}, checkpoint),
                                     evaluator=evaluator, helper_loader=helper_loader)
    return load, execution, tmp_path/'reports/closeout17_20261018.json', checkpoint


def test_independent_loader_accepts_completed_baseline_without_any_19_winner(baseline_files, monkeypatch):
    load, _, _, checkpoint = baseline_files
    monkeypatch.setattr(runner.engine, 'load_eligible', lambda *a, **k: pytest.fail('old19 gate invoked'))
    bundle = load()
    assert bundle['identity']['checkpoint_sha256'] == runner.sha(checkpoint)
    assert bundle['identity']['baseline_all_36_reference_pass'] is False
    assert bundle['old_release']['config'] == 'training/configs/pilot03d.json'
    assert 'sources_sha256' in bundle['old_release']


@pytest.mark.parametrize('fault', ['summary_qualification', 'partial_checkpoint', 'stage_cleanup', 'artifact', 'aggregate', 'build_source'])
def test_independent_loader_rejects_foreign_or_partial_provenance(baseline_files, fault):
    load, execution, summary, checkpoint = baseline_files
    if fault == 'summary_qualification':
        value = runner.read_json(summary)
        value['full_delay_reference_qualification'] = 'PASSED'
        runner.atomic_json(summary, value)
    elif fault == 'partial_checkpoint':
        checkpoint.write_bytes(b'partial')
    elif fault == 'stage_cleanup':
        path = execution/'report.json'
        value = runner.read_json(path)
        value['stages']['formal']['cleanup_complete'] = False
        runner.atomic_json(path, value)
    elif fault == 'build_source':
        (execution.parents[2]/'runtime19/actor_server19.cpp').write_bytes(b'changed')
    else:
        path = execution/'formal/report.json'
        value = runner.read_json(path)
        value['actual_counts']['policy_transitions'] = 37
        runner.atomic_json(path, value)
        if fault == 'aggregate':
            outer_path = execution/'report.json'
            outer = runner.read_json(outer_path)
            stage = outer['stages']['formal']
            stage['receipt_sha256'] = runner.sha(path)
            stage['artifacts_sha256'][str(path)] = runner.sha(path)
            runner.atomic_json(outer_path, outer)
    with pytest.raises(ValueError):
        load()



def complete_actor_receipt(bundle, count):
    identities = dict(bundle['identity']['actor_identities'])
    ready = {'format': 'd1-cpp-actor19-ready-v1', 'status': 'READY',
             **{k: identities[k] for k in ('binary_sha256', 'model_sha256', 'source_sha256', 'ort_library_sha256')},
             'signature': runner.actor_api.SIGNATURE, 'onnxruntime_version': '1.10.0', 'cpu_threads': 1,
             'max_requests': 21664, 'inference_attempts': 0, 'inference_returns': 0,
             'robot_models_constructed': 0, 'physics_steps_executed': 0}
    ended = {'format': 'd1-cpp-actor19-closed-v1', 'status': 'CLOSED', 'requests': count,
             'inference_attempts': count, 'inference_returns': count, 'robot_models_constructed': 0, 'physics_steps_executed': 0}
    return {'format': 'd1-cpp-actor19-adapter-v1', 'status': 'CLOSED', 'identities': identities,
            'process_started': True, 'error': None, 'returncode': 0, 'requests_attempted': count,
            'requests_completed': count, 'inference_attempts': count, 'inference_returns': count,
            'cleanup_complete': True, 'ready': ready, 'eof_receipt': ended}


@pytest.mark.parametrize('field', ['model_sha256', 'binary_sha256', 'source_sha256', 'adapter_sha256', 'ort_library_sha256'])
def test_self_consistent_saved_native_identity_still_must_match_frozen_baseline(saved_worker, field):
    output, child, release, bundle = saved_worker
    child['cpp_actor']['identities'][field] = '0'*64
    if field in child['cpp_actor']['ready']:
        child['cpp_actor']['ready'][field] = '0'*64
    runner.atomic_json(output/'cpp_actor/report.json', child['cpp_actor'])
    with pytest.raises(ValueError, match='identity differs'):
        runner.verify_completed_worker(output, child, release, bundle, helpers=object(), verifier=lambda *a: None)


@pytest.mark.parametrize('section,key,value', [('ready', 'cpu_threads', True), ('ready', 'max_requests', 21600),
    ('ready', 'physics_steps_executed', False), ('ready', 'onnxruntime_version', '1.23.2'),
    ('ready', 'signature', {'inputs': {'obs': [True, 57], 'history': [1, 10, 57]}, 'outputs': {'actions': [1, 16]}}),
    ('eof_receipt', 'requests', True), ('eof_receipt', 'physics_steps_executed', 1),
    (None, 'returncode', False), (None, 'requests_attempted', True)])
def test_saved_ready_eof_typed_fields_rechecked_independently(saved_worker, section, key, value):
    output, child, release, bundle = saved_worker
    target = child['cpp_actor'][section] if section else child['cpp_actor']
    target[key] = value
    runner.atomic_json(output/'cpp_actor/report.json', child['cpp_actor'])
    with pytest.raises(ValueError):
        runner.verify_completed_worker(output, child, release, bundle, helpers=object(), verifier=lambda *a: None)
