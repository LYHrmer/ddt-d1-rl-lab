"""Environment-only exception guards, exercised entirely with fake evidence/processes."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools13'))
import contract10_environment_amendment as adapter  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fake_original(root, monkeypatch):
    original = json.loads((ROOT/'training/configs/contract10_supervisor_draft.json').read_text())
    names = set(original['frozen_inputs']) | set(adapter.runtime().release.REQUIRED_INPUTS)
    for name in names:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        first = Path(name).parts[0]
        source = ROOT/name
        is_source = first.startswith('tools') or first in {'training', 'tests', 'docs', 'src'}
        target.write_bytes(source.read_bytes() if is_source and source.is_file() else b'fake frozen binary or data\n')
    original['frozen_inputs'] = {name: adapter.sha(root/name) for name in names}
    original.update(status='FROZEN_BEFORE_TRAINING', created_utc='2026-10-16T00:00:00+00:00',
                    proposed_budget=dict(adapter.CAP, qualification_native=18000, authorised=True))
    auth = write_json(root/'budget_fake.json', {'format': 'd1-contract10-budget-authorisation-v1', 'contract_id': 10,
        'authorised': True, 'cap': adapter.CAP, 'qualification_native': 18000,
        'start_min_available_ram_bytes': 7*1024**3, 'retries_allowed': False,
        'resume_allowed': False, 'checkpoint_selection': 'final_only'})
    static = write_json(root/'static_fake.json', {'status': 'PASS'})
    original['budget_authorisation'] = {'path': str(auth.relative_to(root)), 'sha256': adapter.sha(auth)}
    original['static_report'] = {'path': str(static.relative_to(root)), 'sha256': adapter.sha(static)}
    path = write_json(root/adapter.ORIGINAL_RELEASE, original)
    monkeypatch.setattr(adapter, 'ORIGINAL_SHA256', adapter.sha(path))
    output = root/adapter.BASE/'scale10_preflight_01'
    (output/'source_snapshot').mkdir(parents=True)
    result = {'status': 'ERROR', 'trained_transitions': 0, 'actual_counts': None, 'per_worker_counts': None,
              'inputs_unchanged': True, 'error': 'PermissionError: [Errno 1] Operation not permitted'}
    write_json(output/'result.json', result)
    protocol_path = root/adapter.BASE/'scale10_preflight_01/protocol.json'
    protocol = {'kind': 'preflight', 'seed': 101, 'envs': 8, 'planned_transitions': 32768,
                'hyperparameters': {'ent_coef': 0.},
                'inputs_sha256': {str(root/name): original['frozen_inputs'][name]
                                  for name in (original['entry'], original['config'])}}
    write_json(protocol_path, protocol)
    forkserver = Path(adapter.sysconfig.get_path('stdlib'))/'multiprocessing/forkserver.py'
    method = next(n for n in adapter.ast.walk(adapter.ast.parse(forkserver.read_text()))
                  if isinstance(n, adapter.ast.FunctionDef) and n.name == 'ensure_running')
    bind = next(n for n in adapter.ast.walk(method) if isinstance(n, adapter.ast.Call)
                and isinstance(n.func, adapter.ast.Attribute) and n.func.attr == 'bind')
    source = root/'.python-deps/stable_baselines3/common/vec_env/subproc_vec_env.py'
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('process.start()\n')
    error_path = root/adapter.BASE/'scale10_preflight_01/error.txt'
    error_path.write_text(f'Traceback (most recent call last):\n  File "{source}", line 122, in __init__\n'
                         '    process.start()\n  File "runtime", line 84, in connect_to_new_process\n'
                         f'  File "{forkserver}", line {bind.lineno}, in ensure_running\n'
                         '    listener.bind(address)\nPermissionError: [Errno 1] Operation not permitted\n')
    report_path = root/adapter.BASE/'scale10_preflight_01_supervisor/report.json'
    report = {'format': 'd1-contract10-supervisor-report-v1', 'kind': 'preflight', 'status': 'FAIL',
              'release_sha256': adapter.sha(path), 'supervisor_sha256': original['supervisor_sha256'],
              'plan': original['plans']['preflight'], 'trainer_result': result, 'trainer_returncode': 1,
              'stop_reason': 'TRAINER_EXIT_NONZERO', 'cleanup_complete': True, 'inputs_unchanged': True,
              'rollouts_compared': 0, 'equivalence': 'INCOMPLETE'}
    write_json(report_path, report)
    for name in adapter.NEW_INPUTS:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes() if (ROOT/name).is_file() else b'fake amendment contract\n')
    return path


def test_zero_steps_requires_the_preworker_failure_path_not_null_counters_alone(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    proof = adapter.audit_zero_failure(path, root=tmp_path)
    assert proof['derived_actual_counts'] == {'policy_transitions': 0, 'native_steps': 0, 'pd_updates': 0}
    assert proof['original_counters_were_null'] is True
    assert proof['basis'] == 'first forkserver listener bind failed before any environment worker could start'


@pytest.mark.parametrize('change', ['counter', 'traceback', 'worker_artifact', 'report', 'source'])
def test_zero_claim_is_rejected_if_any_evidence_disagrees(tmp_path, monkeypatch, change):
    path = fake_original(tmp_path, monkeypatch)
    output = tmp_path/adapter.BASE/'scale10_preflight_01'
    if change == 'counter':
        result = json.loads((output/'result.json').read_text()); result['trained_transitions'] = 1
        write_json(output/'result.json', result)
    elif change == 'traceback': (output/'error.txt').write_text('PermissionError, but location unknown')
    elif change == 'worker_artifact': (output/'env_00.counts.json').write_text('{}')
    elif change == 'report':
        target = tmp_path/adapter.BASE/'scale10_preflight_01_supervisor/report.json'
        report = json.loads(target.read_text()); report['cleanup_complete'] = False; write_json(target, report)
    else: (tmp_path/'tools09/train_d1_ppo09.py').write_text('changed')
    with pytest.raises(SystemExit):
        adapter.audit_zero_failure(path, root=tmp_path)


def test_default_plan_never_creates_authorisation_freeze_probe_or_training(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, 'ipc_probe', lambda: pytest.fail('plan must not start IPC'))
    plan = adapter.plan(path, root=tmp_path)
    assert plan['mode'] == 'PLAN_ONLY' and plan['authorised'] is False
    assert not (tmp_path/adapter.OWNER_RECORD).exists()
    assert not (tmp_path/adapter.AMENDED_RELEASE).exists()


def test_execute_without_owner_record_cannot_probe_or_spawn(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match='owner'):
        adapter.execute(path, root=tmp_path, probe=lambda: pytest.fail('no IPC before authorisation'))


def test_failed_ipc_admission_does_not_freeze_or_claim_the_new_attempt(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    result = adapter.execute(path, root=tmp_path, authorise=True,
                             probe=lambda: {'status': 'FAIL', 'models_loaded': 0, 'physics_steps_executed': 0})
    assert result == 2
    assert (tmp_path/adapter.OWNER_RECORD).is_file()
    assert not (tmp_path/adapter.AMENDED_RELEASE).exists()
    assert not (tmp_path/adapter.BASE/'execution_02').exists()


def test_context_routes_new_paths_and_restores_every_global_on_exception(tmp_path):
    runtime = adapter.runtime()
    before = (runtime.release.RELEASE_PATH, copy.deepcopy(runtime.release.PLANS),
              copy.deepcopy(runtime.runner.OUTPUTS), runtime.runner.Journal, runtime.runner.commands)
    with pytest.raises(RuntimeError):
        with adapter.adapted_context(tmp_path):
            assert runtime.release.RELEASE_PATH == str(adapter.AMENDED_RELEASE)
            assert runtime.release.PLANS['scale']['output'].endswith('_02')
            args = runtime.runner.commands({'config': 'config.json', 'plans': runtime.release.PLANS},
                                           tmp_path/adapter.AMENDED_RELEASE, tmp_path)
            assert args['verdict'][2].endswith('tools13/contract10_environment_amendment.py')
            assert args['verdict'][3] == 'judge'
            journal = runtime.runner.Journal(tmp_path/adapter.BASE/'execution_01', 'fake-release')
            assert journal.directory == tmp_path/adapter.BASE/'execution_02'
            raise RuntimeError('test restoration')
    after = (runtime.release.RELEASE_PATH, runtime.release.PLANS, runtime.runner.OUTPUTS,
             runtime.runner.Journal, runtime.runner.commands)
    assert before == after


def test_new_release_binds_old_inputs_new_code_contract_owner_and_original_failure(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    proof = adapter.audit_zero_failure(path, root=tmp_path)
    owner = adapter.record_owner_approval(proof, root=tmp_path)
    amended = adapter.build_release(path, owner, proof, root=tmp_path)
    original = json.loads(path.read_text())
    assert original['frozen_inputs'].items() <= amended['frozen_inputs'].items()
    assert set(adapter.NEW_INPUTS) <= amended['frozen_inputs'].keys()
    assert str(adapter.OWNER_RECORD) in amended['frozen_inputs']
    assert amended['proposed_budget'] == original['proposed_budget']
    assert amended['plans']['preflight']['output'].endswith('_02')
    assert amended['plans']['scale']['output'].endswith('_02')


def test_only_one_new_attempt_and_no_approval_record_overwrite(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    proof = adapter.audit_zero_failure(path, root=tmp_path)
    owner = adapter.record_owner_approval(proof, root=tmp_path)
    before = owner.read_bytes()
    assert adapter.record_owner_approval(proof, root=tmp_path) == owner
    assert owner.read_bytes() == before
    (tmp_path/adapter.BASE/'scale10_preflight_02_supervisor').mkdir()
    with pytest.raises(SystemExit, match='attempt'):
        adapter.build_release(path, owner, proof, root=tmp_path)


def frozen_amendment(root, monkeypatch):
    path = fake_original(root, monkeypatch)
    proof = adapter.audit_zero_failure(path, root=root)
    owner = adapter.record_owner_approval(proof, root=root)
    amended = adapter.build_release(path, owner, proof, root=root)
    target = write_json(root/adapter.AMENDED_RELEASE, amended)
    return target, amended


def test_actual_frozen_legacy_loader_accepts_only_the_contextual_new_release(tmp_path, monkeypatch):
    _, amended = frozen_amendment(tmp_path, monkeypatch)
    assert adapter.load_amended(tmp_path) == amended
    with pytest.raises(SystemExit, match='release path'):
        adapter.runtime().judge.load_release(tmp_path/adapter.AMENDED_RELEASE, tmp_path)


@pytest.mark.parametrize('change', ['entry', 'config', 'contract', 'resources', 'budget', 'steps',
                                   'output', 'extra_input', 'extra_field', 'journal', 'retry', 'static_scope'])
def test_amended_release_is_an_exact_environment_only_field_closure(tmp_path, monkeypatch, change):
    target, value = frozen_amendment(tmp_path, monkeypatch)
    if change == 'entry': value['entry'] = value['config']
    elif change == 'config': value['config'] = value['contract']
    elif change == 'contract': value['contract'] = adapter.NEW_INPUTS[-1]
    elif change == 'resources': value['resources']['start_min_available_ram_bytes'] -= 1
    elif change == 'budget': value['proposed_budget']['native'] += 1
    elif change == 'steps': value['plans']['scale']['steps'] -= 4096
    elif change == 'output': value['plans']['scale']['output'] += '_retry'
    elif change == 'extra_input':
        (tmp_path/'unreviewed').write_text('extra')
        value['frozen_inputs']['unreviewed'] = adapter.sha(tmp_path/'unreviewed')
    elif change == 'extra_field': value['new_scientific_choice'] = True
    elif change == 'journal': value['environment_amendment']['journal'] += '_retry'
    elif change == 'retry': value['environment_amendment']['retries_allowed'] = True
    elif change == 'static_scope': value['environment_amendment']['original_static_scope'] = 'new paths approved by old static'
    write_json(target, value)
    with pytest.raises(SystemExit, match='amendment'):
        adapter.load_amended(tmp_path)


def test_successful_fake_admission_freezes_only_02_and_enters_legacy_runner_in_context(tmp_path, monkeypatch):
    path = fake_original(tmp_path, monkeypatch)
    old_bytes = path.read_bytes()
    modules = adapter.runtime()
    calls = []

    def run(release_path, root):
        assert release_path == root/adapter.AMENDED_RELEASE
        assert modules.release.RELEASE_PATH == str(adapter.AMENDED_RELEASE)
        assert modules.release.PLANS['preflight']['output'].endswith('_02')
        calls.append(True)
        return 0

    monkeypatch.setattr(modules.runner, 'execute', run)
    probe = lambda: {'status': 'PASS', 'models_loaded': 0, 'physics_steps_executed': 0,
                     'returncode': 0, 'cleanup_complete': True}
    assert adapter.execute(path, root=tmp_path, authorise=True, probe=probe) == 0
    assert calls == [True] and path.read_bytes() == old_bytes
    assert (tmp_path/adapter.AMENDED_RELEASE).is_file()
    assert modules.release.RELEASE_PATH == str(adapter.ORIGINAL_RELEASE)


@pytest.mark.parametrize('field,value', [('authorised', False), ('zero_proof_sha256', '0'*64),
                                       ('approved_cap', {'transitions': 1, 'native': 10}),
                                       ('new_attempts_allowed', 2), ('resume_training_allowed', True),
                                       ('scientific_changes', ['entropy'])])
def test_explicit_owner_record_cannot_expand_or_replace_the_exception(tmp_path, monkeypatch, field, value):
    path = fake_original(tmp_path, monkeypatch)
    proof = adapter.audit_zero_failure(path, root=tmp_path)
    owner = adapter.record_owner_approval(proof, root=tmp_path)
    document = json.loads(owner.read_text()); document[field] = value; write_json(owner, document)
    with pytest.raises(SystemExit, match='owner'):
        adapter.execute(path, root=tmp_path, probe=lambda: pytest.fail('must not start IPC'))


def test_stdlib_ipc_probe_owns_a_session_and_requires_clean_group_exit(monkeypatch):
    observed = {}

    class Process:
        pid = 987654321
        returncode = 0

        def __init__(self, argv, **kwargs):
            observed.update(argv=argv, kwargs=kwargs)

        def communicate(self, **_kwargs):
            return json.dumps({'status': 'PASS', 'models_loaded': 0, 'physics_steps_executed': 0}), ''

    monkeypatch.setattr(adapter.subprocess, 'Popen', Process)
    monkeypatch.setattr(adapter, 'probe_group_members', lambda _pgid: [])
    result = adapter.ipc_probe()
    assert result['status'] == 'PASS' and result['cleanup_complete'] is True
    assert observed['kwargs']['start_new_session'] is True
    assert observed['argv'][-1] == '_ipc_worker'


def test_ipc_timeout_kills_only_its_own_session_and_never_passes(monkeypatch):
    killed = []

    class Process:
        pid = 987654321
        returncode = -9
        calls = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def communicate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise adapter.subprocess.TimeoutExpired('fake-probe', kwargs['timeout'])
            return '', ''

    monkeypatch.setattr(adapter.subprocess, 'Popen', Process)
    monkeypatch.setattr(adapter, 'probe_group_members', lambda pgid: [pgid] if not killed else [])
    monkeypatch.setattr(adapter.os, 'killpg', lambda pgid, sig: killed.append((pgid, sig)))
    result = adapter.ipc_probe()
    assert result['status'] == 'FAIL' and result['timed_out'] is True
    assert killed == [(Process.pid, adapter.signal.SIGTERM)]


@pytest.mark.parametrize('during_spawn', [False, True])
def test_ipc_interrupt_and_spawn_race_always_clean_only_the_probe_group(tmp_path, monkeypatch, during_spawn):
    killed = []
    handlers = {sig: adapter.signal.getsignal(sig) for sig in (adapter.signal.SIGINT, adapter.signal.SIGTERM)}

    class Process:
        pid = 987654321
        returncode = -15

        def __init__(self, *_args, **_kwargs):
            if during_spawn:
                adapter.signal.getsignal(adapter.signal.SIGTERM)(adapter.signal.SIGTERM, None)

        def communicate(self, **_kwargs):
            if not killed:
                raise KeyboardInterrupt()
            return '', ''

    monkeypatch.setattr(adapter.subprocess, 'Popen', Process)
    monkeypatch.setattr(adapter, 'probe_group_members', lambda pgid: [pgid] if not killed else [])
    monkeypatch.setattr(adapter.os, 'killpg', lambda pgid, sig: killed.append((pgid, sig)))
    result = adapter.ipc_probe()
    assert result['status'] == 'FAIL' and result['cleanup_complete'] is True
    assert killed == [(Process.pid, adapter.signal.SIGTERM)]
    assert all(adapter.signal.getsignal(sig) == handler for sig, handler in handlers.items())
