"""Exercise supervised lifecycle and new model boundaries with synthetic actors."""
from __future__ import annotations

import copy
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools22 import run_low_lr22 as r  # noqa: E402


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.c, 'ROOT', tmp_path)
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = {'output_root': 'attempt', 'source_identity': {'new': True},
            'budget': r.c.expected_counts(r.c.TRANSITIONS+43200)}
    state = SimpleNamespace(now=0., stages=[], cleaned=[], fault=None, qualified=False, memory=8 << 30)

    def factory(argv, **kwargs):
        stage = 'train' if 'train_low_lr22.py' in argv[2] else argv[argv.index('--stage')+1]
        state.stages.append(stage)
        output = Path(argv[argv.index('--output')+1])
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['D1_STAGE22_NONCE']
        output.mkdir()
        value = {'stage': stage, 'plan_sha256': r.c.sha(path), 'source_identity': plan['source_identity'],
                 'status': 'COMPLETE', 'actual_counts': r.c.expected_counts(r.c.STAGES[stage]['transitions']),
                 'all_36_reference_pass': state.qualified}
        if state.fault == 'partial':
            value['actual_counts'] = None
            value['known_actual_counts'] = r.c.expected_counts(4096)
        r.c.atomic_json(output/'protocol.json', {'stage': stage, 'plan_sha256': r.c.sha(path),
                                                'source_identity': plan['source_identity']})
        value['protocol_sha256'] = r.c.sha(output/'protocol.json')
        r.c.atomic_json(output/('result.json' if stage == 'train' else 'report.json'), value)
        process = SimpleNamespace(returncode=1 if state.fault == 'partial' else 0)
        process.poll = lambda: process.returncode
        if state.fault == 'signal':
            signal.raise_signal(signal.SIGTERM)
        if state.fault == 'resource':
            state.memory = (4 << 30)-1
        return process

    def cleanup(process):
        state.cleaned.append(process)
        return {'remaining': {'owned': 'alive'} if state.fault == 'cleanup' else {}}

    def verify(plan, path, stage, directory, child):
        if state.fault == 'verifier':
            raise ValueError('independent reader rejected this record')
        return r.projected_counts(child['actual_counts'], r.c.expected_counts(r.c.STAGES[stage]['transitions']))

    def sleep(seconds):
        state.now += seconds
    state.kwargs = dict(plan_path=path, loader=lambda p: copy.deepcopy(plan),
        memory_reader=lambda: {'effective_available_bytes': state.memory}, factory=factory,
        cleanup=cleanup, adopter=lambda: None, verifier=verify, clock=lambda: state.now, sleeper=sleep)
    return state


@pytest.mark.parametrize('qualified', [False, True])
def test_parent_serializes_train_then_export_cpp_and_full_reference(execution, qualified):
    execution.qualified = qualified
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE' and value['error'] is None
    assert execution.stages == ['train', 'export', 'cpp', 'formal']+(['holdout'] if qualified else [])
    assert value['actual_counts'] == r.c.expected_counts(r.c.TRANSITIONS+(43200 if qualified else 21600))
    assert value['full_delay_qualification'] is qualified
    assert value['automatic_model_replacement'] is False and value['active_stage'] is None
    assert len([p for p in execution.cleaned if p is not None]) == len(execution.stages)


@pytest.mark.parametrize('fault', ['partial', 'signal', 'resource', 'cleanup', 'verifier'])
def test_interruption_failure_cannot_continue_or_grant_qualification(execution, fault):
    execution.fault = fault
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and value['full_delay_qualification'] is False
    assert execution.stages == ['train']
    assert value['cleanup_complete'] is (fault != 'cleanup')
    if fault == 'partial':
        assert value['actual_counts'] is None
        assert value['known_actual_counts'] == r.c.expected_counts(4096)


def test_low_start_memory_consumes_no_attempt(execution):
    execution.memory = (6 << 30)-1
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'NOT_READY' and not execution.stages


def test_attempt_directory_is_never_reused(execution):
    r.execute(**execution.kwargs)
    with pytest.raises(ValueError, match='no retry'):
        r.execute(**execution.kwargs)


def test_totals_rebuilt_from_facts_do_not_double_charge():
    stages = {'train': {'actual_counts': r.c.expected_counts(4096)},
              'formal': {'actual_counts': None, 'known_actual_counts': r.c.expected_counts(17)}}
    assert r.totals(stages) == r.totals(stages) == r.c.expected_counts(4113)


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_noncore_worker_counters_are_validated_before_projection(bad):
    value = dict(r.c.expected_counts(600), resets=1, forward_calls=601)
    assert r.projected_counts(value, r.CAP) == r.c.expected_counts(600)
    value['resets'] = bad
    with pytest.raises(ValueError, match='counter'):
        r.projected_counts(value, r.CAP)


@pytest.mark.parametrize('field', ['num_timesteps', 'learning_rate', 'n_envs', 'aux_coef'])
def test_export_cannot_accept_old_final_or_wrong_lr(field):
    model = SimpleNamespace(**{key: r.c.HYPERPARAMETERS[key] for key in
        ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
         'max_grad_norm', 'learning_rate')}, num_timesteps=r.c.INITIAL+r.c.TRANSITIONS,
        aux_coef=0., seed=181, n_envs=4, device='cpu', get_env=lambda: None,
        clip_range=lambda _: .2, normalize_advantage=True, clip_range_vf=None, target_kl=None, use_sde=False)
    r.check_model(model, {'versions': {'synthetic': True}}, {'synthetic': True})
    setattr(model, field, {'num_timesteps':2097152, 'learning_rate':3e-4, 'n_envs':1, 'aux_coef':1.}[field])
    with pytest.raises(ValueError):
        r.check_model(model, {'versions': {'synthetic': True}}, {'synthetic': True})


@pytest.mark.parametrize('bad', [None, 'policy_sha256', 'fixture_sha256', 'runtime_binary_sha256', 'schema_sha256'])
def test_cpp_identity_must_bridge_current_files_to_frozen_runtime(tmp_path, monkeypatch, bad):
    from tools17 import export_completed17 as export
    from tools17 import verify_cpp17 as cpp
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    export_dir, directory = tmp_path/'export', tmp_path/'cpp'
    export_dir.mkdir()
    (directory/'cpp_native').mkdir(parents=True)
    (export_dir/'policy.onnx').write_bytes(b'synthetic-current-model')
    r.c.atomic_json(export_dir/'golden_fixtures.json', {'fixtures': [], 'seed': 1})
    native = directory/'cpp_native/report.json'
    r.c.atomic_json(native, {'status':'synthetic'})
    identities = {'policy_sha256':r.c.sha(export_dir/'policy.onnx'),
                  'fixture_sha256':r.c.sha(export_dir/'golden_fixtures.json')}
    release = {'sources_sha256':{}}
    for name, key in (('runtime03/build/d1_runtime03','runtime_binary_sha256'),
                      ('runtime03/schema03d.json','schema_sha256')):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        identities[key] = release['sources_sha256'][name] = r.c.sha(path)
    exported = {'onnx_sha256':identities['policy_sha256'], 'fixtures_sha256':identities['fixture_sha256'],
                'onnx_node_counts':{},'onnx_signature':{},'parity':{}}
    r.c.atomic_json(export_dir/'report.json', exported)
    child = {**exported,'identities':identities, 'comparisons':64,'max_absolute_error':0.,
             'absolute_tolerance':1e-5,'cxx_ort_1_10_status':'PASS',
             'export_report_sha256':r.c.sha(export_dir/'report.json'), 'cpp_native_report_sha256':r.c.sha(native)}
    monkeypatch.setattr(export, 'cpp_fixtures', lambda _: {})
    monkeypatch.setattr(cpp, 'check_fixtures', lambda _: {})
    monkeypatch.setattr(cpp, 'check_native', lambda *args: 0.)
    if bad is not None:
        identities[bad] = '0'*64
        with pytest.raises(ValueError, match='CPP22'):
            r.verify_numeric('cpp', child, directory, old_release=release)
    else:
        r.verify_numeric('cpp', child, directory, old_release=release)
