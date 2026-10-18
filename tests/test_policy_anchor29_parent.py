"""Bounded fake processes and receipts only: no model, native call or physics."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools29 import run_anchor29 as r  # noqa: E402


def cases(*, passed=True, length=600):
    return [dict(sequence=name, delay_ticks=delay, reset_seed=seed, policy_transitions=length,
                 done_reason='HORIZON_12S' if length == 600 else 'TERMINATED',
                 failure=length != 600, reference_thresholds_met=passed, J=.2,
                 actual_counts=r.c.expected_counts(length))
            for name, delay, seed in r.prior.matrix('formal')]


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.c, 'ROOT', tmp_path)
    monkeypatch.setattr(r.os, 'getpriority', lambda *_: 10)
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = dict(output_root=r.c.OUTPUT, source_identity={'fixture': True}, budget=r.c.BUDGET,
                vector_layout=r.c.VECTOR_LAYOUT, runtime_dependencies_sha256={'fake.py': 'c'*64},
                base_checkpoint={'sha256': 'a'*64}, base_protocol={'sha256': 'b'*64})
    monkeypatch.setattr(r.c, 'stage_path', lambda plan, stage: tmp_path/plan['output_root']/
                        r.c.split_stage(stage)[0]/r.c.split_stage(stage)[1])
    state = SimpleNamespace(now=0., stages=[], cleaned=[], fault=None, fault_stage='train_control',
                            memory=6 << 30, samples=[], active=False, qualified=True,
                            source_drift=False, loader_calls=0, cleanup_delay=0., completed=False,
                            short_control=False, before_claim_delay=0., final_delay=0., output_race=False,
                            reference_calls=0, reference_change=False)

    def loader(_):
        state.loader_calls += 1
        if state.loader_calls == 2:
            state.now += state.before_claim_delay
            if state.output_race:
                folder = tmp_path/plan['output_root']
                folder.mkdir(parents=True)
                (folder/'report.json').write_text('keep original')
        if len(state.stages) == len(r.c.STAGE_ORDER):
            state.now += state.final_delay
        if state.source_drift:
            raise ValueError('source drift')
        return copy.deepcopy(plan)

    def memory():
        value = state.samples.pop(0) if state.samples else state.memory
        return {'effective_available_bytes': value}

    def sleeper(duration):
        assert not state.active or duration <= 1
        state.now += duration

    def factory(argv, **kwargs):
        assert not state.active
        arm = argv[argv.index('--arm')+1]
        kind = 'train' if argv[2].endswith('train_anchor29.py') else argv[argv.index('--stage')+1]
        stage = kind+'_'+arm
        state.stages.append(stage)
        assert kwargs['start_new_session'] is True and kwargs['env']['D1_STAGE29_NONCE']
        assert kwargs['env']['OMP_NUM_THREADS'] == kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
        if state.fault == 'factory':
            raise InterruptedError('factory lost handle')
        directory = Path(argv[argv.index('--output')+1])
        directory.mkdir(parents=True)
        value = dict(r.identity(plan, path, stage), status='COMPLETE', error=None, cleanup_complete=True,
                     inputs_unchanged=True, seed=r.c.SEED, envs=1, hyperparameters=r.c.HYPERPARAMETERS,
                     vector_layout=plan['vector_layout'], runtime_dependencies_sha256=plan['runtime_dependencies_sha256'],
                     anchor_change=r.c.anchor_change(arm), teacher_policy_used=r.c.ARMS[arm] != 0.,
                     actual_counts=r.c.expected_counts(r.c.STAGES[stage]['transitions']))
        value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        protocol = dict(value, base_checkpoint=plan['base_checkpoint'], base_protocol=plan['base_protocol'])
        if kind == 'train':
            value['format'] = 'd1-anchor29-training-result-v1'
            protocol['format'] = 'd1-anchor29-training-protocol-v1'
        elif kind == 'formal':
            length = 100 if state.short_control and arm == 'control' else 600
            rows = cases(passed=state.qualified and length == 600, length=length)
            value.update(all_36_reference_pass=state.qualified and length == 600, cases=rows,
                         actual_counts=r.c.expected_counts(length*36))
            value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        r.c.atomic_json(directory/'protocol.json', protocol)
        value['protocol_sha256'] = r.c.sha(directory/'protocol.json')
        if state.fault == 'partial' and stage == state.fault_stage:
            value.update(status='INCOMPLETE', actual_counts=dict(policy_transitions=1, native_steps=3, pd_updates=2),
                         known_actual_counts=dict(policy_transitions=1, native_steps=3, pd_updates=2))
        if state.fault == 'foreign' and stage == state.fault_stage:
            value['arm'] = 'another'
        if state.fault == 'missing' and stage == state.fault_stage:
            value['actual_counts'] = None
        r.c.atomic_json(directory/('result.json' if kind == 'train' else 'report.json'), value)
        if state.fault == 'signal':
            signal.raise_signal(signal.SIGTERM)
        state.active = True
        return SimpleNamespace(poll=lambda: 1 if state.fault in ('partial', 'foreign') else 0,
                               returncode=1 if state.fault in ('partial', 'foreign') else 0)

    def cleanup(process):
        state.cleaned.append(process)
        state.active = False
        state.now += state.cleanup_delay
        if state.fault == 'cleanup':
            raise RuntimeError('cleanup failed')
        return {'remaining': {} if state.fault != 'residual' else {17: 3}}

    def verifier(_, __, stage, directory, child):
        if state.fault == 'missing' and stage == state.fault_stage:
            raise ValueError('missing counters')
        return r.counts(child['actual_counts'], r.c.expected_counts(r.c.STAGES[stage]['transitions']))

    def initial_checker(*_):
        if state.fault == 'pair':
            raise ValueError('paired initial differs')
        return {'first_rollout_nonreward_equal': True}

    def references(_):
        state.reference_calls += 1
        value = {'status': 'COMPLETE', 'cleanup_complete': True, 'error': None,
                 'cases': cases(), 'all_36_reference_pass': True}
        if state.reference_change and state.reference_calls > 1:
            value['cases'][0]['J'] = 2.
        return {'baseline17': copy.deepcopy(value), 'random25': copy.deepcopy(value)}

    def run(**kwargs):
        return r.execute(path, loader=loader, memory_reader=memory, factory=factory, cleanup=cleanup,
                         adopter=lambda: None, verifier=verifier, initial_checker=initial_checker,
                         reference_loader=references, clock=lambda: state.now, sleeper=sleeper, **kwargs)

    return SimpleNamespace(path=path, plan=plan, state=state, run=run, loader=loader,
                           memory=memory, sleeper=sleeper, factory=factory, cleanup=cleanup)


def test_default_cli_has_no_plan_read_no_spawn_or_framework(capsys, monkeypatch):
    monkeypatch.setattr(r.c, 'load_plan', lambda *_: pytest.fail('default loaded plan'))
    monkeypatch.setattr(r.subprocess, 'Popen', lambda *_args, **_kwargs: pytest.fail('default spawned'))
    assert r.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'PLAN_ONLY' and value['physics_steps_executed'] == 0
    assert value['budget'] == r.c.expected_counts(108736)
    assert value['teacher_sample_rows_max'] == 32768 and value['cpp_closed_loop_qualified'] is False


def test_complete_pair_is_serial_exact_and_not_holdout_or_cpp_loop(execution):
    value = execution.run()
    assert value['status'] == 'COMPLETE' and value['actual_counts'] == r.c.BUDGET
    assert execution.state.stages == list(r.c.STAGE_ORDER)
    assert len(execution.state.cleaned) == 9
    assert value['original_36_pass'] == {'control': True, 'anchor': True}
    assert value['paired_initial']['first_rollout_nonreward_equal'] is True
    assert value['holdout_executed'] is value['cpp_closed_loop_qualified'] is value['simulation_candidate_qualified'] is False
    assert set(value['comparison']['reference_pass_counts']) == {'baseline17', 'random25', 'control', 'anchor'}
    assert set(value['reference_comparisons']) == {'baseline17', 'random25'}
    assert execution.state.reference_calls == 2


def test_task_failures_still_collect_both_arms(execution):
    execution.state.qualified = False
    value = execution.run()
    assert value['status'] == 'COMPLETE' and execution.state.stages[-1] == 'formal_anchor'
    assert value['original_36_pass'] == {'control': False, 'anchor': False}


def test_legal_short_control_cases_still_collect_anchor(execution):
    execution.state.short_control = True
    value = execution.run()
    assert value['status'] == 'COMPLETE' and execution.state.stages[-1] == 'formal_anchor'
    assert value['original_36_pass'] == {'control': False, 'anchor': True}
    assert value['actual_counts'] == r.c.expected_counts(108736-18000)
    assert value['comparison']['common_full_cases'] == 0


def test_admission_below_start_but_above_reserve_does_not_claim(execution):
    execution.state.memory = 3 << 30
    value = execution.run()
    assert value['status'] == 'NOT_READY' and execution.state.now == 300
    assert not execution.state.stages and not (r.ROOT/r.c.OUTPUT).exists()
    assert value['samples'] == 61


def test_admission_three_samples_ten_seconds_required(execution):
    execution.state.samples = [4 << 30, 4 << 30, (4 << 30)-1, 4 << 30, 4 << 30, 4 << 30]
    value = execution.run()
    assert value['status'] == 'COMPLETE' and value['admission']['elapsed_s'] == 25
    assert value['admission']['qualified_samples'] == 3


@pytest.mark.parametrize('memory', [None, True, -1, (2 << 30)-1])
def test_invalid_or_below_reserve_admission_stops_immediately(execution, memory):
    execution.state.memory = memory
    value = execution.run()
    assert value['status'] == 'NOT_READY' and execution.state.now == 0
    assert not execution.state.stages


def test_admission_gap_resets_stability(execution):
    state = execution.state
    def sleeper(duration):
        state.now += 61 if state.now == 0 else duration
    result = r.admission(execution.plan, execution.path, memory_reader=execution.memory,
                         clock=lambda: state.now, sleeper=sleeper, started=0, control={'received': None})
    assert result['status'] == 'ADMITTED' and result['elapsed_s'] == 71


def test_existing_output_never_retries(execution):
    (r.ROOT/r.c.OUTPUT).mkdir(parents=True)
    with pytest.raises(ValueError, match='no retry'):
        execution.run()
    assert not execution.state.stages


def test_slow_source_read_invalidates_stability_before_claim(execution):
    execution.state.before_claim_delay = 8
    value = execution.run()
    assert value['status'] == 'NOT_READY' and not execution.state.stages
    assert not (r.ROOT/r.c.OUTPUT).exists()


def test_claim_race_never_overwrites_an_existing_attempt(execution):
    execution.state.output_race = True
    value = execution.run()
    assert value['status'] == 'NOT_READY' and not execution.state.stages
    assert (r.ROOT/r.c.OUTPUT/'report.json').read_text() == 'keep original'


@pytest.mark.parametrize('fault', ['pair', 'cleanup', 'residual', 'factory', 'signal', 'missing'])
def test_failure_stops_and_owned_cleanup_is_attempted(execution, fault):
    execution.state.fault = fault
    original = signal.getsignal(signal.SIGTERM)
    value = execution.run()
    assert value['status'] == 'INCOMPLETE'
    assert len(execution.state.stages) <= 2 and execution.state.cleaned
    assert signal.getsignal(signal.SIGTERM) == original
    if fault in ('factory', 'missing'):
        assert value['actual_counts'] is None


def test_trusted_partial_interval_is_not_rounded_or_lost(execution):
    execution.state.fault = 'partial'
    value = execution.run()
    assert value['status'] == 'INCOMPLETE'
    assert value['actual_counts'] == value['known_actual_counts'] == dict(policy_transitions=1, native_steps=3, pd_updates=2)


def test_foreign_partial_is_unknown(execution):
    execution.state.fault = 'foreign'
    value = execution.run()
    assert value['actual_counts'] is None and value['known_actual_counts'] == r.ZERO


def test_partial_with_another_anchor_intervention_is_rejected(execution):
    execution.state.fault = 'partial'
    execution.run()
    directory = r.c.stage_path(execution.plan, 'train_control')
    child = r.c.read_json(directory/'result.json')
    child['anchor_change']['direction'] = 'KL(current || base)'
    with pytest.raises(ValueError, match='training scope'):
        r.partial_counts(execution.plan, execution.path, 'train_control', directory, child)


def test_stage_cleanup_deadline_is_enforced_after_exit(execution):
    execution.state.cleanup_delay = 601
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and execution.state.stages == ['train_control']
    assert value['known_actual_counts'] == r.c.expected_counts(32768)


def test_late_final_source_read_cannot_complete(execution):
    execution.state.final_delay = 1801
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and value['actual_counts'] == r.c.BUDGET
    assert value['elapsed_s'] >= 1800


def test_changed_saved_reference_cannot_complete(execution):
    execution.state.reference_change = True
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and 'references changed' in value['error']
    assert value['actual_counts'] == r.c.BUDGET


def test_failed_memory_sample_is_persisted_and_included_in_minimum(execution):
    execution.state.samples = [6 << 30]*4+[(2 << 30)-1]
    value = execution.run()
    assert value['status'] == 'INCOMPLETE' and not execution.state.stages
    assert value['minimum_effective_available_bytes'] == (2 << 30)-1
    assert value['stop_sample']['effective_available_bytes'] == (2 << 30)-1


def test_guard_failure_keeps_diagnostics():
    class Error(ValueError):
        snapshot = {'effective_available_bytes': 5 << 30, 'owned': {'events': {'oom_kill': 1}}, 'errors': ['OOM']}
    def reader():
        raise Error('OOM')
    row = r.read_sample(reader)
    assert row['owned']['events']['oom_kill'] == 1
    with pytest.raises(ValueError, match='guard'):
        r.require_memory(row, 2 << 30)


@pytest.mark.parametrize('value', [dict(policy_transitions=True, native_steps=10, pd_updates=5),
                                  dict(policy_transitions=1, native_steps=10, pd_updates=5, resets=1.5),
                                  dict(policy_transitions=1, native_steps=10, pd_updates=5, resets=True)])
def test_counts_do_not_accept_bool_or_noninteger_extra(value):
    with pytest.raises(ValueError):
        r.counts(value, r.CAP, partial=True)


def test_paired_initial_and_first_buffer_must_both_match(monkeypatch):
    bundle = dict(protocol={'initial_state': {'Adam': 'a'}, 'versions': {'torch': 'x'}},
                  result={'first_rollout_nonreward': {'sha256': 'b'}})
    values = [copy.deepcopy(bundle), copy.deepcopy(bundle)]
    monkeypatch.setattr(r, 'completed_training', lambda _plan, _path, arm: values[list(r.c.ARMS).index(arm)])
    assert r.paired_initials({}, 'unused')['first_rollout_nonreward_equal'] is True
    values[1]['result']['first_rollout_nonreward']['sha256'] = 'other'
    with pytest.raises(ValueError, match='trajectory'):
        r.paired_initials({}, 'unused')


def test_run_stage_checks_sources_before_creating_output(tmp_path):
    def loader(_):
        raise ValueError('source mismatch')
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='source mismatch'):
        r.run_stage(tmp_path/'plan', 'control', 'formal', output, loader=loader)
    assert not output.exists()


def test_run_stage_rejects_wrong_parent_before_model_or_output(tmp_path):
    def validate(*_):
        raise ValueError('parent nonce')
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='parent nonce'):
        r.run_stage(tmp_path/'plan', 'control', 'export', output, loader=lambda _: {}, launch_validator=validate,
                    training_loader=lambda *_: pytest.fail('model metadata loaded'))
    assert not output.exists()


@pytest.fixture
def exported(tmp_path, monkeypatch):
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = {'source_identity': {'fake': True}}
    bundle = {'identity': {'training_identity': {'checkpoint_sha256': 'a'*64}}, 'old_release': {},
              'protocol': {}, 'checkpoint': tmp_path/'policy.zip'}
    state = SimpleNamespace(now=0., calls=[])
    output = tmp_path/'control/export'
    def numeric(kind, actual, parent, directory):
        assert kind == 'export' and actual is bundle and parent == directory.parent
        (directory/'policy.onnx').write_bytes(b'fake numeric artifact')
        (directory/'golden_fixtures.json').write_text('{}')
        return {'onnx_sha256': r.c.sha(directory/'policy.onnx')}
    def run(**kwargs):
        return r.run_stage(path, 'control', 'export', output, loader=lambda _: plan,
                           launch_validator=lambda *_: None, memory_reader=lambda: {'effective_available_bytes': 5 << 30},
                           numeric=numeric, training_loader=lambda *_: bundle, clock=lambda: state.now, **kwargs)
    monkeypatch.setattr(r, 'completed_training', lambda *_: bundle)
    monkeypatch.setattr(r, 'verify_numeric', lambda *args, **kwargs: state.calls.append(args[0]))
    return SimpleNamespace(path=path, plan=plan, bundle=bundle, output=output, state=state, run=run)


def test_real_stage_runner_and_readback_bind_exact_final_checkpoint(exported):
    child = exported.run()
    assert child['status'] == 'COMPLETE' and child['actual_counts'] == r.ZERO
    assert r.verify_stage(exported.plan, exported.path, 'export_control', exported.output, child) == r.ZERO
    assert exported.state.calls == ['export']
    child['checkpoint_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='checkpoint'):
        r.verify_stage(exported.plan, exported.path, 'export_control', exported.output, child)


def test_nontraining_partial_cannot_claim_exact_zero_with_nonzero_known(exported):
    child = exported.run()
    child['known_actual_counts'] = r.c.expected_counts(1)
    with pytest.raises(ValueError):
        r.partial_counts(exported.plan, exported.path, 'export_control', exported.output, child)


def test_stage_output_is_exclusive(exported):
    exported.run()
    with pytest.raises(ValueError, match='exclusive'):
        exported.run()


def test_stage_numeric_return_after_deadline_is_incomplete(exported, monkeypatch):
    original = r.c.atomic_json
    def write(path, value):
        original(path, value)
        if path.name == 'protocol.json':
            exported.state.now = 181
    monkeypatch.setattr(r.c, 'atomic_json', write)
    assert exported.run()['status'] == 'INCOMPLETE'


def test_actual_export_model_guard_rejects_wrong_coef():
    hyper = r.c.HYPERPARAMETERS
    model = SimpleNamespace(**{key: hyper[key] for key in (
        'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'learning_rate')},
        num_timesteps=r.c.INITIAL+r.c.TRANSITIONS, aux_coef=0., seed=r.c.SEED, n_envs=1, device='cpu',
        get_env=lambda: None, anchor_coef=1., clip_range=lambda _: .2, normalize_advantage=True,
        clip_range_vf=None, target_kl=.005, use_sde=False)
    r.check_model(model, {'anchor_coef': 1., 'versions': {}}, {})
    model.anchor_coef = 0.
    with pytest.raises(ValueError, match='model/runtime'):
        r.check_model(model, {'anchor_coef': 1., 'versions': {}}, {})
