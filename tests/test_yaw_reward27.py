"""Private frozen-code binding and fake-resource/process checks only."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools27 import run_yaw_reward27 as r  # noqa: E402


def test_private_binding_changes_only_enumerated_identity_nodes_not_math_or_old_globals():
    from tools26 import common26, runtime26, train_yaw_reward26, run_yaw_reward26
    original = (common26.OUTPUT, common26.FORMAT, common26.command, runtime26.c, train_yaw_reward26.common, run_yaw_reward26.c)
    bound = r.bindings()
    for name in r.FROZEN26:
        old = ast.parse((r.ROOT/'tools26'/(name+'.py')).read_text())
        derived = r.derived_tree(name)
        reverse_labels = {new: old for old, new in r.LABELS.items()}
        class Undo(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) is str and node.value in reverse_labels:
                    node.value = reverse_labels[node.value]
                return node
            def visit_ImportFrom(self, node):
                if node.module in (r.PACKAGE, r.PACKAGE+'.runtime26'):
                    node.module = 'tools26'+node.module[len(r.PACKAGE):]
                return node
            def visit_keyword(self, node):
                if node.arg == 'D1_STAGE27_NONCE': node.arg = 'D1_STAGE26_NONCE'
                return self.generic_visit(node)
        assert ast.dump(Undo().visit(derived), include_attributes=False) == ast.dump(old, include_attributes=False)
    assert original == (common26.OUTPUT, common26.FORMAT, common26.command, runtime26.c, train_yaw_reward26.common, run_yaw_reward26.c)
    assert bound.common is not common26 and bound.runtime.c is bound.train.common is bound.runner.c is bound.common
    assert bound.train.run.__kwdefaults__['loader'] is bound.common.load_plan
    assert bound.train.run.__kwdefaults__['launch_validator'] is bound.common.validate_launch
    assert bound.runner.execute.__kwdefaults__['loader'] is bound.common.load_plan
    assert bound.runtime.Runtime26.create_vec.__globals__['c'] is bound.common
    assert set(bound.common.SOURCES) == set(common26.SOURCES)|set(r.AUTHORED)
    assert bound.common.OUTPUT == 'evidence/yaw_reward27/execution_01'
    assert bound.common.PLAN == 'evidence/yaw_reward27/plan_01.json'
    assert bound.common.BUDGET == common26.BUDGET == dict(policy_transitions=326944, native_steps=3269440, pd_updates=1634720)
    assert bound.common.HYPERPARAMETERS == common26.HYPERPARAMETERS
    assert bound.common.ARMS == common26.ARMS and bound.common.STAGE_ORDER == common26.STAGE_ORDER
    assert bound.common.RESOURCES == common26.RESOURCES
    assert bound.runtime.__file__ == runtime26.__file__
    assert bound.derivation['original_file_identity_retained'] is True
    for stage in bound.common.STAGE_ORDER:
        argv = bound.common.command(r.ROOT/bound.common.PLAN, stage, r.ROOT/bound.common.OUTPUT)
        assert argv[2] == str(r.ROOT/'tools27/run_yaw_reward27.py')
        assert argv[argv.index('--stage')+1] == bound.common.split_stage(stage)[1]
    parent_code = bound.runner.execute.__code__
    assert 'D1_STAGE26_NONCE' not in str(parent_code.co_consts)
    assert 'D1_STAGE27_NONCE' in str(parent_code.co_consts)


def test_sha_gate_rejects_changed_frozen_source(tmp_path):
    (tmp_path/'tools26').mkdir()
    (tmp_path/'tools26/common26.py').write_text('# changed')
    with pytest.raises(ValueError, match='frozen26'):
        r.derived_tree('common26', tmp_path)


def test_default_cli_is_cold_and_does_not_prepare_or_claim():
    code = "import sys;from tools27.run_yaw_reward27 import main;assert main([])==0;assert not any(n.partition('.')[0] in {'torch','numpy','gymnasium','mujoco','stable_baselines3'} for n in sys.modules)"
    process = subprocess.run([sys.executable, '-B', '-c', code], cwd=r.ROOT, check=True, capture_output=True, text=True)
    value = json.loads(process.stdout)
    assert value['status'] == 'PLAN_ONLY' and value['physics_steps_executed'] == 0 and value['training_started'] is False
    assert value['admission']['minimum_effective_bytes'] == 8 << 30


@pytest.fixture
def admission(tmp_path, monkeypatch):
    real = r.bindings()
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(os, 'getpriority', lambda *args: 10)
    plan_path = tmp_path/'plan.json'; plan_path.write_text('{}')
    plan = {'source_identity': {'fake': True}, 'output_root': real.common.OUTPUT, 'budget': real.common.BUDGET}
    state = SimpleNamespace(now=0., mode=None, factories=[], cleanup=[], reads=0, loads=0)
    def loader(path):
        state.loads += 1
        if state.mode == 'source' and state.loads > 1:
            return dict(plan, source_identity={'changed': True})
        if state.mode == 'slow_source' and state.loads == 2:
            state.now = 301.
        if state.mode == 'proof_gap' and state.loads == 2:
            state.now += 6.
        if state.mode == 'late_final_sources' and state.loads == 3:
            state.now = 3901.
        return copy.deepcopy(plan)
    def memory():
        state.reads += 1
        ram = 8 << 30
        if state.mode == 'reset' and state.now == 30.: ram = 7 << 30
        if state.mode == 'timeout': ram = 7 << 30
        if state.mode == 'reserve': ram = (4 << 30)-1
        if state.mode == 'unknown': ram = None
        if state.mode == 'lost' and state.loads >= 2: ram = (8 << 30)-1
        return {'available_ram_bytes': ram, 'cgroup_remaining_bytes': None, 'effective_available_bytes': ram}
    def sleep(seconds):
        state.now += seconds
        if state.mode == 'signal': signal.raise_signal(signal.SIGTERM)
    def factory(argv, **kwargs):
        state.factories.append((argv, kwargs))
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['D1_PARENT27_NONCE'] and kwargs['env']['D1_PARENT27_TICKET_SHA256']
        assert state.now >= 60. and not (tmp_path/plan['output_root']).exists()
        if state.mode == 'spawn_race': raise InterruptedError('no handle returned')
        if state.mode == 'spawn_signal': signal.raise_signal(signal.SIGTERM)
        output = tmp_path/plan['output_root']; output.mkdir(parents=True)
        counts = dict(policy_transitions=305344, native_steps=3053440, pd_updates=1526720)
        child = {'format': real.common.EXECUTION_FORMAT, 'status': 'COMPLETE', 'error': None,
            'source_identity': plan['source_identity'], 'plan_sha256': r.sha(plan_path), 'cleanup_complete': True,
            'actual_counts': counts, 'known_actual_counts': counts}
        if state.mode == 'foreign': child['format'] = 'd1-yaw-reward26-execution-v1'
        if state.mode == 'unknown_complete': child['actual_counts'] = None
        real.common.atomic_json(output/'report.json', child)
        if state.mode == 'late_exit': state.now += 3601.
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    def cleanup(process):
        state.cleanup.append(process)
        if state.mode == 'cleanup_exception': raise OSError('fake cleanup')
        if state.mode == 'late_cleanup': state.now = 3901.
        return {'remaining': {'fake': True} if state.mode == 'cleanup_remaining' else {}}
    state.kwargs = dict(plan_path=plan_path, bundle=real, loader=loader, memory_reader=memory,
        factory=factory, cleanup=cleanup, adopter=lambda: None, clock=lambda: state.now, sleeper=sleep,
        sidecar=tmp_path/'evidence/yaw_reward27/admissions/fake')
    state.plan, state.path = plan, plan_path
    return state


@pytest.mark.parametrize('mode,elapsed', [(None, 60.), ('reset', 95.)])
def test_continuous_admission_then_one_parent_spawn_only(admission, mode, elapsed):
    admission.mode = mode
    result = r.execute(**admission.kwargs)
    assert result['status'] == 'COMPLETE' and result['execution_started'] is True
    assert admission.now == elapsed and len(admission.factories) == 1
    assert result['actual_counts'] == result['known_actual_counts'] == dict(policy_transitions=305344, native_steps=3053440, pd_updates=1526720)
    assert len(admission.cleanup) == 1 and result['cleanup_complete'] is True


@pytest.mark.parametrize('mode', ['timeout', 'reserve', 'unknown', 'lost', 'source', 'slow_source', 'proof_gap', 'signal'])
def test_failed_or_lost_admission_never_claims_training_or_spawns(admission, mode):
    admission.mode = mode
    result = r.execute(**admission.kwargs)
    assert result['status'] in ('NOT_READY', 'INCOMPLETE') and result['execution_started'] is False
    assert not admission.factories and admission.cleanup == [None]
    assert not (r.ROOT/admission.plan['output_root']).exists()
    assert not (admission.kwargs['sidecar']/'launch_ticket.json').exists()
    assert admission.now <= 301.
    assert (admission.kwargs['sidecar']/'report.json').is_file()
    if mode == 'reserve': assert result['stop_sample']['effective_available_bytes'] == (4 << 30)-1


@pytest.mark.parametrize('mode', ['spawn_race', 'spawn_signal', 'foreign', 'late_exit', 'late_cleanup',
                                 'late_final_sources', 'unknown_complete', 'cleanup_remaining', 'cleanup_exception'])
def test_parent_failures_cleanup_owned_tree_once_and_fail_closed(admission, mode):
    admission.mode = mode
    result = r.execute(**admission.kwargs)
    assert result['status'] == 'INCOMPLETE' and len(admission.factories) == 1
    assert len(admission.cleanup) == 1
    if mode == 'spawn_race':
        assert admission.cleanup == [None] and result['actual_counts'] is None
        assert result['spawn_attempted'] is True and result['execution_started'] is None
    if mode == 'foreign': assert result['actual_counts'] is None
    if mode == 'unknown_complete':
        assert result['actual_counts'] is None and result['known_actual_counts']['policy_transitions'] == 305344
    if mode.startswith('cleanup_'): assert result['cleanup_complete'] is False


def test_existing_training_attempt_and_sidecar_are_never_overwritten(admission):
    output = r.ROOT/admission.plan['output_root']; output.mkdir(parents=True)
    with pytest.raises(ValueError, match='already claimed'): r.execute(**admission.kwargs)
    output.rmdir(); sidecar = admission.kwargs['sidecar']; sidecar.mkdir(parents=True)
    with pytest.raises(ValueError, match='exclusive'): r.execute(**admission.kwargs)
    assert not admission.factories


def test_parent_requires_actual_ready_samples_nonce_source_and_outer_parent(admission, monkeypatch):
    r.execute(**admission.kwargs)
    api = admission.kwargs['bundle'].common
    sidecar = admission.kwargs['sidecar']
    ticket = api.read_json(sidecar/'launch_ticket.json')
    receipt = api.read_json(sidecar/'report.json'); receipt['status'] = 'RUNNING_PARENT'
    api.atomic_json(sidecar/'report.json', receipt)
    monkeypatch.setattr(os, 'getppid', lambda: ticket['outer_pid'])
    monkeypatch.setenv('D1_PARENT27_NONCE', ticket['nonce'])
    monkeypatch.setenv('D1_PARENT27_TICKET_SHA256', api.sha(sidecar/'launch_ticket.json'))
    assert r.validate_parent_ticket(admission.plan, admission.path, sidecar, api) == ticket
    monkeypatch.setenv('D1_PARENT27_NONCE', 'foreign')
    with pytest.raises(ValueError, match='ticket'): r.validate_parent_ticket(admission.plan, admission.path, sidecar, api)
    monkeypatch.setenv('D1_PARENT27_NONCE', ticket['nonce'])
    ticket['source_identity'] = {'foreign': True}; api.atomic_json(sidecar/'launch_ticket.json', ticket)
    monkeypatch.setenv('D1_PARENT27_TICKET_SHA256', api.sha(sidecar/'launch_ticket.json'))
    with pytest.raises(ValueError, match='ticket'): r.validate_parent_ticket(admission.plan, admission.path, sidecar, api)
