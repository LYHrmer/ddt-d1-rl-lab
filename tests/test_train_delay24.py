"""Plain counters and fake models only; no NN, SB3 worker, robot or physics."""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools24 import train_delay24 as t  # noqa: E402

VERSIONS = dict.fromkeys(('python', 'torch', 'sb3', 'numpy', 'gymnasium', 'mujoco'), 'fake')
GIB = 1 << 30


class Clock:
    now = 0.
    def __call__(self):
        return self.now


def initial_state():
    return {'num_timesteps': t.INITIAL, 'optimizer_restored': True, 'updates': 1024,
        'policy_state_sha256': 'a'*64, 'optimizer_state_sha256': 'b'*64,
        'optimizer_state_sha256_before_lr': 'b'*64, 'optimizer_state_sha256_after_lr': 'c'*64,
        'optimizer_without_lr_sha256_before': 'd'*64, 'optimizer_without_lr_sha256_after': 'd'*64,
        'optimizer_momentum_preserved': True, 'learning_rate_before': 3e-4,
        'learning_rate_after': 1e-4, 'lr_schedule_endpoints': [1e-4, 1e-4], 'target_kl_before': None, 'target_kl_after': .005,
        'optimizer_state_sha256_before_target_kl': 'c'*64, 'optimizer_state_sha256_after_target_kl': 'c'*64,
        'target_kl_changes_optimizer_state': False}


class FakeArray:
    """An untrusted step observation; deliberately not JSON serialisable."""


class Runtime:
    def __init__(self, clock, fault=None):
        self.clock, self.fault = clock, fault
        self.spawn_calls = self.metadata_sends = 0
        self.learn_calls = 0
        self.vec = self.model = None
    def prepare(self):
        self.versions = dict(VERSIONS)
        if self.fault == 'import':
            raise ImportError('synthetic failed import')
        if self.fault == 'versions':
            self.versions['torch'] = 'changed version'
        if self.fault == 'slow_import':
            self.clock.now = 1801.
    def create_vec(self, plan, transitions, output, root, mode):
        assert transitions == 262144 and mode in ('zero', 'uniform_0_3')
        self.mode = mode
        self.spawn_calls += 1
        if self.fault == 'spawn':
            raise PermissionError('synthetic forkserver failure')
        self.vec = SimpleNamespace(steps=0, envs=4, waiting=False, dirty=False, closed=False, processes=[])
        if self.fault == 'spawn_signal':
            signal.raise_signal(signal.SIGTERM)
        return self.vec
    def create_model(self, vec, plan, hyper, output):
        assert hyper == t.HYPERPARAMETERS and hyper['learning_rate'] == 1e-4
        self.initial_identity = initial_state()
        if self.fault == 'momentum':
            self.initial_identity['optimizer_without_lr_sha256_after'] = 'e'*64
        class Model:
            num_timesteps = t.INITIAL
            target_kl = .005
            def save(model, path):
                if self.fault == 'save':
                    raise OSError('synthetic checkpoint failure')
                Path(path).write_bytes(f'fake:{model.num_timesteps}'.encode())
                if self.fault == 'save_late':
                    self.clock.now = 1801.
        self.model = Model()
        if self.fault == 'target_lie':
            self.model.target_kl = None
        return self.model
    def policy_sha(self, model):
        return f'{model.num_timesteps:064x}'
    def counters(self, vec):
        if vec.waiting or vec.dirty:
            vec.dirty = True
            raise RuntimeError('unsafe pending step: metadata refused before send')
        self.metadata_sends += 1
        if self.fault == 'counter_array' and vec.steps >= 8192:
            return [({'actor': FakeArray()}, .1, False, {}, {})]*4
        rows = [dict(t.common.expected_counts(vec.steps//4), resets=1, forward_calls=vec.steps//4+3) for _ in range(4)]
        if self.fault == 'counter_extra' and vec.steps >= 8192:
            rows[0]['resets'] = 1.5
        return rows
    def delay_state(self, vec):
        if vec.waiting or vec.dirty:
            raise RuntimeError('unsafe pending step: delay RPC refused')
        self.metadata_sends += 1
        rows = [{'last_selection': {'mode': self.mode, 'rank': rank, 'selected_delay_ticks': 0},
                 'delay_counts': {'0': 1, '1': 0, '2': 0, '3': 0}} for rank in range(4)]
        if self.fault == 'delay_array':
            rows[0]['untrusted'] = FakeArray()
        return rows
    def rpc_snapshot(self, vec):
        return {'state': 'NOT_CREATED'} if vec is None else {
            'dirty': vec.dirty or self.fault == 'dirty_after_close', 'waiting': vec.waiting, 'closed': vec.closed}
    def learn(self, model, transitions, watcher):
        self.learn_calls += 1
        if self.fault == 'plan_changed':
            path = watcher.output.parents[3]/'plan.json'
            path.write_text(path.read_text()+'\n')
        if not watcher.poll(force=True):
            return
        limit = 4096 if self.fault in ('partial', 'waiting_signal') else transitions
        infos = [{'training_phase': 'drive_backward', 'reward_terms': {'contributions_per_tick': {'vx': .1}}}]*4
        for _ in range(limit//4096):
            for _ in range(1024):
                self.vec.steps += 4
                model.num_timesteps = t.INITIAL+self.vec.steps
                if not watcher.on_step(infos, [.1]*4, self.vec.steps):
                    return
            self.clock.now += .1
            watcher.on_rollout_end(self.vec.steps)
        if self.fault == 'waiting_signal':
            self.vec.waiting = True
            signal.raise_signal(signal.SIGINT)
        if self.fault == 'late':
            self.clock.now = 1801.


@pytest.fixture
def case(tmp_path):
    (tmp_path/'base.zip').write_bytes(b'original completed control and Adam')
    t.atomic_json(tmp_path/'base_protocol.json', {'hyperparameters': t.ORIGINAL_HYPERPARAMETERS, 'versions': VERSIONS})
    plan = {'config': 'pilot03d.json', 'envs': 4, 'seed': 181, 'output_root': 'evidence/run',
        'stages': {key: dict(t.common.STAGES[key]) for key in ('train_zero','train_random')},
        'base_checkpoint': {'path': str(tmp_path/'base.zip'), 'sha256': t.sha(tmp_path/'base.zip'), 'trained_transitions': t.INITIAL},
        'base_protocol': {'path': str(tmp_path/'base_protocol.json'), 'sha256': t.sha(tmp_path/'base_protocol.json')},
        'hyperparameters': t.HYPERPARAMETERS, 'source_identity': {'git_commit': 'fake', 'sources_sha256': {}},
        'resources': {'start_min_effective_bytes': 6*GIB, 'reserve_bytes': 4*GIB, 'poll_s': 1.}}
    t.atomic_json(tmp_path/'plan.json', plan)
    return tmp_path, tmp_path/'plan.json', plan


def run_fake(case, *, fault=None, memory=None, arm='zero'):
    root, path, plan = case
    clock, runtime = Clock(), None
    runtime = Runtime(clock, fault)
    def close(vec):
        vec.closed = True
        return (False, 'synthetic close failed') if fault == 'close' else (True, None)
    result = t.run(path, arm, root/plan['output_root']/arm/'train', root=root, runtime=runtime,
        loader=lambda p: t.common.read_json(p), launch_validator=lambda *args: None,
        memory_reader=memory or (lambda: {'available_ram_bytes': 8*GIB, 'effective_available_bytes': 7*GIB,
                                         'cgroup_remaining_bytes': None}), clock=clock, closer=close)
    return result, runtime, root/plan['output_root']/arm/'train'


def test_complete_fixed64_rollouts_restores_offset_and_saves_only_final(case):
    result, runtime, output = run_fake(case)
    assert result['status'] == 'COMPLETE' and result['error'] is None
    assert result['new_trained_transitions'] == 262144 and result['final_num_timesteps'] == 1310720
    assert result['actual_counts']['policy_transitions'] == 262144 and result['cleanup_complete']
    assert result['eligible_for_evaluation'] and runtime.vec.closed and not (output/'partial_policy.zip').exists()
    assert result['initial_state'] == initial_state()
    assert result['checkpoint_sha256'] == t.sha(output/'policy.zip')
    rows = [json.loads(line) for line in (output/'ledger.jsonl').read_text().splitlines()]
    assert len(rows) == 64 and rows[0]['transitions'] == 4096 and rows[-1]['transitions'] == 262144
    assert all(row['phase_rewards']['drive_backward']['ticks'] == 4096 for row in rows)
    protocol = t.common.read_json(output/'protocol.json')
    assert protocol['initial_state'] == initial_state() and protocol['hyperparameters']['learning_rate'] == 1e-4
    assert protocol['reset_num_timesteps'] is False and protocol['warm_start'] is True


@pytest.mark.parametrize('fault', ['import', 'versions', 'slow_import', 'spawn', 'spawn_signal', 'momentum',
                                  'partial', 'late', 'plan_changed', 'save', 'save_late', 'close', 'dirty_after_close'])
def test_faults_never_export_partial_as_complete_and_keep_receipt(case, fault):
    prior = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    result, runtime, output = run_fake(case, fault=fault)
    assert result['status'] != 'COMPLETE' and not result['eligible_for_evaluation']
    assert (output/'result.json').is_file() and not (output/'policy.zip').exists()
    assert prior == {number: signal.getsignal(number) for number in prior}
    if fault in ('import', 'versions', 'slow_import'):
        assert runtime.spawn_calls == 0
    if fault == 'spawn':
        assert result['actual_counts'] is None and not result['cleanup_complete']
        assert result['rpc_guard24']['state'] == 'CONSTRUCTION_FAILED'
        assert result['rpc_guard24']['ownership_unknown'] is True
    if fault == 'plan_changed':
        assert result['inputs_unchanged'] is False
    if fault == 'late':
        assert result['actual_counts']['policy_transitions'] == 262144


@pytest.mark.parametrize('fault', ['counter_array', 'counter_extra', 'delay_array', 'waiting_signal'])
def test_interrupted_rpc_or_nonjson_candidates_cannot_pollute_failure_receipt(case, fault):
    result, runtime, output = run_fake(case, fault=fault)
    saved = t.common.read_json(output/'result.json')
    assert result['status'] != 'COMPLETE' and saved == result and runtime.vec.closed
    if fault.startswith('counter') or fault == 'waiting_signal':
        assert result['worker_counts'] is result['actual_counts'] is None
        assert result['known_actual_counts']['policy_transitions'] == 4096
    if fault == 'delay_array':
        assert result['delay_state18'] is None and result['actual_counts']['policy_transitions'] == 262144
    if fault == 'waiting_signal':
        assert runtime.metadata_sends == 1 and result['rpc_guard24']['dirty'] is True


def test_new4gib_reserve_records_real_drop_at3_point5gib(case):
    calls = [0]
    def memory():
        calls[0] += 1
        value = 7*GIB if calls[0] <= 3 else 7*(1 << 29)
        return {'available_ram_bytes': 8*GIB, 'effective_available_bytes': value, 'cgroup_remaining_bytes': value}
    result, runtime, output = run_fake(case, memory=memory)
    assert result['status'] == 'RAM_RESERVE_BELOW_4_GIB' and runtime.vec.closed
    assert result['stop_sample']['effective_available_bytes'] == 7*(1 << 29)
    assert t.common.read_json(output/'stop_sample.json')['reason'] == 'RAM_RESERVE_BELOW_4_GIB'


def test_admission_refusal_is_cold_and_does_not_claim_output(case):
    root, path, plan = case
    runtime = Runtime(Clock())
    with pytest.raises(ValueError, match='NOT_READY'):
        t.run(path, 'zero', root/plan['output_root']/'zero/train', root=root, runtime=runtime,
            loader=lambda p: plan, launch_validator=lambda *args: None,
            memory_reader=lambda: {'effective_available_bytes': 5*GIB})
    assert runtime.spawn_calls == 0 and not (root/plan['output_root']/'zero/train').exists()


def test_fixed_output_nonce_and_no_retry_are_checked_before_prepare(case):
    root, path, plan = case
    runtime = Runtime(Clock())
    with pytest.raises(ValueError, match='exclusive'):
        t.run(path, 'zero', root/'other', root=root, runtime=runtime, loader=lambda p: plan)
    def reject(*args):
        raise ValueError('missing parent nonce')
    with pytest.raises(ValueError, match='nonce'):
        t.run(path, 'zero', root/plan['output_root']/'zero/train', root=root, runtime=runtime, loader=lambda p: plan,
              launch_validator=reject)
    run_fake(case)
    with pytest.raises(ValueError, match='exclusive'):
        run_fake(case)




@pytest.mark.parametrize('arm', ['zero', 'random'])
def test_both_arms_use_their_actual_delay_mode_and_same_budget(case, arm):
    result, runtime, output = run_fake(case, arm=arm)
    assert result['status'] == 'COMPLETE' and runtime.mode == t.common.ARMS[arm]
    assert result['arm'] == arm and result['stage'] == 'train_'+arm and result['mode'] == runtime.mode
    assert result['hyperparameters']['target_kl'] == .005
    assert t.common.read_json(output/'protocol.json')['target_kl_change']['normal_batch_stop_strict_above'] == .0075


@pytest.mark.parametrize('fault', [None, 'old_target', 'optimizer', 'policy'])
def test_actual_kl_setter_changes_real_attribute_only_and_keeps_original_lr_loader(tmp_path, monkeypatch, fault):
    from tools22 import train_low_lr22 as old
    from tools22 import common22
    state = {'step': 1024, 'exp_avg': .125}
    optimizer = SimpleNamespace(state_dict=lambda: {'state': state, 'param_groups': [{'lr': 1e-4}]})
    policy_sha = ['a'*64]
    class Model:
        def __init__(self):
            self.policy = SimpleNamespace(optimizer=optimizer)
            self._target = .01 if fault == 'old_target' else None
        @property
        def target_kl(self):return self._target
        @target_kl.setter
        def target_kl(self, value):
            self._target = value
            if fault == 'optimizer':state['exp_avg'] = .25
            if fault == 'policy':policy_sha[0] = 'f'*64
    model = Model()
    runtime = SimpleNamespace(initial_identity=initial_state(), policy_sha=lambda _: policy_sha[0])
    runtime.initial_identity['optimizer_state_sha256_after_lr'] = t.optimizer_sha(optimizer)
    calls = []
    def load(runtime_arg, vec, plan, hyper, output):
        assert runtime_arg is runtime and hyper == common22.HYPERPARAMETERS and 'target_kl' not in hyper
        calls.append(plan)
        return model
    monkeypatch.setattr(old, 'load_low_lr_model22', load)
    if fault:
        with pytest.raises(ValueError):t.load_kl_model24(runtime, object(), {'real24': True}, t.HYPERPARAMETERS, tmp_path)
    else:
        result = t.load_kl_model24(runtime, object(), {'real24': True}, t.HYPERPARAMETERS, tmp_path)
        assert result is model and result.target_kl == .005 and calls == [{'real24': True}]
        assert runtime.initial_identity['target_kl_before'] is None
        assert runtime.initial_identity['optimizer_state_sha256_before_target_kl'] == runtime.initial_identity['optimizer_state_sha256_after_target_kl']


def test_missing_kl_change_proof_cannot_be_eligible():
    value = initial_state()
    assert t.initial_identity_valid(value)
    value['target_kl_after'] = None
    assert not t.initial_identity_valid(value)


def test_real_model_wrong_kl_is_refused_before_any_learn_despite_claimed_proof(case):
    result, runtime, output = run_fake(case, fault='target_lie')
    assert runtime.initial_identity['target_kl_after'] == .005
    assert runtime.model.target_kl is None and runtime.learn_calls == 0
    assert result['status'] != 'COMPLETE' and not result['eligible_for_evaluation']
    assert result['actual_counts']['policy_transitions'] == 0
    assert not (output/'policy.zip').exists()


def test_help_stays_cold(capsys):
    with pytest.raises(SystemExit) as exited:t.main(['--help'])
    assert exited.value.code == 0 and '--arm' in capsys.readouterr().out
