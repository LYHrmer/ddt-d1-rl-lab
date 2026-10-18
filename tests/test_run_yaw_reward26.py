"""Fake processes/metadata only; no NN, native inference or physical environment."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools26 import run_yaw_reward26 as r  # noqa: E402


def cases(kind='formal', *, passed=True, score=.2):
    return [dict(sequence=name, delay_ticks=delay, reset_seed=seed, policy_transitions=600,
                 done_reason='HORIZON_12S', failure=False, reference_thresholds_met=passed, J=score)
            for name, delay, seed in r.prior.matrix(kind)]


def formal(*, passed=True, score=.2):
    return dict(status='COMPLETE', error=None, cleanup_complete=True, all_36_reference_pass=passed,
                cases=cases(passed=passed, score=score))


def test_actual_factory_monitor_and_metadata_chain_use_changed_reward_and_original_counters(tmp_path, monkeypatch):
    """Execute production wrappers with fake boundaries; no framework is imported.

    Gym attribute lookup and SB3 env_method are their installed source methods.
    Only label math, vector storage and the physical base are replaced here.
    """
    import ast
    import importlib.util
    import os
    from types import ModuleType
    from tools25.low_memory_runtime25 import checked_dummy_class25
    from tools26 import runtime26 as runtime

    root = Path(__file__).resolve().parents[1]

    def source(package, relative):
        path = root/'.python-deps'/package/relative
        if not path.is_file():
            spec = importlib.util.find_spec(package)
            if spec is None:
                pytest.skip('framework source unavailable for pure AST integration test')
            path = Path(spec.origin).parent/relative
        return path

    def extract(path, class_name, method=None, namespace=None):
        node = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
        if method:
            node = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == method)
        namespace = {} if namespace is None else namespace
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)  # noqa: S102 -- locked source, fake dependencies only
        return namespace[method or class_name]

    gym_core = source('gymnasium', 'core.py')
    env_lookup = extract(gym_core, 'Env', 'get_wrapper_attr')
    wrapper_lookup = extract(gym_core, 'Wrapper', 'get_wrapper_attr')
    vector_lookup = extract(source('stable_baselines3', 'common/vec_env/dummy_vec_env.py'), 'DummyVecEnv', 'env_method')

    class Wrapper:
        get_wrapper_attr = wrapper_lookup
        def __init__(self, env): self.env = env
        @property
        def unwrapped(self): return self.env.unwrapped
        def reset(self, **kwargs): return self.env.reset(**kwargs)
        def close(self): return self.env.close()

    class Monitor(Wrapper):
        def __init__(self, env, filename): super().__init__(env); self.rewards = []
        def step(self, action):
            out = self.env.step(action); self.rewards.append(out[1]); return out

    class Env:
        get_wrapper_attr = env_lookup
        def __init__(self, **kwargs):
            self.kwargs, self.closed, self._episode_steps = kwargs, False, 0
            self._training_schedule = {'segments': [(0, 50, (0., 0., 0.)), (50, 1000, (0., 0., .2))]}
            self.counts = dict(policy_transitions=0, native_steps=0, pd_updates=0, resets=0, forward_calls=0)
        @property
        def unwrapped(self): return self
        def reset(self, **kwargs):
            self.counts.update(resets=1, forward_calls=2); return 'actor', {}
        def step(self, action):
            self._episode_steps += 1
            for key, delta in (('policy_transitions', 1), ('native_steps', 10), ('pd_updates', 5), ('forward_calls', 1)):
                self.counts[key] += delta
            raw, weights = {'wz_tracking': .25, 'vx_tracking': .4}, {'wz_tracking': 2., 'vx_tracking': 2.}
            second = {key: raw[key]*weights[key] for key in raw}
            terms = dict(raw=raw, weights=weights, contributions_per_second=second,
                         contributions_per_tick={key: value*.02 for key, value in second.items()}, policy_dt_s=.02)
            return 'actor', float(sum(second.values())*.02), False, False, {'reward_terms': terms}
        def counters(self): return dict(self.counts)
        def get_delay_state18(self):
            return {'delay_counts': {'0': 1, '1': 0, '2': 0, '3': 0}, 'last_selection': dict(
                mode='uniform_0_3', seed=182, rank=0, episode_index=0, stream=18018,
                selected_delay_ticks=0, selection='uniform_0_3')}
        def close(self): self.closed = True

    class Dummy:
        env_method = vector_lookup
        def __init__(self, env_fns): self.envs = [make() for make in env_fns]; self.num_envs = 1
        def _get_target_envs(self, indices): return self.envs
        def reset(self): return self.envs[0].reset()
        def step_async(self, actions): self.action = actions[0]
        def step_wait(self): return self.envs[0].step(self.action)
        def close(self): self.envs[0].close()

    writes = []
    counted = extract(root/'tools09/train_d1_ppo09.py', 'CountedMonitor', namespace={
        'Monitor': Monitor, 'Path': Path, 'os': os, 'atomic_json': lambda path, value: writes.append(value)})
    labels = extract(root/'training/d1_rl/velocity_labels15.py', 'VelocityLabelWrapper', namespace={
        'gym': SimpleNamespace(Wrapper=Wrapper, Env=Env), 'VELOCITY_LABEL_CONTRACT': 'fake-label-math-only'})
    labels.__init__ = Wrapper.__init__
    labels._label = lambda self, actor: {'actor': actor, 'velocity_target': 'label'}
    for name, attributes in {
        'gymnasium': {'Wrapper': Wrapper}, 'd1_rl': {'__path__': []},
        'd1_rl.delay_env18': {'D1Delay18Env': Env},
        'd1_rl.velocity_labels15': {'VelocityLabelWrapper': labels},
        'train_d1_ppo09': {'CountedMonitor': counted},
    }.items():
        module = ModuleType(name); module.__dict__.update(attributes); monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    monkeypatch.setattr(runtime, 'checked_dummy_class25', lambda: checked_dummy_class25(Dummy))
    actual = runtime.Runtime26(); actual.arm = 'yaw3'
    vec = actual.create_vec(dict(envs=1, seed=182, config='training/configs/pilot03d.json'),
                            runtime.c.TRANSITIONS, tmp_path, root, 'uniform_0_3')
    vec.reset()
    for _ in range(128):
        vec.step_async(['action']); out = vec.step_wait()
    assert out[4]['training_phase'] == 'drive_turn_left'
    assert out[4]['reward_terms']['weights']['wz_tracking'] == 3.
    expected = dict(policy_transitions=128, native_steps=1280, pd_updates=640, resets=1, forward_calls=130)
    assert actual.counters(vec) == [expected]
    assert actual.delay_state(vec)[0]['delay_counts'] == {'0': 1, '1': 0, '2': 0, '3': 0}
    reward = actual.reward_state(vec)
    assert reward['steps'] == 128 and reward['extra_reward_sum'] == pytest.approx(.005*128)
    monitor = vec.envs[0].env.env
    assert monitor.rewards == pytest.approx([.031]*128)
    assert [value['status'] for value in writes] == ['created', 'running'] and writes[-1]['counts'] == expected
    state = vec.single25.snapshot()
    assert state['rpc_attempts'] == state['rpc_returns'] == state['rpc_validated'] == 3
    vec.close()
    assert writes[-1]['status'] == 'closed' and writes[-1]['counts'] == expected
    assert vec.envs[0].unwrapped.closed and vec.single25.snapshot()['closed'] is True
    assert vec.single25.snapshot()['dirty'] is False


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.c, 'ROOT', tmp_path)
    original_path = r.c.stage_path
    monkeypatch.setattr(r.c, 'stage_path', lambda plan, stage: original_path(plan, stage, root=tmp_path))
    monkeypatch.setattr(r.os, 'getpriority', lambda *args: 10)
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = {'output_root': r.c.OUTPUT, 'source_identity': {'fixture': True}, 'budget': r.c.BUDGET,
            'base_checkpoint': {'sha256': 'a'*64}, 'base_protocol': {'sha256': 'b'*64},
            'vector_layout': r.c.VECTOR_LAYOUT, 'runtime_dependencies_sha256': {'fake.py': 'c'*64}}
    state = SimpleNamespace(now=0., stages=[], cleaned=[], fault=None, fault_stage='train_control',
                            memory=8 << 30, qualified={'control': True, 'yaw3': True},
                            scores={'control': .2, 'yaw3': .1}, holdout_pass=True, wait=None)

    def factory(argv, **kwargs):
        arm = argv[argv.index('--arm')+1]
        kind = 'train' if argv[2].endswith('train_yaw_reward26.py') else argv[argv.index('--stage')+1]
        stage = kind+'_'+arm
        state.stages.append(stage)
        assert kwargs['start_new_session'] is True and kwargs['env']['D1_STAGE26_NONCE']
        output = Path(argv[argv.index('--output')+1])
        output.mkdir(parents=True)
        value = dict(r.identity(plan, path, stage), seed=182, envs=1, hyperparameters=r.c.HYPERPARAMETERS,
            vector_layout=plan['vector_layout'], runtime_dependencies_sha256=plan['runtime_dependencies_sha256'],
            reward_change=r.c.reward_change(arm), status='COMPLETE', error=None, cleanup_complete=True,
            inputs_unchanged=True, actual_counts=r.c.expected_counts(r.c.STAGES[stage]['transitions']))
        value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        if kind in ('formal', 'holdout'):
            passed = state.qualified[arm] if kind == 'formal' else state.holdout_pass
            value.update(all_36_reference_pass=passed, cases=cases(kind, passed=passed, score=state.scores[arm]))
        value['format'] = 'd1-yaw-reward26-training-result-v1' if kind == 'train' else 'd1-yaw-reward26-stage-v1'
        protocol = {key: value[key] for key in ('stage', 'arm', 'mode', 'yaw_tracking_weight', 'plan_sha256',
            'source_identity', 'seed', 'envs', 'hyperparameters', 'vector_layout', 'runtime_dependencies_sha256', 'reward_change')}
        protocol.update(format='d1-yaw-reward26-training-protocol-v1' if kind == 'train' else 'd1-yaw-reward26-stage-v1',
                        base_checkpoint=plan['base_checkpoint'], base_protocol=plan['base_protocol'])
        if state.fault == 'natural_short' and stage == 'formal_control':
            value['cases'][0].update(policy_transitions=563, done_reason='TERMINATED', failure=True,
                                      reference_thresholds_met=False, J=100.)
            value.update(all_36_reference_pass=False, actual_counts=r.c.expected_counts(21600-37))
            value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        broken = state.fault if stage == state.fault_stage else None
        if broken == 'partial':
            value.update(status='INCOMPLETE', error='injected', actual_counts=None,
                         known_actual_counts=r.c.expected_counts(4096))
        if broken == 'partial_interval':
            value.update(status='INCOMPLETE', error='injected',
                         actual_counts=dict(policy_transitions=1, native_steps=3, pd_updates=2, resets=1, forward_calls=2))
            value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        if broken == 'foreign_partial':
            value['source_identity'] = {'foreign': True}
        r.c.atomic_json(output/'protocol.json', protocol)
        value['protocol_sha256'] = r.c.sha(output/'protocol.json')
        r.c.atomic_json(output/('result.json' if kind == 'train' else 'report.json'), value)
        proc = SimpleNamespace(returncode=2 if broken in ('partial', 'partial_interval', 'foreign_partial') else 0)
        proc.poll = lambda: proc.returncode
        if broken == 'signal':
            signal.raise_signal(signal.SIGTERM)
        if broken == 'resource':
            state.memory = (4 << 30)-1
        if broken == 'late':
            state.now += r.c.STAGES[stage]['max_wall_s']+1
        if broken == 'spawn_race':
            raise InterruptedError('factory failed before returning handle')
        if stage == 'train_control' and state.wait:
            state.memory = (6 << 30)-1
        return proc

    def cleanup(proc):
        state.cleaned.append(proc)
        broken = state.fault if state.stages and state.stages[-1] == state.fault_stage else None
        if broken == 'cleanup_exception':
            raise OSError('owned cleanup error')
        if proc is not None and broken == 'late_cleanup':
            state.now += r.c.STAGES[state.fault_stage]['max_wall_s']+1
        if proc is not None and broken == 'cleanup_signal':
            signal.raise_signal(signal.SIGTERM)
        return {'remaining': {'owned': 'alive'} if broken == 'cleanup' else {}}

    def verify(plan, path, stage, directory, child):
        if state.fault == 'verifier' and stage == state.fault_stage:
            raise ValueError('independent stage reader rejected record')
        return r.projected_counts(child['actual_counts'], r.c.expected_counts(r.c.STAGES[stage]['transitions']))

    def loader(_):
        if len(state.stages) == 1 and state.fault == 'source_drift':
            return dict(plan, source_identity={'changed': True})
        if len(state.stages) == 1 and state.fault == 'late_source_check':
            state.now = r.c.RESOURCES['total_wall_s']+1
        return copy.deepcopy(plan)

    def sleeper(seconds):
        state.now += seconds
        if state.wait == 'rising':
            state.memory = 6 << 30
        if state.wait == 'below_reserve':
            state.memory = (4 << 30)-1
        if state.wait == 'unknown':
            state.memory = None
        if state.wait == 'signal':
            signal.raise_signal(signal.SIGTERM)

    state.kwargs = dict(plan_path=path, loader=loader, memory_reader=lambda: {'effective_available_bytes': state.memory},
        factory=factory, cleanup=cleanup, adopter=lambda: None, verifier=verify,
        initial_checker=lambda *args: {'initial_state': {'same': True}, 'first_rollout_nonreward': {'same': True},
                                      'first_rollout_nonreward_equal': True},
        reference_loader=lambda plan: {'baseline17': formal(score=.3), 'random25': formal(score=.25)},
        clock=lambda: state.now, sleeper=sleeper)
    state.output = tmp_path/r.c.OUTPUT
    state.path, state.plan = path, plan
    return state


@pytest.mark.parametrize('eligible,score,winner', [
    ({'control': True, 'yaw3': True}, {'control': .2, 'yaw3': .1}, 'yaw3'),
    ({'control': True, 'yaw3': True}, {'control': .1, 'yaw3': .1}, 'control'),
    ({'control': True, 'yaw3': False}, {'control': .2, 'yaw3': .01}, 'control'),
    ({'control': False, 'yaw3': True}, {'control': .01, 'yaw3': .2}, 'yaw3'),
    ({'control': False, 'yaw3': False}, {'control': .1, 'yaw3': .2}, None),
])
def test_only_developer_eligible_winner_sees_one_holdout(execution, eligible, score, winner):
    execution.qualified, execution.scores = eligible, score
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE' and value['error'] is None
    assert execution.stages[:8] == list(r.c.STAGE_ORDER[:8])
    assert execution.stages[8:] == ([] if winner is None else ['holdout_'+winner])
    assert value['selection']['selected_arm'] == winner and value['selection']['uses_holdout'] is False
    assert value['actual_counts'] == r.c.expected_counts(326944 if winner else 305344)
    assert value['simulation_candidate_qualified'] is (winner is not None)
    assert value['cpp_closed_loop_qualified'] is value['ros_hardware_qualified'] is False
    assert value['automatic_model_replacement'] is False
    assert value['first_rollout_nonreward_equal'] is True


def test_winner_holdout_failure_never_falls_back_or_reselects(execution):
    execution.holdout_pass = False
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE' and value['selection']['selected_arm'] == 'yaw3'
    assert execution.stages[8:] == ['holdout_yaw3']
    assert value['holdout_36_pass'] is False and value['simulation_candidate_qualified'] is False
    assert value['stages']['holdout_control']['status'] == 'SKIPPED_NOT_SELECTED'


def test_task_failure_retains_paired_other_arm_and_cannot_win_on_low_j(execution):
    execution.fault = 'natural_short'
    execution.scores['control'] = .0001
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE' and 'formal_yaw3' in execution.stages
    assert value['selection']['selected_arm'] == 'yaw3'
    assert value['comparison']['common_full_cases'] == 35
    assert value['actual_counts'] == r.c.expected_counts(326944-37)


@pytest.mark.parametrize('fault', ['signal', 'resource', 'cleanup', 'cleanup_exception', 'verifier',
                                  'late', 'spawn_race', 'late_cleanup', 'cleanup_signal', 'source_drift'])
def test_engineering_fault_stops_round_and_cleans_only_owned_processes(execution, fault):
    execution.fault = fault
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and value['simulation_candidate_qualified'] is False
    assert execution.stages == ['train_control'] and execution.cleaned
    assert value['cleanup_complete'] is (fault not in ('cleanup', 'cleanup_exception'))
    if fault == 'spawn_race':
        assert None in execution.cleaned
    if fault == 'resource':
        assert value['stop_reason'] == 'RAM_BELOW_4_GIB'
        assert value['stop_sample']['effective_available_bytes'] == (4 << 30)-1
        assert value['minimum_effective_available_bytes'] == (4 << 30)-1


@pytest.mark.parametrize('fault,actual,known', [
    ('partial', None, r.c.expected_counts(4096)),
    ('partial_interval', {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2},
                         {'policy_transitions': 1, 'native_steps': 3, 'pd_updates': 2}),
    ('foreign_partial', None, r.ZERO),
])
def test_partial_counts_preserve_real_interval_and_foreign_stays_unknown(execution, fault, actual, known):
    execution.fault = fault
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and execution.stages == ['train_control']
    assert value['actual_counts'] == actual and value['known_actual_counts'] == known


def test_low_start_memory_creates_no_attempt(execution):
    execution.memory = (6 << 30)-1
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'NOT_READY' and not execution.stages and not execution.output.exists()


def test_attempt_cannot_be_reused(execution):
    r.execute(**execution.kwargs)
    with pytest.raises(ValueError, match='no retry'):
        r.execute(**execution.kwargs)


def test_admission_wait_is_bounded_and_spawns_only_once_after_ram_rises(execution):
    execution.wait = 'rising'
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE' and execution.stages.count('train_yaw3') == 1
    assert value['admission_waits'][0]['duration_s'] == 5
    assert value['admission_waits'][0]['status'] == 'ADMITTED'


@pytest.mark.parametrize('mode', ['timeout', 'below_reserve', 'unknown', 'signal'])
def test_failed_admission_wait_never_claims_or_spawns_next_stage(execution, mode):
    execution.wait = mode
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and execution.stages == ['train_control']
    assert 'train_yaw3' not in value['stages']
    assert value['admission_waits'][0]['status'] == 'STOPPED'
    assert value['elapsed_s'] <= 300
    if mode == 'unknown':
        assert value['stop_reason'] == 'EFFECTIVE_MEMORY_UNKNOWN'
        assert value['stop_sample']['effective_available_bytes'] is None
        assert value['minimum_effective_available_bytes'] > 0


def test_slow_source_check_after_wait_cannot_cross_global_deadline_and_spawn(execution):
    execution.wait, execution.fault = 'rising', 'late_source_check'
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and execution.stages == ['train_control']
    assert 'train_yaw3' not in value['stages']


def test_facts_rebuild_counts_without_double_charge():
    facts = {'train': {'actual_counts': r.c.expected_counts(131072)},
             'formal': {'actual_counts': None, 'known_actual_counts': r.c.expected_counts(600)}}
    assert r.totals(facts) == r.totals(facts) == r.c.expected_counts(131672)


@pytest.mark.parametrize('bad', [True, -1, 1.5])
def test_projection_checks_additional_counter_fields(bad):
    values = dict(r.c.expected_counts(1), forward_calls=2, resets=bad)
    with pytest.raises(ValueError, match='counter'):
        r.projected_counts(values, r.CAP)


def test_default_main_is_cold_plan_only(monkeypatch, capsys):
    monkeypatch.setattr(r, 'execute', lambda *a, **k: pytest.fail('must not execute'))
    monkeypatch.setattr(r.c, 'load_plan', lambda *a, **k: pytest.fail('must not load source/model'))
    assert r.main([]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['status'] == 'PLAN_ONLY' and value['budget'] == r.c.expected_counts(326944)
    assert value['execution_started'] is False and value['physics_steps_executed'] == 0


@pytest.mark.parametrize('field,wrong', [('num_timesteps', 1310720), ('seed', 181), ('n_envs', 4),
                                       ('learning_rate', .0003), ('target_kl', None), ('aux_coef', 1.)])
def test_numeric_export_requires_this_complete_checkpoint(field, wrong):
    model = SimpleNamespace(**{key: r.c.HYPERPARAMETERS[key] for key in (
        'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm', 'learning_rate')},
        num_timesteps=1441792, aux_coef=0., seed=182, n_envs=1, device='cpu', get_env=lambda: None,
        clip_range=lambda _: .2, normalize_advantage=True, clip_range_vf=None, target_kl=.005, use_sde=False)
    r.check_model(model, {'versions': {'fixture': True}}, {'fixture': True})
    setattr(model, field, wrong)
    with pytest.raises(ValueError):
        r.check_model(model, {'versions': {'fixture': True}}, {'fixture': True})


def test_pairing_rejects_different_first_rollout_before_any_numeric_stage(monkeypatch):
    bundles = {arm: {'protocol': {'initial_state': {'same': True}, 'versions': {'same': True}},
                     'result': {'first_rollout_nonreward': {'sha': arm}}} for arm in r.c.ARMS}
    monkeypatch.setattr(r, 'completed_training', lambda plan, path, arm: bundles[arm])
    with pytest.raises(ValueError, match='non-reward trajectories'):
        r.paired_initials({}, Path('not-read'))


def test_stage_import_failure_has_exclusive_structured_receipt(tmp_path):
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = {'source_identity': {'fixture': True}}
    bundle = {'identity': {'training_identity': {'checkpoint_sha256': 'a'*64}}}
    output = tmp_path/'export'
    def failure(*args):
        raise ImportError('injected optional dependency missing')
    value = r.run_stage(path, 'control', 'export', output, loader=lambda _: plan,
        launch_validator=lambda *args: None, memory_reader=lambda: {'effective_available_bytes': 8 << 30},
        training_loader=lambda *args: bundle, numeric=failure, clock=lambda: 0.)
    assert value['status'] == 'INCOMPLETE' and value['actual_counts'] is None
    assert (output/'report.json').is_file() and (output/'error.txt').is_file()
    with pytest.raises(ValueError, match='new26'):
        r.run_stage(path, 'control', 'export', output, loader=lambda _: plan,
                    launch_validator=lambda *args: None, training_loader=lambda *args: bundle)


def test_comparison_retains_failures_and_uses_common_full_denominator():
    first, second = formal(score=.2), formal(score=.1)
    first['cases'][0].update(policy_transitions=100, done_reason='TERMINATED', J=100., reference_thresholds_met=False)
    value = r.compare_reports({'first': first, 'second': second})
    assert value['common_full_cases'] == 35
    assert value['common_full_mean_J']['first'] == pytest.approx(.2)
    assert value['all36_mean_J_including_early_failure_penalties']['first'] > 2


@pytest.fixture
def export_receipt(tmp_path, monkeypatch):
    path = tmp_path/'plan.json'
    path.write_text('{}')
    directory = tmp_path/'control/export'
    directory.mkdir(parents=True)
    plan = {'source_identity': {'same': True}}
    bundle = {'identity': {'role': 'completed-control', 'training_identity': {'checkpoint_sha256': 'a'*64}},
              'old_release': {'frozen': True}}
    monkeypatch.setattr(r, 'completed_training', lambda *args: bundle)
    calls = []
    monkeypatch.setattr(r, 'verify_numeric', lambda *args, **kwargs: calls.append((args, kwargs)))
    child = dict(r.identity(plan, path, 'export_control'), status='COMPLETE', error=None,
                 cleanup_complete=True, inputs_unchanged=True, completed_training=bundle['identity'],
                 checkpoint_sha256='a'*64, actual_counts=dict(r.ZERO), known_actual_counts=dict(r.ZERO))
    protocol = dict(r.identity(plan, path, 'export_control'), max_wall_s=r.c.STAGES['export_control']['max_wall_s'],
                    budget_max=dict(r.ZERO), completed_training=copy.deepcopy(bundle['identity']), checkpoint_sha256='a'*64)
    r.c.atomic_json(directory/'protocol.json', protocol)
    child['protocol_sha256'] = r.c.sha(directory/'protocol.json')
    return SimpleNamespace(path=path, directory=directory, plan=plan, child=child, protocol=protocol, calls=calls)


def test_numeric_reader_receives_this_arm_final_and_frozen_assets(export_receipt):
    f = export_receipt
    assert r.verify_stage(f.plan, f.path, 'export_control', f.directory, f.child) == r.ZERO
    assert f.calls[0][1]['old_release'] == {'frozen': True}


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'completed_training', 'source_identity', 'yaw_tracking_weight'])
def test_self_consistent_foreign_export_cannot_replace_this_arm_final(export_receipt, field):
    f = export_receipt
    for value in (f.child, f.protocol):
        value[field] = {'foreign': True} if field in ('completed_training', 'source_identity') else (
            3. if field == 'yaw_tracking_weight' else 'b'*64)
    r.c.atomic_json(f.directory/'protocol.json', f.protocol)
    f.child['protocol_sha256'] = r.c.sha(f.directory/'protocol.json')
    with pytest.raises(ValueError):
        r.verify_stage(f.plan, f.path, 'export_control', f.directory, f.child)
    assert not f.calls


@pytest.mark.parametrize('where', ['actual_counts', 'known_actual_counts'])
def test_numerical_zero_receipt_cannot_use_bool_as_counter(export_receipt, where):
    f = export_receipt
    f.child[where]['resets'] = False
    with pytest.raises(ValueError, match='counter'):
        r.verify_stage(f.plan, f.path, 'export_control', f.directory, f.child)


def test_report_write_interrupt_after_stage_fact_cannot_double_charge(execution, monkeypatch):
    atomic = r.c.atomic_json
    state = {'triggered': False}
    def interrupt(path, value):
        if (Path(path) == execution.output/'report.json' and not state['triggered']
                and value.get('stages', {}).get('train_control', {}).get('status') == 'COMPLETE'):
            state['triggered'] = True
            raise OSError('report interrupted after unique actual fact')
        return atomic(path, value)
    monkeypatch.setattr(r.c, 'atomic_json', interrupt)
    value = r.execute(**execution.kwargs)
    assert state['triggered'] and value['status'] == 'INCOMPLETE'
    assert execution.stages == ['train_control']
    assert value['known_actual_counts'] == value['actual_counts'] == r.c.expected_counts(131072)


@pytest.mark.parametrize('bad', [True, 1.0])
def test_partial_known_counter_type_cannot_hide_behind_dictionary_equality(execution, bad):
    execution.fault = 'partial_interval'
    factory = execution.kwargs['factory']
    def malformed(argv, **kwargs):
        process = factory(argv, **kwargs)
        output = Path(argv[argv.index('--output')+1])
        path = output/'result.json'
        value = r.c.read_json(path)
        value['known_actual_counts']['resets'] = bad
        r.c.atomic_json(path, value)
        return process
    execution.kwargs['factory'] = malformed
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and value['actual_counts'] is None
    assert value['known_actual_counts'] == r.ZERO


def test_reference_source_failure_happens_before_any_child(execution):
    def failure(_):
        raise ValueError('old reference trace SHA changed')
    execution.kwargs['reference_loader'] = failure
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and execution.stages == []
    assert value['actual_counts'] == r.ZERO and None in execution.cleaned


def test_reference_cannot_change_after_physical_stages(execution):
    reference_loader = execution.kwargs['reference_loader']
    count = 0
    def changing(plan):
        nonlocal count
        count += 1
        value = reference_loader(plan)
        if count == 2:
            value['random25']['cases'][0]['J'] += 1
        return value
    execution.kwargs['reference_loader'] = changing
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and value['simulation_candidate_qualified'] is False
    assert value['actual_counts'] == r.c.expected_counts(326944)
