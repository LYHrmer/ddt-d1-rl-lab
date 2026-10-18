"""Pure fake processes and metadata; no model, runtime session or physical worker."""
from __future__ import annotations
import copy
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools25 import run_delay_comparison25 as r  # noqa: E402


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.c, 'ROOT', tmp_path)
    original_path = r.c.stage_path
    monkeypatch.setattr(r.c, 'stage_path', lambda plan, key: original_path(plan, key, root=tmp_path))
    monkeypatch.setattr(r.os, 'getpriority', lambda *args: 10)
    path = tmp_path/'plan.json'
    path.write_text('{}')
    plan = {'output_root': r.c.OUTPUT, 'source_identity': {'new': True}, 'arms': r.c.ARMS,
            'base_checkpoint': {'sha256': 'a'*64}, 'base_protocol': {'sha256': 'b'*64},
            'vector_layout':r.c.VECTOR_LAYOUT, 'runtime_dependencies_sha256':{'fake.py':'a'*64},
            'hyperparameters': r.c.HYPERPARAMETERS, 'budget': r.c.expected_counts(610688)}
    state = SimpleNamespace(now=0., stages=[], cleaned=[], fault=None, qualified=False, memory=8 << 30)
    def factory(argv, **kwargs):
        arm = argv[argv.index('--arm')+1]
        kind = 'train' if 'train_delay25.py' in argv[2] else argv[argv.index('--stage')+1]
        stage = kind+'_'+arm
        state.stages.append(stage)
        output = Path(argv[argv.index('--output')+1])
        assert kwargs['start_new_session'] is True and kwargs['env']['D1_STAGE25_NONCE']
        output.mkdir(parents=True)
        value = {'stage': stage, 'arm': arm, 'mode': r.c.ARMS[arm], 'seed': 181, 'envs': 1,
            'plan_sha256': r.c.sha(path), 'source_identity': plan['source_identity'], 'hyperparameters': r.c.HYPERPARAMETERS,
            'vector_layout':r.c.VECTOR_LAYOUT, 'runtime_dependencies_sha256':plan['runtime_dependencies_sha256'],
            'status': 'COMPLETE', 'actual_counts': r.c.expected_counts(r.c.STAGES[stage]['transitions']),
            'all_36_reference_pass': state.qualified,
            'cases': [{'J': .1 if arm=='random' else .2, 'sequence': sequence, 'delay_ticks': delay,
                       'reset_seed': seed, 'policy_transitions': 600,
                       'reference_thresholds_met': state.qualified}
                      for sequence, delay, seed in r.prior.matrix('formal')]}
        value['known_actual_counts'] = copy.deepcopy(value['actual_counts'])
        value['format'] = 'd1-delay-comparison25-training-result-v1' if kind=='train' else 'd1-delay-comparison25-stage-v1'
        if state.fault == 'partial':
            value['actual_counts'] = None
            value['known_actual_counts'] = r.c.expected_counts(4096)
        if state.fault == 'natural_short' and kind=='formal' and arm=='zero':
            value['actual_counts'] = r.c.expected_counts(21563)
            value['all_36_reference_pass'] = False
            value['cases'][0]['policy_transitions'] = 563
        protocol = {k:value[k] for k in ('stage','arm','mode','seed','envs','plan_sha256','source_identity','hyperparameters','vector_layout','runtime_dependencies_sha256')}
        protocol.update(format='d1-delay-comparison25-training-protocol-v1' if kind=='train' else 'd1-delay-comparison25-stage-v1',
            base_checkpoint=plan['base_checkpoint'], base_protocol=plan['base_protocol'])
        r.c.atomic_json(output/'protocol.json', protocol)
        value['protocol_sha256'] = r.c.sha(output/'protocol.json')
        r.c.atomic_json(output/('result.json' if kind=='train' else 'report.json'), value)
        process = SimpleNamespace(returncode=1 if state.fault=='partial' else 0)
        process.poll = lambda: process.returncode
        if state.fault=='signal':signal.raise_signal(signal.SIGTERM)
        if state.fault=='resource':state.memory = (4 << 30)-1
        if state.fault=='late':state.now += 1801.
        if state.fault=='spawn_race':raise InterruptedError('spawn returned no handle')
        return process
    def cleanup(process):
        state.cleaned.append(process)
        if state.fault=='cleanup_exception':raise OSError('cleanup failed')
        if state.fault=='late_cleanup' and process is not None:state.now += 1801.
        if state.fault=='cleanup_signal' and process is not None:signal.raise_signal(signal.SIGTERM)
        return {'remaining': {'owned':'alive'} if state.fault=='cleanup' else {}}
    def verify(plan, path, stage, directory, child):
        if state.fault=='verifier':raise ValueError('independent reader rejected record')
        return r.projected_counts(child['actual_counts'], r.c.expected_counts(r.c.STAGES[stage]['transitions']))
    state.kwargs = dict(plan_path=path, loader=lambda p: copy.deepcopy(plan),
        memory_reader=lambda: {'effective_available_bytes':state.memory}, factory=factory,
        cleanup=cleanup, adopter=lambda:None, verifier=verify, initial_checker=lambda *args:{'same':True},
        clock=lambda:state.now, sleeper=lambda seconds:setattr(state,'now',state.now+seconds))
    return state


@pytest.mark.parametrize('qualified',[False,True])
def test_both_arms_run_in_fixed_order_with_conditional_holdouts(execution, qualified):
    execution.qualified = qualified
    value = r.execute(**execution.kwargs)
    assert value['status']=='COMPLETE' and value['error'] is None
    assert execution.stages==list(r.c.STAGE_ORDER if qualified else r.c.STAGE_ORDER[:-2])
    assert value['actual_counts']==r.c.expected_counts(610688 if qualified else 567488)
    assert value['selection']['selected_arm']==('random' if qualified else None)
    assert value['automatic_model_replacement'] is False and value['active_stage'] is None


@pytest.mark.parametrize('fault',['partial','signal','resource','cleanup','cleanup_exception','verifier','late','spawn_race','late_cleanup','cleanup_signal'])
def test_engineering_failures_stop_whole_round_and_keep_owned_cleanup(execution, fault):
    execution.fault=fault
    value=r.execute(**execution.kwargs)
    assert value['status']=='INCOMPLETE' and value['selection'] is None
    assert execution.stages==['train_zero'] and execution.cleaned
    assert value['cleanup_complete'] is (fault not in ('cleanup','cleanup_exception'))
    if fault=='partial':
        assert value['actual_counts'] is None and value['known_actual_counts']==r.c.expected_counts(4096)
    if fault=='spawn_race':assert None in execution.cleaned


def test_legal_short_zero_case_does_not_prevent_random_arm_evaluation(execution):
    execution.fault='natural_short'
    value=r.execute(**execution.kwargs)
    assert value['status']=='COMPLETE' and 'formal_random' in execution.stages
    assert value['original_36_pass']=={'zero':False,'random':False}
    assert value['actual_counts']==r.c.expected_counts(567488-37)


def test_low_start_memory_consumes_no_attempt(execution):
    execution.memory=(6 << 30)-1
    result=r.execute(**execution.kwargs)
    assert result['status']=='NOT_READY' and not execution.stages


def test_attempt_directory_is_never_reused(execution):
    r.execute(**execution.kwargs)
    with pytest.raises(ValueError,match='no retry'):r.execute(**execution.kwargs)


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


@pytest.mark.parametrize('field', ['num_timesteps', 'learning_rate', 'n_envs', 'aux_coef', 'target_kl'])
def test_export_cannot_accept_old_final_or_wrong_lr(field):
    model = SimpleNamespace(**{key: r.c.HYPERPARAMETERS[key] for key in
        ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef',
         'max_grad_norm', 'learning_rate')}, num_timesteps=r.c.INITIAL+r.c.TRANSITIONS,
        aux_coef=0., seed=181, n_envs=1, device='cpu', get_env=lambda: None,
        clip_range=lambda _: .2, normalize_advantage=True, clip_range_vf=None, target_kl=.005, use_sde=False)
    r.check_model(model, {'versions': {'synthetic': True}}, {'synthetic': True})
    setattr(model, field, {'num_timesteps':2097152, 'learning_rate':3e-4, 'n_envs':4, 'aux_coef':1., 'target_kl':None}[field])
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
        with pytest.raises(ValueError, match='CPP25'):
            r.verify_numeric('cpp', child, directory, old_release=release)
    else:
        r.verify_numeric('cpp', child, directory, old_release=release)


@pytest.mark.parametrize('winner',[None,'zero','random','tie'])
def test_selection_needs_both_gates_and_only_final_models(winner):
    formals={a:{'all_36_reference_pass':True,'cases':[{'J': .2}]*36} for a in r.c.ARMS}
    holdouts={a:{'all_36_reference_pass':True} for a in r.c.ARMS}
    if winner is None:holdouts['zero']['all_36_reference_pass']=holdouts['random']['all_36_reference_pass']=False
    elif winner!='tie':formals[winner]['cases']=[{'J':.1}]*36
    result=r.select_candidate(formals,holdouts)
    assert result['selected_arm']==('zero' if winner=='tie' else winner)
    assert result['automatic_model_replacement'] is False


def test_two_arm_initial_states_must_match_not_only_checkpoint_labels(monkeypatch):
    bundles={a:{'protocol':{'initial_state':{'policy':'a','optimizer':'b'},'versions':{'fake':'1'}}} for a in r.c.ARMS}
    monkeypatch.setattr(r,'completed_training',lambda plan,path,arm:bundles[arm])
    assert r.paired_initials({},Path('unused'))=={'policy':'a','optimizer':'b'}
    bundles['random']['protocol']['initial_state']['optimizer']='wrong'
    with pytest.raises(ValueError,match='same original'):r.paired_initials({},Path('unused'))


def test_stop_after_fact_commit_cannot_double_count(execution,monkeypatch):
    original=r.prior.artifacts
    def stop(directory):
        value=original(directory)
        if directory.name=='train':signal.raise_signal(signal.SIGTERM)
        return value
    monkeypatch.setattr(r.prior,'artifacts',stop)
    result=r.execute(**execution.kwargs)
    assert result['status']=='INCOMPLETE' and result['known_actual_counts']==r.c.expected_counts(262144)


def test_cli_default_is_plan_only(capsys):
    assert r.main([])==0 and 'PLAN_ONLY' in capsys.readouterr().out


@pytest.mark.parametrize('foreign', [False, True])
def test_nontrain_receipts_need_actual_expected_arm_final(tmp_path, monkeypatch, foreign):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    plan = {'output_root': 'new25', 'source_identity': {'bound': True}}
    path = tmp_path/'plan.json'
    r.c.atomic_json(path, plan)
    directory = tmp_path/'new25/zero/export'
    directory.mkdir(parents=True)
    bundle = {'identity': {'role': 'zero-final', 'training_identity': {'checkpoint_sha256': 'a'*64}}}
    protocol = dict(r.identity(plan, path, 'export_zero'), max_wall_s=180, budget_max=r.ZERO,
                    completed_training={'role': 'foreign'} if foreign else bundle['identity'], checkpoint_sha256='a'*64)
    r.c.atomic_json(directory/'protocol.json', protocol)
    child = dict(protocol, status='COMPLETE', error=None, cleanup_complete=True, inputs_unchanged=True,
                 protocol_sha256=r.c.sha(directory/'protocol.json'), actual_counts=r.ZERO)
    if foreign:
        monkeypatch.setattr(r, 'completed_training', lambda *args: bundle)
        with pytest.raises(ValueError, match='model differs'):r.verify_stage(plan, path, 'export_zero', directory, child)
    else:
        with pytest.raises(FileNotFoundError):r.verify_stage(plan, path, 'export_zero', directory, child)


def test_partial_exact_zero_cannot_overwrite_a_known_physical_prefix(execution):
    execution.fault = 'partial'
    r.execute(**execution.kwargs)
    path = execution.kwargs['plan_path']
    plan = execution.kwargs['loader'](path)
    directory = r.c.stage_path(plan, 'train_zero')
    child = r.c.read_json(directory/'result.json')
    child['actual_counts'] = dict(r.ZERO)
    with pytest.raises(ValueError, match='exact/known'):
        r.partial_counts(plan, path, 'train_zero', directory, child)


def test_fair_comparison_keeps_short_tasks_but_excludes_them_from_common_full():
    rows = [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed,
             'J': .2, 'reference_thresholds_met': True, 'policy_transitions': 600}
            for name, delay, seed in r.prior.matrix('formal')]
    formals = {'zero': {'cases': copy.deepcopy(rows)}, 'random': {'cases': copy.deepcopy(rows)}}
    formals['random']['cases'][0].update(J=.001, reference_thresholds_met=False, policy_transitions=17)
    result = r.compare_final_arms(formals)
    assert result['common_full_pairs'] == 35
    assert result['reference_pass_counts'] == {'zero':36, 'random':35}
    assert result['common_full_mean_J']['zero'] == result['common_full_mean_J']['random']


@pytest.fixture
def complete_single_training(tmp_path, monkeypatch):
    """Exercise the real parent completion reader with coherent fake artifacts."""
    from tools25.low_memory_runtime25 import initial_identity_valid25
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    initial = {**r.c.BASE_INITIAL_STATE,
        'optimizer_state_sha256_before_lr': r.c.BASE_INITIAL_STATE['optimizer_state_sha256'],
        'optimizer_state_sha256_after_lr':'c'*64,
        'optimizer_without_lr_sha256_before':'d'*64,'optimizer_without_lr_sha256_after':'d'*64,
        'optimizer_momentum_preserved':True,'learning_rate_before':3e-4,'learning_rate_after':1e-4,
        'lr_schedule_endpoints':[1e-4,1e-4], 'target_kl_before':None,'target_kl_after':.005,
        'target_kl_changes_optimizer_state':False,'optimizer_state_sha256_before_target_kl':'c'*64,
        'optimizer_state_sha256_after_target_kl':'c'*64,
        'rollout_layout':{'backend':'DummyVecEnv','envs':1,'n_steps':4096,'rollout_transitions':4096,
            'buffer_class':'DictRolloutBuffer','actor_shape':[4096,1,627],'velocity_target_shape':[4096,1,3],
            'actions_shape':[4096,1,16],'last_obs_cleared':True,'interprocess_rpc':False,'environment_subprocesses':0}}
    assert initial_identity_valid25(initial)
    plan = {'output_root':r.c.OUTPUT,'source_identity':{'synthetic':True},
        'base_initial_state':r.c.BASE_INITIAL_STATE,'base_checkpoint':{'sha256':'a'*64},
        'base_protocol':{'sha256':'b'*64},'vector_layout':r.c.VECTOR_LAYOUT,
        'runtime_dependencies_sha256':{'base.py':'f'*64}}
    path = tmp_path/'plan.json'
    r.c.atomic_json(path,plan)
    directory = tmp_path/r.c.OUTPUT/'zero/train'
    directory.mkdir(parents=True)
    versions = dict.fromkeys(('python','torch','sb3','numpy','gymnasium','mujoco'),'fake')
    versions['sb3'] = '2.7.0'
    common = {'stage':'train_zero','arm':'zero','mode':'zero','seed':181,'envs':1,
        'plan_sha256':r.c.sha(path),'source_identity':plan['source_identity'],
        'hyperparameters':r.c.HYPERPARAMETERS,'versions':versions,'initial_state':initial,
        'vector_layout':r.c.VECTOR_LAYOUT,'runtime_dependencies_sha256':plan['runtime_dependencies_sha256'],
        'checkpoint_selection':'final_only'}
    protocol = dict(common,format='d1-delay-comparison25-training-protocol-v1',
        base_checkpoint=plan['base_checkpoint'],base_protocol=plan['base_protocol'],
        planned_new_transitions=r.c.TRANSITIONS,planned_native_steps=r.c.TRANSITIONS*10,
        warm_start=True,reset_num_timesteps=False,teacher_policy_used=False)
    r.c.atomic_json(directory/'protocol.json',protocol)
    (directory/'policy.zip').write_bytes(b'completed single-env fake checkpoint')
    counts = dict(r.c.expected_counts(r.c.TRANSITIONS),resets=1,forward_calls=r.c.TRANSITIONS+3)
    result = dict(common,format='d1-delay-comparison25-training-result-v1',status='COMPLETE',error=None,
        eligible_for_evaluation=True,cleanup_complete=True,inputs_unchanged=True,aux_coef=0.,
        initial_num_timesteps=r.c.INITIAL,new_trained_transitions=r.c.TRANSITIONS,
        final_num_timesteps=r.c.INITIAL+r.c.TRANSITIONS,
        protocol_sha256=r.c.sha(directory/'protocol.json'),checkpoint_sha256=r.c.sha(directory/'policy.zip'),
        actual_counts=counts,known_actual_counts=counts,worker_counts=[counts],per_worker_counts=[counts],
        delay_state18=[{'delay_counts':{'0':1,'1':0,'2':0,'3':0},
                        'last_selection':{'rank':0,'mode':'zero','selected_delay_ticks':0}}],
        single_process_guard25={'backend':'DummyVecEnv','interprocess_rpc':False,'environment_subprocesses':0,
            'closed':True,'dirty':False,'waiting':False,'inflight':None,'construction_complete':True,
            'metadata_attempts':66,'metadata_returns':66,'metadata_validated':66})
    monkeypatch.setattr(r.c,'base_inputs',lambda: ({},{'config':'fake'}, {'versions':versions}))
    return plan,path,directory,protocol,result


@pytest.mark.parametrize('mutation',[None,'checkpoint','arm','four_workers','counts','layout','dependencies',
                                    'dirty','pending','unclosed','subprocess','unvalidated','versions','delay'])
def test_real_single_completion_reader_binds_every_training_seam(complete_single_training,mutation):
    plan,path,directory,_protocol,result = complete_single_training
    if mutation == 'checkpoint':result['checkpoint_sha256'] = 'e'*64
    elif mutation == 'arm':result['arm'] = 'random'
    elif mutation == 'four_workers':result['worker_counts'] = result['per_worker_counts'] = result['worker_counts']*4
    elif mutation == 'counts':result['actual_counts'] = dict(result['actual_counts'],resets=True)
    elif mutation == 'layout':result['initial_state']['rollout_layout']['n_steps'] = 1024
    elif mutation == 'dependencies':result['runtime_dependencies_sha256'] = {}
    elif mutation == 'dirty':result['single_process_guard25']['dirty'] = True
    elif mutation == 'pending':result['single_process_guard25']['waiting'] = True
    elif mutation == 'unclosed':result['single_process_guard25']['closed'] = False
    elif mutation == 'subprocess':result['single_process_guard25']['environment_subprocesses'] = True
    elif mutation == 'unvalidated':result['single_process_guard25']['metadata_validated'] = 65
    elif mutation == 'versions':result['versions'] = dict(result['versions'],torch='foreign')
    elif mutation == 'delay':result['delay_state18'][0]['last_selection']['mode'] = 'uniform_0_3'
    r.c.atomic_json(directory/'result.json',result)
    if mutation:
        with pytest.raises(ValueError):r.completed_training(plan,path,'zero')
    else:
        bundle = r.completed_training(plan,path,'zero')
        assert bundle['checkpoint'] == directory/'policy.zip'
        assert bundle['identity']['training_identity']['checkpoint_sha256'] == result['checkpoint_sha256']


def test_closed_stage_can_wait_for_rising_admission_without_claiming_or_respawning(execution):
    def memory():
        value = (8 << 30) if not execution.stages else ((4 << 30) if execution.now < 5 else (5.84*(1 << 30)))
        if execution.now >= 10:
            value = 6 << 30
        return {'effective_available_bytes':int(value)}
    observed = []
    def sleep(seconds):
        journal = r.c.read_json(r.ROOT/r.c.OUTPUT/'report.json')
        observed.append((seconds, list(journal['stages']), journal['admission_waits'][-1]['status']))
        execution.now += seconds
    execution.kwargs.update(memory_reader=memory,sleeper=sleep)
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'COMPLETE'
    assert execution.stages == list(r.c.STAGE_ORDER[:-2])
    assert observed == [(5.,['train_zero'],'WAITING_FOR_ADMISSION')]*2
    wait = value['admission_waits'][0]
    assert wait['stage'] == 'train_random' and wait['status'] == 'ADMITTED' and wait['duration_s'] == 10
    assert [row['effective_available_bytes'] for row in wait['samples']] == [4 << 30,int(5.84*(1 << 30)),6 << 30]


@pytest.mark.parametrize('fault',['timeout','below4','unknown','signal','global_wall','source_changed','late_source_check'])
def test_admission_wait_stops_without_starting_or_retrying_next_stage(execution,monkeypatch,fault):
    def memory():
        value = 8 << 30 if not execution.stages else int(5.84*(1 << 30))
        if execution.now >= 5:
            if fault == 'below4':value = (4 << 30)-1
            elif fault == 'unknown':value = None
            elif fault in ('source_changed','late_source_check'):value = 6 << 30
        return {'effective_available_bytes':value}
    execution.kwargs['memory_reader'] = memory
    if fault == 'signal':
        execution.kwargs['sleeper'] = lambda seconds:signal.raise_signal(signal.SIGTERM)
    elif fault == 'global_wall':
        original = r.c.stage_path
        def late(plan,key):
            if key == 'train_random':execution.now = 5399.
            return original(plan,key)
        monkeypatch.setattr(r.c,'stage_path',late)
    elif fault in ('source_changed','late_source_check'):
        original = execution.kwargs['loader']
        def changed(path):
            plan = original(path)
            if execution.now >= 5:
                if fault == 'late_source_check':execution.now = 5401.
                else:plan['source_identity'] = {'changed':True}
            return plan
        execution.kwargs['loader'] = changed
    value = r.execute(**execution.kwargs)
    assert value['status'] == 'INCOMPLETE' and execution.stages == ['train_zero']
    assert set(value['stages']) == {'train_zero'} and value['stages']['train_zero']['status'] == 'COMPLETE'
    assert value['known_actual_counts'] == value['actual_counts'] == r.c.expected_counts(262144)
    assert value['admission_waits'] and value['admission_waits'][0]['stage'] == 'train_random'
    wait = value['admission_waits'][0]
    if fault in ('source_changed','late_source_check'):
        assert wait['status'] == 'ADMITTED'
    else:
        assert wait['status'] == 'STOPPED'
    if fault == 'timeout':
        assert wait['duration_s'] == 300 and 'wait exhausted' in value['error']
    if fault == 'global_wall':
        assert value['elapsed_s'] == 5400 and wait['duration_s'] == 1
