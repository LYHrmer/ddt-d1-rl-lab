"""Only synthetic files/processes. No checkpoint load, ORT, environment or physics."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools21 import closeout_completed18 as c


class Clock:
    now = 0.
    def __call__(self):
        return self.now


@pytest.fixture
def old(tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    plan = {'output_root': 'old18', 'source_identity': {'sha': 'synthetic'}}
    plan_path = root/c.PLAN18
    plan_path.parent.mkdir(parents=True)
    c.atomic_json(plan_path, plan)
    run = root/'old18'
    run.mkdir()
    outer = {'status': 'STOPPED', 'cleanup_complete': True, 'plan_sha256': c.sha(plan_path),
             'source_identity': plan['source_identity'], 'stages': {'train_random': {'status': 'INCOMPLETE'}},
             'actual_counts': None}
    initial = {'optimizer_restored': True, 'policy_state_sha256': 'a'*64, 'optimizer_state_sha256': 'b'*64}
    for stage, n, rows in (('preflight_zero',32768,8),('train_zero',1048576,256)):
        directory = run/stage
        directory.mkdir()
        protocol = {'stage': stage, 'mode': 'zero', 'plan_sha256': c.sha(plan_path), 'source_identity': plan['source_identity']}
        c.atomic_json(directory/'protocol.json', protocol)
        counts = c.expected_counts(n//4)
        workers = [dict(counts, resets=1, forward_calls=2) for _ in range(4)]
        result = dict(protocol, status='COMPLETE', error=None, cleanup_complete=True, inputs_unchanged=True,
                      new_trained_transitions=n, protocol_sha256=c.sha(directory/'protocol.json'),
                      initial_state=initial, worker_counts=workers, per_worker_counts=workers,
                      actual_counts={key: sum(row[key] for row in workers) for key in workers[0]})
        c.atomic_json(directory/'result.json', result)
        (directory/'ledger.jsonl').write_text(''.join(json.dumps({'rollout': i+1, 'deterministic': {'steps':4096*(i+1)}})+'\n'
                                                     for i in range(rows)))
        (directory/'policy.zip').write_bytes(b'not a real model')
        outer['stages'][stage] = {'status':'COMPLETE', 'cleanup_complete':True, 'exit_code':0,
            'actual_counts':c.expected_counts(n), 'receipt_sha256':c.sha(directory/'result.json'),
            'artifacts_sha256':c.artifacts(directory)}
    (run/'train_random').mkdir()
    (run/'train_random/partial_policy.zip').write_bytes(b'NEVER READ OR LOAD')
    c.atomic_json(run/'report.json', outer)
    def training_reader(api, value, plan_sha, arm):
        assert arm == 'zero' and value == plan and plan_sha == c.sha(plan_path)
        path = run/'train_zero'
        return path/'policy.zip', c.read_json(path/'protocol.json'), c.read_json(path/'result.json'), {
            'checkpoint_sha256':c.sha(path/'policy.zip')}
    api = SimpleNamespace(load_plan=lambda path:plan, verify_inputs=lambda value:({'sources_sha256':{}},))
    return SimpleNamespace(root=root, run=run, plan=plan, api=api, reader=training_reader)


def load_old(old):
    return c.load_completed_zero(old.root, api=old.api, training_reader=old.reader)


def reseal(old, stage):
    path = old.run/'report.json'
    outer = c.read_json(path)
    directory = old.run/stage
    outer['stages'][stage].update(receipt_sha256=c.sha(directory/'result.json'), artifacts_sha256=c.artifacts(directory))
    c.atomic_json(path, outer)


def test_stopped_parent_does_not_invalidate_independently_complete_zero(old, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != 'partial_policy.zip', 'random partial was read'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    bundle = load_old(old)
    assert bundle['checkpoint'] == old.run/'train_zero/policy.zip'
    assert bundle['identity']['first_eight_rollouts_equal'] is True
    assert bundle['identity']['random_partial_used'] is False
    assert bundle['identity']['delay_randomisation_contrast'] == 'INCOMPLETE'


@pytest.mark.parametrize('fault',['artifact','receipt','stage','cleanup','workers','initial','prefix','short_prefix'])
def test_incomplete_or_tampered_old_zero_is_rejected(old, fault):
    directory = old.run/'train_zero'
    if fault == 'artifact':
        (directory/'policy.zip').write_bytes(b'changed')
    elif fault in ('receipt','stage','cleanup','workers','initial'):
        result = c.read_json(directory/'result.json')
        if fault == 'receipt':
            result['new_trained_transitions'] = 100
        elif fault == 'stage':
            result['stage'] = 'train_random'
        elif fault == 'cleanup':
            result['cleanup_complete'] = False
        elif fault == 'workers':
            result['worker_counts'][0]['native_steps'] -= 1
        else:
            result['initial_state'] = {}
        c.atomic_json(directory/'result.json', result)
        reseal(old, 'train_zero')
    else:
        path = old.run/'preflight_zero/ledger.jsonl'
        rows = [json.loads(row) for row in path.read_text().splitlines()]
        if fault == 'prefix':
            rows[3]['deterministic']['steps'] += 1
        else:
            rows.pop()
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        reseal(old,'preflight_zero')
    with pytest.raises(ValueError):
        load_old(old)


@pytest.fixture
def execution(tmp_path):
    root = tmp_path/'repo'
    root.mkdir()
    state = SimpleNamespace(root=root, output=root/c.OUTPUT, stages=[], cleaned=[], memories=[],
                            clock=Clock(), fault=None, formal_pass=True, holdout_pass=True,
                            sources={'synthetic':'new21'}, bundle={'identity':{'synthetic':'zero18'}})
    class Process:
        returncode = 0
        pid = 123456789
        def poll(self):
            return None if state.fault == 'timeout' else self.returncode
    def factory(argv, **kwargs):
        assert kwargs['start_new_session'] is True and kwargs['env']['D1_STAGE21_NONCE']
        stage = argv[argv.index('--stage')+1]
        state.stages.append(stage)
        if state.fault == 'spawn':
            raise InterruptedError('synthetic Popen before handle')
        if state.fault == 'spawn_signal':
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        directory = state.output/stage
        directory.mkdir()
        release_path = state.output/'release.json'
        release = c.read_json(release_path)
        identity = c.stage_identity(stage,release,release_path)
        protocol = dict(identity, budget_max=c.CAP if stage in ('formal','holdout') else c.ZERO,
                        scenarios=c.matrix(stage) if stage in ('formal','holdout') else [], max_wall_s=c.STAGES[stage])
        c.atomic_json(directory/'protocol.json',protocol)
        child = dict(identity,status='COMPLETE',error=None,cleanup_complete=True,inputs_unchanged=True,
                     new_training_transitions=0,actual_counts=c.CAP if stage in ('formal','holdout') else c.ZERO,
                     known_actual_counts=c.CAP if stage in ('formal','holdout') else c.ZERO,
                     protocol_sha256=c.sha(directory/'protocol.json'),
                     all_36_reference_pass=state.formal_pass if stage=='formal' else state.holdout_pass)
        if state.fault == 'foreign':
            child['completed_zero18'] = {'foreign':True}
        if state.fault == 'partial' and stage == 'formal':
            child.update(status='INCOMPLETE',error='interrupted interval',actual_counts={'policy_transitions':1,'native_steps':3,'pd_updates':2},
                         known_actual_counts={'policy_transitions':1,'native_steps':3,'pd_updates':2})
        if state.fault == 'intermediate' and stage == 'formal':
            child.update(status='INCOMPLETE',error=None,actual_counts=None,known_actual_counts=c.expected_counts(600))
        c.atomic_json(directory/'report.json',child)
        process = Process()
        if state.fault == 'child_failure' or state.fault in ('partial','intermediate') and stage == 'formal':
            process.returncode = 1
        if state.fault == 'late_exit':
            state.clock.now += c.STAGES[stage]+1
        return process
    def cleanup(process):
        assert all(signal.getsignal(sig)==signal.SIG_IGN or signal.getsignal(sig).__name__=='handler'
                   for sig in (signal.SIGINT,signal.SIGTERM))
        state.cleaned.append(process)
        if state.fault == 'cleanup_signal' and process is not None:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        if state.fault == 'late_cleanup':
            state.clock.now = 901.
        return {'remaining':{'owned':1} if state.fault=='cleanup' else {}}
    def memory():
        value = state.memories.pop(0) if state.memories else 8<<30
        if isinstance(value, BaseException):
            raise value
        return {'effective_available_bytes':value}
    def validator(directory,stage,child,release,path):
        c.check_stage_identity(child,stage,release,path)
        assert child['status']=='COMPLETE'
    state.kwargs = {'root':root, 'loader':lambda:state.bundle, 'source_loader':lambda:state.sources, 'memory_reader':memory,
                    'factory':factory, 'cleanup':cleanup, 'adopter':lambda:None, 'clock':state.clock,
                    'sleeper':lambda duration:setattr(state.clock,'now',state.clock.now+duration), 'validator':validator}
    return state


def test_default_plan_has_zero_model_physics_and_no_source_read(monkeypatch, capsys):
    monkeypatch.setattr(c,'load_completed_zero',lambda:pytest.fail('plan loaded old weights'))
    assert c.main([])==0
    value=json.loads(capsys.readouterr().out)
    assert value['status']=='PLAN_ONLY' and value['new_training_transitions']==value['physics_steps_executed']==0
    assert value['onnx_sessions_created']==0 and value['budget']==c.BUDGET


def test_cold_import_does_not_import_model_or_physics_library():
    code='from tools21 import closeout_completed18;import sys;assert not {"torch","mujoco","onnxruntime","stable_baselines3"}&sys.modules.keys()'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10,check=False)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('qualified',[True,False])
def test_fixed_order_holdout_conditional_and_complete_budget(execution,qualified):
    execution.formal_pass=qualified
    report=c.execute(**execution.kwargs)
    assert report['status']=='COMPLETE' and report['cleanup_complete']
    assert execution.stages==(['export','cpp','formal','holdout'] if qualified else ['export','cpp','formal'])
    assert report['actual_counts']==(c.BUDGET if qualified else c.CAP)
    assert report['full_delay_qualification'] is qualified
    if not qualified:
        assert report['stages']['holdout']['status']=='SKIPPED_NO_QUALIFIED_CANDIDATE'


@pytest.mark.parametrize('value',[6*(1<<30)-1,4<<30])
def test_admission_below_six_does_not_claim(execution,value):
    execution.memories=[value]
    result=c.execute(**execution.kwargs)
    assert result['status']=='NOT_READY' and not execution.output.exists() and not execution.stages


@pytest.mark.parametrize('value',[4*(1<<30)-1,OSError('memory unknown')])
def test_runtime_four_gib_or_unknown_stops_and_cleans(execution,value):
    execution.memories=[8<<30,8<<30,value]
    result=c.execute(**execution.kwargs)
    assert result['status']=='INCOMPLETE' and execution.stages==['export'] and execution.cleaned
    assert result['actual_counts']==c.ZERO  # the bound export receipt already measured zero physics


@pytest.mark.parametrize('fault',['spawn','spawn_signal','cleanup_signal','timeout','cleanup','late_exit','late_cleanup','child_failure','foreign'])
def test_lifecycle_faults_cannot_become_complete_or_retry(execution,fault):
    prior={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    execution.fault=fault
    result=c.execute(**execution.kwargs)
    assert result['status']=='INCOMPLETE' and execution.stages==['export']
    assert execution.cleaned and (fault!='spawn' or execution.cleaned[0] is None)
    assert prior=={sig:signal.getsignal(sig) for sig in prior}
    with pytest.raises(ValueError,match='no retry'):
        c.execute(**execution.kwargs)


def test_mid_native_interval_preserves_exact_known_total(execution):
    execution.fault='partial'
    report=c.execute(**execution.kwargs)
    assert execution.stages==['export','cpp','formal'] and report['status']=='INCOMPLETE'
    assert report['known_actual_counts']=={'policy_transitions':1,'native_steps':3,'pd_updates':2}
    assert report['actual_counts']==report['known_actual_counts'] and not report['full_delay_qualification']


def test_intermediate_receipt_preserves_completed_case_but_total_stays_unknown(execution):
    execution.fault='intermediate'
    report=c.execute(**execution.kwargs)
    assert report['status']=='INCOMPLETE' and execution.stages==['export','cpp','formal']
    assert report['known_actual_counts']==c.expected_counts(600) and report['actual_counts'] is None


def test_report_interrupt_after_formal_completion_cannot_double_count(execution,monkeypatch):
    write=c.atomic_json
    injected=[False]
    def interrupt(path,value):
        if Path(path)==execution.output/'report.json' and not injected[0] and value['stages'].get('formal',{}).get('status')=='COMPLETE':
            injected[0]=True
            raise OSError('synthetic report interrupted after aggregate')
        return write(path,value)
    monkeypatch.setattr(c,'atomic_json',interrupt)
    report=c.execute(**execution.kwargs)
    assert injected==[True] and report['status']=='INCOMPLETE'
    assert report['known_actual_counts']==report['actual_counts']==c.CAP
    assert execution.stages==['export','cpp','formal']


def test_exact_four_gib_runtime_is_allowed_after_six_gib_admission(execution):
    execution.memories=[8<<30,8<<30,4<<30]
    assert c.execute(**execution.kwargs)['status']=='COMPLETE'


@pytest.mark.parametrize('bad',[True,-1,1.5])
def test_real_five_key_counter_payload_rejects_invalid_noncore_values(bad):
    counts=dict(c.expected_counts(600),resets=1,forward_calls=601)
    assert c.valid_counts(counts)
    counts['resets']=bad
    assert not c.valid_counts(counts)


def test_stage_export_failure_preserves_exclusive_receipt_without_physics(tmp_path):
    release_path=tmp_path/'release.json'
    release={'format':'d1-completed-zero21-release-v1','budget':c.BUDGET,'limits':c.LIMITS,'stage_limits_s':c.STAGES,
             'output_root':c.OUTPUT,'new_training_transitions':0,'retry_or_resume_allowed':False,
             'source_identity':{},'completed_zero18':{'zero':'complete'}}
    c.atomic_json(release_path,release)
    def fail(*args):
        raise ValueError('synthetic numeric failure')
    result=c.run_stage(release_path,'export',tmp_path/'export',loader=lambda:{'identity':release['completed_zero18']},
        source_checker=lambda proof:None,launch_validator=lambda *args:None,
        memory_reader=lambda:{'effective_available_bytes':8<<30},numeric=fail)
    assert result['status']=='INCOMPLETE' and result['actual_counts']==c.ZERO and result['cleanup_complete']
    assert (tmp_path/'export/report.json').is_file()


def test_source_gate_requires_committed_exact_files(tmp_path):
    proof={'git_commit':'a'*40,'sources_sha256':{name:'b'*64 for name in c.SOURCES}}
    with pytest.raises((OSError,ValueError)):
        c.check_sources(proof,root=tmp_path)


@pytest.mark.parametrize('fault',[None,'short','partial','construct','close','oracle'])
def test_direct_saved_evaluation_keeps_full_matrix_or_real_partial_counts(tmp_path,monkeypatch,fault):
    output=tmp_path/'formal'
    output.mkdir()
    c.atomic_json(output/'protocol.json',{})
    report={'cases':[],'cleanup_complete':True,'actual_counts':None}
    facts={'envs':[],'predictions':0}
    original_write=c.atomic_json
    intermediate=[]
    def write(path,value):
        if Path(path)==output/'report.json' and len(value['cases'])==1:
            intermediate.append((value['actual_counts'],dict(value['known_actual_counts'])))
        return original_write(path,value)
    monkeypatch.setattr(c,'atomic_json',write)
    class Env:
        def __init__(self):
            self.counts=dict(c.ZERO)
            facts['envs'].append(self)
        def step(self,action):
            assert action=='actual predictor output'
            self.counts=c.expected_counts(self.counts['policy_transitions']+1)
            if fault=='partial':
                self.counts.update(native_steps=3,pd_updates=2)
                raise OSError('interrupted native interval')
        def counters(self):
            return self.counts
        def close(self):
            if fault=='close':
                raise OSError('close failed')
    def make(*args):
        if fault=='construct':
            raise OSError('constructor failed')
        return Env()
    def predict(obs):
        facts['predictions']+=1
        if fault=='oracle':
            raise ValueError('predict failed before physical step')
        return 'actual predictor output'
    def episode(env,predict,name,delay,seed,path,remaining):
        n=3 if fault=='short' and len(facts['envs'])==1 else 600
        for tick in range(n):
            env.step(predict(('observation',tick)))
        path.write_bytes(b'synthetic trace')
        return {'sequence':name,'delay_ticks':delay,'reset_seed':seed,'actual_counts':env.counters(),
                'reference_thresholds_met':n==600,'failure':n<600,
                'done_reason':'HORIZON_12S' if n==600 else 'TERMINATED'}
    rt=SimpleNamespace(predict=predict,make_env=make,backend={'synthetic':True})
    helper=SimpleNamespace(make_scenarios=lambda phase:c.matrix('formal'),THRESHOLDS={'fixed':True},
                           SEQUENCES={name:[] for name,_,_ in c.matrix('formal')},run_episode=episode)
    bundle={'old_release':{},'protocol':{},'identity':{'training_identity':{'checkpoint_sha256':'a'*64}}}
    args=(('formal',bundle,tmp_path,output,report,lambda *args:None))
    kwargs={'runtime_factory':lambda *args:rt,'helpers':helper,'verifier':lambda *args:None}
    if fault in ('partial','construct','close','oracle'):
        with pytest.raises((OSError,ValueError)):
            c.evaluate(*args,**kwargs)
        assert report['actual_counts']==({'policy_transitions':1,'native_steps':3,'pd_updates':2} if fault=='partial'
            else None if fault=='construct' else c.expected_counts(600) if fault=='close' else c.ZERO)
        assert report['cleanup_complete'] is (fault not in ('close','construct'))
    else:
        c.evaluate(*args,**kwargs)
        assert len(report['cases'])==len(facts['envs'])==36
        assert report['actual_counts']==c.expected_counts(21003 if fault=='short' else 21600)
        assert report['all_36_reference_pass'] is (fault is None)
        assert facts['predictions']==report['actual_counts']['policy_transitions']
        assert intermediate==[(None,c.expected_counts(3 if fault=='short' else 600))]


@pytest.mark.parametrize('bad',['identity','matrix','counts','flag','thresholds'])
def test_parent_evaluation_readback_rejects_forged_summary(tmp_path,bad):
    output=tmp_path/'formal'
    output.mkdir()
    release_path=tmp_path/'release.json'
    release={'source_identity':{},'completed_zero18':{'training_identity':{'checkpoint_sha256':'a'*64}}}
    c.atomic_json(release_path,release)
    identity=c.stage_identity('formal',release,release_path)
    helper=SimpleNamespace(THRESHOLDS={'fixed':True},SEQUENCES={'fixed':'commands'})
    protocol=dict(identity,budget_max=c.CAP,scenarios=c.matrix('formal'),max_wall_s=300,backend={'synthetic':True},
                  thresholds=helper.THRESHOLDS,sequences=helper.SEQUENCES,checkpoint_sha256='a'*64)
    rows=[{'sequence':name,'delay_ticks':delay,'reset_seed':seed,'actual_counts':c.expected_counts(600),
           'reference_thresholds_met':True,'failure':False,'done_reason':'HORIZON_12S'} for name,delay,seed in c.matrix('formal')]
    child=dict(identity,status='COMPLETE',error=None,cleanup_complete=True,inputs_unchanged=True,new_training_transitions=0,
               actual_counts=c.CAP,cases=rows,all_36_reference_pass=True,backend=protocol['backend'])
    if bad=='identity':
        child['completed_zero18']={}
    elif bad=='matrix':
        rows[1]['reset_seed']=2001
    elif bad=='counts':
        rows[0]['actual_counts']=c.expected_counts(599)
    elif bad=='flag':
        rows[0]['reference_thresholds_met']=False
    else:
        protocol['thresholds']={'relaxed':True}
    c.atomic_json(output/'protocol.json',protocol)
    child['protocol_sha256']=c.sha(output/'protocol.json')
    with pytest.raises(ValueError):
        c.verify_stage(output,'formal',child,release,release_path,helpers=helper,verifier=lambda *args:None)
