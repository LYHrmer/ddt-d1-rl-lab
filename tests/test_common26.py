"""26 source, selection, nonce and completed-model gates with fake artifacts."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools26 import common26 as c


def formal(arm,passed=True,j=1.):
    rows=[{'sequence':s,'delay_ticks':d,'reset_seed':seed,'J':j,'reference_thresholds_met':passed,
           'failure':not passed,'done_reason':'HORIZON_12S','policy_transitions':600}
          for s in ('forward_stop','reverse_stop','turn_stop') for d in range(4) for seed in (1001,1002,1003)]
    return {'status':'COMPLETE','error':None,'cleanup_complete':True,'arm':arm,'cases':rows,'all_36_reference_pass':passed}


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    base={'config':'training/configs/pilot03d.json','base_checkpoint':{'path':'completed25.zip','sha256':'a'*64,'trained_transitions':c.INITIAL},
        'base_protocol':{'path':'completed25_protocol.json','sha256':'b'*64},
        'baseline_provenance':{'serialized_entries_sha256':dict(c.BASE_SERIALIZED),'updates':c.BASE_UPDATES},
        'runtime_dependencies_sha256':{'base_class.py':'c'*64}}
    proof={'git_commit':'d'*40,'sources_sha256':{'fake.py':'e'*64}}
    monkeypatch.setattr(c,'base_inputs',lambda:(copy.deepcopy(base),{},{}))
    monkeypatch.setattr(c,'source_identity',lambda:copy.deepcopy(proof))
    monkeypatch.setattr(c,'check_sources',lambda *a,**kw:None)
    plan={**c.fixed(base),'source_identity':copy.deepcopy(proof)}
    path=tmp_path/c.PLAN;path.parent.mkdir(parents=True);c.atomic_json(path,plan)
    return plan,path


def test_new_budget_same_mode_only_weight_difference_and_single_holdout(inputs):
    plan,_=inputs
    assert plan['budget']=={'policy_transitions':326944,'native_steps':3269440,'pd_updates':1634720}
    assert plan['envs']==1 and plan['seed']==182 and c.INITIAL==1310720 and c.TRANSITIONS==131072
    assert all(s['mode']=='uniform_0_3' for s in c.STAGES.values())
    assert c.ARMS=={'control':2.,'yaw3':3.}
    assert plan['selection']['maximum_holdout_arms']==1 and not plan['selection']['holdout_failure_fallback']
    assert plan['selection']['exact_tie']=='control' and not plan['selection']['uses_holdout_for_selection']
    assert plan['hyperparameters']['n_steps']==4096 and plan['hyperparameters']['target_kl']==.005
    assert plan['resources']['total_wall_s']==3600 and plan['resources']['reserve_bytes']==4<<30
    assert plan['base_checkpoint']['trained_transitions']==1310720 and c.BASE_UPDATES==1088
    assert not plan['partial_checkpoint_eligible'] and not plan['retry_or_resume_allowed']
    assert c.REFERENCE_REPORTS['random25']['sha256']=='692ec9d4ae778af487c340f76aaf8290242a4c049309b20a075ab843f1283562'


@pytest.mark.parametrize('mutation',['seed','envs','lr','weight','mode','cap','selection','extra','typed','source'])
def test_load_plan_rejects_unapproved_scope_or_source(inputs,monkeypatch,mutation):
    plan,path=inputs
    if mutation=='seed':plan['seed']=181
    elif mutation=='envs':plan['envs']=4
    elif mutation=='lr':plan['hyperparameters']['learning_rate']=3e-4
    elif mutation=='weight':plan['arms']['yaw3']=2.
    elif mutation=='mode':plan['stages']['train_yaw3']['mode']='zero'
    elif mutation=='cap':plan['budget']['policy_transitions']+=1
    elif mutation=='selection':plan['selection']['exact_tie']='yaw3'
    elif mutation=='extra':plan['resume']=True
    elif mutation=='typed':plan['resources']['poll_s']=True
    else:monkeypatch.setattr(c,'check_sources',lambda _:c.require(False,'source drift'))
    c.atomic_json(path,plan)
    with pytest.raises(ValueError):c.load_plan(path)


@pytest.mark.parametrize('control,yaw,js,winner',[
    (False,False,(1.,.1),None),(True,False,(1.,.1),'control'),
    (False,True,(.1,1.),'yaw3'),(True,True,(1.,1.),'control'),
    (True,True,(1.,.9),'yaw3'),(True,True,(.9,1.),'control')])
def test_selection_uses_eligible_developer36_only_and_exact_tie_control(control,yaw,js,winner):
    value=c.select_candidate({'control':formal('control',control,js[0]),'yaw3':formal('yaw3',yaw,js[1])})
    assert value['selected_arm']==winner and value['selection_finalized'] and not value['uses_holdout']


@pytest.mark.parametrize('fault',['missing','duplicate','seed','partial','nan','passflag','short','unknown_threshold'])
def test_selection_rejects_incomplete_matrix_or_does_not_qualify_bad_cases(fault):
    values={arm:formal(arm) for arm in c.ARMS}
    if fault=='missing':values['yaw3']['cases'].pop()
    elif fault=='duplicate':values['yaw3']['cases'][0]=copy.deepcopy(values['yaw3']['cases'][1])
    elif fault=='seed':values['yaw3']['cases'][0]['reset_seed']=2001
    elif fault=='partial':values['yaw3']['status']='INCOMPLETE'
    elif fault=='nan':values['yaw3']['cases'][0]['J']=float('nan')
    elif fault=='passflag':values['yaw3']['all_36_reference_pass']=False
    elif fault=='short':values['yaw3']['cases'][0]['policy_transitions']=599
    else:values['yaw3']['cases'][0]['reference_thresholds_met']=None
    with pytest.raises(ValueError):c.select_candidate(values)


@pytest.mark.parametrize('key',c.STAGE_ORDER)
def test_arm_stage_paths_and_commands_are_owned26(inputs,key):
    plan,path=inputs;arm,kind=c.split_stage(key);output=c.stage_path(plan,key)
    assert output==c.ROOT/c.OUTPUT/arm/kind
    argv=c.command(path,key,output)
    assert 'tools26' in argv[2] and argv[argv.index('--arm')+1]==arm
    assert ('--stage' in argv)==(kind!='train')


@pytest.fixture
def launch(inputs,monkeypatch):
    plan,path=inputs;monkeypatch.setenv('D1_STAGE26_NONCE','nonce26')
    def build(key):
        output=c.stage_path(plan,key)
        records={k:{'status':'COMPLETE'} for k in c.STAGE_ORDER[:c.STAGE_ORDER.index(key)]}
        records[key]={'status':'STARTED','nonce':'nonce26','command':c.command(path,key,output)}
        journal={'format':c.EXECUTION_FORMAT,'status':'RUNNING','active_stage':key,'supervisor_pid':c.os.getppid(),
            'plan_sha256':c.sha(path),'source_identity':plan['source_identity'],'stages':records}
        return output,journal
    return plan,path,build


@pytest.mark.parametrize('fault',[None,'pid','nonce','argv','sha','source','prior','cross_arm'])
def test_parent_nonce_and_fixed_stage_are_checked_before_any_child(launch,fault):
    plan,path,build=launch;output,journal=build('train_yaw3')
    if fault=='pid':journal['supervisor_pid']+=1
    elif fault=='nonce':journal['stages']['train_yaw3']['nonce']='stale'
    elif fault=='argv':journal['stages']['train_yaw3']['command'].append('--resume')
    elif fault=='sha':journal['plan_sha256']='f'*64
    elif fault=='source':journal['source_identity']={}
    elif fault=='prior':journal['stages']['train_control']['status']='INCOMPLETE'
    elif fault=='cross_arm':output=c.stage_path(plan,'train_control')
    target=c.ROOT/c.OUTPUT/'report.json';target.parent.mkdir(parents=True);c.atomic_json(target,journal)
    if fault:
        with pytest.raises(ValueError):c.validate_launch(plan,path,'train_yaw3',output)
    else:c.validate_launch(plan,path,'train_yaw3',output)


@pytest.mark.parametrize('fault',[None,'wrong_selection','second_holdout','foreign_winner','badskip'])
def test_only_one_selected_winner_can_ever_launch_holdout(launch,fault):
    plan,path,build=launch;output,journal=build('holdout_yaw3')
    values={'control':formal('control',False),'yaw3':formal('yaw3',True)}
    for arm,value in values.items():
        directory=c.stage_path(plan,'formal_'+arm);directory.mkdir(parents=True,exist_ok=True);c.atomic_json(directory/'report.json',value)
    journal['selection']=c.select_candidate(values)
    journal['stages']['holdout_control']={'status':'SKIPPED_NOT_SELECTED'}
    if fault=='wrong_selection':journal['selection']['selected_arm']='control'
    elif fault=='second_holdout':journal['stages']['holdout_control']['status']='INCOMPLETE'
    elif fault=='foreign_winner':
        values['control']=formal('control',True,.1)
        c.atomic_json(c.stage_path(plan,'formal_control')/'report.json',values['control'])
    elif fault=='badskip':journal['stages']['holdout_control']['status']='SKIPPED_NO_QUALIFIED_CANDIDATE'
    c.atomic_json(c.ROOT/c.OUTPUT/'report.json',journal)
    if fault:
        with pytest.raises(ValueError):c.validate_launch(plan,path,'holdout_yaw3',output)
    else:c.validate_launch(plan,path,'holdout_yaw3',output)


def test_prepare_is_new_only_and_default_cli_remains_cold(inputs,monkeypatch,capsys):
    plan,path=inputs;path.unlink();assert c.prepare(path)==plan
    with pytest.raises(ValueError):c.prepare(path)
    monkeypatch.setattr(c,'base_inputs',lambda:pytest.fail('default read artifacts'))
    monkeypatch.setattr(c,'prepare',lambda:pytest.fail('default prepared files'))
    assert c.main([])==0 and json.loads(capsys.readouterr().out)['status']=='PLAN_ONLY'


def test_source_proof_checks_committed_bytes_and_exact_closure(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'SOURCES',('one.py',));p=tmp_path/'one.py';p.write_text('one=1')
    proof={'git_commit':'a'*40,'sources_sha256':{'one.py':c.sha(p)}}
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:p.read_bytes());c.check_sources(proof,tmp_path)
    monkeypatch.setattr(c.subprocess,'check_output',lambda *a,**kw:b'changed')
    with pytest.raises(ValueError,match='source first'):c.check_sources(proof,tmp_path)


def test_actual25_anchor_is_checked_before_any_framework_load(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);p=tmp_path/c.BASE_ROOT/'plan_01.json';p.parent.mkdir(parents=True);p.write_text('{}')
    with pytest.raises(ValueError,match='baseline changed'):c.base_inputs(tmp_path)


@pytest.fixture
def completed(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('common26_pure_train_fixture',ROOT/'tests/test_train_yaw_reward26.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    root,path,plan=fixture.case.__wrapped__(tmp_path)
    plan['output_root']=c.OUTPUT;c.atomic_json(path,plan)
    result,_runtime,directory=fixture.run_fake((root,path,plan))
    assert result['status']=='COMPLETE'
    monkeypatch.setattr(c,'ROOT',root)
    monkeypatch.setattr(c,'base_inputs',lambda:({}, {'sources_sha256':{}}, {'versions':fixture.VERSIONS}))
    return plan,path,directory,result


def test_completed_bundle_binds_actual26_final_checkpoint(completed):
    plan,path,directory,result=completed
    bundle=c.check_completed_training(plan,path,'control')
    assert bundle['checkpoint']==directory/'policy.zip' and bundle['result']==result
    assert bundle['identity']['role']=='completed_yaw_reward26_control_final_only'


@pytest.mark.parametrize('fault',['actual_bool','known_bool','partial','foreign_weight','initial','reward','first','checkpoint'])
def test_completed_gate_rejects_typed_partial_or_foreign_artifacts(completed,fault):
    plan,path,directory,result=completed
    if fault=='actual_bool':result['actual_counts']['resets']=True
    elif fault=='known_bool':result['known_actual_counts']['resets']=True
    elif fault=='partial':result['status']='PARTIAL'
    elif fault=='foreign_weight':result['yaw_tracking_weight']=3.
    elif fault=='initial':result['initial_state']['updates']=1024
    elif fault=='reward':result['reward_state26']['extra_reward_sum']=1.
    elif fault=='first':result['first_rollout_nonreward']['sha256']='0'*64
    else:(directory/'policy.zip').write_text('foreign')
    c.atomic_json(directory/'result.json',result)
    with pytest.raises(ValueError):c.check_completed_training(plan,path,'control')
