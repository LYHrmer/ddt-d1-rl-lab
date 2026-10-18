"""Independent26 final-only yaw reward pair; stdlib source/provenance gates."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools25 import common25 as previous

require,sha,atomic_json = previous.require,previous.sha,previous.atomic_json
read_json,read_memory,expected_counts = previous.read_json,previous.read_memory,previous.expected_counts
INITIAL,TRANSITIONS,SEED = 1310720,131072,182
OUTPUT,PLAN = 'evidence/yaw_reward26/execution_01','evidence/yaw_reward26/plan_01.json'
FORMAT,EXECUTION_FORMAT = 'd1-yaw-reward26-plan-v1','d1-yaw-reward26-execution-v1'
ARMS = {'control':2.0,'yaw3':3.0}
MODE = 'uniform_0_3'
HYPERPARAMETERS = copy.deepcopy(previous.HYPERPARAMETERS)
RESOURCES = {**previous.RESOURCES,'total_wall_s':3600}
STAGE_ORDER = ('train_control','train_yaw3','export_control','cpp_control','formal_control',
               'export_yaw3','cpp_yaw3','formal_yaw3','holdout_control','holdout_yaw3')
KINDS = {'train':(TRANSITIONS,1200),'export':(0,180),'cpp':(0,90),'formal':(21600,300),'holdout':(21600,300)}
STAGES = {key:{'arm':key.rsplit('_',1)[1],'kind':key.rsplit('_',1)[0],'mode':MODE,
    'yaw_tracking_weight':ARMS[key.rsplit('_',1)[1]],'transitions':KINDS[key.rsplit('_',1)[0]][0],
    'max_wall_s':KINDS[key.rsplit('_',1)[0]][1]} for key in STAGE_ORDER}
BUDGET = expected_counts(2*TRANSITIONS+3*21600)
SOURCES = tuple(dict.fromkeys((*previous.SOURCES,'tools25/run_delay_comparison25.py',
    'tools09/train_d1_ppo09.py','tools26/common26.py','tools26/runtime26.py','tools26/train_yaw_reward26.py','tools26/run_yaw_reward26.py',
    'tests/test_common26.py','tests/test_runtime26.py','tests/test_train_yaw_reward26.py','tests/test_run_yaw_reward26.py',
    'docs/yaw_reward_comparison26.md')))
VECTOR_LAYOUT = {k:v for k,v in previous.VECTOR_LAYOUT.items() if k != 'comparable_to24_as_single_variable'}
BASE_ROOT = 'evidence/low_memory_delay25'
BASE_ANCHORS = {'plan_01.json':'9bd618971362d52cbdbbdcc8a41284ca0172df8fdc4195eb2232b3805ab5632b',
 'execution_01/report.json':'c97c22c036dda63a9d30728fea448beba0dba2caf28c3e81bb76f32709ab4ced',
 'execution_01/random/train/result.json':'0ecb4bbd01a4ac0041f55b7ceab0df334a88d3a98fc3eaf0a21e60b4a0e6a4ad',
 'execution_01/random/train/protocol.json':'ea458bb2bf449c06279fcb0299284dcb5d2a345c959ab67c57831297933e81a2',
 'execution_01/random/train/policy.zip':'873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22'}
BASE_UPDATES = 1088
BASE_SERIALIZED = {'data':'2a39cab358f44e846d5959149d8c235e6cabb047efbb7404dd83c516e22b2906',
 'policy.pth':'94a7e39cf74b0a800863035485788ccc9568d558bfbd3f6b2c67ba040d42f9d2',
 'policy.optimizer.pth':'67738d6b2a4b2b14948fe2e0cbfb1d27609ac1f2f3543ded5af0e0b4b93d2111'}
REFERENCE_REPORTS = {
 'baseline17':{'path':'evidence/closeout17/execution_01/formal/report.json','sha256':'4c5a0852de9f883c04f8707ed147e2cfc999b659360050838b3f7307c8435972'},
 'random25':{'path':'evidence/low_memory_delay25/execution_01/random/formal/report.json','sha256':'692ec9d4ae778af487c340f76aaf8290242a4c049309b20a075ab843f1283562'}}
SELECTION = {'checkpoint':'final_only','candidates':list(ARMS),'eligibility':'own_original36_all_thresholds_and_full_horizons',
 'score':'eligible_original36_mean_J','direction':'lower','exact_tie':'control','maximum_holdout_arms':1,
 'holdout_reset_seeds':[2001,2002,2003],'holdout_failure_fallback':False,'uses_holdout_for_selection':False,
 'final_qualification':'selected_original36_and_selected_holdout36_all_pass','automatic_model_replacement':False}


def reward_change(arm):
    require(arm in ARMS,'unknown26 arm')
    return {'component':'wz_tracking','baseline_weight':2.0,'weight':ARMS[arm],
            'policy_dt_s':.02,'scope':'training_reward_only'}


def check_sources(proof,root=ROOT):
    root=Path(root).resolve()
    require(isinstance(proof,dict) and isinstance(proof.get('git_commit'),str)
            and re.fullmatch('[0-9a-f]{40}',proof['git_commit']) is not None,
            'source26 Git identity required')
    require(set(proof.get('sources_sha256',{})) == set(SOURCES),'source26 closure differs')
    for name,digest in proof['sources_sha256'].items():
        path=root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path)==digest,'source26 changed: '+name)
        data=subprocess.check_output(['git','show',proof['git_commit']+':'+name],cwd=root,stderr=subprocess.PIPE,timeout=10)
        require(hashlib.sha256(data).hexdigest()==digest,'commit26 source first: '+name)


def source_identity(root=ROOT):
    root=Path(root).resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()
    proof={'git_commit':commit,'sources_sha256':{name:sha(root/name) for name in SOURCES}}
    check_sources(proof,root)
    return proof


def base_inputs(root=ROOT):
    """Read only completed25 random; never use its partial or initial18 weights."""
    from tools25 import run_delay_comparison25 as owner
    root=Path(root).resolve()
    require(root==ROOT,'baseline26 uses only registered root')
    paths={name:root/BASE_ROOT/name for name in BASE_ANCHORS}
    for name,path in paths.items():
        require(not path.is_symlink() and sha(path)==BASE_ANCHORS[name],'complete25 baseline changed: '+name)
    old=previous.load_plan(paths['plan_01.json'])
    parent=read_json(paths['execution_01/report.json'])
    require(parent.get('status')=='COMPLETE' and parent.get('cleanup_complete') is True and parent.get('error') is None
            and parent.get('plan_sha256')==BASE_ANCHORS['plan_01.json'] and parent.get('source_identity')==old['source_identity'],
            'completed25 parent provenance differs')
    record=parent['stages']['train_random']
    require(record.get('status')=='COMPLETE' and record.get('cleanup_complete') is True,'completed25 random stage not closed')
    artifacts=record.get('artifacts_sha256',{})
    require(bool(artifacts),'completed25 random artifacts missing')
    for name,digest in artifacts.items():
        p=Path(name)
        require(p.is_absolute() and p.is_relative_to(root/BASE_ROOT/'execution_01/random/train')
                and not p.is_symlink() and sha(p)==digest,'completed25 random artifact changed')
    bundle=owner.completed_training(old,paths['plan_01.json'],'random')
    checkpoint=paths['execution_01/random/train/policy.zip']
    require(bundle['checkpoint']==checkpoint,'wrong26 base checkpoint')
    with zipfile.ZipFile(checkpoint) as archive:
        require(all(hashlib.sha256(archive.read(name)).hexdigest()==digest for name,digest in BASE_SERIALIZED.items()),
                'baseline26 serialized policy/Adam changed')
        data=json.loads(archive.read('data'))
        require(data['num_timesteps']==INITIAL and data['_n_updates']==BASE_UPDATES and data['n_envs']==1
                and data['n_steps']==4096 and data['learning_rate']==1e-4 and data['target_kl']==.005
                and data['seed']==181 and data['aux_coef']==0.0,'baseline26 saved algorithm metadata differs')
    for entry in REFERENCE_REPORTS.values():
        require(sha(root/entry['path'])==entry['sha256'],'reference26 report changed')
    desc={'config':old['config'],'base_checkpoint':{'path':str(checkpoint),'sha256':BASE_ANCHORS['execution_01/random/train/policy.zip'],
            'trained_transitions':INITIAL},'base_protocol':{'path':str(paths['execution_01/random/train/protocol.json']),
            'sha256':BASE_ANCHORS['execution_01/random/train/protocol.json']},
          'baseline_provenance':{'anchors':{str(paths[name]):digest for name,digest in BASE_ANCHORS.items()},
             'training_artifacts_sha256':dict(artifacts),'source_identity':old['source_identity'],
             'serialized_entries_sha256':dict(BASE_SERIALIZED),'updates':BASE_UPDATES},
          'runtime_dependencies_sha256':previous.runtime_dependencies(root)}
    require(desc['runtime_dependencies_sha256']==old['runtime_dependencies_sha256'],'baseline26 SB3 bytes changed')
    return desc,bundle['old_release'],bundle['protocol']


def fixed(base):
    return {'format':FORMAT,'owner_statement':'把这些优化点，在这台电脑能力之内的都完成',
       'scope':'independent same-base same-seed single-env yaw weight pair; no retry or resume of25',
       'output_root':OUTPUT,'arms':dict(ARMS),'training_delay_mode':MODE,'seed':SEED,'envs':1,
       'config':base['config'],'vector_layout':dict(VECTOR_LAYOUT),'rollout_transitions':4096,
       'base_checkpoint':copy.deepcopy(base['base_checkpoint']),'base_protocol':copy.deepcopy(base['base_protocol']),
       'baseline_provenance':copy.deepcopy(base['baseline_provenance']),
       'runtime_dependencies_sha256':copy.deepcopy(base['runtime_dependencies_sha256']),
       'hyperparameters':copy.deepcopy(HYPERPARAMETERS),'resources':dict(RESOURCES),
       'stages':copy.deepcopy(STAGES),'stage_order':list(STAGE_ORDER),'budget':dict(BUDGET),
       'one_attempt_per_stage':True,'retry_or_resume_allowed':False,'partial_checkpoint_eligible':False,
       'warm_start':True,'reset_num_timesteps':False,'preflight':False,'checkpoint_selection':'final_only',
       'selection':copy.deepcopy(SELECTION),'reference_reports':copy.deepcopy(REFERENCE_REPORTS),
       'holdout_reset_seeds':[2001,2002,2003],'automatic_model_replacement':False}


def prepare(path=ROOT/PLAN):
    path=Path(path).absolute()
    require(path==ROOT/PLAN and not path.exists() and not path.is_symlink(),'new fixed26 plan only')
    base,_,_=base_inputs()
    value={**fixed(base),'source_identity':source_identity()}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return value


def load_plan(path=ROOT/PLAN):
    path=Path(path).absolute()
    require(path==ROOT/PLAN and not path.is_symlink(),'fixed26 plan required')
    value=read_json(path);base,_,_=base_inputs();expected=fixed(base)
    require(set(value)==set(expected)|{'source_identity'} and
            json.dumps({k:value[k] for k in expected},sort_keys=True,allow_nan=False)==json.dumps(expected,sort_keys=True,allow_nan=False),
            'scope26 changed')
    check_sources(value['source_identity'])
    return value


def split_stage(key):
    require(key in STAGES,'unknown26 stage')
    return STAGES[key]['arm'],STAGES[key]['kind']


def stage_path(plan,key,root=None):
    arm,kind=split_stage(key)
    require(plan.get('output_root')==OUTPUT,'fixed26 output required')
    return Path(ROOT if root is None else root)/OUTPUT/arm/kind


def command(plan_path,key,output):
    arm,kind=split_stage(key)
    script='train_yaw_reward26.py' if kind=='train' else 'run_yaw_reward26.py'
    value=[sys.executable,'-B',str(ROOT/'tools26'/script),'--plan',str(plan_path),'--arm',arm,'--output',str(output)]
    return value if kind=='train' else value+['--stage',kind]


def select_candidate(formals):
    require(set(formals)==set(ARMS),'both26 developer reports required')
    means,eligible={},[]
    matrix={(s,d,z) for s in ('forward_stop','reverse_stop','turn_stop') for d in range(4) for z in (1001,1002,1003)}
    for arm in ARMS:
        value=formals[arm];rows=value.get('cases',[])
        require(value.get('status')=='COMPLETE' and value.get('error') is None and value.get('cleanup_complete') is True
                and len(rows)==36 and {(r['sequence'],r['delay_ticks'],r['reset_seed']) for r in rows}==matrix,
                'complete26 exact developer matrix required')
        require(all(type(r['J']) in (int,float) and math.isfinite(r['J']) for r in rows),'finite26 developer J required')
        passed=all(r.get('reference_thresholds_met') is True and r.get('failure') is False
                   and r.get('done_reason')=='HORIZON_12S' and type(r.get('policy_transitions')) is int
                   and r['policy_transitions']==600 for r in rows)
        require(value.get('all_36_reference_pass') is passed,'developer26 pass flag differs')
        means[arm]=sum(r['J'] for r in rows)/36
        if passed:eligible.append(arm)
    winner=min(eligible,key=lambda arm:(means[arm],list(ARMS).index(arm))) if eligible else None
    return {'eligible_arms':eligible,'original36_mean_J':means,'selected_arm':winner,
            'selection_finalized':True,'uses_holdout':False,'automatic_model_replacement':False}


def validate_launch(plan,plan_path,key,output):
    arm,kind=split_stage(key);path=Path(plan_path).absolute();output=Path(output).absolute()
    require(path==ROOT/PLAN and not path.is_symlink() and read_json(path)==plan and plan.get('format')==FORMAT
            and output==stage_path(plan,key),'fixed26 child path/plan required')
    journal=read_json(ROOT/OUTPUT/'report.json');record=journal.get('stages',{}).get(key,{})
    require(journal.get('format')==EXECUTION_FORMAT and journal.get('status')=='RUNNING'
            and journal.get('active_stage')==key and journal.get('supervisor_pid')==os.getppid()
            and journal.get('plan_sha256')==sha(path) and journal.get('source_identity')==plan['source_identity']
            and record.get('status')=='STARTED' and isinstance(record.get('nonce'),str) and bool(record['nonce'])
            and record['nonce']==os.environ.get('D1_STAGE26_NONCE') and record.get('command')==command(path,key,output),
            'active26 parent required')
    selection=None
    if kind=='holdout':
        formals={a:read_json(stage_path(plan,'formal_'+a)/'report.json') for a in ARMS}
        selection=select_candidate(formals)
        require(journal.get('selection')==selection and selection['selected_arm']==arm,'only selected26 winner may see holdout')
        used=[name for name,r in journal['stages'].items() if name.startswith('holdout_') and not r.get('status','').startswith('SKIPPED')]
        require(used==[key],'holdout26 cannot switch arms or run twice')
    for prior in STAGE_ORDER[:STAGE_ORDER.index(key)]:
        _,prior_kind=split_stage(prior);status=journal['stages'].get(prior,{}).get('status')
        if status=='COMPLETE':continue
        require(prior_kind=='holdout' and selection is not None and status=='SKIPPED_NOT_SELECTED'
                and split_stage(prior)[0]!=selection['selected_arm'],'prior26 stage incomplete or illegal skip')


def valid_first_rollout26(value):
    try:
        require(value.get('format')=='d1-yaw-reward26-first-rollout-v1' and type(value.get('rollout_transitions')) is int
                and value['rollout_transitions']==4096,'first26 rollout identity differs')
        shapes={'observations.actor':[4096,1,627],'observations.velocity_target':[4096,1,3],
                'actions':[4096,1,16],'episode_starts':[4096,1],'values':[4096,1],'log_probs':[4096,1]}
        arrays=value['arrays']
        require(set(arrays)==set(shapes),'first26 buffer fields differ')
        for key,shape in shapes.items():
            row=arrays[key]
            require(row.get('shape')==shape and all(type(x) is int for x in row['shape'])
                    and row.get('format')=='f' and type(row.get('nbytes')) is int
                    and row['nbytes']==math.prod(shape)*4 and re.fullmatch('[0-9a-f]{64}',row.get('sha256','')) is not None,
                    'first26 buffer identity differs')
        digest=hashlib.sha256(json.dumps(arrays,sort_keys=True,allow_nan=False).encode()).hexdigest()
        require(value.get('sha256')==digest,'first26 joined hash differs')
        return True
    except (ValueError,KeyError,TypeError,AttributeError):return False


def check_completed_training(plan,plan_path,arm):
    from tools16.train_comparison16 import complete_counts
    from tools18.train_delay18 import valid_delay_state
    from tools26.runtime26 import (
        checked_worker_counts26,
        initial_identity_valid26,
        valid_reward_state26,
    )
    require(arm in ARMS,'unknown26 completed arm')
    directory=stage_path(plan,'train_'+arm)
    result,protocol=(read_json(directory/name) for name in ('result.json','protocol.json'))
    checkpoint=directory/'policy.zip'
    identity={'stage':'train_'+arm,'arm':arm,'mode':MODE,'seed':SEED,'envs':1,'plan_sha256':sha(plan_path),
              'source_identity':plan['source_identity'],'hyperparameters':HYPERPARAMETERS,
              'yaw_tracking_weight':ARMS[arm],'reward_change':reward_change(arm)}
    require(all(json.dumps(row.get(k),sort_keys=True,allow_nan=False)==json.dumps(v,sort_keys=True,allow_nan=False)
                for row in (result,protocol) for k,v in identity.items()),'training26 identity/settings differ')
    require(result.get('format')=='d1-yaw-reward26-training-result-v1' and protocol.get('format')=='d1-yaw-reward26-training-protocol-v1'
            and result.get('status')=='COMPLETE' and result.get('error') is None and result.get('cleanup_complete') is True
            and result.get('inputs_unchanged') is True and result.get('eligible_for_evaluation') is True
            and type(result.get('aux_coef')) is float and result['aux_coef']==0.0 and result.get('initial_num_timesteps')==INITIAL
            and result.get('new_trained_transitions')==TRANSITIONS and result.get('final_num_timesteps')==INITIAL+TRANSITIONS
            and result.get('protocol_sha256')==sha(directory/'protocol.json') and not checkpoint.is_symlink()
            and result.get('checkpoint_sha256')==sha(checkpoint),'incomplete/partial26 training')
    workers=result.get('per_worker_counts')
    require(workers==result.get('worker_counts') and checked_worker_counts26(workers)==checked_worker_counts26([result.get('actual_counts')])
            and checked_worker_counts26([result.get('actual_counts')])==checked_worker_counts26([result.get('known_actual_counts')])
            and result.get('actual_counts')==result.get('known_actual_counts') and complete_counts(workers,1,TRANSITIONS)
            and valid_delay_state(result.get('delay_state18'),1,MODE),'incomplete26 counter/delay proof')
    require(initial_identity_valid26(result.get('initial_state'),plan) and protocol.get('initial_state')==result['initial_state']
            and protocol.get('base_checkpoint')==plan['base_checkpoint'] and protocol.get('base_protocol')==plan['base_protocol']
            and protocol.get('planned_new_transitions')==TRANSITIONS and protocol.get('planned_native_steps')==TRANSITIONS*10
            and protocol.get('warm_start') is True and protocol.get('reset_num_timesteps') is False
            and protocol.get('teacher_policy_used') is False
            and protocol.get('checkpoint_selection')==result.get('checkpoint_selection')=='final_only','initial26 base/Adam/protocol incomplete')
    guard=result.get('single_process_guard26',{})
    require(guard.get('backend')=='DummyVecEnv' and guard.get('dirty') is False and guard.get('closed') is True
            and guard.get('interprocess_rpc') is False and type(guard.get('environment_subprocesses')) is int
            and guard['environment_subprocesses']==0 and guard.get('waiting') is False and guard.get('inflight') is None
            and guard.get('construction_complete') is True and all(type(guard.get(k)) is int and guard[k]>0
                for k in ('metadata_attempts','metadata_returns','metadata_validated'))
            and guard['metadata_attempts']==guard['metadata_returns']==guard['metadata_validated'],'dirty/incomplete26 guard')
    require(valid_reward_state26(result.get('reward_state26'),arm,TRANSITIONS),'incomplete26 actual reward provenance')
    first=result.get('first_rollout_nonreward')
    require(valid_first_rollout26(first) and result.get('first_rollout_report_sha256')==sha(directory/'first_rollout26.json')
            and read_json(directory/'first_rollout26.json')==first,'incomplete26 first-rollout provenance')
    _,release,base_protocol=base_inputs()
    require(result.get('versions')==protocol.get('versions')==base_protocol['versions']
            and all(row.get('vector_layout')==plan['vector_layout'] and row.get('runtime_dependencies_sha256')==plan['runtime_dependencies_sha256']
                    for row in (result,protocol)),'runtime26 identity differs')
    bound={'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'training_protocol_sha256':sha(directory/'protocol.json'),'training_result_sha256':sha(directory/'result.json')}
    return {'checkpoint':checkpoint,'protocol':protocol,'result':result,'old_release':release,
        'identity':{'role':'completed_yaw_reward26_'+arm+'_final_only','training_identity':bound}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--prepare',action='store_true');modes.add_argument('--plan',action='store_true')
    args=parser.parse_args(argv);value=prepare() if args.prepare else {'budget':BUDGET}
    print(json.dumps({'status':'PREPARED' if args.prepare else 'PLAN_ONLY','plan':str(ROOT/PLAN),'output':str(ROOT/OUTPUT),
        'budget':value['budget'],'training_started':False,'physics_steps_executed':0,'automatic_model_replacement':False},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
