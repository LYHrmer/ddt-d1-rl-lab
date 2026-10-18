"""Fixed216-input, three-policy drift diagnosis; never trains or simulates."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT='evidence/policy_drift28/execution_01'
AUTHORED=('tools28/audit_policy_drift28.py','tests/test_policy_drift28.py','docs/policy_drift28.md')
SOURCES=(*AUTHORED,'training/d1_rl/velocity_policy15.py','tools27/resource_guard27.py')
SCOPE=(f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/'
       'app.slice/d1-policy-drift28-20261018-01.service')
TICKS=(125,150,175,325,350,375,425,450,475,500,525,575)
TASKS=('forward_stop','reverse_stop','turn_stop')
ROWS=216
LIMITS={'actor_batch_calls':3,'actor_sample_rows':648,'linear_batch_calls':18,
        'checkpoint_loads':3,'training_transitions':0,'physics_steps':0,'total_wall_s':120,
        'start_effective_bytes':6<<30,'reserve_effective_bytes':4<<30,'input_file_max_bytes':16<<20}
ROLES={
 'base25':('evidence/low_memory_delay25/execution_01/random','873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22',1310720),
 'control27':('evidence/yaw_reward27/execution_02/control','be727124c05d67fc1d891a8a467818ab8620d32da11f84369e47fff3d4c733ba',1441792),
 'yaw3_27':('evidence/yaw_reward27/execution_02/yaw3','b42dec19c33ef3050c3e2577de72dfad59d931a6c80f880ab5e10c595c5cab57',1441792)}
PINS={
 'evidence/low_memory_delay25/execution_01/random/train/result.json':'0ecb4bbd01a4ac0041f55b7ceab0df334a88d3a98fc3eaf0a21e60b4a0e6a4ad',
 'evidence/low_memory_delay25/execution_01/random/train/protocol.json':'ea458bb2bf449c06279fcb0299284dcb5d2a345c959ab67c57831297933e81a2',
 'evidence/yaw_reward27/execution_02/report.json':'89f6405b4fe4970a517e045eb36df2e9d7424e03dd95ddfb8879b72ab0aeea65',
 'evidence/yaw_reward27/execution_02/control/train/result.json':'cef595e439fcff46e8350c3326345330bbb42f8c8b708eef1d17124159932c13',
 'evidence/yaw_reward27/execution_02/control/train/protocol.json':'1b012333c7acf528380828bdbb3d365e557b6a9a2cde01139382d6a480be083c',
 'evidence/yaw_reward27/execution_02/yaw3/train/result.json':'142ef923174321f59d286d8814b887b181f9f4ba73d760a98270f9edaa5fdfd2',
 'evidence/yaw_reward27/execution_02/yaw3/train/protocol.json':'be1176ced4e7775293d22716cf2518a6fa2bb214915157c220130831b45a0763',
 'evidence/cpp_diagnostic23/execution_01/worker/report.json':'9e42e4ad95e1fb1e3c0e95427cf41c448dfb816767c4060859dfef59499149d0',
 'training/d1_rl/velocity_policy15.py':'02a0618d82799c5bcaa5249bf922536501ddf479262891c15d8f83049902c15c'}
# Dimensions follow the SHA-pinned VelocityMlpExtractor plus Gaussian log_std.
SHAPES={'log_std':(16,)}
for prefix,layouts in (
 ('mlp_extractor.encoder',((627,128),(128,64),(64,3))),
 ('mlp_extractor.policy_net',((60,128),(128,128))),
 ('mlp_extractor.value_net',((627,128),(128,128)))):
    for index,(incoming,outgoing) in enumerate(layouts):
        SHAPES[f'{prefix}.{index*2}.weight']=(outgoing,incoming)
        SHAPES[f'{prefix}.{index*2}.bias']=(outgoing,)
SHAPES.update({'action_net.weight':(16,128),'action_net.bias':(16,),
               'value_net.weight':(1,128),'value_net.bias':(1,)})


def require(value,message):
    if not value:raise ValueError(message)


def identity(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=LIMITS['input_file_max_bytes'],
            'regular bounded input required: '+str(path))
    raw=path.read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read_json(path):return json.loads(Path(path).read_text())


def inspect_inputs(root=ROOT):
    """Small receipts and byte identities only; no tensor or NN operation."""
    root=Path(root);proof={}
    for name,digest in PINS.items():
        entry=identity(root/name);require(entry['sha256']==digest,'fixed diagnostic input changed: '+name)
        proof[name]=entry
    parent=read_json(root/'evidence/yaw_reward27/execution_02/report.json')
    require(parent.get('status')=='COMPLETE' and parent.get('cleanup_complete') is True
        and parent.get('error') is None and parent.get('selected_arm') is None
        and parent.get('simulation_candidate_qualified') is False,'expected completed unqualified27 pair')
    roles={};versions=None
    for role,(relative,digest,final_steps) in ROLES.items():
        directory=root/relative/'train';result=read_json(directory/'result.json');protocol=read_json(directory/'protocol.json')
        checkpoint=directory/'policy.zip';entry=identity(checkpoint);proof[str(checkpoint.relative_to(root))]=entry
        require(entry['sha256']==digest==result.get('checkpoint_sha256')
            and result.get('status')=='COMPLETE' and result.get('error') is None
            and result.get('cleanup_complete') is True and result.get('inputs_unchanged') is True
            and result.get('eligible_for_evaluation') is True and result.get('final_num_timesteps')==final_steps
            and result.get('protocol_sha256')==identity(directory/'protocol.json')['sha256']
            and result.get('versions')==protocol.get('versions'),'partial or foreign policy: '+role)
        versions=protocol['versions'] if versions is None else versions
        require(protocol['versions']==versions,'policy library stacks differ')
        with zipfile.ZipFile(checkpoint) as archive:
            require(archive.read('_stable_baselines3_version')==b'2.7.0','unsupported checkpoint SB3')
        roles[role]={'checkpoint':str(checkpoint),'identity':entry,'result':str(directory/'result.json')}
    reference=root/'evidence/cpp_diagnostic23/execution_01/worker';report=read_json(reference/'report.json')
    require(report.get('status')=='COMPLETE' and report.get('error') is None
        and report.get('cleanup_complete') is True and report.get('inputs_unchanged') is True,'reference23 incomplete')
    selected=[r for r in report['cases'] if r['delay_ticks'] in (0,1)]
    expected={(task,delay,seed) for task in TASKS for delay in (0,1) for seed in (1001,1002,1003)}
    require(len(selected)==18 and {(r['sequence'],r['delay_ticks'],r['reset_seed']) for r in selected}==expected,
            'exact eighteen complete zero/20ms cases required')
    traces=[]
    for row in selected:
        name=f"{row['sequence']}_delay{row['delay_ticks']}_seed{row['reset_seed']}_inference.npz"
        require(row.get('inference_file')==name and row.get('policy_transitions')==row.get('observations')==600
            and row.get('done_reason')=='HORIZON_12S' and row.get('reference_thresholds_met') is True,
            'reference input case incomplete')
        path=reference/name;entry=identity(path);require(entry['sha256']==row['inference_sha256'],'reference input SHA differs')
        proof[str(path.relative_to(root))]=entry
        traces.append({'path':str(path),'sequence':row['sequence'],'delay_ticks':row['delay_ticks'],'reset_seed':row['reset_seed']})
    return {'roles':roles,'traces':traces,'versions':versions,'inputs':proof,
            'qualification_granted':False,'reference_distribution':'old23 control trajectories; not on-policy25/27'}


def source_identity(root=ROOT):
    root=Path(root);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()
    values={}
    for name in SOURCES:
        entry=identity(root/name)
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=root,stderr=subprocess.PIPE,timeout=5)
        require(hashlib.sha256(blob).hexdigest()==entry['sha256'],'commit clean sources before execute: '+name)
        values[name]=entry
    return {'git_commit':commit,'files':values}


def libraries():
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    import numpy as np
    import torch
    from torch.nn import functional
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    return SimpleNamespace(np=np,torch=torch,functional=functional,
        as_numpy=lambda x:x.detach().cpu().numpy().copy(),
        versions={'numpy':np.__version__,'torch':torch.__version__})


def load_state(checkpoint,lib,counters):
    """PyTorch's weights-only loader; never unpickle SB3 data or optimizer."""
    counters['checkpoint_load_attempts']+=1
    with zipfile.ZipFile(checkpoint) as archive:
        item=archive.getinfo('policy.pth');require(item.file_size<2<<20,'oversized policy state')
        raw=archive.read(item)
    state=lib.torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    require(isinstance(state,dict) and set(state)==set(SHAPES),'exact19 policy tensors required')
    weights={}
    for name,shape in SHAPES.items():
        value=state[name]
        require(value.dtype==lib.torch.float32 and value.device.type=='cpu' and tuple(value.shape)==shape,
                'unexpected policy tensor: '+name)
        array=lib.as_numpy(value)
        require(array.dtype==lib.np.float32 and bool(lib.np.isfinite(array).all()),'nonfinite policy tensor: '+name)
        weights[name]=array
    counters['checkpoint_load_returns']+=1
    return weights


def fixed_inputs(traces,np):
    values=[];labels=[]
    for row in traces:
        with zipfile.ZipFile(row['path']) as archive:
            require(sum(v.file_size for v in archive.infolist())<8<<20,'oversized reference NPZ')
        with np.load(row['path'],allow_pickle=False) as archive:obs=archive['observation']
        require(obs.shape==(600,627) and obs.dtype==np.float32 and bool(np.isfinite(obs).all()),'actual627 reference input differs')
        values.append(obs[list(TICKS)].copy())
        labels.extend({**{k:row[k] for k in ('sequence','delay_ticks','reset_seed')},'tick':tick,
                       'window':'drive' if tick<400 else 'stop'} for tick in TICKS)
    batch=np.concatenate(values,axis=0)
    require(batch.shape==(ROWS,627) and len(labels)==ROWS,'exact216 fixed observations required')
    return batch,labels


def actor_batch(observations,weights,lib,counters):
    """One216-row actor invocation, six Linear ops; no critic or aux target."""
    np,t=lib.np,lib.torch
    require(observations.shape==(ROWS,627) and observations.dtype==np.float32
        and bool(np.isfinite(observations).all()),'finite216x627 batch required')
    counters['actor_batch_attempts']+=1;counters['sample_rows_attempted']+=ROWS
    def linear(x,prefix):
        counters['linear_attempts']+=1
        out=lib.functional.linear(x,t.from_numpy(weights[prefix+'.weight']),t.from_numpy(weights[prefix+'.bias']))
        counters['linear_returns']+=1
        return out
    with t.no_grad():
        x=t.from_numpy(observations)
        velocity=linear(t.tanh(linear(t.tanh(linear(x,'mlp_extractor.encoder.0')),'mlp_extractor.encoder.2')),'mlp_extractor.encoder.4')
        features=t.cat((x[:,:57],velocity),dim=1)
        latent=t.tanh(linear(t.tanh(linear(features,'mlp_extractor.policy_net.0')),'mlp_extractor.policy_net.2'))
        raw=lib.as_numpy(linear(latent,'action_net'))
    require(raw.shape==(ROWS,16) and raw.dtype==np.float32 and bool(np.isfinite(raw).all()),'nonfinite raw actor mean')
    counters['actor_batch_returns']+=1;counters['sample_rows_returned']+=ROWS
    return {'raw_mean':raw,'clipped_mean':np.clip(raw,-1.,1.),'log_std':weights['log_std'].copy()}


def comparisons(outputs,labels,np):
    def stats(value):
        require(bool(np.isfinite(value).all()),'nonfinite diagnostic statistic')
        return {'mean':float(np.mean(value)),'median':float(np.median(value)),
                'p95':float(np.quantile(value,.95)),'max':float(np.max(value))}
    reference=outputs['base25'];results={}
    for role in ('control27','yaw3_27'):
        target=outputs[role];a=reference['log_std'].astype(np.float64);b=target['log_std'].astype(np.float64)
        delta=reference['raw_mean'].astype(np.float64)-target['raw_mean'].astype(np.float64)
        kl=np.sum(b-a+(np.exp(2*a)+delta**2)/(2*np.exp(2*b))-.5,axis=1)
        require(bool(np.isfinite(kl).all()) and bool((kl>=-1e-10).all()),'invalid diagonal Gaussian KL')
        kl=np.maximum(kl,0.)
        results[role]={'KL_base25_to_new':stats(kl),'raw_mean_abs_delta':stats(np.abs(delta)),
            'clipped_mean_abs_delta':stats(np.abs(reference['clipped_mean'].astype(np.float64)-target['clipped_mean'])),
            'base_log_std':a.tolist(),'new_log_std':b.tolist(),'log_std_abs_delta':stats(np.abs(a-b)),
            'by_task_window':{task+'_'+window:stats(kl[[i for i,v in enumerate(labels) if v['sequence']==task and v['window']==window]])
                              for task in TASKS for window in ('drive','stop')}}
    return results


def counters_zero():
    return dict.fromkeys(('checkpoint_load_attempts','checkpoint_load_returns','actor_batch_attempts','actor_batch_returns',
                         'sample_rows_attempted','sample_rows_returned','linear_attempts','linear_returns'),0)


def memory():
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from tools27.resource_guard27 import read_memory
    return read_memory(SCOPE)


def write_json(path,value):
    temporary=Path(path).with_suffix('.json.tmp')
    with temporary.open('w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    temporary.replace(path)


def execute(*,root=ROOT,input_loader=inspect_inputs,source_loader=source_identity,library_loader=libraries,
            memory_reader=memory,clock=time.monotonic):
    started=clock();root=Path(root);output=root/OUTPUT
    require(not output.exists() and not output.is_symlink(),'new drift28 output required; no retry')
    bundle=input_loader(root);source=source_loader(root);sample=memory_reader()
    require(sample['effective_available_bytes']>=LIMITS['start_effective_bytes'],'drift28 start RAM below6GiB')
    output.mkdir(parents=True,exist_ok=False)
    report={'schema':'d1-policy-drift28-v1','status':'RUNNING','source_identity':source,
        'inputs':bundle['inputs'],'limits':LIMITS,'known_actual_counts':counters_zero(),
        'training_transitions':0,'physics_steps':0,'qualification_granted':False,'default_model_changed':False,
        'reference_distribution':bundle['reference_distribution'],'error':None,'resource_samples':[sample]}
    write_json(output/'report.json',report)
    def poll():
        require(clock()-started<LIMITS['total_wall_s'],'drift28 wall exceeded')
        value=memory_reader();report['resource_samples'].append(value)
        require(value['effective_available_bytes']>=LIMITS['reserve_effective_bytes'],'drift28 reserve below4GiB')
    try:
        poll();lib=library_loader();poll()
        require(all(lib.versions[k]==bundle['versions'][k] for k in ('numpy','torch')),'tensor library versions differ from saved policies')
        inputs,labels=fixed_inputs(bundle['traces'],lib.np);results={}
        for role in ROLES:
            poll();weights=load_state(bundle['roles'][role]['checkpoint'],lib,report['known_actual_counts'])
            results[role]=actor_batch(inputs,weights,lib,report['known_actual_counts']);poll()
            write_json(output/'report.json',report)
        arrays={'observation':inputs,**{role+'__'+key:value for role,result in results.items() for key,value in result.items()}}
        with (output/'distributions.npz').open('xb') as stream:lib.np.savez(stream,**arrays)
        report.update(comparisons=comparisons(results,labels,lib.np),sample_labels=labels,
            distribution_file=identity(output/'distributions.npz'),versions=lib.versions)
        require(input_loader(root)==bundle and source_loader(root)==source,'drift28 input/source changed')
        counts=report['known_actual_counts']
        require(counts==dict(checkpoint_load_attempts=3,checkpoint_load_returns=3,actor_batch_attempts=3,actor_batch_returns=3,
            sample_rows_attempted=648,sample_rows_returned=648,linear_attempts=18,linear_returns=18),'drift28 actual inference count differs')
        poll();report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='INCOMPLETE',error=type(exc).__name__+': '+str(exc))
    finally:
        report['elapsed_s']=clock()-started
        if report['elapsed_s']>=LIMITS['total_wall_s']:report.update(status='INCOMPLETE',error=report['error'] or 'drift28 final wall exceeded')
        write_json(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');args=parser.parse_args(argv)
    value=execute() if args.execute else {'status':'PLAN_ONLY','output':str(ROOT/OUTPUT),'limits':LIMITS,
        'roles':list(ROLES),'fixed_input_rows':ROWS,'ticks_per_case':TICKS,'qualification_granted':False,
        'libraries_loaded':False,'training_transitions':0,'physics_steps':0}
    print(json.dumps(value,allow_nan=False));return 0 if value['status'] in ('COMPLETE','PLAN_ONLY') else 2


if __name__=='__main__':raise SystemExit(main())
