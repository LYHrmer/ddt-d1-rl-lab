"""74 complete archived full-state response readback; no model or physics calls."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools74.geometry74 import (
    EPSILONS as TRAIN_EPS,
)
from tools74.geometry74 import (
    GROUPS,
    endpoint_outputs,
    group_squared_errors,
    quaternion_audit,
    scalar_qualification,
)
from tools74.runtime74 import run_preserving

PLAN72 = 'docs/experiments72/yaw_weight72_preregistered.json'
RAW72 = 'evidence/yaw_weight72/execution_01'
REPORT72 = RAW72 + '/report.json'
PUBLISHED72 = 'reports/yaw_weight72_results_20261101.json'
LABEL68 = 'evidence/body_response68/admission_01/body_labels96.npz'
REPORT68 = 'evidence/body_response68/admission_01/report.json'
SOURCES = {'58': 'evidence/train_action_response58/collection_01',
           '62': 'evidence/clean_history62/execution_01'}
ROLES = ('old56', 'old64', 'old69', 'fresh')
SEEDS = (64001, 64002, 64003)
EPS = (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4)
OUTPUTS = ('body_vx', 'body_vy', 'gravity_x', 'gravity_y', 'height')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    paths = {PLAN72, REPORT72, PUBLISHED72, LABEL68, REPORT68,
             'tools74/run74.py', 'tools74/geometry74.py', 'tools74/runtime74.py',
             'tests/test_geometry74.py', 'tests/test_runtime74.py',
             'docs/full_response74_plan_20261101.md', 'docs/full_response74_proposal_20261101.md',
             '.github/workflows/full-response74.yml', 'tools54/numerics54.py', 'tools52/dynamics52.py'}
    for source in SOURCES.values():
        paths.add(source+'/report.json')
        for i in range(12):
            paths.add(f'{source}/response_{i:02d}.npz')
            paths.update(f'{source}/branch_{i:02d}_{b:02d}.npz' for b in range(10))
    for seed in SEEDS:
        for arm in ('control','active'):
            for role in ROLES:
                paths.update(f'{RAW72}/{arm}_seed{seed}/first_step_{role}/point_{i:02d}.npz' for i in range(9))
    return paths


def verify(plan):
    require(plan['schema']=='d1-full-response74-v1', 'wrong plan')
    require(plan['model_seeds']==list(SEEDS) and plan['roles']==list(ROLES), 'grid changed')
    require(plan['intervals']==864 and plan['training_centers']==24 and plan['raw_scalar_errors']==38880, 'counts changed')
    require(plan['cpu_seconds']==30 and plan['wall_seconds']==60, 'budget changed')
    require(all(plan[k]==0 for k in ('model_calls','actor_calls','physics_steps','updates')), 'not read-only')
    require(inputs() <= set(plan['source_sha256']), 'missing binding')
    require(sha(ROOT/PLAN72)=='a80e30727ddaedde27d2fa5d52f4dc55b92056cfd04ac08c46c81d5007fab4fa', 'original72 changed')
    old=json.loads((ROOT/PLAN72).read_text())
    require(all(plan['source_sha256'].get(k)==v for k,v in old['source_sha256'].items()), '72 inheritance changed')
    for name,digest in plan['source_sha256'].items():
        require(sha(ROOT/name)==digest, 'source changed: '+name)


def read_npz(path, np):
    with np.load(path, allow_pickle=False) as source:
        return {key:source[key].copy() for key in source.files}


def record_close(report, name, actual, expected):
    gap=abs(actual-expected)
    row=report['readback_parity'].setdefault(name, {'checks':0,'maximum_absolute_gap':0.})
    row['checks']+=1
    row['maximum_absolute_gap']=max(row['maximum_absolute_gap'],gap)
    require(math.isfinite(actual) and math.isfinite(expected)
            and gap<=1e-10+1e-8*abs(expected), 'readback mismatch: '+name)


def aggregate(points):
    intervals=[e for p in points for e in p['eps']]
    require(len(intervals) in (36,144), 'complete role/full grid required')
    groups={g:math.sqrt(math.fsum(e['group_mse'][g] for e in intervals)/len(intervals)) for g in GROUPS}
    outputs={name:math.sqrt(math.fsum(e['outputs'][name]['error']**2 for e in intervals)/len(intervals)) for name in OUTPUTS}
    return {'intervals':len(intervals),'group_normalized_rmse':groups,'output_rmse_physical':outputs,
            'seven_group_normalized_rmse':math.sqrt(math.fsum(v*v for v in groups.values())/7)}


def calculate(report, plan, output):
    resource.setrlimit(resource.RLIMIT_CPU,(28,30))
    require(os.getpriority(os.PRIO_PROCESS,0)==10 and len(os.sched_getaffinity(0))==1, 'Nice10/singleCPU required')
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'single thread required')
    verify(plan)
    import numpy as np
    require(np.__version__=='2.2.6', 'original NumPy required')
    report.update(numpy_version=np.__version__,readback_parity={},models=[],paired=[],
                  original_qualifications_changed=False,full_model_zero_endpoint_available=False,
                  full_training_model_responses_available=False,quaternion_failure_count=0)
    started_cpu,started_wall=time.process_time(),time.monotonic()

    def budget():
        require(time.process_time()-started_cpu<25 and time.monotonic()-started_wall<55,'diagnostic budget exhausted')

    prior=json.loads((ROOT/REPORT72).read_text())
    published=json.loads((ROOT/PUBLISHED72).read_text())
    old68=json.loads((ROOT/REPORT68).read_text())
    require(prior==published['formal'] and prior['status']==prior['verification_after']=='COMPLETE', 'complete72 required')
    for name in inputs():
        if name.startswith(RAW72+'/'):
            require(sha(ROOT/name)==published['raw_manifest'][name]['sha256'],'published72 binding differs')
    labels=read_npz(ROOT/LABEL68,np)
    scales=np.asarray([.1]*7+[1. if i in (3,7,11,15) else .5 for i in range(16)]
                      +[.3]*6+[5. if i in (3,7,11,15) else 2. for i in range(16)],dtype=np.float32).astype(np.float64).tolist()
    report['physical_scales']=scales
    for source,folder in SOURCES.items():
        previous=json.loads((ROOT/folder/'report.json').read_text())
        require(previous['status']==previous['verification_after']=='COMPLETE','complete training source required')
        for index in range(12):
            budget()
            original=previous['points'][index]
            row={'source':source,'seed':original['seed'],'tick':original['tick'],'status':'INCOMPLETE',
                 'raw45':[],'outputs':{},'quaternions':[]}
            report['training_labels'].append(row)
            raw=read_npz(ROOT/folder/f'response_{index:02d}.npz',np)
            states=[]
            for branch in range(10):
                a=read_npz(ROOT/folder/f'branch_{index:02d}_{branch:02d}.npz',np)['target']
                require(a.dtype==np.float64 and a.shape==(1,45) and np.isfinite(a).all(),'raw F64 training endpoint required')
                states.append(a[0])
            require(np.array_equal(states[0],states[1]),'repeated zero training endpoint differs')
            transformed=[endpoint_outputs(s.tolist()) for s in states]
            for ep,epsilon in enumerate(TRAIN_EPS):
                plus,minus=states[2+2*ep:4+2*ep]
                q=quaternion_audit(plus.tolist(),minus.tolist());row['quaternions'].append(q)
                report['quaternion_failure_count']+=not q['passed']
                for key,value in [('central_state45_fd',(plus-minus)/(2*epsilon)),
                                  ('forward_state45_fd',(plus-states[0])/epsilon),
                                  ('backward_state45_fd',(states[0]-minus)/epsilon)]:
                    require(raw[key].shape==(4,45) and raw[key].dtype==np.float64,'complete F64 training responses required')
                    for actual,expected in zip(value,raw[key][ep],strict=True):
                        record_close(report,'training_raw45_from_original_endpoints',float(actual),float(expected))
            for coordinate in range(45):
                eps=[{'epsilon':e,'central':float(raw['central_state45_fd'][i,coordinate]),
                      'forward':float(raw['forward_state45_fd'][i,coordinate]),
                      'backward':float(raw['backward_state45_fd'][i,coordinate])} for i,e in enumerate(TRAIN_EPS)]
                row['raw45'].append({'coordinate':coordinate,'eps':eps,'qualification':scalar_qualification(eps)})
            require(row['raw45'][28]['qualification']['stable']==original['yaw_stability']['stable'],'old yaw qualification differs')
            for name in OUTPUTS:
                eps=[]
                for ep,e in enumerate(TRAIN_EPS):
                    plus,minus=transformed[2+2*ep:4+2*ep];zero=transformed[0][name]
                    eps.append({'epsilon':e,'central':(plus[name]-minus[name])/(2*e),
                                'forward':(plus[name]-zero)/e,'backward':(zero-minus[name])/e})
                row['outputs'][name]={'eps':eps,'qualification':scalar_qualification(eps)}
                if name.startswith('body_'):
                    channel=name[-2:];column=('vx','vy').index(channel);offset=(0 if source=='58' else 48)+4*index
                    for ep in range(4):
                        record_close(report,'training_body68',eps[ep]['central'],float(labels['target_body'][offset+ep,column]))
                    old=next(p for p in old68['points'] if p['source']==source and p['seed']==row['seed'] and p['tick']==row['tick'])
                    require(row['outputs'][name]['qualification']['stable']==old['channels'][channel]['stability']['stable'],
                            'old body qualification differs')
            row['status']='COMPLETE'
    comparison={(m['seed'],m['arm']):m for m in prior['comparison']['models']}
    for seed in SEEDS:
        for arm in ('control','active'):
            model_points=[]
            for role in ROLES:
                for index in range(9):
                    budget()
                    expected=comparison[(seed,arm)]['first_step'][role]['points'][index]
                    body=comparison[(seed,arm)]['heads'][role]['points'][index]['body']
                    row={'model_seed':seed,'arm':arm,'role':role,'seed':expected['seed'],'tick':expected['tick'],
                         'status':'INCOMPLETE','eps':[]}
                    report['points'].append(row);model_points.append(row)
                    path=ROOT/f'{RAW72}/{arm}_seed{seed}/first_step_{role}/point_{index:02d}.npz'
                    data=read_npz(path,np)
                    require(all(a.dtype==np.float64 and np.isfinite(a).all() for a in data.values()),'raw F64 model endpoints required')
                    for ep,e in enumerate(EPS):
                        prefix=f'epsilon_{ep}_'
                        mp,mm=[data[prefix+'model_next_'+x][0] for x in ('plus','minus')]
                        tp,tm=[data[prefix+'actual_next_'+x] for x in ('plus','minus')]
                        require(all(a.shape==(45,) for a in (mp,mm,tp,tm)),'endpoint shape differs')
                        predicted=data[prefix+'model_state45_fd']
                        require(predicted.shape==(45,),'model response cache shape differs')
                        true=(tp-tm)/(2*e)
                        for a,b in zip((mp-mm)/(2*e),predicted,strict=True):
                            record_close(report,'model_raw45_cache',float(a),float(b))
                        require(float(predicted[28])==expected['eps'][ep]['model_fd']
                                and float(true[28])==expected['eps'][ep]['true_fd'],'yaw72 exact readback differs')
                        q={'model':quaternion_audit(mp.tolist(),mm.tolist()),'actual':quaternion_audit(tp.tolist(),tm.tolist())}
                        report['quaternion_failure_count']+=sum(not v['passed'] for v in q.values())
                        values=[endpoint_outputs(a.tolist()) for a in (mp,mm,tp,tm)]
                        interval={'epsilon':e,'quaternions':q,
                            'raw45_normalized_error':((predicted-true)/scales).tolist(),
                            'group_mse':group_squared_errors(predicted.tolist(),true.tolist(),scales,GROUPS),'outputs':{}}
                        for name in OUTPUTS:
                            fd_model=(values[0][name]-values[1][name])/(2*e)
                            fd_true=(values[2][name]-values[3][name])/(2*e)
                            interval['outputs'][name]={'model_fd':fd_model,'true_fd':fd_true,'error':fd_model-fd_true}
                            if name.startswith('body_'):
                                old=body[name[-2:]]['eps'][ep]
                                record_close(report,'validation_body72_model',fd_model,old['model_fd'])
                                record_close(report,'validation_body72_true',fd_true,old['true_fd'])
                        record_close(report,'height_raw45_identity',interval['outputs']['height']['error'],float(predicted[2]-true[2]))
                        row['eps'].append(interval)
                    row['status']='COMPLETE'
            summary={'seed':seed,'arm':arm,'all':aggregate(model_points),
                     'by_role':{role:aggregate([p for p in model_points if p['role']==role]) for role in ROLES}}
            body_rms=math.sqrt(math.fsum(summary['all']['output_rmse_physical'][c]**2 for c in ('body_vx','body_vy'))/2)/scales[23]
            record_close(report,'aggregate_body72_rmse',body_rms,comparison[(seed,arm)]['body_normalized_rmse'])
            report['models'].append(summary)
    by_key={(m['seed'],m['arm']):m for m in report['models']}
    for seed in SEEDS:
        pair={'seed':seed,'raw_groups':{},'outputs':{}}
        for field,kind in [('group_normalized_rmse','raw_groups'),('output_rmse_physical','outputs')]:
            for name in by_key[(seed,'control')]['all'][field]:
                c,a=[by_key[(seed,arm)]['all'][field][name] for arm in ('control','active')]
                pair[kind][name]={'control':c,'active':a,'difference':a-c,'ratio_active_over_control':a/c if c else None,
                                 'increased_beyond_readback_tolerance':a-c>1e-10+1e-8*abs(c)}
        report['paired'].append(pair)
    report['consistent_increases']={kind:[name for name in report['paired'][0][kind] if
        all(p[kind][name]['increased_beyond_readback_tolerance'] for p in report['paired'])] for kind in ('raw_groups','outputs')}
    raw_quals=[c['qualification'] for p in report['training_labels'] for c in p['raw45']]
    report['training_qualification']={'raw45_qualified':sum(q['stable'] for q in raw_quals),'raw45_required':1080,
        'raw45_all_qualified':all(q['stable'] for q in raw_quals),
        'failure_reason_counts':dict(collections.Counter(reason for q in raw_quals for reason in q['failure_reasons'])),
        'by_output':{name:{'qualified':sum(p['outputs'][name]['qualification']['stable'] for p in report['training_labels']),
                           'required':24} for name in OUTPUTS},'rows_dropped':0}
    require(len(report['training_labels'])==24 and len(report['points'])==216
            and all(p['status']=='COMPLETE' and len(p['eps'])==4 for p in report['points']),'complete74 grid required')
    require(report['quaternion_failure_count']==0,'quaternion representation discontinuity; diagnostic remains incomplete')
    report.update(completed_intervals=864,raw_scalar_errors=38880,training_scalar_channels=1080,
                  descriptive_only=True,repair_validated=False,candidate_authorized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    context={}

    def body(report):
        context['plan']=json.loads(args.plan.read_text());context['sha']=sha(args.plan)
        report.update(schema='d1-full-response74-result-v1',plan_sha256=context['sha'])
        calculate(report,context['plan'],args.output)

    def verify_after():
        require('plan' in context,'plan not loaded')
        verify(context['plan'])
        require(sha(args.plan)==context['sha'],'plan changed')

    report=run_preserving(args.output,body,verify_after)
    print(json.dumps({k:report.get(k) for k in ('status','failure','cpu_seconds','wall_seconds')},indent=2))
    return int(report['status']!='COMPLETE')


if __name__=='__main__':
    raise SystemExit(main())
