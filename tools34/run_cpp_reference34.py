"""Qualified33 candidate driven by new C++ state/ORT with a Python oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools30 import run_notch30 as shared  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402

require, sha, atomic_json, counts = shared.require, shared.sha, shared.atomic_json, shared.COUNTS
OUTPUT = ROOT/'evidence/cpp_reference34/execution_01'
PLAN = ROOT/'evidence/cpp_reference34/plan_01.json'
UNIT = 'd1-cpp-reference34-owned01-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
SOURCES = (*shared.SOURCES, 'tools34/cpp_reference34.py', 'tools34/run_cpp_reference34.py',
           'tests/test_cpp_reference34.py', 'docs/cpp_reference34.md',
           'runtime34/actor_server34.cpp', 'tools33/reference_shape33.py',
           'tools33/run_reference_shape33.py', 'tools19/cpp_actor19.py')


def source(proof=None):
    if proof is None:
        proof = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                 'sources_sha256': {name:sha(ROOT/name) for name in SOURCES}}
    require(set(proof['sources_sha256']) == set(SOURCES), 'source34 closure differs')
    for name, digest in proof['sources_sha256'].items():
        require(sha(ROOT/name) == digest and not (ROOT/name).is_symlink(), 'source34 changed')
        blob = subprocess.check_output(['git','show',proof['git_commit']+':'+name], cwd=ROOT, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit34 source first')
    return proof


def scope():
    return {'format':'d1-cpp-reference34-plan-v1', 'output':str(OUTPUT), 'unit':UNIT,
            'candidate':'C++34 executes qualified33 gain1.5/tau.1 state and ORT; Python same-input oracle only',
            'base':'complete25_random', 'actor_calls_per_tick':2, 'cpp_calls_max':43200, 'python_oracle_calls_max':43200,
            'training_transitions':0, 'budget':counts(43200), 'holdout_only_if_original36_all_pass':True,
            'automatic_model_replacement':False, 'retry_or_resume':False,
            'admission_stable_s':10, 'admission_timeout_s':300, 'execution_wall_s':1200,
            'start_effective_bytes':4<<30, 'reserve_effective_bytes':2<<30,
            'owned_high_bytes':2<<30, 'owned_max_bytes':3<<30, 'owned_swap_max_bytes':0}


from tools34.cpp_reference34 import Runtime, build_actor  # noqa: E402


def inputs():
    identity, bundle = shared.inputs()
    folder = ROOT/'evidence/reference_shape33/execution_01'
    report = shared.read_json(folder/'report.json')
    require(report['status']=='COMPLETE' and report['error'] is None and report['cleanup_complete']
            and report['python_original_and_holdout_qualified'] and report['actual_counts']==counts(43200),
            '34 requires completed qualified33 candidate')
    committed = report['source']['git_commit']
    for name in ('tools33/reference_shape33.py','tools33/run_reference_shape33.py'):
        blob = subprocess.check_output(['git','show',committed+':'+name],cwd=ROOT,timeout=10)
        require(hashlib.sha256(blob).hexdigest()==sha(ROOT/name)==report['source']['sources_sha256'][name],
                'qualified33 algorithm changed')
    artifacts = {str(folder/'report.json'):sha(folder/'report.json')}
    for stage in ('formal','holdout'):
        cases = report['stages'][stage]['cases']
        require([(c['sequence'],c['delay_ticks'],c['reset_seed']) for c in cases]==shared.matrix(stage)
                and all(c['reference_thresholds_met'] and c['policy_transitions']==600 for c in cases),
                '33 qualified full matrix missing')
        for case in cases:
            for filename,digest in ((case['trace_file'],case['trace_sha256']),
                    (case['reference_shape33']['sidecar'],case['reference_shape33']['sha256'])):
                path = folder/stage/filename
                require(sha(path)==digest,'qualified33 evidence changed')
                artifacts[str(path)] = digest
    return {**identity,'qualified33_artifacts':artifacts},bundle


def execute():
    plan = shared.read_json(PLAN)
    require(plan['scope']==scope(), 'scope34 differs')
    source(plan['source'])
    identity,bundle = inputs()
    require(identity==plan['inputs'] and os.getpriority(os.PRIO_PROCESS,0)>=10, 'inputs/Nice differ')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    started,admitted = time.monotonic(),None
    result = {'status':'INCOMPLETE','error':None,'source':plan['source'],'plan_sha256':sha(PLAN),
              'stages':{},'actual_counts':counts(0),'known_actual_counts':counts(0),
              'actor_attempts':0,'actor_returns':0,'cleanup_complete':True,'training_transitions':0,
              'cpp_request_attempts':0,'cpp_request_returns':0,
              'cpp_original_and_holdout_qualified':False,'new_cpp_closedloop_proven':False,
              'automatic_model_replacement':False}
    last = -float('inf')
    def poll(force=False):
        nonlocal last
        now = time.monotonic()
        require(now-(admitted if admitted is not None else started)<(1200 if admitted is not None else 300), 'wall34 limit')
        if force or now-last>=1:
            m = shared.snapshot_memory(SCOPE)
            with (OUTPUT/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps({**m,'elapsed_s':now-started})+'\n')
            shared.check_memory(m)
            require(m['effective_available_bytes']>=2<<30, 'runtime RAM34 reserve')
            last = now
            return m['effective_available_bytes']
    with termination_guard() as control:
        try:
            stable = None
            while True:
                available,now = poll(True),time.monotonic()
                stable = (now if stable is None else stable) if available>=4<<30 else None
                if stable is not None and now-stable>=10:
                    admitted = now
                    break
                time.sleep(1)
            result['cpp_build'] = build_actor(OUTPUT/'build')
            from tools17.evaluate_formal17 import runtime
            for stage in ('formal','holdout'):
                if stage=='holdout' and not shared.eligible(result['stages']['formal']):
                    result['stages'][stage] = {'status':'SKIPPED_ORIGINAL36_FAILED','actual_counts':counts(0)}
                    break
                folder = OUTPUT/stage
                folder.mkdir()
                child = {'status':'INCOMPLETE','error':None,'cases':[],'cleanup_complete':True,
                         'all_36_reference_pass':False,'actual_counts':counts(0),'known_actual_counts':counts(0)}
                result['stages'][stage] = child
                atomic_json(folder/'protocol.json',{'plan_sha256':sha(PLAN),'stage':stage,'budget':counts(21600)})
                rt = None
                def factory(*args):
                    nonlocal rt
                    rt = Runtime(runtime(*args),folder,result['cpp_build'])
                    return rt
                try:
                    shared.evaluate(stage,bundle,shared.PARENT,folder,child,poll,
                                    runtime_factory=factory,verifier=lambda *args:rt.verify(*args))
                    child['status'] = 'COMPLETE'
                except BaseException as exc:
                    child['error'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    if rt is not None:
                        rt.close_actor()
                        child['cleanup_complete'] &= rt.cpp_cleanup
                        child['cpp_attempts'],child['cpp_returns'] = rt.cpp_attempts,rt.cpp_returns
                        child['actor_attempts'],child['actor_returns'] = rt.attempts,rt.returns
                        result['actor_attempts'] += rt.attempts
                        result['actor_returns'] += rt.returns
                        result['cpp_request_attempts'] += rt.cpp_attempts
                        result['cpp_request_returns'] += rt.cpp_returns
                    atomic_json(folder/'report.json',child)
                    result['known_actual_counts'] = shared.known_counts(result['stages'])
                    result['actual_counts'] = (None if any(r['actual_counts'] is None for r in result['stages'].values())
                                               else result['known_actual_counts'])
                    result['cleanup_complete'] = all(r.get('cleanup_complete',True) for r in result['stages'].values())
                    atomic_json(OUTPUT/'report.json',result)
            require(inputs()[0]==identity and sha(PLAN)==result['plan_sha256'], 'inputs34 changed')
            source(plan['source'])
            poll(True)
            require(result['cleanup_complete'], 'cleanup34 incomplete')
            require(result['actual_counts'] is not None and
                    result['cpp_request_attempts']==result['cpp_request_returns']==
                    result['actor_attempts']==result['actor_returns']==result['actual_counts']['policy_transitions'],
                    '34 requires one completed C++ inference and Python oracle per physical tick')
            holdout = result['stages']['holdout']
            result['cpp_original_and_holdout_qualified'] = (shared.eligible(result['stages']['formal']) and
                holdout.get('status')=='COMPLETE' and holdout.get('all_36_reference_pass') is True and
                holdout.get('actual_counts')==counts(21600))
            result['new_cpp_closedloop_proven'] = result['cpp_original_and_holdout_qualified']
            result.update(status='COMPLETE',inputs_unchanged=True)
        except BaseException as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
            (OUTPUT/'error.txt').write_text(traceback.format_exc())
        finally:
            control['defer'] = True
            if control['received']:
                result.update(status='INCOMPLETE',error=control['received'])
            result['elapsed_s'] = time.monotonic()-started
            atomic_json(OUTPUT/'report.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--prepare',action='store_true')
    g.add_argument('--execute',action='store_true')
    args = p.parse_args()
    if args.execute:
        r = execute()
        print(json.dumps({k:r[k] for k in ('status','error','actual_counts','cpp_original_and_holdout_qualified')},indent=2))
        return int(r['status']!='COMPLETE')
    if args.prepare:
        require(not PLAN.exists() and not OUTPUT.exists(), 'single34 attempt only')
        identity,_ = inputs()
        value = {'scope':scope(),'source':source(),'inputs':identity}
        PLAN.parent.mkdir(parents=True,exist_ok=True)
        with PLAN.open('x') as stream:
            json.dump(value,stream,indent=2)
    print(json.dumps({'status':'PREPARED' if args.prepare else 'PLAN_ONLY',**scope()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
