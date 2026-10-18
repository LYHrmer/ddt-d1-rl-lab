"""One same-weight full-matrix D1 mirror projection, using original evaluator."""
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
OUTPUT = ROOT/'evidence/symmetry31/execution_01'
PLAN = ROOT/'evidence/symmetry31/plan_01.json'
UNIT = 'd1-symmetry31-owned01-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
SOURCES = (*shared.SOURCES, 'tools31/symmetry31.py', 'tools31/run_symmetry31.py',
           'tests/test_symmetry31.py', 'docs/symmetry31.md')


def source(proof=None):
    if proof is None:
        proof = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                 'sources_sha256': {name:sha(ROOT/name) for name in SOURCES}}
    require(set(proof['sources_sha256']) == set(SOURCES), 'source31 closure differs')
    for name, digest in proof['sources_sha256'].items():
        require(sha(ROOT/name) == digest and not (ROOT/name).is_symlink(), 'source31 changed')
        blob = subprocess.check_output(['git','show',proof['git_commit']+':'+name], cwd=ROOT, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit31 source first')
    return proof


def scope():
    return {'format':'d1-symmetry31-plan-v1', 'output':str(OUTPUT), 'unit':UNIT,
            'candidate':'mean(pi(o), mirror_action(pi(mirror_all_11_frames(o))))',
            'base':'complete25_random', 'actor_calls_per_tick':2, 'actor_calls_max':86400,
            'training_transitions':0, 'budget':counts(43200), 'holdout_only_if_original36_all_pass':True,
            'automatic_model_replacement':False, 'retry_or_resume':False,
            'admission_stable_s':10, 'admission_timeout_s':300, 'execution_wall_s':900,
            'start_effective_bytes':4<<30, 'reserve_effective_bytes':2<<30,
            'owned_high_bytes':2<<30, 'owned_max_bytes':3<<30, 'owned_swap_max_bytes':0}


class Runtime:
    def __init__(self, original):
        self.original, self.case = original, None
        self.rows, self.attempts, self.returns = [], 0, 0
        self.backend = {**original.backend, 'kind':'python_ort_complete25_mirror31',
                        'actor_calls_per_tick':2, 'new_cpp_closedloop_proven':False}

    def make_env(self, name, sequence, delay, seed):
        require(not self.rows, 'unverified previous case')
        self.case = (name,delay,seed)
        return self.original.make_env(name,sequence,delay,seed)

    def predict(self, obs):
        import numpy as np
        from tools31.symmetry31 import mirror_observation, combine
        require(self.case is not None and len(self.rows)<600, 'case inference cap')
        values = []
        for value in (np.asarray(obs).copy(), mirror_observation(obs)):
            self.attempts += 1
            require(self.attempts<=43200, 'stage actor call cap')
            values.append(self.original.predict(value).copy())
            self.returns += 1
        issued = combine(*values)
        self.rows.append((values[0],values[1],issued.copy()))
        return issued

    def verify(self, output, item, helpers):
        import numpy as np
        from tools17.evaluate_formal17 import verify_trace
        from tools31.symmetry31 import combine
        verify_trace(output,item,helpers)
        require(self.case == (item['sequence'],item['delay_ticks'],item['reset_seed']), 'case identity differs')
        arrays = np.asarray(self.rows, dtype=np.float32)
        path = output/(Path(item['trace_file']).stem+'_symmetry.npz')
        np.savez_compressed(path, actions=arrays)
        with np.load(path,allow_pickle=False) as saved:
            actual = saved['actions']
        require(np.array_equal(actual,arrays) and len(actual)==item['policy_transitions'], 'sidecar count/readback differs')
        issued = np.asarray([combine(a,b) for a,b,_ in actual])
        require(np.array_equal(issued,actual[:,2]), 'projection replay differs')
        with np.load(output/item['trace_file'],allow_pickle=False) as trace:
            require(np.array_equal(issued,trace['issued_action']), 'physical issue differs')
            delay = item['delay_ticks']
            applied = np.zeros_like(issued)
            if delay == 0:
                applied[:] = issued
            elif len(issued)>delay:
                applied[delay:] = issued[:-delay]
            require(np.array_equal(applied,trace['applied_action']), 'FIFO differs')
        item['symmetry31'] = {'sidecar':path.name,'sha256':sha(path),'projection_exact':True,
                              'fifo_exact':True,'actor_calls':2*len(actual)}
        self.rows.clear()


def execute():
    plan = shared.read_json(PLAN)
    require(plan['scope']==scope(), 'scope31 differs')
    source(plan['source'])
    identity,bundle = shared.inputs()
    require(identity==plan['inputs'] and os.getpriority(os.PRIO_PROCESS,0)>=10, 'inputs/Nice differ')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    started,admitted = time.monotonic(),None
    result = {'status':'INCOMPLETE','error':None,'source':plan['source'],'plan_sha256':sha(PLAN),
              'stages':{},'actual_counts':counts(0),'known_actual_counts':counts(0),
              'actor_attempts':0,'actor_returns':0,'cleanup_complete':True,'training_transitions':0,
              'python_original_and_holdout_qualified':False,'new_cpp_closedloop_proven':False,
              'automatic_model_replacement':False}
    last = -float('inf')
    def poll(force=False):
        nonlocal last
        now = time.monotonic()
        require(now-(admitted if admitted is not None else started)<(900 if admitted is not None else 300), 'wall31 limit')
        if force or now-last>=1:
            m = shared.snapshot_memory(SCOPE)
            with (OUTPUT/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps({**m,'elapsed_s':now-started})+'\n')
            shared.check_memory(m)
            require(m['effective_available_bytes']>=2<<30, 'runtime RAM31 reserve')
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
                    rt = Runtime(runtime(*args))
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
                        child['actor_attempts'],child['actor_returns'] = rt.attempts,rt.returns
                        result['actor_attempts'] += rt.attempts
                        result['actor_returns'] += rt.returns
                    atomic_json(folder/'report.json',child)
                    result['known_actual_counts'] = shared.known_counts(result['stages'])
                    result['actual_counts'] = (None if any(r['actual_counts'] is None for r in result['stages'].values())
                                               else result['known_actual_counts'])
                    result['cleanup_complete'] = all(r.get('cleanup_complete',True) for r in result['stages'].values())
                    atomic_json(OUTPUT/'report.json',result)
            require(shared.inputs()[0]==identity and sha(PLAN)==result['plan_sha256'], 'inputs31 changed')
            source(plan['source'])
            poll(True)
            require(result['cleanup_complete'], 'cleanup31 incomplete')
            holdout = result['stages']['holdout']
            result['python_original_and_holdout_qualified'] = (shared.eligible(result['stages']['formal']) and
                holdout.get('status')=='COMPLETE' and holdout.get('all_36_reference_pass') is True and
                holdout.get('actual_counts')==counts(21600))
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
        print(json.dumps({k:r[k] for k in ('status','error','actual_counts','python_original_and_holdout_qualified')},indent=2))
        return int(r['status']!='COMPLETE')
    if args.prepare:
        require(not PLAN.exists() and not OUTPUT.exists(), 'single31 attempt only')
        identity,_ = shared.inputs()
        value = {'scope':scope(),'source':source(),'inputs':identity}
        PLAN.parent.mkdir(parents=True,exist_ok=True)
        with PLAN.open('x') as stream:
            json.dump(value,stream,indent=2)
    print(json.dumps({'status':'PREPARED' if args.prepare else 'PLAN_ONLY',**scope()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
