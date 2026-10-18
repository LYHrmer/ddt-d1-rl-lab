"""One fixed causal yaw reference shaper; original physical commands and gates."""
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
OUTPUT = ROOT/'evidence/reference_shape33/execution_01'
PLAN = ROOT/'evidence/reference_shape33/plan_01.json'
UNIT = 'd1-reference-shape33-owned01-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
SOURCES = (*shared.SOURCES, 'tools33/reference_shape33.py', 'tools33/run_reference_shape33.py',
           'tests/test_reference_shape33.py', 'docs/reference_shape33.md')


def source(proof=None):
    if proof is None:
        proof = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                 'sources_sha256': {name:sha(ROOT/name) for name in SOURCES}}
    require(set(proof['sources_sha256']) == set(SOURCES), 'source33 closure differs')
    for name, digest in proof['sources_sha256'].items():
        require(sha(ROOT/name) == digest and not (ROOT/name).is_symlink(), 'source33 changed')
        blob = subprocess.check_output(['git','show',proof['git_commit']+':'+name], cwd=ROOT, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit33 source first')
    return proof


def scope():
    return {'format':'d1-reference-shape33-plan-v1', 'output':str(OUTPUT), 'unit':UNIT,
            'candidate':'gain1.5 yaw command followed by causal tau.1s smoothing with actual filtered history; physical commands unchanged',
            'base':'complete25_random', 'actor_calls_per_tick':1, 'actor_calls_max':43200,
            'training_transitions':0, 'budget':counts(43200), 'holdout_only_if_original36_all_pass':True,
            'automatic_model_replacement':False, 'retry_or_resume':False,
            'admission_stable_s':10, 'admission_timeout_s':300, 'execution_wall_s':900,
            'start_effective_bytes':4<<30, 'reserve_effective_bytes':2<<30,
            'owned_high_bytes':2<<30, 'owned_max_bytes':3<<30, 'owned_swap_max_bytes':0}


class Runtime:
    def __init__(self, original):
        self.original, self.case = original, None
        self.rows, self.attempts, self.returns = [], 0, 0
        self.backend = {**original.backend, 'kind':'python_ort_complete25_reference_shape33',
                        'actor_calls_per_tick':1, 'new_cpp_closedloop_proven':False,
                        'yaw_command_gain':1.5, 'reference_time_constant_s':.1, 'physical_command_modified':False}

    def make_env(self, name, sequence, delay, seed):
        require(not self.rows, 'unverified previous case')
        from tools33.reference_shape33 import ReferenceShaper
        self.shaper = ReferenceShaper()
        self.case = (name,delay,seed)
        return self.original.make_env(name,sequence,delay,seed)

    def predict(self, obs):
        import numpy as np
        from tools33.reference_shape33 import YAW_COLUMNS
        require(self.case is not None and len(self.rows)<600, 'case inference cap')
        effective = self.shaper(obs)
        self.attempts += 1
        require(self.attempts<=21600, 'stage actor call cap')
        issued = self.original.predict(effective).copy()
        self.returns += 1
        self.rows.append((np.asarray(obs)[YAW_COLUMNS].copy(),effective[YAW_COLUMNS].copy(),issued.copy()))
        return issued

    def verify(self, output, item, helpers):
        import numpy as np
        from tools17.evaluate_formal17 import verify_trace
        verify_trace(output,item,helpers)
        require(self.case == (item['sequence'],item['delay_ticks'],item['reset_seed']), 'case identity differs')
        arrays = {name:np.asarray([r[i] for r in self.rows],dtype=np.float32)
                  for i,name in enumerate(('observed_yaw_commands','effective_yaw_commands','issued'))}
        path = output/(Path(item['trace_file']).stem+'_reference_shape.npz')
        np.savez_compressed(path, **arrays)
        with np.load(path,allow_pickle=False) as saved:
            actual = {k:saved[k] for k in saved.files}
        require(all(np.array_equal(actual[k],arrays[k]) for k in arrays)
                and len(actual['issued'])==item['policy_transitions'], 'sidecar count/readback differs')
        import math
        raw = actual['observed_yaw_commands']
        target = np.clip(raw*np.float32(1.5),-.25,.25)
        require(np.all(raw[0]==raw[0,0]), 'constant initial prehistory required')
        expected = np.empty_like(raw)
        history = target[0,1:].copy()
        state = float(target[0,0])
        for i in range(len(raw)):
            if i:
                require(np.array_equal(raw[i,1:],np.r_[raw[i-1,2:],raw[i-1,0]]), 'raw history differs')
            state = math.exp(-.2)*state + (1-math.exp(-.2))*float(target[i,0])
            expected[i] = np.r_[np.float32(state),history]
            history = np.r_[history[1:],np.float32(state)]
        require(np.array_equal(expected,actual['effective_yaw_commands']), 'command preprocessing differs')
        issued = actual['issued']
        with np.load(output/item['trace_file'],allow_pickle=False) as trace:
            require(np.array_equal(issued,trace['issued_action']), 'physical issue differs')
            delay = item['delay_ticks']
            applied = np.zeros_like(issued)
            if delay == 0:
                applied[:] = issued
            elif len(issued)>delay:
                applied[delay:] = issued[:-delay]
            require(np.array_equal(applied,trace['applied_action']), 'FIFO differs')
        item['reference_shape33'] = {'sidecar':path.name,'sha256':sha(path),'preprocessing_exact':True,
                                  'fifo_exact':True,'actor_calls':len(issued),'physical_command_modified':False}
        self.rows.clear()


def execute():
    plan = shared.read_json(PLAN)
    require(plan['scope']==scope(), 'scope33 differs')
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
        require(now-(admitted if admitted is not None else started)<(900 if admitted is not None else 300), 'wall33 limit')
        if force or now-last>=1:
            m = shared.snapshot_memory(SCOPE)
            with (OUTPUT/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps({**m,'elapsed_s':now-started})+'\n')
            shared.check_memory(m)
            require(m['effective_available_bytes']>=2<<30, 'runtime RAM33 reserve')
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
            require(shared.inputs()[0]==identity and sha(PLAN)==result['plan_sha256'], 'inputs33 changed')
            source(plan['source'])
            poll(True)
            require(result['cleanup_complete'], 'cleanup33 incomplete')
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
        require(not PLAN.exists() and not OUTPUT.exists(), 'single33 attempt only')
        identity,_ = shared.inputs()
        value = {'scope':scope(),'source':source(),'inputs':identity}
        PLAN.parent.mkdir(parents=True,exist_ok=True)
        with PLAN.open('x') as stream:
            json.dump(value,stream,indent=2)
    print(json.dumps({'status':'PREPARED' if args.prepare else 'PLAN_ONLY',**scope()},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
