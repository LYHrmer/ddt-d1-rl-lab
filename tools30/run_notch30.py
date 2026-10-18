"""One fixed causal-notch candidate, completed25 weights, original task gates."""
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

from tools26 import common26 as base  # noqa: E402
from tools21.closeout_completed18 import evaluate, matrix, known_counts  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402
from tools27.resource_guard27 import snapshot_memory, check_memory  # noqa: E402
from tools30.notch30 import NotchFilter30  # noqa: E402

require, sha, read_json, atomic_json = base.require, base.sha, base.read_json, base.atomic_json
COUNTS = base.expected_counts
OUTPUT = ROOT / 'evidence/causal_notch30/execution_01'
PLAN = ROOT / 'evidence/causal_notch30/plan_01.json'
PARENT = ROOT / base.BASE_ROOT / 'execution_01/random'
UNIT = 'd1-causal-notch30-owned01-20261018-01.service'
SCOPE = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}'
SOURCES = tuple(dict.fromkeys((*base.SOURCES, 'tools27/resource_guard27.py',
    'tools30/notch30.py', 'tools30/run_notch30.py', 'tests/test_notch30.py',
    'tests/test_run_notch30.py', 'docs/causal_notch30.md', '.github/workflows/static-checks.yml')))


def source_identity(proof=None):
    if proof is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        proof = {'git_commit': commit, 'sources_sha256': {name: sha(ROOT/name) for name in SOURCES}}
    require(set(proof['sources_sha256']) == set(SOURCES), 'source closure differs')
    for name, digest in proof['sources_sha256'].items():
        path = ROOT/name
        require(not path.is_symlink() and sha(path) == digest, 'source changed: '+name)
        blob = subprocess.check_output(['git', 'show', proof['git_commit']+':'+name], cwd=ROOT,
                                       stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit source first: '+name)
    return proof


def inputs():
    descriptor, release, protocol = base.base_inputs()
    outer = read_json(PARENT.parent/'report.json')
    bound = {}
    for kind in ('export', 'cpp', 'formal'):
        record = outer['stages'][kind+'_random']
        require(record['status'] == 'COMPLETE' and record['cleanup_complete'] is True,
                'incomplete baseline '+kind)
        require(bool(record['artifacts_sha256']), 'missing baseline artifacts')
        for name, digest in record['artifacts_sha256'].items():
            path = Path(name)
            require(path.is_relative_to(PARENT/kind) and not path.is_symlink() and sha(path) == digest,
                    'baseline artifact changed: '+name)
            bound[name] = digest
    require(str(PARENT/'export/policy.onnx') in bound, 'baseline ONNX unbound')
    identity = {'base_checkpoint': descriptor['base_checkpoint'],
                'base_protocol': descriptor['base_protocol'], 'completed25_artifacts_sha256': bound,
                'baseline_provenance': descriptor['baseline_provenance'],
                'runtime_dependencies_sha256': descriptor['runtime_dependencies_sha256']}
    bundle = {'old_release': release, 'protocol': protocol,
              'identity': {'training_identity': {'checkpoint_sha256': descriptor['base_checkpoint']['sha256']}}}
    return identity, bundle


def fixed():
    return {'format': 'd1-causal-notch30-plan-v1', 'output': str(OUTPUT),
            'filter': NotchFilter30().receipt(), 'candidate_count': 1,
            'training_transitions': 0, 'budget': COUNTS(43200),
            'formal_matrix': matrix('formal'), 'holdout_matrix': matrix('holdout'),
            'holdout_requires': 'all_original36_pass_and_full_horizon',
            'unit': UNIT, 'owned_scope': SCOPE, 'memory_high_bytes': 2 << 30,
            'memory_max_bytes': 3 << 30, 'memory_swap_max_bytes': 0,
            'start_effective_bytes': 4 << 30, 'reserve_effective_bytes': 2 << 30,
            'admission_stable_s': 10, 'admission_timeout_s': 300, 'execution_wall_s': 900,
            'retry_or_resume': False, 'automatic_model_replacement': False,
            'new_cpp_closedloop_proven': False}


def prepare():
    require(not PLAN.exists() and not OUTPUT.exists(), 'single new attempt only')
    identity, _ = inputs()
    plan = {**fixed(), 'inputs': identity, 'source_identity': source_identity()}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    with PLAN.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return plan


def load_plan():
    plan = read_json(PLAN)
    expected = fixed()
    require(set(plan) == set(expected) | {'inputs', 'source_identity'} and
            json.dumps({k: plan[k] for k in expected}, sort_keys=True) == json.dumps(expected, sort_keys=True),
            'fixed30 plan differs')
    source_identity(plan['source_identity'])
    identity, bundle = inputs()
    require(identity == plan['inputs'], 'completed25 inputs differ')
    return plan, bundle


class FilteredRuntime:
    """Wrap one actor call per tick; episode identity never controls the filter."""
    def __init__(self, original):
        self.original = original
        self.filter = NotchFilter30()
        self.rows = []
        self.case = None
        self.backend = {**original.backend, 'kind': 'python_ort_cpu_fixed25_causal_notch30',
                        'action_postprocess': self.filter.receipt(),
                        'new_cpp_closedloop_proven': False}

    def make_env(self, name, sequence, delay, seed):
        require(not self.rows, 'previous episode not verified')
        self.filter.reset()
        self.case = (name, delay, seed)
        return self.original.make_env(name, sequence, delay, seed)

    def predict(self, observation):
        import numpy as np
        require(self.case is not None and len(self.rows) < 600, 'episode inference budget exceeded')
        raw = np.asarray(self.original.predict(observation), dtype=np.float32).copy()
        issued = np.asarray(self.filter.step(raw), dtype=np.float32)
        self.rows.append((raw, self.filter.last_linear, issued.copy()))
        return issued

    def arrays(self):
        import numpy as np
        return {name: np.asarray([r[i] for r in self.rows], dtype=dtype).reshape(-1, 16)
                for i, (name, dtype) in enumerate((('raw', np.float32), ('linear', np.float64),
                                                  ('issued', np.float32)))}

    def verify(self, output, item, helpers):
        import numpy as np
        from tools17.evaluate_formal17 import verify_trace
        verify_trace(output, item, helpers)
        require(self.case == (item['sequence'], item['delay_ticks'], item['reset_seed']), 'wrong filter case')
        path = output/(Path(item['trace_file']).stem+'_filter.npz')
        np.savez_compressed(path, **self.arrays())
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        with np.load(output/item['trace_file'], allow_pickle=False) as trace:
            receipt = verify_filter(arrays, trace['issued_action'], trace['applied_action'], item['delay_ticks'])
        require(receipt == self.filter.receipt(), 'filter replay receipt differs')
        item['filter'] = {'sidecar': path.name, 'sha256': sha(path), 'replay_exact': True,
                          'issued_matches_physical_trace': True, 'fifo_exact': True, **receipt}
        self.rows.clear()


def verify_filter(arrays, issued_trace, applied_trace, delay):
    import numpy as np
    require(set(arrays) == {'raw', 'linear', 'issued'} and type(delay) is int and 0 <= delay <= 3,
            'invalid filter sidecar')
    n = len(arrays['raw'])
    require(1 <= n <= 600, 'invalid filter horizon')
    for name, dtype in (('raw', np.float32), ('linear', np.float64), ('issued', np.float32)):
        require(arrays[name].shape == (n, 16) and arrays[name].dtype == dtype and
                bool(np.isfinite(arrays[name]).all()), 'invalid sidecar '+name)
    f = NotchFilter30()
    for i, raw in enumerate(arrays['raw']):
        issued = np.asarray(f.step(raw), dtype=np.float32)
        require(np.array_equal(issued, arrays['issued'][i]) and
                np.array_equal(np.asarray(f.last_linear), arrays['linear'][i]), 'filter recurrence differs')
    require(np.array_equal(arrays['issued'], issued_trace), 'filter output differs from physical issue')
    expected = np.zeros_like(arrays['issued'])
    if delay == 0:
        expected[:] = arrays['issued']
    elif n > delay:
        expected[delay:] = arrays['issued'][:-delay]
    require(np.array_equal(expected, applied_trace), 'physical FIFO differs')
    return f.receipt()


def eligible(report):
    rows = report.get('cases', [])
    return (report.get('status') == 'COMPLETE' and report.get('cleanup_complete') is True and
            report.get('actual_counts') == COUNTS(21600) and report.get('all_36_reference_pass') is True and
            len(rows) == 36 and
            {(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in rows} == set(matrix('formal')) and
            all(r['reference_thresholds_met'] is True and r['failure'] is False and
                r['done_reason'] == 'HORIZON_12S' and r['policy_transitions'] == 600 for r in rows))


def execute():
    plan, bundle = load_plan()
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'owned process must use Nice10 or higher')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    plan_sha = sha(PLAN)
    started, admitted, last = time.monotonic(), None, -float('inf')
    report = {'format': 'd1-causal-notch30-result-v1', 'status': 'INCOMPLETE', 'error': None,
              'plan_sha256': plan_sha, 'source_identity': plan['source_identity'],
              'stages': {}, 'actual_counts': COUNTS(0), 'known_actual_counts': COUNTS(0),
              'cleanup_complete': True, 'inputs_unchanged': False, 'training_transitions': 0,
              'python_original_and_holdout_qualified': False, 'automatic_model_replacement': False,
              'new_cpp_closedloop_proven': False}

    def poll(force=False):
        nonlocal last
        now = time.monotonic()
        require(now - (admitted or started) < (900 if admitted is not None else 300), 'wall budget exceeded')
        if force or now-last >= 1:
            sample = snapshot_memory(SCOPE)
            sample['elapsed_s'] = now-started
            with (OUTPUT/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            check_memory(sample)
            require(sample['effective_available_bytes'] >= 2 << 30, 'host runtime reserve')
            last = now
            return sample['effective_available_bytes']
        return None

    with termination_guard() as control:
        try:
            atomic_json(OUTPUT/'report.json', report)
            stable = None
            while True:
                available = poll(True)
                now = time.monotonic()
                stable = (stable if stable is not None else now) if available >= 4 << 30 else None
                if stable is not None and now-stable >= 10:
                    admitted = now
                    break
                time.sleep(1)
            from tools17.evaluate_formal17 import runtime
            for stage in ('formal', 'holdout'):
                if stage == 'holdout' and not eligible(report['stages']['formal']):
                    report['stages'][stage] = {'status': 'SKIPPED_ORIGINAL36_FAILED', 'actual_counts': COUNTS(0)}
                    break
                directory = OUTPUT/stage
                directory.mkdir()
                child = {'status': 'INCOMPLETE', 'error': None, 'cases': [], 'cleanup_complete': True,
                         'all_36_reference_pass': False, 'actual_counts': COUNTS(0), 'known_actual_counts': COUNTS(0)}
                report['stages'][stage] = child
                atomic_json(directory/'protocol.json', {'plan_sha256': plan_sha, 'stage': stage,
                    'source_identity': plan['source_identity'], 'budget': COUNTS(21600)})
                rt = None
                def factory(*args):
                    nonlocal rt
                    rt = FilteredRuntime(runtime(*args))
                    return rt
                def verifier(*args):
                    rt.verify(*args)
                try:
                    evaluate(stage, bundle, PARENT, directory, child, poll,
                             runtime_factory=factory, verifier=verifier)
                    require(child['cleanup_complete'], 'environment cleanup incomplete')
                    child['status'] = 'COMPLETE'
                except BaseException as exc:
                    child['error'] = f'{type(exc).__name__}: {exc}'
                    raise
                finally:
                    atomic_json(directory/'report.json', child)
                    report['known_actual_counts'] = known_counts(report['stages'])
                    report['actual_counts'] = (None if any(r['actual_counts'] is None for r in report['stages'].values())
                                               else report['known_actual_counts'])
                    report['cleanup_complete'] = all(r.get('cleanup_complete', True) for r in report['stages'].values())
                    atomic_json(OUTPUT/'report.json', report)
                    if rt is not None and rt.rows:
                        import numpy as np
                        np.savez_compressed(directory/'incomplete_filter.npz', **rt.arrays())
            require(inputs()[0] == plan['inputs'] and sha(PLAN) == plan_sha, 'inputs changed during run')
            source_identity(plan['source_identity'])
            poll(True)
            report['inputs_unchanged'] = True
            report['status'] = 'COMPLETE'
            holdout = report['stages'].get('holdout', {})
            report['python_original_and_holdout_qualified'] = (eligible(report['stages']['formal']) and
                holdout.get('status') == 'COMPLETE' and holdout.get('all_36_reference_pass') is True and
                holdout.get('actual_counts') == COUNTS(21600))
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (OUTPUT/'error.txt').write_text(traceback.format_exc())
        finally:
            control['defer'] = True
            report['elapsed_s'] = time.monotonic()-started
            if control['received']:
                report.update(status='INCOMPLETE', error=control['received'])
            atomic_json(OUTPUT/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = execute()
        print(json.dumps({key: result[key] for key in ('status', 'error', 'actual_counts',
                         'python_original_and_holdout_qualified')}, indent=2))
        return 0 if result['status'] == 'COMPLETE' else 1
    if args.prepare:
        prepare()
    print(json.dumps({'status': 'PREPARED' if args.prepare else 'PLAN_ONLY', 'plan': str(PLAN),
                      'budget_max': COUNTS(43200), 'training_transitions': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
