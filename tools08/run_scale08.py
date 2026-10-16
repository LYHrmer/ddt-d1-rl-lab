#!/usr/bin/env python3
"""Contract 08: run the scale training and check it against the frozen 03d run as it goes.

The training itself is tools08/train_d1_ppo08.py, a byte copy of the frozen trainer with
three lines changed. Started from the same seed, its first 1,048,576 steps should repeat
the frozen 03d training exactly, so this launcher compares every rollout the trainer
writes with the frozen ledger and, at the first mismatch, interrupts the trainer. A run
that diverged would confound "more steps" with "a different entry point", and stopping
early keeps the rest of the budget unspent.

Step counts, seed and output directories are fixed by the kind of run, not taken from
the command line. Pure helpers at the top run without a simulator; `run` starts physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FROZEN_LEDGER = 'evidence/career_contract_03/pilot03d_01/ledger.jsonl'
CONFIG = 'training/configs/pilot03d.json'
ENTRY = 'tools08/train_d1_ppo08.py'
RELATIVE_TOLERANCE = 1e-9
PLANS = {
    'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': 8,
                  'output': 'evidence/career_contract_08/scale08_preflight_01'},
    'scale': {'steps': 8388608, 'max_wall': 9000, 'compare_rollouts': 256,
              'output': 'evidence/career_contract_08/scale08_train_01'},
}
SEED = 101
ENVS = 8
THREAD_ENV = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
              'PYTHONDONTWRITEBYTECODE': '1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, rel=RELATIVE_TOLERANCE):
    """Floats equal to a relative tolerance; zero only matches zero."""
    if a == b:
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= rel*max(abs(a), abs(b))


def compare_rollout(new, frozen, rel=RELATIVE_TOLERANCE):
    """Differences between one rollout of the new run and the same rollout of the frozen run.

    Timing and memory fields differ between any two runs and are ignored. What the
    simulation and the learner produced must agree: the rollout index, the transition
    count, the native and reset counters, and every per-term reward contribution.
    """
    problems = []
    for key in ('rollout', 'transitions'):
        if new.get(key) != frozen.get(key):
            problems.append(f'{key}: {new.get(key)} != {frozen.get(key)}')
    if new.get('actual_counts') != frozen.get('actual_counts'):
        problems.append(f"actual_counts: {new.get('actual_counts')} != {frozen.get('actual_counts')}")
    a, b = new.get('reward_contribution_mean_per_tick', {}), frozen.get('reward_contribution_mean_per_tick', {})
    if set(a) != set(b):
        problems.append(f'reward terms differ: {sorted(set(a) ^ set(b))}')
    for term in sorted(set(a) & set(b)):
        if not close(float(a[term]), float(b[term]), rel):
            problems.append(f'{term}: {a[term]!r} != {b[term]!r}')
    return problems


def complete_lines(text):
    """Only lines ending in a newline: the trainer may be halfway through writing the last one."""
    return [line for line in text.split('\n')[:-1] if line.strip()]


def read_frozen(path, count):
    rows = [json.loads(line) for line in complete_lines(Path(path).read_text())]
    if len(rows) < count:
        raise SystemExit(f'frozen ledger has {len(rows)} rollouts, fewer than the {count} to compare')
    return rows[:count]


def trainer_env(base=None):
    """The environment the trainer runs in: the caller's, with threads and import path fixed."""
    return dict(os.environ if base is None else base, **THREAD_ENV,
                PYTHONPATH=f"{ROOT/'.python-deps'}:{ROOT/'training'}")


def command(kind, output, python=sys.executable):
    plan = PLANS[kind]
    return [python, '-B', str(ROOT/ENTRY), '--output', str(output), '--config', str(ROOT/CONFIG),
            '--arm', 'B', '--seed', str(SEED), '--envs', str(ENVS), '--steps', str(plan['steps']),
            '--max-wall', str(plan['max_wall']), '--kind', kind]


def verify_release(release_path, root=ROOT):
    release = json.loads(Path(release_path).read_text())
    for path, digest in release['frozen_inputs'].items():
        target = Path(root)/path
        if not target.is_file() or sha(target) != digest:
            raise SystemExit(f'frozen input changed or missing: {path}')
    if release['launcher_sha256'] != sha(__file__):
        raise SystemExit('this launcher differs from the one frozen in the release receipt')
    return release


def require_preflight_passed(path):
    report = json.loads(Path(path).read_text())
    if report.get('kind') != 'preflight' or report.get('equivalence') != 'PASS':
        raise SystemExit('the scale run needs a preflight receipt whose equivalence check passed')
    return report


def run(kind, release_path, preflight_receipt=None, python=sys.executable, poll_s=2.):
    verify_release(release_path)
    if kind == 'scale':
        require_preflight_passed(preflight_receipt)
    plan = PLANS[kind]
    output = ROOT/plan['output']
    if output.exists():
        raise SystemExit(f'output already exists: {output}')
    frozen = read_frozen(ROOT/FROZEN_LEDGER, plan['compare_rollouts'])
    receipt_dir = output.parent/(output.name+'_launcher')
    receipt_dir.mkdir(parents=True, exist_ok=False)
    env = trainer_env()
    started = time.monotonic()
    process = subprocess.Popen(command(kind, output, python), cwd=ROOT, env=env,
                               stdout=(receipt_dir/'trainer_stdout.txt').open('w'), stderr=subprocess.STDOUT)
    compared, mismatch, interrupted = 0, None, False
    ledger = output/'ledger.jsonl'
    while process.poll() is None:
        if compared < len(frozen) and ledger.is_file():
            rows = complete_lines(ledger.read_text())
            while compared < min(len(rows), len(frozen)):
                problems = compare_rollout(json.loads(rows[compared]), frozen[compared])
                if problems:
                    mismatch = {'rollout': compared+1, 'problems': problems[:10]}
                    break
                compared += 1
            if mismatch and not interrupted:
                process.send_signal(signal.SIGINT)
                interrupted = True
        time.sleep(poll_s)
    returncode = process.wait()
    if mismatch is None and compared < len(frozen) and ledger.is_file():
        rows = complete_lines(ledger.read_text())
        while compared < min(len(rows), len(frozen)):
            problems = compare_rollout(json.loads(rows[compared]), frozen[compared])
            if problems:
                mismatch = {'rollout': compared+1, 'problems': problems[:10]}
                break
            compared += 1
    result_path = output/'result.json'
    result = json.loads(result_path.read_text()) if result_path.is_file() else None
    equivalence = ('FAIL' if mismatch else 'PASS' if compared == len(frozen) else 'INCOMPLETE')
    report = {'format': 'd1-contract08-launcher-v1', 'kind': kind,
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'command': command(kind, output, 'python3'), 'plan': plan,
              'equivalence': equivalence, 'rollouts_compared': compared,
              'rollouts_required': len(frozen), 'first_mismatch': mismatch,
              'interrupted_by_launcher': interrupted, 'trainer_returncode': returncode,
              'trainer_result': result, 'wall_s': time.monotonic()-started,
              'release_sha256': sha(release_path), 'launcher_sha256': sha(__file__)}
    (receipt_dir/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: report[k] for k in ('kind', 'equivalence', 'rollouts_compared',
                                             'interrupted_by_launcher', 'trainer_returncode')}, indent=2))
    return 0 if equivalence == 'PASS' and result and result['status'] == 'COMPLETE' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=sorted(PLANS))
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--preflight-receipt', type=Path, help='required for the scale run')
    args = parser.parse_args()
    return run(args.kind, args.release, args.preflight_receipt)


if __name__ == '__main__':
    raise SystemExit(main())
