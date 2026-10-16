#!/usr/bin/env python3
"""Judge contract 08 from its receipts: gates first, then the pre-registered predictions.

Reuses the gate logic that already decides the contract 04 ledger, so a task passes
here exactly when it would pass there, and takes the thresholds from the frozen
qualification gates rather than restating them. The 8x policy is judged against the
frozen 03d 1x evaluation with the same evaluator, seed and delay. Reads receipts only;
executes no physics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))
sys.path.insert(0, str(ROOT/'tools08'))

from build_final_ledger04 import qualification_verdict, task_gate_summary  # noqa: E402
from release_pilot04b import QUALIFICATION_GATES  # noqa: E402
import release_scale08 as release  # noqa: E402

TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_episodes(eval_dir):
    rows = [json.loads(line) for line in (Path(eval_dir)/'episodes.jsonl').read_text().splitlines() if line.strip()]
    by_task = {row['sequence']: row for row in rows if row['delay_ticks'] == 0 and row['reset_seed'] == 7001}
    missing = [t for t in TASKS if t not in by_task]
    if missing:
        raise SystemExit(f'evaluation lacks tasks {missing}')
    return by_task


def task_results(episodes, gates=QUALIFICATION_GATES):
    out = {}
    for task in TASKS:
        summary = task_gate_summary(episodes[task], gates)
        passed = not summary['failed_gates']
        if passed is not bool(episodes[task]['reference_thresholds_met']):
            raise ValueError(f'{task}: gate summary disagrees with the evaluator flag')
        out[task] = {'passed': passed, 'gates_passed': summary['gates_passed'],
                     'gates_applicable': summary['gates_applicable'],
                     'blocking_gate': summary['blocking_gate'],
                     'stop_displacement_m': episodes[task]['stop_displacement_m'],
                     'stable_wz_mae_rad_s': episodes[task]['stable_wz_mae_rad_s'],
                     'early_failure': episodes[task]['failure'] or episodes[task]['done_reason'] != 'HORIZON_12S'}
    return out


def late_return_change(progress_csv, fraction=.25, edge=20):
    values = []
    for row in csv.DictReader(Path(progress_csv).open()):
        cell = row.get('rollout/ep_rew_mean')
        values.append(float(cell) if cell not in ('', None) else math.nan)
    tail = values[int(len(values)*(1-fraction)):]
    first = [v for v in tail[:edge] if not math.isnan(v)]
    last = [v for v in tail[-edge:] if not math.isnan(v)]
    if not first or not last:
        return None
    a, b = sum(first)/len(first), sum(last)/len(last)
    return {'iterations': len(values), 'first_mean': a, 'last_mean': b, 'relative_change': (b-a)/abs(a)}


def judge(new, base, late, launcher_report, predictions=release.PREDICTIONS):
    by_id = {p['id']: p for p in predictions}
    if launcher_report.get('equivalence') != 'PASS':
        return {p: 'NOT_EVALUABLE' for p in by_id} | {'E1': 'NOT_MET'}
    passed = sum(new[t]['passed'] for t in TASKS)
    out = {'E1': 'MET'}
    out['P1'] = 'MET' if passed >= by_id['P1']['threshold'] else 'NOT_MET'
    out['P2'] = 'MET' if new['forward_stop']['stop_displacement_m'] < by_id['P2']['threshold'] else 'NOT_MET'
    wz = new['turn_stop']['stable_wz_mae_rad_s']
    out['P3'] = 'NOT_EVALUABLE' if wz is None else ('MET' if wz < by_id['P3']['threshold'] else 'NOT_MET')
    out['P4'] = 'REPORTED' if late else 'NOT_EVALUABLE'
    out['P5'] = 'MET' if new['reverse_stop']['passed'] else ('NOT_MET' if base['reverse_stop']['passed'] else 'NOT_EVALUABLE')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-dir', type=Path, required=True, help='8x policy, frozen evaluator')
    parser.add_argument('--baseline-eval-dir', type=Path, required=True, help='03d 1x, frozen evaluator')
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--launcher-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    new, base = task_results(read_episodes(args.eval_dir)), task_results(read_episodes(args.baseline_eval_dir))
    launcher_report = json.loads(args.launcher_report.read_text())
    late = late_return_change(args.train_dir/'progress.csv')
    verdicts = judge(new, base, late, launcher_report)
    passed = sum(new[t]['passed'] for t in TASKS)
    early = sum(new[t]['early_failure'] for t in TASKS)
    report = {'format': 'd1-contract08-verdict-v1', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'predictions': verdicts,
              'qualification': {'tasks_passed': passed, 'tasks_total': len(TASKS), 'early_failures': early,
                                'overall': qualification_verdict(passed, len(TASKS), early, len(TASKS))},
              'tasks_8x': new, 'tasks_03d_1x': base, 'late_return_change': late,
              'equivalence': {k: launcher_report.get(k) for k in ('equivalence', 'rollouts_compared', 'rollouts_required')},
              'inputs': {'eval_episodes_sha256': sha(args.eval_dir/'episodes.jsonl'),
                         'baseline_episodes_sha256': sha(args.baseline_eval_dir/'episodes.jsonl'),
                         'progress_sha256': sha(args.train_dir/'progress.csv'),
                         'launcher_report_sha256': sha(args.launcher_report)},
              'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verdict.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'predictions': verdicts, 'qualification': report['qualification']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
