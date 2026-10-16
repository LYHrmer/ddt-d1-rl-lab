#!/usr/bin/env python3
"""Judge contract 09 from its receipts, with the rules written before its result exists.

Q1 and Q2 come from the training replay receipt (tools08/replay_training_episodes08.py),
the qualification predictions from the frozen evaluator's episodes through the contract 08
gate logic, which already handles a task that ends early. The thresholds are the ones
frozen in the contract 09 release. Reads receipts and traces only; executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for sub in ('tools08', 'tools09'):
    sys.path.insert(0, str(ROOT/sub))

import describe_eval08 as describe  # noqa: E402
import release_scale09 as release  # noqa: E402
import verify_scale08 as gates  # noqa: E402

LAST_QUARTER = slice(12, 16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def last_quarter_std(replay):
    bins = replay['ppo_timeline']
    if len(bins) != 16:
        raise SystemExit('the replay receipt must have 16 slices of training')
    values = [b['train/std'] for b in bins[LAST_QUARTER]]
    if any(v is None for v in values):
        raise SystemExit('a last-quarter slice has no std')
    return sum(values)/len(values)


def last_quarter_forward_ends(replay):
    return sum(b['early_ends_by_segment'].get('drive_forward', 0) for b in replay['timeline'][LAST_QUARTER])


def drive_speeds(eval_dir):
    out = {}
    for task, direction in (('forward_stop', 1.), ('reverse_stop', -1.)):
        with np.load(Path(eval_dir)/f'{task}_delay0_seed7001.npz') as handle:
            out[task] = describe.describe({k: handle[k] for k in handle.files}, direction)['drive_last_second_speed_along_command_m_s']
    return out


def judge(std, forward_ends, tasks, launcher_report, predictions=release.PREDICTIONS):
    by_id = {p['id']: p for p in predictions}
    if launcher_report.get('equivalence') != 'PASS':
        return {'E1': 'NOT_MET'} | {p: 'NOT_EVALUABLE' for p in by_id}
    passed = sum(t['passed'] for t in tasks.values())
    return {'E1': 'MET',
            'Q1': 'MET' if std < by_id['Q1']['threshold'] else 'NOT_MET',
            'Q2': 'MET' if forward_ends <= by_id['Q2']['threshold'] else 'NOT_MET',
            'Q3': 'MET' if passed >= by_id['Q3']['threshold'] else 'NOT_MET',
            'Q4': 'NOT_MET' if tasks['forward_stop']['early_failure'] else 'MET',
            'Q5': 'REPORTED'}


def interpretation(verdicts, table=release.INTERPRETATION):
    if verdicts.get('Q1') == 'NOT_MET':
        return table['Q1 not met']
    if verdicts.get('Q1') == 'MET':
        return table['Q1 met, Q2 met' if verdicts['Q2'] == 'MET' else 'Q1 met, Q2 not met']
    return 'not evaluable: the equivalence gate did not pass'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--replay', type=Path, required=True, help='training replay report.json')
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--launcher-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text())
    tasks = gates.task_results(gates.read_episodes(args.eval_dir))
    launcher_report = json.loads(args.launcher_report.read_text())
    std, forward_ends = last_quarter_std(replay), last_quarter_forward_ends(replay)
    verdicts = judge(std, forward_ends, tasks, launcher_report)
    passed = sum(t['passed'] for t in tasks.values())
    early = sum(t['early_failure'] for t in tasks.values())
    report = {'format': 'd1-contract09-verdict-v1', 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'predictions': verdicts, 'interpretation': interpretation(verdicts),
              'readings': {'last_quarter_mean_std': std, 'last_quarter_forward_drive_early_ends': forward_ends,
                           'drive_last_second_speed_m_s': drive_speeds(args.eval_dir)},
              'qualification': {'tasks_passed': passed, 'tasks_total': 3, 'early_failures': early,
                                'overall': gates.qualification_verdict(passed, 3, early, 3)},
              'tasks': tasks, 'baseline_08': release.BASELINE_08,
              'inputs': {'replay_sha256': sha(args.replay), 'episodes_sha256': sha(args.eval_dir/'episodes.jsonl'),
                         'launcher_report_sha256': sha(args.launcher_report)},
              'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verdict.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'predictions': verdicts, 'interpretation': report['interpretation'],
                      'qualification': report['qualification']}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
