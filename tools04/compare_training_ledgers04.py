#!/usr/bin/env python3
"""Compare what the three trainings were rewarded and penalised for, term by term.

The frozen evaluation traces cannot reproduce every reward term (leg pose, torque and
height error need joint angles and torques they do not store). The training ledgers
record the mean per-tick contribution of every term, so this reads them instead.

Two limits apply to every number. It is an average over the whole training command
distribution, standing and turning included, not the drive phase alone. And it
describes the end of training only through the mean of the last rollouts. Reads
ledgers and progress tables only; imports no simulator and executes no physics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

TICKS_PER_SECOND = 50.
LAST_ROLLOUTS = 20
CONVERGENCE_FRACTION = .25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ledger(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def per_second_means(rows, last=LAST_ROLLOUTS):
    """Mean contribution per second of each term over the last `last` rollouts; NaN where a term is absent."""
    tail = rows[-last:]
    terms = sorted({t for r in rows for t in r['reward_contribution_mean_per_tick']})
    return {t: float(np.mean([r['reward_contribution_mean_per_tick'].get(t, np.nan) for r in tail]))
            * TICKS_PER_SECOND for t in terms}


def difference(a, b):
    """b minus a per term; None when either side lacks the term, so a new term is not read as zero."""
    return {t: (None if (np.isnan(a.get(t, np.nan)) or np.isnan(b.get(t, np.nan))) else b[t]-a[t])
            for t in sorted(set(a) | set(b))}


def tracking_shortfall_ratio(new_term_per_second, shortfall_per_second):
    """How many times larger the new term's cost is than the tracking it gave up; None when no shortfall."""
    if shortfall_per_second is None or shortfall_per_second >= 0:
        return None
    return abs(new_term_per_second)/abs(shortfall_per_second)


def read_progress(path, column='rollout/ep_rew_mean'):
    values = []
    for row in csv.DictReader(Path(path).open()):
        values.append(float(row[column]) if row[column] not in ('', None) else np.nan)
    return np.array(values)


def late_trend(values, fraction=CONVERGENCE_FRACTION, edge=20):
    """Relative change of the mean over the last `edge` iterations against the first `edge` of the final fraction."""
    y = np.asarray(values, dtype=float)
    tail = y[int(len(y)*(1.-fraction)):]
    first, last = np.nanmean(tail[:edge]), np.nanmean(tail[-edge:])
    return {'iterations_in_tail': int(len(tail)), 'first_mean': float(first), 'last_mean': float(last),
            'relative_change': float((last-first)/(abs(first)+1e-12))}


def progress_at(values, fraction, width=8):
    y = np.asarray(values, dtype=float)
    return float(np.nanmean(y[max(0, int(len(y)*fraction)-width):int(len(y)*fraction)+1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train', action='append', default=[], metavar='LABEL=DIR')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    runs = dict(item.split('=', 1) for item in args.train)
    means, trends, curves, hashes = {}, {}, {}, {}
    for label, directory in runs.items():
        directory = Path(directory)
        means[label] = per_second_means(read_ledger(directory/'ledger.jsonl'))
        rewards = read_progress(directory/'progress.csv')
        trends[label] = late_trend(rewards)
        curves[label] = {f'{int(f*100)}%': progress_at(rewards, f) for f in (.25, .5, .75, 1.)}
        hashes[label] = {'ledger_sha256': sha(directory/'ledger.jsonl'),
                         'progress_sha256': sha(directory/'progress.csv')}
    base = means[args.baseline]
    versus = {label: difference(base, means[label]) for label in means if label != args.baseline}
    notes = {}
    for label, diff in versus.items():
        new_terms = [t for t, v in diff.items() if v is None and not np.isnan(means[label].get(t, np.nan))]
        notes[label] = {'terms_only_in_this_version': new_terms,
                        'new_term_cost_over_tracking_shortfall': {
                            t: tracking_shortfall_ratio(means[label][t], diff.get('vx_tracking')) for t in new_terms}}
    report = {'format': 'd1-contract04-training-ledger-comparison-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'transitions_consumed': 0,
              'scope': 'whole training command distribution, mean of the last rollouts; not the drive phase',
              'last_rollouts': LAST_ROLLOUTS, 'baseline': args.baseline,
              'per_second_by_term': means, 'difference_from_baseline': versus, 'notes': notes,
              'return_curve_at_fraction_of_training': curves, 'late_trend': trends,
              'inputs': hashes, 'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'), 'versions': sorted(means)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
