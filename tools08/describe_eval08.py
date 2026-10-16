#!/usr/bin/env python3
"""Describe what each task of an evaluation actually did, beyond its pass or fail.

A displacement gate can be met by moving back and forth, and a task can fail by falling
long before its stop window. For each task this receipt reports the path walked after the
cut against the straight-line displacement, how often the motion reversed, how fast the
robot really drove before the cut, its lean then, and how the lean evolved until the end.
Reads evaluation traces only; imports no simulator and executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

CUT = 400
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def after_cut_motion(boundary_xy, cut=CUT):
    xy = np.asarray(boundary_xy, dtype=float)
    if len(xy)-1 <= cut:
        return None
    steps = np.diff(xy[cut:], axis=0)
    path = float(np.hypot(*steps.T).sum())
    euclid = float(np.hypot(*(xy[-1]-xy[cut])))
    sx = np.sign(steps[:, 0]); sx = sx[sx != 0]
    return {'euclidean_m': euclid, 'path_m': path, 'path_over_euclidean': path/euclid if euclid else None,
            'x_direction_reversals': int(np.sum(sx[1:] != sx[:-1]))}


def describe(trace, direction, cut=CUT):
    vx = np.asarray(trace['body_vx_m_s'], dtype=float)
    pitch = np.asarray(trace['base_pitch_rad'], dtype=float)
    n = len(vx)
    return {'ticks': n, 'ended_early': n < 600,
            'drive_last_second_speed_along_command_m_s': float(direction*vx[cut-50:cut].mean()) if direction else None,
            'drive_last_second_pitch_rad': float(np.median(pitch[cut-50:cut])),
            'pitch_at_end_rad': float(pitch[-1]),
            'peak_abs_pitch_after_cut_rad': float(np.abs(pitch[cut:]).max()) if n > cut else None,
            'after_cut': after_cut_motion(trace['boundary_xy_m'], cut)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', action='append', default=[], metavar='LABEL=DIR')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    out, inputs = {}, {}
    for label, directory in (item.split('=', 1) for item in args.eval):
        out[label] = {}
        for task, direction in zip(TASKS, (1., -1., 0.)):
            path = Path(directory)/f'{task}_delay0_seed7001.npz'
            with np.load(path) as handle:
                out[label][task] = describe({k: handle[k] for k in handle.files}, direction)
            inputs[f'{label}/{task}'] = sha(path)
    report = {'format': 'd1-contract08-evaluation-description-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'cut_tick': CUT, 'versions': out, 'inputs': inputs,
              'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
