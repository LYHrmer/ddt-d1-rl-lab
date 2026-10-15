#!/usr/bin/env python3
"""How 04b's body lean depends on speed, measured against its own resting lean.

An earlier reading divided plateau pitch by speed and called the result a constant of
proportionality. That folded in the lean the robot already has standing still, so this
receipt reports pitch relative to the resting baseline, the slope between the measured
plateaus and the intercept that slope implies, so the shape of the relation is visible
rather than assumed. It also records the body state at the moment the wheels were forced
to zero in two different measurements, because their braking differed. Reads traces only;
imports no simulator and executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

REST = (25, 100)
PLATEAU = (350, 400)
SETTLED = (550, 600)
ENTRY = (395, 400)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def median(values, window):
    return float(np.median(np.asarray(values, dtype=float)[window[0]:window[1]]))


def entry_speed(trace, window=ENTRY):
    lo, hi = window
    return float(np.hypot(np.asarray(trace['body_vx_m_s'], dtype=float)[lo:hi],
                          np.asarray(trace['body_vy_m_s'], dtype=float)[lo:hi]).mean())


def lean_point(trace):
    rest, plateau = median(trace['base_pitch_rad'], REST), median(trace['base_pitch_rad'], PLATEAU)
    speed = entry_speed(trace)
    return {'speed_m_s': speed, 'rest_pitch_rad': rest, 'plateau_pitch_rad': plateau,
            'settled_pitch_rad': median(trace['base_pitch_rad'], SETTLED),
            'lean_above_rest_rad': plateau-rest,
            'ratio_including_rest_rad_per_m_s': plateau/speed if speed else None,
            'ratio_above_rest_rad_per_m_s': (plateau-rest)/speed if speed else None}


def two_point_line(p1, p2):
    """Slope between two plateaus and the pitch it predicts at zero speed."""
    dv = p2['speed_m_s']-p1['speed_m_s']
    if dv == 0:
        return None
    slope = (p2['plateau_pitch_rad']-p1['plateau_pitch_rad'])/dv
    return {'slope_rad_per_m_s': slope, 'intercept_at_zero_speed_rad': p1['plateau_pitch_rad']-slope*p1['speed_m_s']}


def through_rest(line, rest, tolerance=.02):
    """Does the line through the two plateaus also pass through the resting lean?"""
    return None if line is None else bool(abs(line['intercept_at_zero_speed_rad']-rest) <= tolerance)


def release_state_from_probe(drive_rows):
    last = drive_rows[-1]
    return {'tick': last['tick'], 'speed_m_s': float(np.hypot(last['vx'], last['vy'])),
            'tilt_rad': float(last['tilt']), 'height_m': float(last['height'])}


def release_state_from_trace(trace, cut):
    k = cut-1
    return {'tick': k, 'speed_m_s': float(np.hypot(trace['body_vx_m_s'][k], trace['body_vy_m_s'][k])),
            'tilt_rad': float(trace['tilt_rad'][k]), 'height_m': float(trace['base_height_m'][k])}


def load(path):
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--low', type=Path, required=True, help='trace at the lower drive command')
    parser.add_argument('--high', type=Path, required=True, help='trace at the higher drive command')
    parser.add_argument('--probe-drive', type=Path, required=True, help='appendix A drive.jsonl')
    parser.add_argument('--forced-zero', type=Path, required=True, help='05M run A trace')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    low, high = lean_point(load(args.low)), lean_point(load(args.high))
    line = two_point_line(low, high)
    drive = [json.loads(x) for x in args.probe_drive.read_text().splitlines() if x.strip()]
    report = {'format': 'd1-contract05-lean-analysis-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'transitions_consumed': 0,
              'windows_ticks': {'rest': REST, 'plateau': PLATEAU, 'settled': SETTLED, 'entry': ENTRY},
              'low_command': low, 'high_command': high, 'line_through_plateaus': line,
              'line_passes_through_rest_within_0p02_rad': through_rest(line, low['rest_pitch_rad']),
              'wheels_forced_to_zero_release_state': {
                  'appendix_a_probe': release_state_from_probe(drive),
                  'contract_05m_run_a': release_state_from_trace(load(args.forced_zero), 400)},
              'note': ('a zero wheel action is a zero wheel speed target, i.e. passive damping braking '
                       'through kd; active reverse braking was never measured'),
              'inputs': {p.name+'@'+p.parent.name: sha(p) for p in (args.low, args.high, args.probe_drive, args.forced_zero)},
              'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
