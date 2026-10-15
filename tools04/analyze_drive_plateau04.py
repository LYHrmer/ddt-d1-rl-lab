#!/usr/bin/env python3
"""Where does the drive phase settle, and how hard does the policy push the wheels there?

Contract 05M's revision 1 rests on one observation: 04b's drive phase levels off well
below the 0.2 m/s command. This tool puts that observation into a receipt. It reads
frozen evaluation traces only, imports no simulator and executes no physics.

For each trace and for the last second before the cut it reports the signed speed
along the command, the speed trend, the time to reach 90 percent of the command, the
policy's mean wheel target against the rolling rate that speed implies, and the
resulting wheel torque. A torque near the limit would mean the plateau is actuator
saturation; a torque far below it means the policy chose not to push harder.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
WHEEL_ACTION_SCALE = 5.
WHEEL_RADIUS_M = .087
WHEEL_KD = .5
TORQUE_LIMIT_N_M = 12.
TICK_S = .02
DRIVE_START_TICK = 100
CUT_TICK = 400
WINDOW_TICKS = 50
COMMAND_M_S = .2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_speed(vx, direction):
    return direction*np.asarray(vx, dtype=float)


def speed_trend(speed, lo, hi, tick_s=TICK_S):
    """Least-squares slope of the speed over ticks lo..hi, in m/s per second."""
    t = np.arange(lo, hi)*tick_s
    return float(np.polyfit(t, np.asarray(speed, dtype=float)[lo:hi], 1)[0])


def time_to_fraction(speed, command, fraction, start=DRIVE_START_TICK, stop=CUT_TICK, tick_s=TICK_S):
    """Seconds after the drive starts until the speed first reaches fraction*command; None if never."""
    reached = np.flatnonzero(np.asarray(speed, dtype=float)[start:stop] >= fraction*abs(command))
    return float(reached[0]*tick_s) if reached.size else None


def wheel_push(issued_action, speed, direction, lo, hi):
    """Mean wheel target, the rolling rate the speed implies, and the torque the gap produces."""
    wheels = np.asarray(issued_action, dtype=float)[lo:hi][:, list(WHEEL_ACTION_INDEX)]
    target = float(direction*wheels.mean()*WHEEL_ACTION_SCALE)
    rolling = float(np.asarray(speed, dtype=float)[lo:hi].mean()/WHEEL_RADIUS_M)
    torque = WHEEL_KD*(target-rolling)
    return {'mean_abs_action': float(np.abs(wheels).mean()), 'wheel_target_rad_s': target,
            'rolling_rate_rad_s': rolling, 'target_minus_rolling_rad_s': target-rolling,
            'torque_per_wheel_n_m': torque, 'torque_limit_n_m': TORQUE_LIMIT_N_M,
            'fraction_of_torque_limit': abs(torque)/TORQUE_LIMIT_N_M}


def analyse(path, direction, command=COMMAND_M_S, window=WINDOW_TICKS, cut=CUT_TICK):
    with np.load(path) as handle:
        vx, action = handle['body_vx_m_s'], handle['issued_action']
    speed = signed_speed(vx, direction)
    lo, hi = cut-window, cut
    seconds = {str(k): float(speed[DRIVE_START_TICK+50*(k-1):DRIVE_START_TICK+50*k].mean())
               for k in range(1, 7)}
    mean = float(speed[lo:hi].mean())
    return {'command_m_s': command, 'speed_by_second_after_drive_start_m_s': seconds,
            'last_second_mean_speed_m_s': mean,
            'last_second_fraction_of_command': mean/command,
            'last_second_trend_m_s_per_s': speed_trend(speed, lo, hi),
            'seconds_to_90_percent_of_command': time_to_fraction(speed, command, .9),
            'wheel_push_last_second': wheel_push(action, speed, direction, lo, hi),
            'trace_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', action='append', default=[], metavar='LABEL=DIR')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    versions = {}
    for item in args.eval:
        label, directory = item.split('=', 1)
        versions[label] = {task: analyse(Path(directory)/f'{task}_stop_delay0_seed7001.npz',
                                         1. if task == 'forward' else -1.)
                           for task in ('forward', 'reverse')}
    report = {'format': 'd1-contract04-drive-plateau-analysis-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'transitions_consumed': 0,
              'constants': {'wheel_action_scale': WHEEL_ACTION_SCALE, 'wheel_radius_m': WHEEL_RADIUS_M,
                            'wheel_kd': WHEEL_KD, 'torque_limit_n_m': TORQUE_LIMIT_N_M,
                            'drive_start_tick': DRIVE_START_TICK, 'cut_tick': CUT_TICK},
              'versions': versions, 'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'), 'versions': sorted(versions)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
