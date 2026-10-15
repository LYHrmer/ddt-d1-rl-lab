#!/usr/bin/env python3
"""Put run A of contract 05M beside the policy-wheel arm it is compared with.

The pre-registered verdict is a single number per prediction. This receipt keeps
the evidence that sits behind it, and it is computed from the two traces rather than
copied by hand: the speed of each arm at the same ticks after the cut, the decay time
constant fitted over several windows (a decay that is not exponential gives a different
constant in each window), the displacement each arm accumulates and the share the wheel
action accounts for, and the body posture each ends with. Reads traces only; imports no
simulator and executes no physics.
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
sys.path.insert(0, str(ROOT/'tools04'))

import analyze_cut_to_zero04 as analyser  # noqa: E402

CUT_TICK = 400
TICKS = (0, 2, 5, 10, 15, 25, 50, 100, 199)
WINDOWS = ((0, 10), (5, 25), (10, 50), (25, 100))
FIT_FLOOR = .005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def speed_series(trace):
    return np.hypot(np.asarray(trace['body_vx_m_s'], dtype=float), np.asarray(trace['body_vy_m_s'], dtype=float))


def speeds_at(speed, cut, ticks=TICKS):
    return {str(k): float(speed[cut+k]) for k in ticks if cut+k < len(speed)}


def window_taus(speed, cut, windows=WINDOWS, floor=FIT_FLOOR):
    return {f'{lo}-{hi}': analyser.decay_time_constant(speed, cut, lo, hi, floor=floor) for lo, hi in windows}


def displacement(boundary_xy, cut, horizon=200):
    xy = np.asarray(boundary_xy, dtype=float)
    end = min(len(xy)-1, cut+horizon)
    steps = np.hypot(*np.diff(xy[cut:end+1], axis=0).T)
    return {'euclidean_m': float(np.hypot(*(xy[end]-xy[cut]))), 'path_m': float(steps.sum())}


def share_explained(with_policy_wheels, with_forced_zero):
    """Fraction of the policy arm's displacement that disappears when its wheels are forced to zero."""
    if not with_policy_wheels:
        return None
    return float(1.-with_forced_zero/with_policy_wheels)


def posture_after(trace, cut, lo, hi):
    window = slice(cut+lo, cut+hi)
    return {'pitch_rad': float(np.median(np.asarray(trace['base_pitch_rad'], dtype=float)[window])),
            'height_m': float(np.median(np.asarray(trace['base_height_m'], dtype=float)[window]))}


def describe(trace, cut=CUT_TICK):
    speed = speed_series(trace)
    entry = float(speed[cut-5:cut].mean())
    return {'entry_speed_m_s': entry, 'speed_by_tick_after_cut_m_s': speeds_at(speed, cut),
            'tau_s_by_fit_window_ticks': window_taus(speed, cut),
            'displacement_200_ticks': displacement(trace['boundary_xy_m'], cut),
            'ticks_until_speed_below_0p02': analyser.ticks_until_speed_below(speed[:len(trace['boundary_xy_m'])-1], cut),
            'pitch_on_drive_plateau_rad': float(np.median(np.asarray(trace['base_pitch_rad'], dtype=float)[cut-50:cut])),
            'peak_abs_pitch_rad': {'before_cut': float(np.abs(np.asarray(trace['base_pitch_rad'], dtype=float)[:cut]).max()),
                                   'after_cut': float(np.abs(np.asarray(trace['base_pitch_rad'], dtype=float)[cut:]).max())},
            'pitch_per_unit_entry_speed_rad_per_m_s': float(np.median(np.asarray(trace['base_pitch_rad'], dtype=float)[cut-50:cut]))/entry,
            'posture_ticks_20_to_50': posture_after(trace, cut, 20, 50),
            'posture_ticks_150_to_200': posture_after(trace, cut, 150, 200)}


def load(path):
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy-arm', type=Path, required=True, help='frozen evaluation trace .npz')
    parser.add_argument('--forced-zero-arm', type=Path, required=True, help='run A trace .npz')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    policy, forced = describe(load(args.policy_arm)), describe(load(args.forced_zero_arm))
    report = {'format': 'd1-contract05m-arm-comparison-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'transitions_consumed': 0, 'cut_tick': CUT_TICK,
              'policy_wheel_arm': policy, 'forced_zero_wheel_arm': forced,
              'share_of_euclidean_displacement_explained_by_policy_wheel_action': share_explained(
                  policy['displacement_200_ticks']['euclidean_m'], forced['displacement_200_ticks']['euclidean_m']),
              'share_of_path_explained_by_policy_wheel_action': share_explained(
                  policy['displacement_200_ticks']['path_m'], forced['displacement_200_ticks']['path_m']),
              'note': ('post-hoc description of the evidence behind the pre-registered verdict, '
                       'not a pre-registered prediction'),
              'inputs': {'policy_arm_sha256': sha(args.policy_arm), 'forced_zero_arm_sha256': sha(args.forced_zero_arm)},
              'tool_sha256': sha(__file__), 'analyser_sha256': sha(analyser.__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
