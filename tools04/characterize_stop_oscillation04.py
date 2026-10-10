#!/usr/bin/env python3
"""Characterise the zero-command stop behaviour in a native C++ trace.

Reads one runtime03 trace.jsonl (one record per 2 ms native step) and describes
what the robot does while the command is zero: monotone creep, undamped
oscillation around a point, or genuinely at rest. Pure arithmetic, no physics.

Written after 04a replaced 03d's monotone forward creep with a limit cycle, so
the next contract can target the actual behaviour instead of the previous one.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

NATIVE_DT = .002
WHEEL_ACTION_INDEX = (3, 7, 11, 15)


def load_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text().strip().split('\n')]
    return {'sim_time': np.array([r['sim_time'] for r in rows]),
            'command': np.array([r['command'] for r in rows]),
            'position': np.array([r['position'] for r in rows]),
            'body_vx': np.array([r['body_vx'] for r in rows]),
            'body_velocity': np.array([r['body_velocity'] for r in rows]),
            'body_wz': np.array([r['body_wz'] for r in rows]),
            'issued_action': np.array([r['issued_action'] for r in rows]),
            'policy_count': np.array([r['policy_count'] for r in rows])}


def zero_dominant_frequency(signal, dt_s):
    """Dominant nonzero spectral component of the mean-removed signal."""
    centred = signal - signal.mean()
    if centred.size < 8 or not np.any(centred):
        return None
    spectrum = np.abs(np.fft.rfft(centred))
    freqs = np.fft.rfftfreq(centred.size, dt_s)
    index = int(np.argmax(spectrum[1:]))+1
    return {'dominant_hz': float(freqs[index]),
            'dominant_amplitude': float(2*spectrum[index]/centred.size),
            'rms': float(np.sqrt(np.mean(centred**2)))}


def describe_window(trace, lo, hi):
    mask = (trace['sim_time'] > lo) & (trace['sim_time'] <= hi)
    if mask.sum() < 4:
        return None
    t = trace['sim_time'][mask]
    xy = trace['position'][mask][:, :2]
    vx = trace['body_vx'][mask]
    vy = trace['body_velocity'][mask][:, 1]
    wz = trace['body_wz'][mask]
    speed = np.hypot(vx, vy)
    path = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum())
    euclid = float(np.hypot(xy[-1, 0]-xy[0, 0], xy[-1, 1]-xy[0, 1]))
    # Direction reversals of a signal that is actually moving, not sensor noise.
    moving = np.abs(vx) > .002
    reversals = int(np.count_nonzero(np.diff(np.sign(vx[moving])) != 0)) if moving.sum() > 1 else 0
    centroid = xy.mean(axis=0)
    radius = np.hypot(xy[:, 0]-centroid[0], xy[:, 1]-centroid[1])
    return {
        'window_s': [lo, hi], 'samples': int(mask.sum()),
        'commanded': {'vx': float(trace['command'][mask][:, 0].mean()),
                      'wz': float(trace['command'][mask][:, 2].mean())},
        'euclidean_displacement_m': euclid, 'path_length_m': path,
        'path_over_euclid': float(path/euclid) if euclid > 1e-9 else None,
        'excursion_radius_about_centroid_m': {'mean': float(radius.mean()),
                                              'max': float(radius.max())},
        'body_vx': {'mean': float(vx.mean()), 'std': float(vx.std()),
                    'abs_mean': float(np.abs(vx).mean()), 'abs_max': float(np.abs(vx).max())},
        'body_vy': {'mean': float(vy.mean()), 'std': float(vy.std()),
                    'abs_mean': float(np.abs(vy).mean())},
        'body_wz': {'mean': float(wz.mean()), 'std': float(wz.std()),
                    'abs_mean': float(np.abs(wz).mean())},
        'horizontal_speed_mean_m_s': float(speed.mean()),
        'vx_sign_reversals': reversals,
        'vx_reversals_per_second': float(reversals/(hi-lo)),
        'spectrum_body_vx': zero_dominant_frequency(vx, NATIVE_DT),
        'spectrum_body_wz': zero_dominant_frequency(wz, NATIVE_DT),
        # Mean-vs-oscillation split: a monotone creep has mean >> std, a limit
        # cycle has std >> |mean|.
        'oscillation_to_drift_ratio': (float(vx.std()/abs(vx.mean()))
                                       if abs(vx.mean()) > 1e-9 else None),
        'wheel_targets_rad_s': {
            'net_mean': float((trace['issued_action'][mask][:, list(WHEEL_ACTION_INDEX)]*5.).mean()),
            'per_wheel_mean': [float(v) for v in
                               (trace['issued_action'][mask][:, list(WHEEL_ACTION_INDEX)]*5.).mean(axis=0)],
            'net_std': float((trace['issued_action'][mask][:, list(WHEEL_ACTION_INDEX)]*5.
                              ).mean(axis=1).std()),
            'negative_fraction_of_policy_ticks': float(np.mean(
                (trace['issued_action'][mask][:, list(WHEEL_ACTION_INDEX)]*5.).mean(axis=1) < 0.))},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace', type=Path, action='append', required=True,
                        help='runtime03 trace.jsonl; repeatable')
    parser.add_argument('--label', action='append', required=True,
                        help='one label per --trace, in the same order')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    if len(args.trace) != len(args.label):
        parser.error('--trace and --label must come in pairs')
    traces = {}
    for path, label in zip(args.trace, args.label):
        trace = load_trace(path)
        traces[label] = {
            'trace': str(path),
            'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'native_steps': int(trace['sim_time'].size),
            'policy_ticks': int(trace['policy_count'].max()),
            'windows': {name: describe_window(trace, lo, hi) for name, (lo, hi) in
                        (('drive_7_to_8s', (7., 8.)), ('brake_8_to_9s', (8., 9.)),
                         ('settle_9_to_11s', (9., 11.)), ('final_11_to_12s', (11., 12.)))}}
    report = {'format': 'd1-contract04-stop-oscillation-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'native_dt_s': NATIVE_DT,
              'reading_guide': ('oscillation_to_drift_ratio below about 1 is a monotone drift; '
                                'well above 1 is a limit cycle whose mean cancels. '
                                'negative_fraction_of_policy_ticks shows whether the policy ever '
                                'commands reverse wheel velocity, i.e. brakes actively.'),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'traces': traces}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'),
                      'labels': sorted(traces), 'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
