#!/usr/bin/env python3
"""Offline diagnosis of the frozen pilot03b/03c/03d fixed-evaluation traces.

Reads only the existing .npz traces recorded by the 03 evaluations. It runs no
MuJoCo, loads no policy, and integrates zero native steps, so it consumes no
budget from any pool and cannot alter the frozen receipts.

Purpose: separate fact from hypothesis for two open failures of 03d, the
forward braking overshoot and the positive/negative turn asymmetry, before any
new bounded contract picks a single training change.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

TICK_S = .02
# runtime03/schema03d.json: *_foot_joint are the wheels (kp 0, kd .5, scale 5 rad/s).
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
LEG_ACTION_INDEX = tuple(i for i in range(16) if i not in WHEEL_ACTION_INDEX)
# evaluation_contract.json stable_windows_s
WINDOW = {'translation': (4., 8.), 'turn_positive': (3., 5.), 'turn_negative': (6., 8.),
          'stop_last_second': (11., 12.), 'stop_displacement': (8., 12.)}


def window_mask(time_s, span):
    return (time_s >= span[0]) & (time_s < span[1])


def stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {'n': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
            'min': float(values.min()), 'max': float(values.max()),
            'abs_mean': float(np.abs(values).mean())}


def zero_command_segment(trace, time_s, span):
    """Drift while the commanded velocity is exactly zero."""
    mask = window_mask(time_s, span)
    return {'span_s': list(span),
            'vx': stats(trace['body_vx_m_s'][mask]), 'vy': stats(trace['body_vy_m_s'][mask]),
            'wz': stats(trace['body_wz_rad_s'][mask]),
            'speed_mean_m_s': float(np.hypot(trace['body_vx_m_s'][mask],
                                             trace['body_vy_m_s'][mask]).mean())}


def stop_phase(trace, time_s):
    """Reconstruct the 8..12 s stop phase the displacement gate scores."""
    mask = window_mask(time_s, (8., 12.0001))
    x = trace['base_x_m'][mask]
    y = trace['base_y_m'][mask]
    vx = trace['body_vx_m_s'][mask]
    vy = trace['body_vy_m_s'][mask]
    speed = np.hypot(vx, vy)
    t = time_s[mask]
    # Gate metric is start-to-end straight-line distance; path length exposes
    # drift that cancels out and overshoot that does not.
    euclid = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    below = np.flatnonzero(speed < .02)
    settle_s = float(t[below[0]]-t[0]) if below.size else None
    # Signed progress along the pre-stop heading separates overshoot from recoil.
    pre = window_mask(time_s, (7., 8.))
    heading_vx = float(trace['body_vx_m_s'][pre].mean())
    direction = 1. if heading_vx >= 0 else -1.
    yaw0 = float(trace['base_yaw_rad'][mask][0])
    forward_axis = np.array([np.cos(yaw0), np.sin(yaw0)])*direction
    progress = (np.stack([x-x[0], y-y[0]], axis=1) @ forward_axis)
    per_second = []
    for lo in (8., 9., 10., 11.):
        seg = window_mask(t, (lo, lo+1.0001))
        if seg.sum() < 2:
            per_second.append(None)
            continue
        sx, sy = x[seg], y[seg]
        per_second.append({'span_s': [lo, lo+1.],
                           'euclid_m': float(np.hypot(sx[-1]-sx[0], sy[-1]-sy[0])),
                           'path_m': float(np.hypot(np.diff(sx), np.diff(sy)).sum()),
                           'speed_mean_m_s': float(speed[seg].mean()),
                           'vx_mean_m_s': float(vx[seg].mean())})
    return {'euclidean_displacement_m': euclid, 'path_length_m': path,
            'path_minus_euclid_m': path-euclid,
            'speed_at_8s_m_s': float(speed[0]), 'vx_at_8s_m_s': float(vx[0]),
            'settle_to_under_0p02_s': settle_s,
            'pre_stop_7_to_8s_vx_mean_m_s': heading_vx,
            'signed_progress_final_m': float(progress[-1]),
            'signed_progress_max_m': float(progress.max()),
            'signed_progress_min_m': float(progress.min()),
            'recoil_after_overshoot_m': float(progress.max()-progress[-1]),
            'vx_sign_reversals': int(np.count_nonzero(np.diff(np.sign(vx[np.abs(vx) > .005])) != 0)),
            'per_second': per_second}


def wheel_action(trace, time_s, span):
    """Wheel channels are velocity targets in rad/s after the 5.0 scale."""
    mask = window_mask(time_s, span)
    issued = np.asarray(trace['issued_action'], dtype=float)[mask]
    wheels = issued[:, list(WHEEL_ACTION_INDEX)]*5.
    legs = issued[:, list(LEG_ACTION_INDEX)]*.5
    return {'span_s': list(span),
            'wheel_target_rad_s_mean': stats(wheels.mean(axis=1)),
            'wheel_target_rad_s_per_wheel_mean': [float(v) for v in wheels.mean(axis=0)],
            'wheel_target_abs_max_rad_s': float(np.abs(wheels).max()),
            'wheel_clip_fraction': float(np.mean(np.abs(issued[:, list(WHEEL_ACTION_INDEX)]) > .999)),
            'wheel_tick_to_tick_abs_change_mean_rad_s': float(np.abs(np.diff(wheels, axis=0)).mean())
            if wheels.shape[0] > 1 else None,
            'leg_offset_rad_abs_mean': float(np.abs(legs).mean()),
            'leg_clip_fraction': float(np.mean(np.abs(issued[:, list(LEG_ACTION_INDEX)]) > .999)),
            'tau_saturation_total': int(np.asarray(trace['tau_saturation_count'])[mask].sum())}


def turn_asymmetry(trace, time_s):
    """Positive and negative yaw windows share amplitude 0.2 rad/s by contract."""
    out = {}
    for name, span, cmd in (('positive', WINDOW['turn_positive'], .2),
                            ('negative', WINDOW['turn_negative'], -.2)):
        mask = window_mask(time_s, span)
        wz = np.asarray(trace['body_wz_rad_s'])[mask]
        out[name] = {'span_s': list(span), 'commanded_wz_rad_s': cmd,
                     'wz': stats(wz), 'mae_rad_s': float(np.abs(wz-cmd).mean()),
                     'tracking_gain': float(wz.mean()/cmd),
                     'shortfall_rad_s': float(cmd-wz.mean()) if cmd > 0 else float(wz.mean()-cmd)}
    out['gain_difference_positive_minus_negative'] = (out['positive']['tracking_gain']
                                                      - out['negative']['tracking_gain'])
    # Response to the commanded sign flip at 5 s.
    rise = {}
    for name, start, cmd in (('step_at_2s_to_plus', 2., .2), ('step_at_5s_to_minus', 5., -.2)):
        mask = window_mask(time_s, (start, start+1.5))
        wz = np.asarray(trace['body_wz_rad_s'])[mask]
        t = time_s[mask]
        reached = np.flatnonzero((wz/cmd) >= .9)
        rise[name] = {'commanded_wz_rad_s': cmd,
                      'time_to_90pct_s': float(t[reached[0]]-start) if reached.size else None,
                      'wz_at_plus_0p5s_rad_s': float(wz[min(25, wz.size-1)]),
                      'max_fraction_of_command_reached': float((wz/cmd).max())}
    out['step_response'] = rise
    return out


def analyse(npz_path, sequence):
    with np.load(npz_path) as handle:
        trace = {key: handle[key] for key in handle.files}
    time_s = np.asarray(trace['policy_interval_start_s'], dtype=float)
    report = {'trace': str(npz_path), 'sequence': sequence,
              'trace_sha256': hashlib.sha256(Path(npz_path).read_bytes()).hexdigest(),
              'policy_ticks': int(time_s.size),
              'fallen_any': bool(np.any(trace['fallen'])),
              'illegal_contact_any': bool(np.any(trace['illegal_body_contact'])),
              'tau_saturation_total': int(np.asarray(trace['tau_saturation_count']).sum()),
              'zero_command_0_to_2s': zero_command_segment(trace, time_s, (0., 2.)),
              'zero_command_last_second': zero_command_segment(trace, time_s, (11., 12.)),
              'stop_phase_8_to_12s': stop_phase(trace, time_s),
              'action_during_drive_7_to_8s': wheel_action(trace, time_s, (7., 8.)),
              'action_during_stop_8_to_9s': wheel_action(trace, time_s, (8., 9.)),
              'action_during_stop_11_to_12s': wheel_action(trace, time_s, (11., 12.))}
    if sequence == 'turn_stop':
        report['turn_asymmetry'] = turn_asymmetry(trace, time_s)
    else:
        mask = window_mask(time_s, WINDOW['translation'])
        cmd = np.asarray(trace['command_during_interval'], dtype=float)[mask][:, 0]
        vx = np.asarray(trace['body_vx_m_s'])[mask]
        report['drive_4_to_8s'] = {'commanded_vx_mean_m_s': float(cmd.mean()),
                                   'vx': stats(vx), 'mae_m_s': float(np.abs(vx-cmd).mean()),
                                   'tracking_gain': float(vx.mean()/cmd.mean())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval-dir', type=Path, action='append', required=True,
                        help='a frozen pilot03*_python_eval_01 directory; repeatable')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()

    versions = {}
    for directory in args.eval_dir:
        label = directory.name.split('_')[0]
        entries = {}
        for sequence in ('forward_stop', 'reverse_stop', 'turn_stop'):
            npz = directory/f'{sequence}_delay0_seed7001.npz'
            entries[sequence] = analyse(npz, sequence) if npz.is_file() else {'missing': str(npz)}
        versions[label] = entries

    report = {'format': 'd1-contract04-offline-trace-diagnosis-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': 'offline reanalysis of frozen 03 evaluation traces',
              'physics_steps_executed': 0, 'transitions_consumed': 0,
              'policy_inference_calls': 0, 'mujoco_imported': False,
              'note': ('Descriptive statistics only. It identifies where the failures occur; '
                       'it does not by itself prove a cause, and it changes no 03 receipt.'),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'versions': versions}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'),
                      'versions': sorted(versions), 'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
