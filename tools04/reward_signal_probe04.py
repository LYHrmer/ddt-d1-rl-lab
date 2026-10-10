#!/usr/bin/env python3
"""Recompute the frozen 03c/03d reward terms along the frozen evaluation traces.

Pure arithmetic on the recorded .npz metrics: no MuJoCo, no policy, no native
step. The point is to measure how much reward signal each term actually supplies
in the two regions where 03d failed - the 8..9 s braking transient and the
11..12 s residual creep - instead of arguing about the formulas in prose.

Terms that depend on quantities the trace does not store (torque, contacts,
leg joint angles) are reported as unavailable rather than guessed.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

TICK_S = .02
WHEEL_ACTION_INDEX = (3, 7, 11, 15)


def load_config(path):
    return json.loads(Path(path).read_text())['reward']


def terms_along_trace(trace, cfg):
    """Per-tick value of every reward term reconstructable from the trace."""
    vx = np.asarray(trace['body_vx_m_s'], dtype=float)
    vy = np.asarray(trace['body_vy_m_s'], dtype=float)
    wz = np.asarray(trace['body_wz_rad_s'], dtype=float)
    height = np.asarray(trace['base_height_m'], dtype=float)
    cmd = np.asarray(trace['command_during_interval'], dtype=float)
    roll = np.asarray(trace['base_roll_rad'], dtype=float)
    pitch = np.asarray(trace['base_pitch_rad'], dtype=float)
    issued = np.asarray(trace['issued_action'], dtype=float)

    zero_cmd = (cmd[:, 0] == 0.) & (cmd[:, 2] == 0.)
    speed_sq = vx*vx + vy*vy
    standstill = np.where(zero_cmd,
                          np.exp(-speed_sq/cfg['standstill_linear_error_scale_m2_s2']
                                 - wz*wz/cfg['standstill_yaw_error_scale_rad2_s2']),
                          0.)
    # gravity_xy^2 == sin(tilt)^2 components; recover from roll/pitch.
    gravity_xy_sq = np.sin(roll)**2*np.cos(pitch)**2 + np.sin(pitch)**2
    rate = np.full(issued.shape[0], np.nan)
    rate[1:] = np.mean(np.square(issued[1:]-issued[:-1]), axis=1)
    per_second = {
        'vx_tracking': cfg['vx_tracking_weight']*np.exp(-(vx-cmd[:, 0])**2/cfg['vx_tracking_error_scale_m2_s2']),
        'wz_tracking': cfg['wz_tracking_weight']*np.exp(-(wz-cmd[:, 2])**2/cfg['wz_tracking_error_scale_rad2_s2']),
        'alive': np.full(vx.shape, cfg['alive_weight']),
        'body_vy_squared': cfg['body_vy_squared_weight']*vy*vy,
        'gravity_xy_squared': cfg['gravity_xy_squared_weight']*gravity_xy_sq,
        'action_rate_mean_squared': cfg['action_rate_mean_squared_weight']*rate,
        'standstill_tracking': cfg['standstill_tracking_weight']*standstill,
    }
    unavailable = ['height_error_squared (nominal height not stored in trace)',
                   'torque_sum_squared (per-native torques not stored)',
                   'illegal_body_contact (boolean stored, weight applied below)',
                   'leg_pose_deviation_sum_squared (joint angles not stored)']
    return per_second, zero_cmd, speed_sq, unavailable, issued, height


def sensitivity(cfg, speeds):
    """d(term)/d(speed) per unit speed, for the two speed-dependent terms.

    Shows which term is actually able to push the residual creep toward zero.
    """
    out = []
    for v in speeds:
        standstill_d = (cfg['standstill_tracking_weight']
                        * np.exp(-v*v/cfg['standstill_linear_error_scale_m2_s2'])
                        * (-2*v/cfg['standstill_linear_error_scale_m2_s2']))
        vx_track_d = (cfg['vx_tracking_weight']
                      * np.exp(-v*v/cfg['vx_tracking_error_scale_m2_s2'])
                      * (-2*v/cfg['vx_tracking_error_scale_m2_s2']))
        out.append({'speed_m_s': float(v),
                    'standstill_value': float(cfg['standstill_tracking_weight']
                                              * np.exp(-v*v/cfg['standstill_linear_error_scale_m2_s2'])),
                    'standstill_d_per_m_s': float(standstill_d),
                    'vx_tracking_value_at_zero_command': float(
                        cfg['vx_tracking_weight']*np.exp(-v*v/cfg['vx_tracking_error_scale_m2_s2'])),
                    'vx_tracking_d_per_m_s': float(vx_track_d)})
    return out


def window(time_s, lo, hi):
    return (time_s >= lo) & (time_s < hi)


def summarise(npz_path, cfg):
    with np.load(npz_path) as handle:
        trace = {k: handle[k] for k in handle.files}
    time_s = np.asarray(trace['policy_interval_start_s'], dtype=float)
    per_second, zero_cmd, speed_sq, unavailable, issued, _ = terms_along_trace(trace, cfg)
    regions = {'drive_7_to_8s': (7., 8.), 'brake_8_to_9s': (8., 9.), 'creep_11_to_12s': (11., 12.)}
    out = {'trace': str(npz_path), 'unavailable_terms': unavailable, 'regions': {}}
    for name, (lo, hi) in regions.items():
        mask = window(time_s, lo, hi)
        entry = {'zero_command_fraction': float(zero_cmd[mask].mean()),
                 'speed_mean_m_s': float(np.sqrt(speed_sq[mask]).mean()),
                 'reconstructable_sum_per_second': 0.}
        for term, values in per_second.items():
            value = float(np.nanmean(values[mask]))
            entry[term] = value
            entry['reconstructable_sum_per_second'] += value
        wheels = issued[mask][:, list(WHEEL_ACTION_INDEX)]*5.
        entry['wheel_target_net_mean_rad_s'] = float(wheels.mean())
        left = (wheels[:, 0]+wheels[:, 2])/2.
        right = (wheels[:, 1]+wheels[:, 3])/2.
        entry['wheel_right_minus_left_rad_s'] = float((right-left).mean())
        out['regions'][name] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=Path('training/configs/pilot03d.json'))
    parser.add_argument('--eval-dir', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    cfg = load_config(args.config)
    versions = {}
    for directory in args.eval_dir:
        entries = {}
        for sequence in ('forward_stop', 'reverse_stop', 'turn_stop'):
            npz = directory/f'{sequence}_delay0_seed7001.npz'
            if npz.is_file():
                entries[sequence] = summarise(npz, cfg)
        versions[directory.name.split('_')[0]] = entries
    report = {'format': 'd1-contract04-reward-signal-probe-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'transitions_consumed': 0, 'mujoco_imported': False,
              'reward_config': str(args.config),
              'reward_config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
              'reward_weights': cfg,
              'note': ('Reward terms recomputed from frozen trace metrics. Terms needing '
                       'per-native torque, contacts or joint angles are listed as unavailable '
                       'and excluded from the partial sum.'),
              'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'speed_sensitivity': sensitivity(cfg, (0., .005, .0145, .02, .04, .05, .1, .1965)),
              'versions': versions}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'),
                      'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
