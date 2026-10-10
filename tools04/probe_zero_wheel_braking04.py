#!/usr/bin/env python3
"""Measure passive wheel braking with the legs under policy control.

tools04/probe_passive_braking04.py held the legs at the default pose and the
robot sagged from 0.457 m to 0.324 m and ended on an illegal body contact after
0.7 s, so it never reached a clean rolling state. That rules out the rigid,
four-wheels-loaded body that contract 04b G6 assumed, but it does not measure
the braking distance.

This probe keeps the legs under the trained policy and overrides only the four
wheel actions to exactly zero during the coast. That is exactly G6's claim: if
the policy's wheel actions were zero while it was moving, how far would it
travel before stopping?

Consumes real native steps, charged to the 04b engineering pool.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))

TICK_S = .02
NATIVE_PER_TICK = 10
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
WHEEL_KD = .5
WHEEL_RADIUS = .087


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    """Repo-relative when inside the tree, absolute otherwise; never raises."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def record(info, env, tick, zero_wheels, raw_action, applied):
    return {'tick': tick, 't': tick*TICK_S,
            'vx': float(info['body_vx_m_s']), 'vy': float(info['body_vy_m_s']),
            'wz': float(info['body_wz_rad_s']),
            'x': float(info['base_x_m']), 'y': float(info['base_y_m']),
            'height': float(info['base_height_m']), 'tilt': float(info['tilt_rad']),
            'fallen': bool(info['fallen']),
            'illegal': bool(info['illegal_body_contact']),
            'wheels_zeroed': bool(zero_wheels),
            'policy_wheel_action': [float(v) for v in raw_action[list(WHEEL_ACTION_INDEX)]],
            'applied_wheel_action': [float(v) for v in applied[list(WHEEL_ACTION_INDEX)]],
            'wheel_qvel_rad_s': [float(v) for v in
                                 env.data.qvel[env.dof_ids][list(WHEEL_ACTION_INDEX)]]}


def analyse(coast):
    speed = np.array([np.hypot(r['vx'], r['vy']) for r in coast])
    x = np.array([r['x'] for r in coast])
    y = np.array([r['y'] for r in coast])
    t = np.array([r['t'] for r in coast]) - coast[0]['t']
    below = np.flatnonzero(speed < .02)
    usable = speed > 1e-4
    tau = None
    if usable.sum() >= 6:
        slope = np.polyfit(t[usable], np.log(speed[usable]), 1)[0]
        tau = float(-1./slope) if slope < 0 else None
    return {'speed_at_release_m_s': float(speed[0]),
            'speed_at_end_m_s': float(speed[-1]),
            'coast_seconds': float(t[-1]),
            'euclidean_distance_m': float(np.hypot(x[-1]-x[0], y[-1]-y[0])),
            'path_length_m': float(np.hypot(np.diff(x), np.diff(y)).sum()),
            'time_to_under_0p02_m_s': float(t[below[0]]) if below.size else None,
            'distance_until_under_0p02_m': (
                float(np.hypot(x[below[0]]-x[0], y[below[0]]-y[0])) if below.size else None),
            'fitted_time_constant_s': tau,
            'peak_tilt_rad': float(max(r['tilt'] for r in coast)),
            'any_fall_or_illegal': any(r['fallen'] or r['illegal'] for r in coast),
            'speed_series_m_s': [float(v) for v in speed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--command-vx', type=float, default=.2)
    parser.add_argument('--drive-ticks', type=int, default=100)
    parser.add_argument('--coast-ticks', type=int, default=65)
    parser.add_argument('--seed', type=int, default=7001)
    parser.add_argument('--factory', default='d1_rl.factory04b',
                        help='module exposing environment_class for the probed config')
    parser.add_argument('--native-budget', type=int, required=True,
                        help='remaining 04b engineering pool; the run aborts if the plan exceeds it')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    planned = (args.drive_ticks+args.coast_ticks)*NATIVE_PER_TICK
    if planned > args.native_budget:
        raise SystemExit(f'Planned {planned} native steps exceeds the {args.native_budget} remaining')

    import importlib
    import onnxruntime as ort
    from d1_rl.interface import OBS_DIM, HISTORY_LENGTH
    environment_class = importlib.import_module(args.factory).environment_class

    config = args.config.resolve()
    session = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    env = environment_class(config)(config=config, arm='B', seed=args.seed, rank=0)
    drive, coast = [], []
    status = 'OK'
    try:
        observation, _ = env.reset(seed=args.seed)
        env.set_command(args.command_vx, 0.)
        for tick in range(args.drive_ticks+args.coast_ticks):
            coasting = tick >= args.drive_ticks
            if coasting:
                # The command stays zero from the release tick, exactly like the gate.
                env.set_command(0., 0.)
            flat = np.asarray(observation, dtype=np.float32)
            obs = flat[:OBS_DIM].reshape(1, OBS_DIM)
            history = flat[OBS_DIM:].reshape(1, HISTORY_LENGTH, OBS_DIM)
            raw = np.asarray(session.run(['actions'], {'obs': obs, 'history': history})[0][0],
                             dtype=np.float32)
            applied = raw.copy()
            if coasting:
                applied[list(WHEEL_ACTION_INDEX)] = 0.
            observation, _, terminated, truncated, info = env.step(applied)
            row = record(info, env, tick, coasting, raw, applied)
            (coast if coasting else drive).append(row)
            if terminated or truncated:
                status = 'TERMINATED' if terminated else 'TRUNCATED'
                break
        counts = env.counters()
    finally:
        env.close()

    coefficient = 4.*WHEEL_KD/(WHEEL_RADIUS**2)/51.5
    measured = analyse(coast) if coast else None
    report = {
        'format': 'd1-contract04b-policy-legs-zero-wheels-probe-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'purpose': ('measure the braking distance that contract 04b G6 predicted, with the '
                    'legs under policy control and only the four wheel actions forced to zero'),
        'budget_pool': '04b engineering/preflight',
        'planned_native_steps': planned, 'actual_counts': counts,
        'policy': relative(args.onnx), 'policy_sha256': sha(args.onnx),
        'config': relative(config), 'config_sha256': sha(config),
        'seed': args.seed, 'command_vx_during_drive': args.command_vx,
        'wheel_actions_forced_to_zero_during_coast': True,
        'leg_actions_from_policy_throughout': True,
        'mid_episode_state_restoration_used': False,
        'status': status,
        'drive_phase': {'ticks': len(drive),
                        'final_speed_m_s': float(np.hypot(drive[-1]['vx'], drive[-1]['vy']))
                        if drive else None,
                        'final_wheel_qvel_rad_s': drive[-1]['wheel_qvel_rad_s'] if drive else None,
                        'net_policy_wheel_action': float(np.mean(drive[-1]['policy_wheel_action']))
                        if drive else None,
                        'peak_tilt_rad': max((r['tilt'] for r in drive), default=None)},
        'g6_prediction': {'linear_damping_per_second': coefficient,
                          'time_constant_s': 1./coefficient,
                          'assumption': 'rigid upright body, four wheels loaded, rolling without slip'},
        'measured_coast': measured,
        'tool_sha256': sha(__file__)}
    if measured:
        report['g6_verdict'] = {
            'predicted_time_constant_s': 1./coefficient,
            'measured_time_constant_s': measured['fitted_time_constant_s'],
            'ratio_measured_over_predicted': (measured['fitted_time_constant_s']*coefficient
                                              if measured['fitted_time_constant_s'] else None),
            'predicted_distance_from_release_speed_m': measured['speed_at_release_m_s']/coefficient,
            'measured_distance_until_under_0p02_m': measured['distance_until_under_0p02_m'],
            'gate_m': .05,
            'zero_wheel_action_alone_meets_gate': (
                measured['distance_until_under_0p02_m'] is not None
                and measured['distance_until_under_0p02_m'] < .05)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    (args.output/'drive.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in drive))
    (args.output/'coast.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in coast))
    print(json.dumps({'output': str(args.output/'report.json'), 'status': status,
                      'actual_native_steps': counts['native_steps'],
                      'g6_verdict': report.get('g6_verdict')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
