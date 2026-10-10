#!/usr/bin/env python3
"""Measure the real passive braking distance of the frozen D1 wheel interface.

Contract 04b section 8 directs that, if the wheel actions were driven to near
zero (P2 and P3 met) while P1 still failed, the interface arithmetic itself must
be remeasured rather than the reward retuned. This is that measurement.

The arithmetic under test, from docs/career_execution_contract_04b.md G6:
wheels have kp=0, kd=0.5 N m/(rad/s), r=0.087 m and the robot is 48.15 kg
(51.5 kg including wheel inertia), so a wheel action of zero should behave as a
damper with a = -5.49 v, tau = 0.195 s, braking from 0.1965 m/s in 0.0383 m.

Method, with no policy and no mid-episode state restoration: reset the frozen
environment, hold one fixed forward wheel action until the speed settles, then
switch to an exactly zero action and record the coast. Leg actions are zero
throughout, which is the default standing pose under kp=60.

This consumes real native steps and is charged to the 04b engineering pool.
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
# runtime03/schema03d.json plus robot.xml, as used by the G6 arithmetic.
WHEEL_KD = .5
WHEEL_RADIUS = .087
ROBOT_MASS_KG = 48.1469


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predicted_damping(mass_kg):
    """a = -(4 kd / r^2 / m) v, i.e. the G6 model."""
    coefficient = 4.*WHEEL_KD/(WHEEL_RADIUS**2)/mass_kg
    return {'effective_mass_kg': mass_kg,
            'linear_damping_per_second': coefficient,
            'time_constant_s': 1./coefficient,
            'braking_distance_from_0p1965_m': .1965/coefficient,
            'braking_distance_from_0p2210_m': .2210/coefficient}


def run(env, action, ticks, log):
    wheel_action = np.zeros(16, dtype=np.float32)
    wheel_action[list(WHEEL_ACTION_INDEX)] = action
    for _ in range(ticks):
        _, _, terminated, truncated, info = env.step(wheel_action)
        metrics = info['metrics'] if 'metrics' in info else info
        log.append({'t': len(log)*TICK_S,
                    'vx': float(metrics['body_vx_m_s']), 'vy': float(metrics['body_vy_m_s']),
                    'x': float(metrics['base_x_m']), 'y': float(metrics['base_y_m']),
                    'wz': float(metrics['body_wz_rad_s']),
                    'height': float(metrics['base_height_m']), 'tilt': float(metrics['tilt_rad']),
                    'fallen': bool(metrics['fallen']),
                    'wheel_qvel': [float(v) for v in env.data.qvel[env.dof_ids][list(WHEEL_ACTION_INDEX)]],
                    'action': float(action)})
        if terminated or truncated:
            return 'TERMINATED' if terminated else 'TRUNCATED'
    return 'OK'


def analyse_coast(coast):
    speed = np.array([np.hypot(r['vx'], r['vy']) for r in coast])
    x = np.array([r['x'] for r in coast])
    y = np.array([r['y'] for r in coast])
    t = np.array([r['t'] for r in coast])
    t = t - t[0]
    distance = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))
    path = float(np.hypot(np.diff(x), np.diff(y)).sum())
    below = np.flatnonzero(speed < .02)
    usable = speed > 1e-4
    tau = None
    if usable.sum() >= 6:
        slope = np.polyfit(t[usable], np.log(speed[usable]), 1)[0]
        tau = float(-1./slope) if slope < 0 else None
    return {'speed_at_release_m_s': float(speed[0]),
            'speed_at_end_m_s': float(speed[-1]),
            'coast_seconds': float(t[-1]),
            'euclidean_distance_m': distance, 'path_length_m': path,
            'time_to_under_0p02_m_s': float(t[below[0]]) if below.size else None,
            'distance_until_under_0p02_m': (
                float(np.hypot(x[below[0]]-x[0], y[below[0]]-y[0])) if below.size else None),
            'fitted_time_constant_s': tau,
            'implied_linear_damping_per_second': (1./tau) if tau else None,
            'speed_series_m_s': [float(v) for v in speed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT/'training/configs/pilot03d.json',
                        help='frozen base config; the physics and interface are version independent')
    parser.add_argument('--wheel-action', type=float, default=.46,
                        help='fixed normalised wheel action for the acceleration phase '
                             '(0.46 x 5 rad/s x 0.087 m is about 0.20 m/s)')
    parser.add_argument('--accelerate-ticks', type=int, default=100)
    parser.add_argument('--coast-ticks', type=int, default=75)
    parser.add_argument('--seed', type=int, default=7001)
    parser.add_argument('--native-budget', type=int, default=2000,
                        help='04b engineering pool cap; the run aborts if the plan exceeds it')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    planned = (args.accelerate_ticks+args.coast_ticks)*NATIVE_PER_TICK
    if planned > args.native_budget:
        raise SystemExit(f'Planned {planned} native steps exceeds the {args.native_budget} cap')
    if not 0. < args.wheel_action <= 1.:
        raise SystemExit('The acceleration wheel action must lie in (0, 1]')

    from d1_rl.factory03 import environment_class
    config = args.config.resolve()
    env = environment_class(config)(config=config, arm='B', seed=args.seed, rank=0)
    accelerate, coast = [], []
    try:
        env.reset(seed=args.seed)
        status_accelerate = run(env, args.wheel_action, args.accelerate_ticks, accelerate)
        status_coast = run(env, 0., args.coast_ticks, coast) if status_accelerate == 'OK' else 'SKIPPED'
        counts = env.counters()
    finally:
        env.close()

    report = {
        'format': 'd1-contract04b-passive-braking-probe-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'purpose': ('remeasure the passive wheel braking distance that contract 04b G6 '
                    'predicted, as section 8 directs when P2/P3 are met and P1 is not'),
        'budget_pool': '04b engineering/preflight',
        'planned_native_steps': planned,
        'actual_counts': counts,
        'policy_inference_calls': 0,
        'mid_episode_state_restoration_used': False,
        'method': ('fresh reset, then a fixed forward wheel action with zero leg actions, '
                   'then an exactly zero 16-dimensional action; no policy, no teacher'),
        'config': str(config.relative_to(ROOT)), 'config_sha256': sha(config),
        'seed': args.seed, 'wheel_action_during_acceleration': args.wheel_action,
        'phase_status': {'accelerate': status_accelerate, 'coast': status_coast},
        'g6_prediction': {
            'bare_mass': predicted_damping(ROBOT_MASS_KG),
            'with_wheel_inertia_estimate': predicted_damping(51.5),
            'note': 'assumes rolling without slip and all four wheels loaded'},
        'measured_coast': analyse_coast(coast) if coast else None,
        'acceleration_phase': {
            'final_speed_m_s': float(np.hypot(accelerate[-1]['vx'], accelerate[-1]['vy']))
            if accelerate else None,
            'final_wheel_qvel_rad_s': accelerate[-1]['wheel_qvel'] if accelerate else None,
            'commanded_wheel_rad_s': args.wheel_action*5.,
            'peak_tilt_rad': max((r['tilt'] for r in accelerate), default=None),
            'any_fall': any(r['fallen'] for r in accelerate)},
        'tool_sha256': sha(__file__)}
    verdict = report['measured_coast']
    if verdict:
        predicted = report['g6_prediction']['with_wheel_inertia_estimate']['time_constant_s']
        measured = verdict['fitted_time_constant_s']
        report['g6_verdict'] = {
            'predicted_time_constant_s': predicted,
            'measured_time_constant_s': measured,
            'ratio_measured_over_predicted': (measured/predicted) if measured else None,
            'measured_braking_distance_to_0p02_m': verdict['distance_until_under_0p02_m'],
            'gate_m': .05,
            'passive_braking_alone_meets_gate': (
                verdict['distance_until_under_0p02_m'] is not None
                and verdict['distance_until_under_0p02_m'] < .05)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    (args.output/'accelerate.jsonl').write_text(
        ''.join(json.dumps(r)+'\n' for r in accelerate))
    (args.output/'coast.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in coast))
    print(json.dumps({'output': str(args.output/'report.json'),
                      'actual_native_steps': counts['native_steps'],
                      'phase_status': report['phase_status'],
                      'g6_verdict': report.get('g6_verdict')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
