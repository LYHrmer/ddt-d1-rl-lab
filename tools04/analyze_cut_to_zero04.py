#!/usr/bin/env python3
"""Offline look at what happens in the first ticks after a command drops to zero.

Contract 04b ended with a measured fact and an open question. Forcing the four
wheel actions to zero brakes the robot in about 0.0073 m, well inside the 0.05 m
gate, yet 04b's policy left 70 percent of its forward stop displacement in the
8 to 9 s window. This tool reads only frozen evaluation traces, the zero-wheel
probe receipt and the training monitor logs, and answers three questions without
a single physics step:

  1. After the cut, how many ticks pass before the wheel action actually reaches
     zero, and how does the braking compare with the zero-wheel reference?
  2. Where in time does the motion happen, measured against the 10-frame history
     that is the only part of the actor input able to say "the command was just
     cut"?
  3. How many cut events, and how many ticks in the window after one, did training
     actually contain?

It imports no simulator and builds no environment.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WHEEL_ACTION_INDEX = (3, 7, 11, 15)
WHEEL_ACTION_SCALE = 5.
WHEEL_RADIUS_M = .087                   # runtime03 schema / robot.xml wheel collision cylinder
HISTORY_LENGTH = 10                     # frames kept in the actor input
ZERO_THRESHOLD = .02                    # contract 04b prediction P2, mean(a^2) per tick
ZERO_HOLD_TICKS = 5
PROGRESS_TICKS = (5, 10, 25, 50, 65)    # 65 is as long as the zero-wheel probe coasted
PATH_BINS = ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200))
EXPOSURE_WINDOWS = (10, 25, 60)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- pure helpers, covered by tests/test_cut_to_zero_analysis.py ---------------

def nonzero_command(command):
    """True where the commanded vx or wz is nonzero; vy is always zero here."""
    command = np.asarray(command, dtype=float)
    return (np.abs(command[:, 0]) > 0.) | (np.abs(command[:, 2]) > 0.)


def find_cut_ticks(command):
    """Decision ticks at which the command is zero and the previous one was not."""
    moving = nonzero_command(command)
    return [int(k) for k in range(1, len(moving)) if moving[k-1] and not moving[k]]


def history_frames_carrying_command(ticks_after_cut, history_length=HISTORY_LENGTH):
    """How many of the history frames still hold the pre-cut nonzero command.

    Tick 0 is the first decision that sees the zero command. Its history is the
    ten previous observations, all taken before the cut, so all ten carry it.
    By tick 10 the history holds only zero-command frames and says nothing about
    the cut. After that the actor can only tell it has just stopped from its own
    joint and wheel velocities.
    """
    if ticks_after_cut < 0:
        raise ValueError('ticks_after_cut must be nonnegative')
    # No upper clamp is needed: with a nonnegative tick the difference cannot
    # exceed history_length. A clamp was here once and verified unreachable.
    return max(0, history_length-ticks_after_cut)


def zeroing_latency(mean_square, start, threshold=ZERO_THRESHOLD, hold=ZERO_HOLD_TICKS):
    """Ticks after `start` until the wheel action stays below threshold for `hold` ticks.

    Returns None when it never does within the available trace.
    """
    values = np.asarray(mean_square, dtype=float)
    for tick in range(start, len(values)-hold+1):
        if np.all(values[tick:tick+hold] < threshold):
            return int(tick-start)
    return None


def decay_time_constant(speed, start, lo, hi, floor=.005, tick_s=.02):
    """Time constant of an exponential fit to |speed| over ticks start+lo .. start+hi.

    Samples under `floor` are dropped so sensor-scale noise cannot dominate a log
    fit. Returns None when fewer than 4 samples remain or the speed is not falling.
    """
    values = np.asarray(speed, dtype=float)[start+lo:start+hi]
    times = (np.arange(len(values))+lo)*tick_s
    keep = values > floor
    if keep.sum() < 4:
        return None
    slope = np.polyfit(times[keep], np.log(values[keep]), 1)[0]
    return float(-1./slope) if slope < 0. else None


def velocity_feedback_fit(net_target, velocity, start, lo, hi):
    """Least squares of the net wheel target on body velocity: target = slope v + intercept.

    The wheels are a pure damper, torque proportional to (target - wheel speed), and
    a rolling robot has wheel speed v/r. So a slope of 1/r makes the policy follow the
    motion and brake nothing, a slope of 0 is a full brake, and a nonzero intercept is a
    bias that survives at standstill. r2 is None when the target is constant.
    """
    y = np.asarray(net_target, dtype=float)[start+lo:start+hi]
    x = np.asarray(velocity, dtype=float)[start+lo:start+hi]
    if len(x) < 4 or np.ptp(x) == 0.:
        return None
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    total = float(((y-y.mean())**2).sum())
    residual = float(((y-design@np.array([slope, intercept]))**2).sum())
    # A constant target has a total variance of floating point noise, not exactly
    # zero, and dividing by it yields a meaningless r_squared far outside [0, 1].
    # Compare against the scale of the data instead of against zero.
    constant = total <= 1e-18*max(float((y**2).sum()), 1e-300)
    return {'n': int(len(x)), 'slope_rad_s_per_m_s': float(slope),
            'intercept_rad_s': float(intercept),
            'r_squared': None if constant else 1.-residual/total,
            'velocity_mean_m_s': float(x.mean()), 'target_mean_rad_s': float(y.mean())}


def damping_scale_from_slope(slope, wheel_radius=WHEEL_RADIUS_M):
    """Fraction of the zero-action damping that survives a velocity-following slope.

    torque = kd (s v - v/r) = -(kd/r)(1 - s r) v, so the damping is (1 - s r) times
    the zero-action value; 1 at s = 0 and 0 at s = 1/r, and above 1 for a negative slope.
    """
    return 1.-slope*wheel_radius


def ticks_until_speed_below(speed, start, limit=.02, hold=1):
    """Ticks after `start` until |speed| first drops below `limit` and stays for `hold` ticks.

    None when it never does inside the trace. This is how long the robot keeps
    moving after the cut, which the 10-frame history cannot span if it exceeds 10.
    """
    values = np.asarray(speed, dtype=float)
    for tick in range(start, len(values)-hold+1):
        if np.all(values[tick:tick+hold] < limit):
            return int(tick-start)
    return None


def signed_progress(boundary_xy, cut, heading, ticks):
    """Displacement from the cut boundary projected on the entry heading, per tick count."""
    boundary_xy = np.asarray(boundary_xy, dtype=float)
    heading = np.asarray(heading, dtype=float)
    norm = float(np.linalg.norm(heading))
    if norm == 0.:
        raise ValueError('heading must be nonzero')
    unit = heading/norm
    out = {}
    for k in ticks:
        if cut+k < len(boundary_xy):
            out[k] = float((boundary_xy[cut+k]-boundary_xy[cut]) @ unit)
    return out


def path_length_shares(boundary_xy, cut, bins=PATH_BINS):
    """Distance travelled inside each window after the cut, and its share of the total."""
    boundary_xy = np.asarray(boundary_xy, dtype=float)
    steps = np.hypot(*np.diff(boundary_xy[cut:], axis=0).T)
    lengths = {}
    for lo, hi in bins:
        lengths[f'{lo}-{hi}'] = float(steps[lo:hi].sum()) if lo < len(steps) else 0.
    total = sum(lengths.values())
    return {'path_m': lengths,
            'share': {k: (v/total if total > 0. else None) for k, v in lengths.items()},
            'total_m': total}


def path_density_per_tick(shares, bins=PATH_BINS):
    """Share of the total path per tick, for windows of different widths.

    Comparing raw shares across windows of 10, 15, 25, 50 and 100 ticks reads a
    wide window's larger share as "spread out" when its per-tick density may be
    far lower than a narrow window's. Dividing by the width removes that.
    """
    density = {}
    for lo, hi in bins:
        share = shares.get(f'{lo}-{hi}')
        density[f'{lo}-{hi}'] = None if share is None else share/(hi-lo)
    return density


def front_loading_ratio(density, early='0-10', late='100-200'):
    """How many times denser the path is in the early window than the late one."""
    a, b = density.get(early), density.get(late)
    if a is None or b is None or b == 0.:
        return None
    return a/b


def first_order_displacement(entry_speed, time_constant):
    """Stop distance of an exponential decay: the integral of v0 exp(-t/tau) is v0 tau."""
    if time_constant is None:
        return None
    return abs(entry_speed)*time_constant


def required_time_constant(entry_speed, displacement_limit):
    """Largest time constant that keeps a first-order stop inside the limit."""
    if entry_speed == 0.:
        raise ValueError('entry_speed must be nonzero')
    return displacement_limit/abs(entry_speed)


def windowed_velocity_feedback(net_target, velocity, start, windows):
    """The velocity feedback fit over several windows, each tagged with its speed range.

    One window's slope describes only the speeds that occur inside it. A fit over
    the slow creep after the stop says nothing about braking, and the two can even
    disagree in sign, so every fit carries the range it was measured over.
    """
    out = {}
    for lo, hi in windows:
        fit = velocity_feedback_fit(net_target, velocity, start, lo, hi)
        if fit is None:
            out[f'{lo}-{hi}'] = None
            continue
        segment = np.asarray(velocity, dtype=float)[start+lo:start+hi]
        fit['speed_range_m_s'] = [float(np.abs(segment).min()), float(np.abs(segment).max())]
        fit['damping_scale_vs_zero_action'] = damping_scale_from_slope(fit['slope_rad_s_per_m_s'])
        out[f'{lo}-{hi}'] = fit
    return out


def sample_cut_ticks(spec, count, seed=0):
    """Cut ticks of the frozen 03d course: two per full episode, in either mode.

    Mirrors D1Task03dEnv._make_training_schedule from its config ranges alone, so
    no environment is built. Translation cuts after the first drive and after the
    reverse drive; turn cuts after the paired +/- pair and after the second pair.
    """
    rng = np.random.default_rng(seed)
    first_zero = spec['first_zero_ticks']
    translation, turn = spec['translation'], spec['turn']
    mode_is_translation = rng.random(count) < spec['mode_probabilities']['translation']

    def integers(bounds, n):
        return rng.integers(bounds[0], bounds[1]+1, size=n)

    d = integers(translation['motion_ticks_inclusive'], count)
    g = integers(translation['middle_zero_ticks_inclusive'], count)
    t_first = first_zero+d
    t_second = first_zero+d+g+d
    d1 = integers(turn['motion_ticks_inclusive'], count)
    d2 = integers(turn['motion_ticks_inclusive'], count)
    gg = integers(turn['middle_zero_ticks_inclusive'], count)
    u_first = first_zero+2*d1
    u_second = first_zero+2*d1+gg+2*d2
    first = np.where(mode_is_translation, t_first, u_first)
    second = np.where(mode_is_translation, t_second, u_second)
    return np.stack([first, second], axis=1)


def cut_exposure(cuts, lengths, windows=EXPOSURE_WINDOWS):
    """Expected cut events and post-cut ticks seen, given the episode lengths that occurred.

    An episode that ended early (a fall) loses the cuts scheduled after its end.
    """
    cuts = np.asarray(cuts)
    lengths = np.asarray(lengths)
    unique, counts = np.unique(lengths, return_counts=True)
    events = 0.
    ticks = {w: 0. for w in windows}
    for length, n in zip(unique, counts):
        events += n*float((cuts < length).sum(axis=1).mean())
        for w in windows:
            ticks[w] += n*float(np.minimum(w, np.maximum(0, length-cuts)).sum(axis=1).mean())
    total_ticks = float(lengths.sum())
    return {'episodes': int(len(lengths)), 'recorded_ticks': int(total_ticks),
            'expected_cut_events': events,
            'expected_ticks_in_window': {str(w): ticks[w] for w in windows},
            'share_of_ticks_in_window': {str(w): (ticks[w]/total_ticks if total_ticks else None)
                                         for w in windows}}


# --- I/O -----------------------------------------------------------------------

def read_monitor_lengths(directory):
    lengths = []
    for path in sorted(Path(directory).glob('env_*.monitor.csv')):
        with path.open() as handle:
            rows = [r for r in handle if not r.startswith('#')]
        for row in csv.DictReader(rows):
            lengths.append(int(row['l']))
    return lengths


def analyse_trace(npz_path, receipt_row):
    with np.load(npz_path) as handle:
        trace = {k: handle[k] for k in handle.files}
    command = trace['command_during_interval']
    cuts = find_cut_ticks(command)
    if not cuts:
        return {'cuts': [], 'note': 'no motion-to-zero transition in this trace'}
    cut = cuts[-1]                       # the scored stop is the last cut
    boundary = trace['boundary_xy_m']
    wheels = np.asarray(trace['issued_action'], dtype=float)[:, list(WHEEL_ACTION_INDEX)]
    mean_square = np.mean(np.square(wheels), axis=1)
    net_target = wheels.mean(axis=1)*WHEEL_ACTION_SCALE
    horizon = len(boundary)-1
    after = slice(cut, horizon)

    entry = slice(max(0, cut-5), cut)
    vx = np.asarray(trace['body_vx_m_s'], dtype=float)
    vy = np.asarray(trace['body_vy_m_s'], dtype=float)
    entry_speed = float(np.hypot(vx[entry], vy[entry]).mean())
    entry_vx = float(vx[entry].mean())
    entry_wz = float(np.asarray(trace['body_wz_rad_s'], dtype=float)[entry].mean())
    yaw = float(np.asarray(trace['base_yaw_rad'], dtype=float)[cut-1])
    direction = 1. if entry_vx >= 0. else -1.
    heading = direction*np.array([np.cos(yaw), np.sin(yaw)])
    translation = abs(entry_vx) > .05

    ticks_available = horizon-cut
    result = {
        'cut_tick': cut, 'cut_time_s': float(trace['policy_interval_start_s'][cut]),
        'ticks_after_cut_available': ticks_available,
        'translation_task': bool(translation),
        'entry_speed_m_s': entry_speed, 'entry_vx_m_s': entry_vx, 'entry_wz_rad_s': entry_wz,
        'wheel_action_mean_square_per_tick': {
            f'{lo}-{hi}': float(mean_square[cut+lo:cut+hi].mean())
            for lo, hi in ((0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200))
            if cut+lo < horizon},
        'net_wheel_target_rad_s': {
            f'{lo}-{hi}': float(net_target[cut+lo:cut+hi].mean())
            for lo, hi in ((0, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200))
            if cut+lo < horizon},
        'zeroing_latency_ticks': zeroing_latency(mean_square[:horizon], cut),
        'history_frames_carrying_command': {str(k): history_frames_carrying_command(k)
                                            for k in (0, 5, 9, 10, 25)},
        'path_length': path_length_shares(boundary, cut),
        'final_euclidean_displacement_m': float(np.hypot(*(boundary[horizon]-boundary[cut]))),
    }
    if translation:
        result['signed_progress_along_entry_heading_m'] = {
            str(k): v for k, v in signed_progress(boundary, cut, heading, PROGRESS_TICKS+(200,)).items()}
        progress = signed_progress(boundary, cut, heading, (ticks_available,))
        result['signed_progress_at_end_m'] = progress.get(ticks_available)
    if translation:
        speed = np.hypot(vx, vy)
        result['decay_time_constant_s'] = {'ticks_10_to_50': decay_time_constant(speed, cut, 10, 50)}
        result['ticks_until_speed_below_0p02_m_s'] = ticks_until_speed_below(speed[:horizon], cut)
        result['seconds_until_speed_below_0p02_m_s'] = (
            None if result['ticks_until_speed_below_0p02_m_s'] is None
            else result['ticks_until_speed_below_0p02_m_s']*.02)
        fit = velocity_feedback_fit(net_target, vx, cut, 25, 200)
        result['velocity_feedback_fit_ticks_25_to_200'] = fit
        result['velocity_feedback_by_window'] = windowed_velocity_feedback(
            net_target, vx, cut, ((0, 25), (0, 50), (10, 50), (25, 100), (25, 200), (100, 200)))
        density = path_density_per_tick(result['path_length']['share'])
        result['path_density_share_per_tick'] = density
        result['front_loading_ratio_0_10_over_100_200'] = front_loading_ratio(density)
        tau = result['decay_time_constant_s']['ticks_10_to_50']
        predicted = first_order_displacement(entry_speed, tau)
        result['first_order_stop'] = {
            'entry_speed_m_s': entry_speed, 'time_constant_s': tau,
            'predicted_displacement_m': predicted,
            'measured_path_m': result['path_length']['total_m'],
            'measured_euclidean_m': result['final_euclidean_displacement_m'],
            'predicted_over_path': (predicted/result['path_length']['total_m']
                                    if predicted and result['path_length']['total_m'] else None),
            'time_constant_needed_for_0p05_m': required_time_constant(entry_speed, .05)
            if entry_speed else None}
        if fit is not None:
            scale = damping_scale_from_slope(fit['slope_rad_s_per_m_s'])
            result['damping_scale_vs_zero_action'] = scale
    if receipt_row is not None and receipt_row.get('stop_displacement_m') is not None:
        recomputed = result['final_euclidean_displacement_m']
        result['receipt_stop_displacement_m'] = receipt_row['stop_displacement_m']
        result['matches_receipt_to_1e-9'] = bool(
            abs(recomputed-receipt_row['stop_displacement_m']) < 1e-9)
    return result


def zero_wheel_reference(probe_dir):
    """Braking curve with the wheel actions forced to zero, from the 04b appendix probe."""
    probe_dir = Path(probe_dir)
    drive = [json.loads(x) for x in (probe_dir/'drive.jsonl').read_text().splitlines()]
    coast = [json.loads(x) for x in (probe_dir/'coast.jsonl').read_text().splitlines()]
    release_xy = np.array([drive[-1]['x'], drive[-1]['y']])
    release_speed = float(np.hypot(drive[-1]['vx'], drive[-1]['vy']))
    recent = np.array([[r['x'], r['y']] for r in drive[-10:]])
    heading = recent[-1]-recent[0]
    xy = np.array([[r['x'], r['y']] for r in coast])
    boundary = np.vstack([release_xy, xy])
    progress = signed_progress(boundary, 0, heading, PROGRESS_TICKS)
    speeds = np.array([np.hypot(r['vx'], r['vy']) for r in coast])
    tau = decay_time_constant(speeds, 0, 0, 8, floor=.01)
    return {'release_speed_m_s': release_speed, 'coast_ticks': len(coast),
            'decay_time_constant_s_first_8_ticks': tau,
            'implied_damping_per_s': (1./tau) if tau else None,
            'progress_m': {str(k): v for k, v in progress.items()},
            'equivalent_time_s': {str(k): v/release_speed for k, v in progress.items()},
            'caveat': ('release speed is far below the 0.13 to 0.2 m/s of the traces, so scaling '
                       'progress by entry speed assumes the braking is linear in speed; and the '
                       'speed rebounded after braking because forcing only the wheels to zero is '
                       'an inconsistent action pair for the legs')}


def compare_with_reference(trace_result, reference):
    """Actual progress against what zero-wheel braking would give at the same entry speed."""
    if not trace_result.get('translation_task'):
        return None
    speed = trace_result['entry_speed_m_s']
    actual = trace_result['signed_progress_along_entry_heading_m']
    out = {}
    for key, equivalent in reference['equivalent_time_s'].items():
        if key in actual and trace_result['entry_vx_m_s'] != 0.:
            predicted = equivalent*speed
            sign = 1. if trace_result['entry_vx_m_s'] >= 0. else -1.
            out[key] = {'actual_m': actual[key], 'zero_wheel_predicted_m': predicted,
                        'actual_over_predicted': (actual[key]/predicted if predicted else None)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', action='append', required=True, metavar='LABEL=DIR',
                        help='a frozen *_python_eval_01 directory; repeatable')
    parser.add_argument('--probe', type=Path, required=True,
                        help='zero_wheel_braking_probe_* directory')
    parser.add_argument('--train', action='append', default=[], metavar='LABEL=DIR',
                        help='a training run directory holding env_*.monitor.csv; repeatable')
    parser.add_argument('--config', type=Path, default=ROOT/'training/configs/pilot03d.json',
                        help='frozen config whose paired_schedule describes the course')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args(argv)

    reference = zero_wheel_reference(args.probe)
    versions = {}
    for item in args.eval:
        label, directory = item.split('=', 1)
        directory = Path(directory)
        rows = {}
        for line in (directory/'episodes.jsonl').read_text().strip().split('\n'):
            row = json.loads(line)
            rows[row['sequence']] = row
        per_task = {}
        for sequence in ('forward_stop', 'reverse_stop', 'turn_stop'):
            npz = directory/f'{sequence}_delay0_seed7001.npz'
            if not npz.is_file():
                per_task[sequence] = {'missing': str(npz)}
                continue
            traced = analyse_trace(npz, rows.get(sequence))
            traced['versus_zero_wheel_reference'] = compare_with_reference(traced, reference)
            traced['trace_sha256'] = sha(npz)
            per_task[sequence] = traced
        versions[label] = per_task

    spec = json.loads(args.config.read_text())['commands']['paired_schedule']
    cuts = sample_cut_ticks(spec, 20000)
    training = {}
    for item in args.train:
        label, directory = item.split('=', 1)
        lengths = read_monitor_lengths(directory)
        if lengths:
            training[label] = cut_exposure(cuts, lengths)

    report = {
        'format': 'd1-contract04-cut-to-zero-analysis-v1',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'physics_steps_executed': 0, 'transitions_consumed': 0,
        'mujoco_imported': False,
        'history_frames': HISTORY_LENGTH, 'history_span_s': HISTORY_LENGTH*.02,
        'zeroing_threshold_mean_square': ZERO_THRESHOLD, 'zeroing_hold_ticks': ZERO_HOLD_TICKS,
        'cut_tick_ranges_in_course': {
            'first_cut': [int(cuts[:, 0].min()), int(cuts[:, 0].max())],
            'second_cut': [int(cuts[:, 1].min()), int(cuts[:, 1].max())],
            'note': 'Monte Carlo over the frozen config ranges, 20000 samples, seed 0'},
        'zero_wheel_reference': reference,
        'versions': versions, 'training_exposure': training,
        'config': str(args.config.relative_to(ROOT)) if args.config.is_relative_to(ROOT)
        else str(args.config),
        'config_sha256': sha(args.config), 'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output/'report.json'),
                      'versions': sorted(versions), 'training': sorted(training),
                      'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
