#!/usr/bin/env python3
"""Offline assessment of real ROS command silence and native physical stopping."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def load(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_drive_vx(protocol):
    value = float(protocol['vx_m_s_until_stop'])
    if not np.isfinite(value) or not 0. < abs(value) <= .3:
        raise ValueError('Publisher drive vx must be finite, nonzero, and within [-0.3, 0.3]')
    return value


def evaluate(args):
    report, initial = load(args.run/'report.json'), load(args.run/'initial_state.json')
    sender, protocol = load(args.publisher_run/'report.json'), load(args.publisher_run/'protocol.json')
    drive_vx = protocol_drive_vx(protocol)
    schema = load(args.schema)
    native = read_rows(args.run/'trace.jsonl')
    events = read_rows(args.publisher_run/'publisher_events.jsonl')
    received = read_rows(args.publisher_run/'received_truth.jsonl')
    if len(native) > 6000:
        raise ValueError('Native trace exceeds one 6000-step pilot deployment allowance')
    for row in native:
        values = np.asarray([row['sim_time'], row['body_wz'], row['tilt'], row['base_z'],
                             *row['position'], *row['body_velocity'], *row['command']], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite truth in native trace')
    identity = {
        'model': report['model_sha256'] == args.expected_model_sha256,
        'policy': report['policy_sha256'] == args.expected_policy_sha256,
        'schema': report['schema_sha256'] == args.expected_schema_sha256 == sha(args.schema),
        'embedded_schema': report['schema'] == schema,
        'binary': report['runtime_binary_sha256'] == args.expected_runtime_sha256,
        'initial': initial['schema_sha256'] == report['schema_sha256'] and initial['model_sha256'] == report['model_sha256'],
        'libraries': report['mujoco_version'] == '3.3.0' and report['onnxruntime_version'] == '1.10.0',
        'live_mode': report['mode'] == 'live',
        'requested_12s_6000native': report['seconds_requested'] == 12. and report['planned_native_cap'] == 6000,
        'periods': report['native_dt'] == .002 and report['pd_dt'] == .004 and report['policy_dt'] == .02,
        'actual_counts': report['native_steps_attempted'] == report['native_steps_completed'] == len(native),
        'durable_counts': (args.run/'native_step_counter.bin').read_bytes() == struct.pack('=QQ', len(native), len(native)),
        'contiguous_count': [r['native_count'] for r in native] == list(range(1, len(native)+1)),
        'actual_time': all(abs(r['sim_time']-.002*r['native_count']) < 1e-10 for r in native),
        'pd_policy_counts': report['pd_updates'] == (len(native)+1)//2 and report['policy_updates'] == (len(native)+9)//10,
        'domain_and_topics': protocol['domain'] == 84 and protocol['cmd_topic'] == '/d1_selftrained/cmd_vel' and protocol['truth_topic'] == '/d1_selftrained/truth',
        'real_publisher_protocol': protocol['format'] == 'd1-real-ros-watchdog-probe-v1' and protocol['stop_after_8s_sends_no_zero_message'] is True,
        'unchanged_2_8_12s_windows': protocol['zero_command_until_sim_s'] == 2. and protocol['stop_publishing_at_sim_s'] == 8. and protocol['native_runtime_horizon_sim_s'] == 12.,
        'frozen_sender_script': protocol['script_sha256'] == sha(Path(__file__).parent/'live_watchdog03.py'),
    }
    publications = [e for e in events if e['event'] == 'publish']
    stop_events = [e for e in events if e['event'] == 'stop_all_publications']
    silence = bool(len(stop_events) == 1 and publications and
                   all(e['monotonic_s'] < stop_events[0]['monotonic_s'] for e in publications) and
                   all(e['latest_truth_sim_time_s'] < 8. for e in publications))
    identity['no_publications_after_recorded_stop'] = silence
    identity['sender_commands_follow_schedule'] = all(
        e['command'] == ([0., 0., 0.] if e['latest_truth_sim_time_s'] < 2. else [drive_vx, 0., 0.]) for e in publications)
    identity['native_commands_match_requested_drive_or_zero'] = all(
        np.allclose(r['command'], [0., 0., 0.], rtol=0., atol=1e-9) or
        np.allclose(r['command'], [drive_vx, 0., 0.], rtol=0., atol=1e-9) for r in native)
    motion = False
    zero_index = None
    for i, row in enumerate(native):
        cmd = np.asarray(row['command'], dtype=np.float64)
        if cmd.shape != (3,) or not np.isfinite(cmd).all():
            raise ValueError('Invalid command in native trace')
        if abs(cmd[0]) > 1e-9:
            motion = True
        elif motion and row['sim_time']-.002 >= 8.-1e-9:
            zero_index = i
            break
    values = {'actual_nonzero_to_zero_command_time_s': None,
              'stop_publishing_at_8_to_actual_zero_duration_sim_s': None,
              'path_length_from_8_to_actual_zero_m': None,
              'displacement_from_8_to_actual_zero_m': None,
              'stop_displacement_from_actual_zero_to_12_m': None,
              'stop_last_second_horizontal_speed_m_s': None,
              'stop_last_second_abs_wz_rad_s': None,
              'observer_last_motion_publish_to_zero_truth_delay_wall_s': None}
    zero_time = None
    if zero_index is not None:
        zero_time = native[zero_index]['sim_time']-.002
        before_zero = np.asarray((native[zero_index-1]['position'] if zero_index else initial['qpos'][:3])[:2], dtype=np.float64)
        values['actual_nonzero_to_zero_command_time_s'] = zero_time
        values['stop_publishing_at_8_to_actual_zero_duration_sim_s'] = zero_time-8.
        if len(native) >= 4000 and zero_index >= 4000:
            positions = np.asarray([row['position'][:2] for row in native[3999:zero_index]], dtype=np.float64)
            values['path_length_from_8_to_actual_zero_m'] = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            values['displacement_from_8_to_actual_zero_m'] = float(np.linalg.norm(before_zero-positions[0]))
        if len(native) == 6000:
            values['stop_displacement_from_actual_zero_to_12_m'] = float(np.linalg.norm(np.asarray(native[-1]['position'][:2])-before_zero))
        last_publication = sender.get('last_motion_publication')
        zero_receipts = [r for r in received if r['truth']['sim_time'] >= zero_time and r['truth']['sim_time'] >= 8. and r['truth']['command'] == [0., 0., 0.]]
        if last_publication and zero_receipts:
            values['observer_last_motion_publish_to_zero_truth_delay_wall_s'] = zero_receipts[0]['received_monotonic_s']-last_publication['monotonic_s']
    last_second = [r for r in native if r['native_count'] % 10 == 0 and r['sim_time'] > 11.+1e-10]
    if len(last_second) == 50:
        values['stop_last_second_horizontal_speed_m_s'] = float(np.mean([np.hypot(r['body_velocity'][0], r['body_velocity'][1]) for r in last_second]))
        values['stop_last_second_abs_wz_rad_s'] = float(np.mean([abs(r['body_wz']) for r in last_second]))
    thresholds = {'stop_last_second_horizontal_speed_m_s': .02,
                  'stop_last_second_abs_wz_rad_s': .03,
                  'stop_displacement_from_actual_zero_to_12_m': .05}
    flags = {'completed_12s': len(native) == 6000 and report['status'] == 'COMPLETED' and report['terminal_reason'] == 'TIME_LIMIT',
             'no_fall_or_illegal_support': bool(native) and all(r['tilt'] <= .6 and r['base_z'] >= report['h_nom']-.18 and not r['illegal_body_support_since_policy'] for r in native),
             'real_motion_command_was_applied': motion,
             'actual_watchdog_zero_after_silence_observed': zero_index is not None and silence and report['watchdog_stale_policy_ticks'] > 0,
             'zero_held_for_remainder': zero_index is not None and all(r['command'] == [0., 0., 0.] for r in native[zero_index:]),
             'real_ros_capture_completed': sender['status'] == 'CAPTURE_COMPLETE' and len(received) > 0}
    for key, threshold in thresholds.items():
        flags[key] = values[key] is not None and values[key] <= threshold
    valid = all(identity.values())
    return {'status': 'INVALID' if not valid else ('PASS' if all(flags.values()) else 'FAIL'),
            'physics_steps_executed_by_evaluator': 0, 'actual_native_steps': report['native_steps_completed'],
            'drive_vx_m_s_from_publisher_protocol': drive_vx,
            'scope': 'live_watchdog_engineering_only',
            'qualifies_overall_pilot': False,
            'identities_and_counts': identity, 'physical_and_watchdog_flags': flags,
            'metrics': values, 'thresholds': thresholds,
            'runtime_terminal_reason': report['terminal_reason'], 'publisher_terminal_reason': sender['terminal_reason'],
            'watchdog_triggered': zero_index is not None,
            'untriggered_or_early_failure_consumes_trial_no_automatic_retry': True,
            'timing_limitation': 'External last-publication and truth-receipt timestamps measure observer delay; the frozen runtime does not log native callback-receive age. This does not independently measure the internal exact 0.5s timeout.',
            'stopping_reference': 'Displacement gate begins at the native trace actual nonzero-to-zero command boundary. Motion and elapsed time from publication stop at 8s until that boundary are reported separately, never omitted.',
            'input_sha256': {str(p): sha(p) for p in [args.run/'report.json', args.run/'trace.jsonl',
                args.run/'initial_state.json', args.publisher_run/'report.json', args.publisher_run/'protocol.json',
                args.publisher_run/'publisher_events.jsonl', args.publisher_run/'received_truth.jsonl', args.schema]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publisher-run', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    for name in ('model', 'policy', 'schema', 'runtime'):
        parser.add_argument(f'--expected-{name}-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = evaluate(args)
    except Exception as exc:
        result = {'status': 'INVALID', 'physics_steps_executed_by_evaluator': 0,
                  'error': f'{type(exc).__name__}: {exc}'}
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
