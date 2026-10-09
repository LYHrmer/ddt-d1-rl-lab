#!/usr/bin/env python3
"""Real ROS command publisher for one already-authorized 12 s live watchdog trial.

No MuJoCo import/integration occurs here, but publishing commands controls the
separately running native simulator. Root must release and count that trial.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def bounded_drive_vx(value):
    value = float(value)
    if not math.isfinite(value) or not 0. < abs(value) <= .3:
        raise ValueError('drive-vx must be finite, nonzero, and within [-0.3, 0.3] m/s')
    return value


def command_at_sim_time(sim_time, drive_vx):
    """Fixed 0..2 zero, 2..8 drive, then no publication (not a zero message)."""
    if sim_time >= 8.:
        return None
    return [0. if sim_time < 2. else drive_vx, 0., 0.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-live-publisher', action='store_true', required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=25.)
    parser.add_argument('--drive-vx', type=bounded_drive_vx, default=.2,
                        help='2..8s vx command in m/s, nonzero within +/-0.3; default +0.2, reverse probe -0.1')
    args = parser.parse_args()
    if not 12 <= args.max_wall_seconds <= 30:
        parser.error('Wall cap must be between 12 and 30 seconds')
    if os.environ.get('ROS_DOMAIN_ID') != '84':
        parser.error('This dedicated trial requires explicit ROS_DOMAIN_ID=84')
    args.output.mkdir(parents=True, exist_ok=False)
    # Lazy imports make --help purely offline and keep other offline tools light.
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    started = time.monotonic()
    events = (args.output/'publisher_events.jsonl').open('x')
    truth_stream = (args.output/'received_truth.jsonl').open('x')
    requested = {'format': 'd1-real-ros-watchdog-probe-v1', 'domain': 84,
                 'cmd_topic': '/d1_selftrained/cmd_vel', 'truth_topic': '/d1_selftrained/truth',
                 'publication_period_wall_s': .05, 'schedule_basis': 'last received native truth sim_time',
                 'zero_command_until_sim_s': 2., 'vx_m_s_until_stop': args.drive_vx,
                 'stop_publishing_at_sim_s': 8., 'native_runtime_horizon_sim_s': 12.,
                 'max_wall_seconds': args.max_wall_seconds,
                 'stop_after_8s_sends_no_zero_message': True,
                 'scope': 'live_watchdog_engineering_only_not_overall_pilot_qualification',
                 'no_physics_steps_in_publisher': True,
                 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'protocol.json').write_text(json.dumps(requested, indent=2)+'\n')
    state = {'latest': None, 'received_at': None, 'error': None, 'truth_count': 0,
             'tx_count': 0, 'motion_count': 0, 'last_motion': None, 'stop_at': None}

    class Probe(Node):
        def __init__(self):
            super().__init__('d1_watchdog_probe03b')
            self.sender = self.create_publisher(Twist, requested['cmd_topic'], 1)
            self.receiver = self.create_subscription(String, requested['truth_topic'], self.receive, 100)

        def receive(self, message):
            now = time.monotonic()
            try:
                value = json.loads(message.data)
                simulation_time = float(value['sim_time'])
                if not 0 < simulation_time <= 12.000001:
                    raise ValueError('Truth time is outside the 12 s trial')
                if state['latest'] is not None and simulation_time <= state['latest']['sim_time']:
                    raise ValueError('Truth timestamps duplicated or regressed')
                state['latest'], state['received_at'] = value, now
                state['truth_count'] += 1
                truth_stream.write(json.dumps({'received_monotonic_s': now, 'truth': value}, allow_nan=False)+'\n')
                truth_stream.flush()
            except Exception as exc:
                state['error'] = f'{type(exc).__name__}: {exc}'

    rclpy.init(args=[])
    node = Probe()
    next_publication = started
    terminal = 'WALL_LIMIT'
    try:
        while rclpy.ok() and time.monotonic()-started < args.max_wall_seconds:
            rclpy.spin_once(node, timeout_sec=.01)
            now = time.monotonic()
            if state['error']:
                terminal = 'INVALID_TRUTH'
                break
            if state['latest'] is None:
                continue
            if (node.count_publishers(requested['truth_topic']) != 1 or
                    node.count_publishers(requested['cmd_topic']) != 1 or
                    node.sender.get_subscription_count() != 1):
                # Discovery can settle before movement; ambiguous endpoints stop
                # all transmissions and leave native watchdog to request zero.
                if state['motion_count']:
                    terminal = 'ROS_ENDPOINT_COUNT_CHANGED'
                    break
                continue
            sim_time = state['latest']['sim_time']
            if sim_time >= 8. and state['stop_at'] is None:
                state['stop_at'] = {'monotonic_s': now, 'latest_truth_sim_time_s': sim_time}
                events.write(json.dumps({'event': 'stop_all_publications', **state['stop_at']})+'\n')
                events.flush()
            if sim_time >= 12.-1e-8:
                terminal = 'TRUTH_REACHED_12S'
                break
            if state['stop_at'] is not None:
                continue
            if now-state['received_at'] > .25:
                continue
            if now < next_publication:
                continue
            next_publication = now+.05
            command = command_at_sim_time(sim_time, args.drive_vx)
            if command is None:
                continue
            vx = command[0]
            message = Twist()
            message.linear.x = vx
            node.sender.publish(message)
            event = {'event': 'publish', 'monotonic_s': now,
                     'latest_truth_sim_time_s': sim_time, 'command': command}
            events.write(json.dumps(event)+'\n')
            events.flush()
            state['tx_count'] += 1
            if vx:
                state['motion_count'] += 1
                state['last_motion'] = event
    except BaseException as exc:
        state['error'] = f'{type(exc).__name__}: {exc}'
        terminal = 'PUBLISHER_EXCEPTION'
    finally:
        # Intentionally do not publish zero in finally: that would conceal the
        # requested stale-command watchdog experiment.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        events.close()
        truth_stream.close()
    result = {'status': 'CAPTURE_COMPLETE' if terminal == 'TRUTH_REACHED_12S' else 'INCOMPLETE',
              'terminal_reason': terminal, 'error': state['error'],
              'physics_steps_executed_by_publisher': 0,
              'drive_vx_m_s': args.drive_vx,
              'scope': 'live_watchdog_engineering_only_not_overall_pilot_qualification',
              'commands_sent': state['tx_count'], 'motion_commands_sent': state['motion_count'],
              'truth_messages_received': state['truth_count'],
              'last_motion_publication': state['last_motion'], 'stop_publications': state['stop_at'],
              'last_truth_sim_time_s': None if state['latest'] is None else state['latest']['sim_time'],
              'elapsed_monotonic_s': time.monotonic()-started,
              'physical_trial_still_counts_if_incomplete': True,
              'interpretation': 'Real ROS publication/receipt only. Physical stopping is scored separately against the native trace. Observer timestamps are not the native internal callback-receive age.'}
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'CAPTURE_COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
