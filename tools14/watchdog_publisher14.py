#!/usr/bin/env python3
"""Publish the fixed watchdog trial with complete startup and cleanup receipts.

Publishing controls a separately running native simulator. This entry grants no
trial allowance. Its v14 receipts are incompatible with evaluate_watchdog03.py.
Imports remain lazy; this round verifies injected ROS bindings only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / 'runtime03/live_watchdog03.py'
LEGACY_SHA256 = '814b4101be302cff307c35c45a853209d2ea064583dbaf64eb7799cdc9f8c825'
CMD_TOPIC = '/d1_selftrained/cmd_vel'
TRUTH_TOPIC = '/d1_selftrained/truth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def bounded_drive_vx(value):
    value = float(value)
    if not math.isfinite(value) or not 0. < abs(value) <= .3:
        raise ValueError('drive-vx must be finite, nonzero, and within [-0.3, 0.3] m/s')
    return value


def command_at_sim_time(sim_time, drive_vx):
    if sim_time >= 8.:
        return None
    return [0. if sim_time < 2. else drive_vx, 0., 0.]


def load_ros():
    """Load real ROS only after the exclusive receipt directory is claimed."""
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String
    return SimpleNamespace(rclpy=rclpy, Context=Context, Executor=SingleThreadedExecutor,
                           Node=Node, Twist=Twist, String=String)


def capture(output, drive_vx=.2, max_wall_seconds=25., *, loader=None, clock=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never alter an existing trial.
    report = {'format': 'd1-watchdog-publisher-receipt-v14', 'status': 'INCOMPLETE',
              'terminal_reason': None, 'capture_terminal_reason': None, 'error': None,
              'failure_stage': None, 'cleanup_complete': True, 'cleanup_errors': [],
              'cleanup_attempts': [], 'commands_sent': 0, 'motion_commands_sent': 0,
              'truth_messages_received': 0, 'stale_truth_publication_skips': 0,
              'last_motion_publication': None, 'stop_publications': None,
              'last_truth_sim_time_s': None, 'elapsed_monotonic_s': None,
              'physics_steps_executed_by_publisher': 0, 'trial_authorised_by_tool': False,
              'qualifies_overall_pilot': False, 'legacy_evaluator_compatible': False,
              'physical_trial_still_counts_if_incomplete': True,
              'scope': 'live_watchdog_engineering_only_not_overall_pilot_qualification',
              'ros_binding_source': 'injected' if loader is not None else 'default_lazy_imports',
              'ros_context_created': False, 'ros_node_created': False,
              'entry_sha256': None, 'legacy_publisher_sha256': None}
    clock = clock or time.monotonic
    context = node = executor = events = truth_stream = None
    state = {'latest': None, 'received_at': None, 'error': None, 'last_clock': None}
    started = None
    stage = 'configuration'
    terminal = 'WALL_LIMIT'

    def stamp():
        now = float(clock())
        if not math.isfinite(now) or (state['last_clock'] is not None and now < state['last_clock']):
            raise ValueError('monotonic clock is nonfinite or regressed')
        state['last_clock'] = now
        return now

    def receive(message):
        try:
            now = stamp()
            value = json.loads(message.data)
            raw_time = value['sim_time']
            if type(raw_time) not in (int, float):
                raise ValueError('Truth sim_time must be a JSON number')
            sim_time = float(raw_time)
            if not math.isfinite(sim_time) or not 0. < sim_time <= 12.000001:
                raise ValueError('Truth time is outside the 12 s trial')
            if state['latest'] is not None and sim_time <= state['latest']['sim_time']:
                raise ValueError('Truth timestamps duplicated or regressed')
            # Reject an unrecordable row before counting or using it for commands.
            row = json.dumps({'received_monotonic_s': now, 'truth': value}, allow_nan=False)
            truth_stream.write(row + '\n')
            truth_stream.flush()
            state['latest'], state['received_at'] = value, now
            report['truth_messages_received'] += 1
        except Exception as exc:
            state['error'] = f'{type(exc).__name__}: {exc}'

    def event(value):
        events.write(json.dumps(value, allow_nan=False) + '\n')
        events.flush()

    try:
        report['entry_sha256'] = sha(__file__)
        report['legacy_publisher_sha256'] = sha(LEGACY)
        if report['legacy_publisher_sha256'] != LEGACY_SHA256:
            raise ValueError('frozen legacy publisher identity changed')
        drive_vx = bounded_drive_vx(drive_vx)
        if not math.isfinite(max_wall_seconds) or not 12. <= max_wall_seconds <= 30.:
            raise ValueError('Wall cap must be between 12 and 30 seconds')
        protocol = {'format': 'd1-real-ros-watchdog-probe-v14', 'domain': 84,
                    'cmd_topic': CMD_TOPIC, 'truth_topic': TRUTH_TOPIC,
                    'publication_period_wall_s': .05, 'truth_freshness_limit_wall_s': .25,
                    'schedule_basis': 'last received native truth sim_time',
                    'zero_command_until_sim_s': 2., 'vx_m_s_until_stop': drive_vx,
                    'stop_publishing_at_sim_s': 8., 'native_runtime_horizon_sim_s': 12.,
                    'max_wall_seconds': max_wall_seconds, 'stop_after_8s_sends_no_zero_message': True,
                    'legacy_evaluator_compatible': False, 'trial_authorised_by_tool': False,
                    'scope': report['scope'], 'no_physics_steps_in_publisher': True,
                    'script_sha256': report['entry_sha256'], 'legacy_publisher_sha256': LEGACY_SHA256}
        write_json(output / 'protocol.json', protocol)
        if os.environ.get('ROS_DOMAIN_ID') != '84':
            raise ValueError('This dedicated trial requires explicit ROS_DOMAIN_ID=84')
        started = stamp()
        stage = 'capture_open'
        events = (output / 'publisher_events.jsonl').open('x')
        truth_stream = (output / 'received_truth.jsonl').open('x')
        stage = 'imports'
        ros = (loader or load_ros)()
        stage = 'context'
        context = ros.Context()
        report['ros_context_created'] = True
        stage = 'init'
        ros.rclpy.init(args=[], context=context)
        stage = 'node'
        node = ros.Node('d1_watchdog_probe14', context=context)
        report['ros_node_created'] = True
        sender = node.create_publisher(ros.Twist, CMD_TOPIC, 1)
        node.create_subscription(ros.String, TRUTH_TOPIC, receive, 100)
        stage = 'executor'
        executor = ros.Executor(context=context)
        executor.add_node(node)
        stage = 'running'
        next_publication = started
        while True:
            if not context.ok():
                terminal = 'ROS_SHUTDOWN'
                break
            if stamp() - started >= max_wall_seconds:
                break
            executor.spin_once(timeout_sec=.01)
            now = stamp()
            if state['error']:
                report['error'] = state['error']
                terminal = 'INVALID_TRUTH'
                break
            if not context.ok():
                terminal = 'ROS_SHUTDOWN'
                break
            if state['latest'] is None:
                continue
            sim_time = state['latest']['sim_time']
            if sim_time >= 8. and report['stop_publications'] is None:
                report['stop_publications'] = {'monotonic_s': now, 'latest_truth_sim_time_s': sim_time}
                event({'event': 'stop_all_publications', **report['stop_publications']})
            # A received valid final truth remains captured when its producer exits.
            if sim_time >= 12. - 1e-8:
                terminal = 'TRUTH_REACHED_12S'
                break
            if (node.count_publishers(TRUTH_TOPIC) != 1 or node.count_publishers(CMD_TOPIC) != 1
                    or sender.get_subscription_count() != 1):
                if report['motion_commands_sent']:
                    terminal = 'ROS_ENDPOINT_COUNT_CHANGED'
                    break
                continue
            if report['stop_publications'] is not None:
                continue
            if now - state['received_at'] > .25:
                report['stale_truth_publication_skips'] += 1
                continue
            if now < next_publication:
                continue
            next_publication = now + .05
            command = command_at_sim_time(sim_time, drive_vx)
            if command is None:
                continue
            message = ros.Twist()
            message.linear.x = command[0]
            sender.publish(message)
            # Count a successful publish even if recording subsequently fails.
            report['commands_sent'] += 1
            publication = {'event': 'publish', 'monotonic_s': now,
                           'latest_truth_sim_time_s': sim_time, 'command': command}
            if command[0]:
                report['motion_commands_sent'] += 1
                report['last_motion_publication'] = publication
            event(publication)
    except BaseException as exc:
        report['failure_stage'] = stage
        report['error'] = f'{type(exc).__name__}: {exc}'
        terminal = {'configuration': 'CONFIGURATION_FAILED', 'capture_open': 'CAPTURE_OPEN_FAILED',
                    'imports': 'ROS_IMPORT_FAILED', 'context': 'ROS_CONTEXT_FAILED',
                    'init': 'ROS_INIT_FAILED', 'node': 'ROS_NODE_FAILED',
                    'executor': 'ROS_EXECUTOR_FAILED'}.get(stage, 'PUBLISHER_EXCEPTION')
    finally:
        report['capture_terminal_reason'] = terminal

        def clean(name, action):
            report['cleanup_attempts'].append(name)
            try:
                if action() is False:
                    raise RuntimeError('cleanup did not complete')
            except BaseException as exc:
                report['cleanup_errors'].append({'resource': name, 'error': f'{type(exc).__name__}: {exc}'})

        # Never send zero here: doing so would conceal the silence experiment.
        if executor is not None:
            clean('executor', lambda: executor.shutdown(timeout_sec=1.))
        if node is not None:
            clean('node', node.destroy_node)
        if context is not None:
            clean('context', context.try_shutdown)  # Only this dedicated context.
        if events is not None:
            clean('publisher_events', events.close)
        if truth_stream is not None:
            clean('received_truth', truth_stream.close)
        report['cleanup_complete'] = not report['cleanup_errors']
        if not report['cleanup_complete'] and terminal == 'TRUTH_REACHED_12S':
            terminal = 'CLEANUP_FAILED'
        report['terminal_reason'] = terminal
        report['status'] = 'CAPTURE_COMPLETE' if terminal == 'TRUTH_REACHED_12S' else 'INCOMPLETE'
        report['last_truth_sim_time_s'] = None if state['latest'] is None else state['latest']['sim_time']
        if started is not None and state['last_clock'] is not None:
            report['elapsed_monotonic_s'] = state['last_clock'] - started
        report['interpretation'] = ('Publication/receipt capture only; native physical stopping is not scored. '
                                    'Observer timestamps do not measure native internal callback age. '
                                    'The wall cap bounds the capture loop, not blocked ROS import or native calls.')
        write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'CAPTURE_COMPLETE' else 1


def main(argv=None, *, loader=None, clock=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-live-publisher', action='store_true', required=True,
                        help='explicit execution intent; separate physical-trial authorisation is still required')
    parser.add_argument('--max-wall-seconds', type=float, default=25.)
    parser.add_argument('--drive-vx', type=bounded_drive_vx, default=.2)
    args = parser.parse_args(argv)
    return capture(args.output, args.drive_vx, args.max_wall_seconds, loader=loader, clock=clock)


if __name__ == '__main__':
    raise SystemExit(main())
