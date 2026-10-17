import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tools14/watchdog_publisher14.py'


def load_entry():
    spec = importlib.util.spec_from_file_location('watchdog_publisher14_tested', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeROS:
    """Fake ROS bindings drive the real publisher's main and receive callback."""

    def __init__(self, frames=(), failure=None):
        self.frames = list(frames)
        self.failure = failure
        self.now = 0.
        self.current = {}
        self.calls = []
        self.contexts = []
        self.nodes = []
        self.publications = []
        owner = self

        class Context:
            def __init__(self):
                owner.fail('context')
                self.active = False
                owner.contexts.append(self)

            def ok(self):
                return self.active

            def try_shutdown(self):
                owner.calls.append('context_shutdown')
                self.active = False
                owner.fail('context_shutdown')

        class Twist:
            def __init__(self):
                self.linear = SimpleNamespace(x=0., y=0., z=0.)
                self.angular = SimpleNamespace(x=0., y=0., z=0.)

        class Sender:
            def publish(self, message):
                owner.fail('publish')
                owner.publications.append((owner.now, message.linear.x))

            def get_subscription_count(self):
                return owner.current.get('subscriptions', 1)

        class Node:
            def __init__(self, name, *, context):
                owner.fail('node')
                assert context in owner.contexts
                self.context = context
                owner.nodes.append(self)

            def create_publisher(self, *args):
                owner.fail('publisher')
                return Sender()

            def create_subscription(self, cls, topic, callback, depth):
                owner.fail('subscription')
                self.callback = callback
                return object()

            def count_publishers(self, topic):
                return owner.current.get('truth_publishers' if topic.endswith('/truth') else 'cmd_publishers', 1)

            def destroy_node(self):
                owner.calls.append('node_destroy')
                owner.fail('node_destroy')

        class Executor:
            def __init__(self, *, context):
                owner.fail('executor')
                self.context = context

            def add_node(self, node):
                self.node = node
                owner.fail('add_node')

            def spin_once(self, *, timeout_sec):
                assert timeout_sec == .01
                owner.current = owner.frames.pop(0) if owner.frames else {'wall': 31.}
                owner.now = owner.current.get('wall', owner.now + .1)
                if 'exception' in owner.current:
                    raise owner.current['exception']
                if owner.current.get('shutdown'):
                    self.context.active = False
                if 'truth' in owner.current:
                    truth = owner.current['truth']
                    data = truth if isinstance(truth, str) else json.dumps(truth)
                    self.node.callback(SimpleNamespace(data=data))

            def shutdown(self, *, timeout_sec):
                assert timeout_sec == 1.
                owner.calls.append('executor_shutdown')
                owner.fail('executor_shutdown')
                return owner.failure != 'executor_timeout'

        def init(*, args, context):
            assert args == []
            owner.calls.append('init')
            context.active = True  # Also tests cleanup after partial init.
            owner.fail('init')

        self.bindings = SimpleNamespace(rclpy=SimpleNamespace(init=init), Context=Context,
                                        Node=Node, Executor=Executor, Twist=Twist, String=object)

    def fail(self, stage):
        if stage == self.failure:
            raise RuntimeError('injected ' + stage)

    def loader(self):
        self.fail('imports')
        return self.bindings

    def clock(self):
        return self.now


def invoke(tmp_path, monkeypatch, fake, **kwargs):
    monkeypatch.setenv('ROS_DOMAIN_ID', '84')
    output = tmp_path / 'capture'
    entry = load_entry()
    argv = ['--output', str(output), '--execute-live-publisher']
    if 'drive_vx' in kwargs:
        argv += ['--drive-vx', str(kwargs['drive_vx'])]
    code = entry.main(argv, loader=fake.loader, clock=fake.clock)
    return code, json.loads((output / 'report.json').read_text()), output


@pytest.mark.parametrize('failure,reason', [
    ('imports', 'ROS_IMPORT_FAILED'), ('context', 'ROS_CONTEXT_FAILED'),
    ('init', 'ROS_INIT_FAILED'), ('node', 'ROS_NODE_FAILED'),
    ('publisher', 'ROS_NODE_FAILED'), ('subscription', 'ROS_NODE_FAILED'),
    ('executor', 'ROS_EXECUTOR_FAILED'), ('add_node', 'ROS_EXECUTOR_FAILED'),
])
def test_every_startup_failure_has_a_receipt(tmp_path, monkeypatch, failure, reason):
    fake = FakeROS(failure=failure)
    code, report, output = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['status'] == 'INCOMPLETE'
    assert report['terminal_reason'] == reason
    assert report['commands_sent'] == 0 and fake.publications == []
    assert report['trial_authorised_by_tool'] is False
    assert report['physics_steps_executed_by_publisher'] == 0
    assert (output / 'protocol.json').is_file()
    assert all(not context.active for context in fake.contexts)
    if failure in ('publisher', 'subscription', 'executor', 'add_node'):
        assert 'node_destroy' in fake.calls


def test_success_uses_exact_windows_and_never_sends_final_zero(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 1.}},
                    {'wall': .2, 'truth': {'sim_time': 2.}},
                    {'wall': .3, 'truth': {'sim_time': 7.99}},
                    {'wall': .4, 'truth': {'sim_time': 8.}},
                    {'wall': .5, 'truth': {'sim_time': 12.}}])
    code, report, output = invoke(tmp_path, monkeypatch, fake, drive_vx=-.1)
    assert code == 0 and report['status'] == 'CAPTURE_COMPLETE'
    assert fake.publications == [(.1, 0.), (.2, -.1), (.3, -.1)]
    assert report['stop_publications']['latest_truth_sim_time_s'] == 8.
    assert report['cleanup_complete'] is True
    assert fake.calls[-3:] == ['executor_shutdown', 'node_destroy', 'context_shutdown']
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['format'] == 'd1-real-ros-watchdog-probe-v14'
    assert protocol['legacy_evaluator_compatible'] is False
    assert protocol['legacy_publisher_sha256'] == '814b4101be302cff307c35c45a853209d2ea064583dbaf64eb7799cdc9f8c825'


def test_stale_truth_skips_then_fresh_truth_resumes(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}}, {'wall': .4},
                    {'wall': .5, 'truth': {'sim_time': 3.1}},
                    {'wall': .6, 'truth': {'sim_time': 12.}}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 0 and fake.publications == [(.1, .2), (.5, .2)]
    assert report['stale_truth_publication_skips'] == 1


@pytest.mark.parametrize('bad_truth', [
    {'sim_time': float('nan')}, {'sim_time': float('inf')}, {'sim_time': 0.},
    {'sim_time': 12.1}, {'sim_time': 3.}, {'sim_time': 2.9},
    {'sim_time': '3.1'}, {'sim_time': True}, 'not-json',
    {'sim_time': 3.1, 'unused_field': float('nan')},
])
def test_invalid_duplicate_or_regressed_truth_stops_publications(tmp_path, monkeypatch, bad_truth):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}}, {'wall': .2, 'truth': bad_truth}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'INVALID_TRUTH'
    assert fake.publications == [(.1, .2)]


@pytest.mark.parametrize('changed', ['truth_publishers', 'cmd_publishers', 'subscriptions'])
def test_endpoint_change_after_motion_stops(tmp_path, monkeypatch, changed):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}},
                    {'wall': .2, 'truth': {'sim_time': 4.}, changed: 0}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'ROS_ENDPOINT_COUNT_CHANGED'
    assert fake.publications == [(.1, .2)]


def test_ambiguous_endpoints_before_motion_can_settle(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}, 'cmd_publishers': 2},
                    {'wall': .2, 'truth': {'sim_time': 3.1}},
                    {'wall': .3, 'truth': {'sim_time': 12.}}])
    code, _, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 0 and fake.publications == [(.2, .2)]


def test_final_truth_is_complete_even_if_native_endpoint_has_left(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}},
                    {'wall': .2, 'truth': {'sim_time': 12.}, 'truth_publishers': 0}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 0 and report['terminal_reason'] == 'TRUTH_REACHED_12S'
    assert fake.publications == [(.1, .2)]
    assert report['qualifies_overall_pilot'] is False


def test_ros_shutdown_is_distinct_and_does_not_publish_again(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}},
                    {'wall': .2, 'truth': {'sim_time': 4.}, 'shutdown': True}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'ROS_SHUTDOWN'
    assert report['elapsed_monotonic_s'] < 25.
    assert fake.publications == [(.1, .2)]


@pytest.mark.parametrize('failure', ['executor_shutdown', 'executor_timeout', 'node_destroy', 'context_shutdown'])
def test_cleanup_failures_cannot_claim_capture_success_and_do_not_skip_other_cleanup(tmp_path, monkeypatch, failure):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 12.}}], failure=failure)
    code, report, output = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['status'] == 'INCOMPLETE'
    assert report['terminal_reason'] == 'CLEANUP_FAILED' and report['cleanup_complete'] is False
    assert report['capture_terminal_reason'] == 'TRUTH_REACHED_12S'
    assert fake.calls[-3:] == ['executor_shutdown', 'node_destroy', 'context_shutdown']
    assert (output / 'received_truth.jsonl').read_text()


def test_stream_close_failure_still_closes_other_stream_and_writes_receipt(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 12.}}])
    original_open = Path.open
    closed = []

    class CloseFailure:
        def __init__(self, stream, name):
            self.stream, self.name = stream, name

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.stream.close()
            closed.append(self.name)
            if self.name == 'publisher_events.jsonl':
                raise OSError('injected close failure')

    def open_path(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name in ('publisher_events.jsonl', 'received_truth.jsonl') and args and args[0] == 'x':
            return CloseFailure(stream, path.name)
        return stream

    monkeypatch.setattr(Path, 'open', open_path)
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'CLEANUP_FAILED'
    assert closed == ['publisher_events.jsonl', 'received_truth.jsonl']
    assert any(e['resource'] == 'publisher_events' for e in report['cleanup_errors'])


@pytest.mark.parametrize('filename', ['publisher_events.jsonl', 'received_truth.jsonl'])
def test_stream_open_failure_has_a_receipt_before_ros_is_loaded(tmp_path, monkeypatch, filename):
    original_open = Path.open
    fake = FakeROS()

    def open_path(path, *args, **kwargs):
        if path.name == filename and args and args[0] == 'x':
            raise OSError('injected open failure')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_path)
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'CAPTURE_OPEN_FAILED'
    assert fake.contexts == [] and report['cleanup_complete'] is True


def test_publish_exception_keeps_zero_successful_call_count_and_a_receipt(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}}], failure='publish')
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'PUBLISHER_EXCEPTION'
    assert report['commands_sent'] == 0 and report['cleanup_complete'] is True


def test_preexisting_context_is_never_shut_down(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 12.}}])
    unrelated = fake.bindings.Context()
    unrelated.active = True
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 0 and unrelated.active is True
    assert len(fake.contexts) == 2 and fake.contexts[1].active is False
    assert fake.calls.count('context_shutdown') == 1 and report['cleanup_complete'] is True


def test_wall_limit_has_a_receipt_without_truth_or_publication(tmp_path, monkeypatch):
    fake = FakeROS()
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'WALL_LIMIT'
    assert report['truth_messages_received'] == 0 and fake.publications == []


@pytest.mark.parametrize('value', [float('nan'), -.1])
def test_clock_faults_have_a_finite_receipt(tmp_path, monkeypatch, value):
    fake = FakeROS([{'wall': .1, 'truth': {'sim_time': 3.}}, {'wall': value}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and report['terminal_reason'] == 'PUBLISHER_EXCEPTION'
    assert fake.publications == [(.1, .2)]


def test_interrupt_is_recorded_and_owned_resources_are_cleaned(tmp_path, monkeypatch):
    fake = FakeROS([{'wall': .1, 'exception': KeyboardInterrupt()}])
    code, report, _ = invoke(tmp_path, monkeypatch, fake)
    assert code == 1 and 'KeyboardInterrupt' in report['error']
    assert report['cleanup_complete'] is True and fake.publications == []


def test_wrong_domain_fails_before_loading_ros_with_receipt(tmp_path, monkeypatch):
    entry = load_entry()
    output = tmp_path / 'wrong_domain'
    monkeypatch.setenv('ROS_DOMAIN_ID', '1')
    def forbidden():
        raise AssertionError('must not import ROS')

    assert entry.main(['--output', str(output), '--execute-live-publisher'], loader=forbidden) == 1
    assert json.loads((output / 'report.json').read_text())['terminal_reason'] == 'CONFIGURATION_FAILED'


def test_existing_output_is_not_overwritten_or_loaded(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'report.json'
    marker.write_text('keep')
    entry = load_entry()
    with pytest.raises(FileExistsError):
        entry.main(['--output', str(output), '--execute-live-publisher'], loader=lambda: pytest.fail('loaded ROS'))
    assert marker.read_text() == 'keep'


def test_help_does_not_import_ros_or_create_output():
    result = subprocess.run([sys.executable, '-B', str(ENTRY), '--help'], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0 and '--execute-live-publisher' in result.stdout


def test_real_cli_default_loader_imports_only_fake_ros_and_records_init_failure(tmp_path):
    stubs = tmp_path / 'stubs'
    sources = {
        'rclpy/__init__.py': 'def init(*, args, context):\n context.active=True\n raise RuntimeError("CLI fake init failure")\n',
        'rclpy/context.py': 'class Context:\n active=False\n def ok(self): return self.active\n def try_shutdown(self): self.active=False\n',
        'rclpy/node.py': 'class Node: pass\n',
        'rclpy/executors.py': 'class SingleThreadedExecutor: pass\n',
        'geometry_msgs/__init__.py': '', 'geometry_msgs/msg.py': 'class Twist: pass\n',
        'std_msgs/__init__.py': '', 'std_msgs/msg.py': 'class String: pass\n',
    }
    for name, text in sources.items():
        target = stubs / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    output = tmp_path / 'cli_receipt'
    env = dict(os.environ, ROS_DOMAIN_ID='84', PYTHONPATH=str(stubs), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-B', str(ENTRY), '--output', str(output), '--execute-live-publisher'],
                            env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 1
    receipt = json.loads((output / 'report.json').read_text())
    assert receipt['terminal_reason'] == 'ROS_INIT_FAILED' and 'CLI fake init failure' in receipt['error']
    assert receipt['commands_sent'] == 0 and receipt['cleanup_complete'] is True
