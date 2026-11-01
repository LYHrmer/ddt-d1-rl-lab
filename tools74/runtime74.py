"""Preserve a readback prefix on exceptions and catchable termination signals.

The callback mutates a JSON-serializable report in place. Each output directory
and receipt is created exclusively: failed runs are neither overwritten nor
retried. The caller decides its process exit code from the returned status.
"""
from __future__ import annotations

import json
import signal
import time
from pathlib import Path


def _write_exclusive(path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(serialized)


def _signal_failure(signum, _frame):
    raise SystemExit(f'received {signal.Signals(signum).name} ({signum})')


def run_preserving(output: Path, body_callback, verify_after_callback=None):
    """Run once, retaining all appended points even if the callback is stopped.

    Invoke from the main thread so Python can install the signal handlers.
    ``verify_after_callback`` has no arguments and runs after success or body
    failure. Its exception never replaces the original body failure.
    """
    cpu_start, wall_start = time.process_time(), time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'INCOMPLETE', 'model_calls': 0, 'actor_calls': 0,
              'physics_steps': 0, 'updates': 0, 'points': [], 'training_labels': [], 'failure': None}
    _write_exclusive(output / 'initial_incomplete.json', report)
    previous_handlers = {}
    try:
        try:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU):
                previous_handlers[signum] = signal.signal(signum, _signal_failure)
            body_callback(report)
            report['status'] = 'COMPLETE'
        except BaseException as exc:  # noqa: BLE001 - termination must preserve the partial prefix.
            report['status'] = 'INCOMPLETE'
            report['failure'] = f'{type(exc).__name__}: {exc}'
        finally:
            if verify_after_callback is not None:
                try:
                    verify_after_callback()
                    report['verification_after'] = 'COMPLETE'
                except BaseException as exc:  # noqa: BLE001 - verification revokes successful body completion.
                    report['status'] = 'INCOMPLETE'
                    report['verification_after'] = 'INCOMPLETE'
                    report['verification_failure'] = f'{type(exc).__name__}: {exc}'
            report['cpu_seconds'] = time.process_time() - cpu_start
            report['wall_seconds'] = time.monotonic() - wall_start
            _write_exclusive(output / 'report.json', report)
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    return report
