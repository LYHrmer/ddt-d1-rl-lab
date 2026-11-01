"""Exercise real catchable signals with an owned, model-free child process."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools74.runtime74 import run_preserving

ROOT = Path(__file__).resolve().parents[1]
SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU)
CHILD = r'''
import signal
import sys
from pathlib import Path
from tools74.runtime74 import run_preserving

output, ready = Path(sys.argv[1]), Path(sys.argv[2])
def body(report):
    report['points'].append({'index': 0, 'status': 'INCOMPLETE', 'readback_prefix': [3, 7]})
    report['training_labels'].append({'source': 'saved', 'status': 'COMPLETE'})
    ready.write_text('ready', encoding='utf-8')
    while True:
        signal.pause()

report = run_preserving(output, body)
raise SystemExit(0 if report['status'] == 'COMPLETE' else 1)
'''


@pytest.mark.parametrize('signum', (signal.SIGTERM, signal.SIGXCPU))
def test_real_signal_retains_unfinished_readback_and_final_failure(tmp_path, signum):
    output, ready = tmp_path / 'run', tmp_path / 'ready'
    process = subprocess.Popen([sys.executable, '-c', CHILD, str(output), str(ready)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f'child exited before readiness: {stdout}\n{stderr}')
            if time.monotonic() >= deadline:
                pytest.fail('child did not finish the first prefix within10 seconds')
            time.sleep(.01)
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 1, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    initial = json.loads((output / 'initial_incomplete.json').read_text())
    report = json.loads((output / 'report.json').read_text())
    assert initial['status'] == 'INCOMPLETE' and initial['points'] == initial['training_labels'] == []
    assert report['status'] == 'INCOMPLETE'
    assert 'SystemExit' in report['failure'] and signal.Signals(signum).name in report['failure']
    assert report['points'] == [{'index': 0, 'status': 'INCOMPLETE', 'readback_prefix': [3, 7]}]
    assert report['training_labels'] == [{'source': 'saved', 'status': 'COMPLETE'}]
    assert all(report[key] == 0 for key in ('model_calls', 'actor_calls', 'physics_steps', 'updates'))
    assert report['cpu_seconds'] >= 0 and report['wall_seconds'] >= 0


def test_success_receipt_postverification_and_signal_handlers_are_restored(tmp_path):
    before = {signum: signal.getsignal(signum) for signum in SIGNALS}
    called = []

    def body(report):
        report['points'].append({'status': 'COMPLETE', 'value': 4})

    def verify_after():
        called.append('verified')

    output = tmp_path / 'success'
    report = run_preserving(output, body, verify_after)
    assert report['status'] == report['verification_after'] == 'COMPLETE'
    assert report['failure'] is None and called == ['verified']
    assert json.loads((output / 'report.json').read_text()) == report
    assert {signum: signal.getsignal(signum) for signum in SIGNALS} == before


@pytest.mark.parametrize('body_fails', (False, True))
def test_verification_failure_revokes_success_without_hiding_body_failure(tmp_path, body_fails):
    before = {signum: signal.getsignal(signum) for signum in SIGNALS}

    def body(report):
        report['points'].append({'status': 'INCOMPLETE' if body_fails else 'COMPLETE'})
        if body_fails:
            raise RuntimeError('original readback failure')

    def verify_after():
        raise ValueError('late identity mismatch')

    output = tmp_path / 'failed'
    report = run_preserving(output, body, verify_after)
    assert report['status'] == report['verification_after'] == 'INCOMPLETE'
    assert report['verification_failure'] == 'ValueError: late identity mismatch'
    assert report['failure'] == ('RuntimeError: original readback failure' if body_fails else None)
    assert len(report['points']) == 1
    assert json.loads((output / 'report.json').read_text()) == report
    assert {signum: signal.getsignal(signum) for signum in SIGNALS} == before


def test_existing_run_cannot_be_overwritten_or_restarted(tmp_path):
    output = tmp_path / 'once'
    run_preserving(output, lambda report: report['points'].append({'value': 1}))
    original = (output / 'report.json').read_bytes()
    called = []
    with pytest.raises(FileExistsError):
        run_preserving(output, lambda report: called.append('rerun'))
    assert called == [] and (output / 'report.json').read_bytes() == original
