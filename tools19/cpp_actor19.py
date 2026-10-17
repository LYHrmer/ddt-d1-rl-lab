"""Bounded JSONL transport for the independent C++ CPU actor; cold at import."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runtime19/actor_server19.cpp'
ORT_LIBRARY = Path('/usr/lib/libonnxruntime.so').resolve()
SIGNATURE = {'inputs': {'obs': [1, 57], 'history': [1, 10, 57]}, 'outputs': {'actions': [1, 16]}}
HEADERS = tuple(Path('/usr/include')/name for name in
                ('onnxruntime_cxx_api.h', 'onnxruntime_cxx_inline.h', 'onnxruntime_c_api.h', 'nlohmann/json.hpp'))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def lifecycle():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools16 import run_comparison16
    return run_comparison16


def cleanup_owned(process):
    return lifecycle().cleanup(process, {'interrupt_grace_s': 1., 'terminate_grace_s': .5,
                                        'kill_grace_s': .5}, natural_exit=process.poll() is not None)


def build_server(output, *, compiler='g++', runner=subprocess.run, clock=time.monotonic):
    """Compile only. No server invocation, ORT session, robot, or physics."""
    output = Path(output).absolute()
    require(not output.exists(), 'build output must be new')
    compiler_path = Path(shutil.which(compiler) or compiler).resolve(strict=True)
    inputs = [SOURCE, Path(__file__).resolve(), ORT_LIBRARY, compiler_path, *HEADERS]
    identities = {str(path): sha(path) for path in inputs}
    output.mkdir(parents=True, exist_ok=False)
    binary = output/'actor_server19'
    command = [str(compiler_path), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Wpedantic', '-ffp-contract=off',
               '-DD1_ACTOR19_SOURCE_SHA256="'+identities[str(SOURCE)]+'"', str(SOURCE),
               '-I/usr/include', '-L/usr/lib', '-Wl,-rpath,/usr/lib', '-lonnxruntime', '-lcrypto', '-ldl',
               '-o', str(binary)]
    report = {'format': 'd1-cpp-actor19-build-v1', 'status': 'FAIL', 'command': command,
              'inputs_sha256': identities, 'binary_sha256': None, 'error': None,
              'server_started': False, 'onnx_sessions_created': 0, 'robot_models_constructed': 0,
              'physics_steps_executed': 0}
    started = clock()
    try:
        with (output/'compiler.stdout.log').open('xb') as stdout, (output/'compiler.stderr.log').open('xb') as stderr:
            completed = runner(command, stdout=stdout, stderr=stderr, timeout=120, check=False)
        require(completed.returncode == 0 and binary.is_file(), 'C++ compilation failed')
        require(all(sha(path) == digest for path, digest in identities.items()), 'build input changed during compilation')
        report.update(status='PASS', binary_sha256=sha(binary))
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['elapsed_s'] = clock()-started
        report['child_max_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        report['rss_scope'] = 'maximum of completed children in this Python process; fresh build CLI isolates compiler'
        atomic_json(output/'build.json', report)
    return report


class PipeTransport:
    """No reader threads: bounded nonblocking writes and newline reads."""
    def __init__(self, process, clock=time.monotonic):
        self.process, self.clock, self.buffer = process, clock, b''
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)

    def write(self, data, timeout):
        deadline, offset = self.clock()+timeout, 0
        with selectors.DefaultSelector() as ready:
            ready.register(self.process.stdin, selectors.EVENT_WRITE)
            while offset < len(data):
                remaining = deadline-self.clock()
                if remaining <= 0 or not ready.select(remaining):
                    raise TimeoutError('C++ request write timed out')
                try:
                    offset += os.write(self.process.stdin.fileno(), data[offset:])
                except BlockingIOError:
                    pass

    def read(self, timeout):
        deadline = self.clock()+timeout
        with selectors.DefaultSelector() as ready:
            ready.register(self.process.stdout, selectors.EVENT_READ)
            while b'\n' not in self.buffer:
                remaining = deadline-self.clock()
                if remaining <= 0 or not ready.select(remaining):
                    raise TimeoutError('C++ response timed out')
                try:
                    block = os.read(self.process.stdout.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not block:
                    raise EOFError('C++ exited before a complete protocol line')
                self.buffer += block
                require(len(self.buffer) <= 65536, 'C++ protocol line is too large')
        value, self.buffer = self.buffer.split(b'\n', 1)
        return value

    def close_input(self):
        self.process.stdin.close()


def parse_line(line):
    def constant(value):
        raise ValueError('nonfinite protocol constant: '+value)
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate protocol key')
            value[key] = item
        return value
    value = json.loads(line, parse_constant=constant, object_pairs_hook=pairs)
    require(isinstance(value, dict), 'C++ protocol must be one JSON object')
    return value


class CppActor19:
    def __init__(self, binary, model, *, output, expected_binary_sha256, expected_model_sha256,
                 expected_source_sha256, max_requests=21600, timeout_s=2., startup_timeout_s=30.,
                 process_factory=subprocess.Popen, transport_factory=PipeTransport,
                 cleanup_function=cleanup_owned, source=SOURCE, ort_library=ORT_LIBRARY):
        require(type(max_requests) is int and 1 <= max_requests <= 21664, 'invalid bounded request cap')
        require(type(timeout_s) in (float, int) and 0 < timeout_s <= 10
                and type(startup_timeout_s) in (float, int) and 0 < startup_timeout_s <= 60,
                'invalid request or startup timeout')
        self.binary, self.model, self.source = (Path(path).resolve() for path in (binary, model, source))
        self.output = Path(output).absolute()
        self.max_requests, self.timeout, self.startup_timeout = max_requests, timeout_s, startup_timeout_s
        self.process_factory, self.transport_factory, self.cleanup_function = process_factory, transport_factory, cleanup_function
        self.identities = {'binary_sha256': expected_binary_sha256, 'model_sha256': expected_model_sha256,
                           'source_sha256': expected_source_sha256, 'ort_library_sha256': sha(ort_library),
                           'adapter_sha256': sha(Path(__file__))}
        self.paths = {'binary_sha256': self.binary, 'model_sha256': self.model, 'source_sha256': self.source,
                      'ort_library_sha256': Path(ort_library).resolve(), 'adapter_sha256': Path(__file__).resolve()}
        require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for value in self.identities.values()), 'explicit SHA256 identities required')
        self._unchanged()
        self.process = self.transport = self.stderr = None
        self.started = self.closed = False
        self._stats = {'format': 'd1-cpp-actor19-adapter-v1', 'status': 'NOT_STARTED',
                       'identities': dict(self.identities), 'process_started': False,
                       'requests_attempted': 0, 'requests_completed': 0,
                       'inference_attempts': None, 'inference_returns': None,
                       'cleanup_complete': True, 'cleanup_signals': [], 'error': None,
                       'ready': None, 'eof_receipt': None}

    @property
    def stats(self):
        return copy.deepcopy(self._stats)

    def _unchanged(self):
        require(all(path.is_file() and sha(path) == self.identities[key] for key, path in self.paths.items()),
                'actor source, binary, model, adapter or ORT library changed')

    def _receive(self, timeout):
        raw = self.transport.read(timeout)
        require(isinstance(raw, bytes) and len(raw) <= 65536, 'invalid C++ protocol transport')
        with (self.output/'stdout.jsonl').open('ab') as stream:
            stream.write(raw+b'\n')
        return parse_line(raw)

    def start(self):
        require(not self.started and not self.closed, 'actor cannot be restarted')
        self._unchanged()
        require(not self.output.exists(), 'actor output must be new; no overwrite')
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = True
        command = [str(self.binary), '--policy', str(self.model), '--expected-model-sha256', self.identities['model_sha256'],
                   '--expected-source-sha256', self.identities['source_sha256'], '--max-requests', str(self.max_requests)]
        self._stats['command'] = command
        self._stats['status'] = 'STARTING'
        atomic_json(self.output/'report.json', self._stats)
        with lifecycle().termination_guard() as control:
            try:
                self.stderr = (self.output/'stderr.log').open('xb')
                control['defer'] = True
                try:
                    self.process = self.process_factory(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=self.stderr, bufsize=0, start_new_session=True,
                        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    self._stats['process_started'] = True
                finally:
                    control['defer'] = False
                require(control['received'] is None, 'termination during actor startup')
                self.transport = self.transport_factory(self.process)
                ready = self._receive(self.startup_timeout)
                require(ready.get('format') == 'd1-cpp-actor19-ready-v1' and ready.get('status') == 'READY'
                        and all(ready.get(key) == self.identities[key]
                                for key in ('binary_sha256', 'model_sha256', 'source_sha256', 'ort_library_sha256'))
                        and ready.get('signature') == SIGNATURE and ready.get('onnxruntime_version') == '1.10.0'
                        and ready.get('cpu_threads') == 1 and ready.get('max_requests') == self.max_requests,
                        'C++ startup identity, signature or CPU settings differ')
                require(all(type(ready.get(key)) is int and ready[key] == 0
                            for key in ('inference_attempts', 'inference_returns', 'robot_models_constructed', 'physics_steps_executed')),
                        'C++ startup counters differ')
                self._stats.update(status='READY', ready=ready, inference_attempts=0, inference_returns=0)
                atomic_json(self.output/'report.json', self._stats)
            except BaseException as exc:
                self._failed(exc)
                raise
        return self

    def predict(self, observation):
        require(self.started and not self.closed and self._stats['status'] == 'READY', 'actor is not ready')
        try:
            import numpy as np
            require(isinstance(observation, np.ndarray) and observation.dtype == np.float32
                    and observation.shape == (627,) and np.isfinite(observation).all(),
                    'actor input must be finite float32 shape (627,)')
            seq = self._stats['requests_attempted']
            require(seq < self.max_requests, 'actor request cap exhausted')
            payload = {'format': 'd1-cpp-actor19-request-v1', 'seq': seq, 'actor': observation.tolist()}
            self._stats['requests_attempted'] += 1
            self._stats.update(inference_attempts=None, inference_returns=None)
            started = time.monotonic()
            self.transport.write(json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()+b'\n', self.timeout)
            remaining = self.timeout-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('C++ per-request deadline exhausted during write')
            response = self._receive(remaining)
            if response.get('format') == 'd1-cpp-actor19-failure-v1' and response.get('status') == 'FAIL':
                attempted, returned = response.get('inference_attempts'), response.get('inference_returns')
                require(type(attempted) is int and type(returned) is int and 0 <= returned <= attempted <= seq+1,
                        'invalid C++ failure counters')
                self._stats.update(inference_attempts=attempted, inference_returns=returned, failure_receipt=response)
                raise ValueError('C++ inference failed: '+str(response.get('error')))
            require(set(response) == {'format', 'status', 'seq', 'actions', 'inference_attempts', 'inference_returns'}
                    and response['format'] == 'd1-cpp-actor19-response-v1' and response['status'] == 'OK'
                    and type(response['seq']) is int and response['seq'] == seq
                    and all(type(response[key]) is int and response[key] == seq+1
                            for key in ('inference_attempts', 'inference_returns')), 'C++ response sequence or counters differ')
            self._stats.update(inference_attempts=response['inference_attempts'], inference_returns=response['inference_returns'])
            values = response['actions']
            require(isinstance(values, list) and len(values) == 16
                    and all(type(value) in (int, float) and math.isfinite(value) and -1 <= value <= 1 for value in values),
                    'C++ actions must be finite and clipped shape16')
            action = np.asarray(values, dtype=np.float32)
            self._stats.update(requests_completed=seq+1, inference_attempts=response['inference_attempts'],
                               inference_returns=response['inference_returns'])
            return action
        except BaseException as exc:
            self._failed(exc)
            raise

    def _cleanup(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            if self.process is not None:
                try:
                    signals, complete = self.cleanup_function(self.process)
                    self._stats.update(cleanup_signals=signals, cleanup_complete=complete,
                                       returncode=self.process.returncode)
                except BaseException as exc:
                    self._stats.update(cleanup_complete=False, error=(self._stats['error'] or '')+f'; cleanup {exc}')
            for stream in (getattr(self.process, 'stdin', None), getattr(self.process, 'stdout', None), self.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except BaseException as exc:
                    self._stats.update(cleanup_complete=False, error=(self._stats['error'] or '')+f'; stream close {exc}')
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _failed(self, exc):
        self.closed = True
        self._stats.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        self._cleanup()
        atomic_json(self.output/'report.json', self._stats)

    def close(self):
        if self.closed or not self.started:
            return self.stats
        try:
            self.transport.close_input()
            ended = self._receive(self.timeout)
            count = self._stats['requests_completed']
            require(ended.get('format') == 'd1-cpp-actor19-closed-v1' and ended.get('status') == 'CLOSED'
                    and all(type(ended.get(key)) is int and ended[key] == count
                            for key in ('requests', 'inference_attempts', 'inference_returns'))
                    and all(type(ended.get(key)) is int and ended[key] == 0
                            for key in ('robot_models_constructed', 'physics_steps_executed'))
                    and self._stats['requests_attempted'] == count, 'C++ EOF accounting differs')
            self.process.wait(timeout=self.timeout)
            require(self.process.returncode == 0, 'C++ exit code is not zero')
            self._unchanged()
            self._stats.update(status='CLOSED', eof_receipt=ended)
        except BaseException as exc:
            self._failed(exc)
            return self.stats
        self.closed = True
        self._cleanup()
        if not self._stats['cleanup_complete']:
            self._stats['status'] = 'FAIL'
        atomic_json(self.output/'report.json', self._stats)
        return self.stats

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, trace):
        if value is not None and not self.closed:
            self._failed(value)
        else:
            self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if not args.build:
        print(json.dumps({'status': 'PLAN_ONLY', 'server_started': False, 'onnx_sessions_created': 0,
                          'source': str(SOURCE), 'source_sha256': sha(SOURCE)}))
        return 0
    if args.output is None:
        parser.error('--build requires an exclusive new --output')
    report = build_server(args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
