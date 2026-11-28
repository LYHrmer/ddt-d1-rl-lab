"""Run the single registered fixed-action replay under the shared D1 resource guard."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools91.launch91 import PRESSURE, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import ready, wait_for_admission
from tools114.common import BASE, ROOT, registration


def locations(attempt):
    require(1 <= attempt <= 99, 'invalid admission attempt')
    tag = f'replay_{attempt:02d}'
    return BASE / tag, BASE / ('guard_' + tag), 'd1-parking114-' + tag.replace('_', '-')


def worker(attempt):
    from tools27.resource_guard27 import read_memory
    out, _, unit = locations(attempt)
    scope = scope_for(unit)
    require(Path('/proc/self/cgroup').read_text().strip() == '0::' + scope and
            os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1 and
            all(os.environ.get(k) == '1' for k in THREADS) and os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            read_memory(scope)['effective_available_bytes'] >= 4 << 30, 'worker admission mismatch')
    sha = registration()
    require(not any(BASE.glob('replay_??/reservation.json')), 'preserve the previously started replay')
    out.mkdir(exist_ok=False)
    header = dict(registration_sha256=sha, unit=unit, reserved_at_unix_ns=time.time_ns())
    write(out / 'reservation.json', header)
    try:
        from tools114.audit import audit_archives
        from tools114.replay import diagnose
        projection = audit_archives()
        replay = diagnose(out)
        write(out / 'report.json', dict(**header, status='COMPLETE', model_replay=replay,
                                        archived_projection=projection, completed_at_unix_ns=time.time_ns()))
    except BaseException as error:
        write(out / 'worker_partial.json', dict(**header, status='PARTIAL', error=repr(error)))
        raise


def launch():
    registration()
    require(digest(PRESSURE) == GUARD_SHA, 'changed resource supervisor')
    BASE.mkdir(exist_ok=True, parents=True)
    env = dict.fromkeys(THREADS, '1')
    env.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT / '.python-deps', ROOT / 'training', ROOT))),
               CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', MUJOCO_GL='disable')
    with (ROOT / 'evidence/uphill89/serial.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(1, 100):
            out, guard, unit = locations(attempt)
            if guard.exists() or out.exists():
                receipt = read(guard / 'report.json')
                if receipt['status'] == 'COMPLETE':
                    require(read(out / 'report.json')['status'] == 'COMPLETE', 'missing completed replay')
                    return
                require(not receipt['child_started'] and not out.exists(), 'preserve the started replay')
                continue
            if not ready():
                wait_for_admission()
            cmd = ['systemd-run', '--user', '--wait', '--collect', '--pipe', '--slice=app.slice', '--unit=' + unit,
                   '-p', 'Nice=10', '-p', 'CPUWeight=25', '-p', 'OOMPolicy=kill', '-p', 'MemoryHigh=2G',
                   '-p', 'MemoryMax=3G', '-p', 'MemorySwapMax=0', f'--working-directory={ROOT}',
                   *(f'--setenv={k}={v}' for k, v in env.items()), sys.executable, '-B', str(PRESSURE),
                   '--scope', scope_for(unit), '--output', str(guard), '--timeout', '300', '--',
                   sys.executable, '-B', '-m', 'tools114.run', 'worker', '--attempt', str(attempt)]
            print(f'Dispatch {unit}', flush=True)
            code = subprocess.run(cmd).returncode
            receipt = read(guard / 'report.json')
            if code == 0:
                require(receipt['status'] == 'COMPLETE' and receipt['child_reaped'] and
                        read(out / 'report.json')['status'] == 'COMPLETE', 'missing final resource or replay receipt')
                return
            reason = str(receipt['error'])
            if receipt['child_started'] or out.exists() or not any(
                    key in reason for key in ('effective memory', 'admission not sustained', 'PSI')):
                raise RuntimeError(unit + ': ' + reason)
            wait_for_admission()
    raise RuntimeError('no available admission slot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'worker'))
    parser.add_argument('--attempt', type=int, default=1)
    opts = parser.parse_args()
    if opts.mode == 'launch':
        launch()
    else:
        worker(opts.attempt)
