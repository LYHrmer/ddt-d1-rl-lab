"""Run the registered lidar/encoder comparison in serial, memory-guarded user units."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools27.resource_guard27 import read_memory
from tools91.launch91 import PRESSURE, ROOT, THREADS, digest, read, require, write
from tools92.launch92 import scope_for
from tools109.run109 import GUARD_SHA
from tools111.remaining import wait_for_admission


BASE, PUBLIC = ROOT / 'evidence/compound116', ROOT / 'reports/compound116'
PLANS = ROOT / 'docs/experiments116'
SEEDS = (111101, 111103, 111107)
NATIVE_STAGES = tuple(f'native_{s}' for s in SEEDS)
STAGES = ('wiring', *NATIVE_STAGES, 'video')
TIMEOUT = 600


def protocol(stage):
    require(stage in STAGES, 'unregistered compound stage')
    family = stage.split('_')[0]
    anchor, implementation = PLANS / 'design.json', PLANS / f'{family}.json'
    design, code = read(anchor), read(implementation)
    require(design['seeds'] == list(SEEDS) and design['supervisor_sha256'] == GUARD_SHA
            and code['family'] == family and code['design_sha256'] == digest(anchor), 'changed registration')
    for item in (design, code):
        for name, sha in item['sources_sha256'].items():
            require(digest(ROOT / name) == sha, 'changed registered input: ' + name)
    return digest(implementation)


def locations(stage, attempt):
    require(stage in STAGES and 1 <= attempt <= 99, 'invalid compound run')
    tag = f'{stage}_{attempt:02d}'
    return BASE / tag, BASE / ('guard_' + tag), 'd1-compound116-' + tag.replace('_', '-')


def completed(stage):
    for path in sorted(BASE.glob(stage + '_??/report.json')):
        record = read(path)
        out, guard, unit = locations(stage, record['attempt'])
        if not (guard / 'report.json').is_file():
            continue
        receipt = read(guard / 'report.json')
        if receipt['status'] != 'COMPLETE':
            continue
        header = read(out / 'reservation.json')
        require(path.parent == out and record['status'] == 'COMPLETE' and record['stage'] == stage
                and record['unit'] == unit and record['registration_sha256'] == protocol(stage)
                and all(record[k] == v for k, v in header.items()) and receipt['child_started']
                and receipt['returncode'] == 0 and receipt['child_reaped'] and receipt['owned_cgroup_children_empty']
                and receipt['scope'] == scope_for(unit) and receipt['supervisor_sha256'] == GUARD_SHA,
                'unbound stage completion')
        for name, sha in record['files'].items():
            require(Path(name).name == name and digest(out / name) == sha, 'changed worker output')
        return out, record
    return None


def eligible(stage):
    if stage == 'wiring':
        return True
    if stage == 'video':
        return completed(NATIVE_STAGES[0]) is not None
    wiring = completed('wiring')
    if wiring is None:
        return False
    result = wiring[1]['result']
    return result['passed'] is True and result['ticks'] == 1800 and len(result['cases']) == len(SEEDS)


def worker(stage, attempt):
    out, _, unit = locations(stage, attempt)
    scope = scope_for(unit)
    require(Path('/proc/self/cgroup').read_text().strip() == '0::' + scope
            and os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1
            and all(os.environ.get(k) == '1' for k in THREADS) and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and read_memory(scope)['effective_available_bytes'] >= 4 << 30, 'worker admission mismatch')
    require(eligible(stage), 'source stage unavailable or wiring gate closed')
    header = dict(stage=stage, attempt=attempt, unit=unit, registration_sha256=protocol(stage),
                  reserved_at_unix_ns=time.time_ns())
    out.mkdir(exist_ok=False)
    write(out / 'reservation.json', header)
    start = time.monotonic()
    try:
        if stage == 'wiring':
            from tools116.wiring import audit
            result = audit()
        elif stage == 'video':
            from tools116.video import render
            result = render(out, completed(NATIVE_STAGES[0])[0])
        else:
            from tools116.native import episode
            result = episode(out, int(stage.split('_')[1]))
            require(result['trace_sha256'] == digest(out / 'trace.npz'), 'worker trajectory hash differs')
        files = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'reservation.json'}
        write(out / 'report.json', dict(**header, status='COMPLETE', result=result, files=files,
                                        completed_at_unix_ns=time.time_ns(), elapsed_s=time.monotonic()-start))
    except BaseException as exc:
        write(out / 'worker_partial.json', dict(**header, status='PARTIAL', error=repr(exc)))
        raise


def launch(stage):
    registered = protocol(stage)
    require(digest(PRESSURE) == GUARD_SHA, 'changed memory and pressure supervisor')
    BASE.mkdir(parents=True, exist_ok=True)
    settings = dict.fromkeys(THREADS, '1')
    settings.update(PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT))),
                    MUJOCO_GL='disable', CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    if stage == 'video':
        from tools102.mesa_video102 import GL_ENV
        settings.update(GL_ENV)
    with (ROOT / 'evidence/uphill89/serial.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if completed(stage):
            return True
        if not eligible(stage):
            print('Closed prerequisite:', stage, flush=True)
            return False
        for attempt in range(1, 100):
            out, guard, unit = locations(stage, attempt)
            admission = BASE / f'admission_{out.name}.json'
            if guard.exists():
                prior = read(guard/'report.json')
                if prior['child_started'] or out.exists():
                    return False
                continue
            require(not out.exists(), 'preserve existing stage: ' + out.name)
            if admission.exists():
                continue
            print('Waiting for headroom', unit, flush=True)
            try:
                wait_for_admission()
            except TimeoutError as exc:
                write(admission, dict(stage=stage, attempt=attempt, unit=unit, child_started=False,
                                      registration_sha256=registered, error=str(exc), time_ns=time.time_ns()))
                continue
            cmd = ['systemd-run', '--user', '--wait', '--collect', '--pipe', '--slice=app.slice', '--unit='+unit,
                   '-p', 'Nice=10', '-p', 'CPUWeight=25', '-p', 'OOMPolicy=kill', '-p', 'MemoryHigh=2G',
                   '-p', 'MemoryMax=3G', '-p', 'MemorySwapMax=0', f'--working-directory={ROOT}',
                   *(f'--setenv={k}={v}' for k,v in settings.items()), sys.executable, '-B', str(PRESSURE),
                   '--scope', scope_for(unit), '--output', str(guard), '--timeout', str(TIMEOUT), '--',
                   sys.executable, '-B', '-m', 'tools116.run', 'worker', stage, '--attempt', str(attempt)]
            print('Dispatch', unit, flush=True)
            code = subprocess.run(cmd).returncode
            receipt = read(guard/'report.json')
            if code == 0:
                require(completed(stage), 'missing successful completion')
                return True
            if receipt['child_started'] or out.exists():
                print('Retained started attempt', stage, receipt.get('error'), flush=True)
                return False
            require(any(t in str(receipt['error']) for t in ('effective memory', 'admission not sustained', 'PSI')),
                    'non-resource start failure')
    raise RuntimeError('no fresh attempt directory for ' + stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'worker'))
    parser.add_argument('stage', choices=(*STAGES, 'development'))
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.stage, args.attempt)
    else:
        stages = ('wiring', *NATIVE_STAGES) if args.stage == 'development' else (args.stage,)
        outcomes = [launch(s) for s in stages]
        raise SystemExit(0 if all(outcomes) else 2)
