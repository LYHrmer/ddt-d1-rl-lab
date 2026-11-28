"""Publish and independently score the paired speed and conditional parking trials."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools109.run109 import GUARD_SHA
from tools111.control import SPEED_GAIN
from tools111.results import install
from tools112.surface_results import install_json
from tools113.results import compare_tree
from tools114.results import resource_summary
from tools115.archive import _status
from tools116.archive import LIMIT, RECEIPTS
from tools116.response_replay import INDEX as SOURCE_INDEX, INDEX_SHA
from tools117.run import (BASE, PUBLIC, SEEDS, SPEED_STAGES, PARKING_STAGES, STAGES, TIMEOUT,
                          locations, protocol)


MOVIE = ROOT / 'docs/assets/compound117.mp4'


def split_tag(tag):
    stage, _, number = tag.rpartition('_')
    require(stage in STAGES and re.fullmatch(r'[0-9]{2}', number) and int(number) > 0, 'unknown attempt ' + tag)
    return stage, int(number)


def folder(tag, status):
    return PUBLIC / split_tag(tag)[0] if status == 'COMPLETE' else PUBLIC / 'attempts' / tag


def collect():
    """Install once. Interrupted binary outputs stay in their original directory with public digests."""
    tags = set()
    for path in BASE.iterdir():
        if path.is_dir():
            tags.add(path.name.removeprefix('guard_'))
        elif path.name.startswith('admission_') and path.suffix == '.json':
            tags.add(path.stem.removeprefix('admission_'))
    entries = {}
    for tag in sorted(tags):
        stage, _ = split_tag(tag)
        worker, guard = BASE/tag, BASE/('guard_' + tag)
        paths = {p.name: p for p in worker.glob('*') if p.is_file()}
        for path, key in ((guard/'report.json', 'guard.json'), (guard/'resources.jsonl', 'resources.jsonl'),
                          (BASE/('admission_' + tag + '.json'), 'admission.json')):
            if path.is_file():
                require(key not in paths, 'overlapping worker and supervisor output')
                paths[key] = path
        status = _status(read(paths['guard.json']) if 'guard.json' in paths else None,
                         'reservation.json' in paths, 'admission.json' in paths)
        files, partial = {}, {}
        if status != 'COMPLETE':
            for path in guard.glob('*.log'):
                paths['guard_' + path.name] = path
        for key, path in sorted(paths.items()):
            sha = digest(path)
            if status != 'COMPLETE' and path.suffix not in ('.json', '.jsonl', '.log'):
                partial[key] = sha
            else:
                target = MOVIE if status == 'COMPLETE' and stage == 'video' and key == MOVIE.name else folder(tag,status)/key
                install(path, target, sha)
                files[key] = sha
        entries[tag] = dict(status=status, files=files, local_partial_sha256=partial)
    return entries


def check_attempt(tag, entry, registration):
    stage, number = split_tag(tag)
    unit = locations(stage, number)[2]
    files, partial, status = entry['files'], entry['local_partial_sha256'], entry['status']
    directory = folder(tag, status)
    external = {MOVIE.name} if stage == 'video' and status == 'COMPLETE' else set()
    local_names = {p.name for p in directory.glob('*') if p.is_file()}
    require(not local_names & external and local_names | external == set(files), 'unlisted attempt output')
    require(not set(files) & set(partial), 'ambiguous public/partial output')
    for name, sha in (files | partial).items():
        require(Path(name).name == name and re.fullmatch('[0-9a-f]{64}', sha), 'invalid artifact name or digest')
        if name in files:
            require(digest(MOVIE if name in external else directory/name) == sha, 'changed artifact ' + name)
    header = read(directory/'reservation.json') if 'reservation.json' in files else None
    guard = read(directory/'guard.json') if 'guard.json' in files else None
    admission = read(directory/'admission.json') if 'admission.json' in files else None
    require(status == _status(guard, header is not None, admission is not None), 'inconsistent attempt status')
    expected = dict(stage=stage, attempt=number, unit=unit, registration_sha256=registration)
    start = end = resources = error = None
    if header is not None:
        require(expected.items() <= header.items() and header['reserved_at_unix_ns'] > 0, 'unbound reservation')
        start = header['reserved_at_unix_ns']
    if admission is not None:
        require(guard is None and header is None and expected.items() <= admission.items()
                and admission['child_started'] is False, 'unbound pre-admission')
        error = admission.get('error') or admission.get('reason')
    if guard is not None:
        require(guard['supervisor_sha256'] == GUARD_SHA and guard['schema'] == 'pressure-supervisor83-v1'
                and guard['timeout_s'] == TIMEOUT and guard['new_physics_steps_in_supervisor'] == 0
                and guard['scope'].endswith('/app.slice/' + unit + '.service') and guard['command'][1:] ==
                ['-B', '-m', 'tools117.run', 'worker', stage, '--attempt', str(number)], 'different guarded worker')
        error = guard.get('error')
        if status == 'COMPLETE':
            resources = resource_summary(directory, guard)
            samples = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
            require(all(s['self_scope'] == s['owned']['path'] == guard['scope'] for s in samples)
                    and min(s['effective_available_bytes'] for s in samples) == resources['minimum_effective_bytes'],
                    'resource samples do not belong to this worker')
    if 'report.json' in files:
        report = read(directory/'report.json')
        require(header is not None and all(report.get(k) == v for k,v in header.items()) and
                all(files.get(k, partial.get(k)) == v for k,v in report['files'].items()), 'unbound worker outputs')
    if status == 'COMPLETE':
        require(RECEIPTS <= set(files) and report['status'] == 'COMPLETE' and not partial,
                'missing complete worker or guardian receipts')
        outputs = set(files) - RECEIPTS
        require(set(report['files']) == outputs and ((stage != 'video' and outputs == {'trace.npz'}) or
                (stage == 'video' and MOVIE.name in outputs and all(k == MOVIE.name or
                 k.endswith(('.json','.jsonl')) and 'partial' not in k for k in outputs))), 'wrong complete outputs')
        end = report['completed_at_unix_ns']
        require(end >= start and math.isfinite(report['elapsed_s']) and report['elapsed_s'] >= 0, 'invalid worker clock')
    elif 'worker_partial.json' in files:
        receipt = read(directory/'worker_partial.json')
        require(header is not None and all(receipt.get(k) == v for k,v in header.items())
                and receipt['status'] == 'PARTIAL', 'unbound partial worker')
        error = receipt.get('error') or error
    return dict(stage=stage, attempt=number, status=status, reserved_at_unix_ns=start, completed_at_unix_ns=end,
                resources=resources, error=error)


def select(attempts):
    stages = {}
    for stage in STAGES:
        tags = sorted(k for k,r in attempts.items() if r['stage'] == stage)
        require([attempts[k]['attempt'] for k in tags] == list(range(1,len(tags)+1)), 'missing prior attempt')
        started = [k for k in tags if attempts[k]['status'] != 'PRE_ADMISSION_FAILURE']
        require(not started or started == tags[-1:], 'started stage repeated: ' + stage)
        tag = started[0] if started else None
        stages[stage] = dict(status=attempts[tag]['status'] if tag else 'NOT_STARTED', attempt=tag,
                             admission_failures=[k for k in tags if k != tag])
    clocks = sorted((r['reserved_at_unix_ns'], r['completed_at_unix_ns']) for r in attempts.values()
                    if r['completed_at_unix_ns'] is not None)
    require(all(left[1] <= right[0] for left,right in zip(clocks,clocks[1:])), 'overlapping workers')
    return stages


def native(stage, record):
    from tools117.score import score_trace
    family, digits = stage.split('_')
    seed, parking = int(digits), family == 'parking'
    status = record['status']
    if status != 'COMPLETE':
        counts = None
        if record['attempt']:
            directory = folder(record['attempt'], status)
            if (directory/'partial_counts.json').is_file():
                counts = read(directory/'partial_counts.json')['counts']
            elif (directory/'report.json').is_file():
                counts = read(directory/'report.json')['result']['counts']
        return dict(seed=seed, parking=parking, stage_status=status, counts=counts,
                    complete_horizon=False, driving_pass=False, gate_pass=False)
    directory = PUBLIC/stage
    recorded = read(directory/'report.json')['result']
    trace = directory/'trace.npz'
    reference = PUBLIC/f'speed_{seed}'/'trace.npz' if parking else SOURCE_INDEX.parent/f'native_{seed}'/'trace.npz'
    measured = score_trace(trace, reference, seed, parking=parking)
    require(measured['reference_sha256'] == digest(reference) and measured['trace_sha256'] == digest(trace)
            and measured['seed'] == seed and recorded['parking'] is parking and recorded['status'] == 'COMPLETE'
            and recorded['layout'] == 'development' and recorded['arm'] == 'both' and recorded['gain'] == SPEED_GAIN,
            'wrong paired course, intervention or trace')
    for key in ('seed','counts','trace_sha256','reference_sha256','motion','lidar','stop','first_failed_section',
                'driving_pass','gate_pass'):
        compare_tree(measured[key], recorded[key])
    model = read(ROOT/'reports/speed110/results.json')['preflight']
    require(recorded['actor_sha256'] == model['model_sha256'] and recorded['actuator_config_sha256'] == model['config_sha256']
            and recorded['mujoco_version'] == '3.3.0', 'different policy, actuators or simulator')
    with np.load(trace, allow_pickle=False) as z:
        compare_tree(float(z['nominal_base_height_m']), recorded['nominal_base_height_m'])
    return dict(measured, stage_status=status, parking=parking)


def gates(cases, attempts, stages):
    """Use only the three independently rescored speed cases to unlock wheel projection."""
    require([r['seed'] for r in cases] == list(SEEDS), 'wrong development roster')
    ready = all(r['stage_status'] == 'COMPLETE' and r['complete_horizon'] and r['driving_pass'] for r in cases)
    parking = ready and any(not r['gate_pass'] for r in cases)
    for stage in (*PARKING_STAGES, 'video'):
        scheduled = bool(stages[stage]['attempt'] or stages[stage]['admission_failures'])
        dependencies = (SPEED_STAGES[0],) if stage == 'video' else SPEED_STAGES
        valid = stages[SPEED_STAGES[0]]['status'] == 'COMPLETE' if stage == 'video' else parking
        require(not scheduled or valid, 'stage dispatched before its gate: ' + stage)
        if valid and stages[stage]['attempt']:
            latest = max(attempts[stages[k]['attempt']]['completed_at_unix_ns'] for k in dependencies)
            begin = attempts[stages[stage]['attempt']]['reserved_at_unix_ns']
            require(begin is None or begin >= latest, 'stage started before prerequisites ended: ' + stage)
    return parking


def pending_reason(stage, record, speed, opened):
    if record['admission_failures']:
        return 'no worker admitted'
    if stage in PARKING_STAGES and not opened:
        if not all(c['stage_status'] == 'COMPLETE' and c['complete_horizon'] for c in speed):
            return 'three full speed trials unavailable'
        if not all(c['driving_pass'] for c in speed):
            return 'speed, perception or safety gate failed'
        return 'all three speed trials already stopped'
    if stage == 'video' and speed[0]['stage_status'] != 'COMPLETE':
        return 'preselected source unavailable'
    return 'eligible, no recorded attempt'


def totals(cases):
    used = dict.fromkeys(LIMIT, 0)
    unknown = []
    complete = [c for c in cases if c['stage_status'] == 'COMPLETE']
    for case in cases:
        counts = case['counts']
        if counts is None:
            if case['stage_status'] not in ('NOT_STARTED','INTERRUPTED_BEFORE_RESERVATION'):
                unknown.append(case['seed'])
            continue
        require(counts.keys() == LIMIT.keys() and all(type(v) is int and 0 <= v <= LIMIT[k] for k,v in counts.items()),
                'case exceeds its registered budget')
        for k,v in counts.items():
            used[k] += v
    return dict(complete_stages=len(complete), full_horizons=sum(c['complete_horizon'] for c in complete),
                driving_passes=sum(c['driving_pass'] for c in complete), joint_passes=sum(c['gate_pass'] for c in complete),
                confirmed_counts=used, unknown_partial_seeds=unknown,
                counts_upper_bound={k:v+len(unknown)*LIMIT[k] for k,v in used.items()})


def film(stage, case):
    if stage['status'] != 'COMPLETE':
        require(not MOVIE.exists(), 'public video without complete worker')
        return None
    row = read(PUBLIC/'video/report.json')['result']
    frames, fps, stream = row['frames'], row['fps'], row['stream']
    require(case['stage_status'] == 'COMPLETE' and case['seed'] == row['seed'] == SEEDS[0]
            and row['status'] == 'COMPLETE' and row['filename'] == MOVIE.name
            and row['native_report_sha256'] == digest(PUBLIC/SPEED_STAGES[0]/'report.json')
            and row['report_sha256'] == [digest(SOURCE_INDEX.parent / f"native_{case['seed']}" / 'report.json'),
                                         row['native_report_sha256']]
            and row['trace_sha256'] == [case['reference_sha256'],case['trace_sha256']]
            and row['renderer_sha256'] == digest(ROOT/'tools117/video.py') and 'llvmpipe' in row['driver']['renderer'].lower()
            and row['sha256'] == digest(MOVIE) and row['cover_frames'] == row['physics'] == 0
            and 0 < row['bytes'] == MOVIE.stat().st_size <= 5_000_000
            and row['last_ticks'] == [case['reference_motion']['ticks'], case['motion']['ticks']],
            'wrong first-pair video source')
    for label, motion in zip(row['measurements'], (case['reference_motion'],case['motion']), strict=True):
        times = motion['gate_times_s']
        compare_tree(times[2] if len(times) >= 3 else None, label['entry_s'])
        compare_tree(times[3] if len(times) == 4 else None, label['exit_s'])
        compare_tree(motion['sections_m_s'][-1] if len(times) == 4 else None, label['section_m_s'])
    require(frames > 0 and fps == 25 and int(stream['nb_read_frames']) == frames
            and (stream['width'],stream['height'],stream['codec_name'],stream['pix_fmt'],stream['r_frame_rate'])
            == (1280,720,'h264','yuv420p','25/1') and float(stream['duration']) == row['duration_s']
            and abs(row['duration_s'] - frames/fps) <= .001, 'invalid encoded movie')
    clock, previous = 0, -1
    for shot in row['shots']:
        require(previous < shot['tick'] <= max(row['last_ticks']) and shot['first_frame'] == clock
                and shot['frames'] > 0, 'out-of-order replay frames')
        previous, clock = shot['tick'], clock+shot['frames']
    require(clock == frames and previous == max(row['last_ticks']), 'missing replay frames')
    return dict(file=str(MOVIE.relative_to(ROOT)), **{k:row[k] for k in ('seed','desktop_filename','trace_sha256',
                'native_report_sha256','last_ticks','frames','fps','duration_s','bytes','sha256')})


def index(entries):
    require(digest(SOURCE_INDEX) == INDEX_SHA, 'changed original round-116 index')
    registrations = {s:protocol(s) for s in STAGES}
    attempts = {tag:check_attempt(tag, entry, registrations[split_tag(tag)[0]]) for tag,entry in entries.items()}
    stages = select(attempts)
    expected = {tag for tag,entry in entries.items() if entry['status'] != 'COMPLETE' and entry['files']}
    require({p.name for p in (PUBLIC/'attempts').glob('*') if p.is_dir()} == expected
            and {p.name for p in PUBLIC.glob('*') if p.is_dir() and p.name != 'attempts'} ==
            {s for s,r in stages.items() if r['status'] == 'COMPLETE'}, 'unlisted public attempt or stage')
    speed = [native(stage, stages[stage]) for stage in SPEED_STAGES]
    source_cases = {r['seed']:r for r in read(SOURCE_INDEX)['cases']}
    for row in speed:
        if row['stage_status'] == 'COMPLETE':
            source = source_cases[row['seed']]
            require(source['stage_status'] == 'COMPLETE' and source['complete_horizon']
                    and source['trace_sha256'] == row['reference_sha256'], 'wrong matching speed reference')
    opened = gates(speed, attempts, stages)
    for stage, record in stages.items():
        if record['status'] == 'NOT_STARTED':
            record['reason'] = pending_reason(stage, record, speed, opened)
    parking = [native(stage, stages[stage]) for stage in PARKING_STAGES]
    video = film(stages['video'],speed[0])
    a, b = totals(speed), totals(parking)
    summary = dict(speed=a, parking=b, parking_gate_open=opened,
                   confirmed_counts={k:a['confirmed_counts'][k]+b['confirmed_counts'][k] for k in LIMIT},
                   counts_upper_bound={k:a['counts_upper_bound'][k]+b['counts_upper_bound'][k] for k in LIMIT},
                   admission_failures=[tag for tag,r in attempts.items() if r['status'] == 'PRE_ADMISSION_FAILURE'],
                   partial_stages=[s for s,r in stages.items() if r['status'] not in ('COMPLETE','NOT_STARTED')],
                   not_started=[s for s,r in stages.items() if r['status'] == 'NOT_STARTED'],
                   blind_episodes=0, world_updates=0, policy_updates=0)
    return dict(registrations=registrations, reference_index_sha256=INDEX_SHA, attempts=entries, attempt_results=attempts,
                stages=stages, speed_cases=speed, parking_cases=parking, video=video, summary=summary)


def publish():
    entries = collect()
    require(entries, 'no evidence to publish')
    install_json(PUBLIC/'results.json', index(entries))


def verify():
    saved = read(PUBLIC/'results.json')
    rebuilt = index(saved['attempts'])
    compare_tree(rebuilt, saved)
    return rebuilt['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    if parser.parse_args().publish:
        publish()
    if (PUBLIC/'results.json').is_file():
        print(json.dumps(verify(), indent=2, allow_nan=False))
    else:
        require(not PUBLIC.exists(), 'partial public archive without its index')
        print('Registered stages', {s:protocol(s) for s in STAGES})
