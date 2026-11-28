"""Publish and independently score the registered three-pair compound-course experiment."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from tools91.launch91 import ROOT, digest, read, require
from tools109.run109 import GUARD_SHA
from tools111.results import install
from tools112.surface_results import install_json
from tools113.results import compare_tree
from tools114.results import resource_summary
from tools115.archive import _status
from tools116.run import BASE, NATIVE_STAGES, PLANS, PUBLIC, SEEDS, STAGES, TIMEOUT, locations, protocol


MOVIE = ROOT / 'docs/assets/compound116.mp4'
RECEIPTS = {'reservation.json', 'report.json', 'guard.json', 'resources.jsonl'}
LIMIT = dict(decisions=600, servo=3000, physics=6000, packets=72, ray_queries=74880)
GATES = ('horizon', 'forward_speed', 'one_metre_speeds', 'lane', 'all_wheels_exited',
         'step_top_support', 'stopped', 'fall_free', 'legal_support')


def _key(tag):
    stage, _, digits = tag.rpartition('_')
    require(stage in STAGES and re.fullmatch('[0-9]{2}', digits) and int(digits) > 0, 'unknown attempt ' + tag)
    return stage, int(digits)


def _directory(tag, status):
    return PUBLIC / _key(tag)[0] if status == 'COMPLETE' else PUBLIC / 'attempts' / tag


def _collect():
    tags = set()
    for path in BASE.glob('*'):
        if path.is_dir():
            tags.add(path.name.removeprefix('guard_'))
        elif path.name.startswith('admission_') and path.suffix == '.json':
            tags.add(path.stem.removeprefix('admission_'))
    entries = {}
    for tag in sorted(tags):
        stage, _ = _key(tag)
        worker, guard = BASE / tag, BASE / ('guard_' + tag)
        paths = {p.name: p for p in worker.glob('*') if p.is_file()}
        for origin, name in ((guard/'report.json', 'guard.json'), (guard/'resources.jsonl', 'resources.jsonl'),
                             (BASE/('admission_' + tag + '.json'), 'admission.json')):
            if origin.is_file():
                require(name not in paths, 'overlapping worker and guard evidence')
                paths[name] = origin
        status = _status(read(paths['guard.json']) if 'guard.json' in paths else None,
                         'reservation.json' in paths, 'admission.json' in paths)
        folder = _directory(tag, status)
        if status != 'COMPLETE':
            for log in guard.glob('*.log'):
                paths['guard_' + log.name] = log
        files, partial = {}, {}
        for name, origin in sorted(paths.items()):
            sha = digest(origin)
            if status != 'COMPLETE' and origin.suffix not in ('.json', '.jsonl', '.log'):
                partial[name] = sha
                continue
            target = MOVIE if stage == 'video' and name == MOVIE.name else folder / name
            install(origin, target, sha)
            files[name] = sha
        entries[tag] = dict(status=status, files=files, local_partial_sha256=partial)
    return entries


def _ledger(tag, entry, registration):
    stage, number = _key(tag)
    unit = locations(stage, number)[2]
    folder = _directory(tag, entry['status'])
    files, partial = entry['files'], entry['local_partial_sha256']
    external = {MOVIE.name} if stage == 'video' and entry['status'] == 'COMPLETE' else set()
    require(set(files) == {p.name for p in folder.glob('*') if p.is_file()} | external, 'unaccounted public artifact')
    require(not set(partial) & set(files) and all(Path(k).name == k and re.fullmatch('[0-9a-f]{64}', v)
            for k, v in partial.items()), 'invalid retained partial artifact')
    for name, sha in files.items():
        target = MOVIE if name in external else folder / name
        require(Path(name).name == name and digest(target) == sha, 'changed compound evidence ' + name)
    require(not external or not (folder / MOVIE.name).exists(), 'unexpected duplicate public video')
    guard = read(folder/'guard.json') if 'guard.json' in files else None
    header = read(folder/'reservation.json') if 'reservation.json' in files else None
    dispatch = read(folder/'admission.json') if 'admission.json' in files else None
    status = _status(guard, header is not None, dispatch is not None)
    require(entry['status'] == status, 'wrong attempt status')
    expected = dict(stage=stage, attempt=number, unit=unit, registration_sha256=registration)
    if header is not None:
        require(all(header[k] == v for k,v in expected.items()) and header['reserved_at_unix_ns'] > 0,
                'wrong worker reservation')
    resources, ended, error = None, None, None
    if guard is not None:
        require(guard['supervisor_sha256'] == GUARD_SHA and guard['schema'] == 'pressure-supervisor83-v1'
                and guard['timeout_s'] == TIMEOUT and guard['new_physics_steps_in_supervisor'] == 0
                and guard['scope'].endswith('/app.slice/' + unit + '.service') and guard['command'][1:] ==
                ['-B', '-m', 'tools116.run', 'worker', stage, '--attempt', str(number)], 'wrong resource supervisor')
        error = guard.get('error')
        if status == 'COMPLETE':
            resources = resource_summary(folder, guard)
            samples = [json.loads(line) for line in (folder/'resources.jsonl').read_text().splitlines()]
            require(all(s['self_scope'] == s['owned']['path'] == guard['scope'] for s in samples)
                    and min(s['effective_available_bytes'] for s in samples) == resources['minimum_effective_bytes'],
                    'unbound resource measurements')
        else:
            resources = {k:guard.get(k) for k in ('minimum_effective_bytes','peak_group_bytes','child_started','returncode')}
    if dispatch is not None:
        require(all(dispatch[k] == v for k,v in expected.items()) and dispatch['child_started'] is False
                and guard is None and header is None, 'unbound pre-admission receipt')
        error = dispatch.get('error') or dispatch.get('reason')
    if status == 'COMPLETE':
        report = read(folder/'report.json')
        outputs = set(files) - RECEIPTS
        native = stage in NATIVE_STAGES and outputs == {'trace.npz'}
        video = stage == 'video' and MOVIE.name in outputs and all(
            name == MOVIE.name or name.endswith(('.json', '.jsonl')) for name in outputs)
        require((stage == 'wiring' and not outputs or native or video) and not partial and RECEIPTS <= set(files)
                and not any('partial' in name for name in outputs)
                and report['status'] == 'COMPLETE' and all(report.get(k) == v for k,v in header.items())
                and report['files'] == {k: files[k] for k in outputs}
                and math.isfinite(report['elapsed_s']) and report['elapsed_s'] >= 0
                and report['completed_at_unix_ns'] >= header['reserved_at_unix_ns'], 'wrong completed stage receipt')
        ended = report['completed_at_unix_ns']
    elif 'worker_partial.json' in files:
        receipt = read(folder/'worker_partial.json')
        require(header is not None and all(receipt.get(k) == v for k,v in header.items())
                and receipt['status'] == 'PARTIAL', 'unbound partial receipt')
        error = receipt.get('error') or error
    if status != 'COMPLETE' and 'report.json' in files:
        report = read(folder/'report.json')
        require(header is not None and all(report.get(k) == v for k,v in header.items())
                and all(files.get(k, partial.get(k)) == sha for k,sha in report.get('files', {}).items()),
                'unbound worker output in incomplete supervisor')
    return dict(stage=stage, attempt=number, status=status, resources=resources, error=error,
                reserved_at_unix_ns=header['reserved_at_unix_ns'] if header else None, completed_at_unix_ns=ended)


def _select(attempts):
    result = {}
    for stage in STAGES:
        tags = sorted(k for k,v in attempts.items() if v['stage'] == stage)
        require([attempts[k]['attempt'] for k in tags] == list(range(1,len(tags)+1)), 'missing prior attempt')
        started = [k for k in tags if attempts[k]['status'] != 'PRE_ADMISSION_FAILURE']
        require(not started or started == tags[-1:], 'repeated a started stage')
        tag = started[0] if started else None
        result[stage] = dict(status=attempts[tag]['status'] if tag else 'NOT_STARTED', attempt=tag,
                             admission_failures=[k for k in tags if k != tag])
    return result


def _wiring(stage):
    if stage['status'] != 'COMPLETE':
        return None
    from tools116.wiring import audit
    replay = audit()
    compare_tree(replay, read(PUBLIC/'wiring/report.json')['result'])
    require(replay['passed'] and replay['ticks'] == 1800 and [r['seed'] for r in replay['cases']] == list(SEEDS)
            and all(replay[k] == 0 for k in ('new_actor_forwards','physics','policy_updates','world_updates')),
            'wrong archived wiring inputs or budget')
    return replay


def _segment_readback(trace, motion):
    """Describe the 5–6 m interval and final half-second after scoring the registered gates."""
    gates = motion['gate_times_s']
    last_metre = None
    if len(gates) == 4:
        t = trace['time'][:-1]
        interval = (t >= gates[2]) & (t < gates[3])
        trim = trace['velocity_trim'][interval]
        drop = (trace['base_cmd'] - trace['preview_actor_vx'])[interval]
        if len(trim):
            last_metre = dict(samples=len(trim), trim_min_med_max_m_s=np.quantile(trim,[0.,.5,1.]).tolist(),
                              radar_reduction_min_max_m_s=[float(drop.min()),float(drop.max())])
    tail = None
    if motion['ticks'] == 600:
        applied = 5. * trace['applied'][575:,3::4].astype(float)
        power = trace['first_torque_nm'][575:,3::4] * trace['encoder_wheel_rad_s'][575:]
        tail = dict(wheel_samples=int(applied.size), negative_wheel_targets=int((applied < 0.).sum()),
                    first_servo_positive_power_samples=int((power > 0.).sum()),
                    wheel_saturation_flags=int(trace['wheel_saturation'][575:].sum()))
    return dict(last_metre=last_metre, stop_tail=tail)


def _native(seed, stage):
    if stage['status'] != 'COMPLETE':
        counts = None
        if stage['attempt']:
            folder = _directory(stage['attempt'], stage['status'])
            if (folder/'partial_counts.json').is_file():
                counts = read(folder/'partial_counts.json')['counts']
            elif (folder/'report.json').is_file():
                counts = read(folder/'report.json').get('result', {}).get('counts')
        return dict(seed=seed, stage_status=stage['status'], counts=counts, complete_horizon=False, gate_pass=False,
                    motion=None, lidar=None, first_failed_section=None)
    from tools111.control import SPEED_GAIN
    from tools116.score import score_candidate
    folder = PUBLIC / f'native_{seed}'
    row = read(folder/'report.json')['result']
    score = score_candidate(folder/'trace.npz', seed)
    model = read(ROOT/'reports/speed110/results.json')['preflight']
    require(row['status'] == 'COMPLETE' and row['seed'] == seed and row['layout'] == 'development'
            and row['arm'] == 'both' and row['gain'] == SPEED_GAIN and row['trim'] == 'brake_aware_after_lidar'
            and row['actor_sha256'] == model['model_sha256'] and row['actuator_config_sha256'] == model['config_sha256']
            and row['mujoco_version'] == '3.3.0', 'wrong native actor, intervention or simulator')
    for key in ('counts','motion','effects','net_actor_effects','trace_sha256','reference_sha256'):
        compare_tree(score[key], row[key])
    wiring = score['wiring']
    n, before = score['counts']['decisions'], wiring['matched_raw_before_input']
    applied = min(n, before+1)
    with np.load(folder/'trace.npz', allow_pickle=False) as trace:
        segment_readback = _segment_readback(trace, score['motion'])
        compare_tree(float(trace['nominal_base_height_m']), row['nominal_base_height_m'])
        packets = int(np.count_nonzero(trace['capture_tick'] < min(n,applied+1)))
    compare_tree(dict(first_changed_actor_frame_tick=wiring['first_changed_input_tick'], equal_issued_ticks=before,
                      equal_applied_ticks=applied, equal_state_endpoints=applied+1, equal_lidar_packets=packets),
                 row['paired_prefix'])
    return dict(stage_status='COMPLETE', segment_readback=segment_readback, **score)


def _film(stage, case):
    if stage['status'] != 'COMPLETE':
        return None
    require(case['stage_status'] == 'COMPLETE' and case['seed'] == SEEDS[0], 'film requires its fixed native source')
    row = read(PUBLIC/'video/report.json')['result']
    stream = row['stream']
    frames, fps = row['frames'], row['fps']
    source = PUBLIC / NATIVE_STAGES[0] / 'report.json'
    require(row['status'] == 'COMPLETE' and row['filename'] == MOVIE.name and row['seed'] == case['seed']
            and row['trace_sha256'] == [case['reference_sha256'], case['trace_sha256']]
            and row['last_ticks'] == [case['reference_motion']['ticks'], case['motion']['ticks']]
            and row['renderer_sha256'] == digest(ROOT/'tools116/video.py') and row['native_report_sha256'] == digest(source)
            and row['sha256'] == digest(MOVIE) and 0 < row['bytes'] == MOVIE.stat().st_size <= 5_000_000
            and row['cover_frames'] == row['physics'] == 0 and 'llvmpipe' in row['driver']['renderer'].lower(),
            'wrong film source, renderer or artifact')
    require(frames > 0 and fps == 25 and int(stream['nb_read_frames']) == frames
            and (stream['width'],stream['height'],stream['codec_name'],stream['pix_fmt'],stream['r_frame_rate'])
            == (1280,720,'h264','yuv420p','25/1') and abs(float(stream['duration']) - frames/fps) <= .001
            and float(stream['duration']) == row['duration_s'], 'wrong encoded stream receipt')
    count, previous = 0, -1
    for shot in row['shots']:
        require(previous < shot['tick'] <= max(row['last_ticks']) and shot['first_frame'] == count
                and shot['frames'] > 0, 'invalid native film clock')
        count, previous = count+shot['frames'], shot['tick']
    require(count == frames and previous == max(row['last_ticks']), 'incomplete film timeline')
    return dict(file=str(MOVIE.relative_to(ROOT)), **{k:row[k] for k in ('desktop_filename','seed','trace_sha256',
                'native_report_sha256','last_ticks','frames','fps','duration_s','bytes','sha256')})


def summarize(cases):
    require([c['seed'] for c in cases] == list(SEEDS), 'wrong registered native roster')
    complete = [c for c in cases if c['stage_status'] == 'COMPLETE']
    counts = dict.fromkeys(LIMIT, 0)
    unknown = []
    for case in cases:
        used = case['counts']
        if used is None:
            if case['stage_status'] not in ('NOT_STARTED','INTERRUPTED_BEFORE_RESERVATION'):
                unknown.append(case['seed'])
            continue
        require(used.keys() == LIMIT.keys() and all(type(v) is int and 0 <= v <= LIMIT[k] for k,v in used.items()),
                'native counts exceed the registered per-case budget')
        for k,v in used.items():
            counts[k] += v
    by_seed = []
    for c in complete:
        failed = list(c['motion']['failed'])
        if not c['lidar']['speed_active']:
            failed.append('pre_contact_lidar_speed')
        if not c['lidar']['lift_active']:
            failed.append('pre_contact_lidar_lift')
        by_seed.append(dict(seed=c['seed'], failed=failed, first_failed_section=c['first_failed_section']))
    return dict(registered_pairs=len(SEEDS), completed_native_stages=len(complete),
                full_horizons=sum(c['complete_horizon'] for c in complete),
                motion_gate_passes={g:sum(g not in c['motion']['failed'] for c in complete) for g in GATES},
                motion_passes=sum(c['motion_pass'] for c in complete),
                lidar_speed_passes=sum(c['lidar']['speed_active'] for c in complete),
                lidar_lift_passes=sum(c['lidar']['lift_active'] for c in complete),
                lidar_passes=sum(c['lidar_pass'] for c in complete), combined_passes=sum(c['gate_pass'] for c in complete),
                failures_by_seed=by_seed, confirmed_native_counts=counts, unknown_partial_native_seeds=unknown,
                native_counts_upper_bound={k:v+len(unknown)*LIMIT[k] for k,v in counts.items()},
                policy_updates=0, world_updates=0, world_model_steps=0)


def _stage_order(attempts):
    stages = _select(attempts)
    intervals = sorted((a['reserved_at_unix_ns'], a['completed_at_unix_ns']) for a in attempts.values()
                       if a['completed_at_unix_ns'] is not None)
    require(all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:])), 'overlapping completed workers')
    for name in STAGES:
        source = 'wiring' if name in NATIVE_STAGES else NATIVE_STAGES[0] if name == 'video' else None
        s = stages[name]
        allowed = source is None or stages[source]['status'] == 'COMPLETE'
        if s['status'] == 'NOT_STARTED':
            require(not allowed or s['admission_failures'], 'eligible stage has no outcome: ' + name)
            s['reason'] = 'pre-admission failures' if s['admission_failures'] else 'source unavailable: ' + source
        elif source:
            require(allowed, 'prerequisite not completed: ' + source)
            begin = attempts[s['attempt']]['reserved_at_unix_ns']
            end = attempts[stages[source]['attempt']]['completed_at_unix_ns']
            require(begin is None or begin >= end, 'stage started before its prerequisite completed')
    return stages


def index(entries):
    registrations = {s:protocol(s) for s in STAGES}
    attempts = {tag:_ledger(tag, entry, registrations[_key(tag)[0]]) for tag,entry in entries.items()}
    stages = _stage_order(attempts)
    observed = {p.name for p in PUBLIC.glob('attempts/*') if p.is_dir()}
    expected = {tag for tag,e in entries.items() if e['status'] != 'COMPLETE' and e['files']}
    require(observed == expected and {s for s in STAGES if (PUBLIC/s).exists()} ==
            {s for s,v in stages.items() if v['status'] == 'COMPLETE'}, 'unlisted public attempt or stage')
    wiring = _wiring(stages['wiring'])
    cases = [_native(seed, stages[name]) for seed,name in zip(SEEDS,NATIVE_STAGES)]
    video = _film(stages['video'], cases[0])
    require((video is not None) == MOVIE.is_file(), 'public movie availability differs from its completed stage')
    summary = summarize(cases)
    design = read(PLANS/'design.json')
    budget = design['budget']
    require(budget['episodes'] == len(cases) and budget['blind'] == budget['world_updates'] == budget['policy_updates'] == 0
            and design['preselected_video_seed'] == SEEDS[0] and design['thresholds']['all_development'] == len(cases)
            and all(total <= budget[k] == len(cases)*LIMIT[k] for k,total in summary['native_counts_upper_bound'].items()),
            'outcomes exceed or differ from the registered design')
    for case in cases:
        if case['stage_status'] == 'COMPLETE':
            compare_tree(dict(trace_sha256=case['reference_sha256'], case_status=case['reference_case_status'],
                              unit_status=case['reference_parent_unit_status']), design['references'][str(case['seed'])])
    if video:
        require(video['bytes'] <= design['video_max_bytes'], 'registered film budget exceeded')
    summary.update(admission_failures=[k for k,v in attempts.items() if v['status'] == 'PRE_ADMISSION_FAILURE'],
                   partial_stages=[k for k,v in stages.items() if v['status'] not in ('COMPLETE','NOT_STARTED')],
                   not_started=[k for k,v in stages.items() if v['status'] == 'NOT_STARTED'],
                   all_three_development_pairs_passed=summary['combined_passes'] == len(cases), blind_episodes=0)
    return dict(design_sha256=digest(PLANS/'design.json'), registrations=registrations, attempts=entries,
                attempt_results=attempts, stages=stages, wiring=wiring, cases=cases, video=video, summary=summary,
                scope='Three development starts paired to round 111. Speed reduction holds the encoder correction '
                      'fixed at the same tick. Leg lift is measured on the applied servo target before contact.')


def publish():
    entries = _collect()
    require(entries, 'no experiment receipts to publish')
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
    print(json.dumps(verify(), indent=2, allow_nan=False))
