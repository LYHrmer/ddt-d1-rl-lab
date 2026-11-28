"""Publish each independent start and reconstruct its original control and course gates."""
import argparse
import hashlib
import json
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
from tools117.archive import RECEIPTS, totals
from tools118.run import BASE, NATIVE_STAGES, PUBLIC, SEEDS, STAGES, TIMEOUT, locations, protocol


MOVIE = ROOT / 'docs/assets/compound118.mp4'
PARENT = ROOT / 'reports/compound117'
PARENT_COMMIT = '621ba6dcc7cd633ffb11485708efb6b1202d9c28'


def key(tag):
    stage, _, number = tag.rpartition('_')
    require(stage in STAGES and re.fullmatch('[0-9]{2}', number) and int(number) > 0, 'unknown attempt ' + tag)
    return stage, int(number)


def directory(tag, status):
    return PUBLIC/key(tag)[0] if status == 'COMPLETE' else PUBLIC/'attempts'/tag


def collect():
    tags = {p.name.removeprefix('guard_') for p in BASE.glob('*') if p.is_dir()}
    tags.update(p.stem.removeprefix('admission_') for p in BASE.glob('admission_*.json'))
    entries = {}
    for tag in sorted(tags):
        stage, number = key(tag)
        worker, guard, _ = locations(stage,number)
        paths = {p.name:p for p in worker.glob('*') if p.is_file()}
        for path, name in ((guard/'report.json','guard.json'), (guard/'resources.jsonl','resources.jsonl'),
                           (BASE/f'admission_{tag}.json','admission.json')):
            if path.is_file():
                require(name not in paths, 'overlapping worker and supervisor evidence')
                paths[name] = path
        guardian = read(paths['guard.json']) if 'guard.json' in paths else None
        status = _status(guardian,'reservation.json' in paths,'admission.json' in paths)
        if status != 'COMPLETE':
            paths.update({'guard_'+p.name:p for p in guard.glob('*.log')})
        files, retained = {}, {}
        for name, path in sorted(paths.items()):
            sha = digest(path)
            if status != 'COMPLETE' and path.suffix not in ('.json','.jsonl','.log'):
                retained[name] = sha
                continue
            dest = MOVIE if stage == 'video' and status == 'COMPLETE' and name == MOVIE.name else directory(tag,status)/name
            install(path,dest,sha)
            files[name] = sha
        entries[tag] = dict(status=status,files=files,local_partial_sha256=retained)
    return entries


def attempt(tag, entry, registration):
    stage, number = key(tag)
    status, files, retained = entry['status'], entry['files'], entry['local_partial_sha256']
    local = directory(tag,status)
    movie = {MOVIE.name} if stage == 'video' and status == 'COMPLETE' else set()
    require({p.name for p in local.glob('*') if p.is_file()} == files.keys()-movie and not files.keys() & retained.keys(),
            'unlisted or ambiguous public output')
    for name, sha in (files | retained).items():
        require(Path(name).name == name and re.fullmatch('[0-9a-f]{64}', sha), 'invalid artifact name or digest')
        if name in files:
            require(sha == digest(MOVIE if name in movie else local/name), 'changed public evidence ' + name)
    guard = read(local/'guard.json') if 'guard.json' in files else None
    header = read(local/'reservation.json') if 'reservation.json' in files else None
    admission = read(local/'admission.json') if 'admission.json' in files else None
    unit = locations(stage,number)[2]
    expected = dict(stage=stage,attempt=number,unit=unit,registration_sha256=registration)
    require(_status(guard,header is not None,admission is not None) == status, 'different recorded attempt status')
    start = end = error = resources = None
    if header is not None:
        require(expected.items() <= header.items() and header['reserved_at_unix_ns'] > 0, 'wrong reservation')
        start = header['reserved_at_unix_ns']
    if admission is not None:
        require(guard is None and header is None and expected.items() <= admission.items()
                and admission['child_started'] is False, 'wrong pre-admission')
        error = admission.get('error') or admission.get('reason')
    if guard is not None:
        require(guard['supervisor_sha256'] == GUARD_SHA and guard['schema'] == 'pressure-supervisor83-v1'
                and guard['scope'].endswith('/app.slice/'+unit+'.service') and guard['timeout_s'] == TIMEOUT
                and guard['new_physics_steps_in_supervisor'] == 0 and
                guard['command'][1:] == ['-B','-m','tools118.run','worker',stage,'--attempt',str(number)],
                'wrong supervisor, stage or resource policy')
        error = guard.get('error')
    if 'report.json' in files:
        report = read(local/'report.json')
        require(header is not None and header.items() <= report.items() and
                all(files.get(k,retained.get(k)) == v for k,v in report['files'].items()), 'unbound worker result')
    if status == 'COMPLETE':
        outputs = {MOVIE.name} if movie else {'trace.npz'}
        require(files.keys() == RECEIPTS | outputs and not retained and report['status'] == 'COMPLETE'
                and report['files'].keys() == outputs and report['elapsed_s'] >= 0
                and start <= report['completed_at_unix_ns'], 'wrong complete output')
        resources = resource_summary(local,guard)
        samples = [json.loads(line) for line in (local/'resources.jsonl').read_text().splitlines()]
        require(all(s['self_scope'] == s['owned']['path'] == guard['scope'] for s in samples)
                and min(s['effective_available_bytes'] for s in samples) == resources['minimum_effective_bytes'],
                'resource samples do not belong to blind stage')
        end = report['completed_at_unix_ns']
    if 'worker_partial.json' in files:
        partial = read(local/'worker_partial.json')
        require(header is not None and header.items() <= partial.items() and partial['status'] == 'PARTIAL',
                'unbound partial worker')
        error = partial.get('error') or error
    return dict(stage=stage,attempt=number,status=status,start_ns=start,end_ns=end,resources=resources,error=error)


def stages_for(attempts):
    stages = {}
    for stage in STAGES:
        tags = sorted(k for k,r in attempts.items() if r['stage'] == stage)
        require([attempts[k]['attempt'] for k in tags] == list(range(1,len(tags)+1)), 'missing earlier attempt')
        started = [k for k in tags if attempts[k]['status'] != 'PRE_ADMISSION_FAILURE']
        require(not started or started == tags[-1:], 'blind stage retried after starting')
        tag = started[0] if started else None
        stages[stage] = dict(status=attempts[tag]['status'] if tag else 'NOT_STARTED', attempt=tag,
                             admission_failures=[k for k in tags if k != tag])
    clocks = sorted((r['start_ns'],r['end_ns']) for r in attempts.values() if r['end_ns'] is not None)
    require(all(l[1] <= r[0] for l,r in zip(clocks,clocks[1:])), 'overlapping experiment workers')
    film, first = stages['video'], stages[NATIVE_STAGES[0]]
    if film['attempt'] or film['admission_failures']:
        require(first['status'] == 'COMPLETE', 'video without its preselected source')
        begin = attempts[film['attempt']]['start_ns'] if film['attempt'] else None
        require(begin is None or begin >= attempts[first['attempt']]['end_ns'], 'video preceded its source')
    return stages


def reset_hash(trace):
    with np.load(trace, allow_pickle=False) as z:
        measured = np.concatenate((z['qpos'][0],z['qvel'][0])).astype('<f8')
    return hashlib.sha256(measured.tobytes()).hexdigest()


def case(seed, stage):
    from tools118.score import score_trace
    status, tag = stage['status'], stage['attempt']
    if status != 'COMPLETE':
        counts = None
        if tag:
            local = directory(tag,status)
            if (local/'partial_counts.json').is_file():
                counts = read(local/'partial_counts.json')['counts']
            elif (local/'report.json').is_file():
                counts = read(local/'report.json')['result']['counts']
        return dict(seed=seed,stage_status=status,counts=counts,complete_horizon=False,driving_pass=False,gate_pass=False)
    local = directory(tag,status)
    trace = local/'trace.npz'
    recorded = read(local/'report.json')['result']
    row = score_trace(trace,seed)
    for k in ('seed','trace_sha256','counts','motion','lidar','stop','gate_pass','driving_pass','first_failed_section'):
        compare_tree(row[k],recorded[k])
    model = read(ROOT/'reports/speed110/results.json')['preflight']
    require(recorded['parking'] is True and recorded['status'] == 'COMPLETE' and row['seed'] == seed
            and (recorded['layout'],recorded['arm'],recorded['gain']) == ('development','both',SPEED_GAIN)
            and recorded['actor_sha256'] == model['model_sha256'] and recorded['actuator_config_sha256'] == model['config_sha256']
            and recorded['mujoco_version'] == '3.3.0' and reset_hash(trace) == row['reset']['fingerprint_sha256'],
            'different reset, course, controller or native model')
    with np.load(trace, allow_pickle=False) as z:
        compare_tree(float(z['nominal_base_height_m']),recorded['nominal_base_height_m'])
    return dict(row,stage_status=status)


def summarize(cases, development_resets):
    seeds = [int.from_bytes(hashlib.sha256(f'D1-compound118:{PARENT_COMMIT}:{i}'.encode('ascii')).digest()[:4],'big')
             for i in (1,2,3)]
    require(seeds == list(SEEDS) == [r['seed'] for r in cases], 'unregistered blind seed order or derivation')
    complete = [r for r in cases if r['stage_status'] == 'COMPLETE']
    fingerprints = {r['reset']['fingerprint_sha256'] for r in complete}
    require(len(fingerprints) == len(complete) and not fingerprints & set(development_resets), 'reused native reset')
    summary = totals(cases)
    failures = []
    for r in complete:
        failed = list(r['motion']['failed'])
        if not r['lidar']['speed_active']:
            failed.append('pre_contact_lidar_speed')
        if not r['lidar']['lift_active']:
            failed.append('pre_contact_lidar_lift')
        require(bool(failed) != r['gate_pass'] and (r['complete_horizon'] or not r['gate_pass']), 'inconsistent blind outcome')
        if failed:
            failures.append(dict(seed=r['seed'],failed=failed,first_failed_section=r['first_failed_section']))
    return dict(summary,registered_blind_starts=len(SEEDS),distinct_completed_resets=len(fingerprints),
                early_terminated_seeds=[r['seed'] for r in complete if not r['complete_horizon']],
                failures=failures,stop_passes=sum('stopped' not in r['motion']['failed'] and r['complete_horizon'] for r in complete),
                lidar_passes=sum(r['lidar']['passed'] for r in complete),
                all_three_blind_passed=summary['joint_passes'] == len(SEEDS),world_updates=0,policy_updates=0)


def index(entries):
    registration = protocol(STAGES[0])
    require(all(protocol(s) == registration for s in STAGES), 'more than one blind registration')
    attempts = {tag:attempt(tag,e,registration) for tag,e in entries.items()}
    stages = stages_for(attempts)
    require({p.name for p in PUBLIC.glob('*') if p.is_dir() and p.name != 'attempts'} ==
            {s for s,r in stages.items() if r['status'] == 'COMPLETE'} and
            {p.name for p in (PUBLIC/'attempts').glob('*') if p.is_dir()} ==
            {tag for tag,e in entries.items() if e['status'] != 'COMPLETE' and e['files']}, 'unlisted public trial')
    cases = [case(seed,stages[s]) for seed,s in zip(SEEDS,NATIVE_STAGES,strict=True)]
    parent = read(PARENT/'results.json')['parking_cases']
    require(len(parent) == 3 and all(r['gate_pass'] and r['complete_horizon'] for r in parent), 'unqualified development source')
    dev_resets = set()
    for row in parent:
        path = PARENT/f"parking_{row['seed']}"/'trace.npz'
        require(digest(path) == row['trace_sha256'], 'changed development reference')
        dev_resets.add(reset_hash(path))
    summary = summarize(cases,dev_resets)
    summary.update(admission_failures=[k for k,r in attempts.items() if r['status'] == 'PRE_ADMISSION_FAILURE'],
                   partial_stages=[s for s,r in stages.items() if r['status'] not in ('COMPLETE','NOT_STARTED')],
                   not_started=[s for s,r in stages.items() if r['status'] == 'NOT_STARTED'])
    video = None
    if stages['video']['status'] == 'COMPLETE':
        from tools118.video import verify
        row = read(PUBLIC/'video/report.json')['result']
        verify(row,MOVIE,PUBLIC/NATIVE_STAGES[0])
        require(row['trace_sha256'] == cases[0]['trace_sha256'] and row['seed'] == SEEDS[0], 'wrong blind replay source')
        video = dict(file=str(MOVIE.relative_to(ROOT)), **{k:row[k] for k in (
            'seed','trace_sha256','source_report_sha256','desktop_filename','frames','fps','duration_s','bytes','sha256')})
    require(MOVIE.is_file() == (video is not None), 'public blind movie without its completed stage')
    return dict(registration_sha256=registration,attempts=entries,attempt_results=attempts,
                stages=stages,cases=cases,summary=summary,video=video)


def publish():
    entries = collect()
    require(entries, 'no blind receipts to publish')
    install_json(PUBLIC/'results.json',index(entries))


def verify():
    saved = read(PUBLIC/'results.json')
    rebuilt = index(saved['attempts'])
    compare_tree(rebuilt,saved)
    return rebuilt['summary']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    if parser.parse_args().publish:
        publish()
    if (PUBLIC/'results.json').is_file():
        print(json.dumps(verify(),indent=2,allow_nan=False))
    else:
        require(not PUBLIC.exists(),'incomplete public blind evidence')
        print('Registered blind starts', SEEDS, protocol(STAGES[0]))
