"""Publish the saved braking suffixes and verify their inputs, transitions and parking-target audit."""
import argparse
import json
from pathlib import Path

from tools91.launch91 import digest, read, require
from tools109.run109 import GUARD_SHA
from tools111.results import install
from tools112.surface_results import install_json
from tools113.results import compare_tree
from tools114.common import BASE, PUBLIC, registration
from tools114.replay import BUNDLE, verify as verify_replay


INDEX = PUBLIC / 'replay.json'


def copy_files(source, target, names):
    result = {}
    for name in names:
        file = source / name
        result[name] = digest(file)
        install(file, target / name, result[name])
    return result


def publish():
    reservations = sorted(BASE.glob('replay_??/reservation.json'))
    require(len(reservations) == 1, 'expected exactly one replay reservation')
    stage = reservations[0].parent
    selected_guard = 'guard_' + stage.name
    permit = read(BASE / selected_guard / 'report.json')
    require(permit['status'] == 'COMPLETE' and permit['child_started'], 'preserve incomplete replay as partial')
    names = ('reservation.json', 'report.json', BUNDLE)
    require({p.name for p in stage.iterdir() if p.is_file()} == set(names), 'unexpected replay files')
    files = copy_files(stage, PUBLIC / 'replay', names)
    guards = {}
    for folder in sorted(BASE.glob('guard_replay_??')):
        available = [n for n in ('report.json', 'resources.jsonl') if (folder / n).is_file()]
        guards[folder.name] = copy_files(folder, PUBLIC / 'resources' / folder.name, available)
    index = dict(registration_sha256=registration(), files=files, guard=selected_guard, guards=guards)
    verify(index)
    install_json(INDEX, index)


def check_files(folder, files):
    for name, expected in files.items():
        require(Path(name).name == name and digest(folder / name) == expected, 'changed evidence file ' + name)


def resource_summary(folder, receipt):
    require(receipt['status'] == 'COMPLETE' and receipt['returncode'] == 0 and receipt['child_reaped'] and
            receipt['owned_cgroup_children_empty'] and len(receipt['cpu_affinity']) == 1, 'incomplete resource guard')
    readings = [json.loads(line) for line in (folder / 'resources.jsonl').read_text().splitlines()]
    startup = [r['effective_available_bytes'] for r in readings if r['stage'] == 'startup']
    require(startup and startup[-1] >= 4 << 30, 'inadequate memory at admission')
    for sample in readings:
        own = sample['owned']
        require((own['high_bytes'], own['max_bytes'], own['swap_max_bytes'], own['swap_current_bytes']) ==
                (2 << 30, 3 << 30, 0, 0) and sample['effective_available_bytes'] >= 2 << 30 and
                not own['events']['oom_kill'] and not sample['errors'], 'resource limits or reserve violated')
    return dict(startup_effective_bytes=startup[-1], minimum_effective_bytes=receipt['minimum_effective_bytes'],
                peak_group_bytes=receipt['peak_group_bytes'])


def verify(index):
    from tools114.audit import audit_archives
    require(index['registration_sha256'] == registration(), 'different parking registration')
    require(set(index['files']) == {'reservation.json', 'report.json', BUNDLE}, 'missing replay evidence')
    folder = PUBLIC / 'replay'
    check_files(folder, index['files'])
    reservation = read(folder / 'reservation.json')
    report = read(folder / 'report.json')
    require(report['status'] == 'COMPLETE' and reservation['registration_sha256'] == registration() and
            all(report[k] == v for k, v in reservation.items()) and
            report['completed_at_unix_ns'] >= reservation['reserved_at_unix_ns'], 'incomplete parking replay')
    selected = index['guard']
    require(selected in index['guards'] and reservation['unit'] == 'd1-parking114-' + selected[6:].replace('_', '-'),
            'wrong replay supervisor')
    for name, files in index['guards'].items():
        require(name.startswith('guard_replay_') and len(name) == len('guard_replay_00') and
                name[-2:].isdecimal() and 'report.json' in files and
                set(files) <= {'report.json', 'resources.jsonl'}, 'invalid supervisor reference')
        directory = PUBLIC / 'resources' / name
        check_files(directory, files)
        guard = read(directory / 'report.json')
        require(guard['supervisor_sha256'] == GUARD_SHA and guard['new_physics_steps_in_supervisor'] == 0,
                'different resource supervisor')
        if name == selected:
            require('resources.jsonl' in files and guard['child_started'], 'missing admission readings')
            usage = resource_summary(directory, guard)
        else:
            require(not guard['child_started'] and guard['status'] != 'COMPLETE', 'unaccounted replay worker')
    checks = verify_replay(folder, report['model_replay'])
    compare_tree(audit_archives(), report['archived_projection'])
    return dict(**checks, model_steps=report['model_replay']['model_steps'], new_native_episodes=0,
                policy_updates=0, world_updates=0, resources=usage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    if parser.parse_args().publish:
        publish()
    if INDEX.exists():
        print(json.dumps(verify(read(INDEX)), sort_keys=True))
    else:
        require(not PUBLIC.exists(), 'incomplete public parking archive')
        print('Parking protocol registered:', registration())
