"""Check the optional parking replay, including its single-CPU rendering receipt."""
import json
from pathlib import Path

from tools91.launch91 import ROOT, read, require
from tools109.run109 import GUARD_SHA
from tools114.results import resource_summary


PUBLIC = ROOT / 'reports/compound117_parking'
MOVIE = ROOT / 'docs/assets/compound117_parking.mp4'
NATIVE = ROOT / 'reports/compound117'
UNIT = 'd1-compound117-parking-film-01'


def check_guard(directory):
    guardian = read(directory/'guard.json')
    command = guardian['command']
    require(guardian['schema'] == 'pressure-supervisor83-v1' and guardian['supervisor_sha256'] == GUARD_SHA
            and guardian['timeout_s'] == 600 and guardian['child_started'] and guardian['new_physics_steps_in_supervisor'] == 0
            and guardian['scope'].endswith('/app.slice/' + UNIT + '.service') and len(command) == 6
            and command[1:-1] == ['-B', '-m', 'tools117.parking_video', '--output']
            and Path(command[-1]).is_absolute() and Path(command[-1]).parts[-3:] == ('evidence','compound117_parking','video_01'),
            'wrong parking video supervisor, worker or output')
    summary = resource_summary(directory, guardian)
    samples = [json.loads(line) for line in (directory/'resources.jsonl').read_text().splitlines()]
    require(all(s['self_scope'] == s['owned']['path'] == guardian['scope'] for s in samples)
            and min(s['effective_available_bytes'] for s in samples) == summary['minimum_effective_bytes'],
            'resource samples do not belong to parking video unit')
    return summary


def verify():
    if not PUBLIC.exists() and not MOVIE.exists():
        return None
    require(MOVIE.is_file() and all((PUBLIC/name).is_file() for name in ('video.json','guard.json','resources.jsonl')),
            'incomplete parking video publication')
    require((NATIVE/'results.json').is_file() and all((NATIVE/stage/name).is_file()
            for stage in ('speed_111101','parking_111101') for name in ('report.json','guard.json','trace.npz')),
            'parking video requires the two published native trajectories')
    from tools117.parking_video import verify as verify_movie
    verify_movie(PUBLIC/'video.json', MOVIE)
    return check_guard(PUBLIC)


if __name__ == '__main__':
    result = verify()
    print('Parking video has not been published.' if result is None else json.dumps(result, sort_keys=True))
