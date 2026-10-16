#!/usr/bin/env python3
"""Replay the command course of every training episode and join it with what happened.

Each training episode's command course comes from SeedSequence([seed + rank, rank,
episode index, 1]) and nothing else, so it can be regenerated offline by calling the
environment's own _make_training_schedule on that stream; nothing is reimplemented. The
Monitor logs record each finished episode's return and length in order, and a length
below the episode horizon means the episode ended early (a fall or illegal contact).
Joining the two says, for every training episode, which course it ran and whether, and
in which segment, it ended early. Builds no environment and executes no physics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))

HORIZON = 1000
PHASES = ((0., .125), (.125, .25), (.25, .5), (.5, .75), (.75, 1.))
FINE_BINS = 16
PPO_COLUMNS = ('train/approx_kl', 'train/clip_fraction', 'train/std', 'train/explained_variance', 'rollout/ep_rew_mean')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_stream(base_seed, rank, episode_index):
    return np.random.default_rng(np.random.SeedSequence([base_seed, rank, episode_index, 1]))


def schedule(config, base_seed, rank, episode_index):
    """The course the environment itself would draw, from its own method."""
    from d1_rl.task03d import D1Task03dEnv
    fake = SimpleNamespace(config=config, _command_rng=episode_stream(base_seed, rank, episode_index))
    return D1Task03dEnv._make_training_schedule(fake)


def read_monitor(path):
    lines = Path(path).read_text().splitlines()
    rows = list(csv.DictReader(lines[1:]))
    return [{'r': float(r['r']), 'l': int(r['l'])} for r in rows]


def segment_at(course, tick):
    """Index and command of the segment that contains a tick; None past the end."""
    for index, (begin, end, command) in enumerate(course['segments']):
        if begin <= tick < end:
            return index, command
    return None


def label(course, segment_index):
    """A readable name for where in the course an episode ended."""
    begin, end, (vx, _vy, wz) = course['segments'][segment_index]
    if segment_index == 0:
        return 'initial_zero'
    if segment_index == len(course['segments'])-1:
        return 'final_zero'
    if vx == 0. and wz == 0.:
        return 'middle_zero'
    if course['mode'] == 'translation':
        return 'drive_forward' if vx > 0 else 'drive_backward'
    return 'turn_left' if wz > 0 else 'turn_right'


def join(config, base_seed, rank, monitor_rows, steps_per_env):
    out, done_steps = [], 0
    for k, row in enumerate(monitor_rows):
        course = schedule(config, base_seed, rank, k)
        early = row['l'] < HORIZON
        where = label(course, segment_at(course, row['l']-1)[0]) if early else None
        out.append({'episode': k, 'mode': course['mode'], 'first_sign': course['sign'],
                    'amplitude': course['parameters']['amplitudes'][0], 'length': row['l'], 'return': row['r'],
                    'ended_early': early, 'ended_in': where, 'training_fraction': done_steps/steps_per_env})
        done_steps += row['l']
    return out


def phase_of(fraction, phases=PHASES):
    for lo, hi in phases:
        if lo <= fraction < hi:
            return f'{lo:g}-{hi:g}'
    return f'{phases[-1][0]:g}-{phases[-1][1]:g}'


def aggregate(episodes, phases=PHASES):
    """Early-end rate and mean return by phase of training and by course."""
    table = {}
    for e in episodes:
        group = 'turn' if e['mode'] == 'turn' else ('forward_first' if e['first_sign'] > 0 else 'backward_first')
        cell = table.setdefault(phase_of(e['training_fraction'], phases), {}).setdefault(
            group, {'episodes': 0, 'ended_early': 0, 'return_sum': 0., 'ended_in': {}})
        cell['episodes'] += 1
        cell['return_sum'] += e['return']
        if e['ended_early']:
            cell['ended_early'] += 1
            cell['ended_in'][e['ended_in']] = cell['ended_in'].get(e['ended_in'], 0)+1
    for groups in table.values():
        for cell in groups.values():
            cell['early_rate'] = cell['ended_early']/cell['episodes']
            cell['mean_return'] = cell.pop('return_sum')/cell['episodes']
    return table


def config_file_of(protocol):
    """The configuration file the training recorded, refused if it no longer matches the recorded content."""
    candidates = [p for p in protocol['inputs_sha256'] if p.endswith('.json') and '/training/configs/' in p
                  and json.loads(Path(p).read_text()) == protocol['config']]
    if len(candidates) != 1:
        raise SystemExit(f'cannot identify the training configuration among {candidates}')
    if sha(candidates[0]) != protocol['inputs_sha256'][candidates[0]]:
        raise SystemExit('the training configuration changed since the run')
    return Path(candidates[0])


def verify_against_environment(config_path, config, seed, ranks, episodes_per_rank):
    """Build the real environment, reset it as SB3 did, and compare its own courses with the replay.

    Reset places the robot and builds the first observation; it advances no simulated time,
    which the native-step counter must confirm.
    """
    from d1_rl.factory03 import environment_class
    checked, mismatches, natives = 0, [], {}
    for rank in ranks:
        env = environment_class(config)(config=config_path, arm='B', seed=seed, rank=rank)
        try:
            for k in range(episodes_per_rank):
                env.reset(seed=seed+rank) if k == 0 else env.reset()
                env._update_command()
                real, mine = env._training_schedule, schedule(config, seed+rank, rank, k)
                if (real['mode'], real['sign'], real['segments']) != (mine['mode'], mine['sign'], mine['segments']):
                    mismatches.append({'rank': rank, 'episode': k})
                checked += 1
            natives[rank] = env.counters()['native_steps']
        finally:
            env.close()
    return {'episodes_checked': checked, 'mismatches': mismatches, 'native_steps_by_rank': natives,
            'passed': not mismatches and all(v == 0 for v in natives.values())}


def fine_timeline(episodes, bins=FINE_BINS):
    """Per slice of training: episodes run and early ends by the segment they ended in."""
    out = []
    for b in range(bins):
        lo, hi = b/bins, (b+1)/bins
        # A start fraction is taken before the episode's own length is added, so it is always below 1.
        sel = [e for e in episodes if lo <= e['training_fraction'] < hi]
        ends = {}
        for e in sel:
            if e['ended_early']:
                ends[e['ended_in']] = ends.get(e['ended_in'], 0)+1
        out.append({'from': lo, 'to': hi, 'translation_episodes': sum(e['mode'] == 'translation' for e in sel),
                    'turn_episodes': sum(e['mode'] == 'turn' for e in sel), 'early_ends_by_segment': ends})
    return out


def ppo_timeline(progress_csv, bins=FINE_BINS, columns=PPO_COLUMNS):
    """Mean of each PPO diagnostic over the same slices of training; empty cells are skipped."""
    rows = list(csv.DictReader(Path(progress_csv).open()))
    n, out = len(rows), []
    for b in range(bins):
        chunk = rows[int(b*n/bins):int((b+1)*n/bins)]
        entry = {'from': b/bins, 'to': (b+1)/bins}
        for c in columns:
            values = [float(r[c]) for r in chunk if r.get(c) not in ('', None)]
            entry[c] = sum(values)/len(values) if values else None
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--baseline-train-dir', type=Path, help='a shorter run with the same seed; its logs must be a prefix')
    parser.add_argument('--verify-episodes', type=int, default=0, help='episodes per rank to check against the real environment')
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    protocol = json.loads((args.train_dir/'protocol.json').read_text())
    config, seed, envs = protocol['config'], protocol['seed'], protocol['envs']
    steps_per_env = protocol['planned_transitions']//envs
    episodes, prefix = [], {}
    for rank in range(envs):
        rows = read_monitor(args.train_dir/f'env_{rank:02d}.monitor.csv')
        if args.baseline_train_dir:
            base = read_monitor(args.baseline_train_dir/f'env_{rank:02d}.monitor.csv')
            prefix[rank] = {'baseline_episodes': len(base), 'identical_prefix': rows[:len(base)] == base}
        for e in join(config, seed+rank, rank, rows, steps_per_env):
            e['rank'] = rank
            episodes.append(e)
    verification = None
    if args.verify_episodes:
        config_path = config_file_of(protocol)
        verification = verify_against_environment(config_path, config, seed, range(envs), args.verify_episodes)
    if verification is not None and not verification['passed']:
        raise SystemExit(f'replay does not match the real environment: {verification}')
    report = {'format': 'd1-contract08-training-episode-replay-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'physics_steps_executed': 0, 'seed': seed, 'envs': envs, 'steps_per_env': steps_per_env,
              'episodes': len(episodes), 'baseline_prefix_check': prefix, 'replay_verified_against_environment': verification,
              'by_phase_and_course': aggregate(episodes),
              'timeline': fine_timeline(episodes),
              'ppo_timeline': ppo_timeline(args.train_dir/'progress.csv'),
              'inputs': {f'env_{r:02d}.monitor.csv': sha(args.train_dir/f'env_{r:02d}.monitor.csv') for r in range(envs)},
              'tool_sha256': sha(__file__)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    with (args.output/'episodes.jsonl').open('w') as stream:
        for e in episodes:
            stream.write(json.dumps(e)+'\n')
    print(json.dumps({'episodes': len(episodes), 'prefix_identical': all(p['identical_prefix'] for p in prefix.values()) if prefix else None}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
