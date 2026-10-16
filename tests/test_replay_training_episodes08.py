"""Tests for tools08/replay_training_episodes08.py. No model, no physics.

The replay is only worth anything if it regenerates the course the environment really
used. That is checked against the real environment by the static run; here the pure
parts are pinned: the seed stream, the join with the monitor log, segment naming and the
two timelines.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))
sys.path.insert(0, str(ROOT/'training'))

import replay_training_episodes08 as rp  # noqa: E402

CONFIG = json.loads((ROOT/'training/configs/pilot03d.json').read_text())


def test_the_stream_is_the_environments_episode_stream_one():
    a = rp.episode_stream(102, 1, 5).random(3)
    b = np.random.default_rng(np.random.SeedSequence([102, 1, 5, 1])).random(3)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(rp.episode_stream(102, 1, 5).random(3), rp.episode_stream(102, 1, 6).random(3))


def test_a_schedule_is_reproducible_and_covers_the_episode():
    a, b = rp.schedule(CONFIG, 101, 0, 3), rp.schedule(CONFIG, 101, 0, 3)
    assert a == b and a['segments'][0][0] == 0 and a['segments'][-1][1] == rp.HORIZON


def test_translation_courses_go_out_and_come_back_with_the_same_amplitude():
    seen = 0
    for k in range(60):
        c = rp.schedule(CONFIG, 101, 0, k)
        if c['mode'] == 'translation':
            vx = [seg[2][0] for seg in c['segments'] if seg[2][0] != 0.]
            assert len(vx) == 2 and vx[0] == -vx[1] and np.sign(vx[0]) == c['sign']
            seen += 1
    assert seen > 10


def test_course_signs_are_balanced_over_many_episodes():
    signs = [rp.schedule(CONFIG, 101+r, r, k)['sign'] for r in range(8) for k in range(100)]
    assert abs(np.mean(signs)) < .1


def course(mode='translation', sign=1):
    if mode == 'translation':
        segs = ((0, 50, (0., 0., 0.)), (50, 300, (sign*.2, 0., 0.)), (300, 420, (0., 0., 0.)),
                (420, 670, (-sign*.2, 0., 0.)), (670, 1000, (0., 0., 0.)))
    else:
        segs = ((0, 50, (0., 0., 0.)), (50, 150, (0., 0., sign*.2)), (150, 250, (0., 0., -sign*.2)),
                (250, 1000, (0., 0., 0.)))
    return {'mode': mode, 'sign': sign, 'parameters': {'amplitudes': [.2]}, 'segments': segs}


@pytest.mark.parametrize('tick, name', [(10, 'initial_zero'), (100, 'drive_forward'), (350, 'middle_zero'),
                                        (500, 'drive_backward'), (900, 'final_zero')])
def test_segments_are_named_by_what_was_commanded(tick, name):
    c = course()
    assert rp.label(c, rp.segment_at(c, tick)[0]) == name


def test_turn_segments_are_named_by_turn_direction():
    c = course('turn')
    assert rp.label(c, rp.segment_at(c, 100)[0]) == 'turn_left' and rp.label(c, rp.segment_at(c, 200)[0]) == 'turn_right'


def test_segment_at_past_the_end_is_none():
    assert rp.segment_at(course(), 1000) is None


def test_join_marks_early_ends_and_where_they_happened(monkeypatch):
    monkeypatch.setattr(rp, 'schedule', lambda config, seed, rank, k: course(sign=1 if k == 0 else -1))
    out = rp.join(CONFIG, 101, 0, [{'r': 1., 'l': 120}, {'r': 2., 'l': 1000}], steps_per_env=2000)
    assert out[0]['ended_early'] and out[0]['ended_in'] == 'drive_forward' and out[0]['first_sign'] == 1
    assert not out[1]['ended_early'] and out[1]['ended_in'] is None
    assert out[0]['training_fraction'] == 0. and out[1]['training_fraction'] == pytest.approx(120/2000)


def test_the_last_tick_of_a_short_episode_decides_its_segment(monkeypatch):
    monkeypatch.setattr(rp, 'schedule', lambda *a: course())
    assert rp.join(CONFIG, 101, 0, [{'r': 0., 'l': 51}], 2000)[0]['ended_in'] == 'drive_forward'
    assert rp.join(CONFIG, 101, 0, [{'r': 0., 'l': 50}], 2000)[0]['ended_in'] == 'initial_zero'


def test_aggregate_groups_by_course_and_phase():
    eps = [{'mode': 'translation', 'first_sign': 1, 'training_fraction': .9, 'return': 2., 'ended_early': True, 'ended_in': 'drive_forward'},
           {'mode': 'translation', 'first_sign': 1, 'training_fraction': .95, 'return': 4., 'ended_early': False, 'ended_in': None},
           {'mode': 'turn', 'first_sign': -1, 'training_fraction': .1, 'return': 1., 'ended_early': False, 'ended_in': None}]
    out = rp.aggregate(eps)
    cell = out['0.75-1']['forward_first']
    assert cell['episodes'] == 2 and cell['early_rate'] == .5 and cell['mean_return'] == 3. and cell['ended_in'] == {'drive_forward': 1}
    assert out['0-0.125']['turn']['early_rate'] == 0.


def test_fine_timeline_counts_each_episode_once():
    eps = [{'mode': 'translation', 'training_fraction': f, 'ended_early': i % 2 == 0, 'ended_in': 'drive_forward'}
           for i, f in enumerate(np.linspace(0, .999, 64))]
    out = rp.fine_timeline(eps, bins=4)
    assert sum(b['translation_episodes'] for b in out) == 64
    assert sum(b['early_ends_by_segment'].get('drive_forward', 0) for b in out) == 32


def test_ppo_timeline_averages_slices_and_skips_empty_cells(tmp_path):
    path = tmp_path/'progress.csv'
    rows = ['train/std,train/approx_kl'] + [f'{v},' if i == 0 else f'{v},{v/10}' for i, v in enumerate(range(8))]
    path.write_text('\n'.join(rows)+'\n')
    out = rp.ppo_timeline(path, bins=2, columns=('train/std', 'train/approx_kl'))
    assert out[0]['train/std'] == pytest.approx(1.5) and out[1]['train/std'] == pytest.approx(5.5)
    assert out[0]['train/approx_kl'] == pytest.approx(.2)


def test_monitor_reading_skips_the_metadata_line(tmp_path):
    path = tmp_path/'m.csv'; path.write_text('#{"t_start": 0}\nr,l,t\n1.5,44,0.9\n2.0,1000,3.0\n')
    assert rp.read_monitor(path) == [{'r': 1.5, 'l': 44}, {'r': 2., 'l': 1000}]
