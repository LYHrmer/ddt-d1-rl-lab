"""Synthetic body-response admission tests; no source assets, models or physics."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def lib():
    for path in (ROOT, ROOT / '.python-deps'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    np = pytest.importorskip('numpy')
    pytest.importorskip('torch')
    from tools68 import labels68
    return SimpleNamespace(np=np, mod=labels68)


def states(lib, slopes=(-1., -2.)):
    np = lib.np
    values = np.zeros((10, 45), dtype=np.float64)
    values[:, 2], values[:, 3] = .5, 1.
    values[:, 23:25] = (.3, -.2)
    for index, epsilon in enumerate(lib.mod.EPSILONS):
        values[2 + 2 * index, 23:25] += epsilon * np.array(slopes)
        values[3 + 2 * index, 23:25] -= epsilon * np.array(slopes)
        values[2 + 2 * index, 28], values[3 + 2 * index, 28] = .1 * epsilon, -.1 * epsilon
    return values


def all_points(lib, *, slopes=(-1., -2.)):
    rows, arrays = [], []
    for source in lib.mod.SOURCES:
        for seed in lib.mod.SEEDS:
            for tick in lib.mod.TICKS:
                channels, response = lib.mod.body_response(states(lib, slopes))
                rows.append({'source': source, 'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'channels': channels})
                arrays.append(response)
    return rows, arrays


def test_body_transform_uses_inverse_wxyz_rotation_and_normalizes_quaternion(lib):
    np, mod = lib.np, lib.mod
    raw = np.zeros((3, 45), dtype=np.float64)
    raw[:, 23:26] = [1., 0., 0.]
    raw[0, 3] = 1.
    raw[1, 3:7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    raw[2, 3:7] = 7 * raw[1, 3:7]
    actual = mod.body_velocity(raw)
    np.testing.assert_allclose(actual, [[1., 0.], [0., -1.], [0., -1.]], rtol=0, atol=1e-14)
    assert actual.dtype == np.float64


def test_sub_float32_poststate_difference_survives_body_transform(lib):
    np, mod = lib.np, lib.mod
    raw = states(lib, (0., 0.))
    raw[:, 23] = 1.
    raw[2, 23], raw[3, 23] = 1. + 2. ** -36, 1. - 2. ** -36
    assert np.float32(raw[2, 23]) == np.float32(raw[3, 23])
    _, arrays = mod.body_response(raw)
    assert arrays['central_body_vxy_fd'][0, 0] == 2. ** -36 / mod.EPSILONS[0]
    assert arrays['central_body_vxy_fd'][0, 0] != 0.


def test_transform_before_difference_retains_quaternion_contribution(lib):
    np, mod = lib.np, lib.mod
    raw = states(lib, (0., 0.))
    raw[:, 23:26] = [1., 0., 0.]
    for index, epsilon in enumerate(mod.EPSILONS):
        for order, angle in ((2 + 2 * index, 2 * epsilon), (3 + 2 * index, -2 * epsilon)):
            raw[order, 3:7] = [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]
    channels, arrays = mod.body_response(raw)
    assert np.count_nonzero(raw[2, 23:26] - raw[3, 23:26]) == 0
    np.testing.assert_allclose(arrays['central_body_vxy_fd'][:, 1], -2., rtol=1e-8, atol=1e-10)
    assert channels['vy']['stability']['stable']
    assert not channels['vx']['stability']['stable']


@pytest.mark.parametrize('qualified', [48, 47])
def test_admission_requires_every_one_of_48_state_channel_qualifications(lib, qualified):
    rows, _ = all_points(lib)
    if qualified == 47:
        rows[-1]['channels']['vy']['stability']['stable'] = False
    result = lib.mod.qualification(rows)
    assert result['qualified_state_channels'] == qualified
    assert result['admitted'] is (qualified == 48)
    assert result['status'] == ('ADMITTED' if qualified == 48 else 'REJECTED_INCOMPLETE_BODY_RESPONSE_QUALIFICATION')
    assert result['rows_dropped'] == result['fits_authorized_by_this_result'] == 0


def test_negative_and_unknown_channels_remain_in_all96_labels(lib):
    np, mod = lib.np, lib.mod
    points, responses = all_points(lib, slopes=(-1., 0.))
    labels = {}
    for source_index, source in enumerate(mod.SOURCES):
        s = np.zeros((48, 45), dtype=np.float64)
        s[:, 0] = source_index * 12 + np.repeat(np.arange(12), 4)
        s[:, 3] = 1.
        epsilon = np.tile(mod.EPSILONS, 12)
        direction = np.zeros(16)
        direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
        plus = (epsilon[:, None] * direction).astype(np.float32)
        minus = -plus
        labels[source] = {'s': s, 'a0': np.zeros((48, 16), np.float32), 'a_plus': plus, 'a_minus': minus,
            'v': ((plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon[:, None])).astype(np.float32),
            'target': np.full(48, -.2, np.float64), 'epsilon': epsilon,
            'seed': np.repeat(mod.SEEDS, 12), 'tick': np.tile(np.repeat(mod.TICKS, 4), 4)}
    assembled = mod.assemble_labels(labels, points, responses)
    assert assembled['target_body'].shape == (96, 2) and assembled['target_body'].dtype == np.float64
    assert (assembled['target_body'][:, 0] < 0).all() and (assembled['target_body'][:, 1] == 0).all()
    assert assembled['body_channel_qualified'][:, 0].all() and not assembled['body_channel_qualified'][:, 1].any()
    assert mod.qualification(points)['qualified_state_channels'] == 24
    np.testing.assert_array_equal(assembled['source_id'], np.repeat([58, 62], 48))
    np.testing.assert_array_equal(assembled['epsilon'], np.tile(mod.EPSILONS, 24))
    assert len(np.unique(assembled['s'], axis=0)) == 24


def test_missing_or_partial_state_cannot_be_qualified_as_complete(lib):
    rows, _ = all_points(lib)
    with pytest.raises(ValueError, match='all24 complete'):
        lib.mod.qualification(rows[:-1])
    rows[3]['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError, match='all24 complete'):
        lib.mod.qualification(rows)


@pytest.mark.parametrize('invalid', ['float32', 'zero_quaternion', 'different_zero'])
def test_body_response_rejects_lost_precision_or_invalid_baseline(lib, invalid):
    raw = states(lib)
    if invalid == 'float32':
        raw = raw.astype(lib.np.float32)
    elif invalid == 'zero_quaternion':
        raw[:, 3:7] = 0.
    else:
        raw[1, 23] += .01
    with pytest.raises(ValueError):
        lib.mod.body_response(raw)


def test_read_point_recovers_original_yaw_before_creating_body_targets(lib, tmp_path):
    np, mod = lib.np, lib.mod
    raw = states(lib)
    for order in range(10):
        np.savez(tmp_path / f'branch_00_{order:02d}.npz', s=raw[:1], applied_action=np.zeros((1, 16), np.float32),
                 target=raw[order:order + 1])
    central, forward, backward = [], [], []
    for index, epsilon in enumerate(mod.EPSILONS):
        plus, minus = raw[2 + 2 * index:4 + 2 * index]
        central.append((plus - minus) / (2 * epsilon))
        forward.append((plus - raw[0]) / epsilon)
        backward.append((raw[0] - minus) / epsilon)
    arrays = {key: np.asarray(value) for key, value in (
        ('central_state45_fd', central), ('forward_state45_fd', forward), ('backward_state45_fd', backward))}
    np.savez(tmp_path / 'response_00.npz', **arrays)
    old = {'seed': 53103, 'tick': 110, 'yaw28': [
        {'epsilon': epsilon, 'central': central[index][28], 'forward': forward[index][28], 'backward': backward[index][28]}
        for index, epsilon in enumerate(mod.EPSILONS)]}
    yaw = {'target': arrays['central_state45_fd'][:, 28]}
    row, output = mod.read_point(tmp_path, '58', 0, yaw, old)
    assert row['old_yaw_response_exact'] and not row['old_yaw_qualification_used_for_body']
    np.testing.assert_array_equal(output['post_state45'], raw)
    wrong = copy.deepcopy(old)
    wrong['yaw28'][0]['central'] += .01
    with pytest.raises(ValueError, match='original yaw JSON differs'):
        mod.read_point(tmp_path, '58', 0, yaw, wrong)
