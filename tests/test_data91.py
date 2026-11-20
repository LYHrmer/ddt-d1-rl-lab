"""Test physical-step alignment and sealed branch preparation without simulator or model fits."""
import hashlib

import numpy as np
import pytest

from tools91 import data91
from training.d1_rl.interface import DEFAULT_Q


def rollout():
    ticks = np.array([1, 2, 3, 1, 2, 1, 2])
    n = len(ticks)
    issued = np.tile(np.arange(1, n + 1, dtype=np.float32)[:, None] / 10, (1, 16))
    applied = np.zeros_like(issued)
    applied[ticks > 1] = issued[np.flatnonzero(ticks > 1) - 1]
    return dict(qpos=np.tile(np.arange(n, dtype=float)[:, None], (1, 23)),
                qvel=np.tile(-np.arange(n, dtype=float)[:, None], (1, 22)),
                issued=issued, applied=applied, episode_tick=ticks, terrain=np.array([0, 0, 0, 1, 1, 2, 2]),
                terminated=np.arange(n) == 4, truncated=np.arange(n) == 2)


def test_fit_pairs_keep_terminal_interval_exclude_reset_edges_and_drop_unfinished_tail():
    rows = rollout()
    parts, episodes, tail = data91._episode_pairs(rows, 8)
    assert [e["id"] for e in episodes] == [8, 9] and tail == 2
    assert not episodes[0]["terminated"] and episodes[0]["truncated"]
    assert episodes[1]["terminated"] and not episodes[1]["truncated"]
    a = {key: np.concatenate([chunk[key] for chunk in parts]) for key in parts[0]}
    np.testing.assert_array_equal(a["episode"], [8, 8, 9])
    np.testing.assert_array_equal(a["tick"], [1, 2, 1])
    np.testing.assert_array_equal(a["s"][:, 0], [0, 1, 3])
    np.testing.assert_array_equal(a["target"][:, 0], [1, 2, 4])
    np.testing.assert_array_equal(a["target"][:, 23], [-1, -2, -4])
    np.testing.assert_array_equal(a["a"], rows["applied"][[1, 2, 4]])
    np.testing.assert_array_equal(a["done"], [False, True, True])


@pytest.mark.parametrize("column,row,value", [
    ("episode_tick", 2, 4), ("terrain", 1, 1), ("applied", 4, 0.), ("truncated", 2, False),
])
def test_fit_pairs_reject_corrupt_clock_terrain_fifo_or_terminal(column, row, value):
    data = rollout()
    data[column][row] = value
    with pytest.raises(ValueError):
        data91._episode_pairs(data, 0)


def test_moving_start_replays_last_issue_and_exposes_only_the_cut(tmp_path):
    q = np.zeros((4, 23))
    q[:, 3], q[:, 7:] = 1., DEFAULT_Q
    v = np.zeros((4, 22))
    v[:, 3] = [0., .8, -.2, np.nan]  # The fourth endpoint lies after the branch.
    issued = np.zeros((4, 16), dtype=np.float32)
    issued[1:3, 0] = [.25, -.125]
    applied = np.vstack((issued[0], issued[:-1]))
    archive = tmp_path / "trace.npz"
    np.savez_compressed(archive, qpos=q, qvel=v, issued_action=issued, applied_action=applied,
                        raw_cpp_action=issued, time_s=[0., .02, .04, np.nan], command_vx_m_s=np.zeros(4))

    def predict(body, packet):
        assert body.dtype == np.float32 and not packet.any()
        prior = body.reshape(11, 57)
        assert prior[0, 41] == issued[1, 0] and not prior[1:, 41:].any()
        return issued[2]

    start, error = data91._moving_start(archive, 2, predict, 1e-5)
    assert error == 0.
    np.testing.assert_array_equal(start["state45"], np.r_[q[2], v[2]])
    np.testing.assert_array_equal(start["fifo16"], issued[2])
    np.testing.assert_allclose(start["previous_filtered"], [.97 * -.2 + .03 * (.97 * .8), 0., 0.])
    frames = start["observation627"].reshape(11, 57)
    assert frames[0, 41] == issued[2, 0] and frames[-1, 41] == issued[1, 0]
    assert not frames[1:-1, 41:].any()
    with pytest.raises(ValueError, match="parent action disagreement"):
        data91._moving_start(archive, 2, lambda *_: np.ones(16), 1e-5)


def test_standing_start_requires_only_initial_members_and_verifies_history_digest(tmp_path):
    q = np.r_[[1.10, 0., .45, 1., 0., 0., 0.], DEFAULT_Q]
    v = np.zeros(22)
    obs = np.zeros(643, np.float32)
    obs[:627].reshape(11, 57)[:, 5] = -1.
    file = tmp_path / "trace.npz"
    np.savez_compressed(file, initial_qpos=q, initial_qvel=v, initial_obs=obs)
    digest = hashlib.sha256(q.tobytes() + v.tobytes() + obs.tobytes()).hexdigest()
    start = data91._standing_start(file, digest)
    np.testing.assert_array_equal(start["state45"], np.r_[q, v])
    np.testing.assert_array_equal(start["observation627"], obs[:627])
    assert not start["previous_filtered"].any() and not start["fifo16"].any()
    with pytest.raises(ValueError, match="reset identity"):
        data91._standing_start(file, "other")
    obs[57 + 5] = -.5
    np.savez_compressed(file, initial_qpos=q, initial_qvel=v, initial_obs=obs)
    digest = hashlib.sha256(q.tobytes() + v.tobytes() + obs.tobytes()).hexdigest()
    with pytest.raises(ValueError, match="reset actor history"):
        data91._standing_start(file, digest)


def test_starts_refuse_to_reuse_an_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        data91.prepare_starts(tmp_path)
