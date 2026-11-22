from hashlib import sha256

import numpy as np
import pytest

from tools101.common101 import PUBLIC
from tools103.data103 import ACTION_SHA, TAPE_TICKS, _applied_counts, _check_actions, _packet_rows, _windows, window_inputs


def command_tapes():
    actuator = np.arange(16, dtype=np.float32) / 16
    a = (TAPE_TICKS[:, None].astype(np.float32) + actuator[None, :]) / 200
    return np.stack((a, -a))


def test_both_clocks_use_equal_length_chronological_windows_and_terminal_has_no_target():
    actions = command_tapes()
    captures = np.array([25, 34, 42])
    windows = _windows(actions, captures, 2)
    assert windows.shape == (2, 3, 2, 2, 9, 17) and windows.dtype == np.float32
    expected = [
        ((0, 0, 0), 16, 25), ((0, 1, 0), 25, 34), ((0, 0, 1), 18, 27), ((0, 1, 1), 27, 36),
        ((1, 0, 0), 26, 34), ((1, 1, 0), 34, 42), ((1, 0, 1), 28, 36), ((1, 1, 1), 36, 44),
    ]
    for (row, arm, clock), start, stop in expected:
        block = windows[:, row, arm, clock]
        length = stop - start
        np.testing.assert_array_equal(block[:, :length, :16], actions[:, start - 16:stop - 16])
        assert (block[:, :length, 16] == 1).all() and (block[:, length:] == 0).all()
    assert not windows[:, -1].any()
    with pytest.raises(ValueError, match="interval"):
        _windows(actions, np.array([25, 42]), 2)
    with pytest.raises(ValueError, match="crosses"):
        _windows(actions, np.array([24, 33]), 2)


def test_public_packet_rows_reuse_the_same_native_commands_across_mirrors_and_visibility(tmp_path):
    actions = command_tapes()
    captures = np.array([25, 34, 42])
    info = dict(episode_roster=[dict(source_seed=91, episode=2), dict(source_seed=92, episode=2)],
                layouts=["left", "right"], rows=24)
    data = dict(capture_ticks=captures, delay_ticks=2)
    index = np.repeat(np.arange(2), 12)
    stamps = np.tile(captures, 8)
    packet = dict(row_id=np.arange(24), series_id=np.repeat(np.arange(8), 3), capture_tick=stamps,
                  delivery_tick=stamps + 2, source_seed=np.array([91, 92])[index], episode=np.full(24, 2),
                  layout=np.tile(np.repeat(["left", "right"], 6), 2),
                  visibility=np.tile(np.repeat(["clear", "masked"], 3), 4),
                  action_history=actions[index[:, None], stamps[:, None] - 16 + np.array([0, 1])])
    path = tmp_path / "packets.npz"
    np.savez(path, **packet)
    episodes, frames = _packet_rows(path, data, info, actions)
    result = _windows(actions, captures, 2)[episodes, frames]
    np.testing.assert_array_equal(result[:3], result[9:12])
    np.testing.assert_array_equal(result[:3, :, :, :, :16], -result[12:15, :, :, :, :16])
    np.testing.assert_array_equal(result[:, 0, 0, :, 16], result[:, 1, 1, :, 16])
    packet["action_history"][9] = 0
    np.savez(path, **packet)
    with pytest.raises(ValueError, match="delivery history"):
        _packet_rows(path, data, info, actions)


def test_unseen_intermediate_command_detected_even_when_packet_history_still_matches(monkeypatch):
    actions = command_tapes()
    monkeypatch.setitem(ACTION_SHA, "train", sha256(actions.astype("<f4").tobytes()).hexdigest())
    _check_actions(actions, "train", 2)
    actions[0, 30 - 16, 1] += 0.01
    with pytest.raises(ValueError, match="independently recorded"):
        _check_actions(actions, "train", 2)


def test_actuator_audit_distinguishes_a_repeated_command_from_same_tick_actuation():
    issued = np.zeros((4, 16), dtype=np.float32)
    issued[:, 7] = [0., 0.25, 0.25, -0.5]
    applied = np.vstack((np.zeros_like(issued[:1]), issued[:-1]))
    rows = np.array([1, 2, 3])
    assert _applied_counts(issued, applied, rows) == (3, 1)
    with pytest.raises(ValueError, match="one-tick-delayed"):
        _applied_counts(issued, issued, rows)


def test_published_holdout_cannot_be_opened_as_training_packets(tmp_path):
    with pytest.raises(ValueError, match="wrong packet split"):
        window_inputs("train", PUBLIC / "holdout.npz", tmp_path / "never-opened.npz")
