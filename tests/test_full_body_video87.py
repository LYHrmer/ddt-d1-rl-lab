"""Timestamp and source-pair checks. No renderer, physics or policy calls."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools87.video87 import DT, FRAMES, read_pair, read_trace, source_tick


def write_episode(tmp_path, arm, seed, count=301, terminated=False):
    path = tmp_path / arm
    path.mkdir()
    t = np.arange(count) * DT
    np.savez(path / "trace.npz", time_s=t, qpos=np.zeros((count, 23)), qvel=np.zeros((count, 22)),
             body_vx_m_s=np.zeros(count), command_vx_m_s=np.where(t <= 1, 0, .2),
             raw_ranges16=np.zeros((count, 16)))
    info = {"arm": arm, "terrain": "up", "seed": seed, "training_seed": 812,
            "status": "COMPLETE", "protocol_sha256": "fixture",
            "full_horizon": count == 301, "terminated": terminated, "fallen": terminated,
            "trace_sha256": hashlib.sha256((path / "trace.npz").read_bytes()).hexdigest()}
    (path / "metrics.json").write_text(json.dumps(info))
    return path


def test_video_covers_six_seconds_without_extrapolating_recorded_poses(tmp_path):
    left = write_episode(tmp_path, "baseline", 811)
    right = write_episode(tmp_path, "range", 811, 82, terminated=True)
    records = read_pair(left, right)
    indices = [source_tick(i) for i in range(FRAMES)]
    assert len(indices) == 150 and indices[0] == 0 and indices[-1] == 300
    assert np.all(np.diff(indices) > 0)
    assert records[0].time[-1] == 6.0 and records[1].time[-1] == 81 * DT
    assert records[1].command[0] == 0 and records[1].command[-1] == .2


def test_video_rejects_different_evaluation_seeds(tmp_path):
    with pytest.raises(ValueError, match="evaluation seed"):
        read_pair(write_episode(tmp_path, "baseline", 811), write_episode(tmp_path, "range", 812))


def test_video_rejects_interrupted_trajectory_without_a_terminal_event(tmp_path):
    with pytest.raises(ValueError, match="end time"):
        read_trace(write_episode(tmp_path, "range", 811, 91), "range")


def test_video_rejects_modified_complete_trajectory(tmp_path):
    folder = write_episode(tmp_path, "range", 811)
    with (folder / "trace.npz").open("ab") as changed:
        changed.write(b"unexpected trailing bytes")
    with pytest.raises(ValueError, match="digest"):
        read_trace(folder, "range")
