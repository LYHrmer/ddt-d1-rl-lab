"""No robot, renderer, inference, or physics: reject incomplete capture inputs."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools35 import record_cpp_showcase35 as record
from tools35 import render_showcase35 as render


def test_state_capture_copies_synchronized_post_step_state():
    class Environment:
        data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(1), time=0.)

        def reset(self, **kwargs):
            return "observation", {}

        def step(self, action):
            self.data.qpos[:] = [1., 2.]
            self.data.qvel[:] = [3.]
            self.data.time = .02
            return "next", 0., False, False, {}

    wrapped = record.StateCapture(Environment())
    wrapped.reset(seed=1001)
    wrapped.step(None)
    assert np.array_equal(wrapped.rows[0][0], [0., 0.])
    assert np.array_equal(wrapped.rows[1][0], [1., 2.])
    assert np.array_equal(wrapped.rows[1][1], [3.])
    assert wrapped.rows[1][2] == .02
    assert not wrapped.inflight


def test_interrupted_native_step_keeps_unknown_total_and_completed_prefix(tmp_path):
    class Environment:
        data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(1), time=0.)

        def __init__(self):
            self.count = 0

        def counters(self):
            return {"native_steps": self.count}

        def reset(self, **kwargs):
            return None, {}

        def step(self, action):
            self.count += 1
            raise InterruptedError("synthetic native-step interruption")

    with (tmp_path / "states.jsonl").open("x") as stream:
        wrapped = record.StateCapture(Environment(), stream)
        wrapped.reset()
        with pytest.raises(InterruptedError):
            wrapped.step(None)
        assert wrapped.inflight
        assert wrapped.known_counts == {"native_steps": 0}
        assert wrapped.counters() == {"native_steps": 1}
    assert len((tmp_path / "states.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("rows,offset", [(600, 0.), (601, .001)])
def test_partial_or_shifted_state_clock_is_refused(rows, offset):
    with pytest.raises(ValueError):
        record.validate_complete_states(np.zeros((rows, 2)), np.zeros((rows, 1)),
                                        np.arange(rows) * .02 + offset, 2, 1)


def fixture_recording(tmp_path):
    folder = tmp_path / "case"
    folder.mkdir()
    for name in ("model.mjb", "states.npz"):
        (folder / name).write_bytes(b"fixture")
    manifest = {"format": "mujoco-state-showcase-v1", "status": "COMPLETE", "states": 601,
                "duration_s": 12., "new_qualification_claimed": False,
                "artifacts_sha256": {name: record.sha(folder / name) for name in ("model.mjb", "states.npz")}}
    report = {"status": "COMPLETE", "cleanup_complete": True, "error": None}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "report.json").write_text(json.dumps(report))
    (tmp_path / "report.json").write_text(json.dumps(report))
    return folder


def test_complete_case_from_partial_attempt_cannot_be_rendered(tmp_path):
    folder = fixture_recording(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({
        "status": "INCOMPLETE", "cleanup_complete": True, "error": "other case failed"}))
    with pytest.raises(ValueError, match="complete closed"):
        render.load_recording(folder)


def test_modified_recording_is_refused(tmp_path):
    folder = fixture_recording(tmp_path)
    (folder / "states.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        render.load_recording(folder)


def test_frame_selection_contains_start_motion_and_final_stop():
    indices = render.frame_indices(np.arange(601) * .02)
    assert indices[0] == 0 and indices[-1] == 600
    assert 100 in indices and 250 in indices and 400 in indices
    assert len(indices) == 241
    assert max(abs(index * .02 - frame * .05) for frame, index in enumerate(indices)) <= .0100000001
