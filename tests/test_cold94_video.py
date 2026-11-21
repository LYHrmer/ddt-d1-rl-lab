"""Check which physical motions and endpoints can appear in the cold-start replay."""
import json

import numpy as np
import pytest

from tools94.video94 import _source_metadata, load_pairs, run


def recording(path, offset, ticks, *, fallen=False, x0=1.10, reset=948101, horizon=50, exploration=-1):
    states = np.zeros((ticks + 1, 45))
    states[:, 0] = x0 - .002 * np.arange(ticks + 1)
    states[:, 3] = 1.
    flags = np.zeros(ticks, dtype=bool)
    flags[-1] = fallen
    clean = np.zeros(ticks, dtype=bool)
    np.savez(path, state45=states, fallen=flags, illegal=clean, terminated=flags, truncated=clean,
             reset_seed=reset, requested_ticks=horizon, explore_seed=exploration, wheel_offset_rad_s=offset)


def manifest(directory, **overrides):
    row = dict(seed=94011, reset=948101, round=1, kind="calibration", left="neg.npz", right="pos.npz",
               status="MODEL_GATE_FAILED")
    row.update(overrides)
    path = directory / "manifest.json"
    path.write_text(json.dumps({"pairs": [row]}), encoding="utf-8")
    return path


def test_dual_timeline_keeps_early_terminal_state(tmp_path):
    recording(tmp_path / "neg.npz", -.20, 6, fallen=True)
    recording(tmp_path / "pos.npz", .20, 50)
    pair, = load_pairs(manifest(tmp_path))
    assert pair.last_tick == 50
    left, right = (_source_metadata(a) for a in pair.arms)
    assert left["ticks"] == 6 and left["stop_reason"] == "跌倒"
    assert left["first_second_rollback_mm"] is None
    assert right["first_second_rollback_mm"] == pytest.approx(100.)


@pytest.mark.parametrize("change, error", [({"left": "pos.npz"}, "wheel intervention"),
                                            ({"reset": 948102}, "reset"),
                                            ({"kind": "policy"}, "registered policy group"),
                                            ({"left_sha256": "0" * 64}, "hash")])
def test_reject_mislabelled_calibration(tmp_path, change, error):
    recording(tmp_path / "neg.npz", -.20, 50)
    recording(tmp_path / "pos.npz", .20, 50)
    with pytest.raises(ValueError, match=error):
        load_pairs(manifest(tmp_path, **change))


def test_distinct_starts_preserve_failure_without_encoding(tmp_path):
    recording(tmp_path / "neg.npz", -.20, 50)
    recording(tmp_path / "pos.npz", .20, 50, x0=1.11)
    path, out = manifest(tmp_path), tmp_path / "fresh"
    with pytest.raises(ValueError, match="initial state"):
        run(path, out)
    result = json.loads((out / "video.json").read_bytes())
    assert result["status"] == "PARTIAL" and result["frames"] == result["cover_frames"] == 0
    assert not (out / "video.mp4").exists()
    with pytest.raises(FileExistsError):
        run(path, out)


def test_policy_labels_require_held_out_episodes_without_exploration(tmp_path):
    for filename in ("neg.npz", "pos.npz"):
        recording(tmp_path / filename, 0., 100, reset=949002, horizon=100)
    path = manifest(tmp_path, reset=949002, round=2, kind="policy", status="EVALUATED")
    pair, = load_pairs(path)
    assert pair.last_tick == 100 and pair.kind == "policy"
    assert _source_metadata(pair.arms[0])["first_second_rollback_mm"] == pytest.approx(100.)
    recording(tmp_path / "neg.npz", 0., 100, reset=949002, horizon=100, exploration=94011)
    with pytest.raises(ValueError, match="deterministic"):
        load_pairs(path)
