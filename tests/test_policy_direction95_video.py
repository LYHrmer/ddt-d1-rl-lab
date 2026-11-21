"""Cross-check the public D1 movie against its nine measured parameter probes."""
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tools95.score95 import RESETS, SEEDS


ROOT = Path(__file__).resolve().parents[1]


def mp4_box(data, kind):
    """Read a child box without loading a video decoder."""
    at = 0
    while at < len(data):
        size, tag = struct.unpack_from(">I4s", data, at)
        header = 8
        if size == 0:
            size = len(data) - at
        elif size == 1:
            size = struct.unpack_from(">Q", data, at + 8)[0]
            header = 16
        assert header <= size <= len(data) - at, tag
        if tag == kind:
            return data[at + header:at + size]
        at += size
    raise AssertionError(f"MP4 box missing: {kind}")


def test_published_replay_matches_native_sources_scores_and_frame_count():
    movie = ROOT / "docs/assets/policy_direction95.mp4"
    receipt = ROOT / "reports/policy_direction95_video.json"
    report = ROOT / "reports/policy_direction95.json"
    curves = ROOT / "reports/policy_direction95_curves.npz"
    products = (movie, receipt, report, curves)
    if not any(p.exists() for p in products):
        pytest.skip("policy-direction results have not been published in this checkout")
    assert all(p.is_file() for p in products)
    video = json.loads(receipt.read_bytes())
    trial = json.loads(report.read_bytes())
    media = movie.read_bytes()
    assert video["status"] == "COMPLETE" and video["file"] == movie.name
    assert 0 < len(media) == video["bytes"] <= 5_000_000
    assert sha256(media).hexdigest() == video["video_sha256"]
    assert sha256(curves.read_bytes()).hexdigest() == trial["curves_sha256"]
    assert (video["left_alpha"], video["right_alpha"], video["reference_alpha"]) == (-1, 1, 0)
    assert video["measured_panels"] == 18 and video["cover_frames"] == 0
    assert all(video[name] == 0 for name in ("new_physics_steps", "model_steps", "policy_calls", "training_updates"))

    cells = {(row["training_seed"], row["reset"]): row for row in trial["cells"]}
    checklist = {(seed, reset) for seed in SEEDS for reset in RESETS}
    assert len(trial["cells"]) == len(video["pairs"]) == len(checklist) == 9
    assert set(cells) == {(p["seed"], p["reset"]) for p in video["pairs"]} == checklist
    pos = 0
    with np.load(curves, allow_pickle=False) as data:
        for pair in video["pairs"]:
            seed, reset = pair["seed"], pair["reset"]
            cell = cells[seed, reset]
            for side, tag in (("left", "m1"), ("right", "p1"), ("zero", "zero")):
                arm = pair[side]
                native = cell["arms"][tag]["native"]
                folder = f"seed_{seed}/reset_{reset}/{tag}"
                assert arm["status"] == "RECORDED"
                assert arm["source"] == folder + "/native/trace.npz" and arm["receipt"] == folder + "/native.json"
                for name in ("source_sha256", "receipt_sha256"):
                    assert len(bytes.fromhex(arm[name])) == 32 and arm[name] == native[name]
                x = data[native["curve_key"]]
                assert arm["ticks"] == native["ticks"] == len(x) - 1 and x[0] == cell["origin_x_m"]
                displacement = (x - x[0]) * 1000
                assert arm["end_displacement_mm"] == pytest.approx(displacement[-1])
                assert arm["mean_absolute_drift_mm"] == pytest.approx(np.abs(displacement[1:]).mean())
                for flag, note in (("illegal", "非法接触"), ("fallen", "跌倒")):
                    assert (note in arm["stop_reason"]) == native["flags"][flag]
                assert bool(arm["stop_reason"]) == (not native["safe"])
            assert pair["first_frame"] == pos
            end = pos + max(pair["left"]["ticks"], pair["right"]["ticks"])
            assert pair["endpoint_frame"] == end
            assert pair["last_frame"] == end + video["endpoint_hold_frames"]
            pos = pair["last_frame"] + 1

    tables = memoryview(media)
    for name in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
        tables = mp4_box(tables, name)
    sample_count = struct.unpack_from(">I", mp4_box(tables, b"stsz"), 8)[0]
    avc = mp4_box(mp4_box(tables, b"stsd")[8:], b"avc1")
    stream = video["stream"]
    assert struct.unpack_from(">HH", avc, 24) == (stream["width"], stream["height"])
    assert sample_count == pos == video["frames"] == int(stream["nb_read_frames"])
    assert stream["r_frame_rate"] == "25/1" and video["fps"] == 25
    assert video["playback_speed"] == video["source_dt_s"] * video["fps"] == 0.5
    assert video["duration_s"] == pytest.approx(pos / video["fps"])
