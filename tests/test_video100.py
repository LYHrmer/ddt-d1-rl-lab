"""Check the D1 top-view comparison from the same delivered ranges as the official estimator."""
from itertools import product
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from test_policy_direction95_video import mp4_box
from tools100 import video100 as v
from tools100.score100 import score_rows


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


@pytest.fixture
def recording(tmp_path):
    protocol = v.PROTOCOL
    plan = json.loads(protocol.read_bytes())
    directions = plan["budget"]["directions_per_cloud"]
    sparse = np.arange(8, dtype=np.int64)
    keep = np.ones(directions, dtype=bool)
    keep[0] = False
    points, query, seeds, layouts, ticks, rows = [], [], [], [], [], []
    for s, l, t in product(plan["seeds"], plan["layouts"], plan["capture_ticks"]):
        target = 30. if l == "left" else -30. if l == "right" else 0.
        packed = np.full((directions, 3), np.nan)
        # Sparse returns miss a small bump under the queries. Dense-only returns land on both queries.
        packed[0], packed[1] = (.77, .2, -.4), (.77, -.2, -.4)
        packed[8], packed[9] = (.7, .2, -.4 + max(target, 0.)/1000), (.7, -.2, -.4 - min(target, 0.)/1000)
        points.append(packed)
        query.append([[.7, .2], [.7, -.2]])
        seeds.append(s)
        layouts.append(l)
        ticks.append(t)
        for vis in plan["visibilities"]:
            rows.append(dict(seed=s, layout=l, visibility=vis, capture_tick=t, delivery_tick=t+2, truth_mm=target,
                             sparse_mm=0. if vis == "clear" else None, dense_mm=target))
    archive, report = tmp_path / "clouds.npz", tmp_path / "report.json"
    np.savez_compressed(archive, points=points, query_xy=query, cloud_seed=seeds, cloud_layout=layouts,
                        capture_tick=ticks, sparse_indices=sparse, angle_keep=keep)
    write_json(report, dict(status="COMPLETE", cloud_sha256=v.digest(archive), protocol_sha256=v.digest(protocol),
                            rows=rows, score=score_rows(rows, plan)))
    return archive, report


def test_both_arms_same_packet_wheels_and_scale_with_missing_side(recording):
    cases = v.load_cases(*recording)
    assert [(c.seed, c.layout, c.visibility) for c in cases] == list(
        product((87201, 87202, 87203), ("left", "right", "flat"), ("masked", "clear"))
    )
    assert all(len(c.packets) == 12 for c in cases)
    masked, clear = cases[:2]
    for p, q in zip(masked.packets, clear.packets, strict=True):
        assert p.capture == q.capture == p.delivery-2 and p.truth == q.truth == 30.
        assert np.array_equal(p.query, q.query) and np.allclose(v.wheel_origin(p.query), [.35, 0.])
        assert p.estimates == (None, 30.) and q.estimates == (0., 30.)
        assert p.summary()["passed"] == [False, True] and q.summary()["passed"] == [False, True]
        assert p.summary()["points_shown"] == [1, 3] and q.summary()["points_shown"] == [2, 4]
    flat = next(c for c in cases if c.layout == "flat" and c.visibility == "clear")
    assert flat.packets[0].summary()["passed"] == [True, True]
    assert v.side_text(None) == "无估计" and "齐平" in v.side_text(-10.) and "右高" in v.side_text(-10.001)
    near, far, half = v.common_extent(cases)
    assert near <= 0 < .35 < far and half >= .2
    joined = np.array([[.5, .1, 0], [.5, -.1, .03], [.5, .149, 0], [.5, .151, 0]])
    assert v.nearest_mask(joined, [.5,0]).tolist() == [True, True, False, False]
    assert not v.nearest_mask(joined, [.8,0]).any()


def test_one_frame_contains_both_measurements_and_failure_feedback(recording):
    pytest.importorskip("PIL.Image")
    try:
        fonts = v.chinese_font(30, True), v.chinese_font(23, True), v.chinese_font(17)
    except OSError:
        pytest.skip("Noto CJK fonts unavailable on the test runner")
    cases = v.load_cases(*recording)
    image = v.frame(cases[0], 0, 1, 18, v.common_extent(cases), [0, 12], fonts)
    assert image.size == (v.WIDTH, v.HEIGHT)
    from PIL import ImageColor
    pixels = np.asarray(image)
    for color in (v.GOOD, v.BAD, v.TARGET):
        assert (pixels == ImageColor.getrgb(color)).all(axis=-1).sum() > 10


def test_different_report_partial_packet_grid_or_different_predictions_rejected(recording, tmp_path):
    archive, report = recording
    source = json.loads(report.read_bytes())
    for override, error in [(dict(status="PARTIAL"), "incomplete input"), (dict(cloud_sha256="0"*64), "cloud checksum"),
                            (dict(rows=source["rows"][:-1]), "missing, duplicate")]:
        write_json(report, {**source, **override})
        with pytest.raises(ValueError, match=error):
            v.load_cases(archive, report)
    source["rows"][0]["dense_mm"] = 8.
    source["score"] = score_rows(source["rows"], json.loads(v.PROTOCOL.read_bytes()))
    write_json(report, source)
    receipt = tmp_path/"movie.json"
    movie = tmp_path/"movie.mp4"
    with pytest.raises(ValueError, match="estimate differs from recorded returns"):
        v.render(archive, report, movie, receipt)
    assert json.loads(receipt.read_bytes())["status"] == "PARTIAL" and not movie.exists()
    with pytest.raises(FileExistsError):
        v.render(archive, report, movie, receipt)


def test_public_video_every_shown_packet_and_mp4_sample_table():
    root = Path(__file__).resolve().parents[1]
    movie = root/"docs/assets/pointcloud100.mp4"
    receipt = root/"reports/pointcloud100_video.json"
    archive, report = root/"reports/pointcloud100_clouds.npz", root/"reports/pointcloud100.json"
    if not any(p.exists() for p in (movie, receipt, archive, report)):
        pytest.skip("the point-cloud replay and movie have not been published")
    video = json.loads(receipt.read_bytes())
    assert video["status"] == "COMPLETE" and video["file"] == movie.name
    assert (video["protocol_sha256"], video["report_sha256"], video["cloud_sha256"], video["video_sha256"]) == (
        v.digest(v.PROTOCOL), v.digest(report), v.digest(archive), v.digest(movie))
    assert 0 < movie.stat().st_size == video["bytes"] <= 5_000_000
    assert all(video[k] == 0 for k in ("cover_frames", "new_rays", "new_physics_steps", "policy_updates",
                                        "world_updates", "perception_updates"))
    cases = v.load_cases(archive, report)
    assert len(cases) == len(video["cases"]) == 18 and video["shared_extent_m"] == list(v.common_extent(cases))
    next_frame = 0
    for logged, c in zip(video["cases"], cases, strict=True):
        assert (logged["seed"], logged["layout"], logged["visibility"]) == (c.seed, c.layout, c.visibility)
        assert logged["packets"] == [p.summary() for p in c.packets]
        assert logged["passed_counts"] == [sum(p["passed"][i] for p in logged["packets"]) for i in (0, 1)]
        assert len(c.packets) == 12 and logged["first_frame"] == next_frame
        next_frame += len(c.packets)*video["frames_per_packet"] + video["endpoint_hold_frames"]
        assert logged["last_frame"] == next_frame-1
    table = memoryview(movie.read_bytes())
    for kind in (b"moov", b"trak", b"mdia", b"minf", b"stbl"):
        table = mp4_box(table, kind)
    frames = struct.unpack_from(">I", mp4_box(table,b"stsz"), 8)[0]
    avc = mp4_box(mp4_box(table,b"stsd")[8:], b"avc1")
    assert struct.unpack_from(">HH", avc, 24) == (v.WIDTH, v.HEIGHT)
    assert frames == next_frame == video["frames"] == int(video["stream"]["nb_read_frames"])
    assert video["duration_s"] == pytest.approx(frames / v.FPS)
