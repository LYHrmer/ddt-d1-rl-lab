from itertools import product

import numpy as np
import pytest

from tools94.common94 import digest
from tools101.score101 import (EPISODES, LAYOUTS, SEEDS, SOURCE_SEED, TICKS, VISIBILITIES, paired,
                               score_archives, seed_gate, validate_holdout)
from tools101.video101 import clips_for
from tools101.publish101 import compare_geometric_views


def test_angular_mask_comparison_pairs_physical_captures_even_if_view_order_changes(tmp_path):
    data = dict(cloud_id=np.array([8, 9, 9, 8]), visibility=np.array(["clear", "clear", "masked", "masked"]),
                geometric_mm=np.array([30., 0., 0., np.nan]),
                valid=np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=bool))
    path = tmp_path / "heldout.npz"
    np.savez(path, **data)
    assert compare_geometric_views(path) == dict(captures=2, clear_valid_counts=[2], masked_valid_counts=[1],
                                                 identical_finite_estimates=1)
    data["cloud_id"][:] = 8
    corrupt = tmp_path / "duplicated.npz"
    np.savez(corrupt, **data)
    with pytest.raises(ValueError, match="one-to-one"):
        compare_geometric_views(corrupt)


def registered_rows():
    keys = list(product(EPISODES, LAYOUTS, VISIBILITIES, TICKS))
    episode, layout, vis, tick = map(np.array, zip(*keys, strict=True))
    side = np.where(layout == LAYOUTS[0], "left", "right")
    truth = np.where(tick == TICKS[4], np.where(side == "left", 30., -30.), 0.)
    cloud_ids = {key: index for index, key in enumerate(product(EPISODES, LAYOUTS, TICKS))}
    series_ids = {key: index for index, key in enumerate(product(EPISODES, LAYOUTS, VISIBILITIES))}
    cloud = [cloud_ids[e, l, t] for e, l, _, t in keys]
    series = [series_ids[e, l, v] for e, l, v, _ in keys]
    return {"row_id": np.arange(len(keys)), "source_seed": np.full(len(keys), SOURCE_SEED), "episode": episode,
            "layout": layout, "side": side, "visibility": vis, "capture_tick": tick, "delivery_tick": tick+2,
            "cloud_id": np.array(cloud), "series_id": np.array(series), "target_mm": truth, "geometric_mm": truth.copy()}


def test_equal_mae_missing_sides_and_magnitude_are_distinct():
    truth = np.array([0., 30., -30., 30., 30., 0.])
    base = np.array([0., 30., -30., 30., 30., 0.])
    pred = np.array([11., np.nan, 15., 5., 18., 0.])
    stats = paired(truth, base, pred)
    assert stats["both_available"] == 5
    assert stats["baseline_mae_both_mm"] == 0
    assert stats["predictive_mae_both_mm"] == (11+45+25+12)/5
    assert stats["predictive"]["raised_missed_side"] == 3
    assert stats["predictive"]["raised_wrong_side"] == 1
    assert stats["predictive"]["raised_called_flat"] == 1
    assert stats["predictive"]["raised_missing"] == 1
    assert stats["predictive"]["raised_correct_side_bad_magnitude"] == 1
    assert stats["predictive"]["flat_false_positives"] == 1
    assert stats["regressed"] == 5
    same = paired(truth, base, base)
    checks = seed_gate(dict.fromkeys(VISIBILITIES, same), {v: {"neutral": 1, "left": 1, "right": 1} for v in VISIBILITIES})
    assert checks["clear_paired_mae"] and not checks["masked_paired_mae"]


def test_full_grid_rejects_duplicate_missing_and_visibility_label_leakage():
    rows = registered_rows()
    validate_holdout(rows)
    with pytest.raises(ValueError):
        validate_holdout({k: a[:-1] for k, a in rows.items()})
    bad = {k: a.copy() for k, a in rows.items()}
    bad["capture_tick"][1], bad["delivery_tick"][1] = bad["capture_tick"][0], bad["delivery_tick"][0]
    with pytest.raises(ValueError, match="duplicate"):
        validate_holdout(bad)
    rows["target_mm"][0] = 30
    with pytest.raises(ValueError, match="visibility"):
        validate_holdout(rows)


def test_clips_include_all_three_seeds_and_first_registered_regression():
    rows = registered_rows()
    truth = rows["target_mm"]
    b = np.tile(truth, (3, 1))
    p = b.copy()
    p[0, 200] = -30 if truth[200] == 0 else 0.
    p[2, 5] = -30 if truth[5] == 0 else 0.
    clips = clips_for(rows, SEEDS, b, p)
    assert [(c.model, len(c.rows)) for c in clips[:-1]] == [(k, 16) for k in range(3) for _ in range(4)]
    assert all(np.unique(rows["episode"][c.rows]).tolist() == [EPISODES[0]] for c in clips[:-1])
    assert clips[-1].model == 0 and clips[-1].rows[clips[-1].regression] == 200


def test_public_score_rebuilds_sensor_comparator_and_physical_reference(tmp_path):
    rows = registered_rows()
    n = len(rows["row_id"])
    ranges = np.zeros((n, 520))
    ranges[:, (0, 260)] = .5
    az, depression = np.deg2rad([-24, 33])
    offset_xy = .5 * np.cos(depression) * np.array([np.cos(az), np.sin(az)])
    queries = np.array([[.355, .19], [.355, -.19]]) + offset_xy
    world = np.zeros((n, 2, 2))
    world[rows["target_mm"] > 0, 0] = [.785, .220]
    world[rows["target_mm"] < 0, 1] = [.785, -.220]
    rows.update(geometry_depth_m64=ranges, depth_m=ranges.astype(np.float32), valid=ranges > 0,
                capture_body_to_level=np.broadcast_to(np.eye(3), (n, 3, 3)), query_xy=np.tile(queries[None], (n, 1, 1)),
                world_query_xy=world, geometric_mm=np.zeros(n))
    holdout, pred = tmp_path / "holdout.npz", tmp_path / "pred.npz"
    estimates = np.tile(rows["target_mm"], (len(SEEDS), 1))
    def persist():
        np.savez_compressed(holdout, **rows)
        np.savez_compressed(pred, holdout_sha256=digest(holdout), seeds=SEEDS, baseline_mm=estimates, predictive_mm=estimates)
    persist()
    report = score_archives(holdout, pred)
    assert report["geometry_readback"]["geometric_estimates_verified"] == n
    assert report["passing_seeds"] == 0
    assert report["geometric_reference"]["masked"]["raised_missed_side"] == len(EPISODES) * len(LAYOUTS)
    world[0, 0] = [.785, .220]
    persist()
    with pytest.raises(ValueError, match="physical query"):
        score_archives(holdout, pred)
