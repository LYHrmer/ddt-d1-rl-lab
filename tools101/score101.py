"""Score paired predictions on the registered new arrivals, using only NumPy and the public records."""
from __future__ import annotations

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import json
import math

import numpy as np

from tools94.common94 import ROOT, digest, read
from tools100.readback100 import expected_indices, independent_estimate, reference_delta_mm
from tools100.score100 import TOLERANCE_MM, grade
from tools101.common101 import PUBLIC, protocol


SEEDS = (101011, 101023, 101037)
SOURCE_SEED = 89037
EPISODES = (0, 3, 6, 9, 12)
LAYOUTS = ("unseen_left", "unseen_right")
VISIBILITIES = ("clear", "masked")
TICKS = (25, 34, 42, 50, 59, 67, 75, 84, 92, 100, 109, 117, 125, 134, 142, 150)
METADATA = ("row_id", "source_seed", "episode", "layout", "side", "visibility", "capture_tick",
            "delivery_tick", "cloud_id", "series_id", "target_mm", "geometric_mm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_holdout(rows):
    """Reject even a shared omission of a packet, history, condition or entire trajectory."""
    expected = set(product(EPISODES, LAYOUTS, VISIBILITIES, TICKS))
    n = len(expected)
    require(all(rows[k].shape == (n,) for k in METADATA), "wrong number or shape of held-out rows")
    integers = ("row_id", "source_seed", "episode", "capture_tick", "delivery_tick", "cloud_id", "series_id")
    require(all(rows[k].dtype.kind in "iu" for k in integers), "integer identities and clocks required")
    require(np.array_equal(rows["row_id"], np.arange(n)), "prediction columns must index consecutive row IDs")
    require(np.all(rows["source_seed"] == SOURCE_SEED), "unknown held-out source")
    require(np.all(rows["delivery_tick"] == rows["capture_tick"] + 2), "invalid packet delay")
    require(rows["target_mm"].dtype.kind == "f" and np.isfinite(rows["target_mm"]).all(),
            "truth must use finite millimetres")
    require(rows["geometric_mm"].dtype.kind == "f" and not np.isinf(rows["geometric_mm"]).any(),
            "invalid geometric measurement")

    grid = set()
    scene_to_cloud = {}
    series = {}
    for index in range(n):
        episode, layout, vis, tick = (rows[k][index].item() for k in ("episode", "layout", "visibility", "capture_tick"))
        key = episode, layout, vis, tick
        require(key in expected and key not in grid, f"duplicate or unregistered arrival {key!r}")
        grid.add(key)
        side = "left" if layout == "unseen_left" else "right"
        target = float(rows["target_mm"][index])
        require(rows["side"][index] == side and target in (0., 30. if side == "left" else -30.),
                f"inconsistent side or reference height at {key!r}")
        scene = episode, layout, tick
        pair = int(rows["cloud_id"][index]), target
        require(scene_to_cloud.setdefault(scene, pair) == pair, "visibility changed physical cloud or truth")
        series_no = int(rows["series_id"][index])
        require(series.setdefault((episode, layout, vis), series_no) == series_no, "sequence identity changes within episode")
    require(grid == expected, "incomplete held-out grid")
    require(len({v[0] for v in scene_to_cloud.values()}) == len(scene_to_cloud), "physical clouds must have unique IDs")
    require(len(set(series.values())) == len(series), "sequence spans reset, layout or visibility")


def audit_geometry(holdout_path, rows, plan):
    """Rebuild physical labels and the geometry-only comparator from archived range and measured pose."""
    sensor_plan = read(ROOT / plan["data"]["sensor_base"])
    geom = sensor_plan["geometry"]
    fixture, = plan["data"]["layouts"]["holdout"]
    geometry = {**geom, **fixture}
    require(tuple(plan["seeds"]) == SEEDS and tuple(plan["capture_ticks"]) == TICKS
            and tuple(plan["visibilities"]) == VISIBILITIES and tuple(plan["data"]["source_seeds"]["holdout"]) == (SOURCE_SEED,)
            and {fixture["id"] + "_left", fixture["id"] + "_right"} == set(LAYOUTS),
            "registration and audit roster disagree")
    n = len(rows["row_id"])
    with np.load(holdout_path, allow_pickle=False) as archive:
        world, query, rot = (archive[k] for k in ("world_query_xy", "query_xy", "capture_body_to_level"))
        raw, rays, valid = (archive[k] for k in ("geometry_depth_m64", "depth_m", "valid"))
    require(world.shape == query.shape == (n, 2, 2) and rot.shape == (n, 3, 3)
            and all(np.isfinite(v).all() for v in (world, query, rot))
            and np.allclose(rot @ rot.transpose(0, 2, 1), np.eye(3), atol=1e-9, rtol=0)
            and np.allclose(np.linalg.det(rot), 1., atol=1e-9, rtol=0), "invalid pose or query coordinates")
    require(raw.shape == rays.shape == valid.shape == (n, 520) and raw.dtype == np.float64
            and rays.dtype == np.float32 and valid.dtype == bool and np.array_equal(raw.astype(np.float32), rays)
            and np.all(raw[~valid] == 0) and np.isfinite(raw).all()
            and np.all((raw[valid] >= geom["range_min_m"]) & (raw[valid] <= geom["range_max_m"])),
            "measurements differ from the exact first-return audit archive")
    az, dep = np.meshgrid(np.deg2rad(geom["azimuth_deg"]), np.deg2rad(geom["depression_deg"]))
    rays_body = np.column_stack(((np.cos(dep)*np.cos(az)).ravel(), (np.cos(dep)*np.sin(az)).ravel(), -np.sin(dep).ravel()))
    rays_body = np.tile(rays_body, (2, 1))
    mounts = np.repeat(geom["mounts_m"], len(rays_body)//2, axis=0)
    _, keep = expected_indices(geom)
    for i in range(n):
        reference = reference_delta_mm(geometry, rows["side"][i], world[i])
        require(rows["target_mm"][i] == reference, f"wheel-track label differs from physical query at row {i}")
        points = (mounts + raw[i, :, None]*rays_body) @ rot[i]
        estimate = independent_estimate(points[valid[i]], query[i], sensor_plan["estimator"])
        stored = rows["geometric_mm"][i]
        if estimate is None:
            require(np.isnan(stored), f"spurious geometric estimate at row {i}")
        else:
            delta = abs(estimate - stored)
            require(delta <= 1e-8, f"geometric readback differs at row {i}")
    for cloud in np.unique(rows["cloud_id"]):
        idx = rows["cloud_id"] == cloud
        clear, = np.flatnonzero(idx & (rows["visibility"] == "clear"))
        masked, = np.flatnonzero(idx & (rows["visibility"] == "masked"))
        require(np.array_equal(valid[masked], valid[clear] & keep)
                and np.array_equal(raw[masked], np.where(keep, raw[clear], 0.))
                and all(np.array_equal(a[clear], a[masked]) for a in (world, query, rot)),
                "visibility did not apply just the registered angular mask to one shared cloud")
    return {"physical_labels_verified": n, "geometric_estimates_verified": n, "geometric_tolerance_mm": 1e-8}


def read_inputs(holdout_path, predictions_path, expected_seeds=None):
    seeds = tuple(expected_seeds) if expected_seeds is not None else SEEDS
    require(len(seeds) == len(set(seeds)) == 3, "expected three distinct fit seeds")
    with np.load(holdout_path, allow_pickle=False) as archive:
        rows = {k: archive[k] for k in METADATA}
        valid = archive["valid"]
    validate_holdout(rows)
    with np.load(predictions_path, allow_pickle=False) as pred:
        require(pred["holdout_sha256"].shape == () and pred["holdout_sha256"].item() == digest(holdout_path),
                "prediction columns refer to a different holdout")
        require(pred["seeds"].dtype.kind in "iu" and np.array_equal(pred["seeds"], seeds), "missing or reordered fit seeds")
        baseline, predictive = pred["baseline_mm"], pred["predictive_mm"]
    require(all(a.shape == (len(seeds), len(rows["row_id"])) and a.dtype.kind == "f" for a in (baseline, predictive)),
            "expected paired, per-seed predictions in millimetres")
    require(not np.isinf(baseline).any() and not np.isinf(predictive).any(), "infinite prediction")
    require(valid.shape == (len(rows["row_id"]), 520) and valid.dtype == bool, "invalid range availability grid")
    usable = valid.any(axis=1)
    history_available = np.zeros(usable.shape, dtype=bool)
    for series in np.unique(rows["series_id"]):
        indices = np.flatnonzero(rows["series_id"] == series)
        order = indices[np.argsort(rows["delivery_tick"][indices])]
        for position, row in enumerate(order):
            history_available[row] = np.any(usable[order[max(0, position-2):position+1]])
    for arm in (baseline, predictive):
        require(np.all(np.isfinite(arm) == history_available[None, :]),
                "missing a model output with usable inputs, or producing one with an empty history")
    return rows, seeds, baseline, predictive


def errors_mm(truth, estimate, valid):
    errors = np.abs(np.asarray(truth, dtype=np.float64)[valid] - np.asarray(estimate, dtype=np.float64)[valid])
    return math.fsum(errors) / len(errors) if len(errors) else None


def qualifying(truth, estimates):
    return np.array([grade(t, e if np.isfinite(e) else None) for t, e in zip(truth, estimates, strict=True)], dtype=bool)


def arm_stats(truth, estimates, passed):
    available = np.isfinite(estimates)
    raised = np.abs(truth) > TOLERANCE_MM
    labelled_high = np.abs(estimates) > TOLERANCE_MM
    right_side = available & labelled_high & (np.signbit(truth) == np.signbit(estimates))
    return {
        "packets": len(truth), "available": int(available.sum()), "missing": int((~available).sum()),
        "passed": int(passed.sum()), "failed": int((~passed).sum()),
        "mae_available_mm": errors_mm(truth, estimates, available),
        "raised_targets": int(raised.sum()), "raised_passed": int((raised & passed).sum()),
        "raised_missed_side": int((raised & ~right_side).sum()),
        "raised_wrong_side": int((raised & available & labelled_high & ~right_side).sum()),
        "raised_called_flat": int((raised & available & ~labelled_high).sum()),
        "raised_missing": int((raised & ~available).sum()),
        "raised_correct_side_bad_magnitude": int((raised & right_side & ~passed).sum()),
        "flat_targets": int((~raised).sum()),
        "flat_false_positives": int((~raised & available & labelled_high).sum()),
    }


def paired(truth, baseline, predictive):
    require(truth.ndim == 1 and truth.shape == baseline.shape == predictive.shape, "mismatched packet columns")
    b = qualifying(truth, baseline)
    p = qualifying(truth, predictive)
    common = np.isfinite(baseline) & np.isfinite(predictive)
    return {
        "baseline": arm_stats(truth, baseline, b), "predictive": arm_stats(truth, predictive, p),
        "both_available": int(common.sum()), "baseline_mae_both_mm": errors_mm(truth, baseline, common),
        "predictive_mae_both_mm": errors_mm(truth, predictive, common),
        "gained": int((~b & p).sum()), "regressed": int((b & ~p).sum()),
        "both_passed": int((b & p).sum()), "both_failed": int((~b & ~p).sum()),
    }


def seed_gate(conditions, truth_classes):
    checks = {}
    for vis in VISIBILITIES:
        stats = conditions[vis]
        bm, pm = stats["baseline_mae_both_mm"], stats["predictive_mae_both_mm"]
        checks[f"{vis}_paired_mae"] = (pm is not None and bm is not None
                                      and (pm < bm if vis == "masked" else pm <= bm))
        checks[f"{vis}_qualified_non_decreasing"] = stats["predictive"]["passed"] >= stats["baseline"]["passed"]
        checks[f"{vis}_missing_non_increasing"] = stats["predictive"]["missing"] <= stats["baseline"]["missing"]
        checks[f"{vis}_truth_coverage"] = all(truth_classes[vis].values())
    return checks


def score_archives(holdout_path, predictions_path, expected_seeds=None):
    rows, seeds, baseline, predictive = read_inputs(holdout_path, predictions_path, expected_seeds)
    geometry_audit = audit_geometry(holdout_path, rows, protocol())
    truth = rows["target_mm"]
    masks = {v: rows["visibility"] == v for v in VISIBILITIES}
    truth_classes = {v: {"neutral": int((truth[m] == 0).sum()), "left": int((truth[m] > TOLERANCE_MM).sum()),
                         "right": int((truth[m] < -TOLERANCE_MM).sum())} for v, m in masks.items()}
    by_seed = []
    by_episode_layout = []
    for seed, b, p in zip(seeds, baseline, predictive, strict=True):
        conditions = {v: paired(truth[m], b[m], p[m]) for v, m in masks.items()}
        checks = seed_gate(conditions, truth_classes)
        by_seed.append({"seed": int(seed), "passed": all(checks.values()), "checks": checks,
                        "overall": paired(truth, b, p), **conditions})
        for episode, layout, vis in product(EPISODES, LAYOUTS, VISIBILITIES):
            idx = (rows["episode"] == episode) & (rows["layout"] == layout) & masks[vis]
            by_episode_layout.append({"seed": int(seed), "source_seed": SOURCE_SEED, "episode": episode,
                                      "layout": layout, "visibility": vis, **paired(truth[idx], b[idx], p[idx])})
    geom = rows["geometric_mm"]
    geometric_reference = {}
    for name, idx in {"overall": np.ones(len(truth), dtype=bool), **masks}.items():
        geometric_reference[name] = arm_stats(truth[idx], geom[idx], qualifying(truth[idx], geom[idx]))
    all_truth = np.tile(truth, len(seeds))
    return {
        "passed": all(item["passed"] for item in by_seed), "passing_seeds": sum(item["passed"] for item in by_seed),
        "fit_seeds": list(seeds), "native_source_seed": SOURCE_SEED, "native_episodes": len(EPISODES),
        "unique_clouds": len(np.unique(rows["cloud_id"])), "heldout_arrivals": len(truth),
        "truth_counts": truth_classes, "geometry_readback": geometry_audit,
        "overall": paired(all_truth, baseline.ravel(), predictive.ravel()),
        "by_seed": by_seed, "by_episode_layout": by_episode_layout, "geometric_reference": geometric_reference,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=PUBLIC / "holdout.npz")
    parser.add_argument("--predictions", type=Path, default=PUBLIC / "predictions.npz")
    parser.add_argument("--check", nargs="?", const=PUBLIC / "report.json", type=Path)
    args = parser.parse_args()
    score = score_archives(args.holdout, args.predictions)
    if args.check is not None:
        require(json.loads(args.check.read_text())["score"] == score, "recomputed score differs from report")
    print(json.dumps({"passed": score["passed"], "passing_seeds": score["passing_seeds"],
                      "heldout_arrivals": score["heldout_arrivals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
