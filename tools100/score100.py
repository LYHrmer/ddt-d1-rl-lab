"""Shared local height estimate and paired packet scoring for the D1 point-cloud replay."""

from itertools import product
import math
from numbers import Integral, Real

import numpy as np


RADIUS_M = 0.15
TIE_ATOL_M = 1e-12
TOLERANCE_MM = 10.0


def estimate_delta_mm(points_xyz_m, queries_xy_m):
    """Estimate left minus right at two queries in the same gravity-aligned frame as the returns."""
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.shape == (0,):
        points = points.reshape(0, 3)
    queries = np.asarray(queries_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("expected N by 3 XYZ returns")
    if queries.shape != (2, 2) or not np.isfinite(queries).all():
        raise ValueError("expected two finite XY queries in left, right order")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return None

    heights = []
    for query in queries:
        offset = points[:, :2] - query
        distances = np.hypot(offset[:, 0], offset[:, 1])
        nearest = float(np.min(distances))
        if nearest > RADIUS_M:
            return None
        tied = (distances <= RADIUS_M) & (distances - nearest <= TIE_ATOL_M)
        z = points[tied, 2]
        heights.append(math.fsum(z) / len(z))
    return (heights[0] - heights[1]) * 1000.0


def _finite_mm(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("expected finite millimetres or an explicit null for an unavailable estimate")
    return float(value)


def _side(mm):
    if mm > TOLERANCE_MM:
        return "left"
    if mm < -TOLERANCE_MM:
        return "right"
    return "flat"


def grade(truth_mm, estimate_mm):
    """Classify each height independently, then enforce magnitude on elevated targets."""
    truth_mm = _finite_mm(truth_mm)
    if estimate_mm is None:
        return False
    estimate_mm = _finite_mm(estimate_mm)
    target = _side(truth_mm)
    return _side(estimate_mm) == target and (
        target == "flat" or abs(estimate_mm - truth_mm) <= TOLERANCE_MM
    )


def _arm_stats(truth, estimates, grades):
    errors = [abs(t - e) for t, e in zip(truth, estimates, strict=True) if e is not None]
    flat_indices = [i for i, t in enumerate(truth) if _side(t) == "flat"]
    flat_hits = sum(grades[i] for i in flat_indices)
    flat_false_positives = sum(
        estimates[i] is not None and _side(estimates[i]) != "flat" for i in flat_indices
    )
    passed = sum(grades)
    return {
        "packets": len(truth),
        "available": len(errors),
        "missing": len(truth) - len(errors),
        "passed": passed,
        "failed": len(truth) - passed,
        "mae_available_mm": math.fsum(errors) / len(errors) if errors else None,
        "flat_targets": len(flat_indices),
        "flat_hits": flat_hits,
        "flat_false_positives": flat_false_positives,
        "raised_targets": len(truth) - len(flat_indices),
        "raised_hits": passed - flat_hits,
    }


def _paired(rows):
    truth = [r["truth_mm"] for r in rows]
    sparse = [r["sparse_mm"] for r in rows]
    dense = [r["dense_mm"] for r in rows]
    sparse_ok = [grade(t, p) for t, p in zip(truth, sparse, strict=True)]
    dense_ok = [grade(t, p) for t, p in zip(truth, dense, strict=True)]
    paired = list(zip(sparse_ok, dense_ok, strict=True))
    common = [i for i in range(len(rows)) if sparse[i] is not None and dense[i] is not None]
    return {
        "sparse": _arm_stats(truth, sparse, sparse_ok),
        "dense": _arm_stats(truth, dense, dense_ok),
        "gained": sum(not a and b for a, b in paired),
        "regressed": sum(a and not b for a, b in paired),
        "both_passed": sum(a and b for a, b in paired),
        "both_failed": sum(not a and not b for a, b in paired),
        "both_available": len(common),
        "sparse_mae_both_available_mm": math.fsum(abs(truth[i] - sparse[i]) for i in common) / len(common)
        if common else None,
        "dense_mae_both_available_mm": math.fsum(abs(truth[i] - dense[i]) for i in common) / len(common)
        if common else None,
    }


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _paired_grid(rows, plan):
    seeds, ticks = tuple(plan["seeds"]), tuple(plan["capture_ticks"])
    layouts, visibilities = tuple(plan["layouts"]), tuple(plan["visibilities"])
    if (len(seeds) != 3 or len(set(seeds)) != 3 or not all(_integer(s) for s in seeds)
            or not 1 <= len(ticks) <= 12 or not all(_integer(t) and t >= 0 for t in ticks)
            or tuple(sorted(set(ticks))) != ticks):
        raise ValueError("expected three unique seeds and 1 to 12 registered capture ticks")
    if (len(layouts) != 3 or set(layouts) != {"flat", "left", "right"}
            or len(visibilities) != 2 or set(visibilities) != {"clear", "masked"}
            or not _integer(plan["delay_ticks"]) or plan["delay_ticks"] != 2):
        raise ValueError("unexpected layouts, visibilities or packet delay")

    expected = set(product(seeds, layouts, visibilities, ticks))
    grid = {}
    for row in rows:
        key = (row["seed"], row["layout"], row["visibility"], row["capture_tick"])
        if not _integer(row["seed"]) or not _integer(row["capture_tick"]) or key not in expected or key in grid:
            raise ValueError(f"duplicate or unregistered packet {key}")
        if not _integer(row["delivery_tick"]) or row["delivery_tick"] != row["capture_tick"] + plan["delay_ticks"]:
            raise ValueError(f"invalid packet delivery time for {key}")
        truth = _finite_mm(row["truth_mm"])
        if row["layout"] == "flat" and _side(truth) != "flat":
            raise ValueError(f"flat layout with a raised target at {key}")
        for name in ("sparse_mm", "dense_mm"):
            if row[name] is not None:
                _finite_mm(row[name])
        grid[key] = row
    if grid.keys() != expected:
        raise ValueError(f"incomplete packet grid: received {len(grid)} of {len(expected)}")
    for seed, layout, tick in product(seeds, layouts, ticks):
        if grid[seed, layout, "clear", tick]["truth_mm"] != grid[seed, layout, "masked", tick]["truth_mm"]:
            raise ValueError("changing only the angular mask changed the reference height")
    return [grid[key] for key in product(seeds, layouts, visibilities, ticks)]


def score_rows(rows, plan):
    """Recompute the registered decision from the complete grid of paired, delivered packets."""
    rows = _paired_grid(rows, plan)
    by_seed = []
    for seed in plan["seeds"]:
        samples = [r for r in rows if r["seed"] == seed]
        masked_raised = [r for r in samples if r["visibility"] == "masked" and r["layout"] in ("left", "right")]
        high = [r for r in masked_raised if _side(r["truth_mm"]) != "flat"]
        masked = _paired(masked_raised)
        clear = _paired([r for r in samples if r["visibility"] == "clear"])
        flat = _paired([r for r in samples if r["layout"] == "flat"])
        high_counts = {
            side: sum(r["layout"] == side and _side(r["truth_mm"]) == side for r in high)
            for side in ("left", "right")
        }
        checks = {
            "both_raised_sides_covered": min(high_counts.values()) > 0,
            "masked_raised_improved": masked["dense"]["passed"] > masked["sparse"]["passed"],
            "clear_failures_nonincreasing": clear["dense"]["failed"] <= clear["sparse"]["failed"],
            "flat_false_positives_nonincreasing":
                flat["dense"]["flat_false_positives"] <= flat["sparse"]["flat_false_positives"],
        }
        by_seed.append({
            "seed": seed, "overall": _paired(samples), "masked_raised_layouts": masked,
            "masked_true_high": _paired(high), "masked_high_counts": high_counts,
            "clear": clear, "flat_layout": flat, "checks": checks, "passed": all(checks.values()),
        })
    by_cell = []
    for seed, layout, visibility in product(plan["seeds"], plan["layouts"], plan["visibilities"]):
        selected = [r for r in rows if (r["seed"], r["layout"], r["visibility"]) == (seed, layout, visibility)]
        by_cell.append({"seed": seed, "layout": layout, "visibility": visibility, **_paired(selected)})
    return {"passed": all(s["passed"] for s in by_seed), "overall": _paired(rows),
            "by_seed": by_seed, "by_cell": by_cell}
