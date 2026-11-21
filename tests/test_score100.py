from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from tools100.score100 import RADIUS_M, TIE_ATOL_M, TOLERANCE_MM, estimate_delta_mm, grade, score_rows


PLAN = {"seeds": [1, 2, 3], "layouts": ["flat", "left", "right"], "visibilities": ["clear", "masked"],
        "capture_ticks": [9, 17], "delay_ticks": 2}


def test_public_registration_matches_estimator_and_scoring_constants():
    path = Path(__file__).resolve().parents[1] / "docs/experiments100/pointcloud_density.json"
    plan = json.loads(path.read_text())
    assert plan["estimator"]["radius_m"] == RADIUS_M
    assert plan["estimator"]["tie_tolerance_m"] == TIE_ATOL_M
    assert plan["score"]["tolerance_mm"] == TOLERANCE_MM
    assert (len(plan["seeds"]) * len(plan["layouts"]) * len(plan["visibilities"]) *
            len(plan["capture_ticks"])) == plan["budget"]["paired_rows"]


def paired_rows():
    rows = []
    for seed, layout, visibility, tick in product(
        PLAN["seeds"], PLAN["layouts"], PLAN["visibilities"], PLAN["capture_ticks"]
    ):
        truth = {"flat": 0.0, "left": 30.0, "right": -30.0}[layout] if tick == 17 else 0.0
        rows.append({"seed": seed, "layout": layout, "visibility": visibility,
                     "capture_tick": tick, "delivery_tick": tick + 2, "truth_mm": truth,
                     "sparse_mm": truth if visibility == "clear" else 0.0, "dense_mm": truth})
    return rows


def test_gravity_frame_height_estimate_uses_nearest_ties_and_ignores_invalid_returns():
    query = np.array([[0.0, 0.2], [0.0, -0.2]])
    cloud = np.array([[0.0, 0.2, np.nan], [np.nan, -0.2, 0.0],
                      [-0.1, 0.2, 0.01], [0.1 + 5e-13, 0.2, 0.03], [0.11, 0.2, 0.1],
                      [0.15, -0.2, -0.01]])
    assert estimate_delta_mm(cloud, query) == pytest.approx(30.0)
    assert estimate_delta_mm(cloud[::-1], query[::-1]) == pytest.approx(-30.0)
    assert estimate_delta_mm(cloud[2:], query) == pytest.approx(30.0)
    shift = np.array([3.0, 2.0, 0.4])
    # The nearest point on the right lies exactly on the search radius. Use
    # an interior return for the translation check to avoid boundary roundoff.
    cloud[-1, 0] = 0.1
    assert estimate_delta_mm(cloud + shift, query + shift[:2]) == pytest.approx(30.0)


def test_missing_and_invalid_estimator_inputs():
    query = [[0.0, 0.2], [0.0, -0.2]]
    assert estimate_delta_mm([], query) is None
    assert estimate_delta_mm([[0.0, 0.2, 0.0], [0.150001, -0.2, 0.0]], query) is None
    assert estimate_delta_mm([[np.nan, np.nan, np.nan]], query) is None
    with pytest.raises(ValueError, match="XYZ"):
        estimate_delta_mm([[1.0, 2.0]], query)
    with pytest.raises(ValueError, match="queries"):
        estimate_delta_mm([], [[0.0, 0.2], [np.nan, -0.2]])


@pytest.mark.parametrize("truth,prediction,passed", [
    (0.0, None, False), (0.0, -10.0, True), (0.0, 10.0, True), (0.0, 10.001, False),
    (10.0, -10.0, True), (10.001, 10.0, False),
    (21.0, 11.0, True), (-21.0, -11.0, True), (21.0, 10.9999, False),
    (30.0, -30.0, False), (-30.0, -40.0, True), (-30.0, -40.001, False),
])
def test_geometry_grade_boundary_and_sign(truth, prediction, passed):
    assert grade(truth, prediction) is passed


def test_complete_grid_preserves_all_planned_denominators_and_pairings():
    rows = paired_rows()
    result = score_rows(rows, PLAN)
    assert result == score_rows(reversed(rows), PLAN)
    assert result["passed"] and len(result["by_cell"]) == 18
    assert result["overall"]["sparse"]["packets"] == 36
    assert result["overall"]["gained"] == 6 and result["overall"]["regressed"] == 0
    for seed in result["by_seed"]:
        primary = seed["masked_raised_layouts"]
        assert primary["sparse"]["packets"] == primary["dense"]["packets"] == 4
        assert primary["sparse"]["passed"] == 2 and primary["dense"]["passed"] == 4
        assert primary["sparse"]["mae_available_mm"] == 15.0
        assert seed["masked_true_high"]["sparse"]["packets"] == 2
        assert seed["masked_high_counts"] == {"left": 1, "right": 1}
        assert seed["clear"]["dense"]["failed"] == 0
        assert seed["flat_layout"]["dense"]["flat_false_positives"] == 0


def test_each_seed_must_improve_even_when_pooled_gain_is_positive():
    rows = paired_rows()
    for row in rows:
        if row["seed"] == 2 and row["visibility"] == "masked":
            row["dense_mm"] = row["sparse_mm"]
    result = score_rows(rows, PLAN)
    assert result["overall"]["gained"] > 0 and result["overall"]["regressed"] == 0
    assert [s["passed"] for s in result["by_seed"]] == [True, False, True]
    assert not result["passed"]


def test_missing_clear_estimate_is_failure_and_masked_flat_warning_is_counted():
    rows = paired_rows()
    for row in rows:
        if row["seed"] == 1 and row["layout"] == "flat" and row["capture_tick"] == 9:
            row["dense_mm"] = None if row["visibility"] == "clear" else 11.0
    result = score_rows(rows, PLAN)
    s = result["by_seed"][0]
    assert not result["passed"]
    assert not s["checks"]["clear_failures_nonincreasing"]
    assert not s["checks"]["flat_false_positives_nonincreasing"]
    flat = s["flat_layout"]["dense"]
    assert (flat["missing"], flat["flat_false_positives"], flat["flat_hits"], flat["failed"]) == (1, 1, 2, 2)
    assert s["flat_layout"]["both_available"] == 3


def test_no_raised_observations_for_one_side_prevents_claim():
    rows = paired_rows()
    for row in rows:
        if row["seed"] == 1 and row["layout"] == "right":
            row["truth_mm"] = row["dense_mm"] = row["sparse_mm"] = 0.0
    s = score_rows(rows, PLAN)["by_seed"][0]
    assert s["checks"]["masked_raised_improved"] and not s["passed"]
    assert s["masked_high_counts"] == {"left": 1, "right": 0}


def test_public_score_rejects_partial_duplicate_unsynchronized_or_nonfinite_rows():
    rows = paired_rows()
    with pytest.raises(ValueError, match="incomplete"):
        score_rows(rows[:-1], PLAN)
    with pytest.raises(ValueError, match="duplicate"):
        score_rows(rows + [dict(rows[0])], PLAN)
    for override in ({"delivery_tick": 40}, {"dense_mm": np.nan}, {"truth_mm": 30.0}, {"seed": True}):
        with pytest.raises(ValueError):
            score_rows([dict(rows[0], **override), *rows[1:]], PLAN)
    # Both conditions were sampled against exactly the same terrain.
    with pytest.raises(ValueError, match="reference"):
        score_rows([dict(row, truth_mm=31.0) if row["layout"] == "left" and row["visibility"] == "masked"
                    and row["capture_tick"] == 17 else row for row in rows], PLAN)
