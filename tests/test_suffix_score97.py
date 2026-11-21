import math

import numpy as np
import pytest

from tools97 import publish97
from tools95.score95 import agrees
from tools97.score97 import ARMS, CUT, END, METHODS, MODELS, RESETS, SEEDS, score_cell, suffix_return, summarize


def cell_arms(scales=(2., 1.25, 1.), offsets=(0., 0., 0.)):
    arms = {}
    for tag, alpha in ARMS.items():
        actual = {"return_m": -0.25 + alpha/64, "valid_numeric_suffix": True}
        models = {m: {k: {"return_m": -0.25 + off + alpha*scale/64, "valid_numeric_suffix": True}
                      for k, scale, off in zip(METHODS, scales, offsets)} for m in MODELS}
        arms[tag] = {"native": actual, "models": models}
    return arms


def grid(factory):
    return [{"training_seed": s, "reset": r, "scores": score_cell(factory(s, r))} for s in SEEDS for r in RESETS]


def test_suffix_uses_17_to_32_and_origin_at_physical_reset():
    origin = 1.25
    x = [origin] + [50.] * 16 + [origin - .125]*8 + [origin + .25]*8
    assert len(x) == 33
    assert suffix_return(x, origin, 0) == suffix_return(x[CUT:], origin, CUT) == -.1875
    assert suffix_return(x[:-1], origin, 0) is None
    assert suffix_return(x[CUT:-1], origin, CUT) is None
    assert suffix_return(x[:-1] + [math.nan], origin, 0) is None
    with pytest.raises(ValueError):
        suffix_return(x, origin, 17)


def test_symmetric_relative_error_includes_quarter_and_zero_is_not_a_direction():
    arms = cell_arms(scales=(1., .75, -1.))
    scored = score_cell(arms)["paired"]
    for r in ("0.5", "1.0"):
        assert scored["fixed"]["radii"][r] == {"native_fd_m": 1/64, "model_fd_m": .75/64, "same_sign": True,
                "within_25pct": True, "absolute_slope_error_m": .25/64, "relative_slope_error": .25}
        assert scored["anchored"]["radii"][r]["same_sign"] is False
        assert scored["anchored"]["radii"][r]["relative_slope_error"] == 2.
    for tag in ARMS:
        arms[tag]["native"]["return_m"] = -1
        arms[tag]["models"]["paired"]["closed"]["return_m"] = -1
    flat = score_cell(arms)["paired"]["closed"]["radii"]["0.5"]
    assert flat["same_sign"] is False and flat["within_25pct"] is False and flat["relative_slope_error"] is None


def test_attribution_is_paired_within_model_and_requires_every_seed():
    result = summarize(grid(lambda s, r: cell_arms()))
    assert result["primary_family"] == "paired"
    for model in MODELS:
        summary = result["models"][model]
        assert all(x["seeds_supported"] == 3 and x["all_three_seeds_support"] for x in summary["comparisons"].values())
        assert summary["methods"]["anchored"]["same_sign"] == summary["methods"]["anchored"]["within_25pct"] == 18

    def equal_one_weak_seed(s, r):
        arms = cell_arms()
        if s == SEEDS[0]:
            for arm in arms.values():
                arm["models"]["paired"]["fixed"] = arm["models"]["paired"]["closed"].copy()
        return arms
    decision = summarize(grid(equal_one_weak_seed))["models"]
    assert decision["paired"]["comparisons"]["closed_to_fixed"]["seeds_supported"] == 2
    assert not decision["paired"]["comparisons"]["closed_to_fixed"]["all_three_seeds_support"]
    assert decision["control"]["comparisons"]["closed_to_fixed"]["all_three_seeds_support"]
    assert decision["paired"]["comparisons"]["fixed_to_anchored"]["all_three_seeds_support"]
    baseline = summarize(grid(lambda s, r: cell_arms(offsets=(0., .001 if s == SEEDS[0] else 0., 0.))))
    weak = baseline["models"]["paired"]["comparisons"]["closed_to_fixed"]["by_seed"][0]
    assert weak["median_relative_slope_error_strictly_lower"] and not weak["median_neutral_error_nonincreasing"] and not weak["supported"]


def test_incomplete_arm_excluded_even_with_raw_numeric_x_at_last_tick():
    def partial(s, r):
        arms = cell_arms()
        if (s, r) == (SEEDS[0], RESETS[0]):
            arms["ph"]["models"]["paired"]["anchored"]["valid_numeric_suffix"] = False
        return arms
    cells = grid(partial)
    h = cells[0]["scores"]["paired"]["anchored"]
    assert h["radii"]["0.5"]["model_fd_m"] is None and h["radii"]["1.0"]["model_fd_m"] == 1/64
    summaries = summarize(cells)["models"]["paired"]["comparisons"]
    assert summaries["closed_to_fixed"]["all_three_seeds_support"]
    partial = summaries["fixed_to_anchored"]["by_seed"][0]
    assert partial["after"]["radii_planned"] == 6 and partial["after"]["slope_errors_available"] == 5
    assert partial["after"]["median_relative_slope_error"] is None and not partial["all_three_cells_complete"] and not partial["supported"]
    with pytest.raises(ValueError):
        summarize(cells[:-1])


def test_archived_terminal_physical_coordinate_is_observed_but_remains_unsafe():
    x = np.linspace(1.25, 1., END+1)
    flags = np.zeros((END, 4), bool)
    flags[-1, 1:3] = True
    row = {"curve_key": "x", "source_sha256": "source", "flags_key": "flags", "safe": False, "status": "TERMINAL", "error": None}
    actual = publish97.parent_row(row, {"x": x, "flags": flags}, x[0], True)
    assert actual["valid_numeric_suffix"] and not actual["safe"] and actual["ticks"] == 16
    assert actual["flag_ticks"]["illegal"] == [32] and actual["return_m"] == suffix_return(x, x[0], 0)
    short = publish97.parent_row(row, {"x": x[:-1], "flags": flags[:-1]}, x[0], True)
    assert not short["valid_numeric_suffix"] and short["return_m"] is None


def test_published_replay_cannot_change_applied_tape_or_promote_rejected_endpoint():
    start = np.zeros(45, np.float32)
    start[0] = 1.25
    tape = np.zeros((16, 16), np.float32)
    tape[:, 0] = np.arange(16)/32
    x = np.array([start[0], 1.20], np.float32)
    sources = {"native_trace_sha256": "native", "forecast96_sha256": "forecast", "checkpoint_sha256": "world"}
    row = {**sources, "receipt_sha256": "receipt", "curve_key": "x", "start_state_sha256": publish97._f32_sha(start),
           "applied_tape_sha256": publish97._f32_sha(tape), "applied_prefix_sha256": publish97._f32_sha(tape[:2]),
           "attempted_model_steps": 2, "completed_model_steps": 2, "retained_model_steps": 1, "return_m": None,
           "ticks": 1, "last_tick": 17, "rejected_endpoint_tick": 18, "error": "invalid next state", "status": "PARTIAL", "valid_numeric_suffix": False}
    seen = set()
    publish97.verify_replay(row, {"x": x}, seen, "x", start, tape, sources, 1.25)
    assert seen == {"x"}
    with pytest.raises(ValueError):
        publish97.verify_replay(dict(row, status="COMPLETE", valid_numeric_suffix=True), {"x": x}, set(), "x", start, tape, sources, 1.25)
    wrong_tape = tape.copy()
    wrong_tape[0] = tape[1]
    with pytest.raises(ValueError):
        publish97.verify_replay(row, {"x": x}, set(), "x", start, wrong_tape, sources, 1.25)


def test_reported_posthoc_translation_of_the_saved_curves():
    """Re-score the fixed prediction after translating x, without changing its shape or running a model."""
    if not publish97.REPORT.exists() and not publish97.CURVES.exists():
        pytest.skip("public replay has not been produced yet")
    results = publish97.read(publish97.REPORT)
    passed = dict.fromkeys(MODELS, 0)
    with np.load(publish97.CURVES, allow_pickle=False) as curves:
        for cell in results["cells"]:
            for model in MODELS:
                returns = {}
                for tag, arm in cell["arms"].items():
                    fixed = arm["models"][model]["fixed"]
                    assert fixed["valid_numeric_suffix"]
                    x = curves[fixed["curve_key"]].astype(np.float64)
                    native_cut_x = float(curves[arm["native_cut_key"]][0])
                    translated = x + (native_cut_x - x[0])
                    returns[tag] = suffix_return(translated, cell["origin_x_m"], CUT)
                for radius, lo, hi in (("0.5", "mh", "ph"), ("1.0", "m1", "p1")):
                    predicted = (returns[hi] - returns[lo]) / (ARMS[hi] - ARMS[lo])
                    actual = cell["scores"][model]["fixed"]["radii"][radius]["native_fd_m"]
                    passed[model] += agrees(predicted, actual)
    assert passed == {"control": 5, "paired": 11}
