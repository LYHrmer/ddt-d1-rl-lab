"""Compare a sealed policy-parameter direction with paired native returns (metres)."""
from __future__ import annotations

import math


SEEDS = (94011, 94023, 94037)
RESETS = (95101, 95102, 95103)
ARMS = {"m1": -1.0, "mh": -0.5, "zero": 0.0, "ph": 0.5, "p1": 1.0}
HORIZON = 32


def reward_m(x, origin, horizon=HORIZON):
    if len(x) <= horizon:
        return None
    values = [abs(float(a) - float(origin)) for a in x[1:horizon + 1]]
    if not all(map(math.isfinite, values)):
        return None
    return -math.fsum(values) / horizon


def same_sign(left, right):
    return None if left is None or right is None else left * right > 0


def agrees(left, right):
    if left is None or right is None:
        return None
    return same_sign(left, right) and abs(left - right) <= 0.25 * max(abs(left), abs(right))


def difference(left, right):
    return None if left is None or right is None else left - right


def radius_score(native, model, ad, low, high):
    width = ARMS[high] - ARMS[low]
    dn = difference(native[high], native[low])
    dm = difference(model[high], model[low])
    native_slope = None if dn is None else dn / width
    model_slope = None if dm is None else dm / width
    gain = difference(native[high], native["zero"])
    return {
        "native_slope_m": native_slope, "model_slope_m": model_slope,
        "native_positive": None if native_slope is None else native_slope > 0,
        "ad_sign_matches_native": same_sign(ad, native_slope),
        "ad_matches_native_25pct": agrees(ad, native_slope),
        "ad_matches_model_25pct": agrees(ad, model_slope),
        "forward_gain_m": gain, "forward_beats_zero": None if gain is None else gain > 0,
    }


def score_cell(records):
    """A terminal H32 endpoint retains its numeric reward and fails the safety gate."""
    native, model = {}, {}
    for key in ARMS:
        arm = records.get(key, {})
        native[key] = arm.get("native", {}).get("return_m")
        model[key] = arm.get("model", {}).get("return_m")
    ad = records.get("zero", {}).get("model", {}).get("directional_derivative")
    half = radius_score(native, model, ad, "mh", "ph")
    full = radius_score(native, model, ad, "m1", "p1")
    both = (half, full)
    gates = {
        "five_native_complete_safe": all(records.get(k, {}).get("native", {}).get("safe") is True
                                         and native[k] is not None for k in ARMS),
        "five_model_complete": all(records.get(k, {}).get("model", {}).get("safe") is True
                                   and model[k] is not None for k in ARMS),
        "native_both_positive": all(r["native_positive"] is True for r in both),
        "native_slopes_stable_25pct": agrees(half["native_slope_m"], full["native_slope_m"]),
        "ad_matches_native_both_25pct": all(r["ad_matches_native_25pct"] is True for r in both),
        "ad_matches_model_both_25pct": all(r["ad_matches_model_25pct"] is True for r in both),
        "both_forward_beats_zero": all(r["forward_beats_zero"] is True for r in both),
    }
    return {"model_ad_m": ad, "radii": {"0.5": half, "1.0": full}, "gates": gates,
            "passed": all(v is True for v in gates.values())}


def summarize(cells):
    expected = {(s, r) for s in SEEDS for r in RESETS}
    keys = {(c["training_seed"], c["reset"]) for c in cells}
    if keys != expected or len(cells) != len(expected):
        raise ValueError("expected one record, including missing arms, for each of the nine cells")
    scores = [c["score"] for c in cells]
    radii = [r for score in scores for r in score["radii"].values()]
    count = sum(score["passed"] for score in scores)
    return {
        "cells_planned": len(cells), "radii_planned": len(radii), "cells_passed": count,
        "qualified": count == len(cells),
        "safe_five_arm_cells": sum(s["gates"]["five_native_complete_safe"] for s in scores),
        "positive_model_ad_cells": sum(s["model_ad_m"] is not None and s["model_ad_m"] > 0
                                       for s in scores),
        "both_forward_beat_zero_cells": sum(s["gates"]["both_forward_beats_zero"] for s in scores),
        "native_slopes_available": sum(r["native_slope_m"] is not None for r in radii),
        "native_slopes_positive": sum(r["native_positive"] is True for r in radii),
        "ad_native_comparable": sum(r["ad_sign_matches_native"] is not None for r in radii),
        "ad_native_same_sign": sum(r["ad_sign_matches_native"] is True for r in radii),
        "ad_native_within_25pct": sum(r["ad_matches_native_25pct"] is True for r in radii),
        "ad_model_comparable": sum(r["ad_matches_model_25pct"] is not None for r in radii),
        "ad_model_within_25pct": sum(r["ad_matches_model_25pct"] is True for r in radii),
        "forward_zero_comparable": sum(r["forward_beats_zero"] is not None for r in radii),
        "forward_beats_zero": sum(r["forward_beats_zero"] is True for r in radii),
    }
