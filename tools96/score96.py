"""Score the two world models against one common set of blind native transitions."""

from math import isfinite
from statistics import median

from tools95.score95 import ARMS, HORIZON, SEEDS, difference, reward_m, score_cell


MODELS = ("control", "paired")
TRAIN_RESETS = (96101, 96102, 96103)
RESETS = (96201, 96202, 96203)
RADII = ("0.5", "1.0")


def relative_error(a, b):
    delta = difference(a, b)
    if delta is None or max(abs(a), abs(b)) == 0:
        return None
    return abs(delta) / max(abs(a), abs(b))


def score_models(arms):
    result = {}
    for model in MODELS:
        pairs = {tag: {"native": value.get("native", {}), "model": value.get(model, {})}
                 for tag, value in arms.items()}
        score = score_cell(pairs)
        for measured in score["radii"].values():
            measured["ad_native_relative_error"] = relative_error(score["model_ad_m"], measured["native_slope_m"])
        zero = pairs.get("zero", {})
        delta = difference(zero.get("model", {}).get("return_m"), zero.get("native", {}).get("return_m"))
        score["zero_absolute_return_error_m"] = abs(delta) if delta is not None else None
        result[model] = score
    return result


def model_counts(scores):
    if any(set(s["radii"]) != set(RADII) for s in scores):
        raise ValueError("both planned radii are required, including unavailable values")
    radii = [r for s in scores for r in s["radii"].values()]
    result = {
        "cells_planned": len(scores), "radii_planned": 2 * len(scores),
        "cells_passed": sum(s["passed"] is True for s in scores),
        "all_cells_passed": all(s["passed"] is True for s in scores),
        "all_five_native_safe_cells": sum(s["gates"]["five_native_complete_safe"] for s in scores),
        "all_five_model_safe_cells": sum(s["gates"]["five_model_complete"] for s in scores),
        "both_forward_beat_zero_cells": sum(s["gates"]["both_forward_beats_zero"] for s in scores),
    }
    for label, field in (("native_positive", "native_positive"), ("ad_native_sign", "ad_sign_matches_native"),
                         ("ad_native_25pct", "ad_matches_native_25pct"), ("ad_model_25pct", "ad_matches_model_25pct"),
                         ("forward_beats_zero", "forward_beats_zero")):
        result[label] = {"comparable": sum(r[field] is not None for r in radii), "passed": sum(r[field] is True for r in radii)}
    return result


def complete_median(values, count):
    if len(values) != count or any(v is None or not isfinite(v) for v in values):
        return None
    return median(values)


def summarize(cells):
    keys = {(c["training_seed"], c["reset"]) for c in cells}
    if keys != {(s, r) for s in SEEDS for r in RESETS} or len(cells) != 9:
        raise ValueError("expected one record for every planned seed and blind reset")
    by_seed = []
    for seed in SEEDS:
        per_model = {}
        for model in MODELS:
            scores = [c["scores"][model] for c in cells if c["training_seed"] == seed]
            radial = [s["radii"][r]["ad_native_relative_error"] for s in scores for r in RADII]
            baseline = [s["zero_absolute_return_error_m"] for s in scores]
            per_model[model] = {"counts": model_counts(scores), "slope_error_available": sum(v is not None for v in radial),
                                "slope_error_median": complete_median(radial, 6),
                                "zero_error_available": sum(v is not None for v in baseline),
                                "zero_error_median_m": complete_median(baseline, 3)}
        control, paired = per_model["control"], per_model["paired"]
        slope_diff = difference(paired["slope_error_median"], control["slope_error_median"])
        zero_diff = difference(paired["zero_error_median_m"], control["zero_error_median_m"])
        by_seed.append({"training_seed": seed, "models": per_model,
                        "slope_median_strictly_improved": None if slope_diff is None else slope_diff < 0,
                        "zero_median_noninferior": None if zero_diff is None else zero_diff <= 0})
    totals = {m: model_counts([c["scores"][m] for c in cells]) for m in MODELS}
    gates = {"paired_all_nine_cells": totals["paired"]["all_cells_passed"],
             "every_seed_slope_median_strictly_improved": all(s["slope_median_strictly_improved"] is True for s in by_seed),
             "every_seed_zero_median_noninferior": all(s["zero_median_noninferior"] is True for s in by_seed)}
    return {"worlds": totals, "by_seed": by_seed, "gates": gates, "candidate_supported_by_scores": all(gates.values())}
