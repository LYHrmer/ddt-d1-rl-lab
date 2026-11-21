"""Finite-difference diagnostics for the frozen late trajectory. All returns are in metres."""

from math import fsum, isfinite

from tools95.score95 import agrees, difference, same_sign
from tools96.score96 import ARMS, MODELS, RESETS, SEEDS, complete_median, relative_error


CUT, END = 16, 32
METHODS = ("closed", "fixed", "anchored")
RADII = (("0.5", "mh", "ph"), ("1.0", "m1", "p1"))
COMPARISONS = (("closed", "fixed"), ("fixed", "anchored"))


def suffix_return(x, physical_origin, first_tick):
    """Read x[17:33] in physical clock coordinates; x may itself start at tick 0 or 16."""
    if first_tick not in (0, CUT):
        raise ValueError("expected a full trajectory or a suffix beginning at tick 16")
    if physical_origin is None or not isfinite(physical_origin) or len(x) <= END-first_tick:
        return None
    deviations = [abs(float(value) - physical_origin) for value in x[CUT+1-first_tick:END+1-first_tick]]
    return -fsum(deviations) / (END-CUT) if all(map(isfinite, deviations)) else None


def valid_return(row):
    return row["return_m"] if row["valid_numeric_suffix"] else None


def slope(negative, positive, width):
    delta = difference(valid_return(positive), valid_return(negative))
    return None if delta is None else delta / width


def score_cell(arms):
    if set(arms) != set(ARMS):
        raise ValueError("all five arm records are required, including missing arms")
    native = {k: arm["native"] for k, arm in arms.items()}
    results = {}
    for model in MODELS:
        methods = results[model] = {}
        for method in METHODS:
            prediction = {k: arm["models"][model][method] for k, arm in arms.items()}
            radial = {}
            for radius, lo, hi in RADII:
                width = ARMS[hi] - ARMS[lo]
                measured = slope(native[lo], native[hi], width)
                forecast = slope(prediction[lo], prediction[hi], width)
                error = difference(forecast, measured)
                radial[radius] = {"native_fd_m": measured, "model_fd_m": forecast,
                                  "same_sign": same_sign(forecast, measured), "within_25pct": agrees(forecast, measured),
                                  "absolute_slope_error_m": None if error is None else abs(error),
                                  "relative_slope_error": relative_error(forecast, measured)}
            baseline = difference(valid_return(prediction["zero"]), valid_return(native["zero"]))
            methods[method] = {
                "all_native_and_model_suffixes_valid": all(valid_return(r) is not None for r in (*native.values(), *prediction.values())),
                "radii": radial, "neutral_absolute_error_m": None if baseline is None else abs(baseline),
            }
    return results


def method_metrics(scores):
    radii = [r for s in scores for r in s["radii"].values()]
    n = len(scores)
    if any(set(s["radii"]) != {"0.5", "1.0"} for s in scores):
        raise ValueError("both radius records are required")
    rel = [r["relative_slope_error"] for r in radii]
    neutral = [s["neutral_absolute_error_m"] for s in scores]
    return {"cells_planned": n, "radii_planned": 2*n,
            "five_arm_cells_complete": sum(s["all_native_and_model_suffixes_valid"] for s in scores),
            "sign_comparable": sum(r["same_sign"] is not None for r in radii),
            "same_sign": sum(r["same_sign"] is True for r in radii), "within_25pct": sum(r["within_25pct"] is True for r in radii),
            "slope_errors_available": sum(x is not None for x in rel), "neutral_errors_available": sum(x is not None for x in neutral),
            "median_relative_slope_error": complete_median(rel, 2*n),
            "median_absolute_slope_error_m": complete_median([r["absolute_slope_error_m"] for r in radii], 2*n),
            "median_neutral_absolute_error_m": complete_median(neutral, n)}


def summarize(cells):
    if len(cells) != 9 or {(c["training_seed"], c["reset"]) for c in cells} != {(s, r) for s in SEEDS for r in RESETS}:
        raise ValueError("expected exactly one entry for each of the nine original blind cells")
    models = {}
    for model in MODELS:
        totals = {k: method_metrics([c["scores"][model][k] for c in cells]) for k in METHODS}
        by_seed = {s: {k: method_metrics([c["scores"][model][k] for c in cells if c["training_seed"] == s]) for k in METHODS} for s in SEEDS}
        comparisons = {}
        for before, after in COMPARISONS:
            decisions = []
            for seed, stats in by_seed.items():
                a, b = stats[before], stats[after]
                slope_change = difference(b["median_relative_slope_error"], a["median_relative_slope_error"])
                neutral_change = difference(b["median_neutral_absolute_error_m"], a["median_neutral_absolute_error_m"])
                full = a["five_arm_cells_complete"] == b["five_arm_cells_complete"] == 3
                slope_lower = None if slope_change is None else slope_change < 0
                zero_not_worse = None if neutral_change is None else neutral_change <= 0
                decisions.append({"training_seed": seed, "before": a, "after": b, "all_three_cells_complete": full,
                                  "median_relative_slope_error_strictly_lower": slope_lower,
                                  "median_neutral_error_nonincreasing": zero_not_worse,
                                  "supported": full and slope_lower is True and zero_not_worse is True})
            comparisons[before+"_to_"+after] = {"by_seed": decisions, "seeds_supported": sum(d["supported"] for d in decisions),
                                                "all_three_seeds_support": all(d["supported"] for d in decisions)}
        models[model] = {"methods": totals, "comparisons": comparisons}
    return {"primary_family": "paired", "models": models}
