"""Blind H16 physical-state errors and the original, independent H32 policy-direction gates."""

import numpy as np

from tools95.score95 import ARMS, SEEDS, difference, score_cell
from tools96.score96 import complete_median, model_counts, relative_error


RESETS = (98201, 98202, 98203)
MODELS = ("control", "endpoint")
TICK = 16
WHEEL = (3, 7, 11, 15)
GROUPS = {
    "position": tuple(range(3)), "quaternion": tuple(range(3, 7)), "joint_position": tuple(range(7, 23)),
    "base_linear_velocity": tuple(range(23, 26)), "base_angular_velocity": tuple(range(26, 29)),
    "leg_velocity": tuple(29+i for i in range(16) if i not in WHEEL), "wheel_velocity": tuple(29+i for i in WHEEL),
}
# The frozen training vector is float32. Convert its actual represented values to float64 for this readback.
PHYSICAL_SCALE = np.array([.1]*7 + [1. if i in WHEEL else .5 for i in range(16)] + [.3]*6 +
                          [5. if i in WHEEL else 2. for i in range(16)], dtype=np.float32).astype(np.float64)


def state_error(native, forecast, usable=True):
    empty = {"available": False, "normalized_mse": None, "group_mse": dict.fromkeys(GROUPS)}
    if not usable or native is None or forecast is None:
        return empty
    actual, predicted = (np.asarray(x, dtype=np.float64) for x in (native, forecast))
    if actual.shape != (45,) or predicted.shape != (45,) or not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        return empty
    squared = ((predicted-actual)/PHYSICAL_SCALE)**2
    if not np.isfinite(squared).all():
        return empty
    values = {g: float(np.mean(squared[list(indices)])) for g, indices in GROUPS.items()}
    return {"available": True, "normalized_mse": float(np.mean(list(values.values()))), "group_mse": values}


def h32_scores(arms):
    if set(arms) != set(ARMS):
        raise ValueError("record all five blind arms, including absent ones")
    scores = {}
    for m in MODELS:
        score = score_cell({k: {"native": a["native"], "model": a[m]} for k, a in arms.items()})
        for radius in score["radii"].values():
            radius["ad_native_relative_error"] = relative_error(score["model_ad_m"], radius["native_slope_m"])
        delta = difference(arms["zero"][m].get("return_m"), arms["zero"]["native"].get("return_m"))
        score["zero_absolute_return_error_m"] = None if delta is None else abs(delta)
        scores[m] = score
    return scores


def h16_metrics(errors, n):
    return {"planned": n, "comparable": sum(e["available"] for e in errors),
            "median_normalized_mse": complete_median([e["normalized_mse"] for e in errors], n),
            "median_group_mse": {g: complete_median([e["group_mse"][g] for e in errors], n) for g in GROUPS}}


def summarize(cells):
    if len(cells) != 9 or {(c["training_seed"], c["reset"]) for c in cells} != {(s, r) for s in SEEDS for r in RESETS}:
        raise ValueError("required blind grid consists of nine unique seed/reset pairs")
    h16, h32 = [], []
    for seed in SEEDS:
        grid = [c for c in cells if c["training_seed"] == seed]
        if any(set(c["arms"]) != set(ARMS) for c in grid):
            raise ValueError("incomplete arm roster")
        endpoints, directions = {}, {}
        for m in MODELS:
            endpoints[m] = h16_metrics([a["h16"][m] for c in grid for a in c["arms"].values()], 15)
            scores = [c["scores"][m] for c in grid]
            radial = [r["ad_native_relative_error"] for s in scores for r in s["radii"].values()]
            zeros = [s["zero_absolute_return_error_m"] for s in scores]
            directions[m] = {"counts": model_counts(scores), "slope_errors_available": sum(x is not None for x in radial),
                             "median_relative_slope_error": complete_median(radial, 6), "neutral_errors_available": sum(x is not None for x in zeros),
                             "median_neutral_absolute_error_m": complete_median(zeros, 3)}
        d = difference(endpoints["endpoint"]["median_normalized_mse"], endpoints["control"]["median_normalized_mse"])
        slope = difference(directions["endpoint"]["median_relative_slope_error"], directions["control"]["median_relative_slope_error"])
        neutral = difference(directions["endpoint"]["median_neutral_absolute_error_m"], directions["control"]["median_neutral_absolute_error_m"])
        h16.append({"training_seed": seed, "worlds": endpoints, "strictly_improved": None if d is None else d < 0})
        h32.append({"training_seed": seed, "worlds": directions, "slope_median_strictly_improved": None if slope is None else slope < 0,
                    "neutral_median_noninferior": None if neutral is None else neutral <= 0})
    worlds = {m: model_counts([c["scores"][m] for c in cells]) for m in MODELS}
    h16_pass = all(x["strictly_improved"] is True for x in h16)
    h32_pass = worlds["endpoint"]["all_cells_passed"]
    return {"h16": {"by_seed": h16, "all_three_strictly_improved": h16_pass}, "worlds": worlds,
            "h32_attribution": {"by_seed": h32, "all_three_slope_medians_strictly_improved": all(x["slope_median_strictly_improved"] is True for x in h32),
                                "all_three_neutral_medians_noninferior": all(x["neutral_median_noninferior"] is True for x in h32)},
            "endpoint_h32_all_nine": h32_pass, "candidate_supported_by_scores": h16_pass and h32_pass}
