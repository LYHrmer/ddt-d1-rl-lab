"""Apply the registered H16 and H32 criteria to the shared round-99 blind grid."""

from tools95.score95 import difference, score_cell
from tools96.score96 import complete_median, model_counts, relative_error
from tools98.score98 import h16_metrics
from tools99.common99 import ARMS, MODELS, RESETS, SEEDS


def h32_scores(arms):
    if set(arms) != set(ARMS):
        raise ValueError("expected all five registered probes, including unavailable arms")
    scores = {}
    for model in MODELS:
        score = score_cell({tag: {"native": a["native"], "model": a[model]} for tag, a in arms.items()})
        for radial in score["radii"].values():
            radial["ad_native_relative_error"] = relative_error(score["model_ad_m"], radial["native_slope_m"])
        delta = difference(arms["zero"][model]["return_m"], arms["zero"]["native"]["return_m"])
        score["zero_absolute_return_error_m"] = None if delta is None else abs(delta)
        scores[model] = score
    return scores


def summarize(cells):
    expected = {(s, r) for s in SEEDS for r in RESETS}
    if len(cells) != len(expected) or {(c["training_seed"], c["reset"]) for c in cells} != expected:
        raise ValueError("expected the nine registered, unique round-99 blind cells")
    if any(set(c["arms"]) != set(ARMS) or set(c["scores"]) != set(MODELS) for c in cells):
        raise ValueError("a planned blind arm or world is absent")
    h16, h32 = [], []
    for seed in SEEDS:
        group = [c for c in cells if c["training_seed"] == seed]
        endpoints, slopes = {}, {}
        for model in MODELS:
            endpoints[model] = h16_metrics([a["h16"][model] for c in group for a in c["arms"].values()], 15)
            scores = [c["scores"][model] for c in group]
            radial = [r["ad_native_relative_error"] for s in scores for r in s["radii"].values()]
            neutral = [s["zero_absolute_return_error_m"] for s in scores]
            slopes[model] = {"counts": model_counts(scores), "slope_errors_available": sum(x is not None for x in radial),
                             "median_relative_slope_error": complete_median(radial, 6),
                             "neutral_errors_available": sum(x is not None for x in neutral),
                             "median_neutral_absolute_error_m": complete_median(neutral, 3)}
        state_delta = difference(endpoints["late"]["median_normalized_mse"], endpoints["control"]["median_normalized_mse"])
        slope_delta = difference(slopes["late"]["median_relative_slope_error"], slopes["control"]["median_relative_slope_error"])
        neutral_delta = difference(slopes["late"]["median_neutral_absolute_error_m"], slopes["control"]["median_neutral_absolute_error_m"])
        h16.append({"training_seed": seed, "worlds": endpoints, "strictly_improved": None if state_delta is None else state_delta < 0})
        h32.append({"training_seed": seed, "worlds": slopes,
                    "slope_median_strictly_improved": None if slope_delta is None else slope_delta < 0,
                    "neutral_median_noninferior": None if neutral_delta is None else neutral_delta <= 0})
    worlds = {m: model_counts([c["scores"][m] for c in cells]) for m in MODELS}
    endpoint = all(s["strictly_improved"] is True for s in h16)
    direction = worlds["late"]["all_cells_passed"]
    improved = all(s["slope_median_strictly_improved"] is True for s in h32)
    return {"h16": {"by_seed": h16, "all_three_strictly_improved": endpoint}, "worlds": worlds,
            "h32_attribution": {"by_seed": h32, "all_three_slope_medians_strictly_improved": improved,
                                "all_three_neutral_medians_noninferior": all(s["neutral_median_noninferior"] is True for s in h32)},
            "late_h32_all_nine": direction, "slope_improvement_supported": improved,
            "candidate_supported_by_scores": endpoint and direction,
            "all_three_criteria_passed": improved and endpoint and direction}
