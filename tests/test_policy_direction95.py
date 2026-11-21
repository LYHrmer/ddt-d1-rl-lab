"""Check the registered policy-direction gates and independently read the wheel archive."""

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools95.score95 import ARMS, HORIZON, RESETS, SEEDS, agrees, reward_m, score_cell, summarize


ROOT = Path(__file__).resolve().parents[1]
SLOPE = 1 / 128


def linear_probes():
    records = {}
    for arm, scale in ARMS.items():
        value = -0.125 + SLOPE * scale
        records[arm] = {source: {"return_m": value, "safe": True} for source in ("native", "model")}
    records["zero"]["model"]["directional_derivative"] = SLOPE
    return records


def complete_grid():
    return [{"training_seed": seed, "reset": reset, "score": score_cell(linear_probes())}
            for seed in SEEDS for reset in RESETS]


def test_agreement_requires_finite_nonzero_same_sign_and_includes_quarter_boundary():
    just_outside = math.nextafter(3.0, 0.0)
    for sign in (-1, 1):
        for a, b in ((3.0, 4.0), (4.0, 3.0)):
            assert agrees(sign * a, sign * b) is True
        assert agrees(sign * just_outside, sign * 4.0) is False
        assert agrees(sign * 4.0, sign * just_outside) is False
    assert agrees(-4, 3) is False and agrees(0, 0) is False
    assert agrees(None, 1) is None and agrees(1, None) is None
    for invalid in (math.nan, math.inf, -math.inf):
        assert agrees(invalid, 1) is not True


def test_reward_uses_every_post_reset_sample_and_the_physical_origin():
    origin = 1.25
    x = [origin] + [origin + 0.125] * 16 + [origin - 0.25] * 16
    assert HORIZON == 32 and reward_m(x + [1000], origin) == -0.1875
    assert reward_m(x[:-1], origin) is None
    assert reward_m(x[:-1] + [math.nan], origin) is None


@pytest.mark.parametrize("source,gate", [("native", "five_native_complete_safe"), ("model", "five_model_complete")])
def test_short_branch_retains_missing_return_even_if_record_marked_safe(source, gate):
    probes = linear_probes()
    probes["ph"][source]["return_m"] = reward_m([1.25] * HORIZON, 1.25)
    score = score_cell(probes)
    assert score["radii"]["0.5"][source + "_slope_m"] is None
    assert score["radii"]["1.0"][source + "_slope_m"] == SLOPE
    assert score["gates"][gate] is False and score["passed"] is False


def test_illegal_contact_at_horizon_preserves_return_and_fails_safety_gate():
    probes = linear_probes()
    origin = 1.25
    arm = probes["ph"]["native"]
    x = [origin] + [origin + abs(arm["return_m"])] * HORIZON
    arm.update(return_m=reward_m(x, origin), safe=False, illegal=True, terminated=True)
    score = score_cell(probes)
    half = score["radii"]["0.5"]
    assert half["native_slope_m"] == SLOPE and half["forward_gain_m"] == SLOPE / 2
    assert [k for k, passed in score["gates"].items() if not passed] == ["five_native_complete_safe"]
    assert score["passed"] is False


@pytest.mark.parametrize("low,high,radius", [("mh", "ph", "0.5"), ("m1", "p1", "1.0")])
@pytest.mark.parametrize("excess", [0, SLOPE])
def test_each_positive_arm_must_beat_zero_even_with_correct_central_difference(low, high, radius, excess):
    probes = linear_probes()
    shift = SLOPE * ARMS[high] + excess
    for branch in (low, high):
        probes[branch]["native"]["return_m"] -= shift
    score = score_cell(probes)
    result = score["radii"][radius]
    assert result["native_slope_m"] == SLOPE and result["ad_matches_native_25pct"] is True
    assert result["forward_gain_m"] == -excess and result["forward_beats_zero"] is False
    assert [k for k, passed in score["gates"].items() if not passed] == ["both_forward_beats_zero"]
    assert score["passed"] is False


def test_nonfinite_model_derivative_cannot_qualify_valid_native_responses():
    probes = linear_probes()
    probes["zero"]["model"]["directional_derivative"] = math.inf
    score = score_cell(probes)
    assert score["gates"]["both_forward_beats_zero"] is True
    assert score["gates"]["ad_matches_model_both_25pct"] is False and score["passed"] is False


def test_all_nine_cells_and_eighteen_radii_must_be_accounted_for():
    cells = complete_grid()
    assert SEEDS == (94011, 94023, 94037) and RESETS == (95101, 95102, 95103)
    full = summarize(list(reversed(cells)))
    assert full["qualified"] is True and full["cells_planned"] == full["cells_passed"] == 9
    assert full["radii_planned"] == full["native_slopes_available"] == full["ad_native_within_25pct"] == 18

    probes = linear_probes()
    del probes["mh"]
    cells[0]["score"] = score_cell(probes)
    partial = summarize(cells)
    assert partial["qualified"] is False and partial["cells_planned"] == 9 and partial["cells_passed"] == 8
    assert partial["radii_planned"] == 18 and partial["native_slopes_available"] == 17
    assert partial["ad_native_comparable"] == partial["ad_native_within_25pct"] == 17
    assert partial["forward_beats_zero"] == 18  # The missing negative arm does not erase either positive arm.
    with pytest.raises(ValueError):
        summarize(cells[:-1])
    with pytest.raises(ValueError):
        summarize([*cells[:-1], cells[0]])

    del cells[1]["score"]["radii"]["0.5"]
    with pytest.raises(ValueError):
        summarize(cells)


def test_public_wheel_counts_and_each_archived_central_difference():
    recheck = json.loads((ROOT / "reports/cold_start95_wheel_direction.json").read_bytes())
    run = json.loads((ROOT / "reports/cold_start94.json").read_bytes())
    for path, digest in recheck["source_sha256"].items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest
    horizons = recheck["horizons"]
    assert set(horizons) == {"8", "16", "24", "30", "31", "32"}
    rows = {}
    for h, horizon in horizons.items():
        rows[int(h)] = {(r["seed"], r["reset"], r["radius_rad_s"]): r for r in horizon["pairs"]}
        assert len(rows[int(h)]) == len(horizon["pairs"]) == 18
    native_terminal_ticks = []
    with np.load(ROOT / "reports/cold_start94_curves.npz", allow_pickle=False) as curves:
        for cell in run["calibration"]:
            prefix = cell["endpoints_prefix"]
            origin = float(curves[prefix + "_native_m10"][0])
            for radius in (0.1, 0.2):
                key = (cell["seed"], cell["reset"], radius)
                for source in ("native", "model"):
                    minus, plus = (curves[f"{prefix}_{source}_{side}{round(100 * radius)}"].astype(np.float64)
                                   for side in ("m", "p"))
                    if source == "native":
                        native_terminal_ticks.extend((len(minus) - 1, len(plus) - 1))
                    for h, by_cell in rows.items():
                        r = by_cell[key]
                        if min(len(minus), len(plus)) <= h:
                            assert r[source] is None
                            continue
                        # This vector form computes the paired reward difference before averaging.
                        gain = np.mean(np.abs(minus[1:h + 1] - origin) - np.abs(plus[1:h + 1] - origin))
                        assert r[source] == pytest.approx(float(gain) / (2 * radius), rel=1e-12, abs=1e-14)
                        if h == recheck["primary_horizon"]:
                            assert r[source + "_positive_minus_negative_return_mm"] == pytest.approx(1000 * gain)
    assert min(native_terminal_ticks) - 1 == recheck["primary_horizon"] == 30
    earned = {h: sum(r["native"] * r["model"] > 0 for r in by_cell.values()
                     if r["native"] is not None and r["model"] is not None) for h, by_cell in rows.items()}
    assert earned == {8: 13, 16: 8, 24: 7, 30: 6, 31: 6, 32: 6}
    for h, n in earned.items():
        summary = horizons[str(h)]["summary"]["model_vs_native"]
        assert summary == {"comparable": 17 if h == 32 else 18, "same_sign": n, "within_25pct": 0}
    triangle = recheck["h32_triangle"]
    assert triangle["all_available"]["autograd_vs_model"]["within_25pct"] == 14
    assert triangle["all_three_available"]["autograd_vs_native"]["same_sign"] == 2
    assert triangle["all_three_available"]["autograd_vs_native"]["comparable"] == 13
    assert len(triangle["omitted_from_all_three"]) == 5
