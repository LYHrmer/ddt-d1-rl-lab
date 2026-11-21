import math

import pytest

from tools96.score96 import ARMS, MODELS, RESETS, SEEDS, relative_error, reward_m, score_models, summarize


def arms(control_scale=2.0, paired_scale=1.0, control_offset=0., paired_offset=0.):
    slope, base = 1/128, -1/8
    rows = {}
    for tag, a in ARMS.items():
        rows[tag] = {"native": {"safe": True, "return_m": base + a*slope}}
        for model, scale, offset in zip(MODELS, (control_scale, paired_scale), (control_offset, paired_offset)):
            rows[tag][model] = {"safe": True, "return_m": base + offset + a*slope*scale,
                                "directional_derivative": scale*slope if a == 0 else None}
    return rows


def grid(make):
    return [{"training_seed": seed, "reset": reset, "scores": score_models(make(seed, reset))}
            for seed in SEEDS for reset in RESETS]


def test_symmetric_relative_error_handles_zeros_sign_changes_and_missing_data():
    assert relative_error(3, 4) == relative_error(4, 3) == 0.25
    assert relative_error(0, 4) == 1 and relative_error(-4, 4) == 2
    assert relative_error(0, 0) is None and relative_error(math.inf, 4) is None
    assert relative_error(None, 1) is None


def test_candidate_needs_all_nine_gates_and_both_seedwise_comparisons():
    result = summarize(grid(lambda s, r: arms()))
    assert result["candidate_supported_by_scores"] and all(result["gates"].values())
    assert result["worlds"]["control"]["ad_native_25pct"] == {"comparable": 18, "passed": 0}
    assert result["worlds"]["paired"]["ad_native_25pct"] == {"comparable": 18, "passed": 18}
    for entry in result["by_seed"]:
        assert entry["slope_median_strictly_improved"] is True and entry["zero_median_noninferior"] is True
        for name, expected in (("control", .5), ("paired", 0.)):
            m = entry["models"][name]
            assert m["slope_error_median"] == expected and m["slope_error_available"] == 6
            assert m["zero_error_median_m"] == 0 and m["zero_error_available"] == 3

    one_tie = summarize(grid(lambda s, r: arms(control_scale=1.0 if s == SEEDS[0] else 2.0)))
    assert one_tie["worlds"]["paired"]["cells_passed"] == 9 and not one_tie["candidate_supported_by_scores"]
    assert not one_tie["by_seed"][0]["slope_median_strictly_improved"]
    worse_baseline = summarize(grid(lambda s, r: arms(paired_offset=.001 if s == SEEDS[0] else 0.)))
    assert not worse_baseline["by_seed"][0]["zero_median_noninferior"] and not worse_baseline["candidate_supported_by_scores"]


def test_short_arms_preserve_fixed_denominators_and_do_not_supply_partial_medians():
    cells = grid(lambda s, r: arms())
    rows = arms()
    rows["zero"]["paired"].update(return_m=reward_m([1.1]*32, 1.1), safe=False, directional_derivative=None)
    cells[0]["scores"] = score_models(rows)
    result = summarize(cells)
    candidate = result["worlds"]["paired"]
    assert candidate["cells_planned"] == 9 and candidate["radii_planned"] == 18 and candidate["cells_passed"] == 8
    seed = result["by_seed"][0]
    assert seed["models"]["paired"]["slope_error_available"] == 4 and seed["models"]["paired"]["zero_error_available"] == 2
    assert seed["models"]["paired"]["slope_error_median"] is None and seed["models"]["paired"]["zero_error_median_m"] is None
    assert seed["zero_median_noninferior"] is None and seed["slope_median_strictly_improved"] is None
    assert not result["candidate_supported_by_scores"]
    with pytest.raises(ValueError):
        summarize(cells[:-1])


def test_shared_terminal_h32_return_remains_numeric_and_both_worlds_fail_native_safety():
    rows = arms(control_scale=1.)
    rows["m1"]["native"]["safe"] = False
    scores = score_models(rows)
    for score in scores.values():
        assert score["radii"]["1.0"]["native_slope_m"] == 1/128
        assert not score["passed"] and not score["gates"]["five_native_complete_safe"]
    # A correct centred slope can coexist with a positive arm which fails to beat zero.
    rows = arms()
    for tag in ("mh", "ph"):
        rows[tag]["native"]["return_m"] -= 1/256
    score = score_models(rows)["paired"]
    assert score["radii"]["0.5"]["ad_matches_native_25pct"] and not score["gates"]["both_forward_beats_zero"]


def test_public_checker_preserves_missing_cells_and_refuses_duplicate_publication(tmp_path, monkeypatch):
    from tools96 import publish96 as publisher

    monkeypatch.setattr(publisher, "BASE", tmp_path / "unrun")
    monkeypatch.setattr(publisher, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(publisher, "CURVES", tmp_path / "curves.npz")
    report = publisher.publish()
    assert publisher.check() == report
    assert report["counts"]["model_steps_total"] == 0 and report["counts"]["native_episodes"] == 0
    assert report["summary"]["worlds"]["paired"]["radii_planned"] == 18 and not report["qualified_with_provenance"]
    assert all(s["slope_median_strictly_improved"] is None for s in report["summary"]["by_seed"])
    with pytest.raises(FileExistsError):
        publisher.publish()
