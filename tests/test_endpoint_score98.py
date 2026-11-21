import ast
import math

import numpy as np
import pytest

from tools98 import publish98 as pub
from tools98.score98 import ARMS, GROUPS, MODELS, PHYSICAL_SCALE, RESETS, SEEDS, TICK, h32_scores, state_error, summarize


def test_public_numpy_groups_and_scales_match_frozen_torch_source_exactly():
    # Load only frozen numeric constants; the tested path has no Torch or physics dependency.
    file = pub.ROOT/"tools52/dynamics52.py"
    names = {"WHEEL_JOINTS", "LEG_JOINTS", "GROUPS", "PHYSICAL_SCALE"}
    nodes = [n for n in ast.parse(file.read_bytes()).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    source = {"np": np}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(file), "exec"), source)
    assert source["GROUPS"] == GROUPS and np.array_equal(source["PHYSICAL_SCALE"].astype(np.float64), PHYSICAL_SCALE)
    assert sorted(i for g in GROUPS.values() for i in g) == list(range(45))


def test_h16_groups_have_equal_weight_and_quaternion_sign_is_not_aligned():
    truth, forecast = np.zeros(45, dtype=np.float64), np.zeros(45, dtype=np.float32)
    forecast[list(GROUPS["joint_position"])] = PHYSICAL_SCALE[list(GROUPS["joint_position"])]
    result = state_error(truth, forecast)
    assert result["group_mse"]["joint_position"] == 1 and result["normalized_mse"] == 1/7
    truth[3], forecast[3] = 1, -1
    result = state_error(truth, forecast)
    assert result["group_mse"]["quaternion"] == (2 / PHYSICAL_SCALE[3])**2/4
    assert result["normalized_mse"] == pytest.approx((result["group_mse"]["quaternion"]+1)/7)
    forecast[0] = math.nan
    assert state_error(truth, forecast)["normalized_mse"] is None
    assert not state_error(truth[:-1], truth)["available"]


def arms_at(seed, reset):
    arms = {}
    actual = np.zeros(45, np.float64)
    for key, alpha in ARMS.items():
        value = -.125 + alpha/128
        arms[key] = {"native": {"return_m": value, "safe": True}, "h16": {}}
        for m, error in (("control", 2.), ("endpoint", 1.)):
            arms[key][m] = {"return_m": value, "safe": True, "directional_derivative": 1/128 if alpha == 0 else None}
            arms[key]["h16"][m] = state_error(actual, np.full(45, error))
    return arms


def grid(factory=arms_at):
    cells = []
    for s in SEEDS:
        for r in RESETS:
            arms = factory(s, r)
            cells.append({"training_seed": s, "reset": r, "arms": arms, "scores": h32_scores(arms)})
    return cells


def test_primary_needs_three_strict_15_arm_medians_and_nine_h32_cells_only():
    # The two H32 worlds are identical; the separately reported slope-median strict decrease fails.
    cells = grid()
    result = summarize(cells)
    assert result["candidate_supported_by_scores"] and result["h16"]["all_three_strictly_improved"]
    assert not result["h32_attribution"]["all_three_slope_medians_strictly_improved"]
    for s in result["h16"]["by_seed"]:
        assert s["worlds"]["endpoint"]["comparable"] == s["worlds"]["control"]["planned"] == 15
    for c in cells[:3]:
        for a in c["arms"].values():
            a["h16"]["endpoint"] = a["h16"]["control"].copy()
    assert not summarize(cells)["candidate_supported_by_scores"]
    cells[0]["arms"]["ph"]["h16"]["endpoint"] = state_error(None, None)
    summary = summarize(cells)["h16"]["by_seed"][0]
    assert summary["strictly_improved"] is None and summary["worlds"]["endpoint"]["comparable"] == 14
    assert summary["worlds"]["endpoint"]["median_normalized_mse"] is None
    with pytest.raises(ValueError):
        summarize(cells[:-1])


def test_observed_terminal_or_later_failed_model_can_supply_h16_but_not_h32():
    state = np.zeros(45, np.float32)
    physical = {"status": "TERMINAL", "ticks": TICK, "error": None}
    assert pub.h16_status(physical, state, True) == "OBSERVED"
    assert pub.h16_status(dict(physical, status="PARTIAL", error="trace exception"), state, True) == "EXCEPTIONAL_NATIVE"
    failed = {"status": "PARTIAL", "ticks": TICK, "error": "LearnFailure: predicted fall", "attempted_model_steps": TICK}
    assert pub.h16_status(failed, state, False) == "REJECTED_AT_16"
    assert pub.h16_status(dict(failed, ticks=TICK+1, attempted_model_steps=TICK+1), state, False) == "ACCEPTED"
    assert pub.h16_status(dict(failed, ticks=TICK-1), np.empty(0), False) == "UNAVAILABLE"
    cells = grid()
    cells[0]["arms"]["zero"]["native"]["safe"] = False
    cells[0]["scores"] = h32_scores(cells[0]["arms"])
    result = summarize(cells)
    assert result["h16"]["all_three_strictly_improved"] and result["worlds"]["endpoint"]["cells_passed"] == 8
    assert not result["candidate_supported_by_scores"]


def test_incomplete_checkpoint_cannot_supply_h16_primary_even_with_valid_arrays():
    arm = {"native": {"h16_status": "OBSERVED", "h16_key": "a"}, "endpoint": {"h16_status": "ACCEPTED", "h16_key": "b"}}
    vectors = {"a": np.zeros(45, np.float64), "b": np.ones(45, np.float32)}
    assert not pub.endpoint_metrics(arm, "endpoint", {"status": "PARTIAL", "arms": {"endpoint": {"status": "PARTIAL"}}}, vectors)["available"]
    assert not pub.endpoint_metrics(arm, "endpoint", {"status": "PARTIAL", "arms": {"endpoint": {"status": "COMPLETE"}}}, vectors)["available"]


def test_public_checker_accounts_for_all_missing_blind_paths_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(pub, "BASE", tmp_path/"absent")
    monkeypatch.setattr(pub, "REPORT", tmp_path/"report.json")
    monkeypatch.setattr(pub, "CURVES", tmp_path/"curves.npz")
    report = pub.publish()
    assert pub.check() == report and not report["qualified_with_provenance"]
    assert report["counts"]["native_arms_planned"] == 45 and report["counts"]["forecast_arms_planned"] == 90
    assert report["counts"]["model_steps_total"] == 0
    with pytest.raises(FileExistsError):
        pub.publish()
