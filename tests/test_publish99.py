"""The blind round-99 gates and public verifier need only NumPy and synthetic inputs."""
import copy

import numpy as np
import pytest

from tools95.score95 import reward_m
from tools98.score98 import state_error
from tools99 import publish99 as pub
from tools99.score99 import h32_scores, summarize


def grid(control_ad):
    cells = []
    for seed in pub.SEEDS:
        for reset in pub.RESETS:
            arms = {}
            for tag, alpha in pub.ARMS.items():
                native = -.125 + alpha/128
                a = {"native": {"safe": True, "return_m": native}, "h16": {}}
                for model, predicted_ad in (("control", control_ad), ("late", 1/128)):
                    a[model] = {"safe": True, "return_m": native, "directional_derivative": predicted_ad if tag == "zero" else None}
                    a["h16"][model] = state_error(np.zeros(45), np.full(45, 1. if model == "late" else 2.))
                arms[tag] = a
            cells.append({"training_seed": seed, "reset": reset, "arms": arms, "scores": h32_scores(arms)})
    return cells


def test_sensitivity_attribution_and_two_registered_promotion_gates_are_distinct():
    tied = summarize(grid(1/128))
    assert tied["candidate_supported_by_scores"] and not tied["slope_improvement_supported"]
    assert not tied["all_three_criteria_passed"]
    cells = grid(2/128)
    assert summarize(cells)["all_three_criteria_passed"]
    cells[0]["arms"]["ph"]["native"]["safe"] = False
    cells[0]["scores"] = h32_scores(cells[0]["arms"])
    result = summarize(cells)
    assert result["slope_improvement_supported"] and result["h16"]["all_three_strictly_improved"]
    assert result["worlds"]["late"]["cells_passed"] == 8 and not result["candidate_supported_by_scores"]
    for c in cells[:3]:
        for a in c["arms"].values():
            a["h16"]["late"] = a["h16"]["control"]
    assert not summarize(cells)["h16"]["all_three_strictly_improved"]


def test_missing_comparisons_are_kept_in_the_denominator_and_only_new_resets_score():
    cells = grid(2/128)
    cells[0]["arms"]["ph"]["native"]["return_m"] = None
    cells[0]["arms"]["mh"]["h16"]["late"] = state_error(None, None)
    cells[0]["scores"] = h32_scores(cells[0]["arms"])
    result = summarize(cells)
    seed = result["h32_attribution"]["by_seed"][0]
    assert seed["worlds"]["late"]["slope_errors_available"] == 5
    assert seed["worlds"]["late"]["median_relative_slope_error"] is None and seed["slope_median_strictly_improved"] is None
    items = result["h16"]["by_seed"][0]["worlds"]["late"]
    assert items["comparable"] == 14 and items["planned"] == 15 and items["median_normalized_mse"] is None
    assert not result["all_three_criteria_passed"]
    cells[0]["reset"] = 98201
    with pytest.raises(ValueError, match="round-99 blind cells"):
        summarize(cells)


def test_training_normalization_uses_only_each_seeds_three_collection_pairs():
    resets = (96101, 96102, 96103)
    bank = {}
    for i, reset in enumerate(resets):
        for tag, sign in (("m1", -1), ("p1", 1)):
            key = f"collection_94011_{reset}_{tag}_x"
            curve = np.full(33, i + .5)
            curve[2:] += sign * np.arange(1, 32) / 4096
            bank[key] = curve
            bank[key+"_flags"] = np.zeros((32, 4), bool)
    bank["collection_94011_96103_m1_x_flags"][-1, [1, 2]] = True
    target, flagged = pub.training_labels(bank, 94011, resets)
    np.testing.assert_array_equal(target, np.tile(2*np.arange(16, 32)/4096, (3, 1)))
    assert flagged == [{"reset": 96103, "arm": "m1", "tick": 32, "flags": ["illegal", "terminated"]}]
    bank["collection_94011_96103_m1_x_flags"][-2, 2] = True
    with pytest.raises(ValueError, match="training prefix"):
        pub.training_labels(bank, 94011, resets)


def synthetic_terminal(base, recorded_reset):
    folder = base/"native_01/seed_94011/reset_99201/zero"
    (folder/"native").mkdir(parents=True)
    state = np.zeros((33, 45), np.float64)
    state[:, 0] = 1.1 - np.arange(33)/256
    actions = np.zeros((32, 16), np.float32)
    flags = np.zeros((32, 4), bool)
    flags[-1, [1, 2]] = True
    path = folder/"native/trace.npz"
    np.savez(path, state45=state, reset_seed=recorded_reset, requested_ticks=32, completed_transitions=32,
             explore_seed=-1, exploration16=actions, command3=np.zeros((32, 3)), wheel_offset_rad_s=0.,
             applied16=actions, issued16=actions, raw16=actions, obs643=np.zeros((33, 643), np.float32),
             filtered3=np.zeros((33, 3)), ray_due=np.zeros(33, bool), ray_raw16=np.zeros((33, 16), np.float32),
             policy_steps=32, servo_steps=160, native_steps=320, **{k: flags[:, i] for i, k in enumerate(pub.FLAGS)})
    pub.write(folder/"native.json", {"trace_file": "native/trace.npz", "trace_sha256": pub.digest(path), "error": None,
              "ticks": 32, "safe": False, "return_m": reward_m(state[:, 0], state[0, 0]),
              **{k: bool(flags[:, i].any()) for i, k in enumerate(pub.FLAGS)}})


def test_public_only_readback_accounts_for_terminal_and_missing_paths_and_refuses_overwrite(tmp_path):
    base, output, curves_path = tmp_path/"raw", tmp_path/"report.json", tmp_path/"curves.npz"
    synthetic_terminal(base, 99201)
    report = pub.publish(base, output, curves_path)
    assert pub.check(output, curves_path) == report
    counts = report["counts"]
    assert counts["native_arms_planned"] == 45 and counts["forecast_arms_planned"] == 90
    assert counts["blind_native_episodes"] == 1 and counts["blind_forecasts"] == 0 and not report["sources_complete"]
    arm = report["cells"][0]["arms"]["zero"]
    assert arm["native"]["status"] == "TERMINAL" and arm["native"]["return_m"] < 0 and arm["native"]["h16_status"] == "OBSERVED"
    assert not arm["h16"]["late"]["available"] and not report["qualified_with_provenance"]
    with pytest.raises(FileExistsError):
        pub.publish(base, output, curves_path)
    tampered = copy.deepcopy(report)
    tampered["cells"][0]["arms"]["zero"]["native"]["receipt"]["safe"] = True
    with np.load(curves_path, allow_pickle=False) as data, pytest.raises(ValueError, match="original|recorded bytes"):
        pub.verify_public(tampered, data)


def test_raw_native_previous_rounds_blind_reset_is_rejected_before_publishing(tmp_path):
    synthetic_terminal(tmp_path/"raw", 98201)
    with pytest.raises(ValueError, match="native prefixes"):
        pub.publish(tmp_path/"raw", tmp_path/"report.json", tmp_path/"curves.npz")
    assert not (tmp_path/"report.json").exists() and not (tmp_path/"curves.npz").exists()


@pytest.mark.parametrize(("ticks", "h16"), [(16, "REJECTED_AT_16"), (17, "ACCEPTED"), (32, "ACCEPTED")])
def test_forecast_prefix_and_neutral_derivative_remain_bound_to_the_original_receipt(tmp_path, ticks, h16):
    x0 = 1.1
    state = np.zeros((ticks+1, 45), np.float32)
    state[:, 0] = x0 - np.arange(ticks+1)/256
    start = {"state45": np.zeros((1, 45), np.float64)}
    start["state45"][0, 0] = x0
    path = tmp_path/"zero.npz"
    np.savez(path, state45=state)
    full = ticks == 32
    raw = {"seed": 94011, "reset": 99201, "model": "late", "alpha": 0., "file": path.name, "ticks": ticks,
           "return_m": reward_m(state[:, 0], x0), "safe": full, "attempted_model_steps": ticks, "completed_model_steps": ticks,
           "backward_attempted": int(full), "backward_completed": int(full), "directional_derivative": .001 if full else None,
           "state_sha256": pub.digest(path), "checkpoint_sha256": "a"*64, "error": None if full else "LearnFailure: predicted fall"}
    pub.write(tmp_path/"zero.json", raw)
    curves = {}
    row = pub.model_arm(tmp_path, 94011, 99201, "late", "zero", start, curves)
    stages = {"predictions": {"status": "UNSEALED"}}
    pub.verify_sample(row, curves, set(), stages, x0, 94011, 99201, "zero", "late")
    assert row["h16_status"] == h16 and row["safe"] == full
    if full:
        # Editing both public representations together still fails the original receipt check.
        row["directional_derivative"] = -.001
        curves[row["ad_key"]][0] = -.001
        with pytest.raises(ValueError, match="derivative"):
            pub.verify_sample(row, curves, set(), stages, x0, 94011, 99201, "zero", "late")
