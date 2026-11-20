import numpy as np
import pytest

from tools90.stand90 import LEG_INDICES, WHEEL_INDICES, motion_metrics, pair_metrics, run, verify_actions, wheel_decision


SCALES = np.full(4, 5.)


def action(wheels):
    raw = np.linspace(-.9, .9, 16, dtype=np.float32)
    raw[WHEEL_INDICES] = np.asarray(wheels) / SCALES
    return raw


def test_common_increment_has_physical_sign_and_preserves_all_wheel_differences():
    raw = action([-3., -2., 0., 1.])
    saved = raw.copy()
    issued, requested, feasible = wheel_decision(raw, SCALES, "common", 0, 50)
    assert requested == pytest.approx(1.) and feasible == requested
    np.testing.assert_allclose(issued[WHEEL_INDICES] * SCALES, [-2., -1., 1., 2.], atol=3e-7)
    np.testing.assert_array_equal(issued[LEG_INDICES], saved[LEG_INDICES])
    np.testing.assert_array_equal(raw, saved)


@pytest.mark.parametrize("wheels,requested,feasible", [
    ([-5., -4., -3., 4.75], 1.8125, .25),
    ([-5., -4., -3., 5.], 1.75, 0.),
    ([-1., 1., 2., 3.], 0., 0.),
])
def test_headroom_or_positive_mean_at_last_stand_issue(wheels, requested, feasible):
    raw = action(wheels)
    issued, r, d = wheel_decision(raw, SCALES, "common", 49, 50)
    assert r == pytest.approx(requested) and d == pytest.approx(feasible)
    np.testing.assert_allclose((issued[WHEEL_INDICES] - raw[WHEEL_INDICES]) * SCALES, feasible, atol=3e-7)
    assert np.abs(issued).max() <= 1.
    if not feasible:
        np.testing.assert_array_equal(issued, raw)
    for arm, tick in (("original", 0), ("original", 49), ("common", 50), ("common", 99)):
        untouched, r, d = wheel_decision(raw, SCALES, arm, tick, 50)
        np.testing.assert_array_equal(untouched, raw)
        assert r == d == 0.


def test_fifo_carries_final_hold_command_into_first_moving_interval():
    raw = np.tile(action([-1., -2., -3., -4.]), (52, 1))
    issued, requested, feasible = zip(*(wheel_decision(row, SCALES, "common", i, 50) for i, row in enumerate(raw)))
    applied = np.vstack((np.zeros(16), issued[:-1]))
    rows = {"policy_raw": raw, "issued": issued, "applied": applied, "requested_rad_s": requested,
            "feasible_rad_s": feasible, "applied_wheel_target_rad_s": applied[:, WHEEL_INDICES] * SCALES}
    result = verify_actions(rows, SCALES, "common", 50)
    assert result["effective_issued_ticks"] == 50
    assert result["effective_applied_ticks_first_s"] == 49
    assert result["applied_increment_first_moving_tick_rad_s"] == pytest.approx(2.5)
    assert not np.array_equal(applied[50], raw[50])
    np.testing.assert_array_equal(applied[51], raw[50])
    rows["applied"] = np.asarray(issued)
    with pytest.raises(ValueError, match="FIFO"):
        verify_actions(rows, SCALES, "common", 50)


def test_rollback_uses_world_endpoint_at_one_second_and_keeps_peak_separate():
    x = np.full(100, 1.12)
    x[12], x[49], x[50] = .8, 1.06, 1.089
    result = motion_metrics(1.10, x, 50, 100)
    assert result["rollback_m"] == pytest.approx(.04)
    assert result["largest_stand_rollback_m"] == pytest.approx(.3)
    assert result["signed_moving_dx_m"] == pytest.approx(.06)
    assert motion_metrics(1.10, x[:49], 50, 100)["rollback_m"] is None
    assert motion_metrics(1.10, x[:99], 50, 100)["signed_moving_dx_m"] is None
    assert motion_metrics(1.10, np.full(100, 1.12), 50, 100)["rollback_m"] == 0.


def test_pair_is_clean_matched_and_passes_the_unrounded_threshold():
    base = {"parent_seed": 87011, "reset_seed": 90101, "initial_state_sha256": "a",
            "policy_sha256": "b", "clean": True}
    original = dict(base, arm="original", rollback_m=.03)
    common = dict(base, arm="common", rollback_m=0.)
    assert pair_metrics(original, common, .03)["passed"]
    assert not pair_metrics(dict(original, rollback_m=np.nextafter(.03, 0)), common, .03)["passed"]
    assert not pair_metrics(original, dict(common, clean=False), .03)["passed"]
    incomplete = pair_metrics(original, dict(common, rollback_m=None, clean=False), .03)
    assert incomplete["rollback_reduction_m"] is None and not incomplete["passed"]
    with pytest.raises(ValueError, match="initial state"):
        pair_metrics(original, dict(common, initial_state_sha256="changed"), .03)


def test_existing_attempt_is_never_reused_even_if_empty(tmp_path):
    with pytest.raises(FileExistsError):
        run(tmp_path)
