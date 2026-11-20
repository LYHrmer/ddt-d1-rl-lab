import numpy as np
import pytest

from tools88.probe88 import PLAN, WHEELS, apply_intervention, mean_score_speed, passes_pair
from tools87.run87 import read


def test_first_issue_only_offsets_wheels_and_preserves_actor_output():
    plan = read(PLAN)
    raw = np.linspace(-0.8, 0.8, 16, dtype=np.float32)
    raw[WHEELS] = [0.95, -0.4, 0.3, 0.85]
    saved = raw.copy()
    previous, clipped = apply_intervention(raw, 179, "wheel_plus", plan)
    np.testing.assert_array_equal(previous, raw)
    assert not clipped.any()
    original, _ = apply_intervention(raw, 180, "original", plan)
    np.testing.assert_array_equal(original, raw)
    changed, clipped = apply_intervention(raw, 180, "wheel_plus", plan)
    legs = np.setdiff1d(np.arange(16), WHEELS)
    np.testing.assert_array_equal(changed[legs], raw[legs])
    np.testing.assert_allclose(changed[WHEELS], [1.0, -0.3, 0.4, 0.95], atol=1e-7)
    np.testing.assert_array_equal(clipped, [True, False, False, False])
    np.testing.assert_array_equal(raw, saved)


def test_scoring_window_uses_100_measured_endpoints():
    endpoints = np.arange(301, dtype=float)
    assert mean_score_speed(endpoints, [200, 300]) == np.mean(np.arange(201, 301))
    assert mean_score_speed(endpoints[:-1], [200, 300]) is None
    endpoints[235] = np.nan
    with pytest.raises(ValueError):
        mean_score_speed(endpoints, [200, 300])


def test_no_threshold_rounding_and_both_arms_must_complete_cleanly():
    assert passes_pair(0.0200, 0.02, True, True)
    assert not passes_pair(0.0199999, 0.02, True, True)
    assert not passes_pair(0.04, 0.02, True, False)
    assert not passes_pair(None, 0.02, False, True)
