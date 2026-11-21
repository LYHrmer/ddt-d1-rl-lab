import numpy as np
import pytest

from tools94.score94 import OFFSETS, agrees, calibration, control


def trajectories():
    ticks = np.arange(51)
    return {k: 1.10 - .003*ticks + .001*ticks*(k/100) for k in OFFSETS}


def test_two_radii_use_physical_target_units_and_signed_effects():
    native = trajectories()
    result = calibration(native, native, .0165, dict.fromkeys(OFFSETS, True))
    assert result["passed"]
    assert result["radii"]["10"]["native_effect_mm"] == pytest.approx(10.)
    assert result["radii"]["20"]["native_gradient"] == pytest.approx(.0165)
    assert not calibration(native, native, -.0165, dict.fromkeys(OFFSETS, True))["passed"]


def test_shared_predictions_cannot_pass_on_direction_alone():
    x = trajectories()
    off = {k: v - np.linspace(0, .050, len(v)) for k,v in x.items()}
    score = calibration(x, off, .0165, dict.fromkeys(OFFSETS, True))
    assert score["gates"]["effect_sign"] and not score["gates"]["all_arms_within_30mm"]
    assert not score["passed"]
    assert not calibration(x, {**off, 20:off[20][:8]}, None, dict.fromkeys(OFFSETS, True))["available"]
    assert not agrees(0., 0.)


def test_forward_launch_does_not_count_as_standing():
    a = 1.-.002*np.arange(101)
    b = 1.+.004*np.arange(101)
    score = control(a,b,True)
    assert score["rollback_reduction_mm"] == pytest.approx(100.)
    assert not score["passed"]
    assert control(a, np.ones(101), True)["passed"]
    assert not control(a, np.ones(101), False)["passed"]
