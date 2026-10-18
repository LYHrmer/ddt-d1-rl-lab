import cmath
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools30.notch30 import NotchFilter30, coefficients  # noqa: E402


def response(frequency):
    b, a = coefficients()
    z = cmath.exp(-2j * math.pi * frequency / 50)
    return sum(v*z**i for i, v in enumerate(b)) / sum(v*z**i for i, v in enumerate(a))


def test_filter_has_unit_dc_and_nyquist_gain_and_zero_at_measured_frequency():
    assert abs(response(0)-1) < 1e-14
    assert abs(response(25)-1) < 1e-14
    assert abs(response(5)) < 1e-14
    assert abs(response(.5)) > .99
    _, a = coefficients()
    poles = [(-a[1]+sign*cmath.sqrt(a[1]**2-4*a[2]))/2 for sign in (-1,1)]
    assert all(abs(p) < 1 for p in poles)


def test_constant_first_action_has_no_startup_or_cross_channel_transient():
    f = NotchFilter30(); x = tuple((k-8)/10 for k in range(16))
    for _ in range(300):
        assert f.step(x) == pytest.approx(x, abs=1e-14)
    assert f.receipt()['clipped_components'] == 0
    assert f.receipt()['calls'] == 300


def test_actual_recurrence_removes_five_hz_without_removing_dc():
    f = NotchFilter30(); out = []
    for n in range(1000):
        x = .2 + .1*math.sin(2*math.pi*5*n/50)
        out.append(f.step([x]*16)[0])
    assert max(abs(x-.2) for x in out[500:]) < 1e-13


def test_reset_and_causal_prefix_do_not_leak_later_or_prior_episode_samples():
    a, b = NotchFilter30(), NotchFilter30()
    prefix = [[.2*math.sin(i+j) for j in range(16)] for i in range(20)]
    assert [a.step(x) for x in prefix] == [b.step(x) for x in prefix]
    a.step([1.]*16); b.step([-1.]*16)
    a.reset(); b.reset()
    assert a.receipt()['calls'] == b.receipt()['calls'] == 0
    assert a.step(prefix[0]) == b.step(prefix[0]) == tuple(prefix[0])


def test_transient_clipping_preserves_internal_linear_state_and_counts():
    f = NotchFilter30(); f.step([-1.]*16)
    clipped = False
    for _ in range(30):
        issued = f.step([1.]*16)
        assert all(-1 <= x <= 1 for x in issued)
        if any(abs(x)>1 for x in f.last_linear):
            clipped = True
            assert f._y1 == f.last_linear
            assert f._y1 != issued
    assert clipped and f.clipped_components > 0


@pytest.mark.parametrize('value', [[], [0.]*15, [0.]*17, [float('nan')]*16,
                                   [float('inf')]*16, [True]*16, [1.01]*16])
def test_invalid_actions_do_not_advance_state(value):
    f = NotchFilter30()
    with pytest.raises(ValueError):f.step(value)
    assert f.calls == 0 and f.last_linear is None


@pytest.mark.parametrize('values', [(0,5,2),(50,0,2),(50,25,2),(50,5,0),
                                    (True,5,2),(50,float('nan'),2)])
def test_invalid_design_parameters(values):
    with pytest.raises(ValueError):coefficients(*values)
