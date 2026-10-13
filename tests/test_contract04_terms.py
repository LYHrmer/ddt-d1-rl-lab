"""Unit tests for the contract 04 reward terms and the frozen 03d command course.

Pure arithmetic and pure RNG: no MuJoCo model, no environment, no physics step,
so this runs from a clean clone. These are the properties the contracts claim
about their single added term, in executable form.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))

from d1_rl.task04a import anchor_penalty_raw  # noqa: E402
from d1_rl.task04b import wheel_action_penalty_raw  # noqa: E402

CONFIG_04A = json.loads((ROOT/'training/configs/pilot04a.json').read_text())['reward']
CONFIG_04B = json.loads((ROOT/'training/configs/pilot04b.json').read_text())['reward']
KNEE = CONFIG_04A['standstill_anchor_displacement_knee_m']
CAP = CONFIG_04A['standstill_anchor_displacement_saturation_m']
WEIGHT_04A = CONFIG_04A['standstill_anchor_displacement_weight']
SATURATION = CONFIG_04B['wheel_action_mean_squared_saturation']
WEIGHT_04B = CONFIG_04B['wheel_action_mean_squared_weight']

# Per-tick values measured in the 03d and 04a native traces, and the
# approximate value at initialisation with log_std_init=-1.
MEASURED_WHEEL_MEAN_SQUARE = {'03d_forward': .06516, '03d_reverse': .05481,
                              '04a_turn': .17363, 'init': .135}


def positive_floor(reward):
    """vx tracking + wz tracking + alive, the per-second floor at zero command."""
    return (reward['vx_tracking_weight']+reward['wz_tracking_weight']
            + reward['alive_weight'])*.95


# --- 04a anchor displacement term -------------------------------------------

def test_anchor_is_zero_only_at_zero_distance():
    assert anchor_penalty_raw(0., KNEE, CAP) == 0.
    for d in (1e-6, .001, .01, .05, .2):
        assert anchor_penalty_raw(d, KNEE, CAP) > 0.


def test_anchor_is_quadratic_below_the_knee():
    for d in (.01, .02, .03):
        assert np.isclose(anchor_penalty_raw(d, KNEE, CAP), d*d/KNEE)


def test_anchor_value_and_slope_are_continuous_at_the_knee():
    assert np.isclose(anchor_penalty_raw(KNEE, KNEE, CAP), KNEE)
    eps = 1e-7
    left = (anchor_penalty_raw(KNEE, KNEE, CAP)-anchor_penalty_raw(KNEE-eps, KNEE, CAP))/eps
    right = (anchor_penalty_raw(KNEE+eps, KNEE, CAP)-anchor_penalty_raw(KNEE, KNEE, CAP))/eps
    assert np.isclose(left, 2., atol=1e-3) and np.isclose(right, 2., atol=1e-3)


def test_anchor_is_linear_above_the_knee():
    for d in (.06, .08, .095):
        assert np.isclose(anchor_penalty_raw(d, KNEE, CAP), 2.*d-KNEE)


def test_anchor_saturates_and_is_monotone():
    at_cap = anchor_penalty_raw(CAP, KNEE, CAP)
    for d in (CAP, CAP+.01, 1., 100.):
        assert anchor_penalty_raw(d, KNEE, CAP) == at_cap
    rising = [anchor_penalty_raw(d, KNEE, CAP) for d in (0., .01, .03, .06, .09)]
    assert all(a < b for a, b in zip(rising, rising[1:]))


def test_anchor_uses_magnitude_so_sign_does_not_matter():
    assert anchor_penalty_raw(-.03, KNEE, CAP) == anchor_penalty_raw(.03, KNEE, CAP)


def test_anchor_worst_case_stays_below_the_positive_floor():
    """Otherwise ending an episode early could look cheaper than standing still."""
    worst = anchor_penalty_raw(CAP, KNEE, CAP)*abs(WEIGHT_04A)
    assert worst < positive_floor(CONFIG_04A)


def test_anchor_contribution_at_the_gate_threshold():
    # The gate is 0.05 m and the knee is placed on it, so raw == knee there.
    assert np.isclose(anchor_penalty_raw(.05, KNEE, CAP)*WEIGHT_04A, -1.0)


@pytest.mark.parametrize('knee,cap', [(0., .1), (-.05, .1), (.1, .1), (.1, .05)])
def test_anchor_rejects_bad_parameters(knee, cap):
    with pytest.raises(ValueError):
        anchor_penalty_raw(.02, knee, cap)


# --- 04b wheel action magnitude term ----------------------------------------

def test_wheel_term_is_zero_only_for_an_all_zero_action():
    assert wheel_action_penalty_raw([0.]*4, SATURATION) == 0.
    for actions in ([.01, 0., 0., 0.], [0., -.01, 0., 0.], [.05, -.05, .05, -.05]):
        assert wheel_action_penalty_raw(actions, SATURATION) > 0.


def test_wheel_term_equals_the_mean_square_below_saturation():
    for actions in ([.1, -.2, .3, -.4], [.2]*4, [0., .1, 0., -.1]):
        expected = float(np.mean(np.square(actions)))
        assert np.isclose(wheel_action_penalty_raw(actions, SATURATION), expected)


def test_wheel_term_penalises_opposed_patterns_not_just_net_drive():
    """03d held a net bias near zero while each wheel ran at 1 to 2 rad/s."""
    opposed = [.3, -.3, .3, -.3]          # net zero, large magnitudes
    assert np.isclose(float(np.mean(opposed)), 0.)
    assert wheel_action_penalty_raw(opposed, SATURATION) > .08


def test_wheel_term_saturates():
    assert wheel_action_penalty_raw([1.]*4, SATURATION) == SATURATION
    assert wheel_action_penalty_raw([.9]*4, SATURATION) == SATURATION


def test_wheel_term_gradient_stays_live_over_the_whole_measured_range():
    """An earlier 0.09 saturation put 04a's 0.174 and init's 0.135 in the flat
    region; the static checks caught it before training."""
    for name, value in MEASURED_WHEEL_MEAN_SQUARE.items():
        assert value < SATURATION, name


def test_wheel_term_orders_the_measured_behaviours_correctly():
    m = MEASURED_WHEEL_MEAN_SQUARE
    assert m['04a_turn'] > m['03d_forward'] > m['03d_reverse']
    costs = {k: min(v, SATURATION)*abs(WEIGHT_04B) for k, v in m.items()}
    assert all(c > .5 for k, c in costs.items() if k != 'init')


def test_wheel_term_worst_case_stays_below_the_positive_floor():
    worst = SATURATION*abs(WEIGHT_04B)
    assert worst < positive_floor(CONFIG_04B)


@pytest.mark.parametrize('actions,saturation', [
    ([0.]*4, 0.), ([0.]*4, -.1), ([0.]*4, 1.5),
    ([0.]*3, .25), ([0.]*5, .25),
])
def test_wheel_term_rejects_bad_inputs(actions, saturation):
    with pytest.raises(ValueError):
        wheel_action_penalty_raw(actions, saturation)


# --- the two terms are siblings, not a chain --------------------------------

def test_04b_does_not_inherit_the_04a_term():
    assert any('anchor' in k for k in CONFIG_04A)
    assert not any('anchor' in k for k in CONFIG_04B)
    assert any('wheel_action' in k for k in CONFIG_04B)
    assert not any('wheel_action' in k for k in CONFIG_04A)


def test_both_configs_add_exactly_one_term_to_the_frozen_base():
    base = json.loads((ROOT/'training/configs/pilot03d.json').read_text())['reward']
    for config, expected in ((CONFIG_04A, 3), (CONFIG_04B, 2)):
        added = set(config)-set(base)
        removed = set(base)-set(config)
        changed = [k for k in base if k in config and base[k] != config[k]]
        assert len(added) == expected and not removed and not changed


def test_both_terms_are_penalties():
    assert WEIGHT_04A < 0. and WEIGHT_04B < 0.


# --- frozen 03d command course ----------------------------------------------

def test_03d_course_spec_is_unchanged_in_both_contract_04_configs():
    base = json.loads((ROOT/'training/configs/pilot03d.json').read_text())
    for name in ('pilot04a.json', 'pilot04b.json'):
        config = json.loads((ROOT/f'training/configs/{name}').read_text())
        assert config['commands'] == base['commands'], name


def test_03d_course_guarantees_two_motion_to_zero_transitions():
    spec = json.loads(
        (ROOT/'training/configs/pilot03d.json').read_text())['commands']['paired_schedule']
    assert spec['episode_ticks'] == 1000
    assert spec['first_zero_ticks'] == 50
    assert spec['mode_probabilities'] == {'translation': .5, 'turn': .5}
    assert spec['positive_sign_probability'] == .5
    # Translation: 50 zero, D drive, G zero, D reverse, then zero to 1000.
    translation = spec['translation']
    low, high = translation['motion_ticks_inclusive']
    gap_low, gap_high = translation['middle_zero_ticks_inclusive']
    assert (low, high) == (200, 300) and (gap_low, gap_high) == (100, 150)
    # Worst case still leaves a trailing zero segment of at least 200 ticks.
    assert 1000-50-2*high-gap_high >= 200
    # Turn mode: two sign reversals plus a middle gap, same trailing bound.
    turn = spec['turn']
    tlow, thigh = turn['motion_ticks_inclusive']
    assert (tlow, thigh) == (100, 150)
    assert 1000-50-4*thigh-turn['middle_zero_ticks_inclusive'][1] >= 200


def test_03d_course_motion_share_stays_between_40_and_60_percent():
    spec = json.loads(
        (ROOT/'training/configs/pilot03d.json').read_text())['commands']['paired_schedule']
    ticks = spec['episode_ticks']
    translation = spec['translation']['motion_ticks_inclusive']
    turn = spec['turn']['motion_ticks_inclusive']
    # Translation moves for 2D, turn for 4 of the two paired durations.
    for low, high, factor in ((translation[0], translation[1], 2), (turn[0], turn[1], 4)):
        assert .4 <= factor*low/ticks and factor*high/ticks <= .6
