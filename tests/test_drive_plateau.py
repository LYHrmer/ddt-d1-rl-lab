"""Tests for tools04/analyze_drive_plateau04.py, the receipt behind contract 05M's revision 1.

The revision says 04b's drive phase settles at about 0.13 m/s and that the wheel
torque there is far from the limit. Those statements are only as good as the
helpers that produce them, so the helpers are tested on cases with known answers,
and the numbers quoted in the contract are checked against the receipt. Nothing
here loads a model or runs physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from _workstation import needs_workstation_evidence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))

import analyze_drive_plateau04 as plateau  # noqa: E402

RECEIPT = ROOT/'evidence/career_contract_04/drive_plateau_analysis_01/report.json'


# --- signed speed and trend ---------------------------------------------------
def test_signed_speed_follows_the_command_direction():
    np.testing.assert_array_equal(plateau.signed_speed([.1, -.2], 1.), [.1, -.2])
    np.testing.assert_array_equal(plateau.signed_speed([.1, -.2], -1.), [-.1, .2])


def test_trend_of_a_ramp_is_its_slope_and_of_a_plateau_is_zero():
    ramp = np.linspace(0., 1., 101)             # 0.01 per tick = 0.5 per second
    assert plateau.speed_trend(ramp, 10, 60) == pytest.approx(.5)
    assert plateau.speed_trend(np.full(101, .13), 10, 60) == pytest.approx(0., abs=1e-12)
    assert plateau.speed_trend(ramp[::-1], 10, 60) == pytest.approx(-.5)


def test_trend_uses_only_the_requested_ticks():
    speed = np.zeros(100); speed[:30] = np.arange(30)
    assert plateau.speed_trend(speed, 40, 90) == pytest.approx(0., abs=1e-12)


# --- time to reach a fraction of the command -----------------------------------
def test_time_to_fraction_counts_from_the_drive_start_and_includes_the_crossing_tick():
    speed = np.zeros(500); speed[100:] = np.linspace(0., .2, 400)
    t = plateau.time_to_fraction(speed, .2, .5)
    first = int(np.flatnonzero(speed[100:400] >= .1)[0])
    assert t == pytest.approx(first*.02)


def test_time_to_fraction_is_none_when_the_plateau_is_below_the_target():
    speed = np.zeros(500); speed[100:] = .13
    assert plateau.time_to_fraction(speed, .2, .9) is None
    assert plateau.time_to_fraction(speed, .2, .65) == 0.


def test_time_to_fraction_ignores_speed_after_the_cut():
    speed = np.zeros(500); speed[450:] = .3
    assert plateau.time_to_fraction(speed, .2, .9) is None


def test_time_to_fraction_ignores_speed_before_the_drive_starts():
    speed = np.zeros(500); speed[:100] = .3
    assert plateau.time_to_fraction(speed, .2, .9) is None


# --- wheel push -----------------------------------------------------------------
def action_with(value, ticks=400):
    a = np.zeros((ticks, 16)); a[:, [3, 7, 11, 15]] = value
    return a


def test_wheel_push_matches_a_hand_calculation():
    speed = np.full(400, .087)                  # rolling rate 1 rad/s
    out = plateau.wheel_push(action_with(.4), speed, 1., 350, 400)
    assert out['wheel_target_rad_s'] == pytest.approx(2.)           # .4 * 5
    assert out['rolling_rate_rad_s'] == pytest.approx(1.)
    assert out['target_minus_rolling_rad_s'] == pytest.approx(1.)
    assert out['torque_per_wheel_n_m'] == pytest.approx(.5)         # kd * gap
    assert out['fraction_of_torque_limit'] == pytest.approx(.5/12.)


def test_reverse_direction_flips_the_action_sign_but_not_the_torque_magnitude():
    forward = plateau.wheel_push(action_with(.4), np.full(400, .087), 1., 350, 400)
    reverse = plateau.wheel_push(action_with(-.4), np.full(400, .087), -1., 350, 400)
    # speed passed in is already signed along the command, so the same physical case gives the same gap
    assert reverse['torque_per_wheel_n_m'] == pytest.approx(forward['torque_per_wheel_n_m'])
    assert reverse['wheel_target_rad_s'] == pytest.approx(forward['wheel_target_rad_s'])


def test_mean_abs_action_does_not_cancel_opposite_signs():
    a = action_with(.4); a[:, 3] = -.4
    out = plateau.wheel_push(a, np.full(400, .087), 1., 350, 400)
    assert out['mean_abs_action'] == pytest.approx(.4)
    assert out['wheel_target_rad_s'] == pytest.approx(.2*5/1.)       # mean of (-.4,.4,.4,.4) = .2


def test_only_the_four_wheel_columns_are_read():
    a = action_with(.4); a[:, 0] = 1.; a[:, 5] = -1.
    assert plateau.wheel_push(a, np.full(400, .087), 1., 350, 400)['mean_abs_action'] == pytest.approx(.4)


def test_only_the_requested_window_is_read():
    a = action_with(.4); a[:350] = 1.
    assert plateau.wheel_push(a, np.full(400, .087), 1., 350, 400)['mean_abs_action'] == pytest.approx(.4)


# --- the whole analysis on a synthetic trace -------------------------------------
def write_trace(path, plateau_speed, direction=1., action=.45):
    vx = np.zeros(600); vx[100:] = direction*plateau_speed
    act = np.zeros((600, 16)); act[:, [3, 7, 11, 15]] = direction*action
    np.savez_compressed(path, body_vx_m_s=vx, issued_action=act)


def test_a_forward_plateau_is_reported_as_a_fraction_of_the_command(tmp_path):
    write_trace(tmp_path/'t.npz', .13)
    out = plateau.analyse(tmp_path/'t.npz', 1.)
    assert out['last_second_mean_speed_m_s'] == pytest.approx(.13)
    assert out['last_second_fraction_of_command'] == pytest.approx(.65)
    assert out['seconds_to_90_percent_of_command'] is None
    assert out['last_second_trend_m_s_per_s'] == pytest.approx(0., abs=1e-12)


def test_a_reverse_plateau_is_reported_with_the_same_sign_convention(tmp_path):
    write_trace(tmp_path/'t.npz', .13, direction=-1.)
    out = plateau.analyse(tmp_path/'t.npz', -1.)
    assert out['last_second_mean_speed_m_s'] == pytest.approx(.13)
    assert out['wheel_push_last_second']['wheel_target_rad_s'] == pytest.approx(.45*5)


def test_the_six_drive_seconds_are_the_six_consecutive_windows(tmp_path):
    vx = np.zeros(600)
    for k in range(6):
        vx[100+50*k:150+50*k] = .01*(k+1)
    np.savez_compressed(tmp_path/'t.npz', body_vx_m_s=vx, issued_action=np.zeros((600, 16)))
    out = plateau.analyse(tmp_path/'t.npz', 1.)
    assert [round(out['speed_by_second_after_drive_start_m_s'][str(k)], 6) for k in range(1, 7)] == [
        .01, .02, .03, .04, .05, .06]


def test_the_last_second_is_exactly_the_last_fifty_ticks(tmp_path):
    """A plateau gives the same mean for any window length, so the speed must change inside it."""
    vx = np.zeros(600); vx[100:] = .10; vx[375:400] = .20      # last 25 ticks differ from the 25 before
    np.savez_compressed(tmp_path/'t.npz', body_vx_m_s=vx, issued_action=np.zeros((600, 16)))
    out = plateau.analyse(tmp_path/'t.npz', 1.)
    assert plateau.WINDOW_TICKS*plateau.TICK_S == pytest.approx(1.0)
    assert out['last_second_mean_speed_m_s'] == pytest.approx(.15)      # 25 ticks of .10 and 25 of .20
    assert out['last_second_mean_speed_m_s'] != pytest.approx(.20)


def test_torque_fraction_is_a_magnitude_even_when_the_wheel_target_is_below_the_rolling_rate():
    speed = np.full(400, .174)                  # rolling rate 2 rad/s
    out = plateau.wheel_push(action_with(.2), speed, 1., 350, 400)   # target 1 rad/s, below rolling
    assert out['torque_per_wheel_n_m'] == pytest.approx(-.5)
    assert out['fraction_of_torque_limit'] == pytest.approx(.5/12.)
    assert out['fraction_of_torque_limit'] > 0


def test_the_analysis_records_the_trace_digest(tmp_path):
    write_trace(tmp_path/'t.npz', .13)
    assert plateau.analyse(tmp_path/'t.npz', 1.)['trace_sha256'] == plateau.sha(tmp_path/'t.npz')


def test_the_torque_estimate_is_labelled_as_an_estimate_in_the_source():
    """The traces hold no wheel joint speed, so the torque assumes pure rolling. Say so where it is computed."""
    source = Path(plateau.__file__).read_text()
    assert 'rolling rate that speed implies' in source


def test_the_tool_imports_no_simulator():
    source = Path(plateau.__file__).read_text()
    assert 'import mujoco' not in source and 'onnxruntime' not in source


# --- the numbers quoted in the contract come from the receipt --------------------
@needs_workstation_evidence
def test_numbers_quoted_in_contract_revision_1_agree_with_the_receipt():
    if not RECEIPT.is_file():
        pytest.skip('drive plateau receipt not on this machine')
    receipt = json.loads(RECEIPT.read_text())['versions']
    text = (ROOT/'docs/career_execution_contract_05m.md').read_text()
    b = receipt['04b']
    assert f"forward {b['forward']['last_second_mean_speed_m_s']:.3f}" in text
    assert f"reverse {b['reverse']['last_second_mean_speed_m_s']:.3f}" in text
    assert f"{b['forward']['wheel_push_last_second']['target_minus_rolling_rad_s']:.2f}" in text
    low = min(receipt['03d'][t]['wheel_push_last_second']['mean_abs_action'] for t in ('forward', 'reverse'))
    high = max(receipt['03d'][t]['wheel_push_last_second']['mean_abs_action'] for t in ('forward', 'reverse'))
    assert f'03d {low:.2f}–{high:.2f}' in text
    low = min(receipt['04a'][t]['wheel_push_last_second']['mean_abs_action'] for t in ('forward', 'reverse'))
    high = max(receipt['04a'][t]['wheel_push_last_second']['mean_abs_action'] for t in ('forward', 'reverse'))
    assert f'04a {low:.2f}–{high:.2f}' in text
    mean_a = np.mean([b[t]['wheel_push_last_second']['mean_abs_action'] for t in ('forward', 'reverse')])
    assert f'04b {mean_a:.2f}' in text
    assert all(r['wheel_push_last_second']['fraction_of_torque_limit'] < .05
               for v in receipt.values() for r in v.values())
