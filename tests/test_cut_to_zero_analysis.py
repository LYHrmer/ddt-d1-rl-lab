"""Tests for the pure helpers of tools04/analyze_cut_to_zero04.py.

The analysis argues about a 10-frame, 200 ms window, so the places where that
window starts and ends are tested exactly, and the course sampler is checked
against bounds that follow from the frozen config rather than from the tool.
Nothing here loads a model or runs physics.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))

import analyze_cut_to_zero04 as cut  # noqa: E402

SPEC = json.loads((ROOT/'training/configs/pilot03d.json').read_text())['commands']['paired_schedule']


def command_rows(*segments):
    """Build a (n, 3) command array from (ticks, vx, wz) segments."""
    return np.array([[vx, 0., wz] for ticks, vx, wz in segments for _ in range(ticks)])


# --- cut detection ------------------------------------------------------------

def test_a_single_drive_has_one_cut_at_the_first_zero_tick():
    command = command_rows((100, 0., 0.), (300, .2, 0.), (200, 0., 0.))
    assert cut.find_cut_ticks(command) == [400]


def test_no_cut_when_the_command_never_stops():
    assert cut.find_cut_ticks(command_rows((10, .2, 0.), (10, -.2, 0.))) == []


def test_a_sign_reversal_is_not_a_cut_because_the_command_never_reaches_zero():
    """The turn task goes +0.2 then -0.2 directly; only the final drop counts."""
    command = command_rows((50, 0., 0.), (150, 0., .2), (150, 0., -.2), (100, 0., 0.))
    assert cut.find_cut_ticks(command) == [350]


def test_two_separate_motions_give_two_cuts():
    command = command_rows((50, 0., 0.), (100, .2, 0.), (60, 0., 0.), (100, -.2, 0.), (80, 0., 0.))
    assert cut.find_cut_ticks(command) == [150, 310]


def test_motion_from_tick_zero_is_handled():
    assert cut.find_cut_ticks(command_rows((5, .1, 0.), (5, 0., 0.))) == [5]


def test_a_wz_only_command_counts_as_motion():
    assert cut.nonzero_command(np.array([[0., 0., .1], [0., 0., 0.]])).tolist() == [True, False]


# --- the history window --------------------------------------------------------

@pytest.mark.parametrize('k,expected', [(0, 10), (1, 9), (5, 5), (9, 1), (10, 0), (11, 0),
                                        (25, 0), (1000, 0)])
def test_history_carries_the_pre_cut_command_for_exactly_ten_ticks(k, expected):
    assert cut.history_frames_carrying_command(k) == expected


def test_history_window_is_two_hundred_milliseconds_at_the_policy_rate():
    assert cut.HISTORY_LENGTH*.02 == pytest.approx(.2)


def test_negative_ticks_after_the_cut_are_rejected():
    with pytest.raises(ValueError):
        cut.history_frames_carrying_command(-1)


def test_a_custom_history_length_is_respected():
    assert cut.history_frames_carrying_command(3, history_length=4) == 1


# --- zeroing latency -----------------------------------------------------------

def test_latency_is_measured_from_the_cut_not_from_the_start_of_the_trace():
    values = np.concatenate([np.full(10, .1), np.full(3, .1), np.full(20, .001)])
    assert cut.zeroing_latency(values, start=10) == 3


def test_latency_is_zero_when_already_quiet_at_the_cut():
    assert cut.zeroing_latency(np.full(20, .0), start=5) == 0


def test_latency_requires_the_quiet_to_hold_not_a_single_dip():
    values = np.array([.5, .001, .5, .5, .001, .001, .001, .001, .001, .5])
    assert cut.zeroing_latency(values, start=0, hold=5) == 4


def test_latency_is_none_when_it_never_settles():
    assert cut.zeroing_latency(np.full(30, .1), start=0) is None


def test_latency_is_none_when_the_trace_ends_before_the_hold_completes():
    assert cut.zeroing_latency(np.array([.5, .001, .001]), start=0, hold=5) is None


def test_the_threshold_is_strictly_below_not_equal():
    assert cut.zeroing_latency(np.full(10, cut.ZERO_THRESHOLD), start=0) is None


# --- progress along the entry heading -----------------------------------------

def test_progress_projects_on_the_heading_and_keeps_the_sign():
    boundary = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.1, 0.]])
    out = cut.signed_progress(boundary, 0, np.array([1., 0.]), (1, 2, 3))
    assert out == {1: pytest.approx(.1), 2: pytest.approx(.2), 3: pytest.approx(.1)}


def test_progress_exposes_rebound_that_euclidean_distance_would_hide():
    """Overshoot then return: the signed value at the end is small, the peak is not."""
    boundary = np.array([[0., 0.], [-.05, 0.], [-.03, 0.], [.0, 0.], [.02, 0.]])
    out = cut.signed_progress(boundary, 0, np.array([-1., 0.]), (1, 2, 3, 4))
    assert out[1] == pytest.approx(.05) and out[4] == pytest.approx(-.02)


def test_progress_normalises_a_non_unit_heading():
    """Callers pass unit headings today, but the function is public and a scaled
    heading must not silently scale the answer."""
    boundary = np.array([[0., 0.], [.1, 0.]])
    unit = cut.signed_progress(boundary, 0, np.array([1., 0.]), (1,))[1]
    scaled = cut.signed_progress(boundary, 0, np.array([7., 0.]), (1,))[1]
    assert unit == pytest.approx(.1) and scaled == pytest.approx(.1)


def test_progress_ignores_motion_perpendicular_to_the_heading():
    boundary = np.array([[0., 0.], [0., .3]])
    assert cut.signed_progress(boundary, 0, np.array([1., 0.]), (1,))[1] == pytest.approx(0.)


def test_progress_omits_tick_counts_beyond_the_trace():
    boundary = np.zeros((3, 2))
    assert set(cut.signed_progress(boundary, 0, np.array([1., 0.]), (1, 2, 50))) == {1, 2}


def test_a_zero_heading_is_rejected():
    with pytest.raises(ValueError):
        cut.signed_progress(np.zeros((3, 2)), 0, np.zeros(2), (1,))


def test_path_length_counts_back_and_forth_motion_in_full():
    boundary = np.array([[0., 0.], [.1, 0.], [0., 0.], [.1, 0.]])
    out = cut.path_length_shares(boundary, 0, bins=((0, 3),))
    assert out['total_m'] == pytest.approx(.3)


def test_path_shares_sum_to_one_over_all_motion():
    boundary = np.cumsum(np.full((300, 2), .001), axis=0)
    shares = cut.path_length_shares(boundary, 0)['share']
    assert sum(v for v in shares.values() if v is not None) == pytest.approx(1., abs=.01)


def test_path_shares_are_none_rather_than_a_division_by_zero_when_nothing_moved():
    out = cut.path_length_shares(np.zeros((50, 2)), 0)
    assert out['total_m'] == 0. and all(v is None for v in out['share'].values())


# --- the course sampler: bounds follow from the frozen config ----------------

def test_sampler_returns_two_cuts_per_episode_in_increasing_order():
    cuts = cut.sample_cut_ticks(SPEC, 2000)
    assert cuts.shape == (2000, 2) and np.all(cuts[:, 1] > cuts[:, 0])


def test_sampler_stays_inside_the_ranges_the_config_allows():
    cuts = cut.sample_cut_ticks(SPEC, 20000)
    first_zero = SPEC['first_zero_ticks']
    t, u = SPEC['translation'], SPEC['turn']
    low = min(first_zero+t['motion_ticks_inclusive'][0], first_zero+2*u['motion_ticks_inclusive'][0])
    high = max(first_zero+t['motion_ticks_inclusive'][1], first_zero+2*u['motion_ticks_inclusive'][1])
    assert cuts[:, 0].min() >= low and cuts[:, 0].max() <= high


def test_second_cut_always_leaves_at_least_the_final_zero_segment():
    cuts = cut.sample_cut_ticks(SPEC, 20000)
    assert SPEC['episode_ticks']-cuts[:, 1].max() >= 200


def test_the_two_cuts_are_separated_by_a_real_zero_gap_not_adjacent():
    """The course holds zero for at least `middle_zero_ticks` between a motion and
    the next, so two cuts a tick apart would mean the pairing was lost."""
    cuts = cut.sample_cut_ticks(SPEC, 20000)
    gap_floor = min(SPEC['translation']['middle_zero_ticks_inclusive'][0],
                    SPEC['turn']['middle_zero_ticks_inclusive'][0])
    minimum_motion = min(SPEC['translation']['motion_ticks_inclusive'][0],
                         2*SPEC['turn']['motion_ticks_inclusive'][0])
    assert np.min(cuts[:, 1]-cuts[:, 0]) >= gap_floor+minimum_motion


def test_sampler_is_deterministic_for_a_seed():
    assert np.array_equal(cut.sample_cut_ticks(SPEC, 500, seed=3), cut.sample_cut_ticks(SPEC, 500, seed=3))
    assert not np.array_equal(cut.sample_cut_ticks(SPEC, 500, seed=3),
                              cut.sample_cut_ticks(SPEC, 500, seed=4))


# --- exposure -------------------------------------------------------------------

def test_full_length_episodes_see_two_cuts_each():
    cuts = cut.sample_cut_ticks(SPEC, 5000)
    out = cut.cut_exposure(cuts, [1000]*100)
    assert out['expected_cut_events'] == pytest.approx(200.)


def test_windows_are_two_w_per_thousand_ticks_when_cuts_do_not_overlap():
    cuts = cut.sample_cut_ticks(SPEC, 5000)
    out = cut.cut_exposure(cuts, [1000]*100)
    for w in (10, 25, 60):
        assert out['share_of_ticks_in_window'][str(w)] == pytest.approx(2*w/1000, rel=1e-6)


def test_episodes_that_end_before_the_first_cut_contribute_nothing():
    cuts = cut.sample_cut_ticks(SPEC, 5000)
    out = cut.cut_exposure(cuts, [100]*50)
    assert out['expected_cut_events'] == 0.


def test_an_episode_ending_between_the_two_cuts_keeps_only_the_first():
    cuts = np.array([[300, 700]])
    assert cut.cut_exposure(cuts, [500])['expected_cut_events'] == pytest.approx(1.)


def test_a_window_is_truncated_by_the_end_of_the_episode():
    cuts = np.array([[300, 700]])
    out = cut.cut_exposure(cuts, [305], windows=(10,))
    assert out['expected_ticks_in_window']['10'] == pytest.approx(5.)


# --- decay time constant ---------------------------------------------------------

def test_decay_time_constant_recovers_a_known_exponential():
    ticks = np.arange(120)
    speed = .2*np.exp(-(ticks*.02)/.3)
    assert cut.decay_time_constant(speed, 0, 0, 100, floor=.0) == pytest.approx(.3, rel=1e-6)


def test_decay_time_constant_respects_the_window_offset():
    ticks = np.arange(300)
    speed = np.where(ticks < 100, 5., .2*np.exp(-((ticks-100)*.02)/.25))
    assert cut.decay_time_constant(speed, 100, 0, 100, floor=.0) == pytest.approx(.25, rel=1e-6)


def test_decay_time_constant_drops_samples_below_the_floor():
    speed = np.concatenate([.2*np.exp(-np.arange(30)*.02/.3), np.full(60, 1e-4)])
    with_floor = cut.decay_time_constant(speed, 0, 0, 90, floor=.005)
    assert with_floor == pytest.approx(.3, rel=1e-3)


def test_decay_time_constant_is_none_when_the_speed_is_not_falling():
    assert cut.decay_time_constant(np.linspace(.01, .2, 80), 0, 0, 80, floor=.0) is None


def test_decay_time_constant_is_none_with_too_few_samples_above_the_floor():
    assert cut.decay_time_constant(np.full(60, .001), 0, 0, 60) is None


# --- velocity feedback fit -------------------------------------------------------

def test_velocity_feedback_fit_recovers_slope_and_intercept():
    v = np.linspace(-.05, .05, 60)
    out = cut.velocity_feedback_fit(2.5*v+.4, v, 0, 0, 60)
    assert out['slope_rad_s_per_m_s'] == pytest.approx(2.5)
    assert out['intercept_rad_s'] == pytest.approx(.4)
    assert out['r_squared'] == pytest.approx(1.)


def test_a_constant_bias_with_no_velocity_dependence_has_zero_r_squared():
    """The 03d pattern: the target ignores the speed entirely."""
    rng = np.random.default_rng(0)
    v = rng.normal(0., .01, 400)
    out = cut.velocity_feedback_fit(np.full(400, .5)+rng.normal(0., .05, 400), v, 0, 0, 400)
    assert abs(out['slope_rad_s_per_m_s']) < 3. and out['intercept_rad_s'] == pytest.approx(.5, abs=.02)
    assert out['r_squared'] < .05


def test_velocity_feedback_fit_window_selects_the_right_ticks():
    v = np.concatenate([np.zeros(10), np.linspace(-.05, .05, 50)])
    y = np.concatenate([np.full(10, 99.), 2.*np.linspace(-.05, .05, 50)])
    out = cut.velocity_feedback_fit(y, v, 10, 0, 50)
    assert out['slope_rad_s_per_m_s'] == pytest.approx(2.) and out['intercept_rad_s'] == pytest.approx(0., abs=1e-9)


def test_velocity_feedback_fit_is_none_when_the_velocity_never_changes():
    assert cut.velocity_feedback_fit(np.arange(30.), np.full(30, .01), 0, 0, 30) is None


def test_velocity_feedback_fit_r_squared_is_none_when_the_target_is_constant():
    v = np.linspace(-.05, .05, 30)
    assert cut.velocity_feedback_fit(np.full(30, .3), v, 0, 0, 30)['r_squared'] is None


# --- damping scale -----------------------------------------------------------------

def test_a_zero_slope_keeps_all_of_the_damping():
    assert cut.damping_scale_from_slope(0.) == pytest.approx(1.)


def test_following_the_rolling_speed_removes_all_of_the_damping():
    assert cut.damping_scale_from_slope(1./cut.WHEEL_RADIUS_M) == pytest.approx(0., abs=1e-12)


def test_half_the_rolling_slope_halves_the_damping():
    assert cut.damping_scale_from_slope(.5/cut.WHEEL_RADIUS_M) == pytest.approx(.5)


def test_a_negative_slope_brakes_harder_than_a_zero_action():
    assert cut.damping_scale_from_slope(-4.5) > 1.


def test_wheel_radius_matches_the_frozen_runtime_schema():
    import json as _json
    schema = _json.loads((ROOT/'runtime03/schema03d.json').read_text())
    assert 'wheel' in schema['wheel_scale_speed_note'].lower()
    assert '.087' in schema['wheel_scale_speed_note'] and cut.WHEEL_RADIUS_M == .087


def test_r_squared_stays_inside_zero_to_one_for_any_fit_that_has_variance():
    rng = np.random.default_rng(1)
    for _ in range(50):
        v = rng.normal(0., .02, 80)
        y = rng.normal(0., 1., 80)*rng.uniform(.01, 5.)+rng.uniform(-2., 2.)
        r2 = cut.velocity_feedback_fit(y, v, 0, 0, 80)['r_squared']
        assert r2 is not None and -1e-9 <= r2 <= 1.+1e-9


def test_a_tiny_but_real_variation_is_not_mistaken_for_a_constant():
    v = np.linspace(-.05, .05, 60)
    out = cut.velocity_feedback_fit(1e-6*v+1., v, 0, 0, 60)
    assert out['r_squared'] is not None and out['r_squared'] > .99


# --- how long the robot keeps moving after the cut ---------------------------------

def test_ticks_until_speed_below_counts_from_the_cut():
    speed = np.concatenate([np.full(100, .3), np.linspace(.3, 0., 40), np.zeros(20)])
    result = cut.ticks_until_speed_below(speed, start=100, limit=.02)
    assert result is not None and 36 <= result <= 40


def test_ticks_until_speed_below_is_zero_when_already_slow_at_the_cut():
    assert cut.ticks_until_speed_below(np.full(50, .001), start=10) == 0


def test_ticks_until_speed_below_is_none_when_it_never_slows():
    assert cut.ticks_until_speed_below(np.full(50, .1), start=0) is None


def test_ticks_until_speed_below_can_require_the_slowness_to_hold():
    speed = np.array([.1, .01, .1, .1, .01, .01, .01])
    assert cut.ticks_until_speed_below(speed, start=0, limit=.02, hold=1) == 1
    assert cut.ticks_until_speed_below(speed, start=0, limit=.02, hold=3) == 4


def test_the_threshold_is_strictly_below():
    assert cut.ticks_until_speed_below(np.full(10, .02), start=0, limit=.02) is None


def test_motion_outlasts_the_history_window_in_a_slow_stop():
    """The point of the figure: a stop longer than 10 ticks leaves the history behind."""
    speed = np.concatenate([np.full(50, .13), np.linspace(.13, 0., 80)])
    ticks = cut.ticks_until_speed_below(speed, start=50)
    assert ticks > cut.HISTORY_LENGTH


# --- per-tick density: windows of different widths are not comparable by share ---

def test_density_divides_each_share_by_its_window_width():
    shares = {'0-10': .3, '10-25': .3, '25-50': .25, '50-100': .1, '100-200': .05}
    out = cut.path_density_per_tick(shares)
    assert out['0-10'] == pytest.approx(.03) and out['10-25'] == pytest.approx(.02)
    assert out['100-200'] == pytest.approx(.0005)


def test_equal_shares_in_unequal_windows_are_not_equal_density():
    """The error this exists to prevent: 27 and 25 percent look alike, 10 and 15 ticks are not."""
    out = cut.path_density_per_tick({'0-10': .27, '10-25': .25, '25-50': .22,
                                     '50-100': .19, '100-200': .07})
    # Raw shares 27, 25, 22, 19 percent are nearly flat; the densities are not.
    ordered = [out[k] for k in ('0-10', '10-25', '25-50', '50-100', '100-200')]
    assert all(a > b for a, b in zip(ordered, ordered[1:]))
    # Ratios for these exact inputs are 3.07 and 7.11; assert a margin below each.
    assert out['0-10'] > 3.*out['25-50'] and out['0-10'] > 7.*out['50-100']


def test_a_truly_uniform_path_has_equal_density_everywhere():
    widths = {'0-10': 10, '10-25': 15, '25-50': 25, '50-100': 50, '100-200': 100}
    shares = {k: w/200. for k, w in widths.items()}
    values = list(cut.path_density_per_tick(shares).values())
    assert max(values)-min(values) < 1e-12


def test_density_keeps_a_missing_window_as_none():
    assert cut.path_density_per_tick({'0-10': .5})['10-25'] is None


def test_front_loading_ratio_compares_early_to_late_density():
    density = {'0-10': .03, '100-200': .0005}
    assert cut.front_loading_ratio(density) == pytest.approx(60.)


def test_front_loading_ratio_is_none_when_a_window_is_missing_or_zero():
    assert cut.front_loading_ratio({'0-10': .03, '100-200': None}) is None
    assert cut.front_loading_ratio({'0-10': .03, '100-200': 0.}) is None


# --- first order stop: displacement is the entry speed times the time constant ---

def test_the_integral_of_an_exponential_decay_is_speed_times_tau():
    tau, v0 = .4, .15
    t = np.linspace(0., 20., 200001)
    integrate = getattr(np, 'trapezoid', None) or np.trapz     # numpy 2 renamed trapz
    numeric = integrate(v0*np.exp(-t/tau), t)
    assert cut.first_order_displacement(v0, tau) == pytest.approx(numeric, rel=1e-4)


def test_first_order_displacement_ignores_the_direction_of_motion():
    assert cut.first_order_displacement(-.2, .5) == pytest.approx(.1)


def test_first_order_displacement_is_none_without_a_time_constant():
    assert cut.first_order_displacement(.2, None) is None


def test_the_needed_time_constant_is_the_limit_over_the_entry_speed():
    assert cut.required_time_constant(.132, .05) == pytest.approx(.05/.132)
    assert cut.required_time_constant(-.132, .05) == pytest.approx(.05/.132)


def test_required_time_constant_rejects_zero_speed():
    with pytest.raises(ValueError):
        cut.required_time_constant(0., .05)


def test_meeting_the_needed_time_constant_exactly_meets_the_limit():
    v0 = .2
    assert cut.first_order_displacement(v0, cut.required_time_constant(v0, .05)) == pytest.approx(.05)


# --- windowed velocity feedback -----------------------------------------------------

def test_each_window_reports_the_speed_range_it_was_fitted_over():
    v = np.concatenate([np.zeros(5), np.linspace(.2, .0, 100)])
    y = 3.*v+.1
    out = cut.windowed_velocity_feedback(y, v, 5, ((0, 40), (60, 100)))
    early, late = out['0-40'], out['60-100']
    assert early['speed_range_m_s'][1] > late['speed_range_m_s'][1]
    assert early['slope_rad_s_per_m_s'] == pytest.approx(3.) and late['slope_rad_s_per_m_s'] == pytest.approx(3.)


def test_windows_can_disagree_and_both_are_reported():
    """A fast braking phase and a slow creep follow different laws; neither may be hidden."""
    v = np.concatenate([np.linspace(.13, .05, 40), np.linspace(.05, .0, 160)])
    y = np.concatenate([30.*np.linspace(.13, .05, 40)-3., 5.*np.linspace(.05, .0, 160)])
    out = cut.windowed_velocity_feedback(y, v, 0, ((0, 40), (40, 200)))
    assert out['0-40']['slope_rad_s_per_m_s'] == pytest.approx(30., rel=1e-3)
    assert out['40-200']['slope_rad_s_per_m_s'] == pytest.approx(5., rel=1e-3)


def test_a_degenerate_window_is_none_not_omitted():
    v = np.full(60, .01)
    assert cut.windowed_velocity_feedback(np.arange(60.), v, 0, ((0, 30),))['0-30'] is None


def test_each_window_carries_its_own_damping_scale():
    v = np.linspace(-.05, .05, 60)
    out = cut.windowed_velocity_feedback(0.*v, v, 0, ((0, 60),))
    assert out['0-60']['damping_scale_vs_zero_action'] == pytest.approx(1.)


def test_each_window_computes_its_own_damping_scale_from_its_own_slope():
    """Two windows with different slopes: a damping scale copied or hardcoded cannot
    match both. A zero slope would not catch it, since the correct value is then 1."""
    v = np.concatenate([np.linspace(.13, .05, 50), np.linspace(.05, .0, 150)])
    y = np.concatenate([5.*np.linspace(.13, .05, 50), 2.*np.linspace(.05, .0, 150)])
    out = cut.windowed_velocity_feedback(y, v, 0, ((0, 50), (50, 200)))
    assert out['0-50']['damping_scale_vs_zero_action'] == pytest.approx(1.-5.*cut.WHEEL_RADIUS_M, abs=1e-6)
    assert out['50-200']['damping_scale_vs_zero_action'] == pytest.approx(1.-2.*cut.WHEEL_RADIUS_M, abs=1e-6)
    assert out['0-50']['damping_scale_vs_zero_action'] != out['50-200']['damping_scale_vs_zero_action']
