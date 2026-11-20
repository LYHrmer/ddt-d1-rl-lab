import numpy as np
import pytest

from tools87.scoring87 import score, select_video


def test_short_trial_charges_missing_time_even_with_perfect_measured_tracking():
    course = dict(policy_hz=50, ticks=300, stand_ticks=50, command_forward_m_s=.2, toe_x_m=.6)
    n = 100
    velocity = np.r_[np.zeros(51), np.full(n - 50, .2)]
    zeros = np.zeros(n + 1)
    trace = dict(time_s=np.arange(n + 1) / 50, body_vx_m_s=velocity, command_vx_m_s=velocity,
                 front_contact_x_m=np.full(n + 1, np.nan), fallen=np.arange(n + 1) == n,
                 illegal_body_contact=zeros, roll_rad=zeros, pitch_rad=zeros, body_x_m=zeros,
                 raw_ranges16=np.zeros((n + 1, 16)), wheel_saturation_count=zeros, mask_action_max_abs=zeros)
    result = score(trace, course, .4)
    assert result["observed_speed_rmse_m_s"] == 0.
    assert result["fixed_speed_rmse_m_s"] == pytest.approx(np.sqrt(200 * .2**2 / 250))
    assert result["missing_motion_samples"] == 200
    assert result["fallen"] and not result["course_completed"] and not result["reached_mark"]


def test_representative_video_uses_middle_qualified_candidate_and_matched_parent():
    cases = []
    for seed, error in ((1, .005), (2, .06), (3, .03)):
        for arm in ("baseline", "range"):
            cases.append(dict(case=f"{seed}_{arm}", seed=seed, terrain="up", arm=arm, training_seed=seed*10,
                              course_completed=True, reached_mark=True, fixed_speed_rmse_m_s=error))
    selected = select_video(cases)
    assert selected["seed"] == 3 and selected["range"] == "3_range" and selected["baseline"] == "3_baseline"
