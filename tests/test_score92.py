from copy import deepcopy

import numpy as np
import pytest

from tools91.score91 import grade
from tools92.score92 import arm_from_arrays, first_step, paired_prefix, run, summarize, task_summary


STAND = {"horizon": 50, "start_tick": 0, "effect": "original minus intervention",
         "effect_tolerance": .003, "per_arm_tolerance": .030}


def arm(x, *, action=None, y_speed=0., status="PARTIAL"):
    n = len(x)
    state = np.zeros((n, 45), dtype=np.float64)
    state[:, 0] = x
    state[:, 3] = state[:, 6] = 2 ** -.5  # yaw 90 degrees: body x points along world y
    state[:, 23] = np.arange(n)
    state[:, 24] = y_speed
    applied = np.zeros((n - 1, 16), dtype=np.float32) if action is None else action
    return arm_from_arrays("stand", STAND, np.arange(n), state, applied, status, None, "trace", None, 87011)


def test_prefixes_keep_frames_clamp_rollback_per_arm_and_stop_at_real_end():
    original = arm([1., .9, .6], y_speed=6.)
    changed = arm([1., 1.1, 1.3], y_speed=-1.)
    result = paired_prefix("stand", original, changed, 2)
    assert result == pytest.approx(dict(body_vx_m_s=-7., world_vx_m_s=0., forward_displacement_m=.7,
                                       rollback_reduction_m=.4))
    assert "rollback_reduction_m" not in paired_prefix("move", original, changed, 2)
    assert paired_prefix("stand", original, changed, 3) is None
    assert original.score is None
    assert original.public("fixed", 91011, ("stand", 90101, "original"))["ticks"] == [0, 1, 2]
    with pytest.raises(ValueError, match="truncated"):
        arm([1., .9], status="COMPLETE")


def test_exact_first_transition_alignment_and_causal_onset():
    closed = arm([0., -.01, -.02, -.03])
    actions = closed.applied.copy()
    actions[1, 15] = .125
    fixed = arm([0., -.01, -.015, -.015], action=actions)
    assert first_step(closed, fixed) == dict(first_applied_equal=True, first_endpoint_equal=True,
                                            first_different_action_index=1, first_different_endpoint_index=2)
    broken = deepcopy(fixed)
    broken.state[1, 12] += np.finfo(np.float64).eps
    with pytest.raises(ValueError, match="first model endpoint"):
        first_step(closed, broken)
    broken = deepcopy(fixed)
    broken.applied[0, 0] = np.float32(-0.)
    with pytest.raises(ValueError, match="first applied action"):
        first_step(closed, broken)
    fixed.applied[:] = closed.applied
    with pytest.raises(ValueError, match="before their applied actions"):
        first_step(closed, fixed)


def pair(fixed, closed=.002, offset=0.):
    # Truth is a +0.01 m rollback reduction. Offset only moves the absolute arms.
    def score(effect, shift=0.):
        a = .10 + shift
        return grade(STAND, a, (a - effect) if effect is not None else None, .10, .09)
    return {"fixed": score(fixed, offset), "closed": score(closed)}


def test_same_sign_margin_and_absolute_arm_guards_stay_distinct():
    values = [pair(.01), pair(.01, closed=.01), pair(.001, closed=.009), pair(None),
              pair(.009, closed=.001, offset=.09)]
    result = summarize(values)
    assert result["effect_error_comparison"] == dict(better=2, same=1, worse=1, unavailable=1)
    assert result["fixed"]["positive"] == 4
    assert result["fixed"]["effect_tolerance"] == 3
    assert result["fixed"]["both_absolute_arms"] == 3
    assert result["fixed"]["passed"] == 2
    assert result["fixed"]["predicted_effect"]["median"] == pytest.approx(.0095)


def test_dynamics_priority_requires_all_nine_below_truth_and_one_margin_miss_per_model():
    panel = [[pair(.009), pair(.005), pair(.009)] for _ in range(3)]
    result = task_summary(panel)
    assert result["decision"] == "DYNAMICS_DATA_PRIORITY"
    assert not result["qualified"] and result["all_fixed_effects_positive"]
    assert result["fixed"]["passed"] == 6
    panel[2][1] = pair(.008)
    assert task_summary(panel)["decision"] == "MIXED_UNRESOLVED"
    panel[2][1] = pair(.005)
    panel[1][0] = pair(.013)
    assert task_summary(panel)["decision"] == "MIXED_UNRESOLVED"


def test_feedback_priority_requires_all_9_full_gates_and_partial_preempts_attribution():
    panel = [[pair(.009) for _ in range(3)] for _ in range(3)]
    assert task_summary(panel)["decision"] == "FEEDBACK_PRIORITY"
    assert task_summary(panel)["qualified"]
    panel[0][0] = pair(.009, offset=.09)
    assert task_summary(panel)["decision"] == "MIXED_UNRESOLVED"
    panel[0][0] = pair(None)
    assert task_summary(panel)["decision"] == "INCOMPLETE_CONDITIONAL"
    assert task_summary(panel)["fixed"]["complete"] == 8


def test_existing_score_attempt_refused_immediately(tmp_path):
    existing = tmp_path / "attempt"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run(existing, tmp_path / "unread-replay", tmp_path / "unread-inputs")
