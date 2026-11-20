from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from tools87.run87 import read
from tools89.eval89 import PLAN, matrix, score_window, summarize


def clean_cases(plan):
    return [dict(arm=arm, training_seed=train, terrain=terrain, seed=reset,
                 score_vx_m_s=0.04 if arm == "mixed" else 0.,
                 fixed_speed_rmse_m_s=0.05, fallen=False, illegal_contact=False,
                 course_completed=True, reached_mark=True)
            for arm, train, terrain, reset in matrix(plan)]


def test_three_by_three_uphill_and_matched_retention():
    plan = read(PLAN)
    design = matrix(plan)
    counts = Counter(terrain for _, _, terrain, _ in design)
    assert len(set(design)) == len(design) == 30 and counts == {"up": 18, "flat": 6, "down": 6}
    for i, train in enumerate(plan["training"]["seeds"]):
        for arm in plan["training"]["arms"]:
            assert {s for a, t, terrain, s in design if a == arm and t == train and terrain == "up"} == set(
                plan["evaluation"]["reset_seeds"])
            assert {s for a, t, terrain, s in design if a == arm and t == train and terrain in ("flat", "down")} == {
                plan["evaluation"]["reset_seeds"][i]}


def test_speed_uses_only_100_actual_endpoints_and_preserves_missing_window():
    speeds = np.arange(301, dtype=float)
    assert score_window({"body_vx_m_s": speeds}, [200, 300]) == 250.5
    assert score_window({"body_vx_m_s": speeds[:-1]}, [200, 300]) is None
    speeds[299] = np.nan
    with pytest.raises(ValueError):
        score_window({"body_vx_m_s": speeds}, [200, 300])


def test_all_three_training_means_must_pass_and_course_mark_is_separate():
    plan = read(PLAN)
    rows = clean_cases(plan)
    first = plan["training"]["seeds"][0]
    group = [item for item in rows if item["arm"] == "mixed" and item["terrain"] == "up" and item["training_seed"] == first]
    for r, value in zip(group, (0.0, 0.025, 0.035)):
        r["score_vx_m_s"] = value
    group[0]["reached_mark"] = False
    result = summarize(rows, plan)
    assert result["uphill_speed_pass"] and not result["uphill_stable_pass"]
    assert result["uphill_clean_marks"] == {"original": 9, "mixed": 8}
    group[-1]["score_vx_m_s"] = 0.03499
    result = summarize(rows, plan)
    assert [r["passes"] for r in result["uphill_paired_train_seed_means"]] == [False, True, True]
    assert not result["uphill_speed_pass"] and result["retention_pass"]


def test_early_termination_is_preserved_and_neither_an_incomplete_nor_duplicate_matrix_scores():
    plan = read(PLAN)
    rows = clean_cases(plan)
    up = next(r for r in rows if r["terrain"] == "up" and r["arm"] == "mixed")
    up.update(score_vx_m_s=None, course_completed=False, fallen=True)
    verdict = summarize(rows, plan)
    assert not verdict["uphill_speed_pass"] and not verdict["uphill_stable_pass"]
    assert verdict["uphill_paired_train_seed_means"][0]["paired_delta_vx_m_s"] is None
    assert verdict["total_falls"] == 1
    with pytest.raises(ValueError):
        summarize(rows[:-1], plan)
    duplicate = deepcopy(rows)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError):
        summarize(duplicate, plan)


def test_retention_uses_fixed_horizon_per_terrain_and_requires_all_episodes_safe():
    plan = read(PLAN)
    rows = clean_cases(plan)
    down = [r for r in rows if r["terrain"] == "down" and r["arm"] == "mixed"]
    down[0]["fixed_speed_rmse_m_s"] += 0.029
    assert summarize(rows, plan)["retention_pass"]
    down[0]["fixed_speed_rmse_m_s"] += 0.002
    result = summarize(rows, plan)
    assert result["pace_retention"]["flat"]["passes"] and not result["pace_retention"]["down"]["passes"]
    down[0].update(fixed_speed_rmse_m_s=0., course_completed=False, illegal_contact=True)
    result = summarize(rows, plan)
    assert not result["retention_pass"] and result["total_illegal_contacts"] == 1
