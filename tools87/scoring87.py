"""Scores and video selection for the registered six-second D1 course."""
from __future__ import annotations

from statistics import median

import numpy as np


def score(trace, course, mark):
    hz, budget, stand = course["policy_hz"], course["ticks"], course["stand_ticks"]
    ticks = len(trace["time_s"]) - 1
    vx = np.asarray(trace["body_vx_m_s"][1:], dtype=float)
    target = np.asarray(trace["command_vx_m_s"][1:], dtype=float)
    expected = np.where(np.arange(ticks) < stand, 0.0, course["command_forward_m_s"])
    if (not 0 < ticks <= budget or vx.shape != (ticks,) or not np.isfinite(vx).all()
            or not np.allclose(target, expected, atol=1e-7, rtol=0)
            or not np.allclose(trace["time_s"], np.arange(ticks + 1) / hz, atol=1e-8, rtol=0)):
        raise ValueError("recorded motion interval, velocity or command differs from the course")
    error = vx[stand:] - target[stand:]
    missing = budget - stand - error.size
    charged = float(np.dot(error, error)) + missing * course["command_forward_m_s"] ** 2
    contacts = np.asarray(trace["front_contact_x_m"][1:], dtype=float)
    loaded = contacts[np.isfinite(contacts)]
    toe = course["toe_x_m"]
    observed_past = None if not loaded.size else float(loaded.max() - toe)
    fallen = bool(np.any(trace["fallen"]))
    illegal = bool(np.any(trace["illegal_body_contact"]))
    roll_pitch = np.abs([trace["roll_rad"], trace["pitch_rad"]])
    valid = np.asarray(trace["raw_ranges16"])[1:, 8:] > 0.5
    return {
        "ticks": ticks, "duration_s": ticks / hz, "full_horizon": ticks == budget,
        "fallen": fallen, "illegal_contact": illegal, "terminated": fallen or illegal,
        "course_completed": ticks == budget and not fallen and not illegal,
        "fixed_speed_rmse_m_s": float(np.sqrt(charged / (budget - stand))),
        "observed_speed_rmse_m_s": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
        "motion_samples": int(error.size), "missing_motion_samples": missing,
        "mean_forward_speed_while_commanded_m_s": float(np.mean(vx[stand:])) if error.size else None,
        "peak_roll_or_pitch_rad": float(roll_pitch.max()),
        "max_loaded_front_contact_past_toe_m": observed_past,
        "reached_mark": observed_past is not None and observed_past >= mark,
        "wheel_saturated_ticks": int(np.count_nonzero(trace["wheel_saturation_count"])),
        "ticks_with_valid_rays": int(valid.any(axis=1).sum()),
        "ticks_with_all_rays_valid": int(valid.all(axis=1).sum()),
        "valid_ray_fraction": float(valid.mean()),
        "max_action_change_with_terrain_masked": float(np.max(trace["mask_action_max_abs"])),
        "end_body_x_m": float(trace["body_x_m"][-1]),
    }


def select_video(cases):
    """Pick an informative matched recording without using it to rank policies."""
    refs = {(c["terrain"], c["seed"]): c for c in cases if c["arm"] == "baseline"}
    pairs = [(refs[(c["terrain"], c["seed"])], c) for c in cases if c["arm"] == "range"]
    if not pairs:
        raise ValueError("no held-out range/baseline pair")
    for qualified in (True, False):
        for terrain in ("up", "down", "flat"):
            available = [(base, candidate) for base, candidate in pairs
                         if candidate["terrain"] == terrain and (not qualified or all(
                             x["course_completed"] and (terrain == "flat" or x["reached_mark"])
                             for x in (base, candidate)))]
            if available:
                middle = median(c["fixed_speed_rmse_m_s"] for _, c in available)
                base, chosen = min(available, key=lambda pair: (
                    abs(pair[1]["fixed_speed_rmse_m_s"] - middle), pair[1]["training_seed"]))
                return {"baseline": base["case"], "range": chosen["case"], "terrain": terrain,
                        "seed": chosen["seed"], "training_seed": chosen["training_seed"],
                        "qualified_pair_for_display": qualified, "group_median_rmse_m_s": middle}
