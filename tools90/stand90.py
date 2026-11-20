"""One-shot paired uphill stand with deterministic range policies and the native PD/FIFO."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from tools89.common89 import PLAN as PLAN89, ROOT, parent_folder, read, registered_plan, sha, write
from training.d1_rl.interface import ACTOR_DIM, LEG_INDICES, WHEEL_INDICES

PLAN = ROOT / "docs/experiments90/stand_wbc.json"
ARMS = ("original", "common")


def wheel_decision(raw, wheel_scales, arm, tick, stand_ticks):
    """Add a feasible common velocity to the four issued targets during the hold."""
    raw = np.asarray(raw, dtype=np.float32)
    scales = np.asarray(wheel_scales, dtype=np.float64)
    if (raw.shape != (16,) or not np.isfinite(raw).all() or (np.abs(raw) > 1).any()
            or not np.array_equal(scales, [5.] * 4)):
        raise ValueError("finite bounded 16D policy action and four physical 5 rad/s scales required")
    if arm not in ARMS or tick < 0:
        raise ValueError("unknown arm or negative decision tick")
    issued = raw.copy()
    if arm == "original" or tick >= stand_ticks:
        return issued, 0., 0.

    targets = scales * raw[WHEEL_INDICES]
    requested = max(0., -float(np.mean(targets)))
    feasible = min(requested, float(np.min(scales - targets)))
    if feasible:
        issued[WHEEL_INDICES] = (targets + feasible) / scales
    return issued, requested, feasible


def motion_metrics(initial_x_m, endpoints_x_m, stand_ticks, ticks_per_episode):
    """Endpoint index zero follows the first integrated interval."""
    xs = np.asarray(endpoints_x_m, dtype=np.float64)
    if (xs.ndim != 1 or len(xs) > ticks_per_episode or not np.isfinite(xs).all()
            or not np.isfinite(initial_x_m) or not 0 < stand_ticks < ticks_per_episode):
        raise ValueError("invalid stand or forward interval")
    x1 = float(xs[stand_ticks - 1]) if len(xs) >= stand_ticks else None
    x2 = float(xs[-1]) if len(xs) == ticks_per_episode else None
    return {
        "initial_world_x_m": float(initial_x_m), "world_x_at_1s_m": x1, "world_x_at_2s_m": x2,
        "signed_stand_dx_m": None if x1 is None else x1 - initial_x_m,
        "rollback_m": None if x1 is None else max(0., initial_x_m - x1),
        "largest_stand_rollback_m": None if x1 is None else max(0., initial_x_m - float(xs[:stand_ticks].min())),
        "signed_moving_dx_m": None if x2 is None else x2 - x1,
    }


def verify_actions(rows, scales, arm, stand_ticks):
    """Independently recompute all four additions and align them with the actual PD targets."""
    raw, issued, applied = (np.asarray(rows[k]) for k in ("policy_raw", "issued", "applied"))
    n = len(raw)
    if (not n or any(x.shape != (n, 16) for x in (raw, issued, applied))
            or any(not np.isfinite(x).all() or (np.abs(x) > 1).any() for x in (raw, issued, applied))
            or not np.array_equal(issued[:, LEG_INDICES], raw[:, LEG_INDICES])
            or not np.array_equal(applied[0], np.zeros(16))
            or not np.array_equal(applied[1:], issued[:-1])):
        raise ValueError("invalid action, changed leg target or lost one-tick FIFO")
    raw_targets = raw[:, WHEEL_INDICES] * scales
    active = (np.arange(n) < stand_ticks) & (arm == "common")
    requested = np.where(active, np.maximum(0., -raw_targets.mean(axis=1)), 0.)
    feasible = np.minimum(requested, (5. - raw_targets).min(axis=1))
    actual = issued[:, WHEEL_INDICES] * scales - raw_targets
    applied_targets = np.asarray(rows["applied_wheel_target_rad_s"])
    if (not np.array_equal(requested, rows["requested_rad_s"])
            or not np.array_equal(feasible, rows["feasible_rad_s"])
            or not np.array_equal(raw[feasible == 0.], issued[feasible == 0.])
            or not np.allclose(actual, feasible[:, None], rtol=0., atol=3e-7)
            or not np.array_equal(applied_targets, applied[:, WHEEL_INDICES] * scales)):
        raise ValueError("physical common increment or applied PD wheel target differs")
    # At tick zero the FIFO contains the reset action. At tick 50 it still
    # contains the final action issued under the stationary command at tick 49.
    delayed_raw_targets = np.vstack((np.zeros(4), raw_targets[:-1]))
    applied_increment = (applied_targets - delayed_raw_targets).mean(axis=1)
    return {
        "requested_issued_ticks": int(np.sum(requested > 0.)),
        "headroom_limited_issued_ticks": int(np.sum(feasible < requested)),
        "fully_blocked_issued_ticks": int(np.sum((requested > 0.) & (feasible == 0.))),
        "effective_issued_ticks": int(np.sum((actual != 0.).any(axis=1))),
        "effective_applied_ticks_first_s": int(np.count_nonzero(applied_increment[:stand_ticks])),
        "mean_applied_increment_first_s_rad_s": float(applied_increment[:stand_ticks].mean()),
        "max_applied_increment_rad_s": float(applied_increment.max()),
        "applied_increment_first_moving_tick_rad_s": (
            float(applied_increment[stand_ticks]) if n > stand_ticks else None),
    }


def pair_metrics(original, common, threshold):
    if (original["arm"], common["arm"]) != ARMS or any(
        original[k] != common[k] for k in ("reset_seed", "parent_seed", "policy_sha256", "initial_state_sha256")
    ):
        raise ValueError("arms must share the exact initial state, reset and checkpoint")
    a, b = original["rollback_m"], common["rollback_m"]
    reduction = None if a is None or b is None else a - b
    return {"reset_seed": original["reset_seed"], "parent_seed": original["parent_seed"],
            "original_rollback_m": a, "common_rollback_m": b, "rollback_reduction_m": reduction,
            "passed": bool(original["clean"] and common["clean"] and reduction is not None and reduction >= threshold)}


def save_trace(path, initial, rows):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **initial, **rows)


def episode(out, arm, seed, parent_seed, predict, policy_sha, plan):
    from tools89.course89 import make_env

    n_target = plan["ticks_per_episode"]
    stand_ticks = plan["stand_ticks"]
    folder = out / f"{arm}_{seed}"
    folder.mkdir()
    env = None
    rows, initial = {}, {}
    try:
        env = make_env("up", seed, curriculum=True, slope_start_x_m=plan["slope_start_x_m"],
                       perception=True, max_transitions=n_target)
        _, first = env.reset(seed=seed)
        obs, reset = env.reset()
        scales = env.action_scales[WHEEL_INDICES].copy()
        if (first["slope_start"] or first["episode_index"] != 0 or not reset["slope_start"]
                or reset["episode_index"] != 1 or reset["delay_ticks"] != 1
                or reset["start_x_m"] != plan["slope_start_x_m"] or env.data.time != 0.
                or env.counters()["native_steps"] != 0):
            raise ValueError("second uphill reset or delay differs")
        initial = {"initial_qpos": env.data.qpos.copy(), "initial_qvel": env.data.qvel.copy(),
                   "initial_obs": obs.copy(), "free_qpos_index": env.free_qpos,
                   "wheel_action_scales_rad_s": scales, "wheel_dof_indices": env.dof_ids[WHEEL_INDICES].copy()}
        initial_hash = hashlib.sha256()
        for key in ("initial_qpos", "initial_qvel", "initial_obs"):
            initial_hash.update(initial[key].tobytes())
        for tick in range(n_target):
            packet = env.last_raw_features.copy()
            if obs.shape != (ACTOR_DIM + 16,) or not np.isfinite(obs).all() or not np.array_equal(obs[ACTOR_DIM:], packet):
                raise ValueError("invalid causal range-policy input")
            observation = obs.copy()
            raw = predict(observation[:ACTOR_DIM], packet)
            issued, requested, feasible = wheel_decision(raw, scales, arm, tick, stand_ticks)
            obs, reward, terminated, truncated, info = env.step(issued)
            row = {
                "time_s": float(env.data.time), "qpos": env.data.qpos.copy(), "qvel": env.data.qvel.copy(),
                "observation": observation, "ranges": packet, "policy_raw": raw,
                "issued": env._last_policy_action.copy(), "applied": np.asarray(info["applied_action"]),
                "requested_rad_s": requested, "feasible_rad_s": feasible,
                "applied_wheel_target_rad_s": info["wheel_target_rad_s"], "last_torque_nm": info["last_torque_nm"],
                "command": info["command_during_interval"], "body_vx_m_s": info["body_vx_m_s"],
                "pitch_rad": info["base_pitch_rad"], "roll_rad": info["base_roll_rad"],
                "wheel_contacts_past_toe": info["loaded_wheels_past_toe"], "wheel_saturation_count": info["wheel_saturation_count"],
                "reward": reward, "fallen": info["fallen"], "illegal": info["illegal_body_contact"],
                "terminated": terminated, "truncated": truncated,
            }
            for name, value in row.items():
                rows.setdefault(name, []).append(value)
            if terminated or truncated:
                break
        rows = {k: np.asarray(v) for k, v in rows.items()}
        n = len(rows["time_s"])
        counts = env.counters()
        if (not all(np.isfinite(v).all() for v in rows.values())
                or not np.allclose(rows["time_s"], .02 * np.arange(1, n + 1), rtol=0., atol=1e-9)
                or not np.allclose(rows["command"][:, 0], np.where(np.arange(n) < stand_ticks, 0., .2), rtol=0., atol=1e-8)
                or np.any(rows["command"][:, 1:] != 0)
                or (counts["policy_transitions"], counts["pd_updates"], counts["native_steps"]) != (n, 5*n, 10*n)
                or (n == n_target and not rows["ranges"][:, 8:].any())):
            raise ValueError("timing, command, counters, perception or finite-state check failed")
        action_metrics = verify_actions(rows, scales, arm, stand_ticks)
        motion = motion_metrics(float(initial["initial_qpos"][env.free_qpos]), rows["qpos"][:, env.free_qpos], stand_ticks, n_target)
        clean = n == n_target and not rows["fallen"].any() and not rows["illegal"].any()
        status = "COMPLETE" if clean else "TERMINATED"
        trace = folder / ("trace.npz" if clean else "partial_trace.npz")
        save_trace(trace, initial, rows)
        case = {
            "arm": arm, "reset_seed": seed, "parent_seed": parent_seed, "policy_sha256": policy_sha,
            "initial_state_sha256": initial_hash.hexdigest(), "trace": str(trace.relative_to(out)), "trace_sha256": sha(trace),
            "status": status, "clean": bool(clean), "ticks": n, "pd_updates": counts["pd_updates"], "native_steps": counts["native_steps"],
            "fallen": bool(rows["fallen"].any()), "illegal": bool(rows["illegal"].any()),
            "wheel_torque_saturated_ticks": int(np.count_nonzero(rows["wheel_saturation_count"])), **motion, **action_metrics,
        }
        write(folder / "metrics.json", case)
        return case
    except BaseException as exc:
        if not (folder / "trace.npz").exists() and not (folder / "partial_trace.npz").exists():
            save_trace(folder / "partial_trace.npz", initial, rows)
        write(folder / "partial.json", {"error": repr(exc), "status": "PARTIAL",
              "physical_counters": env.counters() if env is not None else {}, "rows": len(rows.get("time_s", []))})
        raise
    finally:
        if env is not None:
            env.close()


def run(out: Path):
    """Guarded worker entry point. Claim the evidence directory before any inference or simulation."""
    out = Path(out)
    out.mkdir()
    cases, pairs = [], []
    try:
        from tools87.ppo87 import load_predictor

        specification = read(PLAN)
        plan = specification["stand"]
        prior = registered_plan()
        parents, resets = plan["parent_seeds"], plan["reset_seeds"]
        if (ROOT / specification["parent89"] != PLAN89 or sha(PLAN89) != specification["parent89_sha256"]
                or prior["parent_protocol"] != specification["parent87"]
                or prior["parent_protocol_sha256"] != specification["parent87_sha256"]):
            raise ValueError("parent provenance differs from the registered protocols")
        if (tuple(plan["arms"]) != ARMS or len(parents) != 3 or len(set(parents)) != 3
                or len(resets) != 3 or len(set(resets)) != 3):
            raise ValueError("exactly three registered pairs required")
        for parent_seed, seed in zip(parents, resets):
            folder = parent_folder(parent_seed, prior)
            policy_sha = sha(folder / "policy.zip")
            predict = load_predictor(folder)
            for arm in ARMS:
                cases.append(episode(out, arm, seed, parent_seed, predict, policy_sha, plan))
            pairs.append(pair_metrics(cases[-2], cases[-1], plan["min_rollback_reduction_m"]))
        report = {"status": "COMPLETE" if all(r["clean"] for r in cases) else "PARTIAL",
                  "protocol_sha256": sha(PLAN), "passed": all(p["passed"] for p in pairs),
                  "cases": cases, "pairs": pairs, "policy_transitions": sum(r["ticks"] for r in cases),
                  "pd_updates": sum(r["pd_updates"] for r in cases), "native_steps": sum(r["native_steps"] for r in cases),
                  "training_steps": 0}
        write(out / "report.json", report)
        return report
    except BaseException as exc:
        write(out / "partial.json", {"status": "PARTIAL", "error": repr(exc), "cases": cases, "pairs": pairs})
        raise
