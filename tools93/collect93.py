"""Collect the missing first-second physical arms of the crossed D1 stand panel."""
import hashlib
from pathlib import Path

import numpy as np

from tools89.common89 import parent_folder, registered_plan
from tools90.stand90 import ARMS, PLAN as PLAN90, save_trace, verify_actions, wheel_decision
from tools93.common93 import PLAN_SHA, ROOT, checked_plan, digest, read, require, write
from training.d1_rl.interface import ACTOR_DIM, WHEEL_INDICES


ARCHIVE = ROOT / "evidence/wbc90/stand_01"
HORIZON = 50
PARENTS = (87011, 87023, 87037)
RESETS = (90101, 90102, 90103)
INITIAL = ("initial_qpos", "initial_qvel", "initial_obs")
INTERFACE = ("free_qpos_index", "wheel_action_scales_rad_s", "wheel_dof_indices")
CLOCKS = ("policy_transitions", "pd_updates", "native_steps")


def new_arms():
    diagonal = set(zip(PARENTS, RESETS, strict=True))
    return [(parent, reset, arm) for parent in PARENTS for reset in RESETS
            if (parent, reset) not in diagonal for arm in ARMS]


def identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def array_digest(arrays):
    h = hashlib.sha256()
    for value in arrays:
        h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def reference_starts(prior):
    public = read(ROOT / "reports/stand_wbc90.json")
    stage = read(ARCHIVE / "report.json")
    old = read(PLAN90)["stand"]
    require(tuple(old["parent_seeds"]) == PARENTS and tuple(old["reset_seeds"]) == RESETS
            and tuple(old["arms"]) == ARMS and old["stand_ticks"] == HORIZON
            and old["ticks_per_episode"] == 100 and old["slope_start_x_m"] == 1.10, "90 protocol differs")
    require(digest(ARCHIVE / "report.json") == public["stand_sha256"] and stage == public["stand"]
            and stage["status"] == "COMPLETE" and stage["protocol_sha256"] == digest(PLAN90),
            "incomplete or changed physical reference")
    expected = [(p, s, a) for p, s in zip(PARENTS, RESETS, strict=True) for a in ARMS]
    require([(r["parent_seed"], r["reset_seed"], r["arm"]) for r in stage["cases"]] == expected,
            "archived diagonal inventory differs")
    starts, receipts = {}, {}
    for row in stage["cases"]:
        parent, reset, arm = row["parent_seed"], row["reset_seed"], row["arm"]
        name = f"{arm}_{reset}/trace.npz"
        path = ARCHIVE / name
        require(row["trace"] == name and not path.is_symlink() and digest(path) == row["trace_sha256"]
                and row["clean"] and row["status"] == "COMPLETE" and row["ticks"] == 100
                and row["policy_sha256"] == prior["parent_policy_sha256"][str(parent)], "diagonal source differs")
        with np.load(path, allow_pickle=False) as raw:
            current = {k: raw[k] for k in (*INITIAL, *INTERFACE)}
            for k in ("applied", "qpos", "qvel"):
                current["first_" + k] = raw[k][0]
        initial_sha = array_digest(current[k] for k in INITIAL)
        require(initial_sha == row["initial_state_sha256"] and not current["initial_qvel"].any()
                and not current["first_applied"].any(), "diagonal initial state, observation or FIFO differs")
        if reset in starts:
            require(all(identical(current[k], starts[reset][k]) for k in current),
                    "archived physical arms differ before intervention reaches the actuators")
        else:
            starts[reset] = current
        receipts[name] = row["trace_sha256"]
    return starts, receipts


def _start(env, reset, initial):
    _, first = env.reset(seed=reset)
    obs, second = env.reset()
    initial.update(initial_qpos=env.data.qpos.copy(), initial_qvel=env.data.qvel.copy(),
                   initial_obs=obs.copy(), free_qpos_index=np.asarray(env.free_qpos),
                   wheel_action_scales_rad_s=env.action_scales[WHEEL_INDICES].copy(),
                   wheel_dof_indices=env.dof_ids[WHEEL_INDICES].copy())
    require(first["episode_index"] == 0 and not first["slope_start"] and second["episode_index"] == 1
            and second["slope_start"] and second["start_x_m"] == 1.10 and second["delay_ticks"] == 1
            and env.data.time == 0. and all(env.counters()[k] == 0 for k in CLOCKS), "wrong second uphill reset")


def _row(env, obs, packet, raw, requested, feasible, result):
    next_obs, reward, terminated, truncated, info = result
    row = dict(time_s=float(env.data.time), qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
               observation=obs, ranges=packet, policy_raw=raw, issued=env._last_policy_action.copy(),
               applied=np.asarray(info["applied_action"]), requested_rad_s=requested, feasible_rad_s=feasible,
               applied_wheel_target_rad_s=info["wheel_target_rad_s"], last_torque_nm=info["last_torque_nm"],
               measured_wheel_speed_rad_s=env.data.qvel[env.dof_ids[WHEEL_INDICES]].copy(),
               command=info["command_during_interval"], body_vx_m_s=info["body_vx_m_s"], pitch_rad=info["base_pitch_rad"],
               roll_rad=info["base_roll_rad"], wheel_contacts_past_toe=info["loaded_wheels_past_toe"],
               wheel_saturation_count=info["wheel_saturation_count"], reward=reward, fallen=info["fallen"],
               illegal=info["illegal_body_contact"], terminated=terminated, truncated=truncated)
    return next_obs, row


def _check_finished(rows, initial, arm, clocks):
    n = len(rows["time_s"])
    require(0 < n <= HORIZON and tuple(clocks[k] for k in CLOCKS) == (n, 5*n, 10*n), "wrong physical clocks")
    require(all(len(v) == n and np.isfinite(v).all() for v in rows.values())
            and np.allclose(rows["time_s"], np.arange(1, n+1)*.02, rtol=0, atol=1e-9)
            and rows["command"].shape == (n, 3) and not rows["command"].any(), "invalid first-second records")
    require(rows["qpos"].shape == (n, 23) and rows["qvel"].shape == (n, 22)
            and identical(rows["measured_wheel_speed_rad_s"], rows["qvel"][:, initial["wheel_dof_indices"]])
            and rows["last_torque_nm"].shape == (n, 16) and rows["wheel_contacts_past_toe"].shape == (n, 4)
            and rows["wheel_contacts_past_toe"].dtype == np.bool_
            and rows["observation"].shape == (n, ACTOR_DIM + 16) and rows["ranges"].shape == (n, 16)
            and identical(rows["observation"][0], initial["initial_obs"])
            and identical(rows["observation"][:, ACTOR_DIM:], rows["ranges"]), "state, contact or policy input mismatch")
    require(n != HORIZON or rows["ranges"][:, 8:].any(), "no valid range feature in full stand")
    return verify_actions(rows, initial["wheel_action_scales_rad_s"], arm, HORIZON)


def episode(out, parent, reset, arm, policy_sha, predict, make_env, reference):
    require((parent, reset, arm) in new_arms(), "only registered missing physical arms may run")
    folder = out / f"parent{parent}" / f"{arm}_{reset}"
    folder.mkdir(parents=True, exist_ok=False)
    env, initial, rows = None, {}, {}
    case = dict(parent_seed=parent, reset_seed=reset, arm=arm, policy_sha256=policy_sha, initial_state_sha256=None,
                first_applied_sha256=None, first_endpoint_sha256=None, status="PARTIAL", clean=False, reason=None,
                ticks=0, policy_calls=0, **dict.fromkeys(CLOCKS, 0))
    try:
        env = make_env("up", reset, curriculum=True, slope_start_x_m=1.10, perception=True, max_transitions=HORIZON)
        _start(env, reset, initial)
        case["initial_state_sha256"] = array_digest(initial[k] for k in INITIAL)
        require(all(identical(v, reference[k]) for k, v in initial.items()),
                "reset or interface differs from same-reset archived diagonal")
        obs = initial["initial_obs"]
        for tick in range(HORIZON):
            observation, packet = obs.copy(), env.last_raw_features.copy()
            require(observation.shape == (ACTOR_DIM+16,) and np.isfinite(observation).all()
                    and identical(observation[ACTOR_DIM:], packet), "invalid causal policy observation")
            case["policy_calls"] += 1
            raw = predict(observation[:ACTOR_DIM], packet)
            issued, requested, feasible = wheel_decision(raw, initial["wheel_action_scales_rad_s"], arm, tick, HORIZON)
            obs, row = _row(env, observation, packet, raw, requested, feasible, env.step(issued))
            for key, value in row.items():
                rows.setdefault(key, []).append(value)
            case["ticks"] += 1
            if tick == 0:
                case["first_applied_sha256"] = array_digest([row["applied"]])
                case["first_endpoint_sha256"] = array_digest([row["qpos"], row["qvel"]])
                require(all(identical(row[k], reference["first_" + k]) for k in ("applied", "qpos", "qvel")),
                        "first applied action or physical endpoint differs from same-reset diagonal")
            if row["terminated"] or row["truncated"] or row["fallen"] or row["illegal"]:
                break
        rows = {k: np.asarray(v) for k, v in rows.items()}
        case.update(_check_finished(rows, initial, arm, env.counters()))
        case.update(fallen=bool(rows["fallen"].any()), illegal=bool(rows["illegal"].any()),
                    terminated=bool(rows["terminated"].any()), truncated=bool(rows["truncated"].any()),
                    wheel_torque_saturated_ticks=int(np.count_nonzero(rows["wheel_saturation_count"])))
        if case["ticks"] == HORIZON and not any(case[k] for k in ("fallen", "illegal", "terminated", "truncated")):
            x = int(initial["free_qpos_index"])
            x0, x1 = float(initial["initial_qpos"][x]), float(rows["qpos"][-1, x])
            case.update(status="COMPLETE", clean=True, initial_world_x_m=x0, world_x_at_1s_m=x1,
                        signed_stand_dx_m=x1-x0, rollback_m=max(0., x0-x1))
        else:
            case.update(status="TERMINATED", reason="episode terminated or truncated in registered stand window")
    except BaseException as exc:
        case["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if env is not None:
                clocks = env.counters()
                case.update({k: int(clocks[k]) for k in CLOCKS})
            trace = folder / ("trace.npz" if case["clean"] else "partial_trace.npz")
            save_trace(trace, initial, rows)
            case.update(trace=str(trace.relative_to(out)), trace_sha256=digest(trace))
            write(folder / "case.json", case)
        finally:
            if env is not None:
                env.close()
    return case


def run(out):
    """Claim a new directory and acquire the twelve off-diagonal one-second arms exactly once."""
    out = Path(out)
    out.mkdir(exist_ok=False)
    seal = dict(protocol_sha256=PLAN_SHA, status="PARTIAL", cases=[], policy_updates=0,
                world_model_updates=0, policy_calls=0, **dict.fromkeys(CLOCKS, 0))
    try:
        from tools87.ppo87 import load_predictor
        from tools89.course89 import make_env

        plan = checked_plan()
        prior = registered_plan()
        require(tuple(plan["parents"]) == PARENTS and tuple(plan["resets"]) == RESETS
                and tuple(plan["arms"]) == ARMS and plan["horizon"] == HORIZON
                and plan["policies_sha256"] == prior["parent_policy_sha256"],
                "93 grid, policies or horizon differ from the registered physical comparison")
        anchors, receipts = reference_starts(prior)
        seal["diagonal_trace_sha256"] = receipts
        folders = {p: parent_folder(p, prior) for p in PARENTS}
        policy_hashes = {p: digest(folders[p] / "policy.zip") for p in PARENTS}
        predictors = {p: load_predictor(folders[p]) for p in PARENTS}
        for parent, reset, arm in new_arms():
            try:
                row = episode(out, parent, reset, arm, policy_hashes[parent], predictors[parent], make_env, anchors[reset])
            except BaseException:
                path = out / f"parent{parent}" / f"{arm}_{reset}/case.json"
                if path.is_file():
                    seal["cases"].append(read(path))
                raise
            seal["cases"].append(row)
        if len(seal["cases"]) == 12 and all(r["clean"] for r in seal["cases"]):
            totals = [sum(r[k] for r in seal["cases"]) for k in ("policy_calls", *CLOCKS)]
            require(totals == [600, 600, 3000, 6000], "full panel differs from registered work")
            seal["status"] = "COMPLETE"
        else:
            seal["reason"] = "missing or incomplete physics arm"
    except BaseException as exc:
        seal["reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for k in ("policy_calls", *CLOCKS):
            seal[k] = sum(row[k] for row in seal["cases"])
        write(out / "seal.json", seal)
        (out / "seal.json").chmod(0o444)
    return seal
