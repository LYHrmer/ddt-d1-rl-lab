"""Collect each registered reset once and preserve measurements even when an episode ends early."""
from __future__ import annotations

import hashlib
import numpy as np

from tools94.common94 import arrays, digest, write
from tools109.cruise109 import (ACTOR_DIM, CHECKPOINT, CHECKPOINT_SHA, TICKS, FlatCruise109,
                                WHEEL_INDICES, command_at, issued_action)
from tools109.score109 import score


def sample(folder, arm, seed, predict):
    folder.mkdir(exist_ok=False)
    write(folder / "reservation.json", {"seed": seed, "arm": arm})
    env = None
    rows = {k: [] for k in ("time_s", "qpos", "qvel", "wheels_actual", "command", "command_history",
                             "ranges", "nominal", "issued", "wheel_target", "wheel_saturated",
                             "last_wheel_torque", "tilt", "fallen", "illegal")}

    def boundary():
        rows["time_s"].append(float(env.data.time))
        rows["qpos"].append(env.data.qpos.copy())
        rows["qvel"].append(env.data.qvel.copy())
        rows["wheels_actual"].append(env.data.qvel[env.dof_ids[WHEEL_INDICES]].copy())

    try:
        env = FlatCruise109(arm, seed)
        obs, _ = env.reset(seed=seed)
        if env.free_qpos != 0 or env.free_dof != 0:
            raise ValueError("unexpected floating-base layout")
        boundary()
        for k in range(TICKS):
            command = command_at(k)
            # The current frame precedes ten history frames ordered oldest first.
            ticks = [k] + [max(0, i) for i in range(k-10, k)]
            channels = obs[:ACTOR_DIM].reshape(11, 57)[:, 6]
            expected = np.asarray([2.*command_at(i) for i in ticks], dtype=np.float32)
            if not np.array_equal(channels, expected):
                raise ValueError("actor command or its history differs from the physical course")
            actor = predict(obs[:ACTOR_DIM], obs[ACTOR_DIM:])
            action = issued_action(arm, actor, command)
            ranges = obs[ACTOR_DIM:].copy()
            obs, _, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(info["command_during_interval"], [command, 0., 0.])
            boundary()
            values = dict(command=command, command_history=channels.copy(), ranges=ranges, nominal=actor.copy(),
                          issued=action, wheel_target=env.action_scales[WHEEL_INDICES]*env._applied_action[WHEEL_INDICES],
                          wheel_saturated=env._last_saturation[WHEEL_INDICES].copy(),
                          last_wheel_torque=env._last_tau[WHEEL_INDICES].copy(), tilt=info["tilt_rad"],
                          fallen=info["fallen"], illegal=info["illegal_body_contact"])
            for name, val in values.items():
                rows[name].append(val)
            if terminated or truncated:
                break
        trace = {k: np.asarray(v) for k, v in rows.items()}
        verdict = score(trace)
        counts = env.counters()
        n = verdict["ticks"]
        if (counts["native_steps"], counts["pd_updates"], counts["policy_transitions"]) != (10*n, 5*n, n):
            raise ValueError("incorrect native or servo clock")
        path = folder / "trace.npz"
        arrays(path, trace)
        reset = hashlib.sha256(trace["qpos"][0].tobytes()+trace["qvel"][0].tobytes()).hexdigest()
        report = {"status": "COMPLETE", "arm": arm, "seed": seed, "initial_state_sha256": reset,
                  "control": "actor_legs_referenced_wheels" if arm == "reference24" else "actor_all_16",
                  "counts": counts, "trace_sha256": digest(path), "score": verdict,
                  "peak_tilt_rad": float(np.max(trace["tilt"]))}
        write(folder / "report.json", report)
        return report
    except BaseException as exc:
        arrays(folder / "partial_trace.npz", {k: np.asarray(v) for k,v in rows.items()})
        write(folder / "partial.json", {"status": "PARTIAL", "error": repr(exc),
                                       "counts": env.counters() if env is not None else {}})
        raise
    finally:
        if env is not None:
            env.close()


def collect(out, arms, seeds):
    from tools89.train89 import load_predictor89

    if CHECKPOINT.is_symlink() or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("wrong speed109 actor")
    predict = load_predictor89(CHECKPOINT.parent)
    rows = []
    for seed in seeds:
        for arm in arms:
            rows.append(sample(out / f"{seed}_{arm}", arm, seed, predict))
    return rows
