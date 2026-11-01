"""Original-task data collection for experiment53, with durable failure evidence.

This module does not start an experiment at import.  All numerical/physical work
is owned by the serial53 runner.  Existing52 helpers remain unchanged; only the
registered tasks, seeds, duration and capture points differ from its collection.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from tools33.reference_shape33 import ReferenceShaper
from tools52 import adapter52 as adapter
from tools52 import control52 as control
from tools52.run_pilot52 import actor_admission, save_npz, save_points  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ROLES = ("train", "development", "probe")
TASKS = ("forward_stop", "reverse_stop", "turn_stop")
POINT_TICKS = (110, 260, 410)
EPISODE_TICKS = 600
DATA_SEEDS = {"train": list(range(53101, 53113)),
              "development": list(range(53201, 53204)),
              "probe": list(range(53301, 53304))}
COUNTERS = ("policy_transitions", "native_steps", "pd_updates", "forward_calls", "resets")
DATA_KEYS = ("s", "a", "target", "episode", "tick", "done")
require, write_json = control.require, control.write_json


def helpers(plan):
    """Load the original SHA-bound SEQUENCES and evaluation helpers without PPO."""
    from tools16.evaluate_comparison16 import frozen_helpers
    return frozen_helpers({"sources_sha256": {
        name: identity["sha256"] for name, identity in plan["identities"].items()}})


def make_env(cap, audit_step=None):
    """Construct one original pilot03d zero-delay environment, with optional audit.

    audit_step(env, issued_action, result) runs after the original step returns.
    The callback lives in a class-method closure, never in snapshot-copied
    env.__dict__.  It must only read the result/state, without forward/observe.
    A zero-command placeholder is replaced by a bound task before each reset.
    """
    from d1_rl.factory03 import environment_class
    cls = environment_class(ROOT / "training/configs/pilot03d.json")
    if audit_step is not None:
        require(callable(audit_step), "audit_step must be callable")

        class AuditedEnv(cls):
            def step(self, action):
                result = super().step(action)
                audit_step(self, action, result)
                return result

        cls = AuditedEnv
    env = cls(config=ROOT / "training/configs/pilot03d.json", arm="B", seed=53000,
              rank=0, max_transitions=cap, eval_delay_ticks=0,
              eval_command_sequence=[{"start_s": 0., "vx": 0., "wz": 0.}])
    require(env.model.nq == 23 and env.model.nv == 22
            and env.free_qpos == 0 and env.free_dof == 0, "unexpected physical dimensions")
    require(np.array_equal(env.qpos_ids, np.arange(7, 23))
            and np.array_equal(env.dof_ids, np.arange(6, 22)), "joint mapping differs")
    return env


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _event(stream, value):
    """Flush each event: completed rows establish a durable work lower bound."""
    stream.write(json.dumps(_plain(value), allow_nan=False) + "\n")
    stream.flush()


def _delta(after, before):
    return {key: int(after[key] - before[key]) for key in COUNTERS}


def _record(env, state, issued, result, episode, tick, *, final_tick):
    _obs, reward, terminated, truncated, info = result
    row = {"s": state.copy(), "a": env._applied_action.copy(),
           "target": control.physical(env), "episode": int(episode), "tick": int(tick),
           "done": bool(terminated or truncated or final_tick),
           "command_during_interval": np.asarray(info["command_during_interval"], dtype=np.float64),
           "next_command": env._command.copy(), "issued_action": np.asarray(issued).copy(),
           "applied_action": env._applied_action.copy(), "reward": float(reward),
           "fallen": bool(info["fallen"]), "illegal_body_contact": bool(info["illegal_body_contact"]),
           "base_height_m": float(info["base_height_m"]), "tilt_rad": float(info["tilt_rad"]),
           "terminated": bool(terminated), "truncated": bool(truncated)}
    row.update({key: int(value) for key, value in env.counters().items() if key in COUNTERS})
    return row


def _arrays(rows):
    """Stable empty/nonempty shapes, including a reset/first-step failure."""
    widths = {"s": 45, "a": 16, "target": 45, "command_during_interval": 3,
              "next_command": 3, "issued_action": 16, "applied_action": 16}
    boolean = {"done", "fallen", "illegal_body_contact", "terminated", "truncated"}
    integer = {"episode", "tick", *COUNTERS}
    names = (*DATA_KEYS, "command_during_interval", "next_command", "issued_action",
             "applied_action", "reward", "fallen", "illegal_body_contact", "base_height_m",
             "tilt_rad", "terminated", "truncated", *COUNTERS)
    result = {}
    for name in names:
        dtype = np.bool_ if name in boolean else np.int64 if name in integer else np.float64
        values = np.asarray([row[name] for row in rows], dtype=dtype)
        result[name] = values.reshape(len(rows), widths[name]) if name in widths else values
    return result


def _termination(row):
    if row is None:
        return []
    return [key for key in ("fallen", "illegal_body_contact", "terminated", "truncated") if row[key]]


def preflight(env, actor, output, plan):
    """Check160 original turn_stop steps, then repeat one32-step branch twice.

    Every completed mapping transition and failed gate is saved before raising.
    Full-state capture/restore remains the original52 implementation; the global
    make_env audit callback records branch-step diagnostics for the runner.
    """
    output = Path(output)
    spec = helpers(plan).SEQUENCES["turn_stop"]
    before = env.counters()
    rows, branch_results = [], []
    errors = {"observation": 0., "sensors": 0., "reference": 0.}
    report = {"status": "INCOMPLETE", "seed": 53000, "task": "turn_stop",
              "mapping_steps": 0, "branches": [{"status": "NOT_RUN"} for _ in range(2)],
              "counts_before": before, "failure": None}
    write_json(output / "preflight_initial.json", report)
    with (output / "preflight_events.jsonl").open("x") as events:
        try:
            env.eval_command_sequence = env._validate_eval_sequence(spec)
            _event(events, {"event": "RESET_BEGIN", "seed": 53000, "counts": env.counters()})
            env.reset(seed=53000)
            shaper = ReferenceShaper()
            for tick in range(160):
                raw = env.current_observation()
                with torch.no_grad():
                    predicted, _ = adapter.reference_step(control.tensor(raw[:57][None]),
                        control.tensor(raw[57:].reshape(1, 10, 57)), control.cache_tensor(shaper))
                oracle = copy.deepcopy(shaper)(raw)
                errors["reference"] = max(errors["reference"],
                    float(np.max(np.abs(predicted[0].numpy() - oracle))))
                previous_filter = env._filtered_gyro.copy()
                state = control.physical(env)
                action = control.real_action(env, shaper, actor, 0.)
                _event(events, {"event": "STEP_BEGIN", "tick": tick, "counts": env.counters(),
                                "command": env._command, "issued_action": action, "s": state})
                result = env.step(action)
                row = _record(env, state, action, result, 0, tick, final_tick=False)
                rows.append(row)
                report["mapping_steps"] = len(rows)
                _event(events, {"event": "STEP_COMPLETE", **row})
                require(not row["terminated"] and not row["truncated"],
                        "admission trajectory terminated: " + ",".join(_termination(row)))
                obs = result[0]
                state = row["target"]
                with torch.no_grad():
                    frame, filtered = adapter.build_observation(control.tensor(state[None], double=True),
                        control.tensor(env._command[None], double=True), control.tensor(action[None]),
                        control.tensor(previous_filter[None], double=True))
                errors["observation"] = max(errors["observation"],
                    float(np.max(np.abs(frame[0].numpy() - obs[:57]))),
                    float(np.max(np.abs(filtered[0].numpy() - env._filtered_gyro))))
                errors["sensors"] = max(errors["sensors"],
                    float(np.max(np.abs(state[26:29] - env._sensor("trunk_gyro")))),
                    float(np.max(np.abs(state[3:7] - env._sensor("trunk_quat")))),
                    float(np.max(np.abs(state[23:26] - env._sensor("frame_vel")))))
                _event(events, {"event": "MAPPING_CHECK", "tick": tick, "errors": errors})
                require(errors["observation"] <= 1e-6 and errors["sensors"] <= 1e-8
                        and errors["reference"] <= 1e-7, "state/observation/reference mapping failed")
            start = control.point(env, shaper, spec, 53000)
            branch_before = env.counters()
            for index in range(2):
                report["branches"][index] = {"status": "RUNNING"}
                _event(events, {"event": "BRANCH_BEGIN", "index": index, "counts": env.counters()})
                branch = control.real_branch(env, shaper, actor, start, 0.)
                branch_results.append(branch)
                report["branches"][index] = branch[0]
                save_npz(output / f"preflight_branch_{index}.npz", {
                    "states": branch[1], "observations": branch[2], "rewards": branch[3]})
                _event(events, {"event": "BRANCH_COMPLETE", "index": index,
                                "result": branch[0], "counts": env.counters()})
                require(branch[0]["status"] == "COMPLETE", "admission restore branch terminated")
            first, second = branch_results
            exact = first[0] == second[0] and all(np.array_equal(a, b)
                for a, b in zip(first[1:], second[1:], strict=True))
            save_npz(output / "restore_parity.npz", {"first_states": first[1], "second_states": second[1],
                "first_observations": first[2], "second_observations": second[2],
                "first_rewards": first[3], "second_rewards": second[3]})
            report["same_snapshot_32_step_exact"] = exact
            require(exact, "counterfactual restore is not exact")
            require(_delta(env.counters(), branch_before)["policy_transitions"] == 64,
                    "restore refunded work")
            require(_delta(env.counters(), before)["policy_transitions"] == 224,
                    "preflight transition count differs")
            report.update(status="COMPLETE", branch_work_not_refunded=True,
                          mjdata_narena_bytes=int(env.data.narena), mjdata_nbuffer_bytes=int(env.data.nbuffer),
                          formal_snapshot_count=18)
        except BaseException as exc:
            report["failure"] = f"{type(exc).__name__}: {exc}"
            report["termination_reasons"] = _termination(rows[-1] if rows else None)
            _event(events, {"event": "FAILED", "failure": report["failure"], "counts": env.counters()})
            raise
        finally:
            report.update(counts_after=env.counters(), actual_counts=_delta(env.counters(), before),
                          observation_max_abs_difference=errors["observation"],
                          sensor_max_abs_difference=errors["sensors"],
                          reference_max_abs_difference=errors["reference"])
            save_npz(output / "preflight_mapping.npz", _arrays(rows))
            write_json(output / "mapping_errors.json", errors)
            write_json(output / "preflight_diagnostics.json", report)
    return report


def collect(env, actor, output, plan):
    """Collect18 complete original12s cases; first failure stops the entire phase.

    Returns (datasets, points) only after all18 cases finish.  Every case keeps
    its full diagnostic NPZ and final JSON.  Events are flushed per transition,
    including attempted-step starts, so a killed process has an explicit
    completed-prefix lower bound rather than an invented zero-work report.
    """
    output = Path(output)
    require(plan["data_seeds"] == DATA_SEEDS, "53 registered role seeds differ")
    sequences = helpers(plan).SEQUENCES
    require(set(sequences) == set(TASKS), "original three-task set differs")
    cases = [{"role": role, "episode": index, "seed": seed,
              "task": TASKS[index % 3] if role == "train" else "turn_stop",
              "status": "NOT_RUN", "steps": 0, "complete_12s_trial": False}
             for role in ROLES for index, seed in enumerate(plan["data_seeds"][role])]
    before = env.counters()
    phase = {"status": "INCOMPLETE", "cases": cases, "counts_before": before, "failure": None}
    write_json(output / "collection_initial.json", phase)
    datasets, points = {}, {role: [] for role in ROLES}
    role_rows = {role: [] for role in ROLES}
    with (output / "collection_events.jsonl").open("x") as events:
        try:
            for case in cases:
                role, seed, index = case["role"], case["seed"], case["episode"]
                spec = sequences[case["task"]]
                context = {key: case[key] for key in ("role", "seed", "episode", "task")}
                rows, start_counts = [], env.counters()
                case.update(status="RUNNING", counts_before=start_counts)
                _event(events, {"event": "CASE_BEGIN", **context, "counts": start_counts})
                try:
                    env.eval_command_sequence = env._validate_eval_sequence(spec)
                    env.reset(seed=seed)
                    shaper = ReferenceShaper()
                    noise_rng = np.random.default_rng(seed + 900000)
                    for tick in range(EPISODE_TICKS):
                        if role != "train" and tick in POINT_TICKS:
                            point = control.point(env, shaper, spec, seed)
                            points[role].append(point)
                            name = f"snapshot_{role}_{seed}_{tick:03d}.pkl"
                            save_points(output / name, [point], env)
                            _event(events, {"event": "SNAPSHOT_SAVED", **context,
                                            "tick": tick, "file": name, "counts": env.counters()})
                        state = control.physical(env)
                        noise = noise_rng.normal(0., .02, 16) if role == "train" else None
                        issued = control.real_action(env, shaper, actor, 0., noise)
                        _event(events, {"event": "STEP_BEGIN", **context, "tick": tick,
                                        "counts": env.counters(), "command": env._command,
                                        "issued_action": issued, "s": state})
                        result = env.step(issued)
                        row = _record(env, state, issued, result, index, tick,
                                      final_tick=tick == EPISODE_TICKS - 1)
                        rows.append(row)
                        _event(events, {"event": "STEP_COMPLETE", **context, **row})
                        if row["terminated"] or row["truncated"]:
                            case["status"] = "TERMINATED"
                            raise RuntimeError("53 collection stopped at first early termination: "
                                               + ",".join(_termination(row)))
                    actual = _delta(env.counters(), start_counts)
                    require(len(rows) == 600 and actual["policy_transitions"] == 600
                            and actual["native_steps"] == 6000 and actual["pd_updates"] == 3000,
                            "complete collection counters differ")
                    case.update(status="COMPLETE", complete_12s_trial=True)
                except BaseException as exc:
                    if case["status"] != "TERMINATED":
                        case["status"] = "INCOMPLETE"
                    case["failure"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    role_rows[role].extend(rows)
                    case.update(steps=len(rows), counts_after=env.counters(),
                                actual_counts=_delta(env.counters(), start_counts),
                                completed_prefix_counts={"policy_transitions": len(rows),
                                    "native_steps": len(rows) * 10, "pd_updates": len(rows) * 5},
                                termination_reasons=_termination(rows[-1] if rows else None),
                                terminated=bool(rows and rows[-1]["terminated"]),
                                truncated=bool(rows and rows[-1]["truncated"]))
                    case["trace_file"] = f"collection_{role}_{seed}.npz"
                    save_npz(output / case["trace_file"], _arrays(rows))
                    write_json(output / f"collection_{role}_{seed}.json", case)
                    _event(events, {"event": "CASE_END", **case})
                if index == len(plan["data_seeds"][role]) - 1:
                    data = _arrays(role_rows[role])
                    datasets[role] = {key: data[key] for key in DATA_KEYS}
                    save_npz(output / f"{role}_transitions.npz", datasets[role])
                    if role != "train":
                        require(len(points[role]) == 9, "missing registered branch point")
                        save_points(output / f"{role}_snapshots.pkl", points[role], env)
            require(_delta(env.counters(), before)["policy_transitions"] == 10800,
                    "collection phase transition count differs")
            phase["status"] = "COMPLETE"
        except BaseException as exc:
            phase["failure"] = f"{type(exc).__name__}: {exc}"
            for case in cases:
                if case["status"] == "NOT_RUN":
                    case["reason"] = "stopped after first collection failure; no retry or continuation"
            for role in ROLES:
                if role_rows[role] and role not in datasets:
                    data = _arrays(role_rows[role])
                    save_npz(output / f"{role}_transitions_partial.npz", {key: data[key] for key in DATA_KEYS})
            _event(events, {"event": "COLLECTION_FAILED", "failure": phase["failure"],
                            "counts": env.counters()})
            raise
        finally:
            phase.update(counts_after=env.counters(), actual_counts=_delta(env.counters(), before))
            write_json(output / "collection.json", phase)
    return datasets, points
