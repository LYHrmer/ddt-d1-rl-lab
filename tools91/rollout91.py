"""Predict the two archived interventions from pre-intervention state only."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools87.run87 import PLAN as PLAN87, ROOT, cpp_actor, parent_inputs, read, sha, write
from tools88.probe88 import PLAN as PLAN88, apply_intervention
from tools90.stand90 import wheel_decision
from training.d1_rl.interface import (
    ACTION_SCALE, WHEEL_INDICES, build_observation, inverse_rotate, pack_actor_input,
)


# Only these six snapshot files cross the data boundary into the predictor.
STARTS = (
    ("move", 88101, 25), ("move", 88102, 25), ("move", 88103, 25),
    ("stand", 90101, 87011), ("stand", 90102, 87023), ("stand", 90103, 87037),
)
ARMS = {"move": ("original", "wheel_plus"), "stand": ("original", "common")}
START_TICK = {"move": 180, "stand": 0}
END_TICK = {"move": 300, "stand": 50}
START_FIELDS = {
    "state45": (np.float64, (45,)), "observation627": (np.float32, (627,)),
    "previous_filtered": (np.float64, (3,)), "fifo16": (np.float32, (16,)),
    "start_tick": (np.int64, ()), "parent_seed": (np.int64, ()), "reset_seed": (np.int64, ()),
}


@dataclass(frozen=True)
class Start:
    task: str
    reset_seed: int
    parent_seed: int
    tick: int
    state: np.ndarray
    observation: np.ndarray
    filtered: np.ndarray
    fifo: np.ndarray


def _load_start(path, task, seed, parent):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(START_FIELDS):
            raise ValueError(f"unexpected inputs in {path.name}")
        values = {name: archive[name] for name in START_FIELDS}
    for key, (dtype, shape) in START_FIELDS.items():
        data = values[key]
        if data.dtype != dtype or data.shape != shape or not np.isfinite(data).all():
            raise ValueError(f"invalid {key} in {path.name}")
    tick = START_TICK[task]
    if tuple(int(values[k]) for k in ("start_tick", "reset_seed", "parent_seed")) != (tick, seed, parent):
        raise ValueError(f"wrong cut, reset or parent in {path.name}")
    obs, fifo, state = values["observation627"], values["fifo16"], values["state45"]
    command = [0.4 if task == "move" else 0., 0., 0.]
    if (not np.all(obs.reshape(11, 57)[:, 6:9] == np.asarray(command, dtype=np.float32))
            or not np.array_equal(obs[41:57], fifo) or (np.abs(fifo) > 1).any()
            or abs(np.linalg.norm(state[3:7]) - 1) > 1e-6):
        raise ValueError(f"invalid command history, FIFO or pose at {path.name}")
    return Start(task, seed, parent, tick, state, obs, values["previous_filtered"], fifo)


class _PredictedRanges:
    """Frozen 87 ray geometry and delayed sampling on the imagined articulated pose."""

    def __init__(self, geometry):
        import mujoco

        self.model = geometry
        self.data = mujoco.MjData(geometry)
        self.base_body = geometry.body("base_link").id
        self._pending = None
        self.sensor_capture_tick = self.sensor_delivery_tick = -1
        self.last_raw_features = np.zeros(16, dtype=np.float32)

    def _measure(self):
        from tools87.terrain87 import D1Terrain87

        return D1Terrain87._measure(self)

    def at(self, tick, state):
        import mujoco
        from tools87.terrain87 import D1Terrain87, capture_due

        if capture_due(tick):
            self.data.qpos[:] = state[:23]
            mujoco.mj_kinematics(self.model, self.data)
        D1Terrain87._update_measurement(self, tick)
        return self.last_raw_features.copy()


def _body_vx(state):
    return float(inverse_rotate(state[3:7], state[23:26])[0])


def _fallen(state, minimum_z, maximum_tilt):
    gravity = inverse_rotate(state[3:7], [0., 0., -1.])
    tilt = np.arccos(np.clip(-gravity[2], -1., 1.))
    return bool(state[2] < minimum_z or tilt > maximum_tilt)


def _score(task, ticks, states):
    """Score only complete windows. State rows contain the initial endpoint."""
    if ticks[-1] != END_TICK[task]:
        return None
    if task == "stand":
        return max(0., float(states[0][0] - states[-1][0]))
    if ticks[0] != 180 or len(states) != 121:
        raise ValueError("moving score requires every endpoint 201..300")
    velocities = [_body_vx(s) for t, s in zip(ticks, states, strict=True) if t > 200]
    return float(np.mean(velocities, dtype=np.float64))


def _roll_arm(start, arm, model, predict, ranges, plan88, limits, folder):
    """Roll a fresh FIFO, gyro filter, observation history and range queue per arm."""
    import torch

    state, fifo = start.state.copy(), start.fifo.copy()
    current = start.observation[:57].copy()
    history = start.observation[57:].reshape(10, 57).copy()
    filtered = start.filtered.copy()
    stop = END_TICK[start.task]
    states, endpoints, issued_ticks = [state.copy()], [start.tick], []
    observations, packets, raw_actions, issued_actions, applied_actions = [], [], [], [], []
    rejected_state = []
    status, error = "COMPLETE", None
    terrain = torch.tensor([1], dtype=torch.int64, device="cpu")
    try:
        with torch.no_grad():
            for tick in range(start.tick, stop):
                body = pack_actor_input(current, history)
                features = ranges.at(tick, state) if ranges is not None else np.zeros(16, dtype=np.float32)
                raw = np.asarray(predict(body, features), dtype=np.float32)
                if start.task == "move":
                    issued, _ = apply_intervention(raw, tick, arm, plan88)
                else:
                    issued, _, _ = wheel_decision(raw, ACTION_SCALE[WHEEL_INDICES], arm, tick, 50)
                applied = fifo.copy()
                issued_ticks.append(tick)
                observations.append(body)
                packets.append(features.copy())
                raw_actions.append(raw.copy())
                issued_actions.append(issued.copy())
                applied_actions.append(applied)
                try:
                    result = model.step(torch.tensor(state[None], dtype=torch.float32),
                                        torch.from_numpy(applied[None]), terrain)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    status, error = "MODEL_ERROR", f"{type(exc).__name__}: {exc}"
                    break
                if (not torch.is_tensor(result) or result.dtype != torch.float32
                        or result.device.type != "cpu" or result.shape != (1, 45)):
                    raise ValueError("expected one CPU float32 world-model state [1,45]")
                next_state = result[0].detach().numpy().astype(np.float64)
                if not np.isfinite(next_state).all():
                    status, error = "NONFINITE", f"nonfinite state at endpoint {tick + 1}"
                    rejected_state.append(next_state)
                    break
                if abs(np.linalg.norm(next_state[3:7]) - 1) > 1e-4:
                    status, error = "DIVERGED", f"invalid unit quaternion at endpoint {tick + 1}"
                    rejected_state.append(next_state)
                    break
                state = next_state
                states.append(state.copy())
                endpoints.append(tick + 1)
                if _fallen(state, *limits):
                    status, error = "PREDICTED_FALL", f"predicted root height or tilt at endpoint {tick + 1}"
                    break
                fifo = issued.copy()
                history = np.concatenate((history[1:], current[None]))
                command = [.2 if tick + 1 >= 50 else 0., 0., 0.]
                current, filtered = build_observation(
                    state[26:29], state[3:7], command, state[7:23], state[29:45], issued, filtered,
                )
    except BaseException as exc:
        status, error = "PARTIAL", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trace = folder / ("trace.npz" if status == "COMPLETE" else "partial_trace.npz")
        with trace.open("xb") as stream:
            np.savez_compressed(
                stream, endpoint_tick=np.asarray(endpoints, dtype=np.int64),
                state45=np.asarray(states, dtype=np.float64), decision_tick=np.asarray(issued_ticks, dtype=np.int64),
                observation627=np.asarray(observations, dtype=np.float32).reshape(-1, 627),
                ranges16=np.asarray(packets, dtype=np.float32).reshape(-1, 16),
                raw16=np.asarray(raw_actions, dtype=np.float32).reshape(-1, 16),
                issued16=np.asarray(issued_actions, dtype=np.float32).reshape(-1, 16),
                applied16=np.asarray(applied_actions, dtype=np.float32).reshape(-1, 16),
                rejected_state45=np.asarray(rejected_state, dtype=np.float64).reshape(-1, 45),
            )
        case = {
            "task": start.task, "reset_seed": start.reset_seed, "parent_seed": start.parent_seed, "arm": arm,
            "status": status, "error": error, "attempted_steps": len(issued_ticks), "steps": len(endpoints) - 1,
            "last_endpoint": endpoints[-1],
            "score": _score(start.task, endpoints, states) if status == "COMPLETE" else None,
            "trace": str(Path(folder.name) / trace.name), "trace_sha256": sha(trace),
        }
        write(folder / "case.json", case)
    return case


def predict_suite(model, starts_dir, out):
    """Exclusive, one-model prediction entry point. Returns the saved pre-score report."""
    from tools87.ppo87 import load_predictor
    from tools87.terrain87 import load_model
    from tools89.common89 import parent_folder, registered_plan

    out, starts_dir = Path(out), Path(starts_dir)
    out.mkdir(exist_ok=False)
    report = {"status": "PARTIAL", "cases": [], "pairs": [], "max_imagined_steps": 1020,
              "physics_steps": 0, "training_steps": 0, "contact_labels": "unavailable"}
    try:
        plan88, plan87, plan89 = read(PLAN88), read(PLAN87), registered_plan()
        binary = parent_inputs(plan87)
        geometry = load_model("up")
        cfg = read(ROOT / "training/configs/pilot03d.json")
        default_pose = read(ROOT / "training/configs/diagnostic03b_zero.json")["static_derivation"]
        if default_pose["model_xml_sha256"] != cfg["robot_xml_sha256"]:
            raise ValueError("static nominal height is for a different robot")
        limits = (default_pose["nominal_root_height_m"] - cfg["termination"]["base_height_below_nominal_by_m"],
                  cfg["termination"]["tilt_above_rad"])
        for task, seed, parent in STARTS:
            start_path = starts_dir / f"{task}_{seed}.npz"
            start = _load_start(start_path, task, seed, parent)
            policy = load_predictor(parent_folder(parent, plan89)) if task == "stand" else None
            for arm in ARMS[task]:
                folder = out / f"{task}_{seed}_{arm}"
                folder.mkdir()
                # The only C++ actor memory is yaw command shaping. At this cut its
                # entire history is exactly zero, equal to fresh-process initialization.
                actor = cpp_actor(folder, plan87, binary) if task == "move" else nullcontext(policy)
                sensor = _PredictedRanges(geometry) if task == "stand" else None
                with actor as predict:
                    case = _roll_arm(start, arm, model, predict, sensor, plan88, limits, folder)
                case["start_sha256"] = sha(start_path)
                report["cases"].append(case)
            original, changed = report["cases"][-2:]
            a, b = original["score"], changed["score"]
            complete = a is not None and b is not None
            effect = (b - a if task == "move" else a - b) if complete else None
            report["pairs"].append({"task": task, "reset_seed": seed, "parent_seed": parent, "complete": complete,
                                    "original_score": a, "intervention_score": b, "predicted_effect": effect})
        report["status"] = "COMPLETE"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["complete_arms"] = sum(c["status"] == "COMPLETE" for c in report["cases"])
        report["imagined_steps"] = sum(c["attempted_steps"] for c in report["cases"])
        write(out / "predictions.json", report)
    return report
