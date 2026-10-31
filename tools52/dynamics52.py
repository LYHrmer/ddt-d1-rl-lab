"""CPU dynamics for the 52 experiment; the 20 ms action seam includes PD.

Only fit_model's explicitly supplied training episodes determine fitted values.
Logged actions condition model fitting, not a later policy-gradient rollout.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

STATE_DIM = 45
ACTION_DIM = 16
INPUT_DIM = STATE_DIM + ACTION_DIM
HORIZON = 8
UPDATES = 1000
BATCH_SIZE = 64
BATCH_SEED = 52000
WHEEL_JOINTS = (3, 7, 11, 15)
LEG_JOINTS = tuple(i for i in range(16) if i not in WHEEL_JOINTS)
ZERO_FEATURES = (0, 1, *(7 + i for i in WHEEL_JOINTS))
GROUPS = {
    "position": tuple(range(3)),
    "quaternion": tuple(range(3, 7)),
    "joint_position": tuple(range(7, 23)),
    "base_linear_velocity": tuple(range(23, 26)),
    "base_angular_velocity": tuple(range(26, 29)),
    "leg_velocity": tuple(29 + i for i in LEG_JOINTS),
    "wheel_velocity": tuple(29 + i for i in WHEEL_JOINTS),
}
PHYSICAL_SCALE = np.array(
    [0.1] * 7 + [1.0 if i in WHEEL_JOINTS else 0.5 for i in range(16)]
    + [0.3] * 6 + [5.0 if i in WHEEL_JOINTS else 2.0 for i in range(16)],
    dtype=np.float32,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_data(data):
    """Require complete, contiguous episodes; a terminal target is permitted."""
    keys = ("s", "a", "target", "episode", "tick", "done")
    require(all(k in data for k in keys), "missing dynamics data field")
    result = {k: np.asarray(data[k]) for k in keys}
    n = len(result["s"])
    require(n > 0, "empty dynamics data")
    for key, width in (("s", 45), ("a", 16), ("target", 45)):
        require(result[key].shape == (n, width), f"invalid {key} shape")
        require(np.isfinite(result[key]).all(), f"nonfinite {key}")
    for key in ("episode", "tick", "done"):
        require(result[key].shape == (n,), f"invalid {key} shape")
    for key in ("episode", "tick"):
        require(np.issubdtype(result[key].dtype, np.integer), f"{key} must be integer")
        require(np.all(result[key] >= 0), f"negative {key}")
    require(result["done"].dtype == np.bool_, "done must be boolean")
    require(np.all(np.abs(result["a"]) <= 1.0 + 1e-7), "action exceeds executed clip")
    for key in ("s", "target"):
        norms = np.linalg.norm(result[key][:, 3:7], axis=1)
        require(np.allclose(norms, 1.0, atol=1e-5, rtol=0), "nonunit data quaternion")
    for episode in np.unique(result["episode"]):
        indices = np.flatnonzero(result["episode"] == episode)
        require(np.all(np.diff(indices) == 1), "episode rows must be contiguous")
        require(np.array_equal(result["tick"][indices], np.arange(len(indices))),
                "episode ticks must start at zero and be contiguous")
        flags = result["done"][indices]
        require(bool(flags[-1]) and not np.any(flags[:-1]),
                "incomplete episode or transition after done")
        require(np.allclose(result["target"][indices[:-1]], result["s"][indices[1:]],
                            atol=1e-6, rtol=0), "discontinuous episode state")
    return result


def window_indices(data, horizon=HORIZON, stride=1):
    """Return windows within complete episodes, never across a done/reset."""
    require(type(horizon) is int and horizon > 0, "invalid horizon")
    require(type(stride) is int and stride > 0, "invalid stride")
    arrays = validate_data(data)
    windows = []
    for episode in np.unique(arrays["episode"]):
        rows = np.flatnonzero(arrays["episode"] == episode)
        for start in range(0, len(rows) - horizon + 1, stride):
            windows.append(rows[start:start + horizon])
    require(bool(windows), "no complete windows of requested horizon")
    return np.stack(windows)


def input_features(states, actions):
    """Remove absolute x/y and wheel phase only from predictor features."""
    features = np.concatenate((states, actions), axis=-1).copy()
    features[..., list(ZERO_FEATURES)] = 0
    return features


def fit_ridge(data):
    """Fit fixed ridge and input statistics using these training episodes only."""
    arrays = validate_data(data)
    x = input_features(arrays["s"], arrays["a"]).astype(np.float64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-3)
    z = np.c_[(x - mean) / scale, np.ones(len(x))]
    y = ((arrays["target"].astype(np.float64) - arrays["s"])
         / PHYSICAL_SCALE.astype(np.float64))
    penalty = 0.001 * np.eye(INPUT_DIM + 1)
    penalty[-1, -1] = 0
    coef = np.linalg.solve(z.T @ z / len(z) + penalty, z.T @ y / len(z))
    require(np.isfinite(coef).all(), "nonfinite ridge coefficients")
    return {"input_mean": mean.astype(np.float32),
            "input_scale": scale.astype(np.float32),
            "ridge_coef": coef.astype(np.float32)}


class Dynamics52(nn.Module):
    """Frozen ridge plus a residual; forward remains differentiable in inputs."""

    def __init__(self, input_mean=None, input_scale=None, ridge_coef=None):
        super().__init__()
        values = {
            "input_mean": np.zeros(INPUT_DIM) if input_mean is None else input_mean,
            "input_scale": np.ones(INPUT_DIM) if input_scale is None else input_scale,
            "ridge_coef": np.zeros((INPUT_DIM + 1, STATE_DIM)) if ridge_coef is None else ridge_coef,
        }
        for name, value in values.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).clone())
        mask = torch.ones(INPUT_DIM)
        mask[list(ZERO_FEATURES)] = 0
        self.register_buffer("feature_mask", mask)
        self.register_buffer("output_scale", torch.from_numpy(PHYSICAL_SCALE.copy()))
        self.residual = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, STATE_DIM),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, states, actions):
        require(states.ndim == 2 and states.shape[1] == STATE_DIM, "state must be [B,45]")
        require(actions.shape == (len(states), ACTION_DIM), "action must be [B,16]")
        require(states.device.type == actions.device.type == "cpu", "CPU tensors required")
        require(states.dtype == actions.dtype == self.input_mean.dtype,
                "state/action dtype must match the model (float32 by default)")
        x = torch.cat((states, actions), dim=-1) * self.feature_mask
        z = (x - self.input_mean) / self.input_scale
        delta = z @ self.ridge_coef[:-1] + self.ridge_coef[-1] + self.residual(z)
        result = states + delta * self.output_scale
        quat = result[:, 3:7]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        require(bool(torch.isfinite(result).all()) and bool((norm > 1e-9).all()),
                "nonfinite dynamics prediction or zero quaternion")
        return torch.cat((result[:, :3], quat / norm, result[:, 7:]), dim=-1)

    def predict(self, states, actions):
        return self(states, actions)

    def evaluate(self, data):
        return evaluate(self, data)


def rollout(model, initial, actions, *, return_trajectory=False):
    require(actions.ndim == 3 and actions.shape[0] == len(initial)
            and actions.shape[1] > 0 and actions.shape[2] == ACTION_DIM,
            "actions must be [B,H,16]")
    state = initial
    trajectory = []
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step])
        trajectory.append(state)
    return torch.stack(trajectory, dim=1) if return_trajectory else state


def group_loss(predicted, target):
    """Equal mean across the seven physical groups, all steps, and the batch."""
    require(predicted.shape == target.shape and predicted.shape[-1] == STATE_DIM,
            "loss states must have matching final dimension 45")
    scale = torch.as_tensor(PHYSICAL_SCALE, device=predicted.device, dtype=predicted.dtype)
    error = (predicted - target) / scale
    return torch.stack([error[..., list(indices)].square().mean()
                        for indices in GROUPS.values()]).mean()


def sequence_loss(model, initial, actions, targets):
    predicted = rollout(model, initial, actions, return_trajectory=True)
    return group_loss(predicted, targets)


def fit_model(data, seedspec, output, cpu_deadline):
    """One fixed final-only fit. The caller serializes the three model seeds."""
    seed = seedspec["seed"] if isinstance(seedspec, dict) else seedspec
    require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    require(np.isfinite(cpu_deadline), "finite absolute process CPU deadline required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    require(started < cpu_deadline, "CPU budget exhausted before fit")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays = validate_data(data)
    windows = torch.from_numpy(window_indices(arrays))
    baseline = fit_ridge(arrays)
    torch.manual_seed(seed)
    model = Dynamics52(**baseline)
    tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32)
               for key in ("s", "a", "target")}
    sampler = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=3e-4)
    final_loss = None
    with (output / "updates.jsonl").open("x", buffering=1) as stream:
        for update in range(UPDATES):
            require(time.process_time() < cpu_deadline, "CPU budget exhausted during fit")
            chosen = torch.randint(len(windows), (BATCH_SIZE,), generator=sampler)
            indices = windows[chosen]
            loss = sequence_loss(model, tensors["s"][indices[:, 0]],
                                 tensors["a"][indices], tensors["target"][indices])
            require(bool(torch.isfinite(loss)), "nonfinite multistep loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.residual.parameters(), 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            final_loss = float(loss.detach())
            stream.write(json.dumps({"update": update + 1, "loss": final_loss,
                                     "gradient_norm_before_clip": float(norm),
                                     "cpu_seconds": time.process_time() - started},
                                    allow_nan=False) + "\n")
    require(time.process_time() < cpu_deadline, "CPU budget exhausted after fit")
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "nonfinite final model")
    model.eval()
    model.requires_grad_(False)
    with (output / "final_state_dict.pt").open("xb") as stream:
        torch.save(model.state_dict(), stream)
    receipt = {
        "status": "COMPLETE", "seed": seed, "updates": UPDATES, "horizon": HORIZON,
        "batch_size": BATCH_SIZE, "batch_seed": BATCH_SEED, "learning_rate": 3e-4,
        "gradient_clip_norm": 1.0, "ridge_lambda": 0.001,
        "training_transitions": len(arrays["s"]), "training_windows": len(windows),
        "training_episodes": np.unique(arrays["episode"]).tolist(),
        "final_training_minibatch_loss": final_loss,
        "cpu_seconds": time.process_time() - started,
        "weights_file": "final_state_dict.pt", "checkpoint_selection": "final_only",
        "loss_groups": list(GROUPS), "physical_scale45": PHYSICAL_SCALE.tolist(),
        "zero_input_features": list(ZERO_FEATURES),
        "pd_semantics": "implicit in learned executed-action to 20 ms state transition",
    }
    with (output / "fit_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return model, receipt


def evaluate(model, data):
    """Episode-equal diagnostics; no selection threshold or fitted statistic."""
    arrays = validate_data(data)
    scores = {}
    with torch.no_grad():
        for horizon in (1, HORIZON):
            indices = window_indices(arrays, horizon, stride=horizon)
            rows = []
            for episode in np.unique(arrays["episode"]):
                selected = indices[arrays["episode"][indices[:, 0]] == episode]
                require(len(selected) > 0, "episode too short for evaluation horizon")
                predicted = []
                for first in range(0, len(selected), 256):
                    batch = selected[first:first + 256]
                    value = rollout(model, torch.as_tensor(arrays["s"][batch[:, 0]],
                                                          dtype=torch.float32),
                                    torch.as_tensor(arrays["a"][batch], dtype=torch.float32))
                    predicted.append(value.detach().numpy())
                prediction = np.concatenate(predicted)
                truth = arrays["target"][selected[:, -1]]
                error = prediction - truth
                normalized = error / PHYSICAL_SCALE
                grouped = {name: float(np.mean(normalized[:, list(group)] ** 2))
                           for name, group in GROUPS.items()}
                physical = {name: float(np.sqrt(np.mean(error[:, list(group)] ** 2)))
                            for name, group in GROUPS.items()}
                dot = np.sum(prediction[:, 3:7] * truth[:, 3:7], axis=1)
                angle = 2 * np.arccos(np.clip(np.abs(dot), 0, 1))
                rows.append({"episode": int(episode), "windows": len(selected),
                             "state_rmse45": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                             "group_mse": grouped, "group_rmse_physical": physical,
                             "mean_group_mse": float(np.mean(list(grouped.values()))),
                             "orientation_rmse_rad": float(np.sqrt(np.mean(angle ** 2)))})
            scores[str(horizon)] = {
                "horizon": horizon, "physical_seconds": 0.02 * horizon,
                "episodes": rows,
                "state_rmse45": np.mean([r["state_rmse45"] for r in rows], axis=0).tolist(),
                "group_mse": {name: float(np.mean([r["group_mse"][name] for r in rows]))
                              for name in GROUPS},
                "group_rmse_physical": {
                    name: float(np.mean([r["group_rmse_physical"][name] for r in rows]))
                    for name in GROUPS},
                "mean_group_mse": float(np.mean([r["mean_group_mse"] for r in rows])),
                "orientation_rmse_rad": float(np.mean([r["orientation_rmse_rad"] for r in rows])),
            }
    return scores
