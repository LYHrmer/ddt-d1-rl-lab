"""CPU plant models at the applied-action / 20 ms whole-body PD boundary."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools52 import dynamics52 as d52
from tools56.fit56 import mapping_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_SEEDS = (91011, 91023, 91037)
BATCH_SEED = 91000
HORIZON, BATCH_SIZE, UPDATES = 8, 64, 300
RIDGE_LAMBDA, LEARNING_RATE, CLIP_NORM = 0.001, 3e-4, 1.0
OFFSETS_M = (0.25, 0., -0.25)  # front, center, rear along world x of the registered road
TOE_X_M, LENGTH_M, UP_GRADE = 0.60, 2.0, math.tan(math.radians(4.0))
INPUT_DIM = d52.INPUT_DIM + 2 * len(OFFSETS_M)
OLD_FOLDER = Path("evidence/world_vz75/execution_01")
OLD_REPORT_SHA256 = "26768d7378479f8038b89f793efab3cadcd801b2b9867c7e5daf49d15b9e3f79"
OLD_WEIGHTS_SHA256 = {
    64001: "2bf421e7a0d386e8b40de20613fff18154711fab3ee0f99ee6e7f444b8651719",
    64002: "1d05ee44067d4b033f3604a54975f4727d7ce2d9ec4109c84ae463a9e9a3804f",
    64003: "80afdbc19b856e8805eb0d4ef1b0156c3ee2076f5c3f601a2d3ee49aee872a16",
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _check_terrain(terrain, size):
    d52.require(isinstance(terrain, torch.Tensor) and terrain.device.type == "cpu"
                and terrain.dtype == torch.long and terrain.shape == (size,)
                and bool(((terrain >= 0) & (terrain <= 2)).all()),
                "terrain must be a CPU int64 [B] tensor with flat=0, up=1, down=2")


def terrain_features(states, terrain):
    """Three [road height minus predicted base z, dh/dx] pairs on the fixed 87 map.

    At each corner use the slope on its right, also for the height's autograd
    derivative. Only the dynamics consumes these map-derived features.
    """
    _check_terrain(terrain, len(states))
    x = states[:, :1] + states.new_tensor(OFFSETS_M)
    grade = ((terrain == 1).to(states.dtype) - (terrain == 2).to(states.dtype))[:, None] * UP_GRADE
    ramp = (x >= TOE_X_M) & (x < TOE_X_M + LENGTH_M)
    along = torch.where(x < TOE_X_M, 0., torch.where(ramp, x - TOE_X_M, LENGTH_M))
    return torch.stack((grade * along - states[:, 2:3], grade * ramp), dim=-1).flatten(1)


def input_features(states, applied, terrain):
    d52.require(states.ndim == 2 and states.shape[1] == d52.STATE_DIM
                and applied.shape == (len(states), d52.ACTION_DIM)
                and states.device.type == applied.device.type == "cpu"
                and states.dtype == applied.dtype and states.is_floating_point(),
                "CPU floating state [B,45] and applied action [B,16] required")
    masked = states.clone()
    masked[:, list(d52.ZERO_FEATURES)] = 0.
    return torch.cat((masked, applied, terrain_features(states, terrain)), dim=1)


def training_arrays(data):
    # The first stored endpoint of a complete 89 episode can supply s only
    # for its second interval. Keep its real clock for the input digest.
    source_tick, episode = np.asarray(data["tick"]), np.asarray(data["episode"])
    d52.require(source_tick.ndim == 1 and source_tick.shape == episode.shape
                and np.issubdtype(source_tick.dtype, np.integer) and (source_tick >= 0).all(),
                "source ticks must be a nonnegative integer vector aligned with episodes")
    tick = source_tick.astype(np.int64)
    for identifier in np.unique(episode):
        rows = np.flatnonzero(episode == identifier)
        tick[rows] -= tick[rows[0]]
    arrays = d52.validate_data({**data, "tick": tick})
    ids = np.asarray(data["terrain"])
    d52.require(ids.shape == arrays["episode"].shape and np.issubdtype(ids.dtype, np.integer)
                and ((0 <= ids) & (ids <= 2)).all(), "invalid terrain IDs in training data")
    same_episode = arrays["episode"][1:] == arrays["episode"][:-1]
    d52.require((ids[1:][same_episode] == ids[:-1][same_episode]).all(),
                "terrain changed within a training episode")
    return {**arrays, "terrain": ids, "source_tick": source_tick}


def fit_ridge(arrays):
    """Fit only the supplied training transitions; match 52 scaling and unpenalized bias."""
    x = input_features(torch.as_tensor(arrays["s"], dtype=torch.float32),
                       torch.as_tensor(arrays["a"], dtype=torch.float32),
                       torch.as_tensor(arrays["terrain"], dtype=torch.long)).numpy().astype(np.float64)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-3)
    z = np.c_[(x - mean) / scale, np.ones(len(x))]
    y = (arrays["target"].astype(np.float64) - arrays["s"]) / d52.PHYSICAL_SCALE
    penalty = RIDGE_LAMBDA * np.eye(INPUT_DIM + 1)
    penalty[-1, -1] = 0.
    coef = np.linalg.solve(z.T @ z / len(z) + penalty, z.T @ y / len(z))
    d52.require(np.isfinite(coef).all(), "nonfinite terrain ridge coefficients")
    return {"input_mean": mean.astype(np.float32), "input_scale": scale.astype(np.float32),
            "ridge_coef": coef.astype(np.float32)}


class TerrainDynamics91(nn.Module):
    def __init__(self, ridge=None):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(INPUT_DIM))
        self.register_buffer("input_scale", torch.ones(INPUT_DIM))
        self.register_buffer("ridge_coef", torch.zeros(INPUT_DIM + 1, d52.STATE_DIM))
        self.register_buffer("output_scale", torch.from_numpy(d52.PHYSICAL_SCALE.copy()))
        if ridge is not None:
            for name in ("input_mean", "input_scale", "ridge_coef"):
                getattr(self, name).copy_(torch.from_numpy(ridge[name]))
        self.residual = nn.Sequential(
            nn.Linear(INPUT_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, d52.STATE_DIM),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def step(self, states, applied, terrain):
        d52.require(states.dtype == self.input_mean.dtype, "state dtype must match model")
        features = input_features(states, applied, terrain)
        z = (features - self.input_mean) / self.input_scale
        delta = z @ self.ridge_coef[:-1] + self.ridge_coef[-1] + self.residual(z)
        predicted = states + delta * self.output_scale
        quat = predicted[:, 3:7]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        d52.require(bool(torch.isfinite(predicted).all()) and bool((norm > 1e-9).all()),
                    "nonfinite dynamics prediction or zero quaternion")
        return torch.cat((predicted[:, :3], quat / norm, predicted[:, 7:]), dim=-1)


class LegacyDynamics75(nn.Module):
    def __init__(self):
        super().__init__()
        self.plant = d52.Dynamics52()

    def step(self, states, applied, terrain):
        _check_terrain(terrain, len(states))
        return self.plant(states, applied)


def rollout_steps(model, initial, applied, terrain):
    """Predict each endpoint from the previous prediction and its applied action."""
    state, states = initial, []
    for action in applied.unbind(dim=1):
        state = model.step(state, action, terrain)
        states.append(state)
    return torch.stack(states, dim=1)


def load_old_models(root=ROOT):
    """The three complete control75 models, with the same strict checks as run80."""
    folder = Path(root) / OLD_FOLDER
    records = folder / "report.json"
    d52.require(not records.is_symlink() and _sha(records) == OLD_REPORT_SHA256, "75 source report identity")
    report = json.loads(records.read_text())
    d52.require(report["status"] == "COMPLETE", "partial 75 source report")
    result = []
    for seed, expected in OLD_WEIGHTS_SHA256.items():
        row, = [m for m in report["models"] if m["seed"] == seed and m["arm"] == "control"]
        fit = row["fit"]
        d52.require(row["status"] == fit["status"] == "COMPLETE" and fit["updates"] == fit["updates_attempted"] == 1000
                    and row["directory"] == f"control_seed{seed}", "partial or mismatched control75 fit")
        relative = f"control_seed{seed}/final_state_dict.pt"
        weights = folder / relative
        d52.require(not weights.is_symlink() and _sha(weights) == report["sealed_artifacts_sha256"][relative] == expected,
                    "control75 weights identity")
        model = LegacyDynamics75()
        model.plant.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        digest = mapping_sha256(model.plant.state_dict())
        d52.require(digest == fit["final_model_state_sha256"], "control75 parameter mapping identity")
        receipt = {"status": "COMPLETE", "seed": seed, "weights_sha256": expected,
                   "weights_file": str(OLD_FOLDER / relative), "model_sha256": digest,
                   "source_report_sha256": OLD_REPORT_SHA256, "updates": 1000}
        result.append((seed, model.eval().requires_grad_(False), receipt))
    return result


def _fit_one(seed, folder, tensors, windows, ridge, data_sha):
    folder.mkdir()
    torch.manual_seed(seed)
    model = TerrainDynamics91(ridge)
    optimizer = torch.optim.Adam(model.residual.parameters(), lr=LEARNING_RATE)
    sampler = torch.Generator(device="cpu").manual_seed(BATCH_SEED)
    completed = attempted = 0
    try:
        with (folder / "updates.jsonl").open("x", buffering=1) as log:
            for attempted in range(1, UPDATES + 1):
                indices = windows[torch.randint(len(windows), (BATCH_SIZE,), generator=sampler)]
                state = tensors["s"][indices[:, 0]]
                terrain = tensors["terrain"][indices[:, 0]]
                targets = tensors["target"][indices]
                actions = tensors["a"][indices]
                loss = d52.group_loss(rollout_steps(model, state, actions, terrain), targets)
                d52.require(bool(torch.isfinite(loss)), "nonfinite 91 training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.residual.parameters(), CLIP_NORM, error_if_nonfinite=True)
                optimizer.step()
                completed = attempted
                log.write(json.dumps({"update": completed, "loss": float(loss.detach()),
                                      "gradient_norm_before_clip": float(norm)}, allow_nan=False) + "\n")
        d52.require(all(bool(p.isfinite().all()) for p in model.parameters()), "nonfinite final 91 model")
        model.eval().requires_grad_(False)
        weights = folder / "final_state_dict.pt"
        with weights.open("xb") as stream:
            torch.save(model.state_dict(), stream)
        receipt = {"status": "COMPLETE", "seed": seed, "updates": completed, "updates_attempted": attempted,
                   "horizon": HORIZON, "batch_size": BATCH_SIZE, "batch_seed": BATCH_SEED,
                   "ridge_lambda": RIDGE_LAMBDA, "learning_rate": LEARNING_RATE, "clip_norm": CLIP_NORM,
                   "training_sha256": data_sha, "training_windows": len(windows),
                   "training_transitions": len(tensors["s"]), "weights_file": weights.name,
                   "weights_sha256": _sha(weights), "model_sha256": mapping_sha256(model.state_dict())}
        _write(folder / "fit.json", receipt)
        return seed, model, receipt
    except BaseException as exc:
        _write(folder / "partial.json", {"status": "PARTIAL", "seed": seed, "updates": completed,
                                       "updates_attempted": attempted, "error": f"{type(exc).__name__}: {exc}"})
        raise


def fit_models(data, out):
    """Fit the three final-only 91 candidates serially in a new output directory.

    The registered runner admits the resource-limited process and supplies only
    complete mixed89 training episodes with globally unique episode IDs.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays = training_arrays(data)
    digest = mapping_sha256(arrays)
    windows = torch.from_numpy(d52.window_indices(arrays, HORIZON))
    _write(out / "training.json", {"status": "STARTED", "seeds": MODEL_SEEDS, "training_sha256": digest,
                                  "training_transitions": len(arrays["s"]), "training_windows": len(windows),
                                  "training_episodes": np.unique(arrays["episode"]).tolist()})
    ridge = fit_ridge(arrays)
    tensors = {k: torch.as_tensor(arrays[k], dtype=torch.float32) for k in ("s", "a", "target")}
    tensors["terrain"] = torch.as_tensor(arrays["terrain"], dtype=torch.long)
    result = [_fit_one(seed, out / f"seed{seed}", tensors, windows, ridge, digest) for seed in MODEL_SEEDS]
    _write(out / "complete.json", {"status": "COMPLETE", "training_sha256": digest,
                                  "fit_receipts_sha256": {str(s): _sha(out / f"seed{s}/fit.json") for s in MODEL_SEEDS}})
    return result


def load_new_models(out):
    """Reload only a complete three-seed set with the recorded final weight identities."""
    out = Path(out)
    seal = json.loads((out / "complete.json").read_text())
    d52.require(seal["status"] == "COMPLETE" and set(seal["fit_receipts_sha256"]) == {str(s) for s in MODEL_SEEDS},
                "partial 91 model set")
    result = []
    for seed in MODEL_SEEDS:
        path = out / f"seed{seed}/fit.json"
        d52.require(not path.is_symlink() and _sha(path) == seal["fit_receipts_sha256"][str(seed)],
                    "91 fit receipt identity")
        receipt = json.loads(path.read_text())
        d52.require(receipt["status"] == "COMPLETE" and receipt["seed"] == seed
                    and receipt["updates"] == receipt["updates_attempted"] == UPDATES
                    and receipt["training_sha256"] == seal["training_sha256"]
                    and receipt["weights_file"] == "final_state_dict.pt", "partial or mismatched 91 fit")
        weights = path.parent / "final_state_dict.pt"
        d52.require(not weights.is_symlink() and _sha(weights) == receipt["weights_sha256"], "91 weights identity")
        model = TerrainDynamics91()
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        d52.require(mapping_sha256(model.state_dict()) == receipt["model_sha256"], "91 parameter mapping identity")
        result.append((seed, model.eval().requires_grad_(False), receipt))
    return result
