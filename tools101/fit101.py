"""Matched offline terrain estimators. Only the next-packet latent loss weight varies."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools56.fit56 import array_sha256, mapping_sha256
from tools94.common94 import arrays, digest, read, write


SEEDS = (101011, 101023, 101037)
WEIGHTS = {"baseline": 0., "predictive": 1.}
UPDATES, BATCH_SIZE, HISTORY = 50, 48, 3
HEIGHT_SCALE_MM = 30.


def _protocol(plan):
    if (tuple(plan["seeds"]) != SEEDS or plan["fit"]["updates_per_arm"] != UPDATES
            or plan["fit"]["batch_size"] != BATCH_SIZE):
        raise ValueError("unexpected seed, update count or batch size")


def temporal_indices(series, capture):
    """Index delivered history and the immediately following capture. -1 means absent."""
    if (series.ndim != 1 or not len(series) or series.shape != capture.shape
            or not np.issubdtype(series.dtype, np.integer) or not np.issubdtype(capture.dtype, np.integer)):
        raise ValueError("invalid packet chronology")
    history = np.full((len(series), HISTORY), -1, dtype=np.int64)
    successor = np.full(len(series), -1, dtype=np.int64)
    boundaries = np.concatenate(([0], np.flatnonzero(series[1:] != series[:-1]) + 1, [len(series)]))
    if len(set(series[boundaries[:-1]].tolist())) != len(boundaries) - 1:
        raise ValueError("a packet series recurs after its boundary")
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        clocks = capture[start:stop]
        block = (clocks * 6) // 50
        # The first capture can be any registered cut into a trajectory, but all
        # following rows must be consecutive acquisitions at the native 6 Hz.
        legal_capture = (clocks == 0) | (block != ((clocks - 1) * 6) // 50)
        if np.any(clocks < 0) or not np.all(legal_capture) or not np.all(np.diff(block) == 1):
            raise ValueError("nonconsecutive or repeated new packet")
        for row in range(start, stop):
            past = np.arange(max(start, row - HISTORY + 1), row + 1)
            history[row, -len(past):] = past
        successor[start:stop - 1] = np.arange(start + 1, stop)
    return torch.from_numpy(history), torch.from_numpy(successor)


class DeliveredPackets:
    """Student whitelist. No terrain labels, simulator poses, geometry or future observations."""

    def __init__(self, archive):
        series, capture = archive["series_id"], archive["capture_tick"]
        delivery = archive["delivery_tick"]
        if delivery.shape != capture.shape or not np.array_equal(delivery, capture + 2):
            raise ValueError("unexpected range delivery latency")
        self.history, self.successor = temporal_indices(series, capture)
        n = len(series)
        self.depth = torch.from_numpy(archive["depth_m"])
        self.valid = torch.from_numpy(archive["valid"])
        self.proprio = torch.from_numpy(archive["proprio"])
        actions = torch.from_numpy(archive["action_history"])
        if (self.depth.shape != (n, 520) or self.valid.shape != self.depth.shape or self.valid.dtype != torch.bool
                or self.proprio.shape != (n, 38) or actions.shape != (n, 2, 16)):
            raise ValueError("invalid dense depth, proprioception or issued action dimensions")
        if any(x.dtype != torch.float32 or not torch.isfinite(x).all()
               for x in (self.depth, self.proprio, actions)):
            raise ValueError("expected finite float32 student observations")
        if ((self.depth[~self.valid] != 0).any() or (self.depth[self.valid] <= 0).any()
                or actions.abs().max() > 1.000001):
            raise ValueError("inconsistent range validity or an unclipped issued action")
        self.actions = actions.reshape(n, 32)


def _mirror_eyes(rays):
    """Place the right sensor in the left sensor's azimuth order, retaining depression."""
    grid = rays.reshape(*rays.shape[:-1], 2, 20, 13)
    left = grid[..., 0, :, :].flatten(-2)
    right = grid[..., 1, :, :].flip([-1]).flatten(-2)
    return torch.stack((left, right), dim=-2)


class TerrainEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("depth_mean", torch.zeros(260))
        self.register_buffer("depth_scale", torch.ones(260))
        self.register_buffer("proprio_mean", torch.zeros(38))
        self.register_buffer("proprio_scale", torch.ones(38))
        self.eye = nn.Linear(520, 16)
        self.body = nn.Linear(38, 8)
        self.memory = nn.GRUCell(72, 32)
        self.height = nn.Linear(32, 1)
        self.predictor = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 40))

    @torch.no_grad()
    def normalize_from_training(self, packets):
        values, valid = _mirror_eyes(packets.depth).double(), _mirror_eyes(packets.valid)
        count = valid.sum((0, 1)).clamp(min=1)
        mean = torch.where(valid, values, 0.).sum((0, 1)) / count
        variance = torch.where(valid, (values - mean).square(), 0.).sum((0, 1)) / count
        self.depth_mean.copy_(mean)
        self.depth_scale.copy_(variance.sqrt().clamp(min=.015))
        self.proprio_mean.copy_(packets.proprio.double().mean(0))
        self.proprio_scale.copy_(packets.proprio.double().std(0, unbiased=False).clamp(min=.05))

    def encode(self, packets, rows):
        valid = _mirror_eyes(packets.valid[rows])
        depth = (_mirror_eyes(packets.depth[rows]) - self.depth_mean) / self.depth_scale
        depth = torch.where(valid, depth.clamp(-5., 5.), 0.)
        eyes = self.eye(torch.cat((depth, valid.to(depth.dtype)), dim=-1)).tanh()
        left, right = eyes.unbind(-2)
        proprio = ((packets.proprio[rows] - self.proprio_mean) / self.proprio_scale).clamp(-5., 5.)
        return torch.cat(((left + right) * .5, (left - right) * .5, self.body(proprio).tanh()), dim=-1)

    def observe(self, packets, rows):
        history = packets.history[rows]
        present = history >= 0
        indices = history.clamp(min=0)
        features = self.encode(packets, indices)
        inputs = torch.cat((features, packets.actions[indices]), dim=-1)
        state = inputs.new_zeros((len(rows), self.memory.hidden_size))
        for t in range(HISTORY):
            proposed = self.memory(inputs[:, t], state)
            state = torch.where(present[:, t, None], proposed, state)
        supported = (packets.valid[indices].any(-1) & present).any(-1)
        return self.height(state).squeeze(-1), state, features[:, -1], supported

    def next_features(self, state, actions):
        return self.predictor(torch.cat((state, actions), dim=-1))


def _modality_mse(prediction, target):
    depth, body = (prediction - target).square().split((32, 8), dim=-1)
    return depth.mean(-1), body.mean(-1)


def _anti_collapse(z):
    deficit = F.relu(.5 - (z.var(0, unbiased=False) + 1e-4).sqrt())
    return .5 * (deficit[:32].mean() + deficit[32:].mean())


def losses(model, packets, target_mm, rows):
    estimate, hidden, current, _ = model.observe(packets, rows)
    terrain = F.mse_loss(estimate, target_mm[rows] / HEIGHT_SCALE_MM)
    regularizer = _anti_collapse(current)
    next_rows = packets.successor[rows]
    paired = next_rows >= 0
    if paired.any():
        with torch.no_grad():
            targets = model.encode(packets, next_rows[paired])
        prediction = model.next_features(hidden[paired], packets.actions[rows[paired]])
        depth, body = _modality_mse(prediction, targets)
        forecast = .5 * (depth.mean() + body.mean())
    else:
        forecast = current.new_zeros(())
    return terrain, regularizer, forecast, int(paired.sum())


def batch_schedule(target_mm, visibility, seed):
    if (target_mm.ndim != 1 or visibility.shape != target_mm.shape or not np.isfinite(target_mm).all()
            or set(visibility.tolist()) != {"clear", "masked"}):
        raise ValueError("invalid training strata")
    side = np.where(target_mm > 10., 1, np.where(target_mm < -10., -1, 0))
    cells = [np.flatnonzero((side == sign) & (visibility == view))
             for sign in (-1, 0, 1) for view in ("clear", "masked")]
    if any(len(cell) == 0 for cell in cells):
        raise ValueError("an entire registered training stratum is empty")
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(np.concatenate([rng.choice(cell, BATCH_SIZE // 6, replace=True)
                                                      for cell in cells])) for _ in range(UPDATES)])


def _checkpoint(path, model, seed, arm, count, train_sha256, initial_hash, schedule_hash):
    with path.open("xb") as out:
        torch.save({"model": model.state_dict(), "seed": seed, "arm": arm, "lambda_next": WEIGHTS[arm],
                    "updates": count, "train_sha256": train_sha256,
                    "initial_state_sha256": initial_hash, "batch_schedule_sha256": schedule_hash}, out)
    return {"file": path.name, "sha256": digest(path), "state_sha256": mapping_sha256(model.state_dict()),
            "updates": count, "lambda_next": WEIGHTS[arm]}


def _fit_pair(packets, target_mm, visibility, seed, folder, train_hash, log):
    torch.manual_seed(seed)
    parent = TerrainEstimator()
    parent.normalize_from_training(packets)
    initial_hash = mapping_sha256(parent.state_dict())
    schedule = batch_schedule(target_mm.numpy(), visibility, seed)
    schedule_hash = array_sha256(schedule)
    models = {arm: copy.deepcopy(parent) for arm in WEIGHTS}
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=.003, betas=(.9, .999), eps=1e-8,
                                        weight_decay=0., foreach=False) for arm, model in models.items()}
    counts = {arm: 0 for arm in models}
    first, last, checkpoints = {}, {}, {}
    try:
        for batch in schedule:
            rows = torch.from_numpy(batch)
            for arm, model in models.items():
                optimizer = optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                terrain, regularizer, forecast, pairs = losses(model, packets, target_mm, rows)
                objective = terrain + .1 * regularizer + WEIGHTS[arm] * forecast
                if not torch.isfinite(objective):
                    raise FloatingPointError("non-finite perception objective")
                objective.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                counts[arm] += 1
                last[arm] = {"terrain_mse": float(terrain.detach()), "variance_penalty": float(regularizer.detach()),
                             "next_feature_mse": float(forecast.detach()), "pairs": pairs}
                first.setdefault(arm, last[arm])
                event = {"seed": seed, "arm": arm, "update": counts[arm], "batch_sha256": array_sha256(batch),
                         "gradient_norm": float(norm), **last[arm]}
                log.write(json.dumps(event, allow_nan=False) + "\n")
                log.flush()
        if first["baseline"] != first["predictive"]:
            raise ValueError("paired arms did not start on equal losses and rows")
        for arm, model in models.items():
            if any(not torch.isfinite(p).all() for p in model.state_dict().values()):
                raise FloatingPointError("non-finite final perception weights")
            path = folder / f"seed_{seed}_{arm}.pt"
            checkpoints[arm] = _checkpoint(path, model, seed, arm, counts[arm], train_hash, initial_hash, schedule_hash)
            checkpoints[arm].update(first=first[arm], last=last[arm])
    except Exception:
        for arm, model in models.items():
            _checkpoint(folder / f"partial_seed_{seed}_{arm}.pt", model, seed, arm, counts[arm],
                        train_hash, initial_hash, schedule_hash)
        raise
    return {"initial_state_sha256": initial_hash, "batch_schedule_sha256": schedule_hash, "arms": checkpoints}


def fit(train_npz, output_folder, plan):
    """Fit six isolated CPU estimators on one archive; the API accepts training rows only."""
    _protocol(plan)
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    train_hash = digest(train_npz)
    write(folder / "fit_started.json", {"seeds": SEEDS, "train_sha256": train_hash})
    report = {"status": "PARTIAL", "seeds": list(SEEDS), "train_sha256": train_hash, "per_seed": {}}
    try:
        with np.load(train_npz, allow_pickle=False) as data:
            packets = DeliveredPackets(data)
            target_mm = torch.from_numpy(data["target_mm"]).float()
            visibility = data["visibility"]
        if len(target_mm) != len(packets.history):
            raise ValueError("training labels are not aligned to packets")
        with (folder / "updates.jsonl").open("x") as log:
            for seed in SEEDS:
                report["per_seed"][str(seed)] = _fit_pair(packets, target_mm, visibility, seed, folder, train_hash, log)
        if digest(train_npz) != train_hash:
            raise ValueError("input archive changed during fitting")
        report.update(status="COMPLETE", updates=sum(arm["updates"] for seed in report["per_seed"].values()
                                                     for arm in seed["arms"].values()))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        write(folder / "fit.json", report)
        raise
    write(folder / "fit.json", report)
    return report


def _load_model(folder, report, seed, arm):
    pair = report["per_seed"][str(seed)]
    entry = pair["arms"][arm]
    path = folder / f"seed_{seed}_{arm}.pt"
    if (entry["file"] != path.name or digest(path) != entry["sha256"] or entry["updates"] != UPDATES
            or entry["lambda_next"] != WEIGHTS[arm]):
        raise ValueError("corrupt or incomplete perception checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"seed": seed, "arm": arm, "lambda_next": WEIGHTS[arm], "updates": UPDATES,
                "train_sha256": report["train_sha256"], "batch_schedule_sha256": pair["batch_schedule_sha256"],
                "initial_state_sha256": pair["initial_state_sha256"]}
    if any(payload[key] != value for key, value in expected.items()) or mapping_sha256(payload["model"]) != entry["state_sha256"]:
        raise ValueError("perception training identity differs")
    model = TerrainEstimator().eval().requires_grad_(False)
    model.load_state_dict(payload["model"], strict=True)
    return model


@torch.no_grad()
def _estimates(model, packets):
    estimates = []
    for rows in torch.arange(len(packets.history)).split(128):
        height, _, _, supported = model.observe(packets, rows)
        if not torch.isfinite(height).all():
            raise FloatingPointError("non-finite learned terrain estimate")
        estimates.append(torch.where(supported, height * HEIGHT_SCALE_MM, torch.nan))
    return torch.cat(estimates).numpy()


@torch.no_grad()
def _latent_diagnostics(model, packets, visibility):
    result = {}
    for view in ("clear", "masked"):
        rows = torch.from_numpy(np.flatnonzero(visibility == view))
        rows = rows[packets.successor[rows] >= 0]
        if not len(rows):
            raise ValueError(f"no held-out new-packet pairs in {view}")
        sums = torch.zeros(4, dtype=torch.float64)
        for chunk in rows.split(128):
            _, state, current, _ = model.observe(packets, chunk)
            target = model.encode(packets, packets.successor[chunk])
            prediction = model.next_features(state, packets.actions[chunk])
            errors = (*_modality_mse(prediction, target), *_modality_mse(current, target))
            sums += torch.stack([error.double().sum() for error in errors])
        mse = (sums / len(rows)).tolist()
        result[view] = dict(pairs=len(rows), predicted_depth_mse=mse[0], predicted_proprio_mse=mse[1],
                            persistence_depth_mse=mse[2], persistence_proprio_mse=mse[3])
    return result


def predict(holdout_npz, fit_folder, output_folder, plan):
    """Frozen students emit one estimate at every delivery, in original row order."""
    _protocol(plan)
    folder, dest = Path(fit_folder), Path(output_folder)
    dest.mkdir(parents=True, exist_ok=True)
    report = read(folder / "fit.json")
    if (report["status"] != "COMPLETE" or report["seeds"] != list(SEEDS)
            or set(report["per_seed"]) != {str(seed) for seed in SEEDS}
            or report["updates"] != len(SEEDS) * len(WEIGHTS) * UPDATES):
        raise ValueError("refusing incomplete perception fit")
    inputs_hash, fit_hash = digest(holdout_npz), digest(folder / "fit.json")
    if inputs_hash == report["train_sha256"]:
        raise ValueError("held-out archive is the training archive")
    write(dest / "predict_started.json", {"holdout_sha256": inputs_hash, "fit_sha256": fit_hash})
    with np.load(holdout_npz, allow_pickle=False) as archive:
        packets = DeliveredPackets(archive)
        visibility = archive["visibility"]
    values = {"seeds": np.asarray(SEEDS), "holdout_sha256": np.array(inputs_hash), "fit_sha256": np.array(fit_hash)}
    diagnostic = {"fit_sha256": fit_hash, "holdout_sha256": inputs_hash, "per_seed": {}}
    for arm in WEIGHTS:
        predictions = []
        for seed in SEEDS:
            model = _load_model(folder, report, seed, arm)
            predictions.append(_estimates(model, packets))
            if arm == "predictive":
                diagnostic["per_seed"][str(seed)] = _latent_diagnostics(model, packets, visibility)
        values[f"{arm}_mm"] = np.stack(predictions)
    if digest(holdout_npz) != inputs_hash:
        raise ValueError("held-out archive changed during prediction")
    write(dest / "latent.json", diagnostic)
    arrays(dest / "predictions.npz", values)
    return dest / "predictions.npz"
