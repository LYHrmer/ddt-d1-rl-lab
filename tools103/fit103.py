"""Compare action windows on the same frozen D1 observation embeddings."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools56.fit56 import array_sha256, mapping_sha256
from tools91.launch91 import require
from tools94.common94 import arrays, digest, read, write
from tools101.common101 import PUBLIC as PARENT
from tools101.fit101 import DeliveredPackets, SEEDS, _load_model, _modality_mse


ARMS = ("past", "transition")
UPDATES, BATCH_SIZE = 50, 48
PAIR_COUNTS = {"train": 960, "holdout": 300}
PARENT_FIT_SHA = "49a294a0d5197472d749928c625848f212be635262e3384f3f3a78f6958ae56f"


class ForecastHead(nn.Module):
    """Each modality sees the current hidden state and its own action clock."""

    def __init__(self):
        super().__init__()
        inputs = 32 + 9 * 17
        self.depth = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 32))
        self.body = nn.Sequential(nn.Linear(inputs, 32), nn.Tanh(), nn.Linear(32, 8))

    def forward(self, hidden, window):
        depth_input = torch.cat((hidden, window[:, 0].flatten(1)), dim=1)
        body_input = torch.cat((hidden, window[:, 1].flatten(1)), dim=1)
        return torch.cat((self.depth(depth_input), self.body(body_input)), dim=1)


def batch_schedule(pair_count, seed):
    """Indices into the eligible chronological row list. Each epoch visits every pair."""
    require(pair_count > 0 and pair_count % BATCH_SIZE == 0, "incomplete epoch of transition pairs")
    rng = np.random.default_rng(seed)
    samples = UPDATES * BATCH_SIZE
    epochs = (samples + pair_count - 1) // pair_count
    order = np.concatenate([rng.permutation(pair_count) for _ in range(epochs)])
    return order[:samples].reshape(UPDATES, BATCH_SIZE)


def _parent():
    path = PARENT / "fit/fit.json"
    require(digest(path) == PARENT_FIT_SHA, "round 101 fit identity changed")
    report = read(path)
    require(report["status"] == "COMPLETE" and report["seeds"] == list(SEEDS)
            and report["updates"] == len(SEEDS) * 2 * 50, "incomplete round 101 encoders")
    return report


def _dataset(split, packet_file, tape_file):
    from tools103.data103 import window_inputs

    windows = window_inputs(split, packet_file, tape_file)
    with np.load(packet_file, allow_pickle=False) as archive:
        packets = DeliveredPackets(archive)
        row_ids = archive["row_id"]
    rows = torch.nonzero(packets.successor >= 0).flatten()
    require(np.array_equal(row_ids, np.arange(len(row_ids))) and len(rows) == PAIR_COUNTS[split],
            "incomplete or reordered packet pairs")
    require(windows.shape == (len(row_ids), 2, 2, 9, 17) and windows.dtype == np.float32
            and np.isfinite(windows).all(), "invalid paired action window tensor")
    return packets, rows, torch.from_numpy(windows)[rows]


@torch.no_grad()
def frozen_features(model, packets, rows):
    """Encode each packet once, so a successor target equals that packet's own current feature."""
    all_rows = torch.arange(len(packets.history))
    features = torch.cat([model.encode(packets, chunk) for chunk in all_rows.split(128)])
    hidden = torch.cat([model.observe(packets, chunk)[1] for chunk in rows.split(128)])
    require(torch.isfinite(hidden).all() and torch.isfinite(features).all(), "non-finite frozen feature")
    return hidden, features[rows], features[packets.successor[rows]]


def _save_head(path, head, identity, count):
    state = head.state_dict()
    with path.open("xb") as f:
        torch.save(dict(model=state, updates=count, **identity), f)
    return dict(file=path.name, sha256=digest(path), state_sha256=mapping_sha256(state), updates=count)


def _fit_pair(seed, frozen, packets, rows, windows, folder, inputs, log, pair):
    hidden, _, target = frozen_features(frozen, packets, rows)
    schedule = batch_schedule(len(rows), seed)
    torch.manual_seed(seed)
    initial = ForecastHead()
    # Hash the original packet row IDs, including the gaps between separate series.
    pair.update(initial_state_sha256=mapping_sha256(initial.state_dict()),
                batch_schedule_sha256=array_sha256(rows.numpy()[schedule]), arms={})
    identity = dict(seed=seed, initial_state_sha256=pair["initial_state_sha256"],
                    batch_schedule_sha256=pair["batch_schedule_sha256"], encoder_sha256=pair["encoder_sha256"],
                    **inputs)
    heads = {arm: copy.deepcopy(initial) for arm in ARMS}
    optimizers = {arm: torch.optim.Adam(head.parameters(), lr=.003, betas=(.9, .999), eps=1e-8,
                                       weight_decay=0., foreach=False) for arm, head in heads.items()}
    counts = dict.fromkeys(ARMS, 0)
    try:
        for batch in schedule:
            idx = torch.from_numpy(batch)
            batch_hash = array_sha256(rows.numpy()[batch])
            for arm_id, arm in enumerate(ARMS):
                head, optimizer = heads[arm], optimizers[arm]
                optimizer.zero_grad(set_to_none=True)
                prediction = head(hidden[idx], windows[idx, arm_id])
                depth, body = _modality_mse(prediction, target[idx])
                loss = .5 * (depth.mean() + body.mean())
                require(torch.isfinite(loss), "non-finite prediction loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                counts[arm] += 1
                event = dict(seed=seed, arm=arm, update=counts[arm], batch_sha256=batch_hash,
                             depth_mse=depth.mean().item(), proprio_mse=body.mean().item(),
                             gradient_norm=float(norm))
                log.write(json.dumps(event, allow_nan=False) + "\n")
                log.flush()
        for arm, head in heads.items():
            require(all(torch.isfinite(p).all() for p in head.parameters()), "non-finite final head weights")
            pair["arms"][arm] = _save_head(folder / f"seed_{seed}_{arm}.pt", head,
                                           dict(arm=arm, **identity), counts[arm])
    except BaseException:
        for arm, head in heads.items():
            pair["arms"][arm] = _save_head(folder / f"partial_seed_{seed}_{arm}.pt", head,
                                           dict(arm=arm, **identity), counts[arm])
        raise


def fit(train_npz, train_tape, output_folder):
    """Fit only the six forecast heads. Both arms share each seed's observations and targets."""
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "fit.json").exists() or (folder / "updates.jsonl").exists() or any(folder.glob("*.pt")):
        raise FileExistsError(f"preserving existing fit in {folder}")
    write(folder / "fit_started.json", dict(seeds=list(SEEDS), arms=list(ARMS)))
    report = dict(status="PARTIAL", seeds=list(SEEDS), arms=list(ARMS), per_seed={})
    try:
        parent = _parent()
        inputs = dict(train_sha256=digest(train_npz), train_tape_sha256=digest(train_tape),
                      encoder_fit_sha256=PARENT_FIT_SHA)
        require(inputs["train_sha256"] == parent["train_sha256"], "fit requires the archived training split")
        report.update(inputs)
        packets, rows, windows = _dataset("train", train_npz, train_tape)
        with (folder / "updates.jsonl").open("x") as log:
            for seed in SEEDS:
                pair = dict(encoder_sha256=parent["per_seed"][str(seed)]["arms"]["predictive"]["sha256"])
                report["per_seed"][str(seed)] = pair
                frozen = _load_model(PARENT / "fit", parent, seed, "predictive")
                _fit_pair(seed, frozen, packets, rows, windows, folder, inputs, log, pair)
        require(digest(train_npz) == inputs["train_sha256"] and digest(train_tape) == inputs["train_tape_sha256"],
                "training input changed during fit")
        report.update(status="COMPLETE", update_log_sha256=digest(folder / "updates.jsonl"))
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["updates"] = sum(head["updates"] for pair in report["per_seed"].values()
                                for head in pair.get("arms", {}).values())
        write(folder / "fit.json", report)
    return report


def _load_head(folder, report, seed, arm):
    pair = report["per_seed"][str(seed)]
    entry = pair["arms"][arm]
    path = folder / f"seed_{seed}_{arm}.pt"
    require(entry["file"] == path.name and entry["updates"] == UPDATES and entry["sha256"] == digest(path),
            "incomplete or changed forecast head checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"arm": arm, "seed": seed, "updates": UPDATES, "encoder_fit_sha256": PARENT_FIT_SHA,
                "train_sha256": report["train_sha256"], "train_tape_sha256": report["train_tape_sha256"],
                "encoder_sha256": pair["encoder_sha256"], "initial_state_sha256": pair["initial_state_sha256"],
                "batch_schedule_sha256": pair["batch_schedule_sha256"]}
    require(all(payload.get(key) == value for key, value in expected.items())
            and mapping_sha256(payload["model"]) == entry["state_sha256"], "forecast head identity differs")
    head = ForecastHead().eval().requires_grad_(False)
    head.load_state_dict(payload["model"], strict=True)
    return head


@torch.no_grad()
def predict(holdout_npz, holdout_tape, fit_folder, output_file):
    """Emit paired forecasts on the development archive; subsequent observations are targets only."""
    folder, dest = Path(fit_folder), Path(output_file)
    if dest.exists():
        raise FileExistsError(f"preserving previous predictions in {dest}")
    parent = _parent()
    fit_path = folder / "fit.json"
    fit_hash = digest(fit_path)
    report = read(fit_path)
    require(report["status"] == "COMPLETE" and report["seeds"] == list(SEEDS) and report["arms"] == list(ARMS)
            and set(report["per_seed"]) == {str(seed) for seed in SEEDS}
            and report["encoder_fit_sha256"] == PARENT_FIT_SHA and report["train_sha256"] == parent["train_sha256"]
            and report["updates"] == len(SEEDS) * len(ARMS) * UPDATES
            and report["update_log_sha256"] == digest(folder / "updates.jsonl"), "incomplete or changed predictor fit")
    holdout_hash, tape_hash = digest(holdout_npz), digest(holdout_tape)
    require(holdout_hash != report["train_sha256"] and tape_hash != report["train_tape_sha256"],
            "development forecasts require the held-out data and action tape")
    packets, rows, windows = _dataset("holdout", holdout_npz, holdout_tape)
    currents, targets, predictions = [], [], []
    for seed in SEEDS:
        pair = report["per_seed"][str(seed)]
        require(set(pair["arms"]) == set(ARMS)
                and pair["encoder_sha256"] == parent["per_seed"][str(seed)]["arms"]["predictive"]["sha256"],
                "incomplete pair or wrong frozen encoder")
        frozen = _load_model(PARENT / "fit", parent, seed, "predictive")
        hidden, current, target = frozen_features(frozen, packets, rows)
        currents.append(current.numpy())
        targets.append(target.numpy())
        forecasts = []
        for arm_id, arm in enumerate(ARMS):
            head = _load_head(folder, report, seed, arm)
            forecast = head(hidden, windows[:, arm_id])
            require(torch.isfinite(forecast).all(), "non-finite development forecast")
            forecasts.append(forecast.numpy())
        predictions.append(np.stack(forecasts))
    require(digest(holdout_npz) == holdout_hash and digest(holdout_tape) == tape_hash and digest(fit_path) == fit_hash,
            "inputs changed during development forecasting")
    arrays(dest, dict(row_id=rows.numpy(), seeds=np.asarray(SEEDS), arms=np.asarray(ARMS),
                      current=np.stack(currents), target=np.stack(targets), prediction=np.stack(predictions),
                      holdout_sha256=holdout_hash, holdout_tape_sha256=tape_hash, fit_sha256=fit_hash))
    return dest
