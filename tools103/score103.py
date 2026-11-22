"""Compare both action windows against identical frozen next-packet features."""
from __future__ import annotations

import numpy as np

from tools94.common94 import digest
from tools101.score101 import (EPISODES, LAYOUTS, METADATA, SEEDS, SOURCE_SEED, TICKS,
                               VISIBILITIES, require, validate_holdout)


ARMS = ("past", "transition")
MODALITIES = {"depth": slice(0, 32), "proprio": slice(32, 40)}


def _intervals(rows, row_id):
    """Only the first 15 rows of each complete 16-packet stream have a successor."""
    validate_holdout(rows)
    series = rows["series_id"].reshape(-1, len(TICKS))
    capture = rows["capture_tick"].reshape(series.shape)
    require(np.all(series == series[:, :1]) and np.all(capture == TICKS),
            "packets are out of chronological order or cross a series boundary")
    starts = np.flatnonzero(rows["capture_tick"] != TICKS[-1])
    require(row_id.dtype.kind in "iu" and np.array_equal(row_id, starts),
            "forecast row IDs must contain every within-series interval exactly once in order")
    native = np.unique(np.column_stack((rows["episode"][starts], rows["capture_tick"][starts],
                                        rows["capture_tick"][starts + 1])), axis=0)
    require(len(starts) == len(native) * len(LAYOUTS) * len(VISIBILITIES),
            "mirror or visibility copies are missing")
    return starts, native


def _summary(errors, selected):
    # Each column is one arrival pair. Feature coordinates were averaged within
    # each modality first, so each selected pair has the same weight.
    past, transition, persistence = errors[:, selected].mean(axis=1).tolist()
    require(np.isfinite([past, transition, persistence]).all(), "nonfinite MSE")
    reduction = (1.0 - transition / past) * 100.0 if past > 0 else None
    require(reduction is None or np.isfinite(reduction), "nonfinite relative MSE")
    return {"past_mse": past, "transition_mse": transition, "persistence_mse": persistence,
            "transition_minus_past_mse": transition - past, "reduction_pct": reduction}


def _gate(views):
    checks = {}
    for modality in MODALITIES:
        masked, clear = views["masked"][modality], views["clear"][modality]
        ref, new = masked["past_mse"], masked["transition_mse"]
        checks[f"masked_{modality}_at_least_5pct"] = ref > 0 and new <= 0.95 * ref
        checks[f"masked_{modality}_beats_persistence"] = new < masked["persistence_mse"]
        checks[f"clear_{modality}_no_regression"] = clear["transition_mse"] <= clear["past_mse"]
    return checks


def score(holdout_npz, forecast_npz):
    """Grade complete paired forecasts on the five round-101 development episodes."""
    with np.load(holdout_npz, allow_pickle=False) as data:
        require(set(METADATA).issubset(data.files), "incomplete development packet roster")
        rows = {key: data[key] for key in METADATA}
    with np.load(forecast_npz, allow_pickle=False) as forecast:
        require({"row_id", "holdout_sha256", "seeds", "arms", "current", "target", "prediction"}
                .issubset(forecast.files), "incomplete forecast archive")
        require(forecast["holdout_sha256"].shape == ()
                and forecast["holdout_sha256"].item() == digest(holdout_npz),
                "forecast belongs to a different holdout file")
        require(forecast["seeds"].dtype.kind in "iu" and np.array_equal(forecast["seeds"], SEEDS),
                "missing or reordered fit seeds")
        require(np.array_equal(forecast["arms"], ARMS), "missing or reordered action windows")
        starts, native = _intervals(rows, forecast["row_id"])
        current, target, prediction = (forecast[k] for k in ("current", "target", "prediction"))
    shape = (len(SEEDS), len(starts), 40)
    require(current.shape == target.shape == shape
            and prediction.shape == (len(SEEDS), len(ARMS), len(starts), 40),
            "incorrect forecast or latent target shape")
    require(all(v.dtype.kind == "f" and np.isfinite(v).all() for v in (current, target, prediction)),
            "expected finite floating-point forecasts and targets on every interval")
    linked = np.diff(starts) == 1
    require(np.allclose(target[:, :-1, :][:, linked], current[:, 1:, :][:, linked],
                        atol=1e-6, rtol=1e-6), "next-packet target disagrees with the next frozen encoding")
    visibility, episodes = rows["visibility"][starts], rows["episode"][starts]
    views = {v: visibility == v for v in VISIBILITIES}
    by_seed = {}
    for i, seed in enumerate(SEEDS):
        # The two heads and persistence share the very same target tensor.
        estimates = np.concatenate((prediction[i], current[i, None]), axis=0).astype(np.float64)
        squared = np.square(estimates - target[i].astype(np.float64))
        errors = {name: squared[..., columns].mean(axis=-1) for name, columns in MODALITIES.items()}
        metrics = {v: {name: _summary(err, mask) for name, err in errors.items()}
                   for v, mask in views.items()}
        by_episode = {}
        for episode in EPISODES:
            use = episodes == episode
            by_episode[str(episode)] = {
                v: {name: _summary(err, mask & use) for name, err in errors.items()}
                for v, mask in views.items()}
        checks = _gate(metrics)
        by_seed[str(seed)] = {**metrics, "checks": checks, "passed": all(checks.values()),
                              "by_episode": by_episode}

    return {"fit_seeds": list(SEEDS), "native_source_seed": SOURCE_SEED, "native_episodes": len(EPISODES),
            "native_intervals": len(native), "native_decisions_in_intervals": int(np.sum(native[:, 2] - native[:, 1])),
            "paired_rows": len(starts), "pairs_per_view": {v: int(mask.sum()) for v, mask in views.items()},
            "by_seed": by_seed, "passing_seeds": sum(x["passed"] for x in by_seed.values()),
            "passed": all(x["passed"] for x in by_seed.values())}
