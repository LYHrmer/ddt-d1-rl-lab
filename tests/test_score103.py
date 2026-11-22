from itertools import product
import json

import numpy as np
import pytest

from tools94.common94 import digest
from tools101.score101 import EPISODES, LAYOUTS, SEEDS, SOURCE_SEED, TICKS, VISIBILITIES
from tools103.score103 import _gate, score


def example(tmp_path):
    keys = list(product(EPISODES, LAYOUTS, VISIBILITIES, TICKS))
    episode, layout, visibility, capture = map(np.array, zip(*keys, strict=True))
    n = len(keys)
    clouds = {key: i for i, key in enumerate(product(EPISODES, LAYOUTS, TICKS))}
    series = {key: i for i, key in enumerate(product(EPISODES, LAYOUTS, VISIBILITIES))}
    data = dict(row_id=np.arange(n), source_seed=np.full(n, SOURCE_SEED),
                episode=episode, layout=layout, side=np.where(layout == LAYOUTS[0], "left", "right"),
                visibility=visibility, capture_tick=capture, delivery_tick=capture + 2,
                cloud_id=np.array([clouds[e, l, t] for e, l, v, t in keys]),
                series_id=np.array([series[e, l, v] for e, l, v, t in keys]),
                target_mm=np.zeros(n), geometric_mm=np.zeros(n))
    path = tmp_path / "holdout.npz"
    np.savez(path, **data)
    starts = np.flatnonzero(capture != TICKS[-1])
    increments = np.repeat([1/16, 1/8], [32, 8])
    latent = np.tile(np.arange(len(TICKS)), n // len(TICKS))[:, None] * increments
    states = np.stack([latent + i / 2 for i in range(len(SEEDS))])
    current, target = states[:, starts], states[:, starts + 1]
    prediction = np.repeat((target + 2 * increments)[:, None], 2, axis=1)
    masked = visibility[starts] == "masked"
    prediction[:, 1, masked] = target[:, masked] + 0.5 * increments
    forecast = dict(holdout_sha256=digest(path), seeds=np.array(SEEDS), arms=np.array(["past", "transition"]),
                    row_id=starts, current=current, target=target, prediction=prediction)
    return path, data, forecast


def test_score_mse_by_modality_visibility_seed_and_distinct_native_interval(tmp_path):
    held, _, forecast = example(tmp_path)
    output = tmp_path / "forecast.npz"
    np.savez(output, **forecast)
    result = score(held, output)
    json.dumps(result, allow_nan=False)
    assert result["passed"] and result["passing_seeds"] == 3
    assert (result["native_intervals"], result["native_decisions_in_intervals"]) == (75, 625)
    assert result["paired_rows"] == 300 and result["pairs_per_view"] == {"clear": 150, "masked": 150}
    for seed in SEEDS:
        row = result["by_seed"][str(seed)]
        assert row["masked"]["depth"]["past_mse"] == (1/8)**2
        assert row["masked"]["depth"]["transition_mse"] == (1/32)**2
        assert row["masked"]["proprio"]["persistence_mse"] == (1/8)**2
        assert row["masked"]["proprio"]["reduction_pct"] == 93.75
        assert row["clear"]["depth"]["transition_minus_past_mse"] == 0
        assert all(v["masked"]["depth"]["transition_minus_past_mse"] < 0
                   for v in row["by_episode"].values())


def test_a_regression_on_one_seed_and_modality_is_retained(tmp_path):
    held, data, forecast = example(tmp_path)
    starts = forecast["row_id"]
    subset = (data["visibility"][starts] == "masked") & (data["episode"][starts] == EPISODES[-1])
    forecast["prediction"][1, 1, subset, 32:] = forecast["target"][1, subset, 32:] + 1
    out = tmp_path / "forecast.npz"
    np.savez(out, **forecast)
    result = score(held, out)
    assert not result["passed"] and result["passing_seeds"] == 2
    item = result["by_seed"][str(SEEDS[1])]
    assert item["checks"]["masked_depth_at_least_5pct"]
    assert not item["checks"]["masked_proprio_at_least_5pct"]
    # Each episode contributed equally many pairs of the two mirrored layouts.
    assert item["masked"]["proprio"]["transition_mse"] == pytest.approx((1 + 4 * (1/16)**2) / 5)
    assert item["by_episode"][str(EPISODES[-1])]["masked"]["proprio"]["transition_minus_past_mse"] > 0


def test_masked_boundary_persistence_and_zero_reference_have_separate_gates():
    values = {"past_mse": 100., "transition_mse": 95., "persistence_mse": 99.}
    views = {v: {m: values.copy() for m in ("depth", "proprio")} for v in VISIBILITIES}
    assert all(_gate(views).values())
    views["masked"]["depth"]["transition_mse"] = np.nextafter(95., np.inf).item()
    views["masked"]["proprio"].update(past_mse=0., transition_mse=0., persistence_mse=0.)
    views["clear"]["depth"]["transition_mse"] = 100.
    checks = _gate(views)
    assert not checks["masked_depth_at_least_5pct"]
    assert not checks["masked_proprio_at_least_5pct"] and not checks["masked_proprio_beats_persistence"]
    assert checks["clear_depth_no_regression"]
    views["clear"]["depth"]["transition_mse"] = 101.
    assert not _gate(views)["clear_depth_no_regression"]


@pytest.mark.parametrize("fault", [
    "seed_order", "arm_order", "missing_interval", "across_reset", "nonfinite", "wrong_target",
    "different_holdout", "interleaved_roster", "wrong_shape",
])
def test_rejects_incomplete_or_unpaired_forecasts(tmp_path, fault):
    held, data, forecast = example(tmp_path)
    if fault == "seed_order":
        forecast["seeds"] = forecast["seeds"][::-1]
    elif fault == "arm_order":
        forecast["arms"] = forecast["arms"][::-1]
    elif fault == "missing_interval":
        forecast["row_id"] = forecast["row_id"][:-1]
    elif fault == "across_reset":
        forecast["row_id"][14] += 1  # End of the first stream, followed by another stream.
    elif fault == "nonfinite":
        forecast["prediction"][0, 1, 39, 0] = np.nan
    elif fault == "wrong_target":
        forecast["target"][0, 5, 0] += 1
    elif fault == "different_holdout":
        forecast["holdout_sha256"] = "0" * 64
    elif fault == "interleaved_roster":
        for key in set(data) - {"row_id"}:
            data[key][[5, 16]] = data[key][[16, 5]]
        np.savez(held, **data)
        forecast["holdout_sha256"] = digest(held)
    else:
        forecast["target"] = forecast["target"][:, :, :-1]
    out = tmp_path / "forecast.npz"
    np.savez(out, **forecast)
    with pytest.raises(ValueError):
        score(held, out)
