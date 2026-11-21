import json

import pytest

from tools99 import common99, run99


def test_registration_counts_and_independent_resets():
    plan = common99.protocol()
    budget = plan["budget"]
    assert len(common99.SEEDS) * len(common99.MODELS) * budget["updates_per_arm"] == budget["world_updates"] == 150
    assert (18 * 16 + 6 * 16) * budget["world_updates"] == budget["fit_model_steps"] == 57_600
    assert len(common99.SEEDS) * len(common99.RESETS) * len(common99.ARMS) == budget["blind_native_episodes"] == 45
    assert budget["blind_forecasts"] * common99.HORIZON + budget["fit_model_steps"] == budget["model_steps_total"] == 60_480
    assert set(common99.RESETS).isdisjoint((*plan["paired_training_resets"], 98201, 98202, 98203))


def test_shared_native_answers_require_sealed_matched_predictions(tmp_path, monkeypatch):
    fits, predictions = (tmp_path / name for name in ("fit", "predictions"))
    fits.mkdir()
    predictions.mkdir()
    for seed in common99.SEEDS:
        dest = fits / f"seed_{seed}"
        dest.mkdir()
        status = "COMPLETE" if seed == common99.SEEDS[0] else "PARTIAL"
        (dest / "receipt.json").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(run99, "checked_stage", {"fit": fits, "predictions": predictions}.__getitem__)
    monkeypatch.setattr(run99, "measure", lambda folder, preview, seeds, resets, tags: (preview, seeds, resets, tags))
    (predictions / "seal.json").write_text('{"forecasts": 29}')
    with pytest.raises((RuntimeError, ValueError), match="seal all forecasts"):
        run99.native(tmp_path)
    (predictions / "seal.json").write_text('{"forecasts": 30}')
    assert run99.native(tmp_path) == (predictions, [common99.SEEDS[0]], common99.RESETS, tuple(common99.ARMS))
