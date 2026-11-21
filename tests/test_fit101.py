"""Synthetic packet chronology, gradient and sealed-checkpoint regressions."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools101 import fit101 as fit


def toy(seed=19):
    rng = np.random.default_rng(seed)
    stamps = np.array([0, 9, 17, 25, 34, 42])
    n = 2 * len(stamps)
    valid = rng.uniform(size=(n, 520)) > .05
    return dict(series_id=np.repeat([10, 30], len(stamps)), capture_tick=np.tile(stamps, 2),
                delivery_tick=np.tile(stamps, 2) + 2,
                depth_m=np.where(valid, rng.uniform(.15, 1.4, (n, 520)), 0.).astype(np.float32), valid=valid,
                proprio=rng.normal(size=(n, 38)).astype(np.float32),
                action_history=rng.uniform(-.5, .5, (n, 2, 16)).astype(np.float32),
                target_mm=np.tile([-30., 0., 30.], 4).astype(np.float32),
                visibility=np.repeat(["clear", "masked"], len(stamps)))


def test_causal_new_packets_restart_after_episode_and_reject_held_or_skipped_captures():
    data = toy()
    history, successor = fit.temporal_indices(data["series_id"], data["capture_tick"])
    assert history[[0, 2, 5, 6, 7]].tolist() == [[-1, -1, 0], [0, 1, 2], [3, 4, 5], [-1, -1, 6], [-1, 6, 7]]
    assert successor[[0, 5, 6, 11]].tolist() == [1, -1, 7, -1]
    for bad_capture in (9, 25, 18):
        copy = data["capture_tick"].copy()
        copy[2] = bad_capture
        with pytest.raises(ValueError, match="packet"):
            fit.temporal_indices(data["series_id"], copy)


def test_future_features_are_targets_only_and_action_conditioning_receives_gradient():
    data = toy()
    packets = fit.DeliveredPackets(data)
    model = fit.TerrainEstimator()
    model.normalize_from_training(packets)
    packets.proprio.requires_grad_()
    packets.actions.requires_grad_()
    row = torch.tensor([1])
    observed = model.observe(packets, row)[0].detach()
    _, _, forecast, pairs = fit.losses(model, packets, torch.from_numpy(data["target_mm"]), row)
    assert pairs == 1
    forecast.backward()
    assert packets.proprio.grad[row].abs().sum() > 0 and packets.proprio.grad[2:].count_nonzero() == 0
    assert packets.actions.grad[row].abs().sum() > 0 and packets.actions.grad[2:].count_nonzero() == 0
    assert model.predictor[0].weight.grad[:, 32:].abs().sum() > 0
    assert model.eye.weight.grad.abs().sum() > 0
    with torch.no_grad():
        packets.proprio[2].add_(3.)
    recomputed = model.observe(packets, row)[0].detach()
    _, _, changed, _ = fit.losses(model, packets, torch.from_numpy(data["target_mm"]), row)
    assert torch.equal(observed, recomputed) and changed.item() != forecast.item()
    _, _, last_forecast, pairs = fit.losses(model, packets, torch.from_numpy(data["target_mm"]), torch.tensor([5, 11]))
    assert pairs == 0 and last_forecast.item() == 0.


def test_missing_entire_history_is_unavailable_and_no_privileged_field_needed():
    data = toy()
    del data["target_mm"], data["visibility"]
    data["valid"][[0, 2, 3, 4, 5, 6]] = False
    data["depth_m"][~data["valid"]] = 0.
    packets = fit.DeliveredPackets(data)
    estimate = fit._estimates(fit.TerrainEstimator(), packets)
    assert np.array_equal(np.flatnonzero(np.isnan(estimate)), [0, 4, 5, 6])
    data["depth_m"][0, 5] = 1.
    with pytest.raises(ValueError, match="range validity"):
        fit.DeliveredPackets(data)


def test_sampling_balances_sign_and_visibility_and_is_exactly_paired():
    data = toy()
    schedule = fit.batch_schedule(data["target_mm"], data["visibility"], 11)
    assert np.array_equal(schedule, fit.batch_schedule(data["target_mm"], data["visibility"], 11))
    assert not np.array_equal(schedule, fit.batch_schedule(data["target_mm"], data["visibility"], 12))
    for sign in (-30., 0., 30.):
        for view in ("clear", "masked"):
            count = ((data["target_mm"][schedule] == sign) & (data["visibility"][schedule] == view)).sum(1)
            assert np.all(count == 8)
    with pytest.raises(ValueError, match="empty"):
        fit.batch_schedule(np.zeros_like(data["target_mm"]), data["visibility"], 11)


def test_one_synthetic_paired_update_can_be_sealed_and_replayed_without_holdout_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "SEEDS", (101011,))
    monkeypatch.setattr(fit, "UPDATES", 1)
    monkeypatch.setattr(fit, "BATCH_SIZE", 6)
    plan = {"seeds": [101011], "fit": {"updates_per_arm": 1, "batch_size": 6}}
    train, test = tmp_path / "train.npz", tmp_path / "holdout.npz"
    np.savez(train, **toy())
    held = toy(33)
    del held["target_mm"]
    np.savez(test, **held)
    root = tmp_path / "fit"
    receipt = fit.fit(train, root, plan)
    arms = receipt["per_seed"]["101011"]["arms"]
    assert receipt["updates"] == 2 and arms["baseline"]["first"] == arms["predictive"]["first"]
    assert arms["baseline"]["state_sha256"] != arms["predictive"]["state_sha256"]
    with pytest.raises(FileExistsError):
        fit.fit(train, root, plan)
    with pytest.raises(ValueError, match="training archive"):
        fit.predict(train, root, tmp_path / "same", plan)
    dest = tmp_path / "pred"
    output = fit.predict(test, root, dest, plan)
    with np.load(output, allow_pickle=False) as data:
        assert data["baseline_mm"].shape == data["predictive_mm"].shape == (1, 12)
        assert np.isfinite(data["predictive_mm"]).all() and str(data["holdout_sha256"]) == fit.digest(test)
    latent = json.loads((dest / "latent.json").read_text())["per_seed"]["101011"]
    assert latent["clear"]["pairs"] == latent["masked"]["pairs"] == 5
    with pytest.raises(FileExistsError):
        fit.predict(test, root, dest, plan)
    with (root / arms["baseline"]["file"]).open("ab") as out:
        out.write(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint"):
        fit.predict(test, root, tmp_path / "damaged", plan)


def test_variance_penalty_detects_collapsed_modalities_and_mse_weights_are_explicit():
    repeated = torch.zeros((2, 40))
    assert fit._anti_collapse(repeated) > .48
    spread = torch.cat((-torch.ones((1, 40)), torch.ones((1, 40))))
    assert fit._anti_collapse(spread).item() == 0.
    offset = repeated.clone()
    offset[:, :32] = 2.
    offset[:, 32:] = -3.
    depth, body = fit._modality_mse(offset, repeated)
    assert depth.tolist() == [4., 4.] and body.tolist() == [9., 9.]
