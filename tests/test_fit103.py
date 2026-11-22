"""Synthetic checks of the clocks, frozen targets, matched updates and sealed heads."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools101.fit101 import DeliveredPackets, TerrainEstimator
from tools103 import fit103 as fit


def toy_packets(seed):
    rng = np.random.default_rng(seed)
    capture = np.tile([25, 34, 42, 50, 59], 2)
    n = len(capture)
    return dict(row_id=np.arange(n), series_id=np.repeat([0, 1], 5), capture_tick=capture,
                delivery_tick=capture + 2, valid=np.ones((n, 520), dtype=bool),
                depth_m=rng.uniform(.2, 1.2, (n, 520)).astype(np.float32),
                proprio=rng.normal(size=(n, 38)).astype(np.float32),
                action_history=rng.uniform(-1., 1., (n, 2, 16)).astype(np.float32))


def test_shuffle_covers_two_complete_epochs_then_half_with_stable_seed():
    order = fit.batch_schedule(960, 101011)
    assert order.shape == (50, 48)
    for epoch in (order[:20], order[20:40]):
        np.testing.assert_array_equal(np.sort(epoch.ravel()), np.arange(960))
    assert len(np.unique(order[40:])) == 480
    np.testing.assert_array_equal(order, fit.batch_schedule(960, 101011))
    assert not np.array_equal(order, fit.batch_schedule(960, 101023))


def test_each_modality_uses_only_its_own_clock():
    torch.manual_seed(11)
    head = fit.ForecastHead()
    hidden = torch.zeros(3, 32)
    window = torch.zeros(3, 2, 9, 17)
    window[:, :, :8, -1] = 1.
    original = head(hidden, window)
    for clock, own, other in ((0, slice(0, 32), slice(32, 40)), (1, slice(32, 40), slice(0, 32))):
        shifted = window.clone()
        shifted[:, clock, 2, 6] = .7
        predicted = head(hidden, shifted)
        assert torch.equal(predicted[:, other], original[:, other])
        assert not torch.equal(predicted[:, own], original[:, own])


def test_successor_changes_target_without_entering_the_current_encoder_state():
    model = TerrainEstimator().eval().requires_grad_(False)
    packets = DeliveredPackets(toy_packets(29))
    rows = torch.nonzero(packets.successor >= 0).flatten()
    model.normalize_from_training(packets)
    hidden, current, target = fit.frozen_features(model, packets, rows)
    assert rows.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert torch.equal(current[1], target[0])
    packets.proprio[1].add_(1.)
    packets.depth[1].mul_(.5)
    new_hidden, new_current, new_target = fit.frozen_features(model, packets, rows)
    assert torch.equal(hidden[0], new_hidden[0]) and torch.equal(current[0], new_current[0])
    assert not torch.equal(target[0], new_target[0]) and torch.equal(new_current[1], new_target[0])
    assert not any(t.requires_grad for t in (hidden, current, target))


@pytest.fixture
def tiny_fit(tmp_path, monkeypatch):
    seed = 101011
    train, holdout = tmp_path / "train.npz", tmp_path / "holdout.npz"
    np.savez(train, **toy_packets(1))
    np.savez(holdout, **toy_packets(2))
    train_tape, holdout_tape = tmp_path / "train-actions", tmp_path / "holdout-actions"
    train_tape.write_bytes(b"synthetic fit")
    holdout_tape.write_bytes(b"synthetic test")
    torch.manual_seed(seed)
    model = TerrainEstimator().eval().requires_grad_(False)
    initial = fit.mapping_sha256(model.state_dict())
    parent = dict(train_sha256=fit.digest(train), per_seed={str(seed): {"arms": {"predictive": {"sha256": initial}}}})
    monkeypatch.setattr(fit, "SEEDS", (seed,))
    monkeypatch.setattr(fit, "UPDATES", 2)
    monkeypatch.setattr(fit, "BATCH_SIZE", 4)
    monkeypatch.setattr(fit, "_parent", lambda: parent)
    monkeypatch.setattr(fit, "_load_model", lambda *args: model)

    def load_synthetic(split, packet_file, tape):
        with np.load(packet_file, allow_pickle=False) as archive:
            packets = DeliveredPackets(archive)
        rows = torch.nonzero(packets.successor >= 0).flatten()
        rng = np.random.default_rng(4)
        # Identical training windows test equality of every other aspect of the protocol.
        window = rng.uniform(-1., 1., (len(rows), 1, 2, 9, 17)).astype(np.float32)
        window[..., -1] = 1.
        windows = np.repeat(window, 2, axis=1)
        if split == "holdout":
            windows[:, 1, :, :, :16] *= -1.
        return packets, rows, torch.from_numpy(windows)

    monkeypatch.setattr(fit, "_dataset", load_synthetic)
    return seed, model, initial, train, train_tape, holdout, holdout_tape, tmp_path / "fit"


def test_identical_conditioning_gives_identical_heads_and_complete_checked_readback(tiny_fit, tmp_path):
    seed, model, initial, train, train_tape, holdout, holdout_tape, folder = tiny_fit
    receipt = fit.fit(train, train_tape, folder)
    assert receipt["status"] == "COMPLETE" and receipt["updates"] == 4
    pair = receipt["per_seed"][str(seed)]
    past, aligned = (pair["arms"][arm] for arm in fit.ARMS)
    assert past["state_sha256"] == aligned["state_sha256"] and pair["encoder_sha256"] == initial
    events = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
    assert len(events) == 4
    assert all(a["batch_sha256"] == b["batch_sha256"] and a["depth_mse"] == b["depth_mse"]
               for a, b in zip(events[::2], events[1::2]))
    output = tmp_path / "forecast.npz"
    fit.predict(holdout, holdout_tape, folder, output)
    assert fit.mapping_sha256(model.state_dict()) == initial
    with np.load(output, allow_pickle=False) as values:
        assert values["prediction"].shape == (1, 2, 8, 40)
        assert values["row_id"].tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
        assert values["arms"].tolist() == list(fit.ARMS)
        assert values["holdout_tape_sha256"].item() == fit.digest(holdout_tape)
        np.testing.assert_array_equal(values["current"][0, 1], values["target"][0, 0])
        assert not np.array_equal(values["prediction"][0, 0], values["prediction"][0, 1])
    with pytest.raises(FileExistsError):
        fit.predict(holdout, holdout_tape, folder, output)
    with pytest.raises(FileExistsError):
        fit.fit(train, train_tape, folder)
    with (folder / past["file"]).open("ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        fit.predict(holdout, holdout_tape, folder, tmp_path / "after.npz")


def test_failed_pair_keeps_both_heads_and_refuses_to_predict(tiny_fit, tmp_path, monkeypatch):
    seed, _, _, train, train_tape, holdout, holdout_tape, folder = tiny_fit
    adam_step = torch.optim.Adam.step
    calls = []

    def fail_second(optimizer):
        if calls:
            raise ValueError("synthetic interruption")
        calls.append(1)
        return adam_step(optimizer)

    monkeypatch.setattr(torch.optim.Adam, "step", fail_second)
    with pytest.raises(ValueError, match="synthetic interruption"):
        fit.fit(train, train_tape, folder)
    receipt = json.loads((folder / "fit.json").read_text())
    assert receipt["status"] == "PARTIAL" and receipt["updates"] == 1
    arms = receipt["per_seed"][str(seed)]["arms"]
    assert [arms[a]["updates"] for a in fit.ARMS] == [1, 0]
    for entry in arms.values():
        assert entry["file"].startswith("partial_") and fit.digest(folder / entry["file"]) == entry["sha256"]
    with pytest.raises(ValueError, match="incomplete"):
        fit.predict(holdout, holdout_tape, folder, tmp_path / "partial.npz")
    with pytest.raises(FileExistsError):
        fit.fit(train, train_tape, folder)
