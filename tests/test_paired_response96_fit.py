"""Analytic and synthetic tests for H8 sampling, tiny displacements and restored Adam."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools96 import fit96


def test_ordered_complete_h8_pairs_keep_sub_float32_motion(monkeypatch):
    def native(_folder, reset, tag):
        n = 10 if tag == "m1" else 12
        s = np.zeros((n + 1, 45), dtype=np.float64)
        s[:, 0] = 1.1 + (.001 * np.arange(n + 1))
        if tag == "p1":
            s[:, 0] += 2e-9 * np.maximum(0, np.arange(n + 1) - 1)
        t = {"state45": s, "applied16": np.zeros((n, 16), dtype=np.float32)}
        for field in ("obs643", "filtered3", "time_s", "sensor_capture_tick", "sensor_delivery_tick",
                      "ray_due", "ray_raw16", "pending_capture_tick", "pending_ray16"):
            t[field] = np.zeros(n + 1)
        return t, {"reset": reset, "tag": tag}

    monkeypatch.setattr(fit96, "_native", native)
    banks, target, roster = fit96._pair_bank(fit96.ROOT, 94011, {})
    assert roster.tolist() == [[r, j] for r in fit96.RESETS for j in range(3)]
    assert all(side[0].shape == (9, 45) and side[1].shape == (9, 8, 16) for side in banks)
    assert target.dtype == torch.float64 and target.shape == (9, 8)
    assert target[0].numpy() == pytest.approx(np.arange(8)*2e-9, abs=3e-16)
    assert target[2].numpy() == pytest.approx(np.arange(1, 9)*2e-9, abs=3e-16)
    assert float(target.square().mean()) < fit96.FLOOR_M2


def test_incomplete_pair_cannot_be_replaced_or_resumed(tmp_path, monkeypatch):
    def missing(*_args):
        raise ValueError("collection error")
    monkeypatch.setattr(fit96, "_native", missing)
    sources = {}
    with pytest.raises(ValueError, match="three matched"):
        fit96._pair_bank(tmp_path, 94011, sources)
    assert [r["reset"] for r in sources["new_pairs"]] == list(fit96.RESETS)
    monkeypatch.setattr(fit96, "_parent", missing)
    receipt = fit96.fit_seed(94011, tmp_path, tmp_path / "fit")
    assert receipt["status"] == "PARTIAL" and receipt["attempted_model_steps"] == 0
    assert all(a["status"] == "NOT_RUN" for a in receipt["arms"].values())
    with pytest.raises(FileExistsError):
        fit96.fit_seed(94011, tmp_path, tmp_path / "fit")


def test_fixed_batch_selection_with_replacement():
    seed = 94011
    old, paired = fit96._batch_plan(seed, 4, 2)
    reference = np.random.default_rng(seed + 960000)
    for a, b in zip(old, paired, strict=True):
        np.testing.assert_array_equal(a, reference.integers(4, size=16, dtype=np.int64))
        np.testing.assert_array_equal(b, reference.integers(2, size=8, dtype=np.int64))
    assert old.shape == (50, 16) and paired.shape == (50, 8)


class ScalarWorld(torch.nn.Module):
    """Toy x receives one linear increment from the issued synthetic input."""
    def __init__(self):
        super().__init__()
        self.residual = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("fixed_coeff", torch.ones(1))

    def step(self, state, applied, terrain):
        return torch.cat((state[:, :1] + self.residual(applied[:, :1]), state[:, 1:]), dim=1)


def test_warmstart_forks_share_batch_and_moments_without_aliasing(tmp_path, monkeypatch):
    parent = ScalarWorld()
    with torch.no_grad():
        parent.residual.weight.fill_(.1)
    opt = torch.optim.Adam(parent.residual.parameters(), lr=fit96.learn94.LR)
    parent.residual.weight.grad = torch.ones_like(parent.residual.weight)
    opt.step()
    opt.state[parent.residual.weight]["step"].fill_(100)
    packet = {"network": copy.deepcopy(parent.state_dict()), "optimizer": copy.deepcopy(opt.state_dict())}
    source = {"world94": {"weights_sha256": fit96.mapping_sha256(packet["network"])},
              "original_adam_sha256": fit96._adam_hash(packet["optimizer"]),
              "ridge_and_stats_sha256": fit96.mapping_sha256(dict(parent.named_buffers()))}

    def fresh(state):
        world = ScalarWorld()
        world.load_state_dict(state)
        return world.eval().requires_grad_(False)

    def bank(a):
        start = np.zeros((1, 45))
        start[0, 0] = 1.1
        start[0, 3] = 1.
        act = np.zeros((1, 8, 16))
        act[0, :, 0] = .01 * a
        target = np.repeat(start[:, None, :], 8, axis=1)
        target[0, :, 0] += .002 * a * np.arange(1, 9)
        return fit96._bank(start, act, target)

    monkeypatch.setattr(fit96, "_fresh_world", fresh)
    monkeypatch.setattr(fit96, "UPDATES", 1)
    oi, pi = fit96._batch_plan(94011, 1, 1)
    target_d = (.004 * torch.arange(1, 9, dtype=torch.float64)).unsqueeze(0)
    results = []
    for name in fit96.ARMS:
        row = fit96._fit_arm(name, tmp_path/name, packet, source, bank(0.), [bank(-1.), bank(1.)],
                             target_d, float(target_d.square().mean()), oi, pi)
        assert row["status"] == "COMPLETE", row["error"]
        assert row["completed_model_steps"] == 256 and row["updates_completed"] == 1
        assert row["initial_adam_sha256"] == source["original_adam_sha256"]
        assert fit96._adam_hash(packet["optimizer"]) == source["original_adam_sha256"]
        results.append(row)
    control, paired = (row["losses"][0] for row in results)
    assert control["state_loss"] == paired["state_loss"] and control["response_mse_m2"] == paired["response_mse_m2"]
    assert control["loss"] == control["state_loss"]
    assert paired["loss"] == pytest.approx(paired["state_loss"] + paired["normalized_pair_loss"])
    assert results[0]["checkpoint"]["weights_sha256"] != results[1]["checkpoint"]["weights_sha256"]
