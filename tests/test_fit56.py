"""Small synthetic56 pairing/loss tests; no1000-update fit, assets or physics."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lib():
    for directory in (ROOT, ROOT / ".python-deps"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from tools56 import fit56
    return SimpleNamespace(np=np, torch=torch, fit=fit56)


def data_fixture(lib, lengths=(40, 45)):
    np = lib.np
    n = sum(lengths)
    state = np.zeros((n, 45), np.float32)
    state[:, 3] = 1.
    episode = np.concatenate([np.full(length, index, np.int64) for index, length in enumerate(lengths)])
    tick = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    done = np.zeros(n, np.bool_)
    done[np.cumsum(lengths) - 1] = True
    return {"s": state, "a": np.zeros((n, 16), np.float32), "target": state.copy(),
            "episode": episode, "tick": tick, "done": done}


def test_both_arms_share_exact_starts_and_never_cross_a_done_or_reset(lib):
    data = data_fixture(lib)
    common = lib.fit.common_window_indices(data)
    assert common.shape == (40 - 32 + 1 + 45 - 32 + 1, 32)
    assert lib.np.array_equal(data["episode"][common], data["episode"][common[:, :1]].repeat(32, axis=1))
    assert not lib.np.any(data["done"][common[:, :-1]])
    assert lib.np.array_equal(data["tick"][common], data["tick"][common[:, :1]] + lib.np.arange(32))
    windows = lib.torch.from_numpy(common)
    chosen8 = lib.fit.sampled_window_rows(len(windows), updates=4, batch_size=5)
    chosen32 = lib.fit.sampled_window_rows(len(windows), updates=4, batch_size=5)
    assert lib.torch.equal(chosen8, chosen32)
    assert lib.fit.array_sha256(chosen8.numpy()) == lib.fit.array_sha256(chosen32.numpy())
    for first, second in zip(chosen8, chosen32, strict=True):
        short = lib.fit.selected_indices(windows, first, 8)
        long = lib.fit.selected_indices(windows, second, 32)
        assert lib.torch.equal(short, long[:, :8])
        assert lib.torch.equal(short[:, 0], long[:, 0])
    # Windows ending later than the common32 boundary cannot sneak into the8 arm.
    short_only_starts = set(range(9, 33)) | set(range(54, 78))
    assert not short_only_starts.intersection(common[:, 0].tolist())


def test_sampler_is_independent_of_model_rng_and_hash_records_values(lib):
    first = lib.fit.sampled_window_rows(13, updates=3, batch_size=4)
    lib.torch.manual_seed(56003)
    lib.torch.rand(100)
    second = lib.fit.sampled_window_rows(13, updates=3, batch_size=4)
    assert lib.torch.equal(first, second)
    changed = second.clone()
    changed[0, 0] = (changed[0, 0] + 1) % 13
    assert lib.fit.array_sha256(first.numpy()) != lib.fit.array_sha256(changed.numpy())
    data = data_fixture(lib)
    original = lib.fit.mapping_sha256(data)
    data["a"][0, 0] = .01
    assert lib.fit.mapping_sha256(data) != original


def test_horizon32_has_late_action_and_target_gradients_while8_excludes_them(lib):
    torch = lib.torch
    data = data_fixture(lib, lengths=(32,))
    windows = torch.from_numpy(lib.fit.common_window_indices(data))
    chosen = torch.tensor([0], dtype=torch.int64)
    initial = torch.tensor(data["s"])

    class Response(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, state, action):
            basis = torch.nn.functional.one_hot(torch.tensor(28), 45).to(state)
            return state + self.gain * action[:, :1] * basis

    results = {}
    for horizon in (8, 32):
        model = Response()
        actions = torch.zeros((32, 16))
        actions[16, 0] = .2
        actions.requires_grad_()
        targets = torch.tensor(data["target"], requires_grad=True)
        loss = lib.fit.paired_sequence_loss(model, {"s": initial, "a": actions, "target": targets},
                                           windows, chosen, horizon)
        loss.backward()
        results[horizon] = {"loss": loss.item(), "gain": model.gain.grad.item(),
                            "actions": actions.grad.clone(), "targets": targets.grad.clone()}
    assert results[8]["loss"] == results[8]["gain"] == 0.
    assert torch.count_nonzero(results[8]["actions"][8:]) == 0
    assert torch.count_nonzero(results[8]["targets"][8:]) == 0
    assert results[32]["loss"] > 0 and results[32]["gain"] > 0
    assert results[32]["actions"][16, 0] > 0
    assert torch.count_nonzero(results[32]["targets"][16:, 28]) > 0


def test_incomplete_episodes_cannot_produce_pairing_windows(lib):
    data = data_fixture(lib)
    data["done"][-1] = False
    with pytest.raises(ValueError, match="incomplete episode"):
        lib.fit.common_window_indices(data)


def test_expired_budget_writes_incomplete_evidence_without_model_or_update(lib, monkeypatch, tmp_path):
    output = tmp_path / "expired"
    monkeypatch.setattr(lib.fit.time, "process_time", lambda: 10.)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("expired admission must not construct a model or fit ridge")

    monkeypatch.setattr(lib.fit.frozen, "fit_ridge", forbidden)
    monkeypatch.setattr(lib.fit.frozen, "Dynamics52", forbidden)
    with pytest.raises(TimeoutError, match="before fit"):
        lib.fit.fit_model({}, 56001, 8, output, cpu_deadline=9.)
    partial = json.loads((output / "partial_receipt.json").read_text())
    assert partial["status"] == "INCOMPLETE" and partial["eligible_for_evaluation"] is False
    assert partial["updates"] == partial["updates_attempted"] == 0
    assert (output / "updates.jsonl").read_text() == ""
    assert not (output / "final_state_dict.pt").exists()
    assert not (output / "fit_receipt.json").exists()
    original = (output / "partial_receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        lib.fit.fit_model({}, 56001, 8, output, cpu_deadline=99.)
    assert (output / "partial_receipt.json").read_bytes() == original
