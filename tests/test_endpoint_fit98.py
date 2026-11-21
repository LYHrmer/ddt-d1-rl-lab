"""Round-98 tensor, archive and optimizer tests with synthetic traces and a scalar toy plant."""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools98 import fit98


def physical_batch():
    states = np.zeros((18, 17, 45), dtype=np.float64)
    states[:, :, 0] = 1.1
    states[:, :, 2] = .4
    states[:, :, 3] = 1.
    acts = np.full((18, 16, 16), .1, dtype=np.float32)
    acts[:, 0] = 0.
    return states, acts


def test_source_reads_only_the_h16_prefix_and_uses_applied_actions(tmp_path):
    n = 20
    states = np.zeros((n+1, 45), dtype=np.float64)
    states[17:, 0] = np.nan
    issued = np.repeat(np.arange(n, dtype=np.float32)[:, None]/32., 16, axis=1)
    fired = np.vstack((np.zeros_like(issued[:1]), issued[:-1]))
    fired[16:] = np.nan
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    path = tmp_path / "trace.npz"
    np.savez(path, state45=states, applied16=fired, issued16=issued, terminated=done,
             truncated=np.zeros(n, dtype=bool), reset_seed=17, completed_transitions=n)
    old_hash = fit98.digest(path)
    s, a = fit98._prefix(path, old_hash, 17)
    assert s.shape == (17, 45) and np.isfinite(s).all()
    np.testing.assert_array_equal(a, fired[:16])
    np.testing.assert_array_equal(a[-1], issued[14])
    with path.open("ab") as f:
        f.write(b"change")
    with pytest.raises(ValueError, match="changed source"):
        fit98._prefix(path, old_hash, 17)


def test_endpoint_is_all_groups_at_tick_16_and_pair_labels_are_float64():
    states, applied = physical_batch()
    states[15:, 2:9, 0] += 2.e-9 * np.arange(1, 8)[None, :]
    # Deliberate unsupervised discrepancy at ticks 9-15 and a single wheel rate error at tick 16.
    states[:, 9:16, 2] += 2.
    states[:, 16, 44] = 5.
    batch = fit98._batch(states, applied)
    baseline = batch["start"][:, None].expand(18, 16, 45).clone()
    state, mse, paired, endpoint = fit98._losses(baseline, batch, 2e-8)
    assert batch["paired_x_m"].dtype == torch.float64
    assert batch["paired_x_m"][0].numpy() == pytest.approx([0., *list(2e-9*np.arange(1, 8))], abs=2e-16)
    assert state.item() == pytest.approx(0., abs=1e-14) and mse.item() > 0.
    assert paired.item() == pytest.approx(mse.item()/2e-8)
    # One of four wheel velocities is off by its physical scale (5 rad/s); then average seven groups.
    assert endpoint.item() == pytest.approx(1/28)
    states[15, 1, 3] = .8
    with pytest.raises(ValueError, match="first zero-applied endpoint"):
        fit98._batch(states, applied)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = torch.nn.Linear(1, 1, bias=False)
        self.register_buffer("unchanged", torch.ones(1))

    def step(self, state, applied, terrain):
        assert terrain.tolist() == [1]*18
        return torch.cat((state[:, :44], state[:, 44:] + self.residual(applied[:, :1])), dim=-1)


def toy_source():
    parent = Toy()
    opt = torch.optim.Adam(parent.residual.parameters(), lr=fit98.learn94.LR)
    parent.residual.weight.grad = torch.ones_like(parent.residual.weight)
    opt.step()
    opt.state[parent.residual.weight]["step"].fill_(150)
    with torch.no_grad():
        parent.residual.weight.zero_()
    packet = {"network": copy.deepcopy(parent.state_dict()), "optimizer": copy.deepcopy(opt.state_dict())}
    source = {"parent_checkpoint": {"weights_sha256": fit98.mapping_sha256(packet["network"]),
                                     "adam_sha256": fit98.fit96._adam_hash(packet["optimizer"])},
              "ridge_and_stats_sha256": fit98.mapping_sha256(dict(parent.named_buffers()))}
    return packet, source


def test_two_forks_differ_only_by_final_term_and_never_change_parent(tmp_path, monkeypatch):
    packet, refs = toy_source()
    states, applied = physical_batch()
    states[:, 16, 44] = 2.

    def source(_seed, record):
        record.update(refs)
        return packet, 1e-7, states, applied

    def fresh(weights):
        world = Toy()
        world.load_state_dict(weights)
        return world.eval().requires_grad_(False)

    monkeypatch.setattr(fit98, "_training_sources", source)
    monkeypatch.setattr(fit98.fit96, "_fresh_world", fresh)
    monkeypatch.setattr(fit98, "UPDATES", 1)
    path = tmp_path / "trial"
    result = fit98.fit_seed(94011, path)
    assert result["status"] == "COMPLETE", result["error"]
    assert result["updates_completed"] == 2 and result["attempted_model_steps"] == result["completed_model_steps"] == 576
    assert result["ppo_updates"] == result["policy_updates"] == 0
    parents = []
    for name, weight in fit98.WEIGHTS.items():
        arm = result["arms"][name]
        l = arm["losses"][0]
        assert arm["initial_adam_sha256"] == refs["parent_checkpoint"]["adam_sha256"]
        assert arm["updates_completed"] == 1 and arm["completed_model_steps"] == 288
        assert l["loss"] == pytest.approx(l["state_loss_h8"] + l["normalized_pair_loss_h8"] + weight*l["endpoint_loss_h16"])
        assert not any(p.requires_grad for p in fit98.load_fit(94011, path, name).parameters())
        parents.append(arm["checkpoint"]["weights_sha256"])
    assert parents[0] != parents[1] and fit98.mapping_sha256(packet["network"]) == refs["parent_checkpoint"]["weights_sha256"]
    assert fit98.fit96._adam_hash(packet["optimizer"]) == refs["parent_checkpoint"]["adam_sha256"]
    with pytest.raises(FileExistsError):
        fit98.fit_seed(94011, path)


def test_failed_model_calls_are_counted_and_not_accepted(tmp_path, monkeypatch):
    packet, refs = toy_source()
    def fails_after_first(weights):
        world = Toy()
        world.load_state_dict(weights)
        original, calls = world.step, []
        def step(*args):
            calls.append(1)
            if len(calls) == 2:
                raise ValueError("injected fault")
            return original(*args)
        world.step = step
        return world.eval().requires_grad_(False)
    monkeypatch.setattr(fit98.fit96, "_fresh_world", fails_after_first)
    batch = fit98._batch(*physical_batch())
    row = fit98._fit_arm("endpoint", tmp_path/"branch", packet, refs, batch, 1e-7)
    assert row["status"] == "PARTIAL" and "injected fault" in row["error"]
    assert row["attempted_model_steps"] == 36 and row["completed_model_steps"] == 18
    assert row["updates_attempted"] == 1 and row["updates_completed"] == 0
    assert row["checkpoint"] is None and (tmp_path/"branch"/"partial_world.pt").is_file()
