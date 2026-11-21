"""Algebra and archive semantics using a scalar analytic fixture, without D1 rollouts."""
import numpy as np
import pytest
import torch

from tools95 import model95 as direction


def _reset():
    start = {name: np.zeros((1, *shape)) for name, shape in direction.START_SHAPES.items()}
    start["state45"][0, 0] = 1.10
    start["ray_due"] = np.array([True])
    return start


def _scalar():
    net = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(1.)
    return net.eval().requires_grad_(False)


def _quadratic(_world, actor, starts, horizon, offset, meter):
    """Each noninitial x-offset equals (w+1)^2, so R'(1) == -4."""
    assert horizon == 32 and offset == 0. and all(k == 0 for _, k in starts)
    origin = torch.tensor([t["state45"][0, 0] for t, _ in starts], dtype=torch.float64)
    w = actor.weight.reshape(())
    offsets = torch.cat([w.new_zeros(1), (w + 1.).square().expand(horizon)])
    x = origin[:, None] + offsets[None, :]
    states = torch.cat((x[..., None], x.new_zeros(len(starts), horizon + 1, 44)), dim=-1)
    meter["imagined_steps_attempted"] += len(starts) * horizon
    meter["imagined_steps"] += len(starts) * horizon
    return states, origin


def test_analytic_derivative_uses_same_origin_forward_and_closed_loop_variants(monkeypatch):
    monkeypatch.setattr(direction.learn94, "_roll", _quadratic)
    parent = _scalar()
    bundle = direction.SealedDirection(0, parent, None, {"weight": torch.tensor([[-.1]])})
    arms = {alpha: direction.predict_arm(bundle, _reset(), alpha) for alpha in direction.ALPHAS}
    assert all(arm["safe"] and arm["ticks"] == 32 and arm["completed_model_steps"] == 32
               and len(arm["state45"]) == 33 for arm in arms.values())
    assert sum(arm["backward_completed"] for arm in arms.values()) == 1
    assert arms[0.]["directional_derivative"] == pytest.approx(.4)
    for h in (.5, 1.):
        fd = (arms[h]["return_m"] - arms[-h]["return_m"]) / (2*h)
        assert fd == pytest.approx(.4, abs=1e-6)
        assert arms[h]["return_m"] > arms[0.]["return_m"]
    assert parent.weight.item() == 1. and not parent.weight.requires_grad


def test_predicted_fall_preserves_short_prefix_without_full_horizon_result(monkeypatch):
    def stop(_world, _actor, _starts, _horizon, _offset, meter):
        meter.update(imagined_steps=3, imagined_steps_attempted=3)
        raise direction.learn94.LearnFailure("predicted fall", dict(meter), np.zeros((1, 4, 45)))

    monkeypatch.setattr(direction.learn94, "_roll", stop)
    bundle = direction.SealedDirection(0, _scalar(), None, {"weight": torch.tensor([[.1]])})
    result = direction.predict_arm(bundle, _reset(), 0.)
    assert not result["safe"] and "predicted fall" in result["error"]
    assert result["ticks"] == 3 and len(result["state45"]) == 4 and result["attempted_model_steps"] == 3
    assert result["return_m"] is None and result["directional_derivative"] is None
    assert result["backward_attempted"] == 0


def test_shadow_adam_seals_only_delta_and_rejects_reuse_or_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(direction, "ROOT", tmp_path)
    actor, world = _scalar(), _scalar()
    seed, sources = 0, {"starts": [],
                        "initial": {"weights_sha256": direction.mapping_sha256(actor.state_dict())},
                        "world": {"weights_sha256": direction.mapping_sha256(world.state_dict())}}
    for reset in range(3):
        name = f"start_{reset}.npz"
        np.savez(tmp_path / name, reset_seed=reset, **_reset())
        sources["starts"].append({"path": name, "reset": reset})
    monkeypatch.setattr(direction, "_sources", lambda _seed: sources)
    monkeypatch.setattr(direction, "_load_parents", lambda _seed, _source: (actor, world))
    monkeypatch.setattr(direction.learn94, "_roll", _quadratic)
    monkeypatch.setattr(direction.learn94, "_parameter_norms", lambda v: {"total": direction._norm(v)})
    dest = tmp_path / "direction"
    receipt = direction.construct(seed, dest)
    assert receipt["status"] == "COMPLETE" and receipt["completed_model_steps"] == 96
    assert receipt["shadow_adam_completed"] == receipt["backward_completed"] == 1
    assert receipt["policy_updates"] == receipt["world_updates"] == 0
    assert receipt["preclip_norm"] == pytest.approx(4.) and receipt["development_directional_derivative"] > 0.
    bundle = direction.load_sealed(seed, dest)
    assert bundle.delta["weight"].item() == pytest.approx(-direction.learn94.LR, abs=1e-7)
    assert direction.variant(bundle, 1.).weight.item() < actor.weight.item() == 1.
    with pytest.raises(FileExistsError):
        direction.construct(seed, dest)
    with (dest / "delta.pt").open("ab") as stream:
        stream.write(b"change")
    with pytest.raises(ValueError, match="direction file changed"):
        direction.load_sealed(seed, dest)
