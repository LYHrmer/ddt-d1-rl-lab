"""The 87 perception comparison uses the same original actor, critic and Adam."""
import io
import json
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from gymnasium import spaces

from tools87 import ppo87 as pilot


def test_sensor_connection_and_critic_isolation():
    torch.manual_seed(7)
    network = pilot.RangeMlp()
    proprio = torch.randn(4, pilot.ACTOR)
    measured = torch.ones(4, pilot.RANGES)
    full = torch.cat((proprio, measured), dim=1)
    blind = torch.cat((proprio, torch.zeros_like(measured)), dim=1)
    assert torch.equal(network.forward_actor(full), network.forward_actor(blind))
    network.forward_actor(blind).sum().backward()
    assert torch.count_nonzero(network.range_projection.weight.grad) == 0
    network.zero_grad()
    network.forward_actor(full).sum().backward()
    assert torch.count_nonzero(network.range_projection.weight.grad) > 0
    with torch.no_grad():
        network.range_projection.weight.fill_(.01)
    assert not torch.equal(network.forward_actor(full), network.forward_actor(blind))
    assert torch.equal(network.forward_critic(full), network.forward_critic(blind))


def test_warm_start_preserves_individual_adam_slots(tmp_path, monkeypatch):
    torch.manual_seed(5)
    source_space = spaces.Dict({
        "actor": spaces.Box(-np.inf, np.inf, (627,), np.float32),
        "velocity_target": spaces.Box(-np.inf, np.inf, (3,), np.float32),
    })
    original = pilot.VelocityActorCriticPolicy(source_space, pilot.action_space(), lambda _: 1e-4)
    for i, parameter in enumerate(original.parameters()):
        parameter.grad = torch.full_like(parameter, (i + 1) / 100)
    original.optimizer.step()
    path = tmp_path / "policy.zip"
    with zipfile.ZipFile(path, "w") as storage:
        for name, state in (("policy.pth", original.state_dict()),
                            ("policy.optimizer.pth", original.optimizer.state_dict())):
            buf = io.BytesIO()
            torch.save(state, buf)
            storage.writestr(name, buf.getvalue())
    digest = pilot.sha(path)
    (tmp_path / "result.json").write_text(json.dumps(dict(
        status="COMPLETE", checkpoint_sha256=digest, cleanup_complete=True, eligible_for_evaluation=True,
        hyperparameters=dict(pilot.PPO_KWARGS, device="cpu", cpu_threads=1, aux_coef=0., log_std_init=-1.))))
    monkeypatch.setattr(pilot, "PARENT", path)
    monkeypatch.setattr(pilot, "COMPLETE25_SHA256", digest)
    policy = pilot.FullBodyPolicy(pilot.observation_space(), pilot.action_space(), lambda _: 1e-4,
                                  **pilot.POLICY_KWARGS)
    report = pilot.warm_start(policy)
    assert report["inherited_adam_parameters"] == len(list(original.parameters()))
    assert policy.mlp_extractor.range_projection.weight not in policy.optimizer.state
    for name, source in original.named_parameters():
        dest = dict(policy.named_parameters())[name]
        assert torch.equal(dest, source)
        for slot in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(policy.optimizer.state[dest][slot], original.optimizer.state[source][slot])


def test_deployment_mask_and_yaw_contract(tmp_path):
    x = np.zeros(pilot.ACTOR, np.float32)
    z = np.ones(pilot.RANGES, np.float32)
    assert np.array_equal(pilot.actor_input(x, z, "range")[-16:], z)
    assert not pilot.actor_input(x, z, "masked").any()
    x[pilot.YAW_COLUMNS[-1]] = .01
    with pytest.raises(ValueError, match="yaw"):
        pilot.actor_input(x, z, "range")
    (tmp_path / "report.json").write_text(json.dumps({"status": "PARTIAL", "arm": "range"}))
    with pytest.raises(ValueError, match="unqualified"):
        pilot.load_predictor(tmp_path)
