"""Random tensors and hand-filled rollouts only; never env construction or learn."""
from __future__ import annotations

import numpy as np
import pytest

th = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.logger import configure

from d1_rl import velocity_policy15 as candidate


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = th.get_num_threads()
    th.set_num_threads(1)
    th.manual_seed(1501)
    yield
    th.set_num_threads(old)


def observation_space():
    return spaces.Dict({"actor": spaces.Box(-np.inf, np.inf, (627,), np.float32),
                        "velocity_target": spaces.Box(-np.inf, np.inf, (3,), np.float32)})


def action_space():
    return spaces.Box(-1., 1., (16,), np.float32)


def policy():
    return candidate.VelocityActorCriticPolicy(observation_space(), action_space(), lambda _: 1e-3)


def observations(n=8):
    return {"actor": th.randn(n, 627)*.1, "velocity_target": th.randn(n, 3)*.2}


def algorithm(aux_coef=1., **kwargs):
    result = candidate.AuxiliaryVelocityPPO(env=None, aux_coef=aux_coef, n_steps=8,
        batch_size=8, n_epochs=1, learning_rate=1e-3, seed=1501, _init_setup_model=False, **kwargs)
    result.observation_space, result.action_space, result.n_envs = observation_space(), action_space(), 1
    result._setup_model()
    result.set_logger(configure(folder=None, format_strings=[]))
    return result


def fill_rollout(model):
    sample = observations(model.n_steps)
    model.rollout_buffer.reset()
    for index in range(model.n_steps):
        obs = {key: value[index:index+1] for key, value in sample.items()}
        with th.no_grad():
            actions, values, log_prob = model.policy(obs)
        model.rollout_buffer.add({key: value.numpy() for key, value in obs.items()},
            actions.numpy(), np.array([.1+index*.01], np.float32), np.array([index == 0]), values, log_prob)
    model.rollout_buffer.compute_returns_and_advantage(last_values=th.zeros(1), dones=np.array([True]))
    return sample


def test_target_does_not_enter_actor_value_or_logprob_graph():
    net, obs = policy(), observations()
    original = {key: value.clone().requires_grad_() for key, value in obs.items()}
    changed = {"actor": original["actor"], "velocity_target": th.ones_like(obs["velocity_target"])*9999}
    a, v, p = net(original, deterministic=True)
    b, w, q = net(changed, deterministic=True)
    assert th.equal(a, b) and th.equal(v, w) and th.equal(p, q)
    assert th.equal(net.predict_values(original), net.predict_values(changed))
    assert th.equal(net.get_distribution(original).distribution.mean,
                    net.get_distribution(changed).distribution.mean)
    (a.sum()+v.sum()).backward()
    assert original["velocity_target"].grad is None
    assert original["actor"].grad is not None
    # Target corruption is detected in the auxiliary path; actor is independent.
    changed["velocity_target"].fill_(float("nan"))
    assert th.equal(net(changed, deterministic=True)[0], a)
    with pytest.raises(ValueError, match="velocity_target"):
        net.auxiliary_velocity_loss(changed)


def test_actor_encoder_retains_ppo_gradient_even_without_auxiliary_loss():
    net, obs = policy(), observations()
    net.zero_grad()
    net(obs, deterministic=True)[0].square().sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in net.mlp_extractor.encoder.parameters()) > 0


def test_auxiliary_mse_has_encoder_gradients_and_reduces_fixed_batch_loss():
    net, obs = policy(), observations()
    obs["velocity_target"].requires_grad_()
    before = net.auxiliary_velocity_loss(obs).item()
    with th.no_grad():
        prediction = net.estimate_velocity(obs["actor"])
        expected = ((prediction-obs["velocity_target"])**2).mean()
    assert before == pytest.approx(expected.item())
    for _ in range(12):
        net.optimizer.zero_grad()
        loss = net.auxiliary_velocity_loss(obs)
        loss.backward()
        assert net.mlp_extractor.encoder[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in net.mlp_extractor.value_net.parameters())
        assert all(p.grad is None for p in net.mlp_extractor.policy_net.parameters())
        assert obs["velocity_target"].grad is None
        net.optimizer.step()
    assert net.auxiliary_velocity_loss(obs).item() < before*.5


@pytest.mark.parametrize("aux_coef", [0., .7])
def test_same_optimizer_updates_hand_filled_aligned_dict_buffer(aux_coef):
    model = algorithm(aux_coef)
    fill_rollout(model)
    assert type(model.rollout_buffer) is DictRolloutBuffer
    before = {name: value.detach().clone() for name, value in model.policy.named_parameters()}
    model.train()
    assert model._n_updates == 1 and model.logger.name_to_value["train/aux_coef"] == aux_coef
    assert np.isfinite(model.logger.name_to_value["train/auxiliary_velocity_mse"])
    assert any(not th.equal(before[name], value) for name, value in model.policy.named_parameters()
               if name.startswith("mlp_extractor.encoder"))
    assert set(model.policy.optimizer.state) <= set(model.policy.parameters())


def test_zero_aux_update_matches_unmodified_sb3_ppo_train():
    local = algorithm(0., clip_range_vf=.2, ent_coef=.01)
    reference = algorithm(0., clip_range_vf=.2, ent_coef=.01)
    reference.policy.load_state_dict(local.policy.state_dict())
    fill_rollout(local)
    # Identical source rollout; comparison differs only in train implementation.
    reference.rollout_buffer = local.rollout_buffer
    np.random.seed(1502)
    local.train()
    np.random.seed(1502)
    PPO.train(reference)
    for name, value in local.policy.state_dict().items():
        assert th.equal(value, reference.policy.state_dict()[name]), name
    for key in ("train/loss", "train/entropy_loss", "train/value_loss", "train/approx_kl"):
        assert local.logger.name_to_value[key] == reference.logger.name_to_value[key]


def test_auxiliary_term_is_really_connected_to_total_optimizer_loss():
    control, treatment = algorithm(0., vf_coef=0., ent_coef=0.), algorithm(1., vf_coef=0., ent_coef=0.)
    treatment.policy.load_state_dict(control.policy.state_dict())
    sample = fill_rollout(control)
    control.rollout_buffer.advantages[:] = 0.
    control.rollout_buffer.returns[:] = control.rollout_buffer.values
    treatment.rollout_buffer = control.rollout_buffer
    control.lr_schedule = treatment.lr_schedule = lambda _: 1e-4
    before = {name: value.detach().clone() for name, value in control.policy.named_parameters()}
    mse_before = treatment.policy.auxiliary_velocity_loss(sample).item()
    np.random.seed(1515)
    control.train()
    np.random.seed(1515)
    treatment.train()
    # With policy, value and entropy gradients suppressed, only the connected
    # auxiliary term can update weights. This kills a logged-but-unused MSE.
    assert all(th.equal(before[name], value) for name, value in control.policy.named_parameters())
    changed_encoder = []
    for name, value in treatment.policy.named_parameters():
        if name.startswith("mlp_extractor.encoder"):
            changed_encoder.append(not th.equal(before[name], value))
        else:
            assert th.equal(before[name], value), name
    assert any(changed_encoder), "auxiliary MSE was not connected to the total loss"
    assert treatment.policy.auxiliary_velocity_loss(sample).item() < mse_before


def test_kl_stop_prevents_both_ppo_and_auxiliary_update():
    model = algorithm(1., target_kl=1e-6)
    fill_rollout(model)
    model.rollout_buffer.log_probs[:] += 10
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model.train()
    assert model._n_updates == 1
    assert all(th.equal(before[name], value) for name, value in model.policy.state_dict().items())
    assert not model.policy.optimizer.state


def test_save_load_preserves_aux_coef_optimizer_and_deterministic_predictions(tmp_path):
    model = algorithm(.37)
    obs = fill_rollout(model)
    model.train()
    array_obs = {key: value.numpy() for key, value in obs.items()}
    prediction = model.predict(array_obs, deterministic=True)[0]
    path = tmp_path/"candidate.zip"
    model.save(path)
    restored = candidate.AuxiliaryVelocityPPO.load(path, device="cpu")
    assert restored.aux_coef == .37 and restored._n_updates == 1
    assert np.array_equal(prediction, restored.predict(array_obs, deterministic=True)[0])
    assert len(restored.policy.optimizer.state) == len(model.policy.optimizer.state) > 0
    assert th.equal(model.policy.predict_values(obs), restored.policy.predict_values(obs))


@pytest.mark.parametrize("fault", ["shape", "nan", "inf"])
def test_bad_actor_rejected(fault):
    net, obs = policy(), observations()
    if fault == "shape":
        obs["actor"] = obs["actor"][:, :-1]
    else:
        obs["actor"][0, 0] = float(fault)
    with pytest.raises(ValueError, match="actor"):
        net(obs)


@pytest.mark.parametrize("fault", ["shape", "nan", "dtype"])
def test_bad_target_rejected_by_auxiliary_loss(fault):
    net, obs = policy(), observations()
    if fault == "shape": obs["velocity_target"] = obs["velocity_target"][:, :2]
    if fault == "nan": obs["velocity_target"][0, 0] = float("nan")
    if fault == "dtype": obs["velocity_target"] = obs["velocity_target"].double()
    with pytest.raises(ValueError, match="velocity_target"):
        net.auxiliary_velocity_loss(obs)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True])
def test_invalid_aux_coefficient_is_rejected(value):
    with pytest.raises(ValueError, match="aux_coef"):
        candidate.AuxiliaryVelocityPPO(aux_coef=value, _init_setup_model=False)


def test_fixed_candidate_rejects_different_policy_space_or_algorithm_contract(monkeypatch):
    with pytest.raises(ValueError, match="override"):
        candidate.VelocityActorCriticPolicy(observation_space(), action_space(), lambda _: .001, normalize_images=True)
    with pytest.raises(ValueError, match="actor space"):
        bad = observation_space(); bad.spaces["actor"] = spaces.Box(-1., 1., (626,), np.float32)
        candidate.VelocityActorCriticPolicy(bad, action_space(), lambda _: .001)
    with pytest.raises(ValueError, match="actions"):
        candidate.VelocityActorCriticPolicy(observation_space(), spaces.Discrete(16), lambda _: .001)
    with pytest.raises(ValueError, match="gSDE"):
        candidate.AuxiliaryVelocityPPO(use_sde=True, _init_setup_model=False)
    monkeypatch.setattr(candidate.sb3, "__version__", "2.8.0")
    with pytest.raises(ValueError, match="2.7.0"):
        policy()
