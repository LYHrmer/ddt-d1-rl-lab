"""CPU-only SB3 2.7.0 velocity-estimator candidate, with no robot imports.

Dict observations carry ``actor`` (627 float32 values: current 57 then ten
previous frames, oldest first) and ``velocity_target`` (3 float32 body-frame
linear velocities in m/s, sampled at that same observation boundary). Only the
auxiliary MSE reads the target. Neither the actor nor critic receives truth.

The encoder intentionally retains the PPO actor gradient, unlike the upstream
detached-latent design: aux_coef=0 is a usable same-architecture control. There
is no feature/target normalization, recurrent state, privileged critic, or gSDE.
This module only defines networks/updates; it never constructs an environment.
The stable import identity is d1_rl.velocity_policy15 for checkpoint portability.
"""
from __future__ import annotations

import math

import numpy as np
import stable_baselines3 as sb3
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

ACTOR_DIM = 627
FRAME_DIM = 57
VELOCITY_DIM = 3
ACTION_DIM = 16
SB3_VERSION = "2.7.0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_spaces(observation_space, action_space=None):
    require(isinstance(observation_space, spaces.Dict)
            and set(observation_space.spaces) == {"actor", "velocity_target"},
            "expected Dict observation with actor and velocity_target only")
    for name, size in (("actor", ACTOR_DIM), ("velocity_target", VELOCITY_DIM)):
        space = observation_space[name]
        require(isinstance(space, spaces.Box) and space.shape == (size,)
                and space.dtype == np.dtype(np.float32), f"{name} space must be float32 ({size},)")
    if action_space is not None:
        require(isinstance(action_space, spaces.Box) and action_space.shape == (ACTION_DIM,)
                and action_space.dtype == np.dtype(np.float32)
                and np.all(action_space.low == -1) and np.all(action_space.high == 1),
                "only float32 Box[-1,1] 16D actions are supported")


def check_actor(actor):
    require(isinstance(actor, th.Tensor) and actor.ndim == 2 and actor.shape[1] == ACTOR_DIM
            and actor.shape[0] > 0 and actor.dtype == th.float32 and actor.device.type == "cpu"
            and bool(th.isfinite(actor).all()), "actor must be finite CPU float32 [batch,627]")
    return actor


class ActorOnlyExtractor(BaseFeaturesExtractor):
    """Select actor only: target values are never touched on inference paths."""

    def __init__(self, observation_space):
        check_spaces(observation_space)
        super().__init__(observation_space, features_dim=ACTOR_DIM)

    def forward(self, observations):
        require(isinstance(observations, dict) and "actor" in observations, "missing actor observation")
        return check_actor(observations["actor"])


class VelocityMlpExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(ACTOR_DIM, 128), nn.Tanh(), nn.Linear(128, 64),
                                     nn.Tanh(), nn.Linear(64, VELOCITY_DIM))
        self.policy_net = nn.Sequential(nn.Linear(FRAME_DIM+VELOCITY_DIM, 128), nn.Tanh(),
                                        nn.Linear(128, 128), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(ACTOR_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.latent_dim_pi = self.latent_dim_vf = 128

    def forward_actor(self, features):
        # No detach: PPO can train the encoder in both treatment and aux=0 arm.
        velocity = self.encoder(features)
        return self.policy_net(th.cat((features[:, :FRAME_DIM], velocity), dim=1))

    def forward_critic(self, features):
        return self.value_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class VelocityActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        require(sb3.__version__ == SB3_VERSION, "velocity candidate requires stable-baselines3==2.7.0")
        check_spaces(observation_space, action_space)
        fixed = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}, "activation_fn": nn.Tanh,
                 "features_extractor_class": ActorOnlyExtractor, "share_features_extractor": True,
                 "normalize_images": False, "use_sde": False}
        for key, value in fixed.items():
            require(key not in kwargs or kwargs[key] == value, f"unsupported policy override: {key}")
            kwargs[key] = value
        require(not kwargs.get("squash_output") and not kwargs.get("features_extractor_kwargs"),
                "squashing and custom feature kwargs are not supported")
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = VelocityMlpExtractor().to(self.device)

    def estimate_velocity(self, actor):
        return self.mlp_extractor.encoder(check_actor(actor))

    def auxiliary_velocity_loss(self, observations):
        actor = check_actor(observations["actor"])
        target = observations["velocity_target"]
        require(isinstance(target, th.Tensor) and target.shape == (actor.shape[0], VELOCITY_DIM)
                and target.dtype == th.float32 and target.device == actor.device
                and bool(th.isfinite(target).all()), "velocity_target must be finite CPU float32 [batch,3]")
        # Mean across both batch and all three velocity components, in (m/s)^2.
        return F.mse_loss(self.estimate_velocity(actor), target.detach())


class AuxiliaryVelocityPPO(PPO):
    """SB3 2.7.0 PPO plus one coefficient-weighted MSE in the same update.

    GAE and observation/target alignment use the unmodified DictRolloutBuffer.
    aux_coef is saved by SB3 together with algorithm attributes, not a callback.
    ``env=None, _init_setup_model=False`` supports loading/hand-filled tests.
    """

    def __init__(self, policy=VelocityActorCriticPolicy, env=None, *, aux_coef=1.0, device="cpu", **kwargs):
        require(sb3.__version__ == SB3_VERSION, "velocity candidate requires stable-baselines3==2.7.0")
        require(policy is VelocityActorCriticPolicy, "only VelocityActorCriticPolicy is supported")
        require(th.device(device).type == "cpu", "velocity candidate is CPU only")
        require(not kwargs.get("use_sde", False), "gSDE is outside the candidate contract")
        require(kwargs.get("rollout_buffer_class") in (None, DictRolloutBuffer), "DictRolloutBuffer is required")
        self.aux_coef = self.checked_aux_coef(aux_coef)
        super().__init__(policy, env, device=device, **kwargs)

    @staticmethod
    def checked_aux_coef(value):
        require(not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0, "aux_coef must be finite and nonnegative")
        return float(value)

    def _setup_model(self):
        require(self.device.type == "cpu", "velocity candidate is CPU only")
        self.aux_coef = self.checked_aux_coef(self.aux_coef)
        check_spaces(self.observation_space, self.action_space)
        super()._setup_model()
        require(isinstance(self.policy, VelocityActorCriticPolicy)
                and type(self.rollout_buffer) is DictRolloutBuffer, "unexpected policy or rollout buffer")

    def train(self):
        # Adapted from stable-baselines3 2.7.0 PPO.train (MIT license,
        # https://github.com/DLR-RM/stable-baselines3/blob/v2.7.0/stable_baselines3/ppo/ppo.py).
        # Copyright (c) 2019 Antonin Raffin; full permission notice is retained
        # in licenses/stable-baselines3-MIT.txt and THIRD_PARTY_NOTICES.md.
        # Continuous Box only: the upstream discrete-action cast is inapplicable.
        # The only optimization change is aux_coef * auxiliary MSE in total loss.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, auxiliary_losses, clip_fractions = [], [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
                ratio = th.exp(log_prob-rollout_data.old_log_prob)
                policy_loss_1 = advantages*ratio
                policy_loss_2 = advantages*th.clamp(ratio, 1-clip_range, 1+clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio-1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values+th.clamp(
                        values-rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                auxiliary_loss = self.policy.auxiliary_velocity_loss(rollout_data.observations)
                auxiliary_losses.append(auxiliary_loss.item())
                loss = (policy_loss+self.ent_coef*entropy_loss+self.vf_coef*value_loss
                        + self.aux_coef*auxiliary_loss)
                require(bool(th.isfinite(loss)), "nonfinite PPO or auxiliary loss")
                with th.no_grad():
                    log_ratio = log_prob-rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio)-1)-log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5*self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        for name, values in (("entropy_loss", entropy_losses), ("policy_gradient_loss", pg_losses),
                             ("value_loss", value_losses), ("auxiliary_velocity_mse", auxiliary_losses),
                             ("approx_kl", approx_kl_divs), ("clip_fraction", clip_fractions)):
            self.logger.record("train/"+name, np.mean(values))
        self.logger.record("train/aux_coef", self.aux_coef)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
