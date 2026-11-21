"""Cold-start D1 actor for model-based policy updates on CPU."""
from __future__ import annotations

from operator import index

import numpy as np
import torch
from torch import nn


OBSERVATIONS = 643
ACTIONS = 16


class ColdActor(nn.Module):
    def __init__(self):
        super().__init__()
        options = dict(device="cpu", dtype=torch.float32)
        self.input_layer = nn.Linear(OBSERVATIONS, 64, **options)
        self.hidden_layer = nn.Linear(64, 64, **options)
        self.action_head = nn.Linear(64, ACTIONS, **options)
        for layer, gain in ((self.input_layer, 1.0), (self.hidden_layer, 1.0), (self.action_head, 0.05)):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != OBSERVATIONS:
            raise ValueError("D1 actor expects a batch of 643 observations")
        hidden = self.input_layer(observation).tanh()
        hidden = self.hidden_layer(hidden).tanh()
        return self.action_head(hidden).tanh()


def make_actor(seed: int) -> ColdActor:
    """Initialize every policy parameter within a private CPU RNG scope."""
    seed = index(seed)
    if seed < 0:
        raise ValueError("training seed must be nonnegative")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ColdActor()


@torch.no_grad()
def predict(actor: nn.Module, np_obs643: np.ndarray) -> np.ndarray:
    """Deterministic native-sampling interface. Training calls the actor directly."""
    observation = np.asarray(np_obs643, dtype=np.float32)
    if observation.shape != (OBSERVATIONS,) or not np.isfinite(observation).all():
        raise ValueError("expected one finite 643-component observation")
    action = actor(torch.from_numpy(observation.copy()).unsqueeze(0))
    if action.shape != (1, ACTIONS) or not torch.isfinite(action).all() or (action.abs() > 1).any():
        raise ValueError("cold actor must emit one bounded finite 16-component action")
    return action[0].numpy().copy()
