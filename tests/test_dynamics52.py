"""Synthetic boundaries and gradient tests; no simulator or fitting loop."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tools52 import dynamics52 as dynamics


def episodes(length=12):
    values = {key: [] for key in ("s", "a", "target", "episode", "tick", "done")}
    for episode in range(2):
        states = np.zeros((length + 1, 45), dtype=np.float32)
        states[:, 3] = 1
        states[:, 2] = 0.5
        states[:, 0] = np.arange(length + 1) * 0.01
        states[:, 23] = np.arange(length + 1) * 0.02 + episode * 0.1
        values["s"].extend(states[:-1])
        values["target"].extend(states[1:])
        values["a"].extend(np.zeros((length, 16), dtype=np.float32))
        values["episode"].extend([episode] * length)
        values["tick"].extend(range(length))
        values["done"].extend([False] * (length - 1) + [True])
    return {key: np.asarray(value) for key, value in values.items()}


def test_loss_backpropagates_through_early_predicted_states():
    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(1.2))

        def forward(self, state, action):
            velocity = self.gain * state[:, 23:24] + action[:, :1]
            return torch.cat((state[:, :23], velocity, state[:, 24:]), dim=-1)

    model = Recurrent()
    initial = torch.zeros(1, 45)
    initial[:, 3] = 1
    initial[:, 23] = 0.2
    actions = torch.zeros(1, 8, 16, requires_grad=True)
    with torch.no_grad():
        target = dynamics.rollout(model, initial, actions, return_trajectory=True)
        target[:, -1, 23] += 0.1
    dynamics.sequence_loss(model, initial, actions, target).backward()
    # Earlier errors are zero: this gradient can only come through the final step.
    assert actions.grad[0, 0, 0].abs() > 0
    assert model.gain.grad.abs() > 0


def test_windows_never_cross_episode_or_done():
    data = episodes()
    indices = dynamics.window_indices(data, 8)
    assert len(indices) == 10
    assert all(len(set(data["episode"][row])) == 1 for row in indices)
    assert not data["done"][indices[:, :-1]].any()
    data["done"][4] = True
    with pytest.raises(ValueError, match="after done"):
        dynamics.window_indices(data, 8)


def test_partial_episode_and_discontinuous_state_are_rejected():
    data = episodes()
    data["done"][-1] = False
    with pytest.raises(ValueError, match="incomplete episode"):
        dynamics.window_indices(data)
    data = episodes()
    data["s"][5, 23] += 1
    with pytest.raises(ValueError, match="discontinuous episode"):
        dynamics.window_indices(data)


def test_heldout_values_do_not_fit_statistics_or_ridge():
    data = episodes()
    training = data["episode"] == 0
    before = dynamics.fit_ridge({key: value[training] for key, value in data.items()})
    data["s"][~training, 23:] = 1e6
    data["target"][~training, 23:] = -1e6
    data["a"][~training] = 0.9
    after = dynamics.fit_ridge({key: value[training] for key, value in data.items()})
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["input_mean"][23] == pytest.approx(data["s"][training, 23].mean())
    assert np.all(before["input_mean"][list(dynamics.ZERO_FEATURES)] == 0)


def test_quaternion_projection_is_differentiable_and_does_not_mutate_input():
    model = dynamics.Dynamics52()
    with torch.no_grad():
        model.residual[-1].bias[4] = 2
    state = torch.zeros(1, 45)
    state[:, 3] = 1
    state.requires_grad_()
    before = state.detach().clone()
    predicted = model(state, torch.zeros(1, 16))
    torch.testing.assert_close(predicted[:, 3:7].norm(dim=-1), torch.ones(1))
    predicted[:, 4].sum().backward()
    assert state.grad[:, 3:7].abs().sum() > 0
    torch.testing.assert_close(state.detach(), before)
