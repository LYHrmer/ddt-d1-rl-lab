"""Frozen C/A fusion preserving conditional body-xy in real arithmetic.

Both models receive the same current state/action on every call. Only the
three world-velocity coordinates are fused. This is a single-step identity;
neither multi-step trajectories nor floating-point gates are guaranteed.
"""
from __future__ import annotations

import torch
from torch import nn


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenPair78(nn.Module):
    """Keep complete frozen predictors; retain all input derivative paths."""

    def __init__(self, control, donor):
        super().__init__()
        _require(isinstance(control, nn.Module) and isinstance(donor, nn.Module),
                 'control and donor must be complete torch modules')
        self.control = control
        self.donor = donor
        self.eval().requires_grad_(False)

    @property
    def input_mean(self):
        """Expose the frozen evaluator's dtype reference without a new buffer."""
        return self.control.input_mean

    def forward(self, states, actions):
        _require(isinstance(states, torch.Tensor) and isinstance(actions, torch.Tensor),
                 'state/action tensors required')
        _require(states.ndim == 2 and states.shape[0] > 0 and states.shape[1] == 45
                 and actions.shape == (len(states), 16), 'state[B,45] and action[B,16] required')
        _require(states.device.type == actions.device.type == 'cpu'
                 and states.dtype in (torch.float32, torch.float64) and actions.dtype == states.dtype,
                 'matching CPU float32/float64 state/action required')
        _require(bool(torch.isfinite(states).all() and torch.isfinite(actions).all()),
                 'finite state/action required')
        _require(not self.training and not self.control.training and not self.donor.training,
                 'frozen pair must remain in eval mode')
        _require(all(not p.requires_grad for p in self.parameters()), 'model parameters must remain frozen')
        _require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == states.dtype)
                     for t in (*self.parameters(), *self.buffers())), 'model/input CPU dtype mismatch')
        control, donor = self.control(states, actions), self.donor(states, actions)
        _require(all(isinstance(value, torch.Tensor) and value.shape == states.shape
                     and value.device.type == 'cpu' and value.dtype == states.dtype
                     and bool(torch.isfinite(value).all()) for value in (control, donor)),
                 'both models must return finite matching CPU state[B,45]')
        quaternion = control[:, 3:7]
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        _require(bool(torch.isfinite(norm).all() and (norm > 0).all()), 'control quaternion norm must be finite and positive')
        quaternion = quaternion / norm
        vertical = states.new_tensor([0., 0., 1.]).expand_as(quaternion[:, 1:])
        cross = torch.linalg.cross(quaternion[:, 1:], vertical, dim=-1)
        column = vertical + 2 * quaternion[:, :1] * cross + 2 * torch.linalg.cross(
            quaternion[:, 1:], cross, dim=-1)
        denominator = column[:, 2:3]
        _require(bool(torch.isfinite(column).all() and (denominator.abs() >= .5).all()),
                 'abs(Rzz) must be at least 0.5 for every row')
        correction = (donor[:, 25:26] - control[:, 25:26]) / denominator
        horizontal = control[:, 23:25] + column[:, :2] * correction
        result = torch.cat((control[:, :23], horizontal, donor[:, 25:26], control[:, 26:]), dim=-1)
        _require(bool(torch.isfinite(result).all()), 'nonfinite fused state')
        return result

    def predict(self, states, actions):
        return self(states, actions)
