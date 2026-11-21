"""Differentiable late signed-displacement supervision on the round-98 reset batch."""
from __future__ import annotations

import math

import torch

from tools94 import learn94
from tools98 import fit98


def _tensor(name, value, shape, dtype=None):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype not in (torch.float32, torch.float64) or (dtype is not None and value.dtype != dtype) or
            value.shape != shape or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite CPU {dtype or 'floating'} tensor of shape {shape}")


def paired_response(x, origin):
    """Return [3,T] positive-minus-negative signed departure from each arm's reset, in metres."""
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("expected a [6,T] floating position tensor with at least one endpoint")
    _tensor("positions", x, (6, x.shape[1]))
    _tensor("reset coordinates", origin, (6,))
    departure = x.double() - origin.double().unsqueeze(1)
    return departure[3:] - departure[:3]


def late_response_loss(predicted_response, physical_response):
    """Mean squared signed-response error divided by the fixed training-label mean square."""
    _tensor("predicted late response", predicted_response, (3, 16))
    _tensor("native late response", physical_response, (3, 16))
    target = physical_response.detach().double()
    denominator = target.square().mean().clamp_min(fit98.fit96.FLOOR_M2)
    loss = (predicted_response.double() - target).square().mean() / denominator
    if not bool(denominator.isfinite()) or not bool(loss.isfinite()):
        raise ValueError("nonfinite late displacement loss or denominator")
    return loss


def _advance(world, state, applied, terrain, meter):
    prediction = learn94._step(world, state, applied, terrain, meter)
    _tensor("predicted state", prediction, state.shape, state.dtype)
    return prediction


def training_terms(world, prefix_batch, late_actions, late_targets, early_denom, *, include_late, meter):
    """Roll all 18 training prefixes to H16, then the same six paired trajectories to H32."""
    if type(include_late) not in (bool, int, float) or include_late not in (0, 1):
        raise ValueError("late response weight must be zero or one")
    if isinstance(early_denom, torch.Tensor):
        _tensor("frozen round-96 denominator", early_denom, ())
        early_denom = early_denom.detach().item()
    if not isinstance(early_denom, (float, int)) or not math.isfinite(early_denom) or early_denom <= 0.:
        raise ValueError("expected a positive finite round-96 denominator")
    _tensor("reset states", prefix_batch["start"], (18, 45), torch.float32)
    _tensor("prefix applied actions", prefix_batch["applied"], (18, 16, 16), torch.float32)
    _tensor("prefix state targets", prefix_batch["target"], (18, 16, 45), torch.float32)
    _tensor("early physical response", prefix_batch["paired_x_m"], (3, 8), torch.float64)
    _tensor("late applied actions", late_actions, (6, 16, 16), torch.float32)
    _tensor("late physical response", late_targets, (3, 16), torch.float64)
    if not bool((prefix_batch["applied"].abs() <= 1.).all() and (late_actions.abs() <= 1.).all()):
        raise ValueError("applied training actions exceed the native clip")

    state = prefix_batch["start"]
    terrain = torch.ones(18, dtype=torch.long, device=state.device)
    prefix = []
    for applied in prefix_batch["applied"].unbind(1):
        state = _advance(world, state, applied, terrain, meter)
        prefix.append(state)
    labels = {"start": prefix_batch["start"], "target": prefix_batch["target"].detach(),
              "paired_x_m": prefix_batch["paired_x_m"].detach()}
    state_h8, _, response_h8, state_h16 = fit98._losses(torch.stack(prefix, dim=1), labels, early_denom)

    # The six H16 endpoints remain on the prefix graph. The physical labels never seed a model step.
    state = state[12:18]
    late_x = []
    for applied in late_actions.unbind(1):
        state = _advance(world, state, applied, terrain[12:18], meter)
        late_x.append(state[:, 0])
    response = paired_response(torch.stack(late_x, dim=1), prefix_batch["start"][12:18, 0])
    response_h17_32 = late_response_loss(response, late_targets)
    total = state_h8 + response_h8 + state_h16 + float(include_late) * response_h17_32
    if not bool(total.isfinite()):
        raise ValueError("nonfinite combined training loss")
    return {"total": total, "state_h8": state_h8, "response_h8": response_h8, "state_h16": state_h16,
            "response_h17_32": response_h17_32}
