"""Synthetic signed-response and gradient checks for the two round-99 objectives."""
import pytest

torch = pytest.importorskip("torch")

from tools52.dynamics52 import group_loss
from tools99.loss99 import late_response_loss, paired_response, training_terms


def signed_targets():
    return (torch.arange(48, dtype=torch.float64).reshape(3, 16) - 23.5) / 1024


@pytest.mark.parametrize(("gain", "expected"), [(1., 0.), (0., 1.), (-1., 4.)])
def test_normalized_loss_preserves_sign_and_magnitude(gain, expected):
    target = signed_targets()
    assert late_response_loss(gain * target, target).item() == pytest.approx(expected)


def test_zero_label_scale_has_a_finite_floor_and_only_predictions_receive_gradients():
    target = torch.zeros(3, 16, dtype=torch.float64, requires_grad=True)
    prediction = torch.full_like(target, 1e-6, requires_grad=True)
    loss = late_response_loss(prediction, target)
    assert loss.item() == pytest.approx(1e-12 / 1e-10)
    prediction_grad, target_grad = torch.autograd.grad(loss, (prediction, target), allow_unused=True)
    assert target_grad is None
    torch.testing.assert_close(prediction_grad, torch.full_like(target, 2e-6 / (48 * 1e-10)))


def test_each_arm_uses_its_own_reset_even_for_sub_float32_displacements():
    origin = torch.tensor([4, -2, 3, -.5, 1, 2], dtype=torch.float64)
    shift = torch.tensor([8, 4, -2, 16, -8, 1], dtype=torch.float64)
    displacement = torch.arange(96, dtype=torch.float64).reshape(6, 16) * 2.**-36
    x = origin[:, None] + displacement
    expected = displacement[3:] - displacement[:3]
    torch.testing.assert_close(paired_response(x, origin), expected, rtol=0, atol=0)
    torch.testing.assert_close(paired_response(x + shift[:, None], origin + shift),
                               expected, rtol=0, atol=0)


def test_both_predictions_carry_opposite_gradients_and_match_a_scalar_difference():
    target = signed_targets()
    origin = torch.tensor([0, .25, .5, 1, 2, 4], dtype=torch.float64)

    def positions(theta):
        return origin[:, None] + torch.cat((-theta * target / 2, theta * target / 2))

    def objective(x):
        return late_response_loss(paired_response(x, origin), target)

    theta = torch.tensor(.25, dtype=torch.float64, requires_grad=True)
    x = positions(theta)
    loss = objective(x)
    scalar_grad, arm_grad = torch.autograd.grad(loss, (theta, x))
    assert loss.item() == pytest.approx(.5625)
    assert scalar_grad.item() == pytest.approx(-1.5)
    h = 2.**-12
    numerical = (objective(positions(.25 + h)) - objective(positions(.25 - h))) / (2*h)
    assert numerical.item() == pytest.approx(scalar_grad.item(), abs=1e-10)
    expected = 2 * (.25 - 1.) * target / (target.numel() * target.square().mean())
    torch.testing.assert_close(arm_grad[3:], expected)
    torch.testing.assert_close(arm_grad[:3], -expected)


class ToyWheel:
    """The gain enters only through action 0. The suffix uses action 1 exclusively."""

    def __init__(self):
        self.prefix_gain = torch.tensor(.25, requires_grad=True)
        self.inputs = []

    def step(self, state, action, terrain):
        assert terrain.tolist() == [1] * len(state)
        self.inputs.append(state.detach().clone())
        x = state[:, :1] + self.prefix_gain * action[:, :1] + action[:, 1:2]
        return torch.cat((x, state[:, 1:]), dim=1)


def synthetic_batch():
    start = torch.zeros(18, 45)
    start[:, 0] = torch.arange(18) / 8
    start[:, 3] = 1.
    start[15:] = start[12:15]
    applied = torch.zeros(18, 16, 16)
    applied[:12, 1:, 0] = torch.arange(1, 13)[:, None] / 64
    applied[12:, 1:, 0] = torch.tensor([-1, -2, -3, 2, 3, 4])[:, None] / 32
    cumulative = applied[:, :, 0].cumsum(1)
    truth = start[:, None, :].expand(-1, 16, -1).clone()
    truth[:, :, 0] += .75 * cumulative
    # A distinct observed endpoint makes any replacement of the predicted boundary visible.
    truth[:, -1, 0] += 2. + torch.arange(18) / 32
    early_delta = truth[:, :8, 0].double() - start[:, :1].double()
    prefix = {"start": start, "target": truth, "applied": applied.requires_grad_(),
              "paired_x_m": early_delta[15:] - early_delta[12:15]}
    late_actions = torch.zeros(6, 16, 16)
    late_actions[:, :, 1] = torch.tensor([-2, -4, -6, 3, 6, 9])[:, None] * torch.arange(1, 17) / 4096
    suffix_x = late_actions[:, :, 1].double().cumsum(1)
    pair_excitation = (cumulative[15:, -1] - cumulative[12:15, -1]).double()[:, None]
    physical = .75 * pair_excitation + suffix_x[3:] - suffix_x[:3]
    predicted_prefix = start[:, None, :].expand(-1, 16, -1).clone()
    predicted_prefix[:, :, 0] += .25 * cumulative
    return prefix, late_actions, physical, predicted_prefix, pair_excitation


@pytest.mark.parametrize("include_late", [False, True])
def test_equal_forward_budget_and_one_added_objective(include_late):
    batch, actions, target, predicted, excitation = synthetic_batch()
    world = ToyWheel()
    meter = {"imagined_steps": 0, "imagined_steps_attempted": 0}
    denom = .125
    terms = training_terms(world, batch, actions, target, denom, include_late=include_late, meter=meter)

    assert len(world.inputs) == 32
    assert [len(x) for x in world.inputs] == [18] * 16 + [6] * 16
    assert meter["imagined_steps"] == meter["imagined_steps_attempted"] == 384
    # Before transitions 17..32, each selected row must carry its own free-running state.
    suffix_increment = actions[:, :, 1].cumsum(1) - actions[:, :, 1]
    expected_suffix_inputs = predicted[12:, -1, 0, None] + suffix_increment
    observed_suffix_inputs = torch.stack([s[:, 0] for s in world.inputs[16:]], dim=1)
    torch.testing.assert_close(observed_suffix_inputs, expected_suffix_inputs, rtol=0, atol=0)

    early_displacement = predicted[:, :8, 0].double() - batch["start"][:, :1].double()
    early_response = early_displacement[15:] - early_displacement[12:15]
    state_h8 = group_loss(predicted[:, :8], batch["target"][:, :8])
    response_h8 = (early_response - batch["paired_x_m"]).square().mean() / denom
    state_h16 = group_loss(predicted[:, -1], batch["target"][:, -1])
    # The physical and toy suffixes have identical action increments, so only the
    # different prefix gains (.25 versus .75) contribute to the response error.
    late = ((.25 - .75) * excitation).square().mean() / target.square().mean()
    torch.testing.assert_close(terms["state_h8"], state_h8)
    torch.testing.assert_close(terms["response_h8"], response_h8)
    torch.testing.assert_close(terms["state_h16"], state_h16)
    torch.testing.assert_close(terms["response_h17_32"], late)
    expected = state_h8 + response_h8 + state_h16
    torch.testing.assert_close(terms["total"], expected + late if include_late else expected)

    early_terms = terms["state_h8"] + terms["response_h8"] + terms["state_h16"]
    added_grad, = torch.autograd.grad(terms["total"] - early_terms, world.prefix_gain, retain_graph=True)
    exact = 2 * (.25 - .75) * excitation.square().mean() / target.square().mean()
    assert added_grad.item() == pytest.approx(exact.item() if include_late else 0., rel=1e-5, abs=1e-7)

    if include_late:
        gain_grad, input_grad = torch.autograd.grad(terms["response_h17_32"],
                                                    (world.prefix_gain, batch["applied"]))
        assert gain_grad.item() == pytest.approx(exact.item(), rel=1e-5)
        assert torch.count_nonzero(input_grad[:12]).item() == 0
        assert (input_grad[12:15, :, 0] > 0).all()
        assert (input_grad[15:, :, 0] < 0).all()
