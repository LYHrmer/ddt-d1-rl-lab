"""Synthetic differentiable78 invariants; no archived model or physics run."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def lib():
    for path in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    torch = pytest.importorskip('torch')
    from tools54 import numerics54
    from tools78.model78 import FrozenPair78
    return SimpleNamespace(torch=torch, numeric=numerics54, cls=FrozenPair78)


def synthetic(lib, gain, *, quaternion_scale=1., fixed_quaternion=None, bad_output=None):
    torch = lib.torch

    class Synthetic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(gain, dtype=torch.float64))
            self.register_buffer('input_mean', torch.zeros(61, dtype=torch.float64))
            self.register_buffer('basis', torch.linspace(.01, .45, 45, dtype=torch.float64))
            self.register_buffer('quaternion_scale', torch.tensor(quaternion_scale, dtype=torch.float64))

        def forward(self, state, action):
            base = state + (self.gain + .013 * action[:, 4:5]) * self.basis
            if fixed_quaternion is None:
                q = torch.stack((1. + .02 * action[:, 0] + .01 * self.gain,
                                 .11 + .06 * action[:, 1] + .01 * state[:, 0],
                                 .07 + .08 * action[:, 2] + .01 * self.gain,
                                 .09 + .04 * action[:, 3] + .02 * state[:, 1]), dim=-1)
                q = q * self.quaternion_scale
            else:
                q = state.new_tensor(fixed_quaternion).expand(len(state), 4)
            velocity = torch.stack((.3 + .2 * action[:, 0] + .07 * action[:, 1] + .11 * self.gain + .05 * state[:, 23],
                                    -.2 + .15 * action[:, 1] - .04 * action[:, 2] + .08 * self.gain + .04 * state[:, 24],
                                    .1 + .3 * self.gain + (.07 + .02 * self.gain) * action[:, 3]
                                    + .09 * action[:, 0] * action[:, 2] + .04 * state[:, 25]), dim=-1)
            result = torch.cat((base[:, :3], q, base[:, 7:23], velocity, base[:, 26:]), dim=-1)
            if bad_output == 'shape':
                return result[:, :-1]
            if bad_output == 'dtype':
                return result.float()
            if bad_output == 'nonfinite':
                return result * state.new_tensor(float('nan'))
            return result

    return Synthetic()


def inputs(lib):
    torch = lib.torch
    state = torch.linspace(-.15, .25, 90, dtype=torch.float64).reshape(2, 45)
    action = torch.linspace(-.2, .3, 32, dtype=torch.float64).reshape(2, 16)
    direction = torch.linspace(.3, -.2, 32, dtype=torch.float64).reshape(2, 16)
    return state, action, direction


def body(lib, state):
    return lib.numeric._rotate_inverse(state[:, 3:7], state[:, 23:26])[:, :2]


def test_world25_and_all42_other_coordinates_are_exact_copies(lib):
    torch = lib.torch
    control, donor = synthetic(lib, .2), synthetic(lib, 1.7)
    model = lib.cls(control, donor)
    state, action, direction = inputs(lib)
    c, a, mixed = control(state, action), donor(state, action), model(state, action)
    assert torch.equal(mixed[:, 25], a[:, 25])
    assert torch.equal(mixed[:, :23], c[:, :23]) and torch.equal(mixed[:, 26:], c[:, 26:])
    assert not torch.equal(c[:, :23], a[:, :23])
    assert not torch.equal(mixed[:, 23:25], c[:, 23:25])
    _, actual_jvp = torch.autograd.functional.jvp(lambda x: model(state, x)[:, 25], action, direction)
    _, expected_jvp = torch.autograd.functional.jvp(lambda x: donor(state, x)[:, 25], action, direction)
    torch.testing.assert_close(actual_jvp, expected_jvp, rtol=1e-13, atol=1e-14)
    assert torch.equal(model.predict(state, action), mixed)


@pytest.mark.parametrize('quaternion_scale', [1., 7.])
def test_body_xy_value_and_jvp_keep_rotation_and_correction_input_derivatives(lib, quaternion_scale):
    torch = lib.torch
    control, donor = synthetic(lib, .2, quaternion_scale=quaternion_scale), synthetic(lib, 1.7)
    model = lib.cls(control, donor)
    state, action, direction = inputs(lib)
    expected, expected_jvp = torch.autograd.functional.jvp(lambda x: body(lib, control(state, x)), action, direction)
    actual, actual_jvp = torch.autograd.functional.jvp(lambda x: body(lib, model(state, x)), action, direction)
    _, quaternion_jvp = torch.autograd.functional.jvp(lambda x: control(state, x)[:, 3:7], action, direction)
    assert bool((quaternion_jvp.abs() > 0).any())
    assert bool((donor(state, action)[:, 25] - control(state, action)[:, 25]).abs().min() > .1)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=2e-14)
    torch.testing.assert_close(actual_jvp, expected_jvp, rtol=1e-11, atol=2e-13)
    epsilon = 1e-5
    difference = (body(lib, model(state, action + epsilon * direction))
                  - body(lib, model(state, action - epsilon * direction))) / (2 * epsilon)
    torch.testing.assert_close(actual_jvp, difference, rtol=1e-6, atol=2e-8)


def test_all_six_frozen_h1_cost_values_and_derivatives_match_control(lib):
    torch = lib.torch
    control, donor = synthetic(lib, .2, quaternion_scale=3.), synthetic(lib, 1.7)
    model = lib.cls(control, donor)
    state, action, direction = inputs(lib)
    command = state.new_tensor([[.1, 0., .05], [-.1, 0., -.05]])
    previous = torch.zeros_like(action)

    def components(network, current):
        return lib.numeric.cost_components(network(state, current), command, current, previous, .46)

    expected, expected_jvp = torch.autograd.functional.jvp(lambda x: components(control, x), action, direction)
    actual, actual_jvp = torch.autograd.functional.jvp(lambda x: components(model, x), action, direction)
    assert actual.shape == (2, 6)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_jvp, expected_jvp, rtol=1e-11, atol=1e-11)
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    fused_cost = lib.numeric.cost(model(state, action + theta * direction), command,
                                  action + theta * direction, previous, .46).mean()
    fused_ad, = torch.autograd.grad(fused_cost, theta)
    epsilon = 1e-5
    costs = [lib.numeric.cost(model(state, action + sign * epsilon * direction), command,
             action + sign * epsilon * direction, previous, .46).mean() for sign in (1, -1)]
    torch.testing.assert_close(fused_ad, (costs[0] - costs[1]) / (2 * epsilon), rtol=1e-6, atol=2e-8)
    torch.testing.assert_close(fused_ad, expected_jvp.sum(dim=-1).mean(), rtol=1e-11, atol=1e-11)


def test_complete_predictors_remain_frozen_and_state_dicts_unchanged(lib):
    torch = lib.torch
    control, donor = synthetic(lib, .2), synthetic(lib, 1.7)
    before = [{name: value.clone() for name, value in net.state_dict().items()} for net in (control, donor)]
    model = lib.cls(control, donor)
    assert model.control is control and model.donor is donor
    assert not model.training and not control.training and not donor.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert set(model.state_dict()) == {prefix + '.' + key for prefix, saved in zip(('control', 'donor'), before)
                                      for key in saved}
    state, action, _ = inputs(lib)
    state.requires_grad_(True)
    action.requires_grad_(True)
    gradients = torch.autograd.grad(model(state, action).square().sum(), (state, action))
    assert all(bool(torch.isfinite(g).all()) and bool((g.abs() > 0).any()) for g in gradients)
    assert all(parameter.grad is None for parameter in model.parameters())
    for net, saved in zip((control, donor), before):
        assert set(net.state_dict()) == set(saved)
        assert all(torch.equal(net.state_dict()[key], value) for key, value in saved.items())


def test_nonunit_quaternion_changes_no_original_control_coordinate(lib):
    torch = lib.torch
    control = synthetic(lib, .2, quaternion_scale=7.)
    model = lib.cls(control, synthetic(lib, 1.7))
    state, action, _ = inputs(lib)
    original, mixed = control(state, action), model(state, action)
    assert bool((torch.linalg.vector_norm(original[:, 3:7], dim=-1) > 5).all())
    assert torch.equal(mixed[:, 3:7], original[:, 3:7])
    torch.testing.assert_close(body(lib, mixed), body(lib, original), rtol=1e-12, atol=2e-14)


@pytest.mark.parametrize('quaternion', [(0., 0., 0., 0.), (1., 1., 0., 0.), (1., 0., 1., 0.)])
def test_zero_quaternion_or_small_rzz_fails_explicitly(lib, quaternion):
    model = lib.cls(synthetic(lib, .2, fixed_quaternion=quaternion), synthetic(lib, 1.7))
    state, action, _ = inputs(lib)
    message = 'quaternion norm' if quaternion == (0., 0., 0., 0.) else 'abs\\(Rzz\\)'
    with pytest.raises(ValueError, match=message):
        model(state, action)


def test_negative_rzz_with_sufficient_absolute_margin_is_allowed(lib):
    control = synthetic(lib, .2, fixed_quaternion=(0., 1., 0., 0.))
    donor = synthetic(lib, 1.7)
    model = lib.cls(control, donor)
    state, action, _ = inputs(lib)
    result = model(state, action)
    lib.torch.testing.assert_close(body(lib, result), body(lib, control(state, action)), rtol=0, atol=0)
    assert lib.torch.equal(result[:, 25], donor(state, action)[:, 25])


@pytest.mark.parametrize('kind', ['state_width', 'action_width', 'batch', 'empty', 'dtype', 'nonfinite'])
def test_bad_input_contract_fails(lib, kind):
    model = lib.cls(synthetic(lib, .2), synthetic(lib, 1.7))
    state, action, _ = inputs(lib)
    if kind == 'state_width':
        state = state[:, :-1]
    elif kind == 'action_width':
        action = action[:, :-1]
    elif kind == 'batch':
        action = action[:1]
    elif kind == 'empty':
        state, action = state[:0], action[:0]
    elif kind == 'dtype':
        action = action.float()
    else:
        state[0, 0] = float('nan')
    with pytest.raises(ValueError):
        model(state, action)


@pytest.mark.parametrize('bad_output', ['shape', 'dtype', 'nonfinite'])
def test_bad_model_output_contract_fails(lib, bad_output):
    model = lib.cls(synthetic(lib, .2), synthetic(lib, 1.7, bad_output=bad_output))
    state, action, _ = inputs(lib)
    with pytest.raises(ValueError, match='both models must return'):
        model(state, action)


def test_parameter_dtype_or_trainability_changes_fail_closed(lib):
    model = lib.cls(synthetic(lib, .2), synthetic(lib, 1.7))
    state, action, _ = inputs(lib)
    with pytest.raises(ValueError, match='model/input CPU dtype mismatch'):
        model(state.float(), action.float())
    model.control.gain.requires_grad_(True)
    with pytest.raises(ValueError, match='parameters must remain frozen'):
        model(state, action)
    model.requires_grad_(False).train()
    with pytest.raises(ValueError, match='eval mode'):
        model(state, action)


def test_float32_call_keeps_output_dtype_and_exact_source_fields(lib):
    torch = lib.torch
    model = lib.cls(synthetic(lib, .2), synthetic(lib, 1.7)).float()
    state, action, _ = inputs(lib)
    state, action = state.float(), action.float()
    result = model(state, action)
    control, donor = model.control(state, action), model.donor(state, action)
    assert result.dtype == torch.float32
    assert torch.equal(result[:, :23], control[:, :23]) and torch.equal(result[:, 26:], control[:, 26:])
    assert torch.equal(result[:, 25], donor[:, 25])


def test_constructor_rejects_incomplete_nonmodules(lib):
    with pytest.raises(ValueError, match='complete torch modules'):
        lib.cls(object(), synthetic(lib, 1.7))


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_frozen_precision_interface_uses_control_buffer_without_duplicate(lib, dtype_name):
    torch = lib.torch
    dtype = getattr(torch, dtype_name)

    class SyntheticActorMatchingDtype(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
            self.register_buffer('reference', torch.zeros(1, dtype=dtype))

    model = lib.cls(synthetic(lib, .2), synthetic(lib, 1.7)).to(dtype=dtype)
    actor = SyntheticActorMatchingDtype()
    assert model.input_mean is model.control.input_mean
    assert model.input_mean.dtype == dtype
    assert lib.numeric._precision(model, actor) == dtype
    expected = {prefix + '.' + name for prefix, network in
                (('control', model.control), ('donor', model.donor)) for name in network.state_dict()}
    assert set(model.state_dict()) == expected
    assert 'input_mean' not in model._buffers
    assert {name for name, _ in model.named_buffers() if name.endswith('input_mean')} == {
        'control.input_mean', 'donor.input_mean'}
    with pytest.raises(AttributeError):
        model.input_mean = torch.zeros(61, dtype=dtype)
    other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    with pytest.raises(ValueError, match='actor floating weights/buffers must match model dtype'):
        lib.numeric._precision(model, actor.to(dtype=other_dtype))
