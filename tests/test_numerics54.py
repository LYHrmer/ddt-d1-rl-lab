"""Synthetic precision equivalence checks; no robot, checkpoint asset or physics."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lib():
    for path in (ROOT, ROOT / ".python-deps"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from tools33.reference_shape33 import ReferenceShaper
    from tools52 import adapter52, control52, dynamics52
    from tools54 import numerics54
    return SimpleNamespace(np=np, torch=torch, adapter=adapter52, control=control52,
                           dynamics=dynamics52, new=numerics54, Shaper=ReferenceShaper)


def fixture_pair(lib, *, delay=0):
    torch, np = lib.torch, lib.np
    weights = {}
    for prefix, layout in (("mlp_extractor.encoder", ((0, 128, 627), (2, 64, 128), (4, 3, 64))),
                           ("mlp_extractor.policy_net", ((0, 128, 60), (2, 128, 128)))):
        for index, outgoing, incoming in layout:
            weights[f"{prefix}.{index}.weight"] = torch.full((outgoing, incoming), .001)
            weights[f"{prefix}.{index}.bias"] = torch.full((outgoing,), .01)
    weights["action_net.weight"] = torch.full((16, 128), .001)
    weights["action_net.bias"] = torch.linspace(-.01, .01, 16)
    actor = lib.adapter.actor_from_state_dict(weights)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(54001)
        model = lib.dynamics.Dynamics52()
    with torch.no_grad():
        model.ridge_coef[45 + 3, 28] = .001
        model.ridge_coef[45 + 7, 28] = -.001
        model.ridge_coef[-1, 2] = .00001
    model.eval().requires_grad_(False)
    state = np.linspace(-.01, .01, 45)
    state[:3] = [.02, -.01, .5]
    state[3:7] = [1., .01, -.02, .03]
    state[3:7] /= np.linalg.norm(state[3:7])
    state[7:23] = np.array([0., .8, -1.5, 0.] * 4) + np.linspace(-.002, .002, 16)
    command = np.array([0., 0., .2])
    last = np.linspace(-.02, .02, 16).astype(np.float32)
    frame, filtered = lib.adapter.build_observation(state, command, last, np.array([.02, -.01, .03]))
    frame, filtered = frame.numpy(), filtered.numpy()
    history = np.repeat(frame[None], 10, axis=0)
    shaper = lib.Shaper()
    shaper(np.r_[frame, history.ravel()].astype(np.float32))
    commands = np.repeat(command[None], 33, axis=0)
    commands[8:20, 2] = -.2
    commands[20:, 2] = 0.
    start = {"state": state, "frame": frame, "history": history, "filtered": filtered,
             "last": last, "fifo": np.zeros((delay, 16), np.float32), "shaper": shaper,
             "commands": commands, "nominal_height": .5, "seed": 53301, "tick": 110}
    return model, actor, start


@pytest.mark.parametrize("delay", [0, 2])
def test_float32_closed_loop_states_cost_and_autograd_are_bitwise52(lib, delay):
    model, actor, start = fixture_pair(lib, delay=delay)
    observed = []
    hook = model.register_forward_hook(lambda _module, inputs, output: observed.append((inputs[1], output)))
    old_theta = lib.torch.tensor(.003, requires_grad=True)
    old_value = lib.control.imaginary(model, actor, start, old_theta)
    old_gradient, = lib.torch.autograd.grad(old_value, old_theta)
    hook.remove()
    new_theta = lib.torch.tensor(.003, requires_grad=True)
    new_value, trace = lib.new.imaginary(model, actor, start, new_theta, return_trace=True)
    new_gradient, = lib.torch.autograd.grad(new_value, new_theta)
    assert lib.torch.equal(new_value, old_value)
    assert lib.torch.equal(new_gradient, old_gradient)
    assert lib.torch.equal(trace["states"], lib.torch.cat([item[1] for item in observed]))
    assert lib.torch.equal(trace["applied_actions"], lib.torch.cat([item[0] for item in observed]))
    assert trace["cost_components"].shape == (32, 6) and trace["actor_inputs"].shape == (32, 627)
    assert trace["frames"].dtype == lib.torch.float32
    assert trace["filtered"].dtype == trace["shaper_state"].dtype == lib.torch.float64
    assert lib.torch.equal(trace["costs"].mean(), new_value)
    lib.torch.testing.assert_close(trace["cost_components"].sum(-1), trace["costs"], rtol=1e-6, atol=1e-7)


def test_float32_gradient_receipt_keeps_original_epsilon_and_gates(lib):
    model, actor, start = fixture_pair(lib)
    assert lib.new.FD_EPS == .001
    assert lib.new.gradients(model, actor, [start]) == lib.control.gradients(model, actor, [start])


def test_double_graph_never_falls_back_to_float32_actor_or_model_inputs(lib):
    model32, actor32, start = fixture_pair(lib)
    model, actor = copy.deepcopy(model32).double(), copy.deepcopy(actor32).double()
    # Promotion preserves the already-trained values; no reinitialization or fit.
    for original, promoted in zip(actor32.parameters(), actor.parameters(), strict=True):
        assert lib.torch.equal(original.double(), promoted)
    dtypes = []
    hooks = [module.register_forward_pre_hook(
        lambda _module, inputs: dtypes.extend(value.dtype for value in inputs))
        for module in (actor.encoder[0], actor.policy_net[0], actor.action_head, model)]
    theta = lib.torch.tensor(.003, dtype=lib.torch.float64, requires_grad=True)
    value, trace = lib.new.imaginary(model, actor, start, theta, return_trace=True)
    derivative, = lib.torch.autograd.grad(value, theta)
    for hook in hooks:
        hook.remove()
    assert dtypes and set(dtypes) == {lib.torch.float64}
    assert value.dtype == derivative.dtype == lib.torch.float64
    assert all(item.dtype == lib.torch.float64 for item in trace.values())
    assert bool(lib.torch.isfinite(derivative))
    assert all(value.dtype == lib.torch.float32 for value in actor32.parameters())
    # The old actor.forward hardcodes float32 and cannot execute double weights;
    # the new graph must route through the independently typed actor components.
    with pytest.raises(RuntimeError):
        actor(lib.torch.zeros((1, 627), dtype=lib.torch.float64))


def test_double_constants_retain_sub_float32_height_information(lib):
    state = lib.torch.zeros((1, 45), dtype=lib.torch.float64)
    state[0, 3] = 1.
    state[0, 2] = .5 + 2. ** -40
    command = lib.torch.zeros((1, 3), dtype=lib.torch.float64)
    action = lib.torch.zeros((1, 16), dtype=lib.torch.float64)
    components = lib.new.cost_components(state, command, action, action, .5)
    total = lib.new.cost(state, command, action, action, .5)
    assert components.dtype == total.dtype == lib.torch.float64
    assert components[0, 4].item() == 20 * (2. ** -40) ** 2
    assert total.item() == components[0, 4].item() > 0.


def test_entire_double_graph_matches_central_difference_at_original_epsilon(lib):
    model, actor, start = fixture_pair(lib)
    model, actor = model.double(), actor.double()
    theta = lib.torch.tensor(0., dtype=lib.torch.float64, requires_grad=True)
    value = lib.new.imaginary(model, actor, start, theta)
    analytic, = lib.torch.autograd.grad(value, theta)
    with lib.torch.no_grad():
        plus = lib.new.imaginary(model, actor, start, lib.torch.tensor(.001, dtype=lib.torch.float64))
        minus = lib.new.imaginary(model, actor, start, lib.torch.tensor(-.001, dtype=lib.torch.float64))
        numerical = (plus - minus) / .002
    assert abs(analytic.item()) > 1e-8
    assert numerical.item() == pytest.approx(analytic.item(), rel=1e-7, abs=2e-10)


def test_mixed_model_actor_precision_is_rejected_without_mutating_weights(lib):
    model, actor, start = fixture_pair(lib)
    model.double()
    with pytest.raises(ValueError, match="match model dtype"):
        lib.new.imaginary(model, actor, start, lib.torch.tensor(0., dtype=lib.torch.float64))
    assert actor.action_head.weight.dtype == lib.torch.float32
