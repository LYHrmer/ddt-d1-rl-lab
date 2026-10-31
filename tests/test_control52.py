"""Synthetic closed-loop and gradient-gate checks; no robot assets or physics."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lib():
    for directory in (ROOT, ROOT / ".python-deps"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from tools52 import adapter52, control52
    return SimpleNamespace(np=np, torch=torch, adapter=adapter52, control=control52)


def synthetic_loop(lib, horizon):
    """An observable scalar yaw system embedded in valid state/actor shapes.

    r_next = 0.8*r + 0.2*u; fixed actor u = 0.02 + 0.3*filtered_r.
    The real adapter supplies the gyro filter, history, reference shaping and
    theta residual.  All other coordinates remain fixed.  This is a synthetic
    recurrence for testing the computational graph, not a robot model.
    """
    np, torch = lib.np, lib.torch
    state = np.zeros(45, np.float32)
    state[2], state[3] = 0.5, 1.0
    state[7:23] = [0., .8, -1.5, 0.] * 4
    command = np.array([0., 0., .1], np.float32)
    previous = np.zeros(16, np.float32)
    filtered = np.zeros(3, np.float64)
    frame, _ = lib.adapter.build_observation(state, command, previous, filtered)
    frame = frame.numpy()
    start = {"state": state, "frame": frame,
             "history": np.repeat(frame[None], 10, axis=0),
             "filtered": filtered, "last": previous, "fifo": np.zeros((0, 16), np.float32),
             "shaper": SimpleNamespace(state=None, history=None, raw_history=None),
             "commands": np.repeat(command[None], horizon + 1, axis=0),
             "nominal_height": .5, "seed": 52001, "tick": 65,
             # These must remain irrelevant: future issued actions are computed
             # from model-predicted observations inside the closed loop.
             "logged_future_actions": np.full((horizon, 16), np.nan, np.float32)}

    class Actor:
        def __init__(self):
            self.inputs, self.outputs = [], []

        def __call__(self, observation):
            self.inputs.append(observation)
            direction = observation.new_tensor([0., 0., 0., 1., 0., 0., 0., -1.,
                                                 0., 0., 0., 1., 0., 0., 0., -1.])
            value = (.02 + .3 * observation[..., 2] / .25).unsqueeze(-1) * direction
            self.outputs.append(value)
            return value

    class Model:
        def __init__(self):
            self.actions, self.outputs = [], []

        def __call__(self, state_value, action):
            self.actions.append(action)
            next_yaw = .8 * state_value[..., 28] + .1 * (action[..., 3] - action[..., 7])
            basis = torch.nn.functional.one_hot(torch.tensor(28), 45).to(state_value)
            result = state_value + (next_yaw - state_value[..., 28]).unsqueeze(-1) * basis
            self.outputs.append(result)
            return result

    return start, Actor(), Model()


def test_imaginary_recomputes_actions_from_each_predicted_state(lib, monkeypatch):
    horizon = 4
    monkeypatch.setattr(lib.control, "HORIZON", horizon)
    start, actor, model = synthetic_loop(lib, horizon)
    theta = lib.torch.tensor(.01, requires_grad=True)
    value = lib.control.imaginary(model, actor, start, theta)
    assert bool(lib.torch.isfinite(value))
    assert len(actor.inputs) == len(model.actions) == horizon
    assert actor.inputs[0][0, 2].item() == 0.
    assert actor.inputs[1][0, 2].item() > 0.
    assert not lib.torch.equal(model.actions[0], model.actions[-1])
    previous_filter = lib.torch.zeros(())
    for tick in range(horizon):
        measured_yaw = actor.inputs[tick][0, 2] / .25
        if tick:
            previous_filter = .97 * model.outputs[tick - 1][0, 28] + .03 * previous_filter
        lib.torch.testing.assert_close(measured_yaw, previous_filter)
        expected_u = .02 + .3 * measured_yaw + theta * ((.1 - measured_yaw) / .3).clamp(-1, 1)
        lib.torch.testing.assert_close(model.actions[tick][0, 3], expected_u)
        lib.torch.testing.assert_close(model.actions[tick][0, 7], -expected_u)
        if tick:
            # New raw history ends with the previous current frame.  Its gyro
            # columns survive reference shaping unchanged.
            lib.torch.testing.assert_close(actor.inputs[tick][0, 57 + 9 * 57 + 2],
                                            actor.inputs[tick - 1][0, 2])


def test_gradient_keeps_state_filter_and_actor_feedback_across_multiple_steps(lib, monkeypatch):
    horizon = 5
    monkeypatch.setattr(lib.control, "HORIZON", horizon)
    start, actor, model = synthetic_loop(lib, horizon)
    theta = lib.torch.tensor(0., requires_grad=True)
    value = lib.control.imaginary(model, actor, start, theta)
    actual, = lib.torch.autograd.grad(value, theta, retain_graph=True)
    final_from_first, = lib.torch.autograd.grad(model.outputs[-1][0, 28], model.actions[0])
    assert final_from_first[0, 3].item() > 0.

    # Independent forward-sensitivity recurrence at theta=0.  It includes the
    # direct residual, prior predicted state, gyro-filter memory and actor
    # feedback.  Detaching any of those temporal paths changes this derivative.
    yaw = filtered = dyaw = dfiltered = previous_u = previous_du = 0.
    expected = 0.
    for _ in range(horizon):
        error = (.1 - filtered) / .3
        assert -1 < error < 1
        u = .02 + .3 * filtered
        du = .3 * dfiltered + error
        next_yaw = .8 * yaw + .2 * u
        next_dyaw = .8 * dyaw + .2 * du
        # Four of 16 actions carry +/-u: the original action-rate term is
        # 0.1*(4/16)*(u-previous_u)^2, in addition to yaw tracking error.
        expected += (2 * (next_yaw - .1) / .04 * next_dyaw
                     + .05 * (u - previous_u) * (du - previous_du))
        filtered, dfiltered = (.97 * next_yaw + .03 * filtered,
                               .97 * next_dyaw + .03 * dfiltered)
        yaw, dyaw, previous_u, previous_du = next_yaw, next_dyaw, u, du
    assert actual.item() == pytest.approx(expected / horizon, rel=2e-5, abs=1e-6)


@pytest.mark.parametrize("slope,cubic,identifiable", [
    (2e-4, 0., True),
    (1e-6, 0., False),
    (2e-5, -40., False),
])
def test_gradient_receipt_requires_nontrivial_matching_analytic_and_fd_directions(
        lib, monkeypatch, slope, cubic, identifiable):
    def local_objective(_model, _actor, _point, theta):
        return slope * theta + cubic * theta.pow(3)

    monkeypatch.setattr(lib.control, "imaginary", local_objective)
    row, = lib.control.gradients(None, None, [{"seed": 52001, "tick": 65}])
    assert row["gradient"] == pytest.approx(slope, rel=1e-5)
    assert row["model_central_difference"] == pytest.approx(slope + cubic * 1e-6,
                                                           rel=1e-4, abs=1e-10)
    # The loose numerical-agreement tolerance alone permits small opposite
    # signs; the separate identifiability field must still reject this case.
    assert row["autograd_agrees"] is True
    assert row["direction_identifiable"] is identifiable


def direction_fixture():
    predictions, probes = [], []
    for index in range(9):
        identity = {"seed": 53000 + index // 3, "tick": (65, 205, 305)[index % 3]}
        predictions.append({**identity, "gradient": .2, "model_central_difference": .2,
                            "autograd_agrees": True, "direction_identifiable": True})
        probes.append({**identity, "gradient": .2, "gradient_double_epsilon": .2,
                       "identifiable": True, "repeated_baseline_exact": True,
                       "branches": [{"status": "COMPLETE"} for _ in range(6)]})
    return {"theta": -.01, "test_gradients": [copy.deepcopy(predictions) for _ in range(3)]}, probes


def test_direction_gate_requires_all_three_models_and_at_least_seven_identifiable_points(lib):
    sealed, probes = direction_fixture()
    result = lib.control.compare_directions(sealed, probes)
    assert result["passed"] is True
    assert [row["matches"] for row in result["models"]] == [9, 9, 9]
    for actual in probes[:2]:
        actual["identifiable"] = False
    assert lib.control.compare_directions(sealed, probes)["passed"] is True
    probes[2]["identifiable"] = False
    result = lib.control.compare_directions(sealed, probes)
    assert result["passed"] is False
    assert [row["matches"] for row in result["models"]] == [6, 6, 6]


@pytest.mark.parametrize("analytic,numerical,identifiable", [
    (1e-6, 1e-6, False),
    (2e-5, -2e-5, False),
    (-.2, -.2, True),
])
def test_one_models_unidentifiable_or_wrong_direction_points_cannot_be_counted(
        lib, analytic, numerical, identifiable):
    sealed, probes = direction_fixture()
    for predicted in sealed["test_gradients"][1][:3]:
        predicted.update(gradient=analytic, model_central_difference=numerical,
                         direction_identifiable=identifiable)
    result = lib.control.compare_directions(sealed, probes)
    assert [row["matches"] for row in result["models"]] == [9, 6, 9]
    assert result["models"][1]["passed"] is False
    assert result["passed"] is False


def test_direction_gate_also_requires_locked_theta_and_repeatable_real_baselines(lib):
    sealed, probes = direction_fixture()
    sealed["theta"] = None
    assert lib.control.compare_directions(sealed, probes)["passed"] is False
    sealed["theta"] = -.01
    probes[0]["repeated_baseline_exact"] = False
    assert lib.control.compare_directions(sealed, probes)["passed"] is False


@pytest.mark.parametrize("failed_branch", range(6))
def test_any_terminated_real_branch_fails_even_with_eight_clear_matching_points(lib, failed_branch):
    sealed, probes = direction_fixture()
    probes[-1]["branches"][failed_branch]["status"] = "TERMINATED"
    probes[-1]["identifiable"] = False
    result = lib.control.compare_directions(sealed, probes)
    assert [row["matches"] for row in result["models"]] == [8, 8, 8]
    assert result["passed"] is False
