"""Asset-free fixed-teacher/Jacobian tests; no environment or physical rollout."""
from __future__ import annotations

import copy
import inspect
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
    torch = pytest.importorskip("torch")
    from tools54 import numerics54
    from tools55 import observed55
    return SimpleNamespace(torch=torch, numeric=numerics54, observed=observed55)


def fixture(lib, *, bias=.01, response=.05):
    torch, nn = lib.torch, lib.torch.nn

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(627, 3, bias=False, dtype=torch.float64)
            self.policy_net = nn.Linear(60, 1, bias=False, dtype=torch.float64)
            self.action_head = nn.Linear(1, 16, bias=False, dtype=torch.float64)
            with torch.no_grad():
                for value in self.parameters():
                    value.zero_()

    class BiasedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("input_mean", torch.zeros(61, dtype=torch.float64))
            vector = torch.zeros(45, dtype=torch.float64)
            vector[28] = bias
            self.register_buffer("bias", vector)
            vector = torch.zeros(45, dtype=torch.float64)
            vector[28] = response
            self.register_buffer("response", vector)

        def forward(self, state, action):
            return state + self.bias + action[..., 3:4] * self.response

    actor = Actor().eval().requires_grad_(False)
    model = BiasedLinear().eval().requires_grad_(False)
    state = torch.zeros(45, dtype=torch.float64)
    state[2:4] = torch.tensor([.5, 1.], dtype=torch.float64)
    state[7:23] = torch.tensor([0., .8, -1.5, 0.] * 4, dtype=torch.float64)
    command = torch.tensor([0., 0., .1], dtype=torch.float64)
    last = torch.zeros(16, dtype=torch.float64)
    frame, filtered = lib.numeric._observation(state, command, last, torch.zeros(3, dtype=torch.float64))
    start = {"state": state, "frame": frame, "history": frame.repeat(10, 1), "filtered": filtered,
             "last": last, "fifo": torch.zeros((0, 16), dtype=torch.float64),
             "shaper": SimpleNamespace(state=None, history=None, raw_history=None),
             "commands": command.repeat(33, 1), "nominal_height": .5, "seed": 55301, "tick": 110}
    baseline = state.repeat(32, 1)
    return model, actor, start, baseline


def test_one_step_centering_removes_bias_but_preserves_local_state_and_action_jacobians(lib):
    model, _actor, start, baseline = fixture(lib)
    actions = lib.torch.zeros((1, 16), dtype=lib.torch.float64)
    centered = lib.observed.CenteredModel(model, start["state"][None], actions, baseline[:1])
    assert not centered.correction.requires_grad
    state = start["state"][None].clone().requires_grad_()
    action = actions.clone().requires_grad_()
    output = centered(state, action)
    assert lib.torch.equal(output, baseline[:1])
    state_gradient, action_gradient = lib.torch.autograd.grad(output[0, 28], (state, action))
    assert state_gradient[0, 28].item() == 1.
    assert action_gradient[0, 3].item() == .05
    assert centered.cursor == 1
    with pytest.raises(ValueError, match="clock exceeds"):
        centered(state, action)


def test_no_outer_quaternion_normalization_is_added_to_centered_model(lib):
    model, _actor, start, baseline = fixture(lib)
    with lib.torch.no_grad():
        model.bias.zero_()
        model.response.zero_()
        model.response[3] = .4
    reference_action = lib.torch.zeros((1, 16), dtype=lib.torch.float64)
    centered = lib.observed.CenteredModel(model, start["state"][None], reference_action, baseline[:1])
    action = reference_action.clone().requires_grad_()
    output = centered(start["state"][None], action)
    derivative, = lib.torch.autograd.grad(output[0, 3], action)
    assert derivative[0, 3].item() == .4  # An outer unit normalization would erase this radial derivative.


def test_baseline_and_correction_are_fixed_across_theta_and_external_input_mutation(lib):
    model, actor, start, baseline = fixture(lib)
    model_before = {key: value.clone() for key, value in model.state_dict().items()}
    baseline.requires_grad_()
    estimator = lib.observed.ObservedRollout(model, actor, start, baseline)
    correction = estimator.correction
    teacher = estimator.teacher_trace
    with lib.torch.no_grad():
        baseline[:, 28] = 99.
        start["commands"][:, 2] = -.2
    plus, plus_trace = estimator.rollout(lib.torch.tensor(.01, dtype=lib.torch.float64), return_trace=True)
    minus, minus_trace = estimator.rollout(lib.torch.tensor(-.01, dtype=lib.torch.float64), return_trace=True)
    zero, zero_trace = estimator.rollout(lib.torch.tensor(0., dtype=lib.torch.float64), return_trace=True)
    assert plus != minus
    assert not lib.torch.equal(plus_trace["states"], minus_trace["states"])
    assert lib.torch.equal(estimator.correction, correction)
    for key, value in teacher.items():
        assert lib.torch.equal(estimator.teacher_trace[key], value)
        lib.torch.testing.assert_close(zero_trace[key], value, rtol=0., atol=1e-10)
    assert estimator.admission(zero, zero_trace)["passed"] is True
    assert estimator.cursor == 0 and estimator.branch_calls == 3
    assert baseline.grad is None
    assert all(lib.torch.equal(value, model.state_dict()[key]) for key, value in model_before.items())
    # Diagnostic copies cannot be used to mutate calibration through aliases.
    correction.zero_()
    teacher["states"].zero_()
    assert lib.torch.count_nonzero(estimator.correction) > 0
    assert estimator.teacher_trace["states"][0, 3].item() == 1.


def test_closed_loop_gradient_is_correct_for_biased_linear_system_and_original_fd_passes(lib):
    model, actor, start, baseline = fixture(lib)
    row, traces = lib.observed.predict_gradients(model, actor, start, baseline,
                                                return_trace=True, extra_eps=True)
    # At theta0 the unbiased reference yaw is0.  Each step adds beta*theta/3,
    # so d yaw_t/d theta=(t+1)*beta/3.  The yaw tracking derivative is -5 times
    # that sensitivity; action-smoothness has zero derivative at zero action.
    expected = -5 * .05 / 3 * (32 + 1) / 2
    assert row["gradient"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert row["autograd_agrees"] is True and row["direction_identifiable"] is True
    assert row["admission"]["passed"] is True
    assert max(row["admission"]["max_gaps"].values()) <= 1e-10
    assert [item["epsilon"] for item in row["fd_checks"]] == [.001, .005, .01]
    assert row["model_central_difference"] == row["fd_checks"][0]["model_central_difference"]
    assert all(item["autograd_agrees"] for item in row["fd_checks"])
    assert row["theta_branches"] == 7 and row["branch_model_calls"] == 7 * 32
    assert row["centering_model_calls"] == 32
    assert set(traces["fd_branches"]) == {"0.001", "0.005", "0.01"}
    assert all(not item.requires_grad for item in traces["teacher"].values())


def test_rollout_cursor_resets_even_after_model_failure(lib, monkeypatch):
    model, actor, start, baseline = fixture(lib)
    estimator = lib.observed.ObservedRollout(model, actor, start, baseline)
    original = model.forward
    calls = []

    def broken(state, action):
        calls.append(None)
        if len(calls) == 3:
            raise RuntimeError("synthetic model failure")
        return original(state, action)

    monkeypatch.setattr(model, "forward", broken)
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        estimator.rollout(lib.torch.tensor(0., dtype=lib.torch.float64))
    assert estimator.cursor == 0
    monkeypatch.setattr(model, "forward", original)
    first = estimator.rollout(lib.torch.tensor(0., dtype=lib.torch.float64))
    second = estimator.rollout(lib.torch.tensor(0., dtype=lib.torch.float64))
    assert lib.torch.equal(first, second) and estimator.cursor == 0


def test_admission_failure_keeps_per_field_gaps(lib):
    model, actor, start, baseline = fixture(lib)
    estimator = lib.observed.ObservedRollout(model, actor, start, baseline)
    value, trace = estimator.rollout(lib.torch.tensor(0., dtype=lib.torch.float64), return_trace=True)
    changed = dict(trace)
    changed["filtered"] = trace["filtered"] + 1e-8
    with pytest.raises(lib.observed.TeacherParityError) as failure:
        estimator.admission(value, changed)
    assert failure.value.admission["passed"] is False
    assert failure.value.admission["max_gaps"]["filtered"] == pytest.approx(1e-8)


def test_public_gradient_api_excludes_future_perturbed_trajectories_and_true_fd_labels(lib):
    parameters = inspect.signature(lib.observed.predict_gradients).parameters
    assert tuple(parameters) == ("model64", "actor64", "start", "baseline_states", "return_trace", "extra_eps")
    model, actor, start, baseline = fixture(lib)
    with pytest.raises(TypeError, match="unexpected keyword"):
        lib.observed.predict_gradients(model, actor, start, baseline, true_gradient=123.)
    with pytest.raises(TypeError, match="unexpected keyword"):
        lib.observed.predict_gradients(model, actor, start, baseline, perturbed_states=baseline)

    class UnrelatedEvidence:
        def __deepcopy__(self, _memo):
            raise AssertionError("unrelated future evidence must not be read/copied")

    start["snapshot"] = start["true_gradient"] = start["future_trajectories"] = UnrelatedEvidence()
    estimator = lib.observed.ObservedRollout(model, actor, start, baseline)
    assert estimator.correction.shape == (32, 45)


def test_float32_or_unfrozen_models_are_rejected_without_mutating_them(lib):
    model, actor, start, baseline = fixture(lib)
    actor32 = copy.deepcopy(actor).float()
    with pytest.raises(ValueError, match="match model dtype"):
        lib.observed.ObservedRollout(model, actor32, start, baseline)
    actor.action_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="already be frozen"):
        lib.observed.ObservedRollout(model, actor, start, baseline)
    assert actor.action_head.weight.requires_grad is True
