"""Observed-forward/model-backward gradients for the independent55 comparison.

Only a theta=0 physical baseline enters this module.  The teacher recomputes
reference actions with the frozen54 double graph; the correction is fixed and
detached before any theta branch.  No true finite-difference labels, perturbed
physical trajectories, files or environments are accepted/read here.

This is an oracle-assisted, offline gradient estimator.  It is not a deployable
world model: control execution still uses the fixed actor and scalar residual.
Centering preserves the base model's Jacobian at the same state/action input;
the composite rollout follows a different linearization path near observations.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from tools54 import numerics54 as numeric

HORIZON = 32
BASELINE_TOLERANCE = 1e-10
TRACE_FIELDS = ("states", "actions", "applied_actions", "frames", "filtered",
                "shaper_state", "actor_inputs", "costs", "cost_components")
require = numeric.require


def _fixed_tensor(value):
    """Take an independent CPU double value, never a reference/gradient alias."""
    return numeric._tensor(value, torch.float64, clone=True).detach()


def _frozen_double(model, actor):
    require(numeric._precision(model, actor) == torch.float64, "observed55 requires float64-only modules")
    for name, module in (("model", model), ("actor", actor)):
        require(not module.training, name + " must be in evaluation mode")
        require(all(not value.requires_grad for value in (*module.parameters(), *module.buffers())),
                name + " parameters/buffers must already be frozen")


def _point_copy(start):
    # Deliberately exclude snapshot/MjData, file handles and unrelated evidence.
    point = {name: _fixed_tensor(start[name]) for name in
             ("state", "frame", "history", "filtered", "last", "fifo", "commands")}
    point["nominal_height"] = float(start["nominal_height"])
    require(math.isfinite(point["nominal_height"]), "nonfinite nominal height")
    point["seed"], point["tick"] = int(start["seed"]), int(start["tick"])
    source = start["shaper"]
    if source.state is None:
        point["shaper"] = SimpleNamespace(state=None, history=None, raw_history=None)
    else:
        state = float(source.state)
        require(math.isfinite(state), "nonfinite reference-filter state")
        point["shaper"] = SimpleNamespace(state=state, history=_fixed_tensor(source.history),
                                           raw_history=_fixed_tensor(source.raw_history))
    return point


class Teacher(torch.nn.Module):
    """Replay fixed post-step states while54 recomputes the theta=0 actor chain."""

    def __init__(self, baseline_states):
        super().__init__()
        states = _fixed_tensor(baseline_states)
        require(states.shape == (HORIZON, 45), "baseline_states must be exactly [32,45]")
        norm = torch.linalg.vector_norm(states[:, 3:7], dim=-1)
        require(bool(torch.allclose(norm, torch.ones_like(norm), rtol=0., atol=1e-8)),
                "physical baseline quaternions must be unit length")
        self.register_buffer("input_mean", torch.zeros(61, dtype=torch.float64, device="cpu"))
        self.register_buffer("targets", states)
        self.cursor = 0

    def reset(self):
        self.cursor = 0

    def forward(self, _state, _action):
        require(self.cursor < HORIZON, "teacher clock exceeds baseline horizon")
        value = self.targets[self.cursor:self.cursor + 1]
        self.cursor += 1
        return value


class CenteredModel(torch.nn.Module):
    """F(s,a)+detach(ref_next-F(ref_s,ref_a)); no outer quaternion normalization.

    The calibration tensors are cloned and detached once.  Forward preserves
    the base model's local state/action Jacobians at the same input.  reset() only resets this
    wrapper's time cursor, never the base model or its parameters/buffers.
    """

    def __init__(self, base, reference_states, reference_actions, reference_next):
        super().__init__()
        states, actions, targets = map(_fixed_tensor, (reference_states, reference_actions, reference_next))
        require(states.ndim == 2 and states.shape[1] == 45 and len(states) > 0
                and targets.shape == states.shape and actions.shape == (len(states), 16),
                "centering reference shapes differ")
        require(base.input_mean.dtype == torch.float64 and base.input_mean.device.type == "cpu",
                "centering base must be CPU float64")
        require(not base.training and all(not p.requires_grad for p in (*base.parameters(), *base.buffers())),
                "centering base must be frozen and in evaluation mode")
        self.base = base
        self.register_buffer("input_mean", base.input_mean.detach().clone())
        with torch.no_grad():
            # Match the rollout's batch1 arithmetic at every reference boundary.
            correction = torch.cat([targets[t:t + 1] - base(states[t:t + 1], actions[t:t + 1])
                                    for t in range(len(states))])
        require(bool(torch.isfinite(correction).all()), "nonfinite fixed correction")
        self.register_buffer("correction", correction.detach().clone())
        self.cursor = 0
        self.total_forward_calls = 0

    def reset(self):
        self.cursor = 0

    def forward(self, state, action):
        require(self.cursor < len(self.correction), "centered clock exceeds fixed correction horizon")
        result = self.base(state, action) + self.correction[self.cursor:self.cursor + 1]
        self.cursor += 1
        self.total_forward_calls += 1
        return result


class TeacherParityError(ValueError):
    """A rejected theta=0 chain, retaining its measured admission diagnostics."""

    def __init__(self, admission):
        self.admission = admission
        super().__init__("centered theta=0 chain differs from fixed teacher: " + repr(admission["max_gaps"]))


class ObservedRollout:
    """Reusable fixed calibration; every rollout shares it and restarts at tick0.

    Input baseline_states is exactly the32 post-step physical states at theta0.
    The copied start provides the pre-step state, memory and command schedule.
    No future physical action or perturbed-trajectory inputs are used.
    """

    def __init__(self, model64, actor64, start, baseline_states):
        _frozen_double(model64, actor64)
        self._actor = actor64
        self._start = _point_copy(start)
        teacher = Teacher(baseline_states)
        try:
            with torch.no_grad():
                value, trace = numeric.imaginary(teacher, actor64, self._start,
                    torch.tensor(0., dtype=torch.float64, device="cpu"), return_trace=True, horizon=HORIZON)
            require(teacher.cursor == HORIZON, "teacher did not consume all32 states")
        finally:
            teacher.reset()
        self.teacher_cost = float(value)
        self._teacher_trace = {name: tensor.detach().clone() for name, tensor in trace.items()}
        reference_states = torch.cat((self._start["state"][None], teacher.targets[:-1]))
        self._centered = CenteredModel(model64, reference_states,
                                       self._teacher_trace["applied_actions"], teacher.targets)
        self.branch_calls = 0

    @property
    def correction(self):
        """Return an independent diagnostic copy of the fixed detached correction."""
        return self._centered.correction.detach().clone()

    @property
    def cursor(self):
        return self._centered.cursor

    @property
    def teacher_trace(self):
        return {key: value.detach().clone() for key, value in self._teacher_trace.items()}

    def rollout(self, theta, *, return_trace=False):
        self._centered.reset()
        self.branch_calls += 1
        try:
            result = numeric.imaginary(self._centered, self._actor, self._start, theta,
                                        return_trace=return_trace, horizon=HORIZON)
            require(self._centered.cursor == HORIZON, "branch did not consume all32 fixed corrections")
            return result
        finally:
            self._centered.reset()

    def admission(self, value, trace):
        gaps = {name: float((trace[name].detach() - self._teacher_trace[name]).abs().max())
                for name in TRACE_FIELDS}
        gaps["mean_cost"] = abs(float(value.detach()) - self.teacher_cost)
        receipt = {"passed": max(gaps.values()) <= BASELINE_TOLERANCE,
                   "max_gaps": gaps, "tolerance": BASELINE_TOLERANCE, "horizon": HORIZON,
                   "baseline_shape": [HORIZON, 45], "dtype": "float64", "device": "cpu",
                   "frozen_parameters": True, "correction_detached": not self._centered.correction.requires_grad,
                   "offline_estimation_only": True, "requires_observed_theta0_baseline": True,
                   "outer_quaternion_normalization": False}
        if not receipt["passed"]:
            raise TeacherParityError(receipt)
        return receipt


def _quaternion_deviation(trace):
    return float((torch.linalg.vector_norm(trace["states"].detach()[:, 3:7], dim=-1) - 1.).abs().max())


def predict_gradients(model64, actor64, start, baseline_states, *, return_trace=False, extra_eps=False):
    """Return one numeric.gradients-compatible row plus admission/model-FD checks.

    The fixed selfcheck epsilon=.001 and original direction/agreement thresholds
    are unchanged.  extra_eps=True reports model-only ±.005/.01 diagnostics;
    they never select a direction or alter the original selfcheck gate.
    return_trace=True returns (row, traces), with teacher/centered/fd_branches
    tensor dictionaries.  No file, environment, true-FD-label or perturbed
    physical-trajectory input exists in this interface.
    """
    estimator = ObservedRollout(model64, actor64, start, baseline_states)
    theta = torch.tensor(0., dtype=torch.float64, device="cpu", requires_grad=True)
    value, trace = estimator.rollout(theta, return_trace=True)
    admission = estimator.admission(value, trace)
    derivative, = torch.autograd.grad(value, theta, retain_graph=return_trace)
    analytic = float(derivative)
    require(math.isfinite(analytic), "nonfinite centered policy gradient")
    checks, branches = [], {}
    deviation = _quaternion_deviation(trace)
    for epsilon in ((numeric.FD_EPS, .005, .01) if extra_eps else (numeric.FD_EPS,)):
        with torch.no_grad():
            plus, plus_trace = estimator.rollout(torch.tensor(epsilon, dtype=torch.float64, device="cpu"),
                                                 return_trace=True)
            minus, minus_trace = estimator.rollout(torch.tensor(-epsilon, dtype=torch.float64, device="cpu"),
                                                   return_trace=True)
            numerical = float((plus - minus) / (2 * epsilon))
        require(math.isfinite(numerical), "nonfinite centered model finite difference")
        checks.append({"epsilon": epsilon, "plus_cost": float(plus), "minus_cost": float(minus),
                       "model_central_difference": numerical,
                       "direction_identifiable": abs(analytic) >= 1e-5 and abs(numerical) >= 1e-5
                       and analytic * numerical > 0,
                       "autograd_agrees": abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4})
        deviation = max(deviation, _quaternion_deviation(plus_trace), _quaternion_deviation(minus_trace))
        if return_trace:
            branches[str(epsilon)] = {"plus": plus_trace, "minus": minus_trace}
    first = checks[0]
    row = {"seed": estimator._start["seed"], "tick": estimator._start["tick"], "cost": float(value.detach()),
           "gradient": analytic, "model_central_difference": first["model_central_difference"],
           "direction_identifiable": first["direction_identifiable"], "autograd_agrees": first["autograd_agrees"],
           "admission": admission, "fd_checks": checks, "teacher_cost": estimator.teacher_cost,
           "max_quaternion_norm_deviation": deviation, "theta_branches": estimator.branch_calls,
           "centering_model_calls": HORIZON, "branch_model_calls": estimator._centered.total_forward_calls,
           "scope": "offline observed-forward/model-backward; fixed actor and scalar residual"}
    if not return_trace:
        return row
    return row, {"teacher": estimator.teacher_trace, "centered": trace, "fd_branches": branches}
