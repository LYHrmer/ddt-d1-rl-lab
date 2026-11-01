"""Precision-controlled reproduction of the frozen52 imagined closed loop.

CPU float32 reproduces52's mixed arithmetic: gyro/observation construction and
the reference-filter scalar use float64, while frames, history, actor and model
use float32.  With float64 model/actor weights, all these operations use float64.
The caller promotes copies of the existing weights; this module never changes
weights, the controller, the horizon, the finite-difference step or thresholds.

The learned state transition still represents action-to-PD-to-physics implicitly.
This is a numerical diagnostic, not a differentiable implementation of MuJoCo.
"""
from __future__ import annotations

import math

import torch

HORIZON = 32
FD_EPS = .001
REFERENCE_ALPHA = math.exp(-.02 / .1)
COMPONENT_NAMES = ("vx", "yaw", "lateral", "tilt", "height", "action_smoothness")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, dtype, *, clone=False):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "CPU tensors required")
    result = torch.as_tensor(value, dtype=dtype, device="cpu")
    require(bool(torch.isfinite(result).all()), "nonfinite numerical input")
    return result.clone() if clone else result


def _precision(model, actor):
    reference = model.input_mean
    require(reference.device.type == "cpu" and reference.dtype in (torch.float32, torch.float64),
            "model.input_mean must be CPU float32 or float64")
    dtype = reference.dtype
    for name, module in (("model", model), ("actor", actor)):
        for value in (*module.parameters(), *module.buffers()):
            require(value.device.type == "cpu", name + " must stay on CPU")
            if value.is_floating_point():
                require(value.dtype == dtype, name + " floating weights/buffers must match model dtype")
    return dtype


def actor_forward(actor, observation):
    """Fixed actor [...,627] -> [...,16], following its weights without a cast32.

    Reuse the original encoder627→128→64→3, policy60→128→128 and action head.
    Calling actor.forward would force float32 in the frozen52 adapter, so this
    independent function calls the same component modules directly.
    """
    require(observation.shape[-1] == 627 and observation.device.type == "cpu",
            "actor observation must be CPU [...,627]")
    require(observation.dtype == actor.action_head.weight.dtype,
            "actor observation and weights must have the same dtype")
    velocity = actor.encoder(observation)
    latent = actor.policy_net(torch.cat((observation[..., :57], velocity), dim=-1))
    return actor.action_head(latent).clamp(-1.0, 1.0)


def _rotate_inverse(quat, vector):
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    cross = torch.linalg.cross(quat[..., 1:], vector, dim=-1)
    return vector - 2 * quat[..., :1] * cross + 2 * torch.linalg.cross(quat[..., 1:], cross, dim=-1)


def _cost_parts_and_total(state, command, action, previous, nominal_height):
    require(state.dtype in (torch.float32, torch.float64) and state.device.type == "cpu",
            "cost requires CPU float32 or float64 state")
    require(all(value.dtype == state.dtype and value.device == state.device
                for value in (command, action, previous)), "cost tensors must share dtype/device")
    velocity = _rotate_inverse(state[..., 3:7], state[..., 23:26])
    gravity = _rotate_inverse(state[..., 3:7], state.new_tensor([0., 0., -1.]).expand_as(velocity))
    # Match52's left-associated expression and operation creation order.  A
    # stack(...).sum(-1) would change float32 rounding and backward accumulation.
    vx = ((velocity[..., 0] - command[..., 0]) / .2) ** 2
    yaw = ((state[..., 28] - command[..., 2]) / .2) ** 2
    total = vx + yaw
    lateral = (velocity[..., 1] / .2) ** 2
    total = total + lateral
    tilt = 6 * (gravity[..., :2] ** 2).sum(-1)
    total = total + tilt
    height = 20 * (state[..., 2] - nominal_height) ** 2
    total = total + height
    smooth = .1 * ((action - previous) ** 2).mean(-1)
    total = total + smooth
    return torch.stack((vx, yaw, lateral, tilt, height, smooth), dim=-1), total


def cost_components(state, command, action, previous, nominal_height):
    """Return [...,6] terms: vx, yaw, lateral, tilt, height, action_smoothness.

    Their mathematical sum is cost; use cost() for52's precise left-associated
    floating-point summation rather than a potentially reordered tensor reduce.
    Inputs use one shared CPU dtype and physical units, as in control52.cost.
    """
    return _cost_parts_and_total(state, command, action, previous, nominal_height)[0]


def cost(state, command, action, previous, nominal_height):
    """Return the frozen52 local cost [...], retaining its summation order."""
    return _cost_parts_and_total(state, command, action, previous, nominal_height)[1]


def _cache(shaper, dtype):
    if shaper.state is None:
        return None
    return {"state": _tensor([shaper.state], torch.float64, clone=True),
            "history": _tensor(shaper.history[None], dtype, clone=True),
            "raw_history": _tensor(shaper.raw_history[None], dtype, clone=True)}


def _reference_step(current, history, cache):
    dtype = current.dtype
    raw_yaw, past_yaw = current[..., 8], history[..., 8]
    target = (raw_yaw * 1.5).clamp(-.25, .25)
    if cache is None:
        require(bool((past_yaw == raw_yaw.unsqueeze(-1)).all()), "reset requires constant prehistory")
        old_state = target.to(torch.float64)
        shaped_history = (past_yaw * 1.5).clamp(-.25, .25)
    else:
        require(torch.equal(past_yaw, cache["raw_history"]), "raw command history differs")
        old_state, shaped_history = cache["state"], cache["history"]
    new_state = REFERENCE_ALPHA * old_state + (1.0 - REFERENCE_ALPHA) * target.to(torch.float64)
    yaw = new_state.to(dtype)
    current_shaped = torch.cat((current[..., :8], yaw.unsqueeze(-1), current[..., 9:]), dim=-1)
    past_shaped = torch.cat((history[..., :8], shaped_history.unsqueeze(-1), history[..., 9:]), dim=-1)
    shaped = torch.cat((current_shaped, past_shaped.flatten(start_dim=-2)), dim=-1)
    updated = {"state": new_state,
               "history": torch.cat((shaped_history[..., 1:], yaw.unsqueeze(-1)), dim=-1),
               "raw_history": torch.cat((past_yaw[..., 1:], raw_yaw.unsqueeze(-1)), dim=-1)}
    return shaped, updated


def _observation(state, command, previous_action, previous_filtered):
    dtype = state.dtype
    state = state.to(torch.float64)
    command = command.to(torch.float64)
    previous = previous_filtered.to(torch.float64)
    action = previous_action.to(torch.float64)
    require(not bool((action.abs() > 1.).any()), "previous action must already be clipped")
    quat = state[..., 3:7]
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    require(not bool((norm < 1e-12).any()), "quaternion norm is zero")
    quat = quat / norm
    gravity_world = state.new_tensor([0., 0., -1.]).expand_as(quat[..., 1:])
    cross = torch.linalg.cross(quat[..., 1:], gravity_world, dim=-1)
    gravity = gravity_world - 2.0 * quat[..., :1] * cross + 2.0 * torch.linalg.cross(
        quat[..., 1:], cross, dim=-1)
    filtered = .97 * state[..., 26:29] + .03 * previous
    default_q = state.new_tensor([0., .8, -1.5, 0.] * 4)
    leg_mask = state.new_tensor([1., 1., 1., 0.] * 4)
    offset = (state[..., 7:23] - default_q) * leg_mask
    scaled_command = command.clamp(-1., 1.) * state.new_tensor([2., 2., .25])
    frame = torch.cat((filtered * .25, gravity, scaled_command, offset,
                       state[..., 29:45] * .05, action), dim=-1).to(dtype)
    return frame, filtered


def _residual_action(action, frame, command, theta):
    value = _tensor(theta, action.dtype)
    value = torch.broadcast_to(value, action.shape[:-1])
    error = ((command[..., 2] - frame[..., 2] / .25) / .3).clamp(-1., 1.)
    direction = action.new_tensor([0., 0., 0., 1., 0., 0., 0., -1.,
                                   0., 0., 0., 1., 0., 0., 0., -1.])
    return (action + (value * error).unsqueeze(-1) * direction).clamp(-1., 1.)


def _delay_step(issued, fifo):
    action = issued.clamp(-1., 1.)
    require(fifo.shape[:-2] == action.shape[:-1] and fifo.shape[-1] == 16
            and fifo.shape[-2] in (0, 1, 2, 3) and not bool((fifo.abs() > 1.).any()),
            "FIFO must hold0..3 clipped actions with matching batch shape")
    if fifo.shape[-2] == 0:
        return action, fifo.clone()
    return fifo[..., 0, :], torch.cat((fifo[..., 1:, :], action.unsqueeze(-2)), dim=-2)


def imaginary(model, actor, start, theta, *, return_trace=False, horizon=HORIZON):
    """One saved52/53 point -> scalar mean cost, recomputing actions every tick.

    dtype follows model.input_mean; actor floating weights/buffers must match.
    Copies promoted by .double() preserve every original weight value.  The
    point carries state45, frame57, history10x57, filtered3, last16, FIFO, shaper,
    commands[horizon+1,3], and nominal_height.  horizon defaults to the frozen32.

    return_trace=True returns (value, trace).  Graph-retaining trace tensors are
    costs[H], states[H,45], actions/applied_actions[H,16], cost_components[H,6],
    frames[H,57], actor_inputs[H,627], filtered[H,3] and shaper_state[H].  States
    are post-step; frames/filtered are pre-action; shaper_state is post-shaping.
    """
    require(type(horizon) is int and horizon > 0, "positive integer horizon required")
    dtype = _precision(model, actor)
    state = _tensor(start["state"][None], dtype, clone=True)
    frame = _tensor(start["frame"][None], dtype, clone=True)
    history = _tensor(start["history"][None], dtype, clone=True)
    filtered = _tensor(start["filtered"][None], torch.float64, clone=True)
    previous = _tensor(start["last"][None], dtype, clone=True)
    fifo = _tensor(start["fifo"][None], dtype, clone=True)
    require(state.shape == (1, 45) and frame.shape == (1, 57) and history.shape == (1, 10, 57)
            and filtered.shape == (1, 3) and previous.shape == (1, 16), "point shapes differ")
    require(len(start["commands"]) >= horizon + 1, "point command horizon is incomplete")
    cache = _cache(start["shaper"], dtype)
    terms = []
    trace = {key: [] for key in ("states", "actions", "applied_actions", "cost_components",
                                "frames", "actor_inputs", "filtered", "shaper_state")}
    for tick in range(horizon):
        command = _tensor(start["commands"][tick][None], dtype, clone=True)
        shaped, cache = _reference_step(frame, history, cache)
        issued = _residual_action(actor_forward(actor, shaped), frame, command, theta)
        applied, fifo = _delay_step(issued, fifo)
        history = torch.cat((history[..., 1:, :], frame.unsqueeze(-2)), dim=-2)
        state = model(state, applied)
        components, total = _cost_parts_and_total(state, command, issued, previous, start["nominal_height"])
        terms.append(total)
        if return_trace:
            for key, item in (("states", state), ("actions", issued), ("applied_actions", applied),
                              ("cost_components", components), ("frames", frame), ("actor_inputs", shaped),
                              ("filtered", filtered), ("shaper_state", cache["state"])):
                trace[key].append(item)
        previous = issued
        frame, filtered = _observation(state, _tensor(start["commands"][tick + 1][None], dtype, clone=True),
                                       issued, filtered)
    stacked = torch.stack(terms)
    value = stacked.mean()
    require(bool(torch.isfinite(value)), "nonfinite imagined cost")
    if not return_trace:
        return value
    packed = {key: torch.stack(items).squeeze(1) for key, items in trace.items()}
    packed["costs"] = stacked.squeeze(1)
    return value, packed


def gradients(model, actor, points, *, horizon=HORIZON):
    """Original52 epsilon=.001 and acceptance fields, using model arithmetic dtype."""
    dtype = _precision(model, actor)
    rows = []
    for start in points:
        theta = torch.tensor(0., dtype=dtype, device="cpu", requires_grad=True)
        value = imaginary(model, actor, start, theta, horizon=horizon)
        derivative, = torch.autograd.grad(value, theta)
        with torch.no_grad():
            plus = imaginary(model, actor, start, torch.tensor(FD_EPS, dtype=dtype, device="cpu"), horizon=horizon)
            minus = imaginary(model, actor, start, torch.tensor(-FD_EPS, dtype=dtype, device="cpu"), horizon=horizon)
            numerical = float((plus - minus) / (2 * FD_EPS))
        analytic = float(derivative)
        require(math.isfinite(analytic) and math.isfinite(numerical), "nonfinite policy gradient")
        rows.append({"seed": start["seed"], "tick": start["tick"], "cost": float(value.detach()),
                     "gradient": analytic, "model_central_difference": numerical,
                     "direction_identifiable": abs(analytic) >= 1e-5 and abs(numerical) >= 1e-5
                     and analytic * numerical > 0,
                     "autograd_agrees": abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4})
    return rows
