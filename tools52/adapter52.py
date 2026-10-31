"""Independent CPU interfaces for the bounded model-gradient experiment52.

Importing this module loads no numerical library or robot.  The actor weights
stay fixed while its input graph remains differentiable.  The learned 20 ms
transition model must cover the action-to-PD-to-physics response; this module
does not claim to reproduce PD or MuJoCo inside that model.

The state45 observation mapping assumes qpos=[free xyz, wxyz, joints16] and
qvel=[free linear3, body angular3, joints16].  The runner must verify these
indices and the gyro/quaternion sensor equivalence against the real model
before accepting any rollout.  A state45 vector alone is not an exact MuJoCo
snapshot: use capture/restore for physical counterfactual branches.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

COMPLETE25_SHA256 = "873cd14ff67e7d399472d97a94e082f7a97d068b00db414986443e50fcdb4f22"
REFERENCE_ALPHA = math.exp(-0.02 / 0.1)
_SNAPSHOT_EXCLUDED = {"model", "data", "_counts"}


def _torch():
    import torch
    return torch


def _tensor(value, name, tail):
    """Accept a floating CPU tensor/array, preserving an existing input graph."""
    torch = _torch()
    value = torch.as_tensor(value)
    if (value.device.type != "cpu" or not value.is_floating_point()
            or value.ndim < len(tail) or tuple(value.shape[-len(tail):]) != tuple(tail)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be finite floating CPU [...,{','.join(map(str, tail))}]")
    return value


def actor_from_state_dict(policy_state):
    """Build a fixed CPU actor from SB3 policy tensors; input [...,627] -> [...,16].

    Only the complete encoder, policy network and action head are selected.
    Critic/log_std tensors are irrelevant to deterministic inference.  The
    caller must bind checkpoint provenance when using this lower-level helper.
    """
    torch = _torch()
    nn = torch.nn

    class FixedActor(nn.Module):
        def __init__(self):
            super().__init__()
            def linear(incoming, outgoing):
                return nn.Linear(incoming, outgoing, device="cpu", dtype=torch.float32)
            self.encoder = nn.Sequential(linear(627, 128), nn.Tanh(), linear(128, 64),
                                         nn.Tanh(), linear(64, 3))
            self.policy_net = nn.Sequential(linear(60, 128), nn.Tanh(),
                                            linear(128, 128), nn.Tanh())
            self.action_head = linear(128, 16)

        def forward(self, observation):
            value = _tensor(observation, "actor observation", (627,)).to(torch.float32)
            velocity = self.encoder(value)
            latent = self.policy_net(torch.cat((value[..., :57], velocity), dim=-1))
            return self.action_head(latent).clamp(-1.0, 1.0)

    # Linear initialization is overwritten; do not consume the runner's RNG.
    with torch.random.fork_rng(devices=[]):
        actor = FixedActor()
    prefixes = {"mlp_extractor.encoder.": "encoder.",
                "mlp_extractor.policy_net.": "policy_net.", "action_net.": "action_head."}
    selected = {}
    for name, value in policy_state.items():
        for source, target in prefixes.items():
            if name.startswith(source):
                key = target + name[len(source):]
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                        or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
                    raise ValueError("invalid CPU float32 actor weight: " + name)
                selected[key] = value
                break
    actor.load_state_dict(selected, strict=True)
    actor.eval().requires_grad_(False)
    return actor


def load_actor(checkpoint, *, expected_sha256=COMPLETE25_SHA256):
    """Load only policy.pth with weights_only=True after checking the ZIP SHA.

    No SB3 algorithm, optimizer, pickled metadata or environment is loaded.
    The default digest binds the complete25 random final checkpoint.  A caller
    supplying another digest is responsible for a separate complete provenance
    gate; synthetic unit tests use that explicit facility.
    """
    path = Path(checkpoint)
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint must be a regular, non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("checkpoint SHA256 differs")
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count("policy.pth") != 1:
            raise ValueError("checkpoint must contain exactly one policy.pth")
        payload = archive.read("policy.pth")
    policy_state = _torch().load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(policy_state, dict):
        raise TypeError("policy.pth must be a tensor state dictionary")
    return actor_from_state_dict(policy_state)


@dataclass(frozen=True)
class Snapshot:
    """Reusable in-memory branch point; model identity and budget stay live."""

    model: object
    data: object
    env_type: type
    shaper_type: type | None
    python_state: dict


def capture(env, shaper=None):
    """Copy full MjData plus Python state/RNGs, excluding monotonic work counters.

    deepcopy the Python dictionaries together to retain RNG aliases such as
    _schedule_rng is _command_rng.  copy.copy(MjData) shares the immutable model
    but copies its integration, warmstart, control, diagnostic and derived data.
    No reset, forward, filter update, actor call or physics step is performed.
    """
    python_state = {key: value for key, value in vars(env).items()
                    if key not in _SNAPSHOT_EXCLUDED and not key.startswith("_fast_")}
    return Snapshot(env.model, copy.copy(env.data), type(env),
                    None if shaper is None else type(shaper),
                    copy.deepcopy({"env": python_state,
                                   "shaper": None if shaper is None else vars(shaper)}))


def restore(env, shaper, snapshot):
    """Restore a capture into the same model without refunding env._counts.

    A fresh data copy makes repeated restores independent.  Cached MjData views
    are discarded.  The runner's external counters, wall time and admission
    guard must also remain monotonic across every branch.
    """
    if (not isinstance(snapshot, Snapshot) or env.model is not snapshot.model
            or type(env) is not snapshot.env_type
            or (None if shaper is None else type(shaper)) is not snapshot.shaper_type):
        raise ValueError("snapshot model/environment/shaper identity differs")
    data = copy.copy(snapshot.data)
    state = copy.deepcopy(snapshot.python_state)
    model, counts = env.model, env._counts
    vars(env).clear()
    vars(env).update(state["env"])
    env.model, env.data, env._counts = model, data, counts
    if shaper is not None:
        vars(shaper).clear()
        vars(shaper).update(state["shaper"])


def build_observation(state, raw_command, previous_action, previous_filtered):
    """state [...,45], command [...,3], action [...,16], filter [...,3] -> frame/filter.

    Returns a float32 [...,57] frame and float64 [...,3] updated gyro filter,
    preserving gradients through casts, quaternion normalization and rotation.
    Joint positions use radians, joint velocities rad/s, commands m/s and rad/s.
    State slices must first pass the runner's real-sensor mapping check.
    """
    torch = _torch()
    state = _tensor(state, "state", (45,)).to(torch.float64)
    command = _tensor(raw_command, "command", (3,)).to(torch.float64)
    previous = _tensor(previous_filtered, "previous gyro filter", (3,)).to(torch.float64)
    action = _tensor(previous_action, "previous clipped action", (16,)).to(torch.float64)
    if not (state.shape[:-1] == command.shape[:-1] == previous.shape[:-1] == action.shape[:-1]):
        raise ValueError("observation components must have identical leading shapes")
    if bool((action.abs() > 1.0).any()):
        raise ValueError("previous action must already be clipped")
    quat = state[..., 3:7]
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    if bool((norm < 1e-12).any()):
        raise ValueError("quaternion norm is zero")
    quat = quat / norm
    world_gravity = state.new_tensor([0.0, 0.0, -1.0]).expand_as(quat[..., 1:])
    cross = torch.linalg.cross(quat[..., 1:], world_gravity, dim=-1)
    gravity = world_gravity - 2.0 * quat[..., :1] * cross + 2.0 * torch.linalg.cross(
        quat[..., 1:], cross, dim=-1)
    filtered = 0.97 * state[..., 26:29] + 0.03 * previous
    default_q = state.new_tensor([0.0, 0.8, -1.5, 0.0] * 4)
    leg_mask = state.new_tensor([1.0, 1.0, 1.0, 0.0] * 4)
    offset = (state[..., 7:23] - default_q) * leg_mask
    scaled_command = command.clamp(-1.0, 1.0) * state.new_tensor([2.0, 2.0, 0.25])
    frame = torch.cat((filtered * 0.25, gravity, scaled_command, offset,
                       state[..., 29:45] * 0.05, action), dim=-1).to(torch.float32)
    return frame, filtered


def advance_history(history, current):
    """Append current [...,57] to oldest-first history [...,10,57], dropping oldest."""
    torch = _torch()
    history = _tensor(history, "history", (10, 57)).to(torch.float32)
    current = _tensor(current, "current frame", (57,)).to(torch.float32)
    if history.shape[:-2] != current.shape[:-1]:
        raise ValueError("history and frame leading shapes differ")
    return torch.cat((history[..., 1:, :], current.unsqueeze(-2)), dim=-2)


def reference_step(raw_current, raw_history, cache=None):
    """Mirror33: [...,57] + [...,10,57] -> shaped [...,627] and next tensor cache.

    cache=None denotes reset with constant raw yaw prehistory.  Otherwise cache
    has float64 state [...], float32 history [...,10] (shaped yaw only), and
    raw_history [...,10].  Returned cache corresponds to *after* this actor
    input; its history must not be substituted into the current actor input.
    All differentiable paths, including the filter's temporal graph, are kept.
    """
    torch = _torch()
    current = _tensor(raw_current, "current frame", (57,)).to(torch.float32)
    history = _tensor(raw_history, "raw history", (10, 57)).to(torch.float32)
    if current.shape[:-1] != history.shape[:-2]:
        raise ValueError("current and history leading shapes differ")
    raw_yaw = current[..., 8]
    past_yaw = history[..., 8]
    target = (raw_yaw * 1.5).clamp(-0.25, 0.25)
    if cache is None:
        if not bool((past_yaw == raw_yaw.unsqueeze(-1)).all()):
            raise ValueError("reset requires constant prehistory")
        old_state = target.to(torch.float64)
        shaped_history = (past_yaw * 1.5).clamp(-0.25, 0.25)
    else:
        if set(cache) != {"state", "history", "raw_history"}:
            raise ValueError("reference cache fields differ")
        old_state = torch.as_tensor(cache["state"])
        shaped_history = _tensor(cache["history"], "shaped yaw history", (10,)).to(torch.float32)
        expected_raw = _tensor(cache["raw_history"], "previous raw yaw history", (10,)).to(torch.float32)
        if (old_state.shape != raw_yaw.shape or old_state.dtype != torch.float64
                or old_state.device.type != "cpu" or not bool(torch.isfinite(old_state).all())
                or shaped_history.shape != past_yaw.shape
                or not torch.equal(past_yaw, expected_raw)):
            raise ValueError("reference cache shape/state or raw history differs")
    new_state = REFERENCE_ALPHA * old_state + (1.0 - REFERENCE_ALPHA) * target.to(torch.float64)
    yaw = new_state.to(torch.float32)
    shaped_current = torch.cat((current[..., :8], yaw.unsqueeze(-1), current[..., 9:]), dim=-1)
    shaped_past = torch.cat((history[..., :8], shaped_history.unsqueeze(-1), history[..., 9:]), dim=-1)
    shaped = torch.cat((shaped_current, shaped_past.flatten(start_dim=-2)), dim=-1)
    updated = {"state": new_state,
               "history": torch.cat((shaped_history[..., 1:], yaw.unsqueeze(-1)), dim=-1),
               "raw_history": torch.cat((past_yaw[..., 1:], raw_yaw.unsqueeze(-1)), dim=-1)}
    return shaped, updated


def residual_action(fixed_action, current_obs, raw_command, theta):
    """Bounded yaw residual: [...,16], [...,57], [...,3], scalar/[...] -> [...,16].

    e=clip((raw wz - filtered gyro wz)/0.3,-1,1); add theta*e to wheels
    [3,7,11,15] with directions [+1,-1,+1,-1], then clip all actions to [-1,1].
    Uses the raw command, before reference shaping; leg actions are unchanged.
    """
    torch = _torch()
    action = _tensor(fixed_action, "fixed action", (16,)).to(torch.float32)
    observation = _tensor(current_obs, "current frame", (57,)).to(torch.float32)
    command = _tensor(raw_command, "raw command", (3,)).to(torch.float32)
    if not (action.shape[:-1] == observation.shape[:-1] == command.shape[:-1]):
        raise ValueError("residual component leading shapes differ")
    value = torch.as_tensor(theta)
    if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
        raise ValueError("theta must be finite and CPU")
    value = value.to(torch.float32)
    try:
        value = torch.broadcast_to(value, action.shape[:-1])
    except RuntimeError as exc:
        raise ValueError("theta must be scalar or match batch shape") from exc
    error = ((command[..., 2] - observation[..., 2] / 0.25) / 0.3).clamp(-1.0, 1.0)
    direction = action.new_tensor([0., 0., 0., 1., 0., 0., 0., -1.,
                                   0., 0., 0., 1., 0., 0., 0., -1.])
    return (action + (value * error).unsqueeze(-1) * direction).clamp(-1.0, 1.0)


def delay_step(issued_action, fifo):
    """Clip issued [...,16], apply FIFO [...,delay,16] (delay 0..3), return action/FIFO.

    This is the policy-tick delay only.  It does not model native PD substeps.
    Zero-delay execution returns the current clipped action and an empty FIFO.
    """
    torch = _torch()
    action = _tensor(issued_action, "issued action", (16,)).to(torch.float32).clamp(-1., 1.)
    queue = _tensor(fifo, "delay FIFO", (16,)).to(torch.float32)
    if (queue.ndim < 2 or queue.shape[:-2] != action.shape[:-1]
            or queue.shape[-2] not in (0, 1, 2, 3) or bool((queue.abs() > 1.).any())):
        raise ValueError("FIFO must contain 0..3 clipped actions with matching batch shape")
    if queue.shape[-2] == 0:
        return action, queue.clone()
    return queue[..., 0, :], torch.cat((queue[..., 1:, :], action.unsqueeze(-2)), dim=-2)
