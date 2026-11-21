"""Cold D1 model learning and closed-loop pathwise gradients on a single CPU."""
from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools52 import adapter52, dynamics52
from tools56.fit56 import mapping_sha256
from tools91.model91 import TerrainDynamics91, fit_ridge


WORLD_UPDATES, WORLD_BATCH, FIT_HORIZON = 100, 32, 8
POLICY_UPDATES, POLICY_BATCH, POLICY_HORIZON = 4, 8, 32
LR, CLIP = 3e-4, 1.
FD_FRACTION = 0.25  # central difference at ±0.25 of the actual proposed Adam displacement
WHEELS = (3, 7, 11, 15)
LEGS = tuple(i for i in range(16) if i not in WHEELS)
ROOT = Path(__file__).resolve().parents[1]
require = dynamics52.require


class LearnFailure(RuntimeError):
    """Carries completed work and any predicted endpoints before a failure."""

    def __init__(self, reason, receipt, partial_states=None):
        super().__init__(reason)
        self.receipt = receipt
        self.partial_states = partial_states
        self.partial_model = self.partial_actor = self.partial_optimizer = None


def _work():
    return {"imagined_steps": 0, "imagined_steps_attempted": 0}


def _step(world, states, actions, terrain, meter):
    meter["imagined_steps_attempted"] += len(states)
    predicted = world.step(states, actions, terrain)
    meter["imagined_steps"] += len(states)
    return predicted


def _round_arrays(traces):
    """Include reset→first endpoint and the terminal transition without crossing resets."""
    require(len(traces) == 12, "each collection round requires twelve finished physical episodes")
    seeds, lengths, states, actions, targets, ids, ticks, ends = [], [], [], [], [], [], [], []
    for i, trace in enumerate(traces):
        s, applied, issued = (np.asarray(trace[k]) for k in ("state45", "applied16", "issued16"))
        n = len(applied)
        require(0 < n <= 50 and applied.shape == issued.shape == (n, 16) and s.shape == (n + 1, 45),
                "a collected episode must have N applied and N+1 states")
        terminal, truncated = (np.asarray(trace[k]) for k in ("terminated", "truncated"))
        require(terminal.shape == truncated.shape == (n,) and terminal.dtype == truncated.dtype == bool,
                "missing native per-interval end flags")
        finished = terminal | truncated
        require(not finished[:-1].any() and (n == 50 or finished[-1]), "unfinished or spliced collection")
        require(not applied[0].any() and np.array_equal(applied[1:], issued[:-1]),
                "collected actions violate the one-tick FIFO")
        require(np.isfinite(issued).all() and (np.abs(issued) <= 1.).all(), "invalid issued action")
        seeds.append(int(trace["reset_seed"]))
        lengths.append(n)
        states.append(s[:-1])
        actions.append(applied)
        targets.append(s[1:])
        ids.append(np.full(n, i, dtype=np.int64))
        ticks.append(np.arange(n, dtype=np.int64))
        ends.append(np.arange(n) == n - 1)
    require(len(set(seeds)) == len(seeds), "collection reset seeds must be distinct")
    arrays = {k: np.concatenate(v) for k, v in
              (("s", states), ("a", actions), ("target", targets), ("episode", ids),
               ("tick", ticks), ("done", ends))}
    windows = torch.from_numpy(dynamics52.window_indices(arrays, FIT_HORIZON))
    arrays["terrain"] = np.ones(len(arrays["s"]), dtype=np.int64)
    return arrays, windows, {"seeds": seeds, "episode_lengths": lengths,
                             "transitions": sum(lengths), "windows": len(windows),
                             "sha256": mapping_sha256(arrays)}


def fit_world(train_rounds, seed, previous=None):
    """Fit from native94 collections only. Return a frozen world, Adam and an audit."""
    meter = _work()
    receipt = {"status": "PARTIAL", "updates": 0, "updates_completed": 0, "updates_attempted": 0,
               "seed": int(seed), "batch_size": WORLD_BATCH, "horizon": FIT_HORIZON, "round_data": [],
               "losses": [], "learning_rate": LR, "gradient_clip": CLIP}
    model = optimizer = None
    try:
        require(len(train_rounds) == (1 if previous is None else 2), "unexpected cold world fit round")
        groups = []
        used = set()
        for traces in train_rounds:
            arrays, windows, details = _round_arrays(traces)
            require(used.isdisjoint(details["seeds"]), "training rounds share reset seeds")
            used.update(details["seeds"])
            receipt["round_data"].append(details)
            tensors = {k: torch.as_tensor(arrays[k], dtype=torch.float32) for k in ("s", "a", "target")}
            groups.append((tensors, windows))
        if previous is None:
            ridge = fit_ridge(arrays)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = TerrainDynamics91(ridge)
            optimizer = torch.optim.Adam(model.residual.parameters(), lr=LR)
        else:
            model, optimizer = copy.deepcopy(previous)
            require(isinstance(model, TerrainDynamics91) and isinstance(optimizer, torch.optim.Adam),
                    "previous world model and its Adam optimizer required")
            require({id(p) for g in optimizer.param_groups for p in g["params"]} ==
                    {id(p) for p in model.residual.parameters()}, "previous Adam is bound to a different world")
        model.train().requires_grad_(True)
        receipt["initial_model_sha256"] = mapping_sha256(model.state_dict())
        receipt["ridge_and_stats_sha256"] = mapping_sha256(dict(model.named_buffers()))
        receipt["rows_per_group_per_update"] = per_group = WORLD_BATCH // len(groups)
        receipt["batch_seed"] = batch_seed = int(seed) + 100_000 * len(groups)
        sampler = torch.Generator(device="cpu").manual_seed(batch_seed)
        for i in range(WORLD_UPDATES):
            receipt["updates_attempted"] = i + 1
            batch_s, batch_a, batch_target = [], [], []
            for tensors, windows in groups:
                rows = windows[torch.randint(len(windows), (per_group,), generator=sampler)]
                batch_s.append(tensors["s"][rows[:, 0]])
                batch_a.append(tensors["a"][rows])
                batch_target.append(tensors["target"][rows])
            state, applied, target = torch.cat(batch_s), torch.cat(batch_a), torch.cat(batch_target)
            terrain = torch.ones(len(state), dtype=torch.long)
            predictions = []
            for action in applied.unbind(1):
                state = _step(model, state, action, terrain, meter)
                predictions.append(state)
            loss = dynamics52.group_loss(torch.stack(predictions, dim=1), target)
            require(bool(loss.isfinite()), "nonfinite H8 world loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.residual.parameters(), CLIP, error_if_nonfinite=True)
            optimizer.step()
            receipt["updates"] = receipt["updates_completed"] = i + 1
            receipt["losses"].append({"loss": float(loss.detach()), "gradient_norm": float(norm)})
            require(all(bool(p.isfinite().all()) for p in model.parameters()), "nonfinite fitted world")
        require(receipt["ridge_and_stats_sha256"] == mapping_sha256(dict(model.named_buffers())),
                "ridge or first-round normalization changed")
        optimizer.zero_grad(set_to_none=True)
        model.eval().requires_grad_(False)
        receipt.update(meter, status="COMPLETE", final_model_sha256=mapping_sha256(model.state_dict()))
        return model, optimizer, receipt
    except Exception as exc:
        receipt.update(meter, error=f"{type(exc).__name__}: {exc}")
        failure = LearnFailure(receipt["error"], receipt)
        failure.partial_model, failure.partial_optimizer = model, optimizer
        raise failure from exc


@lru_cache(maxsize=1)
def _ray_meter():
    # One geometry for all serial rays. Never step the simulator here.
    from tools87.terrain87 import load_model
    from tools91.rollout91 import _PredictedRanges

    return _PredictedRanges(load_model("up"))


def _measure_packet(state):
    import mujoco

    geometry = _ray_meter()
    geometry.data.qpos[:] = state[:23]
    mujoco.mj_kinematics(geometry.model, geometry.data)
    return geometry._measure()


def _due(tick):
    return tick == 0 or (tick * 6) // 50 != ((tick - 1) * 6) // 50


class _Ranges:
    """A private delay queue for each branch, restored at its observation boundary."""

    def __init__(self, trace, tick):
        self.visible = np.array(trace["obs643"][tick, 627:], dtype=np.float32, copy=True)
        self.pending = None
        # At most one 6-Hz capture can be in the 2-tick delivery queue.
        for source in range(max(0, tick - 1), tick + 1):
            require(bool(trace["ray_due"][source]) == _due(source), "range capture clock differs at branch")
            if _due(source):
                self.pending = (source + 2, np.array(trace["ray_raw16"][source], dtype=np.float32, copy=True))

    def at(self, tick, state):
        if _due(tick):
            self.pending = (tick + 2, _measure_packet(state))
        if self.pending is not None and self.pending[0] == tick:
            self.visible = self.pending[1]
            self.pending = None
        return self.visible


@lru_cache(maxsize=1)
def _fall_limits():
    pilot = json.loads((ROOT / "training/configs/pilot03d.json").read_text())
    geom = json.loads((ROOT / "training/configs/diagnostic03b_zero.json").read_text())["static_derivation"]
    require(geom["model_xml_sha256"] == pilot["robot_xml_sha256"], "fall limits for another robot")
    return (geom["nominal_root_height_m"] - pilot["termination"]["base_height_below_nominal_by_m"],
            math.cos(pilot["termination"]["tilt_above_rad"]))


def _stack_at(starts, field, dtype):
    return torch.as_tensor(np.stack([t[field][k] for t, k in starts]), dtype=dtype)


def _roll(world, actor, starts, horizon, wheel_offset, meter):
    """Predict a batch from its cut; each branch sees its own imagined sensor values."""
    states = []
    try:
        require(horizon > 0 and all(k in (0, 18) and k + horizon <= 50 for _, k in starts),
                "imagined start outside the stationary command")
        state = _stack_at(starts, "state45", torch.float32)
        initial_world_x = torch.tensor([t["state45"][0, 0] for t, _ in starts], dtype=torch.float64)
        initial_obs = _stack_at(starts, "obs643", torch.float32)
        current = initial_obs[:, :57]
        history = initial_obs[:, 57:627].reshape(-1, 10, 57)
        fifo = current[:, 41:57]
        filtered = _stack_at(starts, "filtered3", torch.float64)
        sensors = [_Ranges(t, k) for t, k in starts]
        terrain = torch.ones(len(starts), dtype=torch.long)
        command = torch.zeros((len(starts), 3), dtype=torch.float64)
        direction = state.new_zeros(16)
        direction[list(WHEELS)] = 1. / 5.  # normalized target per rad/s
        height_limit, upright_cos = _fall_limits()
        states.append(state)
        for h in range(horizon):
            if h == 0:
                features = initial_obs[:, 627:]
            else:
                # First-return geometry, visibility switches and delivery are intentionally
                # nondifferentiated. Numerical probes reevaluate the forward path.
                positions = state.detach().numpy()
                features = torch.from_numpy(np.stack([sensor.at(k + h, pos)
                    for sensor, (_, k), pos in zip(sensors, starts, positions, strict=True)]))
            actor_input = torch.cat((current, history.flatten(1), features), dim=1)
            issued = (actor(actor_input) + direction * wheel_offset).clamp(-1., 1.)
            require(issued.shape == fifo.shape and bool(issued.isfinite().all()), "invalid predicted policy action")
            state = _step(world, state, fifo, terrain, meter)
            require(state.shape == (len(starts), 45) and bool(state.isfinite().all()), "nonfinite model endpoint")
            states.append(state)
            norms = torch.linalg.vector_norm(state[:, 3:7], dim=1)
            require(bool(((norms - 1.).abs() <= 1e-4).all()), "invalid predicted quaternion")
            upright = 1. - 2. * (state[:, 4].square() + state[:, 5].square())
            require(bool(((state[:, 2] >= height_limit) & (upright >= upright_cos)).all()), "predicted fall")
            if h + 1 < horizon:
                history = adapter52.advance_history(history, current)
                current, filtered = adapter52.build_observation(state, command, issued, filtered)
                fifo = issued
        stacked = torch.stack(states, dim=1)
        return stacked, initial_world_x
    except Exception as exc:
        partial = torch.stack(states, dim=1).detach().numpy().copy() if states else None
        raise LearnFailure(f"{type(exc).__name__}: {exc}", dict(meter), partial) from exc


def _reward(states, initial_x, horizon):
    return -(states[:, 1:horizon + 1, 0].double() - initial_x[:, None]).abs().mean()


def probe(world, actor, trace, offset, *, derivative=False):
    """Predict 50 endpoints. A derivative probe returns dR32/d(common wheel rad/s), in metres."""
    meter = _work()
    states = None
    try:
        require(math.isfinite(offset) and (not derivative or offset == 0.), "derivative requires zero wheel offset")
        w = torch.tensor(float(offset), dtype=torch.float32, requires_grad=derivative)
        with torch.set_grad_enabled(derivative):
            states, origin = _roll(world, actor, [(trace, 0)], 50, w, meter)
            gradient = None
            if derivative:
                r = _reward(states, origin, POLICY_HORIZON)
                grad = torch.autograd.grad(r, w, allow_unused=True)[0] if r.requires_grad else None
                gradient = 0. if grad is None else float(grad.detach())
                require(math.isfinite(gradient), "nonfinite H32 wheel-offset derivative")
        return states[0].detach().numpy().copy(), gradient
    except Exception as exc:
        if isinstance(exc, LearnFailure):
            partial = exc.partial_states
        else:
            partial = states.detach().numpy() if states is not None else None
        partial = partial[0].copy() if partial is not None else np.empty((0, 45), dtype=np.float32)
        error = f"{type(exc).__name__}: {exc}"
        raise LearnFailure(error, {**meter, "status": "PARTIAL", "updates": 0, "updates_completed": 0,
                                   "updates_attempted": 0, "error": error}, partial) from exc


def _parameter_norms(values):
    head, bias = values["action_head.weight"], values["action_head.bias"]
    def rows_norm(rows):
        data = torch.cat((head[list(rows)].flatten(), bias[list(rows)]))
        return float(torch.linalg.vector_norm(data.double()))
    return {"leg_output": rows_norm(LEGS), "wheel_output": rows_norm(WHEELS),
            "range_input_columns": float(torch.linalg.vector_norm(values["input_layer.weight"][:, 627:].double()))}


def _sample_starts(traces, sampler):
    require(len(traces) == 12, "policy updates require the twelve same-round training traces")
    # Balance the reset and late cut where possible; never branch at an actual terminal endpoint.
    late = [i for i, trace in enumerate(traces) if len(trace["state45"]) > 19]
    late_count = min(POLICY_BATCH // 2, len(late))
    first = sampler.choice(len(traces), POLICY_BATCH - late_count, replace=False)
    second = sampler.choice(late, late_count, replace=False)
    return [(traces[i], 0) for i in first] + [(traces[i], 18) for i in second]


def _direction_check(world, actor, optimizer, starts, raw_loss_grad, fd_meter):
    # Adam moments, clipping and preconditioning can rotate an update away from -grad.
    # The finite difference therefore follows the candidate Adam step, including prior-round moments.
    candidate, trial_adam = copy.deepcopy((actor, optimizer))
    for new, old in zip(candidate.parameters(), actor.parameters(), strict=True):
        new.grad = old.grad.detach().clone()
    trial_adam.step()
    base = dict(actor.named_parameters())
    delta = {k: p.detach() - base[k].detach() for k, p in candidate.named_parameters()}
    analytic = -sum(float((raw_loss_grad[k].double() * d.double()).sum()) for k, d in delta.items())
    values = []
    with torch.no_grad():
        for alpha in (-FD_FRACTION, FD_FRACTION):
            for k, parameter in candidate.named_parameters():
                parameter.copy_(base[k] + alpha * delta[k])
            states, origin = _roll(world, candidate, starts, POLICY_HORIZON, 0., fd_meter)
            values.append(float(_reward(states, origin, POLICY_HORIZON)))
    difference = (values[1] - values[0]) / (2. * FD_FRACTION)
    magnitude = max(abs(analytic), abs(difference))
    # A numerical floor is required before accepting signs of float32 dynamics predictions.
    # This is a return-difference floor (metres); the derivatives are per whole Adam step.
    floor = max(1e-8, 8. * torch.finfo(torch.float32).eps * max(abs(v) for v in values))
    identifiable = abs(values[1] - values[0]) > floor and abs(analytic) * (2 * FD_FRACTION) > floor
    finite = all(math.isfinite(v) for v in (analytic, difference, *values))
    same_sign = analytic * difference > 0.
    relative = abs(analytic - difference) / magnitude if magnitude else None
    return {"passed": finite and identifiable and same_sign and relative <= .25,
            "identifiable": identifiable, "same_sign": same_sign, "epsilon_in_adam_steps": FD_FRACTION,
            "perturbed_returns_m": values, "required_return_difference_m": floor,
            "analytic_m_per_step": analytic, "finite_difference_m_per_step": difference,
            "relative_gap": relative, "candidate_direction_norms": _parameter_norms(delta)}


def update_actor(world, actor, trace_list, seed, previous_optimizer=None):
    """Four model-gradient steps on a fresh copy; the incoming policy and world stay sealed."""
    train_steps, fd_steps = _work(), _work()
    receipt = {"status": "PARTIAL", "updates": 0, "updates_attempted": 0, "updates_completed": 0,
               "gradients": [], "starts": [], "direction_check": None, "batch_size": POLICY_BATCH,
               "horizon": POLICY_HORIZON, "learning_rate": LR, "clip_norm": CLIP,
               "seed": int(seed), "pose_to_ray_gradient": "stopped", "terminal_value": 0.}
    candidate = optimizer = None

    def finish(status):
        receipt.update(status=status, optimization_work=dict(train_steps), direction_work=dict(fd_steps),
                       imagined_steps=train_steps["imagined_steps"] + fd_steps["imagined_steps"],
                       imagined_steps_attempted=train_steps["imagined_steps_attempted"]
                       + fd_steps["imagined_steps_attempted"])

    try:
        require(not any(p.requires_grad for p in world.parameters()), "freeze the world before actor learning")
        world_sha = mapping_sha256(world.state_dict())
        candidate, optimizer = copy.deepcopy((actor, previous_optimizer))
        candidate.train().requires_grad_(True)
        if optimizer is None:
            optimizer = torch.optim.Adam(candidate.parameters(), lr=LR)
        require(isinstance(optimizer, torch.optim.Adam) and
                {id(p) for g in optimizer.param_groups for p in g["params"]} ==
                {id(p) for p in candidate.parameters()}, "Adam must belong to the supplied policy")
        anchor = {k: p.detach().clone() for k, p in candidate.named_parameters()}
        receipt["initial_policy_sha256"] = mapping_sha256(candidate.state_dict())
        receipt["batch_seed"] = batch_seed = int(seed) + (200_000 if previous_optimizer is None else 300_000)
        sampler = np.random.default_rng(batch_seed)
        gate_failed = False
        for i in range(POLICY_UPDATES):
            receipt["updates_attempted"] = i + 1
            starts = _sample_starts(trace_list, sampler)
            receipt["starts"].append([{"reset_seed": int(t["reset_seed"]), "tick": k} for t, k in starts])
            optimizer.zero_grad(set_to_none=True)
            states, origin = _roll(world, candidate, starts, POLICY_HORIZON, 0., train_steps)
            loss = -_reward(states, origin, POLICY_HORIZON)
            require(bool(loss.isfinite()), "nonfinite imagined return")
            loss.backward()
            raw = {}
            for key, p in candidate.named_parameters():
                require(p.grad is not None and bool(p.grad.isfinite().all()), f"invalid gradient in {key}")
                raw[key] = p.grad.detach().clone()
            norm = nn.utils.clip_grad_norm_(candidate.parameters(), CLIP, error_if_nonfinite=True)
            receipt["gradients"].append({"imagined_return_m": -float(loss.detach()),
                                         "preclip_total_norm": float(norm), **_parameter_norms(raw)})
            if i == 0:
                check = _direction_check(world, candidate, optimizer, starts, raw, fd_steps)
                receipt["direction_check"] = check
                if not check["passed"]:
                    gate_failed = True
                    break
            optimizer.step()
            receipt["updates"] = receipt["updates_completed"] = i + 1
            require(all(bool(p.isfinite().all()) for p in candidate.parameters()), "nonfinite updated policy")
        optimizer.zero_grad(set_to_none=True)
        candidate.eval().requires_grad_(False)
        receipt["final_policy_sha256"] = mapping_sha256(candidate.state_dict())
        delta = {k: p.detach() - anchor[k] for k, p in candidate.named_parameters()}
        changes = receipt["parameter_changes"] = _parameter_norms(delta)
        receipt["frozen_world_sha256"] = mapping_sha256(world.state_dict())
        require(world_sha == receipt["frozen_world_sha256"], "actor learning changed the world model")
        if gate_failed:
            finish("FAILED_DIRECTION")
        elif min(changes["wheel_output"], changes["leg_output"]) == 0.:
            finish("FAILED_PARAMETER_CHANGE")
        else:
            finish("COMPLETE")
        return candidate, optimizer, receipt
    except Exception as exc:
        finish("PARTIAL")
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        failure = LearnFailure(receipt["error"], receipt, getattr(exc, "partial_states", None))
        failure.partial_actor, failure.partial_optimizer = candidate, optimizer
        raise failure from exc
