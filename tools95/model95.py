"""Sealed, one-step Adam directions for the D1 cold policies. No policy is trained."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tools56.fit56 import mapping_sha256
from tools91.model91 import TerrainDynamics91
from tools94 import learn94
from tools94.actor94 import make_actor
from tools94.common94 import ROOT, SEEDS, digest, fit_reset, read, seed_dir, write


HORIZON = 32
ALPHAS = (-1., -.5, 0., .5, 1.)
SUMMARY = ROOT / "reports/cold_start94.json"
SUMMARY_SHA256 = "a6e770eec5cda66cbc9614182ee19afaa02b1e89c08f1797df3ddecb92108872"
PROPOSAL = ROOT / "docs/supermimic_policy_direction95_proposal.md"
START_SHAPES = {"state45": (45,), "obs643": (643,), "filtered3": (3,),
                "ray_due": (), "ray_raw16": (16,)}


def _require(test, message):
    if not test:
        raise ValueError(message)


def _reset_only(trace):
    """Copy only the physical reset and its pending capture. Future labels cannot enter rollout."""
    start = {}
    for key, shape in START_SHAPES.items():
        sample = np.asarray(trace[key][0])
        _require(sample.shape == shape and np.isfinite(sample).all(), f"invalid reset {key}")
        start[key] = sample[np.newaxis].copy()
    _require(bool(start["ray_due"][0]), "reset lacks its capture at tick zero")
    return start


def _sources(seed):
    _require(seed in SEEDS, "unregistered cold-policy seed")
    _require(digest(SUMMARY) == SUMMARY_SHA256, "experiment 94 summary changed")
    public = next(row for row in read(SUMMARY)["seeds"] if row["seed"] == seed)
    root = seed_dir(seed)
    report_file = root / "report.json"
    _require(digest(report_file) == public["raw_report_sha256"], "raw seed receipt changed")
    report = read(report_file)
    _require(report["seed"] == seed and report["totals"]["policy_updates"] == 0 and
             len(report["rounds"]) == 1, "expected the original, unupdated cold policy")
    round_one = report["rounds"][0]
    fit = round_one["world_fit"]
    _require(fit["status"] == "COMPLETE" and fit["seed"] == seed and
             fit["updates_completed"] == 100, "expected the completed first fit")
    for name, file_hash in report["sources"].items():
        _require(digest(ROOT / name) == file_hash, f"frozen source changed: {name}")
    initial = report["initial"]
    model = round_one["world_checkpoint"]
    _require(initial["weights_sha256"] == round_one["collector_weights_sha256"] and
             model["weights_sha256"] == fit["final_model_sha256"], "collector or fitted weights changed")
    for path, item in ((root / "initial.pt", initial), (root / "round1/world.pt", model)):
        _require(path.name == item["file"] and digest(path) == item["sha256"],
                 f"checkpoint file identity differs: {path}")
    inputs = read(root / "round1/training_inputs.json")
    _require(inputs == round_one["fit"] and len(inputs) == 12, "fit roster changed")
    selected = []
    for j, row in enumerate(inputs[:3]):
        reset = fit_reset(seed, 0, j)
        trace_file = root / f"round1/train_{reset}/trace.npz"
        _require(row["reset"] == reset and digest(trace_file) == row["sha256"],
                 f"fit trace identity differs: {reset}")
        selected.append({"reset": reset, "path": str(trace_file.relative_to(ROOT)),
                         "sha256": row["sha256"], "tick": 0})
    return {"summary94_sha256": SUMMARY_SHA256, "seed_report_sha256": public["raw_report_sha256"],
            "sources94": report["sources"], "proposal95_sha256": digest(PROPOSAL),
            "model95_source_sha256": digest(__file__), "initial": initial,
            "world": model, "starts": selected}


def _load_parents(seed, sources):
    root = seed_dir(seed)
    actor = make_actor(seed)
    # Module constructors initialize temporary layers. Preserve the caller's CPU RNG.
    with torch.random.fork_rng(devices=[]):
        world = TerrainDynamics91()
    for net, path, source in ((actor, root / "initial.pt", sources["initial"]),
                              (world, root / "round1/world.pt", sources["world"])):
        data = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(data["network"], strict=True)
        _require(mapping_sha256(net.state_dict()) == source["weights_sha256"],
                 f"checkpoint tensor identity differs: {path}")
        net.eval().requires_grad_(False)
    return actor, world


def _norm(values):
    return float(torch.linalg.vector_norm(torch.cat([p.detach().double().flatten()
                                                    for p in values.values()])))


def _dot(left, right):
    return sum(float((left[k].detach().double() * right[k].detach().double()).sum()) for k in left)


def _meter(receipt, meter):
    receipt["attempted_model_steps"] = meter["imagined_steps_attempted"]
    receipt["completed_model_steps"] = meter["imagined_steps"]


def construct(seed, dest):
    """Write a single immutable direction receipt in a new or already empty directory."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"direction directory is occupied: {dest}")
    # Exclusive create is also the lock if two callers race on an empty directory.
    write(dest / "started.json", {"seed": seed, "horizon": HORIZON,
                                  "max_model_steps": 3 * HORIZON})
    meter = learn94._work()
    receipt = {"seed": seed, "status": "PARTIAL", "horizon": HORIZON,
               "max_model_steps": 3 * HORIZON, "attempted_model_steps": 0, "completed_model_steps": 0,
               "backward_attempted": 0, "backward_completed": 0,
               "shadow_adam_attempted": 0, "shadow_adam_completed": 0,
               "policy_updates": 0, "world_updates": 0, "ppo_updates": 0,
               "learning_rate": learn94.LR, "clip_norm": learn94.CLIP, "pose_to_ray_gradient": "stopped"}
    try:
        sources = receipt["sources"] = _sources(seed)
        actor, world = _load_parents(seed, sources)
        starts = []
        for source in sources["starts"]:
            with np.load(ROOT / source["path"], allow_pickle=False) as archive:
                _require(int(archive["reset_seed"]) == source["reset"], "wrong trace reset")
                starts.append((_reset_only(archive), 0))
        receipt["reset_tensors_sha256"] = [mapping_sha256(start) for start, _ in starts]
        shadow = copy.deepcopy(actor).requires_grad_(True)
        optimizer = torch.optim.Adam(shadow.parameters(), lr=learn94.LR)
        with torch.enable_grad():
            states, origin = learn94._roll(world, shadow, starts, HORIZON, 0., meter)
            reward = learn94._reward(states, origin, HORIZON)
            _require(bool(reward.isfinite()), "nonfinite development return")
            receipt["development_return_m"] = float(reward.detach())
            receipt["backward_attempted"] = 1
            (-reward).backward()
            receipt["backward_completed"] = 1
        raw = {}
        for name, p in shadow.named_parameters():
            _require(p.grad is not None and bool(p.grad.isfinite().all()), f"invalid loss gradient: {name}")
            raw[name] = p.grad.detach().clone()
        receipt["loss_gradient_sha256"] = mapping_sha256(raw)
        receipt["gradient_norms"] = learn94._parameter_norms(raw)
        receipt["preclip_norm"] = float(torch.nn.utils.clip_grad_norm_(
            shadow.parameters(), learn94.CLIP, error_if_nonfinite=True))
        receipt["shadow_adam_attempted"] = 1
        optimizer.step()
        receipt["shadow_adam_completed"] = 1
        base = dict(actor.named_parameters())
        delta = {k: p.detach() - base[k] for k, p in shadow.named_parameters()}
        _require(all(bool(v.isfinite().all()) for v in delta.values()), "nonfinite shadow Adam displacement")
        receipt["delta_norm"] = _norm(delta)
        receipt["delta_norms"] = learn94._parameter_norms(delta)
        receipt["development_directional_derivative"] = -_dot(raw, delta)
        receipt["float32_recomposition_max_abs"] = max(float(
            (base[k] + delta[k] - p.detach()).abs().max()) for k, p in shadow.named_parameters())
        _require(mapping_sha256(actor.state_dict()) == sources["initial"]["weights_sha256"] and
                 mapping_sha256(world.state_dict()) == sources["world"]["weights_sha256"],
                 "a parent changed during the shadow calculation")
        _require(receipt["delta_norm"] > 0., "zero representable Adam displacement")
        with (dest / "delta.pt").open("xb") as stream:
            torch.save(delta, stream)
        receipt["delta_file_sha256"] = digest(dest / "delta.pt")
        receipt["delta_tensors_sha256"] = mapping_sha256(delta)
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        prefix = getattr(exc, "partial_states", None)
        if prefix is not None:
            with (dest / "partial_states.npz").open("xb") as stream:
                np.savez_compressed(stream, state45=prefix)
            receipt["partial_states_sha256"] = digest(dest / "partial_states.npz")
            receipt["partial_endpoint_ticks"] = max(0, prefix.shape[1] - 1)
    _meter(receipt, meter)
    write(dest / "receipt.json", receipt)
    return receipt


@dataclass(frozen=True)
class SealedDirection:
    seed: int
    actor: torch.nn.Module
    world: torch.nn.Module
    delta: dict


def load_sealed(seed, dest):
    dest = Path(dest)
    receipt = read(dest / "receipt.json")
    _require(receipt["status"] == "COMPLETE" and receipt["seed"] == seed, "direction was not sealed")
    sources = _sources(seed)
    _require(receipt["sources"] == sources, "sealed input or implementation identity differs")
    actor, world = _load_parents(seed, sources)
    path = dest / "delta.pt"
    _require(digest(path) == receipt["delta_file_sha256"], "sealed direction file changed")
    delta = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(actor.named_parameters())
    _require(delta.keys() == params.keys() and mapping_sha256(delta) == receipt["delta_tensors_sha256"],
             "sealed direction tensor identity differs")
    for k, d in delta.items():
        _require(d.shape == params[k].shape and d.dtype == params[k].dtype and bool(d.isfinite().all()),
                 f"invalid displacement for {k}")
    return SealedDirection(seed, actor, world, delta)


def variant(bundle, alpha):
    """Independent float32 actor at the registered scale along the sealed displacement."""
    _require(alpha in ALPHAS, "unregistered parameter scale")
    actor = copy.deepcopy(bundle.actor).eval().requires_grad_(False)
    if alpha != 0.:
        with torch.no_grad():
            for name, p in actor.named_parameters():
                p.copy_(p + float(alpha) * bundle.delta[name])
    return actor


def predict_arm(bundle, trace, alpha):
    """Use a single H32 forward pass; alpha=0 also returns the held-out analytic derivative."""
    meter = learn94._work()
    out = {"return_m": None, "directional_derivative": None, "ticks": 0, "safe": False,
           "backward_attempted": 0, "backward_completed": 0}
    endpoints = None
    try:
        start = _reset_only(trace)
        derivative = alpha == 0.
        actor = variant(bundle, alpha).requires_grad_(derivative)
        with torch.set_grad_enabled(derivative):
            states, origin = learn94._roll(bundle.world, actor, [(start, 0)], HORIZON, 0., meter)
            endpoints = states[0].detach().numpy()
            reward = learn94._reward(states, origin, HORIZON)
            _require(bool(reward.isfinite()), "nonfinite held-out return")
            out["return_m"] = float(reward.detach())
            if derivative:
                params = dict(actor.named_parameters())
                out["backward_attempted"] = 1
                grads = torch.autograd.grad(reward, tuple(params.values()))
                out["backward_completed"] = 1
                slope = _dot(dict(zip(params, grads, strict=True)), bundle.delta)
                _require(np.isfinite(slope), "nonfinite held-out directional derivative")
                out["directional_derivative"] = slope
        out["safe"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        prefix = getattr(exc, "partial_states", None)
        if prefix is not None:
            endpoints = prefix[0]
            # Keep the measured score if the final model endpoint itself failed the fall check.
            if len(endpoints) == HORIZON + 1:
                origin = float(start["state45"][0, 0])
                out["return_m"] = -float(np.abs(endpoints[1:, 0].astype(np.float64) - origin).mean())
    out["state45"] = [] if endpoints is None else endpoints.tolist()
    out["ticks"] = max(0, len(out["state45"]) - 1)
    _meter(out, meter)
    return out
