"""Fit both late-response loss weights from one sealed round-98 endpoint and Adam state."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tools56.fit56 import mapping_sha256
from tools94 import learn94
from tools94.common94 import ROOT, SEEDS, read, seed_dir, write
from tools96 import fit96
from tools99 import data99, loss99


WEIGHTS = {"control": 0., "late": 1.}
PREVIOUS_STEP, UPDATES = 175, 25
MODEL_STEPS_PER_UPDATE = 18 * 16 + 6 * 16
COMPONENTS = ("state_h8", "response_h8", "state_h16", "response_h17_32")


def _parent(seed, source):
    from tools99.common99 import PLAN, PLAN_SHA

    fit96.checked(PLAN, PLAN_SHA)
    folder = data99.BASE98 / "fit_01"
    seal = read(fit96.checked(folder / "seal.json", data99.FIT98_SEAL))
    prefix = f"seed_{seed}"
    seed_key = prefix + "/receipt.json"
    arm_key = prefix + "/endpoint/receipt.json"
    checkpoint_key = prefix + "/endpoint/world.pt"
    files = seal["files"]
    previous = read(fit96.checked(folder / seed_key, files[seed_key]))
    arm = read(fit96.checked(folder / arm_key, files[arm_key]))
    fit96.require(seal["stage"] == "fit" and previous["seed"] == seed and
                  previous["status"] == arm["status"] == "COMPLETE" and previous["arms"]["endpoint"] == arm and
                  arm["lambda_endpoint"] == 1. and arm["updates_attempted"] == arm["updates_completed"] == 25 and
                  arm["attempted_model_steps"] == arm["completed_model_steps"] == 7200, "unqualified round-98 parent")
    ref = arm["checkpoint"]
    fit96.require(ref["file"] == "world.pt" and ref["sha256"] == files[checkpoint_key], "wrong endpoint checkpoint")
    path = fit96.checked(folder / checkpoint_key, ref["sha256"])
    packet = torch.load(path, map_location="cpu", weights_only=True)
    fit96.require(mapping_sha256(packet["network"]) == ref["weights_sha256"] and
                  fit96._adam_hash(packet["optimizer"]) == ref["adam_sha256"], "parent numerical identity differs")
    src = previous["source"]
    fit96.checked(seed_dir(seed) / "initial.pt", src["policy94"]["sha256"])
    source.update(protocol_sha256=PLAN_SHA, fit98_seal_sha256=data99.FIT98_SEAL,
                  seed98_receipt_sha256=files[seed_key], endpoint98_receipt_sha256=files[arm_key],
                  parent_checkpoint={**ref, "path": str(path.relative_to(ROOT))},
                  ridge_and_stats_sha256=src["ridge_and_stats_sha256"], policy94=src["policy94"],
                  direction95_tensors_sha256=src["direction95_tensors_sha256"],
                  training_traces=src["training_traces"], prefix98_tensors_sha256=previous["batch"]["tensors_sha256"],
                  early_denominator_m2=previous["normalization"]["denominator_m2"])
    return packet


def _batch_sha(data):
    return mapping_sha256({**data.prefix_batch, "late_actions": data.late_actions, "late_targets": data.late_targets})


def _blank(arm):
    return {"status": "NOT_RUN", "lambda_late": WEIGHTS[arm], "error": None, "checkpoint": None,
            **dict.fromkeys(fit96.CLOCKS, 0), "losses": []}


def _fit_arm(arm, folder, packet, source, data):
    folder.mkdir()
    row = _blank(arm)
    row["status"] = "PARTIAL"
    meter = learn94._work()
    world = optimizer = None
    try:
        world = fit96._fresh_world(packet["network"])
        world.residual.train().requires_grad_(True)
        optimizer = fit96._restore_adam(world.residual, packet["optimizer"], PREVIOUS_STEP)
        row["initial_weights_sha256"] = mapping_sha256(world.state_dict())
        row["initial_adam_sha256"] = fit96.optimizer_sha(optimizer)
        parent = source["parent_checkpoint"]
        fit96.require(row["initial_weights_sha256"] == parent["weights_sha256"] and
                      row["initial_adam_sha256"] == parent["adam_sha256"] and
                      mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"], "wrong fork")
        with (folder / "progress.jsonl").open("x", buffering=1) as log:
            for i in range(1, UPDATES + 1):
                row["updates_attempted"] = i
                terms = loss99.training_terms(world, data.prefix_batch, data.late_actions, data.late_targets,
                                              data.early_denom, include_late=WEIGHTS[arm], meter=meter)
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                norm = torch.nn.utils.clip_grad_norm_(world.residual.parameters(), learn94.CLIP, error_if_nonfinite=True)
                optimizer.step()
                row["updates_completed"] = i
                fit96.require(all(bool(p.isfinite().all()) for p in world.residual.parameters()),
                              "nonfinite fitted residual")
                metrics = {"update": i, **{k: float(terms[k].detach()) for k in (*COMPONENTS, "total")},
                           "preclip_gradient_norm": float(norm)}
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
                row["losses"].append(metrics)
        fit96._validate_adam(optimizer, world.residual, PREVIOUS_STEP + UPDATES)
        fit96.require(mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"] and
                      meter["imagined_steps_attempted"] == meter["imagined_steps"] == UPDATES * MODEL_STEPS_PER_UPDATE,
                      "frozen buffers or fitted model clocks differ")
        optimizer.zero_grad(set_to_none=True)
        row["checkpoint"] = fit96._save(folder / "world.pt", world, optimizer)
        row["status"] = "COMPLETE"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if world is not None and optimizer is not None:
            row["partial_checkpoint"] = fit96._save(folder / "partial_world.pt", world, optimizer)
    row.update(attempted_model_steps=meter["imagined_steps_attempted"], completed_model_steps=meter["imagined_steps"])
    write(folder / "receipt.json", row)
    return row


def fit_seed(seed, dest):
    """Train both arms in a new or empty seed directory. Preserve every partial attempt."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"preserve previous fit: {dest}")
    write(dest / "started.json", {"seed": seed, "updates_per_arm": UPDATES, "models": list(WEIGHTS)})
    receipt = {"seed": seed, "status": "PARTIAL", "error": None, "source": {}, "batch": {}, "normalization": {},
               "policy_updates": 0, "ppo_updates": 0, "flagged_endpoints": [],
               "arms": {arm: _blank(arm) for arm in WEIGHTS}}
    try:
        fit96.require(seed in SEEDS, "unregistered training seed")
        source = receipt["source"]
        packet = _parent(seed, source)
        data = data99.load_training_data(seed)
        prefix_sha = mapping_sha256(data.prefix_batch)
        fit96.require(prefix_sha == source["prefix98_tensors_sha256"] and
                      data.early_denom == source["early_denominator_m2"] and math.isfinite(data.early_denom) and
                      data.early_denom >= fit96.FLOOR_M2, "base training inputs differ")
        late_mse = float(data.late_targets.detach().double().square().mean())
        fit96.require(math.isfinite(late_mse), "nonfinite training response scale")
        receipt["normalization"] = {"early_denominator_m2": data.early_denom, "late_target_mse_m2": late_mse,
                                    "late_denominator_m2": max(late_mse, fit96.FLOOR_M2), "floor_m2": fit96.FLOOR_M2}
        receipt["flagged_endpoints"] = data.flagged_endpoints
        receipt["batch"] = {"prefix_rows": 18, "prefix_ticks": 16, "suffix_rows": 6, "suffix_ticks": 16,
                            "order": ["12 training94 resets", "3 m1 training96 resets", "3 p1 training96 resets"],
                            "tensors_sha256": _batch_sha(data), "prefix_tensors_sha256": prefix_sha,
                            "late_actions_sha256": mapping_sha256({"late_actions": data.late_actions}),
                            "late_targets_sha256": mapping_sha256({"late_targets": data.late_targets})}
        for arm in WEIGHTS:
            receipt["arms"][arm] = _fit_arm(arm, dest / arm, packet, source, data)
        fit96.require(all(row["status"] == "COMPLETE" for row in receipt["arms"].values()), "one or more incomplete forks")
        control, late = (receipt["arms"][arm]["losses"][0] for arm in WEIGHTS)
        fit96.require(all(control[k] == late[k] for k in COMPONENTS), "initial unweighted losses differ between forks")
        ref = source["parent_checkpoint"]
        fit96.require(mapping_sha256(packet["network"]) == ref["weights_sha256"] and
                      fit96._adam_hash(packet["optimizer"]) == ref["adam_sha256"] and
                      _batch_sha(data) == receipt["batch"]["tensors_sha256"], "parent or fixed inputs changed")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    for key in fit96.CLOCKS:
        receipt[key] = sum(arm[key] for arm in receipt["arms"].values())
    write(dest / "receipt.json", receipt)
    return receipt


def load_fit(seed, dest, arm):
    """Return a frozen complete world for prediction after checking its checkpoint and Adam identity."""
    dest = Path(dest)
    fit96.require(seed in SEEDS and arm in WEIGHTS, "unregistered fit")
    receipt = read(dest / "receipt.json")
    fit96.require(receipt["seed"] == seed and receipt["status"] == "COMPLETE" and
                  set(receipt["arms"]) == set(WEIGHTS), "incomplete matched fit")
    for name, weight in WEIGHTS.items():
        sibling = receipt["arms"][name]
        fit96.require(sibling == read(dest / name / "receipt.json") and sibling["status"] == "COMPLETE" and
                      sibling["lambda_late"] == weight and sibling["updates_attempted"] == sibling["updates_completed"] == UPDATES
                      and sibling["attempted_model_steps"] == sibling["completed_model_steps"] == UPDATES * MODEL_STEPS_PER_UPDATE,
                      "incomplete matched fit")
    row = receipt["arms"][arm]
    source = receipt["source"]
    fit96.require(len(row["losses"]) == UPDATES and row["losses"][-1]["update"] == UPDATES
                  and row["initial_weights_sha256"] == source["parent_checkpoint"]["weights_sha256"]
                  and row["initial_adam_sha256"] == source["parent_checkpoint"]["adam_sha256"],
                  "unqualified or incomplete late-response fit")
    ref = row["checkpoint"]
    fit96.require(ref["file"] == "world.pt", "unexpected fitted checkpoint")
    packet = torch.load(fit96.checked(dest / arm / ref["file"], ref["sha256"]), map_location="cpu", weights_only=True)
    world = fit96._fresh_world(packet["network"])
    fit96.require(mapping_sha256(world.state_dict()) == ref["weights_sha256"] and
                  mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"] and
                  fit96._adam_hash(packet["optimizer"]) == ref["adam_sha256"] and
                  len(packet["optimizer"]["state"]) == len(list(world.residual.parameters())) and
                  all(v["step"].item() == PREVIOUS_STEP + UPDATES for v in packet["optimizer"]["state"].values()),
                  "fitted weights, buffers or Adam state differ")
    return world
