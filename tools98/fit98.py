"""An equal-budget H16 endpoint loss on training-reset prefixes of each frozen round-96 paired model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tools52 import dynamics52
from tools56.fit56 import mapping_sha256
from tools94 import learn94
from tools94.common94 import ROOT, SEEDS, digest, fit_reset, read, seed_dir, write
from tools96 import fit96
from tools96.common96 import BASE as BASE96, TRAIN_RESETS


H8, H16, ROWS, UPDATES = 8, 16, 18, 25
PREVIOUS_STEP = 150
WEIGHTS = {"control": 0., "endpoint": 1.}
CHECKS96 = {"collection": "bd8eaba67195d518807c8d4aefa6ada75e0780be7200613dd89e4f7ed28beefb",
            "fit": "c2b3f711af82859c82539b764be9937e884d74958ba67c8f761f0bd39f75094f",
            "public": "4c46b71939cd7df9802f11f78e49a9cd9a56b91636ee329bb0c04c0ceb70c8e4"}
CLOCKS = fit96.CLOCKS


def _prefix(path, expected_hash, reset):
    """Hash the source file, then expose only states 0..16 and already-applied actions 0..15."""
    with np.load(fit96.checked(path, expected_hash), allow_pickle=False) as t:
        states = t["state45"][:H16+1].copy()
        applied = t["applied16"][:H16].copy()
        issued = t["issued16"][:H16].copy()
        fit96.require(int(t["reset_seed"]) == reset and int(t["completed_transitions"]) >= H16 and
                      not (t["terminated"][:H16-1] | t["truncated"][:H16-1]).any(),
                      f"unavailable H16 reset prefix: {path}")
    fit96.require(states.shape == (H16+1, 45) and states.dtype == np.float64 and np.isfinite(states).all()
                  and applied.shape == issued.shape == (H16, 16) and applied.dtype == issued.dtype == np.float32
                  and np.isfinite(issued).all() and (np.abs(issued) <= 1.).all() and not applied[0].any()
                  and np.array_equal(applied[1:], issued[:-1]), f"bad training prefix state or FIFO: {path}")
    return states, applied


def _training_sources(seed, source):
    from tools98.common98 import PLAN, PLAN_SHA

    # Hash protocol dependencies as bytes. Only training files supply numerical inputs or targets.
    plan = read(fit96.checked(PLAN, PLAN_SHA))
    for section in ("parents_sha256", "local_parent_seals_sha256"):
        for name, expected in plan[section].items():
            fit96.checked(ROOT / name, expected)
    fit96.require(plan["arms"] == WEIGHTS and tuple(plan["training_seeds"]) == SEEDS and
                  tuple(plan["paired_training_resets"]) == TRAIN_RESETS and plan["train_horizon"] == H16 and
                  plan["parents_sha256"]["reports/paired_response96.json"] == CHECKS96["public"],
                  "registered training population or model comparison differs")
    source["protocol_sha256"] = PLAN_SHA
    collection, fits = BASE96 / "collection_01", BASE96 / "fit_01"
    col = read(fit96.checked(collection / "seal.json", CHECKS96["collection"]))
    fit_seal = read(fit96.checked(fits / "seal.json", CHECKS96["fit"]))
    fit96.require(col["stage"] == "collection" and fit_seal["stage"] == "fit" and
                  fit_seal["previous_seal_sha256"] == CHECKS96["collection"], "round-96 source chain differs")
    key = f"seed_{seed}"
    receipt_path = key + "/receipt.json"
    parent = read(fit96.checked(fits / receipt_path, fit_seal["files"][receipt_path]))
    arm_path = key + "/paired/receipt.json"
    arm = read(fit96.checked(fits / arm_path, fit_seal["files"][arm_path]))
    fit96.require(parent["seed"] == seed and parent["status"] == arm["status"] == "COMPLETE" and
                  arm == parent["arms"]["paired"] and arm["updates_completed"] == 50 and
                  arm["checkpoint"]["file"] == "world.pt" and arm["lambda_pair"] == 1. and
                  parent["source"]["collection_seal_sha256"] == CHECKS96["collection"], "wrong fitted parent")
    ref = arm["checkpoint"]
    weight_path = key + "/paired/world.pt"
    fit96.require(fit_seal["files"][weight_path] == ref["sha256"], "round-96 checkpoint references differ")
    packet = torch.load(fit96.checked(fits / weight_path, ref["sha256"]), map_location="cpu", weights_only=True)
    fit96.require(mapping_sha256(packet["network"]) == ref["weights_sha256"] and
                  fit96._adam_hash(packet["optimizer"]) == ref["adam_sha256"], "parent weight or optimizer mismatch")
    fit96.checked(seed_dir(seed) / "initial.pt", parent["source"]["policy94"]["sha256"])
    fit96.checked(fit96.SEAL95 / key / "delta.pt", parent["source"]["direction95_file_sha256"])
    source.update(report96_sha256=CHECKS96["public"], collection96_seal_sha256=CHECKS96["collection"],
                  fit96_seal_sha256=CHECKS96["fit"], seed96_receipt_sha256=fit_seal["files"][receipt_path],
                  paired96_receipt_sha256=fit_seal["files"][arm_path], parent_checkpoint={
                      **ref, "path": str((fits / weight_path).relative_to(ROOT))},
                  ridge_and_stats_sha256=parent["source"]["ridge_and_stats_sha256"],
                  policy94=parent["source"]["policy94"], direction95_tensors_sha256=parent["source"]["direction95_tensors_sha256"])
    denom = parent["normalization"]["denominator_m2"]
    fit96.require(np.isfinite(denom) and denom >= fit96.FLOOR_M2 and
                  denom == max(parent["normalization"]["target_mse_m2"], fit96.FLOOR_M2), "invalid fixed denominator")
    prefix_states, prefix_actions = [], []
    old = parent["source"]["old_traces"]
    old_report = read(fit96.checked(seed_dir(seed) / "report.json", parent["source"]["seed_report94_sha256"]))
    roster = old_report["rounds"][0]["fit"]
    fit96.require(len(old) == len(roster) == 12 and old_report["seed"] == seed, "expected twelve original training resets")
    traces = source["training_traces"] = []
    for j, (entry, recorded) in enumerate(zip(old, roster, strict=True)):
        reset = fit_reset(seed, 0, j)
        path = seed_dir(seed) / f"round1/train_{reset}/trace.npz"
        fit96.require(entry["path"] == str(path.relative_to(ROOT)) and
                      reset == entry["reset"] == recorded["reset"] and entry["sha256"] == recorded["sha256"],
                      "original training identity differs")
        state, action = _prefix(path, entry["sha256"], reset)
        prefix_states.append(state)
        prefix_actions.append(action)
        traces.append({"role": "training94", **entry})
    pairs = parent["source"]["new_pairs"]
    fit96.require(len(pairs) == len(TRAIN_RESETS) and [p["reset"] for p in pairs] == list(TRAIN_RESETS),
                  "expected three registered round-96 training resets")
    for tag in ("m1", "p1"):
        for pair in pairs:
            reset, entry = pair["reset"], pair["arms"][tag]
            prefix = key + f"/reset_{reset}/{tag}"
            relative, receipt_file = prefix + "/native/trace.npz", prefix + "/native.json"
            fit96.require(entry["reset"] == reset and entry["arm"] == tag and entry["path"] == relative and
                          entry["sha256"] == col["files"][relative] and entry["receipt_sha256"] == col["files"][receipt_file],
                          "wrong round-96 training trajectory")
            record = read(fit96.checked(collection / receipt_file, entry["receipt_sha256"]))
            fit96.require(record["error"] is None and record["trace_file"] == "native/trace.npz" and
                          record["trace_sha256"] == entry["sha256"] and record["ticks"] >= H16, "partial training trajectory")
            state, action = _prefix(collection / relative, entry["sha256"], reset)
            prefix_states.append(state)
            prefix_actions.append(action)
            traces.append({"role": "training96", "arm": tag, "reset": reset, "sha256": entry["sha256"],
                           "receipt_sha256": entry["receipt_sha256"], "path": str((collection / relative).relative_to(ROOT))})
    return packet, denom, np.stack(prefix_states), np.stack(prefix_actions)


def _batch(states, applied):
    fit96.require(states.shape == (ROWS, H16+1, 45) and states.dtype == np.float64 and
                  applied.shape == (ROWS, H16, 16) and applied.dtype == np.float32,
                  "expected the fixed 18 complete training prefixes")
    fit96.require(np.array_equal(states[12:15, :2], states[15:18, :2]),
                  "pair members do not share their reset and first zero-applied endpoint")
    x = states[:, :, 0]
    target_d = fit96._difference(x[12:15, 1:H8+1], x[12:15, 0], x[15:18, 1:H8+1], x[15:18, 0])
    return {"start": torch.tensor(states[:, 0], dtype=torch.float32), "applied": torch.from_numpy(applied),
            "target": torch.tensor(states[:, 1:], dtype=torch.float32), "paired_x_m": torch.from_numpy(target_d)}


def _losses(predicted, batch, denom):
    state_h8 = dynamics52.group_loss(predicted[:, :H8], batch["target"][:, :H8])
    # Preserve sub-float32 physical displacement labels and subtract each model branch's actual input origin.
    x = predicted[:, :H8, 0].double()
    origin = batch["start"][:, 0].double()
    response = fit96._difference(x[12:15], origin[12:15], x[15:18], origin[15:18])
    mse = (response - batch["paired_x_m"]).square().mean()
    endpoint = dynamics52.group_loss(predicted[:, H16-1], batch["target"][:, H16-1])
    return state_h8, mse, mse / denom, endpoint


def _blank(arm):
    return {"status": "NOT_RUN", "lambda_endpoint": WEIGHTS[arm], "error": None, "checkpoint": None,
            **dict.fromkeys(CLOCKS, 0), "losses": []}


def _fit_arm(arm, folder, packet, source, batch, denom):
    folder.mkdir()
    row = _blank(arm)
    row["status"] = "PARTIAL"
    parent = source["parent_checkpoint"]
    meter = learn94._work()
    world = optimizer = None
    try:
        world = fit96._fresh_world(packet["network"])
        world.residual.train().requires_grad_(True)
        optimizer = fit96._restore_adam(world.residual, packet["optimizer"], PREVIOUS_STEP)
        row["initial_weights_sha256"] = mapping_sha256(world.state_dict())
        row["initial_adam_sha256"] = fit96.optimizer_sha(optimizer)
        fit96.require(row["initial_weights_sha256"] == parent["weights_sha256"] and
                      row["initial_adam_sha256"] == parent["adam_sha256"] and
                      mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"], "wrong fork")
        terrain = torch.ones(ROWS, dtype=torch.long)
        with (folder / "progress.jsonl").open("x", buffering=1) as log:
            for i in range(1, UPDATES + 1):
                row["updates_attempted"] = i
                state, predicted = batch["start"], []
                for action in batch["applied"].unbind(1):
                    state = learn94._step(world, state, action, terrain, meter)
                    predicted.append(state)
                state_loss, mse, paired, endpoint = _losses(torch.stack(predicted, dim=1), batch, denom)
                loss = state_loss + paired
                if WEIGHTS[arm]:
                    loss = loss + endpoint
                fit96.require(bool(loss.isfinite()) and bool(endpoint.isfinite()), "nonfinite fit loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(world.residual.parameters(), learn94.CLIP, error_if_nonfinite=True)
                optimizer.step()
                row["updates_completed"] = i
                fit96.require(all(bool(p.isfinite().all()) for p in world.residual.parameters()), "nonfinite fitted residual")
                metrics = {"update": i, "state_loss_h8": float(state_loss.detach()), "response_mse_m2": float(mse.detach()),
                           "normalized_pair_loss_h8": float(paired.detach()), "endpoint_loss_h16": float(endpoint.detach()),
                           "loss": float(loss.detach()), "preclip_gradient_norm": float(norm)}
                row["losses"].append(metrics)
                log.write(json.dumps(metrics, allow_nan=False) + "\n")
        fit96._validate_adam(optimizer, world.residual, PREVIOUS_STEP + UPDATES)
        fit96.require(mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"] and
                      meter["imagined_steps_attempted"] == meter["imagined_steps"] == UPDATES*H16*ROWS,
                      "frozen buffers or fit clocks differ")
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
    """Fit both 25-update forks in a new or already empty directory; never resume any prefix."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"preserve previous fit: {dest}")
    write(dest / "started.json", {"seed": seed, "updates_per_arm": UPDATES, "models": list(WEIGHTS)})
    receipt = {"seed": seed, "status": "PARTIAL", "error": None, "source": {}, "batch": {}, "normalization": {},
               "policy_updates": 0, "ppo_updates": 0, "arms": {name: _blank(name) for name in WEIGHTS}}
    try:
        fit96.require(seed in SEEDS, "unregistered training seed")
        packet, denom, states, applied = _training_sources(seed, receipt["source"])
        batch = _batch(states, applied)
        receipt["batch"] = {"rows": ROWS, "horizon": H16, "old_episodes": 12, "pairs": 3,
                            "order": ["12 training94 resets", "3 m1 training96 resets", "3 p1 training96 resets"],
                            "tensors_sha256": mapping_sha256(batch), "physical_states_sha256": mapping_sha256({"states": states})}
        receipt["normalization"] = {"denominator_m2": denom, "source": "round96/paired fit receipt"}
        for name in WEIGHTS:
            receipt["arms"][name] = _fit_arm(name, dest/name, packet, receipt["source"], batch, denom)
        fit96.require(all(a["status"] == "COMPLETE" for a in receipt["arms"].values()), "one or more incomplete forks")
        control, endpoint = (receipt["arms"][name]["losses"][0] for name in WEIGHTS)
        fit96.require(all(control[k] == endpoint[k] for k in ("state_loss_h8", "response_mse_m2",
                      "normalized_pair_loss_h8", "endpoint_loss_h16")), "unequal source model or first batch")
        ref = receipt["source"]["parent_checkpoint"]
        fit96.require(ref["weights_sha256"] == mapping_sha256(packet["network"]) and
                      ref["adam_sha256"] == fit96._adam_hash(packet["optimizer"]) and
                      mapping_sha256(batch) == receipt["batch"]["tensors_sha256"], "parent or fixed inputs changed")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    for key in CLOCKS:
        receipt[key] = sum(a[key] for a in receipt["arms"].values())
    write(dest / "receipt.json", receipt)
    return receipt


def load_fit(seed, dest, arm):
    dest = Path(dest)
    fit96.require(seed in SEEDS and arm in WEIGHTS, "unregistered fit")
    receipt = read(dest / "receipt.json")
    row = receipt["arms"][arm]
    fit96.require(receipt["seed"] == seed and row == read(dest / arm / "receipt.json") and row["status"] == "COMPLETE"
                  and row["lambda_endpoint"] == WEIGHTS[arm] and row["updates_attempted"] == row["updates_completed"] == UPDATES
                  and row["completed_model_steps"] == row["attempted_model_steps"] == UPDATES*H16*ROWS
                  and (receipt["status"] == "COMPLETE" or any(a["status"] != "COMPLETE" for a in receipt["arms"].values())),
                  "unqualified or incomplete step-25 fit")
    ref = row["checkpoint"]
    fit96.require(ref["file"] == "world.pt", "unexpected checkpoint")
    data = torch.load(fit96.checked(dest / arm / ref["file"], ref["sha256"]), map_location="cpu", weights_only=True)
    world = fit96._fresh_world(data["network"])
    fit96.require(mapping_sha256(world.state_dict()) == ref["weights_sha256"] and
                  mapping_sha256(dict(world.named_buffers())) == receipt["source"]["ridge_and_stats_sha256"] and
                  fit96._adam_hash(data["optimizer"]) == ref["adam_sha256"] and
                  all(v["step"].item() == PREVIOUS_STEP + UPDATES for v in data["optimizer"]["state"].values()),
                  "fitted weights, buffers or Adam state differ")
    return world
