"""Paired H8 response calibration. Both losses warm-start the same residual model and Adam."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools18.train_delay18 import optimizer_sha
from tools52 import dynamics52
from tools56.fit56 import mapping_sha256
from tools91.model91 import TerrainDynamics91
from tools94 import learn94
from tools94.common94 import ROOT, SEEDS, arrays, digest, fit_reset, read, seed_dir, write
from tools96.common96 import ORIGINAL as SEAL95, PLAN_SHA, TRAIN_RESETS as RESETS


H, UPDATES, OLD_COUNT, PAIR_COUNT = 8, 50, 16, 8
FLOOR_M2 = 1e-10
ARMS = {"control": 0., "paired": 1.}
SEAL95_SHA = "ee58a6fd4b67cf2908c4aa885bffbb286dacfea7003a2ae99acb229939d744eb"
FIELDS = ("state45", "applied16", "issued16", "terminated", "truncated", "reset_seed")
CLOCKS = ("updates_attempted", "updates_completed", "attempted_model_steps", "completed_model_steps")


def require(value, message):
    if not value:
        raise ValueError(message)


def checked(path, expected):
    require(digest(path) == expected, f"changed source file: {path}")
    return path


def _adam_hash(state):
    return optimizer_sha(SimpleNamespace(state_dict=lambda: state))


def _validate_adam(optimizer, residual, step):
    params = list(residual.parameters())
    require(len(optimizer.param_groups) == 1, "one residual Adam group required")
    group = optimizer.param_groups[0]
    require([id(p) for p in group["params"]] == [id(p) for p in params] and
            set(optimizer.state) == set(params) and group["lr"] == learn94.LR and
            tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8 and group["weight_decay"] == 0
            and not group["amsgrad"] and not group["maximize"], "parent Adam configuration differs")
    for p in params:
        state = optimizer.state[p]
        require(state["step"].item() == step and all(
            state[k].shape == p.shape and state[k].dtype == p.dtype and bool(state[k].isfinite().all())
            for k in ("exp_avg", "exp_avg_sq")) and bool((state["exp_avg_sq"] >= 0).all()),
            "invalid parent Adam moment, shape or update count")


def _restore_adam(residual, saved, step):
    # load_state_dict can reuse CPU tensor storage; each branch needs its own copy of both moments.
    optimizer = torch.optim.Adam(residual.parameters(), lr=learn94.LR)
    optimizer.load_state_dict(copy.deepcopy(saved))
    _validate_adam(optimizer, residual, step)
    require(optimizer_sha(optimizer) == _adam_hash(saved), "Adam differs immediately after restore")
    return optimizer


def _fresh_world(network):
    with torch.random.fork_rng(devices=[]):
        world = TerrainDynamics91()
    world.load_state_dict(network, strict=True)
    return world.eval().requires_grad_(False)


def _parent(seed):
    root = seed_dir(seed)
    seal = read(checked(SEAL95 / "seal.json", SEAL95_SHA))
    key = f"seed_{seed}"
    parent95 = read(checked(SEAL95 / key / "receipt.json", seal["files"][key + "/receipt.json"]))
    require(parent95["status"] == "COMPLETE" and parent95["seed"] == seed and
            parent95["delta_file_sha256"] == seal["files"][key + "/delta.pt"], "wrong sealed direction")
    checked(SEAL95 / key / "delta.pt", parent95["delta_file_sha256"])
    refs = parent95["sources"]
    report = read(checked(root / "report.json", refs["seed_report_sha256"]))
    first = report["rounds"][0]
    require(report["seed"] == seed and len(report["rounds"]) == 1 and
            report["initial"] == refs["initial"] and first["world_checkpoint"] == refs["world"] and
            first["world_fit"]["status"] == "COMPLETE" and first["world_fit"]["updates_completed"] == 100,
            "incomplete or mismatched source fit")
    checked(root / "initial.pt", refs["initial"]["sha256"])
    data = torch.load(checked(root / "round1/world.pt", refs["world"]["sha256"]),
                      map_location="cpu", weights_only=True)
    require(mapping_sha256(data["network"]) == refs["world"]["weights_sha256"], "parent tensor mismatch")
    record = {"seed_report94_sha256": refs["seed_report_sha256"], "direction95_seal_sha256": SEAL95_SHA,
              "direction95_receipt_sha256": seal["files"][key + "/receipt.json"],
              "direction95_tensors_sha256": parent95["delta_tensors_sha256"],
              "direction95_file_sha256": parent95["delta_file_sha256"], "policy94": refs["initial"],
              "world94": refs["world"], "original_adam_sha256": _adam_hash(data["optimizer"]),
              "ridge_and_stats_sha256": first["world_fit"]["ridge_and_stats_sha256"]}
    return data, first, record


def _old_bank(seed, first, sources):
    root = seed_dir(seed) / "round1"
    roster = read(root / "training_inputs.json")
    require(roster == first["fit"] and len(roster) == 12, "original fit roster changed")
    traces = []
    sources["old_traces"] = []
    for j, ref in enumerate(roster):
        reset = fit_reset(seed, 0, j)
        require(ref["reset"] == reset, "unexpected old training reset")
        path = checked(root / f"train_{reset}/trace.npz", ref["sha256"])
        with np.load(path, allow_pickle=False) as t:
            traces.append({k: t[k] for k in FIELDS})
        sources["old_traces"].append({"reset": reset, "sha256": ref["sha256"], "path": str(path.relative_to(ROOT))})
    data, rows, details = learn94._round_arrays(traces)
    require(first["world_fit"]["round_data"] == [details], "old training tensor roster differs from source fit")
    idx = rows.numpy()
    sources["old_training_arrays_sha256"] = details["sha256"]
    return _bank(data["s"][idx[:, 0]], data["a"][idx], data["target"][idx]), np.column_stack(
        (np.asarray(details["seeds"])[data["episode"][idx[:, 0]]], data["tick"][idx[:, 0]]))


def _bank(s, a, target):
    return tuple(torch.as_tensor(np.asarray(v), dtype=torch.float32) for v in (s, a, target))


def _native(folder, reset, tag):
    folder = folder / tag
    receipt = read(folder / "native.json")
    require(receipt.get("error") is None and receipt["trace_file"] == "native/trace.npz",
            f"raw {tag} collection exception or partial")
    path = checked(folder / "native/trace.npz", receipt["trace_sha256"])
    with np.load(path, allow_pickle=False) as f:
        trace = dict(f)
    state, applied, issued = (trace[k] for k in ("state45", "applied16", "issued16"))
    n = len(applied)
    flag = {k: trace[k] for k in ("fallen", "illegal", "terminated", "truncated")}
    require(int(trace["reset_seed"]) == reset and int(trace["requested_ticks"]) == 32 and n == receipt["ticks"]
            and n == int(trace["completed_transitions"]) and H <= n <= 32 and
            (int(trace["policy_steps"]), int(trace["servo_steps"]), int(trace["native_steps"])) == (n, 5*n, 10*n)
            and int(trace["explore_seed"]) == -1 and float(trace["wheel_offset_rad_s"]) == 0. and
            not trace["exploration16"].any() and not trace["command3"].any(), f"invalid native clocks or task: {path}")
    require(state.shape == (n+1, 45) and state.dtype == np.float64 and np.isfinite(state).all() and
            applied.shape == issued.shape == (n, 16) and np.isfinite(issued).all() and (np.abs(issued) <= 1).all()
            and not applied[0].any() and np.array_equal(applied[1:], issued[:-1])
            and np.array_equal(trace["obs643"][1:, 41:57], issued), f"invalid state or executed-action FIFO: {path}")
    require(all(v.shape == (n,) and v.dtype == bool for v in flag.values()) and
            not any(v[:-1].any() for v in flag.values()) and (n == 32 or flag["terminated"][-1] or flag["truncated"][-1])
            and all(bool(v.any()) == receipt[k] for k, v in flag.items()) and
            receipt["safe"] == (n == 32 and not any(v.any() for v in flag.values())),
            f"invalid terminal prefix: {path}")
    return trace, {"reset": reset, "arm": tag, "path": str(path.relative_to(folder.parents[2])),
                   "sha256": receipt["trace_sha256"],
                   "receipt_sha256": digest(folder / "native.json"), "ticks": n,
                   **{k: bool(v.any()) for k, v in flag.items()}}


def _pair_bank(collection, seed, sources):
    keys = ("state45", "obs643", "filtered3", "time_s", "sensor_capture_tick", "sensor_delivery_tick",
            "ray_due", "ray_raw16", "pending_capture_tick", "pending_ray16")
    windows, targets, provenance = [[], []], [], []
    pairs = sources["new_pairs"] = []
    for reset in RESETS:
        row = {"reset": reset, "arms": {}, "windows": 0}
        pairs.append(row)
        try:
            traces = []
            for tag in ("m1", "p1"):
                t, row["arms"][tag] = _native(collection / f"seed_{seed}/reset_{reset}", reset, tag)
                traces.append(t)
            minus, plus = traces
            require(all(np.array_equal(minus[k][0], plus[k][0]) for k in keys) and
                    np.array_equal(minus["state45"][1], plus["state45"][1]), "different paired reset or first endpoint")
            n = min(len(t["applied16"]) for t in traces)
            idx = np.arange(n-H+1)[:, None] + np.arange(H)
            for output, t in zip(windows, traces, strict=True):
                s = t["state45"]
                output.append((s[idx[:, 0]], t["applied16"][idx], s[idx+1]))
            target = _difference(windows[0][-1][2][..., 0], windows[0][-1][0][:, 0],
                                 windows[1][-1][2][..., 0], windows[1][-1][0][:, 0])
            targets.append(target)
            provenance.extend((reset, int(start)) for start in idx[:, 0])
            row["windows"] = len(idx)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    require(all(row["windows"] > 0 for row in pairs), "three matched training resets with legal H8 windows required")
    banks = [_bank(*(np.concatenate(field) for field in zip(*parts, strict=True))) for parts in windows]
    return banks, torch.from_numpy(np.concatenate(targets)), np.asarray(provenance, dtype=np.int64)


def _difference(minus_x, minus_origin, plus_x, plus_origin):
    """Each side's departure relative to its own window origin, positive arm minus negative."""
    return (plus_x - plus_origin[:, None]) - (minus_x - minus_origin[:, None])


def _batch_plan(seed, n_old, n_new):
    rng = np.random.default_rng(seed + 960000)
    old, new = [], []
    for _ in range(UPDATES):
        old.append(rng.integers(n_old, size=OLD_COUNT, dtype=np.int64))
        new.append(rng.integers(n_new, size=PAIR_COUNT, dtype=np.int64))
    return np.stack(old), np.stack(new)


def _blank(arm):
    return {"status": "NOT_RUN", "lambda_pair": ARMS[arm], "error": None, "checkpoint": None,
            **dict.fromkeys(CLOCKS, 0)}


def _save(path, world, optimizer):
    data = {"network": world.state_dict(), "optimizer": optimizer.state_dict()}
    with path.open("xb") as out:
        torch.save(data, out)
    return {"file": path.name, "sha256": digest(path), "weights_sha256": mapping_sha256(data["network"]),
            "adam_sha256": optimizer_sha(optimizer)}


def _fit_arm(name, folder, packet, source, old, sides, target_d, denom, old_idx, new_idx):
    row = _blank(name)
    row.update(status="PARTIAL", losses=[])
    folder.mkdir()
    world = optimizer = None
    meter = learn94._work()
    try:
        world = _fresh_world(packet["network"])
        world.residual.train().requires_grad_(True)
        optimizer = _restore_adam(world.residual, packet["optimizer"], 100)
        row["initial_weights_sha256"] = mapping_sha256(world.state_dict())
        row["initial_adam_sha256"] = optimizer_sha(optimizer)
        require(row["initial_weights_sha256"] == source["world94"]["weights_sha256"] and
                row["initial_adam_sha256"] == source["original_adam_sha256"] and
                mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"],
                "branch did not start from the sealed parent")
        terrain = torch.ones(OLD_COUNT + 2*PAIR_COUNT, dtype=torch.long)
        with (folder / "progress.jsonl").open("x", buffering=1) as log:
            for i, (oi, pi) in enumerate(zip(old_idx, new_idx, strict=True), start=1):
                row["updates_attempted"] = i
                start, applied, target = (torch.cat((old[j][oi], sides[0][j][pi], sides[1][j][pi]))
                                          for j in range(3))
                state, predicted = start, []
                for action in applied.unbind(1):
                    state = learn94._step(world, state, action, terrain, meter)
                    predicted.append(state)
                predicted = torch.stack(predicted, dim=1)
                state_loss = dynamics52.group_loss(predicted, target)
                # Keep the original float32 H8 model and state loss. Convert x BEFORE subtracting.
                neg = slice(OLD_COUNT, OLD_COUNT + PAIR_COUNT)
                pos = slice(OLD_COUNT + PAIR_COUNT, None)
                response = _difference(predicted[neg, :, 0].double(), start[neg, 0].double(),
                                       predicted[pos, :, 0].double(), start[pos, 0].double())
                mse = (response - target_d[pi]).square().mean()
                auxiliary = mse / denom
                loss = state_loss + auxiliary if row["lambda_pair"] else state_loss
                require(bool(loss.isfinite()) and bool(auxiliary.isfinite()), "nonfinite paired or state loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(world.residual.parameters(), learn94.CLIP, error_if_nonfinite=True)
                optimizer.step()
                row["updates_completed"] = i
                require(all(bool(p.isfinite().all()) for p in world.residual.parameters()), "nonfinite residual")
                item = {"update": i, "state_loss": float(state_loss.detach()), "response_mse_m2": float(mse.detach()),
                        "normalized_pair_loss": float(auxiliary.detach()), "loss": float(loss.detach()),
                        "preclip_gradient_norm": float(norm)}
                row["losses"].append(item)
                log.write(json.dumps(item, allow_nan=False) + "\n")
        _validate_adam(optimizer, world.residual, 100 + UPDATES)
        require(mapping_sha256(dict(world.named_buffers())) == source["ridge_and_stats_sha256"] and
                meter["imagined_steps"] == meter["imagined_steps_attempted"] == UPDATES * H * len(terrain),
                "frozen buffers or model clocks differ")
        optimizer.zero_grad(set_to_none=True)
        row["checkpoint"] = _save(folder / "world.pt", world, optimizer)
        row["status"] = "COMPLETE"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if world is not None and optimizer is not None:
            row["partial_checkpoint"] = _save(folder / "partial_world.pt", world, optimizer)
    row.update(attempted_model_steps=meter["imagined_steps_attempted"], completed_model_steps=meter["imagined_steps"])
    write(folder / "receipt.json", row)
    return row


def fit_seed(seed, collection, dest):
    """One registered seed and two independent 50-update arms. Destination must be new or empty."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise FileExistsError(f"preserve existing fit: {dest}")
    write(dest / "started.json", {"seed": seed, "arms": list(ARMS), "updates_per_arm": UPDATES})
    receipt = {"seed": seed, "status": "PARTIAL", "error": None, "source": {}, "batch": {}, "normalization": {},
               "policy_updates": 0, "ppo_updates": 0, "arms": {name: _blank(name) for name in ARMS}}
    try:
        require(seed in SEEDS, "unregistered fit seed")
        packet, first, source = _parent(seed)
        receipt["source"] = source
        old, old_roster = _old_bank(seed, first, source)
        collection = Path(collection)
        seal = read(collection / "seal.json")
        require(seal["stage"] == "collection" and seal["protocol_sha256"] == PLAN_SHA and
                seal["direction_seal95_sha256"] == SEAL95_SHA, "collection provenance changed")
        source["collection_seal_sha256"] = digest(collection / "seal.json")
        sides, target_d, new_roster = _pair_bank(collection, seed, source)
        for pair in source["new_pairs"]:
            for ref in pair["arms"].values():
                receipt_file = str(Path(ref["path"]).parents[1] / "native.json")
                require(seal["files"][ref["path"]] == ref["sha256"] and
                        seal["files"][receipt_file] == ref["receipt_sha256"], "pair does not match the collection seal")
        mse = float(target_d.square().mean())
        require(np.isfinite(mse) and target_d.dtype == torch.float64, "invalid physical response normalization")
        denom = max(mse, FLOOR_M2)
        receipt["normalization"] = {"target_mse_m2": mse, "floor_m2": FLOOR_M2, "denominator_m2": denom,
                                    "pair_windows": len(target_d), "physical_samples": int(target_d.numel()),
                                    "response_targets_sha256": mapping_sha256({"plus_minus_x_m": target_d})}
        oi, pi = _batch_plan(seed, len(old_roster), len(new_roster))
        indices = {"old_reset_start": old_roster, "new_reset_start": new_roster,
                   "old_rows": oi, "paired_rows": pi}
        path = dest / "batch_indices.npz"
        arrays(path, indices)
        receipt["batch"] = {"rng_seed": seed + 960000, "updates": UPDATES, "old_per_batch": OLD_COUNT,
                            "pairs_per_batch": PAIR_COUNT, "order": ["old", "negative", "positive"],
                            "old_windows": len(old_roster), "paired_windows": len(new_roster),
                            "file": path.name, "file_sha256": digest(path), "indices_sha256": mapping_sha256(indices)}
        for name in ARMS:
            receipt["arms"][name] = _fit_arm(name, dest / name, packet, source, old, sides, target_d, denom, oi, pi)
        require(all(a["status"] == "COMPLETE" for a in receipt["arms"].values()), "one or more incomplete arms")
        initial = [a["losses"][0] for a in receipt["arms"].values()]
        require(initial[0]["state_loss"] == initial[1]["state_loss"] and
                initial[0]["response_mse_m2"] == initial[1]["response_mse_m2"],
                "unequal common-parent forward pass on the first batch")
        require(_adam_hash(packet["optimizer"]) == source["original_adam_sha256"] and
                mapping_sha256(packet["network"]) == source["world94"]["weights_sha256"],
                "parent changed while its two forks were fitted")
        receipt["status"] = "COMPLETE"
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    for key in CLOCKS:
        receipt[key] = sum(a[key] for a in receipt["arms"].values())
    write(dest / "receipt.json", receipt)
    return receipt


def load_fit(seed, dest, arm):
    """Load only a complete registered step-50 branch and verify weights, buffers and Adam."""
    require(seed in SEEDS and arm in ARMS, "unregistered fit")
    dest = Path(dest)
    receipt = read(dest / "receipt.json")
    row = receipt["arms"][arm]
    require(receipt["seed"] == seed and row == read(dest / arm / "receipt.json") and row["status"] == "COMPLETE"
            and row["updates_attempted"] == row["updates_completed"] == UPDATES and row["lambda_pair"] == ARMS[arm],
            "incomplete or mismatched step-50 arm")
    require(receipt["status"] == "COMPLETE" or any(r["status"] != "COMPLETE" for r in receipt["arms"].values()),
            "the two completed arms failed the joint integrity check")
    ref = row["checkpoint"]
    require(ref["file"] == "world.pt", "unknown fitted checkpoint")
    data = torch.load(checked(dest / arm / ref["file"], ref["sha256"]), map_location="cpu", weights_only=True)
    world = _fresh_world(data["network"])
    require(mapping_sha256(world.state_dict()) == ref["weights_sha256"] and
            mapping_sha256(dict(world.named_buffers())) == receipt["source"]["ridge_and_stats_sha256"] and
            _adam_hash(data["optimizer"]) == ref["adam_sha256"], "fitted tensor identity differs")
    return world
