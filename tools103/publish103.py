"""Publish complete D1 action tapes and verify paired forecasts from public files."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools91.launch91 import require
from tools94.common94 import digest, read, write
from tools100.publish100 import copy_verified
from tools103.data103 import window_inputs
from tools103.run103 import DEPENDENCIES, PARENT, PLAN_SHA, PUBLIC, STAGES, completed, protocol, sources, unit
from tools103.score103 import score


def _files(stage, plan):
    if stage == "prepare":
        return {n: n for n in ("train_actions.npz", "holdout_actions.npz", "audit.json")}
    if stage == "evaluate":
        return {n: n for n in ("forecast.npz", "report.json")}
    names = ["fit.json", "updates.jsonl"]
    names.extend(f"seed_{s}_{a}.pt" for s in plan["seeds"] for a in plan["arms"])
    return {f"models/{n}": f"fit/{n}" for n in names}


def _record(folder, usage):
    return dict(seal=read(folder / "seal.json"), seal_sha256=digest(folder / "seal.json"), resources=usage)


def _check_stages(plan, manifest, stages):
    require(manifest["protocol_sha256"] == PLAN_SHA and set(manifest["stages"]) == set(stages),
            "incomplete publication for the requested stages")
    limits = plan["resources"]
    for stage in stages:
        record = manifest["stages"][stage]
        seal, usage = record["seal"], record["resources"]
        seal_bytes = (json.dumps(seal, indent=2, allow_nan=False) + "\n").encode()
        expected_parents = {p: manifest["stages"][p]["seal_sha256"] for p in DEPENDENCIES[stage]}
        require(sha256(seal_bytes).hexdigest() == record["seal_sha256"]
                and seal["status"] == "COMPLETE" and seal["stage"] == stage and seal["unit"] == unit(stage)
                and seal["protocol_sha256"] == PLAN_SHA and seal["sources"] == sources(stage)
                and seal["parents"] == expected_parents, f"invalid {stage} seal or lineage")
        for source, public in _files(stage, plan).items():
            require(seal["files"][source] == digest(PUBLIC / public), f"published {public} differs from its seal")
        require(usage["startup_effective_bytes"] >= limits["admission_min_bytes"]
                and usage["minimum_effective_bytes"] >= limits["runtime_min_bytes"]
                and len(usage["cpu_affinity"]) == limits["cpu_affinity_count"]
                and usage["owned_cgroup_children_empty"] and usage["supervisor_sha256"] == limits["supervisor_sha256"]
                and usage["scope"].endswith(f"/app.slice/{unit(stage)}.service"), f"invalid {stage} resource receipt")


def _audit_tape(plan, split, audit):
    packet_path, tape_path = PARENT / f"{split}.npz", PUBLIC / f"{split}_actions.npz"
    windows = window_inputs(split, packet_path, tape_path)
    with np.load(tape_path, allow_pickle=False) as tape:
        commands, tape_ticks = tape["actions"], tape["ticks"]
        episode_lookup = dict(zip(zip(tape["source_seed"], tape["episode"], strict=True),
                                  range(len(commands)), strict=True))
        source_seeds = np.unique(tape["source_seed"]).tolist()
        roster_hash = tape["roster_sha256"].item()
    with np.load(packet_path, allow_pickle=False) as packets:
        seed, episode, series, capture, delivery = (packets[k] for k in
                ("source_seed", "episode", "series_id", "capture_tick", "delivery_tick"))
    # Check every slot independently from its original episode, including the
    # padding and the all-empty terminal row. The two target clocks differ by 40 ms.
    counts = plan["data"][split]
    require(windows.shape == (counts["rows"], 2, 2, 9, 17), "incorrect action-window tensor")
    starts = np.flatnonzero(series[:-1] == series[1:])
    terminal = np.ones(len(series), dtype=bool)
    terminal[starts] = False
    require(not windows[terminal].any(), "terminal packet has an action window")
    for row in starts:
        source = commands[episode_lookup[(seed[row], episode[row])]]
        for clock, boundaries in enumerate((capture, delivery)):
            t, next_t = boundaries[row:row + 2]
            width = int(next_t - t)
            require(width in (8, 9), "incorrect exposure length")
            for arm, begin in enumerate((t - width, t)):
                offset = int(begin - tape_ticks[0])
                block = windows[row, arm, clock]
                require(np.array_equal(block[:width, :16], source[offset:offset + width])
                        and np.all(block[:width, 16] == 1) and not block[width:].any(),
                        "published inputs do not match the registered clock and interval")
    native = np.unique(np.column_stack((seed[starts], episode[starts], capture[starts], capture[starts + 1])), axis=0)
    expected = dict(split=split, episodes=len(commands), source_seeds=source_seeds, source_rows=len(series),
                    verified_two_action_histories=len(series), native_intervals=len(native),
                    native_transition_ticks_per_clock=int((native[:, 3] - native[:, 2]).sum()),
                    native_command_rows=len(commands) * len(tape_ticks), packet_pairs=len(starts),
                    terminal_rows=int(terminal.sum()), tape_sha256=digest(tape_path), packets_sha256=digest(packet_path),
                    actions_sha256=plan["data"]["source_action_bytes_sha256"][split], roster_sha256=roster_hash,
                    catalog_sha256=digest(PARENT / "data.json"), forecast_action_column="issued",
                    native_applied_lag_ms=20, native_applied_lag_policy_ticks=1)
    require(all(audit[key] == value for key, value in expected.items())
            and expected["episodes"] == counts["native_episodes"] and len(starts) == counts["pairs"]
            and len(native) == counts["native_intervals"]
            and expected["native_transition_ticks_per_clock"] == counts["native_action_ticks_per_clock"],
            f"published {split} clocks differ from the native audit")
    # The first step's preceding issued command is outside this compact tape.
    same = int((commands[:, 1:] == commands[:, :-1]).all(axis=-1).sum())
    require(audit["native_applied_matches_previous_issued"] == expected["native_command_rows"]
            and same <= audit["native_applied_matches_same_tick_issued"] <= same + len(commands),
            "native actuator lag receipt contradicts the public command tape")
    return {k: expected[k] for k in ("episodes", "native_intervals", "native_transition_ticks_per_clock", "packet_pairs")}


def _check_prepare(plan, manifest):
    _check_stages(plan, manifest, ("prepare",))
    audit = read(PUBLIC / "audit.json")
    require(audit["status"] == "COMPLETE" and audit["protocol_sha256"] == PLAN_SHA
            and set(audit["splits"]) == {"train", "holdout"}
            and all(audit[k] == 0 for k in ("head_updates", "policy_updates", "physics_substeps")),
            "unexpected preparation status or work")
    return {split: _audit_tape(plan, split, audit["splits"][split]) for split in ("train", "holdout")}


def _check_fit(plan):
    from tools56.fit56 import array_sha256
    from tools103.fit103 import batch_schedule

    folder = PUBLIC / "fit"
    fit, parent = read(folder / "fit.json"), read(PARENT / "fit/fit.json")
    log = folder / "updates.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    n = plan["fit"]["updates_per_arm"]
    require(fit["status"] == "COMPLETE" and fit["seeds"] == plan["seeds"] and fit["arms"] == plan["arms"]
            and set(fit["per_seed"]) == {str(s) for s in plan["seeds"]}
            and fit["train_sha256"] == digest(PARENT / "train.npz")
            and fit["train_tape_sha256"] == digest(PUBLIC / "train_actions.npz")
            and fit["encoder_fit_sha256"] == digest(PARENT / "fit/fit.json")
            and fit["update_log_sha256"] == digest(log)
            and len(events) == fit["updates"] == plan["fit"]["total_head_updates"], "incomplete fitting receipts")
    with np.load(PARENT / "train.npz", allow_pickle=False) as data:
        stream = data["series_id"]
        training_rows = np.flatnonzero(stream[:-1] == stream[1:])
    for seed in plan["seeds"]:
        entry = fit["per_seed"][str(seed)]
        batches = training_rows[batch_schedule(len(training_rows), seed)]
        batch_hashes = [array_sha256(batch) for batch in batches]
        require(entry["batch_schedule_sha256"] == array_sha256(batches)
                and entry["encoder_sha256"] == parent["per_seed"][str(seed)]["arms"]["predictive"]["sha256"]
                and set(entry["arms"]) == set(plan["arms"]), "wrong frozen encoder, schedule or model pair")
        for arm in plan["arms"]:
            head = entry["arms"][arm]
            rows = [e for e in events if (e["seed"], e["arm"]) == (seed, arm)]
            require(head["file"] == f"seed_{seed}_{arm}.pt" and head["sha256"] == digest(folder / head["file"])
                    and head["updates"] == n and [r["update"] for r in rows] == list(range(1, n + 1))
                    and [r["batch_sha256"] for r in rows] == batch_hashes
                    and all(np.isfinite([r["depth_mse"], r["proprio_mse"], r["gradient_norm"]]).all() for r in rows),
                    f"incomplete or changed seed {seed} {arm} fit")


def _check_full(plan, manifest):
    _check_stages(plan, manifest, STAGES)
    prep = read(PUBLIC / "prepare_manifest.json")
    require(manifest["prepare_manifest_sha256"] == digest(PUBLIC / "prepare_manifest.json")
            and manifest["stages"]["prepare"] == prep["stages"]["prepare"], "wrong public preparation")
    _check_prepare(plan, prep)
    _check_fit(plan)
    packets, forecast = PARENT / "holdout.npz", PUBLIC / "forecast.npz"
    report = read(PUBLIC / "report.json")
    hashes = dict(fit_sha256=digest(PUBLIC / "fit/fit.json"), holdout_sha256=digest(packets),
                  tape_sha256=digest(PUBLIC / "holdout_actions.npz"), forecast_sha256=digest(forecast))
    counts = dict(head_updates=plan["fit"]["total_head_updates"], policy_updates=0, encoder_updates=0,
                  dynamics_updates=0, native_decisions=0, physics_substeps=0, new_sensor_rays=0)
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA and report["counts"] == counts
            and all(report[key] == value for key, value in hashes.items()), "forecast sources or work counts differ")
    with np.load(forecast, allow_pickle=False) as data:
        require(data["fit_sha256"].item() == hashes["fit_sha256"] and data["holdout_tape_sha256"].item() == hashes["tape_sha256"],
                "different fitted heads or development actions")
    measured = score(packets, forecast)
    require(report["score"] == measured, "public forecasts do not reproduce the sealed scores")
    return dict(passing_seeds=measured["passing_seeds"], paired_rows=measured["paired_rows"], counts=counts)


def replay_heads():
    import torch
    from tools103.fit103 import predict

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with TemporaryDirectory(prefix="d1-time103-check-") as tmp:
        new = predict(PARENT / "holdout.npz", PUBLIC / "holdout_actions.npz", PUBLIC / "fit", Path(tmp) / "forecast.npz")
        with np.load(new, allow_pickle=False) as replay, np.load(PUBLIC / "forecast.npz", allow_pickle=False) as sealed:
            require(set(replay.files) == set(sealed.files), "different forecast fields")
            for key in sealed.files:
                a, b = replay[key], sealed[key]
                require(a.shape == b.shape and a.dtype == b.dtype, f"changed {key} forecast format")
                if a.dtype.kind == "f":
                    np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-5, equal_nan=False, err_msg=key)
                else:
                    np.testing.assert_array_equal(a, b, err_msg=key)


def verify(prepare_only=False, replay=False):
    plan = protocol()
    if prepare_only:
        return _check_prepare(plan, read(PUBLIC / "prepare_manifest.json"))
    result = _check_full(plan, read(PUBLIC / "manifest.json"))
    if replay:
        replay_heads()
        result["replayed_heads"] = len(plan["seeds"]) * len(plan["arms"])
    return result


def edge_errors():
    """Post-hoc depth errors when the delivered wheel-track query changes height."""
    packet_path, forecast_path = PARENT / "holdout.npz", PUBLIC / "forecast.npz"
    report = read(PUBLIC / "report.json")
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA
            and report["holdout_sha256"] == digest(packet_path) and report["forecast_sha256"] == digest(forecast_path),
            "edge readback requires the sealed development predictions")
    with np.load(packet_path, allow_pickle=False) as p, np.load(forecast_path, allow_pickle=False) as f:
        ids, heights = f["row_id"], p["target_mm"]
        require(np.array_equal(f["arms"], ["past", "transition"])
                and np.array_equal(p["series_id"][ids], p["series_id"][ids + 1]), "misaligned forecast intervals")
        masked = p["visibility"][ids] == "masked"
        change = heights[ids] != heights[ids + 1]
        current_native = np.column_stack((p["episode"][ids], p["capture_tick"][ids]))
        target = f["target"][:, None, :, :32].astype(np.float64)
        losses = np.square(f["prediction"][..., :32].astype(np.float64) - target).mean(axis=-1)
        result = {}
        for label, selected in (("changed", masked & change), ("same", masked & ~change)):
            mse = losses[:, :, selected].mean(axis=-1)
            by_seed = {}
            for seed, (past, transition) in zip(f["seeds"], mse, strict=True):
                change_pct = float(100 * (transition / past - 1)) if past > 0 else None
                by_seed[str(seed)] = dict(past=float(past), transition=float(transition), change_pct=change_pct)
            result[label] = dict(pairs=int(selected.sum()), depth_mse_by_seed=by_seed,
                                 native_intervals=len(np.unique(current_native[selected], axis=0)))
    return result


def _save_manifest(name, value):
    with TemporaryDirectory(prefix="d1-time103-manifest-") as tmp:
        path = Path(tmp) / name
        write(path, value)
        return copy_verified(path, PUBLIC / name)


def publish(prepare_only=False):
    plan = protocol()
    stages = ("prepare",) if prepare_only else STAGES
    completed_stages = {stage: completed(stage) for stage in stages}
    records = {}
    for stage, (folder, usage) in completed_stages.items():
        records[stage] = _record(folder, usage)
        for relative, name in _files(stage, plan).items():
            dest = PUBLIC / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_verified(folder / relative, dest)
    prep = dict(protocol_sha256=PLAN_SHA, stages={"prepare": records["prepare"]})
    prep_result = _check_prepare(plan, prep)
    prep_hash = _save_manifest("prepare_manifest.json", prep)
    if prepare_only:
        return prep_result
    manifest = dict(protocol_sha256=PLAN_SHA, prepare_manifest_sha256=prep_hash, stages=records)
    result = _check_full(plan, manifest)
    _save_manifest("manifest.json", manifest)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true", help="publish the completed command tapes and clock audit")
    modes.add_argument("--check-prepare", action="store_true", help="verify the command tapes without any fitted head")
    modes.add_argument("--check", action="store_true", help="verify every published stage and all registered scores")
    modes.add_argument("--edge-errors", action="store_true", help="describe masked errors at changing wheel-track heights")
    parser.add_argument("--replay-heads", action="store_true", help="run the six frozen public heads during --check")
    args = parser.parse_args()
    if args.replay_heads and not args.check:
        parser.error("--replay-heads requires --check")
    if args.edge_errors:
        output = edge_errors()
    elif args.check or args.check_prepare:
        output = verify(args.check_prepare, args.replay_heads)
    else:
        output = publish(args.prepare_only)
    print(json.dumps(output, allow_nan=False, ensure_ascii=False))
