"""Replay the two occluder positions on the same D1 arrivals and frozen estimators."""
from __future__ import annotations

from copy import copy
from itertools import product

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.readback100 import independent_estimate
from tools100.score100 import TOLERANCE_MM, estimate_delta_mm
from tools101.common101 import PUBLIC as SOURCE
from tools101.score101 import METADATA, paired, qualifying, validate_holdout
from tools102.mask102 import calibration, level_points, packet_masks


ARMS = ("baseline", "predictive")
PREDICTIONS = ("original_h3_mm", "moved_h3_mm", "moved_no_past_depth_mm")
PUBLIC_FILES = ("holdout.npz", "predictions.npz", "fit/fit.json")


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def source_hashes(seeds):
    manifest = read(SOURCE / "manifest.json")
    names = [*PUBLIC_FILES, *(f"fit/seed_{s}_{arm}.pt" for s in seeds for arm in ARMS)]
    hashes = {name: digest(SOURCE / name) for name in names}
    for name, sha in hashes.items():
        require(manifest["files"][str((SOURCE / name).relative_to(ROOT))] == sha,
                f"registered round-101 source differs: {name}")
    return hashes


def roster(parent, plan):
    validate_holdout({k: parent[k] for k in METADATA})
    require(set(parent["source_seed"]) == {plan["source_seed"]}, "source differs from round-102 plan")
    index = {(int(e), str(l), str(v), int(t)): i for i, (e, l, v, t) in
             enumerate(zip(*(parent[k] for k in ("episode", "layout", "visibility", "capture_tick")), strict=True))}
    clear, original, sides, streams, slots = [], [], [], [], []
    for stream, (episode, layout, side) in enumerate(product(plan["episodes"], plan["layouts"], plan["target_sides"])):
        for slot, tick in enumerate(plan["capture_ticks"]):
            clear.append(index[episode, layout, "clear", tick])
            original.append(index[episode, layout, "masked", tick])
            sides.append(side)
            streams.append(stream)
            slots.append(slot)
    clear, original = np.array(clear), np.array(original)
    require(len(clear) == 320 and np.array_equal(parent["cloud_id"][clear], parent["cloud_id"][original]),
            "paired positions do not refer to the same full cloud")
    info = {key: parent[key][clear] for key in
            ("cloud_id", "episode", "layout", "capture_tick", "delivery_tick", "target_mm", "query_xy")}
    return dict(**info, clear_row=clear, masked_row=original, mask_side=np.array(sides),
                series_id=np.array(streams), slot=np.array(slots))


def geometry_outputs(points, valid, query_xy, estimator, independent=False):
    estimates = np.full(len(query_xy), np.nan)
    nearest_mm = np.full((len(query_xy), 2), np.nan)
    for i, (xyz, keep, query) in enumerate(zip(points, valid, query_xy, strict=True)):
        retained = xyz[keep]
        if len(retained):
            distances = np.linalg.norm(retained[:, None, :2] - query[None, :, :], axis=2)
            nearest_mm[i] = distances.min(axis=0) * 1000.
        value = (independent_estimate(retained, query, estimator) if independent
                 else estimate_delta_mm(retained, query))
        if value is not None:
            estimates[i] = value
    return estimates, nearest_mm


def build_packets(parent, plan, independent=False):
    sensor = read(ROOT / "docs/experiments100/pointcloud_density.json")
    calibrated = calibration(sensor["geometry"])
    for key in ("window_shape", "original_starts", "moved_starts", "active_mod4"):
        require(plan["mask"][key] == calibrated[key], f"registered mask calibration differs: {key}")
    packet = roster(parent, plan)
    before, moved, active = packet_masks(packet["capture_tick"], packet["mask_side"], sensor["geometry"])
    src = packet["clear_row"]
    valid = parent["valid"][src]
    points = level_points(parent["geometry_depth_m64"][src], valid, parent["capture_body_to_level"][src], sensor["geometry"])
    packet["active"] = active
    for name, blocked in (("original", before), ("moved", moved)):
        require(blocked.shape == valid.shape and blocked.dtype == bool and (blocked.sum(axis=1) == 36).all(),
                "changing position must preserve the 36 masked directions")
        keep = ~blocked
        packet[f"{name}_keep"] = keep
        packet[f"{name}_geometric_mm"], packet[f"{name}_nearest_mm"] = geometry_outputs(
            points, valid & keep, packet["query_xy"], sensor["estimator"], independent)
    require(np.array_equal(active, packet["slot"] % 4 >= 2) and active.sum() == 160
            and np.array_equal(before[~active], moved[~active]) and np.all(np.any(before[active] != moved[active], axis=1))
            and np.array_equal(valid & ~before, parent["valid"][packet["masked_row"]])
            and ((valid & ~before).sum(axis=1) == (valid & ~moved).sum(axis=1)).all(),
            "mask timing, control visibility or retained ray count differs")
    np.testing.assert_allclose(packet["original_geometric_mm"], parent["geometric_mm"][packet["masked_row"]],
                               rtol=0, atol=1e-8, equal_nan=True)
    return packet


def archived_predictions(packet, seeds, source):
    require(np.array_equal(source["seeds"], seeds), "parent fitted seeds differ")
    # [fit seed, training arm, physical cloud crossed with target side].
    return np.stack([source[f"{arm}_mm"][:, packet["masked_row"]] for arm in ARMS], axis=1)


def remove_past_depth(packets):
    """Keep each current packet, both previous action histories and all IMU/joint slots."""
    import torch

    n = len(packets.history)
    result = copy(packets)
    # The extra half contains the identical observation except that its range
    # sensor returned no rays. Only past lookup slots may point into this half.
    result.depth = torch.cat((packets.depth, torch.zeros_like(packets.depth)))
    result.valid = torch.cat((packets.valid, torch.zeros_like(packets.valid)))
    result.proprio = torch.cat((packets.proprio, packets.proprio))
    result.actions = torch.cat((packets.actions, packets.actions))
    result.history = packets.history.clone()
    previous = result.history[:, :-1]
    previous[previous >= 0] += n
    return result


def frozen_predictions(packet, parent, plan, source, progress=None):
    import torch
    from tools101.fit101 import DeliveredPackets, _estimates, _load_model

    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    src = packet["clear_row"]
    valid = parent["valid"][src] & packet["moved_keep"]
    # Strict student whitelist: query positions, layout IDs and target heights
    # are used by the scorer and never passed to the models.
    obs = DeliveredPackets(dict(series_id=packet["series_id"], capture_tick=packet["capture_tick"],
                                delivery_tick=packet["delivery_tick"], valid=valid,
                                depth_m=np.where(valid, parent["depth_m"][src], np.float32(0.)),
                                action_history=parent["action_history"][src], proprio=parent["proprio"][src]))
    ablated = remove_past_depth(obs)
    seeds = plan["seeds"]
    original = archived_predictions(packet, seeds, source)
    full, no_depth = np.empty_like(original), np.empty_like(original)
    fit = read(SOURCE / "fit/fit.json")
    require(fit["status"] == "COMPLETE" and fit["seeds"] == seeds and fit["updates"] == 300,
            "checkpoint family incomplete")
    if progress is not None:
        progress.mkdir()
    for i, seed in enumerate(seeds):
        for j, arm in enumerate(ARMS):
            model = _load_model(SOURCE / "fit", fit, seed, arm)
            for tag, observations, output in (("h3", obs, full), ("no_past_depth", ablated, no_depth)):
                output[i, j] = _estimates(model, observations)
                if progress is not None:
                    # Each completed inference survives a later failure in this attempt.
                    arrays(progress / f"{seed}_{arm}_{tag}.npz", dict(estimate_mm=output[i, j]))
    # Before the first relocation within a sequence all past and current
    # inputs match the public parent, irrespective of the chosen target side.
    prefix = packet["slot"] < 2
    np.testing.assert_allclose(full[:, :, prefix], original[:, :, prefix], atol=.0002, rtol=0, equal_nan=True)
    first = packet["slot"] == 0
    np.testing.assert_allclose(full[:, :, first], no_depth[:, :, first], atol=.0002, rtol=0, equal_nan=True)
    return dict(seeds=np.array(seeds), arms=np.array(ARMS), original_h3_mm=original,
                moved_h3_mm=full, moved_no_past_depth_mm=no_depth)


def compare(truth, a, b, names):
    values = paired(truth, a, b)
    result = {name: values.pop(key) for name, key in zip(names, ("baseline", "predictive"), strict=True)}
    for name, key in zip(names, ("baseline_mae_both_mm", "predictive_mae_both_mm"), strict=True):
        result[f"{name}_paired_mae_mm"] = values.pop(key)
    return {**result, **values}


def geometry_statistics(packet, plan):
    truth, old, new = (packet[k] for k in ("target_mm", "original_geometric_mm", "moved_geometric_mm"))
    active, side, episode = (packet[k] for k in ("active", "mask_side", "episode"))
    qualifies_old = qualifying(truth, old)
    qualifies_new = qualifying(truth, new)
    damaged = active & qualifies_old & ~qualifies_new
    lost = active & np.isfinite(old) & ~np.isfinite(new)
    changed = np.isfinite(old) & np.isfinite(new) & (np.abs(old - new) >= plan["gates"]["change_mm"])
    on_high = ((truth > TOLERANCE_MM) & (side == "left")) | ((truth < -TOLERANCE_MM) & (side == "right"))
    episode_ids = np.unique(episode[damaged]).tolist()
    damaged_sides = sorted(set(side[damaged].tolist()))
    checks = dict(affected_episodes=len(episode_ids) >= plan["gates"]["minimum_episodes"],
                  both_target_sides=damaged_sides == sorted(plan["target_sides"]))
    require(np.array_equal(old[~active], new[~active], equal_nan=True), "inactive geometry changed")
    j = (side == "right").astype(int)
    nearest_old = packet["original_nearest_mm"][np.arange(len(side)), j]
    nearest_new = packet["moved_nearest_mm"][np.arange(len(side)), j]

    def cell(use):
        detected = use & active & (lost | changed)
        values = compare(truth[use], old[use], new[use], ("original", "moved"))
        nearest = use & np.isfinite(nearest_old) & np.isfinite(nearest_new)
        nearest_change = nearest_new[nearest] - nearest_old[nearest]
        values.update(physical_clouds=len(np.unique(packet["cloud_id"][use])),
                      material_change_or_loss=int(detected.sum()), lost_estimate=int((lost & use).sum()),
                      nearest_paired_count=int(nearest.sum()),
                      nearest_original_mean_mm=float(nearest_old[nearest].mean()) if nearest.any() else None,
                      nearest_moved_mean_mm=float(nearest_new[nearest].mean()) if nearest.any() else None,
                      nearest_change_median_mm=float(np.median(nearest_change)) if nearest.any() else None)
        return values

    result = dict(challenge_established=all(checks.values()), checks=checks, damaged_episodes=episode_ids,
                  damaged_target_sides=damaged_sides, all=cell(np.ones(len(old), dtype=bool)),
                  active=cell(active), inactive=cell(~active), active_true_high_on_target=cell(active & on_high),
                  by_episode=[dict(episode=int(e), **cell(active & (episode == e))) for e in plan["episodes"]],
                  by_target_side={s: cell(active & (side == s)) for s in plan["target_sides"]},
                  true_high_on_target_by_side={s: cell(active & on_high & (side == s)) for s in plan["target_sides"]})
    return result, damaged, on_high


def score(packet, predictions, plan):
    truth, active = packet["target_mm"], packet["active"]
    seeds = plan["seeds"]
    require(np.array_equal(predictions["seeds"], seeds) and np.array_equal(predictions["arms"], ARMS),
            "model identities or training arms differ")
    for name in PREDICTIONS:
        value = predictions[name]
        require(value.shape == (len(seeds), len(ARMS), len(truth))
                and value.dtype.kind == "f" and np.isfinite(value).all(),
                f"incomplete frozen inference: {name}")
    geometry, damaged, on_high = geometry_statistics(packet, plan)
    scopes = dict(all=np.ones(len(truth), dtype=bool), active=active, inactive=~active,
                  active_true_high_on_target=active & on_high, geometric_regressions=damaged,
                  geometric_regressions_true_high_on_target=damaged & on_high)
    models = []
    for i, seed in enumerate(seeds):
        by_arm = {}
        for j, arm in enumerate(ARMS):
            old, new, no_past = [predictions[name][i, j] for name in PREDICTIONS]
            arm_report = {}
            for group, selected in scopes.items():
                arm_report[group] = {
                    "position": compare(truth[selected], old[selected], new[selected], ("original_h3", "moved_h3")),
                    "past_depth": compare(truth[selected], no_past[selected], new[selected], ("ablated", "full"))}
            by_arm[arm] = arm_report
        base, predictive = predictions["moved_h3_mm"][i]
        models.append(dict(seed=seed, arms=by_arm,
                           moved_training_loss_comparison={group: paired(truth[m], base[m], predictive[m])
                                                          for group, m in scopes.items()}))
    return dict(geometry=geometry, models=dict(by_seed=models),
                physical_clouds=len(np.unique(packet["cloud_id"])), paired_rows=len(truth),
                active_paired_rows=int(active.sum()), active_physical_clouds=len(np.unique(packet["cloud_id"][active])))


def replay(folder, plan):
    from tools102.common102 import PLAN_SHA

    hashes = source_hashes(plan["seeds"])
    parent, forecasts = (read_npz(SOURCE / n) for n in ("holdout.npz", "predictions.npz"))
    packet = build_packets(parent, plan)
    arrays(folder / "packets.npz", packet)
    predicted = frozen_predictions(packet, parent, plan, forecasts, progress=folder / "model_outputs")
    arrays(folder / "predictions.npz", predicted)
    copies = predicted["original_h3_mm"].size
    clocks = dict(physics_substeps=0, native_decisions=0, new_rays=0, world_updates=0, policy_updates=0,
                  perception_updates=0, archived_model_estimates=copies, new_model_estimates=2 * copies)
    require(copies == plan["budget"]["archived_estimates_mapped"]
            and 2 * copies == plan["budget"]["new_frozen_estimates"], "new or copied inference budget differs")
    report = dict(status="COMPLETE", protocol_sha256=PLAN_SHA, parent_sha256=hashes, clocks=clocks,
                  packets_sha256=digest(folder / "packets.npz"), predictions_sha256=digest(folder / "predictions.npz"),
                  score=score(packet, predicted, plan))
    write(folder / "report.json", report)
    return report


def readback(folder, plan, replay_models=False):
    """Rebuild the paired inputs, independent geometric readouts and all gates from the public 101 archive."""
    from tools102.common102 import PLAN_SHA

    report = read(folder / "report.json")
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA
            and report["parent_sha256"] == source_hashes(plan["seeds"])
            and report["packets_sha256"] == digest(folder / "packets.npz")
            and report["predictions_sha256"] == digest(folder / "predictions.npz"), "broken replay provenance")
    parent, forecasts = (read_npz(SOURCE / n) for n in ("holdout.npz", "predictions.npz"))
    expected = build_packets(parent, plan, independent=True)
    packet = read_npz(folder / "packets.npz")
    require(set(expected) == set(packet), "changed packet schema")
    for key, value in expected.items():
        if key in ("original_geometric_mm", "moved_geometric_mm", "original_nearest_mm", "moved_nearest_mm"):
            np.testing.assert_allclose(packet[key], value, rtol=0, atol=1e-8, equal_nan=True)
        else:
            np.testing.assert_array_equal(packet[key], value)
    predictions = read_npz(folder / "predictions.npz")
    old = archived_predictions(packet, plan["seeds"], forecasts)
    np.testing.assert_array_equal(old, predictions["original_h3_mm"])
    require(report["clocks"] == dict(physics_substeps=0, native_decisions=0, new_rays=0, world_updates=0,
                                     policy_updates=0, perception_updates=0, archived_model_estimates=1920,
                                     new_model_estimates=3840)
            and score(packet, predictions, plan) == report["score"], "budget or recomputed grading differs")
    if replay_models:
        fresh = frozen_predictions(packet, parent, plan, forecasts)
        for key in PREDICTIONS:
            np.testing.assert_allclose(predictions[key], fresh[key], rtol=0, atol=.0002, equal_nan=True)
    return dict(verified_rows=len(old[0, 0]), archived_estimates=old.size,
                verified_model_estimates=2 * old.size if replay_models else 0)
