"""Build paired range histories at native pre-contact poses without advancing the plant."""
from __future__ import annotations

import hashlib
import math
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from tools91.data91 import FIT_KEYS, FIT_STAGE_SHA, PLAN89, PLAN90, _digest, _episode_pairs, _stage
from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.common100 import protocol as sensor_protocol
from tools100.readback100 import reference_delta_mm
from tools100.score100 import estimate_delta_mm
from training.d1_rl.interface import DEFAULT_Q, LEG_INDICES


EPISODES = {89011: (2, 5, 8, 11), 89023: (2, 5, 8, 11), 89037: (0, 3, 6, 9, 12)}
SOURCE = ROOT / "evidence/uphill89/training_01"


def reset_horizontal_pose(source_seed, episode, reset_frame, reset_config):
    """Reproduce the flat reset RNG. Vertical translation cannot change the wheel x extent."""
    rng = np.random.default_rng(np.random.SeedSequence([source_seed, 0, episode, 0]))
    offsets = np.zeros(16)
    offsets[LEG_INDICES] = rng.uniform(*reset_config["leg_joint_perturbation_uniform_rad"], len(LEG_INDICES))
    yaw = rng.uniform(*reset_config["yaw_perturbation_uniform_rad"])
    require(np.array_equal(reset_frame[9:25], offsets.astype(np.float32)),
            "reconstructed reset joints disagree with the original pre-action observation")
    return np.r_[0., 0., 0., math.cos(yaw / 2), 0., 0., math.sin(yaw / 2), DEFAULT_Q + offsets]


def archive_episodes(seed, stop_tick, registered, stage):
    folder = SOURCE / f"mixed_{seed}"
    record = next(item for item in stage["models"] if item["model"] == folder.name)
    receipt_path = folder / "report.json"
    _digest(receipt_path, record["report_sha256"])
    receipt = read(receipt_path)
    source = folder / "train_trace.npz"
    _digest(source, registered["trace_sha256"])
    require(receipt["status"] == "COMPLETE" and receipt["arm"] == "mixed"
            and receipt["training_seed"] == seed and receipt["parent_seed"] == registered["parent"]
            and receipt["protocol_sha256"] == stage["protocol_sha256"]
            and receipt["trace"]["completed_recorded_ticks"] == receipt["budget"] == 4096,
            f"incomplete native archive {folder.name}")
    with np.load(source, allow_pickle=False) as archive:
        native = {key: archive[key] for key in FIT_KEYS}
        actor = archive["proprio627"]
        start_x, slope = archive["start_x_m"], archive["slope_start"]
    require(actor.shape == (4096, 627) and np.isfinite(actor).all(), "invalid native pre-action observations")
    _, episodes, discarded = _episode_pairs(native, first_id=0)
    reset_config = read(ROOT / "training/configs/pilot03d.json")["reset"]
    selected, roster = [], []
    for episode in episodes:
        if episode["terrain"] != 0:
            continue
        idx, a, b = episode["id"], episode["start_row"], episode["end_row_exclusive"]
        require(b - a == 300 and episode["truncated"] and not episode["terminated"]
                and np.all(start_x[a:b] == 0) and not slope[a:b].any()
                and stop_tick < b - a, "unexpected native flat horizon or placement")
        frame = actor[a:a + stop_tick + 1, :57].copy()
        issued = native["issued"][a:a + stop_tick].copy()
        require(not frame[0, 41:57].any() and np.array_equal(frame[1:, 41:57], issued),
                "archived pre-action observations disagree with actually issued actions")
        reset = reset_horizontal_pose(seed, idx, frame[0], reset_config)
        pose_hash = hashlib.sha256(native["qpos"][a:b].tobytes()).hexdigest()
        roster.append(dict(source_seed=seed, episode=idx, first_archive_row=a, end_archive_row_exclusive=b,
                           physical_episode_qpos_sha256=pose_hash))
        selected.append(dict(source_seed=seed, episode=idx, reset=reset, frame=frame, issued=issued,
                             qpos=native["qpos"][a:a + stop_tick].copy()))
    require(tuple(v["episode"] for v in roster) == EPISODES[seed], f"flat episode roster changed in {folder.name}")
    provenance = dict(source_seed=seed, parent_seed=receipt["parent_seed"], path=str(source.relative_to(ROOT)),
                      trace_sha256=registered["trace_sha256"], native_report_sha256=record["report_sha256"],
                      complete_episode_count=len(episodes), excluded_incomplete_tail_rows=discarded,
                      admitted_flat_episodes=roster)
    return selected, provenance


def delivery_inputs(frame, issued, tick):
    """At boundary t, row t starts the upcoming interval. Rows t-2,t-1 have already issued."""
    require(2 <= tick < len(frame) and tick <= len(issued), "delivery or action history crosses the episode")
    observation = frame[tick]
    history = issued[tick - 2:tick]
    require(np.array_equal(history[-1], observation[41:57]), "last issued action has wrong time index")
    return np.r_[observation[:6], observation[9:41]], history


def prefix_bound(scene, episode, edge):
    bounds = [scene.pose(episode["reset"])["wheel_max_x_bound"]]
    bounds.extend(scene.pose(q)["wheel_max_x_bound"] for q in episode["qpos"])
    require(np.isfinite(bounds).all() and max(bounds) < edge, "native prefix reaches a counterfactual wheel contact")
    return dict(source_seed=episode["source_seed"], episode=episode["episode"], reset_wheel_max_x_m=bounds[0],
                first_tick=0, last_tick=len(bounds) - 1, wheel_max_x_m=max(bounds),
                max_at_tick=int(np.argmax(bounds)), minimum_layout_leading_edge_x_m=edge)


def packet_views(scene, episode, layout, side, index, ticks, delay, visibilities, raw_path, *, audit_depth=False):
    """Capture once per tick. Both angular masks refer to that one physical first return."""
    output = []
    with ZipFile(raw_path, mode="x", compression=ZIP_DEFLATED, compresslevel=3) as raw:
        for packet_number, tick in enumerate(ticks):
            # Native rows contain interval-end qpos, e.g. qpos[0] is boundary 1.
            scene.pose(episode["qpos"][tick - 1])
            transform = scene.data.xmat[scene.base].reshape(3, 3).T @ scene.heading
            scan = scene.capture()
            full_valid = np.isfinite(scan["points"]).all(axis=1)
            measured = scan["range_m"]
            require(measured.shape == scene.angle_keep.shape == (520,) and np.isfinite(measured).all()
                    and np.array_equal(full_valid, (measured >= scene.geometry["range_min_m"])
                                       & (measured <= scene.geometry["range_max_m"]) & ~scan["self_hit"]),
                    "native first-return range, validity or self-hit columns disagree")
            # This private file also retains the collected prefix of a failed case.
            rays = np.empty(len(measured), dtype=[("range_m", "<f8"), ("self_hit", "?")])
            rays["range_m"], rays["self_hit"] = measured, scan["self_hit"]
            with raw.open(f"tick_{tick:03d}.npy", "w") as entry:
                np.save(entry, rays, allow_pickle=False)
            delivery_tick = tick + delay
            delivery = scene.pose(episode["qpos"][delivery_tick - 1])
            require(scene.data.time == 0, "offline ray acquisition advanced the physics clock")
            body, actions = delivery_inputs(episode["frame"], episode["issued"], delivery_tick)
            truth = reference_delta_mm(scene.geometry, side, delivery["world_query_xy"])
            for view_index, visibility in enumerate(visibilities):
                valid = full_valid if visibility == "clear" else full_valid & scene.angle_keep
                geometric = estimate_delta_mm(scan["points"][valid], delivery["query_xy"])
                depth = np.where(valid, measured, 0.)
                row = dict(source_seed=episode["source_seed"], episode=episode["episode"],
                           layout=layout, side=side, visibility=visibility, capture_tick=tick,
                           delivery_tick=delivery_tick, series_id=index * len(visibilities) + view_index,
                           cloud_id=index * len(ticks) + packet_number, depth_m=depth.astype(np.float32), valid=valid,
                           proprio=body, action_history=actions, target_mm=np.float32(truth),
                           geometric_mm=np.nan if geometric is None else geometric,
                           query_xy=delivery["query_xy"], world_query_xy=delivery["world_query_xy"],
                           capture_body_to_level=transform)
                if audit_depth:
                    row["geometry_depth_m64"] = depth
                output.append(row)
    return output


def _layouts(specs, base_geometry):
    return [(f"{spec['id']}_{side}", side, {**base_geometry, **{k: v for k, v in spec.items() if k != "id"}})
            for spec in specs for side in ("left", "right")]


def _write_split(folder, split, episodes, plan, geometry):
    from tools100.sensor100 import Sensor100

    ticks, delay, visibilities = plan["capture_ticks"], plan["delay_ticks"], plan["visibilities"]
    layouts = _layouts(plan["data"]["layouts"][split], geometry)
    minimum_edge = min(g["box_center_x_m"] - g["box_half_x_m"] for _, _, g in layouts)
    rows, bounds = [], []
    native_files = {}
    for layout_index, (layout, side, fixture) in enumerate(layouts):
        scene = Sensor100({"geometry": fixture}, side)
        for episode_index, episode in enumerate(episodes):
            if layout_index == 0:
                bounds.append(prefix_bound(scene, episode, minimum_edge))
            source_seed, episode_id = episode["source_seed"], episode["episode"]
            index = episode_index * len(layouts) + layout_index
            raw_path = folder / "native" / f"{split}_{source_seed}_ep{episode_id:02d}_{layout}.npz"
            rows.extend(packet_views(scene, episode, layout, side, index, ticks, delay, visibilities, raw_path,
                                     audit_depth=split == "holdout"))
            native_files[raw_path.name] = digest(raw_path)
        del scene
    rows.sort(key=lambda row: (row["series_id"], row["capture_tick"]))
    subset = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    subset["row_id"] = np.arange(len(rows), dtype=np.int64)
    case_count = len(episodes) * len(layouts)
    series_count = case_count * len(visibilities)
    point_count = case_count * len(ticks)
    require(len(rows) == point_count * len(visibilities)
            and np.array_equal(subset["capture_tick"], np.tile(ticks, series_count))
            and np.array_equal(subset["series_id"], np.repeat(np.arange(series_count), len(ticks)))
            and np.all(np.bincount(subset["cloud_id"], minlength=point_count) == len(visibilities)),
            "paired physical capture or per-series scoring roster is incomplete")
    path = folder / f"{split}.npz"
    arrays(path, subset)
    return dict(rows=len(rows), clouds=point_count, series=series_count, episodes=len(episodes), sha256=digest(path),
                layouts=[name for name, _, _ in layouts], prefix_bounds=bounds,
                native_first_return_files_sha256=native_files)


def prepare(folder, plan):
    """Write one immutable training split and one independent whole-source/layout holdout."""
    from tools87.terrain87 import capture_due

    ticks, delay = plan["capture_ticks"], plan["delay_ticks"]
    expected = [t for t in range(25, 151) if capture_due(t)]
    require(ticks == expected and len(ticks) == 16 and delay == 2 and plan["visibilities"] == ["clear", "masked"],
            "registered native new-packet clocks or angular views differ")
    require(plan["data"]["source_seeds"] == {"train": [89011, 89023], "holdout": [89037]},
            "registered source partitions differ")
    geometry = sensor_protocol()["geometry"]
    parent_stage = _stage(SOURCE, FIT_STAGE_SHA, PLAN89)
    registered = {record["seed"]: record for record in read(PLAN90)["shadow"]["sources"]}
    (folder / "native").mkdir()
    sources, splits, used_poses = [], {}, set()
    for split in ("train", "holdout"):
        selected, roster = [], []
        for seed in plan["data"]["source_seeds"][split]:
            episodes, provenance = archive_episodes(seed, ticks[-1] + delay, registered[seed], parent_stage)
            selected.extend(episodes)
            sources.append(provenance)
            for native_episode in provenance["admitted_flat_episodes"]:
                fingerprint = native_episode["physical_episode_qpos_sha256"]
                require(fingerprint not in used_poses, "duplicate native episode across splits or sources")
                used_poses.add(fingerprint)
                roster.append(native_episode)
        splits[split] = _write_split(folder, split, selected, plan, geometry)
        splits[split]["episode_roster"] = roster
    clouds = sum(v["clouds"] for v in splits.values())
    new_views = sum(v["rows"] for v in splits.values())
    require(clouds == 672 and new_views == 1344, "physical clouds or matched views differ from registration")
    result = dict(schema=1, status="COMPLETE", parent_stage_sha256=FIT_STAGE_SHA, sources=sources, splits=splits,
                  capture_ticks=ticks, delay_ticks=delay,
                  counts=dict(physical_steps=0, policy_updates=0, world_updates=0, perception_updates=0,
                              independent_clouds=clouds, new_packet_views=new_views, rays_per_cloud=520,
                              native_ray_calls=2 * 520 * clouds))
    write(folder / "data.json", result)
    return result
