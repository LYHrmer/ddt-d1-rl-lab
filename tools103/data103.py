"""Recover issued D1 commands on the capture and delivery clocks of round 101."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read
from tools101.common101 import PLAN_SHA as PARENT_PLAN_SHA, PUBLIC as PARENT


MANIFEST_SHA = "3e983f54715c33613bae2a13c7ed3723ab37d05781795ecd0b82b4aaf7e4724d"
# Independently extracted from the previous-action field of the *next* native
# observation in roster order, little-endian float32. This covers intermediate
# decisions that the public 6 Hz packet history does not itself contain.
ACTION_SHA = {
    "train": "52237433aa340ef30aca4a9fcbbd009280237c716d5e9ad8165eefd7c12c2c37",
    "holdout": "bf4ccde012fdbf2b07fa42d8ecbd1eeea7ab8f524f9cb9fae66733953546b471",
}
TAPE_TICKS = np.arange(16, 152, dtype=np.int64)


def _catalog(split, packets_path):
    require(split in ACTION_SHA, "unknown parent split")
    manifest_path = PARENT / "manifest.json"
    require(digest(manifest_path) == MANIFEST_SHA, "changed round-101 manifest")
    manifest = read(manifest_path)
    catalog_path = PARENT / "data.json"
    catalog_sha = manifest["files"]["reports/jeplo101/data.json"]
    require(manifest["protocol_sha256"] == PARENT_PLAN_SHA and digest(catalog_path) == catalog_sha,
            "changed round-101 acquisition record")
    data = read(catalog_path)
    info = data["splits"][split]
    expected = manifest["files"][f"reports/jeplo101/{split}.npz"]
    require(data["status"] == "COMPLETE" and digest(packets_path) == expected == info["sha256"],
            "partial acquisition or wrong packet split")
    roster = info["episode_roster"]
    other = data["splits"]["holdout" if split == "train" else "train"]["episode_roster"]
    for field in ("source_seed", "physical_episode_qpos_sha256"):
        require({e[field] for e in roster}.isdisjoint(e[field] for e in other), "native splits overlap")
    return data, info, catalog_sha


def _metadata(split, data, info, catalog_sha):
    roster = info["episode_roster"]
    sources = {s["source_seed"]: s for s in data["sources"]}
    roster_bytes = json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
    return dict(
        split=np.asarray(split), ticks=TAPE_TICKS, source_seed=np.array([e["source_seed"] for e in roster]),
        episode=np.array([e["episode"] for e in roster]), first_archive_row=np.array([e["first_archive_row"] for e in roster]),
        trace_sha256=np.array([sources[e["source_seed"]]["trace_sha256"] for e in roster]),
        native_report_sha256=np.array([sources[e["source_seed"]]["native_report_sha256"] for e in roster]),
        manifest_sha256=np.asarray(MANIFEST_SHA), catalog_sha256=np.asarray(catalog_sha),
        roster_sha256=np.asarray(sha256(roster_bytes).hexdigest()), packets_sha256=np.asarray(info["sha256"]),
    )


def _applied_counts(issued, applied, rows):
    """Count 16-component row matches on the actuator clock for the admitted tape."""
    require(applied.shape == issued.shape and (rows > 0).all(), "invalid source actuator clock")
    actual = applied[rows]
    previous = int(np.all(actual == issued[rows - 1], axis=1).sum())
    same = int(np.all(actual == issued[rows], axis=1).sum())
    require(previous == len(rows), "applied action differs from the one-tick-delayed issued command")
    return previous, same


def _native_actions(source):
    path = ROOT / source["path"]
    receipt_path = path.with_name("report.json")
    require(digest(path) == source["trace_sha256"] and digest(receipt_path) == source["native_report_sha256"],
            "changed native training trace or receipt")
    record = read(receipt_path)
    require(record["status"] == "COMPLETE" and record["training_seed"] == source["source_seed"]
            and record["parent_seed"] == source["parent_seed"] and record["arm"] == "mixed"
            and record["trace"]["completed_recorded_ticks"] == record["budget"] == 4096,
            "incomplete native caller")
    with np.load(path, allow_pickle=False) as trace:
        issued, applied, next_observations = trace["issued"], trace["applied"], trace["proprio627"]
        terminated, truncated = trace["terminated"], trace["truncated"]
        terrain, ticks, poses = trace["terrain"], trace["episode_tick"], trace["qpos"]
    done = terminated | truncated
    starts = np.flatnonzero(ticks == 1)
    require(issued.shape == (4096, 16) and issued.dtype == np.float32 and next_observations.shape == (4096, 627)
            and np.array_equal(starts, np.r_[0, np.flatnonzero(done[:-1]) + 1]), "invalid source clocks")
    by_episode = {}
    previous_matches = same_matches = 0
    for item in source["admitted_flat_episodes"]:
        episode, start, stop = item["episode"], item["first_archive_row"], item["end_archive_row_exclusive"]
        require(start == starts[episode] and stop - start == 300 and not done[start:stop - 1].any()
                and truncated[stop - 1] and not terminated[stop - 1] and not terrain[start:stop].any()
                and np.array_equal(ticks[start:stop], np.arange(1, 301))
                and sha256(poses[start:stop].tobytes()).hexdigest() == item["physical_episode_qpos_sha256"],
                "partial or misidentified native episode")
        rows = start + TAPE_TICKS
        previous, same = _applied_counts(issued, applied, rows)
        previous_matches += previous
        same_matches += same
        actions = issued[rows]
        require(np.array_equal(actions, next_observations[rows + 1, 41:57]),
                "issued action disagrees with the next pre-action observation")
        by_episode[(source["source_seed"], episode)] = actions
    return by_episode, previous_matches, same_matches


def _check_actions(actions, split, n_episodes):
    require(actions.dtype == np.float32 and actions.shape == (n_episodes, len(TAPE_TICKS), 16)
            and np.isfinite(actions).all() and (np.abs(actions) <= 1.).all(),
            "invalid native 16-D command tape")
    require(sha256(np.ascontiguousarray(actions, dtype="<f4").tobytes()).hexdigest() == ACTION_SHA[split],
            "issued commands differ from independently recorded previous actions")


def _packet_rows(packets_path, data, info, actions):
    """Bind each public row to its actual episode and test its two previously issued actions."""
    roster = info["episode_roster"]
    captures, delay = np.array(data["capture_ticks"]), data["delay_ticks"]
    views = [(i, layout, visibility) for i in range(len(roster)) for layout in info["layouts"]
             for visibility in ("clear", "masked")]
    series = np.repeat(np.arange(len(views)), len(captures))
    episode_index = np.array([i for i, _, _ in views])[series]
    frame_index = np.tile(np.arange(len(captures)), len(views))
    capture = captures[frame_index]
    with np.load(packets_path, allow_pickle=False) as packets:
        require(np.array_equal(packets["row_id"], np.arange(info["rows"]))
                and np.array_equal(packets["series_id"], series) and np.array_equal(packets["capture_tick"], capture)
                and delay == 2 and np.array_equal(packets["delivery_tick"], capture + delay), "wrong packet clocks")
        for field in ("source_seed", "episode"):
            require(np.array_equal(packets[field], np.array([e[field] for e in roster])[episode_index]),
                    f"wrong native {field} for a packet view")
        require(np.array_equal(packets["layout"], np.array([layout for _, layout, _ in views])[series])
                and np.array_equal(packets["visibility"], np.array([vis for _, _, vis in views])[series]),
                "missing or reordered mirrored packet views")
        history_index = capture[:, None] + np.arange(delay) - TAPE_TICKS[0]
        require(np.array_equal(packets["action_history"], actions[episode_index[:, None], history_index]),
                "issued actions differ from the packet's archived delivery history")
    return episode_index, frame_index


def _windows(actions, captures, delay):
    """[episode, packet, past/transition, depth/body, position, 16 commands + validity]."""
    result = np.zeros((len(actions), len(captures), 2, 2, 9, 17), dtype=np.float32)
    for frame, (capture, following) in enumerate(zip(captures[:-1], captures[1:], strict=True)):
        gap = int(following - capture)
        require(gap in (8, 9), "unexpected interval between new packets")
        for clock, tick in enumerate((capture, capture + delay)):
            for arm, first in enumerate((tick - gap, tick)):
                offset = int(first - TAPE_TICKS[0])
                require(0 <= offset and offset + gap <= actions.shape[1], "action window crosses the native tape")
                result[:, frame, arm, clock, :gap, :16] = actions[:, offset:offset + gap]
                result[:, frame, arm, clock, :gap, 16] = 1
    return result


def build_archive(split, destination):
    """Extract one compact public command tape from complete native episodes, exclusive create."""
    packets_path = PARENT / f"{split}.npz"
    data, info, catalog_sha = _catalog(split, packets_path)
    metadata = _metadata(split, data, info, catalog_sha)
    episodes = {}
    previous_matches = same_matches = 0
    for source in data["sources"]:
        if source["source_seed"] in metadata["source_seed"]:
            source_episodes, previous, same = _native_actions(source)
            episodes.update(source_episodes)
            previous_matches += previous
            same_matches += same
    keys = list(zip(metadata["source_seed"], metadata["episode"], strict=True))
    require(set(keys) == set(episodes) and len(keys) == len(episodes), "incomplete split episode roster")
    actions = np.stack([episodes[key] for key in keys])
    _check_actions(actions, split, len(keys))
    _packet_rows(packets_path, data, info, actions)
    arrays(destination, dict(actions=actions, **metadata))
    gaps = np.diff(data["capture_ticks"])
    return dict(split=split, episodes=len(keys), source_seeds=np.unique(metadata["source_seed"]).tolist(),
                source_rows=int(info["rows"]), verified_two_action_histories=int(info["rows"]),
                native_intervals=int(len(keys) * len(gaps)), native_transition_ticks_per_clock=int(len(keys) * gaps.sum()),
                native_command_rows=int(actions.shape[0] * actions.shape[1]), packet_pairs=int(info["series"] * len(gaps)),
                forecast_action_column="issued", native_applied_lag_policy_ticks=1, native_applied_lag_ms=20,
                native_applied_matches_previous_issued=previous_matches, native_applied_matches_same_tick_issued=same_matches,
                terminal_rows=int(info["series"]), tape_sha256=digest(destination), actions_sha256=ACTION_SHA[split],
                roster_sha256=metadata["roster_sha256"].item(), packets_sha256=info["sha256"], catalog_sha256=catalog_sha)


def window_inputs(split, packets_path, tape_path):
    """Reconstruct both timing conditions from public files. The last row in every series stays empty."""
    data, info, catalog_sha = _catalog(split, packets_path)
    expected = _metadata(split, data, info, catalog_sha)
    with np.load(tape_path, allow_pickle=False) as tape:
        require(set(tape.files) == {*expected, "actions"} and all(
            np.array_equal(tape[name], value) for name, value in expected.items()), "wrong command tape or split")
        actions = tape["actions"]
    _check_actions(actions, split, len(expected["episode"]))
    episode_index, frame_index = _packet_rows(Path(packets_path), data, info, actions)
    windows = _windows(actions, np.array(data["capture_ticks"]), data["delay_ticks"])
    return windows[episode_index, frame_index]
