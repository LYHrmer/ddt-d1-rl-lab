"""Read adjacent 6 Hz packets with measured 50 Hz motion from the same D1 episode."""
from hashlib import sha256
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read
from tools101.common101 import PUBLIC as PACKETS, protocol
from tools103.data103 import MANIFEST_SHA, _catalog
from tools104.motion104 import measured_motion
from tools105.data105 import MEASUREMENTS, SPLITS, inputs


# The measured reference was computed from consecutive current observations, independently
# of the received history below. The oracle reference was selected by native episode clock.
# Encoding: each field, in order, as little-endian float64 C-order bytes in episode order.
MEASURED_SHA = "39e3d77ce251b5726990a3fa6f708ced821bd987b474fe9811396a7ea954fd3f"
ORACLE_SHA = "839c150de48a0857e8e44e47ea59a76d07b0a8ff8de29db6641ab5d3ead32959"
SENSOR_COLUMNS = (0, 1, 2, 3, 4, 5, 28, 32, 36, 40)


def _sources():
    plan = protocol()
    rosters, packet_shas = {}, []
    for split, parent in SPLITS.items():
        data, info, catalog_sha = _catalog(parent, PACKETS / f"{parent}.npz")
        rosters[parent] = info["episode_roster"]
        packet_shas.append(info["sha256"])
        seeds = sorted({e["source_seed"] for e in rosters[parent]})
        require(seeds == plan["data"]["source_seeds"][parent], f"incorrect {split} sources")
    sources = sorted(data["sources"], key=lambda s: s["source_seed"])
    episodes = [e for s in sources for e in s["admitted_flat_episodes"]]
    keys = [(e["source_seed"], e["episode"]) for e in episodes]
    require(keys == sorted(set(keys)) and len(keys) == 13 and all(
        [e for e in episodes if e["source_seed"] in plan["data"]["source_seeds"][parent]] == roster
        for parent, roster in rosters.items()), "native episode roster differs from the registered splits")
    require({str(s["source_seed"]): s["trace_sha256"] for s in sources} == plan["data"]["trace_sha256"],
            "native traces differ from the registered parents")
    ticks = np.asarray(data["capture_ticks"], dtype=np.int64)
    require(np.array_equal(ticks, plan["capture_ticks"]) and len(ticks) == 16
            and set(np.diff(ticks)) == {8, 9} and data["delay_ticks"] == plan["delay_ticks"] == 2,
            "unexpected capture or delivery clocks")
    meta = dict(episode_keys=np.array(keys, dtype=np.int64), capture_ticks=ticks,
                manifest_sha256=MANIFEST_SHA, catalog_sha256=catalog_sha, packets_sha256=packet_shas,
                trace_sha256=[s["trace_sha256"] for s in sources],
                native_report_sha256=[s["native_report_sha256"] for s in sources])
    return sources, meta


def _value_sha(fields):
    value = sha256()
    for arr in fields.values():
        value.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return value.hexdigest()


def _native_episodes(sources, field):
    """Verify acquisition identity and reset/completion clocks before exposing one native column."""
    for source in sources:
        path = ROOT / source["path"]
        receipt_path = path.with_name("report.json")
        require(digest(path) == source["trace_sha256"] and digest(receipt_path) == source["native_report_sha256"],
                "native trace or receipt changed")
        receipt = read(receipt_path)
        require(receipt["status"] == "COMPLETE" and receipt["arm"] == "mixed"
                and receipt["training_seed"] == source["source_seed"] and receipt["parent_seed"] == source["parent_seed"]
                and receipt["budget"] == receipt["trace"]["completed_recorded_ticks"] == 4096, "incomplete run")
        with np.load(path, allow_pickle=False) as trace:
            clocks = trace["episode_tick"]
            terminated, truncated = trace["terminated"], trace["truncated"]
            values = trace[field]
        done = terminated | truncated
        starts = np.flatnonzero(clocks == 1)
        require(len(values) == len(clocks) == 4096 and np.array_equal(starts, np.r_[0, np.flatnonzero(done[:-1]) + 1]),
                "unexpected native observation or reset clock")
        for episode in source["admitted_flat_episodes"]:
            a, b = episode["first_archive_row"], episode["end_archive_row_exclusive"]
            require(a == starts[episode["episode"]] and b - a == 300 and np.array_equal(clocks[a:b], np.arange(1, 301))
                    and not done[a:b-1].any() and truncated[b-1] and not terminated[b-1], "incomplete source episode")
            yield values[a:b], episode


def _sensor_tape(actor, captures):
    """Recover only the ten sensor channels available at each successive capture, oldest first."""
    require(actor.ndim == 2 and actor.shape[1] == 627 and actor.dtype == np.float32,
            "invalid native actor observations")
    pieces = []
    for previous, current in zip(captures[:-1], captures[1:], strict=True):
        gap = int(current - previous)
        require(0 < gap <= 10 and 0 <= previous < current < len(actor), "interval outside on-board history")
        history = actor[current].reshape(11, 57)[:, SENSOR_COLUMNS]
        # Index zero holds the current frame. The ten preceding frames follow oldest to newest.
        received = np.concatenate((history[-gap:], history[:1]))
        require(np.array_equal(received, actor[previous:current+1, SENSOR_COLUMNS]),
                "received history disagrees with the consecutive current sensors")
        pieces.append(received if not pieces else received[1:])
    return np.concatenate(pieces).astype(np.float64)


def _write_archive(path, meta, fields, expected):
    require(_value_sha(fields) == expected, "source values differ from the independently selected native frames")
    arrays(path, {**meta, **fields, "content_sha256": expected})
    return dict(archive_sha256=digest(path), content_sha256=expected, episodes=len(meta["episode_keys"]),
                capture_intervals=len(meta["episode_keys"]) * (len(meta["capture_ticks"]) - 1),
                catalog_sha256=meta["catalog_sha256"], native_trace_sha256=meta["trace_sha256"])


def extract_measured(path):
    """Exclusively create the 50 Hz measured tape. Candidate data never read a simulator state column."""
    if Path(path).exists():
        raise FileExistsError(path)
    sources, meta = _sources()
    captures = meta["capture_ticks"]
    tape = np.stack([_sensor_tape(actor, captures) for actor, _ in _native_episodes(sources, "proprio627")])
    fields = dict(gyro_body=tape[:, :, :3] / 0.25, gravity_body=tape[:, :, 3:6], wheel_rad_s=tape[:, :, 6:] / 0.05)
    meta["sample_ticks"] = np.arange(captures[0], captures[-1] + 1, dtype=np.int64)
    return _write_archive(path, meta, fields, MEASURED_SHA)


def extract_oracle(path):
    """Exclusively create the separate oracle tape of [capture, delivery] world xyz + wxyz poses."""
    if Path(path).exists():
        raise FileExistsError(path)
    sources, meta = _sources()
    captures = meta["capture_ticks"]
    # Native qpos is saved at interval end. Its first row corresponds to boundary t=1.
    boundaries = np.column_stack((captures, captures + 2)) - 1
    poses = []
    for qpos, episode in _native_episodes(sources, "qpos"):
        require(sha256(qpos.tobytes()).hexdigest() == episode["physical_episode_qpos_sha256"],
                "oracle pose episode differs from acquisition")
        poses.append(qpos[boundaries, :7])
    return _write_archive(path, meta, {"reference_poses": np.stack(poses)}, ORACLE_SHA)


def _open_archive(path, *, oracle):
    _, meta = _sources()
    captures = meta["capture_ticks"]
    expected = ORACLE_SHA if oracle else MEASURED_SHA
    names = ("reference_poses",) if oracle else MEASUREMENTS
    if not oracle:
        meta["sample_ticks"] = np.arange(captures[0], captures[-1] + 1, dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {*meta, *names, "content_sha256"}
                and all(np.array_equal(archive[name], x) for name, x in meta.items())
                and archive["content_sha256"].item() == expected, "wrong tape type, source roster or acquisition clock")
        fields = {name: archive[name] for name in names}
    episodes = len(meta["episode_keys"])
    shapes = ([(episodes, len(captures), 2, 7)] if oracle else
              [(episodes, len(meta["sample_ticks"]), n) for n in (3, 3, 4)])
    require(all(arr.shape == shape and arr.dtype == np.float64 and np.isfinite(arr).all()
                for arr, shape in zip(fields.values(), shapes, strict=True)) and _value_sha(fields) == expected,
            "invalid or changed archived values")
    return meta, fields


def _indices(rows, meta):
    lookup = {tuple(key): i for i, key in enumerate(meta["episode_keys"])}
    episodes = np.array([lookup.get(pair, -1) for pair in zip(rows["source_seed"], rows["episode"], strict=True)])
    ticks = meta["capture_ticks"]
    frames = np.searchsorted(ticks, rows["capture_tick"])
    require((episodes >= 0).all() and (frames < len(ticks)).all()
            and np.array_equal(ticks[frames], rows["capture_tick"])
            and np.array_equal(rows["capture_tick"] + 2, rows["delivery_tick"]), "unregistered source or packet clock")
    return episodes, frames


def _previous(rows, frames):
    series = rows["series_id"]
    same = series[1:] == series[:-1]
    require(np.array_equal(rows["row_id"], np.arange(len(series)))
            and np.array_equal(frames == 0, np.r_[True, ~same])
            and (np.diff(frames)[same] == 1).all()
            and all((rows[k][:-1][same] == rows[k][1:][same]).all()
                    for k in ("source_seed", "episode", "layout", "side", "visibility")),
            "previous packet crosses a series or capture gap")
    return np.where(frames > 0, np.arange(len(frames)) - 1, -1)


def history_inputs(split, measured_path, windows_path=None):
    """Return round-105 rows, points, queries, previous row (-1 at start), and measured [dx,dy,dyaw].

    Each motion maps current-capture XY into previous-capture coordinates with capture_query().
    The round-105 queries are already corrected for their own 40 ms delivery delay.
    """
    rows, points, queries = inputs(split, windows_path)
    meta, fields = _open_archive(measured_path, oracle=False)
    episodes, frames = _indices(rows, meta)
    previous = _previous(rows, frames)
    offsets = meta["capture_ticks"] - meta["sample_ticks"][0]
    motions = np.full((len(meta["episode_keys"]), len(offsets), 3), np.nan)
    for ep in np.unique(episodes):
        for i in range(1, len(offsets)):
            interval = slice(offsets[i-1], offsets[i] + 1)
            motions[ep, i] = measured_motion(*(fields[k][ep, interval] for k in MEASUREMENTS))
    return rows, points, queries, previous, motions[episodes, frames]


def oracle_capture_delivery(rows, oracle_path):
    """Return [N,2,7] world xyz + wxyz at each packet's capture and delivery, scorer only."""
    meta, fields = _open_archive(oracle_path, oracle=True)
    episodes, frames = _indices(rows, meta)
    return fields["reference_poses"][episodes, frames]


def oracle_pairs(rows, oracle_path):
    """Return [previous capture, current capture] pose pairs. Initial packets carry two NaN poses."""
    meta, fields = _open_archive(oracle_path, oracle=True)
    episodes, frames = _indices(rows, meta)
    previous = _previous(rows, frames)
    own = fields["reference_poses"][episodes, frames, 0]
    pairs = np.full((len(previous), 2, 7), np.nan)
    available = previous >= 0
    pairs[available, 0] = own[previous[available]]
    pairs[available, 1] = own[available]
    return pairs
