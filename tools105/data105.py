"""Rebuild two readers' common, causal inputs from the sealed D1 range packets."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read
from tools100.readback100 import expected_indices
from tools100.sensor100 import _grid
from tools101.common101 import BASE, PUBLIC as PACKETS, protocol
from tools103.data103 import MANIFEST_SHA, _catalog
from tools104.motion104 import PUBLIC as MOTION104, capture_query, measured_motion


PUBLIC = ROOT / "reports/surface105"
SPLITS = {"design": "holdout", "validation": "train"}
ROW_FIELDS = ("row_id", "source_seed", "episode", "layout", "side", "visibility", "capture_tick",
              "delivery_tick", "cloud_id", "series_id")
MEASUREMENTS = ("gyro_body", "gravity_body", "wheel_rad_s")
DESIGN_WINDOWS_SHA = "354335e5d8aabb7021feb36949e5f2edef15df534694a36a9c640bde12267da7"
# Independently obtained from the three consecutive *current* frames at c,c+1,d,
# before extracting any received history. Encoding is defined by measurement_digest().
VALIDATION_MEASUREMENTS_SHA = "e762959a46a5588840b96b4b10e05274c2af394e228ffb96f19f6bec720089c9"


@dataclass
class _Packets:
    rows: dict
    depth: np.ndarray
    valid: np.ndarray
    query: np.ndarray
    proprio: np.ndarray
    keys: np.ndarray
    window: np.ndarray
    geometry: dict
    data: dict
    info: dict
    catalog_sha: str


def measurement_digest(keys, measured):
    value = sha256(np.ascontiguousarray(keys, dtype="<i8").tobytes())
    for field in MEASUREMENTS:
        value.update(np.ascontiguousarray(measured[field], dtype="<f8").tobytes())
    return value.hexdigest()


def _packets(split):
    require(split in SPLITS, "unknown surface-readout split")
    parent = SPLITS[split]
    path = PACKETS / f"{parent}.npz"
    data, info, catalog_sha = _catalog(parent, path)
    plan = protocol()
    geometry = read(ROOT / plan["data"]["sensor_base"])["geometry"]
    ticks = data["capture_ticks"]
    require(ticks == plan["capture_ticks"] and data["delay_ticks"] == plan["delay_ticks"] == 2,
            "unexpected 50 Hz capture/delivery clocks")
    roster = [(e["source_seed"], e["episode"]) for e in info["episode_roster"]]
    keys = np.array([(s, e, c, c + 2) for (s, e), c in product(roster, ticks)], dtype=np.int64)
    layouts = info["layouts"]
    specs = plan["data"]["layouts"]
    other = "holdout" if parent == "train" else "train"
    fixture_fields = ("box_center_x_m", "box_half_x_m", "box_abs_y_m", "box_half_y_m")
    fixtures = {s: {tuple(g[k] for k in fixture_fields) for g in specs[s]} for s in (parent, other)}
    require(sorted({s for s, _ in roster}) == plan["data"]["source_seeds"][parent]
            and layouts == [f"{g['id']}_{side}" for g, side in product(specs[parent], ("left", "right"))]
            and fixtures[parent].isdisjoint(fixtures[other]), "shared geometry or inconsistent source split")
    scenes = list(product(roster, layouts))
    streams = [(scene, vis) for scene in scenes for vis in ("clear", "masked")]
    n = len(streams) * len(ticks)
    require(info["rows"] == n and info["clouds"] * 2 == n and info["series"] == len(streams)
            and roster == sorted(set(roster)) and (len(roster), n) == ((5, 320) if split == "design" else (8, 1024)),
            "incomplete source episode, layout or time roster")
    series = np.repeat(np.arange(len(streams)), len(ticks))
    frame = np.tile(np.arange(len(ticks)), len(streams))
    expected = dict(row_id=np.arange(n), series_id=series, cloud_id=(series // 2) * len(ticks) + frame,
                    capture_tick=keys[:len(ticks), 2][frame], delivery_tick=keys[:len(ticks), 3][frame])
    for field, values in (("source_seed", [r[0][0] for r, _ in streams]), ("episode", [r[0][1] for r, _ in streams]),
                          ("layout", [r[1] for r, _ in streams]), ("side", [r[1].rsplit("_", 1)[1] for r, _ in streams]),
                          ("visibility", [vis for _, vis in streams])):
        expected[field] = np.asarray(values)[series]
    with np.load(path, allow_pickle=False) as archive:
        rows = {field: archive[field] for field in ROW_FIELDS}
        require(all(np.array_equal(rows[field], expected[field]) for field in ROW_FIELDS),
                "packet rows do not match the original sources, geometry or clocks")
        # This whitelist deliberately omits targets, native poses, world queries and the truth-derived leveling matrix.
        depth, valid, query, proprio = (archive[k] for k in ("depth_m", "valid", "query_xy", "proprio"))
        if split == "design":
            require(np.array_equal(archive["geometry_depth_m64"].astype(np.float32), depth),
                    "design ranges differ from the original higher-precision first returns")
    require(depth.shape == valid.shape == (n, 520) and depth.dtype == np.float32 and valid.dtype == bool
            and query.shape == (n, 2, 2) and proprio.shape == (n, 38) and np.isfinite(query).all()
            and np.isfinite(proprio).all() and np.isfinite(depth).all() and (depth[~valid] == 0).all()
            and (depth[valid] >= np.float32(geometry["range_min_m"])).all()
            and (depth[valid] <= np.float32(geometry["range_max_m"])).all(), "invalid measured sensors or queries")
    clear, masked = (np.flatnonzero(rows["visibility"] == view) for view in ("clear", "masked"))
    _, mask = expected_indices(geometry)
    require(np.array_equal(valid[masked], valid[clear] & mask)
            and np.array_equal(depth[masked], np.where(mask, depth[clear], 0.))
            and np.array_equal(query[clear], query[masked]) and np.array_equal(proprio[clear], proprio[masked]),
            "two views do not share a physical return, delivery query or measured state")
    # Each native movement is reused across source geometries, mirrors and visibility masks.
    window = series // (2 * len(layouts)) * len(ticks) + frame
    return _Packets(rows, depth, valid, query, proprio, keys, window, geometry, data, info, catalog_sha)


def _native_windows(source, ticks):
    path = ROOT / source["path"]
    receipt_path = path.with_name("report.json")
    require(digest(path) == source["trace_sha256"] and digest(receipt_path) == source["native_report_sha256"],
            "native trace or receipt differs from the registered source")
    receipt = read(receipt_path)
    require(receipt["status"] == "COMPLETE" and receipt["training_seed"] == source["source_seed"]
            and receipt["parent_seed"] == source["parent_seed"] and receipt["arm"] == "mixed"
            and receipt["budget"] == receipt["trace"]["completed_recorded_ticks"] == 4096, "partial native run")
    with np.load(path, allow_pickle=False) as trace:
        actor, clocks = trace["proprio627"], trace["episode_tick"]
        terminated, truncated, terrain = trace["terminated"], trace["truncated"], trace["terrain"]
    done = terminated | truncated
    starts = np.flatnonzero(clocks == 1)
    require(actor.shape == (4096, 627) and np.array_equal(starts, np.r_[0, np.flatnonzero(done[:-1]) + 1]),
            "invalid native observation or episode clocks")
    frames = []
    for episode in source["admitted_flat_episodes"]:
        a, b = episode["first_archive_row"], episode["end_archive_row_exclusive"]
        require(starts[episode["episode"]] == a and b - a == 300 and not done[a:b-1].any() and truncated[b-1]
                and not terminated[b-1] and not terrain[a:b].any() and np.array_equal(clocks[a:b], np.arange(1, 301)),
                "partial or misidentified native flat episode")
        for c in ticks:
            d = c + 2
            require(a + d < b, "motion window crosses the episode boundary")
            history = actor[a+d].reshape(11, 57)[[-2, -1, 0]]
            require(np.array_equal(history, actor[a+c:a+d+1, :57]), "received history violates the three-tick clock")
            frames.append(history)
    return np.array(frames, dtype=np.float64)


def _audit_native_ranges(p):
    """Check every saved first return before float32 rounding and each visibility mask, without labels."""
    rows, geometry = p.rows, p.geometry
    names = p.info["native_first_return_files_sha256"]
    hashes = {}
    max_roundoff_m = 0.
    _, mask = expected_indices(geometry)
    for stream in np.unique(rows["series_id"][rows["visibility"] == "clear"]):
        clear = np.flatnonzero(rows["series_id"] == stream)
        masked = np.flatnonzero(rows["series_id"] == stream + 1)
        first = clear[0]
        name = f"train_{rows['source_seed'][first]}_ep{rows['episode'][first]:02d}_{rows['layout'][first]}.npz"
        path = BASE / "prepare_01/native" / name
        hashes[name] = digest(path)
        require(hashes[name] == names[name], f"changed native first-return archive {name}")
        with np.load(path, allow_pickle=False) as raw:
            require(sorted(raw.files) == [f"tick_{c:03d}" for c in rows["capture_tick"][clear]],
                    "missing or duplicated archived first-return tick")
            for i, j in zip(clear, masked, strict=True):
                record = raw[f"tick_{rows['capture_tick'][i]:03d}"]
                ranges, self_hit = record["range_m"], record["self_hit"]
                require(ranges.shape == (520,) and ranges.dtype == np.float64 and np.isfinite(ranges).all()
                        and self_hit.dtype == bool, "invalid original range representation")
                visible = (ranges >= geometry["range_min_m"]) & (ranges <= geometry["range_max_m"]) & ~self_hit
                for row, keep in ((i, visible), (j, visible & mask)):
                    require(np.array_equal(p.valid[row], keep)
                            and np.array_equal(p.depth[row], np.where(keep, ranges, 0.).astype(np.float32)),
                            "published range or angular mask differs from its native first return")
                if visible.any():
                    rounding = np.abs(p.depth[i, visible].astype(np.float64) - ranges[visible])
                    max_roundoff_m = max(max_roundoff_m, float(rounding.max()))
    require(hashes == names, "extra or missing native first-return file")
    return dict(files_sha256=hashes, file_count=len(hashes), physical_clouds=p.info["clouds"],
                native_ray_bins_checked=p.info["clouds"] * 520, public_view_bins_checked=p.info["rows"] * 520,
                public_float32_equals_rounded_native=True, max_valid_range_roundoff_m=max_roundoff_m,
                validity_and_mask_exact=True)


def build_windows(destination):
    """Extract the two validation sources and audit their original returns; exclusively create one NPZ."""
    if Path(destination).exists():
        raise FileExistsError(destination)
    p = _packets("validation")
    sources = [s for s in p.data["sources"] if s["source_seed"] in (89011, 89023)]
    source_roster = [e for s in sources for e in s["admitted_flat_episodes"]]
    require(source_roster == p.info["episode_roster"], "native sources do not match the complete validation roster")
    frames = np.concatenate([_native_windows(s, p.data["capture_ticks"]) for s in sources])
    measured = dict(gyro_body=frames[:, :, :3] / 0.25, gravity_body=frames[:, :, 3:6],
                    wheel_rad_s=frames[:, :, 28:41:4] / 0.05)  # fourth encoder in each group of four
    require(measurement_digest(p.keys, measured) == VALIDATION_MEASUREMENTS_SHA,
            "received histories differ from the independently sealed current frames")
    native = _audit_native_ranges(p)
    traces = [s["trace_sha256"] for s in sources]
    arrays(destination, dict(keys=p.keys, **measured, catalog_sha256=p.catalog_sha, packets_sha256=p.info["sha256"],
                             trace_sha256=traces, measurement_sha256=VALIDATION_MEASUREMENTS_SHA))
    return dict(source_seeds=[s["source_seed"] for s in sources], episodes=p.info["episodes"], motion_windows=len(p.keys),
                observations_per_window=3, capture_to_delivery_ticks=2, packets_sha256=p.info["sha256"],
                parent_manifest_sha256=MANIFEST_SHA, catalog_sha256=p.catalog_sha, windows_sha256=digest(destination),
                measurement_sha256=VALIDATION_MEASUREMENTS_SHA,
                source_trace_sha256={str(s["source_seed"]): s["trace_sha256"] for s in sources},
                source_receipt_sha256={str(s["source_seed"]): s["native_report_sha256"] for s in sources},
                native_first_returns=native)


def _measurements(split, path, p):
    with np.load(path, allow_pickle=False) as w:
        if split == "design":
            require(digest(path) == DESIGN_WINDOWS_SHA and np.array_equal(w["keys"], p.keys[:, 1:]),
                    "changed round-104 design windows")
        else:
            traces = [s["trace_sha256"] for s in p.data["sources"] if s["source_seed"] in (89011, 89023)]
            require(set(w.files) == {"keys", *MEASUREMENTS, "catalog_sha256", "packets_sha256", "trace_sha256",
                                     "measurement_sha256"} and np.array_equal(w["keys"], p.keys)
                    and np.array_equal(w["trace_sha256"], traces) and w["catalog_sha256"].item() == p.catalog_sha
                    and w["packets_sha256"].item() == p.info["sha256"]
                    and w["measurement_sha256"].item() == VALIDATION_MEASUREMENTS_SHA, "incorrect validation source")
        measured = {name: w[name] for name in MEASUREMENTS}
    for field, channels in zip(MEASUREMENTS, (3, 3, 4), strict=True):
        a = measured[field]
        require(a.shape == (len(p.keys), 3, channels) and a.dtype == np.float64 and np.isfinite(a).all(),
                "invalid compact measurement dimensions or values")
    if split == "validation":
        require(measurement_digest(p.keys, measured) == VALIDATION_MEASUREMENTS_SHA, "changed validation measurements")
    return measured


def _level(gravity):
    """Columns expressed in the body frame: horizontal forward, horizontal left, and up."""
    norms = np.linalg.norm(gravity, axis=1, keepdims=True)
    require(np.all(norms > 1e-8), "undefined vertical direction")
    up = -gravity / norms
    forward = np.array([1., 0., 0.]) - up[:, :1] * up
    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    require(np.all(norms > 1e-8), "undefined horizontal forward direction")
    forward /= norms
    return np.stack([forward, np.cross(up, forward), up], axis=-1)


def inputs(split, windows_path=None):
    """Return row IDs/strata, measured [N,520,3] XYZ (invalid bins NaN), and [N,2,2] left/right queries.

    Both readers receive only XYZ and query XY. Acquisition identities and strata remain with the scorer.
    """
    p = _packets(split)
    path = Path(windows_path) if windows_path is not None else (MOTION104 if split == "design" else PUBLIC) / "windows.npz"
    measured = _measurements(split, path, p)
    gyro, gravity, wheels = (measured[name] for name in MEASUREMENTS)
    require(np.array_equal(gyro[p.window, -1], p.proprio[:, :3].astype(np.float64) / 0.25)
            and np.array_equal(gravity[p.window, -1], p.proprio[:, 3:6])
            and np.array_equal(wheels[p.window, -1], p.proprio[:, 25:38:4].astype(np.float64) / 0.05),
            "the delivered packet and three-tick measurement windows report different onboard sensors")
    levels = _level(gravity[:, 0])
    motions = [measured_motion(g, u, v) for g, u, v in zip(gyro, gravity, wheels, strict=True)]
    directions, _, _ = _grid(p.geometry)
    mounts = np.repeat(p.geometry["mounts_m"], len(directions), axis=0)
    directions = np.tile(directions, (2, 1))
    points = (mounts + p.depth[:, :, None] * directions) @ levels[p.window]
    points[~p.valid] = np.nan
    queries = np.array([capture_query(xy, motions[j]) for xy, j in zip(p.query, p.window, strict=True)])
    return p.rows, points, queries
