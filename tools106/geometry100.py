"""Secondary round-100 replay with native reference poses for both geometry readers."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np

from tools74.geometry74 import _rotate_inverse
from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read
from tools100.common100 import PLAN_SHA, PUBLIC_CLOUDS, PUBLIC_MANIFEST, PUBLIC_REPORT, protocol
from tools100.readback100 import expected_indices, reference_delta_mm
from tools102.mask102 import level_points
from tools104.motion104 import capture_query, heading, reference_motion, rotation
from tools105.data105 import _level


MANIFEST_SHA = "8f1be5837307ec283830708bc0c431447e80d12bb8116a2d0a179c63a5fdf552"
# Roster-ordered [seed,c,d] int64 and original #87 boundary qpos[[c,d],:7] float64.
POSE_SHA = "4013bef656e9081257600332866ad45475aab797f25aff65645f82b7c4310430"


def _keys(plan):
    return np.array([(s, c, c + plan["delay_ticks"])
                     for s, c in product(plan["seeds"], plan["capture_ticks"])], dtype=np.int64)


def _pose_digest(keys, poses):
    h = sha256(np.ascontiguousarray(keys, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(poses, dtype="<f8").tobytes())
    return h.hexdigest()


def extract_reference100(path):
    """Write the complete set of capture/delivery reference poses from the three native sources."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    plan = protocol(private=True)
    keys = _keys(plan)
    poses = []
    for source in plan["sources"]:
        bounds = keys[keys[:, 0] == source["seed"], 1:]
        with np.load(ROOT / source["path"], allow_pickle=False) as trace:
            times, qpos = trace["time_s"], trace["qpos"]
            require(qpos.shape == (301, 23) and times.shape == (301,)
                    and np.allclose(times[bounds], bounds / plan["policy_hz"], rtol=0, atol=1e-9),
                    "unexpected round-87 pose boundary clocks")
            poses.extend(qpos[bounds, :7])
    poses = np.stack(poses)
    require(_pose_digest(keys, poses) == POSE_SHA, "native reference pose selection changed")
    trace_sha = [s["sha256"] for s in plan["sources"]]
    arrays(path, dict(keys=keys, reference_poses=poses, source_sha256=trace_sha, protocol_sha256=PLAN_SHA))
    return dict(archive_sha256=digest(path), content_sha256=POSE_SHA, protocol_sha256=PLAN_SHA,
                source_seeds=plan["seeds"], source_trace_sha256=trace_sha, motion_windows=len(keys),
                capture_ticks=plan["capture_ticks"], capture_to_delivery_ticks=plan["delay_ticks"],
                observer="native_reference_pose")


def _reference_poses(path, plan):
    keys = _keys(plan)
    with np.load(path, allow_pickle=False) as archive:
        poses = archive["reference_poses"]
        require(set(archive.files) == {"keys", "reference_poses", "source_sha256", "protocol_sha256"}
                and np.array_equal(archive["keys"], keys) and poses.shape == (len(keys), 2, 7)
                and poses.dtype == np.float64 and archive["protocol_sha256"].item() == PLAN_SHA
                and np.array_equal(archive["source_sha256"], [s["sha256"] for s in plan["sources"]])
                and _pose_digest(keys, poses) == POSE_SHA, "incorrect round-100 reference archive")
    return {(int(s), int(c)): pair for (s, c, _), pair in zip(keys, poses, strict=True)}


def _geometry(plan, cloud, poses):
    """Verify coordinates against independent ray and pose fields before exposing either arm's inputs."""
    spec = plan["geometry"]
    keys = list(zip(cloud["cloud_layout"], cloud["cloud_seed"], cloud["capture_tick"], strict=True))
    require(keys == list(product(plan["layouts"], plan["seeds"], plan["capture_ticks"])),
            "unexpected round-100 cloud roster or acquisition order")
    _, keep = expected_indices(spec)
    require(np.array_equal(keep, cloud["angle_keep"]), "unexpected round-100 angular shutter")
    pairs = [poses[seed, tick] for _, seed, tick in keys]
    levels = _level(np.array([_rotate_inverse(c[3:], (0., 0., -1.)) for c, _ in pairs]))
    ranges = cloud["range_m"]
    valid = (ranges >= spec["range_min_m"]) & (ranges <= spec["range_max_m"]) & ~cloud["self_hit"]
    expected = level_points(ranges, valid, levels, spec)
    points = cloud["points"]
    require(points.shape == expected.shape and np.allclose(points, expected, rtol=0, atol=1e-10, equal_nan=True),
            "saved round-100 cloud differs from capture-frame leveled returns")
    query = np.array([capture_query(xy, reference_motion(pair))
                      for xy, pair in zip(cloud["query_xy"], pairs, strict=True)])
    from_world = np.array([(world - c[:2]) @ rotation(heading(c))
                           for world, (c, _) in zip(cloud["world_query_xy"], pairs, strict=True)])
    require(query.shape == (len(keys), 2, 2) and np.allclose(query, from_world, rtol=0, atol=1e-10),
            "delivered wheel queries do not map to the registered capture poses")
    return {key: i for i, key in enumerate(keys)}, points, query, keep


def _rows(plan, cloud, labels, poses):
    lookup, point_sets, queries, keep = _geometry(plan, cloud, poses)
    keys = [(r["layout"], r["seed"], r["capture_tick"], r["visibility"]) for r in labels]
    require(keys == [(*key, vis) for key in lookup for vis in plan["visibilities"]]
            and all(r["delivery_tick"] == r["capture_tick"] + plan["delay_ticks"] for r in labels),
            "unexpected round-100 scoring roster or delivery clock")
    truths = dict(zip(keys, (r["truth_mm"] for r in labels), strict=True))
    row_id = 0
    for seed, layout, vis in product(plan["seeds"], plan["layouts"], plan["visibilities"]):
        previous_pose = None
        for tick in plan["capture_ticks"]:
            key = layout, seed, tick
            i = lookup[key]
            target = truths[(*key, vis)]
            require(target == reference_delta_mm(plan["geometry"], layout, cloud["world_query_xy"][i]),
                    "round-100 archived target disagrees with its delivered world query")
            capture_pose, _ = poses[seed, tick]
            motion = None if previous_pose is None else reference_motion([previous_pose, capture_pose])
            points = point_sets[i].copy()
            if vis == "masked":
                points[~keep] = np.nan
            meta = dict(row_id=row_id, seed=seed, series=(seed, layout, vis), layout=layout, visibility=vis,
                        tick=tick, delivery_tick=tick + plan["delay_ticks"], target_mm=target,
                        reference_motion=motion, capture_pose=capture_pose.copy())
            yield meta, points, queries[i].copy()
            previous_pose = capture_pose
            row_id += 1


def geometry_rows(reference_path):
    """Yield (scoring metadata, 520 capture-frame XYZ, two capture-frame wheel XY).

    The order is seed, layout, visibility, then ascending capture tick. Reference motion
    expresses current-capture XY in the preceding capture frame via capture_query().
    Ground truth, trial identity and absolute pose are for auditing and scoring only.
    """
    plan = protocol()
    require(digest(PUBLIC_MANIFEST) == MANIFEST_SHA, "changed round-100 publication manifest")
    manifest = read(PUBLIC_MANIFEST)
    for path in (PUBLIC_CLOUDS, PUBLIC_REPORT):
        require(digest(path) == manifest["files"][str(path.relative_to(ROOT))], "changed round-100 public input")
    report = read(PUBLIC_REPORT)
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA
            and report["source_traces"] == plan["sources"], "incomplete round-100 source")
    poses = _reference_poses(reference_path, plan)
    fields = ("cloud_layout", "cloud_seed", "capture_tick", "range_m", "self_hit",
              "angle_keep", "points", "query_xy", "world_query_xy")
    with np.load(PUBLIC_CLOUDS, allow_pickle=False) as archive:
        cloud = {name: archive[name] for name in fields}
    yield from _rows(plan, cloud, report["rows"], poses)
