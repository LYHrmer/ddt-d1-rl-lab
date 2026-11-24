"""Retrospective surface labels for the sparse range packets actually delivered before #108 contact."""
from pathlib import Path

import numpy as np

from tools77.geometry77 import _rotate
from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read
from tools100.sensor100 import _grid


BEAMS = ROOT / "docs/experiments100/pointcloud_density.json"
PACKET_FIELDS = ["capture_tick", "delivery_tick", "lead_ms", "top_bits_left", "top_bits_right",
                 "floor_returns", "riser_returns", "unclassified_valid_returns", "invalid_returns"]


def _world(pose, points):
    return pose[:3] + np.array([_rotate(pose[3:7], p, inverse=False) for p in points])


def _packet(features, capture_pose, delivery_pose, mounts, directions, sensor, box):
    fraction, flags = np.split(features, 2)
    valid = flags == 1.
    require(fraction.dtype == np.float32 and ((flags == 0.) | valid).all()
            and (fraction[~valid] == 0.).all(), "invalid sparse actor representation")
    scale = sensor["range_max_m"]
    lengths = fraction.astype(np.float64)*scale
    # One float32 rounding occurred when #87 packed each normalized return. A unit
    # ray maps its half-ULP range interval to an equal or smaller error on any axis.
    tolerance = .5*np.spacing(fraction).astype(np.float64)*scale + 64*np.finfo(float).eps
    require(((lengths[valid] + tolerance[valid] >= sensor["range_min_m"])
             & (lengths[valid] <= scale + tolerance[valid])).all(), "valid return outside sensor range")
    body = mounts + lengths[:, None]*directions
    x, y, z = _world(capture_pose, body).T
    x_delivery, _, z_delivery = _world(delivery_pose, body).T
    front, h = box["front_x_m"], box["height_m"]
    rear, width = front + box["length_m"], box["half_width_m"]
    interior = (x > front+tolerance) & (x < rear-tolerance) & (abs(y) < width-tolerance)
    exterior = (x < front-tolerance) | (x > rear+tolerance) | (abs(y) > width+tolerance)
    top = valid & interior & (abs(z-h) <= tolerance)
    floor = valid & exterior & (abs(z) <= tolerance)
    riser = (valid & (abs(x-front) <= tolerance) & (abs(y) < width-tolerance)
             & (z > tolerance) & (z < h-tolerance))
    other = valid & ~(top | floor | riser)
    # A point on an edge can satisfy two faces geometrically. Keep it unclassified.
    # For a clock check, assess the same labeled beams at the delivery pose.
    capture_residuals = np.r_[abs(z[top]-h), abs(z[floor]), abs(x[riser]-front)]
    delivery_residuals = np.r_[abs(z_delivery[top]-h), abs(z_delivery[floor]), abs(x_delivery[riser]-front)]
    return top, floor, riser, other, ~valid, 1000*(x[top]-front), capture_residuals, delivery_residuals


def _bits(hits):
    return sum(1 << i for i, hit in enumerate(hits) if hit)


def _case(episodes, seed, plan, native, mounts, directions):
    arms = ["original", "flexion"]
    folders = [episodes / f"seed{seed}_{arm}" for arm in arms]
    records = [read(p/"report.json") for p in folders]
    require(all(r["arm"] == arm and r["seed"] == seed and r["geometry"] == plan["geometry"]
                and r["checkpoint_sha256"] == plan["checkpoint_sha256"] for r, arm in zip(records, arms)),
            "wrong precontact pair")
    row = dict(seed=seed, paired_precontact=False, arm_status={r["arm"]: r["status"] for r in records})
    if any(r["status"] != "COMPLETE" or not r["full_horizon"] or r["first_contact_substep"] is None for r in records):
        return row
    r = records[0]
    require(r["first_contact_substep"] == records[1]["first_contact_substep"], "different physical contact clocks")
    paths = [p/"trace.npz" for p in folders]
    require(all(m["trace_file"] == p.name and digest(p) == m["trace_sha256"] for p, m in zip(paths, records)),
            "precontact source digest mismatch")
    stop = r["first_contact_substep"] + 1  # First loaded solver interval's end, in native substeps.
    n = (stop-1)//10 + 1  # Count policy boundaries strictly before actual contact, including reset.
    delay, hz, capture_hz = (native[k] for k in ("delay_ticks", "policy_hz", "sensor_hz"))
    with np.load(paths[0], allow_pickle=False) as orig, np.load(paths[1], allow_pickle=False) as flex:
        for key, cols in (("qpos", slice(None, 7)), ("actor_observation", slice(627, None))):
            require(np.array_equal(orig[key][:n, cols], flex[key][:n, cols]), f"precontact {key} differs")
        for key in ("sensor_capture_tick", "sensor_delivery_tick", "time_s"):
            require(np.array_equal(orig[key][:n], flex[key][:n]), f"paired {key} differs")
        for archive in (orig, flex):
            force = archive["step_force_n"][:stop, :2]
            hits = np.flatnonzero((force >= plan["trigger_n"]).any(axis=1))
            require(hits.size and hits[0] == stop-1, "native front contact mismatch")
        poses, obs = orig["qpos"][:n, :7], orig["actor_observation"][:n, 627:]
        c_log, d_log = orig["sensor_capture_tick"][:n], orig["sensor_delivery_tick"][:n]
        times = orig["time_s"][:n]
    require(hz == 50 and capture_hz == 6 and delay == 2 and obs.shape == (n, 16)
            and np.allclose(times, np.arange(n)/hz, rtol=0., atol=1e-9)
            and np.allclose(np.linalg.norm(poses[:, 3:7], axis=1), 1., rtol=0., atol=1e-12),
            "unexpected acquisition clock, actor channel shape or pose")
    captures = [t for t in range(n-delay) if t == 0 or t*capture_hz//hz != (t-1)*capture_hz//hz]
    require((c_log[:delay] == -1).all() and (d_log[:delay] == -1).all() and not obs[:delay].any(),
            "actor contains measurements before the first delivery")
    require(np.array_equal(np.flatnonzero(d_log == np.arange(n)), np.array(captures)+delay),
            "archive differs from scheduled delivery")
    packets, evidence, capture_errors, delivery_errors = [], [], [], []
    degrees = native["geometry"]["sparse_depression_deg"]
    for i, c in enumerate(captures):
        d = c+delay
        next_d = captures[i+1]+delay if i+1 < len(captures) else n
        require((c_log[d:next_d] == c).all() and (d_log[d:next_d] == d).all()
                and (obs[d:next_d] == obs[d]).all(), "held actor packet or time stamp changed between deliveries")
        top, floor, riser, other, invalid, depth, err_c, err_d = _packet(
            obs[d], poses[c], poses[d], mounts, directions, native["geometry"], plan["geometry"])
        lead_ms = 2*(stop-10*d)
        left, right = top[:4], top[4:]
        packets.append([c, d, lead_ms, _bits(left), _bits(right), int(floor.sum()), int(riser.sum()),
                        int(other.sum()), int(invalid.sum())])
        capture_errors.extend(err_c)
        delivery_errors.extend(err_d)
        if top.any():
            evidence.append(dict(capture_tick=c, delivery_tick=d, delivery_s=d/hz, lead_ms=lead_ms,
                                 top_depressions_deg_left=np.asarray(degrees)[left].tolist(),
                                 top_depressions_deg_right=np.asarray(degrees)[right].tolist(),
                                 top_x_beyond_front_edge_mm=[float(depth.min()), float(depth.max())]))
    row.update(paired_precontact=True, original_trace_sha256=r["trace_sha256"], first_contact_ms=2*stop,
               first_top_packet=evidence[0] if evidence else None, last_top_packet=evidence[-1] if evidence else None,
               precontact_packets=packets, packets_with_top_both_sides=sum(bool(p[3] and p[4]) for p in packets),
               counts=dict(top=sum(p[3].bit_count()+p[4].bit_count() for p in packets),
                           floor=sum(p[5] for p in packets), front_riser=sum(p[6] for p in packets),
                           unclassified_valid=sum(p[7] for p in packets), invalid=sum(p[8] for p in packets)),
               max_capture_pose_labeled_surface_residual_um=float(max(capture_errors))*1e6 if capture_errors else None,
               max_delivery_pose_same_surface_residual_mm=float(max(delivery_errors))*1000 if delivery_errors else None)
    return row


def preview(episodes_dir, plan, split):
    """Return JSON-ready packet evidence, with no physics, new ranging, or policy/model calls."""
    episodes = Path(episodes_dir)
    index, native = read(episodes/"index.json"), read(BEAMS)
    require(split in ("development", "blind") and index["reset_seeds"] == plan[split]
            and index["arms"] == plan["arms"] == ["original", "flexion"]
            and digest(ROOT/"tools87/terrain87.py") == native["parents_sha256"]["tools87/terrain87.py"],
            "unregistered sparse geometry or split")
    directions, sparse, _ = _grid(native["geometry"])
    directions = np.tile(directions, (2, 1))[sparse]
    mounts = np.repeat(native["geometry"]["mounts_m"], 4, axis=0)
    require(directions.shape == mounts.shape == (8, 3), "expected the native sparse eight-beam order")
    rows = [_case(episodes, seed, plan, native, mounts, directions) for seed in plan[split]]
    return dict(split=split, audit="post_score_physical_surface_preview", original_packet_counted_once_per_pair=True,
                provenance="The two actors' delivered range channels and capture poses match before physical contact. Privileged world pose and step geometry label these returns retrospectively. The actor received normalized ranges and validity.",
                coordinate_clock="For each acquisition the original environment refreshed its base kinematics at the policy boundary before measuring. De-normalize the archived packet and rotate its body endpoints at capture; delivery follows two policy ticks later.",
                packet_fields=PACKET_FIELDS, top_mask_bits_in_depression_deg=native["geometry"]["sparse_depression_deg"],
                surface_rule="Classify face interiors only, within half a normalized float32 ULP mapped into metres plus double precision roundoff. Boundary or off-surface returns remain unclassified.",
                rows=rows)
