"""Recompute both arms from the published clouds without MuJoCo or robot assets."""
import argparse
import hashlib
from itertools import product
import json

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, digest, read
from tools100.common100 import (PLAN_SHA, PUBLIC_CLOUDS, PUBLIC_MANIFEST, PUBLIC_MOVIE_RECEIPT, PUBLIC_REPORT,
                                PUBLIC_VIDEO, protocol)
from tools100.score100 import score_rows


def reference_delta_mm(geometry, layout, world_xy):
    if layout == "flat":
        return 0.
    center = np.array([geometry["box_center_x_m"], geometry["box_abs_y_m"] * {"left": 1, "right": -1}[layout]])
    half = np.array([geometry["box_half_x_m"], geometry["box_half_y_m"]])
    inside = (np.abs(world_xy - center) <= half).all(axis=1)
    return float((int(inside[0]) - int(inside[1])) * 1000 * geometry["box_height_m"])


def expected_indices(geometry):
    azimuths = geometry["azimuth_deg"]
    tilts = geometry["depression_deg"]
    angles = list(product(range(2), tilts, azimuths))
    sparse = [angles.index((mount, angle, 0)) for mount in range(2) for angle in geometry["sparse_depression_deg"]]
    lo, hi = geometry["mask_depression_deg"]
    keep = [not (abs(yaw) <= geometry["mask_abs_azimuth_max_deg"] and lo <= tilt <= hi)
            for _, tilt, yaw in angles]
    return np.array(sparse), np.array(keep)


def independent_estimate(points, queries, estimator):
    # Sort the delivered point distances separately for each query. This audit
    # does not use the recording solver's arg-min or its validity mask.
    heights = []
    for query in queries:
        distance = np.linalg.norm(points[:, :2] - query, axis=1)
        order = np.argsort(distance)
        close = order[distance[order] <= estimator["radius_m"]]
        if not len(close):
            return None
        tied = close[distance[close] <= distance[close[0]] + estimator["tie_tolerance_m"]]
        heights.append(np.mean(points[tied, 2]))
    return float(1000 * (heights[0] - heights[1]))


def measured_counts(data, rows):
    seeds = np.unique(data["cloud_seed"])
    layouts = np.unique(data["cloud_layout"])
    ticks = np.unique(data["capture_tick"])
    require(np.all(data["scene_time_s"] == 0), "native simulation clock advanced")
    return {"pairs": len(seeds) * len(layouts) * len({r["visibility"] for r in rows}),
            "packets_per_pair": len(ticks), "paired_rows": len(rows),
            "independent_clouds": len(data["cloud_seed"]), "directions_per_cloud": data["range_m"].shape[1],
            "physical_steps": 0, "policy_updates": 0, "world_updates": 0, "perception_updates": 0}


def verify_pair(plan, report, path):
    require(report["status"] == "COMPLETE" and report["protocol_sha256"] == PLAN_SHA
            and report["cloud_sha256"] == digest(path) and report["source_traces"] == plan["sources"],
            "publication or cloud provenance differs")
    rows = report["rows"]
    metadata = {"protocol_sha256": PLAN_SHA, "cloud_sha256": report["cloud_sha256"],
                "rows": [{k: v for k, v in r.items() if k != "truth_mm"} for r in rows]}
    original = (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode()
    require(hashlib.sha256(original).hexdigest() == report["predictions_sha256"],
            "public estimates differ from the seal written before labeling")
    require(score_rows(rows, plan) == report["score"], "published gate or counts differ")
    geometry = plan["geometry"]
    sparse, keep = expected_indices(geometry)
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(sparse, data["sparse_indices"]) and np.array_equal(keep, data["angle_keep"]),
                "ray grid, subset or occlusion differs from registration")
        counts = measured_counts(data, rows)
        require(counts == report["counts"] == plan["budget"], "scope or budget differs")
        seeds, layouts, ticks = data["cloud_seed"], data["cloud_layout"], data["capture_tick"]
        keys = list(zip(seeds, layouts, ticks, strict=True))
        lookup = {key: i for i, key in enumerate(keys)}
        expected = set(product(plan["seeds"], plan["layouts"], plan["capture_ticks"]))
        require(set(lookup) == expected and len(keys) == len(expected), "missing or repeated source cloud")
        points, queries, world_xy = data["points"], data["query_xy"], data["world_query_xy"]
        ranges, self_hit = data["range_m"], data["self_hit"]
        n = counts["independent_clouds"]
        require(points.shape == (n, len(keep), 3) and ranges.shape == self_hit.shape == points.shape[:2]
                and world_xy.shape == queries.shape == (n, 2, 2) and self_hit.dtype == np.dtype(bool)
                and np.isfinite(ranges).all() and np.isfinite(queries).all() and np.isfinite(world_xy).all(),
                "corrupt packet dimensions or coordinates")
        valid = (ranges >= geometry["range_min_m"]) & (ranges <= geometry["range_max_m"]) & ~self_hit
        require(np.array_equal(valid, np.isfinite(points).all(axis=-1)) and np.isnan(points[~valid]).all(),
                "invalid first-return point was included in the visible cloud")
        prefix = data["prefix_wheel_max_x_m"]
        require(prefix.shape == (len(plan["seeds"]), max(plan["capture_ticks"]) + plan["delay_ticks"] + 1)
                and np.isfinite(prefix).all(), "incomplete precontact pose prefix")
        edge = geometry["box_center_x_m"] - geometry["box_half_x_m"]
        at_packet = data["wheel_max_x_bound"]
        require(at_packet.shape == (n,) and np.isfinite(at_packet).all() and max(prefix.max(), at_packet.max()) < edge,
                "wheel extent reached the raised surface")
        dense = np.arange(len(keep))
        for row in rows:
            i = lookup[row["seed"], row["layout"], row["capture_tick"]]
            require(row["truth_mm"] == reference_delta_mm(geometry, row["layout"], world_xy[i]), "wrong physical label")
            for arm, indices in (("sparse", sparse), ("dense", dense)):
                if row["visibility"] == "masked":
                    indices = indices[keep[indices]]
                value = independent_estimate(points[i, indices], queries[i], plan["estimator"])
                stored = row[arm + "_mm"]
                require((value is None and stored is None) or
                        (value is not None and stored is not None and abs(value - stored) <= 1e-8),
                        "independent point-based estimate differs")
        # Archived #87 acquired in the native stepping loop. This is an
        # engineering cross-check on the flat scene, outside the promotion gate.
        flat = [lookup[seed, "flat", tick] for seed in plan["seeds"] for tick in plan["capture_ticks"]]
        historical = data["legacy_flat_features"].reshape(len(flat), 16)
        recorded = historical[:, 8:] == 1
        replay_valid = valid[flat][:, sparse]
        comparable = replay_valid & recorded
        errors = np.abs(ranges[flat][:, sparse] - historical[:, :8].astype(float)*geometry["range_max_m"])[comparable]
        return dict(precontact_bound_max_m=float(prefix.max()), fixture_leading_edge_m=float(edge),
                    minimum_longitudinal_clearance_m=float(edge-prefix.max()),
                    native_flat_validity_agreement=int((recorded == replay_valid).sum()),
                    native_flat_beams=int(recorded.size), native_flat_comparable=int(errors.size),
                    native_flat_mae_mm=float(errors.mean()*1000) if len(errors) else None,
                    native_flat_max_error_mm=float(errors.max()*1000) if len(errors) else None,
                    dense_angular_kept=int(keep.sum()), sparse_angular_kept=int(keep[sparse].sum()),
                    physically_valid_returns=int(valid.sum()), self_returns=int(self_hit.sum()),
                    independent_estimates=len(rows)*2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    if args.check:
        report = read(PUBLIC_REPORT)
        checks = verify_pair(protocol(), report, PUBLIC_CLOUDS)
        require(report["instrument_checks"] == checks, "published instrument diagnostics differ")
        manifest = read(PUBLIC_MANIFEST)
        artifacts = (PUBLIC_REPORT, PUBLIC_CLOUDS, PUBLIC_VIDEO, PUBLIC_MOVIE_RECEIPT)
        require(manifest["protocol_sha256"] == PLAN_SHA and manifest["files"] ==
                {str(p.relative_to(ROOT)): digest(p) for p in artifacts}, "public file identity differs")
        for source_map in manifest["producers"].values():
            for name, expected in source_map.items():
                require(digest(ROOT / name) == expected, f"producer source differs: {name}")
        print(json.dumps({"independent_estimates": checks["independent_estimates"],
                          "counts": report["counts"], "passed": report["score"]["passed"]}, indent=2))


if __name__ == "__main__":
    main()
