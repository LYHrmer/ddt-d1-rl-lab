"""Capture first returns on fixed poses, then seal both arms before scoring."""
import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.common100 import PLAN_SHA
from tools100.score100 import estimate_delta_mm, score_rows


def load_inputs(plan):
    poses, legacy = {}, []
    ticks = np.asarray(plan["capture_ticks"], dtype=int)
    delivery = ticks + plan["delay_ticks"]
    for source in plan["sources"]:
        with np.load(ROOT / source["path"], allow_pickle=False) as trace:
            times = trace["time_s"]
            require(times.shape == (301,) and np.allclose(times, np.arange(301) / 50, atol=1e-8, rtol=0),
                    "archived pose clock differs")
            poses[source["seed"]] = trace["qpos"][:delivery[-1] + 1].copy()
            # Row i+1 contains the packet supplied at the beginning of interval i.
            legacy.append(trace["raw_ranges16"][delivery + 1].copy())
    return poses, np.stack(legacy)


def acquire(folder, plan):
    from tools100.sensor100 import Sensor100

    raw = folder / "packets"
    raw.mkdir()
    poses, legacy = load_inputs(plan)
    geometry = plan["geometry"]
    leading_edge = geometry["box_center_x_m"] - geometry["box_half_x_m"]
    prefix_bounds = []
    saved = []
    sparse, keep = None, None
    for layout in plan["layouts"]:
        scene = Sensor100(plan, layout)
        if sparse is None:
            sparse = scene.sparse_indices.copy()
            keep = scene.angle_keep.copy()
        require(np.array_equal(sparse, scene.sparse_indices) and np.array_equal(keep, scene.angle_keep),
                "rays or mask vary by layout")
        for seed in plan["seeds"]:
            if layout == "flat":
                bounds = [scene.pose(q)["wheel_max_x_bound"] for q in poses[seed]]
                require(np.all(np.isfinite(bounds)) and max(bounds) < leading_edge,
                        f"precontact archive window inadmissible for {seed}")
                prefix_bounds.append(bounds)
            for tick in plan["capture_ticks"]:
                cap_pose = scene.pose(poses[seed][tick])
                scan = scene.capture()
                delivery = scene.pose(poses[seed][tick + plan["delay_ticks"]])
                wheel_bound = max(cap_pose["wheel_max_x_bound"], delivery["wheel_max_x_bound"])
                require(wheel_bound < leading_edge, "wheel reached the fixture during sensor delay")
                packet = raw / f"{layout}_{seed}_{tick}.npz"
                arrays(packet, dict(cloud_seed=seed, cloud_layout=layout, capture_tick=tick,
                                    points=scan["points"], range_m=scan["range_m"], self_hit=scan["self_hit"],
                                    query_xy=delivery["query_xy"], world_query_xy=delivery["world_query_xy"],
                                    wheel_max_x_bound=wheel_bound, scene_time_s=float(scene.data.time)))
                saved.append(packet)
        del scene
    require(len(saved) == plan["budget"]["independent_clouds"], "incomplete cloud roster")
    collected = {}
    for packet in saved:
        with np.load(packet, allow_pickle=False) as data:
            for name in data.files:
                collected.setdefault(name, []).append(data[name])
    cloud_file = folder / "clouds.npz"
    arrays(cloud_file, dict(**{key: np.stack(value) for key, value in collected.items()},
                            sparse_indices=sparse, angle_keep=keep, prefix_wheel_max_x_m=prefix_bounds,
                            legacy_flat_features=legacy))
    return cloud_file


def predict(cloud_file, output, plan):
    rows = []
    with np.load(cloud_file, allow_pickle=False) as archive:
        sparse = archive["sparse_indices"]
        dense = np.arange(plan["budget"]["directions_per_cloud"])
        keep = archive["angle_keep"]
        seeds, layouts, ticks = (archive[key] for key in ("cloud_seed", "cloud_layout", "capture_tick"))
        for idx, (points, queries) in enumerate(zip(archive["points"], archive["query_xy"], strict=True)):
            for visibility in plan["visibilities"]:
                row = {"seed": int(seeds[idx]), "layout": str(layouts[idx]), "capture_tick": int(ticks[idx]),
                       "delivery_tick": int(ticks[idx]) + plan["delay_ticks"], "visibility": visibility}
                for arm, indices in (("sparse", sparse), ("dense", dense)):
                    delivered = indices if visibility == "clear" else indices[keep[indices]]
                    row[f"{arm}_mm"] = estimate_delta_mm(points[delivered], queries)
                rows.append(row)
    write(output, {"protocol_sha256": PLAN_SHA, "cloud_sha256": digest(cloud_file), "rows": rows})


def replay(folder, plan):
    import mujoco
    from tools100.readback100 import measured_counts, reference_delta_mm, verify_pair

    cloud_file = acquire(folder, plan)
    predictions = folder / "predictions.json"
    predict(cloud_file, predictions, plan)
    sealed = {"predictions_sha256": digest(predictions), "cloud_sha256": digest(cloud_file)}
    write(folder / "forecast_seal.json", sealed)
    with np.load(cloud_file, allow_pickle=False) as archive:
        keys = zip(archive["cloud_seed"], archive["cloud_layout"], archive["capture_tick"], strict=True)
        queries = dict(zip(keys, archive["world_query_xy"], strict=True))
    rows = read(predictions)["rows"]
    for row in rows:
        key = (row["seed"], row["layout"], row["capture_tick"])
        row["truth_mm"] = reference_delta_mm(plan["geometry"], row["layout"], queries[key])
    with np.load(cloud_file, allow_pickle=False) as archive:
        counts = measured_counts(archive, rows)
    report = dict(schema=1, status="COMPLETE", protocol_sha256=PLAN_SHA,
                  mujoco_version=mujoco.__version__, source_traces=plan["sources"], **sealed,
                  counts=counts, rows=rows, score=score_rows(rows, plan))
    report["instrument_checks"] = verify_pair(plan, report, cloud_file)
    require(digest(predictions) == sealed["predictions_sha256"], "predictions changed during labeling")
    write(folder / "report.json", report)
