"""Rebuild the two fixed readers from the sealed mount comparison and grade complete paired streams."""
from itertools import product

import numpy as np

from tools94.common94 import arrays, digest, read, write
from tools100.readback100 import reference_delta_mm
from tools100.score100 import estimate_delta_mm
from tools100.sensor100 import _grid
from tools101.score101 import qualifying, require
from tools104.motion104 import heading, rotation
from tools105.surface105 import surface_delta_mm
from tools106.footprint106 import rear_returns, support
from tools106.readback106 import counts, groups
from tools106.sightline106 import mount_origins
from tools107.capture107 import geometry
from tools107.mount107 import mount_positions
from tools107.visibility107 import PLAN


READERS = {"surface": surface_delta_mm, "nearest": estimate_delta_mm}


def load_clouds(path):
    spec = geometry()
    roster = read(PLAN)["validation"]
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in ("cloud_id", "seed", "layout", "arm", "capture_tick", "delivery_tick",
                "range_m", "self_hit", "bracket_hit", "points", "angle_keep", "query_capture_xy", "world_query_xy",
                "reference_poses", "capture_body_to_level", "mount_world_xyz")}
    keys = list(zip(data["layout"], data["arm"], data["seed"], data["capture_tick"], strict=True))
    expected = list(product(roster["layouts"], roster["arms"], roster["seeds"], roster["ticks"]))
    require(keys == expected and len(keys) == 216 and np.array_equal(data["cloud_id"], np.arange(len(keys)))
            and np.array_equal(data["delivery_tick"], data["capture_tick"] + 2), "incorrect acquisition roster")
    dirs, _, keep = _grid(spec)
    np.testing.assert_array_equal(keep, data["angle_keep"])
    direc = np.tile(dirs, (2, 1))
    valid = ((data["range_m"] >= spec["range_min_m"]) & (data["range_m"] <= spec["range_max_m"]) & ~data["self_hit"])
    require(data["range_m"].dtype == np.float64 and data["range_m"].shape == (216, 520)
            and data["self_hit"].dtype == data["bracket_hit"].dtype == np.bool_
            and not np.any(data["bracket_hit"] & ~data["self_hit"]), "invalid raw returns")
    np.testing.assert_array_equal(np.isfinite(data["points"]), np.broadcast_to(valid[:, :, None], (216, 520, 3)))
    lookup = dict(zip(keys, range(len(keys)), strict=True))
    for i, (layout, arm, seed, tick) in enumerate(keys):
        offsets = mount_positions(str(arm))
        c = data["reference_poses"][i, 0]
        np.testing.assert_allclose(mount_origins(c, offsets), data["mount_world_xyz"][i], atol=1e-9, rtol=0)
        xyz = np.repeat(offsets, len(dirs), axis=0) + direc * data["range_m"][i, :, None]
        np.testing.assert_allclose((xyz @ data["capture_body_to_level"][i])[valid[i]],
                                   data["points"][i, valid[i]], atol=1e-9, rtol=0)
        np.testing.assert_allclose((data["world_query_xy"][i] - c[:2]) @ rotation(heading(c)),
                                   data["query_capture_xy"][i], atol=1e-9, rtol=0)
        j = lookup[("flat", "original", seed, tick)]
        for k in ("reference_poses", "world_query_xy", "query_capture_xy"):
            np.testing.assert_allclose(data[k][i], data[k][j], atol=1e-9, rtol=0)
    return data, lookup


def readings(data):
    result = {name: np.full((216, 2), np.nan) for name in READERS}
    for i, (points, query) in enumerate(zip(data["points"], data["query_capture_xy"], strict=True)):
        for v, view in enumerate((points, points[data["angle_keep"]])):
            for name, reader in READERS.items():
                value = reader(view, query)
                if value is not None:
                    require(np.isfinite(value), "invalid numerical estimate")
                    result[name][i, v] = value
    return {"cloud_id": data["cloud_id"], **result}


def comparison(y, original, forward):
    a, b = qualifying(y, original), qualifying(y, forward)
    return dict(original=counts(y, original), forward=counts(y, forward),
                gained=int((~a & b).sum()), regressed=int((a & ~b).sum()))


def wheel_coverage(data, i, layout, spec):
    wheel = 0 if layout == "left" else 1
    c = data["reference_poses"][i, 0]
    points = data["points"][i, data["angle_keep"]].copy()
    points[:, :2] = points[:, :2] @ rotation(heading(c)).T
    near, distance = rear_returns(points + c[:3], data["world_query_xy"][i, wheel], spec, layout)
    return dict(**support(near), nearest_rear_ground_mm=None if distance is None else round(1000*distance, 5))


def grade_grid(data, lookup, saved):
    roster = read(PLAN)["validation"]
    spec, result, exit_cases = geometry(), {}, []
    for seed in roster["seeds"]:
        keys = list(product(roster["layouts"], roster["ticks"]))
        indices = {arm: np.array([lookup[layout, arm, seed, tick] for layout, tick in keys]) for arm in roster["arms"]}
        y = np.array([reference_delta_mm(spec, layout, data["world_query_xy"][i])
                      for (layout, _), i in zip(keys, indices["original"], strict=True)])
        series = np.array([layout for layout, _ in keys])
        mask = groups(series, y)
        mask["raised"], mask["flat_query"], mask["flat_scene"] = y != 0, y == 0, series == "flat"
        require(mask["exit"].sum() == mask["enter"].sum() == 2, "independent trajectory lacks a paired transition")
        vrows = {}
        for v, vis in enumerate(roster["visibilities"]):
            vrows[vis] = {reader: {
                k: comparison(y[m], saved[reader][indices["original"][m], v], saved[reader][indices["forward"][m], v])
                for k, m in mask.items()} for reader in READERS}
        result[str(seed)] = vrows
        for j in np.flatnonzero(mask["exit"]):
            layout, tick = keys[j]
            entry = dict(seed=seed, layout=layout, capture_tick=tick)
            for arm in roster["arms"]:
                i = indices[arm][j]
                val = saved["surface"][i, 1]
                entry[arm] = dict(surface_mm=None if not np.isfinite(val) else round(float(val), 5),
                                  **wheel_coverage(data, i, layout, spec))
            exit_cases.append(entry)
    checks = {}
    for seed, views in result.items():
        exits = views["masked"]["surface"]["exit"]
        clear = views["clear"]["surface"]["all"]
        flat = views["masked"]["surface"]["flat_query"]
        checks[seed] = dict(masked_exit_no_loss=exits["forward"]["correct"] >= exits["original"]["correct"],
                            clear_failures_no_increase=clear["forward"]["correct"] >= clear["original"]["correct"],
                            masked_flat_false_positives_no_increase=flat["forward"]["flat_false_positive"] <= flat["original"]["flat_false_positive"])
    total_exit = {arm: sum(v["masked"]["surface"]["exit"][arm]["correct"] for v in result.values())
                  for arm in roster["arms"]}
    gain = total_exit["forward"] > total_exit["original"]
    hit_counts = {arm: {k: int(data[k][data["arm"] == arm].sum()) for k in ("self_hit", "bracket_hit")}
                  for arm in roster["arms"]}
    return dict(status="COMPLETE", per_seed=result, masked_exit_cases=exit_cases, exit_correct=total_exit,
                checks=checks, strict_total_exit_gain=gain,
                passed=bool(gain and all(all(c.values()) for c in checks.values())), rays_hitting_robot=hit_counts,
                physical_clouds=216, paired_views=216, paired_masked_views=108, actual_sensor_rays=112320,
                environment_and_self_mj_ray_calls=224640, native_decisions=0, world_updates=0, policy_updates=0)


def score(folder, verify=False):
    data, lookup = load_clouds(folder / "clouds.npz")
    predicted = readings(data)
    path = folder / "predictions.npz"
    if not verify:
        arrays(path, predicted)
    sha = digest(path)
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(predicted), "wrong prediction columns")
        for k in predicted:
            np.testing.assert_allclose(predicted[k], saved[k], atol=1e-7, rtol=0., equal_nan=True)
        report = {**grade_grid(data, lookup, saved), "sealed_predictions_sha256":sha}
    if verify:
        require(report == read(folder / "optical.json"), "optical scores changed")
    else:
        write(folder / "optical.json", report)
