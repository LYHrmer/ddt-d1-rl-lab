"""Frozen flat-motion roster, mounting clearance and paired native first returns."""
from __future__ import annotations

from hashlib import sha256
from itertools import product
from pathlib import Path
from time import monotonic

import numpy as np

from tools91.launch91 import require
from tools94.common94 import ROOT, arrays, digest, read, write
from tools100.common100 import protocol as sensor_protocol
from tools100.readback100 import expected_indices
from tools101.common101 import protocol as fixture_protocol
from tools104.motion104 import capture_query, heading, reference_motion, rotation


SOURCES = {
    89201: (89011, "af975a5469f0473e6f8937e49f4b1bcd887fa47eae71e18864e0a516296000c5"),
    89202: (89023, "6e61637e2d8c24bb055dad38ca00442936a211d5f1e76424bae5119ba43e4193"),
    89203: (89037, "c9192ba0c6b462bfed6dc248aeeecbed0c4ed0e33ef3fa9de02bbba79846c81c"),
}
PREFIX_SHA = {
    89201: "7cf36f333e1d346ad5d254c0bc6cf423af66c080734991763162f6c42114cdd7",
    89202: "0b5fcc2ee496ae3d2e5f8016b65e6740ae05b2871cd26a73b9e5add150e8e1ba",
    89203: "c00c87b2cc81053d0b0647ffb6afde3631c253547752ae735dbd647a421e036f",
}
CAPTURES = (59, 67, 75, 84, 92, 100, 109, 117, 125, 134, 142, 150)
DELAY = 2
LAST = CAPTURES[-1] + DELAY
LAYOUTS = ("flat", "left", "right")
ARMS = ("original", "forward")
RAYS = 520
N_CLOUDS = len(LAYOUTS) * len(ARMS) * len(SOURCES) * len(CAPTURES)


def _pose_sha(qpos):
    return sha256(np.ascontiguousarray(qpos, dtype="<f8").tobytes()).hexdigest()


def geometry():
    plan = read(ROOT / "docs/experiments107/installation.json")
    validation, budget = plan["validation"], plan["budget"]
    require(validation["seeds"] == list(SOURCES) and validation["ticks"] == list(CAPTURES)
            and validation["arms"] == list(ARMS) and validation["layouts"] == list(LAYOUTS)
            and validation["delay_ticks"] == DELAY and budget["clouds_max"] == N_CLOUDS
            and budget["directional_returns_max"] == N_CLOUDS * RAYS
            and budget["mj_ray_calls_max"] == 2 * N_CLOUDS * RAYS, "changed paired acquisition plan")
    base = sensor_protocol()
    require(tuple(base["capture_ticks"]) == CAPTURES and base["delay_ticks"] == DELAY
            and base["policy_hz"] == 50 and base["sensor_hz"] == 6, "changed sensing clocks")
    narrow, = [s for s in fixture_protocol()["data"]["layouts"]["train"] if s["id"] == "narrow"]
    return {**base["geometry"], **{k: v for k, v in narrow.items() if k != "id"}}


def _sources():
    sources = {}
    for seed, (training, expected) in SOURCES.items():
        name = f"mixed_{training}_flat_{seed}"
        folder = ROOT / "evidence/uphill89/evaluate_01" / name
        case = read(folder / "metrics.json")
        file = folder / "trace.npz"
        require(case["status"] == "COMPLETE" and case["case"] == name and case["terrain"] == "flat"
                and case["arm"] == "mixed" and case["training_seed"] == training and case["seed"] == seed
                and case["ticks"] == 300 and case["full_horizon"] and not case["fallen"]
                and not case["illegal_contact"] and case["trace_sha256"] == expected == digest(file),
                f"incomplete or changed native source {name}")
        with np.load(file, allow_pickle=False) as trace:
            qpos, times, axle = trace["qpos"], trace["time_s"], trace["front_axle_x_m"]
            require(qpos.shape == (301, 23) and np.isfinite(qpos).all() and times.shape == axle.shape == (301,)
                    and np.allclose(times, np.arange(301) / 50, rtol=0, atol=1e-8)
                    and np.isfinite(axle[1:]).all() and _pose_sha(qpos[:LAST+1]) == PREFIX_SHA[seed]
                    and not trace["fallen"].any() and not trace["illegal_body_contact"].any(),
                    f"unexpected native boundary clock or state in {name}")
            sources[seed] = qpos[:LAST + 1].copy(), axle[:LAST + 1].copy()
    return sources


def _input_hashes():
    return dict(native_traces={str(k): sha for k, (_, sha) in SOURCES.items()},
                native_prefix_qpos_sha256={str(k): v for k, v in PREFIX_SHA.items()},
                files={p: digest(ROOT / p) for p in (
                    "tools107/capture107.py", "tools107/mount107.py", "tools100/sensor100.py",
                    "docs/experiments100/pointcloud_density.json", "docs/experiments101/predictive_perception.json",
                    "docs/experiments107/installation.json")})


class _Calls:
    """Count real MuJoCo calls and retain returns from an interrupted acquisition."""
    def __init__(self, native):
        self.native = native
        self.in_packet = None
        self.count = 0

    def __getattr__(self, name):
        if name in ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_RungeKutta", "mj_implicit"):
            raise RuntimeError("integration is outside the frozen-pose acquisition")
        return getattr(self.native, name)

    def mj_ray(self, *args):
        require(self.in_packet is not None, "optical call outside a registered packet")
        value = self.native.mj_ray(*args)
        self.count += 1
        self.in_packet.append((float(value), int(args[7][0]), bool(args[5])))
        return value


def _scene(spec, layout, arm):
    from tools107.mount107 import MountedSensor
    scene = MountedSensor(spec, layout, arm)
    require(scene.mj.__version__ == "3.3.0", "expected the repository's original MuJoCo 3.3.0 sensor engine")
    calls = _Calls(scene.mj)
    scene.mj = calls
    return scene, calls


def _sites(scene):
    xyz = scene.data.site_xpos[scene.site_ids]
    body = scene.data.xmat[scene.base].reshape(3, 3)
    np.testing.assert_allclose(xyz, scene.mounts @ body.T + scene.data.xpos[scene.base], rtol=0, atol=1e-10)
    np.testing.assert_allclose(scene.data.site_xmat[scene.site_ids].reshape(2, 3, 3),
                               np.stack([body, body]), rtol=0, atol=1e-10)
    return xyz.copy()


def _check_axle(scene, pose, axial):
    if np.isfinite(axial):
        base_direction = scene.heading[:, 0]
        measured_x = pose["world_query_xy"][:, 0].mean() - scene.geometry["lookahead_m"] * base_direction[0]
        require(abs(measured_x - axial) < 1e-9, "counterfactual query differs from the recorded native axle")


def _prefix(scene, calls, sources, edge):
    minima, clearance = {}, {}
    samples = {k: np.empty((len(sources), LAST+1)) for k in ("visual_m", "collision_m", "wheel_max_x_m")}
    count = 0
    for si, (seed, (qpos, axle)) in enumerate(sources.items()):
        wheel_max = -np.inf
        for tick, (q, reference_x) in enumerate(zip(qpos, axle, strict=True)):
            pose = scene.pose(q)
            _sites(scene)
            _check_axle(scene, pose, reference_x)
            wheel_max = max(wheel_max, pose["wheel_max_x_bound"])
            samples["wheel_max_x_m"][si, tick] = pose["wheel_max_x_bound"]
            gaps = scene.clearance_m()
            for kind in ("visual", "collision"):
                gap = float(gaps[kind + "_m"])
                samples[kind + "_m"][si, tick] = gap
                require(np.isfinite(gap), "nonfinite hardware clearance")
                if kind not in minima or gap < minima[kind]["gap_m"]:
                    minima[kind] = dict(gap_m=gap, seed=seed, tick=tick, pair=gaps["closest_" + kind])
            require(gaps["minimum_m"] == min(gaps["visual_m"], gaps["collision_m"]), "inconsistent clearance bounds")
            count += 1
        clearance[str(seed)] = float(edge - wheel_max)
    require(scene.data.time == 0 and calls.count == 0, "clock or sensor advanced during mechanical qualification")
    minimum = min(m["gap_m"] for m in minima.values())
    return (dict(poses=count, minimum_m=minimum, minimum_by_shape=minima, precontact_clearance_m=clearance,
                 passed=minimum > 0 and min(clearance.values()) > 0), samples)


def prepare_geometry(folder):
    """Exclusively qualify every capture-prefix boundary in the flat scene of each arm (zero rays)."""
    folder = Path(folder)
    start = monotonic()
    write(folder / "geometry_started.json", dict(stage="hardware_clearance", new_rays=0))
    spec, sources, hashes = geometry(), _sources(), _input_hashes()
    edge = spec["box_center_x_m"] - spec["box_half_x_m"]
    audits, samples = {}, []
    for arm in ARMS:
        scene, calls = _scene(spec, "flat", arm)
        audits[arm], values = _prefix(scene, calls, sources, edge)
        samples.append(values)
        write(folder / f"clearance_{arm}.json", audits[arm])
        del scene
    n = len(SOURCES) * (LAST + 1)
    require(all(a["poses"] == n for a in audits.values()), "incomplete pose coverage")
    require(audits["original"]["precontact_clearance_m"] == audits["forward"]["precontact_clearance_m"],
            "changing the bracket moved the wheels")
    sample_path = folder / "clearance_samples.npz"
    arrays(sample_path, dict(arm=ARMS, seed=list(SOURCES), tick=np.arange(LAST+1),
                             qpos=np.stack([q for q, _ in sources.values()]),
                             **{k: np.stack([s[k] for s in samples]) for k in samples[0]}))
    report = dict(status="COMPLETE", passed=all(a["passed"] for a in audits.values()), inputs=hashes,
                  covered_poses_per_arm=n, checked_arm_poses=n*len(ARMS), minimum_m=min(a["minimum_m"] for a in audits.values()),
                  seed_roster=list(SOURCES), prefix_ticks=[0, LAST], capture_ticks=list(CAPTURES), delay_ticks=DELAY,
                  samples_file=sample_path.name, samples_sha256=digest(sample_path), wheel_leading_edge_x_m=edge,
                  clearance_definition="Minimum conservative separation to moving visual or collision geometry, "
                                       "after collision margins, across the fixed support and both housings.",
                  optical_calls=0, physics_steps=0, arms=audits, elapsed_s=monotonic()-start)
    write(folder / "geometry.json", report)
    return report


def check_geometry(folder):
    """Check each published clearance bound, wheel envelope and worst-pose summary without MuJoCo."""
    folder = Path(folder)
    report = read(folder / "geometry.json")
    samples = folder / "clearance_samples.npz"
    require(report["samples_file"] == samples.name and digest(samples) == report["samples_sha256"]
            and report["inputs"] == _input_hashes(), "incorrect public mounting samples or sources")
    spec = geometry()
    edge = spec["box_center_x_m"] - spec["box_half_x_m"]
    require(edge == report["wheel_leading_edge_x_m"], "incorrect obstacle leading edge")
    expected_n = (len(ARMS), len(SOURCES), LAST+1)
    with np.load(samples, allow_pickle=False) as z:
        qpos = z["qpos"]
        require(np.array_equal(z["arm"], ARMS) and np.array_equal(z["seed"], list(SOURCES))
                and np.array_equal(z["tick"], np.arange(LAST+1)) and qpos.shape == (*expected_n[1:], 23)
                and np.isfinite(qpos).all() and all(_pose_sha(q) == PREFIX_SHA[s]
                                                   for s, q in zip(SOURCES, qpos, strict=True)),
                "invalid mounting roster or pose prefixes")
        fields = {k: z[k] for k in ("visual_m", "collision_m", "wheel_max_x_m")}
    require(all(a.shape == expected_n and np.isfinite(a).all() for a in fields.values())
            and np.array_equal(fields["wheel_max_x_m"][0], fields["wheel_max_x_m"][1]),
            "invalid per-pose measurements or different wheel envelopes")
    passes = []
    for ai, arm in enumerate(ARMS):
        saved = report["arms"][arm]
        for kind in ("visual", "collision"):
            gaps = fields[kind + "_m"][ai]
            si, tick = np.unravel_index(np.argmin(gaps), gaps.shape)
            m = saved["minimum_by_shape"][kind]
            require((m["gap_m"], m["seed"], m["tick"]) == (gaps[si, tick], list(SOURCES)[si], tick),
                    "incorrect worst mounting pose")
        clear = edge - fields["wheel_max_x_m"][ai].max(axis=1)
        minimum = min(fields["visual_m"][ai].min(), fields["collision_m"][ai].min())
        passed = bool(minimum > 0 and clear.min() > 0)
        require(saved["minimum_m"] == minimum and saved["precontact_clearance_m"] ==
                dict(zip(map(str, SOURCES), clear, strict=True)) and saved["passed"] == passed
                and saved["poses"] == expected_n[1]*expected_n[2], "incorrect arm clearance or precontact summary")
        passes.append(passed)
    require(report["passed"] == all(passes) and report["covered_poses_per_arm"] == expected_n[1]*expected_n[2]
            and report["checked_arm_poses"] == np.prod(expected_n) and report["minimum_m"] ==
            min(fields["visual_m"].min(), fields["collision_m"].min()) and report["optical_calls"] == report["physics_steps"] == 0,
            "incorrect mechanical admission result")
    return report


def _qualified(path, hashes):
    gate = read(path)
    poses = len(SOURCES) * (LAST + 1)
    require(gate["status"] == "COMPLETE" and gate["passed"] is True and gate["inputs"] == hashes
            and gate["covered_poses_per_arm"] == poses and gate["checked_arm_poses"] == poses * len(ARMS)
            and gate["minimum_m"] > 0 and gate["optical_calls"] == gate["physics_steps"] == 0
            and gate["seed_roster"] == list(SOURCES) and gate["prefix_ticks"] == [0, LAST]
            and gate["capture_ticks"] == list(CAPTURES) and gate["delay_ticks"] == DELAY
            and set(gate["arms"]) == set(ARMS), "missing or unqualified #107 mechanical admission")
    for a in gate["arms"].values():
        require(a["passed"] and a["poses"] == poses and a["minimum_m"] > 0
                and set(a["precontact_clearance_m"]) == {str(k) for k in SOURCES}
                and min(a["precontact_clearance_m"].values()) > 0, "incomplete arm admission")
    return gate


def _capture(scene, calls, arm, layout, seed, tick, cloud_id, qpos, axle):
    delivery = tick + DELAY
    pair = qpos[[tick, delivery], :7]
    capture = scene.pose(qpos[tick])
    _check_axle(scene, capture, axle[tick])
    mount_xyz = _sites(scene)
    level = scene.data.xmat[scene.base].reshape(3, 3).T @ scene.heading
    scan = scene.capture()
    require(len(calls.in_packet) == 2 * RAYS, "unexpected number of native calls in one cloud")
    clear = [v for v, _, include_static in calls.in_packet if include_static]
    first_geom = [g for _, g, include_static in calls.in_packet if include_static]
    robot = np.array([v for v, _, include_static in calls.in_packet if not include_static])
    ranges, self_hit, bracket, points = (scan[k] for k in ("range_m", "self_hit", "bracket_hit", "points"))
    require(ranges.shape == self_hit.shape == bracket.shape == robot.shape == (RAYS,)
            and points.shape == (RAYS, 3) and ranges.dtype == points.dtype == np.float64
            and self_hit.dtype == bracket.dtype == bool and np.isfinite(ranges).all() and np.isfinite(robot).all()
            and np.array_equal(ranges, clear) and np.array_equal(self_hit, (robot >= 0.) & (abs(robot - ranges) <= 1e-8))
            and np.array_equal(bracket, self_hit & np.isin(first_geom, scene.rig_ids)),
            "inconsistent native first-return values or self-hit identity")
    valid = (ranges >= scene.geometry["range_min_m"]) & (ranges <= scene.geometry["range_max_m"]) & ~self_hit
    require(np.array_equal(np.isfinite(points).all(axis=1), valid) and np.isnan(points[~valid]).all(),
            "saved coordinates and native range validity disagree")
    received = scene.pose(qpos[delivery])
    _check_axle(scene, received, axle[delivery])
    local = received["query_xy"]
    world = received["world_query_xy"]
    query = capture_query(local, reference_motion(pair))
    np.testing.assert_allclose(query, (world - pair[0, :2]) @ rotation(heading(pair[0])), rtol=0, atol=1e-10)
    wheel_bound = max(capture["wheel_max_x_bound"], received["wheel_max_x_bound"])
    edge = scene.geometry["box_center_x_m"] - scene.geometry["box_half_x_m"]
    require(scene.data.time == 0 and wheel_bound < edge, "clock advanced or original wheel envelope reached the box")
    return dict(cloud_id=cloud_id, seed=seed, layout=layout, arm=arm, capture_tick=tick, delivery_tick=delivery,
                range_m=ranges, self_hit=self_hit, bracket_hit=bracket, points=points, query_capture_xy=query,
                query_delivery_xy=local, world_query_xy=world, reference_poses=pair, capture_body_to_level=level,
                mount_world_xyz=mount_xyz, wheel_max_x_bound=wheel_bound)


def _archive(raw_paths, path, keep):
    columns = {}
    for raw in raw_paths:
        with np.load(raw, allow_pickle=False) as packet:
            for k in packet.files:
                columns.setdefault(k, []).append(packet[k])
    result = {k: np.stack(v) for k, v in columns.items()}
    require(np.array_equal(result["cloud_id"], np.arange(N_CLOUDS)), "missing native cloud IDs")
    # The fixed robot poses and wheel queries must agree across both optics and all three layouts.
    for name in ("query_capture_xy", "query_delivery_xy", "world_query_xy", "reference_poses",
                 "capture_body_to_level", "wheel_max_x_bound"):
        a = result[name].reshape(len(LAYOUTS)*len(ARMS), len(SOURCES)*len(CAPTURES), *result[name].shape[1:])
        require(np.array_equal(a, np.broadcast_to(a[:1], a.shape)), f"matched pose or query differs: {name}")
    arrays(path, {**result, "angle_keep": keep})


def capture_initial(folder):
    """Acquire both mount arms once after independent clearance admission. Keep each completed cloud on failure."""
    folder = Path(folder)
    start = monotonic()
    hashes = _input_hashes()
    gate_path = folder / "geometry.json"
    _qualified(gate_path, hashes)
    raw = folder / "raw"
    raw.mkdir()
    write(folder / "capture_started.json", dict(geometry_sha256=digest(gate_path), inputs=hashes))
    spec, sources = geometry(), _sources()
    _, keep = expected_indices(spec)
    paths, n_calls = [], 0
    for layout, arm in product(LAYOUTS, ARMS):
        scene, calls = _scene(spec, layout, arm)
        require(np.array_equal(scene.angle_keep, keep), "native ray shutter differs from the registered angles")
        for seed, tick in product(SOURCES, CAPTURES):
            key = f"{len(paths):03d}_{layout}_{arm}_{seed}_{tick}"
            calls.in_packet = []
            try:
                packet = _capture(scene, calls, arm, layout, seed, tick, len(paths), *sources[seed])
                path = raw / (key + ".npz")
                arrays(path, packet)
                paths.append(path)
            except BaseException as exc:
                calls_done = calls.in_packet
                arrays(raw / (key + "_partial.npz"), dict(
                    seed=seed, layout=layout, arm=arm, capture_tick=tick, completed_calls=len(calls_done),
                    range_m=np.array([v for v, _, _ in calls_done], dtype=np.float64),
                    geom_id=np.array([g for _, g, _ in calls_done], dtype=np.int32),
                    include_static=np.array([flag for _, _, flag in calls_done], dtype=bool)))
                write(folder / "capture_partial.json", dict(status="PARTIAL", at_packet=key, complete_clouds=len(paths),
                                                            completed_ray_calls=n_calls+calls.count, error=repr(exc)))
                raise
            finally:
                calls.in_packet = None
        n_calls += calls.count
        del scene
    require(len(paths) == N_CLOUDS and n_calls == 2 * RAYS * N_CLOUDS,
            "paired capture exceeded or undershot its registered roster")
    path = folder / "clouds.npz"
    _archive(paths, path, keep)
    report = dict(status="COMPLETE", inputs=hashes, geometry_sha256=digest(gate_path), cloud_sha256=digest(path),
                  physical_clouds=N_CLOUDS, directions_per_cloud=RAYS, first_return_beams=N_CLOUDS*RAYS,
                  native_ray_calls=n_calls, clear_masked_views=2*N_CLOUDS, physics_steps=0, native_decisions=0,
                  model_updates=0, policy_updates=0, elapsed_s=monotonic()-start,
                  native_packets_sha256={p.name: digest(p) for p in paths})
    write(folder / "capture.json", report)
    return report
