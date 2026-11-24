"""Post-score geometric audit for the eight previously published masked exit abstentions."""
import numpy as np

from tools94.common94 import ROOT, read
from tools100.score100 import RADIUS_M
from tools100.sensor100 import _grid
from tools104.motion104 import capture_query, heading, reference_motion, rotation
from tools105.score105 import PACKETS
from tools105.surface105 import _covers_query


def rear_returns(points, query, fixture, side):
    """True ground behind the rear face, within its original lateral strip and local XY radius."""
    rear = fixture["box_center_x_m"] + fixture["box_half_x_m"]
    lateral = fixture["box_abs_y_m"] * (1 if side == "left" else -1)
    keep = ((points[:, 0] > rear + 1e-6) & (np.abs(points[:, 2]) <= 1e-6)
            & (np.abs(points[:, 1] - lateral) <= fixture["box_half_y_m"] + 1e-6))
    offsets = points[keep, :2] - query
    distance = np.linalg.norm(offsets, axis=1)
    return offsets[distance <= RADIUS_M], float(distance.min()) if len(distance) else None


def support(offsets):
    return {"local_points": len(offsets), "encloses_query": bool(_covers_query(offsets))}


def exit_footprints(rows, previous, motions, queries, poses, unknown_exit_ids):
    """Use privileged fixture dimensions, world queries and pose only to explain the old failure set."""
    plan = read(ROOT / "docs/experiments101/predictive_perception.json")
    fixtures = {g["id"]: g for g in plan["data"]["layouts"]["train"]}
    geometry = read(ROOT / plan["data"]["sensor_base"])["geometry"]
    rays, _, _ = _grid(geometry)
    origins = np.repeat(geometry["mounts_m"], len(rays), axis=0)
    rays = np.tile(rays, (2, 1))
    with np.load(PACKETS["validation"], allow_pickle=False) as archive:
        depth, valid, level, world, local = (archive[k] for k in (
            "depth_m", "valid", "capture_body_to_level", "world_query_xy", "query_xy"))

    def cloud_world(i):
        body = origins + depth[i, :, None].astype(np.float64) * rays
        capture = poses[i, 0]
        xyz = body[valid[i]] @ level[i]
        xyz[:, :2] = xyz[:, :2] @ rotation(heading(capture)).T
        return xyz + capture[:3]

    def is_inside(query, fixture, lateral):
        return bool(abs(query[0] - fixture["box_center_x_m"]) <= fixture["box_half_x_m"] and
                    abs(query[1] - lateral) <= fixture["box_half_y_m"])

    diagnostics = []
    for i in unknown_exit_ids:
        j = int(previous[i])
        if j < 0 or rows["series_id"][j] != rows["series_id"][i]:
            raise ValueError("exit crosses a stream boundary")
        side = str(rows["side"][i])
        wheel = 0 if side == "left" else 1
        fixture = fixtures[str(rows["layout"][i]).rsplit("_", 1)[0]]
        query = world[i, wheel]
        delivery = poses[i, 1]
        np.testing.assert_allclose(local[i] @ rotation(heading(delivery)).T + delivery[:2], world[i], rtol=0., atol=1e-6)
        lateral = fixture["box_abs_y_m"] * (1 if wheel == 0 else -1)
        rear = fixture["box_center_x_m"] + fixture["box_half_x_m"]
        a, nearest = rear_returns(cloud_world(j), query, fixture, side)
        b, _ = rear_returns(cloud_world(i), query, fixture, side)
        measured = capture_query(queries[i], motions[i])
        exact = capture_query(queries[i], reference_motion(poses[[j, i], 0]))
        diagnostics.append(dict(row_id=int(rows["row_id"][i]), source_seed=int(rows["source_seed"][i]),
                                episode=int(rows["episode"][i]), layout=str(rows["layout"][i]),
                                previous_tick=int(rows["capture_tick"][j]), tick=int(rows["capture_tick"][i]),
                                previous_query_on_top=is_inside(world[j, wheel], fixture, lateral),
                                behind_rear_inside_lane=bool(query[0] > rear + 1e-6 and
                                    abs(query[1] - lateral) <= fixture["box_half_y_m"]),
                                query_past_rear_mm=1000. * float(query[0] - rear),
                                previous_nearest_rear_ground_mm=None if nearest is None else 1000. * nearest,
                                measured_reference_query_offset_mm=1000. * float(np.linalg.norm(measured[wheel] - exact[wheel])),
                                previous=support(a), current=support(b), union=support(np.concatenate((a, b)))))
    return diagnostics
