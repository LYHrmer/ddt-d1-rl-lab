from itertools import product

import numpy as np
import pytest

from tools100.sensor100 import _grid
from tools106 import geometry100 as replay


def _pose(xy, yaw, roll):
    c, s = np.cos(yaw / 2), np.sin(yaw / 2)
    a, b = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([*xy, 0.45, c*a, c*b, s*b, s*a])


def _scene():
    spec = dict(azimuth_deg=list(range(-12, 13, 2)), depression_deg=list(range(20, 40)),
                sparse_depression_deg=[20, 25, 30, 35], mask_abs_azimuth_max_deg=2,
                mask_depression_deg=[26, 31], mounts_m=[[.2, .1, 0.], [.2, -.1, 0.]],
                range_min_m=.14, range_max_m=1.45, box_height_m=.03, box_center_x_m=9.70,
                box_abs_y_m=19.92, box_half_x_m=.015, box_half_y_m=.015)
    plan = dict(geometry=spec, layouts=["flat", "left"], seeds=[41], capture_ticks=[5, 14],
                policy_hz=50, delay_ticks=2, visibilities=["clear", "masked"])
    roll = np.pi/6
    poses = {(41, 5): np.array([_pose([10., 20.], np.pi/2, roll), _pose([10., 20.02], np.pi, roll)]),
             (41, 14): np.array([_pose([9.95, 20.12], 3*np.pi/4, roll), _pose([9.94, 20.15], np.pi, roll)])}
    keys = list(product(plan["layouts"], plan["seeds"], plan["capture_ticks"]))
    directions, _, keep = _grid(spec)
    n = len(keys)
    body = np.repeat(spec["mounts_m"], len(directions), axis=0) + .5*np.tile(directions, (2, 1))
    a, b = np.cos(roll), np.sin(roll)
    roll_matrix = np.array([[1., 0., 0.], [0., a, -b], [0., b, a]])
    points = np.tile(body @ roll_matrix.T, (n, 1, 1))
    points[:, :2] = np.nan
    ranges = np.full(points.shape[:2], .5)
    ranges[:, 0] = -1.
    self_hit = np.zeros_like(ranges, dtype=bool)
    self_hit[:, 1] = True
    queries = np.tile([[.3, .1], [.3, -.1]], (n, 1, 1))
    # The delivery heading is 180 degrees at both ticks.
    world = np.array([poses[s, t][1, :2] - q for (_, s, t), q in zip(keys, queries, strict=True)])
    layouts, seeds, ticks = zip(*keys, strict=True)
    cloud = dict(cloud_layout=layouts, cloud_seed=seeds, capture_tick=ticks, points=points,
                 range_m=ranges, self_hit=self_hit, query_xy=queries, world_query_xy=world, angle_keep=keep)
    labels = [dict(layout=l, seed=s, capture_tick=t, delivery_tick=t+2,
                   visibility=vis, truth_mm=30. if (l, t) == ("left", 5) else 0.)
              for l, s, t in keys for vis in plan["visibilities"]]
    return plan, cloud, labels, poses


def test_reference_frames_order_shutter_and_reset_at_each_view_and_layout():
    plan, cloud, labels, poses = _scene()
    rows = list(replay._rows(plan, cloud, labels, poses))
    assert [(r["series"], r["tick"]) for r, _, _ in rows] == [
        ((41, layout, view), tick) for layout, view, tick in product(["flat", "left"], ["clear", "masked"], [5, 14])]
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first[0]["reference_motion"] is None
        np.testing.assert_allclose(second[0]["reference_motion"], [.12, .05, np.pi/4], atol=1e-14)
        np.testing.assert_allclose(first[2], [[-.08, .3], [.12, .3]], atol=1e-14)
        assert first[0]["delivery_tick"] == 7 and second[0]["delivery_tick"] == 16
    np.testing.assert_equal(rows[0][1], cloud["points"][0])
    np.testing.assert_equal(rows[2][1][cloud["angle_keep"]], rows[0][1][cloud["angle_keep"]])
    assert np.isnan(rows[2][1][~cloud["angle_keep"]]).all()
    assert rows[4][0]["target_mm"] == 30 and rows[5][0]["target_mm"] == 0


@pytest.mark.parametrize(("field", "error"), [("query_xy", "wheel queries"), ("points", "leveled returns")])
def test_rejects_wrong_pose_clock_or_point_frame(field, error):
    plan, cloud, labels, poses = _scene()
    cloud[field][0, -1, 0] += .01
    with pytest.raises(ValueError, match=error):
        list(replay._rows(plan, cloud, labels, poses))


def test_extract_uses_native_boundary_rows_and_hashes_the_complete_roster(tmp_path, monkeypatch):
    plan, _, _, poses = _scene()
    qpos = np.zeros((301, 23))
    qpos[:, 3] = 1.
    keys = np.array([[41, 5, 7], [41, 14, 16]])
    for seed, c, d in keys:
        qpos[[c, d], :7] = poses[seed, c]
    np.savez(tmp_path / "source.npz", qpos=qpos, time_s=np.arange(301)/50)
    plan["sources"] = [dict(seed=41, path="source.npz", sha256=replay.digest(tmp_path / "source.npz"))]
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay, "protocol", lambda *, private: plan)
    expected = np.array([poses[41, 5], poses[41, 14]])
    monkeypatch.setattr(replay, "POSE_SHA", replay._pose_digest(keys, expected))
    ref = tmp_path / "reference.npz"
    receipt = replay.extract_reference100(ref)
    assert receipt["motion_windows"] == 2 and receipt["archive_sha256"] == replay.digest(ref)
    loaded = replay._reference_poses(ref, plan)
    for key, pair in poses.items():
        np.testing.assert_equal(loaded[key], pair)
    with pytest.raises(FileExistsError):
        replay.extract_reference100(ref)
    broken = tmp_path / "wrong.npz"
    np.savez(broken, keys=keys, reference_poses=expected[:, ::-1], protocol_sha256=replay.PLAN_SHA,
             source_sha256=[plan["sources"][0]["sha256"]])
    with pytest.raises(ValueError, match="reference archive"):
        replay._reference_poses(broken, plan)
