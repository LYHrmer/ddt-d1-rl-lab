from types import SimpleNamespace
import json

import numpy as np
import pytest

from tools94.common94 import read
from tools100.readback100 import expected_indices
from tools107 import capture107 as recording


class SyntheticEngine:
    def mj_ray(self, model, data, origin, direction, geomgroup, include_static, exclude, hit_id):
        own_housing = direction[0] == 0
        hit_id[0] = 4 if own_housing else (9 if include_static else -1)
        return .4 if include_static or own_housing else -1.


class SyntheticSensor:
    def __init__(self, geometry, arm, fail, gap):
        self.mj = recording._Calls(SyntheticEngine())
        self.geometry, self.fail, self.gap = geometry, fail, gap
        self.mounts = np.array([[.35 if arm == "original" else .42, y, .05] for y in (.19, -.19)])
        self.base, self.site_ids = 0, np.array([0, 1])
        self.rig_ids = np.array([4])
        self.heading = np.eye(3)
        self.data = SimpleNamespace(time=0., xmat=np.eye(3).reshape(1, 9), xpos=np.zeros((1, 3)),
                                    site_xmat=np.tile(np.eye(3).ravel(), (2, 1)))
        _, self.angle_keep = expected_indices(geometry)

    def pose(self, q):
        self.data.xpos[0] = q[:3]
        self.data.site_xpos = self.mounts + q[:3]
        queries = np.array([[.35, .2], [.35, -.2]])
        return dict(query_xy=queries, world_query_xy=queries+q[:2], wheel_max_x_bound=.2+q[0])

    def clearance_m(self):
        return dict(minimum_m=self.gap, visual_m=self.gap, collision_m=self.gap+.01,
                    closest_visual=["leg", "housing"], closest_collision=["wheel", "bar"])

    def capture(self):
        for _ in self.site_ids:
            for static in (True, False):
                for j in range(260):
                    self.mj.mj_ray(None, None, None, [j], None, static, -1, np.empty(1, dtype=int))
                    if self.fail and self.mj.count == 2*recording.RAYS + 17:
                        raise RuntimeError("fixture read failed")
        own = (np.arange(520) % 260) == 0
        points = np.zeros((520, 3))
        points[own] = np.nan
        return dict(points=points, range_m=np.full(520, .4), self_hit=own, bracket_hit=own)


def setup_series(monkeypatch, *, fail=False, front_gap=.014):
    geom = dict(azimuth_deg=list(range(-12, 13, 2)), depression_deg=list(range(20, 40)),
                sparse_depression_deg=[20, 25, 30, 35], mask_abs_azimuth_max_deg=2,
                mask_depression_deg=[26, 31], range_min_m=.14, range_max_m=1.45, lookahead_m=.35,
                box_center_x_m=.75, box_half_x_m=.07)
    qpos = np.zeros((12, 23))
    qpos[:, 3] = 1.
    qpos[:, 0] = np.arange(12) * .01
    source = {41: (qpos, qpos[:, 0].copy())}
    monkeypatch.setattr(recording, "SOURCES", {41: (7, "synthetic")})
    monkeypatch.setattr(recording, "PREFIX_SHA", {41: recording._pose_sha(qpos)})
    monkeypatch.setattr(recording, "CAPTURES", (1, 9))
    monkeypatch.setattr(recording, "LAST", 11)
    monkeypatch.setattr(recording, "N_CLOUDS", 12)
    monkeypatch.setattr(recording, "geometry", lambda: geom)
    monkeypatch.setattr(recording, "_sources", lambda: source)
    monkeypatch.setattr(recording, "_input_hashes", lambda: {"synthetic": True})

    def make_sensor(spec, layout, arm):
        sensor = SyntheticSensor(spec, arm, fail, front_gap if arm == "forward" else .023)
        return sensor, sensor.mj

    monkeypatch.setattr(recording, "_scene", make_sensor)


def test_prefix_gate_and_complete_paired_raw_roster(monkeypatch, tmp_path):
    setup_series(monkeypatch)
    gate = recording.prepare_geometry(tmp_path)
    assert gate["passed"] and gate["covered_poses_per_arm"] == 12 and gate["optical_calls"] == 0
    assert gate["minimum_m"] == .014
    assert recording.check_geometry(tmp_path) == gate
    assert gate["arms"]["original"]["precontact_clearance_m"]["41"] == pytest.approx(.37)
    report = recording.capture_initial(tmp_path)
    assert report["physical_clouds"] == 12 and report["first_return_beams"] == 6240
    assert report["native_ray_calls"] == 12480 and report["physics_steps"] == 0
    assert len(list((tmp_path / "raw").glob("*.npz"))) == 12
    with np.load(tmp_path / "clouds.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["cloud_id"], np.arange(12))
        assert list(zip(z["arm"][:4], z["capture_tick"][:4])) == [
            ("original", 1), ("original", 9), ("forward", 1), ("forward", 9)]
        np.testing.assert_allclose(z["query_capture_xy"][0], [[.37, .2], [.37, -.2]])
        assert z["bracket_hit"].sum() == 24 and z["points"].shape == (12, 520, 3)
        assert z["mount_world_xyz"][2, 0, 0] - z["mount_world_xyz"][0, 0, 0] == pytest.approx(.07)
    with pytest.raises(FileExistsError):
        recording.capture_initial(tmp_path)


def test_nonpositive_clearance_refuses_to_cast(monkeypatch, tmp_path):
    setup_series(monkeypatch, front_gap=0.)
    assert recording.prepare_geometry(tmp_path)["passed"] is False
    assert recording.check_geometry(tmp_path)["passed"] is False
    with pytest.raises(ValueError, match="mechanical admission"):
        recording.capture_initial(tmp_path)
    assert not (tmp_path / "raw").exists()


def test_interrupted_native_packet_preserves_complete_and_partial_returns(monkeypatch, tmp_path):
    setup_series(monkeypatch, fail=True)
    recording.prepare_geometry(tmp_path)
    with pytest.raises(RuntimeError, match="fixture read failed"):
        recording.capture_initial(tmp_path)
    receipt = read(tmp_path / "capture_partial.json")
    assert receipt["complete_clouds"] == 1 and receipt["completed_ray_calls"] == 1057
    assert (tmp_path / "raw/000_flat_original_41_1.npz").exists()
    with np.load(tmp_path / "raw/001_flat_original_41_9_partial.npz") as z:
        assert z["range_m"].shape == (17,) and int(z["completed_calls"]) == 17
    assert not (tmp_path / "clouds.npz").exists()
    with pytest.raises(FileExistsError):
        recording.capture_initial(tmp_path)


def test_no_rays_during_admission_and_no_native_step_in_capture():
    calls = recording._Calls(SyntheticEngine())
    with pytest.raises(ValueError, match="optical call outside"):
        calls.mj_ray()
    with pytest.raises(RuntimeError, match="integration"):
        calls.mj_step()


def test_public_clearance_rejects_changed_worst_pose(monkeypatch, tmp_path):
    setup_series(monkeypatch)
    gate = recording.prepare_geometry(tmp_path)
    gate["arms"]["original"]["minimum_by_shape"]["visual"]["tick"] = 9
    (tmp_path / "geometry.json").write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="incorrect worst mounting pose"):
        recording.check_geometry(tmp_path)
