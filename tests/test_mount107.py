from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from tools94.common94 import read
from tools107.mount107 import CONFIG, MountedSensor, _box_separations, _mounted_scene, mount_positions, parts


def _touch(a, b):
    return np.all(np.maximum(a[0], b[0]) <= np.minimum(a[1], b[1]) + 1e-12)


def test_same_rigid_rail_connects_both_front_cover_points_and_both_housings():
    config = read(CONFIG)
    old, forward = mount_positions("original"), mount_positions("forward")
    np.testing.assert_array_equal(old, [[.355, .190, .055], [.355, -.190, .055]])
    np.testing.assert_allclose(forward - old, [[.07, 0., 0.], [.07, 0., 0.]], rtol=0., atol=1e-14)
    assert forward[0, 0] == pytest.approx(config["front_panel_reference_m"][0][0] + config["forward_standoff_m"])
    with pytest.raises(ValueError, match="Unknown optical position"):
        mount_positions("unregistered")
    for arm in ("original", "forward"):
        boxes = parts(arm)
        for side, panel, lens in zip(("left", "right"), config["front_panel_reference_m"], mount_positions(arm), strict=True):
            pad, post, rail, housing = (boxes[f"mount107_{part}_{side}"] for part in ("pad", "post", "rail", "housing"))
            assert pad[0][0] == panel[0] and _touch((panel, panel), pad)
            assert _touch(pad, post) and _touch(post, boxes["mount107_bridge"])
            assert _touch(boxes["mount107_bridge"], rail) and _touch(rail, housing)
            assert housing[1][0] < lens[0] and housing[0][2] > lens[2]
        for name, bounds in boxes.items():
            shift = np.asarray(bounds) - parts("original")[name]
            expected = [.07, 0., 0.] if arm == "forward" and "housing" in name else [0., 0., 0.]
            np.testing.assert_allclose(shift, [expected, expected], atol=1e-14)


def test_derived_xml_preserves_the_adjacent_frozen_symlink(tmp_path):
    frozen = tmp_path / "frozen.xml"
    contents = b'<mujoco><worldbody><body name="base_link"/></worldbody></mujoco>'
    frozen.write_bytes(contents)
    original = tmp_path / "robot.xml"
    original.symlink_to(frozen)
    scene = tmp_path / "scene87.xml"
    scene.write_text('<mujoco><include file="robot.xml"/><worldbody/></mujoco>')
    modified = _mounted_scene(scene, {}, "flat", "forward")
    include = ET.parse(modified).find("include").attrib["file"]
    base = ET.parse(tmp_path / include).find("./worldbody/body")
    assert original.is_symlink() and frozen.read_bytes() == contents
    assert ET.parse(scene).find("include").attrib["file"] == "robot.xml"
    assert {g.attrib["name"] for g in base.iter("geom")} == set(parts("forward"))
    np.testing.assert_array_equal([np.fromstring(s.attrib["pos"], sep=" ") for s in base.iter("site")],
                                  mount_positions("forward"))
    with pytest.raises(FileExistsError):
        _mounted_scene(scene, {}, "flat", "forward")
    assert frozen.read_bytes() == contents


def test_oriented_box_clearance_distinguishes_rotated_axes_and_overlap():
    rot = np.array([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    centers = np.array([[0., 0., 0.]])
    fixed_centers = np.array([[3., 0., 0.], [0., 3., 0.], [.1, .1, .1]])
    result = _box_separations(centers, rot, np.array([[1., .5, .25]]), fixed_centers, np.full((3, 3), .4))
    np.testing.assert_allclose(result, [[2.1, 1.6, 0.]], atol=1e-14)


def test_capture_uses_actual_sites_and_records_first_geom_on_the_same_call():
    # Predefined responses let the public test verify the call contract without native rays.
    sensor = object.__new__(MountedSensor)
    sensor.model = object()
    sensor.base, sensor.site_ids, sensor.rig_ids = 0, [0, 1], [12]
    sensor.heading = np.eye(3)
    sensor.geometry = {"range_min_m": .14, "range_max_m": 1.45}
    sensor.directions = np.array([[1., 0., 0.], [0., 0., -1.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    site_pos = np.array([[2., 2., 1.], [3., 4., 1.]])
    sensor.data = SimpleNamespace(time=0., xpos=np.array([[1., 0., 0.]]), site_xpos=site_pos,
                                  site_xmat=np.stack([np.eye(3).ravel(), rotation.ravel()]))
    calls = []

    def response(model, data, origin, direction, groups, include_static, excluded, first_id):
        calls.append((origin.copy(), direction.copy(), include_static))
        if direction[2] == -1.:
            first_id[0] = 12
            return .15
        first_id[0] = 1 if include_static else -1
        return .4 if include_static else -1.

    sensor.mj = SimpleNamespace(mj_ray=response)
    packet = sensor.capture()
    assert len(calls) == 8 and [c[2] for c in calls] == [True, True, False, False] * 2
    np.testing.assert_array_equal(calls[0][0], site_pos[0])
    np.testing.assert_array_equal(calls[4][0], site_pos[1])
    np.testing.assert_array_equal(calls[4][1], [0., 1., 0.])
    np.testing.assert_array_equal(packet["bracket_hit"], [False, True, False, True])
    np.testing.assert_array_equal(packet["bracket_hit"], packet["self_hit"])
    np.testing.assert_allclose(packet["points"][[0, 2]], [[1.4, 2., 1.], [2., 4.4, 1.]])
    assert np.isnan(packet["points"][[1, 3]]).all()


def test_clearance_reports_separate_closest_pairs_and_subtracts_native_margins():
    sensor = object.__new__(MountedSensor)
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    base = np.array([2., -1., .6])
    local = np.array([[2., 0., 0.], [0., 2.5, 0.]])
    sensor.base, sensor.heading = 0, rot
    sensor.data = SimpleNamespace(xmat=np.array([rot.ravel()]), xpos=np.array([base]),
                                  geom_xpos=local @ rot.T + base, geom_xmat=np.array([rot.ravel(), rot.ravel()]))
    sensor.model = SimpleNamespace(geom_margin=np.array([.05, .1]))
    sensor.rig_names, sensor._moving = ["housing"], np.array([0, 1])
    sensor._geom_names, sensor._visual = np.array(["visual thigh", "wheel cylinder"]), np.array([True, False])
    sensor._geom_centers, sensor._geom_half = np.zeros((2, 3)), np.array([[.25]*3, [.2]*3])
    sensor._rig_centers, sensor._rig_half_sizes = np.array([[.4, 0., 0.]]), np.full((1, 3), .1)
    clearance = sensor.clearance_m()
    assert clearance["minimum_m"] == clearance["visual_m"] == pytest.approx(1.2)
    assert clearance["collision_m"] == pytest.approx(2.1)
    assert clearance["closest_visual"] == ["visual thigh", "housing"]
    assert clearance["closest_collision"] == ["wheel cylinder", "housing"]
