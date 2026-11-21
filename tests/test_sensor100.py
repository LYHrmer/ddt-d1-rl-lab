"""Synthetic geometry and range arrays, with no robot model or integration."""

from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from tools100 import sensor100


@pytest.fixture
def geometry():
    return {"mounts_m": [[.30, .15, 0.], [.30, -.15, 0.]],
            "azimuth_deg": [-12, 0, 12], "depression_deg": [33, 43, 56, 82],
            "sparse_depression_deg": [82, 56, 43, 33], "mask_abs_azimuth_max_deg": 4,
            "mask_depression_deg": [48, 61], "range_min_m": .14, "range_max_m": 1.45,
            "box_center_x_m": .775, "box_abs_y_m": .22, "box_half_x_m": .075,
            "box_half_y_m": .10, "box_height_m": .030, "lookahead_m": .35}


def test_identical_eight_directions_and_strict_angular_subset(geometry):
    directions, sparse, keep = sensor100._grid(geometry)
    native = np.deg2rad([82, 56, 43, 33])
    expected = np.column_stack((np.cos(native), np.zeros(4), -np.sin(native)))
    dense = np.tile(directions, (2, 1))
    np.testing.assert_allclose(dense[sparse], np.tile(expected, (2, 1)), rtol=0, atol=1e-15)
    assert np.array_equal(keep[sparse], [True, False, True, True] * 2)


def test_box_geometry_and_mirror(geometry):
    for layout, sign in (("left", 1), ("right", -1)):
        world = ET.Element("worldbody")
        sensor100._add_box(world, geometry, layout)
        box = world.find("./geom[@name='bump100']")
        assert box.get("type") == "box"
        np.testing.assert_allclose(np.fromstring(box.get("pos"), sep=" "), [.775, sign * .22, .015])
        np.testing.assert_allclose(np.fromstring(box.get("size"), sep=" "), [.075, .10, .015])
    world = ET.Element("worldbody")
    sensor100._add_box(world, geometry, "flat")
    assert len(world) == 0


@pytest.fixture
def posed_sensor(geometry):
    # Independent quarter-turn frame and wheel axes. The stub receives the
    # same numeric buffers as native position-only kinematics would supply.
    scene = sensor100.Sensor100.__new__(sensor100.Sensor100)
    scene.geometry = geometry
    scene.base = 0
    scene.wheels = np.arange(4)
    scene.mounts = np.array(geometry["mounts_m"])
    scene.directions, scene.sparse_indices, scene.angle_keep = sensor100._grid(geometry)
    scene.heading = None
    base = np.array([[1., .5, .45]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    centers = np.array([[ .83, .7, .05], [1.17, .7, .05], [.83, .3, .05], [1.17, .3, .05]])
    # Wheel cylinder z axis is parallel to world x at this heading.
    wheel = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    scene.model = SimpleNamespace(nq=7, geom_size=np.tile([.05, .01, 0.], (4, 1)),
                                  geom_margin=np.full(4, .001))
    scene.data = SimpleNamespace(qpos=np.zeros(7), xmat=rotation.reshape(1, 9), xpos=base, time=0.,
                                 geom_xpos=centers, geom_xmat=np.tile(wheel.reshape(1, 9), (4, 1)))
    scene.mj = SimpleNamespace(mj_fwdPosition=lambda model, data: None)
    return scene


def test_query_centers_and_envelope_follow_wheel_axes(posed_sensor):
    pose = posed_sensor.pose(np.array([1., .5, .45, np.sqrt(.5), 0., 0., np.sqrt(.5)]))
    np.testing.assert_allclose(pose["query_xy"], [[.55, .17], [.55, -.17]], atol=1e-12)
    np.testing.assert_allclose(pose["world_query_xy"], [[.83, 1.05], [1.17, 1.05]], atol=1e-12)
    assert pose["wheel_max_x_bound"] == pytest.approx(1. + .17 + .01 + .001)


def test_raw_returns_self_occlusion_and_heading_level_points(posed_sensor, monkeypatch):
    scene = posed_sensor
    with pytest.raises(RuntimeError, match="pose"):
        scene.capture()
    scene.pose(np.zeros(7))

    def ray_fixture(mj, model, data, origin, directions, include_static):
        assert np.all(directions[:, 2] < 0.)
        distance = -origin[2] / directions[:, 2] if include_static else np.full(len(directions), -1.)
        if origin[0] < 1.0:  # The left mount faces world +y in this pose.
            distance[1] = .3  # The articulated robot gives the nearest return.
        if include_static:
            distance[2:5] = [-1., .05, 1.8]  # Missing, too close, too far.
        return distance

    monkeypatch.setattr(sensor100, "_rays", ray_fixture)
    packet = scene.capture()
    assert packet["self_hit"].sum() == 1 and packet["self_hit"][1]
    assert np.isnan(packet["points"][1:5]).all()
    np.testing.assert_allclose(packet["range_m"][1:5], [.3, -1., .05, 1.8])
    z = packet["points"][:, 2]
    assert np.allclose(z[np.isfinite(z)], -.45)
    np.testing.assert_allclose(packet["points"][0], scene.mounts[0] + packet["range_m"][0] * scene.directions[0])
