from types import SimpleNamespace

import numpy as np
import pytest

from tools89 import course89 as course


def test_second_and_fourth_uphill_resets_shift_regardless_of_terrain_cycle_offset():
    for seed in (10, 11, 12):
        terrains = [course.TERRAINS[(seed + episode) % 3] for episode in range(18)]
        selected = [course.slope_start_for_episode("cycle", seed, ep) for ep in range(18)]
        assert [flag for kind, flag in zip(terrains, selected) if kind == "up"] == [False, True] * 3
        assert not any(flag for kind, flag in zip(terrains, selected) if kind != "up")
    for ep in range(4):
        assert course.slope_start_for_episode("up", 0, ep) == (ep % 2 == 1)
        assert not course.slope_start_for_episode("flat", 0, ep)
        assert not course.slope_start_for_episode("down", 0, ep)


def test_root_pose_transports_body_orientation_velocity_and_normal_height():
    yaw = 0.17
    pose = np.array([0., -.2, .43, np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
    initial_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    original_velocity = np.array([.1, -.05, .2])
    velocity = original_velocity.copy()
    x = 1.10
    course.place_on_uphill(pose, velocity, x)
    angle = np.deg2rad(4)
    tilt = np.array([[np.cos(angle), 0., -np.sin(angle)], [0., 1., 0.], [np.sin(angle), 0., np.cos(angle)]])
    actual = np.empty(9)
    course.mujoco.mju_quat2Mat(actual, pose[3:7])
    assert np.allclose(actual.reshape(3, 3), tilt @ initial_rotation, atol=1e-14)
    assert np.allclose(velocity, tilt @ original_velocity, atol=1e-14)
    assert pose[:2].tolist() == [x, -.2]
    for offset in ((.25, .16, -.42), (-.23, -.15, -.4)):
        point = pose[:3] + actual.reshape(3, 3) @ offset
        flat_relative_height = .43 + (initial_rotation @ offset)[2]
        height = tilt[:, 2] @ (point - [course.RAMP_START_X_M, 0., 0.])
        assert height == pytest.approx(flat_relative_height, abs=1e-14)


def test_four_wheel_mask_uses_floor_load_threshold_toe_and_wheel_order(monkeypatch):
    toe = course.RAMP_START_X_M
    contacts = [(0, 80, toe, 4.), (0, 70, toe + .1, 1.0), (75, 0, toe - 1e-6, 2.),
                (95, 0, toe + .1, 2.), (70, 95, toe + .3, 500.), (0, 95, toe + .1, 20.)]
    items = [SimpleNamespace(geom1=a, geom2=b, pos=[x, 0., 0.]) for a, b, x, _ in contacts]
    env = SimpleNamespace(current_terrain="up", floor_geom=0, model=object(),
                          data=SimpleNamespace(contact=items, ncon=len(items)))

    def contact_force(model, data, index, force):
        force[0] = contacts[index][3]

    monkeypatch.setattr(course.mujoco, "mj_contactForce", contact_force)
    slots = {70: 0, 75: 1, 80: 2, 95: 3}
    assert course.loaded_wheels_past_toe(env, slots).tolist() == [False, False, True, True]
    env.current_terrain = "flat"
    assert not course.loaded_wheels_past_toe(env, slots).any()


def test_native_hfield_row_edge_can_hit_underside_and_two_interiors_recover_top():
    from tools87.terrain87 import _set_road

    xml = ('<mujoco><asset><hfield name="r" size="4 1.8 .32 .02" nrow="37" ncol="81"/></asset>'
           '<worldbody><geom name="floor" type="hfield" hfield="r" pos=".6 0 -.16"/></worldbody></mujoco>')
    model = course.mujoco.MjModel.from_xml_string(xml)
    data = course.mujoco.MjData(model)
    env = SimpleNamespace(model=model, data=data, floor_geom=model.geom("floor").id)
    for terrain in course.TERRAINS:
        env.current_terrain = terrain
        _set_road(model, terrain)
        course.mujoco.mj_forward(model, data)
        probe = course.road_cross_sections(env, .21358)
        if course.mujoco.__version__ == "3.3.0":
            assert probe["original_row_edge"]["height_m"] == pytest.approx(-.16, abs=1e-8)
            assert not probe["original_row_edge"]["floor_within_tolerance"]
        for x in (.21358, .9913, 1.275, 1.3864):
            probe = course.road_cross_sections(env, x)
            assert len(probe["two_cell_interiors"]) == 2
            assert all(row["floor_within_tolerance"] and row["abs_error_m"] < 2e-6
                       for row in probe["two_cell_interiors"])
        assert data.time == 0.
