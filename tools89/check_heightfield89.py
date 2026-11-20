"""Explicit native geometry regression for the grid-edge height reading in MuJoCo 3.3.0."""
from types import SimpleNamespace

import pytest

from tools87.terrain87 import _set_road
from tools89 import course89 as course


def test_native_hfield_row_edge_can_hit_underside_and_two_interiors_recover_top():
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
