from xml.etree import ElementTree as ET

import numpy as np
import pytest

from tools108.contact108 import KNEES, STEP_NAME, add_step, contact_channels, first_front_contact, intervention


def test_front_calf_pulse_has_the_registered_angle_clock_and_preserves_other_outputs():
    spec = {"pulse_ticks": 16, "pulse_rad": -.08}
    nominal = np.linspace(-.5, .5, 16, dtype=np.float32)
    assert KNEES == [2, 6]
    for tick in range(60):
        control, zero = intervention(nominal, "original", tick, 20, spec)
        flexed, requested = intervention(nominal, "flexion", tick, 20, spec)
        np.testing.assert_array_equal(control, nominal)
        assert zero == 0.
        expected = np.zeros(16)
        if 20 <= tick < 36:
            expected[KNEES] = -.16
        assert requested == pytest.approx(expected[2] * .5)
        np.testing.assert_array_equal(flexed, nominal + expected.astype(np.float32))
    missing, requested = intervention(nominal, "flexion", 42, None, spec)
    np.testing.assert_array_equal(missing, nominal)
    assert requested == 0.
    nominal[KNEES] = [-.95, -.9]
    flexed, _ = intervention(nominal, "flexion", 20, 20, spec)
    np.testing.assert_array_equal(flexed[KNEES], [-1., -1.])
    np.testing.assert_allclose(nominal[KNEES], [-.95, -.9])


def test_true_wheel_step_force_is_distinct_from_floor_and_nonwheel_contacts():
    slots, floor, step = {4: 0, 7: 1, 8: 2, 11: 3}, 0, 1
    # MuJoCo's first frame axis points from geom1 to geom2.
    contacts = [
        (0, 4, 160., [0, 0, 1], [0, 0, 0]),
        (1, 4, .75, [0, 0, 1], [.5, .2, .03]),
        (4, 1, .25, [1, 0, 0], [.45, .2, .02]),
        (8, 1, 5., [0, 0, -1], [.49, .2, .03]),
        (1, 9, 3., [0, 0, 1], [.52, .1, .03]),
        (0, 10, 4., [0, 0, 1], [.5, .1, 0]),
        (4, 7, 30., [1, 0, 0], [.3, 0, .09])]
    row, events = contact_channels(contacts, slots, floor, step)
    np.testing.assert_array_equal(row["step_force_n"], [1., 0., 5., 0.])
    np.testing.assert_array_equal(row["ground_force_n"], [160., 0., 0., 0.])
    np.testing.assert_array_equal(row["step_normal_vector_n"][0], [-.25, 0., .75])
    np.testing.assert_allclose(row["step_contact_pos_m"][0], [.4875, .2, .0275])
    assert np.isnan(row["step_contact_pos_m"][[1, 3]]).all()
    assert row["nonwheel_step_max_n"] == 3. and row["nonwheel_floor_max_n"] == 4.
    assert len(events) == 4 and [event[0] for event in events] == [0, 0, 2, -1]


@pytest.mark.parametrize("native_index,issued_at", [(0, 1), (9, 1), (10, 2), (19, 2)])
def test_first_contact_uses_an_inclusive_front_wheel_threshold_then_the_next_decision(native_index, issued_at):
    f = np.zeros((30, 4))
    f[:, 2] = 100.
    f[0, 1] = .999999
    assert first_front_contact(f, 1.) is None
    f[native_index, 0] = 1.
    result = first_front_contact(f, 1.)
    assert result == native_index
    assert result // 10 + 1 == issued_at


def test_static_box_has_the_registered_front_top_width_and_collision_filters():
    world = ET.Element("worldbody")
    spec = dict(front_x_m=.45, length_m=1., half_width_m=.7, height_m=.03)
    add_step(world, spec)
    geom = world.find(f"geom[@name='{STEP_NAME}']")
    assert geom.attrib["type"] == "box" and geom.attrib["contype"] == geom.attrib["conaffinity"] == "1"
    center, half_size = (np.fromstring(geom.attrib[k], sep=" ") for k in ("pos", "size"))
    np.testing.assert_allclose(center-half_size, [.45, -.7, 0.], atol=1e-15)
    np.testing.assert_allclose(center+half_size, [1.45, .7, .03], atol=1e-15)
    with pytest.raises(ValueError, match="duplicate"):
        add_step(world, spec)
