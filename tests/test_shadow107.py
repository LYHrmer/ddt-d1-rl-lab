import numpy as np
from tools107.shadow107 import exit_from_shadow_m


def test_nominal_raised_face_and_side_open_at_different_slides():
    box_lo, box_hi = np.array([.1, -.1, 0.]), np.array([.4, .1, .03])
    ground = np.array([.41, 0., 0.])
    distance, edge = exit_from_shadow_m(np.array([.21, 0., .3]), np.array([1., 0., 0.]), ground, box_lo, box_hi)
    assert np.isclose(distance, .1) and edge == "top"
    distance, edge = exit_from_shadow_m(np.array([.21, .7, .15]), np.array([1., 0., 0.]), ground, box_lo, box_hi)
    assert np.isclose(distance, .13) and edge == "upper_y"
