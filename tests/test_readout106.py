import numpy as np

from tools106.history_readout import after_abstention, packet_readouts
from tools106.footprint106 import rear_returns, support


def patches(cx, height=0.03):
    return np.array([[cx+x, y+dy, z] for y, z in ((.2, height), (-.2, 0.))
                     for x in (-.012, .012) for dy in (-.012, .012)])


def test_previous_packet_is_queried_at_new_location_and_only_after_abstention():
    queries = np.array([[0., .2], [0., -.2]])
    old = patches(.22)
    motion = np.array([.22, 0., 0.])
    assert after_abstention(None, queries, old, np.zeros(3)) is None
    assert np.isclose(after_abstention(None, queries, old, motion), 30.)
    assert after_abstention(0., queries, old, motion) == 0.
    assert after_abstention(None, queries, old[:4], motion) is None
    lifted = old.copy()
    lifted[:, 2] += .09
    assert np.isclose(after_abstention(None, queries, lifted, motion), 30.)


def test_resets_and_unknowns_do_not_propagate_an_old_value():
    queries = np.tile([[0., .2], [0., -.2]], (4, 1, 1))
    points = np.full((4, 8, 3), np.nan)
    points[0] = patches(0.)
    raw, candidate = packet_readouts(points, queries, [-1, 0, 1, -1], np.zeros((4, 3)))
    np.testing.assert_allclose(raw, [30., np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(candidate, [30., 30., np.nan, np.nan], equal_nan=True)


def test_only_local_rear_strip_ground_can_bound_the_query():
    fixture = dict(box_center_x_m=.4, box_half_x_m=.05, box_abs_y_m=.2, box_half_y_m=.09)
    query = np.array([.49, .2])
    current = np.array([[.48, .18, 0.], [.48, .22, 0.], [.50, .20, .03], [.44, .2, 0.]])
    prior = np.array([[.51, .19, 0.], [.51, .21, 0.], [.48, .30, 0.], [.80, .20, 0.]])
    a, _ = rear_returns(current, query, fixture, "left")
    b, _ = rear_returns(prior, query, fixture, "left")
    assert support(a) == support(b) == dict(local_points=2, encloses_query=False)
    assert support(np.concatenate([a, b])) == dict(local_points=4, encloses_query=True)
