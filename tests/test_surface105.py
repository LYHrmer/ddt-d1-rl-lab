import numpy as np
import pytest

from tools100.score100 import estimate_delta_mm
from tools105.surface105 import surface_delta_mm, surface_heights_m


def grid(xs, ys, height):
    return np.array([[x, y, height(x, y)] for x in xs for y in ys])


def ledge(with_face):
    left = np.linspace(0.16, 0.24, 5)
    ground = grid([-0.09, -0.05, -0.025], left, lambda x, y: 0.)
    top = grid([0.012, 0.035, 0.065, 0.09], left, lambda x, y: 0.047)
    face = np.array([[0., y, z] for y in left for z in (0.004, 0.018, 0.036)])
    right = grid(np.linspace(-0.09, 0.09, 9), -left, lambda x, y: 0.)
    return np.vstack([ground, top, right, face] if with_face else [ground, top, right])


def queries(x):
    return np.array([[x, 0.2], [x, -0.2]])


def test_riser_locates_boundary_without_becoming_the_support_height():
    cloud = ledge(with_face=True)
    q = queries(0.005)
    assert abs(estimate_delta_mm(cloud, q) - 47.) > 10.
    assert surface_delta_mm(cloud, q) == pytest.approx(47.)
    assert surface_delta_mm(cloud, queries(-0.007)) == pytest.approx(0.)
    reflected = cloud * [1, -1, 1] + [0.14, 0.02, -0.39]
    assert surface_delta_mm(reflected[::-1], q + [0.14, 0.02]) == pytest.approx(-47.)


def test_occluded_boundary_abstains_between_observed_supports():
    cloud = ledge(with_face=False)
    assert surface_delta_mm(cloud, queries(0.005)) is None
    assert surface_delta_mm(cloud, queries(0.05)) == pytest.approx(47.)
    assert surface_delta_mm(cloud, queries(-0.06)) == pytest.approx(0.)


def test_slope_is_one_surface_and_each_wheel_keeps_its_height():
    xy = np.linspace(-0.08, 0.08, 9)
    slope = lambda x, y: 0.18*x - 0.2*y - 0.42
    points = np.vstack([grid(xy, xy + 0.2, slope), grid(xy, xy - 0.2, slope), [np.nan]*3])
    q = queries(-0.018)
    assert surface_heights_m(points, q) == pytest.approx([slope(*p) for p in q])
    assert surface_delta_mm(points, q) == pytest.approx(-80.)


def test_stacked_surfaces_and_missing_support_do_not_get_confident_reads():
    xy = np.linspace(-0.07, 0.07, 7)
    flat = grid(xy, xy + 0.2, lambda x, y: 0.)
    assert surface_heights_m(flat, queries(0.)) == (0., None)
    stacked = np.vstack([flat, flat + [0, 0, 0.073]])
    assert surface_heights_m(stacked, queries(0.)) == (None, None)
    assert surface_delta_mm(np.empty((0, 3)), queries(0.)) is None
    assert surface_delta_mm([[0, 0.2, 0.1]], queries(0.)) is None


def test_oblique_visible_face_uses_each_query_position():
    angle = np.deg2rad(18)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = ledge(with_face=True)
    q = queries(0.006)
    points[:, :2] = points[:, :2] @ rot.T
    assert surface_delta_mm(points, q @ rot.T) == pytest.approx(47.)
