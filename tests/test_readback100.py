import json

import numpy as np
import pytest

from tools100.common100 import PLAN
from tools100.readback100 import expected_indices, independent_estimate, reference_delta_mm
from tools100.score100 import estimate_delta_mm


REGISTRATION = json.loads(PLAN.read_text())
GEOMETRY = REGISTRATION["geometry"]


def test_registered_angular_mask_and_sparse_beams_in_original_order():
    sparse, keep = expected_indices(GEOMETRY)
    np.testing.assert_array_equal(sparse, [253, 123, 58, 6, 513, 383, 318, 266])
    assert keep.shape == (520,) and keep.dtype == np.bool_ and keep.sum() == 484
    np.testing.assert_array_equal(keep[sparse], [True, False, True, True] * 2)
    by_mount = keep.reshape(2, 20, 13)
    np.testing.assert_array_equal(by_mount[0], by_mount[1])
    # At elevations 48, 51, 54, 56, 57, 60 the inclusive stripe blocks
    # azimuths -4, 0, +4. Its immediate neighbours are retained.
    assert not by_mount[:, 6:12, 5:8].any()
    assert by_mount[:, 6:12, [4, 8]].all() and by_mount[:, [5, 12], :].all()


@pytest.mark.parametrize("layout, lane, expected", [("left", 0, 30.0), ("right", 1, -30.0)])
def test_label_matches_mirrored_fixture_and_inclusive_world_xy_boundaries(layout, lane, expected):
    center_x, center_y = GEOMETRY["box_center_x_m"], GEOMETRY["box_abs_y_m"]
    half = np.array([GEOMETRY["box_half_x_m"], GEOMETRY["box_half_y_m"]])
    queries = np.array([[center_x, center_y], [center_x, -center_y]])
    assert reference_delta_mm(GEOMETRY, "flat", queries) == 0.0
    assert reference_delta_mm(GEOMETRY, layout, queries) == expected
    assert reference_delta_mm(GEOMETRY, layout, queries[::-1]) == -expected
    center = queries[lane].copy()
    for axis in (0, 1):
        for step in (-1, 1):
            point = queries.copy()
            point[lane, axis] = center[axis] + step * half[axis]
            assert reference_delta_mm(GEOMETRY, layout, point) == expected
            point[lane, axis] += step * 1e-8
            assert reference_delta_mm(GEOMETRY, layout, point) == 0.0


def test_independent_estimator_matches_online_for_ties_missing_returns_and_one_sided_support():
    query = np.array([[0.0, 0.2], [0.0, -0.2]])
    cloud = np.array([[0.05, 0.2, 0.040], [-0.05 - 4e-13, 0.2, 0.020],
                      [0.05 + 2e-12, 0.2, 0.4], [np.nan, np.nan, np.nan],
                      [0.10, -0.2, 0.005], [0.2, -0.2, 0.5]])
    estimator = REGISTRATION["estimator"]
    for points in (cloud, cloud[::-1], cloud[[3, 1, 4, 5, 2, 0]]):
        replay = independent_estimate(points, query, estimator)
        assert replay == pytest.approx(25.0)
        assert estimate_delta_mm(points, query) == pytest.approx(replay)
    for incomplete in (cloud[[0, 1, 2, 3, 5]], cloud[[3]], np.empty((0, 3))):
        assert independent_estimate(incomplete, query, estimator) is None
        assert estimate_delta_mm(incomplete, query) is None
