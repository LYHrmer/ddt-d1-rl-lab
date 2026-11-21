import json
from pathlib import Path

import numpy as np
import pytest

from tools102.mask102 import CAPTURE_TICKS, calibration, level_points, packet_masks


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = json.loads((ROOT / "docs/experiments100/pointcloud_density.json").read_text())["geometry"]


def test_nominal_angular_windows_use_the_registered_left_right_mirror_and_clip_opposite_eye():
    result = calibration(GEOMETRY)
    assert result["window_shape"] == [6, 3]
    assert result["original_starts"] == [[6, 5], [6, 5]]
    assert result["moved_starts"] == {"left": [[12, 8], [4, 10]], "right": [[4, 0], [12, 2]]}
    assert result["nominal_headings_azimuth_depression_deg"]["left"][1][0] > max(GEOMETRY["azimuth_deg"])
    assert result["nominal_headings_azimuth_depression_deg"]["right"][0][0] < min(GEOMETRY["azimuth_deg"])
    assert result["query_in_nominal_base_m"][0][:2] == pytest.approx([.51921393763024, .220310164962739])


def test_two_packet_bursts_translate_each_18_ray_window_without_changing_exposure_or_area():
    ticks = np.tile(CAPTURE_TICKS, 2)
    sides = np.repeat(["left", "right"], len(CAPTURE_TICKS))
    fixed, moved, active = packet_masks(ticks, sides, GEOMETRY)
    np.testing.assert_array_equal(active, np.tile([False, False, True, True], 8))
    np.testing.assert_array_equal(fixed[~active], moved[~active])
    for mask in (fixed, moved):
        eye = mask.reshape(-1, 2, 20, 13)
        assert (eye.sum(axis=(2, 3)) == 18).all()
        assert (eye.any(axis=3).sum(axis=2) == 6).all()
        assert (eye.any(axis=2).sum(axis=2) == 3).all()
    assert not (fixed[active] & moved[active]).any()
    a = moved[2].reshape(2, 20, 13)
    b = moved[len(CAPTURE_TICKS) + 2].reshape(2, 20, 13)
    np.testing.assert_array_equal(a, b[::-1, :, ::-1])
    np.testing.assert_array_equal(fixed[2], fixed[len(CAPTURE_TICKS) + 2])
    # Phase follows the native capture tick, including sliced/out-of-order queries.
    order = np.array([31, 18, 7, 0, 18])
    f, m, active_again = packet_masks(ticks[order], sides[order], GEOMETRY)
    assert np.array_equal(f, fixed[order]) and np.array_equal(m, moved[order])
    assert np.array_equal(active_again, active[order])
    with pytest.raises(ValueError, match="6 Hz"):
        packet_masks([26], "left", GEOMETRY)
    with pytest.raises(ValueError, match="wheel side"):
        packet_masks([25], "choose-by-label", GEOMETRY)


def test_range_reconstruction_preserves_two_mount_origins_and_capture_attitude():
    ranges = np.full((1, 520), .5)
    valid = np.ones(ranges.shape, dtype=bool)
    valid[0, 6] = False
    rot = np.array([[[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]])
    points = level_points(ranges, valid, rot, GEOMETRY)
    phi, theta = np.radians([-24, 33])
    ray = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
    for index, mount in zip((0, 260), GEOMETRY["mounts_m"], strict=True):
        np.testing.assert_allclose(points[0, index], (np.array(mount) + .5 * ray) @ rot[0], atol=1e-12)
    assert np.isnan(points[0, 6]).all() and np.isfinite(points[0, 7]).all()
    with pytest.raises(ValueError, match="validity"):
        level_points(ranges, valid.astype(int), rot, GEOMETRY)
