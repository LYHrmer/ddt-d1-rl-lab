from types import SimpleNamespace

import numpy as np
import pytest

from tools94.common94 import digest, read
from tools100.readback100 import expected_indices
from tools105 import data105 as data


def test_capture_level_is_right_handed_and_obeys_measured_up():
    up = np.array([[0., 0., 1.], [.48, .36, .8]])
    basis = data._level(-up)
    np.testing.assert_allclose(basis[0], np.eye(3))
    np.testing.assert_allclose(basis[:, :, 2], up)
    np.testing.assert_allclose(basis[1].T @ basis[1], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(basis), [1., 1.])
    with pytest.raises(ValueError, match="horizontal forward"):
        data._level(np.array([[-1., 0., 0.]]))


def test_source_archive_preserves_existing_artifacts(tmp_path):
    out = tmp_path / "existing.npz"
    out.write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        data.build_windows(out)
    assert out.read_bytes() == b"partial"


def test_native_audit_checks_original_precision_self_hits_and_angular_mask(tmp_path, monkeypatch):
    geometry = read(data.ROOT / "docs/experiments100/pointcloud_density.json")["geometry"]
    _, angular = expected_indices(geometry)
    raw = np.empty(520, dtype=[("range_m", "<f8"), ("self_hit", "?")])
    raw["range_m"] = 0.40000000003
    raw["self_hit"] = False
    raw["range_m"][0] = -1.
    raw["self_hit"][1] = True
    folder = tmp_path / "prepare_01/native"
    folder.mkdir(parents=True)
    path = folder / "train_89023_ep02_narrow_left.npz"
    np.savez(path, tick_025=raw)
    clear = (raw["range_m"] >= geometry["range_min_m"]) & ~raw["self_hit"]
    valid = np.array([clear, clear & angular])
    p = SimpleNamespace(geometry=geometry, valid=valid, depth=np.where(valid, raw["range_m"], 0.).astype(np.float32),
                        info={"rows": 2, "clouds": 1, "native_first_return_files_sha256": {path.name: digest(path)}},
                        rows={"series_id": np.array([0, 1]), "visibility": np.array(["clear", "masked"]),
                              "source_seed": np.array([89023, 89023]), "episode": np.array([2, 2]),
                              "layout": np.array(["narrow_left"] * 2), "capture_tick": np.array([25, 25])})
    monkeypatch.setattr(data, "BASE", tmp_path)
    audit = data._audit_native_ranges(p)
    assert audit["native_ray_bins_checked"] == 520 and audit["public_view_bins_checked"] == 1040
    assert audit["public_float32_equals_rounded_native"] and 0 < audit["max_valid_range_roundoff_m"] < 1e-7
    p.depth[0, 2] = np.nextafter(p.depth[0, 2], np.float32(1.))
    with pytest.raises(ValueError, match="published range or angular mask"):
        data._audit_native_ranges(p)


def test_common_inputs_preserve_ray_order_and_only_integrate_through_delivery(monkeypatch):
    geometry = read(data.ROOT / "docs/experiments100/pointcloud_density.json")["geometry"]
    # The received encoder field is saved in fixed scale. Use a representable source value.
    speed_scaled = np.float32(.5)
    proprio = np.zeros((1, 38), dtype=np.float32)
    proprio[:, 5] = -1.
    proprio[:, 25:38:4] = speed_scaled
    query = np.array([[[.4, .2], [.4, -.2]]])
    valid = np.ones((1, 520), dtype=bool)
    valid[:, 19] = False
    p = SimpleNamespace(rows={"row_id": np.array([0])}, proprio=proprio, geometry=geometry, window=np.array([0]),
                        depth=np.full((1, 520), .4, dtype=np.float32), valid=valid, query=query)
    gyro = np.zeros((1, 3, 3))
    gravity = np.broadcast_to([0., 0., -1.], (1, 3, 3))
    # The first wheel value is zero, the others 10 rad/s. Trapezoidal travel = (5+10)*0.02*0.087 m.
    wheels = np.broadcast_to(np.array([0., 10., 10.]).reshape(1, 3, 1), (1, 3, 4))
    monkeypatch.setattr(data, "_packets", lambda split: p)
    monkeypatch.setattr(data, "_measurements", lambda split, path, packets:
                        dict(gyro_body=gyro, gravity_body=gravity, wheel_rad_s=wheels))
    _, points, corrected = data.inputs("design")
    np.testing.assert_allclose(corrected, query + [0.0261, 0.])
    assert np.array_equal(np.isfinite(points).all(axis=2), valid) and np.isnan(points[0, 19]).all()
    # The two mounts have the same first beam direction but different offsets.
    np.testing.assert_allclose(points[0, 260] - points[0, 0], np.diff(geometry["mounts_m"], axis=0)[0])
    proprio[:, 25] = np.float32(.4)
    with pytest.raises(ValueError, match="different onboard sensors"):
        data.inputs("design")
