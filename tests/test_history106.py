import numpy as np
import pytest

from tools106 import history106 as h


def _fixture():
    captures = np.array([25, 34, 42])
    rows = dict(row_id=np.arange(6), series_id=np.repeat([0, 1], 3), source_seed=np.full(6, 7),
                episode=np.zeros(6, dtype=int), layout=np.full(6, "small_left"), side=np.full(6, "left"),
                visibility=np.repeat(["clear", "masked"], 3), capture_tick=np.tile(captures, 2),
                delivery_tick=np.tile(captures + 2, 2))
    meta = dict(episode_keys=np.array([[7, 0]]), capture_ticks=captures, sample_ticks=np.arange(25, 43))
    return rows, meta


def test_received_tape_matches_current_sensors_every_tick_and_ignores_other_channels():
    frames = np.full((41, 57), np.nan, dtype=np.float32)
    frames[:, h.SENSOR_COLUMNS] = np.arange(410).reshape(41, 10)
    actor = np.zeros((41, 11, 57), dtype=np.float32)
    actor[:, 0] = frames
    for tick in range(10, 41):
        actor[tick, 1:] = frames[tick-10:tick]
    captures = [10, 19, 27]
    np.testing.assert_array_equal(h._sensor_tape(actor.reshape(41, 627), captures), frames[10:28, h.SENSOR_COLUMNS])
    actor[19, -5, h.SENSOR_COLUMNS[-1]] += 0.5
    with pytest.raises(ValueError, match="consecutive current sensors"):
        h._sensor_tape(actor.reshape(41, 627), captures)
    with pytest.raises(ValueError, match="outside on-board history"):
        h._sensor_tape(actor.reshape(41, 627), [10, 21])


def test_motion_integrates_interior_samples_once_and_resets_at_visibility_boundary(monkeypatch):
    rows, meta = _fixture()
    points = np.zeros((6, 2, 3))
    queries = np.zeros((6, 2, 2))
    gyro, gravity = np.zeros((1, 18, 3)), np.zeros((1, 18, 3))
    gravity[:, :, 2] = -1.
    wheels = np.zeros((1, 18, 4))
    wheels[0, 2:5] = 1. / 0.087
    gyro[0, 11:13, 2] = 1.

    def sensors(path, *, oracle):
        assert path == "sensors" and oracle is False
        return meta, dict(gyro_body=gyro, gravity_body=gravity, wheel_rad_s=wheels)

    monkeypatch.setattr(h, "inputs", lambda split, path: (rows, points, queries))
    monkeypatch.setattr(h, "_open_archive", sensors)
    out_rows, out_points, out_queries, previous, motion = h.history_inputs("design", "sensors")
    assert out_rows is rows and out_points is points and out_queries is queries
    np.testing.assert_array_equal(previous, [-1, 0, 1, -1, 3, 4])
    assert np.isnan(motion[[0, 3]]).all()
    np.testing.assert_allclose(motion[[1, 2]], [[.06, 0., 0.], [0., 0., .04]], atol=1e-14)
    np.testing.assert_array_equal(motion[:3], motion[3:])
    rows["series_id"][:] = 0
    with pytest.raises(ValueError, match="previous packet crosses"):
        h.history_inputs("design", "sensors")


def test_oracle_keeps_capture_and_delivery_distinct_and_rejects_a_wrong_clock(monkeypatch):
    rows, meta = _fixture()
    poses = np.zeros((1, 3, 2, 7))
    poses[0, :, :, 0] = np.column_stack((meta["capture_ticks"], meta["capture_ticks"] + 2))
    poses[..., 3] = 1.

    def scorer(path, *, oracle):
        assert path == "refs" and oracle is True
        return meta, {"reference_poses": poses}

    monkeypatch.setattr(h, "_open_archive", scorer)
    pairs = h.oracle_pairs(rows, "refs")
    separate = h.oracle_capture_delivery(rows, "refs")
    np.testing.assert_array_equal(separate[:3, :, 0], [[25, 27], [34, 36], [42, 44]])
    np.testing.assert_array_equal(pairs[[1, 2, 4, 5], :, 0], [[25, 34], [34, 42], [25, 34], [34, 42]])
    assert np.isnan(pairs[[0, 3]]).all()
    rows["delivery_tick"][1] += 1
    with pytest.raises(ValueError, match="packet clock"):
        h.oracle_pairs(rows, "refs")


def test_archives_check_type_and_reject_a_forged_measurement_receipt(tmp_path, monkeypatch):
    _, meta = _fixture()
    del meta["sample_ticks"]
    monkeypatch.setattr(h, "_sources", lambda: ((), meta.copy()))
    clean = dict(**meta, sample_ticks=np.arange(25, 43), gyro_body=np.zeros((1, 18, 3)),
                 gravity_body=np.tile([0., 0., -1.], (1, 18, 1)), wheel_rad_s=np.zeros((1, 18, 4)))
    clean["content_sha256"] = h._value_sha({k: clean[k] for k in h.MEASUREMENTS})
    monkeypatch.setattr(h, "MEASURED_SHA", clean["content_sha256"])
    archive = tmp_path / "sensors.npz"
    np.savez(archive, **clean)
    _, fields = h._open_archive(archive, oracle=False)
    assert list(fields) == list(h.MEASUREMENTS)
    with pytest.raises(ValueError, match="wrong tape type"):
        h._open_archive(archive, oracle=True)
    clean["wheel_rad_s"][0, 5, 1] = 0.01
    np.savez(archive, **clean)
    with pytest.raises(ValueError, match="changed archived values"):
        h._open_archive(archive, oracle=False)
    np.savez(archive, reference_poses=np.zeros((1, 3, 2, 7)), **clean)
    with pytest.raises(ValueError, match="wrong tape type"):
        h._open_archive(archive, oracle=False)
