from types import SimpleNamespace

import numpy as np
import pytest

from tools87 import terrain87 as terrain


def test_three_roads_share_the_level_approach_and_use_opposite_four_degree_slopes():
    start = terrain.RAMP_START_X_M
    xs = np.array([start - 0.2, start, start + 1, start + 2, start + 3])
    rise = np.tan(np.deg2rad(4.0))
    assert np.allclose(terrain.ground_height("up", xs), [0, 0, rise, 2 * rise, 2 * rise])
    assert np.allclose(terrain.ground_height("down", xs), -terrain.ground_height("up", xs))
    assert np.array_equal(terrain.ground_height("flat", xs), np.zeros_like(xs))
    with pytest.raises(ValueError):
        terrain.ground_height("cycle", 0)


def test_hfield_rows_have_the_same_cross_track_profile_and_a_vertex_at_the_toe():
    rows, cols = terrain.GRID_ROWS, terrain.GRID_COLS
    model = SimpleNamespace(nhfield=1, hfield_nrow=[rows], hfield_ncol=[cols],
                            hfield_data=np.empty(rows * cols, dtype=np.float32))
    xs = terrain.RAMP_START_X_M + np.linspace(-terrain.HALF_WIDTH_X_M, terrain.HALF_WIDTH_X_M, cols)
    assert xs[cols // 2] == terrain.RAMP_START_X_M
    for grade in terrain.TERRAINS:
        terrain._set_road(model, grade)
        field = model.hfield_data.reshape(rows, cols)
        assert np.count_nonzero(field != field[0]) == 0
        assert np.allclose(terrain.FIELD_BASE_Z_M + terrain.FIELD_SCALE_Z_M * field[0],
                           terrain.ground_height(grade, xs), atol=2e-8)


def test_range_mask_rejects_robot_first_hits_and_invalid_ranges():
    distances = [terrain.MIN_RANGE_M, 0.4, 1.3, terrain.MAX_RANGE_M, -1, np.nan, 0.1, 1.6]
    self_hits = [False, True, False, False, False, False, False, False]
    frame = terrain.range_features(distances, self_hits)
    assert frame.dtype == np.float32
    assert np.isfinite(frame).all()
    assert frame[8:].tolist() == [1, 0, 1, 1, 0, 0, 0, 0]
    assert np.allclose(frame[:8] * terrain.MAX_RANGE_M,
                       [terrain.MIN_RANGE_M, 0, 1.3, terrain.MAX_RANGE_M, 0, 0, 0, 0])


def test_first_return_self_filter_uses_distances_and_ignores_semantic_hit_ids(monkeypatch):
    all_distances = [0.8, 0.5, 0.3, -1.0] * 2
    robot_distances = [0.8, 1.0, -1.0, -1.0] * 2
    calls = []

    def cast(model, data, origin, direction, geomgroup, include_static, exclude, hit_id):
        index = len(calls) // 2
        calls.append(include_static)
        hit_id[:] = 1973
        return (all_distances if include_static else robot_distances)[index]

    monkeypatch.setattr(terrain.mujoco, "mj_ray", cast)
    data = SimpleNamespace(xmat=[np.eye(3)], xpos=[np.zeros(3)])
    frame = terrain.D1Terrain87._measure(SimpleNamespace(base_body=0, data=data, model=object()))
    assert calls == [True, False] * 8
    assert frame[8:].tolist() == [0, 1, 1, 0] * 2
    assert np.allclose(frame[:8] * terrain.MAX_RANGE_M, [0, 0.5, 0.3, 0] * 2)


def test_range_capture_delivery_and_actor_mask_on_same_raw_packets():
    zeros = np.zeros(16, dtype=np.float32)
    obs = np.arange(627, dtype=np.float32)
    captures = []

    def measure():
        captures.append(tick)
        return np.full(16, tick + 1, dtype=np.float32)

    stub = SimpleNamespace(_pending=None, _measure=measure, last_raw_features=zeros.copy(),
                           sensor_capture_tick=-1, sensor_delivery_tick=-1, _masked=zeros)
    for tick in range(52):
        terrain.D1Terrain87._update_measurement(stub, tick)
        delivered = [capture for capture in captures if capture + 2 <= tick]
        assert stub.sensor_capture_tick == (delivered[-1] if delivered else -1)
        assert stub.sensor_delivery_tick == (delivered[-1] + 2 if delivered else -1)
        expected = delivered[-1] + 1 if delivered else 0
        assert np.all(stub.last_raw_features == expected)
        for enabled in (True, False):
            stub.perception = enabled
            actor = terrain.D1Terrain87._with_ranges(stub, obs)
            assert np.array_equal(actor[:627], obs)
            assert np.all(actor[627:] == (expected if enabled else 0))
    assert captures == [0, 9, 17, 25, 34, 42, 50]
