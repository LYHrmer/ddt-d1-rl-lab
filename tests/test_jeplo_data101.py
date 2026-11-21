from types import SimpleNamespace

import numpy as np
import pytest

from tools101.data101 import delivery_inputs, packet_views, prefix_bound


def body_history():
    frame = np.arange(12 * 57, dtype=np.float32).reshape(12, 57)
    issued = np.arange(11 * 16, dtype=np.float32).reshape(11, 16)
    frame[1:, 41:57] = issued
    return frame, issued


def test_delivery_uses_predecision_frame_and_only_already_issued_actions():
    frame, issued = body_history()
    body, actions = delivery_inputs(frame, issued, 8)
    assert body.shape == (38,) and np.array_equal(body, np.r_[frame[8, :6], frame[8, 9:41]])
    assert np.array_equal(actions, issued[[6, 7]])
    issued[8:] = -12345
    frame[9:] = -67890
    assert np.array_equal(delivery_inputs(frame, issued, 8)[0], body)
    assert np.array_equal(delivery_inputs(frame, issued, 8)[1], actions)
    with pytest.raises(ValueError, match="crosses the episode"):
        delivery_inputs(frame, issued, 1)
    frame[8, -1] += 1
    with pytest.raises(ValueError, match="wrong time index"):
        delivery_inputs(frame, issued, 8)


class FixedPoseSensor:
    def __init__(self):
        self.calls = []
        self.tick = 0
        self.data = SimpleNamespace(xmat=np.eye(3).reshape(1, 9), time=0.)
        self.heading = np.eye(3)
        self.base = 0
        self.angle_keep = np.ones(520, dtype=bool)
        self.angle_keep[0] = False
        self.geometry = dict(range_min_m=.14, range_max_m=1.45, box_height_m=.03, box_center_x_m=.5,
                             box_abs_y_m=.2, box_half_x_m=.1, box_half_y_m=.1)

    def pose(self, q):
        self.tick = float(q[0])
        query = np.array([[.5, .2], [.5, -.2]])
        return dict(query_xy=query, world_query_xy=query + [self.tick, 0.], wheel_max_x_bound=self.tick)

    def capture(self):
        self.calls.append(self.tick)
        ranges = np.full(520, 0.7)
        ranges[1] = -1.
        self_hit = np.zeros(520, dtype=bool)
        self_hit[2] = True
        points = np.tile([.5, -.2, 0.], (520, 1))
        points[0] = [.5, .2, .03]
        points[1:3] = np.nan
        return dict(range_m=ranges, self_hit=self_hit, points=points)


def test_same_first_return_serves_both_views_with_delayed_labels(tmp_path):
    scene = FixedPoseSensor()
    frame, issued = body_history()
    episode = dict(qpos=np.arange(1, 12)[:, None] / 100., frame=frame, issued=issued, source_seed=12, episode=3)
    path = tmp_path / "native.npz"
    rows = packet_views(scene, episode, "synthetic_left", "left", 0, [3, 7], 2, ["clear", "masked"], path)
    assert scene.calls == [.03, .07]
    for packet, tick in enumerate((3, 7)):
        clear, masked = rows[2 * packet:2 * packet + 2]
        assert clear["cloud_id"] == masked["cloud_id"] == packet
        assert clear["capture_tick"] == tick and clear["delivery_tick"] == tick + 2
        np.testing.assert_allclose(clear["world_query_xy"] - clear["query_xy"], [[(tick + 2) / 100, 0]] * 2)
        assert clear["target_mm"] == masked["target_mm"] == 30
        assert clear["geometric_mm"] == pytest.approx(30.) and np.isnan(masked["geometric_mm"])
        assert clear["depth_m"].dtype == np.float32 and not masked["valid"][:3].any()
        assert np.all(masked["depth_m"][:3] == 0) and np.array_equal(masked["depth_m"][3:], clear["depth_m"][3:])
        assert np.array_equal(clear["action_history"], issued[tick:tick + 2])
    with np.load(path, allow_pickle=False) as native:
        assert native.files == ["tick_003", "tick_007"]
        assert native["tick_003"].dtype.names == ("range_m", "self_hit")
        assert native["tick_003"]["self_hit"][2]
    with pytest.raises(FileExistsError):
        packet_views(scene, episode, "synthetic_left", "left", 0, [3, 7], 2, ["clear", "masked"], path)


def test_precontact_guard_checks_all_intermediate_boundaries():
    episode = dict(reset=np.array([0.]), qpos=np.array([[.01], [.8], [.02]]), source_seed=12, episode=3)
    with pytest.raises(ValueError, match="counterfactual wheel contact"):
        prefix_bound(FixedPoseSensor(), episode, .7)
