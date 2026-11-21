from itertools import product

import numpy as np
import pytest

from tools101.score101 import EPISODES, LAYOUTS, TICKS, VISIBILITIES
from tools102.replay102 import (archived_predictions, geometry_outputs, geometry_statistics,
                                remove_past_depth, roster)


def test_roster_reuses_each_original_prediction_for_both_target_sides_and_keeps_all_arrivals():
    keys = list(product(EPISODES, LAYOUTS, VISIBILITIES, TICKS))
    episode, layout, view, tick = map(np.array, zip(*keys, strict=True))
    series = {key: i for i, key in enumerate(product(EPISODES, LAYOUTS, VISIBILITIES))}
    clouds = {key: i for i, key in enumerate(product(EPISODES, LAYOUTS, TICKS))}
    n = len(keys)
    parent = dict(row_id=np.arange(n), source_seed=np.full(n, 89037), episode=episode, layout=layout,
                  side=np.where(layout == LAYOUTS[0], "left", "right"), visibility=view,
                  capture_tick=tick, delivery_tick=tick + 2, cloud_id=np.array([clouds[e, l, t] for e,l,_,t in keys]),
                  series_id=np.array([series[e,l,v] for e,l,v,_ in keys]), target_mm=np.zeros(n),
                  geometric_mm=np.zeros(n), query_xy=np.zeros((n, 2, 2)))
    plan = dict(source_seed=89037, episodes=list(EPISODES), layouts=list(LAYOUTS),
                target_sides=["left", "right"], capture_ticks=list(TICKS))
    result = roster(parent, plan)
    assert len(np.unique(result["series_id"])) == 20
    assert np.unique(result["cloud_id"], return_counts=True)[1].tolist() == [2] * 160
    assert all(set(result["mask_side"][result["cloud_id"] == i]) == {"left", "right"} for i in range(160))
    assert np.all(parent["visibility"][result["masked_row"]] == "masked")
    assert np.all(parent["visibility"][result["clear_row"]] == "clear")
    values = np.arange(3 * n).reshape(3, n)
    pred = archived_predictions(result, [11, 22, 33],
                                dict(seeds=[11, 22, 33], baseline_mm=values, predictive_mm=values+7))
    np.testing.assert_array_equal(pred[:, 0, :], values[:, result["masked_row"]])
    np.testing.assert_array_equal(pred[:, 1, :], values[:, result["masked_row"]] + 7)


def test_geometry_recovery_loss_threshold_and_distinct_episodes_are_counted_separately():
    old = np.array([30., -30., 30., 5., 30., 0., -30.])
    new = np.array([15., np.nan, 15., 15., 30., 0., -30.])
    pack = dict(original_geometric_mm=old, moved_geometric_mm=new,
                target_mm=np.array([30., -30., 30., 30., 30., 0., -30.]),
                episode=np.array([0, 3, 6, 9, 9, 12, 12]), active=np.array([1, 1, 1, 1, 1, 0, 0], bool),
                mask_side=np.array(["left", "right", "left", "left", "right", "left", "left"]),
                cloud_id=np.arange(7), original_nearest_mm=np.ones((7, 2)), moved_nearest_mm=np.full((7, 2), 9.))
    plan = dict(episodes=[0, 3, 6, 9, 12], target_sides=["left", "right"], gates=dict(minimum_episodes=3, change_mm=10))
    info, damaged, target_high = geometry_statistics(pack, plan)
    assert info["challenge_established"] and info["damaged_episodes"] == [0, 3, 6]
    assert info["active"]["regressed"] == 3 and info["active"]["material_change_or_loss"] == 4
    assert info["active"]["lost_estimate"] == 1 and info["active"]["nearest_change_median_mm"] == 8.
    assert np.flatnonzero(damaged).tolist() == [0, 1, 2] and np.flatnonzero(target_high).tolist() == [0, 1, 2, 3]
    pack["mask_side"][1] = "left"
    assert not geometry_statistics(pack, plan)[0]["challenge_established"]
    pack["moved_geometric_mm"][-1] = 0.
    with pytest.raises(ValueError, match="inactive geometry"):
        geometry_statistics(pack, plan)


def test_independent_solver_preserves_nearest_ties_and_records_missing_observations():
    xyz = np.array([[[-.01, .2, .02], [.01, .2, .04], [0., -.2, 0.]]])
    query = np.array([[[0., .2], [0., -.2]]])
    settings = dict(radius_m=.15, tie_tolerance_m=1e-12)
    for independent in (False, True):
        both, distances = geometry_outputs(xyz, np.array([[True, True, True]]), query, settings, independent)
        assert both[0] == pytest.approx(30.) and distances.tolist() == [[10., 0.]]
        unavailable, farther = geometry_outputs(xyz, np.array([[False, False, True]]), query, settings, independent)
        assert np.isnan(unavailable).all() and farther.tolist() == [[400., 0.]]


def test_past_depth_ablation_preserves_current_returns_and_all_proprioception_and_actions():
    import torch
    from tools101.fit101 import DeliveredPackets

    ticks = np.array(TICKS[:4])
    raw = dict(series_id=np.zeros(4, dtype=int), capture_tick=ticks, delivery_tick=ticks+2,
               depth_m=np.tile(np.linspace(.2, .8, 4, dtype=np.float32)[:, None], (1, 520)),
               valid=np.ones((4, 520), dtype=bool), proprio=np.arange(4*38, dtype=np.float32).reshape(4, 38),
               action_history=np.broadcast_to(np.arange(4, dtype=np.float32)[:, None, None]/4, (4, 2, 16)).copy())
    packets = DeliveredPackets(raw)
    original = packets.history.clone()
    blank = remove_past_depth(packets)
    torch.testing.assert_close(packets.history, original)
    for r in range(len(ticks)):
        for slot, source in enumerate(original[r].tolist()):
            changed = int(blank.history[r, slot])
            if source < 0:
                assert changed == -1
                continue
            torch.testing.assert_close(blank.proprio[changed], packets.proprio[source])
            torch.testing.assert_close(blank.actions[changed], packets.actions[source])
            if slot == 2:
                assert changed == source
                torch.testing.assert_close(blank.depth[changed], packets.depth[source])
                torch.testing.assert_close(blank.valid[changed], packets.valid[source])
            else:
                assert not blank.depth[changed].any() and not blank.valid[changed].any()
