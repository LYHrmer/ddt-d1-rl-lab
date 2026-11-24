import numpy as np
from itertools import product

from tools94.common94 import arrays, read
from tools100.common100 import protocol
from tools100.sensor100 import _grid
from tools107.capture107 import geometry
from tools107.mount107 import mount_positions
from tools107.score107 import comparison, score
from tools107.visibility107 import PLAN, direction_in_view


def test_unknowns_and_wrong_number_are_reported_separately_and_threshold_is_fixed():
    y = np.array([0., 0., -30., 30., 30.])
    a = np.array([np.nan, 0., -30., np.nan, 30.])
    b = np.array([0., 30., -19.99, 20., 30.])
    outcome = comparison(y, a, b)
    assert outcome["original"]["correct"] == outcome["forward"]["correct"] == 3
    assert outcome["gained"] == outcome["regressed"] == 2
    assert outcome["original"]["unknown"] == 2 and outcome["forward"]["unknown"] == 0
    assert outcome["forward"]["flat_false_positive"] == 1 and outcome["forward"]["wrong_numeric"] == 2


def test_clear_direction_can_still_be_outside_shutter_or_view():
    spec = protocol()["geometry"]
    origin, quat = np.zeros(3), [1., 0., 0., 0.]
    open_ray = direction_in_view(np.array([.3, 0., -.3]), origin, quat, spec)
    shut = direction_in_view(np.array([.3, 0., -.4]), origin, quat, spec)
    side = direction_in_view(np.array([.3, .5, -.4]), origin, quat, spec)
    assert open_ray["in_masked_angular_range"]
    assert shut["in_angular_range"] and not shut["in_masked_angular_range"]
    assert not side["in_angular_range"]


def test_whole_public_readback_keeps_every_unknown_and_fails_the_exit_gate(tmp_path):
    roster = read(PLAN)["validation"]
    keys = list(product(roster["layouts"], roster["arms"], roster["seeds"], roster["ticks"]))
    n = len(keys)
    layout, arm, seed, ticks = map(np.asarray, zip(*keys, strict=True))
    x = np.where(ticks < 100, .62, np.where(ticks < 134, .77, .88))
    queries = np.stack([np.column_stack((x, np.full(n, y))) for y in (.21, -.21)], axis=1)
    _, _, keep = _grid(geometry())
    poses = np.zeros((n, 2, 7))
    poses[:, :, 3] = 1.
    arrays(tmp_path / "clouds.npz", dict(cloud_id=np.arange(n), seed=seed, layout=layout, arm=arm,
           capture_tick=ticks, delivery_tick=ticks+2, range_m=np.full((n, 520), -1.),
           self_hit=np.zeros((n, 520), dtype=bool), bracket_hit=np.zeros((n, 520), dtype=bool),
           points=np.full((n, 520, 3), np.nan), angle_keep=keep, query_capture_xy=queries,
           world_query_xy=queries, reference_poses=poses, capture_body_to_level=np.tile(np.eye(3), (n, 1, 1)),
           mount_world_xyz=np.array([mount_positions(str(a)) for a in arm])))
    score(tmp_path)
    score(tmp_path, verify=True)
    report = read(tmp_path / "optical.json")
    assert not report["passed"] and len(report["masked_exit_cases"]) == 6
    assert report["per_seed"]["89201"]["masked"]["surface"]["all"]["forward"]["unknown"] == 36
