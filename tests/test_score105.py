"""Synthetic scans and labels exercise the paired readout gate independently of the validation packets."""
from itertools import product
import json

import numpy as np
import pytest

from tools105 import score105 as scoring


@pytest.fixture
def rows():
    records = []
    for split, seeds in scoring.SEEDS.items():
        row_id = 0
        streams = product(seeds, scoring.EPISODES[split], scoring.LAYOUTS[split], scoring.VISIBILITIES)
        for series, (seed, episode, layout, view) in enumerate(streams):
            side = layout.rsplit("_", 1)[1]
            for j, tick in enumerate(scoring.TICKS):
                target = (30. if side == "left" else -30.) if 2 <= j < 4 else 0.
                records.append(dict(split=split, row_id=row_id, source_seed=seed, episode=episode,
                                    layout=layout, side=side, visibility=view, series_id=series,
                                    cloud_id=(series // 2) * len(scoring.TICKS) + j,
                                    capture_tick=tick, delivery_tick=tick + 2, target_mm=target,
                                    baseline_mm=0. if view == "masked" and j == 2 else target,
                                    candidate_mm=target))
                row_id += 1
    return {key: np.array([r[key] for r in records]) for key in records[0]}


def test_votes_require_both_validation_seeds_and_exclude_design(rows):
    rows["candidate_mm"][rows["source_seed"] == 89037] = np.nan
    good = scoring.score(rows)
    assert good["gate"] == {"validation_seeds": [89011, 89023], "design_seeds": [89037],
                            "seeds_passed": 2, "seeds_required": 2, "passed": True}
    assert good["per_seed"]["89037"]["checks"] is None
    assert good["per_seed"]["89037"]["passed"] is None
    assert (good["counts"]["packet_pairs"], good["counts"]["physical_clouds"],
            good["counts"]["distinct_native_windows"], good["counts"]["native_episodes"]) == (1344, 672, 208, 13)
    neutral = rows["source_seed"] == 89023
    rows["candidate_mm"][neutral] = rows["baseline_mm"][neutral]
    failed = scoring.score(rows)
    assert failed["gate"]["seeds_passed"] == 1 and not failed["gate"]["passed"]
    assert not failed["per_seed"]["89023"]["checks"]["masked_switch_strict_gain"]


def test_transition_groups_own_arrival_and_first_packets(rows):
    design = scoring.score(rows)["per_seed"]["89037"]["visibility"]["masked"]
    assert {key: design[key]["packet_pairs"] for key in design} == {
        "all_arrivals": 160, "first": 10, "switch": 20, "enter": 10, "exit": 10, "stable": 130}
    assert design["all_arrivals"]["distinct_native_windows"] == 80
    assert design["enter"]["gained"] == 10 and design["exit"]["gained"] == 0
    assert design["first"]["baseline"]["passed"] == 10
    assert design["stable"]["baseline"]["raised_targets"] == 10
    assert design["switch"]["gained"] + design["switch"]["both_passed"] == 20


@pytest.mark.parametrize(("view", "tick_index", "rule"), [
    ("masked", 4, "masked_exit_no_loss"), ("masked", 6, "masked_stable_no_extra_failures"),
    ("clear", 0, "clear_all_no_extra_failures"),
])
def test_extra_failures_cannot_be_offset_by_a_switch_gain(rows, view, tick_index, rule):
    match = ((rows["source_seed"] == 89011) & (rows["episode"] == 2) & (rows["layout"] == "narrow_left")
             & (rows["visibility"] == view) & (rows["capture_tick"] == scoring.TICKS[tick_index]))
    assert match.sum() == 1
    rows["candidate_mm"][match] = np.nan
    scored = scoring.score(rows)["per_seed"]["89011"]
    assert not scored["passed"] and not scored["checks"][rule]
    assert scored["checks"]["masked_switch_strict_gain"]
    all_pairs = scored["visibility"][view]["all_arrivals"]
    assert all_pairs["candidate"]["unknown"] == 1 and all_pairs["regressed"] == 1


def test_exit_gains_cannot_buy_a_lost_entry(rows):
    seed = (rows["source_seed"] == 89011) & (rows["visibility"] == "masked")
    entering = seed & (rows["capture_tick"] == scoring.TICKS[2])
    leaving = seed & (rows["capture_tick"] == scoring.TICKS[4])
    rows["baseline_mm"][entering] = rows["target_mm"][entering]
    rows["baseline_mm"][leaving] = 30.
    rows["candidate_mm"][np.flatnonzero(entering)[0]] = 0.
    checks = scoring.score(rows)["per_seed"]["89011"]["checks"]
    assert checks["masked_switch_strict_gain"] and checks["masked_exit_no_loss"]
    assert not checks["masked_enter_no_loss"]


def test_fault_categories_signs_magnitude_missing_and_tolerance(rows):
    m = ((rows["source_seed"] == 89011) & (rows["visibility"] == "masked")
         & (rows["capture_tick"] == scoring.TICKS[2]) & (rows["side"] == "left"))
    rows["candidate_mm"][np.flatnonzero(m)[:5]] = [np.nan, -30., 10., 19., 20.]
    flat = np.flatnonzero((rows["source_seed"] == 89011) & (rows["visibility"] == "masked")
                          & (rows["capture_tick"] == scoring.TICKS[4]))
    rows["candidate_mm"][flat[:3]] = [10., 10.01, np.nan]
    views = scoring.score(rows)["per_seed"]["89011"]["visibility"]["masked"]
    raised = views["enter"]["candidate"]
    assert [raised[k] for k in ("raised_missed_side", "raised_unknown", "raised_wrong_side",
                                "raised_called_flat", "raised_correct_side_bad_magnitude")] == [3, 1, 1, 1, 1]
    assert raised["failed"] == 4
    exit_ = views["exit"]["candidate"]
    assert exit_["flat_false_positives"] == 1 and exit_["unknown"] == 1 and exit_["failed"] == 2


@pytest.mark.parametrize(("field", "value", "message"), [
    ("row_id", 1, "sequence"), ("series_id", 1, "sequence"), ("capture_tick", 26, "latency"),
    ("source_seed", 89023, "roster"), ("cloud_id", 9, "cloud identity"), ("side", "right", "signed"),
    ("target_mm", -30., "signed"), ("target_mm", 30., "physical truth"),
    ("baseline_mm", np.inf, "predictions"), ("candidate_mm", -np.inf, "predictions"),
])
def test_corrupt_row_is_rejected(rows, field, value, message):
    rows[field][0] = value
    with pytest.raises(ValueError, match=message):
        scoring.score(rows)


def test_drop_shared_packet_cannot_improve_denominator(rows):
    with pytest.raises(ValueError, match="sequence"):
        scoring.score({k: v[1:] for k, v in rows.items()})


def test_both_readouts_receive_only_identical_measured_returns_and_queries(rows):
    metadata = {k: rows[k][:2] for k in scoring.IDS}
    points, queries = np.zeros((2, 520, 3)), np.zeros((2, 2, 2))
    points[1, 0, 0] = 1.
    calls = []

    def reader(p, q):
        calls.append((float(p[0, 0]), q.shape))
        assert not p.flags.writeable and not q.flags.writeable
        return None if p[0, 0] == 0 else 3.

    result = scoring.readouts(metadata, points, queries, reader, reader)
    assert calls[:2] == calls[2:] == [(0., (2, 2)), (1., (2, 2))]
    for key in scoring.ARMS:
        np.testing.assert_equal(result[key], [np.nan, 3.])
    with pytest.raises(ValueError, match="identity"):
        scoring.readouts({**metadata, "target_mm": rows["target_mm"][:2]}, points, queries, reader, reader)
    with pytest.raises(ValueError, match="finite millimetres or None"):
        scoring.readouts(metadata, points, queries, reader, lambda p, q: float("inf"))


def test_exclusive_seal_replay_and_reference_join(tmp_path, rows, monkeypatch):
    archives = {}
    for split in scoring.SEEDS:
        path = tmp_path / f"{split}.npz"
        selected = rows["split"] == split
        scoring.arrays(path, {k: rows[k][selected] for k in (*scoring.IDS, "target_mm")})
        archives[split] = path
    monkeypatch.setattr(scoring, "PACKETS", archives)
    monkeypatch.setattr(scoring, "PACKET_SHA", {k: scoring.digest(p) for k, p in archives.items()})
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    monkeypatch.setattr(scoring, "PROTOCOL", protocol)
    (tmp_path / "design_windows.npz").write_bytes(b"design windows")
    (tmp_path / "windows.npz").write_bytes(b"validation windows")
    monkeypatch.setattr(scoring, "DESIGN_WINDOWS", tmp_path / "design_windows.npz")
    prediction = {k: rows[k].copy() for k in scoring.PREDICTION_KEYS}
    monkeypatch.setattr(scoring, "predictions", lambda windows: prediction)
    original_join = scoring.with_truth

    def join_after_seal(forecast):
        assert (tmp_path / "predictions.npz").is_file()
        assert "target_mm" not in forecast
        return original_join(forecast)

    monkeypatch.setattr(scoring, "with_truth", join_after_seal)
    result = scoring.record(tmp_path)
    assert scoring.check(tmp_path) == result
    with pytest.raises(ValueError, match="overwrite"):
        scoring.record(tmp_path)
    prediction["candidate_mm"][0] += 1.
    with pytest.raises(ValueError, match="replay candidate_mm changed"):
        scoring.check(tmp_path)
    prediction["candidate_mm"][0] -= 1.
    prediction["row_id"][0] += 1
    with pytest.raises(ValueError, match="alignment"):
        scoring.with_truth(prediction)
    prediction["row_id"][0] -= 1
    result["gate"]["passed"] = False
    (tmp_path / "report.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="votes or scores"):
        scoring.check(tmp_path)
