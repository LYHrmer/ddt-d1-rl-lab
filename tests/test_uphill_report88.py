"""Recompute the D1 uphill verdict from public 50 Hz measurements, without private assets."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def test_public_endpoints_reproduce_all_six_cases_and_unrounded_verdict():
    plan_file = ROOT / "docs/experiments88/uphill_wheel_probe.json"
    plan = json.loads(plan_file.read_text())
    report = json.loads((ROOT / "reports/uphill_diagnosis88.json").read_text())
    path = ROOT / "reports/uphill88_measurements.csv"
    assert hashlib.sha256(plan_file.read_bytes()).hexdigest() == report["protocol_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report["measurements_sha256"]
    with path.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    cases = {(c["arm"], c["seed"]): c for c in report["cases"]}
    expected = {(arm, seed) for arm in plan["arm_delta_rad_s"] for seed in plan["reset_seeds"]}
    assert set(cases) == expected and len(cases) == len(report["cases"]) == plan["max_episodes"]
    assert {(r["arm"], int(r["seed"])) for r in all_rows} == expected
    assert len(all_rows) == report["policy_transitions"] <= plan["max_policy_transitions"]
    start, stop = plan["primary"]["interval_end_ticks_exclusive_inclusive"]
    limit = plan["independent_course_mark_m_past_toe"]
    pair_passes = []
    for seed in plan["reset_seeds"]:
        speeds, clean = {}, {}
        for arm in plan["arm_delta_rad_s"]:
            case = cases[arm, seed]
            rows = [r for r in all_rows if int(r["seed"]) == seed and r["arm"] == arm]
            assert [int(r["end_tick"]) for r in rows] == list(range(1, case["ticks"] + 1))
            scoring = [float(r["body_vx_m_s"]) for r in rows if start < int(r["end_tick"]) <= stop]
            speeds[arm] = mean(scoring) if len(scoring) == stop - start else None
            if speeds[arm] is None:
                assert case["score_vx_m_s"] is None
            else:
                assert math.isclose(speeds[arm], case["score_vx_m_s"], rel_tol=0, abs_tol=1e-14)
            positions = [float(r["loaded_front_past_toe_m"]) for r in rows if r["loaded_front_past_toe_m"]]
            assert max(positions, default=None) == case["max_loaded_front_contact_past_toe_m"]
            assert (bool(positions) and max(positions) >= limit) == case["reached_mark"]
            clean[arm] = len(rows) == stop and all(r["fallen"] == r["illegal_contact"] == "0" for r in rows)
            assert clean[arm] == case["full_clean"]
            assert sum(int(r["wheel_saturation_count"]) > 0 for r in rows) == case["wheel_saturated_ticks"]
            assert sum(int(r["wheel_offsets_clipped"]) for r in rows) == case["wheel_offsets_clipped"]
        delta = None if None in speeds.values() else speeds["wheel_plus"] - speeds["original"]
        pair = next(p for p in report["pairs"] if p["seed"] == seed)
        if delta is None:
            assert pair["delta_vx_m_s"] is None
        else:
            assert math.isclose(pair["delta_vx_m_s"], delta, rel_tol=0, abs_tol=1e-14)
        passed = all(clean.values()) and delta is not None and delta >= plan["primary"]["per_pair_min_delta_mean_body_vx_m_s"]
        assert passed == pair["passes"]
        pair_passes.append(passed)
    assert all(pair_passes) == report["primary_pass"]
