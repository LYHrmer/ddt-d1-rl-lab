import gzip
import hashlib
import json
from statistics import mean
from math import fsum

import numpy as np
import pytest

from tools91.launch91 import PLAN, PLAN_SHA, ROOT
from tools91.score91 import curves, grade, scalar


def test_scoring_frames_and_signed_displacement():
    q = np.zeros((121, 23))
    q[:, 3] = q[:, 6] = 2 ** -0.5
    v = np.zeros((121, 22))
    v[:, :2] = [10., .15]
    v[:21, 1] = -50.
    xs, vxs = curves(q, v)
    assert scalar("move", np.arange(180, 301), xs, vxs) == pytest.approx(.15)
    with pytest.raises(ValueError):
        scalar("move", np.arange(181, 301), xs[1:], vxs[1:])
    assert scalar("stand", np.arange(51), np.linspace(1., .875, 51), vxs[:51]) == .125
    assert scalar("stand", np.arange(51), np.linspace(1., 2., 51), vxs[:51]) == 0.


def test_signed_gates_require_both_arms_and_strict_null_advantage():
    rules = json.loads(PLAN.read_bytes())["tasks"]
    assert grade(rules["stand"], .172, .165, .172, .165)["passed"]
    assert not grade(rules["stand"], .165, .172, .172, .165)["checks"]["positive"]
    result = grade(rules["move"], .201, .233, .1, .132)
    assert result["checks"]["effect_tolerance"] and not result["checks"]["both_absolute_arms"] and not result["passed"]
    assert not grade(rules["move"], .1, .1, .1, .101)["checks"]["beats_zero_effect"]
    partial = grade(rules["move"], None, .13, .1, .132)
    assert partial["predicted_effect"] is None and partial["effect_abs_error"] is None and not partial["passed"]


def test_public_prediction_curves_reproduce_all_gates_and_failures():
    report = json.loads((ROOT / "reports/terrain_world91.json").read_bytes())
    raw = (ROOT / "reports/terrain_world91_curves.jsonl.gz").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["curves_sha256"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == report["protocol_sha256"] == PLAN_SHA
    plan = json.loads(PLAN.read_bytes())
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    assert len(rows) == report["curve_count"] == 84
    values, total_steps = {}, 0
    for row in rows:
        key = (row["family"], row["model_seed"], row["task"], row["reset_seed"], row["arm"])
        assert key not in values
        ticks, x, vx = (row[k] for k in ("ticks", "world_x_m", "body_vx_m_s"))
        start = 180 if row["task"] == "move" else 0
        span = plan["tasks"][row["task"]]["horizon"]
        assert len(x) == len(ticks) == len(vx) and ticks == list(range(start, start + len(x)))
        assert np.isfinite(x).all() and np.isfinite(vx).all()
        value = None
        if row["status"] == "COMPLETE":
            assert ticks[-1] == start + span
            value = mean(vx[21:]) if row["task"] == "move" else max(0., x[0] - x[-1])
        values[key] = value
    for family, results in report["families"].items():
        models = results["models"]
        seeds = plan["old_models" if family == "archived" else "fit"]["seeds"]
        assert [m["seed"] for m in models] == seeds
        for m in models:
            passed = []
            for pair in m["pairs"]:
                task, reset = pair["task"], pair["reset_seed"]
                rule = plan["tasks"][task]
                arms = ("original", rule["intervention"])
                predictions = [values[(family, m["seed"], task, reset, arm)] for arm in arms]
                truth = [values[("physics", None, task, reset, arm)] for arm in arms]
                for a, b in zip(pair["prediction"], predictions, strict=True):
                    assert a is None if b is None else a == pytest.approx(b, abs=1e-12)
                assert truth == pytest.approx(pair["truth"], abs=1e-12)
                qualifies = False
                if None not in predictions:
                    true_delta = (truth[1] - truth[0]) * (1 if task == "move" else -1)
                    pred_delta = (predictions[1] - predictions[0]) * (1 if task == "move" else -1)
                    error = abs(pred_delta - true_delta)
                    qualifies = (pred_delta > 0 and error < abs(true_delta) and error <= rule["effect_tolerance"]
                                 and max(abs(a-b) for a,b in zip(predictions, truth)) <= rule["per_arm_tolerance"])
                assert qualifies == pair["passed"]
                passed.append(qualifies)
                total_steps += sum(arm["attempted_steps"] for arm in pair["arms"])
            assert len(passed) == 6 and sum(passed) == m["pairs_passed"] and all(passed) == m["passed"]
        assert results["models_passed"] == sum(m["passed"] for m in models)
        assert results["passed"] == all(m["passed"] for m in models)
    assert total_steps == report["imagined_steps"] <= 6120
    assert report["physical_pairs"] == 6 and report["model_updates"] == 900
    assert report["new_physics_steps"] == report["policy_updates"] == 0


def test_public_world_speed_integral_matches_displacements_and_paired_effects():
    report_path = ROOT / "reports/terrain_world91.json"
    audit = json.loads((ROOT / "reports/terrain_world91_kinematics.json").read_bytes())
    assert audit["report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    raw = (ROOT / "reports/terrain_world91_curves.jsonl.gz").read_bytes()
    assert audit["curves_sha256"] == hashlib.sha256(raw).hexdigest()
    assert audit["policy_hz"] == 50 and audit["endpoint_tick"] == list(range(51))
    curves_by_arm = {}
    for line in gzip.decompress(raw).splitlines():
        r = json.loads(line)
        if r["task"] == "stand":
            curves_by_arm[r["family"], r["model_seed"], r["reset_seed"], r["arm"]] = r
    quantities = {}
    for row in audit["arms"]:
        key = row["family"], row["model_seed"], row["reset_seed"], row["arm"]
        xs = curves_by_arm[key]["world_x_m"]
        vs = row["world_vx_m_s"]
        assert key not in quantities and len(vs) == len(xs) == 51
        dx = xs[-1] - xs[0]
        integral = (.5 * (vs[0] + vs[-1]) + fsum(vs[1:-1])) / 50
        assert dx == pytest.approx(row["world_dx_m"], abs=1e-12)
        assert integral == pytest.approx(row["integrated_world_vx_m"], abs=1e-12)
        assert dx - integral == pytest.approx(row["dx_minus_integral_m"], abs=1e-12)
        quantities[key] = (max(0., -dx), max(0., -integral))
    assert quantities.keys() == curves_by_arm.keys() and len(quantities) == 42
    assert len(audit["pairs"]) == 21
    seen = set()
    for pair in audit["pairs"]:
        group = pair["family"], pair["model_seed"], pair["reset_seed"]
        assert group not in seen
        seen.add(group)
        a, b = quantities[group + ("original",)], quantities[group + ("common",)]
        x, v = (original - common for original, common in zip(a, b))
        assert [x, v, x-v] == pytest.approx([pair["effect_x_m"], pair["effect_integrated_vx_m"], pair["difference_m"]], abs=1e-12)
