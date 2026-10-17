#!/usr/bin/env python3
"""Compare six complete, identity-matched pilot cases; no model is loaded.

This is a preregistered exploratory comparison of one training seed. All
original metrics, qualification flags and failure penalties remain visible.
An incomplete horizon never enters either score or an improvement delta.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools16 import evaluate_comparison16 as evaluation

FORMAT = "d1-comparison16-comparison-v1"


def check_backend(backend):
    required = {"kind": "sb3_cpu_deterministic", "algorithm": "d1_rl.velocity_policy15.AuxiliaryVelocityPPO",
                "device": "cpu", "cpu_threads": 1, "target_for_control": False, "sb3": "2.7.0"}
    evaluation.require(isinstance(backend, dict) and all(backend.get(key) == value for key, value in required.items())
                       and all(isinstance(backend.get(key), str) and backend[key] for key in evaluation.VERSION_KEYS),
                       "unsupported evaluation backend")


def trace_metrics(path, item, helpers):
    """Verify the trace bytes and recompute the unchanged historical metrics."""
    return evaluation.verify_episode_trace(path, item, helpers)


def read_evaluation(release_path, release, arm, path, helpers):
    path = Path(path).resolve()
    evaluation.require(path == evaluation.output_root(release) / f"eval_{arm}" / "report.json",
                       "evaluation path is not the registered arm")
    report = evaluation.read_json(path)
    evaluation.require(report.get("format") == evaluation.RESULT_FORMAT, "unknown evaluation schema")
    identity = evaluation.training_identity(release_path, release, arm,
                                            evaluation.output_root(release) / f"train_{arm}" / "policy.zip")
    common = {"release_sha256": evaluation.sha256(release_path), "arm": arm,
              "seed": release["seed"], "envs": release["envs"], "aux_coef": release["arms"][arm]["aux_coef"], **identity}
    evaluation.require(all(report.get(key) == value for key, value in common.items()), "evaluation/training identity mismatch")
    check_backend(report.get("backend"))
    evaluation.require({key: report["backend"][key] for key in evaluation.VERSION_KEYS} == identity["training_versions"],
                       "evaluation runtime differs from training")
    protocol_path = path.parent / "protocol.json"
    protocol = evaluation.read_json(protocol_path)
    evaluation.require(report.get("protocol_sha256") == evaluation.sha256(protocol_path), "evaluation protocol SHA mismatch")
    evaluation.require(protocol == evaluation.evaluation_protocol(release_path, release, arm, identity, report["backend"]),
                       "evaluation protocol/budget/source mismatch")
    evaluation.require(report.get("inputs_unchanged") is True and report.get("cleanup_complete") is True
                       and report.get("error") is None, "evaluation did not finish cleanly")
    elapsed = report.get("elapsed_s")
    evaluation.require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                       and math.isfinite(elapsed) and 0 <= elapsed <= protocol["max_wall_s"], "evaluation wall budget mismatch")
    cases = report.get("cases")
    evaluation.require(isinstance(cases, list) and len(cases) == 3
                       and [item.get("sequence") for item in cases] == list(evaluation.TASKS), "missing/duplicate/reordered pilot cases")
    recomputed = [trace_metrics(path, item, helpers) for item in cases]
    total = {key: sum(item["actual_counts"][key] for item in cases) for key in evaluation.COUNTS}
    evaluation.require(evaluation.counts_match(report.get("actual_counts"), total), "summary counters differ")
    passes = sum(item["nominal_zero_delay_pass"] is True for item in recomputed)
    evaluation.require(type(report.get("zero_delay_nominal_pass_count")) is int
                       and report["zero_delay_nominal_pass_count"] == passes, "qualification count differs from traces")
    full = all(item["policy_transitions"] == 600 and item["done_reason"] == "HORIZON_12S" for item in recomputed)
    evaluation.require(report.get("full_horizons_complete") is full, "full-horizon flag differs from traces")
    evaluation.require(report.get("status") == "COMPLETE" and full and evaluation.counts_match(total, evaluation.COUNTS),
                       "all three complete 12s horizons are required")
    train_protocol = evaluation.read_json(Path(identity["checkpoint"]).parent / "protocol.json")
    return report, protocol, train_protocol


def scores(cases):
    """Only call with the already verified three complete pilot cases."""
    values, parking = [], []
    for case in cases:
        if case["sequence"] == "turn_stop":
            value, threshold = case["stable_wz_mae_rad_s"], .08
        else:
            value, threshold = case["stable_vx_mae_m_s"], .06
        components = [case["stop_last_second_horizontal_speed_m_s"] / .02,
                      case["stop_last_second_abs_wz_rad_s"] / .03, case["stop_displacement_m"] / .05]
        evaluation.require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                               and math.isfinite(x) and x >= 0 for x in [value, *components]), "invalid score metric")
        values.append(value / threshold)
        parking.append(sum(components) / 3)
    return {"tracking": sum(values) / 3, "parking": sum(parking) / 3}


def compare_results(release_path, control, velocity, output):
    output = Path(output).resolve()
    evaluation.require(not output.exists(), "refusing to overwrite comparison")
    release_path = Path(release_path).resolve()
    release = evaluation.load_inputs(release_path)
    evaluation.validate_launch(release, release_path, "compare", output)
    report = {"format": FORMAT, "status": "INCOMPLETE", "verdict": "INCOMPLETE",
              "release_sha256": evaluation.sha256(release_path), "complete_cases_required": 6,
              "scores": None, "relative_reductions": None, "evaluations": {}, "inputs_sha256": {},
              "reasons": [], "scope": "one training seed101; three zero-delay reset7001 cases per arm; exploratory only",
              "statistical_significance_claimed": False, "generalization_claimed": False}
    validated = {}
    try:
        helpers = evaluation.frozen_helpers(release)
        for arm, path in (("control", control), ("velocity", velocity)):
            try:
                report["inputs_sha256"][arm] = evaluation.sha256(path)
                raw = evaluation.read_json(path)
                report["evaluations"][arm] = raw
                validated[arm] = read_evaluation(release_path, release, arm, path, helpers)
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                report["reasons"].append(f"{arm}: {type(exc).__name__}: {exc}")
        if report["reasons"]:
            return report
        c, v = validated["control"][0], validated["velocity"][0]
        evaluation.require(c["backend"] == v["backend"], "paired backends differ")
        c_hyper, v_hyper = (dict(validated[arm][2]["hyperparameters"]) for arm in ("control", "velocity"))
        c_hyper.pop("aux_coef")
        v_hyper.pop("aux_coef")
        evaluation.require(c_hyper == v_hyper, "paired training hyperparameters differ beyond auxiliary coefficient")
        score = {arm: scores(validated[arm][0]["cases"]) for arm in ("control", "velocity")}
        reductions = {key: ((score["control"][key] - score["velocity"][key]) / score["control"][key]
                            if score["control"][key] > 0 else None) for key in ("tracking", "parking")}
        required_reduction = release["comparison"]["required_relative_reduction"]
        improved = {key: value is not None and value >= required_reduction for key, value in reductions.items()}
        healthy = all(case["failure"] is False and case["threshold_flags"]["no_fall_or_illegal_support"] is True
                      for item in (c, v) for case in item["cases"])
        qualification_retained = v["zero_delay_nominal_pass_count"] >= c["zero_delay_nominal_pass_count"]
        both = all(improved.values()) and healthy and qualification_retained
        report.update(status="COMPLETE", verdict=("BOTH_IMPROVED_EXPLORATORY" if both else
                      "TRADEOFF" if any(improved.values()) else "NO_DEMONSTRATED_BENEFIT"),
                      scores=score, relative_reductions=reductions,
                      required_relative_reduction=required_reduction,
                      gates={"all_six_full_horizons": True, "no_fall_or_illegal_support": healthy,
                             "velocity_qualification_count_not_lower": qualification_retained,
                             "tracking_relative_reduction_at_least_5pct": improved["tracking"],
                             "parking_relative_reduction_at_least_5pct": improved["parking"]},
                      score_deltas={key: score["velocity"][key] - score["control"][key]
                                    for key in ("tracking", "parking")},
                      qualification_counts={"control": c["zero_delay_nominal_pass_count"],
                                            "velocity": v["zero_delay_nominal_pass_count"]},
                      tradeoff_definition="at least one score improves by 5%, but the joint score/safety/qualification gate fails")
        evaluation.contract_api().validate_sources(release)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        report.update(status="INCOMPLETE", verdict="INCOMPLETE", scores=None, relative_reductions=None)
        report["reasons"].append(f"{type(exc).__name__}: {exc}")
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        evaluation.write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--velocity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_results(args.release, args.control, args.velocity, args.output)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"status": "INCOMPLETE", "verdict": "INCOMPLETE", "error": f"{type(exc).__name__}: {exc}"}))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
