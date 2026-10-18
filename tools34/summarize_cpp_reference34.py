"""Compact, read-only summary of a completed C++34 execution.

Re-hashes every saved trace/sidecar named by the report and compares the 72 episodes with the
published Python33 report. Runs no model, no physics and no C++.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

STAGES = ("formal", "holdout")


def file_sha(root: Path, name: str) -> str:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {name} under {root}, found {len(matches)}")
    return hashlib.sha256(matches[0].read_bytes()).hexdigest()


def stage_summary(cases: list[dict]) -> dict:
    cpp = [c["cpp_reference34"] for c in cases]
    return {
        "cases": len(cases),
        "full_600_ticks": all(c["policy_transitions"] == 600 for c in cases),
        "failures": sum(bool(c["failure"]) for c in cases),
        "reference_thresholds_met": all(c["reference_thresholds_met"] for c in cases),
        "mean_J": statistics.fmean(c["J"] for c in cases),
        "max_last_second_horizontal_speed_m_s": max(
            c["stop_last_second_horizontal_speed_m_s"] for c in cases
        ),
        "max_stop_displacement_m": max(c["stop_displacement_m"] for c in cases),
        "max_peak_abs_roll_or_pitch_rad": max(c["peak_abs_roll_or_pitch_rad"] for c in cases),
        "cpp_calls": sum(c["cpp_calls"] for c in cpp),
        "python_oracle_calls": sum(c["python_oracle_calls"] for c in cpp),
        "max_action_error": max(c["max_action_error"] for c in cpp),
        "all_effective_inputs_bitwise_equal": all(
            c["full_effective_inputs_bitwise_equal"] for c in cpp
        ),
    }


def summarize(run: Path, python33: Path) -> dict:
    report = json.loads((run / "report.json").read_text())
    ref = json.loads(python33.read_text())
    rows = [json.loads(x) for x in (run / "resources.jsonl").read_text().splitlines() if x]
    owned = [r["owned"] for r in rows if r.get("owned")]
    checked = bad = 0
    trace_equal = j_diff = 0
    for stage in STAGES:
        for new, old in zip(report["stages"][stage]["cases"], ref["stages"][stage]["cases"], strict=True):
            key = ("sequence", "delay_ticks", "reset_seed")
            if tuple(new[k] for k in key) != tuple(old[k] for k in key):
                raise SystemExit("case order differs from Python33")
            for name, sha in (
                (new["trace_file"], new["trace_sha256"]),
                (new["cpp_reference34"]["sidecar"], new["cpp_reference34"]["sha256"]),
                (new["reference_shape33"]["sidecar"], new["reference_shape33"]["sha256"]),
            ):
                checked += 1
                bad += file_sha(run / stage, name) != sha
            trace_equal += new["trace_sha256"] == old["trace_sha256"]
            j_diff = max(j_diff, abs(new["J"] - old["J"]))
    return {
        "schema": "cpp34-small-saved-summary-v1",
        "run": str(run),
        "status": report["status"],
        "error": report["error"],
        "source_commit": report["source"]["git_commit"],
        "plan_sha256": report["plan_sha256"],
        "binary_sha256": report["cpp_build"]["binary_sha256"],
        "report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
        "cpp_original_and_holdout_qualified": report["cpp_original_and_holdout_qualified"],
        "new_cpp_closedloop_proven": report["new_cpp_closedloop_proven"],
        "automatic_model_replacement": report["automatic_model_replacement"],
        "inputs_unchanged": report["inputs_unchanged"],
        "cleanup_complete": report["cleanup_complete"],
        "training_transitions": report["training_transitions"],
        "actual_counts": report["actual_counts"],
        "cpp_request_attempts": report["cpp_request_attempts"],
        "cpp_request_returns": report["cpp_request_returns"],
        "elapsed_s": report["elapsed_s"],
        "stages": {s: stage_summary(report["stages"][s]["cases"]) for s in STAGES},
        "saved_files_rehashed": {"checked": checked, "mismatched": bad},
        "versus_python33": {
            "episodes_compared": 72,
            "trace_sha256_identical": trace_equal,
            "max_abs_J_difference": j_diff,
            "python33_report": str(python33),
        },
        "resources": {
            "samples": len(rows),
            "min_effective_available_bytes": min(r["effective_available_bytes"] for r in rows),
            "max_sampled_owned_current_bytes": max(o["current_bytes"] for o in owned),
            "final_cgroup_events": owned[-1]["events"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("python33_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run, args.python33_report)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
