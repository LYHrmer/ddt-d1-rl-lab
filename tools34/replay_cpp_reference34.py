"""Zero-physics replay of a completed C++34 execution through the real C++34 actor server.

Feeds the saved raw 627-dim actor inputs of a COMPLETE, qualified execution to a fresh C++34
process per episode and requires the returned 16 actions to equal the saved issued actions
bit for bit. It builds no robot model and takes no physics step; it is the repeatable way to
re-check the 33 reference-shaping deployment path on any machine that holds the evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools19 import cpp_actor19 as transport  # noqa: E402
from tools34.cpp_reference34 import SOURCE, CppReferenceActor  # noqa: E402

require, sha, atomic_json = transport.require, transport.sha, transport.atomic_json
DEFAULT_EXECUTION = ROOT / "evidence/cpp_reference34/execution_01"
DEFAULT_POLICY = ROOT / "evidence/low_memory_delay25/execution_01/random/export/policy.onnx"
STAGES = ("formal", "holdout")


def load_execution(execution: Path) -> dict:
    report = json.loads((execution / "report.json").read_text())
    require(
        report["status"] == "COMPLETE"
        and report["error"] is None
        and report["cpp_original_and_holdout_qualified"] is True
        and report["automatic_model_replacement"] is False,
        "replay requires a COMPLETE, qualified C++34 execution",
    )
    return report


def select(report: dict, stages: list[str], names: list[str] | None, limit: int | None):
    chosen = []
    for stage in stages:
        for case in report["stages"][stage]["cases"]:
            stem = Path(case["trace_file"]).stem
            if names is None or stem in names:
                chosen.append((stage, stem, case))
    require(chosen, "no episode selected")
    return chosen[:limit] if limit else chosen


def replay_case(execution: Path, output: Path, report: dict, stage: str, stem: str, case: dict, model: Path):
    sidecar = execution / stage / case["cpp_reference34"]["sidecar"]
    require(sha(sidecar) == case["cpp_reference34"]["sha256"], f"saved sidecar changed: {sidecar.name}")
    with np.load(sidecar, allow_pickle=False) as saved:
        raw, issued = saved["raw_actor627"], saved["cpp_issued16"]
    require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[1] == 627, "saved input shape differs")
    require(issued.shape == (raw.shape[0], 16), "saved action shape differs")
    transport = case["cpp_reference34"]["transport"]["identities"]
    require(
        transport["binary_sha256"] == report["cpp_build"]["binary_sha256"]
        and transport["model_sha256"] == sha(model),
        "replay model or binary differs from the qualified execution",
    )
    actor = CppReferenceActor(
        report["cpp_build"]["binary"],
        model,
        output=output / stage / f"{stem}_replay",
        expected_binary_sha256=transport["binary_sha256"],
        expected_model_sha256=transport["model_sha256"],
        expected_source_sha256=transport["source_sha256"],
        source=SOURCE,
        max_requests=raw.shape[0],
    )
    actor.start()
    bad, worst = 0, 0.0
    for row, expected in zip(raw, issued, strict=True):
        action = actor.predict(row)
        bad += not np.array_equal(action, expected)
        worst = max(worst, float(np.max(np.abs(action.astype(float) - expected.astype(float)))))
    closed = actor.close()
    require(closed["status"] == "CLOSED" and closed["cleanup_complete"], "C++ replay process did not close cleanly")
    return {
        "stage": stage,
        "episode": stem,
        "requests": int(raw.shape[0]),
        "actions_bitwise_equal_to_saved": bad == 0,
        "mismatched_ticks": bad,
        "max_abs_action_difference": worst,
        "requests_completed": closed["requests_completed"],
        "inference_returns": closed["inference_returns"],
        "cleanup_complete": closed["cleanup_complete"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True, help="must not exist")
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--episode", action="append", help="trace stem, e.g. forward_stop_delay0_seed1001")
    parser.add_argument("--limit", type=int, help="replay at most this many episodes")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output directory exists; refusing overwrite")
    execution = args.execution.resolve()
    report = load_execution(execution)
    chosen = select(report, list(STAGES) if args.stage == "all" else [args.stage], args.episode, args.limit)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    results, error = [], None
    try:
        for stage, stem, case in chosen:
            results.append(replay_case(execution, args.output, report, stage, stem, case, args.policy.resolve()))
    except BaseException as exc:  # keep the partial evidence and fail the run
        error = f"{type(exc).__name__}: {exc}"
    complete = error is None and len(results) == len(chosen)
    summary = {
        "format": "d1-cpp-reference34-replay-v1",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "error": error,
        "execution": str(execution),
        "execution_report_sha256": sha(execution / "report.json"),
        "episodes_selected": len(chosen),
        "episodes_replayed": len(results),
        "all_actions_bitwise_equal_to_saved": complete and all(r["actions_bitwise_equal_to_saved"] for r in results),
        "requests": sum(r["requests"] for r in results),
        "physics_steps": 0,
        "robot_models_constructed": 0,
        "elapsed_s": time.monotonic() - started,
        "results": results,
    }
    atomic_json(args.output / "report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return int(not (complete and summary["all_actions_bitwise_equal_to_saved"]))


if __name__ == "__main__":
    raise SystemExit(main())
