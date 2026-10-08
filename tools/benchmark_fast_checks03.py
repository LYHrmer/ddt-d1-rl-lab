#!/usr/bin/env python3
"""Bounded 640-tick D1FastChecksEnv timing against the frozen base profile.

This is an engineering throughput measurement, not training or policy merit.
Run only after the paired validator and its separately allocated budget pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training/configs/pilot03b.json"
FIXTURE = ROOT / "training/configs/diagnostic03b_static_support.json"
FAST = ROOT / "training/d1_rl/fast_checks.py"
BASELINE = ROOT / "evidence/career_contract_03/engineering_timing03b_01"
TOTAL_TICKS = 640
COLD_TICKS = 64
HOT_TICKS = TOTAL_TICKS - COLD_TICKS
NATIVE_CAP = TOTAL_TICKS * 10
WALL_CAP_S = 60.0
TARGET_RATIO = 1.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def checked_baseline(directory: Path) -> tuple[dict, dict]:
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (protocol["mode"] != "timing" or protocol["seed"] != 7001 or
            protocol["max_transitions"] != TOTAL_TICKS or
            protocol["max_native_steps"] != NATIVE_CAP or
            protocol["cold_policy_ticks"] != COLD_TICKS or
            protocol["hot_policy_ticks"] != HOT_TICKS or
            result["status"] != "COMPLETED" or
            result["inputs_unchanged"] is not True or
            result["actual_counts"]["native_steps"] != NATIVE_CAP):
        raise ValueError("Baseline is not the frozen completed 640-tick timing run")
    expected = (CONFIG, FIXTURE, ROOT / "training/d1_rl/env.py",
                ROOT / "training/d1_rl/interface.py")
    for path in expected:
        if protocol["inputs_sha256"].get(str(path)) != sha256(path):
            raise ValueError(f"Baseline source/config differs from current fast-check input: {path}")
    speed = float(result["hot_transitions_per_s"])
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Invalid baseline hot throughput")
    return protocol, result


def run(output: Path, baseline_dir: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    baseline_protocol, baseline_result = checked_baseline(baseline_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), CONFIG, FIXTURE, FAST,
               ROOT / "training/d1_rl/env.py", ROOT / "training/d1_rl/interface.py",
               baseline_dir / "protocol.json", baseline_dir / "result.json"]
    identities = {str(path): sha256(path) for path in sources}
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    action = fixture["actions"][0]
    if action != fixture["static_derivation"]["constant_issued_action"] or len(action) != 16:
        raise ValueError("First fixed-support action differs from frozen static derivation")
    write_json(output / "protocol.json", {
        "format": "d1-fast-checks-timing-03-v1",
        "purpose": "engineering throughput only; no learned policy or control qualification",
        "seed": 7001, "delay_ticks": 0, "command_vx_wz": [0.0, 0.0],
        "action_source": str(FIXTURE), "constant_action_only": True,
        "max_policy_transitions": TOTAL_TICKS,
        "max_native_steps": NATIVE_CAP,
        "max_wall_seconds": WALL_CAP_S,
        "cold_policy_ticks": COLD_TICKS,
        "hot_policy_ticks": HOT_TICKS,
        "target_speedup_ratio": TARGET_RATIO,
        "baseline_hot_transitions_per_s": baseline_result["hot_transitions_per_s"],
        "baseline_protocol_sha256": identities[str(baseline_dir / "protocol.json")],
        "baseline_result_sha256": identities[str(baseline_dir / "result.json")],
        "inputs_sha256": identities,
    })
    sys.path.insert(0, str(ROOT / "training"))
    from d1_rl.fast_checks import D1FastChecksEnv

    env = None
    started = time.monotonic()
    status = "ERROR"
    error = None
    setup_s = cold_s = hot_s = None
    hot_start = None
    completed_ticks = 0
    try:
        env = D1FastChecksEnv(config=CONFIG, seed=7001,
                              eval_delay_ticks=0, max_transitions=TOTAL_TICKS)
        env.reset()
        env.set_command(0.0, 0.0)
        setup_s = time.monotonic() - started
        loop_start = time.monotonic()
        write_json(output / "live_counts.json", {"completed_policy_ticks": 0,
                                                   "actual_counts": env.counters()})
        for tick in range(TOTAL_TICKS):
            if time.monotonic() - started > WALL_CAP_S:
                raise RuntimeError("60 s engineering timing wall cap reached")
            _, _, terminated, truncated, _ = env.step(action)
            completed_ticks = tick + 1
            if completed_ticks == COLD_TICKS:
                hot_start = time.monotonic()
                cold_s = hot_start - loop_start
            if completed_ticks % 16 == 0:
                write_json(output / "live_counts.json", {"completed_policy_ticks": completed_ticks,
                                                           "actual_counts": env.counters()})
            if terminated or truncated:
                status = "EARLY_TERMINATION"
                break
        else:
            hot_s = time.monotonic() - hot_start
            status = "COMPLETED"
        counts = env.counters()
        if (status == "COMPLETED" and
                (counts["native_steps"] != NATIVE_CAP or counts["pd_updates"] != 3200 or
                 counts["policy_transitions"] != TOTAL_TICKS)):
            status = "INVALID_COUNT"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        actual_counts = env.counters() if env is not None else None
        write_json(output / "live_counts.json", {"completed_policy_ticks": completed_ticks,
                                                   "actual_counts": actual_counts,
                                                   "status": status})
        unchanged = all(sha256(Path(path)) == expected for path, expected in identities.items())
        if not unchanged:
            status = "INPUT_CHANGED"
        speed = HOT_TICKS / hot_s if hot_s is not None and hot_s > 0 else None
        baseline_speed = float(baseline_result["hot_transitions_per_s"])
        ratio = speed / baseline_speed if speed is not None else None
        target_met = status == "COMPLETED" and ratio is not None and ratio >= TARGET_RATIO
        result = {"status": status, "error": error,
                  "completed_policy_ticks": completed_ticks,
                  "actual_counts": actual_counts,
                  "actual_native_steps": actual_counts["native_steps"] if actual_counts else 0,
                  "setup_seconds": setup_s, "cold_64_seconds": cold_s,
                  "hot_576_seconds": hot_s,
                  "hot_transitions_per_s": speed,
                  "baseline_hot_transitions_per_s": baseline_speed,
                  "speedup_ratio": ratio,
                  "target_ratio": TARGET_RATIO,
                  "target_met": target_met,
                  "inputs_unchanged": unchanged,
                  "elapsed_total_s": time.monotonic() - started,
                  "not_a_policy_qualification": True}
        write_json(output / "result.json", result)
        if env is not None:
            env.close()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["target_met"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new timing output directory")
    parser.add_argument("--baseline", type=Path, default=BASELINE,
                        help="frozen original D1Env 640-tick timing run")
    args = parser.parse_args()
    return run(args.output.resolve(), args.baseline.resolve())


if __name__ == "__main__":
    sys.exit(main())
