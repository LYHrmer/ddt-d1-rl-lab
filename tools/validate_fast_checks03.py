#!/usr/bin/env python3
"""Bounded step-for-step D1Env versus D1FastChecksEnv equivalence check.

Three fresh seeds and delays use at most 50 policy ticks per side per case:
3 cases x 2 environments x 50 ticks x 10 native steps = 3,000 native steps.
No retries or replacement cases occur after a mismatch or early termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "training/configs/pilot03b.json"
SUPPORT = ROOT / "training/configs/diagnostic03b_static_support.json"
PULSE = ROOT / "training/configs/diagnostic03b_wheel_pulse.json"
FAST = ROOT / "training/d1_rl/fast_checks.py"
TOLERANCE = 1e-5
TICKS_PER_SIDE = 50
NATIVE_PER_TICK = 10
CASES = ((7001, 0, "fixed_support"),
         (7002, 1, "wheel_pulse_first_50"),
         (7003, 2, "fixed_support"))


class Mismatch(AssertionError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json_text(value), encoding="utf-8")
    temp.replace(path)


def compare(left, right, path: str, maximum: list[float]) -> None:
    """Compare complete nested observations/info; reject nonfinite on either side."""
    if hasattr(left, "tolist"):
        left = left.tolist()
    if hasattr(right, "tolist"):
        right = right.tolist()
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            raise Mismatch(f"{path}: mapping keys differ")
        for key in sorted(left):
            compare(left[key], right[key], f"{path}.{key}", maximum)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise Mismatch(f"{path}: vector/shape differs")
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{path}[{index}]", maximum)
        return
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise Mismatch(f"{path}: boolean differs: {left!r} vs {right!r}")
        return
    if isinstance(left, numbers.Integral) and isinstance(right, numbers.Integral):
        if left != right:
            raise Mismatch(f"{path}: integer differs: {left} vs {right}")
        return
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        a, b = float(left), float(right)
        if not math.isfinite(a) or not math.isfinite(b):
            raise Mismatch(f"{path}: NaN/Inf rejected")
        error = abs(a - b)
        maximum[0] = max(maximum[0], error)
        if error > TOLERANCE:
            raise Mismatch(f"{path}: absolute error {error:.9g} > {TOLERANCE}")
        return
    if type(left) is not type(right) or left != right:
        raise Mismatch(f"{path}: values differ: {left!r} vs {right!r}")


def fixture_actions() -> dict[str, list[list[float]]]:
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    pulse = json.loads(PULSE.read_text(encoding="utf-8"))
    constant = support["static_derivation"]["constant_issued_action"]
    if len(constant) != 16 or any(not math.isfinite(float(x)) for x in constant):
        raise ValueError("Invalid fixed-support action")
    if len(support["actions"]) < TICKS_PER_SIDE or len(pulse["actions"]) < TICKS_PER_SIDE:
        raise ValueError("Diagnostic fixture has fewer than 50 actions")
    fixed = support["actions"][:TICKS_PER_SIDE]
    moving = pulse["actions"][:TICKS_PER_SIDE]
    if any(row != constant for row in fixed):
        raise ValueError("Fixed-support fixture is not constant for the first 50 ticks")
    for tick, row in enumerate(moving):
        expected = constant.copy()
        if 20 <= tick < 30:
            for wheel in (3, 7, 11, 15):
                expected[wheel] = 0.10
        if row != expected:
            raise ValueError(f"Wheel-pulse fixture changed at tick {tick}")
    return {"fixed_support": fixed, "wheel_pulse_first_50": moving}


def snapshot(base, fast, case_index: int, tick: int, phase: str,
             prior_native: int) -> dict:
    left = base.counters() if base is not None else None
    right = fast.counters() if fast is not None else None
    native = (left or {}).get("native_steps", 0) + (right or {}).get("native_steps", 0)
    if prior_native + native > len(CASES) * 2 * TICKS_PER_SIDE * NATIVE_PER_TICK:
        raise RuntimeError("Global 3,000-native-step budget exceeded")
    return {"case_index": case_index, "policy_tick": tick, "phase": phase,
            "baseline_counts": left, "fast_counts": right,
            "current_pair_native_steps": native,
            "total_native_steps_both_sides": prior_native + native}


def compare_boundary(base, fast, left_observation, right_observation, path: str,
                     maximum: list[float]) -> None:
    for label, left, right in (
        ("qpos", base.data.qpos, fast.data.qpos),
        ("qvel", base.data.qvel, fast.data.qvel),
        ("qacc_warmstart", base.data.qacc_warmstart, fast.data.qacc_warmstart),
        ("qfrc_applied", base.data.qfrc_applied, fast.data.qfrc_applied),
        ("observation", left_observation, right_observation),
        ("counters", base.counters(), fast.counters()),
    ):
        compare(left, right, f"{path}.{label}", maximum)


def run(output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), CONFIG, SUPPORT, PULSE, FAST,
               ROOT / "training/d1_rl/env.py", ROOT / "training/d1_rl/interface.py",
               ROOT / "training/configs/interface03b.json",
               ROOT / "evidence/interface_audit_03/golden_run_01/schema.json"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    scene = ROOT / config["scene"]
    sources.extend([scene, scene.parent / "robot.xml"])
    hashes = {str(path): sha256(path) for path in sources}
    actions = fixture_actions()
    write_json(output / "protocol.json", {
        "format": "d1-fast-checks-paired-validation-03-v1",
        "purpose": "equivalence only; fixed diagnostic actions are not policy outputs",
        "cases": [{"reset_seed": seed, "delay_ticks": delay, "action_sequence": sequence,
                   "policy_tick_cap_per_side": TICKS_PER_SIDE,
                   "native_step_cap_per_side": TICKS_PER_SIDE * NATIVE_PER_TICK}
                  for seed, delay, sequence in CASES],
        "global_native_step_cap_both_sides": 3000,
        "absolute_tolerance": TOLERANCE,
        "stop_at_first_exception_mismatch_or_early_done": True,
        "inputs_sha256": hashes,
    })
    # Delay heavyweight imports until after --help and protocol identity work.
    sys.path.insert(0, str(ROOT / "training"))
    from d1_rl.env import D1Env
    from d1_rl.fast_checks import D1FastChecksEnv

    base = fast = None
    started = time.monotonic()
    status = "ERROR"
    error = None
    per_case = []
    total_native = 0
    maximum = [0.0]
    case_index = -1
    tick = -1
    try:
        with (output / "checks.jsonl").open("x", encoding="utf-8") as checks:
            for case_index, (seed, delay, sequence) in enumerate(CASES):
                base = D1Env(config=CONFIG, seed=seed, rank=0,
                             eval_delay_ticks=delay, max_transitions=TICKS_PER_SIDE)
                fast = D1FastChecksEnv(config=CONFIG, seed=seed, rank=0,
                                       eval_delay_ticks=delay, max_transitions=TICKS_PER_SIDE)
                left_obs, left_reset = base.reset(seed=seed)
                right_obs, right_reset = fast.reset(seed=seed)
                base.set_command(0.0, 0.0)
                fast.set_command(0.0, 0.0)
                left_obs, right_obs = base.current_observation(), fast.current_observation()
                compare_boundary(base, fast, left_obs, right_obs, f"case{case_index}.reset", maximum)
                compare(left_reset, right_reset, f"case{case_index}.reset_info", maximum)
                write_json(output / "live_counts.json", snapshot(base, fast, case_index, 0,
                                                                  "reset", total_native))
                for tick, action in enumerate(actions[sequence], 1):
                    if time.monotonic() - started > 90:
                        raise RuntimeError("Paired validation wall cap 90 s reached")
                    left = base.step(action)
                    write_json(output / "live_counts.json", snapshot(base, fast, case_index, tick,
                                                                     "baseline_step_returned", total_native))
                    right = fast.step(action)
                    write_json(output / "live_counts.json", snapshot(base, fast, case_index, tick,
                                                                     "fast_step_returned", total_native))
                    compare_boundary(base, fast, left[0], right[0], f"case{case_index}.tick{tick}", maximum)
                    compare(left[1], right[1], f"case{case_index}.tick{tick}.reward", maximum)
                    compare(left[2], right[2], f"case{case_index}.tick{tick}.terminated", maximum)
                    compare(left[3], right[3], f"case{case_index}.tick{tick}.truncated", maximum)
                    compare(left[4], right[4], f"case{case_index}.tick{tick}.info", maximum)
                    checks.write(json.dumps({"case_index": case_index, "policy_tick": tick,
                                             "max_abs_error_so_far": maximum[0],
                                             "baseline_native_steps": base.counters()["native_steps"],
                                             "fast_native_steps": fast.counters()["native_steps"]}) + "\n")
                    checks.flush()
                    write_json(output / "live_counts.json", snapshot(base, fast, case_index, tick,
                                                                     "paired_check_passed", total_native))
                    if left[2] or left[3] or right[2] or right[3]:
                        raise RuntimeError(f"Matched early episode end at case {case_index}, tick {tick}; no refill")
                left_counts, right_counts = base.counters(), fast.counters()
                for counts in (left_counts, right_counts):
                    if (counts["native_steps"] != 500 or counts["pd_updates"] != 250 or
                            counts["policy_transitions"] != 50):
                        raise RuntimeError(f"Unexpected per-side work after 50 ticks: {counts}")
                per_case.append({"case_index": case_index, "reset_seed": seed,
                                 "delay_ticks": delay, "action_sequence": sequence,
                                 "baseline_counts": left_counts, "fast_counts": right_counts,
                                 "max_abs_error_so_far": maximum[0]})
                total_native += left_counts["native_steps"] + right_counts["native_steps"]
                base.close()
                fast.close()
                base = fast = None
            status = "EQUIVALENT" if total_native == 3000 else "INVALID_COUNT"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        active = snapshot(base, fast, case_index, tick, "final_or_failed", total_native)
        write_json(output / "live_counts.json", active)
        total_actual_native = total_native + active["current_pair_native_steps"]
        unchanged = all(sha256(Path(path)) == value for path, value in hashes.items())
        if not unchanged:
            status = "INPUT_CHANGED"
        result = {"status": status, "error": error,
                  "completed_cases": per_case,
                  "active_case_counts": active,
                  "actual_native_steps_both_sides": total_actual_native,
                  "max_abs_error": maximum[0],
                  "absolute_tolerance": TOLERANCE,
                  "inputs_unchanged": unchanged,
                  "elapsed_wall_s": time.monotonic() - started,
                  "source_identity": hashes,
                  "not_a_policy_qualification": True}
        write_json(output / "result.json", result)
        if base is not None:
            base.close()
        if fast is not None:
            fast.close()
    print(json_text(result))
    return 0 if result["status"] == "EQUIVALENT" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory; never resumes or overwrites a failed pair")
    args = parser.parse_args()
    return run(args.output.resolve())


if __name__ == "__main__":
    sys.exit(main())
