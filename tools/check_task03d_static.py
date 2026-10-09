#!/usr/bin/env python3
"""Check the frozen 03d command course and 03c parity with zero integration.

Model load/reset may call mj_forward; mj_step is intercepted for this process.
The receipt is evidence for the static contract only, never task performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from d1_rl.env import D1Env  # noqa: E402
from d1_rl.task03c import D1Task03cEnv  # noqa: E402
from d1_rl.task03d import D1Task03dEnv, _expected_03d_config  # noqa: E402


INPUTS = (
    "tools/check_task03d_static.py",
    "training/d1_rl/task03d.py",
    "training/configs/pilot03d.json",
    "docs/career_execution_contract_03d_addendum.md",
    "training/d1_rl/task03c.py",
    "training/configs/pilot03c.json",
    "training/d1_rl/fast_checks.py",
    "training/d1_rl/env.py",
    "training/d1_rl/interface.py",
    "training/configs/pilot03b.json",
    "training/configs/interface03b.json",
    "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_contract_plane_02.xml",
    "src/ddt_ros2_control/urdfs/d1_description/mujoco/robot.xml",
)
FROZEN = {
    "training/d1_rl/task03c.py": "782f244b0f0916914bb1dcb0235306d45b18e8c36fb65f39cd3518b490b30475",
    "training/configs/pilot03c.json": "562bdc6b26fba7bbb4906f7045ba11cbe70848a3335bd8d5e1659540a3f2b604",
    "training/d1_rl/fast_checks.py": "ee97af7a67a6d964ce4611524475db2a6249d8cd1fcef8f5a09211beec3280cd",
    "training/d1_rl/env.py": "f9f7c96b5f91548239578253de25dee5a3e7fee30a9e882502b205cf082e2c2b",
    "training/d1_rl/interface.py": "69f8020b5cd16d6625fcbd681e975760d04e7a5ff21830589fbb432c57866232",
    "training/configs/pilot03b.json": "3996e6e7c677be59e2852692d3f71ef7d0de8a689b6be3493f9ba466d92a589f",
    "training/configs/interface03b.json": "dee7aaaea980fbfbfb438c85be9e97fc605bb5a375f62f366fccc7edbaa7bff8",
    "docs/career_execution_contract_03d_addendum.md": "4246d1c90f375e73424a1938f130df065f5c3920686af65ddfc11063ce18331c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BoundaryRng:
    """Force all inclusive integer bounds without advancing a real RNG."""

    def __init__(self, mode: str, high: bool):
        self.mode = mode
        self.high = high
        self.random_calls = 0
        self.uniform_calls = 0
        self.integer_calls = 0

    def random(self):
        self.random_calls += 1
        return (0.75 if self.mode == "turn" else 0.25) if self.random_calls == 1 else (0.75 if self.high else 0.25)

    def uniform(self, low, high):
        self.uniform_calls += 1
        return high if self.high else low

    def integers(self, low, high):
        self.integer_calls += 1
        return high - 1 if self.high else low


def check_schedule(course: dict, *, forced: str | None = None, high: bool = False) -> dict:
    segments = course["segments"]
    mode, sign = course["mode"], course["sign"]
    assert mode in ("translation", "turn") and sign in (-1, 1)
    assert segments[0][0] == 0 and segments[-1][1] == 1000
    assert all(a < b and len(cmd) == 3 and cmd[1] == 0.0 for a, b, cmd in segments)
    assert all(left[1] == right[0] for left, right in zip(segments, segments[1:]))
    assert segments[0] == (0, 50, (0.0, 0.0, 0.0))
    assert segments[-1][2] == (0.0, 0.0, 0.0)
    assert 1000 - segments[-1][0] >= 200
    p = course["parameters"]
    nonzero = [seg for seg in segments if seg[2] != (0.0, 0.0, 0.0)]
    motion_ticks = sum(end - begin for begin, end, _ in nonzero)
    assert 400 <= motion_ticks <= 600
    zero_ticks = 1000 - motion_ticks
    assert 400 <= zero_ticks <= 600
    assert sum(segments[i][2] != (0.0, 0.0, 0.0) and
               segments[i + 1][2] == (0.0, 0.0, 0.0)
               for i in range(len(segments) - 1)) == 2
    gap = p["middle_zero_ticks"]
    assert 100 <= gap <= 150
    if mode == "translation":
        assert len(segments) == 5
        amp, = p["amplitudes"]
        d1, d2 = p["motion_ticks"]
        assert 0.1 <= amp <= 0.3 and 200 <= d1 == d2 <= 300
        assert segments[1][2] == (sign * amp, 0.0, 0.0)
        assert segments[2][2] == (0.0, 0.0, 0.0)
        assert segments[3][2] == (-sign * amp, 0.0, 0.0)
        assert [b - a for a, b, _ in segments[1:4]] == [d1, gap, d1]
    else:
        assert len(segments) == 7
        a1, a2 = p["amplitudes"]
        d1, d1pair, d2, d2pair = p["motion_ticks"]
        assert 0.1 <= a1 <= 0.3 and 0.1 <= a2 <= 0.3
        assert 100 <= d1 == d1pair <= 150 and 100 <= d2 == d2pair <= 150
        assert [seg[2] for seg in segments[1:6]] == [
            (0.0, 0.0, sign * a1), (0.0, 0.0, -sign * a1),
            (0.0, 0.0, 0.0), (0.0, 0.0, -sign * a2),
            (0.0, 0.0, sign * a2)]
        assert [b - a for a, b, _ in segments[1:6]] == [d1, d1, gap, d2, d2]
        assert segments[1][2][2] * segments[2][2][2] < 0
        assert segments[4][2][2] * segments[5][2][2] < 0
    if forced is not None:
        assert mode == forced
        assert all(x == (0.3 if high else 0.1) for x in p["amplitudes"])
        lo, hi = (100, 150) if mode == "turn" else (200, 300)
        assert all(d == (hi if high else lo) for d in p["motion_ticks"])
        assert gap == (150 if high else 100)
    return {"mode": mode, "sign": sign, "motion_ticks": motion_ticks,
            "zero_ticks": zero_ticks, "tail_zero_ticks": 1000 - segments[-1][0]}


def checks(tests: list[dict]) -> dict:
    assert D1Task03dEnv.__mro__[1] is D1Task03cEnv
    for method in ("step", "reset", "_reward", "_observe", "_delayed_action",
                   "_contact_diagnostics", "_check_finite_state"):
        assert getattr(D1Task03dEnv, method) is getattr(D1Task03cEnv, method), method
    assert D1Task03dEnv.step is D1Env.step
    expected = _expected_03d_config()
    assert json.loads((ROOT / "training/configs/pilot03d.json").read_text()) == expected
    assert expected["interface_id"] == "d1-selftrained-pilot03b-v2"
    assert expected["reward"] == json.loads((ROOT / "training/configs/pilot03c.json").read_text())["reward"]
    tests.append({"name": "only_training_command_changes", "passed": True})

    a = D1Task03dEnv(seed=7001, arm="R")
    b = D1Task03cEnv(seed=7001, arm="R")
    try:
        obs_a, info_a = a.reset(seed=7001)
        obs_b, info_b = b.reset(seed=7001)
        assert np.array_equal(obs_a, obs_b)
        assert info_a == info_b
        assert np.array_equal(a.data.qpos, b.data.qpos)
        assert np.array_equal(a.data.qvel, b.data.qvel)
        assert np.array_equal(a.data.qacc_warmstart, b.data.qacc_warmstart)
        assert a._delay_ticks == b._delay_ticks
        assert a.counters()["native_steps"] == b.counters()["native_steps"] == 0
        tests.append({"name": "03c_03d_same_fresh_reset_and_independent_reset_delay_rng", "passed": True})

        first = a._training_schedule
        assert first is not None
        first_report = check_schedule(first)
        a._command_rng.random(200)
        repeat_obs, _ = a.reset(seed=7001)
        assert a._training_schedule == first and np.array_equal(repeat_obs, obs_a)
        tests.append({"name": "fixed_seed_replays_full_course_even_after_command_rng_draws",
                      "passed": True, **first_report})

        # Vary seed and verify both modes occur; all 1000 tick lookups are static.
        sampled = []
        for seed in range(7001, 7065):
            a.reset(seed=seed)
            course = a._training_schedule
            sampled.append(check_schedule(course))
            for tick in range(1001):
                a._episode_steps = tick
                a._update_command()
                expected_command = next((cmd for begin, end, cmd in course["segments"]
                                         if begin <= tick < end), (0.0, 0.0, 0.0))
                assert tuple(a._command) == expected_command
        assert {item["mode"] for item in sampled} == {"translation", "turn"}
        tests.append({"name": "64_seed_both_modes_1001_tick_lookup_and_terminal_tick_zero",
                      "passed": True, "sampled_modes": {mode: sum(item["mode"] == mode for item in sampled)
                                                   for mode in ("translation", "turn")}})

        for mode in ("translation", "turn"):
            for high in (False, True):
                draw = BoundaryRng(mode, high)
                a._command_rng = draw
                a._episode_steps = 0
                a._schedule_rng = None
                a._update_command()
                result = check_schedule(a._training_schedule, forced=mode, high=high)
                assert draw.random_calls == 2
                assert draw.uniform_calls == (1 if mode == "translation" else 2)
                assert draw.integer_calls == (2 if mode == "translation" else 3)
                tests.append({"name": f"{mode}_{'upper' if high else 'lower'}_inclusive_bounds",
                              "passed": True, **result})

        a.set_command(0.12, -0.18)
        a._episode_steps = 700
        a._update_command()
        assert tuple(a._command) == (0.12, 0.0, -0.18)
        assert draw.random_calls == 2
        tests.append({"name": "manual_command_precedence_no_course_draw", "passed": True})

        eval_sequence = [{"start_s": 0.0, "vx": 0.0, "wz": 0.0},
                         {"start_s": 2.0, "vx": -0.2, "wz": 0.2}]
        c = D1Task03dEnv(seed=7001, arm="B", eval_command_sequence=eval_sequence)
        try:
            c.reset(seed=7001)
            assert c._training_schedule is None and c._schedule_rng is None
            c._episode_steps = 100
            c._update_command()
            assert tuple(c._command) == (-0.2, 0.0, 0.2)
            c._episode_steps = 1000
            c._update_command()
            assert tuple(c._command) == (-0.2, 0.0, 0.2)
            assert c._training_schedule is None and c.counters()["native_steps"] == 0
            tests.append({"name": "eval_sequence_precedence_zero_course_draw", "passed": True})
        finally:
            c.close()

        # Identical static state, interval command and action must give every
        # reward field exactly the same value; next command differs on purpose.
        a.reset(seed=7001)
        b.reset(seed=7001)
        a.data.qpos[a.qpos_ids[0]] += 0.11
        b.data.qpos[b.qpos_ids[0]] += 0.11
        a.data.qpos[a.free_qpos + 2] += 0.01
        b.data.qpos[b.free_qpos + 2] += 0.01
        a._last_policy_action[:] = 0.2
        b._last_policy_action[:] = 0.2
        sensors = {"trunk_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                   "trunk_gyro": np.array([0.0, 0.0, 0.04]),
                   "frame_vel": np.array([0.03, -0.02, 0.0])}
        for interval in ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, -0.2)):
            a._command[:] = (0.2, 0.0, 0.1)
            b._command[:] = (-0.2, 0.0, -0.1)
            with patch.object(a, "_sensor", side_effect=lambda name: sensors[name]), \
                 patch.object(b, "_sensor", side_effect=lambda name: sensors[name]):
                args = (np.array(interval), np.full(16, -0.1), 12.5, False, 0)
                reward_a = a._reward(*args)
                reward_b = b._reward(*args)
            assert reward_a == reward_b
        tests.append({"name": "03c_03d_all_reward_terms_and_metrics_exact_for_equal_interval_state",
                      "passed": True, "interval_cases": 3})
        return {"03d": a.counters(), "03c": b.counters()}
    finally:
        a.close()
        b.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new evidence directory; existing paths are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = {member: sha256(ROOT / member) for member in INPUTS}
    tests: list[dict] = []
    counts = None
    error = None
    mj_step_attempts = 0

    def forbidden_step(*_args, **_kwargs):
        nonlocal mj_step_attempts
        mj_step_attempts += 1
        raise AssertionError("03d static check forbids mj_step")

    try:
        for member, expected in FROZEN.items():
            assert before[member] == expected, f"Frozen source changed: {member}"
        with patch.object(mujoco, "mj_step", forbidden_step):
            counts = checks(tests)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    after = {member: sha256(ROOT / member) for member in INPUTS}
    unchanged = before == after
    zero_native = counts is not None and all(
        row["native_steps"] == row["policy_transitions"] == row["pd_updates"] == 0
        for row in counts.values())
    status = "PASS" if error is None and unchanged and mj_step_attempts == 0 and zero_native else "FAIL"
    receipt = {
        "status": status, "error": error, "test_count": len(tests), "tests": tests,
        "mj_step_attempts": mj_step_attempts, "actual_counts": counts,
        "physics_native_steps": sum(row["native_steps"] for row in counts.values()) if counts else 0,
        "mujoco_version": mujoco.__version__,
        "input_sha256_before": before, "input_sha256_after": after,
        "inputs_unchanged": unchanged,
        "limits": "Static command/reward/reset checks only; no rollout, training, or task success claim.",
    }
    (output / "report.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "tests": len(tests), "mj_step_attempts": mj_step_attempts,
                      "physics_native_steps": receipt["physics_native_steps"]}, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
