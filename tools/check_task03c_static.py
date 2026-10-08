#!/usr/bin/env python3
"""Reproduce the contract-03c command/reward checks without integration.

This script deliberately permits mj_forward during model loading and reset,
but intercepts mj_step throughout. It never calls env.step or a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from d1_rl.env import D1Env  # noqa: E402
from d1_rl.fast_checks import D1FastChecksEnv  # noqa: E402
from d1_rl.interface import DEFAULT_Q  # noqa: E402
from d1_rl.task03c import D1Task03cEnv, LEG_INDICES, WHEEL_INDICES  # noqa: E402


INPUTS = (
    "tools/check_task03c_static.py",
    "training/d1_rl/task03c.py",
    "training/d1_rl/fast_checks.py",
    "training/d1_rl/env.py",
    "training/d1_rl/interface.py",
    "training/configs/pilot03c.json",
    "training/configs/pilot03b.json",
    "training/configs/interface03b.json",
    "docs/career_execution_contract_03c_addendum.md",
    "src/ddt_ros2_control/urdfs/d1_description/mujoco/scene_contract_plane_02.xml",
    "src/ddt_ros2_control/urdfs/d1_description/mujoco/robot.xml",
)
FROZEN_03B_SHA256 = {
    "training/d1_rl/fast_checks.py": "ee97af7a67a6d964ce4611524475db2a6249d8cd1fcef8f5a09211beec3280cd",
    "training/d1_rl/env.py": "f9f7c96b5f91548239578253de25dee5a3e7fee30a9e882502b205cf082e2c2b",
    "training/d1_rl/interface.py": "69f8020b5cd16d6625fcbd681e975760d04e7a5ff21830589fbb432c57866232",
    "training/configs/pilot03b.json": "3996e6e7c677be59e2852692d3f71ef7d0de8a689b6be3493f9ba466d92a589f",
    "training/configs/interface03b.json": "dee7aaaea980fbfbfb438c85be9e97fc605bb5a375f62f366fccc7edbaa7bff8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Draw:
    """Count one category draw and only the active-axis uniform draws."""

    def __init__(self, category: float):
        self.category = category
        self.random_calls = 0
        self.uniform_calls = 0

    def random(self):
        self.random_calls += 1
        return self.category

    def uniform(self, _low, _high):
        self.uniform_calls += 1
        return (0.11, -0.22)[self.uniform_calls - 1]


def check(env: D1Task03cEnv, tests: list[dict]) -> None:
    assert D1Task03cEnv.__mro__[1] is D1FastChecksEnv
    assert D1Task03cEnv.step is D1Env.step
    assert D1Task03cEnv.reset is D1Env.reset
    assert D1Task03cEnv._observe is D1Env._observe
    assert D1Task03cEnv._delayed_action is D1Env._delayed_action
    assert D1Task03cEnv._contact_diagnostics is D1FastChecksEnv._contact_diagnostics
    assert D1Task03cEnv._check_finite_state is D1FastChecksEnv._check_finite_state
    tests.append({"name": "physical_timing_reset_observation_and_diagnostics_inherited", "passed": True})

    config = env.config
    assert config["version"] == "d1-pure-ppo-pilot-03c"
    assert config["interface_id"] == "d1-selftrained-pilot03b-v2"
    assert config["commands"]["zero_probability"] == 0.2
    assert config["commands"]["category_probabilities"] == {
        "stop": 0.2, "pure_vx": 0.3, "pure_wz": 0.3, "mixed": 0.2}
    assert config["reward"]["wz_tracking_weight"] == 2.0
    assert config["reward"]["wz_tracking_error_scale_rad2_s2"] == 0.04
    assert config["reward"]["gravity_xy_squared_weight"] == -6.0
    assert config["reward"]["height_error_squared_weight"] == -20.0
    assert config["reward"]["action_rate_mean_squared_weight"] == -0.1
    assert config["reward"]["leg_pose_deviation_sum_squared_weight"] == -0.5
    assert config["reward"]["standstill_tracking_weight"] == 1.0
    assert config["reward"]["multiply_total_by_policy_dt_s"] == 0.02
    tests.append({"name": "03c_identity_and_explicit_reward_constants", "passed": True})

    observation, reset_info = env.reset(seed=7001)
    assert observation.shape == (627,) and np.isfinite(observation).all()
    assert env.counters()["native_steps"] == env.counters()["policy_transitions"] == 0
    assert env.get_initial_state()["interface_id"] == config["interface_id"]
    initial_qpos = env.data.qpos.copy()
    initial_delay = reset_info["delay_ticks"]
    env._command_rng.random(100)
    repeat_observation, repeat_info = env.reset(seed=7001)
    assert np.array_equal(initial_qpos, env.data.qpos)
    assert np.array_equal(observation, repeat_observation)
    assert initial_delay == repeat_info["delay_ticks"]
    assert env.counters()["native_steps"] == 0
    tests.append({"name": "fresh_reset_627D_and_independent_command_rng", "passed": True,
                  "reset_seed": 7001, "delay_ticks": initial_delay})

    env._episode_steps = env.first_zero_ticks - 1
    env._command_phase = -1
    draw = Draw(0.8)
    env._command_rng = draw
    env._update_command()
    assert np.array_equal(env._command, [0.0, 0.0, 0.0]) and draw.random_calls == 0
    tests.append({"name": "first_one_second_exact_zero_no_category_draw", "passed": True})

    for category, expected, uniform_calls, name in (
        (0.199, [0.0, 0.0, 0.0], 0, "stop"),
        (0.2, [0.11, 0.0, 0.0], 1, "pure_vx_lower_boundary"),
        (0.499, [0.11, 0.0, 0.0], 1, "pure_vx"),
        (0.5, [0.0, 0.0, 0.11], 1, "pure_wz_lower_boundary"),
        (0.799, [0.0, 0.0, 0.11], 1, "pure_wz"),
        (0.8, [0.11, 0.0, -0.22], 2, "mixed_lower_boundary"),
    ):
        env._episode_steps = env.first_zero_ticks
        env._command_phase = -1
        draw = Draw(category)
        env._command_rng = draw
        env._update_command()
        assert np.array_equal(env._command, expected)
        assert draw.random_calls == 1 and draw.uniform_calls == uniform_calls
        tests.append({"name": name, "passed": True, "category_draw": category,
                      "command_vx_vy_wz": env._command.tolist(),
                      "active_axis_uniform_draws": uniform_calls})

    env.set_command(0.1, -0.2)
    env._episode_steps = 400
    env._update_command()
    assert np.array_equal(env._command, [0.1, 0.0, -0.2])
    env._manual_command = False
    env.eval_command_sequence = [(0.0, 0.0, 0.0), (2.0, -0.2, 0.2)]
    env._episode_steps = 150
    env._update_command()
    assert np.array_equal(env._command, [-0.2, 0.0, 0.2])
    tests.append({"name": "manual_and_eval_sequence_precede_random_sampling", "passed": True})

    assert WHEEL_INDICES == (3, 7, 11, 15)
    assert LEG_INDICES == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)
    env.data.qpos[env.qpos_ids] = DEFAULT_Q
    env.data.qpos[env.qpos_ids[0]] += 0.2
    env.data.qpos[env.qpos_ids[3]] = 100.0
    env.data.qpos[env.free_qpos + 2] = env.nominal_base_height_m + 0.01
    env._last_policy_action[:] = 0.0
    env._last_policy_action[0] = 0.5
    sensors = {"trunk_quat": np.array([1.0, 0.0, 0.0, 0.0]),
               "trunk_gyro": np.array([0.0, 0.0, 0.01]),
               "frame_vel": np.array([0.01, 0.02, 0.0])}
    with patch.object(env, "_sensor", side_effect=lambda name: sensors[name]):
        env._command[:] = [0.2, 0.0, 0.0]  # Different next command must not suppress standstill.
        reward, terms, _ = env._reward(np.array([0.0, 0.0, 0.0]),
                                       np.zeros(16), 100.0, False, 0)
        standstill = math.exp(-(0.01**2 + 0.02**2) / 0.0004 - 0.01**2 / 0.0025)
        hand = (2 * math.exp(-0.01**2 / 0.04) +
                2 * math.exp(-0.01**2 / 0.04) + 0.25 -
                2 * 0.02**2 - 20 * 0.01**2 -
                0.1 * (0.5**2 / 16) - 1e-5 * 100 -
                0.5 * 0.2**2 + standstill) * 0.02
        assert math.isclose(reward, hand, rel_tol=0, abs_tol=1e-12)
        assert math.isclose(terms["raw"]["leg_pose_deviation_sum_squared"], 0.04,
                            rel_tol=0, abs_tol=1e-12)
        assert math.isclose(terms["raw"]["standstill_tracking"], standstill,
                            rel_tol=0, abs_tol=1e-12)
        assert math.isclose(sum(terms["contributions_per_tick"].values()), reward,
                            rel_tol=0, abs_tol=1e-12)
        tests.append({"name": "independent_hand_reward_and_one_policy_dt", "passed": True,
                      "reward": reward, "hand_reward": hand,
                      "standstill_raw": standstill, "leg_pose_raw": 0.04})

        env.data.qpos[env.qpos_ids[3]] = -100.0
        _, wheel_terms, _ = env._reward(np.array([0.0, 0.0, 0.0]),
                                        np.zeros(16), 100.0, False, 0)
        assert wheel_terms["raw"]["leg_pose_deviation_sum_squared"] == terms["raw"][
            "leg_pose_deviation_sum_squared"]
        tests.append({"name": "actual_12_leg_joints_exclude_four_wheel_axes", "passed": True})

        env._command[:] = 0.0  # Different next command must not create standstill credit.
        _, moving_terms, _ = env._reward(np.array([0.2, 0.0, 0.0]),
                                         np.zeros(16), 100.0, False, 0)
        assert moving_terms["raw"]["standstill_tracking"] == 0.0
        tests.append({"name": "standstill_uses_interval_not_next_command", "passed": True})

    assert env.counters()["native_steps"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new evidence directory; existing paths are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = {member: sha256(ROOT / member) for member in INPUTS}
    tests: list[dict] = []
    env = None
    mj_step_attempts = 0
    error = None

    def forbidden_step(*_args, **_kwargs):
        nonlocal mj_step_attempts
        mj_step_attempts += 1
        raise AssertionError("03c static check forbids mj_step")

    try:
        for member, expected in FROZEN_03B_SHA256.items():
            assert before[member] == expected, f"Frozen 03b source changed: {member}"
        with patch.object(mujoco, "mj_step", forbidden_step):
            env = D1Task03cEnv(seed=7001, arm="R")
            check(env, tests)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        counts = env.counters() if env is not None else None
        if env is not None:
            env.close()
        after = {member: sha256(ROOT / member) for member in INPUTS}
        unchanged = before == after
        status = ("PASS" if error is None and unchanged and mj_step_attempts == 0 and
                  counts is not None and counts["native_steps"] == 0 and
                  counts["policy_transitions"] == 0 else "FAIL")
        report = {
            "status": status, "error": error, "test_count": len(tests), "tests": tests,
            "mj_step_attempts": mj_step_attempts,
            "actual_counts": counts,
            "physics_native_steps": counts["native_steps"] if counts else 0,
            "mujoco_version": mujoco.__version__,
            "input_sha256_before": before, "input_sha256_after": after,
            "inputs_unchanged": unchanged,
            "limits": "Zero integration only; does not establish training success, rollout parity, or task performance.",
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
    print(json.dumps({"status": status, "tests": len(tests),
                      "mj_step_attempts": mj_step_attempts,
                      "physics_native_steps": report["physics_native_steps"]},
                     ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
