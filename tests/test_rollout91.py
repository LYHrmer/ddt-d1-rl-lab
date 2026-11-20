"""Causal rollout tests on synthetic states and policies, without integration."""
import numpy as np
import pytest

from tools91 import rollout91 as runner
from training.d1_rl.interface import DEFAULT_Q, WHEEL_INDICES, build_observation, pack_actor_input


def reset_state():
    state = np.zeros(45, dtype=np.float64)
    state[2:4] = [.46, 1.]
    state[7:23] = DEFAULT_Q
    return state


def start_stand():
    state = reset_state()
    fifo = np.zeros(16, np.float32)
    frame, filtered = build_observation(state[26:29], state[3:7], [0.] * 3,
                                        state[7:23], state[29:45], fifo, np.zeros(3))
    obs = pack_actor_input(frame, np.tile(frame, (10, 1)))
    return runner.Start("stand", 90101, 87011, 0, state, obs, filtered, fifo)


def test_cut_rejects_post_boundary_fields_and_mismatched_fifo(tmp_path):
    start = start_stand()
    values = dict(state45=start.state, fifo16=start.fifo, observation627=start.observation,
                  previous_filtered=start.filtered, start_tick=np.int64(0), parent_seed=np.int64(87011),
                  reset_seed=np.int64(90101))
    for name, change, expected in (("good", {}, None), ("future", {"future_policy": [0.]}, "unexpected"),
                                   ("queue", {"fifo16": np.ones(16, np.float32)}, "FIFO")):
        path = tmp_path / f"{name}.npz"
        np.savez(path, **(values | change))
        if expected:
            with pytest.raises(ValueError, match=expected):
                runner._load_start(path, "stand", 90101, 87011)
        else:
            assert np.array_equal(runner._load_start(path, "stand", 90101, 87011).state, start.state)


def feedback(body, features):
    action = np.zeros(16, np.float32)
    action[WHEEL_INDICES] = np.tanh(-.25 + body[9] + features[0])
    action[0] = np.tanh(body[9])  # responds to the imagined hip angle, not previous issued action
    return action


class SyntheticModel:
    def step(self, state, applied, terrain):
        assert terrain.tolist() == [1]
        next_state = state.clone()
        wheel = applied[:, 3]
        next_state[:, 0] += .02 * wheel
        next_state[:, 7] += .1 * wheel
        next_state[:, 23] = wheel
        next_state[:, 26] = .5 * wheel
        return next_state


def run_case(tmp_path, start, arm, model):
    folder = tmp_path / arm
    folder.mkdir()
    return runner._roll_arm(start, arm, model, feedback, None, {}, (.27, .6), folder)


def test_each_arm_recomputes_feedback_from_own_state_fifo_and_gyro(tmp_path):
    pytest.importorskip("torch")
    start = start_stand()
    a, b = [run_case(tmp_path, start, arm, SyntheticModel()) for arm in ("original", "common")]
    assert a["steps"] == b["steps"] == a["attempted_steps"] == b["attempted_steps"] == 50
    assert a["score"] > b["score"] == 0.
    with np.load(tmp_path / a["trace"]) as original, np.load(tmp_path / b["trace"]) as common:
        for trace in (original, common):
            assert np.array_equal(trace["applied16"][0], start.fifo)
            assert np.array_equal(trace["applied16"][1:], trace["issued16"][:-1])
            assert np.array_equal(trace["issued16"][:-1], trace["observation627"][1:, 41:57])
            assert np.array_equal(trace["observation627"][:-1, :57], trace["observation627"][1:, -57:])
            expected_gyro = .03 * trace["observation627"][:-1, 0] + .97 * .25 * trace["state45"][1:-1, 26]
            np.testing.assert_allclose(trace["observation627"][1:, 0], expected_gyro, atol=1e-7)
        assert np.array_equal(original["raw16"][0], common["raw16"][0])
        assert np.array_equal(original["state45"][:2], common["state45"][:2])
        assert not np.array_equal(original["state45"][2], common["state45"][2])
        assert original["raw16"][2, 0] != common["raw16"][2, 0]


@pytest.mark.parametrize("failure,status,kept", [("nan", "NONFINITE", 0), ("quat", "DIVERGED", 0),
                                                  ("fall", "PREDICTED_FALL", 1), ("raise", "MODEL_ERROR", 0)])
def test_failed_state_preserved_and_never_completed(tmp_path, failure, status, kept):
    pytest.importorskip("torch")

    class FailedModel:
        def step(self, state, applied, terrain):
            if failure == "raise":
                raise FloatingPointError("model overflow")
            result = state.clone()
            result[0, {"nan": 0, "quat": 3, "fall": 2}[failure]] = np.nan if failure == "nan" else 0.
            return result

    case = run_case(tmp_path, start_stand(), "original", FailedModel())
    assert (case["status"], case["steps"], case["attempted_steps"], case["score"]) == (status, kept, 1, None)
    assert case["trace"].endswith("partial_trace.npz")
    with np.load(tmp_path / case["trace"]) as rows:
        assert len(rows["rejected_state45"]) == (failure in ("nan", "quat"))
        assert len(rows["applied16"]) == len(rows["decision_tick"]) == 1
        assert len(rows["state45"]) == len(rows["endpoint_tick"]) == 1 + kept


def test_score_uses_body_axis_and_all_100_final_endpoints():
    states = np.tile(reset_state(), (121, 1))
    states[:, 3:7] = [2**-.5, 0., 0., 2**-.5]  # quarter turn, body x points along world y
    states[:, 23] = -99.
    states[:, 24] = np.arange(180, 301)
    assert runner._score("move", list(range(180, 301)), states) == pytest.approx((201 + 300) / 2)
    assert runner._score("move", list(range(180, 300)), states[:-1]) is None


def test_standing_policy_observes_its_own_predicted_ranges(tmp_path):
    pytest.importorskip("torch")

    class Sensor:
        def at(self, tick, state):
            packet = np.zeros(16, dtype=np.float32)
            if tick >= 2:
                packet[0] = .3 + state[0]
            return packet

    start = start_stand()
    for arm in runner.ARMS["stand"]:
        folder = tmp_path / arm
        folder.mkdir()
        runner._roll_arm(start, arm, SyntheticModel(), feedback, Sensor(), {}, (.27, .6), folder)
    with np.load(tmp_path / "original/trace.npz") as a, np.load(tmp_path / "common/trace.npz") as b:
        for trace in (a, b):
            assert not np.any(trace["ranges16"][:2])
            np.testing.assert_allclose(trace["ranges16"][2:, 0], .3 + trace["state45"][2:-1, 0],
                                       rtol=1e-6, atol=1e-7)
            for tick in range(50):
                np.testing.assert_array_equal(trace["raw16"][tick],
                                              feedback(trace["observation627"][tick], trace["ranges16"][tick]))
        assert a["ranges16"][2, 0] != b["ranges16"][2, 0]
