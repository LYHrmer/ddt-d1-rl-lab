"""Fixed-action time alignment and failure retention with synthetic arrays and models."""
import numpy as np
import pytest

from tools92 import replay92 as replay


def _state():
    state = np.zeros(45, dtype=np.float64)
    state[2:4] = [0.45, 1.]
    return state


@pytest.mark.parametrize("task,start,first", [("move", 2, 3), ("stand", 0, 0)])
def test_uses_physical_applied_row_for_each_endpoint(task, start, first):
    issued = np.repeat((np.arange(1, 9, dtype=np.float32) / 16.)[:, None], 16, axis=1)
    applied = np.zeros_like(issued, dtype=np.float64)
    applied[1:] = issued[:-1]
    fifo = applied[first].astype(np.float32)
    actions = replay._action_window(task, issued, applied, fifo, start, 4)
    np.testing.assert_array_equal(actions, applied[first:first + 4])
    assert actions.dtype == np.float32
    assert not np.array_equal(actions, issued[first:first + 4])
    with pytest.raises(ValueError, match="branch FIFO"):
        replay._action_window(task, issued, applied, fifo + 0.125, start, 4)
    applied[first + 1, 5] += np.finfo(np.float64).eps  # would be hidden by early f32 conversion
    with pytest.raises(ValueError, match="one-tick FIFO"):
        replay._action_window(task, issued, applied, fifo, start, 4)


def test_only_initial_state_and_fixed_actions_admitted(tmp_path):
    path = tmp_path / "extra.npz"
    np.savez(path, state45=_state(), applied16=np.zeros((4, 16), np.float32),
             future_state45=np.zeros((4, 45), np.float64))
    with pytest.raises(ValueError, match="unexpected replay inputs"):
        replay._read_input(path, 4)


@pytest.mark.parametrize("task,start,first", [("move", 2, 3), ("stand", 0, 0)])
def test_exports_only_branch_state_and_same_arm_window(tmp_path, monkeypatch, task, start, first):
    state = _state()
    q = np.tile(state[:23], (8, 1))
    q[:, 0] = np.arange(8) + 100
    v = np.zeros((8, 22))
    issued = np.tile(np.arange(1, 9, dtype=np.float32)[:, None] / 10., (1, 16))
    applied = np.zeros_like(issued, dtype=np.float64)
    applied[1:] = issued[:-1]
    source = tmp_path / "physical.npz"
    if task == "move":
        state[0] = q[start, 0]
        np.savez(source, qpos=q, qvel=v, issued_action=issued, applied_action=applied)
    else:
        np.savez(source, initial_qpos=state[:23], initial_qvel=state[23:], issued=issued, applied=applied)

    prefix = replay._prefix

    def bounded(archive, key, rows):
        if key in ("qpos", "qvel"):
            assert task == "move" and rows == start + 1
        else:
            assert key in ("issued", "applied", "issued_action", "applied_action") and rows == first + 4
        return prefix(archive, key, rows)

    monkeypatch.setattr(replay, "_prefix", bounded)
    spec = dict(task=task, start_tick=start, horizon=4)
    fifo = applied[first].astype(np.float32)
    np.testing.assert_array_equal(replay._physical_input(source, spec, state, fifo), applied[first:first + 4])
    wrong_state = state.copy()
    wrong_state[0] += 1.
    with pytest.raises(ValueError, match="branch state"):
        replay._physical_input(source, spec, wrong_state, fifo)


def _case():
    return dict(task="move", reset_seed=6, parent_seed=9, arm="original", start_tick=180,
                horizon=4, model_seed=1, input_sha256="synthetic")


def _tape():
    return np.arange(4 * 16, dtype=np.float32).reshape(4, 16) / 64.


def test_free_running_state_and_exact_tape_no_second_fifo(tmp_path):
    torch = pytest.importorskip("torch")

    class Accumulator:
        def __init__(self):
            self.calls = []

        def step(self, state, action, terrain):
            assert not torch.is_grad_enabled()
            assert state.shape == (1, 45) and action.shape == (1, 16)
            assert state.dtype == action.dtype == torch.float32
            assert state.device.type == action.device.type == terrain.device.type == "cpu"
            assert terrain.tolist() == [1]
            self.calls.append(action[0].numpy().copy())
            out = state.clone()
            out[:, 0] += action[:, 7]
            return out

    case, state, actions = _case(), _state(), _tape()
    model = Accumulator()
    row = replay._roll_case(tmp_path, case, model, state, actions, (.27, .6))
    np.testing.assert_array_equal(model.calls, actions)
    assert (row["status"], row["reason"], row["attempted_steps"], row["accepted_steps"]) == ("COMPLETE", None, 4, 4)
    with np.load(tmp_path / row["trace"], allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["applied16"], actions)
        np.testing.assert_array_equal(trace["endpoint_tick"], np.arange(180, 185))
        np.testing.assert_array_equal(trace["state45"][:, 0], np.r_[0., np.cumsum(actions[:, 7])])
        assert len(trace["rejected_state45"]) == 0
    with pytest.raises(FileExistsError):
        replay._roll_case(tmp_path, case, model, state, actions, (.27, .6))
    assert len(model.calls) == 4


@pytest.mark.parametrize("failure,status,accepted,rejected", [
    ("nan", "NONFINITE", 1, 1), ("quaternion", "DIVERGED", 1, 1),
    ("height", "PREDICTED_FALL", 2, 0), ("tilt", "PREDICTED_FALL", 2, 0),
    ("raise", "MODEL_ERROR", 1, 0),
])
def test_keeps_failed_attempt_and_accepted_prefix(tmp_path, failure, status, accepted, rejected):
    pytest.importorskip("torch")

    class FailsOnSecond:
        def __init__(self):
            self.calls = 0

        def step(self, state, action, terrain):
            self.calls += 1
            out = state.clone()
            if self.calls == 2:
                if failure == "raise":
                    raise FloatingPointError("synthetic overflow")
                if failure == "tilt":
                    out[0, 3:7] = out[0, [4, 3, 5, 6]]
                else:
                    out[0, {"height": 2, "quaternion": 3, "nan": 0}[failure]] = np.nan if failure == "nan" else 0.
            return out

    row = replay._roll_case(tmp_path, _case(), FailsOnSecond(), _state(), _tape(), (.27, .6))
    assert (row["status"], row["attempted_steps"], row["accepted_steps"]) == (status, 2, accepted)
    assert row["reason"] and row["trace"].endswith("partial_trace.npz")
    with np.load(tmp_path / row["trace"], allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["applied16"], _tape()[:2])
        np.testing.assert_array_equal(trace["endpoint_tick"], np.arange(180, 181 + accepted))
        assert trace["state45"].shape == (accepted + 1, 45)
        assert trace["rejected_state45"].shape == (rejected, 45)
