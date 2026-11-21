"""Synthetic clocks, endpoints, event handling and source wiring for round 97."""
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools97 import replay97 as replay


def start(x):
    state = np.zeros(45, dtype=np.float32)
    state[0], state[2], state[3] = x, .5, 1.
    return state


class Toy:
    def __init__(self, fail=None):
        self.inputs = []
        self.fail = fail

    def step(self, state, action, terrain):
        assert torch.is_inference_mode_enabled() and terrain.tolist() == [1]
        self.inputs.append(action[0].numpy().copy())
        future = state.clone()
        future[0, 0] = state[0, 0] + action[0, 0]
        if len(self.inputs) == 3:
            if self.fail == "throw":
                raise RuntimeError("synthetic failed step")
            if self.fail == "fall":
                future[0, 2] = .1
            elif self.fail == "quat":
                future[0, 3] = .98
            elif self.fail == "nonfinite":
                future[0, 0] = float("nan")
        return future


def test_two_start_choices_share_identical_already_applied_suffix():
    tape = np.zeros((16, 16), dtype=np.float32)
    tape[:, 0] = np.arange(1, 17) / 32.
    for x in (1., 2.):
        world = Toy()
        arrays, row = replay._roll(world, start(x), tape, (.2, .7))
        assert row["status"] == "COMPLETE" and row["error"] is None
        assert row["ticks"] == row["attempted_model_steps"] == row["completed_model_steps"] == 16
        assert row["retained_model_steps"] == 16 and row["rejected_endpoint_tick"] is None
        np.testing.assert_array_equal(world.inputs, tape)
        np.testing.assert_array_equal(arrays["state45"][:, 0], x + np.r_[0, tape[:, 0].cumsum()])
        np.testing.assert_array_equal(arrays["applied16"], tape)
        assert replay._f32_sha(tape) == hashlib.sha256(tape.tobytes()).hexdigest()


@pytest.mark.parametrize("mode,completed,returned", [
    ("throw", 2, False), ("fall", 3, True), ("quat", 3, True), ("nonfinite", 3, True)])
def test_invalid_third_step_retains_only_two_accepted_endpoints(mode, completed, returned):
    tape = np.ones((16, 16), dtype=np.float32) / 16.
    data, row = replay._roll(Toy(mode), start(1.), tape, (.2, .7))
    assert row["status"] == "PARTIAL" and row["ticks"] == row["retained_model_steps"] == 2
    assert row["attempted_model_steps"] == 3 and row["completed_model_steps"] == completed
    assert row["rejected_endpoint_tick"] == 19 and row["error"]
    assert data["state45"].shape == (3, 45) and np.isfinite(data["state45"]).all()
    np.testing.assert_array_equal(data["applied16"], tape[:3])
    assert ("rejected_state45" in data) == returned


def test_physical_event_at_final_tick_is_an_observed_endpoint_and_tape_offset_is_exact(tmp_path):
    folder = tmp_path / "native"
    folder.mkdir()
    states = np.repeat(start(1.).astype(np.float64)[None], 33, axis=0)
    states[:, 0] += np.arange(33) * .001
    issued = np.repeat((np.arange(32, dtype=np.float32) / 64.)[:, None], 16, axis=1)
    applied = np.concatenate((np.zeros_like(issued[:1]), issued[:-1]))
    flags = {k: np.zeros(32, dtype=bool) for k in ("fallen", "illegal", "terminated", "truncated")}
    flags["illegal"][-1] = flags["terminated"][-1] = True
    trace_path = folder / "trace.npz"
    replay.arrays(trace_path, dict(state45=states, issued16=issued, applied16=applied, **flags,
                                   reset_seed=96201, requested_ticks=32, completed_transitions=32,
                                   policy_steps=32, servo_steps=160, native_steps=320))
    replay.write(tmp_path / "native.json", dict(error=None, trace_file="native/trace.npz", safe=False,
                 ticks=32, trace_sha256=replay.digest(trace_path), **{k: bool(v.any()) for k, v in flags.items()}))
    reset, cut, tape, _ = replay._recorded_native(tmp_path, 96201)
    assert cut.dtype == np.float32 and cut[0] == np.float32(states[16, 0])
    np.testing.assert_array_equal(tape, applied[16:32])
    np.testing.assert_array_equal(tape[0], issued[15])
    forecasts = states.astype(np.float32)
    forecasts[16, 0] += .05
    replay.arrays(tmp_path / "ph.npz", {"state45": forecasts})
    replay.write(tmp_path / "ph.json", dict(seed=94011, reset=96201, model="paired", alpha=.5, file="ph.npz",
                 ticks=32, safe=True, state_sha256=replay.digest(tmp_path / "ph.npz")))
    own, _ = replay._forecast_cut(tmp_path, 94011, 96201, "paired", "ph", reset)
    np.testing.assert_array_equal(own, forecasts[16])
    assert own[0] != cut[0]
    with pytest.raises(ValueError, match="invalid round-96 forecast"):
        replay._forecast_cut(tmp_path, 94011, 96201, "control", "ph", reset)
    with trace_path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="native receipt differs"):
        replay._recorded_native(tmp_path, 96201)
