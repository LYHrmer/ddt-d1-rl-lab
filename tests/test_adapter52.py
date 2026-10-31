"""Synthetic adapter checks only; no checkpoint asset, robot or MuJoCo steps."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import subprocess
import sys
import zipfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools52 import adapter52 as adapter


@pytest.fixture(scope="module")
def torch():
    if str(ROOT / ".python-deps") not in sys.path:
        sys.path.insert(0, str(ROOT / ".python-deps"))
    return pytest.importorskip("torch")


def synthetic_weights(torch):
    weights = {}
    for prefix, shapes in (("mlp_extractor.encoder", ((0, 128, 627), (2, 64, 128), (4, 3, 64))),
                           ("mlp_extractor.policy_net", ((0, 128, 60), (2, 128, 128)))):
        for index, output, incoming in shapes:
            weights[f"{prefix}.{index}.weight"] = torch.full((output, incoming), 0.001)
            weights[f"{prefix}.{index}.bias"] = torch.full((output,), 0.01)
    weights["action_net.weight"] = torch.full((16, 128), 0.001)
    weights["action_net.bias"] = torch.zeros(16)
    return weights


def test_import_does_not_load_numerical_libraries():
    code = ("import sys; from tools52 import adapter52; "
            "assert not {'torch','mujoco','numpy','stable_baselines3'} & sys.modules.keys()")
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            capture_output=True, text=True, timeout=5, check=False)
    assert result.returncode == 0, result.stderr


class FakeData:
    def __init__(self, model):
        self.model = model
        self.qpos = np.arange(23, dtype=np.float64)
        self.qvel = np.arange(22, dtype=np.float64)
        self.qacc_warmstart = np.arange(22, dtype=np.float64) + 0.25
        self.ctrl = np.array([0.5])
        self.qfrc_applied = np.arange(22, dtype=np.float64) + 1
        self.time = 0.42

    def __copy__(self):
        result = FakeData(self.model)
        for key, value in vars(self).items():
            if key != "model":
                setattr(result, key, copy.deepcopy(value))
        return result


def test_snapshot_restores_data_rng_alias_fifo_and_shaper_without_refunding_work():
    model = object()
    rng = np.random.default_rng(52)
    env = SimpleNamespace(model=model, data=FakeData(model),
                          _counts={"policy_transitions": 4, "native_steps": 40},
                          _command_rng=rng, _schedule_rng=rng,
                          _history=np.zeros((10, 57), np.float32),
                          _delay_fifo=deque([np.full(16, 0.25, np.float32)]),
                          _episode_steps=4, _episode_done=False,
                          _fast_finite_cache=("discard",))
    shaper = SimpleNamespace(state=0.125, history=np.arange(10, dtype=np.float32),
                             raw_history=np.zeros(10, np.float32))
    expected_rng = copy.deepcopy(rng).random(3)
    snapshot = adapter.capture(env, shaper)
    env.data.qacc_warmstart[:] = -1
    env.data.qpos[:] = -1
    env.data.ctrl[:] = -1
    env.data.time = 9
    env._history[:] = -1
    env._delay_fifo.popleft()
    env._command_rng.random(100)
    env._episode_steps = 6
    env._episode_done = True
    env._counts.update(policy_transitions=6, native_steps=60)
    budget = env._counts
    shaper.history[:] = -1
    adapter.restore(env, shaper, snapshot)
    assert env.model is model and env.data.model is model
    assert env._counts is budget and budget == {"policy_transitions": 6, "native_steps": 60}
    assert env._command_rng is env._schedule_rng
    np.testing.assert_array_equal(env._command_rng.random(3), expected_rng)
    np.testing.assert_array_equal(env.data.qacc_warmstart, np.arange(22) + 0.25)
    np.testing.assert_array_equal(env.data.qpos, np.arange(23))
    assert env.data.time == 0.42 and env.data.ctrl[0] == 0.5
    assert env._episode_steps == 4 and env._episode_done is False
    assert "_fast_finite_cache" not in vars(env)
    np.testing.assert_array_equal(env._history, np.zeros((10, 57)))
    np.testing.assert_array_equal(env._delay_fifo[0], np.full(16, 0.25))
    np.testing.assert_array_equal(shaper.history, np.arange(10))
    env.data.qpos[:] = 99
    env._delay_fifo[0][:] = 99
    env._counts["policy_transitions"] = 7
    adapter.restore(env, shaper, snapshot)
    assert env._counts["policy_transitions"] == 7
    np.testing.assert_array_equal(env.data.qpos, np.arange(23))
    np.testing.assert_array_equal(env._delay_fifo[0], np.full(16, 0.25))
    env.model = object()
    with pytest.raises(ValueError, match="identity differs"):
        adapter.restore(env, shaper, snapshot)


def test_actor_weights_are_fixed_but_current_and_history_input_gradients_survive(torch):
    actor = adapter.actor_from_state_dict(synthetic_weights(torch))
    observation = torch.zeros((2, 627), requires_grad=True)
    output = actor(observation)
    assert output.shape == (2, 16)
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    output.sum().backward()
    assert bool(torch.isfinite(observation.grad).all())
    assert bool((observation.grad[:, :57] != 0).any())
    assert bool((observation.grad[:, 57:] != 0).any())
    assert all(parameter.grad is None for parameter in actor.parameters())
    torch.testing.assert_close(actor(torch.zeros(627)), output[0])
    with pytest.raises(RuntimeError, match="Missing key"):
        adapter.actor_from_state_dict({})


def test_loader_uses_only_bound_tensor_member_and_weights_only(torch, tmp_path, monkeypatch):
    payload = io.BytesIO()
    torch.save(synthetic_weights(torch), payload)
    path = tmp_path / "synthetic_policy.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("policy.pth", payload.getvalue())
        archive.writestr("data", "not a valid pickle or metadata document")
        archive.writestr("policy.optimizer.pth", b"must not be read")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    original = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    adapter.load_actor(path, expected_sha256=expected)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    with pytest.raises(ValueError, match="SHA256 differs"):
        adapter.load_actor(path)
    assert len(calls) == 1


def test_observation_matches_frozen_pure_formula_and_has_state_gradients(torch):
    spec = importlib.util.spec_from_file_location("interface52_fixture", ROOT / "training/d1_rl/interface.py")
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)
    state = np.linspace(-0.2, 0.2, 45)
    state[3:7] = [0.9, 0.1, -0.2, 0.3]
    command = np.array([0.2, 0., -0.3])
    action = np.linspace(-0.1, 0.1, 16)
    previous = np.array([0.1, -0.2, 0.3])
    expected, filtered = interface.build_observation(state[26:29], state[3:7], command,
                                                    state[7:23], state[29:45], action, previous)
    value = torch.tensor(state, requires_grad=True)
    actual, actual_filter = adapter.build_observation(value, command, action, previous)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=0, atol=1e-7)
    np.testing.assert_allclose(actual_filter.detach().numpy(), filtered, rtol=0, atol=1e-15)
    assert actual.dtype == torch.float32 and actual_filter.dtype == torch.float64
    actual.square().sum().backward()
    assert bool(torch.isfinite(value.grad).all()) and bool((value.grad != 0).any())
    assert torch.equal(actual[9:25][[3, 7, 11, 15]], torch.zeros(4))


def test_reference_and_oldest_first_history_match33_over_command_changes(torch):
    from tools33.reference_shape33 import ReferenceShaper
    shaper = ReferenceShaper()
    raw_history = torch.zeros((10, 57))
    cache = None
    for tick, yaw in enumerate((0., .05, .05, -.075, 0., .3)):
        current = torch.zeros(57)
        current[0] = tick
        current[8] = yaw
        raw = torch.cat((current, raw_history.flatten()))
        expected = shaper(raw.numpy())
        shaped, cache = adapter.reference_step(current, raw_history, cache)
        np.testing.assert_array_equal(shaped.numpy(), expected)
        np.testing.assert_array_equal(cache["history"].numpy(), shaper.history)
        np.testing.assert_array_equal(cache["raw_history"].numpy(), shaper.raw_history)
        assert cache["state"].item() == shaper.state
        old_history = raw_history
        raw_history = adapter.advance_history(raw_history, current)
        assert torch.equal(raw_history[:-1], old_history[1:])
        assert torch.equal(raw_history[-1], current)
    with pytest.raises(ValueError, match="raw history differs"):
        adapter.reference_step(current, torch.zeros((10, 57)), cache)


def test_reference_cache_retains_temporal_gradient(torch):
    current = torch.zeros(57)
    history = current.repeat(10, 1)
    _, cache = adapter.reference_step(current, history)
    yaw = torch.tensor(0.05, requires_grad=True)
    changed = torch.cat((current[:8], yaw.reshape(1), current[9:]))
    _, cache = adapter.reference_step(changed, history, cache)
    history = adapter.advance_history(history, changed)
    shaped, _ = adapter.reference_step(current, history, cache)
    shaped.sum().backward()
    assert yaw.grad is not None and yaw.grad > 0


def test_residual_uses_raw_command_preserves_legs_and_has_theta_gradient(torch):
    action = torch.zeros((2, 16))
    observation = torch.zeros((2, 57))
    command = torch.tensor([[0., 0., .3], [0., 0., -.3]])
    theta = torch.tensor(.1, requires_grad=True)
    result = adapter.residual_action(action, observation, command, theta)
    wheels = [3, 7, 11, 15]
    torch.testing.assert_close(result[0, wheels], torch.tensor([.1, -.1, .1, -.1]))
    torch.testing.assert_close(result[1, wheels], -result[0, wheels])
    legs = [index for index in range(16) if index not in wheels]
    assert torch.equal(result[:, legs], action[:, legs])
    result[0, 3].backward()
    torch.testing.assert_close(theta.grad, torch.tensor(1.))
    clipped = adapter.residual_action(torch.ones(16), torch.zeros(57), command[0], 2.)
    assert bool((clipped.abs() <= 1.).all())


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_fifo_executes_whole_clipped_action_after_exact_delay(torch, delay):
    queue = torch.zeros((delay, 16))
    expected = deque([torch.zeros(16) for _ in range(delay)])
    for scalar in (.2, -.3, .5, 2., -.6):
        issued = torch.full((16,), scalar)
        expected.append(issued.clamp(-1, 1))
        applied, queue = adapter.delay_step(issued, queue)
        assert torch.equal(applied, expected.popleft())
        assert queue.shape == (delay, 16)
