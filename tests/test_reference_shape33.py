import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools33.reference_shape33 import ReferenceShaper, YAW_COLUMNS  # noqa: E402
from tools33.run_reference_shape33 import Runtime  # noqa: E402


def test_step_reversal_stop_and_exact_past_effective_history():
    shaper = ReferenceShaper()
    raw_history, effective_history = [0.]*10, [np.float32(0)]*10
    state = 0.
    for value in [0.]*12 + [.05]*30 + [-.05]*30 + [0.]*30:
        obs = np.linspace(-1., 1., 627, dtype=np.float32)
        obs[YAW_COLUMNS] = np.array([value, *raw_history], dtype=np.float32)
        before = obs.copy()
        target = float(np.float32(value)*np.float32(1.5))
        state = math.exp(-.2)*state + (1-math.exp(-.2))*target
        result = shaper(obs)
        assert result[8] == np.float32(state)
        assert np.array_equal(result[YAW_COLUMNS[1:]], effective_history)
        keep = np.ones(627, bool); keep[YAW_COLUMNS] = False
        assert np.array_equal(result[keep], before[keep])
        assert np.array_equal(obs, before)
        raw_history = raw_history[1:]+[value]
        effective_history = effective_history[1:]+[result[8]]


def test_constant_prehistory_has_no_startup_kick_and_keeps_encoded_bound():
    obs = np.zeros(627, np.float32); obs[YAW_COLUMNS] = .24
    assert np.array_equal(ReferenceShaper()(obs)[YAW_COLUMNS], np.full(11, .25, np.float32))
    obs[YAW_COLUMNS] = 0
    assert np.array_equal(ReferenceShaper()(obs), obs)


@pytest.mark.parametrize('value', [np.zeros(626, np.float32), np.zeros(627), np.full(627, np.nan, np.float32)])
def test_rejects_invalid_input(value):
    with pytest.raises(ValueError):
        ReferenceShaper()(value)


def test_rejects_wrong_order_or_skipped_history():
    s = ReferenceShaper(); obs = np.zeros(627, np.float32)
    s(obs); obs[8] = .05; s(obs)
    with pytest.raises(ValueError, match='previous ten'):
        s(obs)
    with pytest.raises(ValueError, match='constant prehistory'):
        ReferenceShaper()(obs)


def test_runtime_one_inference_and_episode_state_reset():
    seen = []
    def predict(obs):
        seen.append(obs.copy())
        return np.zeros(16, np.float32)
    rt = Runtime(SimpleNamespace(backend={}, make_env=lambda *a: None, predict=predict))
    rt.make_env('turn_stop', [], 3, 1001)
    obs = np.zeros(627, np.float32)
    rt.predict(obs); obs[8] = .05; rt.predict(obs)
    assert len(seen) == rt.attempts == rt.returns == 2
    assert 0 < seen[-1][8] < .075
    rt.rows.clear()
    rt.make_env('turn_stop', [], 3, 1002)
    obs[:] = 0; rt.predict(obs)
    assert seen[-1][8] == 0
