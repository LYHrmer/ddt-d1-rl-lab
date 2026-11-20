"""Synthetic terrain dynamics and archive gates. No simulator or optimizer run."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tools91 import model91 as m


def states(count, dtype=torch.float32):
    value = torch.zeros(count, 45, dtype=dtype)
    value[:, 2] = 0.5
    value[:, 3] = 1.
    return value


def test_road_height_slope_and_right_derivative_at_both_corners():
    state = states(4, torch.float64)
    state[:, 0] = torch.tensor([0.6, 2.6, 1.0, 1.0], dtype=state.dtype)
    state.requires_grad_()
    grade = math.tan(math.radians(4.))
    values = m.terrain_features(state, torch.tensor([1, 1, 2, 0]))
    expected = [[0.25 * grade - 0.5, grade, -0.5, grade, -0.5, 0.],
                [2 * grade - 0.5, 0., 2 * grade - 0.5, 0., 1.75 * grade - 0.5, grade],
                [-0.65 * grade - 0.5, -grade, -0.40 * grade - 0.5, -grade,
                 -0.15 * grade - 0.5, -grade],
                [-0.5, 0., -0.5, 0., -0.5, 0.]]
    torch.testing.assert_close(values, torch.tensor(expected, dtype=state.dtype))
    values[:, 2].sum().backward()
    torch.testing.assert_close(state.grad[:, 0], torch.tensor([grade, 0., -grade, 0.], dtype=state.dtype))
    torch.testing.assert_close(state.grad[:, 2], -torch.ones(4, dtype=state.dtype))


def test_mask_leaves_applied_targets_and_relative_terrain_context_intact():
    state = states(2)
    state[:, list(m.d52.ZERO_FEATURES)] = torch.arange(12, dtype=state.dtype).reshape(2, 6)
    before = state.clone()
    applied = torch.arange(32, dtype=state.dtype).reshape(2, 16) / 32.
    features = m.input_features(state, applied, torch.tensor([0, 1]))
    assert features.shape == (2, 67)
    assert (features[:, list(m.d52.ZERO_FEATURES)] == 0).all()
    torch.testing.assert_close(features[:, 45:61], applied)
    assert features[1, 61] > features[0, 61]
    torch.testing.assert_close(state, before)
    with pytest.raises(ValueError, match="terrain"):
        m.input_features(state, applied, torch.tensor([-1, 1]))


def test_fixed_ridge_propagates_gradients_via_terrain_at_predicted_x():
    model = m.TerrainDynamics91().requires_grad_(False)
    # First applied target advances x. The central terrain height at the next
    # predicted x affects vx on the following step, exposing a two-step path.
    model.ridge_coef[45, 0] = 1.
    model.ridge_coef[63, 23] = 1.
    start = states(1)
    start[0, 0] = 1.
    actions = torch.zeros(1, 2, 16, requires_grad=True)
    predicted = m.rollout_steps(model, start, actions, torch.tensor([1]))
    predicted[0, -1, 23].backward()
    assert actions.grad[0, 0, 0].item() == pytest.approx(0.1 * m.UP_GRADE * 0.3)
    assert actions.grad[0, 1, 0].item() == 0.
    torch.testing.assert_close(predicted[:, :, 3:7].norm(dim=-1), torch.ones(1, 2))


def test_legacy_interface_ignores_map_and_preserves_applied_action_gradient():
    model = m.LegacyDynamics75().requires_grad_(False)
    model.plant.ridge_coef[45, 23] = 1.
    batch = states(2)
    applied = torch.full((2, 16), 0.2, requires_grad=True)
    prediction = model.step(batch, applied, torch.tensor([0, 2]))
    torch.testing.assert_close(prediction[0], prediction[1], rtol=0., atol=0.)
    prediction[:, 23].sum().backward()
    torch.testing.assert_close(applied.grad[:, 0], torch.full((2,), 0.3))


def training_fixture():
    state = states(18).numpy()
    state[:, 0] = 0.6 + np.arange(len(state)) * 0.005
    return {"s": state[:-1], "target": state[1:], "a": np.zeros((17, 16)), "episode": np.r_[[4]*8, [5]*9],
            "tick": np.r_[np.arange(2, 10), np.arange(10, 19)],
            "done": np.r_[[False]*7, True, [False]*8, True], "terrain": np.r_[[0]*8, [1]*9]}


def test_windows_use_whole_episodes_and_preserve_the_original_clocks():
    data = training_fixture()
    arrays = m.training_arrays(data)
    windows = m.d52.window_indices(arrays)
    assert windows[:, 0].tolist() == [0, 8, 9]
    assert arrays["tick"].tolist() == [*range(8), *range(9)]
    np.testing.assert_array_equal(arrays["source_tick"], data["tick"])
    data["terrain"][10] = 2
    with pytest.raises(ValueError, match="terrain changed"):
        m.training_arrays(data)
    data = training_fixture()
    data["tick"][10] += 1
    with pytest.raises(ValueError, match="contiguous"):
        m.training_arrays(data)
    data = training_fixture()
    data["done"][-1] = False
    with pytest.raises(ValueError, match="incomplete"):
        m.training_arrays(data)


def test_refuse_tampered_legacy_report_and_reusing_output_directory(tmp_path):
    folder = tmp_path / m.OLD_FOLDER
    folder.mkdir(parents=True)
    (folder / "report.json").write_text('{"status":"COMPLETE"}')
    with pytest.raises(ValueError, match="source report identity"):
        m.load_old_models(tmp_path)
    with pytest.raises(FileExistsError):
        m.fit_models({}, tmp_path)
    with pytest.raises(FileNotFoundError):
        m.load_new_models(tmp_path)
