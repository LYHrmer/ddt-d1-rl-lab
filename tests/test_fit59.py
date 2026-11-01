"""Synthetic59 tests: action quantization, mixed gradients, pairing and budgets.

No robot assets, physics or training loop is used. Full48 label fixtures contain
12 independent states ×4 epsilon scales; they are never counted as48 episodes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def lib():
    for path in (ROOT, ROOT / '.python-deps'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    from tools59 import fit59
    return SimpleNamespace(np=np, torch=torch, fit=fit59)


def data_fixture(lib, length=40):
    np = lib.np
    state = np.zeros((length, 45), dtype=np.float32)
    state[:, 3], state[:, 2] = 1., 1.
    action = np.zeros((length, 16), dtype=np.float32)
    action[:, 3] = .2
    done = np.zeros(length, dtype=np.bool_)
    done[-1] = True
    return {'s': state, 'a': action, 'target': state.copy(), 'episode': np.zeros(length, dtype=np.int64),
            'tick': np.arange(length, dtype=np.int64), 'done': done}


@pytest.fixture
def source58(lib, tmp_path):
    np, fit = lib.np, lib.fit
    output = tmp_path / 'synthetic58'
    output.mkdir()
    report = {'status': 'COMPLETE', 'counts': {'policy_transitions': 2520},
              'all_four_complete_baselines_exact': True, 'labels_admitted': 12, 'points': []}
    direction = np.zeros(16, dtype=np.float64)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    deltas = [0., 0., *(sign * epsilon for epsilon in fit.EPSILONS for sign in (1, -1))]
    for index, (seed, tick) in enumerate((seed, tick) for seed in fit.LABEL_SEEDS for tick in fit.LABEL_TICKS):
        state = np.zeros((1, 45), dtype=np.float64)
        state[0, 3], state[0, 2], state[0, 28] = 1., 1. + index / 20., 10.
        base = np.zeros((1, 16), dtype=np.float32)
        base[0, [3, 7, 11, 15]] = (.1, -.2, .3, -.4)
        rate = -.0004 if index == 0 else .003 + index * .001
        targets = []
        for order, delta in enumerate(deltas):
            action = np.clip(base.astype(np.float64) + delta * direction, -1., 1.).astype(np.float32)
            target = state.copy()
            target[0, 28] += delta * rate
            targets.append(target)
            np.savez(output / f'branch_{index:02d}_{order:02d}.npz',
                     s=state, applied_action=action.astype(np.float64), target=target)
        response = np.stack([(targets[2 + 2 * j] - targets[3 + 2 * j])[0] / (2 * epsilon)
                             for j, epsilon in enumerate(fit.EPSILONS)])
        np.savez(output / f'response_{index:02d}.npz', central_state45_fd=response)
        report['points'].append({'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'label_admitted': True,
                                 'branches': [{'status': 'COMPLETE', 'delta': delta} for delta in deltas]})
    (output / 'report.json').write_text(json.dumps(report))
    return output


def test_loader_preserves_small_signed_fd_before_float32_and_all_quantized_directions(lib, source58):
    np, fit = lib.np, lib.fit
    labels = fit.load_labels58(source58)
    assert labels['s'].shape == (48, 45) and labels['v'].dtype == np.float32
    assert labels['target'].dtype == np.float64
    assert list(zip(labels['seed'], labels['tick'], labels['epsilon'], strict=True)) == [
        (seed, tick, epsilon) for seed in fit.LABEL_SEEDS for tick in fit.LABEL_TICKS for epsilon in fit.EPSILONS]
    with np.load(source58 / 'branch_00_02.npz') as plus, np.load(source58 / 'branch_00_03.npz') as minus:
        # Casting physical endpoints first erases this negative action response.
        assert plus['target'][0, 28].astype(np.float32) == minus['target'][0, 28].astype(np.float32)
    assert labels['target'][0] < 0 and labels['target'][0].astype(np.float32) != 0
    expected = ((labels['a_plus'].astype(np.float64) - labels['a_minus'].astype(np.float64))
                / (2 * labels['epsilon'][:, None])).astype(np.float32)
    assert np.array_equal(labels['v'], expected)
    assert labels['v'][0, 3] != np.float32(1.)  # Actual quantization, not ideal d.
    assert np.all(labels['v'][:, [3, 11]] > 0) and np.all(labels['v'][:, [7, 15]] < 0)


def test_incomplete_or_duplicated_label_grid_is_rejected_instead_of_filtering(lib, source58):
    fit = lib.fit
    labels = fit.load_labels58(source58)
    labels['epsilon'][1] = labels['epsilon'][0]
    with pytest.raises(ValueError, match='no dropped/duplicate'):
        fit.validate_labels(labels)
    report = json.loads((source58 / 'report.json').read_text())
    report['points'].pop()
    (source58 / 'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='all12 points'):
        fit.load_labels58(source58)


def toy_model(lib):
    torch = lib.torch

    class StateDependentAffine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, state, action):
            basis = torch.nn.functional.one_hot(torch.tensor(28), 45).to(state)
            increment = self.gain * (1. + state[:, 2]) * action[:, 3]
            return state + increment[:, None] * basis

    return StateDependentAffine()


def test_jvp_mixed_parameter_gradient_is_nonzero_and_analytically_correct(lib):
    torch, fit = lib.torch, lib.fit
    model = toy_model(lib)
    state = torch.zeros((48, 45))
    state[:, 2] = torch.linspace(.1, 1., 48)
    direction = torch.zeros((48, 16))
    direction[:, 3] = 1.
    labels = {'s': state, 'a0': torch.zeros((48, 16)), 'v': direction, 'target': torch.zeros(48)}
    loss = fit.auxiliary_loss(model, labels, create_graph=True)
    derivative, = torch.autograd.grad(loss, model.gain)
    scale = float(fit.frozen.PHYSICAL_SCALE[28])
    expected = 2 * model.gain.detach() * ((1 + state[:, 2]) ** 2).mean() / scale ** 2
    assert derivative.abs() > 0
    torch.testing.assert_close(derivative, expected)
    assert fit.auxiliary_loss(model, labels, create_graph=False).requires_grad is False


def test_lambda_zero_preserves_exact_state_only_parameter_gradient(lib):
    torch, fit = lib.torch, lib.fit
    model = toy_model(lib)
    labels = {'s': torch.ones((48, 45)), 'a0': torch.zeros((48, 16)),
              'v': torch.ones((48, 16)), 'target': torch.zeros(48)}
    state_loss = (model.gain - 1.7).square()
    expected, = torch.autograd.grad(state_loss, model.gain, retain_graph=True)
    auxiliary = fit.auxiliary_loss(model, labels, create_graph=True)
    actual, = torch.autograd.grad(fit.objective(state_loss, auxiliary, 0.), model.gain)
    assert auxiliary.item() > 0 and torch.equal(actual, expected)


def test_initialization_and_sampler_are_paired_and_zero_residual_predictions_share_ridge(lib):
    torch, fit = lib.torch, lib.fit
    data = data_fixture(lib)
    first, ridge_a = fit.initialize_model(data, 59001)
    torch.rand(20)
    second, ridge_b = fit.initialize_model(data, 59001)
    third, ridge_c = fit.initialize_model(data, 59002)
    assert ridge_a == ridge_b == ridge_c
    assert fit.paired.mapping_sha256(first.state_dict()) == fit.paired.mapping_sha256(second.state_dict())
    assert fit.paired.mapping_sha256(first.state_dict()) != fit.paired.mapping_sha256(third.state_dict())
    state, action = torch.tensor(data['s'][:2]), torch.tensor(data['a'][:2])
    assert torch.equal(first(state, action), third(state, action))
    rows_a = fit.paired.sampled_window_rows(6828, updates=3, batch_size=64)
    torch.manual_seed(59003)
    torch.rand(20)
    rows_b = fit.paired.sampled_window_rows(6828, updates=3, batch_size=64)
    assert torch.equal(rows_a, rows_b)


def test_calibration_weights_the_last44_windows_and_never_updates_parameters(lib, source58, monkeypatch):
    np, torch, fit = lib.np, lib.torch, lib.fit
    model, data = toy_model(lib), data_fixture(lib)
    labels = fit.load_labels58(source58)
    windows = np.tile(np.arange(32, dtype=np.int64), (6828, 1))
    monkeypatch.setattr(fit.paired, 'common_window_indices', lambda _data: windows)
    # Deliberately distinct full/last-batch losses expose an unweighted mean.
    monkeypatch.setattr(fit.frozen, 'group_loss', lambda prediction, _target: prediction.new_tensor(float(len(prediction))))
    before = model.gain.detach().clone()
    result = fit.calibrate_weight(model, data, labels, deadline=1e12)
    assert result['S0'] == pytest.approx((106 * 64 * 64 + 44 * 44) / 6828)
    assert result['batches'] == 107 and result['last_batch_windows'] == 44
    assert result['weighted_window_count'] == 6828
    assert result['lambda'] == result['S0'] / result['R0'] > 0
    assert torch.equal(before, model.gain) and model.gain.grad is None


def test_expired_admission_creates_no_fit_or_numeric_work(lib, monkeypatch, tmp_path):
    fit = lib.fit
    monkeypatch.setattr(fit.time, 'process_time', lambda: 10.)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('expired budget must not initialize or calibrate')

    monkeypatch.setattr(fit, 'initialize_model', forbidden)
    monkeypatch.setattr(fit, 'calibrate_weight', forbidden)
    output = tmp_path / 'expired'
    with pytest.raises(TimeoutError, match='before fit admission'):
        fit.fit_model({}, {}, 59001, 'state_only', output, cpu_deadline=9.)
    assert not output.exists()
