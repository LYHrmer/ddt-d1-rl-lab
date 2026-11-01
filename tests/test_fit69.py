"""Synthetic69 objective/label contracts; no original data or candidate fitting."""
from __future__ import annotations

import hashlib
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
    from tools69 import fit69
    return SimpleNamespace(np=np, torch=torch, fit=fit69)


@pytest.fixture
def labels(lib):
    np, fit = lib.np, lib.fit
    state = np.zeros((96, 45), dtype=np.float64)
    state[:, 0], state[:, 3] = np.repeat(np.arange(24), 4), 1.
    state[:, 2] = .5
    epsilon = np.tile(fit.baseline.EPSILONS, 24)
    direction = np.zeros(16)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    plus = (epsilon[:, None] * direction).astype(np.float32)
    minus = -plus
    return {'s': state, 'a0': np.zeros((96, 16), np.float32), 'a_plus': plus, 'a_minus': minus,
            'v': ((plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon[:, None])).astype(np.float32),
            'target': np.tile(np.linspace(-.01, .02, 48), 2), 'epsilon': epsilon,
            'seed': np.tile(np.repeat(fit.baseline.LABEL_SEEDS, 12), 2),
            'tick': np.tile(np.repeat(fit.baseline.LABEL_TICKS, 4), 8),
            'source_id': np.repeat([58, 62], 48), 'source_point_index': np.tile(np.repeat(np.arange(12), 4), 2),
            'target_body': np.column_stack((np.linspace(-.0004, .02, 96), np.linspace(.01, -.02, 96))),
            'body_channel_qualified': np.ones((96, 2), dtype=np.bool_)}


def write_admission(lib, directory, labels, *, admitted=True):
    np, fit = lib.np, lib.fit
    path = directory / 'body_labels96.npz'
    np.savez(path, **{'target_yaw' if key == 'target' else key: value for key, value in labels.items()})
    report = {'schema': 'd1-body-response68-result-v1', 'status': 'COMPLETE', 'verification_after': 'COMPLETE',
              'plan_sha256': fit.ADMISSION_PLAN_SHA256, 'completed_points': 24, 'label_rows': 96, 'states': 24,
              'admission': {'status': 'ADMITTED' if admitted else 'REJECTED_INCOMPLETE_BODY_RESPONSE_QUALIFICATION',
                            'admitted': admitted, 'qualified_state_channels': 48 if admitted else 47,
                            'required_state_channels': 48, 'rows_dropped': 0},
              'label_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'points': [{'source': source, 'seed': seed, 'tick': tick, 'status': 'COMPLETE',
                          'old_yaw_response_exact': True,
                          'channels': {name: {'stability': {'stable': True}} for name in ('vx', 'vy')}}
                         for source in ('58', '62') for seed in fit.baseline.LABEL_SEEDS for tick in fit.baseline.LABEL_TICKS]}
    (directory / 'report.json').write_text(json.dumps(report))


def test_loader_retains_all96_raw_body_labels_and_original64_yaw_identity(lib, labels, tmp_path):
    np, fit = lib.np, lib.fit
    write_admission(lib, tmp_path, labels)
    loaded = fit.load_labels68(tmp_path)
    assert loaded['target_body'].shape == (96, 2) and loaded['target_body'].dtype == np.float64
    np.testing.assert_array_equal(loaded['target_body'], labels['target_body'])
    assert np.float32(10. + labels['epsilon'][0] * labels['target_body'][0, 0]) == \
        np.float32(10. - labels['epsilon'][0] * labels['target_body'][0, 0])
    assert loaded['target_body'][0, 0] == -.0004 and loaded['target_body'][-1, 0] > 0
    old = fit.baseline.validate_labels(fit.yaw_labels(loaded), 'active')
    hashes = fit.label_hashes(loaded)
    assert hashes['labels_sha256'] == fit.paired.mapping_sha256(old)
    changed = {key: value.copy() for key, value in loaded.items()}
    changed['target_body'][0, 0] += .01
    assert fit.label_hashes(changed)['labels_sha256'] == hashes['labels_sha256']
    assert fit.label_hashes(changed)['body_labels_sha256'] != hashes['body_labels_sha256']
    assert fit.label_hashes(changed)['combined_labels_sha256'] != hashes['combined_labels_sha256']


@pytest.mark.parametrize('corruption', ['rejected', 'dropped', 'unknown', 'float32', 'wrong_source', 'sha'])
def test_loader_rejects_incomplete_qualification_or_label_provenance(lib, labels, tmp_path, corruption):
    if corruption == 'dropped':
        labels = {key: value[:-1] for key, value in labels.items()}
    elif corruption == 'unknown':
        labels['body_channel_qualified'][0, 1] = False
    elif corruption == 'float32':
        labels['target_body'] = labels['target_body'].astype(lib.np.float32)
    elif corruption == 'wrong_source':
        labels['source_id'][48:] = 60
    write_admission(lib, tmp_path, labels, admitted=corruption != 'rejected')
    if corruption == 'sha':
        with (tmp_path / 'body_labels96.npz').open('ab') as stream:
            stream.write(b'tampered')
    with pytest.raises(ValueError):
        lib.fit.load_labels68(tmp_path)


def test_body_global96x2_mean_has_correct_nonzero_mixed_parameter_gradient(lib):
    torch, fit = lib.torch, lib.fit

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[23], basis[24] = 1., 2.
            return state + (self.gain * (1 + state[:, 2]) * action[:, 3])[:, None] * basis

    model = Affine()
    state = torch.zeros((96, 45))
    state[:, 2], state[:, 3] = torch.linspace(.1, 1., 96), 2.  # Non-unit identity quaternion.
    direction = torch.zeros((96, 16))
    direction[:, 3] = 1.
    target = torch.stack((torch.linspace(-.3, .1, 96), torch.linspace(.1, -.2, 96)), dim=1)
    data = {'s': state, 'a0': torch.zeros((96, 16)), 'v': direction, 'target': torch.zeros(96), 'target_body': target}
    loss = fit.body_loss(model, data, create_graph=True)
    derivative, = torch.autograd.grad(loss, model.gain)
    factor = (1 + state[:, 2])[:, None] * state.new_tensor([1., 2.])
    scale = float(fit.frozen.PHYSICAL_SCALE[23])
    expected = ((model.gain.detach() * factor - target) / scale).square().mean()
    expected_gradient = 2 * ((model.gain.detach() * factor - target) * factor).mean() / scale ** 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(derivative, expected_gradient)
    assert derivative.abs() > 0 and fit.body_loss(model, data, create_graph=False).requires_grad is False
    changed = dict(data, target_body=target.clone())
    changed['target_body'][48:, 1] += 10.
    assert not torch.equal(fit.body_loss(model, changed, create_graph=False), loss.detach())


def test_body_jvp_differentiates_quaternion_normalized_inverse_rotation(lib):
    torch, fit = lib.torch, lib.fit

    class Rotation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.3))

        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[6] = 1.
            return state + (self.gain * action[:, 3])[:, None] * basis

    state = torch.zeros((96, 45))
    state[:, 3], state[:, 6], state[:, 23] = 2., .25, 1.
    action, direction = torch.zeros((96, 16)), torch.zeros((96, 16))
    action[:, 3], direction[:, 3] = .1, 1.
    data = {'s': state, 'a0': action, 'v': direction, 'target': torch.zeros(96), 'target_body': torch.zeros((96, 2))}
    model = Rotation()
    response = fit.body_jvp(model, data, create_graph=True)
    w, z, gain = 2., .25 + .3 * .1, .3
    expected = state.new_tensor([-4 * w ** 2 * z * gain / (w * w + z * z) ** 2,
                                -2 * w * gain * (w * w - z * z) / (w * w + z * z) ** 2])
    torch.testing.assert_close(response, expected.expand(96, 2), rtol=1e-5, atol=1e-7)
    derivative, = torch.autograd.grad(response.square().mean(), model.gain)
    assert derivative.abs() > 0


def test_control_objective_value_and_gradient_are_bitwise_original64(lib):
    torch, fit = lib.torch, lib.fit
    parameter = torch.tensor(.37, requires_grad=True)
    state, yaw = parameter.square(), parameter.pow(3)
    body = 1e12 * parameter.square()
    old = fit.baseline.objective(state, yaw, fit.FIXED_LAMBDA)
    actual = fit.objective(state, yaw, body, 'control')
    old_grad, = torch.autograd.grad(old, parameter, retain_graph=True)
    actual_grad, = torch.autograd.grad(actual, parameter, retain_graph=True)
    assert torch.equal(old, actual) and torch.equal(old_grad, actual_grad)
    candidate = fit.objective(state, yaw, body, 'active')
    torch.testing.assert_close(candidate, old + fit.FIXED_LAMBDA * body, rtol=0, atol=0)


def test_control_returns_original_graph_without_zero_times_body(lib, monkeypatch):
    torch, fit = lib.torch, lib.fit
    original = torch.tensor(1., requires_grad=True)
    monkeypatch.setattr(fit.baseline, 'objective', lambda *_args: original)
    result = fit.objective(original, original, torch.tensor(float('nan'), requires_grad=True), 'control')
    assert result is original and torch.isfinite(result)


def test_initialization_and_sampling_protocol_are_the_original64_protocol(lib):
    torch, fit = lib.torch, lib.fit
    assert fit.initialize_model is fit.baseline.initialize_model
    assert fit.MODEL_SEEDS == (64001, 64002, 64003) and fit.HORIZON == 8 and fit.UPDATES == 1000 and fit.BATCH_SIZE == 64
    assert fit.BODY_COEFFICIENT == {'control': 0., 'active': fit.FIXED_LAMBDA}
    first = fit.paired.sampled_window_rows(6828, updates=3, batch_size=64)
    torch.manual_seed(64003)
    torch.rand(12)
    assert torch.equal(first, fit.baseline.paired.sampled_window_rows(6828, updates=3, batch_size=64))


def test_expired_budget_creates_no_fit_and_performs_no_initialization(lib, monkeypatch, tmp_path):
    fit = lib.fit
    monkeypatch.setattr(fit.time, 'process_time', lambda: 10.)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('expired admission must not initialize')

    monkeypatch.setattr(fit, 'initialize_model', forbidden)
    output = tmp_path / 'expired69'
    with pytest.raises(TimeoutError, match='before fit admission'):
        fit.fit_model({}, {}, 64001, 'control', output, cpu_deadline=9.)
    assert not output.exists()
