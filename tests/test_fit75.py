"""Synthetic75 label/objective contracts; no archived-data fitting or physics."""
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
    from tools75 import fit75
    return SimpleNamespace(np=np, torch=torch, fit=fit75)


@pytest.fixture
def labels(lib):
    np, fit = lib.np, lib.fit
    state = np.zeros((96, 45), dtype=np.float64)
    state[:, 0], state[:, 3], state[:, 2] = np.repeat(np.arange(24), 4), 1., .5
    epsilon = np.tile(fit.previous.baseline.EPSILONS, 24)
    direction = np.zeros(16)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    plus = (epsilon[:, None] * direction).astype(np.float32)
    minus = -plus
    return {'s': state, 'a0': np.zeros((96, 16), np.float32), 'a_plus': plus, 'a_minus': minus,
            'v': ((plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon[:, None])).astype(np.float32),
            'target': np.tile(np.linspace(-.01, .02, 48), 2), 'epsilon': epsilon,
            'seed': np.tile(np.repeat(fit.previous.baseline.LABEL_SEEDS, 12), 2),
            'tick': np.tile(np.repeat(fit.previous.baseline.LABEL_TICKS, 4), 8),
            'source_id': np.repeat([58, 62], 48), 'source_point_index': np.tile(np.repeat(np.arange(12), 4), 2),
            'target_body': np.column_stack((np.linspace(-.0004, .02, 96), np.linspace(.01, -.02, 96))),
            'body_channel_qualified': np.ones((96, 2), dtype=np.bool_),
            'target_world_vz': np.linspace(-.02, .02, 96, dtype=np.float64),
            'world_vz_channel_qualified': np.ones(96, dtype=np.bool_)}


def write_labels(lib, directory, labels, *, admitted=True):
    path = directory / 'world_vz_labels96.npz'
    lib.np.savez(path, **{'target_yaw' if key == 'target' else key: value for key, value in labels.items()})
    report = {'schema': 'd1-world-vz75-labels-v1', 'status': 'COMPLETE', 'verification_after': 'COMPLETE',
              'label_rows': 96, 'admission': {'admitted': admitted, 'qualified_states': 24, 'required_states': 24,
                                           'rows_dropped': 0},
              'label_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / 'report.json').write_text(json.dumps(report))


def test_loader_preserves96_raw_targets_and_original69_hashes(lib, labels, tmp_path):
    np, fit = lib.np, lib.fit
    write_labels(lib, tmp_path, labels)
    loaded = fit.load_labels75(tmp_path)
    np.testing.assert_array_equal(loaded['target_world_vz'], labels['target_world_vz'])
    assert loaded['target_world_vz'].dtype == np.float64
    assert np.float32(10 + labels['epsilon'][0] * labels['target_world_vz'][0]) == \
        np.float32(10 - labels['epsilon'][0] * labels['target_world_vz'][0])
    assert loaded['target_world_vz'][0] < 0 < loaded['target_world_vz'][-1]
    hashes = fit.label_hashes(loaded)
    original = fit.previous.label_hashes({key: labels[key] for key in fit.BASE_KEYS})
    assert all(hashes[key] == value for key, value in original.items())
    changed = {key: value.copy() for key, value in loaded.items()}
    changed['target_world_vz'][0] += .001
    new = fit.label_hashes(changed)
    assert new['combined_labels_sha256'] == hashes['combined_labels_sha256']
    assert new['world_vz_labels_sha256'] != hashes['world_vz_labels_sha256']
    assert new['combined75_labels_sha256'] != hashes['combined75_labels_sha256']


@pytest.mark.parametrize('corruption', ['f32', 'unknown', 'dropped', 'rejected', 'sha'])
def test_loader_rejects_invalid_or_unbound_labels(lib, labels, tmp_path, corruption):
    if corruption == 'f32':
        labels['target_world_vz'] = labels['target_world_vz'].astype(lib.np.float32)
    elif corruption == 'unknown':
        labels['world_vz_channel_qualified'][95] = False
    elif corruption == 'dropped':
        labels = {key: value[:-1] for key, value in labels.items()}
    write_labels(lib, tmp_path, labels, admitted=corruption != 'rejected')
    if corruption == 'sha':
        with (tmp_path / 'world_vz_labels96.npz').open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        lib.fit.load_labels75(tmp_path)


def test_world_vz_scalar96_mean_retains_mixed_parameter_gradient(lib):
    torch, fit = lib.torch, lib.fit

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[25] = 1.
            return state + (self.gain * (1 + state[:, 2]) * action[:, 3])[:, None] * basis

    state = torch.zeros((96, 45))
    state[:, 2] = torch.linspace(.1, 1., 96)
    state[:, 3], state[:, 4] = 2., 2.  # World-vz must not rotate this to body coordinates.
    direction = torch.zeros((96, 16))
    direction[:, 3] = 2.
    target = torch.linspace(-.2, .1, 96)
    labels = {'s': state, 'a0': torch.zeros((96, 16)), 'v': direction, 'target': torch.zeros(96),
              'target_body': torch.zeros((96, 2)), 'target_world_vz': target}
    model = Affine()
    response = fit.world_vz_jvp(model, labels, create_graph=True)
    factor = 2 * (1 + state[:, 2])
    torch.testing.assert_close(response, model.gain * factor)
    loss = fit.world_vz_loss(model, labels, create_graph=True)
    gradient, = torch.autograd.grad(loss, model.gain)
    scale = float(fit.frozen.PHYSICAL_SCALE[25])
    torch.testing.assert_close(loss, ((model.gain * factor - target) / scale).square().mean())
    torch.testing.assert_close(gradient, 2 * ((model.gain.detach() * factor - target) * factor).mean() / scale ** 2)
    assert gradient.abs() > 0 and model.gain.grad is None
    assert not fit.world_vz_jvp(model, labels, create_graph=False).requires_grad
    changed = dict(labels, target_world_vz=target.clone())
    changed['target_world_vz'][-1] += 2.
    assert not torch.equal(fit.world_vz_loss(model, changed), loss.detach())


def test_control_matches69_active_bitwise_without_zero_times_extra(lib):
    torch, fit = lib.torch, lib.fit
    parameter = torch.tensor(.37, requires_grad=True)
    state, yaw, body = parameter.square(), parameter.pow(3), 13 * parameter.pow(4)
    old = fit.previous.objective(state, yaw, body, 'active')
    control = fit.objective(state, yaw, body, torch.tensor(float('nan'), requires_grad=True), 'control')
    old_gradient, = torch.autograd.grad(old, parameter, retain_graph=True)
    actual_gradient, = torch.autograd.grad(control, parameter)
    assert torch.equal(old, control) and torch.equal(old_gradient, actual_gradient)
    assert torch.isfinite(control)


def test_active_changes_only_gamma_and_helpers_are_original(lib):
    torch, fit = lib.torch, lib.fit
    state, yaw, body, world = [torch.tensor(v, requires_grad=True) for v in (.1, .2, .3, .4)]
    loss = fit.objective(state, yaw, body, world, 'active')
    assert [float(v) for v in torch.autograd.grad(loss, (state, yaw, body, world))] == pytest.approx(
        [1., fit.FIXED_LAMBDA, fit.FIXED_LAMBDA, fit.FIXED_LAMBDA])
    assert fit.MODEL_SEEDS == (64001, 64002, 64003) and (fit.HORIZON, fit.UPDATES, fit.BATCH_SIZE) == (8, 1000, 64)
    assert fit.BODY_COEFFICIENT == fit.YAW_COEFFICIENT == {'control': fit.FIXED_LAMBDA, 'active': fit.FIXED_LAMBDA}
    assert fit.WORLD_COEFFICIENT == {'control': 0., 'active': fit.FIXED_LAMBDA}
    for name in ('initialize_model', 'yaw_jvp', 'auxiliary_loss', 'body_jvp', 'body_loss'):
        assert getattr(fit, name) is getattr(fit.previous, name)
    assert fit.paired is fit.previous.paired


def test_final_diagnostics_keep_all96_rows_targets_order_and_weights(lib, labels, tmp_path):
    torch, fit = lib.torch, lib.fit

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.125), requires_grad=False)

        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[25] = 1.
            return state + self.gain * action[:, 3:4] * basis

    model = Affine()
    before = fit.paired.mapping_sha256(model.state_dict())
    output = tmp_path / 'final75'
    result = fit.final_training_world_vz_diagnostics(model, labels, output)
    assert result['status'] == 'COMPLETE' and len(result['rows']) == 96
    assert result['used_for_training_manipulation_gate'] and not result['used_for_selection']
    assert result['rows'][0]['source'] == '58' and result['rows'][-1]['source'] == '62'
    assert [row['epsilon'] for row in result['rows']] == labels['epsilon'].tolist()
    assert result['rows'][0]['target_float64'] == labels['target_world_vz'][0]
    assert fit.paired.mapping_sha256(model.state_dict()) == before and model.gain.grad is None
    with lib.np.load(output / 'final_train_world_vz.npz', allow_pickle=False) as raw:
        lib.np.testing.assert_array_equal(raw['target_float64'], labels['target_world_vz'])
        assert raw['predicted_world_vz_jvp'].shape == (96,) and raw['target_training_float32'].dtype == lib.np.float32
    with pytest.raises(FileExistsError):
        fit.final_training_world_vz_diagnostics(model, labels, output)


def test_expired_budget_creates_no_fit(lib, monkeypatch, tmp_path):
    fit = lib.fit

    def forbidden(*_args, **_kwargs):
        raise AssertionError('expired admission must not initialize')

    monkeypatch.setattr(fit, 'initialize_model', forbidden)
    output = tmp_path / 'expired75'
    with pytest.raises(TimeoutError, match='before fit admission'):
        fit.fit_model({}, {}, 64001, 'control', output, cpu_deadline=0.)
    assert not output.exists()
