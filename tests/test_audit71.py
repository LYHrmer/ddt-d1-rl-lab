"""Small analytic71 contracts; no archived data, optimizer or physics."""
from __future__ import annotations

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
    from tools71 import audit71
    return SimpleNamespace(np=np, torch=torch, audit=audit71)


def test_uneven_tail_weights_loss_and_f32_gradients_by_samples(lib):
    torch, audit = lib.torch, lib.audit
    parameter = torch.nn.Parameter(torch.tensor(.5))
    parameter.grad = torch.tensor(73.)
    original_grad = parameter.grad
    values, visited = torch.arange(1, 6, dtype=torch.float32), []

    def loss_at(begin, end):
        visited.append((begin, end))
        return (parameter * values[begin:end]).square().mean()

    loss, gradient, sizes = audit.weighted_batch_gradients((parameter,), 5, loss_at, 1e9, batch_size=2)
    assert sizes == [2, 2, 1] and visited == [(0, 2), (2, 4), (4, 5)]
    assert loss == pytest.approx(2.75) and gradient.tolist() == pytest.approx([11.])
    assert gradient.dtype == torch.float64 and gradient.grad_fn is None
    assert parameter.item() == .5 and parameter.grad is original_grad and parameter.grad.item() == 73.


def test_pairwise_sign_zero_norm_and_weighted_projections(lib):
    torch, audit = lib.torch, lib.audit
    gradients = {name: torch.tensor(value, dtype=torch.float64) for name, value in {
        'state': [1., 0.], 'yaw': [-2., 0.], 'body': [0., 3.], 'anchor': [0., 0.]}.items()}
    control, control_gradient = audit.gradient_geometry(gradients, 'control')
    active, active_gradient = audit.gradient_geometry(gradients, 'active')
    assert active['unweighted_pairwise']['state']['yaw']['dot'] == -2.
    assert active['unweighted_pairwise']['state']['yaw']['cosine'] == -1.
    assert active['unweighted_pairwise']['state']['body']['cosine'] == 0.
    assert active['unweighted_pairwise']['state']['anchor']['cosine'] is None
    assert active['objective_projections']['anchor']['cosine_status'] == 'ZERO_NORM_UNDEFINED'
    weight = audit.fit.FIXED_LAMBDA
    torch.testing.assert_close(control_gradient, torch.tensor([1 - 2 * weight, 0.], dtype=torch.float64))
    torch.testing.assert_close(active_gradient, torch.tensor([1 - 2 * weight, 3 * weight], dtype=torch.float64))
    assert control['objective_projections']['body']['dot'] == 0.
    assert control['weighted_component_norms']['body'] == 0.
    assert control['weighted_component_pairwise']['body']['state']['cosine'] is None
    assert active['weighted_component_pairwise']['anchor']['yaw']['cosine'] is None
    assert active['weighted_component_pairwise']['state']['yaw']['dot'] == pytest.approx(-2 * weight)
    assert active['weighted_component_pairwise']['body']['body']['dot'] == pytest.approx(9 * weight ** 2)
    assert active['objective_projections']['body']['dot'] == pytest.approx(9 * weight)
    assert active['objective_projections']['body']['directional_derivative_along_negative_objective_gradient'] == pytest.approx(-9 * weight)


def test_anchor_includes_last_center_with_equal_one_over24_weight(lib):
    np, torch, audit = lib.np, lib.torch, lib.audit
    predicted = torch.zeros((24, 45), requires_grad=True)
    targets = np.zeros((24, 45), dtype=np.float64)
    targets[-1, 28] = 3.
    loss = audit.yaw_anchor_loss(predicted, targets)
    gradient, = torch.autograd.grad(loss, predicted)
    scale = float(audit.frozen.PHYSICAL_SCALE[28])
    assert float(loss.detach()) == pytest.approx((3 / scale) ** 2 / 24, rel=1e-6)
    assert gradient[-1, 28].item() == pytest.approx(-6 / (24 * scale ** 2), rel=1e-6)
    assert torch.count_nonzero(gradient).item() == 1
    with pytest.raises(ValueError, match='all24'):
        audit.yaw_anchor_loss(predicted[:-1], targets[:-1])


def test_raw_f64_value_error_survives_f32_objective_rounding(lib):
    np, torch, audit = lib.np, lib.torch, lib.audit
    targets = np.zeros((24, 45), dtype=np.float64)
    targets[:, 3], targets[:, 28] = 1., 1.
    predicted = torch.as_tensor(targets, dtype=torch.float32).clone()
    targets[:, 28] += 5e-9
    report, arrays = audit.value_diagnostics(predicted, targets)
    assert audit.yaw_anchor_loss(predicted, targets).item() == 0.
    assert report['yaw28_raw_rmse'] == pytest.approx(5e-9, rel=1e-6)
    assert arrays['center_error64'].dtype == np.float64
    assert np.all(arrays['center_error64'][:, 28] < 0)
    assert set(report['frozen_groups']) == set(audit.frozen.GROUPS)
    assert len(report['state45_raw_rmse']) == 45
    assert report['by_source']['58']['centers'] == report['by_source']['62']['centers'] == 12
    targets[12:, 28] += .2
    grouped, _ = audit.value_diagnostics(predicted, targets)
    assert grouped['by_source']['58']['yaw28_raw_rmse'] < 1e-8
    assert grouped['by_source']['62']['yaw28_raw_rmse'] == pytest.approx(.2, abs=1e-8)


@pytest.mark.parametrize('raises', [False, True])
def test_model_guard_preserves_weights_grad_objects_flags_modes_and_rng(lib, raises):
    torch, audit = lib.torch, lib.audit

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.residual = torch.nn.Linear(1, 1, bias=False)
            self.register_buffer('fixed', torch.tensor(4.))

    model = Tiny()
    model.train()
    parameter = next(model.parameters())
    parameter.requires_grad_(False)
    parameter.grad = torch.full_like(parameter, 19.)
    previous_grad, previous_weight = parameter.grad, parameter.detach().clone()
    initial_rng = torch.get_rng_state().clone()

    def work():
        with audit.unchanged_model(model):
            assert parameter.requires_grad
            audit.loss_gradient(model.residual(torch.ones(2, 1)).square().mean(), (parameter,))
            if raises:
                raise RuntimeError('synthetic stop')

    if raises:
        with pytest.raises(RuntimeError, match='synthetic stop'):
            work()
    else:
        work()
    assert not parameter.requires_grad and model.training
    assert parameter.grad is previous_grad and torch.all(parameter.grad == 19.)
    assert torch.equal(parameter, previous_weight) and torch.equal(torch.get_rng_state(), initial_rng)


def test_unused_parameter_has_explicit_zero_gradient_without_grad_mutation(lib):
    torch, audit = lib.torch, lib.audit
    used, unused = torch.nn.Parameter(torch.tensor(2.)), torch.nn.Parameter(torch.tensor(9.))
    missing = {}
    result = audit.loss_gradient(used.square(), (used, unused), unused=missing, parameter_names=['used', 'unused'])
    assert result.tolist() == [4., 0.]
    assert missing == {'indices': [1], 'names': ['unused']}
    assert used.grad is unused.grad is None
    with pytest.raises(ValueError, match='retained parameter graph'):
        audit.loss_gradient(used.square().detach(), (used, unused))
    with pytest.raises(ValueError, match='disconnected from all'):
        audit.loss_gradient(torch.tensor(1., requires_grad=True), (used, unused))
    with pytest.raises(ValueError, match='state/anchor'):
        audit.loss_gradient(used.square(), (used, unused), require_all_used=True)


def test_expired_audit_leaves_partial_progress_without_touching_model(lib):
    progress, artifacts = {}, {}
    with pytest.raises(Exception, match='CPU'):
        lib.audit.audit_model(object(), {}, {}, None, 0., arm='active', artifacts=artifacts, progress=progress)
    assert progress == {'status': 'INCOMPLETE', 'stage': 'admission', 'arm': 'active', 'losses': {}, 'unused_parameters': {}}
    assert artifacts == {}


def test_interrupted_state_second_chunk_retains_first_completed_prefix(lib):
    torch, audit = lib.torch, lib.audit
    parameter = torch.nn.Parameter(torch.tensor(.5))
    partial, arrays = {}, {}

    def loss_at(begin, end):
        if begin:
            raise RuntimeError('second chunk stop')
        return (parameter * torch.arange(1, end + 1, dtype=torch.float32)).square().mean()

    with pytest.raises(RuntimeError, match='second chunk stop'):
        audit.weighted_batch_gradients((parameter,), 5, loss_at, 1e9, batch_size=2,
                                       partial=partial, arrays=arrays, parameter_names=['gain'])
    assert partial['status'] == 'INCOMPLETE' and partial['completed_windows'] == 2
    assert partial['completed_batches'] == 1 and partial['batch_sizes'] == [2]
    assert partial['weighted_loss_sum'] == pytest.approx(1.25)
    assert partial['partial_global_mean_loss'] == pytest.approx(.25)
    assert partial['accumulation_denominator_windows'] == 5
    assert partial['unused_parameters'] == {'indices': [], 'names': []}
    assert arrays['gradient_state_completed_prefix64'].tolist() == pytest.approx([1.])
    assert parameter.grad is None and parameter.item() == .5
