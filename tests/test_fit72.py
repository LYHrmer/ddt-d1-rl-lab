"""Synthetic72 protocol/weight contracts; no candidate fit or archived data."""
from __future__ import annotations

import ast
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
    pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    from tools72 import fit72
    return SimpleNamespace(torch=torch, fit=fit72)


def test_control_value_and_parameter_gradient_are_bitwise69_active(lib):
    torch, fit = lib.torch, lib.fit
    parameter = torch.tensor(.37, requires_grad=True)
    state, yaw, body = parameter.square(), parameter.pow(3), 13. * parameter.pow(4)
    reference = fit.previous.objective(state, yaw, body, 'active')
    control = fit.objective(state, yaw, body, 'control')
    reference_gradient, = torch.autograd.grad(reference, parameter, retain_graph=True)
    control_gradient, = torch.autograd.grad(control, parameter)
    assert torch.equal(reference, control) and torch.equal(reference_gradient, control_gradient)


def test_control_returns69_active_graph_directly(lib, monkeypatch):
    fit = lib.fit
    sentinel, calls = object(), []

    def original(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(fit, 'previous', SimpleNamespace(objective=original))
    assert fit.objective('state', 'yaw', 'body', 'control') is sentinel
    assert calls == [('state', 'yaw', 'body', 'active')]


@pytest.mark.parametrize('arm,multiplier', [('control', 1), ('active', 2)])
def test_only_yaw_derivative_coefficient_changes_and_body_stays_nonzero(lib, arm, multiplier):
    torch, fit = lib.torch, lib.fit
    state, yaw, body = (torch.tensor(value, requires_grad=True) for value in (.125, -.75, 1.25))
    actual = fit.objective(state, yaw, body, arm)
    expected = (state + multiplier * fit.FIXED_LAMBDA * yaw) + fit.FIXED_LAMBDA * body
    assert torch.equal(actual, expected)
    gradients = torch.autograd.grad(actual, (state, yaw, body))
    assert [gradient.item() for gradient in gradients] == pytest.approx(
        [1., multiplier * fit.FIXED_LAMBDA, fit.FIXED_LAMBDA])
    assert fit.YAW_COEFFICIENT[arm] == multiplier * .08980082414438752
    assert fit.BODY_COEFFICIENT[arm] == .08980082414438752


def test_original_helpers_labels_initialization_and_sampling_are_reused(lib):
    torch, fit = lib.torch, lib.fit
    for name in ('initialize_model', 'load_labels68', 'validate_labels', 'labels32', 'label_hashes',
                 'yaw_labels', 'yaw_jvp', 'auxiliary_loss', 'body_jvp', 'body_loss'):
        assert getattr(fit, name) is getattr(fit.previous, name)
    assert fit.MODEL_SEEDS == (64001, 64002, 64003)
    assert (fit.HORIZON, fit.UPDATES, fit.BATCH_SIZE) == (8, 1000, 64)
    assert fit.paired is fit.previous.paired and fit.frozen is fit.previous.frozen
    first = fit.paired.sampled_window_rows(6828, updates=2, batch_size=64)
    torch.manual_seed(64003)
    torch.rand(5)
    assert torch.equal(first, fit.previous.paired.sampled_window_rows(6828, updates=2, batch_size=64))


def test_update_computation_through_adam_step_matches69_source():
    """Guard the frozen loss ordering, zero_grad, backward, clip and step seam."""
    def update_prefix(path):
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_model')
        loop = next(node for node in ast.walk(function)
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'update')
        prefix = []
        for node in loop.body:
            prefix.append(ast.dump(node, include_attributes=False))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and \
                    isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'step':
                break
        return prefix

    assert update_prefix(ROOT / 'tools72/fit72.py') == update_prefix(ROOT / 'tools69/fit69.py')


def test_expired_admission_creates_nothing_and_does_not_initialize(lib, monkeypatch, tmp_path):
    fit = lib.fit

    def forbidden(*_args, **_kwargs):
        raise AssertionError('expired admission must not initialize')

    monkeypatch.setattr(fit, 'initialize_model', forbidden)
    output = tmp_path / 'expired72'
    with pytest.raises(TimeoutError, match='before fit admission'):
        fit.fit_model({}, {}, 64001, 'active', output, cpu_deadline=0.)
    assert not output.exists()


def test_input_failure_preserves_partial_and_existing_directory_cannot_resume(lib, monkeypatch, tmp_path):
    fit = lib.fit

    def invalid(_data):
        raise ValueError('synthetic input failure')

    monkeypatch.setattr(fit, 'frozen', SimpleNamespace(PHYSICAL_SCALE=[.3] * 45, validate_data=invalid))
    monkeypatch.setattr(fit, 'torch', SimpleNamespace(set_num_threads=lambda _n: None,
                        get_num_interop_threads=lambda: 1, use_deterministic_algorithms=lambda _flag: None))
    output = tmp_path / 'partial72'
    with pytest.raises(ValueError, match='synthetic input failure'):
        fit.fit_model({}, {}, 64001, 'active', output, cpu_deadline=1e9)
    receipt = json.loads((output / 'partial_receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_evaluation']
    assert receipt['updates'] == receipt['updates_attempted'] == 0 and not receipt['final_weights_written']
    assert receipt['fixed_lambda'] == receipt['effective_lambda'] == .17960164828877504
    assert receipt['fixed_beta'] == receipt['body_coefficient'] == receipt['reference_lambda'] == .08980082414438752
    assert receipt['control_reference'] == 'complete69 active at the same seed'
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        fit.fit_model({}, {}, 64001, 'active', output, cpu_deadline=1e9)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
