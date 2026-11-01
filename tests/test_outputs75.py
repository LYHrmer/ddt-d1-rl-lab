"""Analytic actual-action world-vz checks; no simulator or policy is used."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools75.outputs75 import EPSILONS, world_vz_comparison


@pytest.fixture(scope='module')
def lib():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / '.python-deps'))
    np, torch = pytest.importorskip('numpy'), pytest.importorskip('torch')
    from tools75 import outputs75
    return SimpleNamespace(np=np, torch=torch, mod=outputs75)


def _analytic(lib):
    np, torch = lib.np, lib.torch
    matrix = np.zeros((45, 16), np.float64)
    matrix[25, 3], matrix[24, 7], matrix[28, 11] = .3, 2., -1.

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('matrix', torch.tensor(matrix, dtype=torch.float64))

        def forward(self, state, action):
            return state + action @ self.matrix.T

    model = Affine().requires_grad_(False)
    state = np.zeros(45, np.float64)
    state[3:7] = [2 ** -.5, 2 ** -.5, 0., 0.]  # Tilt makes rotated-vz a different coordinate.
    state[25] = .125
    axis = np.zeros(16)
    axis[[3, 7, 11, 15]] = [1., -1., 1., -1.]
    center = np.full(16, .125, np.float32)
    points, grids = [], []
    for index in range(9):
        points.append({'seed': index, 'tick': 110, 'state': state.copy(), 'fifo': np.empty((0, 16))})
        grid = []
        for theta in [0., 0., *(sign * e for e in EPSILONS for sign in (1, -1))]:
            action = (center.astype(np.float64) + .1 * theta * axis).astype(np.float32)
            post = state + action.astype(np.float64) @ matrix.T
            observations = np.zeros((32, 627), np.float32)
            observations[:, 41:57] = action
            grid.append({'states': np.tile(post, (32, 1)), 'observations': observations})
        grids.append(grid)
    return SimpleNamespace(model=model, points=points, grids=grids)


def test_world_index25_and_recorded_theta_direction_are_not_rotated_or_rescaled(lib, tmp_path):
    data = _analytic(lib)
    refs = lib.mod.world_vz_references(data.points, data.grids)
    ledger = {}
    lib.mod.evaluate_world_vz(data.model, data.points, data.grids, tmp_path / 'vz', ledger, float('inf'))
    result = world_vz_comparison(ledger['points'], refs)
    assert result['valid_matches'] == result['resolved_count'] == 9
    assert result['passed_9_of_9'] and result['numeric_valid']
    assert result['normalized_mse'] < 1e-20 and result['error_scalar_count'] == 36
    assert all(row['state_index'] == 25 and row['rotation_applied'] is False for row in refs)
    for i, e in enumerate(EPSILONS):
        grid = data.grids[0]
        plus, minus = [grid[j]['observations'][0, 41:57].astype(lib.np.float64) for j in (2 + 2 * i, 3 + 2 * i)]
        actual_v = (plus - minus) / (2 * e)
        expected = .3 * actual_v[3]
        prediction = ledger['points'][0]['eps'][i]
        assert prediction['model_jvp'] == pytest.approx(expected)
        assert prediction['model_fd'] == pytest.approx(expected)
        assert prediction['model_jvp'] == pytest.approx(.03, rel=.01)  # Not the absolute-action gain .3.
        assert prediction['line_selfcheck']['autograd_agrees']
        assert prediction['endpoint_selfcheck']['autograd_agrees']


def test_world_reference_qualification_is_not_inherited_from_yaw(lib):
    data = _analytic(lib)
    for grid in data.grids:
        for branch in grid:
            branch['states'][:, 25] = .125
    refs = lib.mod.world_vz_references(data.points, data.grids)
    assert all(not row['resolved'] for row in refs)
    assert all(row['all_four_wheels_resolved'] for row in refs)


def test_world_reference_keeps_one_sided_and_actual_wheel_resolution_gates(lib):
    data = _analytic(lib)
    baseline = data.grids[0][0]['states'][0, 25]
    data.grids[0][2]['states'][0, 25] = baseline + EPSILONS[0] * .09
    refs = lib.mod.world_vz_references(data.points, data.grids)
    assert not refs[0]['resolved']
    data = _analytic(lib)
    for branch in data.grids[0][2:]:
        branch['observations'][0, 56] = data.grids[0][0]['observations'][0, 56]
    assert not lib.mod.world_vz_references(data.points, data.grids)[0]['resolved']


def _receipts():
    predictions = [{'seed': i, 'tick': 110, 'status': 'COMPLETE', 'eps': [
        {'epsilon': e, 'model_fd': .9, 'numeric_valid': True} for e in EPSILONS]} for i in range(9)]
    refs = [{'seed': i, 'tick': 110, 'resolved': True, 'eps': [
        {'epsilon': e, 'central64': 1.} for e in EPSILONS]} for i in range(9)]
    return predictions, refs


def test_unknown_and_one_bad_scale_are_not_hidden_by_mean_or_discarded():
    predictions, refs = _receipts()
    predictions[0]['eps'][2]['model_fd'] = -.9
    result = world_vz_comparison(predictions, refs)
    assert result['valid_matches'] == 8 and result['points'][0]['passed'] is False
    unknown = copy.deepcopy(refs)
    unknown[0]['resolved'] = False
    result = world_vz_comparison(predictions, unknown)
    assert result['points'][0]['passed'] is None and result['resolved_count'] == 8
    assert result['error_scalar_count'] == 36


@pytest.mark.parametrize('failure', ('magnitude', 'selfcheck', 'nonfinite', 'partial'))
def test_world_comparison_rejects_unqualified_or_invalid_predictions(failure):
    predictions, refs = _receipts()
    if failure == 'magnitude':
        predictions[0]['eps'][0]['model_fd'] = 1e-6
    elif failure == 'selfcheck':
        predictions[0]['eps'][0]['numeric_valid'] = False
    elif failure == 'nonfinite':
        predictions[0]['eps'][0]['model_fd'] = float('nan')
    else:
        predictions[0]['status'] = 'INCOMPLETE'
    if failure in ('nonfinite', 'partial'):
        with pytest.raises(ValueError):
            world_vz_comparison(predictions, refs)
    else:
        result = world_vz_comparison(predictions, refs)
        assert not result['numeric_valid'] and not result['passed_9_of_9']


def test_expired_evaluation_retains_incomplete_receipt_without_retry(lib, tmp_path):
    data = _analytic(lib)
    ledger = {}
    with pytest.raises(ValueError, match='CPU deadline'):
        lib.mod.evaluate_world_vz(data.model, data.points, data.grids, tmp_path / 'expired', ledger, -1.)
    saved = json.loads((tmp_path / 'expired' / 'result.json').read_text())
    assert saved['status'] == 'INCOMPLETE' and saved['completed_points'] == 0
