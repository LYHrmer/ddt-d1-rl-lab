"""Asset-free first-body/H1 tests, using an analytic physical-action fixture."""
from __future__ import annotations

import copy
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
    from tools54 import numerics54
    from tools69 import outputs69
    return SimpleNamespace(np=np, torch=torch, mod=outputs69, numeric=numerics54)


def fixture(lib):
    np, torch, mod = lib.np, lib.torch, lib.mod
    matrix = np.zeros((45, 16), dtype=np.float64)
    matrix[23, 3], matrix[24, 7], matrix[28, 11] = .3, -.2, .2

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('matrix', torch.tensor(matrix, dtype=torch.float64))

        def forward(self, state, action):
            return state + action @ self.matrix.T

    model = Affine().requires_grad_(False)
    initial = np.zeros(45, dtype=np.float64)
    initial[2], initial[3], initial[23], initial[24], initial[28] = .5, 1., .12, .03, .1
    direction = np.zeros(16)
    direction[[3, 7, 11, 15]] = [1., -1., 1., -1.]
    points, grids = [], []
    for index in range(9):
        points.append({'seed': index, 'tick': 110, 'state': initial.copy(), 'fifo': np.empty((0, 16)),
                       'commands': np.tile([.1, 0., .2], (33, 1)), 'last': np.zeros(16, np.float32),
                       'nominal_height': .5})
        branches = []
        for theta in [0., 0., *(sign * ep for ep in mod.EPSILONS for sign in (1, -1))]:
            action = (theta * direction).astype(np.float32)
            post = initial + action.astype(np.float64) @ matrix.T
            observation = np.zeros((32, 627), np.float32)
            observation[:, 41:57] = action
            branches.append({'states': np.tile(post, (32, 1)), 'observations': observation})
        grids.append(branches)
    refs = mod.head_references(points, grids)
    costs = [{'seed': p['seed'], 'tick': p['tick'], 'status': 'COMPLETE',
              **mod.selfcheck(r['cost_h1']['gradient'], r['cost_h1']['gradient'])}
             for p, r in zip(points, refs, strict=True)]
    return SimpleNamespace(model=model, points=points, grids=grids, refs=refs, costs=costs)


def test_real_action_body_fd_and_two_selfchecks_recover_analytic_response(lib, tmp_path):
    data = fixture(lib)
    ledger = {}
    lib.mod.evaluate_heads(data.model, data.points, data.grids, data.costs, tmp_path / 'heads', ledger, float('inf'))
    assert ledger['status'] == 'COMPLETE' and ledger['completed_points'] == 9
    result = lib.mod.head_comparison(ledger['points'], data.refs)
    assert result['body_valid_matches'] == result['body_resolved_count'] == 18
    assert result['cost_h1_valid_matches'] == result['cost_h1_resolved_count'] == 9
    assert result['numeric_valid'] and result['all_references_resolved']
    assert result['body_error_scalar_count'] == 72 and result['body_normalized_mse'] < 1e-24
    for row in ledger['points']:
        for channel in ('vx', 'vy'):
            assert len(row['body'][channel]['eps']) == 4
            assert all(x['line_selfcheck']['autograd_agrees'] and x['endpoint_selfcheck']['autograd_agrees']
                       for x in row['body'][channel]['eps'])


def test_body_channel_reference_is_not_inherited_from_yaw_or_other_body_channel(lib):
    data = fixture(lib)
    # Remove only the physical vx response. Other coordinate qualifications stand.
    for grid in data.grids:
        for branch in grid:
            branch['states'][:, 23] = .12
    refs = lib.mod.head_references(data.points, data.grids)
    assert all(not r['body']['vx']['resolved'] and r['body']['vy']['resolved'] for r in refs)


def test_one_bad_epsilon_or_unknown_reference_cannot_be_hidden_by_mean(lib, tmp_path):
    data = fixture(lib)
    ledger = {}
    lib.mod.evaluate_heads(data.model, data.points, data.grids, data.costs, tmp_path / 'heads', ledger, float('inf'))
    bad = copy.deepcopy(ledger['points'])
    bad[0]['body']['vx']['eps'][0]['model_fd'] *= -1
    result = lib.mod.head_comparison(bad, data.refs)
    assert result['body_valid_matches'] == 17 and not result['body_passed_18_of18']
    assert result['points'][0]['body']['vx']['passed'] is False
    unknown = copy.deepcopy(data.refs)
    unknown[0]['body']['vx']['resolved'] = False
    result = lib.mod.head_comparison(bad, unknown)
    assert result['body_resolved_count'] == 17
    assert result['points'][0]['body']['vx']['passed'] is None
    assert not result['all_references_resolved'] and result['body_error_scalar_count'] == 72


def test_scalar_stability_requires_whole_theta_grid_and_one_sided_agreement(lib):
    mod = lib.mod
    pairs = [(2 * ep, -2 * ep) for ep in mod.EPSILONS]
    assert mod.scalar_reference(0., pairs, True)['resolved']
    pairs[-1] = (4 * mod.EPSILONS[-1], 0.)
    assert not mod.scalar_reference(0., pairs, True)['resolved']
    with pytest.raises(ValueError, match='all four'):
        mod.scalar_reference(0., pairs[:3], True)


def test_h1_free_prediction_never_uses_physical_grid_and_retains_F64_derivative(lib, tmp_path, monkeypatch):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    calls = []

    def imaginary(model, actor, start, theta, *, return_trace, horizon):
        assert return_trace and horizon == 1 and theta.dtype == torch.float64
        calls.append(float(theta.detach()))
        cost = 2. + .0123 * theta + .02 * theta ** 2
        parts = torch.stack((cost, *(cost * 0. for _ in range(5))))[None]
        return cost, {'states': torch.tensor(start['state'][None], dtype=torch.float64),
                      'costs': cost[None], 'cost_components': parts}

    monkeypatch.setattr(lib.numeric, 'imaginary', imaginary)
    ledger = {}
    mod.evaluate_h1_costs(data.model, object(), data.points, tmp_path / 'h1', ledger, float('inf'))
    assert ledger['status'] == 'COMPLETE' and len(calls) == 27
    assert set(calls) == {0., .001, -.001}
    for row in ledger['points']:
        assert row['gradient'] == .0123 and row['autograd_agrees'] and row['direction_identifiable']
        assert row['physical_perturbed_data_used'] is False


def test_expired_head_budget_retains_incomplete_receipt_without_inference(lib, tmp_path):
    data = fixture(lib)
    ledger = {}
    with pytest.raises(ValueError, match='CPU deadline'):
        lib.mod.evaluate_heads(data.model, data.points, data.grids, data.costs, tmp_path / 'expired', ledger, -1.)
    assert ledger['status'] == 'INCOMPLETE' and ledger['completed_points'] == 0
    assert (tmp_path / 'expired' / 'result.json').is_file()
