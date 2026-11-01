"""Synthetic65 accounting/derivative checks; no robot, checkpoint or physics."""
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
    from tools54 import numerics54
    from tools65 import diagnose65
    return SimpleNamespace(np=np, torch=torch, mod=diagnose65, numeric=numerics54)


def test_prefix_fd_keeps_time_order_and_correct_mean_denominator(lib):
    np, mod = lib.np, lib.mod
    slope = np.arange(1., 33.)[:, None] * np.array([[1., -2., 0., .5, .25, -.25]])
    base = np.full((32, 6), 2.)
    epsilon = .000125
    result = mod.prefix_differences(base, base + epsilon * slope, base - epsilon * slope, epsilon)
    np.testing.assert_allclose(result['per_step_central'], slope, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result['cumulative_central'], slope.cumsum(0), rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(result['prefix_mean_central'], slope.cumsum(0) / np.arange(1, 33)[:, None],
                               rtol=1e-10, atol=1e-9)
    assert result['prefix_mean_central'].shape == (32, 6)
    # The H32 denominator is32, not32*6 or the six registered summary horizons.
    assert result['prefix_mean_central'][-1, 0] == pytest.approx(16.5)


def test_roundoff_bound_scales_with_epsilon_and_is_not_a_stability_floor(lib):
    np, mod = lib.np, lib.mod
    parts = np.full((32, 6), .3)
    a = mod.fd_roundoff_bounds(parts, parts, mod.EPSILONS[0])
    b = mod.fd_roundoff_bounds(parts, parts, mod.EPSILONS[1])
    for key in a:
        np.testing.assert_array_equal(a[key], 2 * b[key])
    np.testing.assert_array_equal(a['prefix_mean'], a['cumulative'] / np.arange(1, 33))
    tiny = [{f'prefix_mean_{name}': np.full(32, 1e-6) for name in ('central', 'forward', 'backward')}
            for _ in mod.EPSILONS]
    assert all(not row['resolved'] for row in mod.prefix_stability(tiny, action_resolved=True))


def test_each_prefix_rechecks_all_four_scales_and_keeps_unknown_h32(lib):
    np, mod = lib.np, lib.mod
    grids = [{f'prefix_mean_{name}': np.full(32, .01) for name in ('central', 'forward', 'backward')}
             for _ in mod.EPSILONS]
    grids[0]['prefix_mean_forward'][-1] = -.01
    rows = mod.prefix_stability(grids, action_resolved=True)
    assert len(rows) == 32 and all(row['resolved'] for row in rows[:-1])
    assert rows[-1]['resolved'] is False
    assert all(row['changes64_decision'] is False for row in rows)
    assert [row['horizon'] for row in rows if row['registered_summary_horizon']] == list(mod.HORIZONS)
    assert all(not row['resolved'] for row in mod.prefix_stability(grids, action_resolved=False))


def toy_rollout(lib, *, cubic=0.):
    """Several cancelling component derivatives and an actual recurrent graph."""
    torch = lib.torch

    def rollout(theta):
        state, components = theta.new_tensor(.2), []
        for tick in range(32):
            action = theta * (1. + .1 * state)
            state = .9 * state + action
            first = .5 + state + cubic * theta ** 3
            second = .5 - .7 * state - cubic * theta ** 3
            rest = torch.stack((.1 + .01 * theta * (tick + 1), theta * 0 + .2,
                                .2 + theta * (1. if tick < 3 else -.2), .1 + action.square()))
            components.append(torch.cat((torch.stack((first, second)), rest)))
        parts = torch.stack(components)
        costs = parts[:, 0]
        for index in range(1, 6):
            costs = costs + parts[:, index]
        return costs.mean(), {'cost_components': parts, 'costs': costs}

    return rollout


def test_jvp_retains_recurrent_theta_path_and_restores_total_at_all32_prefixes(lib):
    np, torch, mod = lib.np, lib.torch, lib.mod
    rollout = toy_rollout(lib)
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    scalar, trace = rollout(theta)
    scalar_gradient, = torch.autograd.grad(scalar, theta, retain_graph=True)
    expected_derivatives = np.array([
        [float(torch.autograd.grad(trace['cost_components'][tick, column], theta, retain_graph=True)[0])
         for column in range(6)] for tick in range(32)])
    result, arrays = mod.decompose_model(rollout)
    assert len(result['prefixes']) == 32
    assert result['gradient'] == float(scalar_gradient)
    np.testing.assert_allclose(arrays['per_step_component_derivative'], expected_derivatives, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(arrays['per_step_component_derivative'].sum(-1), arrays['per_step_cost_derivative'],
                               rtol=1e-12, atol=1e-12)
    assert arrays['per_step_component_derivative'][31, 0] > arrays['per_step_component_derivative'][0, 0]
    assert np.count_nonzero(arrays['per_step_component_derivative'][:, 3]) == 0
    assert result['all_prefix_total_selfchecks']
    assert result['all_prefix_component_selfchecks']


@pytest.mark.parametrize('changed', ['trace', 'gradient'])
def test64_identity_failure_stops_before_jvp_and_preserves_zero_trace(lib, changed):
    torch, mod = lib.torch, lib.mod
    rollout = toy_rollout(lib)
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value, trace = rollout(theta)
    derivative, = torch.autograd.grad(value, theta)
    saved_trace = {key: tensor.detach().numpy().copy() for key, tensor in trace.items()}
    saved = {'cost': float(value.detach()), 'gradient': float(derivative)}
    if changed == 'trace':
        saved_trace['costs'][5] += .01
    else:
        saved['gradient'] += .01
    calls, buffer = [], {}

    def counted(theta):
        calls.append(1)
        return rollout(theta)

    with pytest.raises(ValueError, match='differs from64'):
        mod.decompose_model(counted, expected_trace=saved_trace, expected_receipt=saved, arrays=buffer)
    assert len(calls) == 1
    assert buffer['theta0_cost_components'].shape == (32, 6)


def test_failed_component_selfcheck_is_retained_even_when_total_selfcheck_passes(lib):
    result, arrays = lib.mod.decompose_model(toy_rollout(lib, cubic=10000000.))
    assert result['all_prefix_total_selfchecks']
    assert not result['all_prefix_component_selfchecks']
    assert len(result['prefixes']) == 32 and arrays['per_step_component_derivative'].shape == (32, 6)
    assert any(not row['components']['vx']['autograd_agrees'] for row in result['prefixes'])


def physical_fixture(lib, directory):
    np, torch, mod, numeric = lib.np, lib.torch, lib.mod, lib.numeric
    commands = np.zeros((33, 3), dtype=np.float64)
    commands[:, 0] = np.linspace(-.1, .2, 33)
    commands[:, 2] = np.linspace(.2, -.1, 33)
    start = {'fifo': np.empty((0, 16), np.float32), 'commands': commands,
             'last': np.full(16, .02, np.float32), 'nominal_height': .5, 'seed': 64301, 'tick': 110}
    sequence = [0., 0., *(sign * eps for eps in mod.EPSILONS for sign in (1, -1))]
    rows, reference_parts = [], []
    for order, theta in enumerate(sequence):
        state = np.zeros((32, 45), dtype=np.float64)
        state[:, 2], state[:, 3], state[:, 23] = .5, 1., .1
        state[:, 28] = .15 + theta * np.arange(1, 33)
        observation = np.zeros((32, 627), dtype=np.float32)
        observation[:, 41:57] = .001 * np.arange(1, 33)[:, None]
        observation[:, [44, 48, 52, 56]] += np.float32(theta) * np.array([1., -1., 1., -1.], np.float32)
        action = torch.tensor(observation[:, 41:57], dtype=torch.float64)
        previous = torch.cat((torch.tensor(start['last'][None], dtype=torch.float64), action[:-1]))
        states, command = torch.tensor(state), torch.tensor(commands[:32])
        costs = numeric.cost(states, command, action, previous, .5)
        parts = numeric.cost_components(states, command, action, previous, .5).numpy()
        reference_parts.append(parts)
        row = {'status': 'COMPLETE', 'steps': 32, 'theta': theta, 'unsafe_steps': [],
               'cost64': float(costs.mean()), 'per_step_cost64': costs.tolist()}
        rows.append(row)
        (directory / f'grid_00_{order:02d}.json').write_text(json.dumps(row))
        np.savez(directory / f'grid_00_{order:02d}.npz', states=state, observations=observation,
                 rewards=np.zeros(32))
    gradients = []
    for index, eps in enumerate(mod.EPSILONS):
        plus, minus = rows[2 + 2 * index]['cost64'], rows[3 + 2 * index]['cost64']
        gradients.append({'epsilon': eps, 'central64': (plus - minus) / (2 * eps),
                          'forward64': (plus - rows[0]['cost64']) / eps,
                          'backward64': (rows[0]['cost64'] - minus) / eps})
    return start, {'status': 'COMPLETE', 'branches': rows, 'gradients': gradients,
                   'all_four_wheels_resolved': True}, np.stack(reference_parts)


def test_physical_cost_uses_actual_action_previous_action_and_current_interval_command(lib, tmp_path):
    np, mod = lib.np, lib.mod
    start, old, expected_parts = physical_fixture(lib, tmp_path)
    row, arrays = mod.physical_decomposition(start, 0, old, tmp_path)
    assert row['status'] == 'COMPLETE'
    np.testing.assert_array_equal(arrays['branch_components'], expected_parts)
    np.testing.assert_array_equal(arrays['commands'], start['commands'][:32])
    assert arrays['branch_components'][0, 0, 5] > arrays['branch_components'][0, 1, 5]
    assert len(row['prefixes']) == 32 and set(row['component_prefixes']) == set(mod.COMPONENTS)
    assert all(len(prefixes) == 32 for prefixes in row['component_prefixes'].values())
    # Zero height/lateral responses remain unidentifiable, even if total is stable.
    assert all(not prefix['resolved'] for prefix in row['component_prefixes']['height'])
    assert all(item['cost_per_step_exact'] and item['cost_mean_exact'] for item in row['parity'])


def test_partial_physical_branch_is_not_a_diagnostic_candidate(lib, tmp_path):
    start, old, _ = physical_fixture(lib, tmp_path)
    old['branches'][4]['status'] = 'TERMINATED'
    (tmp_path / 'grid_00_04.json').write_text(json.dumps(old['branches'][4]))
    with pytest.raises(ValueError, match='partial/different64 branch'):
        lib.mod.physical_decomposition(start, 0, old, tmp_path)
