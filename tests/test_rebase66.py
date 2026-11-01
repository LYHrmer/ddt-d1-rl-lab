"""Asset-free66 translation tests; no environment, checkpoints or physics."""
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
    from tools65 import diagnose65
    from tools66 import rebase66
    return SimpleNamespace(np=np, torch=torch, numeric=numerics54, prior=diagnose65, mod=rebase66)


def fixture(lib):
    """Analytic affine free trajectory; the scalar cost remains nonlinear."""
    np, torch, numeric = lib.np, lib.torch, lib.numeric
    base = torch.zeros((32, 45), dtype=torch.float64)
    base[:, 2], base[:, 3], base[:, 23], base[:, 28] = .5, 1., .3, .2
    slope = torch.zeros_like(base)
    slope[:, 23] = torch.linspace(.1, .4, 32, dtype=torch.float64)
    slope[:, 28] = -.07
    action0 = torch.full((32, 16), .05, dtype=torch.float64)
    action_slope = torch.arange(1, 33, dtype=torch.float64)[:, None].expand(32, 16) * .001
    commands = np.tile(np.array([.1, 0., .15]), (33, 1))
    start = {'commands': commands, 'last': np.full(16, .02, np.float32), 'nominal_height': .5}
    first = torch.tensor(start['last'][None], dtype=torch.float64)
    command = torch.tensor(commands[:32], dtype=torch.float64)
    calls = []

    def free(theta):
        calls.append(float(theta.detach()))
        states = base + theta * slope
        actions = action0 + theta * action_slope
        previous = torch.cat((first, actions[:-1]))
        parts = numeric.cost_components(states, command, actions, previous, .5)
        costs = numeric.cost(states, command, actions, previous, .5)
        return costs.mean(), {'states': states, 'actions': actions, 'applied_actions': actions,
                              'costs': costs, 'cost_components': parts}

    target, actual = base.clone(), action0.clone()
    target[:, 23], target[:, 28] = .05, .05
    actual += .01
    return SimpleNamespace(free=free, start=start, target=target, actual=actual, calls=calls,
                           state_slope=slope, action_slope=action_slope, base=base, action0=action0)


def test_fixed_origin_is_computed_once_and_shared_across_nonzero_theta(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    rebased, fixed = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    assert data.calls == [0.]
    before = {key: tensor.clone() for key, tensor in fixed.items()}
    observed = []
    for theta in (0., .001, -.001):
        _, trace = rebased(torch.tensor(theta, dtype=torch.float64))
        observed.append(trace)
        torch.testing.assert_close(trace['cost_eval_states'], data.target + theta * data.state_slope,
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(trace['cost_eval_actions'], data.actual + theta * data.action_slope,
                                   rtol=1e-12, atol=1e-12)
    assert data.calls == [0., 0., .001, -.001]
    assert all(torch.equal(fixed[key], value) for key, value in before.items())
    assert not torch.equal(observed[1]['cost_eval_states'], observed[2]['cost_eval_states'])
    assert not torch.equal(observed[1]['costs'], observed[2]['costs'])


def test_theta0_cost_exact_previous_shift_and_no_external_quaternion_normalization(lib):
    torch, numeric, mod = lib.torch, lib.numeric, lib.mod
    data = fixture(lib)
    # Raw non-unit quaternion makes an accidental outer normalization visible.
    data.target[:, 3:7] = torch.tensor([.8, .1, .2, .1], dtype=torch.float64)
    rebased, _ = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    value, trace = rebased(torch.tensor(0., dtype=torch.float64))
    previous = torch.cat((torch.tensor(data.start['last'][None], dtype=torch.float64), data.actual[:-1]))
    command = torch.tensor(data.start['commands'][:32], dtype=torch.float64)
    physical = numeric.cost(data.target, command, data.actual, previous, .5)
    assert torch.equal(trace['cost_eval_states'], data.target)
    assert torch.equal(trace['cost_eval_previous_actions'], previous)
    assert torch.equal(trace['costs'], physical) and torch.equal(value, physical.mean())
    assert torch.equal(trace['states'], data.base)
    assert torch.equal(trace['actions'], data.action0)


def test_real_future_values_cannot_change_free_model_actor_trajectory(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    a, _ = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    shifted_target, shifted_action = data.target.clone(), data.actual.clone()
    shifted_target[:, 23] += .4
    shifted_action += .2
    b, _ = mod.make_rebased_rollout(data.free, data.start, shifted_target, shifted_action)
    value_a, trace_a = a(torch.tensor(.003, dtype=torch.float64))
    value_b, trace_b = b(torch.tensor(.003, dtype=torch.float64))
    assert all(torch.equal(trace_a[key], trace_b[key]) for key in ('states', 'actions', 'applied_actions', 'free_costs'))
    assert value_a != value_b


def test_rebased_cost_gradient_has_analytic_value_and_nonzero_fixed_offset_fd(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    rebased, _ = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value, _ = rebased(theta)
    derivative, = torch.autograd.grad(value, theta)
    previous = torch.cat((torch.tensor(data.start['last'][None], dtype=torch.float64), data.actual[:-1]))
    prior_slope = torch.cat((torch.zeros((1, 16), dtype=torch.float64), data.action_slope[:-1]))
    analytic = (2 * (data.target[:, 23] - .1) * data.state_slope[:, 23] / .2 ** 2
                + 2 * (data.target[:, 28] - .15) * data.state_slope[:, 28] / .2 ** 2
                + .2 * ((data.actual - previous) * (data.action_slope - prior_slope)).mean(-1)).mean()
    plus, _ = rebased(torch.tensor(.001, dtype=torch.float64))
    minus, _ = rebased(torch.tensor(-.001, dtype=torch.float64))
    numerical = (plus - minus) / .002
    assert abs(float(analytic)) > 1e-5
    torch.testing.assert_close(derivative, analytic, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(numerical, analytic, rtol=1e-9, atol=1e-10)


def test_all_state_action_previous_jvp_and_fd_match_free_sensitivity(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    rebased, fixed = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    arrays = {}
    admission = mod.sensitivity_admission(rebased, fixed, arrays=arrays)
    assert admission['status'] == 'COMPLETE' and all(row['passed'] for row in admission['parity'].values())
    np.testing.assert_array_equal(arrays['sensitivity_jvp_states'], data.state_slope.numpy())
    np.testing.assert_array_equal(arrays['sensitivity_jvp_actions'], data.action_slope.numpy())
    np.testing.assert_array_equal(arrays['sensitivity_jvp_cost_eval_previous_actions'][0], np.zeros(16))
    np.testing.assert_array_equal(arrays['sensitivity_jvp_cost_eval_previous_actions'][1:], data.action_slope[:-1].numpy())
    assert arrays['sensitivity_fd_cost_eval_states'].shape == (32, 45)


def test_admission_rejects_detaching_current_cost_state_graph(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    rebased, fixed = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)

    def broken(theta):
        scalar, trace = rebased(theta)
        trace['cost_eval_states'] = trace['cost_eval_states'].detach()
        return scalar, trace

    with pytest.raises(ValueError, match='changed free states JVP'):
        mod.sensitivity_admission(broken, fixed)
    assert data.target.dtype == torch.float64


def test_baseline_inputs_are_cloned_detached_and_all_prefix_components_are_retained(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    data.target.requires_grad_(True)
    rebased, fixed = mod.make_rebased_rollout(data.free, data.start, data.target, data.actual)
    assert all(not tensor.requires_grad for tensor in fixed.values())
    baseline = fixed['baseline_states'].clone()
    with torch.no_grad():
        data.target[:, 23] += 10.
        data.actual += .2
    result, arrays = lib.prior.decompose_model(rebased)
    assert len(result['prefixes']) == 32
    assert all(set(row['components']) == set(mod.COMPONENTS) for row in result['prefixes'])
    assert result['all_prefix_total_selfchecks'] and result['all_prefix_component_selfchecks']
    assert torch.equal(torch.from_numpy(arrays['theta0_cost_eval_states']), baseline)


def test_unknown_total_and_component_references_stay_unknown_independently(lib):
    mod = lib.mod
    valid = {'gradient': 1., 'autograd_agrees': True, 'direction_identifiable': True}
    result = {'prefixes': [{'horizon': h, **valid, 'components': {name: dict(valid) for name in mod.COMPONENTS}}
                           for h in range(1, 33)]}
    references = [{'horizon': h, 'resolved': h != 32, 'reference_gradient': 1.} for h in range(1, 33)]
    physical = {'status': 'COMPLETE', 'prefixes': references,
                'component_prefixes': {name: copy.deepcopy(references) for name in mod.COMPONENTS}}
    physical['component_prefixes']['yaw'][0]['resolved'] = False
    annotated = mod.annotate_references(result, physical)
    assert annotated['prefixes'][0]['valid_direction_match'] is True
    assert annotated['prefixes'][0]['components']['yaw']['valid_direction_match'] is None
    assert annotated['prefixes'][0]['components']['vx']['valid_direction_match'] is True
    assert annotated['prefixes'][-1]['sign_matches_reference'] is None
    assert annotated['prefixes'][-1]['valid_direction_match'] is None


def test_incomplete_baseline_cannot_form_an_oracle_diagnostic(lib):
    data = fixture(lib)
    with pytest.raises(ValueError, match='complete F64 baseline inputs'):
        lib.mod.make_rebased_rollout(data.free, data.start, data.target[:31], data.actual)
