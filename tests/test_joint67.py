"""Synthetic67 direction/timing/chain-rule checks; no robot or physical steps."""
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
    from tools67 import joint67
    return SimpleNamespace(np=np, torch=torch, numeric=numerics54, prior=diagnose65, mod=joint67)


def fixture(lib):
    np, torch = lib.np, lib.torch
    a = np.eye(45)
    a[23, 23], a[28, 28] = .9, .8
    b = np.zeros((45, 16))
    b[23, 7], b[28, 3], b[28, 11] = .2, .3, -.1

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('a', torch.tensor(a, dtype=torch.float64))
            self.register_buffer('b', torch.tensor(b, dtype=torch.float64))

        def forward(self, state, action):
            return state @ self.a.T + action @ self.b.T

    model = Affine().requires_grad_(False)
    initial = np.zeros(45)
    initial[2], initial[3], initial[23], initial[28] = .5, 1., .1, .1
    start = {'state': initial, 'last': np.full(16, .02, np.float32), 'fifo': np.empty((0, 16), np.float32),
             'commands': np.tile(np.array([.1, 0., .2]), (33, 1)), 'nominal_height': .5}
    sequence = [0., 0., *(sign * eps for eps in lib.mod.EPSILONS for sign in (1, -1))]
    branches = []
    for theta in sequence:
        state, states = initial.copy(), []
        observation = np.zeros((32, 627), dtype=np.float32)
        observation[:, 41:57] = .05 + np.arange(32)[:, None] * .001
        observation[:, [44, 48, 52, 56]] += np.float32(theta) * np.array([1., -1., 1., -1.], np.float32)
        for action in observation[:, 41:57]:
            state = state @ a.T + action.astype(np.float64) @ b.T
            states.append(state.copy())
        branches.append({'states': np.stack(states), 'observations': observation, 'rewards': np.zeros(32)})
    return SimpleNamespace(model=model, start=start, branches=branches, a=a, b=b)


def physical_costs(lib, data):
    np, torch, prior, numeric = lib.np, lib.torch, lib.prior, lib.numeric
    costs, parts = [], []
    for branch in data.branches:
        state = torch.tensor(branch['states'], dtype=torch.float64)
        action = torch.tensor(branch['observations'][:, 41:57], dtype=torch.float64)
        previous = torch.cat((torch.tensor(data.start['last'][None], dtype=torch.float64), action[:-1]))
        command = torch.tensor(data.start['commands'][:32], dtype=torch.float64)
        costs.append(numeric.cost(state, command, action, previous, .5).numpy())
        parts.append(numeric.cost_components(state, command, action, previous, .5).numpy())
    arrays = {'branch_costs': np.stack(costs), 'branch_components': np.stack(parts)}
    total_fds, part_fds = [], []
    for ep, epsilon in enumerate(prior.EPSILONS):
        total = prior.prefix_differences(costs[0], costs[2 + 2 * ep], costs[3 + 2 * ep], epsilon)
        component = prior.prefix_differences(parts[0], parts[2 + 2 * ep], parts[3 + 2 * ep], epsilon)
        total_fds.append(total)
        part_fds.append(component)
        arrays.update({f'epsilon_{ep}_total_{key}': value for key, value in total.items()})
        arrays.update({f'epsilon_{ep}_component_{key}': value for key, value in component.items()})
    row = {'status': 'COMPLETE', 'prefixes': prior.prefix_stability(total_fds, action_resolved=True),
           'component_prefixes': {name: prior.prefix_stability(
               [{key: value[:, column] for key, value in fd.items()} for fd in part_fds], action_resolved=True)
               for column, name in enumerate(prior.COMPONENTS)}}
    return row, arrays


def test_prestate_and_previous_are_shifted_separately_on_each_physical_branch(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    directions = mod.build_directions(data.start, data.branches)
    assert not directions['delta_pre_state'][:, 0].any()
    assert not directions['delta_previous_action'][:, 0].any()
    np.testing.assert_array_equal(directions['delta_pre_state'][:, 1:], directions['delta_next_state'][:, :-1])
    np.testing.assert_array_equal(directions['delta_previous_action'][:, 1:], directions['delta_action'][:, :-1])
    assert np.count_nonzero(directions['delta_pre_state'][:, 1]) > 0
    assert directions['delta_action'].shape == (4, 32, 16)
    assert directions['midpoint_drift_next_state'].shape == (4, 32, 45)


def test_state_difference_does_not_round_physical_endpoints_to_float32(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    data.branches[2]['states'][0, 28] = 1. + 2. ** -32
    data.branches[3]['states'][0, 28] = 1. - 2. ** -32
    assert np.float32(data.branches[2]['states'][0, 28]) == np.float32(data.branches[3]['states'][0, 28])
    directions = mod.build_directions(data.start, data.branches)
    assert directions['delta_next_state'][0, 0, 28] == 2. ** -32 / mod.EPSILONS[0]
    plus = data.branches[2]['observations'][:, 41:57].astype(np.float64)
    minus = data.branches[3]['observations'][:, 41:57].astype(np.float64)
    np.testing.assert_array_equal(directions['delta_action'][0], (plus - minus) / (2 * mod.EPSILONS[0]))


def test_affine_joint_response_identifies_model_A_and_B_and_first_B_direction(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    directions = mod.build_directions(data.start, data.branches)
    row, arrays = mod.joint_response(data.model, directions)
    assert row['status'] == 'COMPLETE' and row['later_physical_A_B_separately_identifiable'] is False
    for ep in range(4):
        expected_a = directions['delta_pre_state'][ep] @ data.a.T
        expected_b = directions['delta_action'][ep] @ data.b.T
        np.testing.assert_allclose(arrays[f'epsilon_{ep}_A_delta_state'], expected_a, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays[f'epsilon_{ep}_B_delta_action'], expected_b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(arrays[f'epsilon_{ep}_joint_jvp'], directions['delta_next_state'][ep], rtol=1e-8, atol=1e-9)
        assert arrays[f'epsilon_{ep}_line_selfcheck'].all()
        assert not arrays[f'epsilon_{ep}_A_delta_state'][0].any()
        assert row['epsilons'][ep]['state_line_selfcheck_channels'] == 32 * 45


def test_nonlinear_line_fd_failure_is_saved_without_aborting_remaining_scales(lib):
    torch, mod = lib.torch, lib.mod
    data = fixture(lib)
    directions = mod.build_directions(data.start, data.branches)
    directions['zero_action'][:, 3] = 0.

    class Curved(torch.nn.Module):
        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[28] = 1.
            return state + 10000. * action[:, 3:4] ** 3 * basis

    row, arrays = mod.joint_response(Curved(), directions)
    assert row['status'] == 'COMPLETE' and len(row['epsilons']) == 4
    assert all(not item['all_state_line_selfchecks'] for item in row['epsilons'])
    assert all(arrays[f'epsilon_{ep}_line_selfcheck'].shape == (32, 45) for ep in range(4))
    assert not arrays['epsilon_0_line_selfcheck'][0, 28]


def test_cost_projection_at_real_poststate_has_correct_action_previous_and_residual_identity(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    directions = mod.build_directions(data.start, data.branches)
    physical_row, physical_arrays = physical_costs(lib, data)
    _, response = mod.joint_response(data.model, directions)
    # Inject a known joint-state error; action/previous remain the real tangents.
    for ep in range(4):
        response[f'epsilon_{ep}_joint_jvp'][:, 28] += .1
    row, arrays = mod.project_cost_response(data.start, directions, response, physical_row, physical_arrays)
    assert row['status'] == 'COMPLETE' and row['cost_baseline_exact']
    assert len(row['prefixes']) == 32
    for ep in range(4):
        state_error = arrays[f'epsilon_{ep}_state_error_component_per_step']
        model = arrays[f'epsilon_{ep}_model_component_per_step']
        actual = arrays[f'epsilon_{ep}_actual_component_per_step']
        np.testing.assert_allclose(model - actual, state_error, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(state_error[:, 1], 2 * (directions['zero_next_state'][:, 28] - .2) * .1 / .2 ** 2,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(state_error[:, 5], np.zeros(32))
        np.testing.assert_array_equal(model[:, 5], actual[:, 5])
        assert arrays[f'epsilon_{ep}_actual_component_prefix_mean_minus_physical_fd'].shape == (32, 6)


@pytest.mark.parametrize(('change', 'expected'), [
    ('reference', 'REFERENCE_UNRESOLVED'), ('numeric', 'NUMERIC_UNRESOLVED'),
    ('actual', 'ACTUAL_COST_TANGENT_UNRESOLVED'), ('small_model', 'MODEL_DIRECTION_UNRESOLVED'),
    ('reverse', 'LOCAL_DIRECTION_MISMATCH'), ('match', 'LOCAL_DIRECTION_MATCH'),
])
def test_projection_qualification_distinguishes_unknown_numerics_and_model_direction(lib, change, expected):
    reference, actual, model, valid = {'resolved': True}, [.02] * 4, [.02] * 4, True
    if change == 'reference':
        reference['resolved'] = False
    elif change == 'numeric':
        valid = False
    elif change == 'actual':
        actual[0] = -.02
    elif change == 'small_model':
        model[0] = 1e-8
    elif change == 'reverse':
        model = [-.02] * 4
    row = lib.mod.projection_comparison(reference, [.02] * 4, actual, model, valid)
    assert row['status'] == expected and row['physical_eligibility_changed'] is False
    if expected == 'LOCAL_DIRECTION_MATCH':
        assert row['valid_local_direction_match'] is True
    elif expected == 'LOCAL_DIRECTION_MISMATCH':
        assert row['valid_local_direction_match'] is False
    else:
        assert row['valid_local_direction_match'] is None


def test_midpoint_drift_is_preserved_as_a_separate_even_direction(lib):
    np, mod = lib.np, lib.mod
    data = fixture(lib)
    original = mod.build_directions(data.start, data.branches)
    shifted = copy.deepcopy(data.branches)
    shifted[2]['states'][:, 23] += .001
    shifted[3]['states'][:, 23] += .001
    changed = mod.build_directions(data.start, shifted)
    np.testing.assert_allclose(changed['delta_next_state'][0], original['delta_next_state'][0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(changed['midpoint_drift_next_state'][0, :, 23]
                               - original['midpoint_drift_next_state'][0, :, 23], .001, rtol=1e-12, atol=1e-12)


def test_partial_branch_and_expired_compute_budget_are_not_complete_inputs(lib):
    data = fixture(lib)
    directions = lib.mod.build_directions(data.start, data.branches)
    with pytest.raises(ValueError, match='CPU deadline'):
        lib.mod.joint_response(data.model, directions, deadline=-1.)
    data.branches[4]['states'] = data.branches[4]['states'][:31]
    with pytest.raises(ValueError, match='invalid physical branch'):
        lib.mod.build_directions(data.start, data.branches)
