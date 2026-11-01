"""Analytic response fixtures; no assets, actor, fitting or physical simulator."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def affine_case():
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / '.python-deps'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    np = pytest.importorskip('numpy')
    pytest.importorskip('torch')
    from tools57 import diagnose57

    class Affine:
        def __init__(self, yaw_gain=2.):
            self.yaw_gain = yaw_gain

        def __call__(self, state, action):
            result = state.clone()
            result[:, 28] = .5 * state[:, 28] + self.yaw_gain * action[:, 3] + .125
            result[:, 23] = .25 * state[:, 23] + .75 * action[:, 7] - .0625
            return result

    initial = np.zeros(45)
    initial[3], initial[23], initial[28] = 1., -.1, .2
    base, direction = np.zeros(16), np.zeros(16)
    base[[3, 7]], direction[[3, 7]] = [.37, -.23], [.13, -.07]
    branches = []
    for theta in (0., 0., *(sign * eps for eps in diagnose57.EPSILONS for sign in (1, -1))):
        actual = (base + theta * direction).astype(np.float32)
        states = np.repeat(initial[None], 32, axis=0)
        # Closed-form geometric sums define truth independently of the model.
        for column, decay, gain, action_index, bias in ((28, .5, 2., 3, .125), (23, .25, .75, 7, -.0625)):
            powers = decay ** np.arange(1, 33)
            states[:, column] = powers * initial[column] + (
                gain * float(actual[action_index]) + bias) * (1 - powers) / (1 - decay)
        observations = np.full((32, 627), 99., dtype=np.float32)
        observations[:, 41:57] = actual
        branches.append({'states': states, 'observations': observations})
    return SimpleNamespace(np=np, subject=diagnose57, model=Affine,
                           start={'state': initial}, branches=branches)


def test_teacher_forcing_uses_each_perturbed_branch_prestate(affine_case):
    case = affine_case
    rows, arrays = case.subject.response_diagnostics(case.model(), case.start, case.branches)
    for index, row in enumerate(rows):
        actual = arrays[f'epsilon_{index}_actual_state_fd']
        predicted = arrays[f'epsilon_{index}_model_state_fd']
        assert actual.shape == predicted.shape == (32, 45)
        case.np.testing.assert_allclose(predicted, actual, rtol=0., atol=2e-11)
        assert actual[1, 28] > 1.4 * actual[0, 28]
        assert row['first_step_prestate_exact'] and not row['actor_evaluated']
        assert not case.np.array_equal(arrays[f'epsilon_{index}_plus_pre_states'][1:],
                                       arrays[f'epsilon_{index}_minus_pre_states'][1:])


def test_wrong_action_gain_is_visible_at_the_shared_first_state(affine_case):
    case = affine_case
    rows, _ = case.subject.response_diagnostics(case.model(-2.), case.start, case.branches)
    for row in rows:
        yaw = row['first_step_yaw28']
        assert row['first_step_prestate_exact']
        assert yaw['actual_fd'] > 0 > yaw['model_fd']
        assert yaw['model_fd'] == pytest.approx(-yaw['actual_fd'], rel=0., abs=2e-11)


def test_action_difference_comes_from_quantized_saved_actions(affine_case):
    case = affine_case
    rows, arrays = case.subject.response_diagnostics(case.model(), case.start, case.branches)
    for index, epsilon in enumerate(case.subject.EPSILONS):
        plus, minus = case.branches[2 + 2 * index:4 + 2 * index]
        expected = (plus['observations'][:, 41:57].astype('float64')
                    - minus['observations'][:, 41:57].astype('float64')) / (2 * epsilon)
        case.np.testing.assert_array_equal(arrays[f'epsilon_{index}_actual_action_fd'], expected)
        case.np.testing.assert_array_equal(rows[index]['first_step_actual_action_fd'], expected[0])
    assert abs(rows[0]['first_step_actual_action_fd'][3] - .13) > 1e-7


def test_model_selfcheck_does_not_certify_the_physical_direction():
    from tools57.diagnose57 import gradient_receipt

    row = gradient_receipt({'seed': 1, 'tick': 110}, value=1., analytic=.3, numerical=.3)
    physical_gradient = -.2
    assert row['autograd_agrees'] and row['direction_identifiable']
    assert row['gradient'] * physical_gradient < 0
