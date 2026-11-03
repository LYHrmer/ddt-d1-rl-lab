"""Synthetic80 update contracts; numerical execution belongs to the root runner."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def lib():
    for path in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    torch = pytest.importorskip('torch')
    pytest.importorskip('numpy')
    from tools79 import policy79
    from tools80 import policy80
    return SimpleNamespace(torch=torch, old=policy79, policy=policy80)


def modules(lib, *, recursive=False):
    torch = lib.torch

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(627, 3), torch.nn.Tanh())
            self.policy_net = torch.nn.Sequential(torch.nn.Linear(60, 128), torch.nn.Tanh())
            self.action_head = torch.nn.Linear(128, 16)
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(.003)
                self.action_head.bias.copy_(torch.linspace(-.02, .03, 16))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5 if recursive else .7))
            self.register_buffer('input_mean', torch.zeros(61))

        def forward(self, state, action):
            if recursive:
                return self.gain * state + action[:, 3:4]
            effect = action[:, 3] - action[:, 7] + action[:, 11] - action[:, 15]
            velocity = torch.stack((.9 * state[:, 23] + .1 * action[:, 3],
                                    .9 * state[:, 24], state[:, 25]), dim=-1)
            angular = torch.stack((state[:, 26], state[:, 27], .8 * state[:, 28] + self.gain * effect), dim=-1)
            return torch.cat((state[:, :23], velocity, angular, state[:, 29:]), dim=-1)

    return Model().eval().requires_grad_(False), Actor().eval().requires_grad_(False)


def starts(lib, count=27):
    torch = lib.torch
    points = []
    for index in range(count):
        state, frame = torch.zeros(45), torch.zeros(57)
        state[2], state[3] = .46, 1.
        frame[5], frame[8] = -1., .025
        points.append({'seed': 80101 + index // 3, 'tick': (110, 260, 410)[index % 3],
            'state': state, 'frame': frame, 'history': frame.expand(10, 57).clone(),
            'filtered': torch.zeros(3), 'last': torch.zeros(16), 'fifo': torch.empty(0, 16),
            'shaper': SimpleNamespace(state=None), 'commands': torch.tensor([[0., 0., .1]] * 33),
            'nominal_height': .46, 'initial': .02 + .001 * index, 'factor': 1. + index / 27.})
    return points


def update(lib, model, actor, anchor, points, output):
    return lib.policy.update_actor(model, actor, anchor, points, output, cpu_deadline=time.process_time() + 100.)


def recursive_objective(lib):
    """Thirty-two on-policy recurrent states with an independent scalar oracle."""
    torch = lib.torch

    def objective(model, actor, point, theta, *, horizon):
        assert horizon == 32 and theta.dtype == torch.float64 and float(theta) == 0.
        state = torch.tensor([[point['initial']]], dtype=torch.float64)
        costs = []
        for _ in range(horizon):
            action = actor.action_head(state.expand(1, 128))
            state = model(state, action)
            costs.append(state.square().mean())
        return point['factor'] * torch.stack(costs).mean()

    return objective


def linear_objective(lib):
    def objective(model, actor, _point, theta, *, horizon):
        assert horizon == 32 and float(theta) == 0.
        latent = lib.torch.ones(1, 128, dtype=lib.torch.float64)
        return 1. - model(latent[:, :1] * 0., actor.action_head(latent)).mean()
    return objective


def isolated_objective(monkeypatch, policy, objective):
    # Replace only the new80 module's reference. Frozen79/54 globals stay intact.
    monkeypatch.setattr(policy, 'numeric', SimpleNamespace(imaginary=objective))


def test27_equal_mean_matches_independent_full_recurrence_and_preserves_sources(lib, monkeypatch, tmp_path):
    torch, policy, old = lib.torch, lib.policy, lib.old
    model, actor = modules(lib, recursive=True)
    anchor = copy.deepcopy(actor)
    model.gain.grad = torch.tensor(2.)
    source = [policy.mapping_sha256(m.state_dict()) for m in (model, actor, anchor)]
    original_functions = (old.update_actor, old.project_head, old.numeric.imaginary)
    isolated_objective(monkeypatch, policy, recursive_objective(lib))
    points = starts(lib)
    alpha = float(model.gain) + sum(actor.action_head.weight[3].double().tolist())
    bias = float(actor.action_head.bias[3])
    expected_w = expected_b = expected_loss = detached_b = 0.
    for point in points:
        state, derivative_w, derivative_b = point['initial'], 0., 0.
        for _ in range(32):
            previous = state
            state = alpha * previous + bias
            derivative_w = previous + alpha * derivative_w
            derivative_b = 1. + alpha * derivative_b
            scale = point['factor'] / (32 * 27)
            expected_loss += scale * state * state
            expected_w += scale * 2 * state * derivative_w
            expected_b += scale * 2 * state * derivative_b
            detached_b += scale * 2 * state
    changed, receipt = update(lib, model, actor, anchor, points, tmp_path / 'mean27')
    tensors = torch.load(tmp_path / 'mean27/head_step.pt', weights_only=True)
    expected_weight, expected_bias = torch.zeros(16, 128, dtype=torch.float64), torch.zeros(16, dtype=torch.float64)
    expected_weight[3] = expected_w
    expected_bias[3] = expected_b
    torch.testing.assert_close(tensors['masked_mean_gradient_weight64'], expected_weight, rtol=2e-12, atol=2e-14)
    torch.testing.assert_close(tensors['masked_mean_gradient_bias64'], expected_bias, rtol=2e-12, atol=2e-14)
    assert abs(receipt['mean_cost_before'] - expected_loss) <= 2e-14 + 2e-12 * abs(expected_loss)
    assert abs(expected_b - detached_b) > .01
    assert receipt['gradient_points_completed'] == receipt['point_count'] == 27
    assert receipt['fd_points_completed'] == 54 and receipt['selfcheck']['passed']
    assert receipt['status'] == receipt['verification_after'] == 'COMPLETE'
    assert receipt['updates_attempted'] == receipt['updates_completed'] == 1
    assert receipt['schema'] == 'd1-policy80-update-v1'
    assert source == [policy.mapping_sha256(m.state_dict()) for m in (model, actor, anchor)]
    assert float(model.gain.grad) == 2.
    assert (old.update_actor, old.project_head, old.numeric.imaginary) == original_functions
    assert policy._protected_hash(changed) == policy._protected_hash(actor)
    assert all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in changed.parameters())
    for index in (0, 8, 26):
        prefix = torch.load(tmp_path / 'mean27' / f'gradient_prefix_{index:02d}.pt', weights_only=True)
        assert prefix['denominator'] == 27 and prefix['completed'] == index + 1
    assert json.loads((tmp_path / 'mean27/receipt.json').read_text()) == receipt


def test9_real_frozen54_update_is_bitwise_equivalent_to79(lib, tmp_path):
    torch, policy, old = lib.torch, lib.policy, lib.old
    model, actor = modules(lib)
    anchor, points = copy.deepcopy(actor), starts(lib, 9)
    old_changed, before = old.update_actor(model, actor, anchor, points, tmp_path / 'old79',
                                          cpu_deadline=time.process_time() + 100.)
    new_changed, after = update(lib, model, actor, anchor, points, tmp_path / 'new80')
    assert policy.mapping_sha256(old_changed.state_dict()) == policy.mapping_sha256(new_changed.state_dict())
    for key in ('source_mapping_sha256', 'source_gradient_sha256', 'actor_before_mapping_sha256',
                'anchor_mapping_sha256', 'actor_after_mapping_sha256', 'protected_after_mapping_sha256',
                'mean_cost_before', 'masked_gradient_norm', 'gradient_norm_before_clip', 'selfcheck',
                'initial_row_l1', 'final_row_l1', 'actual_update_norm', 'points', 'fd_points',
                'updates_attempted', 'updates_completed', 'zero_update_roundtrip_exact'):
        assert before[key] == after[key], key
    old_tensors = torch.load(tmp_path / 'old79/head_step.pt', weights_only=True)
    new_tensors = torch.load(tmp_path / 'new80/head_step.pt', weights_only=True)
    assert old_tensors.keys() == new_tensors.keys()
    assert all(torch.equal(old_tensors[key], new_tensors[key]) for key in old_tensors)
    assert before['schema'] == 'd1-policy79-update-v1' and after['schema'] == 'd1-policy80-update-v1'


def test_two27_point_steps_share_original_anchor_and_cannot_accumulate_two_bounds(lib, monkeypatch, tmp_path):
    torch, policy = lib.torch, lib.policy
    model, anchor = modules(lib, recursive=True)
    points = starts(lib)
    isolated_objective(monkeypatch, policy, linear_objective(lib))
    first, receipt1 = update(lib, model, anchor, anchor, points, tmp_path / 'first')
    first_hash = policy.mapping_sha256(first.state_dict())
    second, receipt2 = update(lib, model, first, anchor, points, tmp_path / 'second')
    assert receipt1['anchor_mapping_sha256'] == receipt2['anchor_mapping_sha256'] == policy.mapping_sha256(anchor.state_dict())
    assert receipt2['actor_before_mapping_sha256'] == first_hash == policy.mapping_sha256(first.state_dict())
    assert receipt1['updates_completed'] + receipt2['updates_completed'] == 2
    assert max(receipt1['final_row_l1']) <= .01 and max(receipt2['final_row_l1']) <= .01
    actual = policy.row_l1(second.action_head.weight, second.action_head.bias, anchor.action_head.weight, anchor.action_head.bias)
    assert bool((actual <= .01).all()) and float(actual[0]) > .0099
    artifact = torch.load(tmp_path / 'second/head_step.pt', weights_only=True)
    unprojected = policy.row_l1(artifact['sgd_weight64'], artifact['sgd_bias64'], artifact['anchor_weight64'], artifact['anchor_bias64'])
    assert float(unprojected[0]) > .01
    assert policy._protected_hash(second) == policy._protected_hash(anchor)


@pytest.mark.parametrize('kind', ['low_magnitude', 'zero', 'fd_disagreement'])
def test_failed27_direction_keeps_all_diagnostics_and_zero_updates(lib, monkeypatch, tmp_path, kind):
    policy = lib.policy
    model, actor = modules(lib, recursive=True)
    objective, calls = linear_objective(lib), 0

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = objective(*args, **kwargs)
        if kind == 'low_magnitude':
            return value * 1e-12
        if kind == 'zero':
            return value * 0.
        return value + (1. if 28 <= calls <= 54 else 0.)

    isolated_objective(monkeypatch, policy, failing)
    output = tmp_path / kind
    with pytest.raises(ValueError, match='direction qualification failed'):
        update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert receipt['verification_after'] == 'COMPLETE' and receipt['gradient_points_completed'] == 27
    assert receipt['updates_attempted'] == receipt['updates_completed'] == 0
    assert not receipt['selfcheck']['passed'] and (output / 'head_step.pt').is_file()
    assert (output / 'gradient_prefix_26.pt').is_file() and not (output / 'actor_state_dict.pt').exists()
    if kind == 'low_magnitude':
        assert receipt['selfcheck']['autograd_agrees'] and not receipt['selfcheck']['direction_identifiable']
    if kind == 'fd_disagreement':
        assert receipt['fd_points_completed'] == 54 and not receipt['selfcheck']['autograd_agrees']


def test_mid27_failure_retains_full_denominator_and_completed_prefix(lib, monkeypatch, tmp_path):
    policy = lib.policy
    model, actor = modules(lib, recursive=True)
    objective, calls = linear_objective(lib), 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 17:
            raise RuntimeError('synthetic point17 interruption')
        return objective(*args, **kwargs)

    isolated_objective(monkeypatch, policy, interrupted)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='point17 interruption'):
        update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    prefix = lib.torch.load(output / 'gradient_prefix_15.pt', weights_only=True)
    assert receipt['gradient_points_completed'] == 16 and receipt['updates_attempted'] == 0
    assert prefix['denominator'] == 27 and prefix['completed'] == 16
    assert not (output / 'gradient_prefix_16.pt').exists()
    assert receipt['verification_after'] == 'COMPLETE' and (output / 'partial_actor_state_dict64.pt').is_file()


def test_post_sgd_failure_keeps_real_update_count_without_mutating79(lib, monkeypatch, tmp_path):
    model, actor = modules(lib, recursive=True)
    isolated_objective(monkeypatch, lib.policy, linear_objective(lib))
    original_projection = lib.old.project_head

    def fail(*_args):
        raise RuntimeError('synthetic projection failure')

    monkeypatch.setattr(lib.policy, 'project_head', fail)
    output = tmp_path / 'post_step'
    with pytest.raises(RuntimeError, match='projection failure'):
        update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['updates_completed'] == receipt['updates_attempted'] == 1
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert (output / 'partial_actor_state_dict64.pt').is_file()
    assert lib.old.project_head is original_projection


@pytest.mark.parametrize('kind', ['wrong_count', 'duplicate', 'changed_nonwheel', 'over_bound'])
def test_bad80_input_contract_is_rejected_and_preserved(lib, tmp_path, kind):
    model, actor = modules(lib)
    anchor, points = copy.deepcopy(actor), starts(lib)
    if kind == 'wrong_count':
        points = points[:3]
    elif kind == 'duplicate':
        points[-1] = points[0]
    else:
        with lib.torch.no_grad():
            actor.action_head.bias[0 if kind == 'changed_nonwheel' else 3] += .011
    output = tmp_path / kind
    with pytest.raises(ValueError):
        update(lib, model, actor, anchor, points, output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['updates_completed'] == receipt['updates_attempted'] == 0 and receipt['status'] == 'INCOMPLETE'
    original = (output / 'receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        update(lib, model, actor, anchor, points, output)
    assert (output / 'receipt.json').read_bytes() == original
