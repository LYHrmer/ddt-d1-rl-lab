"""Synthetic79 parameter updates only; root owns bounded numerical execution."""
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
    return SimpleNamespace(torch=torch, policy=policy79)


def modules(lib):
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
            self.gain = torch.nn.Parameter(torch.tensor(.7))
            self.register_buffer('input_mean', torch.zeros(61))

        def forward(self, state, action):
            effect = action[:, 3] - action[:, 7] + action[:, 11] - action[:, 15]
            velocity = torch.stack((.9 * state[:, 23] + .1 * action[:, 3],
                                    .9 * state[:, 24], state[:, 25]), dim=-1)
            angular = torch.stack((state[:, 26], state[:, 27],
                                   .8 * state[:, 28] + self.gain * effect), dim=-1)
            return torch.cat((state[:, :23], velocity, angular, state[:, 29:]), dim=-1)

    return Model().eval().requires_grad_(False), Actor().eval().requires_grad_(False)


def starts(lib, count=3):
    torch = lib.torch
    result = []
    for index in range(count):
        state, frame = torch.zeros(45), torch.zeros(57)
        state[2], state[3] = .46, 1.
        frame[5], frame[8] = -1., .025
        result.append({'seed': 79001, 'tick': index, 'state': state, 'frame': frame,
            'history': frame.expand(10, 57).clone(), 'filtered': torch.zeros(3), 'last': torch.zeros(16),
            'fifo': torch.empty(0, 16), 'shaper': SimpleNamespace(state=None),
            'commands': torch.tensor([[0., 0., .1]] * 33), 'nominal_height': .46,
            'factor': float(index + 1)})
    return result


def short_objective(lib):
    def objective(model, actor, point, theta, *, horizon):
        assert horizon == 32 and theta.dtype == lib.torch.float64 and float(theta) == 0.
        observation = lib.torch.full((1, 627), .1, dtype=lib.torch.float64)
        action = lib.policy.numeric.actor_forward(actor, observation)
        state = point['state'][None].double()
        predicted = model(state, action)
        return point['factor'] * (predicted[:, 28] - .1).square().mean()
    return objective


def call_update(lib, model, actor, anchor, points, output):
    return lib.policy.update_actor(model, actor, anchor, points, output, cpu_deadline=time.process_time() + 60.)


def test_mask_exactly_four_wheel_rows_and516_parameters(lib):
    torch, policy = lib.torch, lib.policy
    weight, bias = policy.mask_gradient(torch.ones(16, 128, dtype=torch.float64), torch.ones(16, dtype=torch.float64))
    assert int((weight != 0).sum() + (bias != 0).sum()) == 516
    assert bool((weight[list(policy.OTHER_ROWS)] == 0).all())
    assert bool((bias[list(policy.OTHER_ROWS)] == 0).all())


def test_projection_is_cumulative_relative_to_original_anchor_and_f32_admitted(lib):
    torch, policy = lib.torch, lib.policy
    anchor_weight = torch.full((16, 128), .03125, dtype=torch.float64)
    anchor_bias = torch.full((16,), -.0625, dtype=torch.float64)
    first_weight, first_bias = anchor_weight.clone(), anchor_bias.clone()
    first_weight[list(policy.WHEEL_ROWS), 0] += .009
    first_weight, first_bias = policy.project_head(first_weight, first_bias, anchor_weight, anchor_bias)
    second_weight, second_bias = first_weight.clone(), first_bias.clone()
    second_bias[list(policy.WHEEL_ROWS)] += .009
    actual_weight, actual_bias = policy.project_head(second_weight, second_bias, anchor_weight, anchor_bias)
    bounds = policy.row_l1(actual_weight.float(), actual_bias.float(), anchor_weight.float(), anchor_bias.float())
    assert bool((bounds <= .01).all()) and bool((bounds > .0099).all())
    assert bool((policy.row_l1(second_weight, second_bias, anchor_weight, anchor_bias) > .01).all())
    assert torch.equal(actual_weight[list(policy.OTHER_ROWS)], anchor_weight[list(policy.OTHER_ROWS)])
    assert torch.equal(actual_bias[list(policy.OTHER_ROWS)], anchor_bias[list(policy.OTHER_ROWS)])


def test_zero_update_double_roundtrip_keeps_f32_mapping_and_actions_exact(lib):
    torch, policy = lib.torch, lib.policy
    _, actor = modules(lib)
    copied = copy.deepcopy(actor).double().float()
    observation = torch.linspace(-.2, .2, 627).reshape(1, 627)
    assert policy.mapping_sha256(copied.state_dict()) == policy.mapping_sha256(actor.state_dict())
    assert torch.equal(policy.numeric.actor_forward(actor, observation), policy.numeric.actor_forward(copied, observation))


@pytest.mark.parametrize('count', [3, 9])
def test_one_step_uses_mean_true_model_input_gradient_and_preserves_sources(lib, monkeypatch, tmp_path, count):
    torch, policy = lib.torch, lib.policy
    model, actor = modules(lib)
    anchor = copy.deepcopy(actor)
    objective = short_objective(lib)
    monkeypatch.setattr(policy.numeric, 'imaginary', objective)
    points = starts(lib, count)
    source_hashes = [policy.mapping_sha256(m.state_dict()) for m in (model, actor, anchor)]
    expected_actor, expected_model = copy.deepcopy(actor).double(), copy.deepcopy(model).double()
    expected_actor.action_head.requires_grad_(True)
    expected_loss = sum(objective(expected_model, expected_actor, point, torch.tensor(0., dtype=torch.float64), horizon=32)
                        for point in points) / count
    expected = policy.mask_gradient(*torch.autograd.grad(expected_loss, tuple(expected_actor.action_head.parameters())))
    updated, receipt = call_update(lib, model, actor, anchor, points, tmp_path / 'update')
    artifacts = torch.load(tmp_path / 'update' / 'head_step.pt', weights_only=True)
    torch.testing.assert_close(artifacts['masked_mean_gradient_weight64'], expected[0], rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(artifacts['masked_mean_gradient_bias64'], expected[1], rtol=1e-13, atol=1e-15)
    clipped = torch.cat((artifacts['clipped_gradient_weight64'].flatten(), artifacts['clipped_gradient_bias64']))
    assert float(torch.linalg.vector_norm(clipped)) <= 1.
    for kind in ('weight', 'bias'):
        torch.testing.assert_close(artifacts[f'sgd_{kind}64'], artifacts[f'head_before_{kind}64']
                                   - .001 * artifacts[f'clipped_gradient_{kind}64'], rtol=1e-13, atol=1e-15)
    assert receipt['status'] == receipt['verification_after'] == 'COMPLETE'
    assert receipt['updates_completed'] == receipt['updates_attempted'] == receipt['updates'] == 1
    assert receipt['gradient_points_completed'] == count and receipt['fd_points_completed'] == 2 * count
    assert receipt['selfcheck']['passed'] and receipt['actual_update_norm'] > 0
    assert source_hashes == [policy.mapping_sha256(m.state_dict()) for m in (model, actor, anchor)]
    assert updated is not actor and all(not p.requires_grad and p.grad is None for p in updated.parameters())
    for name, value in updated.state_dict().items():
        if name in policy.HEAD_KEYS:
            assert torch.equal(value[list(policy.OTHER_ROWS)], actor.state_dict()[name][list(policy.OTHER_ROWS)])
        else:
            assert torch.equal(value, actor.state_dict()[name])
    assert max(receipt['final_row_l1']) <= .01
    assert json.loads((tmp_path / 'update' / 'receipt.json').read_text()) == receipt


def test_real_frozen54_h32_path_backpropagates_to_wheel_head(lib, tmp_path):
    model, actor = modules(lib)
    updated, receipt = call_update(lib, model, actor, copy.deepcopy(actor), starts(lib), tmp_path / 'real54')
    assert receipt['selfcheck']['passed'] and receipt['actual_update_norm'] > 0
    assert lib.policy._protected_hash(updated) == lib.policy._protected_hash(actor)


def test_refitted_model_may_retain_frozen_gradients_without_mutation(lib, monkeypatch, tmp_path):
    model, actor = modules(lib)
    model.gain.grad = lib.torch.full_like(model.gain, 2.)
    before = model.gain.grad.clone()
    monkeypatch.setattr(lib.policy.numeric, 'imaginary', short_objective(lib))
    _, receipt = call_update(lib, model, actor, copy.deepcopy(actor), starts(lib), tmp_path / 'refitted')
    assert receipt['status'] == receipt['verification_after'] == 'COMPLETE'
    assert lib.torch.equal(model.gain.grad, before)
    assert receipt['source_gradient_sha256']['model'] == lib.policy._gradient_hash(model)


@pytest.mark.parametrize('kind', ['low_magnitude', 'zero', 'fd_disagreement'])
def test_failed_direction_retains_diagnostic_and_performs_no_update(lib, monkeypatch, tmp_path, kind):
    policy = lib.policy
    model, actor = modules(lib)
    objective, calls = short_objective(lib), []

    def altered(*args, **kwargs):
        calls.append(None)
        value = objective(*args, **kwargs)
        if kind == 'low_magnitude':
            return value * 1e-12
        if kind == 'zero':
            return value * 0.
        return value + (1. if 4 <= len(calls) <= 6 else 0.)

    monkeypatch.setattr(policy.numeric, 'imaginary', altered)
    before = policy.mapping_sha256(actor.state_dict())
    output = tmp_path / kind
    with pytest.raises(ValueError, match='direction qualification failed'):
        call_update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert receipt['verification_after'] == 'COMPLETE'
    assert receipt['updates_completed'] == receipt['updates_attempted'] == 0
    assert receipt['gradient_points_completed'] == 3
    assert not receipt['selfcheck']['passed'] and (output / 'head_step.pt').is_file()
    assert not (output / 'actor_state_dict.pt').exists()
    assert policy.mapping_sha256(actor.state_dict()) == before
    if kind == 'low_magnitude':
        assert receipt['selfcheck']['autograd_agrees'] and not receipt['selfcheck']['direction_identifiable']
    if kind == 'fd_disagreement':
        assert not receipt['selfcheck']['autograd_agrees'] and receipt['fd_points_completed'] == 6


def test_mid_gradient_failure_keeps_exact_completed_prefix(lib, monkeypatch, tmp_path):
    policy = lib.policy
    model, actor = modules(lib)
    objective, count = short_objective(lib), 0

    def failing(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('synthetic point2 failure')
        return objective(*args, **kwargs)

    monkeypatch.setattr(policy.numeric, 'imaginary', failing)
    output = tmp_path / 'prefix'
    with pytest.raises(RuntimeError, match='point2 failure'):
        call_update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    prefix = lib.torch.load(output / 'gradient_prefix_00.pt', weights_only=True)
    assert receipt['gradient_points_completed'] == 1 and receipt['updates_attempted'] == 0
    assert prefix['completed'] == 1 and prefix['denominator'] == 3
    assert not (output / 'gradient_prefix_01.pt').exists()
    assert receipt['verification_after'] == 'COMPLETE'


def test_post_sgd_failure_retains_attempted_and_completed_counts(lib, monkeypatch, tmp_path):
    policy = lib.policy
    model, actor = modules(lib)
    monkeypatch.setattr(policy.numeric, 'imaginary', short_objective(lib))

    def fail(*_args):
        raise RuntimeError('synthetic projection failure')

    monkeypatch.setattr(policy, 'project_head', fail)
    output = tmp_path / 'after_step'
    with pytest.raises(RuntimeError, match='projection failure'):
        call_update(lib, model, actor, copy.deepcopy(actor), starts(lib), output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['updates_attempted'] == receipt['updates_completed'] == 1
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert (output / 'partial_actor_state_dict64.pt').is_file()
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    assert any(e['stage'] == 'sgd_completed' and e['updates_completed'] == 1 for e in events)


def test_overwrite_and_original_anchor_violations_fail_closed(lib, monkeypatch, tmp_path):
    torch, policy = lib.torch, lib.policy
    model, actor = modules(lib)
    anchor = copy.deepcopy(actor)
    monkeypatch.setattr(policy.numeric, 'imaginary', short_objective(lib))
    output = tmp_path / 'existing'
    output.mkdir()
    with pytest.raises(FileExistsError):
        call_update(lib, model, actor, anchor, starts(lib), output)
    with torch.no_grad():
        actor.action_head.bias[3] += .011
    with pytest.raises(ValueError, match='original-anchor bound'):
        call_update(lib, model, actor, anchor, starts(lib), tmp_path / 'bound')
    assert json.loads((tmp_path / 'bound' / 'receipt.json').read_text())['updates_attempted'] == 0


def test_original_formula_separates_agreement_magnitude_and_sign(lib):
    policy = lib.policy
    assert policy.direction_check(2e-5, 2e-5)['passed']
    assert not policy.direction_check(2e-6, 2e-6)['passed']
    opposite = policy.direction_check(2e-5, -2e-5)
    assert opposite['autograd_agrees'] and not opposite['direction_identifiable']
    assert not policy.direction_check(0., 0.)['passed']
