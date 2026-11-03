"""Synthetic81 objective/endpoint contracts; root owns numerical execution."""
from __future__ import annotations

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
    from tools81 import policy81
    return SimpleNamespace(torch=torch, policy=policy81, old=policy79)


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
            angular = torch.stack((state[:, 26], state[:, 27], .8 * state[:, 28] + self.gain * effect), dim=-1)
            return torch.cat((state[:, :26], angular, state[:, 29:]), dim=-1)

    return Model().eval().requires_grad_(False), Actor().eval().requires_grad_(False)


def starts(lib):
    torch = lib.torch
    points = []
    for seed_index, seed in enumerate(lib.policy.D_SEEDS):
        for tick_index, tick in enumerate(lib.policy.TICKS):
            state, frame = torch.zeros(45), torch.zeros(57)
            command = .1 if tick == 175 else -.1
            state[2], state[3] = .46, 1.
            frame[5], frame[8] = -1., command * .25
            points.append({'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'state': state, 'frame': frame,
                'history': frame.expand(10, 57).clone(), 'filtered': torch.zeros(3), 'last': torch.zeros(16),
                'fifo': torch.empty(0, 16), 'shaper': SimpleNamespace(state=None),
                'commands': torch.tensor([[0., 0., command]] * 33), 'nominal_height': .46,
                'base_yaw': 1. + tick_index + seed_index / 10., 'gain': 1. + tick_index,
                'cost_factor': 1. + seed_index})
    return points


def synthetic_rollout(lib, *, scale=1.):
    """Closed-form independent cost/yaw functions with distinct head directions."""
    torch = lib.torch

    def rollout(_model, actor, point, theta, *, return_trace, horizon):
        assert return_trace and horizon == 32 and theta.dtype == torch.float64 and float(theta) == 0.
        parameter = actor.action_head.bias
        cost = point['cost_factor'] * (1. + scale * parameter[3])
        yaw = point['base_yaw'] + scale * point['gain'] * parameter[7]
        command = torch.as_tensor(point['commands'], dtype=torch.float64)[:32, 2]
        states = torch.zeros(32, 45, dtype=torch.float64)
        states[:, 28] = yaw + command
        components = torch.stack((cost, cost * 0, cost * 0, cost * 0, cost * 0, cost * 0)).expand(32, 6)
        # An explicit zero weight dependence permits complete head autograd.
        value = cost + actor.action_head.weight.sum() * 0.
        states = states + actor.action_head.weight.sum() * 0.
        return value, {'states': states, 'cost_components': components,
                       'actions': torch.zeros(32, 16, dtype=torch.float64)}
    return rollout


def install_rollout(monkeypatch, lib, rollout):
    # Only replace the new81 module's reference, never mutate frozen54 globals.
    monkeypatch.setattr(lib.policy, 'numeric', SimpleNamespace(imaginary=rollout))


def update(lib, model, actor, points, arm, output):
    return lib.policy.update_actor(model, actor, points, arm, output, cpu_deadline=time.process_time() + 120.)


def test_objectives_have_same_six_starts_but_max_is_inside_reset_mean(lib, monkeypatch):
    torch, policy = lib.torch, lib.policy
    model, actor = modules(lib)
    actor = actor.double()
    actor.action_head.requires_grad_(True)
    points = starts(lib)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib))
    result = policy.objective_values(model.double(), actor, list(reversed(points)))
    bias3, bias7 = actor.action_head.bias[3], actor.action_head.bias[7]
    assert float(result['C6'].detach()) == pytest.approx(2. * (1. + float(bias3.detach())), abs=1e-14)
    assert float(result['Q6'].detach()) == pytest.approx(2.1 + 2. * float(bias7.detach()), abs=1e-14)
    assert [(r['seed'], r['tick']) for r in result['points']] == [(p['seed'], p['tick']) for p in points]
    c_grad, = torch.autograd.grad(result['C6'], actor.action_head.bias, retain_graph=True)
    q_grad, = torch.autograd.grad(result['Q6'], actor.action_head.bias)
    assert float(c_grad[3]) == 2. and float(q_grad[7]) == 2.
    assert float(c_grad[7]) == float(q_grad[3]) == 0.


def test_normalization_is_masked_max_row_l1_not_l2_and_scale_invariant(lib):
    torch, policy = lib.torch, lib.policy
    weight, bias = torch.zeros(16, 128, dtype=torch.float64), torch.zeros(16, dtype=torch.float64)
    weight[3, :2], bias[3], bias[7], weight[0, 0] = torch.tensor([2., -3.]), 5., 2., 1e9
    direction, scale = policy.normalize_direction(weight, bias)
    assert scale == 10. and float(direction[0][3, 0]) == -.2 and float(direction[1][3]) == -.5
    assert float(direction[1][7]) == -.2 and float(direction[0][0, 0]) == 0.
    enlarged, _ = policy.normalize_direction(weight * 7., bias * 7.)
    for left, right in zip(direction, enlarged, strict=True):
        torch.testing.assert_close(left, right, atol=0., rtol=1e-15)
    with pytest.raises(ValueError, match='zero or nonfinite'):
        policy.normalize_direction(weight * 0., bias * 0.)


@pytest.mark.parametrize('arm,index', [('C6', 3), ('Q6', 7)])
def test_fixed_f32_and_ideal_endpoints_separate_preserve_sources_and_reload(lib, monkeypatch, tmp_path, arm, index):
    torch, policy = lib.torch, lib.policy
    model, actor = modules(lib)
    model.gain.grad = torch.tensor(2.)
    source = [policy.mapping_sha256(m.state_dict()) for m in (model, actor)]
    old_functions = (lib.old.update_actor, lib.old.project_head, lib.old.numeric.imaginary)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib))
    output = tmp_path / arm
    probes, receipt = update(lib, model, actor, starts(lib), arm, output)
    assert receipt['status'] == receipt['verification_after'] == 'COMPLETE'
    assert receipt['updates_attempted'] == receipt['updates_completed'] == receipt['updates'] == 1
    assert receipt['gradient_points_completed'] == 6 and receipt['fd_points_completed'] == 48
    assert len(receipt['probes']) == len(probes) == 8 and len(receipt['selfchecks']) == 4
    assert all(r['autograd_agrees'] and r['magnitude_floor'] is None for r in receipt['selfchecks'])
    tensors = torch.load(output / 'head_step.pt', weights_only=True)
    assert float(tensors['direction_bias64'][index]) == -1.
    assert int((tensors['direction_bias64'] != 0).sum()) == 1
    for scale_index, epsilon in enumerate(policy.SCALES):
        for sign in (1, -1):
            actual = probes[(scale_index, sign)]
            expected = (actor.action_head.bias[index].double() - sign * epsilon).float()
            assert torch.equal(actual.action_head.bias[index], expected)
            assert policy._protected_hash(actual) == policy._protected_hash(actor)
            ideal = policy.load_probe(actor, output, scale_index, sign, ideal=True)
            loaded = policy.load_probe(actor, output, scale_index, sign)
            assert float(ideal.action_head.bias[index]) == float(actor.action_head.bias[index].double() - sign * epsilon)
            assert policy.mapping_sha256(loaded.state_dict()) == policy.mapping_sha256(actual.state_dict())
            assert ideal.action_head.bias.dtype == torch.float64 and actual.action_head.bias.dtype == torch.float32
            assert float(policy.row_l1(actual.action_head.weight, actual.action_head.bias,
                                      actor.action_head.weight, actor.action_head.bias).max()) <= .01
    assert source == [policy.mapping_sha256(m.state_dict()) for m in (model, actor)]
    assert float(model.gain.grad) == 2. and old_functions == (lib.old.update_actor, lib.old.project_head, lib.old.numeric.imaginary)
    assert json.loads((output / 'receipt.json').read_text()) == receipt
    with pytest.raises(FileExistsError):
        update(lib, model, actor, starts(lib), arm, output)


def test_tiny_nonzero_q_gradient_does_not_import_old_cost_floor(lib, monkeypatch, tmp_path):
    model, actor = modules(lib)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib, scale=1e-12))
    _, receipt = update(lib, model, actor, starts(lib), 'Q6', tmp_path / 'tiny')
    assert receipt['status'] == 'COMPLETE' and 0. < receipt['max_row_l1_gradient'] < 1e-5
    assert receipt['direction_magnitude_floor'] is None


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'partial', 'wrong_D', 'zero'])
def test_invalid_or_zero_direction_preserves_incomplete_receipt(lib, monkeypatch, tmp_path, kind):
    model, actor = modules(lib)
    points = starts(lib)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib, scale=0. if kind == 'zero' else 1.))
    if kind == 'missing':
        points.pop()
    elif kind == 'duplicate':
        points[1] = points[0]
    elif kind == 'partial':
        points[0]['status'] = 'INCOMPLETE'
    elif kind == 'wrong_D':
        for point in points:
            point['seed'] += 100
    output = tmp_path / kind
    with pytest.raises(ValueError):
        update(lib, model, actor, points, 'C6', output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert receipt['updates_attempted'] == receipt['updates_completed'] == 0
    assert not (output / 'actor_state_dict.pt').exists()


def test_interruption_keeps_completed_readback_prefix(lib, monkeypatch, tmp_path):
    model, actor = modules(lib)
    original, calls = synthetic_rollout(lib), 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise SystemExit('synthetic interruption')
        return original(*args, **kwargs)

    install_rollout(monkeypatch, lib, interrupted)
    output = tmp_path / 'partial'
    with pytest.raises(SystemExit, match='synthetic interruption'):
        update(lib, model, actor, starts(lib), 'Q6', output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['gradient_points_completed'] == len(receipt['points']) == 3
    assert receipt['updates_completed'] == 0 and receipt['status'] == 'INCOMPLETE'
    assert (output / 'gradient_point_02.json').is_file() and not (output / 'gradient_point_03.json').exists()


def test_real54_rollout_preserves_recursive_gradient_and_counts(lib, tmp_path):
    model, actor = modules(lib)
    _, receipt = update(lib, model, actor, starts(lib), 'C6', tmp_path / 'real54')
    assert receipt['status'] == 'COMPLETE'
    assert receipt['model_calls_attempted'] == receipt['model_calls'] == 54 * 32
    assert receipt['max_row_l1_gradient'] > 0. and receipt['directional_derivative'] < 0.


def test_endpoint_write_failure_after_main_candidate_revokes_eligibility(lib, monkeypatch, tmp_path):
    model, actor = modules(lib)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib))
    binary = lib.policy._binary

    def fail(path, value):
        if path.name == 'probe_00_plus_ideal64.pt':
            raise OSError('synthetic artifact failure')
        return binary(path, value)

    monkeypatch.setattr(lib.policy, '_binary', fail)
    output = tmp_path / 'write_failure'
    with pytest.raises(OSError, match='synthetic artifact failure'):
        update(lib, model, actor, starts(lib), 'C6', output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['updates_attempted'] == receipt['updates_completed'] == 1
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_execution']
    assert (output / 'probe_00_plus_f32.pt').is_file()
    assert (output / 'partial_endpoint_state_dict.pt').is_file()
    with pytest.raises(ValueError, match='incomplete endpoint'):
        lib.policy.load_probe(actor, output, 0, 1)


def test_late_source_verification_failure_revokes_complete_candidate(lib, monkeypatch, tmp_path):
    model, actor = modules(lib)
    install_rollout(monkeypatch, lib, synthetic_rollout(lib))
    binary = lib.policy._binary

    def changed(path, value):
        digest = binary(path, value)
        if path.name == 'actor_state_dict.pt':
            with lib.torch.no_grad():
                actor.action_head.bias[0] += .1
        return digest

    monkeypatch.setattr(lib.policy, '_binary', changed)
    output = tmp_path / 'late_source_change'
    with pytest.raises(ValueError, match='anchor source changed'):
        update(lib, model, actor, starts(lib), 'C6', output)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and receipt['verification_after'] == 'FAILED'
    assert receipt['updates_completed'] == 1 and not receipt['eligible_for_execution']
    assert (output / 'partial_endpoint_state_dict.pt').is_file()
