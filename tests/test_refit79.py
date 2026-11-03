"""Small synthetic79 replay/objective/failure tests; never a100-step fit."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope='module')
def lib():
    root = Path(__file__).resolve().parents[1]
    for directory in (root, root / '.python-deps'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    np = pytest.importorskip('numpy')
    torch = pytest.importorskip('torch')
    from tools79 import refit79
    return SimpleNamespace(np=np, torch=torch, fit=refit79)


def dataset(lib, episodes, action=.1):
    np = lib.np
    n = episodes * 600
    states = np.zeros((n, 45), dtype=np.float64)
    states[:, 3] = 1.
    actions = np.zeros((n, 16), dtype=np.float64)
    actions[:, 0] = action
    done = np.zeros(n, dtype=np.bool_)
    done[599::600] = True
    return {'s': states, 'a': actions, 'target': states.copy(),
            'episode': np.repeat(np.arange(episodes), 600), 'tick': np.tile(np.arange(600), episodes), 'done': done}


def label_rows(lib):
    np, old = lib.np, lib.fit.previous.baseline
    states = np.zeros((96, 45), dtype=np.float64)
    states[:, 3], states[:, 2] = 1., .5
    states[:, 0] = np.repeat(np.arange(24), 4)
    epsilon = np.tile(old.EPSILONS, 24)
    direction = np.zeros(16)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    plus = (epsilon[:, None] * direction).astype(np.float32)
    minus = -plus
    return {'s': states, 'a0': np.zeros((96, 16), dtype=np.float32), 'a_plus': plus, 'a_minus': minus,
            'v': ((plus.astype(np.float64) - minus.astype(np.float64)) / (2 * epsilon[:, None])).astype(np.float32),
            'target': np.tile(np.linspace(-.01, .02, 48), 2), 'epsilon': epsilon,
            'seed': np.tile(np.repeat(old.LABEL_SEEDS, 12), 2),
            'tick': np.tile(np.repeat(old.LABEL_TICKS, 4), 8), 'source_id': np.repeat([58, 62], 48),
            'source_point_index': np.tile(np.repeat(np.arange(12), 4), 2),
            'target_body': np.zeros((96, 2), dtype=np.float64),
            'body_channel_qualified': np.ones((96, 2), dtype=np.bool_)}


def test_balanced_sampling_is_local_and_never_crosses_episode_edges(lib):
    fit, torch, np = lib.fit, lib.torch, lib.np
    old, new, ow, nw = fit.validate_replay(dataset(lib, 12), dataset(lib, 1))
    before = torch.random.get_rng_state().clone()
    old_rows, new_rows = fit.sample_rows(len(ow), len(nw))
    assert (ow.shape, nw.shape, old_rows.shape, new_rows.shape) == ((6828, 32), (569, 32), (100, 32), (100, 32))
    assert torch.equal(before, torch.random.get_rng_state())
    repeated = fit.sample_rows()
    assert torch.equal(old_rows, repeated[0]) and torch.equal(new_rows, repeated[1])
    for data, windows, sampled in ((old, ow, old_rows), (new, nw, new_rows)):
        selected = windows[sampled].numpy()
        assert np.all(data['episode'][selected] == data['episode'][selected[..., :1]])
        assert np.all(np.diff(data['tick'][selected], axis=-1) == 1)
        assert not data['done'][selected[..., :-1]].any()
        assert selected.min() >= 0 and selected.max() < len(data['s'])
    assert ow[568].tolist() == list(range(568, 600))
    assert ow[569].tolist() == list(range(600, 632))


@pytest.mark.parametrize('problem', ('missing', 'partial', 'early_done', 'tick', 'nan', 'unsafe'))
def test_incomplete_or_invalid_replay_cannot_be_used(lib, problem):
    old, new = dataset(lib, 12), dataset(lib, 1)
    if problem == 'missing':
        del new['target']
    elif problem == 'partial':
        new = {key: value[:-1] for key, value in new.items()}
    elif problem == 'early_done':
        old['done'][100] = True
    elif problem == 'tick':
        new['tick'][50] = 0
    elif problem == 'nan':
        old['s'][0, 0] = lib.np.nan
    else:
        new['fallen'] = lib.np.zeros(600, dtype=lib.np.bool_)
        new['fallen'][599] = True
    with pytest.raises(ValueError):
        lib.fit.validate_replay(old, new)


def test_h8_loss_weights_old_new_equally_and_propagates_recurrent_state(lib):
    fit, torch = lib.fit, lib.torch
    old, new, ow, nw = fit.validate_replay(dataset(lib, 12, .1), dataset(lib, 1, .2))
    tensors = [{k: torch.as_tensor(data[k], dtype=torch.float32) for k in ('s', 'a', 'target')} for data in (old, new)]

    class Integrator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, state, action):
            basis = state.new_zeros(45)
            basis[2] = 1.
            return state + self.gain * action[:, :1] * basis

    model = Integrator()
    rows = torch.arange(32)
    loss = fit.mixed_state_loss(model, *tensors, ow, nw, rows, rows)
    scale = float(fit.frozen.PHYSICAL_SCALE[2])
    expected = 25.5 * (.1 ** 2 + .2 ** 2) / (2 * scale ** 2 * 3 * 7)
    assert float(loss) == pytest.approx(expected, rel=2e-6)
    gradient, = torch.autograd.grad(loss, model.gain)
    assert float(gradient) == pytest.approx(2 * expected, rel=2e-6)


def test_objective_retains_original_yaw_and_body_parameter_gradients(lib):
    torch, fit = lib.torch, lib.fit
    state, yaw, body = [torch.tensor(v, requires_grad=True) for v in (.2, .3, .4)]
    loss = fit.objective(state, yaw, body)
    reference = fit.previous.objective(state, yaw, body, 'active')
    assert torch.equal(loss, reference)
    assert [float(v) for v in torch.autograd.grad(loss, (state, yaw, body))] == pytest.approx(
        [1., fit.FIXED_LAMBDA, fit.FIXED_LAMBDA])


def test_trainable_copy_leaves_source_and_frozen_state_intact(lib):
    torch, fit = lib.torch, lib.fit
    source = fit.frozen.Dynamics52().eval().requires_grad_(False)
    source.residual[0].weight.grad = torch.ones_like(source.residual[0].weight)
    original = fit._source_identity(source)
    model = fit.clone_trainable(source)
    frozen_before = fit.paired.mapping_sha256(fit.frozen_mapping(model))
    assert all(name.startswith('residual.') for name, p in model.named_parameters() if p.requires_grad)
    assert all(not b.requires_grad for b in model.buffers())
    assert all(p.grad is None for p in model.parameters())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.residual[-1].bias.square().sum().add(model.residual[-1].bias.sum()).backward()
    optimizer.step()
    assert fit.paired.mapping_sha256(model.state_dict()) != original['mapping']
    assert fit.paired.mapping_sha256(fit.frozen_mapping(model)) == frozen_before
    assert fit._source_identity(source) == original


def test_expired_budget_retains_zero_count_receipt_and_refuses_overwrite(lib, tmp_path):
    out = tmp_path / 'expired'
    with pytest.raises(TimeoutError):
        lib.fit.refit_model(None, {}, {}, {}, out, cpu_deadline=0.)
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['status'] == 'INCOMPLETE' and not receipt['eligible_for_evaluation']
    assert receipt['updates_completed'] == receipt['updates_attempted'] == 0
    assert (out / 'failure.txt').exists() and not (out / 'final_state_dict.pt').exists()
    with pytest.raises(FileExistsError):
        lib.fit.refit_model(None, {}, {}, {}, out, cpu_deadline=0.)


def test_failure_before_first_optimizer_step_keeps_attempt_and_partial(lib, monkeypatch, tmp_path):
    fit = lib.fit
    source = fit.frozen.Dynamics52().eval().requires_grad_(False)
    original = fit._source_identity(source)

    def fail(*_args):
        raise RuntimeError('synthetic interrupted update')

    monkeypatch.setattr(fit, 'mixed_state_loss', fail)
    out = tmp_path / 'interrupted'
    with pytest.raises(RuntimeError, match='synthetic interrupted update'):
        fit.refit_model(source, dataset(lib, 12), dataset(lib, 1), label_rows(lib), out,
                        cpu_deadline=time.process_time() + 60.)
    receipt = json.loads((out / 'receipt.json').read_text())
    assert (receipt['updates_completed'], receipt['updates_attempted']) == (0, 1)
    assert receipt['status'] == 'INCOMPLETE' and receipt['source_model_unchanged']
    assert receipt['frozen_mapping_sha256_before'] == receipt['frozen_mapping_sha256_after']
    assert (out / 'partial_state_dict.pt').exists() and (out / 'updates.jsonl').read_text() == ''
    assert json.loads((out / 'events.jsonl').read_text())['event'] == 'UPDATE_BEGIN'
    assert fit._source_identity(source) == original
