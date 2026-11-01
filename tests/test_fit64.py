"""Synthetic64 label-source, quantization, global-loss and pairing contracts.

No robot assets, physics, original data or training loop are used. Source62's
old coverage rejection is deliberately retained in the synthetic admission.
"""
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
    from tools64 import fit64
    return SimpleNamespace(np=np, torch=torch, fit=fit64)


@pytest.fixture(scope='module')
def source_files(lib, tmp_path_factory):
    np, fit = lib.np, lib.fit
    parent = tmp_path_factory.mktemp('synthetic64_sources')
    direction = np.zeros(16, dtype=np.float64)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    deltas = [0., 0., *(sign * epsilon for epsilon in fit.EPSILONS for sign in (1, -1))]
    outputs = {}
    for source in fit.SOURCE_KEYS:
        output = parent / source
        output.mkdir()
        outputs[source] = output
        report = {'status': 'COMPLETE', 'counts': {'policy_transitions': 2640 if source == '60' else 2520},
                  'all_four_complete_baselines_exact': True, 'all_four_complete_initial_states_exact': True,
                  'labels_admitted': 12, 'labels_admitted_by_center': {'clean': 12, 'noisy': 12},
                  'cases': [{'seed': seed, 'status': 'COMPLETE', 'steps': 600} for seed in fit.LABEL_SEEDS],
                  'hypothesis': {'status': 'REJECTED', 'coverage': {'status': 'REJECTED', 'supported': False}},
                  'points': []}
        suffix = '_clean' if source == '60' else ''
        for index, (seed, tick) in enumerate((seed, tick) for seed in fit.LABEL_SEEDS for tick in fit.LABEL_TICKS):
            state = np.zeros((1, 45), dtype=np.float64)
            state[0, 3], state[0, 28] = 1., 10.
            state[0, 2] = 1. + index / 20. + (2. if source == '62' else 0.)
            base = np.zeros((1, 16), dtype=np.float32)
            base[0, [3, 7, 11, 15]] = (.1, -.2, .3, -.4)
            base[0, 0] = .02 if source in ('60', '62') else 0.
            rate = -.0004 if index == 0 else (-1 if index % 2 == 0 else 1) * (.003 + index * .001)
            targets = []
            for order, delta in enumerate(deltas):
                action = np.clip(base.astype(np.float64) + delta * direction, -1., 1.).astype(np.float32)
                target = state.copy()
                target[0, 28] += delta * rate
                targets.append(target)
                np.savez(output / f'branch_{index:02d}{suffix}_{order:02d}.npz',
                         s=state, applied_action=action.astype(np.float64), target=target)
            response = np.stack([(targets[2 + 2 * j] - targets[3 + 2 * j])[0] / (2 * epsilon)
                                 for j, epsilon in enumerate(fit.EPSILONS)])
            np.savez(output / f'response_{index:02d}{suffix}.npz', central_state45_fd=response)
            point = {'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'label_admitted': True,
                     'branches': [{'status': 'COMPLETE', 'delta': delta} for delta in deltas]}
            if source == '60':
                report['points'].append({'seed': seed, 'tick': tick, 'status': 'COMPLETE', 'centers': {'clean': point}})
            else:
                report['points'].append(point)
        (output / 'report.json').write_text(json.dumps(report))
    return outputs


@pytest.fixture(scope='module')
def sources(lib, source_files):
    return lib.fit.load_label_sources(*(source_files[key] for key in lib.fit.SOURCE_KEYS))


def test_all_sources_preserve_signed_float64_labels_and_actual_action_directions(lib, source_files, sources):
    np, fit = lib.np, lib.fit
    assert set(sources) == {'58', '60', '62'}
    for source, labels in sources.items():
        assert labels['s'].shape == (48, 45)
        assert labels['target'].dtype == np.float64 and labels['v'].dtype == np.float32
        assert np.any(labels['target'] < 0) and np.any(labels['target'] > 0)
        suffix = '_clean' if source == '60' else ''
        with np.load(source_files[source] / f'branch_00{suffix}_02.npz') as plus, \
                np.load(source_files[source] / f'branch_00{suffix}_03.npz') as minus:
            assert plus['target'][0, 28].astype(np.float32) == minus['target'][0, 28].astype(np.float32)
        assert labels['target'][0] < 0 and labels['target'][0].astype(np.float32) != 0
        expected = ((labels['a_plus'].astype(np.float64) - labels['a_minus'].astype(np.float64))
                    / (2 * labels['epsilon'][:, None])).astype(np.float32)
        assert np.array_equal(labels['v'], expected)
        assert labels['v'][0, 3] != np.float32(1.)
    # The old62 all-negative coverage rejection is not rewritten or used to filter.
    assert json.loads((source_files['62'] / 'report.json').read_text())['hypothesis']['status'] == 'REJECTED'
    assert len(sources['62']['target']) == 48
    assert fit.FIXED_LAMBDA == 0.08980082414438752 and type(fit.FIXED_LAMBDA) is float


def test_composition_changes_only_the_second_complete_source_and_reports_distinct_states(lib, sources):
    np, fit = lib.np, lib.fit
    control, active = (fit.compose_labels(sources, arm) for arm in fit.ARMS)
    for key in fit.LABEL_KEYS:
        assert np.array_equal(control[key][:48], sources['58'][key])
        assert np.array_equal(active[key][:48], sources['58'][key])
        assert np.array_equal(control[key][48:], sources['60'][key])
        assert np.array_equal(active[key][48:], sources['62'][key])
    assert np.array_equal(control['source_id'], np.r_[np.full(48, 58), np.full(48, 60)])
    assert np.array_equal(active['source_id'], np.r_[np.full(48, 58), np.full(48, 62)])
    assert len(np.unique(control['s'], axis=0)) == 12
    assert len(np.unique(active['s'], axis=0)) == 24
    assert np.array_equal(control['s'][:48], control['s'][48:])
    assert not np.array_equal(control['a0'][:48], control['a0'][48:])


@pytest.mark.parametrize('corruption', ['dropped_row', 'duplicate_eps', 'wrong_source', 'wrong_state'])
def test_composition_rejects_missing_grid_wrong_provenance_and_unmatched_control_states(lib, sources, corruption):
    fit = lib.fit
    labels = fit.compose_labels(sources, 'control')
    if corruption == 'dropped_row':
        labels = {key: value[:-1] for key, value in labels.items()}
    elif corruption == 'duplicate_eps':
        labels['epsilon'][49] = labels['epsilon'][48]
    elif corruption == 'wrong_source':
        labels['source_id'][48:] = 62
    else:
        labels['s'][48:52, 2] += .1
    with pytest.raises(ValueError):
        fit.validate_labels(labels, 'control')


def test_partial_source_is_not_accepted_even_when_the_question_is_new(lib, tmp_path):
    output = tmp_path / 'partial62'
    output.mkdir()
    (output / 'report.json').write_text(json.dumps({'status': 'INCOMPLETE'}))
    with pytest.raises(ValueError, match='complete60/62'):
        lib.fit.load_labels62(output)


def test_96_row_global_mean_retains_mixed_parameter_gradients_and_the_second_half(lib):
    torch, fit = lib.torch, lib.fit

    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, state, action):
            basis = torch.nn.functional.one_hot(torch.tensor(28), 45).to(state)
            return state + (self.gain * (1 + state[:, 2]) * action[:, 3])[:, None] * basis

    model = Affine()
    state = torch.zeros((96, 45))
    state[:, 2] = torch.linspace(.1, 1., 96)
    direction = torch.zeros((96, 16))
    direction[:, 3] = 1.
    target = torch.cat((torch.full((48,), -.3), torch.full((48,), .1)))
    labels = {'s': state, 'a0': torch.zeros((96, 16)), 'v': direction, 'target': target}
    loss = fit.auxiliary_loss(model, labels, create_graph=True)
    derivative, = torch.autograd.grad(loss, model.gain)
    factor = 1 + state[:, 2]
    scale = float(fit.frozen.PHYSICAL_SCALE[28])
    expected = ((model.gain.detach() * factor - target) / scale).square().mean()
    expected_gradient = 2 * ((model.gain.detach() * factor - target) * factor).mean() / scale ** 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(derivative, expected_gradient)
    assert derivative.abs() > 0
    changed = dict(labels, target=target.clone())
    changed['target'][48:] += 10.
    assert not torch.equal(fit.auxiliary_loss(model, changed, create_graph=False), loss.detach())
    assert fit.auxiliary_loss(model, labels, create_graph=False).requires_grad is False


def test_initialization_and_sampling_remain_paired_without_label_calibration(lib):
    np, torch, fit = lib.np, lib.torch, lib.fit
    state = np.zeros((40, 45), dtype=np.float32)
    state[:, 3], state[:, 2] = 1., 1.
    action = np.zeros((40, 16), dtype=np.float32)
    done = np.zeros(40, dtype=np.bool_)
    done[-1] = True
    data = {'s': state, 'a': action, 'target': state.copy(), 'episode': np.zeros(40, dtype=np.int64),
            'tick': np.arange(40, dtype=np.int64), 'done': done}
    first, ridge_a = fit.initialize_model(data, 64001)
    torch.rand(20)
    second, ridge_b = fit.initialize_model(data, 64001)
    third, ridge_c = fit.initialize_model(data, 64002)
    assert ridge_a == ridge_b == ridge_c
    assert fit.paired.mapping_sha256(first.state_dict()) == fit.paired.mapping_sha256(second.state_dict())
    assert fit.paired.mapping_sha256(first.state_dict()) != fit.paired.mapping_sha256(third.state_dict())
    assert torch.equal(first(torch.tensor(state[:2]), torch.tensor(action[:2])),
                       third(torch.tensor(state[:2]), torch.tensor(action[:2])))
    first_rows = fit.paired.sampled_window_rows(6828, updates=3, batch_size=64)
    torch.manual_seed(64003)
    torch.rand(20)
    assert torch.equal(first_rows, fit.paired.sampled_window_rows(6828, updates=3, batch_size=64))


def test_expired_admission_creates_no_fit_and_does_no_initialization(lib, monkeypatch, tmp_path):
    fit = lib.fit
    monkeypatch.setattr(fit.time, 'process_time', lambda: 10.)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('expired budget cannot initialize')

    monkeypatch.setattr(fit, 'initialize_model', forbidden)
    output = tmp_path / 'expired64'
    with pytest.raises(TimeoutError, match='before fit admission'):
        fit.fit_model({}, {}, 64001, 'control', output, cpu_deadline=9.)
    assert not output.exists()
