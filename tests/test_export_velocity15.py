"""Random-network numerical checks only: no environment, learning or physics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'tools15/export_velocity15.py'


@pytest.fixture(scope='module')
def entry():
    spec = importlib.util.spec_from_file_location('tested_export_velocity15', ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def lib(entry):
    for directory in (ROOT / '.python-deps', ROOT / 'training'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    # The static CI job deliberately lacks these optional numerical dependencies.
    # The dedicated CPU candidate job installs and runs all of these tests.
    for name in ('torch', 'onnx', 'onnxruntime', 'stable_baselines3', 'gymnasium'):
        pytest.importorskip(name)
    return entry.libraries()


@pytest.fixture(scope='module')
def exported(entry, lib, tmp_path_factory):
    output = tmp_path_factory.mktemp('velocity15_export') / 'random_fixture'
    policy = entry.make_random_policy()
    previous_mode = policy.training
    report = entry.export_actor(policy, output)
    assert policy.training is previous_mode
    return SimpleNamespace(output=output, report=report, policy=policy)


@pytest.fixture
def policy(entry, lib):
    return entry.make_random_policy()


def test_module_import_is_stdlib_only():
    code = ('import importlib.util,sys; '
            f's=importlib.util.spec_from_file_location("light_export",{str(ENTRY)!r}); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'assert not {"torch","onnx","onnxruntime","stable_baselines3","mujoco"} & sys.modules.keys()')
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_cli_requires_explicit_self_check_and_accepts_no_checkpoint(tmp_path):
    output = tmp_path / 'must_not_exist'
    for argv in (['--output', str(output)],
                 ['--self-check', '--output', str(output), '--checkpoint', 'partial_policy.zip']):
        result = subprocess.run([sys.executable, '-B', str(ENTRY), *argv],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 2 and not output.exists()


def test_existing_output_is_rejected_before_dependencies_or_policy_are_touched(entry, tmp_path, monkeypatch):
    output = tmp_path / 'already_exists'
    output.mkdir()
    marker = output / 'numerical_fixture.json'
    marker.write_text('keep')
    def forbidden():
        raise AssertionError('must reject output before importing numerical dependencies')
    monkeypatch.setattr(entry, 'libraries', forbidden)
    with pytest.raises(FileExistsError):
        entry.export_actor(object(), output)
    assert marker.read_text() == 'keep'


def test_full_encoder_export_has_the_exact_signature_and_unchanged_gates(entry, exported, lib):
    report = exported.report
    assert report['status'] == 'PASS' and report['artifact_role'] == 'numerical_fixture'
    assert report['absolute_limits'] == {'sb3_vs_torch': 1e-6, 'torch_vs_ort': 1e-5}
    assert report['onnx_ir_version'] == 7 and report['onnx_opset'] == 11
    assert set(report['onnx_node_counts']).issubset(entry.ALLOWED_ONNX_OPS)
    assert 'Slice' in report['onnx_node_counts']
    assert report['onnx_node_counts']['Gemm'] == 6  # encoder 3, actor 2, head 1; no critic.
    assert [row['name'] for row in report['onnx_signature']['inputs']] == ['obs', 'history']
    assert report['parity']['fixtures'] == 64 and report['parity']['prediction_batch_size'] == 1
    assert report['parity']['sb3_predict_vs_torch_max_abs_error'] <= 1e-6
    assert report['parity']['torch_vs_python_ort_max_abs_error'] <= 1e-5
    assert report['cpu_threads'] == {'torch_intra': 1, 'torch_inter': 1, 'ort_intra': 1, 'ort_inter': 1}
    assert report['release_ready'] is report['qualification_eligible'] is False
    assert report['completed_checkpoint_provenance_checked'] is False
    assert report['checkpoint_loaded_by_exporter'] is report['training_started'] is False
    assert report['physics_steps_executed'] == report['robot_models_constructed'] == 0
    assert report['cxx_ort_1_10_status'] == 'NOT_TESTED'
    assert not (exported.output / 'metadata.json').exists()
    assert json.loads((exported.output / 'numerical_fixture.json').read_text()) == report
    for name, expected in report['artifact_sha256'].items():
        assert entry.sha(exported.output / name) == expected
    model = lib.onnx.load(str(exported.output / 'candidate_actor.onnx'))
    assert model.ir_version == 7


def test_training_targets_do_not_change_actor_or_critic_and_history_has_an_effect(exported):
    assert exported.report['parity']['target_actor_max_abs_change'] == 0.
    assert exported.report['parity']['target_critic_max_abs_change'] == 0.
    assert exported.report['parity']['history_only_pair_max_action_change'] > 1e-8
    rows = json.loads((exported.output / 'golden_fixtures.json').read_text())['fixtures']
    assert len(rows) == 64 and rows[0]['obs'] == rows[1]['obs']
    assert rows[0]['history_oldest_to_newest_previous_only'] != rows[1]['history_oldest_to_newest_previous_only']


def test_target_leak_in_actor_features_is_rejected(entry, lib, policy, exported, monkeypatch):
    original = policy.features_extractor.forward
    def leaking(observations):
        features = original(observations).clone()
        features[:, :3] += observations['velocity_target']
        return features
    monkeypatch.setattr(policy.features_extractor, 'forward', leaking)
    with pytest.raises(ValueError, match='target leaked'):
        entry.golden_parity(entry.actor_module(policy), policy, exported.output / 'candidate_actor.onnx')


def test_target_leak_in_critic_only_is_rejected(entry, policy, exported, monkeypatch):
    original = policy.predict_values
    monkeypatch.setattr(policy, 'predict_values',
                        lambda observations: original(observations) + observations['velocity_target'].sum(dim=1, keepdim=True))
    with pytest.raises(ValueError, match='target leaked'):
        entry.golden_parity(entry.actor_module(policy), policy, exported.output / 'candidate_actor.onnx')


def test_a_graph_that_skips_the_encoder_is_rejected_by_numerical_parity(entry, lib, policy, tmp_path):
    torch = lib.torch
    class WrongActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = policy.mlp_extractor.policy_net
            self.head = policy.action_net

        def forward(self, obs, history):
            # Preserves both signatures and allowed ops, but bypasses all encoder layers.
            wrong_velocity = history.reshape(1, 570)[:, :3]
            return torch.clamp(self.head(self.actor(torch.cat((obs.reshape(1, 57), wrong_velocity), dim=1))), -1., 1.)

    policy.set_training_mode(False)
    path = tmp_path / 'missing_encoder.onnx'
    torch.onnx.export(WrongActor().eval(), (torch.zeros((1, 57)), torch.zeros((1, 10, 57))), str(path),
                      input_names=['obs', 'history'], output_names=['actions'], opset_version=11,
                      dynamo=False, do_constant_folding=True)
    model = lib.onnx.load(str(path))
    model.ir_version = 7
    lib.onnx.save(model, str(path))
    entry.check_onnx(path)  # A valid signature/allowed-op check is insufficient by itself.
    with pytest.raises(ValueError, match='encoder parity failed'):
        entry.golden_parity(entry.actor_module(policy).eval(), policy, path)


@pytest.mark.parametrize('fault', ['encoder_missing', 'encoder_shape', 'action_head', 'double_precision', 'nan_weight'])
def test_invalid_policy_architecture_or_parameters_produces_a_failed_receipt(entry, lib, policy, tmp_path, fault):
    torch = lib.torch
    if fault == 'encoder_missing':
        del policy.mlp_extractor.encoder
    elif fault == 'encoder_shape':
        policy.mlp_extractor.encoder[0] = torch.nn.Linear(626, 128)
    elif fault == 'action_head':
        policy.action_net = torch.nn.Linear(128, 15)
    elif fault == 'double_precision':
        policy.double()
    else:
        with torch.no_grad():
            policy.mlp_extractor.encoder[0].weight[0, 0] = float('nan')
    output = tmp_path / fault
    with pytest.raises(ValueError):
        entry.export_actor(policy, output)
    report = json.loads((output / 'numerical_fixture.json').read_text())
    assert report['status'] == 'FAIL' and report['error']
    assert report['release_ready'] is False and not (output / 'golden_fixtures.json').exists()


@pytest.mark.parametrize('fault', ['obs_shape', 'output_shape', 'float64_input', 'opset', 'ir', 'custom_domain', 'nan_initializer'])
def test_invalid_onnx_signature_version_domain_and_nonfinite_data_are_rejected(entry, lib, exported, tmp_path, fault):
    model = lib.onnx.load(str(exported.output / 'candidate_actor.onnx'))
    if fault == 'obs_shape':
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 58
    elif fault == 'output_shape':
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 15
    elif fault == 'float64_input':
        model.graph.input[0].type.tensor_type.elem_type = lib.onnx.TensorProto.DOUBLE
    elif fault == 'opset':
        model.opset_import[0].version = 12
    elif fault == 'ir':
        model.ir_version = 8
    elif fault == 'custom_domain':
        model.graph.node[0].domain = 'unreviewed'
    else:
        tensor = model.graph.initializer[0]
        array = lib.onnx.numpy_helper.to_array(tensor).copy()
        array.flat[0] = lib.np.nan
        tensor.CopyFrom(lib.onnx.numpy_helper.from_array(array, tensor.name))
    path = tmp_path / (fault + '.onnx')
    lib.onnx.save(model, str(path))
    with pytest.raises(ValueError):
        entry.check_onnx(path)


@pytest.mark.parametrize('fault', ['nan_obs', 'nan_history', 'wrong_dtype', 'wrong_count'])
def test_invalid_fixtures_are_rejected_before_inference(entry, lib, exported, fault):
    obs, history = entry.fixtures()
    if fault == 'nan_obs':
        obs[0, 0] = lib.np.nan
    elif fault == 'nan_history':
        history[0, 0, 0] = lib.np.nan
    elif fault == 'wrong_dtype':
        obs = obs.astype(lib.np.float64)
    else:
        obs = obs[:1]
    with pytest.raises(ValueError, match='fixtures must be finite'):
        entry.golden_parity(entry.actor_module(exported.policy), exported.policy,
                            exported.output / 'candidate_actor.onnx', (obs, history))


def test_nonfinite_sb3_output_is_rejected(entry, lib, policy, exported, monkeypatch):
    monkeypatch.setattr(policy, 'predict', lambda *args, **kwargs: (lib.np.full((1, 16), lib.np.nan), None))
    with pytest.raises(ValueError, match='nonfinite.*SB3 action'):
        entry.golden_parity(entry.actor_module(policy), policy, exported.output / 'candidate_actor.onnx')


def test_sb3_torch_limit_is_not_relaxed(entry, policy, exported):
    actor = entry.actor_module(policy)
    with pytest.raises(ValueError, match='encoder parity failed'):
        entry.golden_parity(lambda obs, history: actor(obs, history) + 2e-6,
                            policy, exported.output / 'candidate_actor.onnx')


def test_torch_ort_limit_is_not_relaxed(entry, lib, policy, exported, monkeypatch):
    original = lib.ort.InferenceSession
    class NoisySession:
        def __init__(self, *args, **kwargs):
            self.session = original(*args, **kwargs)

        def get_inputs(self):
            return self.session.get_inputs()

        def get_outputs(self):
            return self.session.get_outputs()

        def run(self, *args, **kwargs):
            return [self.session.run(*args, **kwargs)[0] + 2e-5]

    monkeypatch.setattr(lib.ort, 'InferenceSession', NoisySession)
    with pytest.raises(ValueError, match='encoder parity failed'):
        entry.golden_parity(entry.actor_module(policy), policy, exported.output / 'candidate_actor.onnx')
