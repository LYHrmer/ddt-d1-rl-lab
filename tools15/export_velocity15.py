#!/usr/bin/env python3
"""Export/check the full candidate velocity encoder and actor as a numerical fixture.

Only --self-check is exposed by the CLI. It constructs a random untrained policy;
no checkpoint is read, no learning occurs and no robot model or physics is run.
This is not the completed-checkpoint provenance or qualification export path.
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
OBS_DIM, HISTORY_LENGTH, ACTOR_DIM, ACTION_DIM = 57, 10, 627, 16
FIXTURE_SEED, FIXTURE_COUNT = 20261017, 64
SB3_TORCH_LIMIT, TORCH_ORT_LIMIT = 1e-6, 1e-5
ALLOWED_ONNX_OPS = {'Slice', 'Concat', 'Gemm', 'Tanh', 'Clip', 'Reshape', 'Constant', 'Identity'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


@functools.lru_cache(maxsize=1)
def libraries():
    # Both the CLI and library use CPU single-thread inference, never CUDA.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    for directory in (ROOT / '.python-deps', ROOT / 'training'):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import stable_baselines3 as sb3
    import torch
    from gymnasium import spaces
    from stable_baselines3.common.distributions import DiagGaussianDistribution
    from stable_baselines3.common.policies import ActorCriticPolicy
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return SimpleNamespace(np=np, onnx=onnx, ort=ort, sb3=sb3, torch=torch, spaces=spaces,
                           ActorCriticPolicy=ActorCriticPolicy, Gaussian=DiagGaussianDistribution)


def check_policy(policy):
    lib = libraries()
    np, torch, spaces = lib.np, lib.torch, lib.spaces
    if not isinstance(policy, lib.ActorCriticPolicy):
        raise ValueError('expected an SB3 ActorCriticPolicy')
    obs = policy.observation_space
    if not isinstance(obs, spaces.Dict) or set(obs.spaces) != {'actor', 'velocity_target'}:
        raise ValueError('policy signature requires Dict actor[627], velocity_target[3]')
    for name, size in (('actor', ACTOR_DIM), ('velocity_target', 3)):
        space = obs[name]
        if not isinstance(space, spaces.Box) or space.shape != (size,) or space.dtype != np.float32:
            raise ValueError('policy signature requires float32 actor[627], velocity_target[3]')
    actions = policy.action_space
    if (not isinstance(actions, spaces.Box) or actions.shape != (ACTION_DIM,)
            or actions.dtype != np.float32 or not np.all(actions.low == -1)
            or not np.all(actions.high == 1) or policy.squash_output
            or not isinstance(policy.action_dist, lib.Gaussian) or policy.use_sde):
        raise ValueError('policy requires unsquashed Gaussian float32 Box[-1,1] actions[16]')

    def sequential(module, dimensions, name):
        expected = []
        for index, (incoming, outgoing) in enumerate(zip(dimensions, dimensions[1:])):
            expected.append((torch.nn.Linear, incoming, outgoing))
            if index < len(dimensions) - 2 or name != 'encoder':
                expected.append((torch.nn.Tanh, None, None))
        if not isinstance(module, torch.nn.Sequential) or len(module) != len(expected):
            raise ValueError(f'{name} has the wrong complete architecture')
        for layer, (kind, incoming, outgoing) in zip(module, expected):
            if type(layer) is not kind or (kind is torch.nn.Linear and
                    (layer.in_features, layer.out_features) != (incoming, outgoing)):
                raise ValueError(f'{name} has the wrong layer or dimensions')

    extractor = policy.mlp_extractor
    sequential(getattr(extractor, 'encoder', None), [627, 128, 64, 3], 'encoder')
    sequential(extractor.policy_net, [60, 128, 128], 'policy_net')
    sequential(extractor.value_net, [627, 128, 128], 'value_net')
    if type(policy.action_net) is not torch.nn.Linear or (
            policy.action_net.in_features, policy.action_net.out_features) != (128, 16):
        raise ValueError('action head must be Linear(128,16)')
    for name, tensor in policy.state_dict().items():
        if tensor.device.type != 'cpu' or (tensor.is_floating_point() and
                (tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()))):
            raise ValueError(f'policy contains a nonfinite or non-CPU-float32 tensor: {name}')


def actor_module(policy):
    torch = libraries().torch

    class CompleteVelocityActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = policy.mlp_extractor.encoder
            self.policy_net = policy.mlp_extractor.policy_net
            self.action_head = policy.action_net

        def forward(self, obs, history):
            features = torch.cat((obs.reshape(1, 57), history.reshape(1, 570)), dim=1)
            estimate = self.encoder(features)
            latent = self.policy_net(torch.cat((features[:, :57], estimate), dim=1))
            return torch.clamp(self.action_head(latent), -1., 1.)

    return CompleteVelocityActor()


def onnx_signature(model):
    def item(value):
        tensor = value.type.tensor_type
        return {'name': value.name, 'elem_type': tensor.elem_type,
                'shape': [d.dim_value for d in tensor.shape.dim]}
    return {'inputs': [item(x) for x in model.graph.input],
            'outputs': [item(x) for x in model.graph.output]}


def check_onnx(path):
    lib = libraries()
    model = lib.onnx.load(str(path))
    if [(x.domain, x.version) for x in model.opset_import] != [('', 11)] or model.ir_version != 7:
        raise ValueError('ONNX requires exactly opset 11 and IR 7')
    if any(node.domain not in ('',) for node in model.graph.node):
        raise ValueError('ONNX custom operator domain is not allowed')
    counts = Counter(node.op_type for node in model.graph.node)
    if not set(counts).issubset(ALLOWED_ONNX_OPS):
        raise ValueError(f'ONNX contains an unsupported operation: {sorted(counts)}')
    signature = onnx_signature(model)
    expected = {'inputs': [
        {'name': 'obs', 'elem_type': lib.onnx.TensorProto.FLOAT, 'shape': [1, 57]},
        {'name': 'history', 'elem_type': lib.onnx.TensorProto.FLOAT, 'shape': [1, 10, 57]}],
        'outputs': [{'name': 'actions', 'elem_type': lib.onnx.TensorProto.FLOAT, 'shape': [1, 16]}]}
    if signature != expected:
        raise ValueError(f'ONNX signature differs: {signature}')
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == lib.onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == lib.onnx.AttributeProto.FLOAT and not lib.np.isfinite(attribute.f):
                raise ValueError('ONNX has a nonfinite floating attribute')
            elif attribute.type == lib.onnx.AttributeProto.FLOATS and not lib.np.isfinite(attribute.floats).all():
                raise ValueError('ONNX has nonfinite floating attributes')
    for tensor in tensors:
        array = lib.onnx.numpy_helper.to_array(tensor)
        if not lib.np.issubdtype(array.dtype, lib.np.number) or not lib.np.isfinite(array).all():
            raise ValueError('ONNX has a nonfinite or nonnumeric tensor')
    lib.onnx.checker.check_model(model)
    return dict(sorted(counts.items())), signature


def fixtures():
    np = libraries().np
    rng = np.random.default_rng(FIXTURE_SEED)
    obs = rng.normal(0., .5, (FIXTURE_COUNT, 57)).astype(np.float32)
    history = rng.normal(0., .5, (FIXTURE_COUNT, 10, 57)).astype(np.float32)
    obs[0] = 0.
    history[0] = 0.
    obs[1] = obs[0]  # Paired history-only change; the previous frames remain distinct.
    return obs, history


def golden_parity(actor, policy, onnx_path, inputs=None):
    lib = libraries()
    np, torch = lib.np, lib.torch
    obs, history = fixtures() if inputs is None else inputs
    if (obs.shape != (64, 57) or history.shape != (64, 10, 57)
            or obs.dtype != np.float32 or history.dtype != np.float32
            or not np.isfinite(obs).all() or not np.isfinite(history).all()):
        raise ValueError('fixtures must be finite float32 with the fixed 64 signatures')
    options = lib.ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = lib.ort.ExecutionMode.ORT_SEQUENTIAL
    session = lib.ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    if ([(x.name, x.shape, x.type) for x in session.get_inputs()] != [
            ('obs', [1, 57], 'tensor(float)'), ('history', [1, 10, 57], 'tensor(float)')]
            or [(x.name, x.shape, x.type) for x in session.get_outputs()] != [('actions', [1, 16], 'tensor(float)')]):
        raise ValueError('ORT did not expose the exact batch-one signature')
    rows = []
    max_sb3_torch = max_torch_ort = 0.
    actor_target_delta = critic_target_delta = 0.

    def finite(value, shape, label):
        array = np.asarray(value)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'nonfinite or wrong-shaped {label}')
        return array

    with torch.no_grad():
        for index in range(64):
            one_obs, one_history = obs[index:index + 1], history[index:index + 1]
            flat = np.concatenate((one_obs, one_history.reshape(1, 570)), axis=1)
            base = {'actor': flat, 'velocity_target': np.zeros((1, 3), dtype=np.float32)}
            altered = {'actor': flat.copy(), 'velocity_target': np.array([[37., -19., 8.]], dtype=np.float32)}
            sb3_action = finite(policy.predict(base, deterministic=True)[0], (1, 16), 'SB3 action')
            altered_action = finite(policy.predict(altered, deterministic=True)[0], (1, 16), 'altered-target action')
            base_tensors, _ = policy.obs_to_tensor(base)
            altered_tensors, _ = policy.obs_to_tensor(altered)
            value = finite(policy.predict_values(base_tensors).cpu().numpy(), (1, 1), 'critic value')
            altered_value = finite(policy.predict_values(altered_tensors).cpu().numpy(), (1, 1), 'altered-target value')
            actor_target_delta = max(actor_target_delta, float(np.max(np.abs(sb3_action - altered_action))))
            critic_target_delta = max(critic_target_delta, float(np.max(np.abs(value - altered_value))))
            if not np.array_equal(sb3_action, altered_action) or not np.array_equal(value, altered_value):
                raise ValueError('velocity_target leaked into actor or critic inference')
            torch_action = finite(actor(torch.from_numpy(one_obs), torch.from_numpy(one_history)).cpu().numpy(),
                                  (1, 16), 'Torch complete actor output')
            ort_action = finite(session.run(['actions'], {'obs': one_obs, 'history': one_history})[0],
                                (1, 16), 'ORT complete actor output')
            max_sb3_torch = max(max_sb3_torch, float(np.max(np.abs(sb3_action - torch_action))))
            max_torch_ort = max(max_torch_ort, float(np.max(np.abs(torch_action - ort_action))))
            if max_sb3_torch > SB3_TORCH_LIMIT or max_torch_ort > TORCH_ORT_LIMIT:
                raise ValueError(f'complete encoder parity failed: SB3/Torch={max_sb3_torch}, Torch/ORT={max_torch_ort}')
            rows.append({'index': index, 'obs': one_obs[0].tolist(),
                         'history_oldest_to_newest_previous_only': one_history[0].tolist(),
                         'sb3_actions': sb3_action[0].tolist(), 'torch_actions': torch_action[0].tolist(),
                         'python_ort_actions': ort_action[0].tolist()})
    return rows, {'sb3_predict_vs_torch_max_abs_error': max_sb3_torch,
                  'torch_vs_python_ort_max_abs_error': max_torch_ort,
                  'target_actor_max_abs_change': actor_target_delta,
                  'target_critic_max_abs_change': critic_target_delta,
                  'history_only_pair_max_action_change': float(np.max(np.abs(
                      np.asarray(rows[1]['torch_actions']) - np.asarray(rows[0]['torch_actions'])))),
                  'prediction_batch_size': 1, 'fixtures': 64}


def policy_state_sha256(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def export_actor(policy, output):
    """Export a supplied CPU policy for numerical checks, never certify provenance."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-velocity-actor-numerical-fixture-v15', 'status': 'FAIL',
              'artifact_role': 'numerical_fixture', 'release_ready': False,
              'qualification_eligible': False, 'completed_checkpoint_provenance_checked': False,
              'checkpoint_loaded_by_exporter': False, 'training_started': False,
              'physics_steps_executed': 0, 'robot_models_constructed': 0,
              'entry_sha256': sha(__file__), 'error': None,
              'history_order': 'oldest_to_newest_previous_only',
              'full_actor_path': 'obs+history -> encoder -> current57+estimated_velocity3 -> actor -> action_head -> clip',
              'absolute_limits': {'sb3_vs_torch': SB3_TORCH_LIMIT, 'torch_vs_ort': TORCH_ORT_LIMIT},
              'cxx_ort_1_10_status': 'NOT_TESTED', 'finite_inputs_required_by_runtime': True}
    was_training = None
    try:
        lib = libraries()
        torch = lib.torch
        check_policy(policy)
        report['policy_state_sha256'] = policy_state_sha256(policy)
        source = Path(inspect.getfile(type(policy))).resolve()
        report['policy_type'] = type(policy).__module__ + '.' + type(policy).__qualname__
        report['policy_source_sha256'] = sha(source)
        report['declared_random_fixture_seed'] = getattr(policy, '_numerical_fixture_seed15', None)
        was_training = policy.training
        policy.set_training_mode(False)
        actor = actor_module(policy).eval()
        onnx_path = output / 'candidate_actor.onnx'
        with torch.no_grad():
            torch.onnx.export(actor, (torch.zeros((1, 57)), torch.zeros((1, 10, 57))), str(onnx_path),
                              input_names=['obs', 'history'], output_names=['actions'], opset_version=11,
                              dynamo=False, do_constant_folding=True)
        model = lib.onnx.load(str(onnx_path))
        model.ir_version = 7
        lib.onnx.save(model, str(onnx_path))
        counts, signature = check_onnx(onnx_path)
        rows, errors = golden_parity(actor, policy, onnx_path)
        fixture_path = output / 'golden_fixtures.json'
        write_json(fixture_path, {'format': 'd1-velocity-two-input-golden-64-v15',
                                 'seed': FIXTURE_SEED, 'count': 64, 'input_order': ['obs', 'history'],
                                 'history_order': report['history_order'], 'fixtures': rows})
        if policy_state_sha256(policy) != report['policy_state_sha256']:
            raise ValueError('policy weights changed during numerical export')
        report.update(status='PASS', onnx_ir_version=7, onnx_opset=11, onnx_node_counts=counts,
                      onnx_signature=signature, parity=errors,
                      artifact_sha256={'candidate_actor.onnx': sha(onnx_path), 'golden_fixtures.json': sha(fixture_path)},
                      versions={'torch': torch.__version__, 'sb3': lib.sb3.__version__,
                                'onnx': lib.onnx.__version__, 'python_ort': lib.ort.__version__},
                      cpu_threads={'torch_intra': torch.get_num_threads(), 'torch_inter': torch.get_num_interop_threads(),
                                   'ort_intra': 1, 'ort_inter': 1})
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if was_training is not None:
            policy.set_training_mode(was_training)
        write_json(output / 'numerical_fixture.json', report)
    return report


def make_random_policy():
    lib = libraries()
    from d1_rl.velocity_policy15 import VelocityActorCriticPolicy
    with lib.torch.random.fork_rng(devices=[]):
        lib.torch.manual_seed(FIXTURE_SEED)
        spaces = lib.spaces
        policy = VelocityActorCriticPolicy(spaces.Dict({
            'actor': spaces.Box(-lib.np.inf, lib.np.inf, (627,), dtype=lib.np.float32),
            'velocity_target': spaces.Box(-lib.np.inf, lib.np.inf, (3,), dtype=lib.np.float32)}),
            spaces.Box(-1., 1., (16,), dtype=lib.np.float32), lambda _: 3e-4)
    policy._numerical_fixture_seed15 = FIXTURE_SEED
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True,
                        help='only a random untrained numerical fixture; no checkpoint export')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = export_actor(make_random_policy(), args.output)
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}', 'physics_steps_executed': 0}))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
