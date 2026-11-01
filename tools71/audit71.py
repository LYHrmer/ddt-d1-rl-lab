"""Training-only71 value/gradient audit; no optimizer, physics, actor or gate.

Caller binds complete69 weights, original53 train and all68 labels/raw zeros,
owns resource admission, persists partials, and serializes all six models.
Gradient geometry is local Euclidean geometry, not Adam/clip update prediction.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager

import numpy as np
import torch

from tools52 import dynamics52 as frozen
from tools54 import numerics54 as numeric
from tools56 import fit56 as paired
from tools69 import fit69 as fit

LOSSES = ('state', 'yaw', 'body', 'anchor')
require = frozen.require


def loss_gradient(loss, parameters, *, unused=None, parameter_names=None, require_all_used=False):
    """Detached F64 flattened gradient; never writes parameter.grad."""
    require(loss.ndim == 0 and bool(torch.isfinite(loss)) and loss.requires_grad,
            'finite scalar loss with retained parameter graph required')
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    missing = [index for index, gradient in enumerate(gradients) if gradient is None]
    if unused is not None:
        unused.update(indices=missing, names=None if parameter_names is None else [parameter_names[i] for i in missing])
    require(len(missing) < len(parameters), 'loss is disconnected from all model parameters')
    require(not require_all_used or not missing, 'state/anchor loss has unused model parameters')
    result = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1).to(torch.float64)
                        for p, g in zip(parameters, gradients, strict=True)])
    require(bool(torch.isfinite(result).all()), 'nonfinite diagnostic gradient')
    return result


def weighted_batch_gradients(parameters, count, loss_at, deadline, *, batch_size=64,
                             partial=None, arrays=None, parameter_names=None):
    """Global mean of chunk means, weighted by actual n/count including tail.

    loss_at(begin,end) constructs one F32 chunk graph. Each gradient is converted
    to F64 before weighting/accumulation; no graph survives its chunk.
    """
    require(count > 0 and batch_size > 0, 'positive sample and chunk counts required')
    total = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
    weighted, sizes = [], []
    partial = {} if partial is None else partial
    partial.update(status='INCOMPLETE', completed_windows=0, completed_batches=0, batch_sizes=[],
                   weighted_loss_sum=0., accumulation_denominator_windows=count)
    for begin in range(0, count, batch_size):
        paired._deadline(deadline, 'state loss/gradient chunks')
        end = min(begin + batch_size, count)
        loss = loss_at(begin, end)
        unused = {}
        gradient = loss_gradient(loss, parameters, unused=unused, parameter_names=parameter_names, require_all_used=True)
        total += gradient * ((end - begin) / count)
        weighted.append(float(loss.detach()) * (end - begin))
        sizes.append(end - begin)
        if arrays is not None:
            arrays['gradient_state_completed_prefix64'] = total.numpy().copy()
        partial.update(completed_windows=end, completed_batches=len(sizes), batch_sizes=sizes.copy(),
                       weighted_loss_sum=math.fsum(weighted), unused_parameters=unused,
                       partial_global_mean_loss=math.fsum(weighted) / count)
        del loss
    partial['status'] = 'COMPLETE'
    return math.fsum(weighted) / count, total, sizes


def yaw_anchor_loss(predicted, raw_zero):
    """Diagnostic-only mean over all24 centers, with target cast to F32."""
    require(predicted.shape == (24, 45) and predicted.dtype == torch.float32
            and np.shape(raw_zero) == (24, 45), 'all24 center predictions/targets required')
    target = torch.as_tensor(raw_zero, dtype=torch.float32, device='cpu').detach()
    return ((predicted[:, 28] - target[:, 28]) / float(frozen.PHYSICAL_SCALE[28])).square().mean()


def gradient_geometry(gradients, arm):
    """All unweighted pairs and projections onto the fixed weighted objective."""
    require(arm in fit.ARMS and set(gradients) == set(LOSSES), 'registered arm/four gradients required')
    require(all(g.ndim == 1 and g.dtype == torch.float64 and g.device.type == 'cpu'
                and bool(torch.isfinite(g).all()) for g in gradients.values()), 'finite F64 CPU gradients required')
    norms = {name: float(torch.linalg.vector_norm(gradients[name])) for name in LOSSES}

    def relation(left, right):
        dot = float(torch.dot(left, right))
        denominator = float(torch.linalg.vector_norm(left)) * float(torch.linalg.vector_norm(right))
        return {'dot': dot, 'cosine': dot / denominator if denominator > 0 else None,
                'cosine_status': 'DEFINED' if denominator > 0 else 'ZERO_NORM_UNDEFINED'}

    pairs = {left: {right: relation(gradients[left], gradients[right]) for right in LOSSES} for left in LOSSES}
    coefficients = {'state': 1., 'yaw': fit.FIXED_LAMBDA, 'body': fit.BODY_COEFFICIENT[arm], 'anchor': 0.}
    weighted = {name: coefficients[name] * gradients[name] for name in LOSSES}
    weighted_norms = {name: float(torch.linalg.vector_norm(weighted[name])) for name in LOSSES}
    weighted_pairs = {left: {right: relation(weighted[left], weighted[right]) for right in LOSSES} for left in LOSSES}
    objective = gradients['state'] + fit.FIXED_LAMBDA * gradients['yaw']
    if arm == 'active':
        objective = objective + fit.BODY_COEFFICIENT[arm] * gradients['body']
    projections = {name: relation(gradients[name], objective) for name in LOSSES}
    for row in projections.values():
        row['directional_derivative_along_negative_objective_gradient'] = -row['dot']
    return {'unweighted_norms': norms, 'unweighted_pairwise': pairs, 'objective_projections': projections,
            'component_coefficients': coefficients, 'weighted_component_norms': weighted_norms,
            'weighted_component_pairwise': weighted_pairs,
            'weighted_objective_norm': float(torch.linalg.vector_norm(objective)),
            'lambda_yaw': fit.FIXED_LAMBDA, 'beta_body': fit.BODY_COEFFICIENT[arm],
            'anchor_coefficient': 0., 'interpretation': 'local Euclidean; not Adam or clipping prediction'}, objective


@contextmanager
def unchanged_model(model):
    """Temporarily enable residual AD; verify original weights/.grad/RNG intact."""
    parameters = tuple(model.parameters())
    flags = [p.requires_grad for p in parameters]
    grads = [(p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters]
    modes = [(module, module.training) for module in model.modules()]
    initial = paired.mapping_sha256(model.state_dict())
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    try:
        for parameter in model.residual.parameters():
            parameter.requires_grad_(True)
        yield initial
    finally:
        for parameter, flag in zip(parameters, flags, strict=True):
            parameter.requires_grad_(flag)
        require(paired.mapping_sha256(model.state_dict()) == initial, 'audit changed weights/buffers')
        require(all(p.grad is original and (copy is None or torch.equal(p.grad, copy))
                    for p, (original, copy) in zip(parameters, grads, strict=True)), 'audit changed parameter.grad')
        require(all(module.training == mode for module, mode in modes), 'audit changed module mode')
        current_numpy = np.random.get_state()
        require(torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
                and current_numpy[0] == numpy_rng[0] and np.array_equal(current_numpy[1], numpy_rng[1])
                and current_numpy[2:] == numpy_rng[2:], 'audit changed RNG state')


def _value_summary(error, normalized, body_error):
    groups = {name: {'raw_rmse': float(error[:, list(indices)].square().mean().sqrt()),
                     'normalized_mse': float(normalized[:, list(indices)].square().mean())}
              for name, indices in frozen.GROUPS.items()}
    return {'centers': len(error), 'state45_raw_rmse': error.square().mean(dim=0).sqrt().tolist(),
              'state45_normalized_rmse': normalized.square().mean(dim=0).sqrt().tolist(),
              'frozen_groups': groups, 'yaw28_raw_rmse': float(error[:, 28].square().mean().sqrt()),
              'body_vxy_raw_rmse': body_error.square().mean(dim=0).sqrt().tolist(),
              'body_vxy_normalized_rmse': (body_error / float(frozen.PHYSICAL_SCALE[23])).square().mean(dim=0).sqrt().tolist()}


def value_diagnostics(predicted, raw_zero):
    """Raw F64 errors over24 and each fixed12-center source; no new forwards."""
    predicted64, target = predicted.detach().to(torch.float64), torch.as_tensor(raw_zero, dtype=torch.float64)
    require(predicted64.shape == target.shape == (24, 45) and bool(torch.isfinite(target).all()), 'all24 finite zeros required')
    error = predicted64 - target
    normalized = error / torch.as_tensor(frozen.PHYSICAL_SCALE, dtype=torch.float64)
    pred_body = numeric._rotate_inverse(predicted64[:, 3:7], predicted64[:, 23:26])[:, :2]
    true_body = numeric._rotate_inverse(target[:, 3:7], target[:, 23:26])[:, :2]
    body_error = pred_body - true_body
    report = _value_summary(error, normalized, body_error)
    report.update(prediction_dtype='float32 promoted to float64', target_dtype='raw float64',
                  by_source={source: _value_summary(error[part], normalized[part], body_error[part])
                             for source, part in (('58', slice(0, 12)), ('62', slice(12, 24)))})
    arrays = {'center_prediction32': predicted.detach().numpy(), 'center_target64': target.numpy(),
              'center_error64': error.numpy(), 'center_normalized_error64': normalized.numpy(),
              'center_body_prediction64': pred_body.numpy(), 'center_body_target64': true_body.numpy(),
              'center_body_error64': body_error.numpy()}
    return report, arrays


def audit_model(model, train, labels, zero_targets24, deadline, *, arm, artifacts=None, progress=None):
    """Return (JSON report, NPZ arrays) for one complete69 model; writes no files.

    zero_targets24 is raw F64 [24,45], in labels[::4] source/seed/tick order.
    All four epsilon rows are retained in yaw/body objectives and provenance.
    Optional artifacts/progress dicts receive completed stages and live forward
    counts before any exception; caller persists them without retry/resumption.
    """
    arrays = {} if artifacts is None else artifacts
    report = {} if progress is None else progress
    report.update(status='INCOMPLETE', stage='admission', arm=arm, losses={}, unused_parameters={})
    require(arm in fit.ARMS and math.isfinite(deadline), 'registered arm/finite CPU deadline required')
    paired._deadline(deadline, 'before71 audit')
    train, labels = frozen.validate_data(train), fit.validate_labels(labels)
    require(len(train['s']) == 7200 and np.array_equal(np.unique(train['episode']), np.arange(12))
            and all(np.count_nonzero(train['episode'] == i) == 600 for i in range(12)), 'original7200 train required')
    windows = paired.common_window_indices(train)
    require(windows.shape == (6828, 32), 'all6828 common32 windows required')
    zeros = np.asarray(zero_targets24)
    require(zeros.shape == (24, 45) and zeros.dtype == np.float64 and np.isfinite(zeros).all(), 'raw F64 zeros required')
    require(all(np.array_equal(np.repeat(labels[key][::4], 4, axis=0), labels[key]) for key in ('s', 'a0')),
            'each center must retain identical s/a0 across all four epsilon rows')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32
                for p in (*model.parameters(), *model.buffers())), 'CPU F32 model required')
    tensors = {key: torch.as_tensor(train[key], dtype=torch.float32) for key in ('s', 'a', 'target')}
    parameters = tuple(model.residual.parameters())
    parameter_names = [name for name, _ in model.residual.named_parameters()]
    counter, stage = {}, 'centers'
    report['forward_work'] = counter

    def count_forward(_module, inputs):
        report['stage'] = stage
        bucket = counter.setdefault(stage, {'calls': 0, 'rows': 0})
        bucket['calls'] += 1
        bucket['rows'] += len(inputs[0])

    arrays.update({key: np.asarray(value).copy() for key, value in labels.items()})
    arrays.update(center_s64=labels['s'][::4].copy(), center_a0=labels['a0'][::4].copy())
    gradients, losses = {}, report['losses']
    hook = model.register_forward_pre_hook(count_forward)
    try:
        with unchanged_model(model) as model_hash, torch.enable_grad():
            report['model_state_sha256'] = model_hash
            predicted = model(torch.as_tensor(labels['s'][::4], dtype=torch.float32),
                              torch.as_tensor(labels['a0'][::4], dtype=torch.float32))
            values, value_arrays = value_diagnostics(predicted, zeros)
            arrays.update(value_arrays)
            report['value_errors'] = values
            anchor = yaw_anchor_loss(predicted, zeros)
            report['unused_parameters']['anchor'] = {}
            losses['anchor'], gradients['anchor'] = float(anchor.detach()), loss_gradient(
                anchor, parameters, unused=report['unused_parameters']['anchor'],
                parameter_names=parameter_names, require_all_used=True)
            arrays['gradient_anchor64'] = gradients['anchor'].numpy()
            del predicted, anchor
            stage = 'state'

            def chunk(begin, end):
                indices = torch.from_numpy(windows[begin:end, :8])
                return frozen.sequence_loss(model, tensors['s'][indices[:, 0]], tensors['a'][indices], tensors['target'][indices])

            report['state_prefix'] = {}
            losses['state'], gradients['state'], sizes = weighted_batch_gradients(
                parameters, len(windows), chunk, deadline, partial=report['state_prefix'], arrays=arrays,
                parameter_names=parameter_names)
            report['unused_parameters']['state'] = report['state_prefix']['unused_parameters']
            arrays['gradient_state64'] = gradients['state'].numpy()
            for name, loss_fn in (('yaw', fit.auxiliary_loss), ('body', fit.body_loss)):
                stage = name
                paired._deadline(deadline, f'before {name} gradient')
                loss = loss_fn(model, labels, create_graph=True)
                report['unused_parameters'][name] = {}
                losses[name], gradients[name] = float(loss.detach()), loss_gradient(
                    loss, parameters, unused=report['unused_parameters'][name], parameter_names=parameter_names)
                arrays[f'gradient_{name}64'] = gradients[name].numpy()
                del loss
            geometry, objective = gradient_geometry(gradients, arm)
            arrays.update({f'gradient_{name}64': gradients[name].numpy() for name in LOSSES})
            arrays['gradient_weighted_objective64'] = objective.numpy()
            arrays['common_windows'] = windows
            paired._deadline(deadline, 'after71 gradient diagnostics')
    finally:
        hook.remove()
    require({key: value['rows'] for key, value in counter.items()} ==
            {'centers': 24, 'state': 54624, 'yaw': 96, 'body': 96}, 'actual model forward row count differs')
    require({key: value['calls'] for key, value in counter.items()} ==
            {'centers': 1, 'state': 856, 'yaw': 1, 'body': 1}, 'actual model forward call count differs')
    report.update({'status': 'COMPLETE', 'stage': 'complete', 'arm': arm, 'model_state_sha256': model_hash,
              'training_data_sha256': paired.mapping_sha256(train), **fit.label_hashes(labels),
              'zero_targets_sha256': paired.array_sha256(zeros), 'common_windows_sha256': paired.array_sha256(windows),
              'windows': 6828, 'state_horizon': 8, 'state_batches': len(sizes), 'last_batch_windows': sizes[-1],
              'weighted_window_count': sum(sizes), 'label_rows': 96, 'centers': 24,
              'source_ids': [58, 62], 'rows_dropped': 0, 'losses': losses, 'value_errors': values,
              'gradient_geometry': geometry, 'forward_work': counter,
              'parameter_order': [{'name': name, 'shape': list(p.shape), 'numel': p.numel()}
                                  for name, p in model.residual.named_parameters()],
              'gradients_are_F32_AD_then_F64_reduce': True, 'weights_grad_flags_modes_RNG_unchanged': True,
              'model_updates': 0, 'physics_steps': 0, 'actor_calls': 0, 'new_training_gate': None})
    return report, arrays
