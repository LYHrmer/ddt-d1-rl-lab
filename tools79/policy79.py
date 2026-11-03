"""One constrained79 wheel-head SGD step through a frozen world model.

Only the four128-weight-plus-bias wheel rows may change. All rollouts use
the original54 finite-H32 local cost, theta=0, and equal point weights. This
module does not establish physical improvement or change existing gates.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from tools54 import numerics54 as numeric
from tools56.fit56 import mapping_sha256

WHEEL_ROWS = (3, 7, 11, 15)
OTHER_ROWS = tuple(i for i in range(16) if i not in WHEEL_ROWS)
HORIZON, LEARNING_RATE, CLIP_NORM = 32, .001, 1.
FD_EPS, INTERNAL_BOUND, EXECUTION_BOUND = .001, .0099998, .01
HEAD_KEYS = ('action_head.weight', 'action_head.bias')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _binary(path, value):
    with path.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _deadline(limit):
    if time.process_time() >= limit:
        raise TimeoutError('policy79 CPU deadline reached')


def _validate_frozen(module):
    require(isinstance(module, torch.nn.Module) and all(not m.training for m in module.modules()),
            'complete eval modules required')
    require(all(not p.requires_grad for p in module.parameters()), 'input modules must have frozen parameters')
    require(all(p.grad is None or (p.grad.device.type == 'cpu' and p.grad.dtype == torch.float32
                                  and bool(torch.isfinite(p.grad).all())) for p in module.parameters()),
            'preexisting source gradients must be finite CPU F32')
    require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                and bool(torch.isfinite(t).all()) for t in (*module.parameters(), *module.buffers())),
            'finite CPU F32 source modules required')


def _head(actor):
    head = actor.action_head
    require(isinstance(head, torch.nn.Linear) and head.weight.shape == (16, 128)
            and head.bias is not None and head.bias.shape == (16,), 'complete16x128+bias action head required')
    return head


def _protected_hash(actor):
    values = dict(actor.state_dict())
    for key in HEAD_KEYS:
        values[key] = values[key][list(OTHER_ROWS)]
    return mapping_sha256(values)


def _gradient_hash(module):
    return mapping_sha256({name: p.grad for name, p in module.named_parameters() if p.grad is not None})


def mask_gradient(weight, bias):
    require(weight.shape == (16, 128) and bias.shape == (16,), 'head gradient shape differs')
    require(bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()), 'nonfinite head gradient')
    weight, bias = weight.detach().clone(), bias.detach().clone()
    weight[list(OTHER_ROWS)] = 0
    bias[list(OTHER_ROWS)] = 0
    return weight, bias


def row_l1(weight, bias, anchor_weight, anchor_bias):
    """Measure the actual stored weights relative to the original anchor in F64."""
    rows = list(WHEEL_ROWS)
    return ((weight.double()[rows] - anchor_weight.double()[rows]).abs().sum(-1)
            + (bias.double()[rows] - anchor_bias.double()[rows]).abs())


def project_head(weight, bias, anchor_weight, anchor_bias):
    """Radially rescale each129-coordinate row delta into its fixed L1 bound."""
    weight, bias = weight.detach().clone(), bias.detach().clone()
    rows = list(WHEEL_ROWS)
    lengths = row_l1(weight, bias, anchor_weight, anchor_bias)
    factors = (INTERNAL_BOUND / lengths.clamp_min(torch.finfo(torch.float64).tiny)).clamp(max=1.)
    weight[rows] = anchor_weight[rows] + (weight[rows] - anchor_weight[rows]) * factors[:, None]
    bias[rows] = anchor_bias[rows] + (bias[rows] - anchor_bias[rows]) * factors
    return weight, bias


def direction_check(analytic, difference):
    require(math.isfinite(analytic) and math.isfinite(difference), 'nonfinite direction diagnostic')
    agreement = abs(analytic - difference) <= 1e-4 + .1 * abs(difference)
    identifiable = abs(analytic) >= 1e-5 and abs(difference) >= 1e-5 and analytic * difference > 0
    return {'gradient': analytic, 'model_central_difference': difference,
            'absolute_error': abs(analytic - difference), 'tolerance': 1e-4 + .1 * abs(difference),
            'autograd_agrees': agreement, 'direction_identifiable': identifiable,
            'magnitude_floor': 1e-5, 'passed': agreement and identifiable}


def update_actor(model, actor, anchor, points, output, *, cpu_deadline):
    """Return an independent frozen F32 actor plus output/receipt.json.

    A caller supplies either9 development or3 online starts and an absolute
    process-CPU deadline. Original models/actors are never mutated. Exclusive
    artifacts and fsynced events retain failures without resume or overwrite.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'schema': 'd1-policy79-update-v1', 'status': 'INCOMPLETE', 'stage': 'created',
               'updates_attempted': 0, 'updates_completed': 0, 'updates': 0, 'planned_updates': 1,
               'gradient_points_completed': 0, 'fd_points_completed': 0, 'points': [], 'fd_points': [],
               'horizon': HORIZON, 'theta': 0., 'learning_rate': LEARNING_RATE, 'momentum': 0.,
               'gradient_clip_norm': CLIP_NORM, 'fd_epsilon': FD_EPS, 'allowed_parameter_count': 516,
               'wheel_rows': list(WHEEL_ROWS), 'projection': 'radial_row_l1',
               'projection_internal_bound': INTERNAL_BOUND, 'execution_bound': EXECUTION_BOUND,
               'point_reduction': 'equal mean', 'eligible_for_execution': False, 'failure': None,
               'verification_after': 'NOT_RUN', 'retry_or_resume': False}
    _json(output / 'initial_incomplete.json', receipt)
    originals, identities, source_gradients, artifacts = {}, {}, {}, {}
    work_model = work_actor = candidate = None
    model64_hash = protected64_hash = caught = None
    with (output / 'events.jsonl').open('x') as events:
        def event(stage):
            receipt['stage'] = stage
            events.write(json.dumps({'stage': stage, 'cpu_seconds': time.process_time() - started,
                **{k: receipt[k] for k in ('updates_attempted', 'updates_completed',
                                         'gradient_points_completed', 'fd_points_completed')}}, allow_nan=False) + '\n')
            events.flush()
            os.fsync(events.fileno())

        def loss(network, point):
            _deadline(cpu_deadline)
            value = numeric.imaginary(work_model, network, point,
                                      torch.zeros((), dtype=torch.float64), horizon=HORIZON)
            require(value.ndim == 0 and value.dtype == torch.float64 and bool(torch.isfinite(value)),
                    'finite scalar F64 imagined cost required')
            return value

        try:
            require(type(cpu_deadline) in (int, float) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
            _deadline(cpu_deadline)
            require(isinstance(points, (list, tuple)) and len(points) in (3, 9), 'exactly3 or9 complete starts required')
            require(len({(p['seed'], p['tick']) for p in points}) == len(points), 'duplicate start identity')
            originals = {'model': model, 'actor': actor, 'anchor': anchor}
            for name, module in originals.items():
                _validate_frozen(module)
                identities[name] = mapping_sha256(module.state_dict())
                source_gradients[name] = _gradient_hash(module)
            head, anchor_head = _head(actor), _head(anchor)
            require(_protected_hash(actor) == _protected_hash(anchor), 'actor protected weights differ from original anchor')
            initial_bounds = row_l1(head.weight, head.bias, anchor_head.weight, anchor_head.bias)
            require(bool((initial_bounds <= EXECUTION_BOUND).all()), 'initial actor exceeds original-anchor bound')
            receipt.update(source_mapping_sha256=identities.copy(), actor_before_mapping_sha256=identities['actor'],
                           source_gradient_sha256=source_gradients.copy(),
                           anchor_mapping_sha256=identities['anchor'], model_before_mapping_sha256=identities['model'],
                           protected_before_mapping_sha256=_protected_hash(actor), point_count=len(points),
                           initial_row_l1=initial_bounds.tolist())
            work_model = copy.deepcopy(model).double().eval().requires_grad_(False)
            work_actor = copy.deepcopy(actor).double().eval().requires_grad_(False)
            for module in (work_model, work_actor):
                for parameter in module.parameters():
                    parameter.grad = None
            head64 = _head(work_actor)
            head64.requires_grad_(True)
            anchor_weight, anchor_bias = anchor_head.weight.detach().double(), anchor_head.bias.detach().double()
            model64_hash, protected64_hash = mapping_sha256(work_model.state_dict()), _protected_hash(work_actor)
            receipt['zero_update_roundtrip_exact'] = mapping_sha256(copy.deepcopy(work_actor).float().state_dict()) == identities['actor']
            require(receipt['zero_update_roundtrip_exact'], 'F64 toF32 zero-update mapping changed')
            artifacts.update(head_before_weight64=head64.weight.detach().clone(), head_before_bias64=head64.bias.detach().clone(),
                             anchor_weight64=anchor_weight, anchor_bias64=anchor_bias)
            gradients = (torch.zeros_like(head64.weight), torch.zeros_like(head64.bias))
            mean_value = 0.
            event('gradient')
            for index, point in enumerate(points):
                value = loss(work_actor, point)
                current = mask_gradient(*torch.autograd.grad(value, (head64.weight, head64.bias), allow_unused=False))
                for accumulated, gradient in zip(gradients, current, strict=True):
                    accumulated.add_(gradient / len(points))
                mean_value += float(value.detach()) / len(points)
                receipt['points'].append({'index': index, 'seed': point['seed'], 'tick': point['tick'],
                    'cost': float(value.detach()), 'masked_gradient_norm': float(torch.linalg.vector_norm(torch.cat(
                        (current[0].flatten(), current[1]))))})
                receipt['gradient_points_completed'] += 1
                artifacts.update(masked_mean_gradient_weight64=gradients[0], masked_mean_gradient_bias64=gradients[1])
                prefix = f'gradient_prefix_{index:02d}.pt'
                _binary(output / prefix, {'denominator': len(points), 'completed': index + 1,
                    'weighted_loss': mean_value, 'weighted_gradient_weight64': gradients[0],
                    'weighted_gradient_bias64': gradients[1]})
                event('gradient_point_complete')
            norm = torch.linalg.vector_norm(torch.cat((gradients[0].flatten(), gradients[1])))
            require(bool(torch.isfinite(norm)), 'nonfinite mean gradient')
            receipt.update(mean_cost_before=mean_value, masked_gradient_norm=float(norm))
            if float(norm) == 0.:
                receipt['selfcheck'] = direction_check(0., 0.) | {'fd_evaluated': False, 'reason': 'zero_gradient_no_unit_direction'}
                raise ValueError('policy direction qualification failed: zero mean gradient')
            direction = tuple(g / norm for g in gradients)
            artifacts.update(unit_direction_weight64=direction[0], unit_direction_bias64=direction[1])
            analytic = float(sum((g * v).sum() for g, v in zip(gradients, direction, strict=True)))
            means = []
            for sign in (1, -1):
                shifted = copy.deepcopy(work_actor).requires_grad_(False)
                with torch.no_grad():
                    for parameter, vector in zip(_head(shifted).parameters(), direction, strict=True):
                        parameter.add_(vector, alpha=sign * FD_EPS)
                    mean = 0.
                    for index, point in enumerate(points):
                        value = float(loss(shifted, point))
                        mean += value / len(points)
                        receipt['fd_points'].append({'sign': sign, 'index': index, 'seed': point['seed'],
                                                    'tick': point['tick'], 'cost': value})
                        receipt['fd_points_completed'] += 1
                        event('fd_point_complete')
                means.append(mean)
            receipt['selfcheck'] = direction_check(analytic, (means[0] - means[1]) / (2 * FD_EPS)) | {
                'fd_evaluated': True, 'plus_mean': means[0], 'minus_mean': means[1],
                'direction': 'unit_normalized_masked_mean_head_gradient', 'projection_used_in_fd': False}
            require(receipt['selfcheck']['passed'], 'policy direction qualification failed')
            _deadline(cpu_deadline)
            optimizer = torch.optim.SGD(head64.parameters(), lr=LEARNING_RATE, momentum=0.)
            for parameter, gradient in zip(head64.parameters(), gradients, strict=True):
                parameter.grad = gradient.clone()
            clipped_norm = torch.nn.utils.clip_grad_norm_(head64.parameters(), CLIP_NORM, error_if_nonfinite=True)
            receipt['gradient_norm_before_clip'] = float(clipped_norm)
            artifacts.update(clipped_gradient_weight64=head64.weight.grad.clone(), clipped_gradient_bias64=head64.bias.grad.clone())
            receipt['updates_attempted'] = 1
            event('sgd_attempt')
            optimizer.step()
            receipt['updates_completed'] = receipt['updates'] = 1
            artifacts.update(sgd_weight64=head64.weight.detach().clone(), sgd_bias64=head64.bias.detach().clone())
            event('sgd_completed')
            projected = project_head(head64.weight, head64.bias, anchor_weight, anchor_bias)
            with torch.no_grad():
                head64.weight.copy_(projected[0])
                head64.bias.copy_(projected[1])
            artifacts.update(projected_weight64=projected[0], projected_bias64=projected[1])
            candidate = copy.deepcopy(work_actor).float().eval().requires_grad_(False)
            for parameter in candidate.parameters():
                parameter.grad = None
            _validate_frozen(candidate)
            actual_head = _head(candidate)
            bounds = row_l1(actual_head.weight, actual_head.bias, anchor_head.weight, anchor_head.bias)
            receipt['final_row_l1'] = bounds.tolist()
            require(bool((bounds <= EXECUTION_BOUND).all()), 'F32 execution exceeds original-anchor bound')
            require(_protected_hash(candidate) == receipt['protected_before_mapping_sha256'], 'protected F32 actor parameters changed')
            artifacts.update(actual_delta_weight64=actual_head.weight.double() - actor.action_head.weight.double(),
                             actual_delta_bias64=actual_head.bias.double() - actor.action_head.bias.double())
            receipt.update(actor_after_mapping_sha256=mapping_sha256(candidate.state_dict()),
                           protected_after_mapping_sha256=_protected_hash(candidate),
                           actual_update_norm=float(torch.linalg.vector_norm(torch.cat((artifacts['actual_delta_weight64'].flatten(),
                                                                                        artifacts['actual_delta_bias64'])))),
                           weights_file='actor_state_dict.pt')
            receipt['weights_file_sha256'] = _binary(output / receipt['weights_file'], candidate.state_dict())
            _deadline(cpu_deadline)
            receipt.update(status='COMPLETE', eligible_for_execution=True)
            event('complete')
        except BaseException as exc:
            caught = exc
            receipt.update(status='INCOMPLETE', eligible_for_execution=False, failure=f'{type(exc).__name__}: {exc}')
            event('failed')
            raise
        finally:
            verification_error = None
            try:
                for name, digest in identities.items():
                    _validate_frozen(originals[name])
                    require(mapping_sha256(originals[name].state_dict()) == digest, name + ' source changed')
                    require(_gradient_hash(originals[name]) == source_gradients[name], name + ' source gradients changed')
                if model64_hash is not None:
                    require(mapping_sha256(work_model.state_dict()) == model64_hash, 'F64 world model changed')
                    require(_protected_hash(work_actor) == protected64_hash, 'protected F64 actor weights changed')
                receipt['source_identity_checks_completed'] = len(identities)
                receipt['verification_after'] = 'COMPLETE' if len(identities) == 3 else 'NOT_COMPLETE'
            except BaseException as exc:  # noqa: BLE001 -- persist interrupted verification without hiding the original failure.
                verification_error = exc
                receipt.update(status='INCOMPLETE', eligible_for_execution=False, verification_after='FAILED',
                               verification_failure=f'{type(exc).__name__}: {exc}')
                if receipt['failure'] is None:
                    receipt['failure'] = receipt['verification_failure']
            artifact_error = None
            try:
                if artifacts:
                    receipt['head_step_file'] = 'head_step.pt'
                    receipt['head_step_file_sha256'] = _binary(output / 'head_step.pt', artifacts)
                if receipt['status'] != 'COMPLETE' and work_actor is not None:
                    receipt['partial_actor_file'] = 'partial_actor_state_dict64.pt'
                    receipt['partial_actor_file_sha256'] = _binary(output / receipt['partial_actor_file'], work_actor.state_dict())
            except BaseException as exc:  # noqa: BLE001 -- still attempt the final receipt when an artifact write fails.
                artifact_error = exc
                receipt.update(status='INCOMPLETE', eligible_for_execution=False,
                               artifact_write_failure=f'{type(exc).__name__}: {exc}')
                if receipt['failure'] is None:
                    receipt['failure'] = receipt['artifact_write_failure']
            receipt['cpu_seconds'] = time.process_time() - started
            _json(output / 'receipt.json', receipt)
            if verification_error is not None and caught is None:
                raise verification_error
            if artifact_error is not None and caught is None:
                raise artifact_error
    return candidate, receipt
