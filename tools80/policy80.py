"""Independent80 fixed-start policy update; frozen79 mathematics is retained.

The formal runner supplies27 equally weighted starts. Nine starts are accepted
only to compare this function with79 on synthetic inputs. No79 globals or
frozen evaluation functions are changed; only this new function admits27.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path

import torch

from tools54 import numerics54 as numeric
from tools56.fit56 import mapping_sha256
from tools79.policy79 import (
    CLIP_NORM,
    EXECUTION_BOUND,
    FD_EPS,
    HORIZON,
    INTERNAL_BOUND,
    LEARNING_RATE,
    WHEEL_ROWS,
    _binary,
    _gradient_hash,
    _head,
    _json,
    _protected_hash,
    _validate_frozen,
    direction_check,
    mask_gradient,
    project_head,
    require,
    row_l1,
)


def _deadline(limit):
    if time.process_time() >= limit:
        raise TimeoutError('policy80 CPU deadline reached')


def update_actor(model, actor, anchor, points, output, *, cpu_deadline):
    """Return an independent frozen F32 actor plus output/receipt.json.

    A caller supplies27 starts (or9 for79 compatibility checks) and an absolute
    process-CPU deadline. Original models/actors are never mutated. Exclusive
    artifacts and fsynced events retain failures without resume or overwrite.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'schema': 'd1-policy80-update-v1', 'status': 'INCOMPLETE', 'stage': 'created',
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
            require(isinstance(points, (list, tuple)) and len(points) in (9, 27), 'exactly9 or27 complete starts required')
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
