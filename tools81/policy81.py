"""Fixed81 C6/Q6 wheel-head directions and independent F32/F64 endpoints.

No optimizer search, projection, critic, model fit or physical evaluation lives
here. Numerical AD/FD diagnostics do not select an endpoint or requalify a model.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import time
from pathlib import Path

import torch

from tools54 import numerics54 as numeric
from tools79.policy79 import (
    HEAD_KEYS,
    OTHER_ROWS,
    WHEEL_ROWS,
    _binary,
    _gradient_hash,
    _head,
    _json,
    _protected_hash,
    _validate_frozen,
    mapping_sha256,
    mask_gradient,
    require,
    row_l1,
)

ARMS = ('C6', 'Q6')
SCALES = (.009, .0045, .00225, .001125)
D_SEEDS = (81107, 81108, 81109)
TICKS = (175, 325)
HORIZON, EXECUTION_BOUND = 32, .01


def _deadline(limit):
    if limit is not None:
        require(type(limit) in (int, float) and math.isfinite(limit), 'finite CPU deadline required')
        if time.process_time() >= limit:
            raise TimeoutError('policy81 CPU deadline reached')


def _ordered_points(points):
    require(isinstance(points, (tuple, list)) and len(points) == 6, 'six complete steady starts required')
    require(all(type(p.get('seed')) is int and type(p.get('tick')) is int for p in points),
            'integer start identities required')
    require(all(p.get('status', 'COMPLETE') == 'COMPLETE' for p in points), 'partial start rejected')
    identities = [(p['seed'], p['tick']) for p in points]
    seeds = sorted({p['seed'] for p in points})
    require(len(seeds) == 3 and sorted(identities) == [(s, t) for s in seeds for t in TICKS],
            'exactly three seeds with both175/325 starts required')
    return sorted(points, key=lambda p: (p['seed'], p['tick']))


def objective_values(model, actor, points, *, cpu_deadline=None, on_point=None):
    """Return differentiable F64 C6/Q6, per_seed_q[3], and six point traces.

    Point rows contain seed/tick, scalar cost/yaw_mae, yaw_error[32] and the
    frozen54 graph-retaining trace. Commands[t] pair with post-state[t]. The
    callback receives (index, row) after each finished rollout, retaining a
    durable prefix if a later rollout fails. V callers may reuse the same graph
    for both objectives; this function does not generate a direction or update.
    """
    rows = []
    for index, point in enumerate(_ordered_points(points)):
        _deadline(cpu_deadline)
        value, trace = numeric.imaginary(model, actor, point, torch.zeros((), dtype=torch.float64),
                                         return_trace=True, horizon=HORIZON)
        require(value.ndim == 0 and value.dtype == torch.float64 and bool(torch.isfinite(value)),
                'finite F64 scalar cost required')
        require(trace['states'].shape == (HORIZON, 45) and trace['states'].dtype == torch.float64,
                'F64 post-state trace32x45 required')
        require(all(bool(torch.isfinite(trace[key]).all()) for key in ('states', 'cost_components', 'actions')),
                'nonfinite model trace')
        command = torch.as_tensor(point['commands'], dtype=torch.float64, device='cpu')
        require(command.ndim == 2 and command.shape[0] >= HORIZON + 1 and command.shape[1] == 3
                and bool(torch.isfinite(command).all()), 'complete finite command horizon required')
        error = trace['states'][:, 28] - command[:HORIZON, 2]
        require(bool(torch.isfinite(error).all()), 'nonfinite yaw trace')
        row = {'seed': point['seed'], 'tick': point['tick'], 'cost': value,
               'yaw_mae': error.abs().mean(), 'yaw_error': error, 'trace': trace}
        rows.append(row)
        if on_point is not None:
            on_point(index, row)
    q = torch.stack([torch.maximum(rows[i]['yaw_mae'], rows[i + 1]['yaw_mae']) for i in (0, 2, 4)])
    return {'C6': torch.stack([r['cost'] for r in rows]).mean(), 'Q6': q.mean(),
            'per_seed_q': q, 'points': rows}


def normalize_direction(weight, bias):
    """Negative masked gradient / maximum wheel-row L1, without clipping."""
    weight, bias = mask_gradient(weight, bias)
    require(weight.dtype == bias.dtype == torch.float64 and weight.device.type == bias.device.type == 'cpu',
            'CPU F64 gradients required')
    lengths = weight[list(WHEEL_ROWS)].abs().sum(-1) + bias[list(WHEEL_ROWS)].abs()
    scale = float(lengths.max())
    require(math.isfinite(scale) and scale > 0., 'zero or nonfinite max-row-L1 gradient')
    return (-weight / scale, -bias / scale), scale


def _actor_with_head(anchor, weight, bias, *, actual):
    require(weight.shape == (16, 128) and bias.shape == (16,)
            and bool(torch.isfinite(weight).all()) and bool(torch.isfinite(bias).all()), 'invalid endpoint head')
    source_head = _head(anchor)
    require(torch.equal(weight[list(OTHER_ROWS)], source_head.weight.to(weight.dtype)[list(OTHER_ROWS)])
            and torch.equal(bias[list(OTHER_ROWS)], source_head.bias.to(bias.dtype)[list(OTHER_ROWS)]),
            'endpoint changes protected head rows')
    actor = copy.deepcopy(anchor).to(dtype=torch.float32 if actual else torch.float64).eval().requires_grad_(False)
    with torch.no_grad():
        _head(actor).weight.copy_(weight)
        _head(actor).bias.copy_(bias)
    for parameter in actor.parameters():
        parameter.grad = None
    return actor


def _ulp_summary(anchor, changed):
    def ordered(value):
        bits = value.detach().contiguous().view(torch.int32).to(torch.int64)
        return torch.where(bits < 0, 0x80000000 - (bits & 0x7fffffff), 0x80000000 + bits)

    distances = (ordered(changed) - ordered(anchor)).abs().flatten()
    nonzero = distances[distances > 0]
    return {'coordinates': int(distances.numel()), 'changed': int(nonzero.numel()),
            'min_nonzero_ulp': int(nonzero.min()) if nonzero.numel() else None,
            'max_ulp': int(distances.max())}


def _point_summary(row):
    error = row['yaw_error'].detach()
    trace = row['trace']
    return {'seed': row['seed'], 'tick': row['tick'], 'cost': float(row['cost'].detach()),
            'yaw_mae': float(row['yaw_mae'].detach()), 'yaw_error': error.tolist(),
            'yaw_error_sign': torch.sign(error).to(torch.int8).tolist(),
            'exact_zero_yaw_errors': int((error == 0).sum()),
            'minimum_absolute_yaw_error': float(error.abs().min()),
            'cost_components_mean': trace['cost_components'].detach().mean(0).tolist(),
            'first_action': trace['actions'][0].detach().tolist(),
            'clipped_action_coordinates': int((trace['actions'].detach().abs() >= 1.).sum())}


def _objective_summary(values):
    points = [_point_summary(row) for row in values['points']]
    choices = []
    for index in (0, 2, 4):
        a, b = points[index:index + 2]
        choices.append({'seed': a['seed'], 'tie': a['yaw_mae'] == b['yaw_mae'],
                        'max_ticks': [p['tick'] for p in (a, b) if p['yaw_mae'] == max(a['yaw_mae'], b['yaw_mae'])]})
    return {'C6': float(values['C6'].detach()), 'Q6': float(values['Q6'].detach()),
            'per_seed_q': values['per_seed_q'].detach().tolist(), 'max_choices': choices, 'points': points}


def update_actor(model, anchor, points, arm, output, *, cpu_deadline):
    """Return {(scale_index, sign): frozen actual-F32 actor}, receipt.

    Exactly one main candidate is constructed at (0,+1); the other endpoints
    are fixed diagnostics, not further optimization steps. Eight ideal-F64
    endpoint heads and their D objective scalars are independently archived.
    Existing hooks are preserved on copies, so the runner counts every call.
    """
    import json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'schema': 'd1-policy81-direction-v1', 'status': 'INCOMPLETE', 'stage': 'created',
               'arm': arm, 'horizon': HORIZON, 'scales': list(SCALES), 'main_probe': [0, 1],
               'point_count': 6, 'allowed_parameter_count': 516, 'wheel_rows': list(WHEEL_ROWS),
               'head_parameter_keys': list(HEAD_KEYS),
               'normalization': 'negative_masked_gradient_div_max_wheel_row_l1',
               'execution_bound': EXECUTION_BOUND, 'projection': False, 'gradient_clip': False,
               'updates_attempted': 0, 'updates_completed': 0, 'updates': 0, 'model_updates': 0,
               'gradient_points_completed': 0, 'fd_points_completed': 0,
               'model_calls_attempted': 0, 'model_calls': 0,
               'points': [], 'fd_points': [], 'probes': [], 'selfchecks': [],
               'eligible_for_execution': False, 'verification_after': 'NOT_RUN', 'failure': None,
               'selfchecks_are_diagnostic_only': True, 'direction_magnitude_floor': None,
               'retry_or_resume': False}
    _json(output / 'initial_incomplete.json', receipt)
    originals = {'model': model, 'anchor': anchor}
    identities, gradient_hashes, artifacts, probes = {}, {}, {}, {}
    work_model = work_actor = None
    last_endpoint = last_endpoint_identity = None
    model64_hash = protected64_hash = caught = None
    hooks = []
    with (output / 'events.jsonl').open('x') as events:
        def event(stage):
            receipt['stage'] = stage
            events.write(json.dumps({'stage': stage, 'cpu_seconds': time.process_time() - started,
                **{key: receipt[key] for key in ('updates_attempted', 'updates_completed',
                    'gradient_points_completed', 'fd_points_completed', 'model_calls_attempted', 'model_calls')}},
                allow_nan=False) + '\n')
            events.flush()
            os.fsync(events.fileno())

        def before_forward(_module, _inputs):
            _deadline(cpu_deadline)
            receipt['model_calls_attempted'] += 1

        def after_forward(_module, _inputs, _result):
            receipt['model_calls'] += 1

        def save_point(kind, **identity):
            def saved(index, row):
                item = {'index': index, **identity, **_point_summary(row)}
                receipt['points' if kind == 'gradient' else 'fd_points'].append(item)
                receipt[kind + '_points_completed'] += 1
                name = ('gradient' if kind == 'gradient' else f"fd_{identity['scale_index']:02d}_{identity['sign']:+d}")
                _json(output / f'{name}_point_{index:02d}.json', item)
                event(kind + '_point_complete')
            return saved

        try:
            require(type(cpu_deadline) in (int, float) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
            _deadline(cpu_deadline)
            require(arm in ARMS, 'arm must be C6 or Q6')
            points = _ordered_points(points)
            require(tuple(sorted({p['seed'] for p in points})) == D_SEEDS, 'only registered D seeds generate directions')
            for name, source in originals.items():
                _validate_frozen(source)
                identities[name] = mapping_sha256(source.state_dict())
                gradient_hashes[name] = _gradient_hash(source)
            anchor_head = _head(anchor)
            require(isinstance(anchor.policy_net, torch.nn.Sequential)
                    and isinstance(anchor.policy_net[-1], torch.nn.Tanh), 'bounded final policy latent required')
            receipt.update(source_mapping_sha256=identities.copy(), source_gradient_sha256=gradient_hashes.copy(),
                           actor_before_mapping_sha256=identities['anchor'], anchor_mapping_sha256=identities['anchor'],
                           protected_before_mapping_sha256=_protected_hash(anchor),
                           point_identities=[{'seed': p['seed'], 'tick': p['tick']} for p in points])
            work_model = copy.deepcopy(model).double().eval().requires_grad_(False)
            work_actor = copy.deepcopy(anchor).double().eval().requires_grad_(False)
            for module in (work_model, work_actor):
                for parameter in module.parameters():
                    parameter.grad = None
            head = _head(work_actor).requires_grad_(True)
            model64_hash, protected64_hash = mapping_sha256(work_model.state_dict()), _protected_hash(work_actor)
            require(mapping_sha256(copy.deepcopy(work_actor).float().state_dict()) == identities['anchor'],
                    'zero-update F32/F64 roundtrip changed source')
            hooks = [work_model.register_forward_pre_hook(before_forward), work_model.register_forward_hook(after_forward)]
            values = objective_values(work_model, work_actor, points, cpu_deadline=cpu_deadline,
                                      on_point=save_point('gradient'))
            receipt['before'] = _objective_summary(values)
            gradients = mask_gradient(*torch.autograd.grad(values[arm], (head.weight, head.bias)))
            direction, norm = normalize_direction(*gradients)
            analytic = float(sum((g * d).sum() for g, d in zip(gradients, direction, strict=True)))
            receipt.update(max_row_l1_gradient=norm, directional_derivative=analytic)
            artifacts.update(anchor_mapping_sha256=identities['anchor'],
                head_before_weight64=head.weight.detach().clone(), head_before_bias64=head.bias.detach().clone(),
                masked_gradient_weight64=gradients[0], masked_gradient_bias64=gradients[1],
                direction_weight64=direction[0], direction_bias64=direction[1])
            receipt['head_step_file'] = 'head_step.pt'
            receipt['head_step_file_sha256'] = _binary(output / receipt['head_step_file'], artifacts)
            event('direction_sealed')
            del values
            for scale_index, scale in enumerate(SCALES):
                endpoint_values = {}
                for sign in (1, -1):
                    _deadline(cpu_deadline)
                    if scale_index == 0 and sign == 1:
                        receipt['updates_attempted'] = 1
                        event('main_candidate_attempt')
                    ideal_weight = artifacts['head_before_weight64'] + sign * scale * direction[0]
                    ideal_bias = artifacts['head_before_bias64'] + sign * scale * direction[1]
                    actual = _actor_with_head(anchor, ideal_weight.float(), ideal_bias.float(), actual=True)
                    last_endpoint = actual
                    last_endpoint_identity = {'scale_index': scale_index, 'sign': sign, 'epsilon': scale}
                    if scale_index == 0 and sign == 1:
                        receipt['updates_completed'] = receipt['updates'] = 1
                        event('main_candidate_constructed')
                    actual_head = _head(actual)
                    bounds = row_l1(actual_head.weight, actual_head.bias, anchor_head.weight, anchor_head.bias)
                    require(bool((bounds <= EXECUTION_BOUND).all()), 'actual F32 endpoint violates original-anchor bound')
                    require(_protected_hash(actual) == receipt['protected_before_mapping_sha256'], 'protected parameters changed')
                    _validate_frozen(actual)
                    tag = f"probe_{scale_index:02d}_{'plus' if sign == 1 else 'minus'}"
                    row = {'scale_index': scale_index, 'epsilon': scale, 'sign': sign, 'status': 'INCOMPLETE',
                           'actual_file': tag + '_f32.pt', 'ideal_file': tag + '_ideal64.pt',
                           'actual_mapping_sha256': mapping_sha256(actual.state_dict()),
                           'final_row_l1': bounds.tolist(),
                           'parameter_ulps': {name: _ulp_summary(getattr(anchor_head, name)[list(WHEEL_ROWS)],
                                                               getattr(actual_head, name)[list(WHEEL_ROWS)])
                                              for name in ('weight', 'bias')}}
                    receipt['probes'].append(row)
                    row['actual_file_sha256'] = _binary(output / row['actual_file'], {
                        'weight': actual_head.weight.detach().clone(), 'bias': actual_head.bias.detach().clone(),
                        'delta_weight64': actual_head.weight.double() - anchor_head.weight.double(),
                        'delta_bias64': actual_head.bias.double() - anchor_head.bias.double()})
                    ideal = _actor_with_head(anchor, ideal_weight, ideal_bias, actual=False)
                    row['ideal_mapping_sha256'] = mapping_sha256(ideal.state_dict())
                    require(_protected_hash(ideal) == protected64_hash, 'protected ideal-F64 parameters changed')
                    row['ideal_file_sha256'] = _binary(output / row['ideal_file'], {'weight': ideal_weight, 'bias': ideal_bias})
                    probes[(scale_index, sign)] = actual
                    with torch.no_grad():
                        result = objective_values(work_model, ideal, points, cpu_deadline=cpu_deadline,
                            on_point=save_point('fd', scale_index=scale_index, sign=sign))
                    row['ideal_D_values'] = _objective_summary(result)
                    endpoint_values[sign] = row['ideal_D_values']
                    row['status'] = 'COMPLETE'
                    event('endpoint_complete')
                fd = (endpoint_values[1][arm] - endpoint_values[-1][arm]) / (2 * scale)
                tolerance = 1e-4 + .1 * abs(fd)
                receipt['selfchecks'].append({'epsilon': scale, 'analytic': analytic, 'ideal_model_fd': fd,
                    'absolute_error': abs(analytic - fd), 'tolerance': tolerance,
                    'autograd_agrees': abs(analytic - fd) <= tolerance,
                    'magnitude_floor': None, 'acceptance_gate': False,
                    'plus_max_choices': endpoint_values[1]['max_choices'],
                    'minus_max_choices': endpoint_values[-1]['max_choices'],
                    'baseline_max_choices': receipt['before']['max_choices'],
                    'max_branch_changed': any(endpoint_values[s]['max_choices'] != receipt['before']['max_choices']
                                              for s in (1, -1)),
                    'abs_sign_pattern_changed': any(
                        row['yaw_error_sign'] != old['yaw_error_sign']
                        for s in (1, -1)
                        for row, old in zip(endpoint_values[s]['points'], receipt['before']['points'], strict=True))})
            require(len(probes) == 8 and all(p['status'] == 'COMPLETE' for p in receipt['probes']), 'endpoint set incomplete')
            receipt['weights_file'] = 'actor_state_dict.pt'
            receipt['weights_file_sha256'] = _binary(output / receipt['weights_file'], probes[(0, 1)].state_dict())
            receipt['actor_after_mapping_sha256'] = mapping_sha256(probes[(0, 1)].state_dict())
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
                    require(_gradient_hash(originals[name]) == gradient_hashes[name], name + ' source gradients changed')
                if model64_hash is not None:
                    require(mapping_sha256(work_model.state_dict()) == model64_hash, 'F64 model changed')
                    require(_protected_hash(work_actor) == protected64_hash, 'protected F64 actor changed')
                receipt['verification_after'] = 'COMPLETE' if len(identities) == 2 else 'NOT_COMPLETE'
            except BaseException as exc:  # noqa: BLE001 -- retain late verification failures and prior body failures.
                verification_error = exc
                receipt.update(status='INCOMPLETE', eligible_for_execution=False, verification_after='FAILED',
                               verification_failure=f'{type(exc).__name__}: {exc}')
                if receipt['failure'] is None:
                    receipt['failure'] = receipt['verification_failure']
            for handle in hooks:
                handle.remove()
            artifact_error = None
            if receipt['status'] != 'COMPLETE' and last_endpoint is not None:
                try:
                    receipt['partial_endpoint_identity'] = last_endpoint_identity
                    receipt['partial_endpoint_file'] = 'partial_endpoint_state_dict.pt'
                    receipt['partial_endpoint_file_sha256'] = _binary(
                        output / receipt['partial_endpoint_file'], last_endpoint.state_dict())
                except BaseException as exc:  # noqa: BLE001 -- preserve the primary failure even when storage fails.
                    artifact_error = exc
                    receipt['partial_artifact_failure'] = f'{type(exc).__name__}: {exc}'
            receipt['cpu_seconds'] = time.process_time() - started
            _json(output / 'receipt.json', receipt)
            if verification_error is not None and caught is None:
                raise verification_error
            if artifact_error is not None and caught is None:
                raise artifact_error
    return probes, receipt


def load_probe(anchor, output, scale_index, sign, *, ideal=False):
    """Rebuild a sealed head endpoint; actual probes stay F32, ideal probes F64."""
    import json

    output = Path(output)
    receipt = json.loads((output / 'receipt.json').read_text())
    require(type(scale_index) is int and 0 <= scale_index < len(SCALES) and type(sign) is int and sign in (-1, 1),
            'invalid registered endpoint identity')
    require(receipt['status'] == receipt['verification_after'] == 'COMPLETE' and receipt['eligible_for_execution'],
            'incomplete endpoint receipt')
    _validate_frozen(anchor)
    require(mapping_sha256(anchor.state_dict()) == receipt['anchor_mapping_sha256'], 'anchor identity differs')
    matches = [r for r in receipt['probes'] if (r['scale_index'], r['sign']) == (scale_index, sign)]
    require(len(matches) == 1 and matches[0]['status'] == 'COMPLETE', 'missing or duplicate sealed endpoint')
    row = matches[0]
    prefix = 'ideal' if ideal else 'actual'
    path = output / row[prefix + '_file']
    require(hashlib.sha256(path.read_bytes()).hexdigest() == row[prefix + '_file_sha256'], 'endpoint file identity differs')
    head = torch.load(path, map_location='cpu', weights_only=True)
    dtype = torch.float64 if ideal else torch.float32
    require(head['weight'].dtype == head['bias'].dtype == dtype and head['weight'].shape == (16, 128)
            and head['bias'].shape == (16,) and bool(torch.isfinite(head['weight']).all())
            and bool(torch.isfinite(head['bias']).all()), 'endpoint head schema differs')
    actor = _actor_with_head(anchor, head['weight'], head['bias'], actual=not ideal)
    require(mapping_sha256(actor.state_dict()) == row[prefix + '_mapping_sha256'], 'endpoint mapping differs')
    return actor
