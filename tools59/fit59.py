"""Proposed59 paired state-only/state+yaw-JVP objective; no run authorization.

The caller must SHA-bind sources, complete53 training data, and every58 input
before calling this module, then verify them afterwards under the serial CPU
resource guard. No heldout data, policy update, physics or checkpoint selection
exists here.48 label rows are12 states ×4 scales, not48 independent samples.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tools52 import dynamics52 as frozen
from tools56 import fit56 as paired

MODEL_SEEDS = (59001, 59002, 59003)
ARMS = ('state_only', 'state_plus_yaw_jvp')
LABEL_SEEDS = (53103, 53106, 53109, 53112)
LABEL_TICKS = (110, 260, 410)
EPSILONS = (.000003125, .00000625, .0000125, .000025)
LABEL_KEYS = ('s', 'a0', 'a_plus', 'a_minus', 'v', 'target', 'seed', 'tick', 'epsilon')
HORIZON, UPDATES, BATCH_SIZE = 8, 1000, 64
require = frozen.require


def validate_labels(labels):
    """Require all48 ordered rows; preserve raw float64 FD targets/provenance."""
    require(all(key in labels for key in LABEL_KEYS), 'missing59 label field')
    values = {key: np.asarray(labels[key]) for key in LABEL_KEYS}
    expected = [(seed, tick, epsilon) for seed in LABEL_SEEDS for tick in LABEL_TICKS for epsilon in EPSILONS]
    require(all(values[key].shape == (48,) for key in ('target', 'seed', 'tick', 'epsilon')), '48 labels required')
    require(all(np.issubdtype(values[key].dtype, np.integer) for key in ('seed', 'tick')), 'integer label provenance required')
    require(list(zip(values['seed'].tolist(), values['tick'].tolist(), values['epsilon'].tolist(), strict=True)) == expected,
            'all12 states ×4 eps required in fixed order; no dropped/duplicate cases')
    for key, width in (('s', 45), ('a0', 16), ('a_plus', 16), ('a_minus', 16), ('v', 16)):
        require(values[key].shape == (48, width), 'invalid label shape: ' + key)
    require(all(np.isfinite(value).all() for value in values.values()), 'nonfinite label')
    require(values['target'].dtype == np.float64 and values['v'].dtype == np.float32,
            'raw target must remain float64; quantized-action direction must be float32')
    for key in ('a0', 'a_plus', 'a_minus'):
        require(np.abs(values[key]).max() <= 1 and np.array_equal(values[key], values[key].astype(np.float32)),
                'actual clipped float32 action required: ' + key)
    direction = np.zeros(16, dtype=np.float64)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    for key, sign in (('a_plus', 1), ('a_minus', -1)):
        expected_action = np.clip(values['a0'].astype(np.float64) + sign * values['epsilon'][:, None] * direction,
                                  -1., 1.).astype(np.float32)
        require(np.array_equal(values[key], expected_action), 'absolute unnormalized wheel perturbation required')
    actual_v = ((values['a_plus'].astype(np.float64) - values['a_minus'].astype(np.float64))
                / (2 * values['epsilon'][:, None])).astype(np.float32)
    require(np.array_equal(actual_v, values['v']), 'v must use actual float32 actions, differenced in float64')
    require(np.array_equal((values['a_plus'].astype(np.float64) + values['a_minus'].astype(np.float64)) / 2,
                           values['a0'].astype(np.float64)), 'physical FD midpoint must equal a0')
    for first in range(0, 48, 4):
        require(all(np.array_equal(values[key][first:first + 4], np.repeat(values[key][first:first + 1], 4, axis=0))
                    for key in ('s', 'a0')), 'each four-eps group must share one state/action')
    return values


def load_labels58(directory):
    """Load own SHA-bound58 artifacts; caller owns all source/raw SHA admission.

    target comes directly from58's float64 central_state45_fd[:,28]. Neither
    next-state endpoint is cast tofloat32 before subtraction. Action v likewise
    subtracts the actual executed float32 values after promotion tofloat64,
    divides by2δ, then casts tofloat32 without normalization or yaw-error gain.
    """
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['status'] == 'COMPLETE' and report['counts']['policy_transitions'] == 2520
            and report['all_four_complete_baselines_exact'] and report['labels_admitted'] == 12,
            'complete58 replay and all12 admitted points required')
    require(len(report['points']) == 12, 'all12 points required')
    rows = {key: [] for key in LABEL_KEYS}
    expected_points = [(seed, tick) for seed in LABEL_SEEDS for tick in LABEL_TICKS]
    for index, (point, identity) in enumerate(zip(report['points'], expected_points, strict=True)):
        require((point['seed'], point['tick']) == identity and point['status'] == 'COMPLETE'
                and point['label_admitted'] and len(point['branches']) == 10
                and all(branch['status'] == 'COMPLETE' for branch in point['branches']), 'ineligible/dropped58 point')
        require([branch['delta'] for branch in point['branches']] ==
                [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))], '58 action grid differs')
        branches = []
        for order in range(10):
            with np.load(directory / f'branch_{index:02d}_{order:02d}.npz', allow_pickle=False) as raw:
                branch = {key: raw[key].copy() for key in ('s', 'applied_action', 'target')}
            require(branch['s'].shape == branch['target'].shape == (1, 45)
                    and branch['applied_action'].shape == (1, 16), 'one complete transition per branch required')
            branches.append(branch)
        require(all(np.array_equal(branches[0][key], branches[1][key]) for key in branches[0]), 'zero branches differ')
        with np.load(directory / f'response_{index:02d}.npz', allow_pickle=False) as raw:
            response = raw['central_state45_fd'].copy()
        require(response.shape == (4, 45) and response.dtype == np.float64
                and np.isfinite(response).all(), 'original float64 response required')
        for order, epsilon in enumerate(EPSILONS):
            plus, minus = branches[2 + 2 * order:4 + 2 * order]
            require(np.array_equal(plus['s'], branches[0]['s']) and np.array_equal(minus['s'], branches[0]['s']),
                    'physical FD must share a pre-state')
            actual = (plus['target'].astype(np.float64) - minus['target'].astype(np.float64))[0] / (2 * epsilon)
            require(np.array_equal(actual, response[order]), '58 saved response differs from raw float64 FD')
            a_plus, a_minus = plus['applied_action'][0], minus['applied_action'][0]
            row = {'s': branches[0]['s'][0], 'a0': branches[0]['applied_action'][0],
                   'a_plus': a_plus, 'a_minus': a_minus,
                   'v': ((a_plus.astype(np.float64) - a_minus.astype(np.float64)) / (2 * epsilon)).astype(np.float32),
                   'target': response[order, 28], 'seed': identity[0], 'tick': identity[1], 'epsilon': epsilon}
            for key, value in row.items():
                rows[key].append(value)
    return validate_labels({key: np.asarray(value) for key, value in rows.items()})


def labels32(labels):
    """Create detached CPU float32 training tensors, after raw-label validation."""
    return {key: torch.as_tensor(labels[key], dtype=torch.float32, device='cpu').detach()
            for key in ('s', 'a0', 'v', 'target')}


def yaw_jvp(model, labels, *, create_graph):
    """J_a model(s,a)[:,28]·v; graph=True retains mixed derivatives to weights."""
    data = labels32(labels)
    require(data['s'].shape == (48, 45) and data['a0'].shape == data['v'].shape == (48, 16)
            and data['target'].shape == (48,), 'all48 auxiliary samples required')
    require(all(bool(torch.isfinite(value).all()) for value in data.values()), 'nonfinite auxiliary tensors')
    _, response = torch.autograd.functional.jvp(lambda action: model(data['s'], action)[:, 28],
                                               data['a0'], data['v'], create_graph=create_graph, strict=False)
    require(bool(torch.isfinite(response).all()), 'nonfinite model yaw JVP')
    return response


def auxiliary_loss(model, labels, *, create_graph=True):
    response = yaw_jvp(model, labels, create_graph=create_graph)
    target = torch.as_tensor(labels['target'], dtype=torch.float32, device='cpu').detach()
    return ((response - target) / float(frozen.PHYSICAL_SCALE[28])).square().mean()


def objective(state_loss, auxiliary, weight):
    require(math.isfinite(weight) and weight >= 0, 'finite nonnegative objective weight required')
    return state_loss + weight * auxiliary


def initialize_model(data, seed):
    """Both arms use the same frozen ridge/statistics and seed-specific weights."""
    require(type(seed) is int and seed in MODEL_SEEDS, 'unregistered59 seed')
    ridge = frozen.fit_ridge(data)
    torch.manual_seed(seed)
    model = frozen.Dynamics52(**ridge)
    require(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                for value in (*model.parameters(), *model.buffers())), 'CPU float32 model required')
    require(all(parameter.requires_grad for parameter in model.parameters()), 'trainable initial parameters required')
    return model, paired.mapping_sha256(ridge)


def calibrate_weight(model, train, labels, deadline):
    """Train-only S0/R0; all common32 windows, H8 loss, and all48 JVP rows.

    Last calibration batch is weighted by its actual number of windows. Initial
    prediction hashes exclude seed-specific hidden weights: zero residual output
    must make predictions and lambda identical across all paired initializations.
    No parameter .grad or optimizer state is populated by calibration. The
    detached scalar balances initial loss contributions, not gradient norms.
    """
    paired._deadline(deadline, 'before lambda calibration')
    data = frozen.validate_data(train)
    validate_labels(labels)
    windows = paired.common_window_indices(data)
    require(windows.shape == (6828, 32), 'lambda calibration requires all6828 original windows')
    tensors = {key: torch.as_tensor(data[key], dtype=torch.float32) for key in ('s', 'a', 'target')}
    initial_hash = paired.mapping_sha256(model.state_dict())
    prediction_hash = hashlib.sha256()
    weighted, sizes = [], []
    with torch.no_grad():
        for first in range(0, len(windows), BATCH_SIZE):
            paired._deadline(deadline, 'during state calibration')
            indices = torch.from_numpy(windows[first:first + BATCH_SIZE, :HORIZON])
            prediction = frozen.rollout(model, tensors['s'][indices[:, 0]], tensors['a'][indices], return_trajectory=True)
            loss = frozen.group_loss(prediction, tensors['target'][indices])
            size = len(indices)
            weighted.append(float(loss) * size)
            sizes.append(size)
            prediction_hash.update(paired.array_sha256(prediction.numpy()).encode())
    state_initial = math.fsum(weighted) / sum(sizes)
    paired._deadline(deadline, 'before JVP calibration')
    response = yaw_jvp(model, labels, create_graph=False)
    target = torch.as_tensor(labels['target'], dtype=torch.float32)
    auxiliary_initial = float(((response - target) / float(frozen.PHYSICAL_SCALE[28])).square().mean())
    require(math.isfinite(state_initial) and state_initial > 0
            and math.isfinite(auxiliary_initial) and auxiliary_initial > 0, 'positive finite S0/R0 required')
    weight = float(state_initial / auxiliary_initial)
    require(math.isfinite(weight) and weight > 0, 'positive finite calibrated lambda required')
    require(paired.mapping_sha256(model.state_dict()) == initial_hash, 'calibration changed model state')
    paired._deadline(deadline, 'after lambda calibration')
    return {'S0': state_initial, 'R0': auxiliary_initial, 'lambda': weight,
            'lambda_sha256': paired.array_sha256(np.asarray([state_initial, auxiliary_initial, weight], dtype=np.float64)),
            'initial_state_predictions_sha256': prediction_hash.hexdigest(),
            'initial_yaw_jvp_sha256': paired.array_sha256(response.detach().numpy()),
            'common_windows': len(windows), 'state_horizon': HORIZON, 'batches': len(sizes),
            'last_batch_windows': sizes[-1], 'weighted_window_count': sum(sizes),
            'label_rows': 48, 'independent_label_states': 12, 'train_only': True,
            'lambda_detached_python_float': True, 'balance': 'initial loss contributions only; not gradient norms'}


def fit_model(data, labels, seed, arm, output, cpu_deadline):
    """One proposed final-only fit; caller owns authorization and identity binds.

    Expired admission creates no fit directory. Once admitted, every failure
    retains a partial receipt and flushed update/event prefix without resuming.
    Both arms evaluate all48 auxiliary rows with create_graph=True every update;
    the only objective difference is effective lambda0 versus calibrated lambda.
    """
    require(type(seed) is int and seed in MODEL_SEEDS and arm in ARMS, 'unregistered59 seed/arm')
    require(isinstance(cpu_deadline, (float, int)) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
    paired._deadline(cpu_deadline, 'before fit admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'status': 'INCOMPLETE', 'seed': seed, 'arm': arm, 'horizon': HORIZON,
               'common_window_horizon': 32, 'planned_updates': UPDATES, 'updates': 0, 'updates_attempted': 0,
               'batch_size': BATCH_SIZE, 'batch_seed': paired.BATCH_SEED, 'learning_rate': 3e-4,
               'gradient_clip_norm': 1., 'ridge_lambda': .001, 'dtype': 'float32', 'device': 'cpu',
               'label_rows_per_update': 48, 'independent_label_states': 12, 'auxiliary_computed_in_both_arms': True,
               'checkpoint_selection': 'final_only', 'retry_or_resume': False, 'eligible_for_evaluation': False,
               'failure': None, 'stage': 'created'}
    paired._write_json(output / 'fit_started.json', receipt)
    completed = attempted = 0
    schedule = None
    with (output / 'fit_events.jsonl').open('x', buffering=1) as events, (output / 'updates.jsonl').open('x', buffering=1) as updates:
        def event(name):
            events.write(json.dumps({'event': name, 'stage': receipt['stage'], 'updates_completed': completed,
                                     'updates_attempted': attempted}, allow_nan=False) + '\n')
            events.flush()

        try:
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            event('START')
            arrays, label_arrays = frozen.validate_data(data), validate_labels(labels)
            require(len(arrays['s']) == 7200 and np.array_equal(np.unique(arrays['episode']), np.arange(12))
                    and all(np.count_nonzero(arrays['episode'] == index) == 600 for index in range(12)),
                    'exact original7200 transitions/12 complete600-step episodes required')
            receipt.update(training_data_sha256=paired.mapping_sha256(arrays), labels_sha256=paired.mapping_sha256(label_arrays),
                           training_transitions=7200, training_episodes=list(range(12)), training_windows=6828)
            receipt['stage'] = 'sampling'
            common = paired.common_window_indices(arrays).astype('<i8', copy=False)
            require(common.shape == (6828, 32), 'common32 windows differ')
            windows = torch.from_numpy(common)
            schedule = paired.sampled_window_rows(len(common))
            sampled = schedule.numpy().astype('<i8', copy=False)
            receipt.update(common_windows_sha256=paired.array_sha256(common),
                           planned_sampling_rows_sha256=paired.array_sha256(sampled))
            with (output / 'sampling_plan.npz').open('xb') as stream:
                np.savez_compressed(stream, common_windows=common, sampled_window_rows=sampled)
            paired._deadline(cpu_deadline, 'before ridge/initialization')
            receipt['stage'] = 'initialization'
            model, receipt['ridge_and_normalization_sha256'] = initialize_model(arrays, seed)
            receipt['initial_model_state_sha256'] = paired.mapping_sha256(model.state_dict())
            receipt['stage'] = 'lambda_calibration'
            receipt['calibration'] = calibrate_weight(model, arrays, label_arrays, cpu_deadline)
            receipt['lambda_sha256'] = receipt['calibration']['lambda_sha256']
            receipt['calibrated_lambda'] = receipt['calibration']['lambda']
            weight = 0. if arm == 'state_only' else receipt['calibrated_lambda']
            receipt['effective_lambda'] = weight
            paired._write_json(output / 'pairing_identity.json', {key: value for key, value in receipt.items()
                               if key.endswith('sha256') or key in ('seed', 'arm', 'calibration', 'effective_lambda')})
            tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32) for key in ('s', 'a', 'target')}
            label_tensors = labels32(label_arrays)
            optimizer = torch.optim.Adam(model.residual.parameters(), lr=3e-4)
            receipt['stage'] = 'updates'
            final = None
            for update in range(UPDATES):
                paired._deadline(cpu_deadline, 'during fit')
                chosen = schedule[update]
                attempted += 1
                event('UPDATE_BEGIN')
                state_loss = paired.paired_sequence_loss(model, tensors, windows, chosen, HORIZON)
                auxiliary = auxiliary_loss(model, label_tensors, create_graph=True)
                loss = objective(state_loss, auxiliary, weight)
                require(bool(torch.isfinite(loss)) and bool(torch.isfinite(auxiliary)), 'nonfinite objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                final = {'state_loss': float(state_loss.detach()), 'auxiliary_loss': float(auxiliary.detach()),
                         'weighted_auxiliary_loss': float((weight * auxiliary).detach()), 'loss': float(loss.detach())}
                updates.write(json.dumps({'update': completed, **final, 'gradient_norm_before_clip': float(norm),
                    'cpu_seconds': time.process_time() - started,
                    'sampled_rows_sha256': paired.array_sha256(chosen.numpy().astype('<i8', copy=False))}, allow_nan=False) + '\n')
                updates.flush()
            receipt['stage'] = 'finalization'
            paired._deadline(cpu_deadline, 'after fit')
            require(completed == attempted == UPDATES and all(bool(torch.isfinite(value).all()) for value in model.parameters()),
                    'incomplete/nonfinite final model')
            model.eval().requires_grad_(False)
            with (output / 'final_state_dict.pt').open('xb') as stream:
                torch.save(model.state_dict(), stream)
            paired._deadline(cpu_deadline, 'after serialization')
            receipt.update(status='COMPLETE', stage='complete', updates=completed, updates_attempted=attempted,
                           eligible_for_evaluation=True, final_training_minibatch=final, weights_file='final_state_dict.pt',
                           final_model_state_sha256=paired.mapping_sha256(model.state_dict()),
                           completed_sampling_rows_sha256=paired.array_sha256(sampled[:completed]),
                           cpu_seconds=time.process_time() - started, loss_groups=list(frozen.GROUPS),
                           physical_scale45=frozen.PHYSICAL_SCALE.tolist(),
                           pd_semantics='implicit in executed-action to20ms state prediction')
            event('COMPLETE')
            paired._write_json(output / 'fit_receipt.json', receipt)
            return model, receipt
        except BaseException as exc:
            receipt.update(status='INCOMPLETE', eligible_for_evaluation=False, updates=completed, updates_attempted=attempted,
                           failure=f'{type(exc).__name__}: {exc}', cpu_seconds=time.process_time() - started,
                           final_weights_written=(output / 'final_state_dict.pt').exists())
            if schedule is not None:
                receipt['completed_sampling_rows_sha256'] = paired.array_sha256(schedule[:completed].numpy().astype('<i8', copy=False))
            event('INCOMPLETE')
            with (output / 'failure.txt').open('x') as stream:
                stream.write(traceback.format_exc())
            paired._write_json(output / 'partial_receipt.json', receipt)
            raise
