"""Proposed75: add only world-vz JVP supervision to complete69 active.

Both arms keep the original state/yaw/body objective, all96 training labels,
seeds, sampling and1000 final-only updates. Caller owns source/resource binds
and requires same-seed control final mapping SHA to reproduce69 active.
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

from tools69 import fit69 as previous

frozen, paired = previous.frozen, previous.paired
MODEL_SEEDS, ARMS = previous.MODEL_SEEDS, ('control', 'active')
FIXED_LAMBDA = previous.FIXED_LAMBDA
YAW_COEFFICIENT = {'control': FIXED_LAMBDA, 'active': FIXED_LAMBDA}
BODY_COEFFICIENT = {'control': FIXED_LAMBDA, 'active': FIXED_LAMBDA}
WORLD_COEFFICIENT = {'control': 0., 'active': FIXED_LAMBDA}
HORIZON, UPDATES, BATCH_SIZE = previous.HORIZON, previous.UPDATES, previous.BATCH_SIZE
LABEL_KEYS, BODY_KEYS = previous.LABEL_KEYS, previous.BODY_KEYS
WORLD_KEYS = ('target_world_vz', 'world_vz_channel_qualified')
BASE_KEYS = (*LABEL_KEYS, 'source_id', *BODY_KEYS)
require, initialize_model = frozen.require, previous.initialize_model
yaw_labels, yaw_jvp, auxiliary_loss = previous.yaw_labels, previous.yaw_jvp, previous.auxiliary_loss
body_jvp, body_loss = previous.body_jvp, previous.body_loss


def validate_labels(labels):
    """All ordered69 labels plus all96 qualified raw F64 world-vz targets."""
    require(all(key in labels for key in (*BASE_KEYS, *WORLD_KEYS)), 'missing75 label field')
    original = previous.validate_labels({key: labels[key] for key in BASE_KEYS})
    target, mask = (np.asarray(labels[key]) for key in WORLD_KEYS)
    require(target.shape == (96,) and target.dtype == np.float64 and np.isfinite(target).all(),
            'all96 raw F64 world-vz targets required')
    require(mask.shape == (96,) and mask.dtype == np.bool_ and mask.all(), 'all24 world-vz states must be qualified')
    return original | dict(zip(WORLD_KEYS, (target, mask), strict=True))


def load_labels75(directory):
    """Read bound complete75 labels without hardcoding a yet-unsealed plan SHA."""
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['schema'] == 'd1-world-vz75-labels-v1'
            and report['status'] == report['verification_after'] == 'COMPLETE'
            and report['label_rows'] == 96, 'complete75 label report required')
    admission = report['admission']
    require(admission['admitted'] is True and admission['qualified_states'] == admission['required_states'] == 24
            and admission['rows_dropped'] == 0, 'all24 world-vz states required without dropped rows')
    path = directory / 'world_vz_labels96.npz'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == report['label_file_sha256'], '75 raw label SHA differs')
    with np.load(path, allow_pickle=False) as raw:
        labels = {key: raw['target_yaw' if key == 'target' else key].copy() for key in (*BASE_KEYS, *WORLD_KEYS)}
    return validate_labels(labels)


def labels32(labels):
    return previous.labels32(labels) | {
        'target_world_vz': torch.as_tensor(labels['target_world_vz'], dtype=torch.float32, device='cpu').detach()}


def world_vz_jvp(model, labels, *, create_graph):
    """CPU F32 J_action model(s,a)[:,25] v for all96 unfiltered label rows."""
    data = labels32(labels)
    require(data['s'].shape == (96, 45) and data['a0'].shape == data['v'].shape == (96, 16)
            and data['target_world_vz'].shape == (96,), 'all96 scalar world-vz rows required')
    _, response = torch.autograd.functional.jvp(lambda action: model(data['s'], action)[:, 25],
                                               data['a0'], data['v'], create_graph=create_graph, strict=False)
    require(bool(torch.isfinite(response).all()), 'nonfinite world-vz JVP')
    return response


def world_vz_loss(model, labels, *, create_graph=True):
    response = world_vz_jvp(model, labels, create_graph=create_graph)
    target = torch.as_tensor(labels['target_world_vz'], dtype=torch.float32, device='cpu').detach()
    return ((response - target) / float(frozen.PHYSICAL_SCALE[25])).square().mean()


def objective(state_loss, yaw_loss, body, world, arm):
    """Control returns69 active directly; no zero-times-world graph is attached."""
    require(arm in ARMS, 'unregistered75 arm')
    original = previous.objective(state_loss, yaw_loss, body, 'active')
    if arm == 'control':
        return original
    return original + WORLD_COEFFICIENT[arm] * world


def label_hashes(labels):
    """Keep69 subset identities unchanged;75-only provenance hashes are separate."""
    return previous.label_hashes({key: labels[key] for key in BASE_KEYS}) | {
        'world_vz_labels_sha256': paired.mapping_sha256({key: labels[key] for key in
            (*WORLD_KEYS, 'source_id', 'seed', 'tick', 'epsilon')}),
        'combined75_labels_sha256': paired.mapping_sha256(labels)}


def final_training_world_vz_diagnostics(model, labels, output):
    """Observe all96 final F32 responses in a new exclusive output directory.

    Targets are stored both raw F64 and exactly cast for training. No update,
    rounding for presentation, label selection or new qualification is applied.
    """
    labels = validate_labels(labels)
    require(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                for value in (*model.parameters(), *model.buffers())), 'final diagnostic requires CPU F32 model')
    model_hash = paired.mapping_sha256(model.state_dict())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'INCOMPLETE', 'label_rows': 96, 'distinct_states': 24, 'rows': [],
               'dtype': 'float32', 'device': 'cpu', 'target_raw_dtype': 'float64',
               'model_state_sha256': model_hash, **label_hashes(labels),
               'parameter_updates': 0, 'used_for_training_manipulation_gate': True,
               'used_for_selection': False, 'failure': None}
    try:
        response = world_vz_jvp(model, labels, create_graph=False).detach()
        target = torch.as_tensor(labels['target_world_vz'], dtype=torch.float32)
        normalized = (response - target) / float(frozen.PHYSICAL_SCALE[25])
        require(response.shape == (96,) and bool(torch.isfinite(normalized).all()), 'nonfinite final world-vz diagnostic')
        for index in range(96):
            receipt['rows'].append({'source': str(int(labels['source_id'][index])),
                'seed': int(labels['seed'][index]), 'tick': int(labels['tick'][index]),
                'epsilon': float(labels['epsilon'][index]), 'target_float64': float(labels['target_world_vz'][index]),
                'target_training_float32': float(target[index]), 'predicted_world_vz_jvp': float(response[index]),
                'normalized_error': float(normalized[index])})
        path = output / 'final_train_world_vz.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, predicted_world_vz_jvp=response.numpy(), target_float64=labels['target_world_vz'],
                                target_training_float32=target.numpy(), normalized_error=normalized.numpy(),
                                **{key: labels[key] for key in ('source_id', 'seed', 'tick', 'epsilon', 'world_vz_channel_qualified')})
        receipt.update(world_vz_loss=float(normalized.square().mean()),
                       normalized_rmse=float(normalized.square().mean().sqrt()),
                       diagnostic_file=path.name, diagnostic_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        require(paired.mapping_sha256(model.state_dict()) == model_hash, 'final diagnostic changed weights')
        receipt['status'] = 'COMPLETE'
    except BaseException as exc:
        receipt['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        paired._write_json(output / 'final_train_world_vz.json', receipt)
    return receipt


def fit_model(data, labels, seed, arm, output, cpu_deadline):
    """One final-only75 fit; the root runner owns resource/source admission.

    Both arms keep69 active state/yaw/body arithmetic; only gamma differs.
    Control logs world-vz loss without attaching that graph to its objective.
    The runner verifies each control final mapping SHA against complete69 active.
    """
    require(type(seed) is int and seed in MODEL_SEEDS and arm in ARMS, 'unregistered75 seed/arm')
    require(isinstance(cpu_deadline, (float, int)) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
    paired._deadline(cpu_deadline, 'before fit admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, beta, gamma = time.process_time(), BODY_COEFFICIENT[arm], WORLD_COEFFICIENT[arm]
    receipt = {'status': 'INCOMPLETE', 'seed': seed, 'arm': arm, 'horizon': HORIZON,
               'common_window_horizon': 32, 'planned_updates': UPDATES, 'updates': 0, 'updates_attempted': 0,
               'batch_size': BATCH_SIZE, 'batch_seed': paired.BATCH_SEED, 'learning_rate': 3e-4,
               'gradient_clip_norm': 1., 'ridge_lambda': .001, 'dtype': 'float32', 'device': 'cpu',
               'label_rows_per_update': 96, 'independent_label_states': 24,
               'state_count_semantics': 'distinct exact physical state vectors, not independent episodes',
               'auxiliary_computed_in_both_arms': True, 'auxiliary_reduction': 'global mean over96 yaw rows',
               'body_computed_in_both_arms': True, 'body_reduction': 'global mean over96x2 body scalar residuals',
               'body_scale': float(frozen.PHYSICAL_SCALE[23]), 'body_target_raw_dtype': 'float64',
               'fixed_lambda': FIXED_LAMBDA, 'effective_lambda': FIXED_LAMBDA, 'reference_lambda': FIXED_LAMBDA,
               'fixed_beta': beta, 'body_coefficient': beta, 'fixed_gamma': gamma, 'world_vz_coefficient': gamma,
               'world_vz_scale': float(frozen.PHYSICAL_SCALE[25]), 'world_vz_computed_in_both_arms': True,
               'world_vz_reduction': 'global mean over96 scalar rows', 'world_vz_target_raw_dtype': 'float64',
               'calibration_performed': False,
               'second_label_source': '62', 'only_objective_variable': 'world-vz JVP coefficient gamma',
               'checkpoint_selection': 'final_only', 'retry_or_resume': False, 'eligible_for_evaluation': False,
               'control_reference': 'complete69 active at the same seed', 'control_reproduction_checked': False,
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
            receipt.update(training_data_sha256=paired.mapping_sha256(arrays), **label_hashes(label_arrays),
                           lambda_sha256=paired.array_sha256(np.asarray([FIXED_LAMBDA], dtype=np.float64)),
                           beta_sha256=paired.array_sha256(np.asarray([beta], dtype=np.float64)),
                           gamma_sha256=paired.array_sha256(np.asarray([gamma], dtype=np.float64)),
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
            common_labels = {key: label_arrays[key][:48] for key in LABEL_KEYS}
            paired._deadline(cpu_deadline, 'before initial function identity')
            with torch.no_grad():
                prediction = model(torch.as_tensor(common_labels['s'], dtype=torch.float32),
                                   torch.as_tensor(common_labels['a0'], dtype=torch.float32))
            initial_yaw = yaw_jvp(model, common_labels, create_graph=False)
            initial_body = body_jvp(model, label_arrays, create_graph=False)
            initial_world = world_vz_jvp(model, label_arrays, create_graph=False)
            receipt.update(initial_common58_predictions_sha256=paired.array_sha256(prediction.numpy()),
                           initial_common58_yaw_jvp_sha256=paired.array_sha256(initial_yaw.detach().numpy()),
                           initial_body_jvp_sha256=paired.array_sha256(initial_body.detach().numpy()),
                           initial_world_vz_jvp_sha256=paired.array_sha256(initial_world.detach().numpy()))
            require(paired.mapping_sha256(model.state_dict()) == receipt['initial_model_state_sha256'],
                    'initial function checks changed parameters')
            paired._write_json(output / 'pairing_identity.json', {key: value for key, value in receipt.items()
                if key.endswith('sha256') or key in ('seed', 'arm', 'fixed_lambda', 'effective_lambda',
                                                     'fixed_beta', 'body_coefficient', 'fixed_gamma', 'world_vz_coefficient',
                                                     'reference_lambda', 'second_label_source')})
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
                body = body_loss(model, label_tensors, create_graph=True)
                world = world_vz_loss(model, label_tensors, create_graph=True)
                loss = objective(state_loss, auxiliary, body, world, arm)
                require(all(bool(torch.isfinite(value)) for value in (loss, auxiliary, body, world)), 'nonfinite objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                final = {'state_loss': float(state_loss.detach()), 'auxiliary_loss': float(auxiliary.detach()),
                         'weighted_auxiliary_loss': float((FIXED_LAMBDA * auxiliary).detach()),
                         'body_loss': float(body.detach()), 'weighted_body_loss': float((beta * body).detach()),
                         'body_coefficient': beta, 'world_vz_loss': float(world.detach()),
                         'weighted_world_vz_loss': float((gamma * world).detach()),
                         'world_vz_coefficient': gamma, 'loss': float(loss.detach())}
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
