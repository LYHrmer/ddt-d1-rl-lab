"""Proposed72: double only yaw-response weight; this module authorizes no run.

Both arms retain complete69 active's body term, all96 labels, seeds, window
sampling and1000 final-only updates. Control must reproduce its same-seed69
active mapping exactly; caller owns source/resource admission and that check.
No frozen-module global is changed and no coefficient is calibrated here.
"""
from __future__ import annotations

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
YAW_COEFFICIENT = {'control': FIXED_LAMBDA, 'active': 2 * FIXED_LAMBDA}
BODY_COEFFICIENT = {'control': FIXED_LAMBDA, 'active': FIXED_LAMBDA}
HORIZON, UPDATES, BATCH_SIZE = previous.HORIZON, previous.UPDATES, previous.BATCH_SIZE
LABEL_KEYS, BODY_KEYS = previous.LABEL_KEYS, previous.BODY_KEYS
require, initialize_model = frozen.require, previous.initialize_model
load_labels68, validate_labels = previous.load_labels68, previous.validate_labels
labels32, label_hashes = previous.labels32, previous.label_hashes
yaw_labels, yaw_jvp, auxiliary_loss = previous.yaw_labels, previous.yaw_jvp, previous.auxiliary_loss
body_jvp, body_loss = previous.body_jvp, previous.body_loss


def objective(state_loss, yaw_loss, body, arm):
    """Control calls69 active unchanged; active changes only its yaw scalar."""
    require(arm in ARMS, 'unregistered72 arm')
    if arm == 'control':
        return previous.objective(state_loss, yaw_loss, body, 'active')
    return previous.baseline.objective(state_loss, yaw_loss, YAW_COEFFICIENT[arm]) + BODY_COEFFICIENT[arm] * body


def fit_model(data, labels, seed, arm, output, cpu_deadline):
    """One final-only72 fit; the root runner owns resource/source admission.

    Control keeps the complete69 active objective graph and training protocol.
    Active doubles only yaw lambda; body beta remains the original lambda.
    The runner checks each control final mapping SHA against complete69 active.
    """
    require(type(seed) is int and seed in MODEL_SEEDS and arm in ARMS, 'unregistered72 seed/arm')
    require(isinstance(cpu_deadline, (float, int)) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
    paired._deadline(cpu_deadline, 'before fit admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, beta = time.process_time(), BODY_COEFFICIENT[arm]
    yaw_weight = YAW_COEFFICIENT[arm]
    receipt = {'status': 'INCOMPLETE', 'seed': seed, 'arm': arm, 'horizon': HORIZON,
               'common_window_horizon': 32, 'planned_updates': UPDATES, 'updates': 0, 'updates_attempted': 0,
               'batch_size': BATCH_SIZE, 'batch_seed': paired.BATCH_SEED, 'learning_rate': 3e-4,
               'gradient_clip_norm': 1., 'ridge_lambda': .001, 'dtype': 'float32', 'device': 'cpu',
               'label_rows_per_update': 96, 'independent_label_states': 24,
               'state_count_semantics': 'distinct exact physical state vectors, not independent episodes',
               'auxiliary_computed_in_both_arms': True, 'auxiliary_reduction': 'global mean over96 yaw rows',
               'body_computed_in_both_arms': True, 'body_reduction': 'global mean over96x2 body scalar residuals',
               'body_scale': float(frozen.PHYSICAL_SCALE[23]), 'body_target_raw_dtype': 'float64',
               'fixed_lambda': yaw_weight, 'effective_lambda': yaw_weight,
               'reference_lambda': FIXED_LAMBDA, 'yaw_coefficient_multiplier': 1 if arm == 'control' else 2,
               'fixed_beta': beta, 'body_coefficient': beta, 'calibration_performed': False,
               'second_label_source': '62', 'only_objective_variable': 'yaw JVP coefficient lambda (1x versus 2x)',
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
                           lambda_sha256=paired.array_sha256(np.asarray([yaw_weight], dtype=np.float64)),
                           beta_sha256=paired.array_sha256(np.asarray([beta], dtype=np.float64)),
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
            receipt.update(initial_common58_predictions_sha256=paired.array_sha256(prediction.numpy()),
                           initial_common58_yaw_jvp_sha256=paired.array_sha256(initial_yaw.detach().numpy()),
                           initial_body_jvp_sha256=paired.array_sha256(initial_body.detach().numpy()))
            require(paired.mapping_sha256(model.state_dict()) == receipt['initial_model_state_sha256'],
                    'initial function checks changed parameters')
            paired._write_json(output / 'pairing_identity.json', {key: value for key, value in receipt.items()
                if key.endswith('sha256') or key in ('seed', 'arm', 'fixed_lambda', 'effective_lambda',
                                                     'fixed_beta', 'body_coefficient', 'second_label_source',
                                                     'reference_lambda', 'yaw_coefficient_multiplier')})
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
                loss = objective(state_loss, auxiliary, body, arm)
                require(all(bool(torch.isfinite(value)) for value in (loss, auxiliary, body)), 'nonfinite objective')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                completed += 1
                final = {'state_loss': float(state_loss.detach()), 'auxiliary_loss': float(auxiliary.detach()),
                         'weighted_auxiliary_loss': float((yaw_weight * auxiliary).detach()),
                         'body_loss': float(body.detach()), 'weighted_body_loss': float((beta * body).detach()),
                         'body_coefficient': beta, 'yaw_coefficient': yaw_weight, 'loss': float(loss.detach())}
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
