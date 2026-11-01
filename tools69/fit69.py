"""Proposed69 paired body-response objective; this module authorizes no run.

Both arms use64 active's state/yaw data, seeds, initialization, sampling and
objective. Active adds one fixed-weight body-vx/vy JVP loss using all admitted68
labels. Control computes body loss for logging but never attaches it to its
objective graph. The caller binds all sources, runs serially within its resource
guard, and checks each control final mapping SHA against the complete64 active.
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
from tools54 import numerics54 as numeric
from tools56 import fit56 as paired
from tools64 import fit64 as baseline

MODEL_SEEDS = baseline.MODEL_SEEDS
ARMS = ('control', 'active')
FIXED_LAMBDA = baseline.FIXED_LAMBDA
BODY_COEFFICIENT = {'control': 0., 'active': FIXED_LAMBDA}
HORIZON, UPDATES, BATCH_SIZE = baseline.HORIZON, baseline.UPDATES, baseline.BATCH_SIZE
LABEL_KEYS = baseline.LABEL_KEYS
BODY_KEYS = ('target_body', 'body_channel_qualified', 'source_point_index')
ADMISSION_PLAN_SHA256 = 'b8ca3c2a3d89e9dda50c6b9df263910a0f7d2f722c9e00993e3ba641d956405e'
require = frozen.require
initialize_model = baseline.initialize_model
yaw_jvp, auxiliary_loss = baseline.yaw_jvp, baseline.auxiliary_loss


def yaw_labels(labels):
    """The exact64 active mapping, excluding body-only provenance/targets."""
    return {key: labels[key] for key in (*LABEL_KEYS, 'source_id')}


def validate_labels(labels):
    """Require all96 ordered58+62 rows and all192 admitted body scalar targets."""
    require(all(key in labels for key in (*LABEL_KEYS, 'source_id', *BODY_KEYS)), 'missing69 label field')
    yaw = baseline.validate_labels(yaw_labels(labels), 'active')
    body = {key: np.asarray(labels[key]) for key in BODY_KEYS}
    require(body['target_body'].shape == (96, 2) and body['target_body'].dtype == np.float64
            and np.isfinite(body['target_body']).all(), 'all raw F64 body targets required')
    mask = body['body_channel_qualified']
    require(mask.shape == (96, 2) and mask.dtype == np.bool_ and mask.all(), 'all body labels must be68-qualified')
    require(np.issubdtype(body['source_point_index'].dtype, np.integer)
            and np.array_equal(body['source_point_index'], np.tile(np.repeat(np.arange(12), 4), 2)),
            'all24 body states/four scales in original order required')
    return yaw | body


def load_labels68(directory):
    """Read exact COMPLETE+ADMITTED68; caller must first SHA-bind its report/file.

    Raw targets remain float64, including values below endpoint float32 spacing.
    The only target rename is68 target_yaw -> the frozen64 key target.
    """
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(report['schema'] == 'd1-body-response68-result-v1'
            and report['status'] == report['verification_after'] == 'COMPLETE'
            and report['plan_sha256'] == ADMISSION_PLAN_SHA256 and report['completed_points'] == 24
            and report['label_rows'] == 96 and report['states'] == 24, 'complete fixed68 admission required')
    admission = report['admission']
    require(admission['status'] == 'ADMITTED' and admission['admitted']
            and admission['qualified_state_channels'] == admission['required_state_channels'] == 48
            and admission['rows_dropped'] == 0, 'all48 body state-channel qualifications required')
    expected = [(source, seed, tick) for source in ('58', '62')
                for seed in baseline.LABEL_SEEDS for tick in baseline.LABEL_TICKS]
    require([(row['source'], row['seed'], row['tick']) for row in report['points']] == expected
            and all(row['status'] == 'COMPLETE' and row['old_yaw_response_exact']
                    and all(row['channels'][channel]['stability']['stable'] for channel in ('vx', 'vy'))
                    for row in report['points']), 'complete unfiltered68 point ledger required')
    path = directory / 'body_labels96.npz'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == report['label_file_sha256'], '68 raw label SHA differs')
    with np.load(path, allow_pickle=False) as raw:
        labels = {key: raw['target_yaw' if key == 'target' else key].copy()
                  for key in (*LABEL_KEYS, 'source_id', *BODY_KEYS)}
    return validate_labels(labels)


def labels32(labels):
    """Frozen yaw tensors plus body targets, converted only after raw admission."""
    return baseline.labels32(labels) | {
        'target_body': torch.as_tensor(labels['target_body'], dtype=torch.float32, device='cpu').detach()}


def body_jvp(model, labels, *, create_graph):
    """J_action phi(model(state,action)) v, all96x2, CPU float32 mixed graph."""
    data = labels32(labels)
    require(data['s'].shape == (96, 45) and data['a0'].shape == data['v'].shape == (96, 16)
            and data['target_body'].shape == (96, 2), 'all96 two-channel body rows required')

    def transformed(action):
        predicted = model(data['s'], action)
        return numeric._rotate_inverse(predicted[:, 3:7], predicted[:, 23:26])[:, :2]

    _, response = torch.autograd.functional.jvp(transformed, data['a0'], data['v'],
                                               create_graph=create_graph, strict=False)
    require(bool(torch.isfinite(response).all()), 'nonfinite model body JVP')
    return response


def body_loss(model, labels, *, create_graph=True):
    """One mean over all96*2 scalar residuals; no sign/source/channel weighting."""
    response = body_jvp(model, labels, create_graph=create_graph)
    target = torch.as_tensor(labels['target_body'], dtype=torch.float32, device='cpu').detach()
    return ((response - target) / float(frozen.PHYSICAL_SCALE[23])).square().mean()


def objective(state_loss, yaw_loss, body, arm):
    """Control returns the original64 objective graph without even a0*body term."""
    require(arm in ARMS, 'unregistered69 arm')
    original = baseline.objective(state_loss, yaw_loss, FIXED_LAMBDA)
    if arm == 'control':
        return original
    return original + BODY_COEFFICIENT[arm] * body


def label_hashes(labels):
    """Legacy64 yaw hash remains comparable; body and combined identities are separate."""
    return {'labels_sha256': paired.mapping_sha256(yaw_labels(labels)),
            'body_labels_sha256': paired.mapping_sha256({key: labels[key] for key in
                (*BODY_KEYS, 'source_id', 'seed', 'tick', 'epsilon')}),
            'combined_labels_sha256': paired.mapping_sha256(labels),
            'common58_labels_sha256': paired.mapping_sha256({key: labels[key][:48] for key in LABEL_KEYS}),
            'second_labels_sha256': paired.mapping_sha256({key: labels[key][48:] for key in LABEL_KEYS})}


def fit_model(data, labels, seed, arm, output, cpu_deadline):
    """One final-only69 fit; the root runner owns resource/source admission.

    Original64 active state/yaw arithmetic is unchanged. Both arms calculate
    body loss but beta0 leaves it disconnected. A complete control is only a
    clean reproduction if the runner verifies its final mapping SHA against64.
    """
    require(type(seed) is int and seed in MODEL_SEEDS and arm in ARMS, 'unregistered69 seed/arm')
    require(isinstance(cpu_deadline, (float, int)) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
    paired._deadline(cpu_deadline, 'before fit admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, beta = time.process_time(), BODY_COEFFICIENT[arm]
    receipt = {'status': 'INCOMPLETE', 'seed': seed, 'arm': arm, 'horizon': HORIZON,
               'common_window_horizon': 32, 'planned_updates': UPDATES, 'updates': 0, 'updates_attempted': 0,
               'batch_size': BATCH_SIZE, 'batch_seed': paired.BATCH_SEED, 'learning_rate': 3e-4,
               'gradient_clip_norm': 1., 'ridge_lambda': .001, 'dtype': 'float32', 'device': 'cpu',
               'label_rows_per_update': 96, 'independent_label_states': 24,
               'state_count_semantics': 'distinct exact physical state vectors, not independent episodes',
               'auxiliary_computed_in_both_arms': True, 'auxiliary_reduction': 'global mean over96 yaw rows',
               'body_computed_in_both_arms': True, 'body_reduction': 'global mean over96x2 body scalar residuals',
               'body_scale': float(frozen.PHYSICAL_SCALE[23]), 'body_target_raw_dtype': 'float64',
               'fixed_lambda': FIXED_LAMBDA, 'effective_lambda': FIXED_LAMBDA,
               'fixed_beta': beta, 'body_coefficient': beta, 'calibration_performed': False,
               'second_label_source': '62', 'only_objective_variable': 'body JVP coefficient beta',
               'checkpoint_selection': 'final_only', 'retry_or_resume': False, 'eligible_for_evaluation': False,
               'control_reference': 'complete64 active at the same seed', 'control_reproduction_checked': False,
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
                                                     'fixed_beta', 'body_coefficient', 'second_label_source')})
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
                         'weighted_auxiliary_loss': float((FIXED_LAMBDA * auxiliary).detach()),
                         'body_loss': float(body.detach()), 'weighted_body_loss': float((beta * body).detach()),
                         'body_coefficient': beta, 'loss': float(loss.detach())}
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
