"""One complete79 feedback refit, preserving the original75 control objective.

The caller binds a complete75 control checkpoint and serial resource guard.
This module copies its weights; it neither resumes an old optimizer nor changes
the source model, ridge, input statistics, physical scales or training labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tools75 import fit75

previous, frozen, paired = fit75.previous, fit75.frozen, fit75.paired
UPDATES, HORIZON, COMMON_HORIZON = 100, 8, 32
BATCH_PER_SOURCE, SAMPLING_SEED = 32, 79000
FIXED_LAMBDA = .08980082414438752
require = frozen.require


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def validate_replay(old_data, online_data):
    """Return two complete datasets and their separate legal32-step windows."""
    arrays, windows = [], []
    for name, data, count, episodes in (('old', old_data, 7200, 12), ('online', online_data, 600, 1)):
        checked = frozen.validate_data(data)
        ids, lengths = np.unique(checked['episode'], return_counts=True)
        require(len(checked['s']) == count and len(ids) == episodes and bool((lengths == 600).all()),
                name + ' requires complete600-step episodes only')
        require(name != 'old' or np.array_equal(ids, np.arange(12)), 'old episode identities must remain0..11')
        for key in ('s', 'a', 'target'):
            require(np.issubdtype(checked[key].dtype, np.floating), name + ' must contain floating transitions')
        for key in ('terminated', 'truncated', 'fallen', 'illegal_body_contact'):
            if key in data:
                flags = np.asarray(data[key])
                require(flags.shape == (count,) and flags.dtype == np.bool_ and not bool(flags.any()),
                        name + ' contains incomplete/unsafe collection flags')
        indices = frozen.window_indices(checked, COMMON_HORIZON)
        require(indices.shape == (episodes * 569, COMMON_HORIZON), name + ' window count differs')
        arrays.append(checked)
        windows.append(torch.as_tensor(indices, dtype=torch.int64, device='cpu'))
    return (*arrays, *windows)


def sample_rows(old_count=6828, online_count=569):
    """Interleave32 old then32 new draws each step, using only a local RNG."""
    require(old_count == 6828 and online_count == 569, 'registered replay window counts required')
    rng = torch.Generator(device='cpu').manual_seed(SAMPLING_SEED)
    old, online = [], []
    for _ in range(UPDATES):
        old.append(torch.randint(old_count, (BATCH_PER_SOURCE,), generator=rng, device='cpu'))
        online.append(torch.randint(online_count, (BATCH_PER_SOURCE,), generator=rng, device='cpu'))
    return torch.stack(old), torch.stack(online)


def mixed_state_loss(model, old, online, old_windows, online_windows, old_rows, online_rows):
    """One original seven-group H8 loss on a balanced64-window batch."""
    require(old_rows.shape == online_rows.shape == (BATCH_PER_SOURCE,), '32+32 window rows required')
    oi, ni = old_windows[old_rows, :HORIZON], online_windows[online_rows, :HORIZON]
    initial = torch.cat((old['s'][oi[:, 0]], online['s'][ni[:, 0]]), dim=0)
    actions = torch.cat((old['a'][oi], online['a'][ni]), dim=0)
    targets = torch.cat((old['target'][oi], online['target'][ni]), dim=0)
    return frozen.sequence_loss(model, initial, actions, targets)


def objective(state_loss, yaw_loss, body_loss):
    """Exact69 active/75 control operation order; no world-vz term attached."""
    require(previous.FIXED_LAMBDA == FIXED_LAMBDA, 'frozen objective coefficient differs')
    return previous.objective(state_loss, yaw_loss, body_loss, 'active')


def frozen_mapping(model):
    return {name: value for name, value in model.state_dict().items() if not name.startswith('residual.')}


def clone_trainable(source_model):
    """Copy a CPU F32 Dynamics52, enabling only its original residual tensors."""
    require(isinstance(source_model, frozen.Dynamics52), 'complete Dynamics52 source required from caller')
    require(all(t.device.type == 'cpu' and t.dtype == torch.float32 and bool(torch.isfinite(t).all())
                for t in (*source_model.parameters(), *source_model.buffers())), 'finite CPU F32 source required')
    model = copy.deepcopy(source_model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('residual.'))
        parameter.grad = None
    for buffer in model.buffers():
        buffer.requires_grad_(False)
    model.train()
    require(any(p.requires_grad for p in model.parameters()), 'empty residual parameter set')
    require(paired.mapping_sha256(model.state_dict()) == paired.mapping_sha256(source_model.state_dict()),
            'copied initial weights differ')
    return model


def _source_identity(model):
    return {'mapping': paired.mapping_sha256(model.state_dict()),
            'gradients': paired.mapping_sha256({n: p.grad for n, p in model.named_parameters() if p.grad is not None}),
            'requires_grad': {n: p.requires_grad for n, p in model.named_parameters()},
            'training': {n: m.training for n, m in model.named_modules()}}


def refit_model(source_model, old_data, online_data, labels, output, *, cpu_deadline):
    """Run100 final-only updates or raise after preserving an ineligible receipt.

    receipt.json is always written after exclusive directory admission, including
    deadline/data failures. Completed counts mean optimizer.step returned; a
    failure midway through an attempted step is never counted as completed.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'schema': 'd1-feedback79-refit-v1', 'status': 'INCOMPLETE', 'stage': 'created',
        'planned_updates': UPDATES, 'updates_completed': 0, 'updates_attempted': 0, 'updates': 0,
        'horizon': HORIZON, 'common_window_horizon': COMMON_HORIZON, 'batch_size': 64,
        'batch_old': 32, 'batch_online': 32, 'batch_seed': SAMPLING_SEED,
        'learning_rate': 3e-4, 'gradient_clip_norm': 1., 'optimizer': 'new Adam; no imported optimizer state',
        'dtype': 'float32', 'device': 'cpu', 'fixed_lambda': FIXED_LAMBDA,
        'body_coefficient': FIXED_LAMBDA, 'world_vz_coefficient': 0., 'label_rows_per_update': 96,
        'checkpoint_selection': 'final_only', 'retry_or_resume': False, 'eligible_for_evaluation': False,
        'source_completeness_scope': 'caller must bind the complete75 control fit receipt and checkpoint',
        'failure': None, 'verification_failure': None, 'source_model_unchanged': None}
    _write(output / 'initial_incomplete.json', receipt)
    model = source_before = primary_error = None
    try:
        require(isinstance(cpu_deadline, (int, float)) and not isinstance(cpu_deadline, bool)
                and math.isfinite(cpu_deadline), 'finite absolute CPU deadline required')
        paired._deadline(cpu_deadline, 'before79 refit admission')
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        source_before = _source_identity(source_model)
        receipt['source_model_state_sha256'] = source_before['mapping']
        receipt['stage'] = 'data_admission'
        old, online, ow, nw = validate_replay(old_data, online_data)
        admitted_labels = previous.validate_labels(labels)
        old_rows, online_rows = sample_rows(len(ow), len(nw))
        receipt.update(old_transitions=7200, online_transitions=600, old_windows=6828, online_windows=569,
            old_data_sha256=paired.mapping_sha256(old), online_data_sha256=paired.mapping_sha256(online),
            old_windows_sha256=paired.array_sha256(ow.numpy()), online_windows_sha256=paired.array_sha256(nw.numpy()),
            sampling_sha256=paired.mapping_sha256({'old_rows': old_rows, 'online_rows': online_rows}),
            **previous.label_hashes(admitted_labels))
        with (output / 'sampling_plan.npz').open('xb') as stream:
            np.savez_compressed(stream, old_windows=ow.numpy(), online_windows=nw.numpy(),
                                old_rows=old_rows.numpy(), online_rows=online_rows.numpy())
        paired._deadline(cpu_deadline, 'before79 source copy')
        model = clone_trainable(source_model)
        receipt.update(initial_model_state_sha256=paired.mapping_sha256(model.state_dict()),
            frozen_mapping_sha256_before=paired.mapping_sha256(frozen_mapping(model)),
            trainable_parameter_names=[n for n, p in model.named_parameters() if p.requires_grad],
            trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
        old_t = {k: torch.as_tensor(old[k], dtype=torch.float32, device='cpu') for k in ('s', 'a', 'target')}
        new_t = {k: torch.as_tensor(online[k], dtype=torch.float32, device='cpu') for k in ('s', 'a', 'target')}
        label_t = previous.labels32(admitted_labels)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=3e-4)
        receipt['stage'] = 'updates'
        with (output / 'events.jsonl').open('x', buffering=1) as events, (output / 'updates.jsonl').open('x', buffering=1) as updates:
            for index in range(UPDATES):
                paired._deadline(cpu_deadline, 'during79 refit')
                receipt['updates_attempted'] += 1
                _append(events, {'event': 'UPDATE_BEGIN', 'updates_attempted': receipt['updates_attempted'],
                                 'updates_completed': receipt['updates_completed']})
                state = mixed_state_loss(model, old_t, new_t, ow, nw, old_rows[index], online_rows[index])
                yaw = previous.auxiliary_loss(model, label_t, create_graph=True)
                body = previous.body_loss(model, label_t, create_graph=True)
                loss = objective(state, yaw, body)
                require(all(bool(torch.isfinite(v)) for v in (state, yaw, body, loss)), 'nonfinite79 loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                receipt['updates_completed'] += 1
                receipt['updates'] = receipt['updates_completed']
                metrics = {'update': index + 1, 'updates_completed': receipt['updates_completed'],
                    'updates_attempted': receipt['updates_attempted'], 'state_loss': float(state.detach()),
                    'yaw_loss': float(yaw.detach()), 'body_loss': float(body.detach()),
                    'weighted_yaw_loss': float((FIXED_LAMBDA * yaw).detach()),
                    'weighted_body_loss': float((FIXED_LAMBDA * body).detach()), 'loss': float(loss.detach()),
                    'gradient_norm_before_clip': float(norm), 'cpu_seconds': time.process_time() - started}
                _append(updates, metrics)
                _append(events, {'event': 'UPDATE_COMPLETE', 'updates_completed': receipt['updates_completed'],
                                 'updates_attempted': receipt['updates_attempted']})
                receipt['final_training_minibatch'] = metrics
        receipt['stage'] = 'finalization'
        paired._deadline(cpu_deadline, 'after79 updates')
        require(receipt['updates_completed'] == receipt['updates_attempted'] == UPDATES,
                'incomplete79 optimizer steps')
        require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()), 'nonfinite final79 weights')
        model.eval().requires_grad_(False)
        receipt['final_model_state_sha256'] = paired.mapping_sha256(model.state_dict())
        receipt['model_changed'] = receipt['final_model_state_sha256'] != receipt['initial_model_state_sha256']
        with (output / 'final_state_dict.pt').open('xb') as stream:
            torch.save(model.state_dict(), stream)
        receipt.update(weights_file='final_state_dict.pt', weights_file_sha256=hashlib.sha256(
            (output / 'final_state_dict.pt').read_bytes()).hexdigest())
        paired._deadline(cpu_deadline, 'after79 serialization')
        receipt.update(status='COMPLETE', stage='complete', eligible_for_evaluation=True)
    except BaseException as exc:
        primary_error = exc
        receipt.update(status='INCOMPLETE', failure=f'{type(exc).__name__}: {exc}', eligible_for_evaluation=False)
        with (output / 'failure.txt').open('x', encoding='utf-8') as stream:
            stream.write(traceback.format_exc())
        raise
    finally:
        verification_error = None
        try:
            if source_before is not None:
                receipt['source_model_unchanged'] = _source_identity(source_model) == source_before
                require(receipt['source_model_unchanged'], '79 refit changed source weights/gradients/flags')
            if model is not None:
                receipt['frozen_mapping_sha256_after'] = paired.mapping_sha256(frozen_mapping(model))
                require(receipt['frozen_mapping_sha256_after'] == receipt['frozen_mapping_sha256_before'],
                        '79 refit changed frozen buffers/parameters')
        except BaseException as exc:  # noqa: BLE001 - late verification must revoke eligibility and preserve the receipt.
            verification_error = exc
            receipt.update(status='INCOMPLETE', eligible_for_evaluation=False,
                           verification_failure=f'{type(exc).__name__}: {exc}')
        if receipt['status'] != 'COMPLETE' and model is not None and not (output / 'final_state_dict.pt').exists():
            try:
                with (output / 'partial_state_dict.pt').open('xb') as stream:
                    torch.save(model.state_dict(), stream)
                receipt['partial_weights_file'] = 'partial_state_dict.pt'
            except BaseException as exc:  # noqa: BLE001 - preserve the primary failure if partial serialization is interrupted.
                receipt['partial_weights_failure'] = f'{type(exc).__name__}: {exc}'
        receipt['cpu_seconds'] = time.process_time() - started
        _write(output / 'receipt.json', receipt)
        if verification_error is not None and primary_error is None:
            raise verification_error
    return model, receipt
