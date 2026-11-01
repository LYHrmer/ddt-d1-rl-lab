"""Proposed64 paired label-source intervention; this module authorizes no run.

Both arms retain the original53 state loss and all48 noisy-center58 labels.
Their second48 labels are60 clean-current/noisy-history (control) or62 clean
history (active).96 rows are averaged together at the fixed complete59 weight;
there is no recalibration, sign filtering, new state-loss data or model change.
The caller must SHA-bind every source/raw input and diagnose both final models
on all three48-row sources without selecting weights.62's original coverage
rejection remains unchanged. Distinct states are not independent episodes.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from tools52 import dynamics52 as frozen
from tools56 import fit56 as paired
from tools59 import fit59 as legacy

MODEL_SEEDS = (64001, 64002, 64003)
ARMS = ('control', 'active')
SOURCE_KEYS = ('58', '60', '62')
SECOND_SOURCE = {'control': '60', 'active': '62'}
FIXED_LAMBDA = 0.08980082414438752
LABEL_SEEDS, LABEL_TICKS, EPSILONS = legacy.LABEL_SEEDS, legacy.LABEL_TICKS, legacy.EPSILONS
LABEL_KEYS = legacy.LABEL_KEYS
HORIZON, UPDATES, BATCH_SIZE = 8, 1000, 64
require = frozen.require
labels32, objective = legacy.labels32, legacy.objective
load_labels58 = legacy.load_labels58


def _load_clean_labels(directory, source):
    """Load all60/62 raw float64 labels; source/root SHA checks belong to caller."""
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    require(source in ('60', '62') and report['status'] == 'COMPLETE', 'complete60/62 source required')
    require(report['counts']['policy_transitions'] == (2640 if source == '60' else 2520),
            'source physical work differs')
    require([case['seed'] for case in report['cases']] == list(LABEL_SEEDS)
            and all(case['status'] == 'COMPLETE' and case['steps'] == 600 for case in report['cases']),
            'all four complete600-step cases required')
    if source == '60':
        require(report['all_four_complete_baselines_exact']
                and report['labels_admitted_by_center']['clean'] == 12, 'complete60 clean-center labels required')
    else:
        require(report['all_four_complete_initial_states_exact'] and report['labels_admitted'] == 12,
                'complete62 reset-admitted labels required')
        # Mixed signs are intentional64 inputs. Do not require62 coverage support.
    expected_points = [(seed, tick) for seed in LABEL_SEEDS for tick in LABEL_TICKS]
    require(len(report['points']) == 12, 'all12 source points required')
    rows = {key: [] for key in LABEL_KEYS}
    deltas = [0., 0., *(sign * epsilon for epsilon in EPSILONS for sign in (1, -1))]
    for index, (point, identity) in enumerate(zip(report['points'], expected_points, strict=True)):
        selected = point['centers']['clean'] if source == '60' else point
        require((point['seed'], point['tick']) == identity and point['status'] == selected['status'] == 'COMPLETE'
                and selected['label_admitted'] and len(selected['branches']) == 10
                and all(branch['status'] == 'COMPLETE' for branch in selected['branches']), 'ineligible source point')
        require([branch['delta'] for branch in selected['branches']] == deltas, 'source action grid differs')
        suffix = '_clean' if source == '60' else ''
        branches = []
        for order in range(10):
            with np.load(directory / f'branch_{index:02d}{suffix}_{order:02d}.npz', allow_pickle=False) as raw:
                branch = {key: raw[key].copy() for key in ('s', 'applied_action', 'target')}
            require(branch['s'].shape == branch['target'].shape == (1, 45)
                    and branch['applied_action'].shape == (1, 16)
                    and branch['target'].dtype == np.float64, 'complete original float64 transition required')
            branches.append(branch)
        require(all(np.array_equal(branches[0][key], branches[1][key]) for key in branches[0]), 'zero branches differ')
        with np.load(directory / f'response_{index:02d}{suffix}.npz', allow_pickle=False) as raw:
            response = raw['central_state45_fd'].copy()
        require(response.shape == (4, 45) and response.dtype == np.float64
                and np.isfinite(response).all(), 'original float64 response required')
        for order, epsilon in enumerate(EPSILONS):
            plus, minus = branches[2 + 2 * order:4 + 2 * order]
            require(np.array_equal(plus['s'], branches[0]['s']) and np.array_equal(minus['s'], branches[0]['s']),
                    'physical FD must share a pre-state')
            actual = (plus['target'].astype(np.float64) - minus['target'].astype(np.float64))[0] / (2 * epsilon)
            require(np.array_equal(actual, response[order]), 'saved response differs from raw float64 FD')
            a_plus, a_minus = plus['applied_action'][0], minus['applied_action'][0]
            row = {'s': branches[0]['s'][0], 'a0': branches[0]['applied_action'][0],
                   'a_plus': a_plus, 'a_minus': a_minus,
                   'v': ((a_plus.astype(np.float64) - a_minus.astype(np.float64)) / (2 * epsilon)).astype(np.float32),
                   'target': response[order, 28], 'seed': identity[0], 'tick': identity[1], 'epsilon': epsilon}
            for key, value in row.items():
                rows[key].append(value)
    return legacy.validate_labels({key: np.asarray(value) for key, value in rows.items()})


def load_labels60(directory):
    return _load_clean_labels(directory, '60')


def load_labels62(directory):
    return _load_clean_labels(directory, '62')


def load_label_sources(directory58, directory60, directory62):
    """Return all three complete48-row sources; no sign/seed/epsilon selection."""
    sources = {'58': load_labels58(directory58), '60': load_labels60(directory60), '62': load_labels62(directory62)}
    require(np.array_equal(sources['58']['s'], sources['60']['s']), '58 and60 must share the replayed states')
    return sources


def validate_labels(labels, arm):
    """Require ordered58+60/62 halves and actual-action float64 FD provenance."""
    require(arm in ARMS and all(key in labels for key in (*LABEL_KEYS, 'source_id')), 'invalid64 arm/label fields')
    values = {key: np.asarray(labels[key]) for key in (*LABEL_KEYS, 'source_id')}
    require(all(len(value) == 96 for value in values.values()), 'all96 auxiliary rows required')
    source_ids = values['source_id']
    require(source_ids.shape == (96,) and np.issubdtype(source_ids.dtype, np.integer)
            and np.array_equal(source_ids, np.r_[np.full(48, 58), np.full(48, int(SECOND_SOURCE[arm]))]),
            'fixed58 plus assigned second source required')
    for begin in (0, 48):
        legacy.validate_labels({key: values[key][begin:begin + 48] for key in LABEL_KEYS})
    if arm == 'control':
        require(np.array_equal(values['s'][:48], values['s'][48:]), 'control must retain identical58/60 states')
    distinct = len(np.unique(values['s'], axis=0))
    require(distinct == (12 if arm == 'control' else 24), 'registered distinct-state count differs')
    return values


def compose_labels(sources, arm):
    """Exactly96 rows: shared58 first, then60(control) or62(active), all signs."""
    require(set(sources) == set(SOURCE_KEYS) and arm in ARMS, 'all three source sets and registered arm required')
    checked = {key: legacy.validate_labels(sources[key]) for key in SOURCE_KEYS}
    require(np.array_equal(checked['58']['s'], checked['60']['s']), '58/60 state identity differs')
    second = checked[SECOND_SOURCE[arm]]
    require(all(checked['58'][key].dtype == second[key].dtype for key in LABEL_KEYS), 'source array dtypes differ')
    result = {key: np.concatenate((checked['58'][key], second[key]), axis=0) for key in LABEL_KEYS}
    result['source_id'] = np.r_[np.full(48, 58, dtype=np.int64), np.full(48, int(SECOND_SOURCE[arm]), dtype=np.int64)]
    return validate_labels(result, arm)


def yaw_jvp(model, labels, *, create_graph):
    """J_a model(s,a)[:,28]·v;96 training rows or48 diagnostic rows, CPU float32."""
    data = labels32(labels)
    count = len(data['s'])
    require(count in (48, 96) and data['s'].shape == (count, 45)
            and data['a0'].shape == data['v'].shape == (count, 16)
            and data['target'].shape == (count,), 'complete48/96-row auxiliary tensors required')
    require(all(bool(torch.isfinite(value).all()) for value in data.values()), 'nonfinite auxiliary tensors')
    _, response = torch.autograd.functional.jvp(lambda action: model(data['s'], action)[:, 28],
                                               data['a0'], data['v'], create_graph=create_graph, strict=False)
    require(bool(torch.isfinite(response).all()), 'nonfinite model yaw JVP')
    return response


def auxiliary_loss(model, labels, *, create_graph=True):
    """One global mean over all96 training rows; no per-sign/source reweighting."""
    response = yaw_jvp(model, labels, create_graph=create_graph)
    target = torch.as_tensor(labels['target'], dtype=torch.float32, device='cpu').detach()
    return ((response - target) / float(frozen.PHYSICAL_SCALE[28])).square().mean()


def initialize_model(data, seed):
    """Unchanged52 ridge/residual/statistics and seed-paired float32 initialization."""
    require(type(seed) is int and seed in MODEL_SEEDS, 'unregistered64 seed')
    ridge = frozen.fit_ridge(data)
    torch.manual_seed(seed)
    model = frozen.Dynamics52(**ridge)
    require(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                for value in (*model.parameters(), *model.buffers())), 'CPU float32 model required')
    require(all(parameter.requires_grad for parameter in model.parameters()), 'trainable initial parameters required')
    return model, paired.mapping_sha256(ridge)


def fit_model(data, labels, seed, arm, output, cpu_deadline):
    """One proposed final-only fit; caller owns authorization and identity binds.

    Expired admission creates no fit directory. Once admitted, every failure
    retains a partial receipt and flushed update/event prefix without resuming.
    Both arms evaluate all96 auxiliary rows with create_graph=True every update.
    Only the second48-row source differs; the inherited59 lambda is fixed and
    never calibrated against these labels. The caller separately diagnoses the
    final weights on each of58/60/62 without selecting a checkpoint.
    """
    require(type(seed) is int and seed in MODEL_SEEDS and arm in ARMS, 'unregistered64 seed/arm')
    require(isinstance(cpu_deadline, (float, int)) and math.isfinite(cpu_deadline), 'finite CPU deadline required')
    paired._deadline(cpu_deadline, 'before fit admission')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.process_time()
    receipt = {'status': 'INCOMPLETE', 'seed': seed, 'arm': arm, 'horizon': HORIZON,
               'common_window_horizon': 32, 'planned_updates': UPDATES, 'updates': 0, 'updates_attempted': 0,
               'batch_size': BATCH_SIZE, 'batch_seed': paired.BATCH_SEED, 'learning_rate': 3e-4,
               'gradient_clip_norm': 1., 'ridge_lambda': .001, 'dtype': 'float32', 'device': 'cpu',
               'label_rows_per_update': 96, 'independent_label_states': 12 if arm == 'control' else 24,
               'state_count_semantics': 'distinct exact physical state vectors, not independent episodes',
               'auxiliary_computed_in_both_arms': True, 'auxiliary_reduction': 'global mean over96 rows',
               'fixed_lambda': FIXED_LAMBDA, 'effective_lambda': FIXED_LAMBDA, 'calibration_performed': False,
               'lambda_source': 'complete59 value, separately SHA-bound by caller',
               'second_label_source': SECOND_SOURCE[arm],
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
            arrays, label_arrays = frozen.validate_data(data), validate_labels(labels, arm)
            require(len(arrays['s']) == 7200 and np.array_equal(np.unique(arrays['episode']), np.arange(12))
                    and all(np.count_nonzero(arrays['episode'] == index) == 600 for index in range(12)),
                    'exact original7200 transitions/12 complete600-step episodes required')
            common_labels = {key: label_arrays[key][:48] for key in LABEL_KEYS}
            second_labels = {key: label_arrays[key][48:] for key in LABEL_KEYS}
            receipt.update(training_data_sha256=paired.mapping_sha256(arrays), labels_sha256=paired.mapping_sha256(label_arrays),
                           common58_labels_sha256=paired.mapping_sha256(common_labels),
                           second_labels_sha256=paired.mapping_sha256(second_labels),
                           lambda_sha256=paired.array_sha256(np.asarray([FIXED_LAMBDA], dtype=np.float64)),
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
            receipt['stage'] = 'fixed_weight_identity'
            weight = FIXED_LAMBDA  # Detached Python constant; never a label-dependent ratio.
            paired._deadline(cpu_deadline, 'before initial common58 function identity')
            with torch.no_grad():
                initial_prediction = model(torch.as_tensor(common_labels['s'], dtype=torch.float32),
                                           torch.as_tensor(common_labels['a0'], dtype=torch.float32))
            initial_jvp = yaw_jvp(model, common_labels, create_graph=False)
            receipt.update(initial_common58_predictions_sha256=paired.array_sha256(initial_prediction.numpy()),
                           initial_common58_yaw_jvp_sha256=paired.array_sha256(initial_jvp.detach().numpy()))
            require(paired.mapping_sha256(model.state_dict()) == receipt['initial_model_state_sha256'],
                    'initial function checks changed parameters')
            paired._write_json(output / 'pairing_identity.json', {key: value for key, value in receipt.items()
                               if key.endswith('sha256') or key in ('seed', 'arm', 'fixed_lambda', 'effective_lambda', 'second_label_source')})
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
