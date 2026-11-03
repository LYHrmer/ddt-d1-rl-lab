"""Independent81 saved-tensor arithmetic audit, using Python's stdlib only."""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import io
import itertools
import json
import math
import pickle
import struct
import time
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

WHEELS = (3, 7, 11, 15)
OTHERS = tuple(i for i in range(16) if i not in WHEELS)
HEAD = ('action_head.weight', 'action_head.bias')
SEEDS = (64001, 64002, 64003)
ARMS = ('C6', 'Q6')
SCALES = (.009, .0045, .00225, .001125)
D_SEEDS = (81107, 81108, 81109)
TICKS = (175, 325)
TASKS = ('forward_stop', 'reverse_stop', 'turn_stop')
PLAN = 'docs/experiments81/objective81_preregistered.json'
DTYPES = {'FloatStorage': ('f', '<f4', 4), 'DoubleStorage': ('d', '<f8', 8),
          'LongStorage': ('q', '<i8', 8), 'IntStorage': ('i', '<i4', 4),
          'BoolStorage': ('?', '|b1', 1), 'ByteStorage': ('B', '|u1', 1)}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class StorageType:
    name: str


@dataclass
class Storage:
    kind: StorageType
    data: bytes


class Tensor:
    def __init__(self, storage, offset, shape, stride):
        self.storage, self.offset, self.shape, self.stride = storage, offset, tuple(shape), tuple(stride)
        if len(self.shape) != len(self.stride) or any(type(x) is not int or x < 0 for x in (*self.shape, *self.stride, offset)):
            raise ValueError('unsupported tensor shape/stride/offset')
        self.code, self.dtype, self.width = DTYPES[storage.kind.name]
        self.count = math.prod(self.shape)
        if self.count > 2_000_000:
            raise ValueError('tensor allocation limit exceeded')

    @cached_property
    def raw(self):
        if self.count == 0:
            return b''
        contiguous, stride = True, 1
        for size, actual in zip(reversed(self.shape), reversed(self.stride)):
            contiguous &= actual == stride
            stride *= size
        if contiguous:
            result = self.storage.data[self.offset * self.width:(self.offset + self.count) * self.width]
        else:
            pieces = []
            for index in itertools.product(*(range(n) for n in self.shape)):
                position = self.offset + sum(i * s for i, s in zip(index, self.stride))
                pieces.append(self.storage.data[position * self.width:(position + 1) * self.width])
            result = b''.join(pieces)
        if len(result) != self.count * self.width:
            raise ValueError('tensor storage is truncated')
        return result

    @cached_property
    def values(self):
        return tuple(item[0] for item in struct.iter_unpack('<' + self.code, self.raw))

    def rows(self, rows):
        if not self.shape:
            raise ValueError('scalar row selection')
        per_row = math.prod(self.shape[1:]) * self.width
        raw = b''.join(self.raw[i * per_row:(i + 1) * per_row] for i in rows)
        return contiguous_tensor(raw, self.storage.kind, (len(rows), *self.shape[1:]))


def contiguous_tensor(raw, kind, shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.insert(0, step)
        step *= size
    return Tensor(Storage(kind, raw), 0, shape, strides)


def rebuild(storage, offset, shape, stride, *_ignored):
    if not isinstance(storage, Storage):
        raise TypeError('unsupported tensor storage')
    return Tensor(storage, offset, shape, stride)


class TensorUnpickler(pickle.Unpickler):
    """An allowlist for torch.save state dictionaries; never imports torch."""
    def __init__(self, stream, archive, prefix):
        super().__init__(stream)
        self.archive, self.prefix, self.storages = archive, prefix, {}

    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch' and name in DTYPES:
            return StorageType(name)
        if module == 'torch._utils' and name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
            return rebuild
        raise pickle.UnpicklingError(f'unsupported global {module}.{name}')

    def persistent_load(self, identifier):
        if not isinstance(identifier, tuple) or len(identifier) != 5 or identifier[0] != 'storage':
            raise pickle.UnpicklingError('unsupported persistent identifier')
        _, kind, key, location, count = identifier
        if not isinstance(kind, StorageType) or location != 'cpu' or type(count) is not int or count < 0:
            raise pickle.UnpicklingError('unsupported storage metadata')
        if not isinstance(key, str) or not key.isdigit():
            raise pickle.UnpicklingError('invalid storage key')
        if key not in self.storages:
            data = self.archive.read(self.prefix + 'data/' + key)
            if len(data) != count * DTYPES[kind.name][2]:
                raise ValueError('storage byte count differs')
            self.storages[key] = Storage(kind, data)
        storage = self.storages[key]
        if storage.kind != kind or len(storage.data) != count * DTYPES[kind.name][2]:
            raise ValueError('inconsistent shared storage')
        return storage


def load_pt_bytes(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 32 << 20:
            raise ValueError('duplicate members or oversized torch archive')
        picks = [name for name in names if name.endswith('/data.pkl')]
        if len(picks) != 1:
            raise ValueError('exactly one tensor pickle required')
        prefix = picks[0][:-len('data.pkl')]
        if archive.read(prefix + 'byteorder') != b'little':
            raise ValueError('only the recorded little-endian archives are supported')
        return TensorUnpickler(io.BytesIO(archive.read(picks[0])), archive, prefix).load()


def mapping_sha(values):
    h = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name]
        if not isinstance(tensor, Tensor):
            raise TypeError('state mapping contains a non-tensor')
        header = json.dumps({'shape': list(tensor.shape), 'dtype': tensor.dtype},
                            sort_keys=True, separators=(',', ':')).encode()
        h.update(name.encode() + b'\0' + header + b'\0' + tensor.raw)
    return h.hexdigest()


def protected_sha(values):
    subset = dict(values)
    for name in HEAD:
        subset[name] = subset[name].rows(OTHERS)
    return mapping_sha(subset)


def norm(values):
    return math.sqrt(math.fsum(value * value for value in values))


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class Audit:
    def __init__(self, repo, run):
        self.repo, self.run = repo, run
        self.checks, self.failures, self.inputs, self.max_gaps = 0, [], {}, {}
        self.check_counts = collections.Counter()
        self.started = time.process_time()
        self.completed_stages = []

    def check(self, label, okay, detail=None):
        self.checks += 1
        self.check_counts[label] += 1
        if not okay:
            self.failures.append({'check': label, 'detail': detail})

    def close(self, label, actual, expected, *, atol=2e-14, rtol=2e-12):
        gap = abs(actual - expected)
        self.max_gaps[label] = max(self.max_gaps.get(label, 0.), gap)
        self.check(label, math.isfinite(actual) and math.isfinite(expected)
                   and gap <= atol + rtol * abs(expected),
                   {'actual': actual, 'expected': expected, 'gap': gap})

    def path(self, path):
        path = Path(path).resolve()
        current = digest(path)
        if str(path) in self.inputs:
            self.check('repeated_input_identity', current == self.inputs[str(path)], str(path))
        self.inputs[str(path)] = current
        return path

    def json(self, path):
        return json.loads(self.path(path).read_text())

    def pt(self, path):
        return load_pt_bytes(self.path(path).read_bytes())

    def tensor(self, tensor, shape, dtype, label):
        self.check(label, isinstance(tensor, Tensor) and tensor.shape == shape and tensor.dtype == dtype)
        self.check('tensor_all_finite', all(math.isfinite(v) for v in tensor.values), label)

    def point_summary(self, row):
        errors = row['yaw_error']
        self.check('saved_yaw_error_horizon', len(errors) == 32 and all(math.isfinite(x) for x in errors))
        self.close('point_yaw_mae_from_errors', row['yaw_mae'], math.fsum(abs(x) for x in errors) / 32)
        self.check('point_yaw_signs', row['yaw_error_sign'] == [1 if x > 0 else -1 if x < 0 else 0 for x in errors])
        self.check('point_exact_zeros', row['exact_zero_yaw_errors'] == sum(x == 0 for x in errors))
        self.close('point_minimum_abs_error', row['minimum_absolute_yaw_error'], min(map(abs, errors)))
        self.check('point_components_and_action_size', len(row['cost_components_mean']) == 6
                   and len(row['first_action']) == 16)
        self.close('point_cost_components_sum', row['cost'], math.fsum(row['cost_components_mean']))

    def objective_summary(self, summary, identities):
        points = summary['points']
        self.check('objective_point_grid', [(p['seed'], p['tick']) for p in points] == identities)
        for row in points:
            self.point_summary(row)
        cost = math.fsum(p['cost'] for p in points) / 6
        q = [max(points[i]['yaw_mae'], points[i + 1]['yaw_mae']) for i in (0, 2, 4)]
        self.close('C6_equal_six_point_mean', summary['C6'], cost)
        self.close('Q6_equal_three_reset_max_mean', summary['Q6'], math.fsum(q) / 3)
        for actual, expected in zip(summary['per_seed_q'], q, strict=True):
            self.close('per_reset_Q_max', actual, expected)
        choices = []
        for i in (0, 2, 4):
            pair = points[i:i + 2]
            top = max(p['yaw_mae'] for p in pair)
            choices.append({'seed': pair[0]['seed'], 'tie': pair[0]['yaw_mae'] == pair[1]['yaw_mae'],
                            'max_ticks': [p['tick'] for p in pair if p['yaw_mae'] == top]})
        self.check('recorded_max_choices', summary['max_choices'] == choices)

    def stage(self, seed, arm, anchor, anchor64, model, published, sealed_by_id):
        folder = self.run / f'seed{seed}/actor_{arm}'
        receipt = self.json(folder / 'receipt.json')
        self.check('published_receipt_exact', receipt == published)
        self.check('stage_complete', receipt['status'] == receipt['verification_after'] == 'COMPLETE'
                   and receipt['eligible_for_execution'] is True and receipt['failure'] is None)
        self.check('stage_protocol', receipt['schema'] == 'd1-policy81-direction-v1' and receipt['arm'] == arm
                   and receipt['horizon'] == 32 and receipt['scales'] == list(SCALES)
                   and receipt['main_probe'] == [0, 1] and receipt['normalization'] == 'negative_masked_gradient_div_max_wheel_row_l1'
                   and receipt['wheel_rows'] == list(WHEELS) and receipt['allowed_parameter_count'] == 516
                   and receipt['execution_bound'] == .01 and receipt['projection'] is False
                   and receipt['gradient_clip'] is False and receipt['model_updates'] == 0
                   and receipt['selfchecks_are_diagnostic_only'] is True and receipt['direction_magnitude_floor'] is None)
        self.check('exact_one_candidate_update', receipt['updates'] == receipt['updates_completed']
                   == receipt['updates_attempted'] == 1)
        anchor_sha = mapping_sha(anchor)
        self.check('original_anchor_mapping', receipt['anchor_mapping_sha256'] == receipt['actor_before_mapping_sha256']
                   == receipt['source_mapping_sha256']['anchor'] == anchor_sha)
        self.check('source_model_mapping', receipt['source_mapping_sha256']['model'] == mapping_sha(model))
        self.check('protected_source_mapping', receipt['protected_before_mapping_sha256'] == protected_sha(anchor))
        h = self.pt(folder / receipt['head_step_file'])
        self.check('head_step_sha', digest(folder / receipt['head_step_file']) == receipt['head_step_file_sha256'])
        expected_keys = {'anchor_mapping_sha256', *(prefix + kind + '64' for prefix in
                        ('head_before_', 'masked_gradient_', 'direction_') for kind in ('weight', 'bias'))}
        self.check('exact_direction_artifact_schema', set(h) == expected_keys and h['anchor_mapping_sha256'] == anchor_sha)
        for name in sorted(expected_keys - {'anchor_mapping_sha256'}):
            self.tensor(h[name], (16, 128) if 'weight' in name else (16,), '<f8', 'direction_tensor_schema')
        gw, gb = h['masked_gradient_weight64'].values, h['masked_gradient_bias64'].values
        dw, db = h['direction_weight64'].values, h['direction_bias64'].values
        self.check('anchor_F64_head_exact', h['head_before_weight64'].values == anchor[HEAD[0]].values
                   and h['head_before_bias64'].values == anchor[HEAD[1]].values)
        row_lengths = [math.fsum(abs(gw[r * 128 + c]) for c in range(128)) + abs(gb[r]) for r in WHEELS]
        scale = max(row_lengths)
        self.check('nonzero_finite_gradient_row_norm', math.isfinite(scale) and scale > 0)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('cannot audit normalization of invalid gradient')
        self.close('max_row_L1_gradient', receipt['max_row_l1_gradient'], scale)
        for i, (g, d) in enumerate(zip(gw + gb, dw + db, strict=True)):
            self.close('negative_max_row_L1_direction', d, -g / scale)
        for r in OTHERS:
            self.check('nonwheel_gradient_and_direction_zero', gb[r] == db[r] == 0.
                       and all(gw[r * 128 + c] == dw[r * 128 + c] == 0. for c in range(128)))
        direction_row_lengths = [math.fsum(abs(dw[r * 128 + c]) for c in range(128)) + abs(db[r]) for r in WHEELS]
        self.close('maximum_direction_row_L1_one', max(direction_row_lengths), 1.)
        analytic = math.fsum(g * d for g, d in zip(gw + gb, dw + db, strict=True))
        self.close('directional_AD_dot_saved_gradient', receipt['directional_derivative'], analytic)
        self.check('ideal_linear_direction_negative', analytic < 0.)
        identities = [(s, t) for s in D_SEEDS for t in TICKS]
        self.check('registered_D_points', receipt['point_identities'] == [{'seed': s, 'tick': t} for s, t in identities])
        self.check('full_gradient_and_FD_point_counts', receipt['point_count'] == receipt['gradient_points_completed'] == 6
                   and receipt['fd_points_completed'] == 48 and len(receipt['points']) == 6 and len(receipt['fd_points']) == 48
                   and receipt['model_calls_attempted'] == receipt['model_calls'] == 1728)
        self.objective_summary(receipt['before'], identities)
        expected_point_files = []
        for index, row in enumerate(receipt['points']):
            self.check('gradient_point_order', (row['index'], row['seed'], row['tick']) == (index, *identities[index]))
            expected_point_files.append(f'gradient_point_{index:02d}.json')
            self.check('gradient_point_receipt_parity', self.json(folder / expected_point_files[-1]) == row)
            self.check('gradient_point_summary_parity', {k: v for k, v in row.items() if k != 'index'}
                       == receipt['before']['points'][index])
        expected_probe_ids = [(k, sign) for k in range(4) for sign in (1, -1)]
        self.check('all_eight_probes_once', [(p['scale_index'], p['sign']) for p in receipt['probes']] == expected_probe_ids)
        endpoints = {}
        for probe in receipt['probes']:
            k, sign = probe['scale_index'], probe['sign']
            epsilon = SCALES[k]
            self.check('registered_probe_identity', probe['epsilon'] == epsilon and probe['status'] == 'COMPLETE')
            actual = self.pt(folder / probe['actual_file'])
            ideal = self.pt(folder / probe['ideal_file'])
            self.check('actual_probe_file_sha', digest(folder / probe['actual_file']) == probe['actual_file_sha256'])
            self.check('ideal_probe_file_sha', digest(folder / probe['ideal_file']) == probe['ideal_file_sha256'])
            self.check('actual_probe_members', set(actual) == {'weight', 'bias', 'delta_weight64', 'delta_bias64'})
            self.check('ideal_probe_members', set(ideal) == {'weight', 'bias'})
            full_actual, full_ideal = dict(anchor), dict(anchor64)
            actual_deltas = []
            for kind, width, key in (('weight', 128, HEAD[0]), ('bias', 1, HEAD[1])):
                shape = (16, 128) if width == 128 else (16,)
                self.tensor(actual[kind], shape, '<f4', 'actual_head_schema')
                self.tensor(actual['delta_' + kind + '64'], shape, '<f8', 'actual_delta_schema')
                self.tensor(ideal[kind], shape, '<f8', 'ideal_head_schema')
                av, iv, dv = actual[kind].values, ideal[kind].values, actual['delta_' + kind + '64'].values
                before, direction = h['head_before_' + kind + '64'].values, h['direction_' + kind + '64'].values
                for i in range(len(before)):
                    self.close('ideal_endpoint_coordinate', iv[i], before[i] + sign * epsilon * direction[i], atol=4e-16, rtol=2e-13)
                    self.check('actual_F32_exact_ideal_cast', struct.pack('<f', av[i]) == struct.pack('<f', iv[i]))
                    self.check('actual_F64_delta_exact', dv[i] == av[i] - anchor[key].values[i])
                    if i // width in OTHERS:
                        self.check('nonwheel_endpoint_exact', av[i] == anchor[key].values[i] and iv[i] == before[i] and dv[i] == 0.)
                actual_deltas.extend(dv)
                full_actual[key], full_ideal[key] = actual[kind], ideal[kind]
                old_vals = [anchor[key].values[r * width + c] for r in WHEELS for c in range(width)]
                new_vals = [av[r * width + c] for r in WHEELS for c in range(width)]
                def ordered(x):
                    bits = struct.unpack('<i', struct.pack('<f', x))[0]
                    return 0x80000000 - (bits & 0x7fffffff) if bits < 0 else 0x80000000 + bits
                ulps = [abs(ordered(a) - ordered(b)) for a, b in zip(new_vals, old_vals, strict=True)]
                nonzero = [x for x in ulps if x]
                expected_ulp = {'coordinates': len(ulps), 'changed': len(nonzero),
                    'min_nonzero_ulp': min(nonzero) if nonzero else None, 'max_ulp': max(ulps)}
                self.check('actual_parameter_ULP_summary', probe['parameter_ulps'][kind] == expected_ulp)
            am, im = mapping_sha(full_actual), mapping_sha(full_ideal)
            self.check('actual_reconstructed_complete_mapping', am == probe['actual_mapping_sha256'])
            self.check('ideal_reconstructed_complete_mapping', im == probe['ideal_mapping_sha256'])
            self.check('protected_actual_mapping', protected_sha(full_actual) == protected_sha(anchor))
            self.check('protected_ideal_mapping', protected_sha(full_ideal) == protected_sha(anchor64))
            bounds = [math.fsum(abs(full_actual[HEAD[0]].values[r * 128 + c] - anchor[HEAD[0]].values[r * 128 + c])
                               for c in range(128)) + abs(full_actual[HEAD[1]].values[r] - anchor[HEAD[1]].values[r]) for r in WHEELS]
            self.check('strict_original_anchor_bound', all(b <= .01 for b in bounds))
            for value, expected in zip(probe['final_row_l1'], bounds, strict=True):
                self.close('saved_actual_row_bound', value, expected)
            sealed = sealed_by_id[(seed, arm, k, sign)]
            self.check('sealed48_probe_mapping', sealed['mapping_sha256'] == am)
            self.objective_summary(probe['ideal_D_values'], identities)
            fd_rows = [r for r in receipt['fd_points'] if (r['scale_index'], r['sign']) == (k, sign)]
            self.check('six_FD_points_per_endpoint', len(fd_rows) == 6)
            for index, point in enumerate(fd_rows):
                name = f'fd_{k:02d}_{sign:+d}_point_{index:02d}.json'
                expected_point_files.append(name)
                self.check('FD_point_order', (point['index'], point['seed'], point['tick']) == (index, *identities[index]))
                self.check('FD_point_file_receipt_parity', self.json(folder / name) == point)
                self.check('FD_point_summary_parity', {key: value for key, value in point.items()
                           if key not in ('index', 'scale_index', 'sign')} == probe['ideal_D_values']['points'][index])
            endpoints[(k, sign)] = {'mapping_sha256': am, 'ideal_mapping_sha256': im, 'row_l1': bounds,
                'actual_delta_dot_gradient': math.fsum(x * y for x, y in zip(actual_deltas, gw + gb, strict=True)),
                'actual_changed_coordinates': sum(x != 0 for x in actual_deltas), 'full_actual': full_actual,
                'values': probe['ideal_D_values']}
        self.check('point_file_grid_exact', sorted(p.name for p in folder.glob('*_point_*.json')) == sorted(expected_point_files))
        self.check('four_diagnostic_selfchecks', len(receipt['selfchecks']) == 4)
        for k, saved in enumerate(receipt['selfchecks']):
            p, m = endpoints[(k, 1)]['values'], endpoints[(k, -1)]['values']
            fd = (p[arm] - m[arm]) / (2 * SCALES[k])
            self.close('selfcheck_analytic', saved['analytic'], analytic)
            self.close('selfcheck_FD_from_saved_objectives', saved['ideal_model_fd'], fd)
            self.close('selfcheck_tolerance', saved['tolerance'], 1e-4 + .1 * abs(fd))
            self.close('selfcheck_absolute_error', saved['absolute_error'], abs(saved['analytic'] - fd))
            self.check('selfcheck_bool_recomputed', saved['autograd_agrees'] == (abs(saved['analytic'] - fd) <= 1e-4 + .1 * abs(fd)))
            self.check('selfchecks_not_acceptance', saved['acceptance_gate'] is False and saved['magnitude_floor'] is None
                       and saved['epsilon'] == SCALES[k])
            before = receipt['before']
            self.check('recorded_max_choice_sides', saved['plus_max_choices'] == p['max_choices']
                       and saved['minus_max_choices'] == m['max_choices'] and saved['baseline_max_choices'] == before['max_choices'])
            self.check('max_branch_switch_bool', saved['max_branch_changed'] == any(x['max_choices'] != before['max_choices'] for x in (p, m)))
            self.check('abs_sign_switch_bool', saved['abs_sign_pattern_changed'] == any(
                a['yaw_error_sign'] != b['yaw_error_sign'] for x in (p, m)
                for a, b in zip(x['points'], before['points'], strict=True)))
        after = self.pt(folder / receipt['weights_file'])
        self.check('main_checkpoint_file_sha', digest(folder / receipt['weights_file']) == receipt['weights_file_sha256'])
        main = endpoints[(0, 1)]
        self.check('main_plus009_full_checkpoint', mapping_sha(after) == receipt['actor_after_mapping_sha256'] == main['mapping_sha256'])
        self.check('main_checkpoint_tensor_keys', set(after) == set(anchor))
        for name in sorted(anchor):
            self.check('main_checkpoint_tensor_exact', after[name].raw == main['full_actual'][name].raw
                       and after[name].dtype == '<f4' and after[name].shape == anchor[name].shape, name)
        events = [json.loads(line) for line in self.path(folder / 'events.jsonl').read_text().splitlines()]
        self.check('main_update_journal_once', sum(e['stage'] == 'main_candidate_attempt' for e in events) == 1
                   and sum(e['stage'] == 'main_candidate_constructed' for e in events) == 1
                   and events[-1]['stage'] == 'complete' and events[-1]['updates_completed'] == 1)
        result = {'seed': seed, 'arm': arm, 'updates_completed': receipt['updates_completed'],
            'gradient_points': 6, 'fd_points': 48, 'model_calls': receipt['model_calls'],
            'gradient_max_row_L1': scale, 'direction_max_row_L1': max(direction_row_lengths),
            'directional_derivative': analytic, 'actual_main_delta_dot_gradient': main['actual_delta_dot_gradient'],
            'main_mapping_sha256': main['mapping_sha256'], 'main_checkpoint_sha256': receipt['weights_file_sha256'],
            'diagnostic_autograd_agrees': [r['autograd_agrees'] for r in receipt['selfchecks']],
            'probes': [{'scale_index': k, 'sign': sign, **{key: value for key, value in row.items() if key not in ('full_actual', 'values')}}
                       for (k, sign), row in sorted(endpoints.items())]}
        self.completed_stages.append(result)
        return result

    def run_audit(self):
        report = self.json(self.run / 'report.json')
        if report.get('status') != 'COMPLETE' or report.get('verification_after') != 'COMPLETE':
            raise ValueError('terminal COMPLETE report required; never upgrade a partial run')
        self.check('complete_formal81_report', report['schema'] == 'd1-objective81-result-v2' and report['stage'] == 'experiment')
        self.check('six_candidate_updates', report['policy_network_updates'] == 6 and report['model_updates'] == 0)
        self.check('full_model_forward_schedule', report['model_calls'] == 23328)
        self.check('original_eligibility_preserved', report['global_candidate_accepted'] is False
                   and report['complete_model_eligibility'] == 'NOT_REQUALIFIED' and report['inherited75_and78_decisions_changed'] is False)
        plan = self.json(self.repo / PLAN)
        self.check('plan_report_binding', digest(self.repo / PLAN) == report['plan_sha256'])
        self.check('registered_protocol', plan['schema'] == 'd1-objective81-preregistered-v2' and plan['arms'] == list(ARMS)
                   and plan['model_seeds'] == list(SEEDS) and plan['scales'] == list(SCALES)
                   and plan['direction_seeds'] == list(D_SEEDS) and plan['starts_per_objective'] == 6
                   and plan['policy_updates'] == 6 and plan['model_updates'] == 0 and plan['primary_scale_index'] == 0 and plan['primary_sign'] == 1)
        for name, expected in plan['source_sha256'].items():
            self.check('all_plan_source_bindings', digest(self.path(self.repo / name)) == expected, name)
        parent = self.json(self.repo / 'docs/experiments80/steady_sampling80_preregistered.json')
        self.check('all1368_parent_bindings_unchanged', len(parent['source_sha256']) == 1368
                   and all(plan['source_sha256'].get(k) == v for k, v in parent['source_sha256'].items()))
        original = self.json(self.repo / 'docs/experiments53/model_gradient53_preregistered.json')
        anchor_path = self.path(self.repo / original['checkpoint'])
        self.check('anchor_direct_source_binding', digest(anchor_path) == plan['source_sha256'][original['checkpoint']])
        with zipfile.ZipFile(anchor_path) as z:
            saved_policy = load_pt_bytes(z.read('policy.pth'))
        prefixes = {'mlp_extractor.encoder.': 'encoder.', 'mlp_extractor.policy_net.': 'policy_net.', 'action_net.': 'action_head.'}
        anchor = {replacement + name[len(prefix):]: value for name, value in saved_policy.items()
                  for prefix, replacement in prefixes.items() if name.startswith(prefix)}
        self.check('anchor_mapping_from_complete25', mapping_sha(anchor) == report['anchor_mapping_sha256'])
        self.check('original_anchor_all_F32', all(t.dtype == '<f4' for t in anchor.values()))
        anchor64 = {name: contiguous_tensor(struct.pack('<' + 'd' * t.count, *t.values), StorageType('DoubleStorage'), t.shape)
                    for name, t in anchor.items()}
        ids = self.json(self.run / 'probe_identities.json')
        expected_ids = {(s, a, k, sign) for s in SEEDS for a in ARMS for k in range(4) for sign in (1, -1)}
        sealed = {(r['model_seed'], r['arm'], r['scale_index'], r['sign']): r for r in ids['probes']}
        self.check('all48_sealed_probe_identities', len(ids['probes']) == len(sealed) == 48 and set(sealed) == expected_ids)
        self.check('sealed_fixed_primary_and_anchor', ids['primary_scale_index'] == 0 and ids['primary_sign'] == 1
                   and ids['baseline_mapping_sha256'] == mapping_sha(anchor))
        required_seals = {'development_sealed_before_updates.json', 'policies_sealed_before_validation.json',
                          'validation_sealed_before_predictions.json', 'predictions_sealed_before_physics.json'}
        self.check('all_stage_seals_recorded', set(report['seals']) == required_seals)
        for name, expected in report['seals'].items():
            seal = self.json(self.run / name)
            self.check('seal_file_sha', digest(self.run / name) == expected, name)
            for member, checksum in seal['artifact_sha256'].items():
                self.check('sealed_member_sha', digest(self.path(self.run / member)) == checksum, member)
        policy_seal = self.json(self.run / 'policies_sealed_before_validation.json')['artifact_sha256']
        self.check('probe_identity_in_policy_seal', policy_seal.get('probe_identities.json') == digest(self.run / 'probe_identities.json'))
        source75 = self.json(self.repo / 'evidence/world_vz75/execution_01/report.json')
        self.check('complete_source75', source75['status'] == source75['verification_after'] == 'COMPLETE')
        self.check('three_models_two_arms', len(report['models']) == 3 and {r['seed'] for r in report['models']} == set(SEEDS)
                   and all(set(r['arms']) == set(ARMS) and r['status'] == 'COMPLETE' for r in report['models']))
        expected_receipts = []
        for seed in SEEDS:
            published = next(r for r in report['models'] if r['seed'] == seed)
            old = next(r for r in source75['models'] if r['seed'] == seed and r['arm'] == 'control')
            member = old['directory'] + '/final_state_dict.pt'
            relative = 'evidence/world_vz75/execution_01/' + member
            model = self.pt(self.repo / relative)
            self.check('source75_frozen_file_sha', digest(self.repo / relative) == source75['sealed_artifacts_sha256'][member]
                       == plan['source_sha256'][relative])
            self.check('source75_complete_model_mapping', mapping_sha(model) == published['source_model_sha256']
                       == old['fit']['final_model_state_sha256'])
            for arm in ARMS:
                expected_receipts.append(f'seed{seed}/actor_{arm}/receipt.json')
                self.stage(seed, arm, anchor, anchor64, model, published['arms'][arm], sealed)
        actual_receipts = sorted(str(p.relative_to(self.run)) for p in self.run.glob('seed*/actor_*/receipt.json'))
        self.check('exact_six_receipt_paths', actual_receipts == sorted(expected_receipts))
        accounting = report['update_accounting']
        self.check('durable_six_update_accounting', accounting['policy_network_updates'] == 6 and accounting['model_updates'] == 0
                   and len(accounting['receipts']) == 6 and sorted(r['path'] for r in accounting['receipts']) == sorted(expected_receipts))
        for row in accounting['receipts']:
            self.check('accounting_receipt_sha', row['status'] == 'COMPLETE' and row['updates_completed'] == row['updates_attempted'] == 1
                       and row['sha256'] == digest(self.run / row['path']))
        progress = [json.loads(line) for line in self.path(self.run / 'progress.jsonl').read_text().splitlines()]
        pseal = next(i for i, r in enumerate(progress) if r['event'] == 'ALL_SIX_CANDIDATES_AND_PROBES_SEALED_BEFORE_V')
        vbegin = next(i for i, r in enumerate(progress) if r['event'] == 'BEGIN' and r['phase'].startswith('validation/'))
        prediction_seal = next(i for i, r in enumerate(progress) if r['event'] == 'ALL_PREDICTIONS_SEALED')
        physical_begin = next(i for i, r in enumerate(progress) if r['event'] == 'BEGIN' and r['phase'].startswith('screen/'))
        self.check('recorded_seal_order', pseal < vbegin < prediction_seal < physical_begin)
        for path, expected in list(self.inputs.items()):
            self.check('all_inputs_unchanged_after_audit', digest(path) == expected, path)
            try:
                name = str(Path(path).relative_to(self.run))
            except ValueError:
                continue
            if name != 'report.json':
                self.check('formal_manifest_matches_audited_artifact', report['raw_artifacts_sha256'].get(name) == expected, name)
        result = self.base_result()
        result.update(audit_complete=True, source_report_sha256=digest(self.run / 'report.json'),
            source_plan_sha256=digest(self.repo / PLAN), plan_source_binding_count=len(plan['source_sha256']),
            completed_stages=len(self.completed_stages), actual_updates_completed=sum(s['updates_completed'] for s in self.completed_stages),
            probes_checked=sum(len(s['probes']) for s in self.completed_stages),
            gradient_points=sum(s['gradient_points'] for s in self.completed_stages), fd_points=sum(s['fd_points'] for s in self.completed_stages),
            maximum_actual_row_l1=max(b for s in self.completed_stages for p in s['probes'] for b in p['row_l1']),
            main_linear_prediction_sign_counts=dict(collections.Counter('negative' if s['actual_main_delta_dot_gradient'] < 0
                else 'positive' if s['actual_main_delta_dot_gradient'] > 0 else 'zero' for s in self.completed_stages)))
        return result

    def base_result(self):
        return {'schema': 'd1-objective81-policy-independent-audit-v1', 'status': 'PASS' if not self.failures else 'FAIL',
            'method': 'stdlib ZIP/allowlisted pickle/struct saved-tensor arithmetic; no numeric library or model calls',
            'source_report': str(self.run / 'report.json'), 'auditor_sha256': digest(__file__),
            'checks': self.checks, 'check_counts': dict(self.check_counts), 'failures': self.failures,
            'maximum_arithmetic_gaps': self.max_gaps, 'stages': self.completed_stages,
            'inputs_sha256': self.inputs, 'new_model_calls': 0, 'new_actor_calls': 0,
            'new_physics_steps': 0, 'new_training_updates': 0, 'original_gates_changed': False,
            'cpu_seconds': time.process_time() - self.started,
            'arithmetic_tolerance': {'default_abs': 2e-14, 'default_rel': 2e-12,
                'ideal_endpoint_abs': 4e-16, 'ideal_endpoint_rel': 2e-13,
                'actual_F32_cast_and_delta': 'exact', 'actual_anchor_row_bound': 'strict <=0.01'},
            'limits': ['Saved gradients are not independently differentiated; this audits their recorded normalization,projection-free endpoints and provenance.',
                'No independent per-point parameter gradients were saved; six-point C6/Q6 reductions are checked from saved values,not rederived BPTT.',
                'All F32 probes are reconstructed by replacing archived heads in the complete original anchor; complete main checkpoints independently verify the reconstruction.',
                'Source files and persisted model mappings are independently verified; runtime in-memory .grad preservation remains the producer receipt.',
                'Recorded seal order is checked against the durable event ledger; no external timestamp attestation is claimed.',
                'IdealF64 selfchecks remain diagnostic; no AD/FD outcome is turned into a physical qualification or selection gate.',
                'Negative actual-delta dot saved-gradient is descriptive local linear prediction,not measured physical improvement.',
                'A rowL1 bound applies at the same actor latent,not to changed closed-loop states or all future observations.',
                'Partial formal or audit stages are never promoted to complete.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/home/lyh/ddt_ros2_ws'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = Audit(args.repo.resolve(), args.run.resolve())
    try:
        result = audit.run_audit()
    except BaseException as exc:
        audit.failures.append({'check': 'audit_exception', 'detail': f'{type(exc).__name__}: {exc}'})
        result = audit.base_result()
        result.update(status='FAIL', audit_complete=False, exception=f'{type(exc).__name__}: {exc}')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'audit_complete', 'checks', 'completed_stages',
        'actual_updates_completed', 'probes_checked', 'maximum_actual_row_l1', 'main_linear_prediction_sign_counts', 'cpu_seconds')}))
    return 0 if result['status'] == 'PASS' and result.get('audit_complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
