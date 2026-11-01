"""Synthetic world-vz label admission; no archived assets or model execution."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools75 import labels75


@pytest.fixture(scope='module')
def lib():
    return SimpleNamespace(np=pytest.importorskip('numpy'), mod=labels75)


def original_labels(lib):
    np, mod = lib.np, lib.mod
    s = np.zeros((96, 45), dtype=np.float64)
    s[:, 0] = np.repeat(np.arange(24), 4)
    s[:, 3] = 1.
    epsilon = np.tile(mod.EPSILONS, 24)
    a0 = np.full((96, 16), .3125, dtype=np.float64)
    direction = np.zeros(16, dtype=np.float64)
    direction[[3, 7, 11, 15]] = (1., -1., 1., -1.)
    plus = (a0 + epsilon[:, None] * direction).astype(np.float32).astype(np.float64)
    minus = (a0 - epsilon[:, None] * direction).astype(np.float32).astype(np.float64)
    labels = {'s': s, 'a0': a0, 'a_plus': plus, 'a_minus': minus,
              'v': ((plus - minus) / (2 * epsilon[:, None])).astype(np.float32),
              'seed': np.tile(np.repeat(mod.SEEDS, 12), 2).astype(np.int64),
              'tick': np.tile(np.repeat(mod.TICKS, 4), 8).astype(np.int64),
              'epsilon': epsilon, 'target_yaw': np.full(96, -.2, dtype=np.float64),
              'target_body': np.full((96, 2), -.3, dtype=np.float64),
              'source_id': np.repeat(np.array([58, 62], dtype=np.int64), 48),
              'source_point_index': np.tile(np.repeat(np.arange(12, dtype=np.int64), 4), 2),
              'body_channel_qualified': np.ones((96, 2), dtype=np.bool_)}
    return labels


def points_and_targets(lib, *, invalid=False):
    rows, targets = [], []
    for source in lib.mod.SOURCES:
        for seed in lib.mod.SEEDS:
            for tick in lib.mod.TICKS:
                value = 0. if invalid and len(rows) == 23 else -.004
                eps = [{'epsilon': e, 'central': value, 'forward': value, 'backward': value} for e in lib.mod.EPSILONS]
                qualification = lib.mod.scalar_qualification(eps)
                rows.append({'source': source, 'seed': seed, 'tick': tick, 'status': 'COMPLETE',
                             'eps': eps, 'qualification': qualification, 'all_four_wheels_nonzero': True,
                             'qualified': qualification['stable']})
                targets.append(lib.np.full(4, value, dtype=lib.np.float64))
    return rows, targets


def fixture_point(lib, tmp_path, *, slope=-.004):
    np, mod = lib.np, lib.mod
    labels = original_labels(lib)
    states = np.zeros((10, 45), dtype=np.float64)
    states[:, 3] = 1.
    states[:, 25] = 1.
    actions = [labels['a0'][0], labels['a0'][0]]
    for ep, epsilon in enumerate(mod.EPSILONS):
        states[2 + 2 * ep, 25] += epsilon * slope
        states[3 + 2 * ep, 25] -= epsilon * slope
        states[2 + 2 * ep, 28] = epsilon * -.2
        states[3 + 2 * ep, 28] = -epsilon * -.2
        actions.extend([labels['a_plus'][ep], labels['a_minus'][ep]])
    branches = []
    deltas = [0., 0., *(sign * e for e in mod.EPSILONS for sign in (1, -1))]
    for order, action in enumerate(actions):
        np.savez(tmp_path / f'branch_00_{order:02d}.npz', s=labels['s'][0:1],
                 target=states[order:order + 1], applied_action=action[None], issued_action=action[None])
        row = {'status': 'COMPLETE', 'delta': deltas[order], 'issued_equals_applied': True}
        if order < 2:
            row['zero_exact'] = True
        else:
            row['actual_action_delta'] = mod.action_ulps(actions[0], action)
        branches.append(row)
    raw = {key: [] for key in ('central_state45_fd', 'forward_state45_fd', 'backward_state45_fd')}
    for ep, e in enumerate(mod.EPSILONS):
        p, m = states[2 + 2 * ep:4 + 2 * ep]
        for key, value in zip(raw, ((p - m) / (2 * e), (p - states[0]) / e, (states[0] - m) / e), strict=True):
            raw[key].append(value)
    raw = {key: np.asarray(value) for key, value in raw.items()}
    labels['target_yaw'][:4] = raw['central_state45_fd'][:, 28]
    np.savez(tmp_path / 'response_00.npz', **raw)
    old = {'status': 'COMPLETE', 'label_admitted': True, 'branches': branches,
           'all_four_wheels_nonzero': True, 'yaw28': [
               {'epsilon': e, **{name: float(raw[key][ep, 28]) for name, key in zip(
                   ('central', 'forward', 'backward'), raw, strict=True)}} for ep, e in enumerate(mod.EPSILONS)]}
    row = {'source': '58', 'seed': 53103, 'tick': 110, 'status': 'INCOMPLETE',
           'eps': [], 'branches_read': 0, 'raw_fd_scalar_checks': 0, 'action_checks': []}
    return labels, states, raw, old, row


def test_plan_is_independent_of_unfinished_training_runner():
    paths = labels75.required_inputs()
    assert 'tools75/labels75.py' in paths and 'tests/test_labels75.py' in paths
    assert not any(name.startswith('tools75/') and name != 'tools75/labels75.py' for name in paths)
    assert sum('/branch_' in name for name in paths) == 240
    assert sum('/response_' in name for name in paths) == 24
    assert not any('yaw_weight72' in name or 'full_response74/diagnostic' in name for name in paths)
    spec = labels75.plan_spec()
    assert (spec['coordinate'], spec['frame'], spec['cpu_seconds'], spec['wall_seconds']) == (25, 'world', 30, 60)


def test_verify_rejects_extra_training_dependency_before_reading_inputs():
    plan = {**labels75.plan_spec(), 'source_sha256': dict.fromkeys(labels75.required_inputs(), '0' * 64)}
    plan['source_sha256']['tools75/run75.py'] = '0' * 64
    with pytest.raises(ValueError, match='exactly its independent inputs'):
        labels75.verify(plan)


def test_original_fields_and_raw_precision_survive_assembly(lib):
    original = original_labels(lib)
    rows, targets = points_and_targets(lib)
    saved = {k: (v.dtype, v.shape, v.tobytes()) for k, v in original.items()}
    output = lib.mod.assemble_labels(original, rows, targets)
    assert list(output) == [*original, 'target_world_vz', 'world_vz_channel_qualified']
    for key, value in original.items():
        assert (output[key].dtype, output[key].shape, output[key].tobytes()) == saved[key]
        assert value.tobytes() == saved[key][2]
        assert output[key] is not value
    assert output['target_world_vz'].dtype == lib.np.float64 and output['target_world_vz'].shape == (96,)
    assert (output['target_world_vz'] < 0).all()
    assert output['world_vz_channel_qualified'].dtype == lib.np.bool_ and output['world_vz_channel_qualified'].all()


@pytest.mark.parametrize('invalid', ['row_order', 'dtype', 'field_order', 'missing_state'])
def test_original_schema_rejects_silent_changes(lib, invalid):
    labels = original_labels(lib)
    if invalid == 'row_order':
        labels['tick'][0] = 260
    elif invalid == 'dtype':
        labels['target_body'] = labels['target_body'].astype(lib.np.float32)
    elif invalid == 'field_order':
        labels = dict(reversed(list(labels.items())))
    else:
        labels['s'][-4:] = labels['s'][-8:-4]
    with pytest.raises(ValueError):
        lib.mod.validate_original(labels)


def test_all_unknown_flags_survive_but_rejected_labels_cannot_be_published(lib):
    rows, targets = points_and_targets(lib, invalid=True)
    result = lib.mod.admission(rows)
    assert not result['admitted'] and result['qualified_states'] == 23 and result['rows_dropped'] == 0
    flags = rows[-1]['qualification']['failure_reasons']
    assert 'CENTRAL_MAGNITUDE_BELOW_1E_5' in flags and 'CENTRAL_SPREAD_UNDEFINED' in flags
    assert 'ONE_SIDED_GAP_UNDEFINED' in flags
    assert len(rows) == 24 and sum(len(p['eps']) for p in rows) == 96
    with pytest.raises(ValueError, match='all24 world-vz states'):
        lib.mod.assemble_labels(original_labels(lib), rows, targets)


def test_missing_partial_or_wheel_unresolved_state_cannot_be_admitted(lib):
    rows, _ = points_and_targets(lib)
    with pytest.raises(ValueError, match='all24 complete'):
        lib.mod.admission(rows[:-1])
    rows[3]['status'] = 'INCOMPLETE'
    with pytest.raises(ValueError, match='all24 complete'):
        lib.mod.admission(rows)
    rows[3].update(status='COMPLETE', qualified=False, all_four_wheels_nonzero=False)
    assert not lib.mod.admission(rows)['admitted']


def test_raw_world25_retains_sub_float32_endpoint_difference_and_wheel_ulps(lib, tmp_path):
    labels, states, raw, old, row = fixture_point(lib, tmp_path)
    assert lib.np.float32(states[2, 25]) == lib.np.float32(states[3, 25])
    output = lib.mod.read_point(tmp_path, '58', 0, labels, old, row)
    lib.np.testing.assert_array_equal(output, raw['central_state45_fd'][:, 25])
    assert row['status'] == 'COMPLETE' and row['qualified'] and row['raw_fd_scalar_checks'] == 540
    assert len(row['eps']) == len(row['action_checks']) == 4 and row['branches_read'] == 10
    assert all(side['wheel_nonzero_count'] == 4 and side['min_nonzero_delta_ulps'] > 0
               for p in row['action_checks'] for side in (p['plus'], p['minus']))


@pytest.mark.parametrize('corruption', ['cached_fd', 'v32', 'ulp', 'wrong_action', 'zero', 'prestate'])
def test_read_point_rejects_raw_or_provenance_mismatch(lib, tmp_path, corruption):
    np, mod = lib.np, lib.mod
    labels, _, raw, old, row = fixture_point(lib, tmp_path)
    if corruption == 'cached_fd':
        raw['central_state45_fd'][0, 25] += .01
        np.savez(tmp_path / 'response_00.npz', **raw)
    elif corruption == 'v32':
        labels['v'][0, 3] += 1.
    elif corruption == 'ulp':
        old['branches'][2]['actual_action_delta']['delta_ulps'][3] += 1.
    else:
        order = 1 if corruption == 'zero' else 2
        path = tmp_path / f'branch_00_{order:02d}.npz'
        branch = mod.read_npz(path)
        if corruption == 'zero':
            branch['target'][0, 25] += .01
        elif corruption == 'prestate':
            branch['s'][0, 0] += .01
        else:
            branch['applied_action'][0, 3] = branch['issued_action'][0, 3] = .125
        np.savez(path, **branch)
    with pytest.raises(ValueError):
        mod.read_point(tmp_path, '58', 0, labels, old, row)
    assert row['status'] == 'INCOMPLETE'


def test_partial_current_point_is_preserved_by_runtime(lib, tmp_path, monkeypatch):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    labels, _, _, old, row = fixture_point(lib, inputs)
    reader = lib.mod.read_npz

    def stop_after_fifth(path):
        if Path(path).name == 'branch_00_05.npz':
            raise SystemExit('synthetic catchable stop')
        return reader(path)

    monkeypatch.setattr(lib.mod, 'read_npz', stop_after_fifth)

    def body(report):
        report['points'].append(row)
        lib.mod.read_point(inputs, '58', 0, labels, old, row)

    output = tmp_path / 'failed'
    report = lib.mod.run_preserving(output, body, lambda: None)
    assert report['status'] == 'INCOMPLETE' and report['verification_after'] == 'COMPLETE'
    assert report['points'][0]['branches_read'] == 5 and 'SystemExit' in report['failure']
    assert not (output / lib.mod.LABEL_FILE).exists()


def test_qualification_reuses_frozen74_function():
    from tools74.geometry74 import scalar_qualification
    assert labels75.scalar_qualification is scalar_qualification


def test_admission_does_not_mutate_source_rows(lib):
    rows, _ = points_and_targets(lib)
    before = copy.deepcopy(rows)
    assert lib.mod.admission(rows)['admitted']
    assert rows == before


@pytest.mark.parametrize('case', ['complete', 'rejected', 'late_source_change'])
def test_cli_publishes_only_after_complete_admission_and_final_verification(lib, tmp_path, monkeypatch, case):
    mod = lib.mod
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(mod.plan_spec()))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['labels75', '--plan', str(plan), '--output', str(output)])
    monkeypatch.setattr(mod.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(mod.os, 'getpriority', lambda *args: 10)
    monkeypatch.setattr(mod.os, 'sched_getaffinity', lambda *args: {0})
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(mod.importlib.metadata, 'version', lambda name: '2.2.6')
    verifications = []

    def verify(*args):
        verifications.append(True)
        if case == 'late_source_change' and len(verifications) == 2:
            raise ValueError('synthetic late source mutation')

    monkeypatch.setattr(mod, 'verify', verify)

    def calculate(report, **kwargs):
        points, targets = points_and_targets(lib, invalid=case == 'rejected')
        report.update(points=points, completed_points=24, label_rows=96,
                      admission={'status': 'PENDING_FINAL_VERIFICATION', 'admitted': False})
        return original_labels(lib), targets

    monkeypatch.setattr(mod, 'calculate', calculate)
    assert mod.main() == (0 if case == 'complete' else 1)
    report = json.loads((output / 'report.json').read_text())
    assert len(verifications) == 2 and len(report['points']) == 24
    assert report['admission']['admitted'] is (case == 'complete')
    assert (output / mod.LABEL_FILE).exists() is (case == 'complete')
    if case == 'late_source_change':
        assert report['status'] == report['verification_after'] == 'INCOMPLETE'
        assert 'late source mutation' in report['verification_failure']
    else:
        assert report['status'] == report['verification_after'] == 'COMPLETE'
        assert report['admission']['qualified_states'] == (23 if case == 'rejected' else 24)


def test_assembly_rejects_target_detached_from_qualification_evidence(lib):
    rows, targets = points_and_targets(lib)
    targets[0] = targets[0] + .001
    with pytest.raises(ValueError, match='differs from admitted response'):
        lib.mod.assemble_labels(original_labels(lib), rows, targets)
