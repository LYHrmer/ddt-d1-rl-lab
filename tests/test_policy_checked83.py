"""Cold checked-entry contracts, without Torch, data archives or model calls."""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tools83 import policy_geometry83_checked as checked

ROOT = Path(__file__).resolve().parents[1]
A, B, C = (hashlib.sha256(value).hexdigest() for value in (b'a', b'b', b'c'))


def structures():
    parent = {'source_sha256': {'/fixed/inherited/v27.json': A, 'old.py': B}}
    archived = {'source_sha256': parent['source_sha256'] | {'already_registered.py': C},
                'spec': copy.deepcopy(checked.SPEC)}
    new_hashes = dict.fromkeys(checked.NEW_SOURCE_NAMES, A)
    plan = {'schema': checked.CHECKED_SCHEMA, 'spec': copy.deepcopy(checked.SPEC),
            'source_admission_parent': checked._parent_identity(),
            'source_sha256': archived['source_sha256'] | new_hashes}
    return plan, parent, archived, new_hashes


def test_corrected_structure_retains_pinned_absolute_and_exact_additions():
    plan, parent, archived, new_hashes = structures()
    result = checked.validate_structure(plan, parent, archived, new_hashes)
    assert result['inherited_absolute_count'] == 1
    assert result['registered_new_entries'] == 1 + len(checked.NEW_SOURCE_NAMES)


@pytest.mark.parametrize('extra', ('unexpected.py', '/new/unregistered.json'))
def test_unregistered_manifest_extra_rejected(extra):
    plan, parent, archived, new_hashes = structures()
    plan['source_sha256'][extra] = A
    with pytest.raises(ValueError, match='extra or missing'):
        checked.validate_structure(plan, parent, archived, new_hashes)


@pytest.mark.parametrize('name', ('/fixed/inherited/v27.json', 'already_registered.py',
                                  'tools83/policy_geometry83_checked.py'))
def test_any_old_or_new_rebinding_is_rejected(name):
    plan, parent, archived, new_hashes = structures()
    plan['source_sha256'][name] = B if plan['source_sha256'][name] != B else C
    with pytest.raises(ValueError, match='rebinding'):
        checked.validate_structure(plan, parent, archived, new_hashes)


def test_new_whitelist_and_scientific_spec_cannot_expand():
    plan, parent, archived, new_hashes = structures()
    extra = new_hashes | {'more.py': A}
    with pytest.raises(ValueError, match='whitelist'):
        checked.validate_structure(plan, parent, archived, extra)
    plan['spec']['model_forward_cap'] = 4032
    with pytest.raises(ValueError, match='science spec'):
        checked.validate_structure(plan, parent, archived, new_hashes)


def test_pinned_parent_identity_cannot_be_replaced():
    plan, parent, archived, new_hashes = structures()
    plan['source_admission_parent']['original81_plan_sha256'] = A
    with pytest.raises(ValueError, match='correction identity'):
        checked.validate_structure(plan, parent, archived, new_hashes)


def test_model_math_objectives_and_point_loading_are_exact_frozen_source_copies():
    original = ast.parse((ROOT / 'tools83/policy_geometry83.py').read_text())
    revised = ast.parse((ROOT / 'tools83/policy_geometry83_checked.py').read_text())

    def functions(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                if isinstance(node, ast.FunctionDef)}

    old, new = functions(original), functions(revised)
    for name in ('project_two_halfspaces', 'row_geometry', '_episodes', '_load_points', '_audit_model'):
        assert old[name] == new[name], name
    original_spec = next(node.value for node in original.body if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'SPEC' for t in node.targets))
    revised_spec = next(node.value for node in revised.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'SPEC' for t in node.targets))
    assert ast.dump(original_spec, include_attributes=False) == ast.dump(revised_spec, include_attributes=False)


def npy_header(shape, dtype='<f8'):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': shape}).encode('latin1') + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header


def test_npy_header_metadata_is_read_without_loading_array_payload():
    assert checked._npy_header(io.BytesIO(npy_header((45,)))) == {
        'descr': '<f8', 'fortran_order': False, 'shape': (45,)}
    with pytest.raises(ValueError, match='magic'):
        checked._npy_header(io.BytesIO(b'not-a-npy'))
    with pytest.raises(ValueError, match='truncated'):
        checked._npy_header(io.BytesIO(npy_header((45,))[:-3]))


def test_npz_metadata_refuses_wrong_point_shape_or_extra_member(tmp_path):
    names = {'state': ((45,), '<f8'), 'frame': ((57,), '<f4'), 'history': ((10, 57), '<f4'),
             'filtered': ((3,), '<f8'), 'last': ((16,), '<f4'), 'fifo': ((0, 16), '<f8'),
             'commands': ((33, 3), '<f8')}
    valid = tmp_path / 'valid.npz'
    with zipfile.ZipFile(valid, 'w') as archive:
        for name, (shape, dtype) in names.items():
            archive.writestr('0_' + name + '.npy', npy_header(shape, dtype))
    assert checked._point_headers(valid, 1, False) == 7
    wrong = tmp_path / 'wrong.npz'
    with zipfile.ZipFile(wrong, 'w') as archive:
        for name, (shape, dtype) in names.items():
            archive.writestr('0_' + name + '.npy', npy_header((44,) if name == 'state' else shape, dtype))
    with pytest.raises(ValueError, match='shape/dtype'):
        checked._point_headers(wrong, 1, False)
    with zipfile.ZipFile(valid, 'a') as archive:
        archive.writestr('unexpected.npy', npy_header((1,)))
    with pytest.raises(ValueError, match='member grid'):
        checked._point_headers(valid, 1, False)


def test_validate_only_cli_dispatch_never_calls_formal_model_audit(monkeypatch, tmp_path, capsys):
    calls = []

    def cold(plan, output):
        calls.append((plan, output))
        return {'status': 'COMPLETE', 'verification_after': 'COMPLETE', 'model_calls': 0,
                'actor_calls': 0, 'physics_steps': 0, 'updates': 0}

    def forbidden(*_args, **_kwargs):
        raise AssertionError('formal audit invoked during validate-only')

    monkeypatch.setattr(checked, 'validate_only', cold)
    monkeypatch.setattr(checked, 'audit', forbidden)
    monkeypatch.setattr(checked.resource, 'setrlimit', lambda *_args: None)
    plan, output = tmp_path / 'plan.json', tmp_path / 'admission'
    monkeypatch.setattr(sys, 'argv', ['checked83', '--validate-only', '--plan', str(plan), '--output', str(output)])
    assert checked.main() == 0
    assert calls == [(plan, output)]
    assert json.loads(capsys.readouterr().out)['validate_only'] is True


def test_checked_module_cold_import_has_no_numerical_dependencies():
    program = (
        'import runpy, sys\n'
        f'runpy.run_path({str(ROOT / "tools83/policy_geometry83_checked.py")!r}, run_name="cold83")\n'
        'assert not any(n.split(".")[0] in ("torch","numpy","mujoco") for n in sys.modules)\n'
    )
    subprocess.run([sys.executable, '-I', '-B', '-c', program], check=True, timeout=10,
                   capture_output=True, text=True)
