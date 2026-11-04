"""Synthetic red/green coverage of inherited absolute source manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools83.source_admission83 import (
    check_file_hashes,
    check_version_metadata,
    read_pinned_json,
    validate_manifest,
)

ABSOLUTE = '/trusted/frozen81/runtime_manifest_v27.json'
A, B, C = (hashlib.sha256(value).hexdigest() for value in (b'a', b'b', b'c'))


def test_red_regression_pinned_inherited_absolute_and_new_relative_are_compatible():
    # Minimal original failure: ROOT / an absolute key preserves the absolute
    # path, as81 allowed;83's later blanket-relative guard rejected it.
    inherited, additions = {ABSOLUTE: A}, {'tools83/new.py': B}
    assert Path('/repository') / ABSOLUTE == Path(ABSOLUTE)
    result = validate_manifest(inherited | additions, inherited, additions)
    assert result['total_entries'] == 2
    assert result['inherited_absolute_count'] == 1
    assert result['inherited_absolute_paths'] == [ABSOLUTE]


def test_relative_only_manifest_remains_compatible():
    result = validate_manifest({'old.py': A, 'new.py': B}, {'old.py': A}, {'new.py': B})
    assert result['inherited_absolute_count'] == 0


@pytest.mark.parametrize('extra', ('extra.py', '/unregistered/extra.py'))
def test_extra_is_not_silently_admitted(extra):
    with pytest.raises(ValueError, match='extra or missing'):
        validate_manifest({'old.py': A, extra: B}, {'old.py': A}, {})


@pytest.mark.parametrize('missing', ('old.py', 'new.py'))
def test_missing_inherited_or_new_binding_rejected(missing):
    manifest = {'old.py': A, 'new.py': B}
    del manifest[missing]
    with pytest.raises(ValueError, match='extra or missing'):
        validate_manifest(manifest, {'old.py': A}, {'new.py': B})


@pytest.mark.parametrize('name', ('old.py', 'new.py', ABSOLUTE))
def test_rebinding_any_registered_source_is_rejected(name):
    inherited, additions = {'old.py': A, ABSOLUTE: A}, {'new.py': B}
    manifest = inherited | additions
    manifest[name] = C
    # Validate the rebinding before any path-mode decision so it cannot hide
    # behind an unrelated inherited absolute name.
    manifest = {name: manifest[name]} | manifest
    with pytest.raises(ValueError, match='rebinding'):
        validate_manifest(manifest, inherited, additions)


@pytest.mark.parametrize('name', ('/new/absolute.py', '../escape.py', 'a/../escape.py',
                                  './new.py', 'a//new.py', '.', '', 'a\\new.py'))
def test_new_names_must_be_canonical_safe_relative(name):
    with pytest.raises(ValueError, match='safe canonical relative'):
        validate_manifest({name: A}, {}, {name: A})


def test_original_plan_hash_is_checked_before_json_or_allowlist_interpretation(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text('not even JSON')
    with pytest.raises(ValueError, match='pinned plan SHA'):
        read_pinned_json(plan, A)


def test_duplicate_json_keys_cannot_replace_an_allowlist_entry(tmp_path):
    raw = b'{"source_sha256": {}, "source_sha256": {"/unexpected": "x"}}'
    plan = tmp_path / 'plan.json'
    plan.write_bytes(raw)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        read_pinned_json(plan, hashlib.sha256(raw).hexdigest())


def test_exact_pinned_bytes_load_without_evaluating_source_files(tmp_path):
    raw = json.dumps({'source_sha256': {ABSOLUTE: A}}).encode()
    plan = tmp_path / 'plan.json'
    plan.write_bytes(raw)
    result = read_pinned_json(plan, hashlib.sha256(raw).hexdigest())
    assert result['source_sha256'] == {ABSOLUTE: A}


def test_file_hash_mismatch_and_new_symlink_escape_rejected(tmp_path):
    root = tmp_path / 'repository'
    root.mkdir()
    (root / 'source.py').write_bytes(b'a')
    with pytest.raises(ValueError, match='hash mismatch'):
        check_file_hashes({'source.py': B}, root, {'source.py'})
    outside = tmp_path / 'outside'
    outside.write_bytes(b'a')
    (root / 'link').symlink_to(outside)
    with pytest.raises(ValueError, match='outside repository'):
        check_file_hashes({'link': A}, root, {'link'})


def test_version_check_uses_only_the_supplied_metadata_reader():
    expected = {'torch': '2.12.1', 'numpy': '2.2.6', 'mujoco': '3.3.0', 'gymnasium': '1.2.0'}
    calls = []

    def metadata_only(name):
        calls.append(name)
        return expected[name]

    assert check_version_metadata(expected, metadata_only) == expected
    assert calls == ['torch', 'numpy', 'mujoco', 'gymnasium']
    with pytest.raises(ValueError, match='metadata mismatch'):
        check_version_metadata(expected, lambda _name: 'wrong')


def test_bad_hash_or_overlapping_addition_rejected():
    with pytest.raises(ValueError, match='SHA256'):
        validate_manifest({'old.py': 'not-sha'}, {'old.py': 'not-sha'}, {})
    with pytest.raises(ValueError, match='overlaps'):
        validate_manifest({'old.py': A}, {'old.py': A}, {'old.py': A})
