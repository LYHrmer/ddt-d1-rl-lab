"""Prepared/frozen boundaries and memory samples on temporary files, zero physics."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools16'))
import contract16 as contract  # noqa: E402


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path/'repo'
    root.mkdir()
    for name in contract.SOURCE_FILES:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('synthetic source '+name)
    old = {}
    for index in range(72):
        name = f'old/input_{index}.txt'
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'closed input {index}')
        old[name] = contract.sha(target)
    old_release = root/contract.OLD_RELEASE
    old_release.parent.mkdir(parents=True, exist_ok=True)
    old_release.write_text(json.dumps({'frozen_inputs': old}))
    monkeypatch.setattr(contract, 'OLD_RELEASE_SHA', contract.sha(old_release))
    monkeypatch.setattr(contract, 'source_revision', lambda root: 'f'*40)
    monkeypatch.setattr(contract, 'validate_revision', lambda release, root: None)
    path = root/'evidence/local_comparison16/prepared.json'
    plan = contract.prepare(path, root=root)
    return root, path, plan


def approval_bundle(prepared):
    root, path, plan = prepared
    auth = contract.authorization(plan, contract.sha(path))
    # This approval is only a fake fixture in tmp_path, never the real project.
    auth.update(approved=True, statement='Synthetic fixture approval only', owner_message_reference='test-fixture')
    auth_path = path.with_name('authorization.json')
    contract.write_exclusive(auth_path, auth)
    static_path = path.with_name('static.json')
    contract.check_prepared(path, static_path, root=root)
    return auth_path, static_path


@pytest.mark.parametrize('envs', [2, 4])
def test_plan_counts_include_both_preflights_training_and_evaluation(envs):
    plan = contract.canonical_plan(envs)
    assert contract.budget(plan) == {'policy_transitions': 2166288, 'native_steps': 21662880,
                                    'pd_updates': 10831440}
    assert 4096//envs in (1024, 2048)
    assert plan['arms'] == {'control': {'aux_coef': 0.}, 'velocity': {'aux_coef': 1.}}


def test_prepare_is_non_executable_and_templates_are_not_approval(prepared):
    root, path, plan = prepared
    assert plan['status'] == contract.PREPARED
    assert contract.load_release(path, require_authorized=False, root=root) == plan
    assert contract.authorization(plan, contract.sha(path))['approved'] is False
    with pytest.raises(ValueError, match='not authorized'):
        contract.load_release(path, root=root)
    assert not (root/plan['output_root']).exists()


def test_authorized_release_requires_unchanged_approval_and_prepared_identity(prepared):
    root, path, plan = prepared
    auth_path, static_path = approval_bundle(prepared)
    output = path.with_name('release.json')
    sealed = contract.freeze(path, auth_path, static_path, output, root=root)
    assert sealed['status'] == contract.AUTHORIZED
    assert contract.load_release(output, root=root) == sealed
    assert not (root/plan['output_root']).exists()
    auth = contract.read_json(auth_path)
    auth['statement'] += ' changed'
    contract.atomic_json(auth_path, auth)
    with pytest.raises(ValueError, match='provenance changed'):
        contract.load_release(output, root=root)


def test_refuse_unsigned_draft_approval(prepared):
    root, path, plan = prepared
    auth_path, static_path = approval_bundle(prepared)
    auth = contract.authorization(plan, contract.sha(path))
    contract.atomic_json(auth_path, auth)
    with pytest.raises(ValueError, match='owner approval'):
        contract.freeze(path, auth_path, static_path, path.with_name('no.json'), root=root)
    assert not path.with_name('no.json').exists()


@pytest.mark.parametrize('fault', ['budget', 'envs', 'prepared_release_sha256', 'retry_or_resume_allowed',
                                 'owner_message_reference', 'statement'])
def test_approval_cannot_change_budget_selection_or_source(fault, prepared):
    root, path, _ = prepared
    auth_path, static_path = approval_bundle(prepared)
    auth = contract.read_json(auth_path)
    auth[fault] = '' if fault in ('statement', 'owner_message_reference') else 'different'
    contract.atomic_json(auth_path, auth)
    with pytest.raises(ValueError):
        contract.freeze(path, auth_path, static_path, path.with_name('no.json'), root=root)


@pytest.mark.parametrize('fault', ['seed', 'resources', 'stages', 'comparison', 'stage_order', 'budget'])
def test_fixed_plan_tampering_is_refused(fault, prepared):
    root, path, plan = prepared
    changed = copy.deepcopy(plan)
    changed[fault] = None
    contract.atomic_json(path, changed)
    with pytest.raises(ValueError):
        contract.load_release(path, require_authorized=False, root=root)


@pytest.mark.parametrize('name', ['old/input_0.txt', 'training/d1_rl/velocity_policy15.py',
                                'tools16/run_comparison16.py'])
def test_old_and_new_sources_cannot_change_after_prepare(name, prepared):
    root, path, _ = prepared
    (root/name).write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        contract.load_release(path, require_authorized=False, root=root)


@pytest.mark.parametrize('fault', ['format', 'status', 'prepared_release_sha256', 'sources_sha256',
                                 'physics_steps_executed'])
def test_freeze_refuses_a_static_report_for_other_inputs_or_with_physics(fault, prepared):
    root, path, _ = prepared
    auth_path, static_path = approval_bundle(prepared)
    report = contract.read_json(static_path)
    report[fault] = 'wrong'
    contract.atomic_json(static_path, report)
    with pytest.raises(ValueError, match='static'):
        contract.freeze(path, auth_path, static_path, path.with_name('no.json'), root=root)


def test_outputs_are_exclusive_and_outside_paths_rejected(prepared, tmp_path):
    root, path, _ = prepared
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        contract.prepare(path, root=root)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='prepare under'):
        contract.prepare(tmp_path/'outside.json', root=root)


@pytest.mark.parametrize('name', ['../secret', '/tmp/secret', 'a//b', 'a/./b', 'a/../b', 'a\\b', '', None])
def test_unsafe_paths_rejected_without_reading_them(name, tmp_path):
    with pytest.raises(ValueError, match='unsafe'):
        contract.relative_file(tmp_path, name)


def test_symlink_source_is_refused(tmp_path):
    (tmp_path/'data').write_text('data')
    (tmp_path/'link').symlink_to(tmp_path/'data')
    with pytest.raises(ValueError, match='symlink'):
        contract.relative_file(tmp_path, 'link')


def memory_tree(tmp_path):
    proc, groups = tmp_path/'proc', tmp_path/'cgroup'
    (proc/'self').mkdir(parents=True)
    (groups/'parent/child').mkdir(parents=True)
    (proc/'meminfo').write_text('MemAvailable: 8388608 kB\nSwapFree: 999999999 kB\n')
    (proc/'self/cgroup').write_text('0::/parent/child\n')
    (groups/'memory.max').write_text('max')
    (groups/'parent/memory.max').write_text(str(10*1024**3))
    (groups/'parent/memory.current').write_text(str(7*1024**3))
    (groups/'parent/child/memory.max').write_text(str(8*1024**3))
    (groups/'parent/child/memory.current').write_text(str(2*1024**3))
    return proc, groups


def test_effective_ram_uses_tightest_parent_limit_and_never_swap(tmp_path):
    proc, groups = memory_tree(tmp_path)
    value = contract.read_memory(proc, groups)
    assert value == {'available_ram_bytes': 8*1024**3, 'cgroup_remaining_bytes': 3*1024**3,
                     'effective_available_bytes': 3*1024**3}
    (groups/'parent/memory.current').write_text(str(11*1024**3))
    assert contract.read_memory(proc, groups)['effective_available_bytes'] == 0


def test_uncapped_groups_use_host_memavailable(tmp_path):
    proc, groups = memory_tree(tmp_path)
    for path in groups.rglob('memory.max'):
        path.write_text('max')
    assert contract.read_memory(proc, groups) == {'available_ram_bytes': 8*1024**3,
        'cgroup_remaining_bytes': None, 'effective_available_bytes': 8*1024**3}


@pytest.mark.parametrize('fault', ['v1', 'missing_scope', 'missing_current', 'negative'])
def test_unknown_or_invalid_memory_accounting_fails_closed(fault, tmp_path):
    proc, groups = memory_tree(tmp_path)
    if fault == 'v1': (proc/'self/cgroup').write_text('1:memory:/parent/child\n')
    if fault == 'missing_scope': (proc/'self/cgroup').write_text('0::/gone\n')
    if fault == 'missing_current': (groups/'parent/memory.current').unlink()
    if fault == 'negative': (groups/'parent/memory.current').write_text('-1')
    with pytest.raises((ValueError, FileNotFoundError)):
        contract.read_memory(proc, groups)


def test_shared_module_import_does_not_import_numerical_or_robot_libraries():
    entry = ROOT/'tools16/contract16.py'
    script = ('import importlib.util,sys; '
              f's=importlib.util.spec_from_file_location("contract",{str(entry)!r}); '
              'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
              'assert not {"torch","mujoco","stable_baselines3","onnxruntime"}&sys.modules.keys()')
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_source_revision_requires_committed_clean_execution_sources(tmp_path):
    env = dict(os.environ, GIT_AUTHOR_NAME='test', GIT_AUTHOR_EMAIL='test@example.invalid',
               GIT_COMMITTER_NAME='test', GIT_COMMITTER_EMAIL='test@example.invalid')
    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    git('init', '-q')
    git('commit', '-q', '--allow-empty', '-m', 'synthetic fixture')
    with pytest.raises(ValueError, match='commit preparation sources'):
        contract.source_revision(tmp_path)
    for name in contract.SOURCE_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source')
    git('add', '--', *contract.SOURCE_FILES)
    git('commit', '-q', '-m', 'synthetic execution sources')
    assert contract.source_revision(tmp_path) == git('rev-parse', 'HEAD').stdout.strip()
    release = {'source_git_commit': contract.source_revision(tmp_path),
               'sources_sha256': {name: contract.sha(tmp_path/name) for name in contract.SOURCE_FILES}}
    contract.validate_revision(release, tmp_path)
    bad = copy.deepcopy(release)
    bad['source_git_commit'] = '0'*40
    with pytest.raises(ValueError, match='commit does not contain'):
        contract.validate_revision(bad, tmp_path)
    bad = copy.deepcopy(release)
    bad['sources_sha256'][contract.SOURCE_FILES[0]] = '0'*64
    with pytest.raises(ValueError, match='commit/source identity'):
        contract.validate_revision(bad, tmp_path)
    (tmp_path/contract.SOURCE_FILES[0]).write_text('dirty')
    with pytest.raises(ValueError, match='uncommitted'):
        contract.source_revision(tmp_path)
