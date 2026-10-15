"""Tests for tools04/check_handoff_integrity.py, on synthetic fixtures only.

Every fixture lives under pytest's tmp_path. The real old repository, the nine
user results and the frozen releases are never opened for writing, and nothing
here trains, builds a robot or integrates physics. What is under test is the
property the tool exists for: a read, structure, hash or check error must
surface as a failure, never as an empty or passing result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools04'))

import check_handoff_integrity as chk  # noqa: E402
from _workstation import needs_workstation_evidence  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True,
                   env={'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
                        'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t',
                        'HOME': str(cwd), 'PATH': '/usr/bin:/bin'})


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, 'init', '-q')
    git(path, 'commit', '-q', '--allow-empty', '-m', 'init')
    return path


# --- relative_path -----------------------------------------------------------

@pytest.mark.parametrize('bad', ['', '/etc/passwd', '../x', 'a/../b', 'a//b', './a',
                                 'a\\b', None, 7])
def test_relative_path_rejects_unsafe_names(tmp_path, bad):
    with pytest.raises(ValueError):
        chk.relative_path(tmp_path, bad)


def test_relative_path_accepts_a_plain_nested_name(tmp_path):
    assert chk.relative_path(tmp_path, 'a/b/c.txt') == tmp_path.resolve()/'a/b/c.txt'


def test_relative_path_rejects_a_symlink_inside_root(tmp_path):
    (tmp_path/'real').mkdir()
    (tmp_path/'link').symlink_to(tmp_path/'real')
    with pytest.raises(ValueError, match='symlink'):
        chk.relative_path(tmp_path, 'link/file.txt')


def test_relative_path_ignores_a_symlinked_ancestor_of_root(tmp_path):
    """A symlinked /home must not make every frozen file look tampered with."""
    real = tmp_path/'real'
    real.mkdir()
    (real/'f.txt').write_text('x')
    alias = tmp_path/'alias'
    alias.symlink_to(real)
    assert chk.relative_path(alias, 'f.txt').name == 'f.txt'


# --- verify_hash_manifest ----------------------------------------------------

def test_manifest_passes_when_every_hash_matches(tmp_path):
    (tmp_path/'a').write_bytes(b'one')
    (tmp_path/'b').write_bytes(b'two')
    out = chk.verify_hash_manifest(
        tmp_path, {'a': digest(b'one'), 'b': digest(b'two')}, 2)
    assert out['status'] == 'PASS' and out['passed'] == out['count'] == 2


def test_manifest_reports_a_changed_file_as_fail_not_as_pass(tmp_path):
    (tmp_path/'a').write_bytes(b'one')
    out = chk.verify_hash_manifest(tmp_path, {'a': digest(b'ONE')})
    assert out['status'] == 'FAIL' and out['files'][0]['status'] == 'FAIL'
    assert out['files'][0]['actual'] == digest(b'one')


def test_manifest_reports_a_missing_file_as_fail(tmp_path):
    out = chk.verify_hash_manifest(tmp_path, {'gone': digest(b'x')})
    assert out['status'] == 'FAIL' and out['files'][0]['actual'] is None


def test_manifest_does_not_count_a_directory_as_a_file(tmp_path):
    (tmp_path/'d').mkdir()
    out = chk.verify_hash_manifest(tmp_path, {'d': digest(b'')})
    assert out['status'] == 'FAIL'


@pytest.mark.parametrize('manifest', [None, [], {}, 'x', {'a': 'short'},
                                      {'a': 'G'*64}, {'a': 5}, {'a': 'A'*64}])
def test_manifest_raises_on_a_malformed_manifest(tmp_path, manifest):
    with pytest.raises(ValueError):
        chk.verify_hash_manifest(tmp_path, manifest)


def test_manifest_raises_when_the_entry_count_is_wrong(tmp_path):
    (tmp_path/'a').write_bytes(b'x')
    with pytest.raises(ValueError, match='expected 2'):
        chk.verify_hash_manifest(tmp_path, {'a': digest(b'x')}, 2)


def test_manifest_refuses_a_path_that_escapes_root(tmp_path):
    with pytest.raises(ValueError):
        chk.verify_hash_manifest(tmp_path, {'../outside': digest(b'x')})


# --- markdown link extraction -----------------------------------------------

def test_prose_only_drops_fenced_and_inline_code():
    source = 'see [real](a.md)\n```\n[fake](b.md)\n```\ninline `[no](c.md)` done\n'
    cleaned = chk.prose_only(source)
    assert 'a.md' in cleaned and 'b.md' not in cleaned and 'c.md' not in cleaned


def test_markdown_targets_finds_inline_reference_and_image_links():
    source = ('[a](one.md) ![img](two.png)\n[ref]: three.md\n'
              '[angle](<four five.md>)\n')
    found = chk.markdown_targets(source)
    assert {'one.md', 'two.png', 'three.md', 'four five.md'} <= set(found)


def test_markdown_targets_keeps_balanced_parentheses():
    assert 'file_(v2).md' in chk.markdown_targets('[x](file_(v2).md)')


def test_markdown_targets_ignores_links_inside_a_tilde_fence():
    assert chk.markdown_targets('~~~\n[x](hidden.md)\n~~~\n') == []


def test_document_links_pass_when_every_local_target_exists(tmp_path):
    repo = make_repo(tmp_path/'repo')
    (repo/'a.md').write_text('[b](b.md) and [web](https://example.com/x)')
    (repo/'b.md').write_text('no links')
    out = chk.check_document_links(repo)
    assert out['status'] == 'PASS' and out['local_checked'] == 1
    assert out['external_targets'] == ['https://example.com/x']


def test_document_links_fail_on_a_broken_local_target(tmp_path):
    repo = make_repo(tmp_path/'repo')
    (repo/'a.md').write_text('[gone](missing.md)')
    out = chk.check_document_links(repo)
    assert out['status'] == 'FAIL' and out['broken'][0]['target'] == 'missing.md'


def test_document_links_do_not_validate_anchors_or_the_network(tmp_path):
    repo = make_repo(tmp_path/'repo')
    (repo/'a.md').write_text('[x](a.md#no-such-heading)')
    out = chk.check_document_links(repo)
    assert out['status'] == 'PASS'
    assert 'not validated' in out['scope']


def test_document_links_raise_outside_a_git_repository(tmp_path):
    """A failed Git call must not read as an empty document list."""
    plain = tmp_path/'plain'
    plain.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        chk.check_document_links(plain)


# --- command -----------------------------------------------------------------

def test_command_raises_on_a_nonzero_exit(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        chk.command(['git', 'rev-parse', 'HEAD'], tmp_path)


def test_command_returns_stdout_on_success(tmp_path):
    repo = make_repo(tmp_path/'repo')
    assert chk.command(['git', 'status', '--porcelain'], repo) == ''


# --- release inputs ----------------------------------------------------------

def _release_fixture(tmp_path, files):
    (tmp_path/'evidence').mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, data in files.items():
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path/name).write_bytes(data)
        manifest[name] = digest(data)
    receipt = tmp_path/'evidence/extra.json'
    receipt.write_text(json.dumps({'inputs_sha256': manifest}))
    return receipt


def test_extra_release_cannot_replace_a_base_receipt():
    base = chk.BASE_RELEASES[0][0]
    with pytest.raises(ValueError, match='duplicate'):
        chk.check_release_inputs(ROOT, [base])


@pytest.mark.parametrize('bad', ['/abs/x.json', 'docs/x.json', 'evidence/x.txt', '../x.json'])
def test_extra_release_must_be_a_relative_evidence_json(bad):
    with pytest.raises(ValueError):
        chk.check_release_inputs(ROOT, [bad])


@needs_workstation_evidence
def test_real_base_releases_still_match_their_pinned_hashes():
    """The three historical receipts and all their inputs are unchanged."""
    out = chk.check_release_inputs(ROOT)
    assert out['status'] == 'PASS'
    counts = [out['receipts'][name]['count'] for name, _, _ in chk.BASE_RELEASES]
    assert counts == [22, 22, 23]


# --- protected files, on a synthetic pair of repositories -------------------

def _protected_fixture(tmp_path, *, tamper=None, extra_user_file=False,
                       track_user_file=False, track_user_file_committed=False,
                       dirty_tree=False, dirty_unfrozen_file=False):
    ddt = tmp_path/'ddt'
    old = make_repo(tmp_path/'old')
    frozen = {f'src/f{i}.py': f'code {i}'.encode() for i in range(3)}
    for name, data in frozen.items():
        (old/name).parent.mkdir(parents=True, exist_ok=True)
        (old/name).write_bytes(data)
    drive = old/'results/my_course_drive_01'
    drive.mkdir(parents=True)
    user = {f'u{i}.json': f'user {i}'.encode() for i in range(2)}
    for name, data in user.items():
        (drive/name).write_bytes(data)
    guard = {'frozen_77_sha256': {k: digest(v) for k, v in frozen.items()},
             'my_course_drive_01': {k: {'sha256': digest(v)} for k, v in user.items()}}
    (ddt/'evidence/career_contract_03').mkdir(parents=True)
    guard_path = ddt/chk.GUARD
    guard_path.write_text(json.dumps({'guard': guard}))
    git(old, 'add', 'src')
    git(old, 'commit', '-qm', 'base')
    if tamper:
        (old/'src/f0.py').write_bytes(b'TAMPERED')
        git(old, 'commit', '-qam', 'tamper')
    if extra_user_file:
        (drive/'unexpected.txt').write_text('x')
    if track_user_file:
        git(old, 'add', '-f', 'results/my_course_drive_01/u0.json')
    if track_user_file_committed:
        # Committed, so `git status` stays clean and only the tracking guard can fire.
        git(old, 'add', '-f', 'results/my_course_drive_01/u0.json')
        git(old, 'commit', '-qm', 'track a user result')
    if dirty_tree:
        (old/'src/f1.py').write_bytes(b'edited but uncommitted')
    if dirty_unfrozen_file:
        # A tracked file that is NOT in the frozen manifest, so the 77-file hash
        # audit still passes and only the old-tree cleanliness guard can fire.
        (old/'README.md').write_text('base')
        git(old, 'add', 'README.md')
        git(old, 'commit', '-qm', 'add unfrozen file')
        (old/'README.md').write_text('edited but uncommitted')
    return ddt, old, digest(guard_path.read_bytes()), len(frozen), len(user)


def _audit(tmp_path, monkeypatch, **kwargs):
    fixture_kwargs = {k: v for k, v in kwargs.items() if k != 'unfrozen'}
    ddt, old, guard_sha, n_frozen, n_user = _protected_fixture(tmp_path, **fixture_kwargs)
    # The production tool pins 77 and 9; the fixture is smaller, so relax only counts.
    real = chk.verify_hash_manifest
    expected = {77: n_frozen, 9: n_user}
    monkeypatch.setattr(chk, 'verify_hash_manifest',
                        lambda root, manifest, count=None: real(root, manifest,
                                                                expected.get(count, count)))
    return chk.check_protected_files(ddt, old, guard_sha256=guard_sha, unfrozen=kwargs.get('unfrozen', {}))


def test_protected_files_pass_on_an_untouched_pair(tmp_path, monkeypatch):
    out = _audit(tmp_path, monkeypatch)
    assert out['status'] == 'PASS'
    assert out['frozen_77']['status'] == 'PASS'
    assert out['user_results_9']['exact_file_set'] is True
    assert out['old_tracked_changes'] == []


def test_protected_files_fail_when_a_frozen_file_is_edited(tmp_path, monkeypatch):
    out = _audit(tmp_path, monkeypatch, tamper=True)
    assert out['status'] == 'FAIL' and out['frozen_77']['status'] == 'FAIL'
    bad = [r['path'] for r in out['frozen_77']['files'] if r['status'] == 'FAIL']
    assert bad == ['src/f0.py']


def test_protected_files_fail_on_an_unexpected_file_among_user_results(tmp_path, monkeypatch):
    out = _audit(tmp_path, monkeypatch, extra_user_file=True)
    assert out['status'] == 'FAIL' and out['user_results_9']['exact_file_set'] is False


def test_protected_files_fail_when_a_user_result_gets_tracked_by_git(tmp_path, monkeypatch):
    out = _audit(tmp_path, monkeypatch, track_user_file=True)
    assert out['status'] == 'FAIL' and out['user_results_9']['tracked']


def test_protected_files_fail_on_an_uncommitted_change_in_the_old_tree(tmp_path, monkeypatch):
    out = _audit(tmp_path, monkeypatch, dirty_tree=True)
    assert out['status'] == 'FAIL' and out['old_tracked_changes']


def test_a_committed_user_result_fails_on_the_tracking_guard_alone(tmp_path, monkeypatch):
    """Isolated: git status is clean, the hashes and the file set are fine."""
    out = _audit(tmp_path, monkeypatch, track_user_file_committed=True)
    assert out['old_tracked_changes'] == []
    assert out['frozen_77']['status'] == 'PASS'
    assert out['user_results_9']['exact_file_set'] is True
    assert out['user_results_9']['tracked'] == ['results/my_course_drive_01/u0.json']
    assert out['status'] == 'FAIL'


def test_an_uncommitted_unfrozen_file_fails_on_the_tree_guard_alone(tmp_path, monkeypatch):
    """Isolated: every frozen hash and the user results are intact."""
    out = _audit(tmp_path, monkeypatch, dirty_unfrozen_file=True)
    assert out['frozen_77']['status'] == 'PASS'
    assert out['user_results_9']['status'] == 'PASS'
    assert out['old_tracked_changes'] and 'README.md' in out['old_tracked_changes'][0]
    assert out['status'] == 'FAIL'


def test_protected_files_refuse_a_guard_receipt_with_the_wrong_pin(tmp_path, monkeypatch):
    ddt, old, _, _, _ = _protected_fixture(tmp_path)
    with pytest.raises(ValueError, match='guard receipt SHA256 changed'):
        chk.check_protected_files(ddt, old, guard_sha256='0'*64)


def test_protected_files_raise_when_the_old_path_is_not_a_repository(tmp_path, monkeypatch):
    ddt, _, guard_sha, _, _ = _protected_fixture(tmp_path)
    with pytest.raises((subprocess.CalledProcessError, ValueError, FileNotFoundError)):
        chk.check_protected_files(ddt, tmp_path/'not_a_repo', guard_sha256=guard_sha)


# --- receipt directory: history is never overwritten ------------------------

def test_receipt_directory_must_be_new_and_below_evidence(tmp_path):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    old.mkdir()
    made = chk.new_receipt_directory(root, old, 'evidence/run_01')
    assert made.is_dir()
    with pytest.raises(FileExistsError):
        chk.new_receipt_directory(root, old, 'evidence/run_01')


@pytest.mark.parametrize('bad', ['elsewhere/run', 'evidence', '../escape', 'evidence/../x'])
def test_receipt_directory_rejects_paths_outside_evidence(tmp_path, bad):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    old.mkdir()
    with pytest.raises(ValueError):
        chk.new_receipt_directory(root, old, bad)


def test_receipt_directory_refuses_a_root_inside_the_old_repository(tmp_path):
    old = tmp_path/'old'
    (old/'evidence').mkdir(parents=True)
    with pytest.raises(ValueError, match='outside the read-only old repository'):
        chk.new_receipt_directory(old, old, 'evidence/x')


def test_receipt_directory_refuses_a_symlinked_evidence_child(tmp_path):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    old.mkdir()
    (root/'evidence/trap').symlink_to(old)
    with pytest.raises(ValueError):
        chk.new_receipt_directory(root, old, 'evidence/trap/run')
    assert list(old.iterdir()) == []


# --- run(): failures are retained, never swallowed --------------------------

def test_run_records_a_failed_output_check_and_still_writes_a_receipt(tmp_path, monkeypatch):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    make_repo(root)
    old.mkdir()
    for name in ('check_release_inputs', 'check_protected_files',
                 'check_document_links', 'check_static'):
        monkeypatch.setattr(chk, name, lambda *a, **k: {'status': 'PASS'})
    report, path = chk.run(root, old, 'not_below_evidence')
    assert report['status'] == 'FAIL'
    assert any(e['check'] == 'output' for e in report['errors'])
    assert path.is_file() and 'handoff_integrity_failures' in path.as_posix()


def test_run_turns_an_exception_in_any_check_into_a_failure(tmp_path, monkeypatch):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    make_repo(root)
    old.mkdir()

    def boom(*a, **k):
        raise OSError('cannot read')

    monkeypatch.setattr(chk, 'check_release_inputs', boom)
    for name in ('check_protected_files', 'check_document_links', 'check_static'):
        monkeypatch.setattr(chk, name, lambda *a, **k: {'status': 'PASS'})
    report, path = chk.run(root, old, 'evidence/run_01')
    assert report['status'] == 'FAIL'
    assert report['errors'][0] == {'check': 'release_inputs',
                                   'error': 'OSError: cannot read'}
    assert json.loads(path.read_text())['status'] == 'FAIL'


def test_run_turns_a_non_pass_status_into_a_failure(tmp_path, monkeypatch):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    make_repo(root)
    old.mkdir()
    monkeypatch.setattr(chk, 'check_document_links', lambda *a, **k: {'status': 'FAIL'})
    for name in ('check_release_inputs', 'check_protected_files', 'check_static'):
        monkeypatch.setattr(chk, name, lambda *a, **k: {'status': 'PASS'})
    report, _ = chk.run(root, old, 'evidence/run_01')
    assert report['status'] == 'FAIL'
    assert [e['check'] for e in report['errors']] == ['document_links']


def test_run_never_overwrites_an_existing_receipt(tmp_path, monkeypatch):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence/run_01').mkdir(parents=True)
    (root/'evidence/run_01/report.json').write_text('history')
    make_repo(root)
    old.mkdir()
    for name in ('check_release_inputs', 'check_protected_files',
                 'check_document_links', 'check_static'):
        monkeypatch.setattr(chk, name, lambda *a, **k: {'status': 'PASS'})
    report, path = chk.run(root, old, 'evidence/run_01')
    assert (root/'evidence/run_01/report.json').read_text() == 'history'
    assert report['status'] == 'FAIL' and 'handoff_integrity_failures' in path.as_posix()


def test_run_reports_zero_physics_and_no_robot_model(tmp_path, monkeypatch):
    root, old = tmp_path/'ddt', tmp_path/'old'
    (root/'evidence').mkdir(parents=True)
    make_repo(root)
    old.mkdir()
    for name in ('check_release_inputs', 'check_protected_files',
                 'check_document_links', 'check_static'):
        monkeypatch.setattr(chk, name, lambda *a, **k: {'status': 'PASS'})
    report, _ = chk.run(root, old, 'evidence/run_01')
    assert report['native_integration_steps_executed'] == 0
    assert report['robot_model_required'] is False


# --- check_static: a green-looking receipt must still be rejected ------------

def _fake_static(monkeypatch, root, **overrides):
    """Make `command` write a static receipt shaped like the real one."""
    report = {'status': 'PASS', 'checks_total': 32, 'failed_checks': [],
              'native_integration_steps_executed': 0, 'robot_model_required': False,
              'checks': [{'status': 'PASS'}]*32,
              'tool_sha256': digest((root/'tools04/check_contract_static_ci.py').read_bytes())}
    report.update(overrides)

    def fake(args, cwd, env=None):
        # check_static already hands the child its own `<output>/static` directory.
        out = Path(args[args.index('--output')+1])
        out.mkdir(parents=True, exist_ok=True)
        (out/'report.json').write_text(json.dumps(report))
        return 'stdout'

    monkeypatch.setattr(chk, 'command', fake)


def _static_root(tmp_path):
    root = tmp_path/'ddt'
    (root/'tools04').mkdir(parents=True)
    (root/'tools04/check_contract_static_ci.py').write_text('# stub\n')
    return root


def test_static_accepts_only_a_fully_passing_zero_physics_receipt(tmp_path, monkeypatch):
    root = _static_root(tmp_path)
    _fake_static(monkeypatch, root)
    out = tmp_path/'out'
    out.mkdir()
    assert chk.check_static(root, out)['status'] == 'PASS'


@pytest.mark.parametrize('override', [
    {'status': 'FAIL'}, {'checks_total': 31}, {'failed_checks': ['x']},
    {'native_integration_steps_executed': 1},
    {'native_integration_steps_executed': True},
    {'native_integration_steps_executed': False},
    {'native_integration_steps_executed': None},
    {'robot_model_required': True},
    {'checks': [{'status': 'PASS'}]*31},
    {'checks': [{'status': 'PASS'}]*31+[{'status': 'FAIL'}]},
    {'tool_sha256': '0'*64},
])
def test_static_rejects_a_receipt_that_only_looks_green(tmp_path, monkeypatch, override):
    root = _static_root(tmp_path)
    _fake_static(monkeypatch, root, **override)
    out = tmp_path/'out'
    out.mkdir()
    assert chk.check_static(root, out)['status'] == 'FAIL'


# --- the real tool never needs the rtk wrapper -------------------------------

def test_tool_does_not_depend_on_the_rtk_wrapper():
    source = (ROOT/'tools04/check_handoff_integrity.py').read_text()
    assert "'rtk'" not in source and '"rtk"' not in source



# --- owner-unfrozen files --------------------------------------------------------
def _sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_apply_unfrozen_substitutes_only_the_named_file():
    manifest = {'a.py': _sha(b'a'), 'b.py': _sha(b'b')}
    out = chk.apply_unfrozen(manifest, {'a.py': {'frozen_sha256': _sha(b'a'), 'current_sha256': _sha(b'A')}})
    assert out == {'a.py': _sha(b'A'), 'b.py': _sha(b'b')} and manifest['a.py'] == _sha(b'a')


def test_an_unfreeze_record_for_a_file_never_frozen_is_refused():
    with pytest.raises(ValueError, match='never frozen'):
        chk.apply_unfrozen({'a.py': _sha(b'a')}, {'new.py': {'frozen_sha256': _sha(b'a'), 'current_sha256': _sha(b'x')}})


def test_an_unfreeze_record_with_the_wrong_frozen_digest_is_refused():
    with pytest.raises(ValueError, match='does not match'):
        chk.apply_unfrozen({'a.py': _sha(b'a')}, {'a.py': {'frozen_sha256': _sha(b'z'), 'current_sha256': _sha(b'A')}})


def test_an_unfreeze_record_with_a_malformed_new_digest_is_refused():
    with pytest.raises(ValueError, match='invalid current'):
        chk.apply_unfrozen({'a.py': _sha(b'a')}, {'a.py': {'frozen_sha256': _sha(b'a'), 'current_sha256': 'xyz'}})


def test_an_unfrozen_file_at_its_new_digest_passes(tmp_path, monkeypatch):
    """The tampered fixture edits src/f0.py; recording exactly that edit makes it pass."""
    ddt, old, *_ = _protected_fixture(tmp_path/'probe', tamper=True)
    frozen = json.loads((ddt/chk.GUARD).read_text())['guard']['frozen_77_sha256']['src/f0.py']
    record = {'src/f0.py': {'frozen_sha256': frozen, 'current_sha256': _sha(b'TAMPERED')}}
    out = _audit(tmp_path/'run', monkeypatch, tamper=True, unfrozen=record)
    assert out['status'] == 'PASS' and out['frozen_77']['unfrozen_by_owner'] == record


def test_an_unfrozen_file_changed_again_after_unfreezing_still_fails(tmp_path, monkeypatch):
    ddt, old, *_ = _protected_fixture(tmp_path/'probe', tamper=True)
    frozen = json.loads((ddt/chk.GUARD).read_text())['guard']['frozen_77_sha256']['src/f0.py']
    record = {'src/f0.py': {'frozen_sha256': frozen, 'current_sha256': _sha(b'SOMETHING ELSE')}}
    out = _audit(tmp_path/'run', monkeypatch, tamper=True, unfrozen=record)
    assert out['status'] == 'FAIL'
    assert [r['path'] for r in out['frozen_77']['files'] if r['status'] == 'FAIL'] == ['src/f0.py']


def test_the_production_unfreeze_record_names_one_file_with_its_recorded_frozen_digest():
    assert list(chk.UNFROZEN) == ['src/wheel_legged_control/experiments.py']
    entry = chk.UNFROZEN['src/wheel_legged_control/experiments.py']
    assert entry['frozen_sha256'] == '5c8a7a273fd3219c549c005a7d6996de66600a4fcf54c409b88c4d3cd49a7ed4'
    assert entry['authorised_by'] == 'repository owner' and entry['date'] == '2026-10-16'
