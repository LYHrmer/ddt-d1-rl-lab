"""Cleanup receipt regressions using disposable /tmp fixtures only.

The archived 17-case validation is kept runnable in ordinary pytest/CI. The
production module is imported from this checkout; every operational path is
replaced before main() runs. Git and disk-space queries are mocked, and real
removals are confined to the current fixture's two disposable directories.
No wheel-legged-control-lab checkout, release archive, or simulator is needed.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools04/cleanup_wheel_legged_local.py'
spec = importlib.util.spec_from_file_location('cleanup_review_target', SOURCE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
REAL_RMTREE = shutil.rmtree
REAL_SHA = mod.sha256


class CleanupReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cleanup-fixture-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(self.temp.name)
        self.repo = self.base/'repo'
        self.repo.mkdir()
        self.work = self.base/'work'
        self.work.mkdir()
        self.receipts = self.base/'ddt/evidence/career_contract_04'
        self.receipts.mkdir(parents=True)
        self.output = self.receipts/'receipt.json'
        self.frozen = self.repo/'src/frozen.py'
        self.frozen.parent.mkdir()
        self.frozen.write_text('protected source\n')
        self.drive = self.repo/'results/my_course_drive_01'
        self.drive.mkdir(parents=True)
        self.user = self.drive/'result.json'
        self.user.write_text('protected user data\n')
        self.targets = [self.repo/'results/d1_budget_study_interrupted_20260911',
                        self.repo/'results/d1_budget_study_interrupted_20260912']
        for target in self.targets:
            target.mkdir()
            (target/'disposable.txt').write_text('interrupted\n')
        (self.repo/'.gitignore').write_text('/results/d1_budget_study_interrupted_*/\n')
        guard = self.base/'guard.json'
        guard.write_text(json.dumps({'guard': {
            'frozen_77_sha256': {'src/frozen.py': REAL_SHA(self.frozen)},
            'my_course_drive_01': {'result.json': {'sha256': REAL_SHA(self.user)}},
        }}))
        self.stack.enter_context(patch.multiple(
            mod, REPO=self.repo, WORK=self.work, RELEASE=self.work/'release',
            GUARD=guard, RECEIPT_ROOT=self.receipts,
            ORIGINAL=self.work/'release/original.tar.gz', GROUP_A=[],
            GROUP_B=self.targets, NEVER_TOUCH=[self.drive, self.repo/'.git']))
        self.git = self.stack.enter_context(patch.object(mod.subprocess, 'run',
                                                 side_effect=self.git_ok))
        self.stack.enter_context(patch.object(mod.shutil, 'disk_usage', return_value=
                                       types.SimpleNamespace(free=1000)))
        self.stack.enter_context(patch.object(mod.shutil, 'rmtree',
                                             side_effect=self.remove_fixture_target))
        # Fail before invocation if a new or changed constant escapes the fixture.
        for name in ('REPO', 'WORK', 'RELEASE', 'GUARD', 'RECEIPT_ROOT', 'ORIGINAL'):
            self.assertTrue(getattr(mod, name).resolve().is_relative_to(self.base))
        for paths in (mod.GROUP_A, mod.GROUP_B, mod.NEVER_TOUCH):
            self.assertTrue(all(p.resolve().is_relative_to(self.base) for p in paths))

    def remove_fixture_target(self, path):
        path = Path(path)
        self.assertIn(path, self.targets)
        self.assertTrue(path.resolve().is_relative_to(self.base))
        REAL_RMTREE(path)

    def git_ok(self, command, **kwargs):
        self.assertEqual(kwargs['cwd'], self.repo)
        self.assertTrue(kwargs['check'])
        self.assertEqual(command[0], 'git')
        self.assertIn(command[1], ('ls-files', 'status'))
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    def run_tool(self, *extra, execute=True, output=None):
        argv = ['cleanup', '--groups', 'B', '--receipt', str(output or self.output)]
        if execute:
            argv.append('--execute')
        argv.extend(extra)
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return mod.main()

    def receipt(self):
        return json.loads(self.output.read_text())

    def test_b_only_executes_without_original_archive(self):
        self.assertEqual(self.run_tool(), 0)
        receipt = self.receipt()
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['mode'], 'EXECUTED')
        self.assertEqual(len(receipt['removed_paths']), 2)
        self.assertNotIn('original_archive_still_identical', receipt['post_checks'])
        self.assertTrue(all(receipt['post_checks'].values()))
        self.assertFalse(mod.ORIGINAL.exists())
        self.assertTrue(all(not t.exists() for t in self.targets))
        self.assertEqual(self.user.read_text(), 'protected user data\n')
        self.assertEqual(self.frozen.read_text(), 'protected source\n')

    def test_dry_run_only_writes_new_receipt(self):
        self.assertEqual(self.run_tool(execute=False), 0)
        self.assertEqual(self.receipt()['mode'], 'DRY_RUN')
        self.assertEqual(self.receipt()['removed_paths'], [])
        self.assertTrue(all(t.exists() for t in self.targets))

    def test_git_ls_files_failure_blocks_deletion(self):
        self.git.side_effect = subprocess.CalledProcessError(128, ['git', 'ls-files'])
        self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.receipt()['status'], 'FAIL')
        self.assertEqual(self.receipt()['mode'], 'BLOCKED')
        self.assertTrue(all(t.exists() for t in self.targets))

    def test_missing_git_blocks_deletion(self):
        self.git.side_effect = FileNotFoundError('git unavailable')
        self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.receipt()['removed_paths'], [])
        self.assertTrue(all(t.exists() for t in self.targets))

    def test_git_status_failure_is_fail_receipt(self):
        def git(command, **kwargs):
            if command[1] == 'status':
                raise subprocess.CalledProcessError(128, command)
            return self.git_ok(command, **kwargs)
        self.git.side_effect = git
        self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.receipt()['status'], 'FAIL')
        self.assertFalse(self.receipt()['post_checks']['repo_tracked_worktree_clean'])

    def test_dirty_git_status_is_fail_receipt(self):
        def git(command, **kwargs):
            result = self.git_ok(command, **kwargs)
            if command[1] == 'status':
                result.stdout = ' M tracked.py\n'
            return result
        self.git.side_effect = git
        self.assertEqual(self.run_tool(), 1)
        self.assertIn('repo_tracked_worktree_clean', self.receipt()['failed_guards'])

    def test_frozen_postcheck_failure_is_fail_receipt(self):
        def remove(path):
            self.remove_fixture_target(path)
            self.frozen.write_text('changed only in temporary fixture\n')
        with patch.object(mod.shutil, 'rmtree', side_effect=remove):
            self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.receipt()['status'], 'FAIL')
        self.assertIn('frozen_77_intact_after', self.receipt()['failed_guards'])

    def test_user_postcheck_failure_is_fail_receipt(self):
        def remove(path):
            self.remove_fixture_target(path)
            self.user.write_text('changed only in temporary fixture\n')
        with patch.object(mod.shutil, 'rmtree', side_effect=remove):
            self.assertEqual(self.run_tool(), 1)
        self.assertIn('user_result_drive_intact_after', self.receipt()['failed_guards'])

    def test_unreadable_frozen_postcheck_is_fail_receipt(self):
        def sha(path, *args):
            if Path(path) == self.frozen and not self.targets[0].exists():
                raise PermissionError('fixture read denied')
            return REAL_SHA(path, *args)
        with patch.object(mod, 'sha256', side_effect=sha):
            self.assertEqual(self.run_tool(), 1)
        self.assertIn('frozen_77_intact_after', self.receipt()['failed_guards'])

    def test_removal_failure_is_recorded(self):
        with patch.object(mod.shutil, 'rmtree', side_effect=PermissionError('denied')):
            self.assertEqual(self.run_tool(), 1)
        self.assertEqual(self.receipt()['status'], 'FAIL')
        self.assertEqual(self.receipt()['mode'], 'EXECUTED')
        self.assertEqual(self.receipt()['removed_paths'], [])
        self.assertTrue(all(t.exists() for t in self.targets))

    def assert_rejected(self, output):
        with self.assertRaises(SystemExit) as stopped:
            self.run_tool(output=output)
        self.assertEqual(stopped.exception.code, 2)
        self.assertTrue(all(t.exists() for t in self.targets))
        self.git.assert_not_called()

    def test_existing_receipt_is_never_overwritten(self):
        self.output.write_text('previous receipt')
        self.assert_rejected(self.output)
        self.assertEqual(self.output.read_text(), 'previous receipt')

    def test_outside_path_is_rejected(self):
        self.assert_rejected(self.user)
        self.assertEqual(self.user.read_text(), 'protected user data\n')

    def test_parent_symlink_escape_is_rejected(self):
        link = self.receipts/'escape'
        link.symlink_to(self.drive, target_is_directory=True)
        self.assert_rejected(link/'new_receipt.json')
        self.assertFalse((self.drive/'new_receipt.json').exists())

    def test_existing_receipt_symlink_is_rejected(self):
        self.output.symlink_to(self.user)
        self.assert_rejected(self.output)
        self.assertEqual(self.user.read_text(), 'protected user data\n')

    def test_dangling_receipt_symlink_is_rejected(self):
        self.output.symlink_to(self.receipts/'not-created.json')
        self.assert_rejected(self.output)
        self.assertFalse((self.receipts/'not-created.json').exists())

    def test_receipt_directory_is_rejected(self):
        self.assert_rejected(self.receipts)

    def test_exclusive_creation_collision_prevents_deletion(self):
        def git(command, **kwargs):
            self.output.write_text('racing writer')
            return self.git_ok(command, **kwargs)
        self.git.side_effect = git
        with self.assertRaises(FileExistsError):
            self.run_tool()
        self.assertEqual(self.output.read_text(), 'racing writer')
        self.assertTrue(all(t.exists() for t in self.targets))


if __name__ == '__main__':
    unittest.main(verbosity=2)
