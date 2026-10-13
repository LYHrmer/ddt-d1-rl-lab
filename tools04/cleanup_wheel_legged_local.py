#!/usr/bin/env python3
"""Reclaim local disk from the wheel-legged-control-lab tree, with proof first.

Two groups, both justified before anything is removed:

  A  The same 4,735,238,370 B archive result_artifacts.tar.gz exists four times:
     the original package, its 36-part split, the 36 parts downloaded back for
     verification, and those parts re-joined. The split and the re-joined copy
     are verified byte-identical to the original before the duplicates go; the
     original and every receipt JSON stay.

  B  results/d1_budget_study_interrupted_2026091{1,2}/ are leftovers of
     interrupted runs. The repository's own .gitignore line 27 declares
     /results/d1_budget_study_interrupted_*/ disposable, and git tracks zero
     files under them. Both facts are rechecked here.

Refuses to touch the 77 protected frozen files or results/my_course_drive_01.
Dry run by default; --execute is required to delete, and it stops at the first
failed guard. Receipts are new files under ddt_ros2_ws/evidence/career_contract_04;
an existing receipt or a path resolving outside that directory is refused.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path('/home/lyh/wheel-legged-control-lab')
WORK = Path('/home/lyh/wheel-legged-control-lab-work')
RELEASE = WORK/'recovery-20260912/release'
GUARD = Path('/home/lyh/ddt_ros2_ws/evidence/career_contract_03/protected_guard_final03_01.json')
RECEIPT_ROOT = Path('/home/lyh/ddt_ros2_ws/evidence/career_contract_04')

ORIGINAL = RELEASE/'build-v080-final-01/package/result_artifacts.tar.gz'
GROUP_A = [RELEASE/'build-v080-final-01/parts',
           RELEASE/'receipts/20260914T015504Z-4905d1e210e6/downloaded',
           RELEASE/'receipts/20260914T015504Z-4905d1e210e6/joined',
           RELEASE/'receipts/20260913T144757Z-cd8aed5549aa/downloaded',
           RELEASE/'receipts/20260913T140822Z-51c4bf8579a8/downloaded']
GROUP_B = [REPO/'results/d1_budget_study_interrupted_20260911',
           REPO/'results/d1_budget_study_interrupted_20260912']
NEVER_TOUCH = [REPO/'results/my_course_drive_01', REPO/'.git', WORK/'gui-20260912']


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_bytes(path):
    total = files = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return total, files


def tracked_count(path):
    out = subprocess.run(['git', 'ls-files', '--', str(path.relative_to(REPO))],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return len([x for x in out.stdout.split('\n') if x.strip()])


def receipt_path(path):
    path = path.expanduser()
    resolved = path.resolve()
    if not resolved.is_relative_to(RECEIPT_ROOT.absolute()) or resolved == RECEIPT_ROOT:
        raise ValueError(f'receipt must resolve inside {RECEIPT_ROOT}')
    if path.is_symlink() or resolved.exists():
        raise ValueError(f'refusing to overwrite an existing receipt: {path}')
    if not resolved.parent.is_dir():
        raise ValueError(f'receipt parent directory does not exist: {resolved.parent}')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--execute', action='store_true',
                        help='actually delete; omit for a dry run')
    parser.add_argument('--groups', default='AB', choices=['A', 'B', 'AB'])
    parser.add_argument('--receipt', type=Path,
                        default=RECEIPT_ROOT/('cleanup_receipt_' + dt.datetime.now(
                            dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json'))
    args = parser.parse_args()
    try:
        args.receipt = receipt_path(args.receipt)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    guards, failed = [], []
    original_sha = None

    def guard(name, ok, detail=None):
        guards.append({'guard': name, 'status': 'PASS' if ok else 'FAIL',
                       'detail': detail or {}})
        if not ok:
            failed.append(name)
        return ok

    # The protected targets must not be inside anything we are about to remove.
    protected = json.loads(GUARD.read_text())['guard']
    frozen77 = list(protected['frozen_77_sha256'])
    guard('no_frozen_77_file_under_results',
          not any(p.startswith('results/') for p in frozen77),
          {'frozen_files': len(frozen77)})
    bad77 = [p for p in frozen77
             if not (REPO/p).is_file() or sha256(REPO/p) != protected['frozen_77_sha256'][p]]
    guard('frozen_77_intact_before', not bad77, {'mismatched': bad77})
    drive = REPO/'results/my_course_drive_01'
    bad_drive = [p for p, meta in protected['my_course_drive_01'].items()
                 if not (drive/p).is_file() or sha256(drive/p) != meta['sha256']]
    guard('user_result_drive_intact_before', not bad_drive, {'mismatched': bad_drive})

    targets = (GROUP_A if 'A' in args.groups else []) + (GROUP_B if 'B' in args.groups else [])
    for keep in NEVER_TOUCH:
        guard(f'never_touch_outside_targets:{keep.name}',
              not any(keep == t or keep.is_relative_to(t) for t in targets),
              {'path': str(keep)})

    plan = {'A': [], 'B': []}
    if 'A' in args.groups:
        guard('original_archive_present', ORIGINAL.is_file(), {'path': str(ORIGINAL)})
        original_size = ORIGINAL.stat().st_size if ORIGINAL.is_file() else 0
        original_sha = sha256(ORIGINAL) if ORIGINAL.is_file() else None
        print(f'original  {original_size} B  sha256 {original_sha}')
        for path in GROUP_A:
            if not path.exists():
                plan['A'].append({'path': str(path), 'state': 'ABSENT'})
                continue
            size, files = tree_bytes(path)
            entry = {'path': str(path), 'bytes': size, 'files': files}
            joined = path/'result_artifacts.tar.gz'
            if joined.is_file():
                entry['verified'] = 'byte_identical_to_original'
                entry['sha256'] = sha256(joined)
                guard(f'joined_copy_identical:{path.parent.name}',
                      entry['sha256'] == original_sha, {'sha256': entry['sha256']})
            else:
                parts = sorted(path.glob('result_artifacts.tar.gz.part*'))
                entry['parts'] = len(parts)
                entry['parts_bytes'] = sum(p.stat().st_size for p in parts)
                # Each part must match the matching slice of the retained original.
                offsets, cursor = {}, 0
                for p in sorted(RELEASE.glob('build-v080-final-01/parts/*.part*')):
                    offsets[p.name] = cursor
                    cursor += p.stat().st_size
                checked = 0
                with open(ORIGINAL, 'rb') as handle:
                    for p in parts:
                        if p.name not in offsets:
                            continue
                        handle.seek(offsets[p.name])
                        if hashlib.sha256(handle.read(p.stat().st_size)).hexdigest() \
                                == sha256(p):
                            checked += 1
                entry['parts_verified_against_original'] = checked
                entry['verified'] = ('all_parts_match_original_slices'
                                     if checked == len(parts) and parts
                                     else f'{checked}/{len(parts)}_matched')
                guard(f'parts_match_original:{path.parent.name}/{path.name}',
                      checked == len(parts) and bool(parts),
                      {'checked': checked, 'parts': len(parts)})
            plan['A'].append(entry)

    if 'B' in args.groups:
        ignore = (REPO/'.gitignore').read_text()
        guard('gitignore_declares_interrupted_disposable',
              '/results/d1_budget_study_interrupted_*/' in ignore)
        for path in GROUP_B:
            if not path.exists():
                plan['B'].append({'path': str(path), 'state': 'ABSENT'})
                continue
            size, files = tree_bytes(path)
            try:
                tracked = tracked_count(path)
            except (OSError, subprocess.CalledProcessError) as error:
                tracked = None
                guard(f'git_ls_files_readable:{path.name}', False,
                      {'error': str(error)})
            guard(f'zero_tracked_files:{path.name}', tracked == 0, {'tracked': tracked})
            plan['B'].append({'path': str(path), 'bytes': size, 'files': files,
                              'git_tracked_files': tracked,
                              'verified': ('gitignored_and_untracked' if tracked == 0
                                           else 'FAILED')})

    planned = sum(e.get('bytes', 0) for g in plan.values() for e in g)
    print(f'\nplanned removal: {planned/2**30:.2f} GiB across '
          f'{sum(len(g) for g in plan.values())} paths')
    for group, entries in plan.items():
        for e in entries:
            print(f"  {group}  {e.get('bytes',0)/2**30:7.2f} GiB  "
                  f"{e.get('verified', e.get('state','?')):36} {e['path']}")
    print('\nguards: ' + ('all passed' if not failed else 'FAILED '+', '.join(failed)))

    before = shutil.disk_usage('/home/lyh')
    removed, freed = [], 0
    # Reserve the receipt before deletion, so output collisions cannot occur
    # after a successful cleanup. Exclusive creation never overwrites a file.
    receipt_handle = args.receipt.open('x')
    execution_started = args.execute and not failed
    if execution_started:
        for group, entries in plan.items():
            for e in entries:
                path = Path(e['path'])
                if not path.exists():
                    continue
                try:
                    shutil.rmtree(path)
                except OSError as error:
                    guard(f'removal_completed:{path}', False, {'error': str(error)})
                    break
                removed.append(e['path'])
                freed += e.get('bytes', 0)
            if failed:
                break
        print(f'\nremoved {len(removed)} paths, {freed/2**30:.2f} GiB')
    elif args.execute:
        print('\nrefusing to delete: a guard failed')

    after = shutil.disk_usage('/home/lyh')
    post = {}
    if execution_started:
        if 'A' in args.groups:
            try:
                post['original_archive_still_identical'] = sha256(ORIGINAL) == original_sha
            except OSError:
                post['original_archive_still_identical'] = False
        try:
            post['frozen_77_intact_after'] = not [
                p for p in frozen77 if not (REPO/p).is_file()
                or sha256(REPO/p) != protected['frozen_77_sha256'][p]]
        except OSError:
            post['frozen_77_intact_after'] = False
        try:
            post['user_result_drive_intact_after'] = not [
                p for p, meta in protected['my_course_drive_01'].items()
                if not (drive/p).is_file() or sha256(drive/p) != meta['sha256']]
        except OSError:
            post['user_result_drive_intact_after'] = False
        try:
            post['repo_tracked_worktree_clean'] = subprocess.run(
                ['git', 'status', '--porcelain', '--untracked-files=no'],
                cwd=REPO, capture_output=True, text=True, check=True).stdout.strip() == ''
        except (OSError, subprocess.CalledProcessError) as error:
            post['repo_tracked_worktree_clean'] = False
            guard('git_status_readable_after', False, {'error': str(error)})
        for name, ok in post.items():
            guard(name, ok)
        print('post-checks: ' + json.dumps(post))

    receipt = {'format': 'wheel-legged-local-cleanup-v1',
               'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'mode': ('EXECUTED' if execution_started else
                        'BLOCKED' if args.execute else 'DRY_RUN'),
               'status': 'FAIL' if failed else 'PASS',
               'groups_requested': args.groups,
               'retained_original': {'path': str(ORIGINAL),
                                     'sha256': original_sha if 'A' in args.groups else None},
               'never_touched': [str(p) for p in NEVER_TOUCH],
               'plan': plan, 'planned_bytes': planned,
               'removed_paths': removed, 'freed_bytes': freed,
               'guards': guards, 'failed_guards': failed,
               'post_checks': post,
               'disk_free_before_bytes': before.free,
               'disk_free_after_bytes': after.free,
               'note': ('No remote synchronization check is performed by this tool. '
                        'The guards and post_checks record the local checks performed; '
                        'removed_paths records only completed removals.'),
               'tool_sha256': sha256(__file__)}
    with receipt_handle:
        receipt_handle.write(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(f'receipt: {args.receipt}')
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
