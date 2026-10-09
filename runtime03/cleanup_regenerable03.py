#!/usr/bin/env python3
"""Prepare/dry-run an exact DDT cache allowlist; deletion requires explicit --execute."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path('/home/lyh/ddt_ros2_ws')
BUILD_AUDIT = ROOT / 'evidence/build_intermediate_audit_20261009_01.json'
CACHE_AUDIT = ROOT / 'evidence/disk_audit_deployment03c_20261009_01.json'
BUILD_ROOTS = (ROOT / 'build', ROOT / 'runtime03/build')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink() or path.resolve() != path:
        raise ValueError(f'Not a regular unredirected path: {path}')
    return {'path': str(path), 'device': st.st_dev, 'inode': st.st_ino,
            'link_count': st.st_nlink, 'size': st.st_size,
            'allocated_bytes': st.st_blocks * 512, 'mtime_ns': st.st_mtime_ns,
            'sha256': digest(path)}


def command(argv):
    result = subprocess.run(['rtk', 'proxy', *argv], text=True, capture_output=True)
    return {'argv': argv, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def git_guard(paths):
    roots = {ROOT}
    for path in paths:
        for parent in path.parents:
            if not parent.is_relative_to(ROOT):
                break
            if (parent / '.git').exists():
                roots.add(parent)
    tracked = set()
    repos = []
    for root in sorted(roots):
        result = command(['git', '-C', str(root), 'ls-files', '-z', '--cached'])
        if result['returncode']:
            raise RuntimeError(result)
        names = result['stdout'].split('\0')
        tracked.update(root / name for name in names if name)
        repos.append({'root': str(root), 'tracked_path_count': sum(bool(name) for name in names)})
    overlap = set(paths) & tracked
    if overlap:
        raise ValueError(f'Tracked candidates forbidden: {sorted(map(str, overlap))}')
    return repos


def symlink_snapshot():
    result = []
    for directory, dirs, files in os.walk(ROOT, followlinks=False):
        if '.git' in dirs:
            dirs.remove('.git')
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink():
                target = path.resolve(strict=False)
                result.append({'path': str(path), 'link_text': os.readlink(path),
                               'resolved_target': str(target),
                               'resolved_target_is_file': target.is_file(),
                               'resolved_target_is_directory': target.is_dir()})
    return sorted(result, key=lambda item: item['path'])


def allowed(record):
    path = Path(record['path'])
    category = record['category']
    if path.resolve() != path or not path.is_relative_to(ROOT):
        raise ValueError(f'Path redirection/outside scope: {path}')
    source = Path(record['source'])
    if not source.is_file():
        raise ValueError(f'Missing reconstruction source: {source}')
    if category in ('elf_relocatable_object', 'object_dependency_file'):
        if not any(path.is_relative_to(root) for root in BUILD_ROOTS):
            raise ValueError(f'Object outside approved build roots: {path}')
        if not path.name.endswith('.o' if category == 'elf_relocatable_object' else '.o.d'):
            raise ValueError(f'Bad object suffix: {path}')
    elif category == 'source_backed_python_bytecode':
        if not path.is_relative_to(ROOT / '.python-deps') or path.parent.name != '__pycache__':
            raise ValueError(f'Bytecode outside dependency cache: {path}')
        name = re.sub(r'\.(?:cpython|pypy)-[^.]+(?:\.opt-\d+)?\.pyc$', '.py', path.name)
        if source != path.parent.parent / name:
            raise ValueError(f'Bytecode source mismatch: {path}')
    else:
        raise ValueError(f'Forbidden category: {category}')


def prepare():
    audit = json.loads(BUILD_AUDIT.read_text())
    cache = json.loads(CACHE_AUDIT.read_text())
    records = []
    protected = set()
    for entry in audit['candidates']:
        if entry['category'] not in ('elf_relocatable_object', 'object_dependency_file'):
            continue
        path = Path(entry['path'])
        current = snapshot(path)
        for key in ('device', 'inode', 'link_count', 'mtime_ns', 'sha256'):
            if current[key] != entry[key]:
                raise ValueError(f'Build audit stale for {path}: {key}')
        records.append({**current, 'category': entry['category'], 'source': entry['source']})
    for entry in cache['reconstructible_source_backed_pyc']:
        path = Path(entry['path'])
        if not path.is_relative_to(ROOT / '.python-deps'):
            continue
        records.append({**snapshot(path), 'category': 'source_backed_python_bytecode', 'source': entry['source']})
    for entry in records:
        allowed(entry)
        if entry['link_count'] != 1:
            raise ValueError(f'Hardlinked candidate: {entry["path"]}')
        protected.add(Path(entry['source']))
    paths = {Path(entry['path']) for entry in records}
    if len(paths) != len(records):
        raise ValueError('Duplicate candidate')
    for entry in audit['linked_targets_preserved'] + audit['archives_preserved']:
        protected.add(Path(entry['path']))
    for root in BUILD_ROOTS:
        for name in ('CMakeCache.txt', 'build.make', 'link.txt', 'Makefile'):
            protected.update(path for path in root.rglob(name) if path.is_file() and not path.is_symlink())
    for relative in ('runtime03/src', 'runtime03/include', 'training/d1_rl', 'training/configs', 'install'):
        for directory, _, files in os.walk(ROOT / relative, followlinks=False):
            for name in files:
                path = Path(directory) / name
                if path.is_file() and not path.is_symlink() and path.suffix != '.pyc':
                    protected.add(path)
    protected.update((ROOT / 'runtime03').glob('schema*.json'))
    links = symlink_snapshot()
    for entry in links:
        target = Path(entry['resolved_target'])
        if target in paths or (entry['resolved_target_is_directory'] and any(p.is_relative_to(target) for p in paths)):
            raise ValueError(f'Symlink exposes candidate: {entry}')
        if entry['resolved_target_is_file'] and target.is_relative_to(ROOT):
            protected.add(target)
    if protected & paths:
        raise ValueError('Protected/candidate collision')
    return {'format': 'ddt-exact-regenerable-allowlist-v1', 'created_utc': now(),
            'mode': 'allowlist_preparation_only_no_deletion', 'workspace': str(ROOT),
            'build_audit_sha256': digest(BUILD_AUDIT), 'cache_audit_sha256': digest(CACHE_AUDIT),
            'generator_sha256': digest(Path(__file__)), 'git_repositories_checked': git_guard(paths),
            'candidate_count': len(records), 'candidate_allocated_bytes': sum(x['allocated_bytes'] for x in records),
            'candidates': records, 'protected_file_sha256': {str(path): digest(path) for path in sorted(protected)},
            'workspace_symlinks': links,
            'rebuild_cost': 'A later CMake/colcon build must recompile 152 translation units and relink targets. Python imports may regenerate bytecode. Wall-time cost has not been measured. Current executables, .so/.a, source, configuration, install targets, models and records remain. Do not rebuild binaries frozen by an active experiment.'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate(manifest):
    if manifest['format'] != 'ddt-exact-regenerable-allowlist-v1' or manifest['workspace'] != str(ROOT):
        raise ValueError('Bad manifest identity')
    paths = []
    for entry in manifest['candidates']:
        allowed(entry)
        path = Path(entry['path'])
        current = snapshot(path)
        if current != {key: entry[key] for key in current}:
            raise ValueError(f'Candidate identity/content changed: {path}')
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate candidates')
    git_guard(paths)
    guards = {path: digest(Path(path)) for path in manifest['protected_file_sha256']}
    if guards != manifest['protected_file_sha256']:
        raise ValueError('Protected source/target changed since allowlist preparation')
    if symlink_snapshot() != manifest['workspace_symlinks']:
        raise ValueError('Workspace symlinks changed since allowlist preparation')
    return paths, guards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--host-idle-confirmed', action='store_true',
                        help='Root operator has independently confirmed host training/build/runtime finished; local process namespace visibility alone is insufficient')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prepare:
        if args.manifest or args.execute:
            parser.error('--prepare cannot combine with --manifest or --execute')
        report = prepare()
    else:
        if args.manifest is None:
            parser.error('--manifest is required for dry-run or execution')
        if args.execute and not args.host_idle_confirmed:
            parser.error('--execute additionally requires --host-idle-confirmed after an independent host process check')
        manifest_sha = digest(args.manifest)
        if args.execute and args.expected_manifest_sha256 != manifest_sha:
            parser.error('--execute requires the exact --expected-manifest-sha256')
        manifest = json.loads(args.manifest.read_text())
        if manifest['generator_sha256'] != digest(Path(__file__)):
            raise ValueError('Cleanup script changed after allowlist preparation; regenerate a new allowlist and dry-run')
        paths, before = validate(manifest)
        processes = command(['ps', '-eo', 'pid=,comm=,args='])
        if processes['returncode']:
            raise RuntimeError(processes)
        relevant = [line for line in processes['stdout'].splitlines()
                    if str(ROOT) in line and any(token in line for token in
                         ('train_d1_ppo.py', 'bounded_training_run.py', 'd1_runtime03 --', 'colcon build'))]
        compilers = command(['ps', '-C', 'cc1,cc1plus,ld,ld.lld,cmake,ninja,make,gmake,c++',
                             '-o', 'pid=,comm=,args='])
        opened = command(['lsof', '-nP', '-Fpn', '--', *map(str, paths)])
        blocked = bool(relevant or compilers['stdout'].strip() or opened['stdout'].strip() or
                       opened['returncode'] not in (0, 1) or compilers['returncode'] not in (0, 1))
        report = {'created_utc': now(), 'mode': 'execute' if args.execute else 'dry_run_no_deletion',
                  'manifest': str(args.manifest), 'manifest_sha256': manifest_sha,
                  'candidate_count': len(paths), 'candidate_allocated_bytes': manifest['candidate_allocated_bytes'],
                  'protected_file_sha256_before': before, 'related_active_processes': relevant,
                  'compiler_process_check': compilers, 'lsof_check': opened,
                  'process_check_limitation': 'Only processes visible in the current PID namespace. Empty ps/lsof results do not prove host idleness; lsof mount warnings are preserved. Execution additionally requires independent host confirmation by the root operator.',
                  'independent_host_idle_confirmed_by_root_operator': args.host_idle_confirmed,
                  'execution_blocked_by_process_snapshot': blocked, 'deleted': [],
                  'rebuild_cost': manifest['rebuild_cost']}
        if args.execute:
            if blocked:
                raise RuntimeError('Active training/build/runtime or file use: deletion refused; rerun dry-run after completion')
            # Reserve receipt and append-only journal before the first unlink.
            # Avoid rewriting a large receipt for every removed cache file.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            journal_path = args.output.with_name(args.output.name + '.deleted.jsonl')
            if journal_path.exists():
                raise FileExistsError(journal_path)
            report['operation_journal'] = str(journal_path)
            with args.output.open('x') as out:
                out.write(json.dumps(report, indent=2) + '\n')
            with journal_path.open('x') as journal:
                for entry in manifest['candidates']:
                    path = Path(entry['path'])
                    current = snapshot(path)
                    if current != {key: entry[key] for key in current}:
                        raise RuntimeError(f'Candidate changed during cleanup: {path}')
                    path.unlink()
                    item = {**entry, 'deleted_utc': now()}
                    report['deleted'].append(item)
                    journal.write(json.dumps(item) + '\n')
                    journal.flush()
            after = {path: digest(Path(path)) for path in before}
            report['protected_file_sha256_after'] = after
            report['protected_files_match'] = after == before
            report['workspace_symlinks_preserved'] = symlink_snapshot() == manifest['workspace_symlinks']
            report['sum_deleted_allocated_bytes'] = sum(item['allocated_bytes'] for item in report['deleted'])
            report['all_candidates_absent'] = all(not path.exists() for path in paths)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            if not report['protected_files_match'] or not report['workspace_symlinks_preserved']:
                raise RuntimeError('Protection mismatch after cleanup; inspect receipt')
            print(json.dumps({key: report[key] for key in ('mode', 'sum_deleted_allocated_bytes', 'protected_files_match', 'workspace_symlinks_preserved')}))
            return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(json.dumps({key: report[key] for key in ('mode', 'candidate_count', 'candidate_allocated_bytes')}))
    if 'execution_blocked_by_process_snapshot' in report:
        print('execution_blocked_by_process_snapshot:', report['execution_blocked_by_process_snapshot'])


if __name__ == '__main__':
    main()
