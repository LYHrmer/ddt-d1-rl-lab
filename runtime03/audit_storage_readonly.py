#!/usr/bin/env python3
"""Inventory physical allocation and source-backed bytecode; delete nothing."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re

ROOT = Path('/home/lyh/ddt_ros2_ws')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    directories = {}
    files = []
    caches = []
    partial_downloads = []
    seen = set()
    total_allocated = total_logical = 0
    ignored_symlinks = 0
    for directory, child_dirs, names in os.walk(ROOT, followlinks=False):
        d = Path(directory)
        ignored_symlinks += sum((d/name).is_symlink() for name in child_dirs)
        for name in names:
            p = d/name
            try:
                s = p.lstat()
            except FileNotFoundError:
                continue
            if p.is_symlink():
                ignored_symlinks += 1
                continue
            key = (s.st_dev, s.st_ino)
            if key in seen:
                continue
            seen.add(key)
            logical, allocated = s.st_size, s.st_blocks * 512
            total_allocated += allocated
            total_logical += logical
            relative = p.relative_to(ROOT)
            entry = {'path': str(p), 'logical_bytes': logical, 'allocated_bytes': allocated,
                     'link_count': s.st_nlink}
            files.append(entry)
            parts = relative.parts
            for depth in range(1, min(3, len(parts))):
                parent = '/'.join(parts[:depth])
                directories[parent] = directories.get(parent, 0) + allocated
            if name.endswith('.pyc') and d.name == '__pycache__':
                source_name = re.sub(r'\.(?:cpython|pypy)-[^.]+(?:\.opt-\d+)?\.pyc$', '.py', name)
                source = d.parent/source_name
                if source.is_file() and s.st_nlink == 1:
                    caches.append({**entry, 'source': str(source),
                                   'inside_build_or_install': bool({'build', 'install'} & set(parts)),
                                   'candidate_kind': 'reconstructible_python_bytecode_with_existing_source'})
            if name.endswith(('.part', '.partial', '.crdownload', '.download')):
                partial_downloads.append(entry)
    processes = []
    interesting = ('ddt_ros2_ws', 'd1_runtime03', 'mujoco_bridge', 'wheel-legged-control-lab')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            raw = (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            cwd = os.readlink(proc/'cwd')
            if not raw or not any(token in raw or token in cwd for token in interesting):
                continue
            if 'audit_storage_readonly.py' in raw:
                continue
            status = (proc/'status').read_text()
            rss = re.search(r'^VmRSS:\s+(\d+) kB', status, re.M)
            ppid = re.search(r'^PPid:\s+(\d+)', status, re.M)
            processes.append({'pid': int(proc.name), 'parent_pid': int(ppid[1]) if ppid else None,
                              'rss_bytes': int(rss[1])*1024 if rss else None,
                              'working_directory': cwd, 'command': raw})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree'):
            mem[key+'_bytes'] = int(value.strip().split()[0])*1024
    statvfs = os.statvfs(ROOT)
    report = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': str(ROOT), 'mode': 'read_only_inventory_no_deletion_or_sparse_conversion',
        'deduplicates_inode_allocation': True, 'symlinks_not_followed': ignored_symlinks,
        'total_unique_file_logical_bytes': total_logical, 'total_unique_file_allocated_bytes': total_allocated,
        'filesystem_available_bytes': statvfs.f_bavail*statvfs.f_frsize,
        'top_directories_allocated': sorted(directories.items(), key=lambda x: -x[1])[:35],
        'top_files': sorted(files, key=lambda x: -x['allocated_bytes'])[:30],
        'reconstructible_source_backed_pyc': caches,
        'reconstructible_source_backed_pyc_allocated_bytes': sum(x['allocated_bytes'] for x in caches),
        'source_backed_pyc_outside_build_install_allocated_bytes': sum(x['allocated_bytes'] for x in caches if not x['inside_build_or_install']),
        'partial_download_candidates_require_owner_confirmation': partial_downloads,
        'partial_download_allocated_bytes': sum(x['allocated_bytes'] for x in partial_downloads),
        'related_processes_snapshot': processes, 'memory_snapshot': mem,
        'process_snapshot_scope': 'Only processes visible inside this Python process namespace; an empty list does not establish host process absence. Inspect host ps separately.',
        'preservation': 'All models, checkpoints, old records, evidence, source archives, install/build symlink dependencies and prior frozen files were preserved. A related process is not classified as orphan solely from this snapshot.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open('x').write(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({key: report[key] for key in ('created_utc', 'total_unique_file_allocated_bytes',
        'filesystem_available_bytes', 'top_directories_allocated',
        'reconstructible_source_backed_pyc_allocated_bytes',
        'source_backed_pyc_outside_build_install_allocated_bytes', 'partial_download_allocated_bytes',
        'related_processes_snapshot', 'memory_snapshot')}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
