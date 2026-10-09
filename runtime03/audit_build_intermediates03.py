#!/usr/bin/env python3
"""Read-only CMake object/probe inventory; never remove files or invoke a build."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat

ROOT = Path('/home/lyh/ddt_ros2_ws')
BUILD_ROOTS = (ROOT / 'build', ROOT / 'runtime03/build')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(path):
    return Path(os.path.abspath(path))


def files_under(root):
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def cmake_build_directory(path):
    parts = path.parts
    return Path(*parts[:parts.index('CMakeFiles')])


def first_dependency(depfile):
    if not depfile.is_file():
        return None
    joined = depfile.read_text(errors='replace').replace('\\\n', ' ')
    first_rule = joined.split('\n', 1)[0]
    if ':' not in first_rule:
        return None
    dependencies = shlex.split(first_rule.split(':', 1)[1])
    if not dependencies:
        return None
    source = Path(dependencies[0])
    return normalized(source if source.is_absolute() else cmake_build_directory(depfile) / source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    compile_commands = {}
    for database in (ROOT / 'build/compile_commands.json', ROOT / 'build/mujoco/compile_commands.json'):
        for record in json.loads(database.read_text()) if database.is_file() else []:
            tokens = record.get('arguments') or shlex.split(record['command'])
            if '-o' not in tokens:
                continue
            directory = Path(record['directory'])
            output = normalized(directory / tokens[tokens.index('-o') + 1])
            source = normalized(directory / record['file'])
            compile_commands[output] = {'database': str(database), 'source': str(source),
                                        'compiler': tokens[0], 'working_directory': str(directory)}

    build_files = sorted(path for root in BUILD_ROOTS for path in files_under(root))
    links = []
    object_targets = defaultdict(list)
    for linkfile in (path for path in build_files if path.name == 'link.txt'):
        directory = cmake_build_directory(linkfile)
        for line in linkfile.read_text().splitlines():
            tokens = shlex.split(line)
            if not tokens:
                continue
            target = None
            if '-o' in tokens:
                target = normalized(directory / tokens[tokens.index('-o') + 1])
            elif 'ar' in Path(tokens[0]).name:
                archive = next((token for token in tokens[1:] if token.endswith('.a')), None)
                if archive:
                    target = normalized(directory / archive)
            if target:
                record = {'link_file': str(linkfile), 'preserved_target': str(target),
                          'target_exists': target.is_file()}
                links.append(record)
                for token in tokens:
                    if token.endswith('.o'):
                        object_targets[normalized(directory / token)].append(record)

    archives = []
    for path in build_files:
        if path.suffix == '.a':
            magic = path.open('rb').read(8)
            archives.append({'path': str(path), 'magic_ascii': magic.decode('ascii', errors='replace'),
                             'thin_archive': magic == b'!<thin>\n', 'allocated_bytes': path.stat().st_blocks * 512})
    thin_archives = [record for record in archives if record['thin_archive']]

    symlinks = []
    for directory, names, files in os.walk(ROOT, followlinks=False):
        for name in names + files:
            path = Path(directory) / name
            if path.is_symlink():
                try:
                    symlinks.append((path, path.resolve(strict=False)))
                except (OSError, RuntimeError):
                    symlinks.append((path, None))

    candidates = []
    excluded = []
    for path in build_files:
        kind = None
        source = None
        object_path = None
        compile_evidence = None
        if path.name.endswith('.o.d'):
            kind = 'object_dependency_file'
            object_path = Path(str(path)[:-2])
            compile_evidence = compile_commands.get(object_path)
            source = Path(compile_evidence['source']) if compile_evidence else first_dependency(path)
        elif path.name.endswith('.o'):
            kind = 'elf_relocatable_object'
            object_path = path
            compile_evidence = compile_commands.get(path)
            source = Path(compile_evidence['source']) if compile_evidence else first_dependency(Path(str(path) + '.d'))
        elif path.name == 'a.out' and path.parent.name in ('CompilerIdC', 'CompilerIdCXX'):
            kind = 'cmake_compiler_id_probe'
            source = path.parent / ('CMakeCCompilerId.c' if path.parent.name == 'CompilerIdC' else 'CMakeCXXCompilerId.cpp')
        elif path.name in ('CMakeDetermineCompilerABI_C.bin', 'CMakeDetermineCompilerABI_CXX.bin'):
            kind = 'cmake_compiler_abi_probe'
            source = Path('/usr/share/cmake-3.22/Modules') / ('CMakeCCompilerABI.c' if path.name.endswith('_C.bin') else 'CMakeCXXCompilerABI.cpp')
        if kind is None:
            continue
        st = path.stat()
        refs = [{'symlink': str(link), 'resolved_target': str(target)} for link, target in symlinks
                if target is not None and (target == path or (target.is_dir() and path.is_relative_to(target)))]
        targets = object_targets.get(object_path, []) if object_path else []
        reasons = []
        if not stat.S_ISREG(st.st_mode):
            reasons.append('not_a_regular_file')
        if st.st_nlink != 1:
            reasons.append('multiple_hardlinks')
        if refs:
            reasons.append('workspace_symlink_exposes_candidate')
        if source is None or not source.is_file():
            reasons.append('source_not_found')
        if object_path and thin_archives:
            reasons.append('thin_archive_present_requires_member_dependency_review')
        if object_path and (not targets or not all(item['target_exists'] for item in targets)):
            reasons.append('existing_link_target_not_proven')
        elf_type = None
        if kind == 'elf_relocatable_object':
            header = path.open('rb').read(20)
            if header[:4] == b'\x7fELF' and header[5] in (1, 2):
                elf_type = int.from_bytes(header[16:18], 'little' if header[5] == 1 else 'big')
            if elf_type != 1:
                reasons.append('not_elf_ET_REL')
        relative = path.relative_to(ROOT)
        group = str(Path(*relative.parts[:2]))
        record = {'path': str(path), 'group': group, 'category': kind,
                  'logical_bytes': st.st_size, 'allocated_bytes': st.st_blocks * 512,
                  'device': st.st_dev, 'inode': st.st_ino, 'link_count': st.st_nlink,
                  'mtime_ns': st.st_mtime_ns, 'sha256': sha(path), 'elf_type': elf_type,
                  'source': str(source) if source else None, 'source_exists': bool(source and source.is_file()),
                  'compile_command_evidence': compile_evidence,
                  'preserved_link_targets': targets, 'workspace_symlink_references': refs,
                  'exclusion_reasons': reasons}
        (excluded if reasons else candidates).append(record)

    totals = defaultdict(lambda: {'count': 0, 'allocated_bytes': 0})
    category_totals = defaultdict(lambda: {'count': 0, 'allocated_bytes': 0})
    for record in candidates:
        for group in (totals[record['group']], category_totals[record['category']]):
            group['count'] += 1
            group['allocated_bytes'] += record['allocated_bytes']
    target_paths = sorted({Path(record['preserved_target']) for record in links if record['target_exists']})
    preserved_targets = [{'path': str(path), 'sha256': sha(path), 'allocated_bytes': path.stat().st_blocks * 512,
                          'workspace_symlink_references': [str(link) for link, target in symlinks if target == path]}
                         for path in target_paths]
    report = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'mode': 'read_only_inventory_no_deletion_no_build_no_physics',
              'scope': [str(path) for path in BUILD_ROOTS],
              'candidate_count': len(candidates),
              'candidate_allocated_bytes': sum(item['allocated_bytes'] for item in candidates),
              'candidate_groups': dict(totals), 'candidate_categories': dict(category_totals),
              'candidates': candidates, 'excluded': excluded,
              'archives_preserved': archives, 'thin_archives_found': thin_archives,
              'all_workspace_symlinks_scanned': len(symlinks),
              'unresolved_symlink_paths': [str(link) for link, target in symlinks if target is None],
              'linked_targets_preserved': preserved_targets,
              'deletion_authorized_by_this_report': False,
              'process_check': 'Not performed inside this namespace. Root must inspect host ps/lsof and revalidate identities immediately before any authorized deletion.',
              'rebuild_cost': 'Removing .o forces recompilation of those translation units and relinking dependent targets on a later build; timing is unmeasured. Removing .o.d requires regenerated dependency metadata. Retained sources, compile databases, build.make, CMake cache, shared libraries, archives and executables remain. Do not rebuild binaries frozen by an active experiment.',
              'exclusions_policy': 'No final executable, shared library, static archive, CMake configuration, source, dependency git data, source archive, evidence, model or checkpoint is a candidate. Compiler probe executables are listed separately from runtime targets.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(json.dumps({key: report[key] for key in ('candidate_count', 'candidate_allocated_bytes',
          'candidate_groups', 'candidate_categories', 'thin_archives_found', 'all_workspace_symlinks_scanned')}, indent=2))
    print('Excluded', len(excluded), dict(Counter(reason for item in excluded for reason in item['exclusion_reasons'])))


if __name__ == '__main__':
    main()
