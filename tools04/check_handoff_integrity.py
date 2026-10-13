#!/usr/bin/env python3
"""Read-only handoff checks; write an exclusive receipt below ddt/evidence.

The baseline receipts are pinned here, not reconstructed from current files.
This command never trains, constructs a robot, integrates physics or cleans up.
External URLs and Markdown anchors are reported but not network-validated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'wheel-legged-control-lab'
GUARD = 'evidence/career_contract_03/protected_guard_final03_01.json'
GUARD_SHA256 = '08533c1d6741b67b781c2180ec75b6278a9752dc11b2d97d40897c3b944682aa'
BASE_RELEASES = (
    ('evidence/career_contract_03/pilot03d_release_01.json', 22,
     'e0162abec0975aac11154960161c110ca93079ec4d7ccea741ceab9ee0a150e1'),
    ('evidence/career_contract_04/pilot04a_release_02.json', 22,
     'b725c9cafb6fa483a92c08a387bf6f7dc7beecbfaae18e20a5b20e6dcc9c819a'),
    ('evidence/career_contract_04/pilot04b_release_01.json', 23,
     'a305497ff0c02fd5e22211524c01acb4b5726590e6220e6cc35f4e4b3662ecb6'),
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def command(args, cwd, env=None):
    """A failed Git/process invocation must never resemble an empty result."""
    return subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                          text=True, capture_output=True, check=True).stdout


def relative_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError(f'invalid relative path: {name!r}')
    p = Path(name)
    if p.is_absolute() or any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError(f'invalid relative path: {name!r}')
    root = Path(root).resolve()
    target = root/p
    # Only components below root are inspected. Ancestors of root (for example a
    # symlinked /home) are the caller's environment, not part of the frozen set.
    inside = [root.joinpath(*p.parts[:i]) for i in range(1, len(p.parts)+1)]
    if any(part.is_symlink() for part in inside):
        raise ValueError(f'symlink path is not a frozen regular file: {name}')
    if not target.resolve().is_relative_to(root):
        raise ValueError(f'path escapes root: {name}')
    return target


def verify_hash_manifest(root, expected, expected_count=None):
    """Reusable fail-closed hash audit. Malformed manifests raise ValueError.

    Missing/changed files are explicit FAIL rows, so callers must inspect status.
    No imports from the training or simulator stack are needed.
    """
    if not isinstance(expected, dict) or not expected:
        raise ValueError('hash manifest must be a nonempty object')
    if expected_count is not None and len(expected) != expected_count:
        raise ValueError(f'expected {expected_count} manifest entries, got {len(expected)}')
    rows = []
    for name, digest in sorted(expected.items()):
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError(f'invalid SHA256 for {name!r}')
        target = relative_path(root, name)
        actual = sha256(target) if target.is_file() else None
        rows.append({'path': name, 'expected': digest, 'actual': actual,
                     'status': 'PASS' if actual == digest else 'FAIL'})
    passed = sum(row['status'] == 'PASS' for row in rows)
    return {'status': 'PASS' if passed == len(rows) else 'FAIL',
            'count': len(rows), 'passed': passed, 'files': rows}


def check_protected_files(ddt_root, old_root, *, guard_sha256=GUARD_SHA256):
    """Audit the 77 frozen files, exact 9 untracked results and old tracked tree.

    The optional digest supports synthetic fixtures. Production callers should
    retain the pinned default. This function writes nothing in either repo.
    """
    guard_path = relative_path(ddt_root, GUARD)
    if sha256(guard_path) != guard_sha256:
        raise ValueError('protected guard receipt SHA256 changed')
    guard = json.loads(guard_path.read_text())['guard']
    frozen = verify_hash_manifest(old_root, guard['frozen_77_sha256'], 77)
    metadata = guard['my_course_drive_01']
    if not isinstance(metadata, dict):
        raise ValueError('user result manifest must be an object')
    expected = {p: m['sha256'] for p, m in metadata.items()}
    drive = relative_path(old_root, 'results/my_course_drive_01')
    user = verify_hash_manifest(drive, expected, 9)
    # os.walk does not follow links. Any additional symlink is still an entry,
    # and therefore cannot hide an unexpected file behind a directory link.
    actual = []
    for directory, dirs, files in os.walk(drive, followlinks=False):
        for name in files + [n for n in dirs if (Path(directory)/n).is_symlink()]:
            actual.append((Path(directory)/name).relative_to(drive).as_posix())
    user['actual_files'] = sorted(actual)
    user['exact_file_set'] = user['actual_files'] == sorted(expected)
    user['tracked'] = command(['git', 'ls-files', '-z', '--',
                               'results/my_course_drive_01'], old_root).split('\0')
    user['tracked'] = [p for p in user['tracked'] if p]
    changes = command(['git', 'status', '--porcelain', '-z',
                       '--untracked-files=no'], old_root).split('\0')
    changes = [p for p in changes if p]
    head = command(['git', 'rev-parse', 'HEAD'], old_root).strip()
    if not re.fullmatch('[0-9a-f]{40,64}', head):
        raise ValueError('old Git HEAD is empty or malformed')
    if not user['exact_file_set'] or user['tracked']:
        user['status'] = 'FAIL'
    return {'status': 'PASS' if frozen['status'] == user['status'] == 'PASS'
            and not changes else 'FAIL', 'guard_receipt_sha256': guard_sha256,
            'frozen_77': frozen, 'user_results_9': user,
            'old_tracked_changes': changes, 'old_head': head}


def check_release_inputs(root, extra_releases=()):
    releases = list(BASE_RELEASES)
    for name in extra_releases:
        if name in {r[0] for r in releases}:
            raise ValueError(f'duplicate release: {name}')
        # Additional receipts cannot replace any of the three historical ones.
        if not name.startswith('evidence/') or not name.endswith('.json'):
            raise ValueError('extra release must be a relative evidence/*.json path')
        releases.append((name, None, None))
    rows = {}
    for name, count, pinned_digest in releases:
        path = relative_path(root, name)
        digest = sha256(path)
        if pinned_digest is not None and digest != pinned_digest:
            raise ValueError(f'release receipt SHA256 changed: {name}')
        data = json.loads(path.read_text())
        rows[name] = verify_hash_manifest(root, data['inputs_sha256'], count)
        rows[name]['release_sha256'] = digest
    return {'status': 'PASS' if all(r['status'] == 'PASS' for r in rows.values())
            else 'FAIL', 'receipts': rows}


def prose_only(source):
    """Remove fenced/indented and inline code before looking for links."""
    result, fence = [], None
    for line in source.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            continue
        if match:
            fence = match[1]
        elif not line.startswith(('    ', '\t')):
            result.append(line)
    return re.sub(r'(`+).*?\1(?!`)', '', '\n'.join(result), flags=re.S)


def markdown_targets(source):
    source = prose_only(source)
    references = {}
    for match in re.finditer(r'^ {0,3}\[([^]\n]+)\]:\s*(<[^>]+>|\S+)',
                             source, re.M):
        references[match[1].strip().casefold()] = match[2].strip('<>')
    targets = list(references.values())
    # Allow balanced parentheses in unquoted destinations, e.g. file_(v2).md.
    for match in re.finditer(r'(?<!!)\[[^]\n]*\]\(|!\[[^]\n]*\]\(', source):
        rest = source[match.end():].lstrip()
        if rest.startswith('<'):
            end = rest.find('>')
            if end >= 0:
                targets.append(rest[1:end])
            continue
        depth, chars = 0, []
        for char in rest:
            if char == ')' and depth == 0 or char.isspace():
                break
            depth += (char == '(') - (char == ')')
            chars.append(char)
        if chars:
            targets.append(''.join(chars))
    return targets


def check_document_links(root):
    names = command(['git', 'ls-files', '-z', '--cached', '--others',
                     '--exclude-standard', '--', '*.md', '*.MD'], root)
    documents = sorted(set(filter(None, names.split('\0'))))
    local, external = [], set()
    for name in documents:
        path = relative_path(root, name)
        for target in markdown_targets(path.read_text()):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                external.add(target)
            elif parts.path:
                destination = path.parent/unquote(parts.path)
                local.append({'document': name, 'target': target,
                              'exists': destination.exists()})
    broken = [row for row in local if not row['exists']]
    return {'status': 'FAIL' if broken else 'PASS', 'documents': len(documents),
            'local_checked': len(local), 'local': local, 'broken': broken,
            'external_targets': sorted(external),
            'scope': 'local filesystem targets; external URLs and anchors not validated'}


def new_receipt_directory(root, old_root, requested):
    root, old_root = Path(root).resolve(), Path(old_root).resolve()
    if root == old_root or root.is_relative_to(old_root):
        raise ValueError('ddt root must be outside the read-only old repository')
    candidate = Path(requested)
    candidate = candidate if candidate.is_absolute() else root/candidate
    evidence = root/'evidence'
    if not candidate.is_relative_to(evidence) or candidate == evidence:
        raise ValueError('output must be a new directory below ddt/evidence')
    # Reject traversal and symlinks before making any directory.
    relative_path(root, candidate.relative_to(root).as_posix())
    resolved = candidate.resolve()
    if resolved == old_root or resolved.is_relative_to(old_root):
        raise ValueError('receipt may not touch the read-only old repository')
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def check_static(root, output):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root/'.python-deps')+os.pathsep+str(root/'training')
    stdout = command([sys.executable, '-B', 'tools04/check_contract_static_ci.py',
                      '--output', output/'static'], root, env)
    (output/'static.stdout.txt').write_text(stdout)
    report = json.loads((output/'static/report.json').read_text())
    passed = (report['status'] == 'PASS' and report['checks_total'] == 32
              and report['failed_checks'] == []
              and type(report['native_integration_steps_executed']) is int
              and report['native_integration_steps_executed'] == 0
              and report['robot_model_required'] is False
              and len(report['checks']) == 32
              and all(c['status'] == 'PASS' for c in report['checks'])
              and report['tool_sha256'] == sha256(root/'tools04/check_contract_static_ci.py'))
    return {'status': 'PASS' if passed else 'FAIL', 'receipt': report}


def run(root, old_root, output, extra_releases=()):
    """Run every independent audit, retaining failures in a fresh receipt."""
    root, old_root = Path(root), Path(old_root)
    report = {'format': 'd1-handoff-integrity-v1',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0, 'robot_model_required': False,
              'status': 'FAIL', 'errors': []}
    try:
        out = new_receipt_directory(root, old_root, output)
    except Exception as exc:
        report['errors'].append({'check': 'output', 'error': f'{type(exc).__name__}: {exc}'})
        out = new_receipt_directory(root, old_root,
                                   'evidence/handoff_integrity_failures/'+uuid.uuid4().hex)
    checks = (
        ('release_inputs', lambda: check_release_inputs(root, extra_releases)),
        ('protected_files', lambda: check_protected_files(root, old_root)),
        ('document_links', lambda: check_document_links(root)),
        ('static', lambda: check_static(root, out)),
    )
    for name, check in checks:
        try:
            report[name] = check()
            if report[name]['status'] != 'PASS':
                report['errors'].append({'check': name, 'error': 'failed checks; see detail'})
        except Exception as exc:
            report['errors'].append({'check': name, 'error': f'{type(exc).__name__}: {exc}'})
    try:
        report['ddt_head'] = command(['git', 'rev-parse', 'HEAD'], root).strip()
        if not re.fullmatch('[0-9a-f]{40,64}', report['ddt_head']):
            raise ValueError('ddt Git HEAD is empty or malformed')
        report['diff_check'] = command(['git', 'diff', '--check'], root)
        report['cached_diff_check'] = command(['git', 'diff', '--cached', '--check'], root)
    except Exception as exc:
        report['errors'].append({'check': 'git', 'error': f'{type(exc).__name__}: {exc}'})
    report['status'] = 'FAIL' if report['errors'] else 'PASS'
    # Exclusive creation protects history even if an unexpected file appeared.
    with (out/'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return report, out/'report.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='new directory below this repository/evidence')
    parser.add_argument('--extra-release', action='append', default=[],
                        help='additional evidence/*.json manifest, never replaces the base three')
    args = parser.parse_args(argv)
    try:
        report, receipt = run(ROOT, OLD, args.output, args.extra_release)
    except Exception as exc:
        print(json.dumps({'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}',
                          'receipt': None}), file=sys.stderr)
        return 1
    print(json.dumps({'status': report['status'], 'errors': report['errors'],
                      'receipt': str(receipt)}, indent=2, ensure_ascii=False))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
