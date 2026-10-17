#!/usr/bin/env python3
"""Prepare/validate a bounded local comparison; this module never starts training."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'd1-local-comparison-v16'
PREPARED = 'PREPARED_NOT_AUTHORIZED'
AUTHORIZED = 'AUTHORIZED_FROZEN'
TEMPLATE = 'training/configs/local_comparison16.json'
OLD_RELEASE = 'evidence/career_contract_10/scale10_release_02.json'
OLD_RELEASE_SHA = 'd3a0ae0a2ba986ec4e9b5f3a2fa2dc58f159c601312885575d0543afe9e56e9c'
STAGE_ORDER = ('preflight_control', 'preflight_velocity', 'train_control',
               'train_velocity', 'eval_control', 'eval_velocity', 'compare')
SOURCE_FILES = (
    TEMPLATE, 'docs/local_comparison16.md', 'tools16/contract16.py',
    'tools16/train_comparison16.py', 'tools16/run_comparison16.py',
    'tools16/evaluate_comparison16.py', 'tools16/compare_results16.py',
    'training/d1_rl/velocity_policy15.py', 'training/d1_rl/velocity_labels15.py',
    'tests/test_contract16.py', 'tests/test_train_comparison16.py',
    'tests/test_run_comparison16.py', 'tests/test_comparison16.py',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    result = json.loads(Path(path).read_text())
    require(isinstance(result, dict), 'expected a JSON object')
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def relative_file(root, name, *, exists=True):
    root = Path(root).resolve()
    require(isinstance(name, str) and name and not name.startswith('/') and '\\' not in name
            and all(part not in ('', '.', '..') for part in name.split('/')), 'unsafe relative path')
    path = root/name
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), f'symlink is not an input/output: {name}')
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), 'path escaped repository')
    if exists:
        require(path.is_file(), f'missing source: {name}')
    return path


def expected_counts(transitions):
    return {'policy_transitions': transitions, 'native_steps': transitions*10,
            'pd_updates': transitions*5}


def budget(release):
    per_arm = sum(release['stages'][stage]['transitions'] for stage in ('preflight', 'train', 'eval'))
    return expected_counts(per_arm*2)


def source_revision(root):
    """New execution code must already have a reviewable, clean Git identity."""
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}', revision), 'invalid source Git revision')
    tracked = set(subprocess.check_output(['git', 'ls-files', '--', *SOURCE_FILES], cwd=root, text=True).splitlines())
    require(set(SOURCE_FILES) <= tracked, 'commit preparation sources before generating a release')
    changed = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *SOURCE_FILES], cwd=root, text=True)
    require(not changed.strip(), 'preparation sources have uncommitted changes')
    return revision


def validate_revision(release, root):
    revision = release['source_git_commit']
    for name in SOURCE_FILES:
        try:
            data = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=root,
                                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'source commit does not contain {name}') from exc
        require(hashlib.sha256(data).hexdigest() == release['sources_sha256'][name],
                f'commit/source identity differs: {name}')


def canonical_plan(envs=4):
    require(type(envs) is int and envs in (2, 4), 'select 2 or 4 environments before the attempt')
    # Read the shipped template, rather than allowing arbitrary per-run overrides.
    result = read_json(ROOT/TEMPLATE)
    result['envs'] = envs
    return result


def check_plan(release):
    require(isinstance(release, dict), 'release must be an object')
    require(type(release.get('envs')) is int and release['envs'] in (2, 4), 'invalid environment count')
    expected = canonical_plan(release['envs'])
    for name, value in expected.items():
        if name == 'status':
            continue
        require(release.get(name) == value, f'fixed comparison field changed: {name}')
    require(release.get('status') in (PREPARED, AUTHORIZED), 'invalid release status')
    require(release.get('budget') == budget(expected), 'budget does not include both arms and preflights/evaluation')
    require(release.get('stage_order') == list(STAGE_ORDER), 'stage order changed')
    require(release.get('one_attempt_per_stage') is True and release.get('retry_or_resume_allowed') is False,
            'one attempt and no resume are required')
    require(isinstance(release.get('source_git_commit'), str)
            and re.fullmatch('[0-9a-f]{40}', release['source_git_commit']), 'missing source Git identity')


def validate_sources(release, root=ROOT):
    sources = release.get('sources_sha256')
    require(isinstance(sources, dict) and set(SOURCE_FILES) <= set(sources), 'incomplete source closure')
    old_path = relative_file(root, OLD_RELEASE)
    require(sha(old_path) == OLD_RELEASE_SHA, 'closed contract10 release identity changed')
    old = read_json(old_path)['frozen_inputs']
    require(len(old) == 72 and set(old) <= set(sources), 'old 72-input closure missing')
    for name, digest in {OLD_RELEASE: OLD_RELEASE_SHA, **old}.items():
        require(sources.get(name) == digest, f'old frozen identity changed: {name}')
    for name, digest in sources.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'invalid SHA256')
        require(sha(relative_file(root, name)) == digest, f'source changed: {name}')
    validate_revision(release, root)


def authorization(release, prepared_sha):
    """An unsigned template, deliberately not an approval or execution capability."""
    return {'format': 'd1-comparison16-owner-authorization-v1', 'approved': False,
            'prepared_release_sha256': prepared_sha, 'envs': release['envs'], 'seed': release['seed'],
            'budget': release['budget'], 'stage_order': release['stage_order'],
            'one_attempt_per_stage': True, 'retry_or_resume_allowed': False,
            'statement': '未批准。需要仓库所有者明确批准本轮新预算后，才可记录批准并冻结。'}


def check_authorization(auth, prepared, prepared_sha):
    require(auth.get('format') == 'd1-comparison16-owner-authorization-v1' and auth.get('approved') is True,
            'new owner approval is required; the draft does not authorize training')
    expected = authorization(prepared, prepared_sha)
    for key in ('prepared_release_sha256', 'envs', 'seed', 'budget', 'stage_order',
                'one_attempt_per_stage', 'retry_or_resume_allowed'):
        require(auth.get(key) == expected[key], f'approval differs from prepared plan: {key}')
    require(isinstance(auth.get('statement'), str) and bool(auth['statement'].strip())
            and isinstance(auth.get('owner_message_reference'), str)
            and bool(auth['owner_message_reference'].strip()), 'record the explicit owner message and reference')


def load_release(path, require_authorized=True, root=ROOT):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'release must be inside repository')
    release = read_json(path)
    check_plan(release)
    validate_sources(release, root)
    if require_authorized:
        require(release['status'] == AUTHORIZED, 'prepared plan is not authorized to execute')
    if release['status'] == AUTHORIZED:
        proof = release.get('approval_provenance', {})
        prepared_path = relative_file(root, proof.get('prepared_path'))
        auth_path = relative_file(root, proof.get('authorization_path'))
        static_path = relative_file(root, proof.get('static_report_path'))
        require(sha(prepared_path) == proof.get('prepared_sha256')
                and sha(auth_path) == proof.get('authorization_sha256')
                and sha(static_path) == proof.get('static_report_sha256'), 'approval provenance changed')
        require(read_json(prepared_path).get('status') == PREPARED, 'approval must bind the prepared plan')
        prepared = load_release(prepared_path, require_authorized=False, root=root)
        check_authorization(read_json(auth_path), prepared, sha(prepared_path))
        expected = copy.deepcopy(prepared)
        expected['status'] = AUTHORIZED
        expected['approval_provenance'] = proof
        require(release == expected, 'authorized release changed prepared settings')
        static = read_json(static_path)
        require(static.get('format') == 'd1-comparison16-static-v1' and static.get('status') == 'PASS'
                and static.get('prepared_release_sha256') == sha(prepared_path)
                and static.get('sources_sha256') == prepared['sources_sha256']
                and static.get('physics_steps_executed') == 0, 'static report does not bind prepared plan')
    return release


def prepare(output, envs=4, root=ROOT):
    root = Path(root).resolve()
    output = Path(output).resolve()
    require(output.is_relative_to(root/'evidence/local_comparison16'), 'prepare under evidence/local_comparison16')
    release = canonical_plan(envs)
    old_path = relative_file(root, OLD_RELEASE)
    require(sha(old_path) == OLD_RELEASE_SHA, 'old release changed')
    old = read_json(old_path)['frozen_inputs']
    release.update(budget=budget(release), stage_order=list(STAGE_ORDER),
                   source_git_commit=source_revision(root),
                   one_attempt_per_stage=True, retry_or_resume_allowed=False,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   sources_sha256={name: sha(relative_file(root, name))
                                   for name in sorted(set(SOURCE_FILES) | set(old) | {OLD_RELEASE})})
    check_plan(release)
    validate_sources(release, root)
    write_exclusive(output, release)
    return release


def check_prepared(path, output, root=ROOT):
    release = load_release(path, require_authorized=False, root=root)
    require(release['status'] == PREPARED, 'static preparation report requires the unapproved draft')
    report = {'format': 'd1-comparison16-static-v1', 'status': 'PASS',
              'prepared_release_sha256': sha(path), 'sources_sha256': release['sources_sha256'],
              'execution_authorized': False, 'physics_steps_executed': 0,
              'robot_model_required': False, 'budget': release['budget'],
              'checks': ['fixed configuration', 'old 72 inputs unchanged', 'source closure',
                         'both arms including preflight/evaluation', 'fixed order/no retry',
                         'prepared status cannot execute']}
    write_exclusive(output, report)
    return report


def freeze(prepared_path, auth_path, static_path, output, root=ROOT):
    prepared = load_release(prepared_path, require_authorized=False, root=root)
    require(prepared['status'] == PREPARED, 'freeze only a prepared draft')
    check_authorization(read_json(auth_path), prepared, sha(prepared_path))
    root = Path(root).resolve()
    result = copy.deepcopy(prepared)
    result['status'] = AUTHORIZED
    result['approval_provenance'] = {
        'prepared_path': str(Path(prepared_path).resolve().relative_to(root)),
        'prepared_sha256': sha(prepared_path),
        'authorization_path': str(Path(auth_path).resolve().relative_to(root)),
        'authorization_sha256': sha(auth_path),
        'static_report_path': str(Path(static_path).resolve().relative_to(root)),
        'static_report_sha256': sha(static_path)}
    # Validate in memory before writing the execution capability.
    static = read_json(static_path)
    require(static.get('format') == 'd1-comparison16-static-v1' and static.get('status') == 'PASS'
            and static.get('prepared_release_sha256') == sha(prepared_path)
            and static.get('sources_sha256') == prepared['sources_sha256']
            and static.get('physics_steps_executed') == 0, 'invalid static preparation receipt')
    output = Path(output).resolve()
    require(output.is_relative_to(root/'evidence/local_comparison16'), 'freeze under evidence/local_comparison16')
    write_exclusive(output, result)
    return result


def read_memory(proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    """Sample host and every visible cgroup-v2 hard limit; never count swap as RAM."""
    proc_root, cgroup_root = Path(proc_root), Path(cgroup_root)
    fields = dict(line.split(':', 1) for line in (proc_root/'meminfo').read_text().splitlines())
    host = int(fields['MemAvailable'].split()[0])*1024
    require(host >= 0, 'invalid host memory sample')
    scopes = (proc_root/'self/cgroup').read_text().splitlines()
    scope = next((line[3:] for line in scopes if line.startswith('0::')), None)
    require(scope is not None, 'cgroup-v2 memory accounting is required for this local plan')
    require('..' not in Path(scope).parts, 'unsafe cgroup scope')
    current, remaining = cgroup_root/scope.lstrip('/'), []
    require(current.is_dir(), 'current cgroup-v2 scope is not visible')
    while True:
        maximum = current/'memory.max'
        if maximum.exists():
            limit = maximum.read_text().strip()
            if limit != 'max':
                used = int((current/'memory.current').read_text())
                require(int(limit) >= 0 and used >= 0, 'invalid cgroup sample')
                remaining.append(max(0, int(limit)-used))
        if current == cgroup_root:
            break
        current = current.parent
    capped = min(remaining) if remaining else None
    return {'available_ram_bytes': host, 'cgroup_remaining_bytes': capped,
            'effective_available_bytes': min(host, capped) if capped is not None else host}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('prepare')
    make.add_argument('--envs', type=int, choices=(2, 4), default=4)
    make.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('check')
    check.add_argument('--release', type=Path, required=True)
    check.add_argument('--output', type=Path, required=True)
    seal = sub.add_parser('freeze')
    for name in ('prepared', 'authorization', 'static-report', 'output'):
        seal.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.output, args.envs)
    elif args.command == 'check':
        result = check_prepared(args.release, args.output)
    else:
        result = freeze(args.prepared, args.authorization, args.static_report, args.output)
    print(json.dumps({'status': result.get('status'), 'execution_started': False,
                      'physics_steps_executed': 0, 'budget': result.get('budget')}, indent=2))


if __name__ == '__main__':
    main()
