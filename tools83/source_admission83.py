"""Source-only admission for the archived83 manifest, never a diagnostic retry.

The root-owned red run recorded the original relative-only rejection. Only
exact inherited absolute names from the pinned81 manifest are now accepted.
No Torch/NumPy/pickle/model/simulator import or numerical rollout is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PHASE = 'GREEN'
PLAN81 = 'docs/experiments81/objective81_preregistered.json'
PLAN81_SHA = '987db250df14869b9f8ce72ab3522d3356de646233341c9a66c4bbccb3d6cf20'
FAILED83_PLAN = 'docs/experiments83/policy_geometry83_preregistered.json'
FAILED83_PLAN_SHA = '616f93d1d337cea4392710b2e66edd05926773c8ed5294e7596c82be1ac3bcce'
PLAN53 = 'docs/experiments53/model_gradient53_preregistered.json'
FAILED83_REPORT = 'evidence/policy_geometry83/execution_01/report.json'
PACKAGES = ('torch', 'numpy', 'mujoco', 'gymnasium')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + str(key))
        result[key] = value
    return result


def read_pinned_json(path, expected_sha):
    """Check exact bytes BEFORE interpreting the inherited path allowlist."""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'pinned plan SHA mismatch: ' + str(path))
    return json.loads(raw, object_pairs_hook=_unique_object)


def _digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'lowercase SHA256 required')


def _safe_relative(name):
    require(isinstance(name, str) and name != '' and '\\' not in name and '\0' not in name,
            'safe canonical relative path required')
    path = Path(name)
    require(not path.is_absolute() and name != '.' and '..' not in path.parts
            and path.as_posix() == name, 'safe canonical relative path required')


def validate_manifest(entries, trusted_inherited, registered_additions):
    """Pure helper; callers must obtain trusted_inherited from pinned81 bytes.

    Additions are an explicit registered whitelist, not arbitrary extra files.
    Existing entry hashes cannot be rebound and no inherited entry can vanish.
    """
    require(all(isinstance(value, dict) for value in (entries, trusted_inherited, registered_additions)),
            'source mappings required')
    require(not (trusted_inherited.keys() & registered_additions.keys()), 'addition overlaps inherited source')
    for name, digest in registered_additions.items():
        _safe_relative(name)
        _digest(digest)
    expected = trusted_inherited | registered_additions
    require(entries.keys() == expected.keys(), 'extra or missing source binding')
    for name, digest in entries.items():
        require(isinstance(name, str), 'string source path required')
        _digest(digest)
        require(digest == expected[name], 'source rebinding rejected: ' + name)
        if Path(name).is_absolute():
            require(name in trusted_inherited and digest == trusted_inherited[name],
                    'only exact pinned inherited absolute bindings are allowed')
        else:
            _safe_relative(name)
    inherited_absolute = sorted(name for name in trusted_inherited if Path(name).is_absolute())
    return {'inherited_entries': len(trusted_inherited), 'registered_new_entries': len(registered_additions),
            'total_entries': len(entries), 'inherited_absolute_count': len(inherited_absolute),
            'inherited_absolute_paths': inherited_absolute}


def check_file_hashes(entries, root, registered_new_names, on_file=None):
    root = Path(root).resolve()
    verified = {}
    for name, expected in sorted(entries.items()):
        path = root / name
        if name in registered_new_names:
            require(path.resolve().is_relative_to(root), 'new relative path resolves outside repository: ' + name)
        actual = sha(path)
        require(actual == expected, 'source hash mismatch: ' + name)
        verified[str(path)] = actual
        if on_file is not None:
            on_file(path, actual)
    return verified


def check_version_metadata(expected, version_getter=None):
    """Only package metadata is queried; importing Torch etc. is forbidden."""
    require(isinstance(expected, dict) and set(expected) == set(PACKAGES), 'fixed package metadata set required')
    require(all(isinstance(value, str) and value for value in expected.values()), 'version strings required')
    get = importlib.metadata.version if version_getter is None else version_getter
    actual = {name: get(name) for name in PACKAGES}
    require(actual == expected, 'package version metadata mismatch')
    return actual


def admit(candidate_path, result, deadline):
    for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
        sys.path.insert(0, str(directory))
    # Pin81 first, before interpreting any allowed absolute source names.
    parent = read_pinned_json(ROOT / PLAN81, PLAN81_SHA)
    require(parent['schema'] == 'd1-objective81-preregistered-v2', 'wrong fixed81 schema')
    inherited = parent['source_sha256']
    archived = read_pinned_json(ROOT / FAILED83_PLAN, FAILED83_PLAN_SHA)
    require(archived['schema'] == 'd1-policy-geometry83-v1', 'wrong archived83 schema')
    additions = {key: value for key, value in archived['source_sha256'].items() if key not in inherited}
    require(len(inherited) == 1390 and len(additions) == 56, 'fixed closure size changed')
    candidate = read_pinned_json(candidate_path, FAILED83_PLAN_SHA)
    result['manifest'] = validate_manifest(candidate['source_sha256'], inherited, additions)
    require(result['manifest']['inherited_absolute_count'] == 6, 'fixed inherited absolute set changed')
    require(candidate == archived, 'candidate is not the unchanged failed83 manifest')

    def retain(path, digest):
        deadline()
        result['inputs_sha256'][str(Path(path).resolve())] = digest
        result['files_verified'] += 1

    for path, digest in ((ROOT / PLAN81, PLAN81_SHA), (ROOT / FAILED83_PLAN, FAILED83_PLAN_SHA),
                         (Path(candidate_path), FAILED83_PLAN_SHA)):
        result['inputs_sha256'][str(path.resolve())] = digest
    check_file_hashes(candidate['source_sha256'], ROOT, set(additions), retain)
    original = read_pinned_json(ROOT / PLAN53, inherited[PLAN53])
    result['version_metadata'] = check_version_metadata(original['versions'])
    failed_path = ROOT / FAILED83_REPORT
    failed_sha = sha(failed_path)
    failed = read_pinned_json(failed_path, failed_sha)
    require(failed['status'] == 'INCOMPLETE' and failed['model_calls'] == failed['actor_calls'] == 0
            and failed['failure'] == 'ValueError: relative binding required'
            and failed['plan_sha256'] == FAILED83_PLAN_SHA, 'archived zero-call failure differs')
    result['inputs_sha256'][str(failed_path)] = failed_sha
    result['archived_failure'] = {key: failed[key] for key in
                                  ('status', 'failure', 'model_calls', 'actor_calls', 'plan_sha256')}
    result['source_compatible'] = True
    result['admission'] = 'SOURCE_MANIFEST_ONLY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT / FAILED83_PLAN)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(IMPLEMENTATION_PHASE == 'GREEN', 'CLI disabled until the red regression is recorded and fixed')
    require(not args.output.exists(), 'exclusive output required; no overwrite or resume')
    cpu_start, wall_start = time.process_time(), time.monotonic()
    result = {'schema': 'source-admission83-v1', 'status': 'INCOMPLETE', 'source_compatible': False,
              'admission': 'NOT_ADMITTED', 'files_verified': 0, 'inputs_sha256': {}, 'failure': None,
              'new_model_calls': 0, 'new_actor_calls': 0, 'new_physics_steps': 0, 'new_updates': 0,
              'scientific_question': 'UNANSWERED', 'execution83_status_unchanged': 'INCOMPLETE',
              'retry_or_resume_authorized': False, 'runtime_model_build_checked': False,
              'hashing_semantics': 'stream opaque file bytes only; no checkpoint/NPZ/pickle deserialization'}
    for path in (Path(__file__).resolve(), ROOT / 'tests/test_source_admission83.py'):
        result['inputs_sha256'][str(path)] = sha(path)
    initial = args.output.with_name(args.output.name + '.initial.json')
    with initial.open('x') as stream:
        json.dump(result, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())

    def stop(signum, _frame):
        raise InterruptedError('source admission interrupted: signal ' + str(signum))

    def deadline():
        require(time.process_time() - cpu_start < 25 and time.monotonic() - wall_start < 55,
                'source admission CPU/wall budget')

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU, signal.SIGALRM):
        signal.signal(signum, stop)
    resource.setrlimit(resource.RLIMIT_CPU, (29, 30))
    signal.alarm(60)
    try:
        admit(args.plan, result, deadline)
        result['status'] = 'COMPLETE'
    except BaseException as exc:  # noqa: BLE001 - interrupted admission must retain its prefix.
        result['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            for path, digest in result['inputs_sha256'].items():
                deadline()
                require(sha(path) == digest, 'source changed after admission: ' + path)
            result['verification_after'] = 'COMPLETE'
        except BaseException as exc:  # noqa: BLE001 - late changes revoke source admission.
            result['status'] = 'INCOMPLETE'
            result['verification_after'] = 'INCOMPLETE'
            result['verification_failure'] = f'{type(exc).__name__}: {exc}'
        if result['status'] != 'COMPLETE':
            result['source_compatible'] = False
            result['admission'] = 'NOT_ADMITTED'
        result['cpu_seconds'] = time.process_time() - cpu_start
        result['wall_seconds'] = time.monotonic() - wall_start
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        signal.alarm(0)
    print(json.dumps({key: result[key] for key in
                      ('status', 'source_compatible', 'admission', 'files_verified', 'scientific_question')}))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
