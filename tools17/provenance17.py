"""Shared, simulator-free provenance checks for final comparison16 control validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = 'reports/local_comparison16_execution.json'
ORDER = ('preflight_control', 'preflight_velocity', 'train_control', 'train_velocity',
         'eval_control', 'eval_velocity', 'compare')
COUNTS = {'policy_transitions': 1048576, 'native_steps': 10485760, 'pd_updates': 5242880}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(value, dict), 'expected a JSON object')
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def exclusive_json(path, value):
    path = Path(path)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def authored_sources(root=ROOT):
    root = Path(root)
    names = {SUMMARY, 'tools15/export_velocity15.py', 'tools16/contract16.py',
             'tools16/evaluate_comparison16.py', 'tools/evaluate_d1_ppo.py',
             'docs/closeout17_contract.md'}
    names.update(p.relative_to(root).as_posix() for p in (root / 'tools17').glob('*.py'))
    names.update(p.relative_to(root).as_posix() for p in (root / 'tests').glob('test_*17.py'))
    return sorted(names)


def source_identity(root=ROOT):
    root = Path(root).resolve()
    names = authored_sources(root)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(re.fullmatch('[0-9a-f]{40}', revision), 'invalid source commit')
    tracked = set(subprocess.check_output(['git', 'ls-files', '--', *names], cwd=root, text=True).splitlines())
    require(set(names) <= tracked, 'commit all closeout sources before real execution')
    proof = {'git_commit': revision, 'sources_sha256': {name: sha256(root / name) for name in names}}
    check_sources(proof, root=root)
    return proof


def check_sources(proof, root=ROOT):
    root = Path(root).resolve()
    require(isinstance(proof, dict) and re.fullmatch('[0-9a-f]{40}', proof.get('git_commit', '')),
            'missing source Git identity')
    sources = proof.get('sources_sha256')
    require(isinstance(sources, dict) and set(sources) == set(authored_sources(root)),
            'closeout source set changed')
    for name, digest in sources.items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'unsafe source path')
        require(sha256(path) == digest, 'source changed: ' + name)
        data = subprocess.check_output(['git', 'show', f'{proof["git_commit"]}:{name}'], cwd=root,
                                       stderr=subprocess.PIPE)
        require(hashlib.sha256(data).hexdigest() == digest, 'uncommitted source: ' + name)


def _check_counts(result, envs):
    total, workers = result.get('actual_counts'), result.get('per_worker_counts')
    require(isinstance(total, dict) and all(type(total.get(k)) is int and total[k] == v
                                           for k, v in COUNTS.items()), 'formal training counters differ')
    require(isinstance(workers, list) and len(workers) == envs, 'missing training workers')
    for worker in workers:
        require(isinstance(worker, dict) and set(worker) == set(total)
                and all(type(v) is int and v >= 0 for v in worker.values())
                and all(worker[k] == v // envs for k, v in COUNTS.items()), 'worker counters differ')
    require(all(type(v) is int and v >= 0 and sum(w[k] for w in workers) == v for k, v in total.items()),
            'worker counters do not sum to training receipt')


def load_completed_control(training_root, *, repository_root=ROOT):
    """Return only the published, completed aux=0 checkpoint, without loading a policy."""
    from tools16.contract16 import load_release, validate_sources
    repository_root = Path(repository_root).resolve()
    training_root = Path(training_root).resolve()
    summary = read_json(repository_root / SUMMARY)
    published = summary['replacement']
    require(summary.get('status') == 'COMPLETE' and published.get('root') == str(training_root),
            'training root is not the completed published replacement')
    base = training_root / 'evidence/local_comparison16'
    release_path = base / 'release_01.json'
    require(sha256(release_path) == published.get('release_sha256'), 'training release SHA changed')
    release = load_release(release_path, require_authorized=True, root=training_root)
    validate_sources(release, root=repository_root)
    require(release['envs'] == 4 and release['seed'] == 101
            and release['arms']['control']['aux_coef'] == 0, 'published control settings changed')
    execution = training_root / release['output_root']
    run = read_json(execution / 'report.json')
    require(run.get('format') == 'd1-comparison16-execution-v1' and run.get('status') == 'COMPLETE'
            and run.get('release_sha256') == sha256(release_path)
            and run.get('failed_stage') is None and run.get('error') is None
            and run.get('active_stage') is None and run.get('unknown_count_stages') == []
            and run.get('actual_counts') == release['budget'] == published['actual_counts']
            and tuple(run.get('stages', {})) == ORDER, 'training comparison was not completed')
    for name, stage in run['stages'].items():
        require(stage.get('status') == 'COMPLETE' and stage.get('returncode') == 0
                and stage.get('cleanup_complete') is True and stage.get('stop_reason') is None,
                'training stage not complete: ' + name)
        for artifact, digest in stage.get('artifacts_sha256', {}).items():
            path = Path(artifact)
            require(path.resolve().is_relative_to(execution) and sha256(path) == digest,
                    'completed training artifact changed: ' + artifact)
    for relative, digest in summary['local_artifacts_sha256'].items():
        path = base / relative
        require(path.resolve().is_relative_to(base) and sha256(path) == digest,
                'published evidence changed: ' + relative)
    assets = read_json(base / 'recovery_mesh_inputs.json')['files']
    for item in assets:
        name = item['path']
        require(isinstance(name, str) and Path(name).name == name, 'unsafe mesh asset path')
        for root in (repository_root, training_root):
            path = root / 'src/ddt_ros2_control/urdfs/d1_description/meshes' / name
            require(sha256(path) == item['sha256'], 'training mesh asset changed: ' + name)
    output = execution / 'train_control'
    checkpoint = output / 'policy.zip'
    require(checkpoint.is_file() and not checkpoint.is_symlink(), 'missing completed policy.zip')
    protocol = read_json(output / 'protocol.json')
    result = read_json(output / 'result.json')
    require(result == run['stages']['train_control'].get('child_receipt'), 'training receipt differs from run')
    common = {'release_sha256': sha256(release_path), 'arm': 'control', 'stage': 'train',
              'envs': 4, 'seed': 101}
    require(all(item.get(k) == v for item in (protocol, result) for k, v in common.items()),
            'training protocol or result identity differs')
    require(protocol.get('format') == 'd1-comparison16-training-protocol-v1'
            and result.get('format') == 'd1-comparison16-training-result-v1'
            and protocol.get('planned_transitions') == COUNTS['policy_transitions']
            and protocol.get('inputs_sha256') == release['sources_sha256']
            and protocol.get('random_initialization') is True and protocol.get('warm_start') is False
            and protocol.get('teacher_policy_used') is False, 'training provenance differs')
    require(result.get('status') == 'COMPLETE' and result.get('eligible_for_evaluation') is True
            and result.get('trained_transitions') == COUNTS['policy_transitions']
            and result.get('cleanup_complete') is True and result.get('inputs_unchanged') is True
            and result.get('error') is None and result.get('aux_coef') == 0,
            'partial or ineligible training checkpoint')
    _check_counts(result, release['envs'])
    require(result.get('protocol_sha256') == sha256(output / 'protocol.json'), 'training protocol SHA differs')
    require(result.get('checkpoint_sha256') == sha256(checkpoint)
            == published['checkpoints_sha256']['control'], 'published checkpoint SHA differs')
    hyper = protocol.get('hyperparameters', {})
    require(hyper.get('aux_coef') == 0 and hyper.get('device') == 'cpu'
            and hyper.get('cpu_threads') == 1 and hyper.get('n_steps') == 1024,
            'control hyperparameters differ')
    require(protocol.get('config') == read_json(repository_root / release['config']), 'training config differs')
    return release, protocol, result, checkpoint
