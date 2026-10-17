"""Source Git identity, completed-checkpoint and partial-accounting adversarial checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

from tools17 import provenance17 as provenance
from tools17 import run_closeout17 as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE, text=True,
                                   env=dict(os.environ, GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
                                            GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid'))


@pytest.fixture
def source_repo(tmp_path):
    root = tmp_path / 'source'
    (root / 'tools17').mkdir(parents=True)
    (root / 'tools17/entry.py').write_text('pass\n')
    for name in provenance.authored_sources(root):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n' if name.endswith('.json') else 'source fixture\n')
    git(root, 'init', '-q')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'fixture sources')
    return root


def test_source_proof_binds_actual_git_blob(source_repo):
    proof = provenance.source_identity(source_repo)
    provenance.check_sources(proof, source_repo)
    assert len(proof['git_commit']) == 40 and 'tools17/entry.py' in proof['sources_sha256']


@pytest.mark.parametrize('change', ['edited', 'forged_hash', 'new_source', 'missing_source', 'fake_revision'])
def test_source_drift_and_forged_revision_cannot_pass(source_repo, change):
    proof = provenance.source_identity(source_repo)
    path = source_repo / 'tools17/entry.py'
    if change in ('edited', 'forged_hash'):
        path.write_text('uncommitted change\n')
        if change == 'forged_hash':
            proof['sources_sha256']['tools17/entry.py'] = provenance.sha256(path)
    elif change == 'new_source':
        (source_repo / 'tools17/unregistered.py').write_text('pass\n')
    elif change == 'missing_source':
        proof['sources_sha256'].pop('tools17/entry.py')
    else:
        proof['git_commit'] = 'f' * 40
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        provenance.check_sources(proof, source_repo)


def bundle(tmp_path, monkeypatch):
    from tools16 import contract16
    root, training = tmp_path / 'source', tmp_path / 'replacement'
    root.mkdir()
    base = training / 'evidence/local_comparison16'
    output = base / 'execution_01'
    target = output / 'train_control'
    target.mkdir(parents=True)
    checkpoint = target / 'policy.zip'
    checkpoint.write_bytes(b'fake policy bytes, never loaded')
    release = {'envs': 4, 'seed': 101, 'arms': {'control': {'aux_coef': 0}},
               'output_root': 'evidence/local_comparison16/execution_01',
               'config': 'training/configs/pilot03d.json', 'sources_sha256': {'fixture': 'a' * 64},
               'budget': {'policy_transitions': 2166288, 'native_steps': 21662880, 'pd_updates': 10831440}}
    write(base / 'release_01.json', release)
    config = {'fixture': True}
    write(root / release['config'], config)
    protocol = {'format': 'd1-comparison16-training-protocol-v1', 'release_sha256': provenance.sha256(base / 'release_01.json'),
                'arm': 'control', 'stage': 'train', 'envs': 4, 'seed': 101,
                'planned_transitions': 1048576, 'inputs_sha256': release['sources_sha256'],
                'random_initialization': True, 'warm_start': False, 'teacher_policy_used': False,
                'hyperparameters': {'aux_coef': 0., 'device': 'cpu', 'cpu_threads': 1, 'n_steps': 1024}, 'config': config}
    write(target / 'protocol.json', protocol)
    workers = [{**{key: value // 4 for key, value in provenance.COUNTS.items()}, 'resets': 2, 'forward_calls': 262150}
               for _ in range(4)]
    counts = {key: sum(worker[key] for worker in workers) for key in workers[0]}
    result = {'format': 'd1-comparison16-training-result-v1', 'release_sha256': protocol['release_sha256'],
              'arm': 'control', 'stage': 'train', 'envs': 4, 'seed': 101, 'status': 'COMPLETE',
              'eligible_for_evaluation': True, 'trained_transitions': 1048576, 'cleanup_complete': True,
              'inputs_unchanged': True, 'error': None, 'aux_coef': 0., 'actual_counts': counts,
              'per_worker_counts': workers, 'protocol_sha256': provenance.sha256(target / 'protocol.json'),
              'checkpoint_sha256': provenance.sha256(checkpoint)}
    write(target / 'result.json', result)
    stages = {name: {'status': 'COMPLETE', 'returncode': 0, 'cleanup_complete': True,
                     'stop_reason': None, 'artifacts_sha256': {}} for name in provenance.ORDER}
    stages['train_control']['child_receipt'] = result
    run = {'format': 'd1-comparison16-execution-v1', 'status': 'COMPLETE',
           'release_sha256': protocol['release_sha256'], 'failed_stage': None, 'error': None,
           'active_stage': None, 'unknown_count_stages': [], 'actual_counts': release['budget'], 'stages': stages}
    write(output / 'report.json', run)
    mesh = b'unchanged mesh'
    mesh_name = 'src/ddt_ros2_control/urdfs/d1_description/meshes/fixture.STL'
    for directory in (root, training):
        path = directory / mesh_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mesh)
    write(base / 'recovery_mesh_inputs.json', {'files': [{'path': 'fixture.STL', 'sha256': hashlib.sha256(mesh).hexdigest()}]})
    summary = {'status': 'COMPLETE', 'replacement': {'root': str(training), 'release_sha256': protocol['release_sha256'],
               'actual_counts': release['budget'], 'checkpoints_sha256': {'control': result['checkpoint_sha256']}},
               'local_artifacts_sha256': {'recovery_mesh_inputs.json': provenance.sha256(base / 'recovery_mesh_inputs.json')}}
    write(root / provenance.SUMMARY, summary)
    monkeypatch.setattr(contract16, 'load_release', lambda *args, **kwargs: release)
    monkeypatch.setattr(contract16, 'validate_sources', lambda *args, **kwargs: None)
    return root, training, protocol, result, run


def test_completed_control_identity_uses_no_model_loader(tmp_path, monkeypatch):
    root, training, protocol, result, _ = bundle(tmp_path, monkeypatch)
    r, p, t, path = provenance.load_completed_control(training, repository_root=root)
    assert r['seed'] == 101 and p == protocol and t == result and path.name == 'policy.zip'


@pytest.mark.parametrize('change', ['root', 'checkpoint', 'partial', 'cleanup', 'worker_float', 'worker_sum',
                                   'run_failed', 'stage_failed', 'receipt', 'config', 'mesh', 'published_hash'])
def test_completed_checkpoint_cannot_be_replaced_or_partially_certified(tmp_path, monkeypatch, change):
    root, training, protocol, result, run = bundle(tmp_path, monkeypatch)
    base = training / 'evidence/local_comparison16'
    output = base / 'execution_01'
    target = output / 'train_control'
    summary = provenance.read_json(root / provenance.SUMMARY)
    if change == 'root':
        summary['replacement']['root'] = str(training / 'different')
    elif change == 'checkpoint':
        (target / 'policy.zip').write_bytes(b'partial or foreign checkpoint')
    elif change == 'partial':
        result['status'] = 'PARTIAL'
    elif change == 'cleanup':
        result['cleanup_complete'] = False
    elif change == 'worker_float':
        result['per_worker_counts'][0]['resets'] = 2.0
    elif change == 'worker_sum':
        result['per_worker_counts'][0]['forward_calls'] += 1
    elif change == 'run_failed':
        run['status'] = 'FAIL'
    elif change == 'stage_failed':
        run['stages']['eval_velocity']['status'] = 'FAIL'
    elif change == 'receipt':
        run['stages']['train_control']['child_receipt'] = {}
    elif change == 'config':
        protocol['config'] = {'different': True}
        write(target / 'protocol.json', protocol)
        result['protocol_sha256'] = provenance.sha256(target / 'protocol.json')
    elif change == 'mesh':
        (root / 'src/ddt_ros2_control/urdfs/d1_description/meshes/fixture.STL').write_bytes(b'changed mesh')
    else:
        summary['replacement']['checkpoints_sha256']['control'] = '0' * 64
    if change != 'receipt':
        run['stages']['train_control']['child_receipt'] = copy.deepcopy(result)
    write(target / 'result.json', result)
    write(output / 'report.json', run)
    write(root / provenance.SUMMARY, summary)
    with pytest.raises(ValueError):
        provenance.load_completed_control(training, repository_root=root)


def test_partial_formal_counts_require_their_own_provenance(tmp_path):
    protocol = tmp_path / 'protocol.json'
    release = {'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True}}
    write(protocol, {'format': 'd1-closeout17-formal-protocol-v1', 'release_sha256': 'b' * 64,
                     'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True}})
    child = {'format': 'd1-closeout17-formal-result-v1', 'release_sha256': 'b' * 64,
             'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True},
             'protocol_sha256': provenance.sha256(protocol), 'status': 'INCOMPLETE', 'inputs_unchanged': False}
    assert runner.minimum_formal_identity(child, release, 'b' * 64, tmp_path)
    child['checkpoint_sha256'] = 'c' * 64
    assert not runner.minimum_formal_identity(child, release, 'b' * 64, tmp_path)


def test_foreign_protocol_cannot_launder_partial_counts(tmp_path):
    protocol = tmp_path / 'protocol.json'
    release = {'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True}}
    write(protocol, {'format': 'd1-closeout17-formal-protocol-v1', 'release_sha256': 'foreign',
                     'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True}})
    child = {'format': 'd1-closeout17-formal-result-v1', 'release_sha256': 'b' * 64,
             'checkpoint_sha256': 'a' * 64, 'source_identity': {'fixture': True},
             'protocol_sha256': provenance.sha256(protocol)}
    assert not runner.minimum_formal_identity(child, release, 'b' * 64, tmp_path)


def test_cleanup_ignores_interrupts_only_while_reaping_owned_group(monkeypatch):
    observed = []
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def fake_cleanup(process, resources, natural_exit):
        observed.extend(signal.getsignal(sig) for sig in before)
        raise RuntimeError('fake cleanup fault')
    monkeypatch.setattr(runner, 'cleanup', fake_cleanup)
    sent, clean, error = runner.cleanup_owned(SimpleNamespace(), False)
    assert sent == [] and clean is False and 'fake cleanup fault' in error
    assert observed == [signal.SIG_IGN, signal.SIG_IGN]
    assert {sig: signal.getsignal(sig) for sig in before} == before


def test_changed_completed_stage_is_rejected(tmp_path):
    file = tmp_path / 'report.json'
    file.write_text('{}')
    record = {'status': 'COMPLETE', 'artifacts_sha256': runner.artifact_hashes(tmp_path)}
    runner.unchanged_stages({'stages': {'export': record}})
    file.write_text('{"status":"forged PASS"}')
    with pytest.raises(ValueError, match='artifact changed'):
        runner.unchanged_stages({'stages': {'export': record}})
