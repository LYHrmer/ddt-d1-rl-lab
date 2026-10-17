"""Frozen inputs and budgets for new, independent delay-robustness work."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools17 import provenance17  # noqa: E402
from tools16.contract16 import atomic_json, expected_counts, read_memory  # noqa: E402,F401

FORMAT = 'd1-independent-delay-optimization18-v1'
OUTPUT = 'evidence/local_optimization18/execution_01'
OWNER = '把这些优化点，在这台电脑能力之内的都完成'
RESOURCES = {'start_min_effective_bytes': 6 * 2**30, 'reserve_bytes': 3 * 2**30,
             'poll_s': 1., 'interrupt_grace_s': 10, 'terminate_grace_s': 5,
             'kill_grace_s': 2, 'process_wall_grace_s': 30}
STAGES = {
    'probe_base': {'kind': 'evaluate', 'transitions': 600, 'max_wall_s': 120},
    'preflight_zero': {'kind': 'train', 'mode': 'zero', 'transitions': 32768, 'max_wall_s': 900},
    'preflight_random': {'kind': 'train', 'mode': 'uniform_0_3', 'transitions': 32768, 'max_wall_s': 900},
    'train_zero': {'kind': 'train', 'mode': 'zero', 'transitions': 1048576, 'max_wall_s': 5400},
    'train_random': {'kind': 'train', 'mode': 'uniform_0_3', 'transitions': 1048576, 'max_wall_s': 5400},
    'export_zero': {'kind': 'export', 'transitions': 0, 'max_wall_s': 180},
    'cpp_zero': {'kind': 'cpp', 'transitions': 0, 'max_wall_s': 90},
    'export_random': {'kind': 'export', 'transitions': 0, 'max_wall_s': 180},
    'cpp_random': {'kind': 'cpp', 'transitions': 0, 'max_wall_s': 90},
    'eval_zero': {'kind': 'evaluate', 'transitions': 21600, 'max_wall_s': 600},
    'eval_random': {'kind': 'evaluate', 'transitions': 21600, 'max_wall_s': 600},
    'holdout': {'kind': 'evaluate', 'transitions': 21600, 'max_wall_s': 600},
}
SOURCE_NAMES = (
    'tools18/common18.py', 'tools18/run_optimization18.py', 'tools18/train_delay18.py',
    'tools18/evaluate_delay18.py', 'tools18/export_delay18.py',
    'training/d1_rl/delay_env18.py', 'tests/test_delay_env18.py',
    'tests/test_train_delay18.py', 'tests/test_common18.py', 'tests/test_evaluate_delay18.py',
    'tests/test_export_delay18.py',
    'docs/local_optimization18.md',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return provenance17.sha256(Path(path))


def read_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'expected JSON object')
    return value


def source_identity():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    proof = {'git_commit': revision, 'sources_sha256': {name: sha(ROOT/name) for name in SOURCE_NAMES}}
    check_sources(proof)
    return proof


def check_sources(proof):
    require(isinstance(proof, dict) and isinstance(proof.get('git_commit'), str)
            and len(proof['git_commit']) == 40, 'missing new18 Git identity')
    sources = proof.get('sources_sha256')
    require(isinstance(sources, dict) and set(sources) == set(SOURCE_NAMES), 'new18 source set changed')
    for name, digest in sources.items():
        path = ROOT/name
        require(not path.is_symlink() and path.resolve().is_relative_to(ROOT), 'unsafe new18 source')
        require(sha(path) == digest, 'new18 source changed: ' + name)
        blob = subprocess.check_output(['git', 'show', f'{proof["git_commit"]}:{name}'], cwd=ROOT,
                                       stderr=subprocess.PIPE)
        require(hashlib.sha256(blob).hexdigest() == digest, 'uncommitted new18 source: ' + name)


def budget():
    return expected_counts(sum(item['transitions'] for item in STAGES.values()))


def verify_inputs(plan):
    check_sources(plan['source_identity'])
    old_release, protocol, result, checkpoint = provenance17.load_completed_control(plan['training_root'])
    expected = plan['base_checkpoint']
    require(str(checkpoint) == expected['path'] and sha(checkpoint) == expected['sha256']
            and result['trained_transitions'] == expected['trained_transitions'] == 1048576,
            'completed baseline checkpoint changed')
    require(plan['base_protocol']['path'] == str(checkpoint.parent/'protocol.json')
            and sha(checkpoint.parent/'protocol.json') == plan['base_protocol']['sha256'], 'baseline protocol changed')
    summary_path = ROOT/'reports/closeout17_20261018.json'
    require(sha(summary_path) == '0321b79882cdb784d03703dbb748223b3a027dda726cbe957e489cfe6414c935',
            'closed closeout17 summary changed')
    require(plan['old_closeout_artifacts_sha256'] == read_json(summary_path)['local_artifacts_sha256'],
            'closed artifact proof differs')
    for name, digest in plan['old_closeout_artifacts_sha256'].items():
        require(sha(ROOT/name) == digest, 'closed closeout17 artifact changed: ' + name)
    provenance17.check_sources(read_json(ROOT/'evidence/closeout17/release_01.json')['source_identity'])
    return old_release, protocol, result, checkpoint


def prepare(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT/'evidence/local_optimization18') and not path.exists(), 'new plan path required')
    training_root = provenance17.read_json(ROOT/provenance17.SUMMARY)['replacement']['root']
    _, _, result, checkpoint = provenance17.load_completed_control(training_root)
    old = read_json(ROOT/'reports/closeout17_20261018.json')
    require(old['full_delay_reference_qualification'] == 'NOT_PASSED', 'unexpected starting qualification')
    plan = {'format': FORMAT, 'owner_statement': OWNER, 'status': 'FROZEN_NEW_SCOPE',
            'authorization_scope': 'New active owner goal; independently capped local work, not an old attempt retry.',
            'training_root': str(training_root), 'config': 'training/configs/pilot03d.json',
            'seed': 181, 'envs': 4, 'rollout_transitions': 4096,
            'base_checkpoint': {'path': str(checkpoint), 'sha256': sha(checkpoint),
                                'trained_transitions': result['trained_transitions']},
            'base_protocol': {'path': str(checkpoint.parent/'protocol.json'),
                              'sha256': sha(checkpoint.parent/'protocol.json')},
            'source_identity': source_identity(), 'resources': RESOURCES,
            'stages': STAGES, 'stage_order': list(STAGES), 'budget': budget(), 'output_root': OUTPUT,
            'one_attempt_per_stage': True, 'retry_or_resume_allowed': False,
            'warm_start': True, 'preflight_weights_discarded': True,
            'selection': 'All36 original reference flags true; lowest mean J among eligible arms, zero tie first.',
            'holdout_reset_seeds': [2001, 2002, 2003],
            'old_closeout_artifacts_sha256': old['local_artifacts_sha256']}
    verify_inputs(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return plan


def load_plan(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT/'evidence/local_optimization18'), 'plan outside new18 scope')
    plan = read_json(path)
    expected = {'format': FORMAT, 'owner_statement': OWNER, 'status': 'FROZEN_NEW_SCOPE',
                'config': 'training/configs/pilot03d.json', 'seed': 181, 'envs': 4,
                'rollout_transitions': 4096, 'resources': RESOURCES, 'stages': STAGES,
                'stage_order': list(STAGES), 'budget': budget(), 'output_root': OUTPUT,
                'one_attempt_per_stage': True, 'retry_or_resume_allowed': False,
                'warm_start': True, 'preflight_weights_discarded': True,
                'selection': 'All36 original reference flags true; lowest mean J among eligible arms, zero tie first.',
                'holdout_reset_seeds': [2001, 2002, 2003]}
    require(all(plan.get(key) == value for key, value in expected.items()), 'fixed optimization scope changed')
    require(set(plan['source_identity']['sources_sha256']) == set(SOURCE_NAMES), 'incomplete new18 source proof')
    verify_inputs(plan)
    return plan


def validate_launch(plan, plan_path, stage, output):
    require(stage in STAGES, 'unknown stage')
    execution = ROOT/plan['output_root']
    require(Path(output).resolve() == execution/stage, 'stage output outside fixed attempt')
    journal = read_json(execution/'report.json')
    record = journal.get('stages', {}).get(stage, {})
    require(journal.get('status') == 'RUNNING' and journal.get('active_stage') == stage
            and journal.get('supervisor_pid') == os.getppid()
            and journal.get('plan_sha256') == sha(plan_path)
            and record.get('status') == 'STARTED'
            and bool(record.get('nonce')) and record['nonce'] == os.environ.get('D1_STAGE18_NONCE'),
            'stage needs its live bounded parent supervision')
    previous = plan['stage_order'][:plan['stage_order'].index(stage)]
    require(all(journal.get('stages', {}).get(name, {}).get('status') == 'COMPLETE' for name in previous),
            'stage prerequisites incomplete')
    from tools18.run_optimization18 import command
    require(record.get('command') == command(Path(plan_path).resolve(), stage, Path(output).resolve()),
            'stage launch command differs')


def completed_training(plan, mode):
    require(mode in ('zero', 'random'), 'unknown trained arm')
    output = ROOT/plan['output_root']/('train_'+mode)
    report = read_json(output/'result.json')
    protocol_path = output/'protocol.json'
    protocol = read_json(protocol_path)
    require(report.get('status') == 'COMPLETE' and report.get('cleanup_complete') is True
            and report.get('inputs_unchanged') is True
            and report.get('new_trained_transitions') == 1048576
            and report.get('initial_num_timesteps') == 1048576
            and report.get('final_num_timesteps') == 2097152
            and report.get('error') is None and report.get('stage') == 'train_'+mode
            and report.get('source_identity') == plan['source_identity']
            and report.get('protocol_sha256') == sha(protocol_path)
            and protocol.get('source_identity') == plan['source_identity']
            and protocol.get('stage') == 'train_'+mode,
            'new training arm incomplete')
    require(report.get('checkpoint_sha256') == sha(output/'policy.zip'), 'new checkpoint changed')
    return output/'policy.zip', report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    require(bool(args.prepare) != bool(args.plan), 'choose prepare or plan')
    plan = prepare(args.prepare) if args.prepare else load_plan(args.plan)
    print(json.dumps({'status': plan['status'], 'training_started': False,
                      'budget': plan['budget'], 'stage_order': plan['stage_order']}, indent=2))


if __name__ == '__main__':
    main()
