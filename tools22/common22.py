"""New bounded low-LR candidate; frozen older attempts are never resumed."""
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
from tools16.contract16 import atomic_json, expected_counts, read_memory  # noqa: E402,F401
from tools18.train_delay18 import HYPERPARAMETERS as BASE_HYPERPARAMETERS  # noqa: E402

INITIAL, TRANSITIONS = 1048576, 262144
OUTPUT = 'evidence/low_lr22/execution_01'
PLAN = 'evidence/low_lr22/plan_01.json'
OLD_PLAN = 'evidence/local_optimization18/plan_01.json'
FORMAT = 'd1-low-lr22-plan-v1'
HYPERPARAMETERS = {**BASE_HYPERPARAMETERS, 'learning_rate': 1e-4}
RESOURCES = {'start_min_effective_bytes': 6 << 30, 'reserve_bytes': 4 << 30,
             'poll_s': 1., 'total_wall_s': 3000}
STAGES = {'train': {'mode': 'zero', 'transitions': TRANSITIONS, 'max_wall_s': 1800},
          'export': {'transitions': 0, 'max_wall_s': 180},
          'cpp': {'transitions': 0, 'max_wall_s': 90},
          'formal': {'transitions': 21600, 'max_wall_s': 300},
          'holdout': {'transitions': 21600, 'max_wall_s': 300}}
SOURCES = ('tools22/common22.py', 'tools22/checked_runtime22.py', 'tools22/train_low_lr22.py',
           'tools22/run_low_lr22.py', 'tests/test_common22.py', 'tests/test_checked_runtime22.py',
           'tests/test_train_low_lr22.py', 'tests/test_run_low_lr22.py', 'docs/low_lr22.md',
           'tools21/closeout_completed18.py', 'tools19/run_cpp_closedloop19.py', 'tools19/cpp_actor19.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON: '+value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    require(isinstance(value, dict), 'JSON object required')
    return value


def check_sources(proof, root=ROOT):
    require(set(proof.get('sources_sha256', {})) == set(SOURCES), 'source22 closure differs')
    for name, digest in proof['sources_sha256'].items():
        path = root/name
        require(not path.is_symlink() and sha(path) == digest, 'source22 changed: '+name)
        blob = subprocess.check_output(['git', 'show', proof['git_commit']+':'+name],
                                       cwd=root, stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit22 sources first: '+name)


def source_identity(root=ROOT):
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip()
    proof = {'git_commit': revision, 'sources_sha256': {name: sha(root/name) for name in SOURCES}}
    check_sources(proof, root)
    return proof


def base_inputs(root=ROOT):
    from tools18 import common18
    old = common18.load_plan(root/OLD_PLAN)
    release, protocol, result, checkpoint = common18.verify_inputs(old)
    require(result['trained_transitions'] == INITIAL and sha(checkpoint) == old['base_checkpoint']['sha256'],
            'only original completed control is accepted')
    require(protocol['hyperparameters'] == BASE_HYPERPARAMETERS, 'baseline hyperparameters differ')
    return old, release, protocol


def fixed(old, root=ROOT):
    return {'format': FORMAT, 'owner_statement': '把这些优化点，在这台电脑能力之内的都完成',
            'scope': 'independent final-only candidate screening; no retry/resume of old attempts',
            'output_root': OUTPUT, 'envs': 4, 'seed': 181, 'config': old['config'],
            'rollout_transitions': 4096, 'base_checkpoint': old['base_checkpoint'],
            'base_protocol': old['base_protocol'], 'baseline_plan_sha256': sha(root/OLD_PLAN),
            'hyperparameters': HYPERPARAMETERS, 'resources': RESOURCES, 'stages': STAGES,
            'stage_order': list(STAGES), 'budget': expected_counts(TRANSITIONS+43200),
            'retry_or_resume_allowed': False, 'partial_checkpoint_eligible': False,
            'preflight': False, 'selection': 'final only; unchanged original36; holdout only after all36 pass',
            'promotion_requires_nominal_9_of_9': True, 'automatic_model_replacement': False,
            'lr_causal_effect_identifiable_against18': False}


def prepare(path=ROOT/PLAN):
    path = Path(path).absolute()
    require(path == ROOT/PLAN and not path.exists() and not path.is_symlink(), 'new fixed22 plan only')
    old, _, _ = base_inputs()
    plan = {**fixed(old), 'source_identity': source_identity()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return plan


def load_plan(path=ROOT/PLAN):
    require(Path(path).absolute() == ROOT/PLAN and not Path(path).is_symlink(), 'fixed22 plan required')
    plan = read_json(path)
    old, _, _ = base_inputs()
    expected = fixed(old)
    require(set(plan) == set(expected)|{'source_identity'}
            and all(plan.get(k) == v for k, v in expected.items()), 'new22 scope changed')
    check_sources(plan['source_identity'])
    return plan


def command(plan_path, stage, output):
    if stage == 'train':
        return [sys.executable, '-B', str(ROOT/'tools22/train_low_lr22.py'),
                '--plan', str(plan_path), '--output', str(output)]
    return [sys.executable, '-B', str(ROOT/'tools22/run_low_lr22.py'),
            '--plan', str(plan_path), '--stage', stage, '--output', str(output)]


def validate_launch(plan, plan_path, stage, output):
    require(stage in STAGES and Path(output) == ROOT/OUTPUT/stage, 'fixed22 child path required')
    journal = read_json(ROOT/OUTPUT/'report.json')
    record = journal.get('stages', {}).get(stage, {})
    require(journal.get('status') == 'RUNNING' and journal.get('active_stage') == stage
            and journal.get('supervisor_pid') == os.getppid() and journal.get('plan_sha256') == sha(plan_path)
            and record.get('status') == 'STARTED' and bool(record.get('nonce'))
            and record['nonce'] == os.environ.get('D1_STAGE22_NONCE')
            and record.get('command') == command(plan_path, stage, Path(output)), 'active22 parent required')
    require(all(journal['stages'].get(name, {}).get('status') == 'COMPLETE'
                for name in list(STAGES)[:list(STAGES).index(stage)]), 'prior22 stage incomplete')
    if stage == 'holdout':
        require(read_json(ROOT/OUTPUT/'formal/report.json').get('all_36_reference_pass') is True,
                'holdout requires unchanged original36 pass')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    value = prepare() if args.prepare else load_plan()
    print(json.dumps({'status': 'PREPARED' if args.prepare else 'VALIDATED', 'budget': value['budget'],
                      'training_started': False, 'physics_steps_executed': 0}, indent=2))
