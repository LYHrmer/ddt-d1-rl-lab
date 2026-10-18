"""Independent two-arm delay comparison25 contracts; cold until explicitly prepared."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools22 import common22 as previous  # noqa: E402

require, sha, atomic_json = previous.require, previous.sha, previous.atomic_json
read_json, read_memory, expected_counts = previous.read_json, previous.read_memory, previous.expected_counts
INITIAL, TRANSITIONS = 1048576, 262144
OUTPUT = 'evidence/low_memory_delay25/execution_01'
PLAN = 'evidence/low_memory_delay25/plan_01.json'
OLD_PLAN = previous.OLD_PLAN
FORMAT = 'd1-delay-comparison25-plan-v1'
EXECUTION_FORMAT = 'd1-delay-comparison25-execution-v1'
ARMS = {'zero': 'zero', 'random': 'uniform_0_3'}
HYPERPARAMETERS = {**previous.HYPERPARAMETERS, 'target_kl': .005, 'n_steps': 4096}
RESOURCES = {'start_min_effective_bytes': 6 << 30, 'reserve_bytes': 4 << 30,
             'poll_s': 1., 'total_wall_s': 5400, 'required_nice': 10,
             'interstage_admission_wait_s': 300, 'admission_poll_s': 5.}
STAGE_ORDER = ('train_zero', 'train_random', 'export_zero', 'cpp_zero', 'formal_zero',
               'export_random', 'cpp_random', 'formal_random', 'holdout_zero', 'holdout_random')
_KINDS = {'train': (TRANSITIONS, 1800), 'export': (0, 180), 'cpp': (0, 90),
          'formal': (21600, 300), 'holdout': (21600, 300)}
STAGES = {key: {'arm': key.rsplit('_', 1)[1], 'kind': key.rsplit('_', 1)[0],
                'mode': ARMS[key.rsplit('_', 1)[1]],
                'transitions': _KINDS[key.rsplit('_', 1)[0]][0],
                'max_wall_s': _KINDS[key.rsplit('_', 1)[0]][1]} for key in STAGE_ORDER}
BUDGET = expected_counts(2*(TRANSITIONS+43200))
SOURCES = ('tools25/common25.py', 'tools25/train_delay25.py', 'tools25/run_delay_comparison25.py',
           'tests/test_common25.py', 'tests/test_train_delay25.py', 'tests/test_run_delay_comparison25.py',
           'docs/low_memory_delay25.md', 'tools25/low_memory_runtime25.py',
           'tests/test_low_memory_runtime25.py', 'training/d1_rl/velocity_policy15.py',
           'training/d1_rl/velocity_labels15.py', 'training/d1_rl/delay_env18.py',
           'tools18/train_delay18.py', 'tools22/common22.py', 'tools22/checked_runtime22.py',
           'tools21/closeout_completed18.py',
           'tools19/run_cpp_closedloop19.py', 'tools19/cpp_actor19.py',
           'tools17/export_completed17.py', 'tools17/evaluate_formal17.py', 'tools17/verify_cpp17.py',
           'tools16/evaluate_comparison16.py', 'tools16/train_comparison16.py',
           'tools16/contract16.py', 'tools16/run_comparison16.py')
BASE_INITIAL_RESULT = 'evidence/local_optimization18/execution_01/train_zero/result.json'
BASE_INITIAL_RESULT_SHA256 = '8e255b2c31f9988cfcee95a249cb1935f504d683becd162d14acc347e361fe43'
BASE_INITIAL_STATE = {'num_timesteps': INITIAL,
                      'policy_state_sha256': 'd73cfee94aeb4615480624ff84261ef62308799a543a12387731fc3bbbefaf81',
                      'optimizer_state_sha256': '26afea6f24a6a242a6651d102ab054eb5c832000dfa0ebe912314ec93bbaedc9',
                      'updates': 1024, 'optimizer_restored': True}
VECTOR_LAYOUT = {'backend': 'DummyVecEnv', 'envs': 1, 'n_steps': 4096,
                 'rollout_transitions': 4096, 'interprocess_rpc': False,
                 'comparable_to24_as_single_variable': False}
RUNTIME_DEPENDENCIES = tuple('.python-deps/stable_baselines3/'+name for name in (
    'common/base_class.py', 'common/on_policy_algorithm.py', 'common/buffers.py',
    'common/save_util.py', 'common/utils.py', 'ppo/ppo.py',
    'common/vec_env/dummy_vec_env.py', 'common/vec_env/base_vec_env.py',
    'common/vec_env/patch_gym.py', 'common/vec_env/util.py'))


def runtime_dependencies(root=ROOT):
    root = Path(root).resolve()
    result = {}
    for name in RUNTIME_DEPENDENCIES:
        path = root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                'unsafe25 runtime dependency: '+name)
        result[name] = sha(path)
    return result


SELECTION = {'checkpoint': 'final_only', 'candidates': ['zero', 'random'],
             'eligibility': 'own_original36_and_holdout36_all_pass',
             'score': 'original36_mean_J', 'direction': 'lower', 'exact_tie': 'zero',
             'scope': 'only_this_run_two_final_checkpoints', 'automatic_model_replacement': False}


def check_sources(proof, root=ROOT):
    root = Path(root).resolve()
    require(isinstance(proof, dict) and isinstance(proof.get('git_commit'), str)
            and re.fullmatch('[0-9a-f]{40}', proof['git_commit']), 'source25 Git identity required')
    sources = proof.get('sources_sha256')
    require(isinstance(sources, dict) and set(sources) == set(SOURCES), 'source25 closure differs')
    for name, digest in sources.items():
        path = root/name
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest)
                and not path.is_symlink() and path.resolve().is_relative_to(root)
                and sha(path) == digest, 'source25 changed: '+name)
        data = subprocess.check_output(['git', 'show', proof['git_commit']+':'+name],
                                       cwd=root, stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(data).hexdigest() == digest, 'commit25 sources first: '+name)


def source_identity(root=ROOT):
    root = Path(root).resolve()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                       text=True, timeout=5).strip()
    proof = {'git_commit': revision, 'sources_sha256': {name: sha(root/name) for name in SOURCES}}
    check_sources(proof, root)
    return proof


def base_inputs(root=ROOT):
    """Reuse only the original-source reader, never a22 plan or execution gate."""
    return previous.base_inputs(root)


def fixed(old, root=ROOT):
    root = Path(root).resolve()
    initial_path = root/BASE_INITIAL_RESULT
    require(not initial_path.is_symlink() and initial_path.resolve().is_relative_to(root)
            and sha(initial_path) == BASE_INITIAL_RESULT_SHA256, 'known baseline initial receipt changed')
    result = read_json(initial_path)
    initial = result.get('initial_state')
    require(result.get('status') == 'COMPLETE' and result.get('stage') == 'train_zero'
            and result.get('mode') == 'zero' and result.get('seed') == 181 and result.get('envs') == 4
            and result.get('base_checkpoint_sha256') == old['base_checkpoint']['sha256']
            and result.get('initial_num_timesteps') == INITIAL and isinstance(initial, dict)
            and all(type(initial.get(k)) is type(v) and initial[k] == v for k, v in BASE_INITIAL_STATE.items()),
            'known original policy/Adam initial state differs')
    return {'format': FORMAT, 'owner_statement': '把这些优化点，在这台电脑能力之内的都完成',
            'scope': 'independent one-env two-arm final-only delay comparison; no retry of24 and no single-variable comparison against24',
            'output_root': OUTPUT, 'arms': dict(ARMS), 'envs': 1, 'seed': 181,
            'vector_layout': dict(VECTOR_LAYOUT),
            'runtime_dependencies_sha256': runtime_dependencies(root),
            'config': old['config'], 'rollout_transitions': 4096,
            'base_checkpoint': copy.deepcopy(old['base_checkpoint']),
            'base_protocol': copy.deepcopy(old['base_protocol']),
            'baseline_plan_sha256': sha(root/OLD_PLAN),
            'base_initial_state': copy.deepcopy(BASE_INITIAL_STATE),
            'base_initial_result': {'path': str(initial_path), 'sha256': BASE_INITIAL_RESULT_SHA256},
            'hyperparameters': copy.deepcopy(HYPERPARAMETERS), 'resources': dict(RESOURCES),
            'stages': copy.deepcopy(STAGES), 'stage_order': list(STAGE_ORDER), 'budget': dict(BUDGET),
            'one_attempt_per_stage': True, 'retry_or_resume_allowed': False,
            'partial_checkpoint_eligible': False, 'warm_start': True, 'reset_num_timesteps': False,
            'preflight': False, 'checkpoint_selection': 'final_only', 'selection': copy.deepcopy(SELECTION),
            'holdout_reset_seeds': [2001, 2002, 2003], 'promotion_requires_nominal_9_of_9': True,
            'automatic_model_replacement': False}


def prepare(path=ROOT/PLAN):
    path = Path(path).absolute()
    require(path == ROOT/PLAN and not path.exists() and not path.is_symlink(), 'new fixed25 plan only')
    old, _, _ = base_inputs()
    plan = {**fixed(old, root=ROOT), 'source_identity': source_identity(root=ROOT)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return plan


def load_plan(path=ROOT/PLAN):
    path = Path(path).absolute()
    require(path == ROOT/PLAN and not path.is_symlink(), 'fixed25 plan required')
    plan = read_json(path)
    old, _, _ = base_inputs()
    expected = fixed(old, root=ROOT)
    require(set(plan) == set(expected)|{'source_identity'}
            and json.dumps({k: plan[k] for k in expected}, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False), 'scope25 changed')
    check_sources(plan['source_identity'], root=ROOT)
    return plan


def split_stage(key):
    require(key in STAGES, 'unknown25 stage')
    return STAGES[key]['arm'], STAGES[key]['kind']


def stage_path(plan, key, root=ROOT):
    arm, kind = split_stage(key)
    require(plan.get('output_root') == OUTPUT, 'fixed25 output required')
    return Path(root)/OUTPUT/arm/kind


def command(plan_path, key, output):
    arm, kind = split_stage(key)
    if kind == 'train':
        return [sys.executable, '-B', str(ROOT/'tools25/train_delay25.py'),
                '--plan', str(plan_path), '--arm', arm, '--output', str(output)]
    return [sys.executable, '-B', str(ROOT/'tools25/run_delay_comparison25.py'),
            '--plan', str(plan_path), '--arm', arm, '--stage', kind, '--output', str(output)]


def validate_launch(plan, plan_path, key, output):
    arm, kind = split_stage(key)
    path, output = Path(plan_path).absolute(), Path(output).absolute()
    require(path == ROOT/PLAN and not path.is_symlink() and plan.get('format') == FORMAT
            and read_json(path) == plan and output == stage_path(plan, key, root=ROOT),
            'fixed25 child path/plan required')
    journal = read_json(ROOT/OUTPUT/'report.json')
    record = journal.get('stages', {}).get(key, {})
    require(journal.get('format') == EXECUTION_FORMAT and journal.get('status') == 'RUNNING'
            and journal.get('active_stage') == key and journal.get('supervisor_pid') == os.getppid()
            and journal.get('plan_sha256') == sha(path) and journal.get('source_identity') == plan['source_identity']
            and record.get('status') == 'STARTED' and isinstance(record.get('nonce'), str) and bool(record['nonce'])
            and record['nonce'] == os.environ.get('D1_STAGE25_NONCE')
            and record.get('command') == command(path, key, output), 'active25 parent required')
    for previous_key in STAGE_ORDER[:STAGE_ORDER.index(key)]:
        prior_arm, prior_kind = split_stage(previous_key)
        status = journal['stages'].get(previous_key, {}).get('status')
        if status == 'COMPLETE':
            continue
        require(prior_kind == 'holdout' and status == 'SKIPPED_NO_QUALIFIED_CANDIDATE'
                and read_json(stage_path(plan, 'formal_'+prior_arm, root=ROOT)/'report.json').get('all_36_reference_pass') is False,
                'prior25 stage incomplete or illegal skip')
    if kind == 'holdout':
        require(read_json(stage_path(plan, 'formal_'+arm, root=ROOT)/'report.json').get('all_36_reference_pass') is True,
                'holdout25 requires own unchanged original36 pass')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        value = prepare()
        status = 'PREPARED'
    else:
        value, status = {'budget': BUDGET}, 'PLAN_ONLY'
    print(json.dumps({'status': status, 'plan': str(ROOT/PLAN), 'output': str(ROOT/OUTPUT),
                      'budget': value['budget'], 'training_started': False, 'physics_steps_executed': 0,
                      'automatic_model_replacement': False}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
