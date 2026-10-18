"""Fixed-base KL comparison: cold provenance, scope and completed-run checks."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools26 import common26 as baseline  # noqa: E402

require, sha, atomic_json = baseline.require, baseline.sha, baseline.atomic_json
read_json, expected_counts = baseline.read_json, baseline.expected_counts
base_inputs = baseline.base_inputs
BASE_UPDATES, BASE_SERIALIZED = baseline.BASE_UPDATES, baseline.BASE_SERIALIZED
BASE_ANCHORS, BASE_ROOT = baseline.BASE_ANCHORS, baseline.BASE_ROOT
REFERENCE_REPORTS = copy.deepcopy(baseline.REFERENCE_REPORTS)
HYPERPARAMETERS = copy.deepcopy(baseline.HYPERPARAMETERS)
VECTOR_LAYOUT = copy.deepcopy(baseline.VECTOR_LAYOUT)
INITIAL, TRANSITIONS, SEED = 1310720, 32768, 183
MODE = 'uniform_0_3'
ARMS = {'control': 0.0, 'anchor': 1.0}
OUTPUT = 'evidence/policy_anchor29/execution_01'
PLAN = 'evidence/policy_anchor29/plan_01.json'
ADMISSIONS = 'evidence/policy_anchor29/admissions'
FORMAT = 'd1-anchor29-plan-v1'
EXECUTION_FORMAT = 'd1-anchor29-execution-v1'
UNIT = 'd1-policy-anchor29-owned01-20261018-01.service'
OWNED_SCOPE = (f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/'
               f'app.slice/{UNIT}')
RESOURCE_POLICY = {'unit': UNIT, 'owned_scope': OWNED_SCOPE,
                   'memory_high_bytes': 2 << 30, 'memory_max_bytes': 3 << 30,
                   'memory_swap_max_bytes': 0, 'memory_oom_group': 1,
                   'oom_policy': 'kill', 'minimum_nice': 10,
                   'covers': 'parent, all stage processes and descendants'}
RESOURCES = {**baseline.RESOURCES, 'start_min_effective_bytes': 4 << 30,
             'reserve_bytes': 2 << 30, 'poll_s': 1, 'total_wall_s': 1800,
             'interstage_admission_wait_s': 300, 'admission_poll_s': 5}
STAGE_ORDER = ('train_control', 'train_anchor', 'export_control', 'cpp_control',
               'formal_control', 'export_anchor', 'cpp_anchor', 'formal_anchor')
KINDS = {'train': (TRANSITIONS, 600), 'export': (0, 180),
         'cpp': (0, 90), 'formal': (21600, 300)}
STAGES = {key: {'arm': key.rsplit('_', 1)[1], 'kind': key.rsplit('_', 1)[0],
               'mode': MODE, 'anchor_coef': ARMS[key.rsplit('_', 1)[1]],
               'transitions': KINDS[key.rsplit('_', 1)[0]][0],
               'max_wall_s': KINDS[key.rsplit('_', 1)[0]][1]}
          for key in STAGE_ORDER}
BUDGET = expected_counts(2 * TRANSITIONS + 2 * 21600)
ANCHOR = {'direction': 'KL(fixed_base || current)', 'reduction': 'sum_action_mean_batch',
          'fixed_base_checkpoint_sha256': BASE_ANCHORS['execution_01/random/train/policy.zip'],
          'teacher_frozen': True, 'teacher_requires_grad': False,
          'teacher_actor_samples_max': TRANSITIONS, 'teacher_batch_calls_max': 8,
          'teacher_samples_per_batch': 4096, 'control_teacher_calls': 0,
          'current_distribution': 'reuse same evaluate_actions forward',
          'reward_yaw_weight': 2.0, 'reward_changed': False,
          'privileged_actor_input_added': False}
SELECTION = {'checkpoint': 'final_only', 'candidates': list(ARMS),
             'eligibility': 'own_original36_all_thresholds_and_full_horizons',
             'maximum_holdout_arms': 0, 'uses_holdout_for_selection': False,
             'qualification_granted_by_this_run': False, 'automatic_model_replacement': False}
SOURCES = tuple(dict.fromkeys((*baseline.SOURCES,
    'tools27/resource_guard27.py', 'tests/test_resource_guard27.py',
    'tools29/common29.py', 'tools29/anchor29.py', 'tools29/ppo_anchor29.py',
    'tools29/train_anchor29.py', 'tools29/run_anchor29.py',
    'tests/test_common29.py', 'tests/test_anchor29.py', 'tests/test_train_anchor29.py',
    'tests/test_policy_anchor29_parent.py', 'docs/policy_anchor29.md')))


def read_memory():
    from tools27.resource_guard27 import read_memory as guarded
    return guarded(OWNED_SCOPE)


def anchor_change(arm):
    require(arm in ARMS, 'unknown29 arm')
    return {**ANCHOR, 'coefficient': ARMS[arm], 'teacher_policy_used': ARMS[arm] != 0.0}


def check_sources(proof, root=ROOT):
    root = Path(root).resolve()
    require(isinstance(proof, dict) and isinstance(proof.get('git_commit'), str)
            and re.fullmatch('[0-9a-f]{40}', proof['git_commit']) is not None,
            'source29 Git identity required')
    require(set(proof.get('sources_sha256', {})) == set(SOURCES), 'source29 closure differs')
    for name, digest in proof['sources_sha256'].items():
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path) == digest,
                'source29 changed: ' + name)
        data = subprocess.check_output(['git', 'show', proof['git_commit'] + ':' + name],
                                       cwd=root, stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(data).hexdigest() == digest, 'commit29 source first: ' + name)


def source_identity(root=ROOT):
    root = Path(root).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip()
    proof = {'git_commit': commit, 'sources_sha256': {name: sha(root / name) for name in SOURCES}}
    check_sources(proof, root)
    return proof


def fixed(base):
    return {'format': FORMAT,
            'owner_statement': '本机低并行有界优化；发挥模型作用，以实际跟踪和停车结果验证',
            'scope': 'same completed25 baseline and Adam; only fixed-base KL coefficient differs',
            'output_root': OUTPUT, 'arms': dict(ARMS), 'training_delay_mode': MODE,
            'seed': SEED, 'envs': 1, 'config': base['config'],
            'vector_layout': dict(VECTOR_LAYOUT), 'rollout_transitions': 4096,
            **{key: copy.deepcopy(base[key]) for key in
               ('base_checkpoint', 'base_protocol', 'baseline_provenance', 'runtime_dependencies_sha256')},
            'hyperparameters': copy.deepcopy(HYPERPARAMETERS), 'resources': dict(RESOURCES),
            'resource_policy': dict(RESOURCE_POLICY), 'anchor': copy.deepcopy(ANCHOR),
            'stages': copy.deepcopy(STAGES), 'stage_order': list(STAGE_ORDER), 'budget': dict(BUDGET),
            'one_attempt_per_stage': True, 'retry_or_resume_allowed': False,
            'partial_checkpoint_eligible': False, 'warm_start': True, 'reset_num_timesteps': False,
            'preflight': False, 'checkpoint_selection': 'final_only',
            'selection': copy.deepcopy(SELECTION), 'reference_reports': copy.deepcopy(REFERENCE_REPORTS),
            'holdout_reset_seeds': [], 'automatic_model_replacement': False}


def prepare(path=ROOT / PLAN):
    path = Path(path).absolute()
    require(path == ROOT / PLAN and not path.exists() and not path.is_symlink(), 'new fixed29 plan only')
    require(not (ROOT / OUTPUT).exists(), '29 execution already exists')
    base, _, _ = base_inputs()
    value = {**fixed(base), 'source_identity': source_identity()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return value


def load_plan(path=ROOT / PLAN):
    path = Path(path).absolute()
    require(path == ROOT / PLAN and not path.is_symlink(), 'fixed29 plan required')
    value = read_json(path)
    base, _, _ = base_inputs()
    expected = fixed(base)
    require(set(value) == set(expected) | {'source_identity'} and
            json.dumps({k: value[k] for k in expected}, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False), 'scope29 changed')
    check_sources(value['source_identity'])
    return value


def split_stage(key):
    require(key in STAGES, 'unknown29 stage')
    return STAGES[key]['arm'], STAGES[key]['kind']


def stage_path(plan, key, root=None):
    arm, kind = split_stage(key)
    require(plan.get('output_root') == OUTPUT, 'fixed29 output required')
    return Path(ROOT if root is None else root) / OUTPUT / arm / kind


def command(plan_path, key, output):
    arm, kind = split_stage(key)
    script = 'train_anchor29.py' if kind == 'train' else 'run_anchor29.py'
    value = [sys.executable, '-B', str(ROOT / 'tools29' / script), '--plan', str(plan_path),
             '--arm', arm, '--output', str(output)]
    return value if kind == 'train' else value + ['--stage', kind]


def validate_launch(plan, plan_path, key, output):
    split_stage(key)
    path, output = Path(plan_path).absolute(), Path(output).absolute()
    require(path == ROOT / PLAN and not path.is_symlink() and read_json(path) == plan
            and plan.get('format') == FORMAT and output == stage_path(plan, key),
            'fixed29 child path/plan required')
    journal = read_json(ROOT / OUTPUT / 'report.json')
    record = journal.get('stages', {}).get(key, {})
    require(journal.get('format') == EXECUTION_FORMAT and journal.get('status') == 'RUNNING'
            and journal.get('active_stage') == key and journal.get('supervisor_pid') == os.getppid()
            and journal.get('plan_sha256') == sha(path)
            and journal.get('source_identity') == plan['source_identity']
            and record.get('status') == 'STARTED'
            and isinstance(record.get('nonce'), str) and bool(record['nonce'])
            and record['nonce'] == os.environ.get('D1_STAGE29_NONCE')
            and record.get('command') == command(path, key, output), 'active29 parent required')
    require(all(journal['stages'].get(prior, {}).get('status') == 'COMPLETE'
                for prior in STAGE_ORDER[:STAGE_ORDER.index(key)]), 'prior29 stage incomplete')


def valid_first_rollout29(value):
    if not isinstance(value, dict) or value.get('format') != 'd1-anchor29-first-rollout-v1':
        return False
    # Only the schema label changes; the six actual pre-update arrays keep their exact checks.
    return baseline.valid_first_rollout26({**value, 'format': 'd1-yaw-reward26-first-rollout-v1'})


def select_candidate(formals):
    """Report development eligibility; this bounded mechanism test has no holdout."""
    require(set(formals) == set(ARMS), 'both29 developer reports required')
    matrix = {(s, d, z) for s in ('forward_stop', 'reverse_stop', 'turn_stop')
              for d in range(4) for z in (1001, 1002, 1003)}
    means, eligible = {}, []
    for arm, value in formals.items():
        rows = value.get('cases', [])
        require(value.get('status') == 'COMPLETE' and value.get('error') is None
                and value.get('cleanup_complete') is True and len(rows) == 36
                and {(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in rows} == matrix,
                'complete29 developer matrix required')
        require(all(type(r['J']) in (int, float) and math.isfinite(r['J']) for r in rows),
                'finite29 developer J required')
        passed = all(r.get('reference_thresholds_met') is True and r.get('failure') is False
                     and r.get('done_reason') == 'HORIZON_12S'
                     and type(r.get('policy_transitions')) is int and r['policy_transitions'] == 600
                     for r in rows)
        require(value.get('all_36_reference_pass') is passed, 'developer29 pass flag differs')
        means[arm] = sum(r['J'] for r in rows) / 36
        if passed:
            eligible.append(arm)
    return {'eligible_arms': eligible, 'original36_mean_J': means, 'selected_arm': None,
            'uses_holdout': False, 'qualification_granted': False, 'automatic_model_replacement': False}


def check_completed_training(plan, plan_path, arm):
    from tools16.train_comparison16 import complete_counts
    from tools18.train_delay18 import valid_delay_state
    from tools29.anchor29 import (
        checked_worker_counts29,
        initial_identity_valid29,
        valid_anchor_receipt29,
        valid_reward_state29,
    )
    require(arm in ARMS, 'unknown29 completed arm')
    directory = stage_path(plan, 'train_' + arm)
    result, protocol = (read_json(directory / name) for name in ('result.json', 'protocol.json'))
    checkpoint = directory / 'policy.zip'
    identity = {'stage': 'train_' + arm, 'arm': arm, 'mode': MODE, 'seed': SEED, 'envs': 1,
                'plan_sha256': sha(plan_path), 'source_identity': plan['source_identity'],
                'hyperparameters': HYPERPARAMETERS, 'anchor_coef': ARMS[arm],
                'anchor_change': anchor_change(arm), 'teacher_policy_used': ARMS[arm] != 0.0}
    require(all(json.dumps(row.get(k), sort_keys=True, allow_nan=False)
                == json.dumps(v, sort_keys=True, allow_nan=False)
                for row in (result, protocol) for k, v in identity.items()),
            'training29 identity/settings differ')
    require(result.get('format') == 'd1-anchor29-training-result-v1'
            and protocol.get('format') == 'd1-anchor29-training-protocol-v1'
            and result.get('status') == 'COMPLETE' and result.get('error') is None
            and result.get('cleanup_complete') is True and result.get('inputs_unchanged') is True
            and result.get('eligible_for_evaluation') is True
            and type(result.get('aux_coef')) is float and result['aux_coef'] == 0.0
            and type(result.get('initial_num_timesteps')) is int and result['initial_num_timesteps'] == INITIAL
            and type(result.get('new_trained_transitions')) is int and result['new_trained_transitions'] == TRANSITIONS
            and type(result.get('final_num_timesteps')) is int and result['final_num_timesteps'] == INITIAL + TRANSITIONS
            and result.get('protocol_sha256') == sha(directory / 'protocol.json')
            and not checkpoint.is_symlink() and result.get('checkpoint_sha256') == sha(checkpoint),
            'incomplete/partial29 training')
    workers = result.get('per_worker_counts')
    require(workers == result.get('worker_counts')
            and checked_worker_counts29(workers) == checked_worker_counts29([result.get('actual_counts')])
            and checked_worker_counts29([result.get('actual_counts')])
                == checked_worker_counts29([result.get('known_actual_counts')])
            and result.get('actual_counts') == result.get('known_actual_counts')
            and complete_counts(workers, 1, TRANSITIONS)
            and valid_delay_state(result.get('delay_state18'), 1, MODE),
            'incomplete29 counter/delay proof')
    require(initial_identity_valid29(result.get('initial_state'), plan)
            and protocol.get('initial_state') == result['initial_state']
            and protocol.get('base_checkpoint') == plan['base_checkpoint']
            and protocol.get('base_protocol') == plan['base_protocol']
            and protocol.get('planned_new_transitions') == TRANSITIONS
            and protocol.get('planned_native_steps') == TRANSITIONS * 10
            and protocol.get('warm_start') is True and protocol.get('reset_num_timesteps') is False
            and protocol.get('checkpoint_selection') == result.get('checkpoint_selection') == 'final_only',
            'initial29 base/Adam/protocol incomplete')
    guard = result.get('single_process_guard29', {})
    require(guard.get('backend') == 'DummyVecEnv' and guard.get('dirty') is False
            and guard.get('closed') is True and guard.get('interprocess_rpc') is False
            and type(guard.get('environment_subprocesses')) is int and guard['environment_subprocesses'] == 0
            and guard.get('waiting') is False and guard.get('inflight') is None
            and guard.get('construction_complete') is True
            and all(type(guard.get(k)) is int and guard[k] > 0
                    for k in ('metadata_attempts', 'metadata_returns', 'metadata_validated'))
            and guard['metadata_attempts'] == guard['metadata_returns'] == guard['metadata_validated'],
            'dirty/incomplete29 guard')
    require(valid_reward_state29(result.get('reward_state29'), TRANSITIONS),
            'incomplete29 unchanged reward provenance')
    require(valid_anchor_receipt29(result.get('anchor29'), ARMS[arm], completed=True),
            'incomplete29 teacher/cache/update proof')
    first = result.get('first_rollout_nonreward')
    require(valid_first_rollout29(first)
            and result.get('first_rollout_report_sha256') == sha(directory / 'first_rollout29.json')
            and read_json(directory / 'first_rollout29.json') == first, 'incomplete29 first-rollout provenance')
    _, release, original_protocol = base_inputs()
    require(result.get('versions') == protocol.get('versions') == original_protocol['versions']
            and all(row.get('vector_layout') == plan['vector_layout']
                    and row.get('runtime_dependencies_sha256') == plan['runtime_dependencies_sha256']
                    for row in (result, protocol)), 'runtime29 identity differs')
    bound = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
             'training_protocol_sha256': sha(directory / 'protocol.json'),
             'training_result_sha256': sha(directory / 'result.json')}
    return {'checkpoint': checkpoint, 'protocol': protocol, 'result': result, 'old_release': release,
            'identity': {'role': 'completed_anchor29_' + arm + '_final_only', 'training_identity': bound}}
