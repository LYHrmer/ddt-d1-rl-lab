#!/usr/bin/env python3
"""Validate contract 10 authorisation and frozen bytes without starting training."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORISATION_FORMAT = 'd1-contract10-budget-authorisation-v1'
CANONICAL_AUTH = 'evidence/career_contract_10/budget_authorisation_02.json'
ORIGINAL_AUTH = 'evidence/career_contract_10/budget_authorisation_01.json'
ORIGINAL_AUTH_SHA256 = '4808ba0cc740a853d263a5fc8d9a17d1ceb4d62714ce22950ea853a5713ffc00'
APPROVAL = '批准合同 10 新预算，内存达标后执行'
CAP = {'transitions': 8421376, 'native': 84231760}
PLANS = {
    'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': 8,
                  'frozen_ledger': 'evidence/career_contract_09/scale09_preflight_01/ledger.jsonl',
                  'output': 'evidence/career_contract_10/scale10_preflight_01'},
    'scale': {'steps': 8388608, 'max_wall': 9000, 'compare_rollouts': 1173,
              'frozen_ledger': 'evidence/career_contract_09/scale09_train_01/ledger.jsonl',
              'output': 'evidence/career_contract_10/scale10_train_01'},
}
PINNED_INPUTS = {
    'tools09/train_d1_ppo09.py': '68ca591970e6c41d9b6cb96b27fd60ef8c1c066f349fb5955f9b97fd74b38198',
    'training/configs/pilot03d.json': '9cd6546e8a8df50c7d578a664003681fc79ddca7fa0a27b80158f0f5b6842bea',
    'tools08/run_scale08.py': 'e15e7cb8a874022071180d4335124a48cd62923602b0e3a1585fa5b9bfa0627b',
    'tools10/supervise_scale10.py': 'f82e3d0255ad8ff7931ad8ddcfe3a6c235b5f50295b6c42e346b1a2a4a599a0b',
    'tools/train_d1_ppo.py': '7dcdd0e8edd5c4458d3d71c74d2d131ce58a4a1bc37c30659ebe701d5937b612',
    'tools/export_d1_ppo.py': '1563d6a96dbd044d844b14a54ab273c754eb8272c0d8ec6ccda774182c6cb27a',
    'tools/evaluate_d1_ppo.py': '47e5b3dd2a806b958c1be5d24a46a54848e6a4981401a07f40b0be3a4107f121',
    'training/d1_rl/factory03.py': '1cec9db3a0875cdb401b0f213f9b0e7a70daa653b1a1d60c3d4ef29d9d406ac1',
    'tools08/replay_training_episodes08.py': '1caeeddcc00816ae8c220f87e2c9e1e1f990920c2ee6e7cfc92e52e6118a8743',
    'tools08/verify_scale08.py': '02061eac2ed28633c1c3c3d1bb573b6e39e7525f591e47c7c90a9b1b69e98277',
}
REQUIRED_INPUTS = tuple(sorted(set(PINNED_INPUTS) | {
    'docs/career_execution_contract_10.md',
    'tools12/check_contract10_static.py', 'tests/test_contract10_static.py',
    'tools12/release_contract10.py', 'tests/test_release_contract10.py',
    'tools12/verify_contract10.py', 'tests/test_verify_contract10.py',
    'tools12/run_contract10.py', 'tests/test_run_contract10.py', 'tests/test_run_contract10_integration.py',
    'tests/test_supervise_scale10.py', 'tools10/export_single_sample10.py', 'tests/test_export_single_sample10.py',
    'tools10/summarize_export10.py', 'tools11/preflight_deployment11.py', 'tests/test_preflight_deployment11.py',
    'tools/run_selftrained03.py', 'runtime03/deployment_variants.py', 'runtime03/build/d1_runtime03',
    'runtime03/schema03d.json', 'runtime04/launch_env04.sh', 'runtime04/ros_env04.sh',
    'runtime04/preflight_ros_imports04.py', 'reports/contract08_summary.json', 'reports/contract09_summary.json',
    'tools09/verify_scale09.py', 'tools04/build_final_ledger04.py', 'tools04/release_pilot04b.py',
    'tools08/release_scale08.py', *(p['frozen_ledger'] for p in PLANS.values()),
}))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_authorisation(auth, path, root):
    require(path == root/CANONICAL_AUTH, 'use the canonical contract 10 authorisation 02 path')
    require(auth.get('format') == AUTHORISATION_FORMAT and type(auth.get('contract_id')) is int
            and auth['contract_id'] == 10 and auth.get('authorised') is True,
            'contract 10 budget is not explicitly authorised')
    require(auth.get('cap') == CAP and all(type(auth['cap'].get(k)) is int for k in CAP)
            and type(auth.get('qualification_native')) is int and auth['qualification_native'] == 18000,
            'authorised transition/native/qualification cap changed')
    require(type(auth.get('start_min_available_ram_bytes')) is int
            and auth['start_min_available_ram_bytes'] == 7*1024**3
            and auth.get('single_attempt') is True and auth.get('retries_allowed') is False
            and auth.get('resume_allowed') is False and auth.get('checkpoint_selection') == 'final_only',
            'authorisation must preserve 7 GiB, a single attempt, no retry/resume and final-only checkpoint')
    source = auth.get('source_record', {})
    require(isinstance(source, dict), 'original approval source_record must be an object')
    original_path = (root/ORIGINAL_AUTH).resolve()
    require(source.get('path') == ORIGINAL_AUTH and source.get('sha256') == ORIGINAL_AUTH_SHA256
            and original_path.is_relative_to(root) and sha(original_path) == ORIGINAL_AUTH_SHA256,
            'original approval 01 bytes/path/hash differ; do not rebaseline them')
    original = json.loads(original_path.read_text())
    require(auth.get('approval_verbatim') == original['approval_verbatim'] == APPROVAL
            and auth.get('contract') == original['contract'] == 'docs/career_execution_contract_10.md'
            and original['cap'] == auth['cap']
            and original['start_min_available_ram_bytes'] == auth['start_min_available_ram_bytes']
            and original['single_attempt'] is True and original['resume_allowed'] is False,
            'canonical authorisation does not preserve the original text approval and limits')
    for field in ('authorised_by', 'source'):
        require(field not in auth or auth[field] == original[field], f'original approval {field} changed')


def check_plan(plan, auth):
    require(plan.get('format') == 'd1-contract10-supervisor-release-v1'
            and type(plan.get('contract_id')) is int and plan['contract_id'] == 10
            and plan.get('status') == 'AUTHORISED_PENDING_FREEZE', 'plan is not authorised pending freeze')
    require(plan.get('entry') == 'tools09/train_d1_ppo09.py'
            and plan.get('config') == 'training/configs/pilot03d.json'
            and plan.get('contract') == 'docs/career_execution_contract_10.md'
            and type(plan.get('seed')) is int and plan['seed'] == 101
            and type(plan.get('envs')) is int and plan['envs'] == 8,
            'the frozen trainer, config, contract, seed or worker count changed')
    require(plan.get('project_cumulative_before') == {'transitions': 18504192, 'native': 185562110}
            and all(type(plan['project_cumulative_before'][key]) is int for key in ('transitions', 'native')),
            'the closed contract 09 cumulative starting ledger changed')
    require(plan.get('plans') == PLANS and all(type(plan['plans'][kind]['steps']) is int
            and type(plan['plans'][kind]['compare_rollouts']) is int for kind in PLANS),
            'steps, wall caps, comparison windows, ledgers or single-attempt output paths changed')
    budget = plan.get('proposed_budget')
    require(budget == dict(CAP, qualification_native=18000, authorised=True)
            and budget['authorised'] is True and all(type(budget[key]) is int
                for key in ('transitions', 'native', 'qualification_native')),
            'proposed budget differs from the approved independent pool')
    transitions = sum(plan['plans'][kind]['steps'] for kind in PLANS)
    require(transitions == budget['transitions'] == auth['cap']['transitions']
            and transitions*10+budget['qualification_native'] == budget['native'] == auth['cap']['native'],
            'the transitions*10 plus qualification native budget equation does not balance')
    resources = plan['resources']
    require(isinstance(resources, dict), 'resources must be an object')
    require(type(resources.get('start_min_available_ram_bytes')) is int
            and resources['start_min_available_ram_bytes'] >= auth['start_min_available_ram_bytes'] >= 7*1024**3,
            'start headroom must retain the authorised minimum 7 GiB')
    for field in ('poll_s', 'interrupt_grace_s', 'terminate_grace_s', 'kill_grace_s', 'process_wall_grace_s'):
        value = resources[field]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f'invalid resource interval: {field}')
    require(0 < resources['poll_s'] <= 2 and resources['kill_grace_s'] > 0
            and resources['process_wall_grace_s'] == 180,
            'resource polling/kill grace or the contracted 180-second process allowance changed')


def check_inputs(plan, root):
    frozen = plan.get('frozen_inputs')
    require(isinstance(frozen, dict) and set(REQUIRED_INPUTS) <= frozen.keys(),
            'required frozen checker/gates/runner/deployment dependencies are missing')
    for name, digest in frozen.items():
        require(isinstance(name, str) and bool(name) and not Path(name).is_absolute()
                and '..' not in Path(name).parts and '\\' not in name and Path(name).as_posix() == name,
                f'frozen input path is not a safe relative path: {name}')
        target = (root/name).resolve()
        require(target.is_relative_to(root) and target.is_file(), f'frozen input is missing/outside repository: {name}')
        require(isinstance(digest, str) and len(digest) == 64
                and all(ch in '0123456789abcdef' for ch in digest) and sha(target) == digest,
                f'frozen input bytes changed: {name}')
    require(all(frozen[name] == digest for name, digest in PINNED_INPUTS.items()),
            'a historical scientific source was rebaselined')
    require(frozen['tools12/check_contract10_static.py'] == sha(__file__)
            and plan.get('supervisor_sha256') == frozen['tools10/supervise_scale10.py'],
            'checker or supervisor identity does not match the frozen plan')
    # Read only the pinned arithmetic assignment; never import the trainer or simulator.
    tree = ast.parse((root/'tools10/supervise_scale10.py').read_text())
    expression = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'RESERVE' for target in node.targets))
    def integer(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
            left, right = integer(node.left), integer(node.right)
            return left*right if isinstance(node.op, ast.Mult) else left**right
        raise ValueError('the supervisor reserve is no longer a constant integer expression')
    reserve = integer(expression)
    require(reserve == 3*1024**3, 'the runtime RAM reserve must remain 3 GiB')
    return reserve


def check(plan_path, authorisation_path, output, root=ROOT):
    root, plan_path, authorisation_path, output = map(lambda p: Path(p).resolve(),
                                                     (root, plan_path, authorisation_path, output))
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'format': 'd1-contract10-static-v1', 'status': 'FAIL', 'checks': [], 'error': None,
               'physics_steps_executed': 0, 'training_started': False,
               'plan_sha256': None, 'budget_authorisation_sha256': None,
               'frozen_inputs': None, 'tool_sha256': sha(__file__)}
    phase = 'read_inputs'
    try:
        plan = json.loads(plan_path.read_text())
        auth = json.loads(authorisation_path.read_text())
        require(isinstance(plan, dict) and isinstance(auth, dict), 'plan and authorisation must be JSON objects')
        receipt.update(plan_sha256=sha(plan_path), budget_authorisation_sha256=sha(authorisation_path),
                       frozen_inputs=plan.get('frozen_inputs'))
        phase = 'budget_authorisation'
        check_authorisation(auth, authorisation_path, root)
        receipt['checks'].append({'check': phase, 'status': 'PASS'})
        receipt['budget_source_record'] = auth['source_record']
        phase = 'fixed_plan_and_budget'
        check_plan(plan, auth)
        receipt['checks'].append({'check': phase, 'status': 'PASS'})
        phase = 'frozen_inputs_and_runtime_reserve'
        receipt['supervisor_ram_reserve_bytes'] = check_inputs(plan, root)
        receipt['checks'].append({'check': phase, 'status': 'PASS'})
        receipt['status'] = 'PASS'
    except (OSError, ValueError, TypeError, KeyError, StopIteration, SyntaxError) as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        receipt['checks'].append({'check': phase, 'status': 'FAIL', 'detail': str(exc)})
    (output/'report.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--budget-authorisation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = check(args.plan, args.budget_authorisation, args.output)
    print(json.dumps({'status': receipt['status'], 'physics_steps_executed': 0}))
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
