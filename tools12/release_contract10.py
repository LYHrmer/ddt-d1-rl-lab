#!/usr/bin/env python3
"""Freeze the approved contract 10 once; read-only checks execute no physics.

The static receipt binds the exact plan and authorisation bytes. The plan binds every
source dependency; this tool refuses changed bytes, another budget, or a prior attempt.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FORMAT = 'd1-contract10-supervisor-release-v1'
AUTHORISATION_FORMAT = 'd1-contract10-budget-authorisation-v1'
RELEASE_PATH = 'evidence/career_contract_10/scale10_release_01.json'
STATIC_TOOL = 'tools12/check_contract10_static.py'
CAP = {'transitions': 8421376, 'native': 84231760}
REQUIRED_INPUTS = (
    'tools12/release_contract10.py', 'tools12/verify_contract10.py',
    'tests/test_release_contract10.py', 'tests/test_verify_contract10.py',
    'tools10/supervise_scale10.py', 'tests/test_supervise_scale10.py',
    STATIC_TOOL, 'tests/test_contract10_static.py',
)
PLANS = {
    'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': 8,
                  'frozen_ledger': 'evidence/career_contract_09/scale09_preflight_01/ledger.jsonl',
                  'output': 'evidence/career_contract_10/scale10_preflight_01'},
    'scale': {'steps': 8388608, 'max_wall': 9000, 'compare_rollouts': 1173,
              'frozen_ledger': 'evidence/career_contract_09/scale09_train_01/ledger.jsonl',
              'output': 'evidence/career_contract_10/scale10_train_01'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def relative_file(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'receipt must be a file inside this repository')
    return {'path': str(path.relative_to(root)), 'sha256': sha(path)}


def check_frozen_inputs(plan, root):
    frozen = plan.get('frozen_inputs')
    require(isinstance(frozen, dict) and bool(frozen), 'missing frozen inputs')
    required = set(REQUIRED_INPUTS) | {
        plan['entry'], plan['config'], plan['contract'], 'tools08/run_scale08.py',
        *(plan['plans'][kind]['frozen_ledger'] for kind in PLANS),
    }
    require(required <= frozen.keys(), 'missing required frozen input')
    for name, digest in frozen.items():
        target = (root/name).resolve()
        require(not Path(name).is_absolute() and target.is_relative_to(root)
                and target.is_file() and sha(target) == digest,
                f'frozen input changed, missing or outside repository: {name}')
    require(frozen['tools12/release_contract10.py'] == sha(__file__), 'release tool identity mismatch')
    require(plan.get('supervisor_sha256') == frozen['tools10/supervise_scale10.py'],
            'supervisor identity mismatch')


def check_authorisation(authorisation):
    require(authorisation.get('format') == AUTHORISATION_FORMAT
            and type(authorisation.get('contract_id')) is int and authorisation['contract_id'] == 10
            and authorisation.get('authorised') is True
            and authorisation.get('cap') == CAP
            and all(type(authorisation['cap'].get(k)) is int for k in CAP)
            and type(authorisation.get('qualification_native')) is int
            and authorisation['qualification_native'] == 18000
            and type(authorisation.get('start_min_available_ram_bytes')) is int
            and authorisation['start_min_available_ram_bytes'] == 7*1024**3
            and authorisation.get('retries_allowed') is False
            and authorisation.get('resume_allowed') is False
            and authorisation.get('checkpoint_selection') == 'final_only',
            'a matching explicit contract 10 budget authorisation is required')


def build(plan_path, authorisation_path, static_path, root=ROOT):
    root = Path(root).resolve()
    plan = json.loads(Path(plan_path).read_text())
    authorisation = json.loads(Path(authorisation_path).read_text())
    static = json.loads(Path(static_path).read_text())
    check_authorisation(authorisation)
    require(plan.get('format') == RELEASE_FORMAT and plan.get('contract_id') == 10
            and plan.get('status') == 'AUTHORISED_PENDING_FREEZE', 'plan is not authorised pending freeze')
    require(plan.get('proposed_budget') == dict(CAP, qualification_native=18000, authorised=True),
            'plan budget differs from authorisation')
    require(plan.get('entry') == 'tools09/train_d1_ppo09.py'
            and plan.get('config') == 'training/configs/pilot03d.json'
            and plan.get('contract') == 'docs/career_execution_contract_10.md'
            and type(plan.get('seed')) is int and plan['seed'] == 101
            and type(plan.get('envs')) is int and plan['envs'] == 8
            and plan.get('plans') == PLANS, 'contract 10 scientific plan changed')
    require(type(plan['resources'].get('start_min_available_ram_bytes')) is int
            and plan['resources']['start_min_available_ram_bytes'] == 7*1024**3,
            'contract 10 requires the approved 7 GiB start threshold')
    check_frozen_inputs(plan, root)
    require(static.get('status') == 'PASS' and type(static.get('physics_steps_executed')) is int
            and static['physics_steps_executed'] == 0 and static.get('training_started') is False
            and static.get('plan_sha256') == sha(plan_path)
            and static.get('budget_authorisation_sha256') == sha(authorisation_path)
            and static.get('tool_sha256') == plan['frozen_inputs'][STATIC_TOOL]
            and static.get('frozen_inputs') == plan['frozen_inputs'],
            'static checks must pass without physics and bind this exact plan, authorisation and frozen inputs')
    for run in PLANS.values():
        output = root/run['output']
        require(not output.exists() and not output.with_name(output.name+'_supervisor').exists(),
                'contract 10 already has a training attempt; a new release cannot retry it')
    receipt = copy.deepcopy(plan)
    receipt.update(status='FROZEN_BEFORE_TRAINING',
                   created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   plan_sha256=sha(plan_path), budget_authorisation=relative_file(root, authorisation_path),
                   static_report=relative_file(root, static_path), release_tool_sha256=sha(__file__),
                   physics_steps_executed_by_this_receipt=0)
    return receipt


def freeze(plan, authorisation, static, output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output == root/RELEASE_PATH, f'the single contract 10 release path is {RELEASE_PATH}')
    require(not output.exists(), 'release already exists; refusing to overwrite')
    receipt = build(plan, authorisation, static, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--budget-authorisation', type=Path, required=True)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = freeze(args.plan, args.budget_authorisation, args.static_report, args.output)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output),
                      'frozen_inputs': len(receipt['frozen_inputs'])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
