#!/usr/bin/env python3
"""Freeze contract 08 before any of its training runs.

Writes the SHA-256 of every input the scale run depends on, the pre-registered
predictions, the gates and the budget. The launcher refuses to start unless its own
SHA equals the one written here. Executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import run_scale08 as launcher  # noqa: E402

CONTRACT = 'docs/career_execution_contract_08.md'
FROZEN_INPUTS = (
    'training/configs/pilot03d.json', 'tools/train_d1_ppo.py', 'tools/export_d1_ppo.py',
    'tools/evaluate_d1_ppo.py', 'training/d1_rl/factory03.py', launcher.ENTRY,
    launcher.FROZEN_LEDGER, CONTRACT, 'tools08/check_scale08_static.py', 'tests/test_run_scale08.py',
)
GATES = {
    'E0': {'what': 'preflight of the new entry, first 8 rollouts equal to the frozen 03d ledger',
           'relative_tolerance': launcher.RELATIVE_TOLERANCE, 'counters': 'exact',
           'if_not_met': 'do not start the scale run; diagnose with the frozen trainer, then end the contract'},
    'E1': {'what': 'scale run, first 256 rollouts equal to the frozen 03d ledger, watched live',
           'relative_tolerance': launcher.RELATIVE_TOLERANCE, 'counters': 'exact',
           'if_not_met': 'interrupt the trainer; no export, no evaluation; contract ends'},
}
PREDICTIONS = [
    {'id': 'P1', 'kind': 'primary', 'metric': 'tasks passed by the final 8x policy', 'threshold': 2,
     'direction': 'at_least', 'baseline_03d_1x': 1,
     'refuted_if': 'at most 1: eight times the training does not improve qualification'},
    {'id': 'P2', 'kind': 'secondary', 'metric': 'forward_stop stop displacement m', 'threshold': .05,
     'direction': 'below', 'baseline_03d_1x': .081},
    {'id': 'P3', 'kind': 'secondary', 'metric': 'turn_stop stable wz MAE rad/s', 'threshold': .08,
     'direction': 'below', 'baseline_03d_1x': .103},
    {'id': 'P4', 'kind': 'descriptive', 'metric': 'relative change of training return over the last 25 percent',
     'baseline_03d_1x': .103, 'note': 'reported, not judged; above +5 percent means 8x has not converged either'},
    {'id': 'P5', 'kind': 'no_regression', 'metric': 'reverse_stop still passes all applicable gates',
     'baseline_03d_1x': True},
]
BUDGET = {'pool': 'contract 08', 'borrowed_from': 'NONE',
          'preflight': {'transitions': 32768, 'native': 327680},
          'preflight_diagnosis_only_if_E0_fails': {'transitions': 32768, 'native': 327680},
          'scale_training': {'transitions': 8388608, 'native': 83886080, 'max_wall_s': 9000},
          'python_qualification_native': 18000, 'onnx_golden_native': 0,
          'cap': {'transitions': 8454144, 'native': 84559440},
          'project_cumulative_before': {'transitions': 5242880, 'native': 52932140},
          'retries_allowed': False, 'resume_allowed': False, 'checkpoint_selection': 'final_only'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(static_report_path, root=ROOT):
    static_report_path = Path(static_report_path)
    static = json.loads(static_report_path.read_text())
    if static.get('status') != 'PASS':
        raise SystemExit('static checks are not PASS; contract 08 cannot be frozen')
    if static.get('native_integration_steps_executed') != 0 or static.get('training_started') is not False:
        raise SystemExit('the static checks ran physics or started training')
    missing = [p for p in FROZEN_INPUTS if not (Path(root)/p).is_file()]
    if missing:
        raise SystemExit(f'frozen inputs missing: {missing}')
    if static.get('new_entry_sha256') != sha(Path(root)/launcher.ENTRY):
        raise SystemExit('the static checks examined a different new entry than the current one')
    if static.get('launcher_sha256') != sha(launcher.__file__):
        raise SystemExit('the static checks examined a different launcher than the current one')
    return {
        'format': 'd1-contract08-release-v1', 'status': 'FROZEN_BEFORE_TRAINING',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'contract': CONTRACT, 'single_change': 'training steps 1,048,576 -> 8,388,608 on the frozen 03d configuration',
        'frozen_inputs': {p: sha(Path(root)/p) for p in FROZEN_INPUTS},
        'launcher_sha256': sha(launcher.__file__), 'release_tool_sha256': sha(__file__),
        'static_report': {'path': str(static_report_path.resolve().relative_to(Path(root).resolve())),
                          'sha256': sha(static_report_path), 'checks': len(static['checks'])},
        'plans': launcher.PLANS, 'seed': launcher.SEED, 'envs': launcher.ENVS,
        'gates': GATES, 'predictions_preregistered': PREDICTIONS, 'budget': BUDGET,
        'not_authorised_by_any_outcome': ['formal three arms x three seeds', 'a second scale run',
                                          'calling a single task pass a qualification pass'],
        'physics_steps_executed_by_this_receipt': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite an existing release receipt: {args.output}')
    receipt = build(args.static_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': receipt['status'],
                      'frozen_inputs': len(receipt['frozen_inputs']), 'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
