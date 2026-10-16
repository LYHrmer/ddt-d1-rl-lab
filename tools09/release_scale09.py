#!/usr/bin/env python3
"""Freeze contract 09 before any of its training runs. Executes no physics.

The contract 08 launcher verifies a release against its own SHA, so this receipt records
that launcher's SHA as `launcher_sha256` and freezes the contract 09 wrapper, entry,
checks and contract text alongside the frozen 03d inputs.
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
sys.path.insert(0, str(ROOT/'tools09'))

import run_scale08 as base  # noqa: E402
import run_scale09 as launcher  # noqa: E402

CONTRACT = 'docs/career_execution_contract_09.md'
FROZEN_INPUTS = (
    'training/configs/pilot03d.json', 'tools/train_d1_ppo.py', 'tools/export_d1_ppo.py',
    'tools/evaluate_d1_ppo.py', 'training/d1_rl/factory03.py', base.FROZEN_LEDGER,
    'tools08/run_scale08.py', 'tests/test_run_scale08.py', 'tools08/verify_scale08.py',
    'tools08/replay_training_episodes08.py', launcher.ENTRY, 'tools09/run_scale09.py',
    'tools09/check_scale09_static.py', 'tests/test_scale09.py', CONTRACT,
)
BASELINE_08 = {'last_quarter_mean_std': .676, 'forward_drive_early_ends_last_quarter': 43,
               'tasks_passed': 1, 'forward_fell_at_s': 9.70, 'drive_speed_forward_backward_m_s': [.062, .281]}
PREDICTIONS = [
    {'id': 'Q1', 'kind': 'mechanism', 'metric': 'mean action std over the last quarter of training',
     'threshold': .45, 'direction': 'below', 'baseline_08': BASELINE_08['last_quarter_mean_std']},
    {'id': 'Q2', 'kind': 'behaviour', 'metric': 'early ends inside forward-drive segments, last quarter of training',
     'threshold': 15, 'direction': 'at_most', 'baseline_08': BASELINE_08['forward_drive_early_ends_last_quarter']},
    {'id': 'Q3', 'kind': 'qualification', 'metric': 'tasks passed by the final policy', 'threshold': 2,
     'direction': 'at_least', 'baseline_08': BASELINE_08['tasks_passed']},
    {'id': 'Q4', 'kind': 'qualification', 'metric': 'forward_stop completes 12 s without a fall', 'baseline_08': False},
    {'id': 'Q5', 'kind': 'descriptive', 'metric': 'drive-phase last-second speed forward / backward, reported only',
     'baseline_08': BASELINE_08['drive_speed_forward_backward_m_s']},
]
INTERPRETATION = {'Q1 met, Q2 met': 'supports: the entropy bonus raises the noise and the noise causes the forward decay (one seed)',
                  'Q1 met, Q2 not met': 'the noise was held down and forward still decayed: the noise is not the cause',
                  'Q1 not met': 'without the entropy bonus the std still rises: the bonus is not the main driver'}
BUDGET = {'pool': 'contract 09', 'borrowed_from': 'NONE',
          'preflight': {'transitions': 32768, 'native': 327680},
          'scale_training': {'transitions': 8388608, 'native': 83886080, 'max_wall_s': 9000},
          'python_qualification_native': 18000, 'onnx_golden_native': 0,
          'cap': {'transitions': 8421376, 'native': 84231760},
          'project_cumulative_before': {'transitions': 13664256, 'native': 137162750},
          'retries_allowed': False, 'resume_allowed': False, 'checkpoint_selection': 'final_only'}
EXPORT_RULE = ('run the frozen exporter; if its absolute 1e-6 SB3-versus-Torch check refuses, record the export '
               'gate as FAIL, skip the C++ golden comparison, mark deployment unverified and evaluate the same '
               'final policy with the frozen evaluator through --checkpoint')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(static_report_path, root=ROOT):
    static_report_path = Path(static_report_path)
    static = json.loads(static_report_path.read_text())
    if static.get('status') != 'PASS':
        raise SystemExit('static checks are not PASS; contract 09 cannot be frozen')
    if static.get('native_integration_steps_executed') != 0 or static.get('training_started') is not False:
        raise SystemExit('the static checks ran physics or started training')
    missing = [p for p in FROZEN_INPUTS if not (Path(root)/p).is_file()]
    if missing:
        raise SystemExit(f'frozen inputs missing: {missing}')
    for field, path in (('new_entry_sha256', launcher.ENTRY), ('wrapper_sha256', 'tools09/run_scale09.py'),
                        ('launcher_sha256', 'tools08/run_scale08.py')):
        if static.get(field) != sha(Path(root)/path):
            raise SystemExit(f'the static checks examined a different {path} than the current one')
    return {
        'format': 'd1-contract09-release-v1', 'status': 'FROZEN_BEFORE_TRAINING',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'contract': CONTRACT,
        'single_change': 'PPO entropy coefficient 0.005 -> 0, everything else as contract 08',
        'frozen_inputs': {p: sha(Path(root)/p) for p in FROZEN_INPUTS},
        'launcher_sha256': sha(Path(root)/'tools08/run_scale08.py'), 'release_tool_sha256': sha(__file__),
        'static_report': {'path': str(static_report_path.resolve().relative_to(Path(root).resolve())),
                          'sha256': sha(static_report_path), 'checks': len(static['checks'])},
        'plans': launcher.PLANS, 'gate': 'rollout 1 equal to the frozen 03d ledger, relative 1e-9, counters exact',
        'predictions_preregistered': PREDICTIONS, 'interpretation_table': INTERPRETATION,
        'export_rule': EXPORT_RULE, 'budget': BUDGET,
        'not_authorised_by_any_outcome': ['formal three arms x three seeds', 'a second run of this contract'],
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
                      'frozen_inputs': len(receipt['frozen_inputs'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
