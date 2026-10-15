#!/usr/bin/env python3
"""Freeze contract 05M before any of its physics runs.

Records the SHA-256 of every input, of the contract text, of the probe and of its
static checks, together with the pre-registered predictions, the budget and the
interpretation table. The probe refuses to run unless its own SHA equals the one
written here, so nothing recorded below can be renegotiated after a result exists.
Executes no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools05'))

import probe_stop_factorisation05 as probe  # noqa: E402

CONTRACT = ROOT/'docs/career_execution_contract_05m.md'


def predictions():
    return [
        {'id': 'P1', 'kind': 'gate', 'metric': 'replay of run A equals the frozen evaluation on every pre-cut tick',
         'channels': ['body_vx_m_s', 'base_x_m', 'base_y_m'], 'ticks': probe.CUT_TICK,
         'max_abs_difference': probe.P1_TOLERANCE, 'if_not_met': 'stop; B and C cancelled'},
        {'id': 'P2', 'kind': 'primary',
         'metric': 'tau of run A over ticks 10..50 after the cut, wheels forced to zero',
         'baseline_policy_arm_s': probe.TAU_POLICY_ARM_S,
         'supported_at_or_below_s': probe.P2_SUPPORTED_AT_OR_BELOW_S,
         'refuted_at_or_above_s': probe.P2_REFUTED_AT_OR_ABOVE_S,
         'between': 'UNCERTAIN: no mechanism conclusion'},
        {'id': 'P3', 'kind': 'secondary', 'metric': 'run A Euclidean displacement over 200 ticks after the cut',
         'threshold_m': probe.P3_DISPLACEMENT_BELOW_M, 'direction': 'below'},
        {'id': 'P4', 'kind': 'linearity', 'metric': 'tau(B) / tau(A)',
         'range': list(probe.P4_RATIO_RANGE)},
        {'id': 'P5', 'kind': 'formula', 'metric': 'predicted / measured path length, runs A and B',
         'range': list(probe.P5_RATIO_RANGE)},
    ]


def build_receipt(static_report_path, contract=CONTRACT, root=ROOT):
    static_report_path = Path(static_report_path)
    static = json.loads(static_report_path.read_text())
    if static.get('status') != 'PASS':
        raise SystemExit('static checks are not PASS; the contract cannot be frozen')
    if static.get('native_integration_steps_executed') != 0:
        raise SystemExit('the static checks ran physics')
    if static.get('probe_sha256') != probe.sha(probe.__file__):
        raise SystemExit('the static checks were run against a different probe than the current one')
    probe.check_frozen_inputs(probe.FROZEN_INPUTS, root)
    own = Path(__file__).resolve()
    tests = root/'tests/test_probe05.py'
    checker = root/'tools05/check_probe05_static.py'
    return {
        'format': 'd1-contract05m-release-v1',
        'status': 'FROZEN_BEFORE_MEASUREMENT',
        'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'contract': str(Path(contract).resolve().relative_to(root)),
        'contract_sha256': probe.sha(contract),
        'kind': 'measurement, not training; 0 transitions',
        'single_question': ('with the legs under policy control, does forcing only the four wheel '
                            'actions to zero after the cut shorten the 04b decay time constant?'),
        'tool_sha256': probe.sha(probe.__file__),
        'release_tool_sha256': probe.sha(own),
        'static_check_tool_sha256': probe.sha(checker),
        'tests_sha256': probe.sha(tests),
        'static_report': {'path': str(static_report_path.resolve().relative_to(root)),
                          'sha256': probe.sha(static_report_path), 'checks': len(static['checks'])},
        'frozen_inputs': {k: probe.sha(Path(root)/k) for k in probe.FROZEN_INPUTS},
        'onnx_sha256': probe.sha(root/'evidence/career_contract_04/pilot04b_export_01/policy.onnx'),
        'config_sha256': probe.sha(root/'training/configs/pilot04b.json'),
        'runs': {name: {'ticks': spec['ticks'], 'cut_tick': spec['cut_tick'],
                        'zero_wheels': spec['zero_wheels'], 'sequence': spec['sequence'],
                        'planned_native': probe.planned_native(name), 'purpose': spec['purpose']}
                 for name, spec in probe.RUNS.items()},
        'seed': probe.SEED,
        'budget': {'pool': '05M engineering', 'cap_native': probe.POOL_CAP_NATIVE,
                   'transitions': 0, 'retries_allowed': False, 'seed_change_allowed': False,
                   'borrowed_from': 'NONE',
                   'order': ['smoke', 'A', 'B and C only after A passes P1']},
        'predictions_preregistered': predictions(),
        'interpretation_table': {
            'P2 SUPPORTED + P3 MET': probe.interpretation('SUPPORTED', 'MET'),
            'P2 SUPPORTED + P3 NOT_MET': probe.interpretation('SUPPORTED', 'NOT_MET'),
            'P2 REFUTED': probe.interpretation('REFUTED', 'MET'),
            'P2 UNCERTAIN': probe.interpretation('UNCERTAIN', 'MET')},
        'not_authorised_by_any_outcome': ['training', 'a new reward term', 'a qualification trial',
                                          'watchdog acceptance', 'terrain or lateral claims'],
        'limits': ['single seed 7001', 'single policy 04b', 'flat ground only', 'Python ORT, not the C++ runner',
                   'forced-zero wheels is not a policy'],
        'physics_steps_executed_by_this_receipt': 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite an existing release receipt: {args.output}')
    receipt = build_receipt(args.static_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'status': receipt['status'],
                      'inputs': len(receipt['frozen_inputs']), 'physics_steps_executed': 0}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
