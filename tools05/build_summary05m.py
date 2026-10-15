#!/usr/bin/env python3
"""Compact, GitHub-readable summary of contract 05M, built from its receipts.

evidence/ is not published, so without this file a reader of the repository cannot see
the one physical measurement that settled whether the policy's wheel action is what
keeps 04b moving after a stop. Every number is read from a receipt. Pre-registered
verdicts and post-hoc observations are kept in separate fields. It also lists the
statements in the earlier published contract 04 summary that this work overturned,
because that file is history and is not rewritten. Reads receipts only; no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT/'evidence/career_contract_05'
LEDGER04 = ROOT/'evidence/career_contract_04/final_delivery_02.json'
RUNS = ('smoke', 'A', 'B', 'C')
RUN_DIRS = {'smoke': 'probe05_smoke_01', 'A': 'probe05_runA_01', 'B': 'probe05_runB_01', 'C': 'probe05_runC_01'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def budget(run_reports, cap):
    used = {name: report['actual_counts']['native_steps'] for name, report in run_reports.items()}
    total = sum(used.values())
    if total > cap:
        raise SystemExit(f'05M used {total} native steps, more than its {cap} cap')
    bad = [name for name, report in run_reports.items() if report['status'] != 'COMPLETE']
    if bad:
        raise SystemExit(f'runs did not complete: {bad}')
    return {'pool': '05M engineering', 'cap_native': cap, 'used_native_by_run': used,
            'used_native_total': total, 'transitions': 0, 'unused_native': cap-total}


def project_cumulative(before, added_native, added_transitions=0):
    return {'transitions': before['transitions']+added_transitions, 'native': before['native']+added_native,
            'before_contract_05m': before, 'added_by_contract_05m': {'transitions': added_transitions,
                                                                    'native': added_native}}


def readings(analysis):
    first = analysis['first_order_stop']
    return {'entry_speed_m_s': analysis['entry_speed_m_s'],
            'tau_s_ticks_10_to_50': analysis['decay_time_constant_s']['ticks_10_to_50'],
            'displacement_200_ticks_euclidean_m': analysis['final_euclidean_displacement_m'],
            'path_200_ticks_m': first['measured_path_m'],
            'first_order_predicted_over_path': first['predicted_over_path'],
            'ticks_until_speed_below_0p02': analysis['ticks_until_speed_below_0p02_m_s']}


CORRECTIONS = [
    {'claim_in_contract04_summary': 'the policy zeroes its wheels only after it has already stopped',
     'status': 'overturned',
     'by': 'zero-physics reanalysis (results M9): wheel action mean(a^2) falls below 0.02 within 10 ticks of the cut'},
    {'claim': 'forward stop displacement is about entry speed times decay constant (as a mechanism)',
     'status': 'demoted to an empirical fit on the policy arm',
     'by': '05M: with the wheels forced to zero the same formula over-predicts by about 1.6 times'},
    {'claim': "04b's lean is about -1.1 to -1.2 rad per m/s of speed",
     'status': 'overturned',
     'by': 'lean_analysis_01: the ratio included the resting lean; above rest it is not constant'},
]


def build(ev=EV, ledger04=LEDGER04):
    release = read(ev/'probe05_release_02.json')
    runs = {name: read(ev/RUN_DIRS[name]/'report.json') for name in RUNS}
    verdict = read(ev/'probe05_verdict_01/verdict.json')
    arms = {'command_0p2': read(ev/'probe05_arms_03/report.json'), 'command_0p3': read(ev/'probe05_arms_04/report.json')}
    lean = read(ev/'lean_analysis_01/report.json')
    money = budget(runs, release['budget']['cap_native'])
    if runs['A'].get('fidelity', {}).get('passed') is not True:
        raise SystemExit('run A did not pass its fidelity gate; nothing downstream is valid')
    return {
        'format': 'd1-contract05m-publication-summary-v1',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'simulation_only': True, 'real_robot_validation': False,
        'contract': release['contract'], 'kind': release['kind'],
        'question': release['single_question'],
        'policy_measured': '04b exported ONNX, Python ORT, seed 7001, flat ground',
        'release_receipt_sha256': sha(ev/'probe05_release_02.json'),
        'release_supersedes': release['supersedes'],
        'pre_registered_verdict': verdict['predictions'],
        'pre_registered_interpretation': verdict['interpretation'],
        'interpretation_limit': ('a zero wheel action is a zero wheel speed target, i.e. passive damping braking; '
                                 'active reverse braking was never measured'),
        'p4_precondition': verdict['p4_precondition'],
        'readings': {name: readings(runs[name]['analysis']) for name in ('A', 'B', 'C')},
        'post_hoc_not_pre_registered': {
            'share_of_displacement_explained_by_policy_wheel_action': {
                k: v['share_of_euclidean_displacement_explained_by_policy_wheel_action'] for k, v in arms.items()},
            'lean': {'rest_pitch_rad': lean['low_command']['rest_pitch_rad'],
                     'lean_above_rest_per_m_s': {'command_0p2': lean['low_command']['ratio_above_rest_rad_per_m_s'],
                                                 'command_0p3': lean['high_command']['ratio_above_rest_rad_per_m_s']},
                     'line_through_plateaus': lean['line_through_plateaus'],
                     'line_passes_through_rest': lean['line_passes_through_rest_within_0p02_rad']},
            'release_state_when_wheels_forced_to_zero': lean['wheels_forced_to_zero_release_state']},
        'budget': money,
        'project_cumulative': project_cumulative(read(ledger04)['project_cumulative'], money['used_native_total']),
        'corrections_to_earlier_published_claims': CORRECTIONS,
        'not_authorised_by_this_result': release['not_authorised_by_any_outcome'],
        'limits': release['limits'],
        'receipt_sha256': {RUN_DIRS[n]: sha(ev/RUN_DIRS[n]/'report.json') for n in RUNS}
        | {'probe05_verdict_01': sha(ev/'probe05_verdict_01/verdict.json'),
           'lean_analysis_01': sha(ev/'lean_analysis_01/report.json'),
           'final_delivery_02': sha(ledger04)},
        'tool_sha256': sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/contract05m_summary.json')
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'native_used': report['budget']['used_native_total'],
                      'project_cumulative_native': report['project_cumulative']['native']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
