#!/usr/bin/env python3
"""Compact, GitHub-readable summary of contract 08, built from its receipts.

evidence/ is not published. Every number here is read from a receipt: the training and
equivalence results, the export failure and its diagnosis, the qualification verdict and
the description of what each task did. Pre-registered verdicts and post-hoc observations
sit in separate fields. The budget is recomputed and must stay within the frozen cap.
Reads receipts only; no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT/'evidence/career_contract_08'
PREVIOUS_SUMMARY = ROOT/'reports/contract05m_summary.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def budget(preflight, training, evaluation, cap):
    used_native = (preflight['trainer_result']['actual_counts']['native_steps']
                   + training['trainer_result']['actual_counts']['native_steps']
                   + evaluation['actual_native_steps'])
    used_transitions = preflight['trainer_result']['trained_transitions'] + training['trainer_result']['trained_transitions']
    if used_native > cap['native'] or used_transitions > cap['transitions']:
        raise SystemExit(f'contract 08 used {used_native} native / {used_transitions} transitions, over its cap')
    return {'used_native': used_native, 'used_transitions': used_transitions, 'cap': cap,
            'by_step': {'preflight_native': preflight['trainer_result']['actual_counts']['native_steps'],
                        'training_native': training['trainer_result']['actual_counts']['native_steps'],
                        'python_qualification_native': evaluation['actual_native_steps'],
                        'preflight_diagnosis_native': 0, 'onnx_golden_native': 0}}


def cumulative(before, used_native, used_transitions):
    return {'transitions': before['transitions']+used_transitions, 'native': before['native']+used_native,
            'before_contract_08': before, 'added_by_contract_08': {'transitions': used_transitions, 'native': used_native}}


def build(ev=EV, previous=PREVIOUS_SUMMARY):
    release = read(ev/'scale08_release_01.json')
    preflight = read(ev/'scale08_preflight_01_launcher/report.json')
    training = read(ev/'scale08_train_01_launcher/report.json')
    evaluation = read(ev/'scale08_python_eval_01/summary.json')
    verdict = read(ev/'scale08_verdict_01/verdict.json')
    diagnosis = read(ev/'scale08_export_diagnosis_01/report.json')
    description = read(ev/'scale08_eval_description_01/report.json')['versions']
    if preflight['equivalence'] != 'PASS' or training['equivalence'] != 'PASS':
        raise SystemExit('an equivalence gate did not pass; the scale result would be confounded')
    money = budget(preflight, training, evaluation, release['budget']['cap'])
    return {
        'format': 'd1-contract08-publication-summary-v1',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'simulation_only': True, 'real_robot_validation': False,
        'contract': release['contract'], 'addendum': 'docs/career_execution_contract_08_addendum_a.md',
        'single_change': release['single_change'],
        'equivalence': {'preflight_rollouts': preflight['rollouts_compared'],
                        'scale_rollouts': training['rollouts_compared'], 'status': 'PASS'},
        'training': {'status': training['trainer_result']['status'],
                     'transitions': training['trainer_result']['trained_transitions'],
                     'wall_minutes': training['wall_s']/60, 'late_return_change': verdict['late_return_change']},
        'export_gate': {'status': 'FAIL', 'error': diagnosis['export_error_verbatim'],
                        'diagnosis': {k: {m: v[m] for m in ('max_abs_difference', 'max_abs_unclipped_action',
                                                            'difference_over_eps_times_peak', 'max_abs_weight')}
                                      for k, v in diagnosis['policies'].items()},
                        'deployment_chain': 'not verified: the C++ golden comparison needs the exporter metadata'},
        'pre_registered_verdict': verdict['predictions'],
        'qualification': verdict['qualification'],
        'tasks': {t: {'8x_passed': verdict['tasks_8x'][t]['passed'], '8x_blocking': verdict['tasks_8x'][t]['blocking_gate'],
                      '1x_passed': verdict['tasks_03d_1x'][t]['passed'], '1x_blocking': verdict['tasks_03d_1x'][t]['blocking_gate']}
                  for t in verdict['tasks_8x']},
        'post_hoc_not_pre_registered': {'what_each_task_did': description},
        'budget': money,
        'project_cumulative': cumulative(read(previous)['project_cumulative'], money['used_native'], money['used_transitions']),
        'limits': ['single seed 101', 'single configuration 03d', 'flat ground only', 'Python evaluator decides',
                   'deployment chain not verified'],
        'receipt_sha256': {name: sha(ev/name) for name in (
            'scale08_release_01.json', 'scale08_preflight_01_launcher/report.json',
            'scale08_train_01_launcher/report.json', 'scale08_python_eval_01/summary.json',
            'scale08_verdict_01/verdict.json', 'scale08_export_diagnosis_01/report.json',
            'scale08_eval_description_01/report.json')},
        'tool_sha256': sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/contract08_summary.json')
    args = parser.parse_args()
    report = build()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(args.output), 'used_native': report['budget']['used_native'],
                      'project_cumulative': {k: report['project_cumulative'][k] for k in ('transitions', 'native')}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
