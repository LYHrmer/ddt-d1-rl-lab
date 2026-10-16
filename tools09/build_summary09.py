#!/usr/bin/env python3
"""Publish the incomplete contract 09 from receipts; never load its partial policy.

The frozen RAM stop occurred before the last quarter of the planned training.
Predictions and qualification remain unevaluated. Inputs, worker counters and the
preflight are checked before adding the actual consumption to the contract 08 total.
Reads files only and refuses to overwrite an existing summary. No physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools09'))

import verify_scale09 as verify  # noqa: E402

RECEIPTS = (
    'scale09_release_01.json', 'scale09_preflight_01_launcher/report.json',
    'scale09_preflight_01/result.json', 'scale09_train_01_launcher/report.json',
    'scale09_train_01/result.json', 'scale09_train_01/protocol.json',
    'scale09_verdict_01/verdict.json',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def counters(result):
    """Count failed steps too, using final worker counters rather than live_status."""
    counts = result['actual_counts']
    for key, value in counts.items():
        if type(value) is not int or value < 0:
            raise ValueError(f'invalid counter: {key}')
        if sum(worker[key] for worker in result['per_worker_counts']) != value:
            raise ValueError(f'worker counters disagree: {key}')
    transitions = counts['policy_transitions']
    if result['trained_transitions'] != transitions or counts['native_steps'] != transitions*10:
        raise ValueError('training/native transition counts disagree')
    return {'transitions': transitions, 'native': counts['native_steps']}


def build(root=ROOT):
    root = Path(root)
    ev = root/'evidence/career_contract_09'
    release, pre, pre_result, launch, result, protocol, verdict = [read(ev/p) for p in RECEIPTS]
    for path, expected in release['frozen_inputs'].items():
        if sha(root/path) != expected:
            raise ValueError(f'frozen input changed: {path}')
    for path, expected in protocol['inputs_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'protocol input changed: {path}')
    for receipt, trainer in ((pre, pre_result), (launch, result)):
        if receipt['release_sha256'] != sha(ev/RECEIPTS[0]):
            raise ValueError('launcher refers to a different release')
        if receipt['trainer_result'] != trainer or trainer['inputs_unchanged'] is not True:
            raise ValueError('launcher/trainer result mismatch or changed inputs')
    if (pre['equivalence'] != 'PASS' or pre_result['status'] != 'COMPLETE'
            or pre['trainer_returncode'] != 0):
        raise ValueError('preflight did not complete and pass equivalence')
    target = release['plans']['scale']['steps']
    if (protocol['planned_transitions'] != target or protocol['hyperparameters']['ent_coef'] != 0
            or protocol['kind'] != 'scale' or protocol['seed'] != 101):
        raise ValueError('training protocol does not match contract 09')
    if result['status'] == 'COMPLETE' or result['trained_transitions'] >= target:
        raise ValueError('this closeout is only for incomplete contract 09 training')
    expected_verdict = verify.judge(None, None, {}, launch)
    if (verdict['predictions'] != expected_verdict
            or verdict['qualification']['overall'] != 'NOT_EVALUATED'
            or verdict['qualification']['tasks_passed'] is not None
            or verdict['inputs']['launcher_report_sha256'] != sha(ev/RECEIPTS[3])):
        raise ValueError('incomplete training must not have predictions or qualification results')
    training_dir = ev/'scale09_train_01'
    if (training_dir/'policy.zip').exists():
        raise ValueError('unexpected final policy for incomplete training')
    for pattern in ('scale09_export_*', 'scale09_python_eval_*', 'scale09_training_replay_*',
                    'scale09_cpp_golden_*', 'scale09_golden_*'):
        if any(ev.glob(pattern)):
            raise ValueError(f'unexpected post-training artifacts: {pattern}')
    pre_used, train_used = counters(pre_result), counters(result)
    used = {key: pre_used[key]+train_used[key] for key in ('transitions', 'native')}
    cap = release['budget']['cap']
    if any(used[key] > cap[key] for key in used):
        raise ValueError('contract 09 exceeds its budget')
    previous = read(root/'reports/contract08_summary.json')['project_cumulative']
    before = release['budget']['project_cumulative_before']
    if any(previous[key] != before[key] for key in used):
        raise ValueError('contract 08 cumulative total differs from contract 09 registration')
    return {
        'format': 'd1-contract09-publication-summary-v1',
        'prepared_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'INCOMPLETE', 'simulation_only': True, 'real_robot_validation': False,
        'contract': release['contract'], 'single_change': release['single_change'],
        'equivalence': {'E0': pre['equivalence'], 'E1': launch['equivalence'],
                        'scale_rollouts_compared': launch['rollouts_compared']},
        'training': {'status': result['status'], 'transitions': train_used['transitions'],
                     'planned_transitions': target, 'fraction_completed': train_used['transitions']/target,
                     'finished_utc': result['finished_utc'], 'wall_minutes': launch['wall_s']/60,
                     'trainer_returncode': launch['trainer_returncode'],
                     'inputs_unchanged': result['inputs_unchanged'],
                     'min_available_ram_bytes_rollout_samples': result['min_available_ram_bytes'],
                     'ram_sample_limit': ('The step callback checks RAM every 64 calls but records the minimum '
                                          'only at rollout ends; the triggering sample is not retained.')},
        'partial_checkpoint': {'path': 'scale09_train_01/partial_policy.zip',
                               'sha256': sha(training_dir/'partial_policy.zip'),
                               'used_for_export_or_evaluation': False},
        'pre_registered_verdict': verdict['predictions'],
        'qualification': verdict['qualification'], 'interpretation': verdict['interpretation'],
        'subsequent_steps': {step: 'NOT_EXECUTED_INCOMPLETE_TRAINING' for step in
                             ('export', 'cpp_golden', 'python_qualification', 'training_episode_replay')},
        'budget': {'used_transitions': used['transitions'], 'used_native': used['native'], 'cap': cap,
                   'by_step': {'preflight': pre_used, 'scale_training': train_used,
                               'python_qualification_native': 0, 'closeout_native': 0}},
        'project_cumulative': {**{key: before[key]+used[key] for key in used},
                               'before_contract_09': before, 'added_by_contract_09': used},
        'frozen_inputs_checked': len(release['frozen_inputs']),
        'protocol_inputs_checked': len(protocol['inputs_sha256']),
        'receipt_sha256': {name: sha(ev/name) for name in RECEIPTS},
        'physics_steps_executed_by_this_tool': 0,
        'limits': ['no retry or resume', 'partial policy is not the final policy',
                   'Q1-Q5 not evaluable; no conclusion about the entropy hypothesis',
                   'unused budget expires with this contract'],
        'tool_sha256': sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/contract09_summary.json')
    args = parser.parse_args()
    report = build()
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'budget': report['budget'],
                      'project_cumulative': report['project_cumulative']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
