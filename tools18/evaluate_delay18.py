"""Bounded evaluation of new delay18 arms using the unchanged formal metrics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools18 import common18 as common  # noqa: E402
from tools17.evaluate_formal17 import runtime, verify_trace  # noqa: E402
from tools16.evaluate_comparison16 import frozen_helpers  # noqa: E402
from tools16.train_comparison16 import termination_scope  # noqa: E402


def matrix(stage):
    if stage == 'probe_base':
        return [('turn_stop', 3, 1001)]
    common.require(stage in ('eval_zero', 'eval_random', 'holdout'), 'not an evaluation stage')
    seeds = (2001, 2002, 2003) if stage == 'holdout' else (1001, 1002, 1003)
    return [(task, delay, seed) for task in ('forward_stop', 'reverse_stop', 'turn_stop')
            for delay in (0, 1, 2, 3) for seed in seeds]


def selection(plan):
    candidates = []
    for mode in ('zero', 'random'):
        report_path = ROOT/plan['output_root']/('eval_'+mode)/'report.json'
        report = common.read_json(report_path)
        common.require(report.get('status') == 'COMPLETE' and len(report.get('cases', [])) == 36
                       and report.get('cleanup_complete') is True and report.get('inputs_unchanged') is True
                       and report.get('source_identity') == plan['source_identity']
                       and report.get('stage') == 'eval_'+mode
                       and [(item['sequence'], item['delay_ticks'], item['reset_seed']) for item in report['cases']]
                           == matrix('eval_'+mode)
                       and all(type(item['J']) in (int, float) and math.isfinite(item['J']) and item['J'] >= 0
                               for item in report['cases']),
                       'candidate evaluation incomplete')
        if report.get('all_36_reference_pass') is True:
            cap = common.expected_counts(21600)
            common.require(all(report.get('actual_counts', {}).get(key) == value for key, value in cap.items())
                           and all(item['reference_thresholds_met'] is True
                                   and item['done_reason'] == 'HORIZON_12S' and item['failure'] is False
                                   and all(item['actual_counts'].get(key) == value
                                           for key, value in common.expected_counts(600).items())
                                   for item in report['cases']),
                           'candidate pass flag contradicts cases')
            candidates.append((sum(item['J'] for item in report['cases'])/36, mode))
    return min(candidates, key=lambda row: (row[0], row[1] != 'zero'))[1] if candidates else None


def inputs(plan, stage):
    execution = ROOT/plan['output_root']
    if stage == 'probe_base':
        export = ROOT/'evidence/closeout17/execution_01/export'
        report = common.read_json(export/'report.json')
        common.require(report['checkpoint_sha256'] == plan['base_checkpoint']['sha256'], 'foreign baseline export')
        common.require(report['onnx_sha256'] == common.sha(export/'policy.onnx'), 'baseline ONNX changed')
        return export/'policy.onnx', plan['base_checkpoint']['sha256'], 'baseline'
    mode = selection(plan) if stage == 'holdout' else stage.removeprefix('eval_')
    common.require(mode is not None, 'no candidate passed all36; holdout not eligible')
    checkpoint, trained = common.completed_training(plan, mode)
    export = execution/('export_'+mode)
    exported = common.read_json(export/'report.json')
    cpp_path = execution/('cpp_'+mode)/'report.json'
    cpp = common.read_json(cpp_path)
    for report in (exported, cpp):
        common.require(report.get('status') == 'PASS' and report.get('checkpoint_sha256') == common.sha(checkpoint)
                       and report.get('source_identity') == plan['source_identity'], 'foreign or failed numerical stage')
    common.require(exported['onnx_sha256'] == cpp['onnx_sha256'] == common.sha(export/'policy.onnx'), 'ONNX mismatch')
    common.require(cpp['export_report_sha256'] == common.sha(export/'report.json'), 'CPP export proof changed')
    common.require(exported['fixtures_sha256'] == cpp['fixtures_sha256'] == common.sha(export/'golden_fixtures.json'),
                   'numerical fixtures changed')
    return export/'policy.onnx', trained['checkpoint_sha256'], mode


def add_counts(total, value, cap):
    common.require(isinstance(value, dict) and all(type(value.get(key)) is int and value[key] >= 0 for key in total),
                   'unknown or invalid physics counters')
    result = {key: total[key]+value[key] for key in total}
    common.require(all(result[key] <= cap[key] for key in total), 'evaluation physical cap exceeded')
    return result


def run(plan_path, stage, output, *, loader=common.load_plan, launch_validator=common.validate_launch,
        runtime_factory=runtime, memory_reader=common.read_memory, clock=time.monotonic):
    started = clock()
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = loader(plan_path)
    launch_validator(plan, plan_path, stage, output)
    common.require(not output.exists(), 'evaluation refuses overwrite')
    old_release, training_protocol, _, _ = common.verify_inputs(plan)
    onnx, checkpoint_sha, mode = inputs(plan, stage)
    input_onnx_sha = common.sha(onnx)
    helpers = frozen_helpers(old_release)
    cases = matrix(stage)
    cap = common.expected_counts(plan['stages'][stage]['transitions'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-delay18-evaluation-v1', 'status': 'INCOMPLETE', 'stage': stage, 'mode': mode,
              'plan_sha256': common.sha(plan_path), 'source_identity': plan['source_identity'],
              'checkpoint_sha256': checkpoint_sha, 'onnx_sha256': input_onnx_sha,
              'planned_cases': len(cases), 'cases': [], 'actual_counts': None,
              'known_actual_counts': dict.fromkeys(cap, 0), 'inputs_unchanged': False, 'cleanup_complete': True,
              'all_36_reference_pass': False, 'new_training_transitions': 0, 'error': None,
              'minimum_effective_bytes': None}
    environment = None
    total = dict.fromkeys(cap, 0)
    accounted, constructing, unknown = False, False, False
    last_poll = -float('inf')

    def poll(force=False):
        nonlocal last_poll
        now = clock()
        common.require(now-started < plan['stages'][stage]['max_wall_s'], 'evaluation wall limit')
        if force or now-last_poll >= 1:
            sample = memory_reader()
            value = sample['effective_available_bytes']
            common.require(type(value) is int and value >= plan['resources']['reserve_bytes'], 'evaluation RAM reserve')
            old_min = report['minimum_effective_bytes']
            report['minimum_effective_bytes'] = min(old_min, value) if old_min is not None else value
            last_poll = now

    with termination_scope():
        try:
            common.require(memory_reader()['effective_available_bytes'] >= plan['resources']['start_min_effective_bytes'],
                           'evaluation start RAM below6GiB')
            rt = runtime_factory(old_release, onnx, training_protocol)
            poll(True)
            protocol = {'format': 'd1-delay18-evaluation-protocol-v1', 'stage': stage, 'mode': mode,
                        'plan_sha256': report['plan_sha256'], 'source_identity': plan['source_identity'],
                        'checkpoint_sha256': checkpoint_sha, 'onnx_sha256': input_onnx_sha,
                        'backend': rt.backend, 'thresholds': helpers.THRESHOLDS, 'budget_max': cap,
                        'scenarios': [{'sequence': name, 'delay_ticks': delay, 'reset_seed': seed}
                                      for name, delay, seed in cases]}
            common.provenance17.exclusive_json(output/'protocol.json', protocol)
            report['protocol_sha256'] = common.sha(output/'protocol.json')
            report['backend'] = rt.backend

            def predict(observation):
                poll()
                return rt.predict(observation)

            for name, delay, seed in cases:
                poll(True)
                accounted, constructing = False, True
                environment = rt.make_env(name, helpers.SEQUENCES[name], delay, seed)
                constructing = False
                item = helpers.run_episode(environment, predict, name, delay, seed,
                                           output/f'{name}_delay{delay}_seed{seed}.npz', cap['native_steps']-total['native_steps'])
                common.require((item['sequence'], item['delay_ticks'], item['reset_seed']) == (name, delay, seed),
                               'case identity mismatch')
                verify_trace(output, item, helpers)
                report['cases'].append(item)
                total = add_counts(total, item['actual_counts'], cap)
                accounted = True
                environment.close()
                environment = None
                report['known_actual_counts'] = dict(total)
                common.atomic_json(output/'report.json', report)
            poll(True)
            loader(plan_path)
            common.require(common.sha(onnx) == input_onnx_sha and inputs(plan, stage) == (onnx, checkpoint_sha, mode),
                           'evaluation model changed')
            report['inputs_unchanged'] = True
            report['all_36_reference_pass'] = (len(cases) == 36 and total == cap
                                              and all(item['reference_thresholds_met'] is True for item in report['cases']))
            report['by_delay_ms'] = {str(delay*20): {'cases': len(items),
                'reference_pass_count': sum(item['reference_thresholds_met'] is True for item in items),
                'failures': sum(item['failure'] is True for item in items)}
                for delay in (0, 1, 2, 3) for items in [[item for item in report['cases'] if item['delay_ticks'] == delay]]}
            report['reference_pass_cases'] = sum(item['reference_thresholds_met'] is True for item in report['cases'])
            report['mean_J'] = sum(item['J'] for item in report['cases'])/len(cases)
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
            if constructing:
                unknown, report['cleanup_complete'] = True, False
            if environment is not None and not accounted:
                try:
                    total = add_counts(total, environment.counters(), cap)
                except BaseException:
                    unknown = True
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if environment is not None:
                try:
                    environment.close()
                except BaseException as exc:
                    report.update(cleanup_complete=False, error=f'cleanup {exc}', status='INCOMPLETE')
            report['known_actual_counts'] = total
            report['actual_counts'] = None if unknown else total
            report['elapsed_s'] = clock()-started
            common.atomic_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stage', choices=('probe_base', 'eval_zero', 'eval_random', 'holdout'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.plan, args.stage, args.output)
    print(json.dumps({key: report.get(key) for key in ('status', 'stage', 'mode', 'reference_pass_cases', 'by_delay_ms', 'error')}, indent=2))
    return 0 if report['status'] == 'COMPLETE' and report['cleanup_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
