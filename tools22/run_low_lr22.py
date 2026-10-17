"""Own bounded training22, numerical export and unchanged reference evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools22 import common22 as c  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402
from tools19.run_cpp_closedloop19 import cleanup_owned_tree, subreaper  # noqa: E402
from tools21 import closeout_completed18 as prior  # noqa: E402

ZERO = c.expected_counts(0)
CAP = c.expected_counts(21600)


def identity(plan, path, stage):
    return {'format': 'd1-low-lr22-stage-v1', 'stage': stage, 'plan_sha256': c.sha(path),
            'source_identity': plan['source_identity']}


def projected_counts(value, cap, *, partial=False):
    c.require(prior.valid_counts(value, cap, partial=partial), 'invalid or unknown22 counters')
    return {key: value[key] for key in ZERO}


def completed_training(plan, plan_path):
    directory = ROOT/plan['output_root']/'train'
    result, protocol = c.read_json(directory/'result.json'), c.read_json(directory/'protocol.json')
    checkpoint = directory/'policy.zip'
    common = {'stage': 'train', 'mode': 'zero', 'seed': 181, 'envs': 4,
              'plan_sha256': c.sha(plan_path), 'source_identity': plan['source_identity'],
              'hyperparameters': c.HYPERPARAMETERS}
    c.require(all(item.get(k) == v for item in (result, protocol) for k, v in common.items()),
              'training22 identity/settings differ')
    c.require(result.get('format') == 'd1-low-lr22-training-result-v1'
              and protocol.get('format') == 'd1-low-lr22-training-protocol-v1'
              and result.get('status') == 'COMPLETE' and result.get('error') is None
              and result.get('eligible_for_evaluation') is True and result.get('cleanup_complete') is True
              and result.get('inputs_unchanged') is True and result.get('aux_coef') == 0.
              and result.get('initial_num_timesteps') == c.INITIAL
              and result.get('new_trained_transitions') == c.TRANSITIONS
              and result.get('final_num_timesteps') == c.INITIAL+c.TRANSITIONS
              and result.get('protocol_sha256') == c.sha(directory/'protocol.json')
              and checkpoint.is_file() and not checkpoint.is_symlink()
              and result.get('checkpoint_sha256') == c.sha(checkpoint), 'incomplete/partial22 training')
    prior.training_counts(result, c.TRANSITIONS)
    c.require(protocol.get('base_checkpoint') == plan['base_checkpoint']
              and protocol.get('base_protocol') == plan['base_protocol']
              and protocol.get('planned_new_transitions') == c.TRANSITIONS
              and protocol.get('warm_start') is True and protocol.get('reset_num_timesteps') is False
              and protocol.get('teacher_policy_used') is False, 'training22 provenance differs')
    initial = result.get('initial_state', {})
    old = c.read_json(ROOT/'evidence/local_optimization18/execution_01/train_zero/result.json')['initial_state']
    c.require(initial.get('optimizer_restored') is True and all(initial.get(k) == old[k]
              for k in ('num_timesteps', 'policy_state_sha256', 'optimizer_state_sha256', 'updates')),
              'original complete policy/Adam not restored')
    c.require(initial.get('optimizer_momentum_preserved') is True
              and initial.get('learning_rate_before') == 3e-4 and initial.get('learning_rate_after') == 1e-4
              and initial.get('optimizer_state_sha256_before_lr') == initial['optimizer_state_sha256']
              and initial.get('optimizer_without_lr_sha256_before') == initial.get('optimizer_without_lr_sha256_after')
              and isinstance(initial.get('optimizer_without_lr_sha256_before'), str)
              and len(initial['optimizer_without_lr_sha256_before']) == 64,
              'LR22 change lacks preserved optimizer momentum proof')
    _, release, _ = c.base_inputs()
    bound = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': c.sha(checkpoint),
             'training_protocol_sha256': c.sha(directory/'protocol.json'),
             'training_result_sha256': c.sha(directory/'result.json')}
    return {'checkpoint': checkpoint, 'protocol': protocol, 'old_release': release,
            'identity': {'role': 'completed_low_lr22_final_only', 'training_identity': bound}}


def check_model(model, protocol, versions):
    c.require(type(model.num_timesteps) is int and model.num_timesteps == c.INITIAL+c.TRANSITIONS
              and type(model.aux_coef) is float and model.aux_coef == 0. and model.seed == 181
              and model.n_envs == 4 and str(model.device) == 'cpu' and model.get_env() is None,
              'loaded model is not completed22')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                'vf_coef', 'max_grad_norm', 'learning_rate'):
        c.require(getattr(model, key) == c.HYPERPARAMETERS[key], 'loaded22 PPO differs: '+key)
    c.require(model.clip_range(0.) == model.clip_range(1.) == .2 and model.normalize_advantage is True
              and model.clip_range_vf is None and model.target_kl is None and model.use_sde is False
              and versions == protocol['versions'], 'unsupported22 model/runtime')


def numerical(stage, bundle, parent, output):
    if stage == 'cpp':
        return prior.numerical(stage, bundle, parent, output)
    from tools17 import export_completed17 as export
    model, versions = export.load_model(bundle['checkpoint'])
    check_model(model, bundle['protocol'], versions)
    value = export.export_numerical(model, output)
    c.atomic_json(output/'golden_fixtures.json', export.cpp_fixtures(value))
    return {'onnx_sha256': c.sha(output/'policy.onnx'), 'fixtures_sha256': c.sha(output/'golden_fixtures.json'),
            'absolute_limits': export.LIMITS, 'versions': {**versions, **value['numerical_versions']},
            'onnx_ir_version': 7, 'onnx_opset': 11, 'cxx_ort_1_10_status': 'NOT_TESTED',
            **{key: value[key] for key in ('onnx_node_counts', 'onnx_signature', 'parity', 'policy_state_sha256')}}


def verify_numeric(stage, child, directory, *, old_release=None):
    from tools17 import export_completed17 as export
    from tools17 import verify_cpp17 as cpp
    parent = directory.parent
    exported = child if stage == 'export' else c.read_json(parent/'export/report.json')
    export_dir = parent/'export'
    c.require(exported.get('onnx_sha256') == child.get('onnx_sha256') == c.sha(export_dir/'policy.onnx')
              and exported.get('fixtures_sha256') == child.get('fixtures_sha256') == c.sha(export_dir/'golden_fixtures.json'),
              'numeric22 artifacts differ')
    golden = c.read_json(export_dir/'golden_fixtures.json')
    export.cpp_fixtures({'onnx_node_counts': exported['onnx_node_counts'],
                        'onnx_signature': exported['onnx_signature'], 'parity': exported['parity'],
                        'rows': golden['fixtures'], 'fixture_seed': golden['seed']})
    fixtures = cpp.check_fixtures(golden)
    if stage == 'cpp':
        native = directory/'cpp_native/report.json'
        old_release = old_release or c.base_inputs()[1]
        c.require(child.get('export_report_sha256') == c.sha(export_dir/'report.json')
                  and child.get('cpp_native_report_sha256') == c.sha(native)
                  and child.get('comparisons') == 64 and child.get('absolute_tolerance') == 1e-5
                  and child.get('identities', {}).get('policy_sha256') == child['onnx_sha256']
                  and child['identities'].get('fixture_sha256') == child['fixtures_sha256']
                  and all(child['identities'].get(key) == old_release['sources_sha256'].get(name) == c.sha(ROOT/name)
                          for name, key in (('runtime03/build/d1_runtime03', 'runtime_binary_sha256'),
                                            ('runtime03/schema03d.json', 'schema_sha256')))
                  and child.get('cxx_ort_1_10_status') == 'PASS', 'CPP22 proof differs')
        maximum = cpp.check_native(c.read_json(native), fixtures, child['identities'])
        c.require(child.get('max_absolute_error') == maximum, 'CPP22 numerical result differs')


def verify_stage(plan, plan_path, stage, directory, child):
    if stage == 'train':
        completed_training(plan, plan_path)
        return projected_counts(child.get('actual_counts'), c.expected_counts(c.TRANSITIONS))
    c.require(all(child.get(k) == v for k, v in identity(plan, plan_path, stage).items())
              and child.get('status') == 'COMPLETE' and child.get('error') is None
              and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
              and child.get('protocol_sha256') == c.sha(directory/'protocol.json'), 'stage22 proof differs')
    protocol = c.read_json(directory/'protocol.json')
    c.require(all(protocol.get(k) == v for k, v in identity(plan, plan_path, stage).items())
              and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s'], 'stage22 protocol differs')
    cap = CAP if stage in ('formal', 'holdout') else ZERO
    counts = projected_counts(child.get('actual_counts'), cap)
    c.require(protocol.get('budget_max') == cap, 'stage22 cap differs')
    if stage in ('export', 'cpp'):
        verify_numeric(stage, child, directory)
        return counts
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import verify_trace
    bundle = completed_training(plan, plan_path)
    helpers = frozen_helpers(bundle['old_release'])
    c.require([(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in child['cases']] == prior.matrix(stage)
              and protocol.get('thresholds') == helpers.THRESHOLDS and protocol.get('sequences') == helpers.SEQUENCES
              and protocol.get('backend') == child.get('backend')
              and protocol.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256'],
              'original22 evaluation identity/matrix differs')
    total = dict(ZERO)
    for row in child['cases']:
        verify_trace(directory, row, helpers)
        actual = projected_counts(row.get('actual_counts'), CAP)
        total = {key: total[key]+actual[key] for key in ZERO}
    full = total == CAP and all(row['reference_thresholds_met'] is True and row['failure'] is False
                               and row['done_reason'] == 'HORIZON_12S' for row in child['cases'])
    c.require(total == counts and child.get('all_36_reference_pass') is full, 'evaluation22 totals/qualification differ')
    return counts


def run_stage(plan_path, stage, output):
    started = time.monotonic()
    plan = c.load_plan(plan_path)
    c.validate_launch(plan, plan_path, stage, output)
    bundle = completed_training(plan, plan_path)
    c.require(stage in ('export', 'cpp', 'formal', 'holdout') and not output.exists(), 'new22 stage only')
    output.mkdir(exist_ok=False)
    report = dict(identity(plan, plan_path, stage), status='INCOMPLETE', error=None, cases=[],
                  actual_counts=None, known_actual_counts=dict(ZERO), cleanup_complete=True,
                  inputs_unchanged=False, all_36_reference_pass=False)
    last = -float('inf')
    def poll(force=False, admission=False):
        nonlocal last
        now = time.monotonic()
        c.require(now-started < c.STAGES[stage]['max_wall_s'], 'stage22 wall limit')
        if force or now-last >= 1.:
            sample = dict(c.read_memory(), elapsed_s=now-started)
            with (output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            c.require(sample['effective_available_bytes'] >= c.RESOURCES['start_min_effective_bytes' if admission else 'reserve_bytes'],
                      'stage22 RAM below 6/4 GiB')
            last = now
    with termination_guard() as control:
        try:
            poll(True, True)
            c.atomic_json(output/'protocol.json', dict(identity(plan, plan_path, stage),
                max_wall_s=c.STAGES[stage]['max_wall_s'], budget_max=CAP if stage in ('formal','holdout') else ZERO))
            report['protocol_sha256'] = c.sha(output/'protocol.json')
            if stage in ('export', 'cpp'):
                report.update(numerical(stage, bundle, output.parent, output))
                report['actual_counts'] = dict(ZERO)
            else:
                prior.evaluate(stage, bundle, output.parent, output, report, poll)
            c.require(c.load_plan(plan_path) == plan and completed_training(plan, plan_path)['identity'] == bundle['identity'],
                      'stage22 inputs changed')
            poll(True)
            report.update(status='COMPLETE', inputs_unchanged=True)
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            report['elapsed_s'] = time.monotonic()-started
            if control['received'] or report['elapsed_s'] >= c.STAGES[stage]['max_wall_s']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; interrupted/wall limit')
            c.atomic_json(output/'report.json', report)
    return report


def totals(stages):
    return {key: sum(row.get('actual_counts', {}).get(key, 0) if row.get('actual_counts') is not None
                     else row.get('known_actual_counts', ZERO)[key] for row in stages.values()) for key in ZERO}


def execute(plan_path=c.ROOT/c.PLAN, *, loader=c.load_plan, memory_reader=c.read_memory,
            factory=subprocess.Popen, cleanup=cleanup_owned_tree, adopter=subreaper,
            verifier=verify_stage, clock=time.monotonic, sleeper=time.sleep):
    started = clock()
    plan = loader(plan_path)
    output = ROOT/plan['output_root']
    c.require(not output.exists() and not output.is_symlink(), 'attempt22 already exists; no retry')
    if memory_reader()['effective_available_bytes'] < 6 << 30:
        return {'status': 'NOT_READY', 'execution_started': False, 'actual_counts': ZERO}
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-low-lr22-execution-v1', 'status': 'RUNNING', 'supervisor_pid': os.getpid(),
              'plan_sha256': c.sha(plan_path), 'source_identity': plan['source_identity'],
              'stages': {}, 'active_stage': None, 'actual_counts': None, 'known_actual_counts': dict(ZERO),
              'cleanup_complete': True, 'error': None, 'original_36_pass': False,
              'holdout_36_pass': False, 'full_delay_qualification': False, 'automatic_model_replacement': False}
    c.atomic_json(output/'report.json', report)
    process, current = None, None
    def sample(admission=False):
        row = dict(memory_reader(), elapsed_s=clock()-started, stage=current)
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        c.require(row['effective_available_bytes'] >= (6 if admission else 4) << 30, 'RAM below 6/4 GiB')
        c.require(clock()-started < c.RESOURCES['total_wall_s'], 'total22 wall limit')
    with termination_guard() as control:
        try:
            adopter()
            for stage, spec in c.STAGES.items():
                if stage == 'holdout' and not report['original_36_pass']:
                    report['stages'][stage] = {'status': 'SKIPPED_NO_QUALIFIED_CANDIDATE', 'actual_counts': dict(ZERO)}
                    continue
                current = stage
                sample(True)
                c.require(loader(plan_path) == plan, 'plan22 changed before stage')
                for old in report['stages'].values():
                    if old.get('status') == 'COMPLETE':
                        c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'earlier22 artifact changed')
                argv = c.command(Path(plan_path), stage, output/stage)
                record = {'status': 'STARTED', 'nonce': secrets.token_hex(24), 'command': argv, 'actual_counts': None}
                report['stages'][stage], report['active_stage'] = record, stage
                c.atomic_json(output/'report.json', report)
                stage_started = clock()
                try:
                    with (output/(stage+'.stdout.log')).open('xb') as stream:
                        control['defer'] = True
                        try:
                            process = factory(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ, D1_STAGE22_NONCE=record['nonce'],
                                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                    PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT)))))
                        finally:
                            control['defer'] = False
                        c.require(control['received'] is None, 'termination during22 spawn')
                        while process.poll() is None:
                            sample()
                            c.require(clock()-stage_started < spec['max_wall_s'], 'child22 wall limit')
                            sleeper(1.)
                        sample()
                        c.require(process.returncode == 0 and clock()-stage_started < spec['max_wall_s'], 'child22 failed/late')
                finally:
                    control['defer'] = True
                    receipt = cleanup(process)
                    record['cleanup_complete'] = receipt.get('remaining') == {}
                    report['cleanup_complete'] &= record['cleanup_complete']
                    process = None
                    control['defer'] = False
                    c.require(record['cleanup_complete'], 'owned22 cleanup incomplete')
                c.require(control['received'] is None, 'termination during22 cleanup')
                child = c.read_json(output/stage/('result.json' if stage == 'train' else 'report.json'))
                actual = verifier(plan, plan_path, stage, output/stage, child)
                record.update(status='COMPLETE', actual_counts=actual, artifacts_sha256=prior.artifacts(output/stage))
                report['known_actual_counts'] = totals(report['stages'])
                projected_counts(report['known_actual_counts'], plan['budget'])
                if stage == 'formal':
                    report['original_36_pass'] = child['all_36_reference_pass']
                elif stage == 'holdout':
                    report['holdout_36_pass'] = child['all_36_reference_pass']
                current = report['active_stage'] = None
                c.atomic_json(output/'report.json', report)
            c.require(loader(plan_path) == plan, 'final22 inputs changed')
            for old in report['stages'].values():
                if old.get('status') == 'COMPLETE':
                    c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'final22 artifact changed')
            sample()
            report.update(status='COMPLETE', full_delay_qualification=report['original_36_pass'] and report['holdout_36_pass'])
        except BaseException as exc:
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            if current in report['stages']:
                record = report['stages'][current]
                record['status'] = 'INCOMPLETE'
                try:
                    child = c.read_json(output/current/('result.json' if current == 'train' else 'report.json'))
                    c.require(child.get('plan_sha256') == c.sha(plan_path)
                              and child.get('source_identity') == plan['source_identity'] and child.get('stage') == current,
                              'foreign partial22 receipt')
                    protocol_path = output/current/'protocol.json'
                    protocol = c.read_json(protocol_path)
                    c.require(child.get('protocol_sha256') == c.sha(protocol_path)
                              and protocol.get('plan_sha256') == c.sha(plan_path)
                              and protocol.get('stage') == current
                              and protocol.get('source_identity') == plan['source_identity'], 'partial22 protocol differs')
                    cap = c.expected_counts(c.STAGES[current]['transitions'])
                    if child.get('actual_counts') is not None:
                        record['actual_counts'] = projected_counts(child['actual_counts'], cap, partial=True)
                    else:
                        record['known_actual_counts'] = projected_counts(child.get('known_actual_counts', ZERO), cap, partial=True)
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            try:
                report['cleanup_complete'] &= cleanup(process).get('remaining') == {}
            except BaseException as exc:
                report.update(cleanup_complete=False, error=(report['error'] or '')+f'; cleanup {exc}')
            report['elapsed_s'] = clock()-started
            if not report['cleanup_complete'] or control['received'] or report['elapsed_s'] >= c.RESOURCES['total_wall_s']:
                report.update(status='INCOMPLETE', full_delay_qualification=False)
            report['known_actual_counts'] = totals(report['stages'])
            report['actual_counts'] = (report['known_actual_counts'] if all(
                row.get('actual_counts') is not None for row in report['stages'].values()) else None)
            c.atomic_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=c.ROOT/c.PLAN)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stage', choices=('export', 'cpp', 'formal', 'holdout'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    c.require(not (args.execute and args.stage), 'choose stage or execute')
    if args.stage:
        c.require(args.output is not None, 'supervised output required')
        value = run_stage(args.plan.resolve(), args.stage, args.output.absolute())
    elif args.execute:
        value = execute(args.plan.resolve())
    else:
        value = {'status': 'PLAN_ONLY', 'budget': c.expected_counts(c.TRANSITIONS+43200),
                 'new_training_transitions': c.TRANSITIONS, 'execution_started': False}
    print(json.dumps({k: value.get(k) for k in ('status', 'actual_counts', 'full_delay_qualification', 'error', 'budget')}))
    raise SystemExit(0 if value['status'] in ('COMPLETE', 'PLAN_ONLY', 'NOT_READY') else 2)
