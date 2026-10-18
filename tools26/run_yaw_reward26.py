"""Own one source-bound yaw-reward comparison and a single preselected holdout."""
from __future__ import annotations

import argparse
import json
import math
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
from tools26 import common26 as c  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402
from tools19.run_cpp_closedloop19 import cleanup_owned_tree, subreaper  # noqa: E402
from tools21 import closeout_completed18 as prior  # noqa: E402
from tools25.run_delay_comparison25 import verify_numeric  # noqa: E402

ZERO = c.expected_counts(0)
CAP = c.expected_counts(21600)


def identity(plan, path, stage):
    arm, _ = c.split_stage(stage)
    return {'format': 'd1-yaw-reward26-stage-v1', 'stage': stage, 'arm': arm,
            'mode': 'uniform_0_3', 'yaw_tracking_weight': c.ARMS[arm],
            'plan_sha256': c.sha(path), 'source_identity': plan['source_identity']}


def projected_counts(value, cap, *, partial=False):
    c.require(prior.valid_counts(value, cap, partial=partial), 'invalid or unknown26 counters')
    return {key: value[key] for key in ZERO}


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def completed_training(plan, path, arm):
    return c.check_completed_training(plan, path, arm)


def check_model(model, protocol, versions):
    c.require(type(model.num_timesteps) is int and model.num_timesteps == c.INITIAL+c.TRANSITIONS
              and type(model.aux_coef) is float and model.aux_coef == 0. and model.seed == 182
              and model.n_envs == 1 and str(model.device) == 'cpu' and model.get_env() is None,
              'loaded model is not completed26')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                'vf_coef', 'max_grad_norm', 'learning_rate'):
        c.require(getattr(model, key) == c.HYPERPARAMETERS[key], 'loaded26 PPO differs: '+key)
    c.require(model.clip_range(0.) == model.clip_range(1.) == .2 and model.normalize_advantage is True
              and model.clip_range_vf is None and model.target_kl == .005 and model.use_sde is False
              and versions == protocol['versions'], 'unsupported26 model/runtime')


def numerical(kind, bundle, parent, output):
    if kind == 'cpp':
        return prior.numerical(kind, bundle, parent, output)
    from tools17 import export_completed17 as export
    model, versions = export.load_model(bundle['checkpoint'])
    check_model(model, bundle['protocol'], versions)
    value = export.export_numerical(model, output)
    c.atomic_json(output/'golden_fixtures.json', export.cpp_fixtures(value))
    return {'onnx_sha256': c.sha(output/'policy.onnx'), 'fixtures_sha256': c.sha(output/'golden_fixtures.json'),
            'absolute_limits': export.LIMITS, 'versions': {**versions, **value['numerical_versions']},
            'onnx_ir_version': 7, 'onnx_opset': 11, 'cxx_ort_1_10_status': 'NOT_TESTED',
            **{key: value[key] for key in ('onnx_node_counts', 'onnx_signature', 'parity', 'policy_state_sha256')}}


def evaluation_artifacts(parent):
    return {name: c.sha(parent/name) for name in (
        'export/report.json', 'export/policy.onnx', 'export/golden_fixtures.json', 'cpp/report.json')}


def verify_stage(plan, path, stage, directory, child):
    arm, kind = c.split_stage(stage)
    bundle = completed_training(plan, path, arm)
    if kind == 'train':
        c.require(child == bundle['result'], 'training26 reader returned another receipt')
        return projected_counts(child.get('actual_counts'), c.expected_counts(c.TRANSITIONS))
    protocol = c.read_json(directory/'protocol.json')
    c.require(all(same_json(value.get(key), expected) for value in (child, protocol)
                  for key, expected in identity(plan, path, stage).items())
              and child.get('status') == 'COMPLETE' and child.get('error') is None
              and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
              and child.get('protocol_sha256') == c.sha(directory/'protocol.json')
              and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s'], 'stage26 identity/status differs')
    c.require(all(value.get('completed_training') == bundle['identity']
                  and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                  for value in (child, protocol)), 'stage26 expected final checkpoint differs')
    cap = CAP if kind in ('formal', 'holdout') else ZERO
    c.require(protocol.get('budget_max') == cap, 'stage26 physical cap differs')
    actual = projected_counts(child.get('actual_counts'), cap)
    projected_counts(child.get('known_actual_counts'), cap)
    c.require(same_json(child.get('known_actual_counts'), child.get('actual_counts')), 'stage26 known/exact differs')
    if kind in ('export', 'cpp'):
        verify_numeric(kind, child, directory, old_release=bundle['old_release'])
        return actual
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import verify_trace
    helpers = frozen_helpers(bundle['old_release'])
    bridge = evaluation_artifacts(directory.parent)
    c.require(all(value.get('numerical_predecessors') == bridge for value in (child, protocol)),
              'evaluation26 numerical predecessor differs')
    verify_numeric('cpp', c.read_json(directory.parent/'cpp/report.json'), directory.parent/'cpp',
                   old_release=bundle['old_release'])
    c.require(protocol.get('thresholds') == helpers.THRESHOLDS and protocol.get('sequences') == helpers.SEQUENCES
              and protocol.get('backend') == child.get('backend')
              and [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in child['cases']]
                  == prior.matrix(kind), 'evaluation26 original matrix/backend differs')
    total = dict(ZERO)
    for row in child['cases']:
        verify_trace(directory, row, helpers)
        delta = projected_counts(row.get('actual_counts'), CAP)
        total = {key: total[key]+delta[key] for key in ZERO}
    full = total == CAP and all(row['reference_thresholds_met'] is True and row['failure'] is False
                               and row['done_reason'] == 'HORIZON_12S' for row in child['cases'])
    c.require(total == actual and child.get('all_36_reference_pass') is full,
              'evaluation26 totals/qualification differs')
    return actual


def run_stage(plan_path, arm, kind, output, *, loader=c.load_plan,
              launch_validator=c.validate_launch, memory_reader=c.read_memory, clock=time.monotonic,
              numeric=numerical, evaluation=prior.evaluate, training_loader=completed_training):
    c.require(arm in c.ARMS and kind in ('export', 'cpp', 'formal', 'holdout'), 'unknown26 stage')
    started = clock()
    stage = kind+'_'+arm
    plan = loader(plan_path)
    launch_validator(plan, plan_path, stage, output)
    bundle = training_loader(plan, plan_path, arm)
    c.require(not output.exists() and not output.is_symlink(), 'new26 stage output required')
    output.mkdir(exist_ok=False)
    report = dict(identity(plan, plan_path, stage), status='INCOMPLETE', error=None, cases=[],
                  actual_counts=None, known_actual_counts=dict(ZERO), cleanup_complete=True,
                  inputs_unchanged=False, all_36_reference_pass=False,
                  completed_training=bundle['identity'],
                  checkpoint_sha256=bundle['identity']['training_identity']['checkpoint_sha256'])
    last = -float('inf')

    def poll(force=False, admission=False):
        nonlocal last
        now = clock()
        c.require(now-started < c.STAGES[stage]['max_wall_s'], 'stage26 wall limit')
        if force or now-last >= 1.:
            sample = dict(memory_reader(), elapsed_s=now-started)
            with (output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            limit = c.RESOURCES['start_min_effective_bytes' if admission else 'reserve_bytes']
            c.require(type(sample.get('effective_available_bytes')) is int
                      and sample['effective_available_bytes'] >= limit, 'stage26 RAM below 6/4 GiB')
            last = now

    with termination_guard() as control:
        try:
            poll(True, True)
            c.atomic_json(output/'protocol.json', dict(identity(plan, plan_path, stage),
                max_wall_s=c.STAGES[stage]['max_wall_s'], budget_max=CAP if kind in ('formal', 'holdout') else ZERO,
                completed_training=bundle['identity'], checkpoint_sha256=report['checkpoint_sha256']))
            report['protocol_sha256'] = c.sha(output/'protocol.json')
            if kind in ('export', 'cpp'):
                report.update(numeric(kind, bundle, output.parent, output))
                report['actual_counts'] = report['known_actual_counts'] = dict(ZERO)
            else:
                bridge = report['numerical_predecessors'] = evaluation_artifacts(output.parent)
                protocol = c.read_json(output/'protocol.json')
                protocol['numerical_predecessors'] = bridge
                c.atomic_json(output/'protocol.json', protocol)
                report['protocol_sha256'] = c.sha(output/'protocol.json')
                evaluation(kind, bundle, output.parent, output, report, poll)
            c.require(loader(plan_path) == plan and training_loader(plan, plan_path, arm)['identity'] == bundle['identity'],
                      'stage26 inputs changed')
            poll(True)
            c.require(report['cleanup_complete'] is True, 'stage26 cleanup incomplete')
            report.update(status='COMPLETE', inputs_unchanged=True)
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            report['elapsed_s'] = clock()-started
            if control['received'] or report['elapsed_s'] >= c.STAGES[stage]['max_wall_s']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; interrupted/wall limit')
            c.atomic_json(output/'report.json', report)
    return report


def totals(stages):
    return prior.known_counts(stages)


def paired_initials(plan, path):
    bundles = [completed_training(plan, path, arm) for arm in c.ARMS]
    initial = bundles[0]['protocol']['initial_state']
    c.require(same_json(initial, bundles[1]['protocol']['initial_state'])
              and same_json(bundles[0]['protocol']['versions'], bundles[1]['protocol']['versions']),
              'two26 arms did not start with identical policy/Adam/runtime')
    rollout = bundles[0]['result'].get('first_rollout_nonreward')
    c.require(isinstance(rollout, dict) and same_json(rollout, bundles[1]['result'].get('first_rollout_nonreward')),
              'first26 rollout non-reward trajectories differ before PPO update')
    return {'initial_state': initial, 'first_rollout_nonreward': rollout, 'first_rollout_nonreward_equal': True}


def case_table(report):
    rows = report['cases']
    keys = [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in rows]
    c.require(keys == prior.matrix('formal') and all(type(row['policy_transitions']) is int
              and 1 <= row['policy_transitions'] <= 600 and type(row['J']) in (int, float)
              and math.isfinite(row['J']) and row['J'] >= 0 for row in rows), 'comparison26 original matrix/J differs')
    return dict(zip(keys, rows))


def compare_reports(reports):
    tables = {name: case_table(report) for name, report in reports.items()}
    common = [key for key in prior.matrix('formal') if all(table[key]['policy_transitions'] == 600
              and table[key].get('done_reason') == 'HORIZON_12S' for table in tables.values())]
    return {'common_full_cases': len(common), 'common_full_keys': common,
            'common_full_mean_J': {name: sum(table[key]['J'] for key in common)/len(common)
                                  if common else None for name, table in tables.items()},
            'reference_pass_counts': {name: sum(row['reference_thresholds_met'] is True for row in table.values())
                                      for name, table in tables.items()},
            'full_horizon_counts': {name: sum(row['policy_transitions'] == 600 and row.get('done_reason') == 'HORIZON_12S'
                                            for row in table.values()) for name, table in tables.items()},
            'all36_mean_J_including_early_failure_penalties': {
                name: sum(row['J'] for row in table.values())/36 for name, table in tables.items()},
            'scope': 'Descriptive common-horizon comparison only; no multi-training-seed significance or new qualification gate.'}


def load_references(plan):
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import verify_trace
    helpers = frozen_helpers(c.base_inputs()[1])
    c.require(set(plan['reference_reports']) == {'baseline17', 'random25'}, 'reference26 set differs')
    reports = {}
    for name, proof in plan['reference_reports'].items():
        path = ROOT/proof['path']
        c.require(not path.is_symlink() and path.resolve().is_relative_to(ROOT) and c.sha(path) == proof['sha256'],
                  'reference26 report changed')
        report = c.read_json(path)
        c.require(report.get('status') == 'COMPLETE' and report.get('cleanup_complete') is True,
                  'reference26 not engineering-complete')
        case_table(report)
        for row in report['cases']:
            verify_trace(path.parent, row, helpers)
        reports[name] = report
    return reports


def partial_counts(plan, path, stage, directory, child):
    arm, kind = c.split_stage(stage)
    protocol_path = directory/'protocol.json'
    protocol = c.read_json(protocol_path)
    base = {'plan_sha256': c.sha(path), 'source_identity': plan['source_identity'],
            'stage': stage, 'arm': arm, 'mode': 'uniform_0_3', 'yaw_tracking_weight': c.ARMS[arm]}
    c.require(all(same_json(value.get(key), expected) for value in (child, protocol) for key, expected in base.items())
              and child.get('protocol_sha256') == c.sha(protocol_path), 'foreign26 partial receipt')
    if kind == 'train':
        c.require(child.get('format') == 'd1-yaw-reward26-training-result-v1'
                  and protocol.get('format') == 'd1-yaw-reward26-training-protocol-v1'
                  and child.get('seed') == protocol.get('seed') == 182
                  and child.get('envs') == protocol.get('envs') == 1
                  and all(same_json(value.get('hyperparameters'), c.HYPERPARAMETERS)
                          and same_json(value.get('vector_layout'), plan['vector_layout'])
                          and same_json(value.get('runtime_dependencies_sha256'), plan['runtime_dependencies_sha256'])
                          and same_json(value.get('reward_change'), c.reward_change(arm)) for value in (child, protocol))
                  and same_json(protocol.get('base_checkpoint'), plan['base_checkpoint'])
                  and same_json(protocol.get('base_protocol'), plan['base_protocol']),
                  'partial26 training provenance differs')
    else:
        bundle = completed_training(plan, path, arm)
        c.require(child.get('format') == protocol.get('format') == 'd1-yaw-reward26-stage-v1'
                  and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s']
                  and protocol.get('budget_max') == (CAP if kind in ('formal', 'holdout') else ZERO)
                  and all(value.get('completed_training') == bundle['identity']
                          and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                          for value in (child, protocol)), 'partial26 stage scope differs')
        if kind in ('formal', 'holdout'):
            c.require(all(value.get('numerical_predecessors') == evaluation_artifacts(directory.parent)
                          for value in (child, protocol)), 'foreign partial26 numerical inputs')
    cap = c.expected_counts(c.STAGES[stage]['transitions'])
    actual = child.get('actual_counts')
    if actual is not None:
        projected_counts(child.get('known_actual_counts'), cap, partial=True)
        c.require(same_json(child.get('known_actual_counts'), actual), 'partial26 exact/known disagree')
        return projected_counts(actual, cap, partial=True), None
    return None, projected_counts(child.get('known_actual_counts', ZERO), cap, partial=True)


def execute(plan_path=c.ROOT/c.PLAN, *, loader=c.load_plan, memory_reader=c.read_memory,
            factory=subprocess.Popen, cleanup=cleanup_owned_tree, adopter=subreaper,
            verifier=verify_stage, initial_checker=paired_initials, reference_loader=load_references,
            clock=time.monotonic, sleeper=time.sleep):
    started = clock()
    plan = loader(plan_path)
    output, plan_path = ROOT/plan['output_root'], Path(plan_path)
    c.require(not output.exists() and not output.is_symlink(), 'attempt26 already exists; no retry')
    c.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '26 requires external Nice10')
    initial_sample = dict(memory_reader())
    memory = initial_sample.get('effective_available_bytes')
    c.require(type(memory) is int and memory >= 0, 'effective26 RAM unknown')
    if memory < c.RESOURCES['start_min_effective_bytes']:
        return {'status': 'NOT_READY', 'execution_started': False, 'actual_counts': dict(ZERO),
                'readiness_sample': initial_sample}
    plan_sha = c.sha(plan_path)
    report = {'format': c.EXECUTION_FORMAT, 'status': 'RUNNING', 'supervisor_pid': os.getpid(),
              'plan_sha256': plan_sha, 'source_identity': plan['source_identity'], 'stages': {},
              'active_stage': None, 'actual_counts': None, 'known_actual_counts': dict(ZERO),
              'cleanup_complete': True, 'error': None, 'automatic_model_replacement': False,
              'admission_waits': [], 'original_36_pass': {arm: False for arm in c.ARMS},
              'selection': None, 'selected_arm': None, 'holdout_36_pass': False,
              'initial_resource_sample': initial_sample, 'minimum_effective_available_bytes': memory,
              'stop_sample': None, 'stop_reason': None,
              'simulation_candidate_qualified': False, 'cpp_closed_loop_qualified': False,
              'ros_hardware_qualified': False}
    process = current = None
    formals, holdout = {}, None

    def sample():
        row = dict(memory_reader(), elapsed_s=clock()-started, stage=current)
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        value = row.get('effective_available_bytes')
        if type(value) is int and value >= 0:
            report['minimum_effective_available_bytes'] = min(report['minimum_effective_available_bytes'], value)
        if type(value) is not int or value < c.RESOURCES['reserve_bytes']:
            unknown = type(value) is not int or value < 0
            report['stop_sample'] = row
            report['stop_reason'] = 'EFFECTIVE_MEMORY_UNKNOWN' if unknown else 'RAM_BELOW_4_GIB'
        c.require(type(value) is int and value >= c.RESOURCES['reserve_bytes'], 'RAM below 4 GiB or unknown')
        if clock()-started >= c.RESOURCES['total_wall_s']:
            report['stop_sample'], report['stop_reason'] = row, 'TOTAL_WALL_LIMIT'
        c.require(clock()-started < c.RESOURCES['total_wall_s'], 'total26 wall limit')
        return row

    def admit():
        c.require(process is None and report['cleanup_complete'] is True and current not in report['stages'],
                  'admission26 requires closed prior stages')
        wait_started, wait = clock(), None
        try:
            while True:
                row = sample()
                duration = clock()-wait_started
                if wait is not None:
                    wait.update(duration_s=duration)
                    wait['samples'].append(row)
                if row['effective_available_bytes'] >= c.RESOURCES['start_min_effective_bytes']:
                    c.require(duration <= c.RESOURCES['interstage_admission_wait_s'], 'admission26 wait exceeded')
                    if wait is not None:
                        wait['status'] = 'ADMITTED'
                        c.atomic_json(output/'report.json', report)
                    return
                if wait is None:
                    wait = {'stage': current, 'status': 'WAITING_FOR_ADMISSION', 'started_elapsed_s': clock()-started,
                            'duration_s': duration, 'samples': [row]}
                    report['admission_waits'].append(wait)
                c.require(duration < c.RESOURCES['interstage_admission_wait_s'], 'admission26 wait exhausted')
                c.atomic_json(output/'report.json', report)
                duration = min(c.RESOURCES['admission_poll_s'], c.RESOURCES['interstage_admission_wait_s']-duration,
                               c.RESOURCES['total_wall_s']-(clock()-started))
                c.require(duration > 0, 'total26 wall limit during admission')
                sleeper(duration)
        except BaseException as exc:
            if wait is not None:
                wait.update(status='STOPPED', duration_s=clock()-wait_started, error=f'{type(exc).__name__}: {exc}')
            raise

    with termination_guard() as control:
        try:
            control['defer'] = True
            output.mkdir(parents=True, exist_ok=False)
            c.atomic_json(output/'report.json', report)
            control['defer'] = False
            c.require(control['received'] is None, 'termination during26 claim')
            adopter()
            references = reference_loader(plan)
            for stage in c.STAGE_ORDER:
                arm, kind = c.split_stage(stage)
                spec, directory = c.STAGES[stage], c.stage_path(plan, stage)
                if kind == 'holdout' and arm != report['selected_arm']:
                    status = 'SKIPPED_NO_QUALIFIED_CANDIDATE' if report['selected_arm'] is None else 'SKIPPED_NOT_SELECTED'
                    report['stages'][stage] = {'status': status, 'actual_counts': dict(ZERO)}
                    continue
                current = stage
                admit()
                c.require(loader(plan_path) == plan and c.sha(plan_path) == plan_sha, 'plan26 changed before stage')
                for old in report['stages'].values():
                    if old.get('status') == 'COMPLETE':
                        c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()),
                                  'earlier26 artifact changed')
                c.require(clock()-started < c.RESOURCES['total_wall_s'], 'total26 wall limit before claim')
                argv = c.command(plan_path, stage, directory)
                record = {'status': 'STARTED', 'nonce': secrets.token_hex(24), 'command': argv, 'actual_counts': None}
                report['stages'][stage], report['active_stage'] = record, stage
                c.atomic_json(output/'report.json', report)
                stage_started = clock()
                try:
                    with (output/(stage+'.stdout.log')).open('xb') as stream:
                        control['defer'] = True
                        try:
                            process = factory(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ, D1_STAGE26_NONCE=record['nonce'],
                                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                    PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT)))))
                        finally:
                            control['defer'] = False
                        c.require(control['received'] is None, 'termination during26 spawn')
                        while process.poll() is None:
                            sample()
                            c.require(clock()-stage_started < spec['max_wall_s'], 'child26 wall limit')
                            sleeper(1.)
                        sample()
                        c.require(process.returncode == 0 and clock()-stage_started < spec['max_wall_s'], 'child26 failed/late')
                finally:
                    control['defer'] = True
                    receipt = cleanup(process)
                    record['cleanup_complete'] = receipt.get('remaining') == {}
                    report['cleanup_complete'] &= record['cleanup_complete']
                    process = None
                    control['defer'] = False
                    c.require(record['cleanup_complete'], 'owned26 cleanup incomplete')
                c.require(control['received'] is None, 'termination during26 cleanup')
                child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                actual = verifier(plan, plan_path, stage, directory, child)
                record.update(status='COMPLETE', actual_counts=actual, artifacts_sha256=prior.artifacts(directory))
                c.require(clock()-stage_started < spec['max_wall_s'], 'child26 late cleanup/readback')
                report['known_actual_counts'] = totals(report['stages'])
                projected_counts(report['known_actual_counts'], plan['budget'])
                if stage == 'train_yaw3':
                    paired = initial_checker(plan, plan_path)
                    report.update(paired_initial_state=paired['initial_state'],
                                  first_rollout_nonreward=paired['first_rollout_nonreward'],
                                  first_rollout_nonreward_equal=paired['first_rollout_nonreward_equal'])
                if kind == 'formal':
                    report['original_36_pass'][arm] = child['all_36_reference_pass']
                    formals[arm] = child
                    if len(formals) == len(c.ARMS):
                        report['selection'] = c.select_candidate(formals)
                        report['selected_arm'] = report['selection']['selected_arm']
                elif kind == 'holdout':
                    c.require(holdout is None and arm == report['selected_arm'], 'more than one26 holdout')
                    holdout = child
                    report['holdout_36_pass'] = child['all_36_reference_pass']
                current = report['active_stage'] = None
                c.atomic_json(output/'report.json', report)
            c.require(loader(plan_path) == plan and c.sha(plan_path) == plan_sha, 'final26 inputs changed')
            c.require(reference_loader(plan) == references, 'comparison26 reference changed')
            for old in report['stages'].values():
                if old.get('status') == 'COMPLETE':
                    c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'final26 artifact changed')
            sample()
            c.require(report['selection'] == c.select_candidate(formals), 'holdout influenced26 selection')
            report.update(status='COMPLETE', comparison=compare_reports({**references, **formals}),
                          paired_arm_comparison=compare_reports(formals),
                          reference_comparisons={name: compare_reports({name: reference, **formals})
                                                 for name, reference in references.items()},
                          simulation_candidate_qualified=report['selected_arm'] is not None and report['holdout_36_pass'] is True)
        except BaseException as exc:
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}', simulation_candidate_qualified=False)
            if current in report['stages']:
                record = report['stages'][current]
                record['status'] = 'INCOMPLETE'
                try:
                    directory = c.stage_path(plan, current)
                    _, kind = c.split_stage(current)
                    child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                    actual, known = partial_counts(plan, plan_path, current, directory, child)
                    if record.get('actual_counts') is None:
                        record['actual_counts'] = actual
                        if known is not None:
                            record['known_actual_counts'] = known
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
                report.update(status='INCOMPLETE', simulation_candidate_qualified=False)
            report['known_actual_counts'] = totals(report['stages'])
            report['actual_counts'] = (report['known_actual_counts'] if all(
                row.get('actual_counts') is not None for row in report['stages'].values()) else None)
            if output.is_dir():
                c.atomic_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=c.ROOT/c.PLAN)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arm', choices=tuple(c.ARMS))
    parser.add_argument('--stage', choices=('export', 'cpp', 'formal', 'holdout'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    c.require(not (args.execute and args.stage), 'choose stage or execute')
    if args.stage:
        c.require(args.output is not None and args.arm is not None, 'supervised arm/output required')
        value = run_stage(args.plan.resolve(), args.arm, args.stage, args.output.absolute())
    elif args.execute:
        value = execute(args.plan.resolve())
    else:
        value = {'status': 'PLAN_ONLY', 'budget': c.expected_counts(326944),
                 'new_training_transitions': c.TRANSITIONS*2, 'execution_started': False,
                 'physics_steps_executed': 0, 'cpp_closed_loop_qualified': False}
    print(json.dumps({key: value.get(key) for key in ('status', 'actual_counts', 'selection', 'error', 'budget',
        'execution_started', 'physics_steps_executed', 'cpp_closed_loop_qualified')}))
    return 0 if value['status'] in ('COMPLETE', 'PLAN_ONLY', 'NOT_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
