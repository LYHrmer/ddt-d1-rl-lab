"""Own one bounded two-arm25 attempt and unchanged numerical/formal checks."""
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
from tools25 import common25 as c  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402
from tools19.run_cpp_closedloop19 import cleanup_owned_tree, subreaper  # noqa: E402
from tools21 import closeout_completed18 as prior  # noqa: E402

ZERO = c.expected_counts(0)
CAP = c.expected_counts(21600)


def identity(plan, path, stage):
    arm, kind = c.split_stage(stage)
    return {'format': 'd1-delay-comparison25-stage-v1', 'stage': stage, 'arm': arm,
            'mode': c.ARMS[arm], 'plan_sha256': c.sha(path),
            'source_identity': plan['source_identity']}


def projected_counts(value, cap, *, partial=False):
    c.require(prior.valid_counts(value, cap, partial=partial), 'invalid or unknown25 counters')
    return {key: value[key] for key in ZERO}


def completed_training(plan, plan_path, arm):
    c.require(arm in c.ARMS, 'unknown25 arm')
    directory = ROOT/plan['output_root']/arm/'train'
    result, protocol = c.read_json(directory/'result.json'), c.read_json(directory/'protocol.json')
    checkpoint = directory/'policy.zip'
    common = {'stage': 'train_'+arm, 'arm': arm, 'mode': c.ARMS[arm], 'seed': 181, 'envs': 1,
              'plan_sha256': c.sha(plan_path), 'source_identity': plan['source_identity'],
              'hyperparameters': c.HYPERPARAMETERS}
    c.require(all(json.dumps(item.get(k), sort_keys=True, allow_nan=False)
                  == json.dumps(v, sort_keys=True, allow_nan=False)
                  for item in (result, protocol) for k, v in common.items()),
              'training25 identity/settings differ')
    c.require(result.get('format') == 'd1-delay-comparison25-training-result-v1'
              and protocol.get('format') == 'd1-delay-comparison25-training-protocol-v1'
              and result.get('status') == 'COMPLETE' and result.get('error') is None
              and result.get('eligible_for_evaluation') is True and result.get('cleanup_complete') is True
              and result.get('inputs_unchanged') is True and result.get('aux_coef') == 0.
              and result.get('initial_num_timesteps') == c.INITIAL
              and result.get('new_trained_transitions') == c.TRANSITIONS
              and result.get('final_num_timesteps') == c.INITIAL+c.TRANSITIONS
              and result.get('protocol_sha256') == c.sha(directory/'protocol.json')
              and checkpoint.is_file() and not checkpoint.is_symlink()
              and result.get('checkpoint_sha256') == c.sha(checkpoint), 'incomplete/partial25 training')
    from tools25.low_memory_runtime25 import checked_worker_counts25
    from tools16.train_comparison16 import complete_counts
    from tools18.train_delay18 import valid_delay_state
    workers = result.get('worker_counts')
    c.require(workers == result.get('per_worker_counts')
              and checked_worker_counts25(workers) == result.get('actual_counts')
              and checked_worker_counts25([result.get('actual_counts')])
                  == checked_worker_counts25([result.get('known_actual_counts')])
              and result.get('known_actual_counts') == result.get('actual_counts')
              and complete_counts(workers, 1, c.TRANSITIONS), 'single25 training counts incomplete')
    c.require(valid_delay_state(result.get('delay_state18'), 1, c.ARMS[arm]),
              'single25 final delay provenance incomplete')
    c.require(protocol.get('base_checkpoint') == plan['base_checkpoint']
              and protocol.get('base_protocol') == plan['base_protocol']
              and protocol.get('planned_new_transitions') == c.TRANSITIONS
              and protocol.get('planned_native_steps') == c.TRANSITIONS*10
              and protocol.get('warm_start') is True and protocol.get('reset_num_timesteps') is False
              and protocol.get('teacher_policy_used') is False, 'training25 provenance differs')
    initial = result.get('initial_state', {})
    old = plan['base_initial_state']
    c.require(initial.get('optimizer_restored') is True and all(initial.get(k) == old[k]
              for k in ('num_timesteps', 'policy_state_sha256', 'optimizer_state_sha256', 'updates', 'optimizer_restored')),
              'original complete policy/Adam not restored')
    c.require(initial.get('optimizer_momentum_preserved') is True
              and initial.get('learning_rate_before') == 3e-4 and initial.get('learning_rate_after') == 1e-4
              and initial.get('optimizer_state_sha256_before_lr') == initial['optimizer_state_sha256']
              and initial.get('optimizer_without_lr_sha256_before') == initial.get('optimizer_without_lr_sha256_after')
              and isinstance(initial.get('optimizer_without_lr_sha256_before'), str)
              and len(initial['optimizer_without_lr_sha256_before']) == 64,
              'LR25 change lacks preserved optimizer momentum proof')
    from tools25.train_delay25 import initial_identity_valid
    c.require(initial_identity_valid(initial) and protocol.get('initial_state') == initial
              and protocol.get('checkpoint_selection') == result.get('checkpoint_selection') == 'final_only'
              and result.get('single_process_guard25', {}).get('dirty') is False
              and result['single_process_guard25'].get('closed') is True
              and result['single_process_guard25'].get('backend') == 'DummyVecEnv'
              and result['single_process_guard25'].get('interprocess_rpc') is False
              and type(result['single_process_guard25'].get('environment_subprocesses')) is int
              and result['single_process_guard25']['environment_subprocesses'] == 0
              and result['single_process_guard25'].get('waiting') is False
              and result['single_process_guard25'].get('inflight') is None
              and result['single_process_guard25'].get('construction_complete') is True
              and all(value.get('vector_layout') == plan['vector_layout']
                      and value.get('runtime_dependencies_sha256') == plan['runtime_dependencies_sha256']
                      for value in (result, protocol)), 'KL25/single-process proof incomplete')
    guard = result['single_process_guard25']
    c.require(all(type(guard.get(key)) is int and guard[key] > 0
                  for key in ('metadata_attempts', 'metadata_returns', 'metadata_validated'))
              and guard['metadata_attempts'] == guard['metadata_returns'] == guard['metadata_validated'],
              'single25 metadata validation incomplete')
    _, release, original_protocol = c.base_inputs()
    c.require(result.get('versions') == protocol.get('versions') == original_protocol['versions'],
              'single25 baseline/runtime versions differ')
    bound = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': c.sha(checkpoint),
             'training_protocol_sha256': c.sha(directory/'protocol.json'),
             'training_result_sha256': c.sha(directory/'result.json')}
    return {'checkpoint': checkpoint, 'protocol': protocol, 'old_release': release,
            'identity': {'role': 'completed_delay_comparison25_'+arm+'_final_only', 'training_identity': bound}}


def check_model(model, protocol, versions):
    c.require(type(model.num_timesteps) is int and model.num_timesteps == c.INITIAL+c.TRANSITIONS
              and type(model.aux_coef) is float and model.aux_coef == 0. and model.seed == 181
              and model.n_envs == 1 and str(model.device) == 'cpu' and model.get_env() is None,
              'loaded model is not completed25')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                'vf_coef', 'max_grad_norm', 'learning_rate'):
        c.require(getattr(model, key) == c.HYPERPARAMETERS[key], 'loaded25 PPO differs: '+key)
    c.require(model.clip_range(0.) == model.clip_range(1.) == .2 and model.normalize_advantage is True
              and model.clip_range_vf is None and model.target_kl == .005 and model.use_sde is False
              and versions == protocol['versions'], 'unsupported25 model/runtime')


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
              'numeric25 artifacts differ')
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
                  and child.get('cxx_ort_1_10_status') == 'PASS', 'CPP25 proof differs')
        maximum = cpp.check_native(c.read_json(native), fixtures, child['identities'])
        c.require(child.get('max_absolute_error') == maximum, 'CPP25 numerical result differs')


def verify_stage(plan, plan_path, stage, directory, child):
    arm, kind = c.split_stage(stage)
    if kind == 'train':
        completed_training(plan, plan_path, arm)
        return projected_counts(child.get('actual_counts'), c.expected_counts(c.TRANSITIONS))
    c.require(all(child.get(k) == v for k, v in identity(plan, plan_path, stage).items())
              and child.get('status') == 'COMPLETE' and child.get('error') is None
              and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
              and child.get('protocol_sha256') == c.sha(directory/'protocol.json'), 'stage25 proof differs')
    protocol = c.read_json(directory/'protocol.json')
    bundle = completed_training(plan, plan_path, arm)
    c.require(all(protocol.get(k) == v for k, v in identity(plan, plan_path, stage).items())
              and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s']
              and all(value.get('completed_training') == bundle['identity']
                      and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                      for value in (protocol, child)), 'stage25 protocol/model differs')
    cap = CAP if kind in ('formal', 'holdout') else ZERO
    counts = projected_counts(child.get('actual_counts'), cap)
    c.require(protocol.get('budget_max') == cap, 'stage25 cap differs')
    if kind in ('export', 'cpp'):
        verify_numeric(kind, child, directory)
        return counts
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import verify_trace
    bridge = evaluation_artifacts(directory.parent)
    c.require(all(value.get('numerical_predecessors') == bridge for value in (child, protocol)),
              'evaluation25 numerical predecessor differs')
    verify_numeric('cpp', c.read_json(directory.parent/'cpp/report.json'), directory.parent/'cpp',
                   old_release=bundle['old_release'])
    helpers = frozen_helpers(bundle['old_release'])
    c.require([(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in child['cases']] == prior.matrix(kind)
              and protocol.get('thresholds') == helpers.THRESHOLDS and protocol.get('sequences') == helpers.SEQUENCES
              and protocol.get('backend') == child.get('backend')
              and protocol.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256'],
              'original25 evaluation identity/matrix differs')
    total = dict(ZERO)
    for row in child['cases']:
        verify_trace(directory, row, helpers)
        actual = projected_counts(row.get('actual_counts'), CAP)
        total = {key: total[key]+actual[key] for key in ZERO}
    full = total == CAP and all(row['reference_thresholds_met'] is True and row['failure'] is False
                               and row['done_reason'] == 'HORIZON_12S' for row in child['cases'])
    c.require(total == counts and child.get('all_36_reference_pass') is full, 'evaluation25 totals/qualification differ')
    return counts


def evaluation_artifacts(parent):
    return {name: c.sha(parent/name) for name in (
        'export/report.json', 'export/policy.onnx', 'export/golden_fixtures.json', 'cpp/report.json')}


def run_stage(plan_path, arm, kind, output, *, loader=c.load_plan,
              launch_validator=c.validate_launch, memory_reader=c.read_memory, clock=time.monotonic,
              numeric=numerical, evaluation=prior.evaluate, training_loader=completed_training):
    c.require(arm in c.ARMS and kind in ('export', 'cpp', 'formal', 'holdout'), 'unknown25 stage')
    stage = kind+'_'+arm
    started = clock()
    plan = loader(plan_path)
    launch_validator(plan, plan_path, stage, output)
    bundle = training_loader(plan, plan_path, arm)
    c.require(kind in ('export', 'cpp', 'formal', 'holdout') and not output.exists(), 'new25 stage only')
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
        c.require(now-started < c.STAGES[stage]['max_wall_s'], 'stage25 wall limit')
        if force or now-last >= 1.:
            sample = dict(memory_reader(), elapsed_s=now-started)
            with (output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            c.require(type(sample.get('effective_available_bytes')) is int and sample['effective_available_bytes'] >= c.RESOURCES['start_min_effective_bytes' if admission else 'reserve_bytes'],
                      'stage25 RAM below 6/4 GiB')
            last = now
    with termination_guard() as control:
        try:
            poll(True, True)
            c.atomic_json(output/'protocol.json', dict(identity(plan, plan_path, stage),
                max_wall_s=c.STAGES[stage]['max_wall_s'], budget_max=CAP if kind in ('formal','holdout') else ZERO,
                completed_training=bundle['identity'], checkpoint_sha256=report['checkpoint_sha256']))
            report['protocol_sha256'] = c.sha(output/'protocol.json')
            if kind in ('export', 'cpp'):
                report.update(numeric(kind, bundle, output.parent, output))
                report['actual_counts'] = dict(ZERO)
            else:
                report['numerical_predecessors'] = evaluation_artifacts(output.parent)
                protocol = c.read_json(output/'protocol.json')
                protocol['numerical_predecessors'] = report['numerical_predecessors']
                c.atomic_json(output/'protocol.json', protocol)
                report['protocol_sha256'] = c.sha(output/'protocol.json')
                evaluation(kind, bundle, output.parent, output, report, poll)
            c.require(loader(plan_path) == plan and training_loader(plan, plan_path, arm)['identity'] == bundle['identity'],
                      'stage25 inputs changed')
            poll(True)
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
    return {key: sum(row.get('actual_counts', {}).get(key, 0) if row.get('actual_counts') is not None
                     else row.get('known_actual_counts', ZERO)[key] for row in stages.values()) for key in ZERO}


def paired_initials(plan, plan_path):
    bundles = [completed_training(plan, plan_path, arm) for arm in c.ARMS]
    c.require(bundles[0]['protocol']['initial_state'] == bundles[1]['protocol']['initial_state']
              and bundles[0]['protocol']['versions'] == bundles[1]['protocol']['versions'],
              'two arms did not start from the same original policy/Adam/runtime')
    return bundles[0]['protocol']['initial_state']


def select_candidate(formals, holdouts):
    """Only the two preregistered final checkpoints; exact ties prefer zero."""
    eligible = [arm for arm in c.ARMS if formals.get(arm, {}).get('all_36_reference_pass') is True
                and holdouts.get(arm, {}).get('all_36_reference_pass') is True]
    means = {arm: sum(row['J'] for row in formals[arm]['cases'])/36 for arm in eligible}
    c.require(all(len(formals[arm]['cases']) == 36 and all(
        isinstance(row['J'], (int, float)) and not isinstance(row['J'], bool) and math.isfinite(row['J'])
        for row in formals[arm]['cases']) for arm in eligible), 'invalid final selection score')
    selected = min(eligible, key=lambda arm: (means[arm], 0 if arm == 'zero' else 1)) if eligible else None
    return {'eligible_arms': eligible, 'original36_mean_J': means, 'selected_arm': selected,
            'automatic_model_replacement': False,
            'scope': 'same-seed two-final screening; no multi-training-seed significance or physical deployment claim'}


def compare_final_arms(formals):
    """Retain task failures and compare only matching complete horizons for fair J."""
    def keyed(report):
        rows = report['cases']
        keys = [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in rows]
        c.require(keys == prior.matrix('formal'), 'comparison25 original matrix differs')
        return dict(zip(keys, rows))
    tables = {arm: keyed(formals[arm]) for arm in c.ARMS}
    common = [key for key in tables['zero'] if all(
        tables[arm][key]['policy_transitions'] == 600 for arm in c.ARMS)]
    return {'reference_pass_counts': {arm: sum(row['reference_thresholds_met'] is True
                for row in formals[arm]['cases']) for arm in c.ARMS},
            'full_horizon_counts': {arm: sum(row['policy_transitions'] == 600
                for row in formals[arm]['cases']) for arm in c.ARMS},
            'common_full_pairs': len(common),
            'common_full_mean_J': {arm: sum(tables[arm][key]['J'] for key in common)/len(common)
                if common else None for arm in c.ARMS},
            'all36_mean_J_including_early_failure_penalties': {
                arm: sum(row['J'] for row in formals[arm]['cases'])/36 for arm in c.ARMS},
            'scope': 'one training seed; task failures retained; partial horizons cannot support an improvement claim'}


def partial_counts(plan, plan_path, stage, directory, child):
    arm, kind = c.split_stage(stage)
    protocol_path = directory/'protocol.json'
    protocol = c.read_json(protocol_path)
    base = {'plan_sha256': c.sha(plan_path), 'source_identity': plan['source_identity'],
            'stage': stage, 'arm': arm, 'mode': c.ARMS[arm]}
    c.require(all(item.get(k) == v for item in (child, protocol) for k, v in base.items())
              and child.get('protocol_sha256') == c.sha(protocol_path), 'foreign25 partial receipt')
    if kind == 'train':
        c.require(child.get('format') == 'd1-delay-comparison25-training-result-v1'
                  and protocol.get('format') == 'd1-delay-comparison25-training-protocol-v1'
                  and child.get('seed') == protocol.get('seed') == 181
                  and child.get('envs') == protocol.get('envs') == 1
                  and child.get('hyperparameters') == protocol.get('hyperparameters') == c.HYPERPARAMETERS
                  and protocol.get('base_checkpoint') == plan['base_checkpoint']
                  and protocol.get('base_protocol') == plan['base_protocol']
                  and all(value.get('vector_layout') == plan['vector_layout']
                          and value.get('runtime_dependencies_sha256') == plan['runtime_dependencies_sha256']
                          for value in (child, protocol)), 'partial25 training provenance differs')
    else:
        bundle = completed_training(plan, plan_path, arm)
        c.require(child.get('format') == protocol.get('format') == 'd1-delay-comparison25-stage-v1'
                  and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s']
                  and protocol.get('budget_max') == (CAP if kind in ('formal','holdout') else ZERO)
                  and all(value.get('completed_training') == bundle['identity']
                          and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                          for value in (child, protocol)),
                  'partial25 stage scope differs')
        if kind in ('formal','holdout'):
            c.require(protocol.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256'],
                      'foreign partial25 model')
            c.require(all(value.get('numerical_predecessors') == evaluation_artifacts(directory.parent)
                          for value in (child, protocol)), 'foreign partial25 numerical inputs')
    cap = c.expected_counts(c.STAGES[stage]['transitions'])
    actual = child.get('actual_counts')
    if actual is not None:
        c.require(child.get('known_actual_counts') == actual, 'partial25 exact/known counters disagree')
        return projected_counts(actual, cap, partial=True), None
    return None, projected_counts(child.get('known_actual_counts', ZERO), cap, partial=True)


def execute(plan_path=c.ROOT/c.PLAN, *, loader=c.load_plan, memory_reader=c.read_memory,
            factory=subprocess.Popen, cleanup=cleanup_owned_tree, adopter=subreaper,
            verifier=verify_stage, initial_checker=paired_initials,
            clock=time.monotonic, sleeper=time.sleep):
    started = clock()
    plan = loader(plan_path)
    output, plan_path = ROOT/plan['output_root'], Path(plan_path)
    c.require(not output.exists() and not output.is_symlink(), 'attempt25 already exists; no retry')
    c.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '25 requires external Nice10 or lower priority')
    initial_memory = memory_reader().get('effective_available_bytes')
    c.require(type(initial_memory) is int and initial_memory >= 0, 'effective25 RAM unknown')
    if initial_memory < 6 << 30:
        return {'status': 'NOT_READY', 'execution_started': False, 'actual_counts': dict(ZERO)}
    output.mkdir(parents=True, exist_ok=False)
    plan_sha = c.sha(plan_path)
    report = {'format': 'd1-delay-comparison25-execution-v1', 'status': 'RUNNING', 'supervisor_pid': os.getpid(),
              'plan_sha256': plan_sha, 'source_identity': plan['source_identity'], 'stages': {},
              'active_stage': None, 'actual_counts': None, 'known_actual_counts': dict(ZERO),
              'cleanup_complete': True, 'error': None, 'automatic_model_replacement': False,
              'admission_waits': [],
              'original_36_pass': {arm: False for arm in c.ARMS},
              'holdout_36_pass': {arm: False for arm in c.ARMS}, 'selection': None}
    c.atomic_json(output/'report.json', report)
    process = current = None
    formals, holdouts = {}, {}
    def sample():
        row = dict(memory_reader(), elapsed_s=clock()-started, stage=current)
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        n = row.get('effective_available_bytes')
        c.require(type(n) is int and n >= c.RESOURCES['reserve_bytes'], 'RAM below 4 GiB or unknown')
        c.require(clock()-started < c.RESOURCES['total_wall_s'], 'total25 wall limit')
        return row

    def admit():
        """No live child, no stage claim: wait only for the preregistered 6-GiB gate."""
        c.require(process is None and report['cleanup_complete'] is True
                  and current not in report['stages'], 'admission25 requires closed prior stages')
        wait_started, wait = clock(), None
        try:
            while True:
                row = sample()
                duration = clock()-wait_started
                if wait is not None:
                    wait.update(duration_s=duration)
                    wait['samples'].append(row)
                if row['effective_available_bytes'] >= c.RESOURCES['start_min_effective_bytes']:
                    c.require(duration <= c.RESOURCES['interstage_admission_wait_s'], 'admission25 wait exceeded')
                    if wait is not None:
                        wait['status'] = 'ADMITTED'
                        c.atomic_json(output/'report.json', report)
                    return
                if wait is None:
                    wait = {'stage': current, 'status': 'WAITING_FOR_ADMISSION',
                            'started_elapsed_s': clock()-started, 'duration_s': duration, 'samples': [row]}
                    report['admission_waits'].append(wait)
                c.require(duration < c.RESOURCES['interstage_admission_wait_s'], 'admission25 wait exhausted')
                c.atomic_json(output/'report.json', report)
                delay = min(c.RESOURCES['admission_poll_s'],
                            c.RESOURCES['interstage_admission_wait_s']-duration,
                            c.RESOURCES['total_wall_s']-(clock()-started))
                c.require(delay > 0, 'total25 wall limit during admission')
                sleeper(delay)
        except BaseException as exc:
            if wait is not None:
                wait.update(status='STOPPED', duration_s=clock()-wait_started,
                            error=f'{type(exc).__name__}: {exc}')
            raise
    with termination_guard() as control:
        try:
            adopter()
            for stage in c.STAGE_ORDER:
                arm, kind = c.split_stage(stage)
                spec, directory = c.STAGES[stage], c.stage_path(plan, stage)
                if kind == 'holdout' and not report['original_36_pass'][arm]:
                    report['stages'][stage] = {'status': 'SKIPPED_NO_QUALIFIED_CANDIDATE', 'actual_counts': dict(ZERO)}
                    continue
                current = stage
                admit()
                c.require(loader(plan_path) == plan and c.sha(plan_path) == plan_sha, 'plan25 changed before stage')
                for old in report['stages'].values():
                    if old.get('status') == 'COMPLETE':
                        c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'earlier25 artifact changed')
                c.require(clock()-started < c.RESOURCES['total_wall_s'], 'total25 wall limit before stage claim')
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
                                start_new_session=True, env=dict(os.environ, D1_STAGE25_NONCE=record['nonce'],
                                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                    PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT)))))
                        finally:
                            control['defer'] = False
                        c.require(control['received'] is None, 'termination during25 spawn')
                        while process.poll() is None:
                            sample()
                            c.require(clock()-stage_started < spec['max_wall_s'], 'child25 wall limit')
                            sleeper(1.)
                        sample()
                        c.require(process.returncode == 0 and clock()-stage_started < spec['max_wall_s'], 'child25 failed/late')
                finally:
                    control['defer'] = True
                    receipt = cleanup(process)
                    record['cleanup_complete'] = receipt.get('remaining') == {}
                    report['cleanup_complete'] &= record['cleanup_complete']
                    process = None
                    control['defer'] = False
                    c.require(record['cleanup_complete'], 'owned25 cleanup incomplete')
                c.require(control['received'] is None, 'termination during25 cleanup')
                child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                actual = verifier(plan, plan_path, stage, directory, child)
                record.update(status='COMPLETE', actual_counts=actual, artifacts_sha256=prior.artifacts(directory))
                c.require(clock()-stage_started < spec['max_wall_s'], 'child25 late cleanup/readback')
                report['known_actual_counts'] = totals(report['stages'])
                projected_counts(report['known_actual_counts'], plan['budget'])
                if stage == 'train_random':
                    report['paired_initial_state'] = initial_checker(plan, plan_path)
                if kind == 'formal':
                    report['original_36_pass'][arm] = child['all_36_reference_pass']
                    formals[arm] = child
                elif kind == 'holdout':
                    report['holdout_36_pass'][arm] = child['all_36_reference_pass']
                    holdouts[arm] = child
                current = report['active_stage'] = None
                c.atomic_json(output/'report.json', report)
            c.require(loader(plan_path) == plan and c.sha(plan_path) == plan_sha, 'final25 inputs changed')
            for old in report['stages'].values():
                if old.get('status') == 'COMPLETE':
                    c.require(all(c.sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'final25 artifact changed')
            sample()
            report.update(status='COMPLETE', selection=select_candidate(formals, holdouts),
                          comparison=compare_final_arms(formals))
        except BaseException as exc:
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            if current in report['stages']:
                record = report['stages'][current]
                record['status'] = 'INCOMPLETE'
                try:
                    directory = c.stage_path(plan, current)
                    _, kind = c.split_stage(current)
                    child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                    actual, known = partial_counts(plan, plan_path, current, directory, child)
                    if record.get('actual_counts') is None:
                        record.update(actual_counts=actual)
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
                report.update(status='INCOMPLETE', selection=None)
            report['known_actual_counts'] = totals(report['stages'])
            report['actual_counts'] = (report['known_actual_counts'] if all(
                row.get('actual_counts') is not None for row in report['stages'].values()) else None)
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
        value = {'status': 'PLAN_ONLY', 'budget': c.expected_counts(610688),
                 'new_training_transitions': c.TRANSITIONS*2, 'execution_started': False}
    print(json.dumps({key: value.get(key) for key in ('status', 'actual_counts', 'selection', 'error', 'budget')}))
    return 0 if value['status'] in ('COMPLETE', 'PLAN_ONLY', 'NOT_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
