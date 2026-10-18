"""One bounded fixed-base anchor pair; default is a zero-physics plan preview."""
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
from tools29 import common29 as c  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402
from tools19.run_cpp_closedloop19 import cleanup_owned_tree, subreaper  # noqa: E402
from tools21 import closeout_completed18 as prior  # noqa: E402
from tools25.run_delay_comparison25 import verify_numeric  # noqa: E402
from tools26.run_yaw_reward26 import compare_reports, load_references  # noqa: E402

ZERO = c.expected_counts(0)
CAP = c.expected_counts(21600)
ADMISSION = {'minimum_effective_bytes': 4 << 30, 'reserve_bytes': 2 << 30,
             'stable_s': 10, 'poll_s': 5, 'minimum_samples': 3, 'maximum_gap_s': 7.5,
             'maximum_wait_s': 300, 'total_wall_s': 2100}


def same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def counts(value, cap, *, partial=False):
    c.require(prior.valid_counts(value, cap, partial=partial), 'unknown/invalid29 counters')
    return {key: value[key] for key in ZERO}


def identity(plan, path, stage):
    arm, _ = c.split_stage(stage)
    return {'format': 'd1-anchor29-stage-v1', 'stage': stage, 'arm': arm,
            'mode': 'uniform_0_3', 'anchor_coef': c.ARMS[arm],
            'plan_sha256': c.sha(path), 'source_identity': plan['source_identity']}


def completed_training(plan, path, arm):
    return c.check_completed_training(plan, path, arm)


def check_model(model, protocol, versions):
    c.require(type(model.num_timesteps) is int and model.num_timesteps == c.INITIAL+c.TRANSITIONS
              and type(model.aux_coef) is float and model.aux_coef == 0. and model.seed == c.SEED
              and model.n_envs == 1 and str(model.device) == 'cpu' and model.get_env() is None,
              'loaded model is not completed29')
    for key in ('n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'ent_coef',
                'vf_coef', 'max_grad_norm', 'learning_rate'):
        c.require(getattr(model, key) == c.HYPERPARAMETERS[key], 'loaded29 PPO differs: '+key)
    c.require(type(model.anchor_coef) is float and model.anchor_coef == protocol['anchor_coef']
              and model.clip_range(0.) == model.clip_range(1.) == .2
              and model.normalize_advantage is True and model.clip_range_vf is None
              and model.target_kl == .005 and model.use_sde is False
              and same(versions, protocol['versions']), 'unsupported29 model/runtime')


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
        c.require(same(child, bundle['result']), 'another29 training receipt returned')
        return counts(child.get('actual_counts'), c.expected_counts(c.TRANSITIONS))
    protocol = c.read_json(directory/'protocol.json')
    c.require(all(same(value.get(key), expected) for value in (child, protocol)
                  for key, expected in identity(plan, path, stage).items())
              and child.get('status') == 'COMPLETE' and child.get('error') is None
              and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
              and child.get('protocol_sha256') == c.sha(directory/'protocol.json')
              and protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s'], 'stage29 identity/status differs')
    c.require(all(same(value.get('completed_training'), bundle['identity'])
                  and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                  for value in (child, protocol)), 'stage29 final checkpoint differs')
    cap = CAP if kind == 'formal' else ZERO
    c.require(protocol.get('budget_max') == cap, 'stage29 budget differs')
    actual = counts(child.get('actual_counts'), cap)
    counts(child.get('known_actual_counts'), cap)
    c.require(same(child['actual_counts'], child['known_actual_counts']), 'stage29 known/exact conflict')
    if kind in ('export', 'cpp'):
        verify_numeric(kind, child, directory, old_release=bundle['old_release'])
        return actual
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import verify_trace
    helpers = frozen_helpers(bundle['old_release'])
    bridge = evaluation_artifacts(directory.parent)
    c.require(all(value.get('numerical_predecessors') == bridge for value in (child, protocol)),
              'formal29 numerical predecessors differ')
    verify_numeric('cpp', c.read_json(directory.parent/'cpp/report.json'), directory.parent/'cpp',
                   old_release=bundle['old_release'])
    c.require(protocol.get('thresholds') == helpers.THRESHOLDS and protocol.get('sequences') == helpers.SEQUENCES
              and protocol.get('backend') == child.get('backend')
              and [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in child['cases']]
                  == prior.matrix('formal'), 'formal29 original matrix/backend differs')
    total = dict(ZERO)
    for row in child['cases']:
        verify_trace(directory, row, helpers)
        delta = counts(row.get('actual_counts'), CAP)
        total = {key: total[key]+delta[key] for key in ZERO}
    full = total == CAP and all(row['reference_thresholds_met'] is True and row['failure'] is False
                               and row['done_reason'] == 'HORIZON_12S' for row in child['cases'])
    c.require(total == actual and child.get('all_36_reference_pass') is full, 'formal29 totals/pass flag differ')
    return actual


def run_stage(plan_path, arm, kind, output, *, loader=c.load_plan, launch_validator=c.validate_launch,
              memory_reader=c.read_memory, clock=time.monotonic, numeric=numerical,
              evaluation=prior.evaluate, training_loader=completed_training):
    c.require(arm in c.ARMS and kind in ('export', 'cpp', 'formal'), 'unknown29 stage')
    started, stage = clock(), kind+'_'+arm
    plan_path, output = Path(plan_path), Path(output)
    plan = loader(plan_path)
    launch_validator(plan, plan_path, stage, output)
    bundle = training_loader(plan, plan_path, arm)
    c.require(not output.exists() and not output.is_symlink(), 'exclusive29 stage output required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(identity(plan, plan_path, stage), status='INCOMPLETE', error=None, cases=[], actual_counts=None,
                  known_actual_counts=dict(ZERO), cleanup_complete=True, inputs_unchanged=False,
                  all_36_reference_pass=False, completed_training=bundle['identity'],
                  checkpoint_sha256=bundle['identity']['training_identity']['checkpoint_sha256'])
    last = -float('inf')

    def poll(force=False, admission=False):
        nonlocal last
        now = clock()
        c.require(now-started < c.STAGES[stage]['max_wall_s'], 'stage29 wall limit')
        if force or now-last >= 1.:
            sample = read_sample(memory_reader)
            sample.update(elapsed_s=now-started)
            append_json(output/'resources.jsonl', sample)
            limit = c.RESOURCES['start_min_effective_bytes' if admission else 'reserve_bytes']
            require_memory(sample, limit)
            last = now

    with termination_guard() as control:
        try:
            poll(True, True)
            c.atomic_json(output/'protocol.json', dict(identity(plan, plan_path, stage),
                max_wall_s=c.STAGES[stage]['max_wall_s'], budget_max=CAP if kind == 'formal' else ZERO,
                completed_training=bundle['identity'], checkpoint_sha256=report['checkpoint_sha256']))
            report['protocol_sha256'] = c.sha(output/'protocol.json')
            if kind in ('export', 'cpp'):
                report.update(numeric(kind, bundle, output.parent, output))
                report['actual_counts'] = report['known_actual_counts'] = dict(ZERO)
            else:
                protocol = c.read_json(output/'protocol.json')
                protocol['numerical_predecessors'] = report['numerical_predecessors'] = evaluation_artifacts(output.parent)
                c.atomic_json(output/'protocol.json', protocol)
                report['protocol_sha256'] = c.sha(output/'protocol.json')
                evaluation(kind, bundle, output.parent, output, report, poll)
            c.require(same(loader(plan_path), plan)
                      and same(training_loader(plan, plan_path, arm)['identity'], bundle['identity']), 'stage29 inputs changed')
            poll(True)
            c.require(report['cleanup_complete'] is True, 'stage29 cleanup incomplete')
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


def read_sample(reader):
    try:
        return dict(reader())
    except Exception as exc:
        if isinstance(getattr(exc, 'snapshot', None), dict):
            return dict(exc.snapshot, resource_guard_error=f'{type(exc).__name__}: {exc}')
        raise


def require_memory(row, minimum):
    value = row.get('effective_available_bytes')
    c.require(not row.get('resource_guard_error') and not row.get('errors')
              and type(value) is int and value >= minimum, '29 RAM/owned-cgroup guard failed or unknown')


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def paired_initials(plan, path):
    bundles = [completed_training(plan, path, arm) for arm in c.ARMS]
    initial = bundles[0]['protocol']['initial_state']
    proof = bundles[0]['result'].get('first_rollout_nonreward')
    c.require(isinstance(proof, dict) and all(same(initial, item['protocol']['initial_state'])
              and same(bundles[0]['protocol']['versions'], item['protocol']['versions'])
              and same(proof, item['result'].get('first_rollout_nonreward')) for item in bundles[1:]),
              '29 initial policy/Adam/runtime or first pre-update trajectory differ')
    return {'initial_state': initial, 'first_rollout_nonreward': proof, 'first_rollout_nonreward_equal': True}


def partial_counts(plan, path, stage, directory, child):
    arm, kind = c.split_stage(stage)
    protocol = c.read_json(directory/'protocol.json')
    base = identity(plan, path, stage)
    if kind == 'train':
        base.pop('format')
    c.require(all(same(value.get(key), expected) for value in (child, protocol) for key, expected in base.items())
              and child.get('protocol_sha256') == c.sha(directory/'protocol.json'), 'foreign29 partial receipt')
    if kind == 'train':
        c.require(child.get('format') == 'd1-anchor29-training-result-v1'
                  and protocol.get('format') == 'd1-anchor29-training-protocol-v1'
                  and all(type(value.get('seed')) is int and value['seed'] == c.SEED
                          and type(value.get('envs')) is int and value['envs'] == 1
                          and same(value.get('hyperparameters'), c.HYPERPARAMETERS)
                          and same(value.get('vector_layout'), plan['vector_layout'])
                          and same(value.get('runtime_dependencies_sha256'), plan['runtime_dependencies_sha256'])
                          and same(value.get('anchor_change'), c.anchor_change(arm))
                          and value.get('teacher_policy_used') is (c.ARMS[arm] != 0.)
                          for value in (child, protocol))
                  and same(protocol.get('base_checkpoint'), plan['base_checkpoint'])
                  and same(protocol.get('base_protocol'), plan['base_protocol']), 'partial29 training scope differs')
    else:
        bundle = completed_training(plan, path, arm)
        c.require(protocol.get('max_wall_s') == c.STAGES[stage]['max_wall_s']
                  and protocol.get('budget_max') == (CAP if kind == 'formal' else ZERO)
                  and all(same(value.get('completed_training'), bundle['identity'])
                          and value.get('checkpoint_sha256') == bundle['identity']['training_identity']['checkpoint_sha256']
                          for value in (child, protocol)), 'partial29 model/budget differs')
        if kind == 'formal':
            c.require(all(value.get('numerical_predecessors') == evaluation_artifacts(directory.parent)
                          for value in (child, protocol)), 'foreign29 partial numerical inputs')
    cap = c.expected_counts(c.STAGES[stage]['transitions'])
    actual = child.get('actual_counts')
    known = counts(child.get('known_actual_counts'), cap, partial=True)
    if actual is None:
        return None, known
    c.require(same(actual, child['known_actual_counts']), 'partial29 exact/known conflict')
    return counts(actual, cap, partial=True), None


def admission(plan, path, *, memory_reader, clock, sleeper, started, control):
    folder = ROOT/c.ADMISSIONS/('admission_'+secrets.token_hex(12))
    folder.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-anchor29-admission-v1', 'status': 'WAITING', 'plan_sha256': c.sha(path),
              'source_identity': plan['source_identity'], 'policy': ADMISSION, 'execution_started': False,
              'physics_steps_executed': 0, 'error': None, 'samples': 0, 'directory': str(folder)}
    first = previous = None
    number = 0
    try:
        while True:
            now = clock()
            c.require(control['received'] is None and now-started <= ADMISSION['maximum_wait_s'], '29 admission interrupted/timeout')
            row = read_sample(memory_reader)
            row['elapsed_s'] = clock()-started
            append_json(folder/'samples.jsonl', row)
            report['samples'] += 1
            require_memory(row, ADMISSION['reserve_bytes'])
            if row['effective_available_bytes'] >= ADMISSION['minimum_effective_bytes']:
                if previous is None or row['elapsed_s']-previous > ADMISSION['maximum_gap_s']:
                    first, number = row['elapsed_s'], 0
                number += 1
                previous = row['elapsed_s']
                if number >= ADMISSION['minimum_samples'] and previous-first >= ADMISSION['stable_s']:
                    c.require(previous <= ADMISSION['maximum_wait_s'], '29 admission observed too late')
                    report.update(status='ADMITTED', qualified_samples=number, stable_s=previous-first)
                    return report
            else:
                first = previous = None
                number = 0
            c.atomic_json(folder/'report.json', report)
            remaining = ADMISSION['maximum_wait_s']-(clock()-started)
            c.require(remaining > 0, '29 admission exhausted')
            sleeper(min(ADMISSION['poll_s'], remaining))
    except BaseException as exc:
        report.update(status='NOT_READY', error=f'{type(exc).__name__}: {exc}')
        return report
    finally:
        report['elapsed_s'] = clock()-started
        c.atomic_json(folder/'report.json', report)


def execute(plan_path=None, *, loader=c.load_plan, memory_reader=c.read_memory, factory=subprocess.Popen,
            cleanup=cleanup_owned_tree, adopter=subreaper, verifier=verify_stage,
            initial_checker=paired_initials, reference_loader=load_references, clock=time.monotonic, sleeper=time.sleep):
    outer_started = clock()
    path = Path(plan_path or ROOT/c.PLAN)
    plan = loader(path)
    plan_sha = c.sha(path)
    output = ROOT/plan['output_root']
    c.require(not output.exists() and not output.is_symlink(), '29 attempt exists; no retry')
    c.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '29 requires external Nice10')
    report, process, current, started, claimed = None, None, None, None, False
    with termination_guard() as control:
        admitted = admission(plan, path, memory_reader=memory_reader, clock=clock, sleeper=sleeper,
                             started=outer_started, control=control)
        if admitted['status'] != 'ADMITTED':
            return admitted
        try:
            c.require(same(loader(path), plan) and c.sha(path) == plan_sha, '29 inputs changed during admission')
            before_claim = read_sample(memory_reader)
            before_claim['elapsed_s'] = clock()-outer_started
            append_json(Path(admitted['directory'])/'samples.jsonl', before_claim)
            require_memory(before_claim, c.RESOURCES['start_min_effective_bytes'])
            c.require(before_claim['elapsed_s']-admitted['elapsed_s'] <= ADMISSION['maximum_gap_s']
                      and clock()-outer_started <= ADMISSION['maximum_wait_s'] and control['received'] is None,
                      '29 total wall/interrupted before claim')
            started = clock()
            report = {'format': c.EXECUTION_FORMAT, 'status': 'RUNNING', 'supervisor_pid': os.getpid(),
                      'plan_sha256': plan_sha, 'source_identity': plan['source_identity'], 'stages': {},
                      'active_stage': None, 'actual_counts': None, 'known_actual_counts': dict(ZERO),
                      'cleanup_complete': True, 'error': None, 'admission': admitted,
                      'minimum_effective_available_bytes': None, 'stop_sample': None,
                      'original_36_pass': {}, 'simulation_candidate_qualified': False,
                      'cpp_closed_loop_qualified': False, 'ros_hardware_qualified': False,
                      'holdout_executed': False, 'automatic_model_replacement': False}
            control['defer'] = True
            output.mkdir(parents=True, exist_ok=False)
            claimed = True
            c.atomic_json(output/'report.json', report)
            control['defer'] = False
            c.require(control['received'] is None, 'termination during29 claim')
            adopter()
            references = reference_loader(plan)

            def sample():
                row = read_sample(memory_reader)
                row.update(elapsed_s=clock()-started, stage=current)
                append_json(output/'resources.jsonl', row)
                value = row.get('effective_available_bytes')
                if type(value) is int and value >= 0:
                    old = report['minimum_effective_available_bytes']
                    report['minimum_effective_available_bytes'] = value if old is None else min(old, value)
                try:
                    require_memory(row, c.RESOURCES['reserve_bytes'])
                    c.require(clock()-started < c.RESOURCES['total_wall_s']
                              and clock()-outer_started < ADMISSION['total_wall_s'], '29 total wall limit')
                except BaseException:
                    report['stop_sample'] = row
                    raise
                return row

            def recheck():
                c.require(same(loader(path), plan) and c.sha(path) == plan_sha, '29 source/plan changed')
                for record in report['stages'].values():
                    if record['status'] == 'COMPLETE':
                        c.require(all(c.sha(name) == digest for name, digest in record['artifacts_sha256'].items()),
                                  'completed29 artifacts changed')
                sample()

            formals = {}
            for current in c.STAGE_ORDER:
                arm, kind = c.split_stage(current)
                spec, directory = c.STAGES[current], c.stage_path(plan, current)
                recheck()
                row = sample()
                wait_started = clock()
                while row['effective_available_bytes'] < c.RESOURCES['start_min_effective_bytes']:
                    c.require(process is None and report['cleanup_complete'] is True, '29 wait requires no live stage')
                    c.require(clock()-wait_started < 300, '29 interstage admission exhausted')
                    c.atomic_json(output/'report.json', report)
                    sleeper(min(5., 300-(clock()-wait_started)))
                    row = sample()
                recheck()
                c.require(control['received'] is None, 'termination before29 spawn')
                record = {'status': 'STARTED', 'actual_counts': None, 'command': c.command(path, current, directory),
                          'nonce': secrets.token_hex(24)}
                report['stages'][current], report['active_stage'] = record, current
                c.atomic_json(output/'report.json', report)
                stage_started = clock()
                stage_failure = None
                try:
                    with (output/(current+'.stdout.log')).open('xb') as stream:
                        control['defer'] = True
                        try:
                            process = factory(record['command'], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ,
                                D1_STAGE29_NONCE=record['nonce'], PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                                MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                PYTHONPATH=os.pathsep.join(map(str, (ROOT/'.python-deps', ROOT/'training', ROOT)))))
                        finally:
                            control['defer'] = False
                        c.require(control['received'] is None, 'termination during29 spawn')
                        while process.poll() is None:
                            sample()
                            c.require(clock()-stage_started < spec['max_wall_s'], '29 child wall limit')
                            sleeper(1.)
                        sample()
                        c.require(process.returncode == 0 and clock()-stage_started < spec['max_wall_s'], '29 child failed/late')
                except BaseException as exc:
                    stage_failure = exc
                    raise
                finally:
                    control['defer'] = True
                    try:
                        record['cleanup_receipt'] = cleanup(process)
                        record['cleanup_complete'] = record['cleanup_receipt'].get('remaining') == {}
                        report['cleanup_complete'] &= record['cleanup_complete']
                        process = None
                        c.require(record['cleanup_complete'], '29 owned cleanup incomplete')
                    except BaseException as exc:
                        record.update(cleanup_complete=False, cleanup_error=f'{type(exc).__name__}: {exc}')
                        report['cleanup_complete'] = False
                        if stage_failure is not None:
                            raise stage_failure from exc
                        raise
                    finally:
                        control['defer'] = False
                c.require(control['received'] is None, 'termination during29 cleanup')
                child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                actual = verifier(plan, path, current, directory, child)
                record.update(status='COMPLETE', actual_counts=actual, artifacts_sha256=prior.artifacts(directory))
                sample()
                c.require(clock()-stage_started < spec['max_wall_s'], '29 child late cleanup/readback')
                report['known_actual_counts'] = prior.known_counts(report['stages'])
                counts(report['known_actual_counts'], plan['budget'])
                if current == 'train_anchor':
                    report['paired_initial'] = initial_checker(plan, path)
                if kind == 'formal':
                    formals[arm] = child
                    report['original_36_pass'][arm] = child['all_36_reference_pass']
                report['active_stage'] = None
                c.atomic_json(output/'report.json', report)
            c.require(same(reference_loader(plan), references), '29 saved references changed')
            recheck()
            report.update(status='COMPLETE', comparison=compare_reports({**references, **formals}),
                          paired_arm_comparison=compare_reports(formals),
                          reference_comparisons={name: compare_reports({name: value, **formals})
                                                 for name, value in references.items()},
                          selection=c.select_candidate(formals))
        except BaseException as exc:
            if not claimed:
                admitted.update(status='NOT_READY', error=f'{type(exc).__name__}: {exc}')
                c.atomic_json(Path(admitted['directory'])/'report.json', admitted)
                return admitted
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            if current in report['stages']:
                record = report['stages'][current]
                record['status'] = 'INCOMPLETE'
                try:
                    _, kind = c.split_stage(current)
                    directory = c.stage_path(plan, current)
                    child = c.read_json(directory/('result.json' if kind == 'train' else 'report.json'))
                    actual, known = partial_counts(plan, path, current, directory, child)
                    if record.get('actual_counts') is None:
                        record['actual_counts'] = actual
                        if known is not None:
                            record['known_actual_counts'] = known
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        finally:
            if claimed:
                control['defer'] = True
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, signal.SIG_IGN)
                try:
                    report['cleanup_complete'] &= cleanup(process).get('remaining') == {}
                    c.require(same(loader(path), plan) and c.sha(path) == plan_sha, '29 final source changed')
                except BaseException as exc:
                    report.update(status='INCOMPLETE', cleanup_complete=False,
                                  error=(report['error'] or '')+f'; final {type(exc).__name__}: {exc}')
                report['elapsed_s'], report['outer_elapsed_s'] = clock()-started, clock()-outer_started
                if (not report['cleanup_complete'] or control['received']
                        or report['elapsed_s'] >= c.RESOURCES['total_wall_s']
                        or report['outer_elapsed_s'] >= ADMISSION['total_wall_s']):
                    report['status'] = 'INCOMPLETE'
                report['known_actual_counts'] = prior.known_counts(report['stages'])
                counts(report['known_actual_counts'], plan['budget'], partial=True)
                report['actual_counts'] = (report['known_actual_counts'] if all(
                    value.get('actual_counts') is not None for value in report['stages'].values()) else None)
                c.atomic_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/c.PLAN)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--arm', choices=tuple(c.ARMS))
    parser.add_argument('--stage', choices=('export', 'cpp', 'formal'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    c.require(sum((args.prepare, args.execute, args.stage is not None)) <= 1, 'choose one29 mode')
    if args.prepare:
        value = c.prepare(args.plan.absolute())
        value = {'status': 'PREPARED', 'budget': value['budget']}
    elif args.stage:
        c.require(args.arm is not None and args.output is not None, 'supervised29 arm/output required')
        value = run_stage(args.plan.absolute(), args.arm, args.stage, args.output.absolute())
    elif args.execute:
        value = execute(args.plan.absolute())
    else:
        value = {'status': 'PLAN_ONLY', 'budget': c.expected_counts(108736), 'new_training_transitions': 65536,
                 'teacher_sample_rows_max': 32768, 'cpp_golden_fixture_rows_per_arm': 64,
                 'execution_started': False, 'physics_steps_executed': 0,
                 'holdout_executed': False, 'cpp_closed_loop_qualified': False}
    print(json.dumps({key: value.get(key) for key in ('status', 'actual_counts', 'error', 'budget',
        'execution_started', 'physics_steps_executed', 'teacher_sample_rows_max', 'cpp_closed_loop_qualified')}))
    return 0 if value['status'] in ('COMPLETE', 'PLAN_ONLY', 'PREPARED', 'NOT_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
