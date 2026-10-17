"""Independent, conditional C++ actor-in-the-loop proof; never trains."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
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
from tools19 import cpp_actor19 as actor_api  # noqa: E402
from tools16.contract16 import read_memory  # noqa: E402
from tools16.train_comparison16 import termination_scope  # noqa: E402
from tools16.run_comparison16 import termination_guard  # noqa: E402

OUTPUT = 'evidence/cpp_closedloop19/execution_01'
PLAN18 = 'evidence/local_optimization18/plan_01.json'
BUDGET = {'policy_transitions': 21600, 'native_steps': 216000, 'pd_updates': 108000}
LIMITS = {'start_effective_bytes': 6 << 30, 'reserve_bytes': 4 << 30,
          'total_wall_s': 900, 'worker_wall_s': 840, 'supervisor_stop_s': 880,
          'poll_s': 1., 'golden_inferences': 64, 'max_cpp_inferences': 21664, 'absolute_tolerance': 1e-5}
SOURCES = ('runtime19/actor_server19.cpp', 'runtime19/.gitignore', 'tools19/cpp_actor19.py', 'tests/test_cpp_actor19.py',
           'tools19/run_cpp_closedloop19.py', 'tests/test_cpp_closedloop19.py', 'docs/cpp_closedloop19.md')
require, sha, atomic_json = actor_api.require, actor_api.sha, actor_api.atomic_json


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    require(isinstance(value, dict), 'expected a JSON object')
    return value


def matrix(seeds=(1001, 1002, 1003)):
    return [(task, delay, seed) for task in ('forward_stop', 'reverse_stop', 'turn_stop')
            for delay in (0, 1, 2, 3) for seed in seeds]


def source_identity(root=ROOT):
    proof = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=10).strip(),
             'sources_sha256': {name: sha(root/name) for name in SOURCES}}
    check_sources(proof, root=root)
    return proof


def check_sources(proof, root=ROOT):
    require(set(proof.get('sources_sha256', {})) == set(SOURCES), 'new19 source closure differs')
    for name, digest in proof['sources_sha256'].items():
        path = root/name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, 'new19 source changed: '+name)
        blob = subprocess.check_output(['git', 'show', f'{proof["git_commit"]}:{name}'], cwd=root,
                                       stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit all new19 sources before execution: '+name)


def valid_counts(value, *, partial=False):
    if (not isinstance(value, dict) or not all(type(item) is int and item >= 0 for item in value.values())
            or not all(type(value.get(key)) is int and 0 <= value[key] <= cap for key, cap in BUDGET.items())):
        return False
    p, n, d = (value[key] for key in ('policy_transitions', 'native_steps', 'pd_updates'))
    return (max(0, p-1)*10 <= n <= 10*p and max(0, p-1)*5 <= d <= 5*p and 0 <= 2*d-n <= 2) if partial else n == 10*p and d == 5*p


def add_counts(total, delta):
    require(isinstance(delta, dict) and all(type(value) is int and value >= 0 for value in delta.values())
            and all(type(delta.get(key)) is int and delta[key] >= 0 for key in BUDGET),
            'unknown physical counter')
    result = {key: total[key]+delta[key] for key in BUDGET}
    require(valid_counts(result, partial=True), 'physical budget or counter feasibility exceeded')
    return result


def complete_reference(report, expected, verifier, directory, helpers):
    require(report.get('status') == 'COMPLETE' and report.get('cleanup_complete') is True
            and report.get('inputs_unchanged') is True and report.get('error') is None
            and [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in report.get('cases', [])] == expected,
            'old18 evaluation matrix or completion differs')
    for row in report['cases']:
        verifier(directory, row, helpers)
    if report.get('all_36_reference_pass') is True:
        require(report.get('actual_counts') == BUDGET and all(
            row['reference_thresholds_met'] is True and row['failure'] is False
            and row['done_reason'] == 'HORIZON_12S'
            and all(type(row['actual_counts'].get(key)) is int and row['actual_counts'][key] == count
                    for key, count in {'policy_transitions': 600, 'native_steps': 6000, 'pd_updates': 3000}.items())
            for row in report['cases']), 'old18 full-pass flag contradicts actual cases')


def verify_inferences(output, item):
    """Tie each saved numerical pair to the action actually issued to physics."""
    import numpy as np
    name = f'{item["sequence"]}_delay{item["delay_ticks"]}_seed{item["reset_seed"]}_inference.npz'
    require(item.get('inference_file') == name and sha(Path(output)/name) == item.get('inference_sha256'),
            'inference artifact identity differs')
    n = item['actual_counts']['policy_transitions']
    require(type(n) is int and 1 <= n <= 600 and item.get('observations') == n, 'inference horizon differs')
    with np.load(Path(output)/name, allow_pickle=False) as data:
        require(set(data.files) == {'observation', 'cpp_actions', 'python_ort_actions', 'max_absolute_error'},
                'inference array names differ')
        arrays = {key: data[key] for key in data.files}
    for key, shape in {'observation': (n, 627), 'cpp_actions': (n, 16), 'python_ort_actions': (n, 16)}.items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == np.float32 and bool(np.isfinite(value).all()),
                'invalid inference array: '+key)
    cpp, oracle = arrays['cpp_actions'], arrays['python_ort_actions']
    require(bool((np.abs(cpp) <= 1).all()) and bool((np.abs(oracle) <= 1).all()), 'inference action range differs')
    delta = np.max(np.abs(cpp-oracle), axis=1).astype(np.float64)
    require(arrays['max_absolute_error'].shape == (n,) and arrays['max_absolute_error'].dtype == np.float64
            and np.array_equal(arrays['max_absolute_error'], delta)
            and bool((delta <= LIMITS['absolute_tolerance']).all()), 'saved numerical parity differs')
    with np.load(Path(output)/item['trace_file'], allow_pickle=False) as trace:
        require(np.array_equal(trace['issued_action'], cpp), 'physics was driven by another action')
    return float(np.max(delta))


def verify_completed_worker(directory, child, release, bundle, *, helpers=None, verifier=None):
    """A zero-physics readback gate, independent of the worker's success flags."""
    import numpy as np
    if helpers is None:
        from tools16.evaluate_comparison16 import frozen_helpers
        helpers = frozen_helpers(bundle['old_release'])
    if verifier is None:
        from tools17.evaluate_formal17 import verify_trace as verifier
    protocol_path = Path(directory)/'protocol.json'
    protocol = read_json(protocol_path)
    require(child.get('format') == 'd1-cpp-closedloop19-worker-v1'
            and child.get('release_sha256') == sha(Path(directory).parent/'release.json')
            and child.get('source_identity') == release['source_identity']
            and child.get('old18_qualification') == bundle['identity']
            and child.get('status') == 'COMPLETE' and child.get('error') is None
            and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
            and child.get('protocol_sha256') == sha(protocol_path)
            and protocol.get('format') == 'd1-cpp-closedloop19-protocol-v1'
            and protocol.get('release_sha256') == child['release_sha256']
            and protocol.get('source_identity') == release['source_identity']
            and protocol.get('old18_qualification') == bundle['identity']
            and protocol.get('budget') == BUDGET and protocol.get('limits') == LIMITS
            and protocol.get('action_driver') == 'cpp_ort1.10_only'
            and [tuple(row) for row in protocol.get('scenarios', [])] == matrix(), 'worker proof identity differs')
    rows = child.get('cases', [])
    require([(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in rows] == matrix(), 'worker matrix differs')
    total, maximum = dict.fromkeys(BUDGET, 0), 0.
    for row in rows:
        require(valid_counts(row.get('actual_counts')) and 1 <= row['actual_counts']['policy_transitions'] <= 600,
                'invalid case counts')
        verifier(directory, row, helpers)
        maximum = max(maximum, verify_inferences(directory, row))
        total = add_counts(total, row['actual_counts'])
    require(child.get('actual_counts') == child.get('known_actual_counts') == total
            and child.get('maximum_same_input_error') == maximum, 'worker aggregate differs')
    golden = child.get('golden', [])
    require(len(golden) == 64 and [row.get('index') for row in golden] == list(range(64)), 'golden indices differ')
    for row in golden:
        fixture = bundle['fixtures'][row['index']]
        actual, oracle = (np.asarray(row.get(key), dtype=np.float32) for key in ('actions', 'python_ort_actions'))
        require(actual.shape == oracle.shape == (16,) and bool(np.isfinite(actual).all())
                and bool(np.isfinite(oracle).all()) and bool((np.abs(actual) <= 1).all())
                and bool((np.abs(oracle) <= 1).all()), 'invalid golden actions')
        delta = max(float(np.max(np.abs(actual-np.asarray(fixture[key], dtype=np.float32))))
                    for key in ('torch_actions', 'python_ort_actions'))
        delta = max(delta, float(np.max(np.abs(actual-oracle))))
        require(type(row.get('max_absolute_error')) in (int, float)
                and row['max_absolute_error'] == delta <= LIMITS['absolute_tolerance'], 'golden numerical proof differs')
    expected = 64+total['policy_transitions']
    actor = child.get('cpp_actor') or {}
    require(actor.get('status') == 'CLOSED' and actor.get('cleanup_complete') is True
            and all(type(actor.get(key)) is int and actor[key] == expected
                    for key in ('requests_completed', 'inference_attempts', 'inference_returns'))
            and type(child.get('oracle_calls')) is int and child['oracle_calls'] == expected,
            'C++ EOF or actual inference counts differ')
    qualified = total == BUDGET and all(row['reference_thresholds_met'] is True for row in rows)
    require(child.get('all_36_reference_pass') is qualified, 'reference qualification differs')
    return total


def load_eligible(root=ROOT, *, common=None, evaluator=None, numerical=None, verifier=None, helper_loader=None):
    if common is None:
        from tools18 import common18 as common
    if evaluator is None:
        from tools18 import evaluate_delay18 as evaluator
    if numerical is None:
        from tools18 import export_delay18 as numerical
    if verifier is None:
        from tools17.evaluate_formal17 import verify_trace as verifier
    if helper_loader is None:
        from tools16.evaluate_comparison16 import frozen_helpers as helper_loader
    plan_path = root/PLAN18
    plan = common.load_plan(plan_path)
    execution = root/plan['output_root']
    outer = read_json(execution/'report.json')
    require(outer.get('status') == 'COMPLETE' and outer.get('cleanup_complete') is True
            and outer.get('error') is None and outer.get('full_delay_qualification') is True
            and outer.get('plan_sha256') == sha(plan_path)
            and outer.get('source_identity') == plan['source_identity']
            and set(outer.get('stages', {})) == set(plan['stage_order']), 'old18 is not completely qualified')
    proof = {str(plan_path): sha(plan_path), str(execution/'report.json'): sha(execution/'report.json')}
    for stage in plan['stage_order']:
        record = outer['stages'][stage]
        require(record.get('status') == 'COMPLETE' and record.get('cleanup_complete') is True,
                'old18 stage incomplete: '+stage)
        directory = execution/stage
        actual = {str(path): sha(path) for path in directory.rglob('*') if path.is_file()}
        require(actual == record.get('artifacts_sha256'), 'old18 stage artifact closure changed: '+stage)
        receipt = directory/('result.json' if plan['stages'][stage]['kind'] == 'train' else 'report.json')
        require(sha(receipt) == record.get('receipt_sha256'), 'old18 stage receipt changed: '+stage)
        proof.update(actual)
    old_release = common.verify_inputs(plan)[0]
    helpers = helper_loader(old_release)
    for arm in ('zero', 'random'):
        complete_reference(read_json(execution/('eval_'+arm)/'report.json'), matrix(), verifier,
                           execution/('eval_'+arm), helpers)
    selected = evaluator.selection(plan)
    require(selected in ('zero', 'random') and outer.get('selected_arm') == selected, 'old18 winner differs')
    selected_report = read_json(execution/('eval_'+selected)/'report.json')
    holdout = read_json(execution/'holdout/report.json')
    complete_reference(holdout, matrix((2001, 2002, 2003)), verifier, execution/'holdout', helpers)
    require(selected_report.get('all_36_reference_pass') is True and holdout.get('all_36_reference_pass') is True
            and holdout.get('mode') == selected and holdout.get('stage') == 'holdout', 'old18 winner did not pass both 36-case sets')
    checkpoint, protocol, _, training = numerical.training_identity(common, plan, sha(plan_path), selected)
    export_dir, exported, fixtures = numerical.inspect_export(common, plan, sha(plan_path), selected, training)
    cpp = read_json(execution/('cpp_'+selected)/'report.json')
    require(cpp.get('status') == 'PASS' and cpp.get('checkpoint_sha256') == sha(checkpoint)
            and holdout.get('checkpoint_sha256') == sha(checkpoint)
            and selected_report.get('checkpoint_sha256') == sha(checkpoint)
            and cpp.get('onnx_sha256') == exported['onnx_sha256'] == holdout.get('onnx_sha256')
            and cpp.get('fixtures_sha256') == exported['fixtures_sha256'], 'old18 numeric/model identities differ')
    build_dir = root/'runtime19/build'
    build = read_json(build_dir/'build.json')
    binary = build_dir/'actor_server19'
    require(build.get('status') == 'PASS' and build.get('server_started') is False
            and sha(binary) == build.get('binary_sha256')
            and build['inputs_sha256'].get(str(root/'runtime19/actor_server19.cpp')) == sha(root/'runtime19/actor_server19.cpp')
            and build['inputs_sha256'].get(str(root/'tools19/cpp_actor19.py')) == sha(root/'tools19/cpp_actor19.py')
            and all(sha(name) == digest for name, digest in build['inputs_sha256'].items()), 'C++ build provenance changed')
    proof.update({str(build_dir/'build.json'): sha(build_dir/'build.json'), str(binary): sha(binary)})
    return {'plan': plan, 'old_release': old_release, 'training_protocol': protocol, 'selected_arm': selected,
            'model': Path(export_dir)/'policy.onnx', 'fixtures': fixtures, 'binary': binary,
            'identity': {'selected_arm': selected, 'checkpoint_sha256': sha(checkpoint),
                         'onnx_sha256': exported['onnx_sha256'], 'fixtures_sha256': exported['fixtures_sha256'],
                         'binary_sha256': sha(binary), 'build_report_sha256': sha(build_dir/'build.json'),
                         'artifacts_sha256': proof}}


def validate_launch(release_path, output):
    root = Path(release_path).parent
    report = read_json(root/'report.json')
    require(Path(output).resolve() == root/'worker' and report.get('status') == 'RUNNING'
            and report.get('release_sha256') == sha(release_path) and report.get('supervisor_pid') == os.getppid()
            and report.get('nonce') == os.environ.get('D1_CPP19_NONCE') and bool(report.get('nonce')),
            'worker requires its active bounded parent')


def run_worker(release_path, output, *, bundle_loader=load_eligible, source_checker=check_sources,
               launch_validator=validate_launch, runtime_factory=None, actor_factory=actor_api.CppActor19,
               helper_loader=None, verifier=None, memory_reader=read_memory, clock=time.monotonic):
    if runtime_factory is None:
        from tools17.evaluate_formal17 import runtime as runtime_factory
    if helper_loader is None:
        from tools16.evaluate_comparison16 import frozen_helpers as helper_loader
    if verifier is None:
        from tools17.evaluate_formal17 import verify_trace as verifier
    started = clock()
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    release = read_json(release_path)
    require(release.get('budget') == BUDGET and release.get('limits') == LIMITS, 'new19 budget changed')
    source_checker(release['source_identity'])
    launch_validator(release_path, output)
    bundle = bundle_loader()
    require(bundle['identity'] == release['old18_qualification'], 'old18 eligibility changed')
    require(not output.exists(), 'worker output must be new')
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-cpp-closedloop19-worker-v1', 'status': 'INCOMPLETE', 'release_sha256': sha(release_path),
              'source_identity': release['source_identity'], 'old18_qualification': bundle['identity'],
              'actual_counts': None, 'known_actual_counts': dict.fromkeys(BUDGET, 0), 'cases': [],
              'golden': [], 'oracle_calls': 0, 'cleanup_complete': True, 'inputs_unchanged': False,
              'error': None, 'cpp_actor': None, 'all_36_reference_pass': False, 'new_training_transitions': 0}
    total = dict.fromkeys(BUDGET, 0)
    environment = actor = None
    accounted, constructing, unknown = False, False, False
    oracle_maximum, last_poll = 0., -float('inf')
    observations, cpp_actions, python_actions, errors = [], [], [], []
    current_name = None

    def poll(force=False, admission=False):
        nonlocal last_poll
        now = clock()
        require(now-started < LIMITS['worker_wall_s'], 'worker wall deadline exhausted')
        if force or now-last_poll >= LIMITS['poll_s']:
            sample = dict(memory_reader(), elapsed_s=now-started, phase='admission' if admission else 'running')
            with (output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            value = sample.get('effective_available_bytes')
            require(type(value) is int and value >= (LIMITS['start_effective_bytes'] if admission else LIMITS['reserve_bytes']),
                    'effective RAM admission/reserve exhausted')
            report['minimum_effective_bytes'] = min(report.get('minimum_effective_bytes', value), value)
            last_poll = now

    def save_inferences():
        if current_name is None:
            return None
        import numpy as np
        path = output/(current_name+'_inference.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, observation=np.asarray(observations, dtype=np.float32).reshape(-1, 627),
                cpp_actions=np.asarray(cpp_actions, dtype=np.float32).reshape(-1, 16),
                python_ort_actions=np.asarray(python_actions, dtype=np.float32).reshape(-1, 16),
                max_absolute_error=np.asarray(errors, dtype=np.float64))
        return {'inference_file': path.name, 'inference_sha256': sha(path), 'observations': len(observations)}

    with termination_scope():
        try:
            poll(True, admission=True)
            rt = runtime_factory(bundle['old_release'], bundle['model'], bundle['training_protocol'])
            helpers = helper_loader(bundle['old_release'])
            require(helpers.make_scenarios('formal') == matrix(), 'frozen formal matrix changed')
            poll(True, admission=True)
            protocol = {'format': 'd1-cpp-closedloop19-protocol-v1', 'release_sha256': report['release_sha256'],
                        'source_identity': release['source_identity'], 'old18_qualification': bundle['identity'],
                        'budget': BUDGET, 'limits': LIMITS,
                        'backend': {'kind': 'cpp_ort1.10_actor_with_same_input_python_ort_oracle', 'oracle': rt.backend},
                        'scenarios': matrix(), 'thresholds': helpers.THRESHOLDS,
                        'action_driver': 'cpp_ort1.10_only', 'oracle': 'same_actual_input_python_ort_no_extra_physics'}
            atomic_json(output/'protocol.json', protocol)
            report['protocol_sha256'] = sha(output/'protocol.json')
            actor = actor_factory(bundle['binary'], bundle['model'], output=output/'cpp_actor',
                expected_binary_sha256=bundle['identity']['binary_sha256'],
                expected_model_sha256=bundle['identity']['onnx_sha256'],
                expected_source_sha256=release['source_identity']['sources_sha256']['runtime19/actor_server19.cpp'],
                max_requests=LIMITS['max_cpp_inferences'])
            actor.start()
            import numpy as np
            for index in range(64):
                poll()
                fixture = bundle['fixtures'][index]
                obs = np.asarray(fixture['obs']+[value for frame in fixture['history_oldest_to_newest_previous_only']
                                                for value in frame], dtype=np.float32)
                actual, reference = actor.predict(obs), rt.predict(obs)
                report['oracle_calls'] += 1
                delta = max(float(np.max(np.abs(actual-np.asarray(fixture[key], dtype=np.float32))))
                            for key in ('torch_actions', 'python_ort_actions'))
                delta = max(delta, float(np.max(np.abs(actual-reference))))
                require(math.isfinite(delta) and delta <= LIMITS['absolute_tolerance'], '64-fixture C++ numerical parity failed')
                report['golden'].append({'index': index, 'actions': actual.tolist(),
                                        'python_ort_actions': reference.tolist(), 'max_absolute_error': delta})

            def predict(observation):
                nonlocal oracle_maximum
                poll()
                actual = actor.predict(observation)
                reference = rt.predict(observation)
                report['oracle_calls'] += 1
                delta = float(np.max(np.abs(actual-reference)))
                require(math.isfinite(delta) and delta <= LIMITS['absolute_tolerance'], 'same-input C++/Python ORT divergence')
                observations.append(observation.copy())
                cpp_actions.append(actual.copy())
                python_actions.append(reference.copy())
                errors.append(delta)
                oracle_maximum = max(oracle_maximum, delta)
                return actual

            for name, delay, seed in matrix():
                poll(True)
                current_name = f'{name}_delay{delay}_seed{seed}'
                observations, cpp_actions, python_actions, errors = [], [], [], []
                accounted, constructing = False, True
                environment = rt.make_env(name, helpers.SEQUENCES[name], delay, seed)
                constructing = False
                poll(True)
                item = helpers.run_episode(environment, predict, name, delay, seed, output/(current_name+'.npz'),
                                           BUDGET['native_steps']-total['native_steps'])
                require((item['sequence'], item['delay_ticks'], item['reset_seed']) == (name, delay, seed), 'wrong returned scenario')
                verifier(output, item, helpers)
                require(len(observations) == item['actual_counts']['policy_transitions'], 'inference/physical step count differs')
                total = add_counts(total, item['actual_counts'])
                accounted = True
                item.update(save_inferences())
                current_name = None
                verify_inferences(output, item)
                report['cases'].append(item)
                environment.close()
                environment = None
                report['known_actual_counts'] = dict(total)
                atomic_json(output/'report.json', report)
            poll(True)
            require(bundle_loader()['identity'] == bundle['identity'], 'old18 inputs changed during closed loop')
            source_checker(release['source_identity'])
            require(sha(release_path) == report['release_sha256'], 'new19 release changed')
            report['inputs_unchanged'] = True
            report['all_36_reference_pass'] = total == BUDGET and all(row['reference_thresholds_met'] is True for row in report['cases'])
            report['maximum_same_input_error'] = oracle_maximum
            report['status'] = 'COMPLETE'
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
            if constructing:
                unknown, report['cleanup_complete'] = True, False
            if environment is not None and not accounted:
                try:
                    total = add_counts(total, environment.counters())
                except BaseException:
                    unknown = True
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if current_name is not None:
                try:
                    report['partial_inference'] = save_inferences()
                except BaseException as exc:
                    report['error'] = (report['error'] or '')+f'; inference archive {exc}'
            if environment is not None:
                try:
                    environment.close()
                except BaseException as exc:
                    report.update(cleanup_complete=False, error=(report['error'] or '')+f'; environment cleanup {exc}')
            if actor is not None:
                try:
                    report['cpp_actor'] = actor.close()
                except BaseException as exc:
                    report.update(cleanup_complete=False, error=(report['error'] or '')+f'; C++ cleanup {exc}')
                report['cleanup_complete'] &= bool(report['cpp_actor']) and report['cpp_actor'].get('cleanup_complete') is True
                if report['status'] == 'COMPLETE' and report['cpp_actor']:
                    expected = 64+total['policy_transitions']
                    require_actor = (report['cpp_actor'].get('status') == 'CLOSED'
                        and report['cpp_actor'].get('requests_completed') == expected
                        and report['cpp_actor'].get('inference_attempts') == expected
                        and report['cpp_actor'].get('inference_returns') == expected
                        and report['oracle_calls'] == expected)
                    if not require_actor:
                        report.update(status='INCOMPLETE', error='C++ EOF/counters or oracle count incomplete')
            report['known_actual_counts'] = total
            report['actual_counts'] = None if unknown else total
            report['elapsed_s'] = clock()-started
            if report['elapsed_s'] >= LIMITS['worker_wall_s']:
                report['error'] = (report['error'] or '')+'; worker wall deadline exhausted during finalization'
            if report['error'] or not report['cleanup_complete']:
                report['status'] = 'INCOMPLETE'
            atomic_json(output/'report.json', report)
    return report


def subreaper():
    # The C++ service has its own session; adoption also covers a killed worker.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot become owned-child subreaper')


def descendants(parent):
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            rows[int(path.parent.name)] = (int(fields[1]), int(fields[19]), fields[0])
        except (OSError, ValueError, IndexError):
            continue
    family = {parent}
    while True:
        added = {pid for pid, (ppid, _, _) in rows.items() if ppid in family}-family
        if not added:
            break
        family.update(added)
    return {pid: rows[pid][1] for pid in family-{parent} if rows[pid][2] not in ('Z', 'X')}


def cleanup_owned_tree(process, *, clock=time.monotonic, sleeper=time.sleep):
    started, sent = clock(), []
    while clock()-started < 7:
        remaining = descendants(os.getpid())
        if not remaining:
            break
        elapsed = clock()-started
        number = signal.SIGINT if elapsed < 3 else signal.SIGTERM if elapsed < 5 else signal.SIGKILL
        for pid, birth in remaining.items():
            try:
                fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[19]) == birth:
                    os.kill(pid, number)
                    sent.append({'pid': pid, 'birth_tick': birth, 'signal': signal.Signals(number).name})
            except (OSError, ValueError, IndexError):
                continue
        sleeper(.05)
    if process is not None:
        process.poll()
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    return {'method': 'own_subreaper_descendants_with_linux_birth_tick', 'signals': sent,
            'remaining': descendants(os.getpid()), 'elapsed_s': clock()-started}


def execute(output=None, *, bundle_loader=load_eligible, source_loader=source_identity, memory_reader=read_memory,
            process_factory=subprocess.Popen, adopter=subreaper, cleanup=cleanup_owned_tree,
            proof_checker=verify_completed_worker, clock=time.monotonic, sleeper=time.sleep, root=ROOT):
    output = Path(output or root/OUTPUT).absolute()
    require(output == root/OUTPUT and not output.exists(), 'new19 fixed output already used or overridden; no retry')
    bundle = bundle_loader()
    sources = source_loader()
    first = memory_reader()
    require(type(first.get('effective_available_bytes')) is int and first['effective_available_bytes'] >= 0,
            'invalid admission resource sample')
    if first['effective_available_bytes'] < LIMITS['start_effective_bytes']:
        return {'status': 'NOT_READY', 'execution_started': False, 'sample': first}
    output.mkdir(parents=True, exist_ok=False)
    release = {'format': 'd1-cpp-closedloop19-release-v1',
               'owner_scope': '把这些优化点，在这台电脑能力之内的都完成', 'source_identity': sources,
               'old18_qualification': bundle['identity'], 'budget': BUDGET, 'limits': LIMITS,
               'one_attempt': True, 'training_allowed': False, 'retry_or_resume_allowed': False}
    atomic_json(output/'release.json', release)
    report = {'format': 'd1-cpp-closedloop19-execution-v1', 'status': 'RUNNING', 'source_identity': sources,
              'release_sha256': sha(output/'release.json'), 'old18_qualification': bundle['identity'],
              'supervisor_pid': os.getpid(), 'nonce': secrets.token_hex(24), 'actual_counts': None,
              'known_actual_counts': dict.fromkeys(BUDGET, 0), 'cleanup_complete': True, 'error': None,
              'all_36_reference_pass': False, 'new_training_transitions': 0,
              'minimum_effective_bytes': first['effective_available_bytes']}
    started = clock()
    process = None

    def sample(phase):
        row = dict(memory_reader(), phase=phase, elapsed_s=clock()-started)
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        value = row.get('effective_available_bytes')
        require(type(value) is int and value >= 0, 'resource sample unknown')
        report['minimum_effective_bytes'] = min(report['minimum_effective_bytes'], value)
        require(value >= LIMITS['reserve_bytes'], 'effective RAM below 4 GiB')

    command = [sys.executable, '-B', str(root/'tools19/run_cpp_closedloop19.py'), '--worker',
               '--release', str(output/'release.json'), '--output', str(output/'worker')]
    report['command'] = command
    atomic_json(output/'report.json', report)
    with termination_guard() as control:
        try:
            sample('initial')
            adopter()
            with (output/'worker.stdout.log').open('xb') as stream:
                control['defer'] = True
                try:
                    process = process_factory(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, D1_CPP19_NONCE=report['nonce'], PYTHONDONTWRITEBYTECODE='1',
                                 OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                finally:
                    control['defer'] = False
                require(control['received'] is None, 'termination during worker spawn')
                report['worker_pid'] = process.pid
                atomic_json(output/'report.json', report)
                while process.poll() is None:
                    sample('running')
                    require(clock()-started < LIMITS['supervisor_stop_s'], 'worker process deadline exhausted')
                    sleeper(LIMITS['poll_s'])
                sample('observed_exit')
                require(clock()-started < LIMITS['supervisor_stop_s'], 'worker exit observed after process deadline')
            require(process.returncode == 0, 'closed-loop worker exited unsuccessfully')
            child_path = output/'worker/report.json'
            child = read_json(child_path)
            require(child.get('format') == 'd1-cpp-closedloop19-worker-v1'
                    and child.get('release_sha256') == report['release_sha256']
                    and child.get('source_identity') == sources and child.get('old18_qualification') == bundle['identity']
                    and child.get('status') == 'COMPLETE' and child.get('error') is None
                    and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
                    and valid_counts(child.get('actual_counts'))
                    and [(row['sequence'], row['delay_ticks'], row['reset_seed']) for row in child['cases']] == matrix()
                    and len(child.get('golden', [])) == 64
                    and [row['index'] for row in child['golden']] == list(range(64)), 'worker proof incomplete')
            protocol_path = output/'worker/protocol.json'
            require(child.get('protocol_sha256') == sha(protocol_path), 'worker protocol changed')
            proof_checker(output/'worker', child, release, bundle)
            require(bundle_loader()['identity'] == bundle['identity'] and source_loader() == sources, 'execution inputs changed')
            require(sha(output/'release.json') == report['release_sha256'], 'release changed during execution')
            report.update(status='COMPLETE', actual_counts=child['actual_counts'],
                known_actual_counts=child['actual_counts'], worker_report_sha256=sha(child_path),
                artifacts_sha256={str(path.relative_to(output)): sha(path) for path in (output/'worker').rglob('*') if path.is_file()},
                all_36_reference_pass=child['all_36_reference_pass'])
        except BaseException as exc:
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            try:
                report['cleanup'] = cleanup(process)
                report['cleanup_complete'] = report['cleanup']['remaining'] == {}
            except BaseException as exc:
                report.update(cleanup_complete=False, error=(report['error'] or '')+f'; owned cleanup {exc}')
            if report['status'] != 'COMPLETE':
                try:
                    child = read_json(output/'worker/report.json')
                    protocol_path = output/'worker/protocol.json'
                    protocol = read_json(protocol_path)
                    require(child.get('release_sha256') == report['release_sha256'] and child.get('source_identity') == sources
                            and child.get('old18_qualification') == bundle['identity']
                            and child.get('protocol_sha256') == sha(protocol_path)
                            and child.get('format') == 'd1-cpp-closedloop19-worker-v1'
                            and protocol.get('format') == 'd1-cpp-closedloop19-protocol-v1'
                            and protocol.get('release_sha256') == report['release_sha256']
                            and protocol.get('source_identity') == sources
                            and protocol.get('old18_qualification') == bundle['identity']
                            and protocol.get('budget') == BUDGET and protocol.get('limits') == LIMITS,
                            'foreign partial worker proof')
                    known = child['known_actual_counts']
                    require(valid_counts(known, partial=True), 'unknown partial subtotal')
                    report['known_actual_counts'] = known
                    if valid_counts(child.get('actual_counts'), partial=True):
                        report['actual_counts'] = child['actual_counts']
                except BaseException:
                    pass
            if not report['cleanup_complete'] or control['received']:
                report['status'] = 'INCOMPLETE'
            report['elapsed_s'] = clock()-started
            if report['elapsed_s'] >= LIMITS['total_wall_s']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; total wall deadline exhausted')
            atomic_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--release', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    require(not (args.worker and args.execute), 'choose one entry mode')
    if args.worker:
        require(args.release is not None and args.output is not None, 'worker requires supervised paths')
        result = run_worker(args.release, args.output)
    elif args.execute:
        result = execute(args.output)
    else:
        print(json.dumps({'status': 'PLAN_ONLY', 'output': str(ROOT/OUTPUT), 'budget': BUDGET, 'limits': LIMITS,
                          'training_started': False, 'onnx_sessions_created': 0, 'physics_steps_executed': 0,
                          'requires': 'completed winning18 original36 plus holdout36; committed new19 sources'}, indent=2))
        return 0
    print(json.dumps({key: result.get(key) for key in ('status', 'actual_counts', 'all_36_reference_pass', 'error')}, indent=2))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
