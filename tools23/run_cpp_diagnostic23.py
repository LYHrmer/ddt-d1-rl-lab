"""Independent baseline17 C++ actor diagnostic. Never trains or grants qualification."""
# Broad catches retain signal/partial evidence before bounded cleanup.
# ruff: noqa: BLE001
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools17 import provenance17 as provenance
from tools19 import run_cpp_closedloop19 as engine

actor_api = engine.actor_api
require, sha, atomic_json = actor_api.require, actor_api.sha, actor_api.atomic_json
read_json, read_memory = engine.read_json, engine.read_memory
matrix, valid_counts, add_counts = engine.matrix, engine.valid_counts, engine.add_counts
verify_inferences = engine.verify_inferences
termination_scope, termination_guard = engine.termination_scope, engine.termination_guard
subreaper, cleanup_owned_tree = engine.subreaper, engine.cleanup_owned_tree
BUDGET = dict(engine.BUDGET)
LIMITS = dict(engine.LIMITS)
OUTPUT = 'evidence/cpp_diagnostic23/execution_01'
BASELINE = 'evidence/closeout17'
TRAINING_ROOT = Path('/home/lyh/ddt_comparison16_replacement_20261017')
CHECKPOINT = '428ff0ce9e8e88e10dabf94801a57b88de7e76d07b42800553b69c7f590e4eff'
SCOPE = {'diagnostic_only': True, 'qualification_eligible': False, 'deployment_qualified': False,
         'baseline_full_delay_qualification': 'NOT_PASSED', 'old19_eligibility_granted': False}
NEW_SOURCES = ('tools23/run_cpp_diagnostic23.py', 'tests/test_cpp_diagnostic23.py', 'docs/cpp_diagnostic23.md')


def source_names(root=ROOT):
    return sorted(set(NEW_SOURCES) | set(engine.SOURCES) | set(provenance.authored_sources(root))
                  | {'reports/closeout17_20261018.json', 'tools16/train_comparison16.py', 'tools16/run_comparison16.py'})


def check_sources(proof, root=ROOT):
    require(isinstance(proof, dict) and re.fullmatch('[0-9a-f]{40}', proof.get('git_commit', '')),
            'committed new23 source identity required')
    require(set(proof.get('sources_sha256', {})) == set(source_names(root)), 'new23 source closure differs')
    for name, digest in proof['sources_sha256'].items():
        path = root/name
        require(not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path) == digest,
                'new23 source changed: '+name)
        data = subprocess.check_output(['git', 'show', f'{proof["git_commit"]}:{name}'], cwd=root,
                                       stderr=subprocess.PIPE, timeout=10)
        require(hashlib.sha256(data).hexdigest() == digest, 'commit new23 sources before execution: '+name)


def source_identity(root=ROOT):
    proof = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=10).strip(),
             'sources_sha256': {name: sha(root/name) for name in source_names(root)}}
    check_sources(proof, root)
    return proof


def load_baseline17(root=ROOT, *, loader=None, evaluator=None, helper_loader=None):
    """Only metadata, hashes and existing NumPy traces. No model/session/environment."""
    if loader is None:
        loader = provenance.load_completed_control
    if evaluator is None:
        from tools17 import evaluate_formal17 as evaluator
    if helper_loader is None:
        from tools16.evaluate_comparison16 import frozen_helpers as helper_loader
    summary_path = root/'reports/closeout17_20261018.json'
    summary = read_json(summary_path)
    require(summary.get('engineering_validation_status') == 'COMPLETE'
            and summary.get('full_delay_reference_qualification') == 'NOT_PASSED'
            and summary.get('checkpoint_sha256') == CHECKPOINT, 'published baseline17 identity differs')
    training_release, training_protocol, training_result, checkpoint = loader(TRAINING_ROOT, repository_root=root)
    require(sha(checkpoint) == CHECKPOINT and training_result.get('status') == 'COMPLETE', 'partial or foreign baseline')
    release_path = root/BASELINE/'release_01.json'
    old = read_json(release_path)
    provenance.check_sources(old['source_identity'], root=root)
    require(old.get('status') == 'AUTHORIZED_FROZEN' and old.get('checkpoint_sha256') == CHECKPOINT
            and old.get('source_identity') == summary.get('source_identity'), 'old17 release identity differs')
    execution = root/old['output_root']
    require(execution == root/BASELINE/'execution_01', 'old17 output differs')
    outer = read_json(execution/'report.json')
    require(outer.get('status') == 'COMPLETE' and outer.get('error') is None
            and set(outer.get('stages', {})) == {'export', 'cpp', 'formal'}, 'old17 incomplete')
    proof = {str(path): sha(path) for path in (summary_path, release_path, execution/'report.json', checkpoint,
             checkpoint.parent/'protocol.json', checkpoint.parent/'result.json')}
    for name, stage in outer['stages'].items():
        receipt = execution/name/'report.json'
        require(stage.get('status') == 'COMPLETE' and stage.get('cleanup_complete') is True
                and stage.get('returncode') == 0 and stage.get('receipt_sha256') == sha(receipt), 'old17 stage changed: '+name)
        for name, digest in stage['artifacts_sha256'].items():
            path = Path(name)
            require(not path.is_symlink() and path.resolve().is_relative_to(execution) and sha(path) == digest,
                    'old17 artifact changed: '+name)
            proof[str(path)] = digest
    model, numerical = evaluator.check_numeric_inputs(execution/'export', execution/'cpp/report.json', old)
    exported = read_json(execution/'export/report.json')
    require(numerical['onnx_sha256'] == summary['onnx']['sha256'] and exported['checkpoint_sha256'] == CHECKPOINT
            and exported['fixtures_sha256'] == summary['onnx']['fixture_sha256'], 'published ONNX/fixture changed')
    from tools17.verify_cpp17 import check_fixtures
    fixtures = check_fixtures(read_json(execution/'export/golden_fixtures.json'))
    formal = read_json(execution/'formal/report.json')
    require(formal.get('status') == 'COMPLETE' and formal.get('cleanup_complete') is True
            and formal.get('inputs_unchanged') is True and formal.get('error') is None
            and formal.get('all_36_reference_pass') is False and formal.get('checkpoint_sha256') == CHECKPOINT
            and formal.get('onnx_sha256') == sha(model)
            and [(v['sequence'], v['delay_ticks'], v['reset_seed']) for v in formal.get('cases', [])] == matrix(),
            'old17 matrix/provenance differs')
    helpers = helper_loader(training_release)
    total = dict.fromkeys(BUDGET, 0)
    for item in formal['cases']:
        evaluator.verify_trace(execution/'formal', item, helpers)
        total = add_counts(total, item['actual_counts'])
    require(formal.get('actual_counts') == total and formal.get('by_delay_ms') == summaries(formal['cases']),
            'old17 aggregate metrics differ')
    build_path = root/'runtime19/build/build.json'
    build, binary = read_json(build_path), root/'runtime19/build/actor_server19'
    require(build.get('status') == 'PASS' and build.get('server_started') is False
            and build.get('onnx_sessions_created') == 0 and build.get('robot_models_constructed') == 0
            and build.get('physics_steps_executed') == 0 and sha(binary) == build.get('binary_sha256')
            and build['inputs_sha256'].get(str(root/'runtime19/actor_server19.cpp')) == sha(root/'runtime19/actor_server19.cpp')
            and build['inputs_sha256'].get(str(root/'tools19/cpp_actor19.py')) == sha(root/'tools19/cpp_actor19.py')
            and all(sha(p) == d for p, d in build['inputs_sha256'].items()), 'C++ build provenance differs')
    actor_identities = {'model_sha256': sha(model), 'binary_sha256': sha(binary),
                        'source_sha256': sha(root/'runtime19/actor_server19.cpp'),
                        'adapter_sha256': sha(root/'tools19/cpp_actor19.py'),
                        'ort_library_sha256': build['inputs_sha256'][str(actor_api.ORT_LIBRARY)]}
    proof.update(build['inputs_sha256'])
    proof.update({str(build_path): sha(build_path), str(binary): sha(binary)})
    return {'old_release': training_release, 'training_protocol': training_protocol, 'model': model, 'binary': binary,
            'fixtures': fixtures, 'identity': {'checkpoint_sha256': CHECKPOINT, 'onnx_sha256': sha(model),
                'fixtures_sha256': exported['fixtures_sha256'], 'binary_sha256': sha(binary),
                'actor_identities': actor_identities, 'baseline_all_36_reference_pass': False, 'baseline_by_delay_ms': summaries(formal['cases']),
                'artifacts_sha256': proof}}


def summaries(rows):
    return {str(delay*20): {'cases': len(selected),
            'full_horizons': sum(row['done_reason'] == 'HORIZON_12S' for row in selected),
            'failures': sum(row['failure'] is True for row in selected),
            'reference_pass_count': sum(row['reference_thresholds_met'] is True for row in selected)}
            for delay in range(4) for selected in [[row for row in rows if row['delay_ticks'] == delay]]}


def valid_scope(value):
    return all(type(value.get(k)) is type(v) and value[k] == v for k, v in SCOPE.items())


def validate_release(release):
    require(release.get('format') == 'd1-cpp-diagnostic23-release-v1'
            and release.get('budget') == BUDGET and release.get('limits') == LIMITS
            and release.get('one_attempt') is True and release.get('training_allowed') is False
            and release.get('retry_or_resume_allowed') is False
            and valid_scope(release), 'new23 release scope/budget differs')


def validate_launch(release_path, output):
    parent = Path(release_path).resolve().parent
    report = read_json(parent/'report.json')
    require(Path(release_path).resolve() == ROOT/OUTPUT/'release.json' and Path(output).resolve() == parent/'worker'
            and report.get('format') == 'd1-cpp-diagnostic23-execution-v1' and report.get('status') == 'RUNNING'
            and report.get('release_sha256') == sha(release_path) and report.get('supervisor_pid') == os.getppid()
            and report.get('nonce') == os.environ.get('D1_CPP23_NONCE') and bool(report.get('nonce')),
            'worker requires its active bounded new23 parent')


def verify_completed_worker(directory, child, release, bundle, *, helpers=None, verifier=None):
    import numpy as np
    validate_release(release)
    if helpers is None:
        from tools16.evaluate_comparison16 import frozen_helpers
        helpers = frozen_helpers(bundle['old_release'])
    if verifier is None:
        from tools17.evaluate_formal17 import verify_trace as verifier
    protocol = read_json(Path(directory)/'protocol.json')
    require(child.get('format') == 'd1-cpp-diagnostic23-worker-v1'
            and child.get('release_sha256') == sha(Path(directory).parent/'release.json')
            and child.get('source_identity') == release['source_identity']
            and child.get('baseline17_provenance') == bundle['identity']
            and child.get('status') == 'COMPLETE' and child.get('error') is None
            and child.get('cleanup_complete') is True and child.get('inputs_unchanged') is True
            and child.get('protocol_sha256') == sha(Path(directory)/'protocol.json')
            and protocol.get('format') == 'd1-cpp-diagnostic23-protocol-v1'
            and protocol.get('release_sha256') == child['release_sha256']
            and protocol.get('source_identity') == release['source_identity']
            and protocol.get('baseline17_provenance') == bundle['identity']
            and protocol.get('budget') == BUDGET and protocol.get('limits') == LIMITS
            and protocol.get('action_driver') == 'cpp_ort1.10_only'
            and [tuple(row) for row in protocol.get('scenarios', [])] == matrix()
            and valid_scope(child) and valid_scope(protocol), 'worker proof identity differs')
    require([(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in child.get('cases', [])] == matrix(), 'worker matrix differs')
    total, maximum = dict.fromkeys(BUDGET, 0), 0.
    for row in child['cases']:
        require(valid_counts(row.get('actual_counts')) and 1 <= row['actual_counts']['policy_transitions'] <= 600, 'invalid case counts')
        verifier(directory, row, helpers)
        maximum = max(maximum, verify_inferences(directory, row))
        total = add_counts(total, row['actual_counts'])
    require(child.get('actual_counts') == child.get('known_actual_counts') == total
            and child.get('maximum_same_input_error') == maximum and child.get('by_delay_ms') == summaries(child['cases']),
            'worker aggregate differs')
    golden = child.get('golden', [])
    require(len(golden) == 64 and [v.get('index') for v in golden] == list(range(64)), 'golden indices differ')
    for row in golden:
        fixture = bundle['fixtures'][row['index']]
        actual, oracle = (np.asarray(row.get(key), dtype=np.float32) for key in ('actions', 'python_ort_actions'))
        require(actual.shape == oracle.shape == (16,) and bool(np.isfinite(actual).all()) and bool(np.isfinite(oracle).all())
                and bool((np.abs(actual) <= 1).all()) and bool((np.abs(oracle) <= 1).all()), 'invalid golden actions')
        delta = max(float(np.max(np.abs(actual-np.asarray(fixture[key], dtype=np.float32))))
                    for key in ('torch_actions', 'python_ort_actions'))
        delta = max(delta, float(np.max(np.abs(actual-oracle))))
        require(type(row.get('max_absolute_error')) in (int, float) and row['max_absolute_error'] == delta <= 1e-5,
                'golden parity differs')
    count = 64+total['policy_transitions']
    actor = child.get('cpp_actor') or {}
    verify_actor_receipt(directory, actor, release, bundle, count)
    require(actor.get('status') == 'CLOSED' and actor.get('cleanup_complete') is True
            and all(type(actor.get(k)) is int and actor[k] == count
                    for k in ('requests_completed', 'inference_attempts', 'inference_returns'))
            and type(child.get('oracle_calls')) is int and child['oracle_calls'] == count, 'C++ EOF/counters differ')
    require(child.get('all_36_reference_pass') is (total == BUDGET and all(r['reference_thresholds_met'] is True for r in child['cases'])),
            'task reference flag differs')
    return total


def verify_actor_receipt(directory, actor, release, bundle, count):
    """Bridge saved service READY/EOF identities to the selected baseline/build."""
    path = Path(directory)/'cpp_actor/report.json'
    identities = bundle['identity']['actor_identities']
    require(actor == read_json(path) and actor.get('format') == 'd1-cpp-actor19-adapter-v1'
            and actor.get('identities') == identities and actor.get('process_started') is True
            and actor.get('error') is None and type(actor.get('returncode')) is int and actor['returncode'] == 0
            and type(actor.get('requests_attempted')) is int and actor['requests_attempted'] == count,
            'saved C++ adapter receipt or model/build identity differs')
    require(identities['source_sha256'] == release['source_identity']['sources_sha256']['runtime19/actor_server19.cpp']
            and identities['adapter_sha256'] == release['source_identity']['sources_sha256']['tools19/cpp_actor19.py']
            and identities['model_sha256'] == bundle['identity']['onnx_sha256']
            and identities['binary_sha256'] == bundle['identity']['binary_sha256'], 'C++ provenance bridge differs')
    ready, ended = actor.get('ready') or {}, actor.get('eof_receipt') or {}
    require(ready.get('format') == 'd1-cpp-actor19-ready-v1' and ready.get('status') == 'READY'
            and all(ready.get(k) == identities[k] for k in ('binary_sha256', 'model_sha256', 'source_sha256', 'ort_library_sha256'))
            and json.dumps(ready.get('signature'), sort_keys=True) == json.dumps(actor_api.SIGNATURE, sort_keys=True)
            and ready.get('onnxruntime_version') == '1.10.0'
            and type(ready.get('cpu_threads')) is int and ready['cpu_threads'] == 1
            and type(ready.get('max_requests')) is int and ready['max_requests'] == LIMITS['max_cpp_inferences']
            and all(type(ready.get(k)) is int and ready[k] == 0 for k in
                    ('inference_attempts', 'inference_returns', 'robot_models_constructed', 'physics_steps_executed')),
            'saved C++ READY differs')
    require(ended.get('format') == 'd1-cpp-actor19-closed-v1' and ended.get('status') == 'CLOSED'
            and all(type(ended.get(k)) is int and ended[k] == count for k in ('requests', 'inference_attempts', 'inference_returns'))
            and all(type(ended.get(k)) is int and ended[k] == 0 for k in ('robot_models_constructed', 'physics_steps_executed')),
            'saved C++ EOF differs')


def run_worker(release_path, output, *, bundle_loader=load_baseline17, source_checker=check_sources,
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
    validate_release(release)
    source_checker(release['source_identity'])
    launch_validator(release_path, output)
    bundle = bundle_loader()
    require(bundle['identity'] == release['baseline17_provenance'], 'baseline17 provenance changed')
    require(not output.exists(), 'worker output must be new')
    output.mkdir(parents=True, exist_ok=False)
    report = {'format': 'd1-cpp-diagnostic23-worker-v1', 'status': 'INCOMPLETE', 'release_sha256': sha(release_path),
              'source_identity': release['source_identity'], 'baseline17_provenance': bundle['identity'],
              'actual_counts': None, 'known_actual_counts': dict.fromkeys(BUDGET, 0), 'cases': [],
              'golden': [], 'oracle_calls': 0, 'cleanup_complete': True, 'inputs_unchanged': False,
              'error': None, 'cpp_actor': None, 'all_36_reference_pass': False, 'new_training_transitions': 0, **SCOPE}
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
            protocol = {'format': 'd1-cpp-diagnostic23-protocol-v1', 'release_sha256': report['release_sha256'],
                        'source_identity': release['source_identity'], 'baseline17_provenance': bundle['identity'],
                        'budget': BUDGET, 'limits': LIMITS,
                        'backend': {'kind': 'cpp_ort1.10_actor_with_same_input_python_ort_oracle', 'oracle': rt.backend},
                        'scenarios': matrix(), 'thresholds': helpers.THRESHOLDS,
                        'action_driver': 'cpp_ort1.10_only', 'oracle': 'same_actual_input_python_ort_no_extra_physics', **SCOPE}
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
            require(bundle_loader()['identity'] == bundle['identity'], 'baseline17 inputs changed during closed loop')
            source_checker(release['source_identity'])
            require(sha(release_path) == report['release_sha256'], 'new23 release changed')
            report['inputs_unchanged'] = True
            report['all_36_reference_pass'] = total == BUDGET and all(row['reference_thresholds_met'] is True for row in report['cases'])
            report['by_delay_ms'] = summaries(report['cases'])
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


def execute(output=None, *, bundle_loader=load_baseline17, source_loader=source_identity, memory_reader=read_memory,
            process_factory=subprocess.Popen, adopter=subreaper, cleanup=cleanup_owned_tree,
            proof_checker=verify_completed_worker, clock=time.monotonic, sleeper=time.sleep, root=ROOT):
    output = Path(output or root/OUTPUT).absolute()
    require(output == root/OUTPUT and not output.exists(), 'new23 fixed output already used or overridden; no retry')
    sources = source_loader()
    bundle = bundle_loader()
    first = memory_reader()
    require(type(first.get('effective_available_bytes')) is int and first['effective_available_bytes'] >= 0,
            'invalid admission resource sample')
    if first['effective_available_bytes'] < LIMITS['start_effective_bytes']:
        return {'status': 'NOT_READY', 'execution_started': False, 'sample': first}
    output.mkdir(parents=True, exist_ok=False)
    release = {'format': 'd1-cpp-diagnostic23-release-v1',
               'owner_scope': '把这些优化点，在这台电脑能力之内的都完成', 'source_identity': sources,
               'baseline17_provenance': bundle['identity'], 'budget': BUDGET, 'limits': LIMITS,
               'one_attempt': True, 'training_allowed': False, 'retry_or_resume_allowed': False, **SCOPE}
    atomic_json(output/'release.json', release)
    report = {'format': 'd1-cpp-diagnostic23-execution-v1', 'status': 'RUNNING', 'source_identity': sources,
              'release_sha256': sha(output/'release.json'), 'baseline17_provenance': bundle['identity'],
              'supervisor_pid': os.getpid(), 'nonce': secrets.token_hex(24), 'actual_counts': None,
              'known_actual_counts': dict.fromkeys(BUDGET, 0), 'cleanup_complete': True, 'error': None,
              'all_36_reference_pass': False, 'new_training_transitions': 0, **SCOPE,
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

    command = [sys.executable, '-B', str(root/'tools23/run_cpp_diagnostic23.py'), '--worker',
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
                        env=dict(os.environ, D1_CPP23_NONCE=report['nonce'], PYTHONDONTWRITEBYTECODE='1',
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
            require(child.get('format') == 'd1-cpp-diagnostic23-worker-v1'
                    and child.get('release_sha256') == report['release_sha256']
                    and child.get('source_identity') == sources and child.get('baseline17_provenance') == bundle['identity']
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
                all_36_reference_pass=child['all_36_reference_pass'], by_delay_ms=child['by_delay_ms'])
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
                            and child.get('baseline17_provenance') == bundle['identity']
                            and child.get('protocol_sha256') == sha(protocol_path)
                            and child.get('format') == 'd1-cpp-diagnostic23-worker-v1'
                            and protocol.get('format') == 'd1-cpp-diagnostic23-protocol-v1'
                            and protocol.get('release_sha256') == report['release_sha256']
                            and protocol.get('source_identity') == sources
                            and protocol.get('baseline17_provenance') == bundle['identity']
                            and protocol.get('budget') == BUDGET and protocol.get('limits') == LIMITS,
                            'foreign partial worker proof')
                    require(valid_scope(child) and valid_scope(protocol), 'partial diagnostic scope differs')
                    known = child['known_actual_counts']
                    require(valid_counts(known, partial=True), 'unknown partial subtotal')
                    report['known_actual_counts'] = known
                    if valid_counts(child.get('actual_counts'), partial=True) and child['actual_counts'] == known:
                        report['actual_counts'] = child['actual_counts']
                except BaseException as exc:
                    report['partial_receipt_error'] = f'{type(exc).__name__}: {exc}'
            if not report['cleanup_complete'] or control['received']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; cleanup or termination incomplete')
            report['elapsed_s'] = clock()-started
            if report['elapsed_s'] >= LIMITS['total_wall_s']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; total wall deadline exhausted')
            atomic_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--release', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        require(args.release is not None and args.output is not None, 'worker requires supervised paths')
        result = run_worker(args.release, args.output)
    elif args.execute:
        require(args.release is None, 'release is created exclusively by the parent')
        result = execute(args.output)
    else:
        require(args.release is None and args.output is None, 'plan uses fixed new23 paths')
        print(json.dumps({'status': 'PLAN_ONLY', 'output': str(ROOT/OUTPUT), 'budget': BUDGET, 'limits': LIMITS, **SCOPE,
            'new_training_transitions': 0, 'onnx_sessions_created': 0, 'robot_models_constructed': 0, 'physics_steps_executed': 0,
            'requires': 'unchanged completed baseline17/control ONNX and committed new23 sources; no old19 eligibility'}, indent=2))
        return 0
    print(json.dumps({key: result.get(key) for key in ('status', 'actual_counts', 'all_36_reference_pass', 'qualification_eligible', 'error')}, indent=2))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
