"""Independent closeout of completed zero18 only; never resumes or trains."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from tools16.contract16 import atomic_json, expected_counts, read_memory
from tools16.run_comparison16 import compare_prefix, termination_guard
from tools19.run_cpp_closedloop19 import cleanup_owned_tree, subreaper

OUTPUT = 'evidence/completed_zero21/execution_01'
PLAN18 = 'evidence/local_optimization18/plan_01.json'
SOURCES = ('tools21/closeout_completed18.py', 'tests/test_closeout_completed18.py',
           'docs/completed_zero21.md', 'tools19/run_cpp_closedloop19.py', 'tools19/cpp_actor19.py')
STAGES = {'export': 180, 'cpp': 90, 'formal': 300, 'holdout': 300}
CAP = expected_counts(21600)
BUDGET = expected_counts(43200)
LIMITS = {'start_effective_bytes': 6 << 30, 'reserve_bytes': 4 << 30, 'total_wall_s': 900, 'poll_s': 1.}
ZERO = expected_counts(0)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    require(isinstance(value, dict), 'JSON object required')
    return value


def artifacts(directory):
    return {str(path): sha(path) for path in Path(directory).rglob('*') if path.is_file()}


def source_identity(root=ROOT):
    proof = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
             'sources_sha256': {name: sha(root/name) for name in SOURCES}}
    check_sources(proof, root=root)
    return proof


def check_sources(proof, root=ROOT):
    require(set(proof.get('sources_sha256', {})) == set(SOURCES), 'new21 source set differs')
    for name, digest in proof['sources_sha256'].items():
        path = root/name
        require(not path.is_symlink() and sha(path) == digest, 'new21 source changed: '+name)
        blob = subprocess.check_output(['git', 'show', proof['git_commit']+':'+name], cwd=root, stderr=subprocess.PIPE)
        require(hashlib.sha256(blob).hexdigest() == digest, 'commit new21 sources first: '+name)


def training_counts(result, transitions):
    workers = result.get('worker_counts')
    require(result.get('per_worker_counts') == workers and isinstance(workers, list) and len(workers) == 4,
            'four identical worker counter views required')
    actual = result.get('actual_counts', {})
    require(set(actual) >= set(ZERO) and all(type(v) is int and v >= 0 for v in actual.values()), 'invalid training counts')
    for row in workers:
        require(isinstance(row, dict) and set(row) == set(actual)
                and all(type(v) is int and v >= 0 for v in row.values())
                and all(row[k] == v for k, v in expected_counts(transitions//4).items()), 'incomplete worker')
    require(all(actual[key] == sum(row[key] for row in workers) for key in actual)
            and all(actual[key] == value for key, value in expected_counts(transitions).items()), 'training count sum differs')


def load_completed_zero(root=ROOT, *, api=None, training_reader=None, prefix_compare=compare_prefix):
    """Read only the completed zero/E0, including their exact artifact manifests."""
    from tools18 import common18, export_delay18
    api = api or common18
    training_reader = training_reader or export_delay18.training_identity
    plan_path = root/PLAN18
    plan = api.load_plan(plan_path)
    run = root/plan['output_root']
    outer = read_json(run/'report.json')
    require(outer.get('status') == 'STOPPED' and outer.get('cleanup_complete') is True
            and outer.get('plan_sha256') == sha(plan_path) and outer.get('source_identity') == plan['source_identity'],
            'stopped18 source/cleanup proof differs')
    bound = {str(plan_path): sha(plan_path), str(run/'report.json'): sha(run/'report.json')}
    results = {}
    for stage, transitions in (('preflight_zero', 32768), ('train_zero', 1048576)):
        directory = run/stage
        record = outer.get('stages', {}).get(stage, {})
        require(record.get('status') == 'COMPLETE' and record.get('cleanup_complete') is True
                and type(record.get('exit_code')) is int and record['exit_code'] == 0
                and record.get('actual_counts') == expected_counts(transitions)
                and artifacts(directory) == record.get('artifacts_sha256')
                and sha(directory/'result.json') == record.get('receipt_sha256'), 'completed zero stage changed: '+stage)
        result, protocol = read_json(directory/'result.json'), read_json(directory/'protocol.json')
        require(result.get('status') == 'COMPLETE' and result.get('error') is None
                and result.get('cleanup_complete') is True and result.get('inputs_unchanged') is True
                and result.get('new_trained_transitions') == transitions
                and result.get('protocol_sha256') == sha(directory/'protocol.json')
                and all(item.get('stage') == stage and item.get('mode') == 'zero'
                        and item.get('source_identity') == plan['source_identity']
                        and item.get('plan_sha256') == sha(plan_path) for item in (result, protocol)),
                'zero stage result/protocol differs')
        training_counts(result, transitions)
        results[stage] = result
        bound.update(record['artifacts_sha256'])
    prefix = prefix_compare(run/'preflight_zero/ledger.jsonl', run/'train_zero/ledger.jsonl')
    initial = results['train_zero'].get('initial_state')
    require(isinstance(initial, dict) and initial.get('optimizer_restored') is True
            and all(isinstance(initial.get(k), str) and len(initial[k]) == 64
                    and set(initial[k]) <= set('0123456789abcdef') for k in ('policy_state_sha256','optimizer_state_sha256')),
            'initial policy/Adam proof missing')
    require(prefix == (8, None) and results['preflight_zero'].get('initial_state') == initial,
            'E0/full first-eight or initial policy/Adam differs')
    checkpoint, protocol, result, training = training_reader(api, plan, sha(plan_path), 'zero')
    require(result == results['train_zero'] and Path(checkpoint) == run/'train_zero/policy.zip', 'only completed final zero accepted')
    release = api.verify_inputs(plan)[0]
    return {'checkpoint': Path(checkpoint), 'protocol': protocol, 'old_release': release,
            'identity': {'role': 'completed_zero18_only', 'training_identity': training,
                         'old18_artifacts_sha256': bound, 'first_eight_rollouts_equal': True,
                         'random_partial_used': False, 'delay_randomisation_contrast': 'INCOMPLETE'}}


def valid_counts(value, cap=CAP, *, partial=False):
    if (not isinstance(value, dict) or not all(type(v) is int and v >= 0 for v in value.values())
            or not all(type(value.get(k)) is int and 0 <= value[k] <= v for k, v in cap.items())):
        return False
    p, n, d = (value[k] for k in ('policy_transitions', 'native_steps', 'pd_updates'))
    return (max(0, p-1)*10 <= n <= p*10 and max(0, p-1)*5 <= d <= p*5 and 0 <= 2*d-n <= 2) if partial else n == p*10 and d == p*5


def matrix(stage):
    seeds = (2001, 2002, 2003) if stage == 'holdout' else (1001, 1002, 1003)
    return [(name, delay, seed) for name in ('forward_stop', 'reverse_stop', 'turn_stop')
            for delay in (0, 1, 2, 3) for seed in seeds]


def known_counts(stages):
    """Rebuild from unique stage facts, including interruptions during report I/O."""
    values = [row['actual_counts'] if row.get('actual_counts') is not None else row.get('known_actual_counts', ZERO)
              for row in stages.values()]
    return {key: sum(value[key] for value in values) for key in ZERO}


def command(stage, release_path, output, root=ROOT):
    return [sys.executable, '-B', str(root/'tools21/closeout_completed18.py'), '--stage', stage,
            '--release', str(release_path), '--output', str(output)]


def validate_release(path):
    release = read_json(path)
    require(release.get('format') == 'd1-completed-zero21-release-v1' and release.get('budget') == BUDGET
            and release.get('stage_limits_s') == STAGES and release.get('limits') == LIMITS
            and release.get('output_root') == OUTPUT and release.get('new_training_transitions') == 0
            and release.get('retry_or_resume_allowed') is False, 'new21 release scope differs')
    return release


def validate_launch(stage, path, output):
    root = Path(path).parent
    journal = read_json(root/'report.json')
    record = journal.get('stages', {}).get(stage, {})
    require(output == root/stage and journal.get('status') == 'RUNNING' and journal.get('active_stage') == stage
            and journal.get('release_sha256') == sha(path) and journal.get('supervisor_pid') == os.getppid()
            and record.get('status') == 'STARTED' and bool(record.get('nonce'))
            and record['nonce'] == os.environ.get('D1_STAGE21_NONCE')
            and record.get('command') == command(stage, path, output), 'stage requires its active bounded parent')
    require(all(journal['stages'].get(name, {}).get('status') == 'COMPLETE'
                for name in list(STAGES)[:list(STAGES).index(stage)]), 'prior stage incomplete')
    if stage == 'holdout':
        require(read_json(root/'formal/report.json').get('all_36_reference_pass') is True, 'holdout requires all original36')


def stage_identity(stage, release, release_path):
    return {'format': 'd1-completed-zero21-stage-v1', 'stage': stage, 'release_sha256': sha(release_path),
            'source_identity': release['source_identity'], 'completed_zero18': release['completed_zero18']}


def check_stage_identity(child, stage, release, release_path):
    require(all(child.get(k) == v for k, v in stage_identity(stage, release, release_path).items()), 'foreign stage receipt')


def numerical(stage, bundle, parent, output):
    """Old numerical arithmetic; new paths and provenance, never old execution gates."""
    from tools17 import export_completed17 as export
    from tools17 import verify_cpp17 as cpp
    from tools18.export_delay18 import check_model
    checkpoint = bundle['checkpoint']
    if stage == 'export':
        model, versions = export.load_model(checkpoint)
        check_model(model, bundle['protocol'], versions)
        value = export.export_numerical(model, output)
        atomic_json(output/'golden_fixtures.json', export.cpp_fixtures(value))
        return {'onnx_sha256': sha(output/'policy.onnx'), 'fixtures_sha256': sha(output/'golden_fixtures.json'),
                'absolute_limits': export.LIMITS, 'versions': {**versions, **value['numerical_versions']},
                'onnx_ir_version': 7, 'onnx_opset': 11, 'cxx_ort_1_10_status': 'NOT_TESTED',
                **{key: value[key] for key in ('onnx_node_counts', 'onnx_signature', 'parity', 'policy_state_sha256')}}
    directory = parent/'export'
    exported = read_json(directory/'report.json')
    require(exported.get('status') == 'COMPLETE' and exported.get('onnx_sha256') == sha(directory/'policy.onnx')
            and exported.get('fixtures_sha256') == sha(directory/'golden_fixtures.json'), 'export changed')
    fixtures = cpp.check_fixtures(read_json(directory/'golden_fixtures.json'))
    binary, schema = ROOT/'runtime03/build/d1_runtime03', ROOT/'runtime03/schema03d.json'
    identities = {'policy_sha256': sha(directory/'policy.onnx'), 'fixture_sha256': sha(directory/'golden_fixtures.json'),
                  'runtime_binary_sha256': sha(binary), 'schema_sha256': sha(schema)}
    require(all(bundle['old_release']['sources_sha256'].get(name) == identities[key]
                for name, key in (('runtime03/build/d1_runtime03', 'runtime_binary_sha256'),
                                  ('runtime03/schema03d.json', 'schema_sha256'))), 'frozen C++ assets changed')
    argv = [str(binary), '--schema', str(schema), '--onnx-golden', str(directory/'golden_fixtures.json'),
            '--policy', str(directory/'policy.onnx'), '--expected-policy-sha256', identities['policy_sha256'],
            '--output', str(output/'cpp_native')]
    env = dict(os.environ, LD_LIBRARY_PATH=f'/opt/ros/humble/lib:{ROOT/"install/mujoco/lib"}'+
               (':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
    receipt = cpp.invoke_cpp(argv, cwd=ROOT, env=env, stdout_path=output/'stdout.txt', stderr_path=output/'stderr.txt', timeout_s=60)
    require(receipt.get('process_started') is True and receipt.get('returncode') == 0 and receipt.get('error') is None
            and receipt.get('cleanup_complete') is True and not receipt.get('cleanup_signals'), 'C++ process failed')
    native = output/'cpp_native/report.json'
    maximum = cpp.check_native(read_json(native), fixtures, identities)
    return {'identities': identities, 'onnx_sha256': identities['policy_sha256'], 'fixtures_sha256': identities['fixture_sha256'],
            'cpp_native_report_sha256': sha(native), 'export_report_sha256': sha(directory/'report.json'),
            'comparisons': 64, 'max_absolute_error': maximum, 'absolute_tolerance': 1e-5, 'cxx_ort_1_10_status': 'PASS'}


def evaluate(stage, bundle, parent, output, report, poll, *, runtime_factory=None, helpers=None, verifier=None):
    from tools16.evaluate_comparison16 import frozen_helpers
    from tools17.evaluate_formal17 import runtime, verify_trace
    rt = (runtime_factory or runtime)(bundle['old_release'], parent/'export/policy.onnx', bundle['protocol'])
    helpers = helpers or frozen_helpers(bundle['old_release'])
    verifier = verifier or verify_trace
    require(helpers.make_scenarios('formal') == matrix('formal'), 'frozen matrix changed')
    report['backend'] = rt.backend
    protocol = read_json(output/'protocol.json')
    protocol.update(backend=rt.backend, thresholds=helpers.THRESHOLDS, sequences=helpers.SEQUENCES,
                    checkpoint_sha256=bundle['identity']['training_identity']['checkpoint_sha256'])
    atomic_json(output/'protocol.json', protocol)
    report['protocol_sha256'] = sha(output/'protocol.json')
    total, env, unknown, constructing, accounted = dict(ZERO), None, False, False, False
    try:
        def predict(observation):
            poll()
            return rt.predict(observation)
        for name, delay, seed in matrix(stage):
            poll(True)
            accounted, constructing = False, True
            env = rt.make_env(name, helpers.SEQUENCES[name], delay, seed)
            constructing = False
            item = helpers.run_episode(env, predict, name, delay, seed, output/f'{name}_delay{delay}_seed{seed}.npz',
                                       CAP['native_steps']-total['native_steps'])
            require((item['sequence'], item['delay_ticks'], item['reset_seed']) == (name, delay, seed), 'wrong scenario')
            verifier(output, item, helpers)
            require(valid_counts(item.get('actual_counts')), 'invalid case counters')
            total = {key: total[key]+item['actual_counts'][key] for key in ZERO}
            require(valid_counts(total), 'stage budget exceeded')
            accounted = True
            report['cases'].append(item)
            env.close()
            env = None
            report['known_actual_counts'] = dict(total)
            atomic_json(output/'report.json', report)
        report['all_36_reference_pass'] = total == CAP and all(
            row['reference_thresholds_met'] is True and row['failure'] is False
            and row['done_reason'] == 'HORIZON_12S' for row in report['cases'])
    except BaseException:
        if constructing:
            unknown, report['cleanup_complete'] = True, False
        if env is not None and not accounted:
            try:
                counts = env.counters()
                require(valid_counts(counts, partial=True), 'partial counters unknown')
                total = {key: total[key]+counts[key] for key in ZERO}
                require(valid_counts(total, partial=True), 'partial cap exceeded')
            except BaseException:  # noqa: BLE001 - counter failures leave the physical total unknown
                unknown = True
        raise
    finally:
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if env is not None:
                try:
                    env.close()
                except BaseException:
                    report['cleanup_complete'] = False
                    raise
        finally:
            for sig, prior in previous.items():
                signal.signal(sig, prior)
            report['known_actual_counts'], report['actual_counts'] = total, None if unknown else total


def run_stage(release_path, stage, output, *, loader=load_completed_zero, source_checker=check_sources,
              launch_validator=validate_launch, memory_reader=read_memory, clock=time.monotonic,
              numeric=numerical, evaluation=evaluate):
    require(stage in STAGES, 'unknown stage')
    started = clock()
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    release = validate_release(release_path)
    source_checker(release['source_identity'])
    launch_validator(stage, release_path, output)
    bundle = loader()
    require(bundle['identity'] == release['completed_zero18'] and not output.exists(), 'source changed or output already used')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(stage_identity(stage, release, release_path), status='INCOMPLETE',
                  actual_counts=None if stage in ('formal','holdout') else dict(ZERO),
                  known_actual_counts=dict(ZERO), error=None, cleanup_complete=True, inputs_unchanged=False,
                  new_training_transitions=0, random_partial_used=False, cases=[], all_36_reference_pass=False)
    last = -float('inf')
    def poll(force=False, admission=False):
        nonlocal last
        now = clock()
        require(now-started < STAGES[stage], 'stage wall limit')
        if force or now-last >= LIMITS['poll_s']:
            sample = dict(memory_reader(), elapsed_s=now-started)
            value = sample.get('effective_available_bytes')
            with (output/'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
            require(type(value) is int and value >= LIMITS['start_effective_bytes' if admission else 'reserve_bytes'], 'RAM reserve/admission')
            last = now
    with termination_guard() as control:
        try:
            poll(True, admission=True)
            protocol = dict(stage_identity(stage, release, release_path), budget_max=CAP if stage in ('formal','holdout') else ZERO,
                            scenarios=matrix(stage) if stage in ('formal','holdout') else [], max_wall_s=STAGES[stage])
            atomic_json(output/'protocol.json', protocol)
            report['protocol_sha256'] = sha(output/'protocol.json')
            if stage in ('export', 'cpp'):
                report['actual_counts'] = None if stage == 'cpp' else dict(ZERO)
                report.update(numeric(stage, bundle, release_path.parent, output))
                report['actual_counts'] = dict(ZERO)
            else:
                evaluation(stage, bundle, release_path.parent, output, report, poll)
            require(loader()['identity'] == bundle['identity'], 'completed zero inputs changed')
            source_checker(release['source_identity'])
            require(sha(release_path) == report['release_sha256'], 'release changed')
            poll(True)
            require(report['cleanup_complete'] is True, 'stage cleanup incomplete')
            report.update(status='COMPLETE', inputs_unchanged=True)
        except BaseException as exc:  # noqa: BLE001 - persist interruption/error before exiting this owned stage
            report['error'] = f'{type(exc).__name__}: {exc}'
            (output/'error.txt').write_text(traceback.format_exc())
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            report['elapsed_s'] = clock()-started
            if control['received']:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; '+control['received'])
            if report['elapsed_s'] >= STAGES[stage]:
                report.update(status='INCOMPLETE', error=(report['error'] or '')+'; final stage wall limit')
            atomic_json(output/'report.json', report)
    return report


def verify_stage(directory, stage, child, release, release_path, *, helpers=None, verifier=None):
    check_stage_identity(child, stage, release, release_path)
    require(child.get('status') == 'COMPLETE' and child.get('error') is None and child.get('cleanup_complete') is True
            and child.get('inputs_unchanged') is True and child.get('new_training_transitions') == 0
            and child.get('protocol_sha256') == sha(directory/'protocol.json'), 'stage proof incomplete')
    protocol = read_json(directory/'protocol.json')
    check_stage_identity(protocol, stage, release, release_path)
    require(protocol.get('budget_max') == (CAP if stage in ('formal','holdout') else ZERO)
            and protocol.get('scenarios') == ([list(row) for row in matrix(stage)] if stage in ('formal','holdout') else [])
            and protocol.get('max_wall_s') == STAGES[stage],
            'stage protocol scope differs')
    require(valid_counts(child.get('actual_counts')) and (stage in ('formal','holdout') or child['actual_counts'] == ZERO), 'invalid actual counts')
    if stage in ('formal','holdout'):
        from tools16.evaluate_comparison16 import frozen_helpers
        from tools17.evaluate_formal17 import verify_trace
        verifier = verifier or verify_trace
        helpers = helpers or frozen_helpers(load_completed_zero()['old_release'])
        require(protocol.get('thresholds') == helpers.THRESHOLDS and protocol.get('sequences') == helpers.SEQUENCES
                and protocol.get('backend') == child.get('backend')
                and protocol.get('checkpoint_sha256') == release['completed_zero18']['training_identity']['checkpoint_sha256'],
                'evaluation backend/science/model identity differs')
        require([(r['sequence'], r['delay_ticks'], r['reset_seed']) for r in child['cases']] == matrix(stage), '36-case matrix incomplete')
        total = dict(ZERO)
        for row in child['cases']:
            verifier(directory, row, helpers)
            require(valid_counts(row.get('actual_counts')), 'case counters invalid')
            total = {k: total[k]+row['actual_counts'][k] for k in ZERO}
        expected = total == CAP and all(row['reference_thresholds_met'] is True and row['failure'] is False
                                       and row['done_reason'] == 'HORIZON_12S' for row in child['cases'])
        require(total == child['actual_counts'] and child['all_36_reference_pass'] is expected, 'qualification/count summary differs')
    else:
        from tools17 import export_completed17 as export
        from tools17 import verify_cpp17 as cpp
        parent = directory.parent
        exported = child if stage == 'export' else read_json(parent/'export/report.json')
        export_dir = parent/'export'
        require(exported.get('onnx_sha256') == child.get('onnx_sha256') == sha(export_dir/'policy.onnx')
                and exported.get('fixtures_sha256') == child.get('fixtures_sha256') == sha(export_dir/'golden_fixtures.json'),
                'numeric artifacts changed')
        golden = read_json(export_dir/'golden_fixtures.json')
        export.cpp_fixtures({'onnx_node_counts': exported['onnx_node_counts'], 'onnx_signature': exported['onnx_signature'],
                             'parity': exported['parity'], 'rows': golden['fixtures'], 'fixture_seed': golden['seed']})
        fixtures = cpp.check_fixtures(golden)
        if stage == 'cpp':
            native = directory/'cpp_native/report.json'
            require(child.get('export_report_sha256') == sha(export_dir/'report.json')
                    and child.get('cpp_native_report_sha256') == sha(native)
                    and child.get('identities', {}).get('policy_sha256') == child['onnx_sha256']
                    and child['identities'].get('fixture_sha256') == child['fixtures_sha256'], 'C++ numeric identity differs')
            maximum = cpp.check_native(read_json(native), fixtures, child['identities'])
            require(child.get('comparisons') == 64 and child.get('max_absolute_error') == maximum
                    and child.get('absolute_tolerance') == 1e-5 and child.get('cxx_ort_1_10_status') == 'PASS', 'C++ numeric result differs')


def execute(*, root=ROOT, loader=load_completed_zero, source_loader=source_identity, memory_reader=read_memory,
            factory=subprocess.Popen, cleanup=cleanup_owned_tree, adopter=subreaper,
            clock=time.monotonic, sleeper=time.sleep, validator=verify_stage):
    output = root/OUTPUT
    require(not output.exists() and not output.is_symlink(), 'attempt already exists; no retry')
    started = clock()
    sources, bundle = source_loader(), loader()
    first = memory_reader()
    require(type(first.get('effective_available_bytes')) is int and first['effective_available_bytes'] >= 0, 'RAM unknown')
    if first['effective_available_bytes'] < LIMITS['start_effective_bytes']:
        return {'status': 'NOT_READY', 'execution_started': False, 'actual_counts': ZERO}
    output.mkdir(parents=True, exist_ok=False)
    release_path = output/'release.json'
    release = {'format': 'd1-completed-zero21-release-v1', 'owner_scope': '把这些优化点，在这台电脑能力之内的都完成',
               'source_identity': sources, 'completed_zero18': bundle['identity'], 'budget': BUDGET,
               'limits': LIMITS, 'stage_limits_s': STAGES, 'output_root': OUTPUT,
               'new_training_transitions': 0, 'retry_or_resume_allowed': False}
    atomic_json(release_path, release)
    report = {'format': 'd1-completed-zero21-execution-v1', 'status': 'RUNNING', 'release_sha256': sha(release_path),
              'source_identity': sources, 'completed_zero18': bundle['identity'], 'supervisor_pid': os.getpid(),
              'stages': {}, 'active_stage': None, 'known_actual_counts': dict(ZERO), 'actual_counts': None,
              'cleanup_complete': True, 'error': None, 'new_training_transitions': 0, 'random_partial_used': False,
              'original_36_pass': False, 'holdout_36_pass': False, 'full_delay_qualification': False}
    atomic_json(output/'report.json', report)
    process, current = None, None
    def sample(admission=False):
        row = dict(memory_reader(), elapsed_s=clock()-started, stage=current)
        with (output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        value = row.get('effective_available_bytes')
        require(type(value) is int and value >= LIMITS['start_effective_bytes' if admission else 'reserve_bytes'], 'effective RAM below 6/4 GiB')
        require(clock()-started < LIMITS['total_wall_s'], 'total wall limit')
    with termination_guard() as control:
        try:
            adopter()
            for stage, stage_wall_s in STAGES.items():
                if stage == 'holdout' and not report['original_36_pass']:
                    report['stages'][stage] = {'status': 'SKIPPED_NO_QUALIFIED_CANDIDATE', 'actual_counts': ZERO}
                    continue
                current = stage
                sample(True)
                require(source_loader() == sources and loader()['identity'] == bundle['identity'], 'source changed before stage')
                for old in report['stages'].values():
                    if old.get('status') == 'COMPLETE':
                        require(all(sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'completed artifact changed')
                argv = command(stage, release_path, output/stage, root)
                record = {'status': 'STARTED', 'nonce': secrets.token_hex(24), 'command': argv, 'actual_counts': None}
                report['stages'][stage], report['active_stage'] = record, stage
                atomic_json(output/'report.json', report)
                stage_started = clock()
                try:
                    with (output/(stage+'.stdout.log')).open('xb') as stream:
                        control['defer'] = True
                        try:
                            process = factory(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, env=dict(os.environ, D1_STAGE21_NONCE=record['nonce'],
                                    PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                                    PYTHONPATH=os.pathsep.join(map(str, (root/'.python-deps', root/'training', root)))))
                        finally:
                            control['defer'] = False
                        require(control['received'] is None, 'interrupted during Popen')
                        while process.poll() is None:
                            sample()
                            require(clock()-stage_started < stage_wall_s, 'stage process wall limit')
                            sleeper(LIMITS['poll_s'])
                        sample()
                        require(clock()-stage_started < stage_wall_s, 'stage exit observed late')
                        require(process.returncode == 0, 'stage child failed')
                finally:
                    control['defer'] = True
                    receipt = cleanup(process)
                    record['cleanup_complete'] = receipt.get('remaining') == {}
                    report['cleanup_complete'] &= record['cleanup_complete']
                    process = None
                    require(record['cleanup_complete'], 'owned cleanup incomplete')
                    control['defer'] = False
                require(control['received'] is None, 'termination requested during stage cleanup')
                child = read_json(output/stage/'report.json')
                validator(output/stage, stage, child, release, release_path)
                record.update(status='COMPLETE', actual_counts=child['actual_counts'], artifacts_sha256=artifacts(output/stage))
                report['known_actual_counts'] = known_counts(report['stages'])
                require(valid_counts(report['known_actual_counts'], BUDGET), 'total cap exceeded')
                if stage == 'formal':
                    report['original_36_pass'] = child['all_36_reference_pass']
                if stage == 'holdout':
                    report['holdout_36_pass'] = child['all_36_reference_pass']
                atomic_json(output/'report.json', report)
                current = None
            require(loader()['identity'] == bundle['identity'] and source_loader() == sources, 'final inputs changed')
            for old in report['stages'].values():
                if old.get('status') == 'COMPLETE':
                    require(all(sha(path) == digest for path, digest in old['artifacts_sha256'].items()), 'final completed artifact changed')
            sample()
            report.update(status='COMPLETE', actual_counts=report['known_actual_counts'],
                          full_delay_qualification=report['original_36_pass'] and report['holdout_36_pass'])
        except BaseException as exc:  # noqa: BLE001 - every interruption needs a terminal supervision receipt
            report.update(status='INCOMPLETE', error=f'{type(exc).__name__}: {exc}')
            if current in report['stages']:
                report['stages'][current]['status'] = 'INCOMPLETE'
                try:
                    child = read_json(output/current/'report.json')
                    check_stage_identity(child, current, release, release_path)
                    protocol_path = output/current/'protocol.json'
                    protocol = read_json(protocol_path)
                    check_stage_identity(protocol, current, release, release_path)
                    require(child.get('protocol_sha256') == sha(protocol_path)
                            and protocol.get('budget_max') == (CAP if current in ('formal','holdout') else ZERO)
                            and protocol.get('max_wall_s') == STAGES[current], 'partial protocol changed')
                    known = child['known_actual_counts']
                    require(valid_counts(known, ZERO if current in ('export','cpp') else CAP, partial=True), 'partial counters unknown')
                    record = report['stages'][current]
                    record['known_actual_counts'] = known
                    if (valid_counts(child.get('actual_counts'), ZERO if current in ('export','cpp') else CAP, partial=True)
                            and child['actual_counts'] == known):
                        record['actual_counts'] = child['actual_counts']
                    else:
                        record['actual_counts'] = None
                    report['known_actual_counts'] = known_counts(report['stages'])
                    require(valid_counts(report['known_actual_counts'], BUDGET, partial=True), 'partial total exceeded')
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        finally:
            control['defer'] = True
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            try:
                cleaned = cleanup(process)
                report['cleanup_complete'] &= cleaned.get('remaining') == {}
            except BaseException as exc:  # noqa: BLE001 - cleanup uncertainty must revoke completion
                report.update(cleanup_complete=False, error=(report['error'] or '')+f'; cleanup {exc}')
            report['elapsed_s'] = clock()-started
            if not report['cleanup_complete'] or control['received'] or report['elapsed_s'] >= LIMITS['total_wall_s']:
                report.update(status='INCOMPLETE', full_delay_qualification=False)
            report['known_actual_counts'] = known_counts(report['stages'])
            report['actual_counts'] = (report['known_actual_counts'] if all(
                row.get('actual_counts') is not None for row in report['stages'].values()) else None)
            atomic_json(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--stage', choices=tuple(STAGES), help=argparse.SUPPRESS)
    parser.add_argument('--release', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(not (args.execute and args.stage), 'choose one mode')
    if args.stage:
        require(args.release is not None and args.output is not None, 'supervised paths required')
        result = run_stage(args.release, args.stage, args.output)
    elif args.execute:
        require(args.output is None and args.release is None, 'fixed independent output required')
        result = execute()
    else:
        print(json.dumps({'status': 'PLAN_ONLY', 'output': str(ROOT/OUTPUT), 'budget': BUDGET, 'limits': LIMITS,
                          'stage_limits_s': STAGES, 'new_training_transitions': 0, 'physics_steps_executed': 0,
                          'onnx_sessions_created': 0, 'random_partial_used': False}))
        return 0
    print(json.dumps({k: result.get(k) for k in ('status','actual_counts','full_delay_qualification','error')}))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
