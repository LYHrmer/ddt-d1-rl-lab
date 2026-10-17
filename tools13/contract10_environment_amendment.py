#!/usr/bin/env python3
"""Review or execute one explicitly authorised environment-only contract 10 exception.

Default is PLAN_ONLY. The explicit authorisation flag is intended only for an owner
invoking this reviewed command or an approval-gated host command. The program cannot
observe the approval UI and does not claim that it can. Original evidence stays intact.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import datetime as dt
import hashlib
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('evidence/career_contract_10')
ORIGINAL_RELEASE = BASE/'scale10_release_01.json'
ORIGINAL_SHA256 = 'fe5061395b6147ae55316d7eb030625576031316ae2f6912888fbbb3afac928f'
AMENDED_RELEASE = BASE/'scale10_release_02.json'
OWNER_RECORD = BASE/'environment_amendment_owner_01.json'
NEW_INPUTS = ('tools13/contract10_environment_amendment.py',
              'tests/test_contract10_environment_amendment.py',
              'docs/career_execution_contract_10_environment_amendment.md')
CAP = {'transitions': 8421376, 'native': 84231760}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def runtime():
    for sub in ('tools12', 'tools10', 'tools11', 'tools08'):
        name = str(ROOT/sub)
        if name not in sys.path:
            sys.path.insert(0, name)
    return SimpleNamespace(release=importlib.import_module('release_contract10'),
                           judge=importlib.import_module('verify_contract10'),
                           runner=importlib.import_module('run_contract10'))


def original_release(path, root):
    require(Path(path).resolve() == root/ORIGINAL_RELEASE and sha(path) == ORIGINAL_SHA256,
            'the original frozen contract 10 release identity changed')
    release = read_json(path)
    require(release.get('status') == 'FROZEN_BEFORE_TRAINING' and release.get('contract_id') == 10
            and release.get('proposed_budget') == dict(CAP, qualification_native=18000, authorised=True)
            and release['resources']['start_min_available_ram_bytes'] == 7*1024**3,
            'original contract or approved cap differs')
    for name, expected in release['frozen_inputs'].items():
        target = (root/name).resolve()
        require(not Path(name).is_absolute() and target.is_relative_to(root)
                and target.is_file() and sha(target) == expected, f'original frozen input changed: {name}')
    return release


def audit_zero_failure(release_path, root=ROOT):
    """Derive zero from the failure boundary; null trainer counters remain unknown fields."""
    root = Path(root).resolve()
    release = original_release(release_path, root)
    output = root/BASE/'scale10_preflight_01'
    result_path, error_path = output/'result.json', output/'error.txt'
    protocol_path = output/'protocol.json'
    report_path = root/BASE/'scale10_preflight_01_supervisor/report.json'
    result, report, protocol = map(read_json, (result_path, report_path, protocol_path))
    require(result.get('status') == 'ERROR' and type(result.get('trained_transitions')) is int
            and result['trained_transitions'] == 0 and result.get('actual_counts') is None
            and result.get('per_worker_counts') is None and result.get('inputs_unchanged') is True
            and result.get('error') == 'PermissionError: [Errno 1] Operation not permitted',
            'original result does not describe the exact pre-worker IPC failure')
    require(report.get('format') == 'd1-contract10-supervisor-report-v1'
            and report.get('kind') == 'preflight' and report.get('status') == 'FAIL'
            and report.get('release_sha256') == sha(release_path)
            and report.get('supervisor_sha256') == release['supervisor_sha256']
            and report.get('plan') == release['plans']['preflight']
            and report.get('trainer_result') == result and report.get('trainer_returncode') == 1
            and report.get('stop_reason') == 'TRAINER_EXIT_NONZERO'
            and report.get('cleanup_complete') is True and report.get('inputs_unchanged') is True
            and type(report.get('rollouts_compared')) is int and report['rollouts_compared'] == 0
            and report.get('equivalence') == 'INCOMPLETE', 'original supervisor evidence disagrees')
    require({p.name for p in output.iterdir()} == {'source_snapshot', 'error.txt', 'protocol.json', 'result.json'}
            and (output/'source_snapshot').is_dir(), 'unexpected worker, model, progress or ledger artifact in failed attempt')
    require(not (root/BASE/'scale10_train_01').exists()
            and not (root/BASE/'scale10_train_01_supervisor').exists(), 'original scale attempt already exists')
    require(protocol.get('kind') == 'preflight' and protocol.get('seed') == 101 and protocol.get('envs') == 8
            and protocol.get('planned_transitions') == 32768
            and protocol.get('hyperparameters', {}).get('ent_coef') == 0., 'failed protocol differs from the frozen E0')
    for name, expected in protocol['inputs_sha256'].items():
        target = Path(name).resolve()
        require(target.is_relative_to(root)
                and release['frozen_inputs'].get(str(target.relative_to(root))) == expected
                and sha(target) == expected, 'failed training protocol source identity differs')
    forkserver = Path(sysconfig.get_path('stdlib'))/'multiprocessing/forkserver.py'
    popen = forkserver.with_name('popen_forkserver.py')
    sb3 = root/'.python-deps/stable_baselines3/common/vec_env/subproc_vec_env.py'
    tree = ast.parse(forkserver.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'ensure_running')
    binds = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'listener' and n.func.attr == 'bind']
    spawns = [n for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == 'spawnv_passfds']
    require(len(binds) == len(spawns) == 1 and binds[0].lineno < spawns[0].lineno,
            'CPython forkserver no longer establishes the listener before server creation')
    trace = error_path.read_text()
    require(f'File "{forkserver}", line {binds[0].lineno}, in ensure_running' in trace
            and 'listener.bind(address)' in trace and 'in connect_to_new_process' in trace
            and f'File "{sb3}", line ' in trace and 'process.start()' in trace
            and trace.rstrip().endswith('PermissionError: [Errno 1] Operation not permitted'),
            'traceback does not identify the pre-worker forkserver bind failure')
    trainer = (root/release['entry']).read_text()
    require('start_method="forkserver"' in trainer and trainer.index('vec=SubprocVecEnv(') < trainer.index('model=PPO(')
            and 'def factory():' in trainer and 'return factory' in trainer
            and 'process.start()' in sb3.read_text()
            and popen.read_text().index('forkserver.connect_to_new_process') < popen.read_text().index('f.write(buf.getbuffer())'),
            'worker construction or delivery order differs from the audited path')
    files = [Path(release_path), result_path, error_path, protocol_path, report_path, sb3]
    return {'format': 'd1-contract10-preworker-zero-proof-v1',
            'basis': 'first forkserver listener bind failed before any environment worker could start',
            'derived_actual_counts': {'policy_transitions': 0, 'native_steps': 0, 'pd_updates': 0},
            'original_counters_were_null': True, 'original_release_sha256': sha(release_path),
            'inputs_sha256': {str(path.relative_to(root)): sha(path) for path in files},
            'external_runtime_sha256': {str(path): sha(path) for path in (forkserver, popen)},
            'forkserver_bind_line': binds[0].lineno, 'forkserver_spawn_line': spawns[0].lineno,
            'physics_steps_executed_by_audit': 0}


def record_owner_approval(proof, root=ROOT):
    """Called only for an explicit owner flag, never by plan/build/read-only commands."""
    root = Path(root).resolve()
    path = root/OWNER_RECORD
    if not path.exists():
        exclusive_json(path, {'format': 'd1-contract10-environment-owner-amendment-v1', 'contract_id': 10,
            'authorised': True, 'authorisation_source': 'explicit_cli_flag_after_host_approval',
            'approval_ui_observable_by_process': False,
            'scope': 'one environment-only replacement after a proved pre-worker IPC failure',
            'original_release_sha256': proof['original_release_sha256'], 'zero_proof_sha256': digest(proof),
            'approved_cap': CAP, 'qualification_native': 18000, 'start_min_available_ram_bytes': 7*1024**3,
            'scientific_changes': [], 'new_attempts_allowed': 1, 'resume_training_allowed': False,
            'command_argv': sys.argv, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    validate_owner(path, proof, root)
    return path


def validate_owner(path, proof, root):
    require(Path(path).resolve() == root/OWNER_RECORD and Path(path).is_file(), 'explicit owner environment exception is required')
    owner = read_json(path)
    require(owner.get('format') == 'd1-contract10-environment-owner-amendment-v1'
            and owner.get('contract_id') == 10 and owner.get('authorised') is True
            and owner.get('authorisation_source') == 'explicit_cli_flag_after_host_approval'
            and owner.get('original_release_sha256') == proof['original_release_sha256']
            and owner.get('zero_proof_sha256') == digest(proof) and owner.get('approved_cap') == CAP
            and owner.get('qualification_native') == 18000 and owner.get('start_min_available_ram_bytes') == 7*1024**3
            and owner.get('scientific_changes') == [] and type(owner.get('new_attempts_allowed')) is int
            and owner['new_attempts_allowed'] == 1 and owner.get('resume_training_allowed') is False,
            'owner amendment does not authorise this exact environment-only exception')


@contextmanager
def adapted_context(root=ROOT):
    """Route paths only, single-threaded; restore all frozen module globals in finally."""
    require(threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
            'environment adaptation requires an isolated single-threaded process')
    root, modules = Path(root).resolve(), runtime()
    release, runner = modules.release, modules.runner
    previous = (release.RELEASE_PATH, release.PLANS, runner.OUTPUTS, runner.Journal, runner.commands)
    old_commands, old_journal = runner.commands, runner.Journal
    try:
        release.RELEASE_PATH = str(AMENDED_RELEASE)
        release.PLANS = copy.deepcopy(previous[1])
        for value in release.PLANS.values():
            value['output'] = value['output'].removesuffix('_01')+'_02'
        runner.OUTPUTS = {name: Path(str(path).removesuffix('_01')+'_02') for name, path in previous[2].items()}

        def journal(directory, release_sha):
            require(Path(directory) == root/BASE/'execution_01', 'unexpected frozen journal path')
            return old_journal(root/BASE/'execution_02', release_sha)

        def commands(release, release_path, root=root):
            result = old_commands(release, release_path, root)
            verdict = result['verdict']
            require(verdict[2] == str(Path(root)/'tools12/verify_contract10.py'), 'frozen judge command changed')
            result['verdict'] = [*verdict[:2], str(Path(root)/NEW_INPUTS[0]), 'judge', *verdict[3:]]
            return result

        runner.Journal, runner.commands = journal, commands
        yield modules
    finally:
        release.RELEASE_PATH, release.PLANS, runner.OUTPUTS, runner.Journal, runner.commands = previous


def build_release(release_path, owner_path, proof, root=ROOT):
    root = Path(root).resolve()
    original = original_release(release_path, root)
    require(proof == audit_zero_failure(release_path, root), 'zero-physics audit changed')
    validate_owner(owner_path, proof, root)
    with adapted_context(root) as modules:
        amended = copy.deepcopy(original)
        amended['plans'] = copy.deepcopy(modules.release.PLANS)
        downstream = list(modules.runner.OUTPUTS.values())
    for plan in amended['plans'].values():
        output = root/plan['output']
        require(not output.exists() and not output.with_name(output.name+'_supervisor').exists(),
                'amended training attempt already exists; no second exception or retry')
    require(not (root/BASE/'execution_02').exists(), 'amended execution attempt already exists')
    require(not any((root/path).exists() for path in downstream),
            'amended downstream output already exists; refusing to spend a new attempt')
    amended['frozen_inputs'] = amended_inputs(original, proof, root)
    amended.update(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   environment_amendment=amendment_info(proof, root))
    return amended


def amended_inputs(original, proof, root):
    frozen = copy.deepcopy(original['frozen_inputs'])
    for name in (*NEW_INPUTS, str(OWNER_RECORD), *proof['inputs_sha256']):
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(), f'missing amended frozen input: {name}')
        frozen[name] = sha(path)
    return frozen


def amendment_info(proof, root):
    return {'format': 'd1-contract10-environment-release-v1',
            'original_release': str(ORIGINAL_RELEASE), 'original_release_sha256': proof['original_release_sha256'],
            'owner_record': str(OWNER_RECORD), 'owner_record_sha256': sha(root/OWNER_RECORD),
            'adapter_sha256': sha(__file__), 'zero_failure_proof': proof,
            'journal': str(BASE/'execution_02'), 'retries_allowed': False,
            'original_static_scope': 'historical check of original frozen inputs only; '
                                     'new amendment paths and inputs are checked by this adapter'}


def load_amended(root=ROOT):
    root = Path(root).resolve()
    proof = audit_zero_failure(root/ORIGINAL_RELEASE, root)
    validate_owner(root/OWNER_RECORD, proof, root)
    release = read_json(root/AMENDED_RELEASE)
    original = original_release(root/ORIGINAL_RELEASE, root)
    changed_fields = {'created_utc', 'plans', 'frozen_inputs', 'environment_amendment'}
    require(set(release) == set(original) | {'environment_amendment'}
            and all(release[key] == value for key, value in original.items() if key not in changed_fields),
            'amendment changed original scientific fields or metadata beyond the permitted environment exception')
    expected_plans = copy.deepcopy(original['plans'])
    for value in expected_plans.values():
        value['output'] = value['output'].removesuffix('_01')+'_02'
    require(release['plans'] == expected_plans, 'amendment changed a plan beyond its unique output path')
    require(release.get('environment_amendment') == amendment_info(proof, root)
            and release['frozen_inputs'] == amended_inputs(original, proof, root),
            'amendment identity or exact frozen input closure changed')
    with adapted_context(root) as modules:
        modules.judge.load_release(root/AMENDED_RELEASE, root)
    return release


def _ipc_echo(connection):
    connection.send({'token': 'forkserver-pipe-ok', 'models_loaded': 0, 'physics_steps_executed': 0})
    connection.close()


def _ipc_worker():
    """Standalone stdlib-only admission child, never the trainer's preload modules."""
    context = mp.get_context('forkserver')
    context.set_forkserver_preload([])
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_ipc_echo, args=(sender,))
    result = {'status': 'FAIL', 'models_loaded': 0, 'physics_steps_executed': 0}
    try:
        child.start(); sender.close()
        require(receiver.poll(5), 'forkserver dummy handshake timed out')
        message = receiver.recv()
        child.join(5)
        require(child.exitcode == 0 and message == {'token': 'forkserver-pipe-ok', 'models_loaded': 0,
                                                   'physics_steps_executed': 0}, 'forkserver dummy handshake failed')
        require(not {'torch', 'mujoco', 'gymnasium', 'stable_baselines3'} & sys.modules.keys(), 'IPC probe imported a model module')
        result['status'] = 'PASS'
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if child.pid is not None:
            if child.is_alive(): child.terminate(); child.join(2)
            if child.is_alive(): child.kill(); child.join(2)
        receiver.close(); sender.close()
    print(json.dumps(result))
    return 0 if result['status'] == 'PASS' else 1


def ipc_probe():
    """Bounded process-group probe: socket, forkserver child and real pipe round trip."""
    for sub in ('tools10', 'tools08'):
        name = str(ROOT/sub)
        if name not in sys.path:
            sys.path.insert(0, name)
    supervisor = importlib.import_module('supervise_scale10')
    process, stdout, stderr, error, signals = None, '', '', None, []
    with supervisor.handle_termination() as termination:
        try:
            termination['defer'] = True
            try:
                process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '_ipc_worker'],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            finally:
                termination['defer'] = False
            if termination['received']:
                raise supervisor.SupervisorTerminated(termination['received'])
            stdout, stderr = process.communicate(timeout=15)
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            if process is not None:
                if error is None:
                    deadline = time.monotonic()+2
                    while probe_group_members(process.pid) and time.monotonic() < deadline:
                        time.sleep(.02)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if not probe_group_members(process.pid):
                        break
                    try:
                        os.killpg(process.pid, sig)
                        signals.append(sig.name)
                    except ProcessLookupError:
                        break
                    deadline = time.monotonic()+1
                    while probe_group_members(process.pid) and time.monotonic() < deadline:
                        time.sleep(.02)
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except BaseException as exc:
                    error = error or f'{type(exc).__name__}: {exc}'
    try:
        result = json.loads(stdout)
    except ValueError:
        result = {'status': 'FAIL', 'models_loaded': 0, 'physics_steps_executed': 0,
                  'error': 'IPC probe returned no complete receipt'}
    result.update(returncode=process.returncode if process else None, stderr=stderr[-2000:], cleanup_signals=signals,
                  cleanup_complete=process is None or not probe_group_members(process.pid),
                  probe_error=error, timed_out=bool(error and error.startswith('TimeoutExpired:')))
    if result['returncode'] != 0 or error or signals or not result['cleanup_complete']:
        result['status'] = 'FAIL'
    return result


def probe_group_members(pgid):
    """Read only /proc; the pgid belongs to the probe's own new session."""
    members = []
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
                members.append(int(stat.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def plan(release_path, root=ROOT):
    root = Path(root).resolve()
    proof = audit_zero_failure(release_path, root)
    authorised = (root/OWNER_RECORD).is_file()
    if authorised:
        validate_owner(root/OWNER_RECORD, proof, root)
    original = original_release(release_path, root)
    with adapted_context(root) as modules:
        original = copy.deepcopy(original); original['plans'] = copy.deepcopy(modules.release.PLANS)
        commands = modules.runner.commands(original, root/AMENDED_RELEASE, root)
    return {'mode': 'PLAN_ONLY', 'authorised': authorised, 'physics_steps_executed': 0,
            'original_zero_failure_proof': proof, 'approved_cap_unchanged': CAP,
            'new_release': str(AMENDED_RELEASE), 'new_journal': str(BASE/'execution_02'),
            'commands': commands, 'required_owner_flag': '--authorise-environment-exception --execute'}


def execute(release_path, root=ROOT, authorise=False, probe=None):
    root = Path(root).resolve()
    proof = audit_zero_failure(release_path, root)
    owner = record_owner_approval(proof, root) if authorise else root/OWNER_RECORD
    validate_owner(owner, proof, root)
    admission = (probe or ipc_probe)()
    require(admission.get('models_loaded') == admission.get('physics_steps_executed') == 0,
            'IPC admission must be a zero-model, zero-physics check')
    record = root/BASE/'environment_readiness_checks'/f'ipc_{time.time_ns()}_{os.getpid()}.json'
    exclusive_json(record, admission)
    if admission.get('status') != 'PASS' or admission.get('cleanup_complete') is not True:
        return 2
    if not (root/AMENDED_RELEASE).exists():
        exclusive_json(root/AMENDED_RELEASE, build_release(release_path, owner, proof, root))
    load_amended(root)
    with adapted_context(root) as modules:
        return modules.runner.execute(root/AMENDED_RELEASE, root)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == '_ipc_worker':
        return _ipc_worker()
    judge_mode = len(sys.argv) > 1 and sys.argv[1] == 'judge'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT/ORIGINAL_RELEASE)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorise-environment-exception', action='store_true')
    if judge_mode:
        parser.add_argument('--supervisor-report', type=Path, required=True)
        parser.add_argument('--replay', type=Path)
        parser.add_argument('--eval-dir', type=Path)
        parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(sys.argv[2:] if judge_mode else None)
    if judge_mode:
        require(args.release.resolve() == ROOT/AMENDED_RELEASE, 'judge needs this amended release')
        load_amended(ROOT)
        with adapted_context(ROOT) as modules:
            result = modules.judge.write_verdict(args.release, args.supervisor_report, args.replay, args.eval_dir, args.output)
        print(json.dumps({'predictions': result['predictions'], 'qualification': result['qualification']}))
        return 0
    require(not args.authorise_environment_exception or args.execute, 'owner exception flag requires --execute')
    if not args.execute:
        print(json.dumps(plan(args.release), indent=2))
        return 0
    return execute(args.release, authorise=args.authorise_environment_exception)


if __name__ == '__main__':
    raise SystemExit(main())
