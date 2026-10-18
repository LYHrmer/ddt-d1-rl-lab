"""Independent27 resource admission around the unchanged26 scientific pipeline.

Cold imports are stdlib-only. Frozen sources are compiled into private modules;
their __file__ honestly remains the original26 file. Only the enumerated receipt,
path, nonce and import bindings change. No old module globals are mutated.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN26 = {
    'common26': '1de130e2178b72c10ee7d818e67d41e5867479ec105e93de483530889e3673fb',
    'runtime26': '974434ea439f7a0c07f96888bf443d38613ec57a89018b06d050c5a898b7b71c',
    'train_yaw_reward26': '58b4ad2fb8144de069d597e152eec47bf0175e312acf821d83feafc19d2623df',
    'run_yaw_reward26': 'c7b72de25cb819d61ac9d3d006bc6ad0253823cbcda5c8eb58bb626e71f33215',
}
AUTHORED = ('tools27/run_yaw_reward27.py', 'tests/test_yaw_reward27.py', 'docs/yaw_reward27.md')
PACKAGE = '_d1_resource_revision27'
LABELS = {
    'evidence/yaw_reward26/execution_01': 'evidence/yaw_reward27/execution_01',
    'evidence/yaw_reward26/plan_01.json': 'evidence/yaw_reward27/plan_01.json',
    'D1_STAGE26_NONCE': 'D1_STAGE27_NONCE',
    'first_rollout26.json': 'first_rollout27.json',
    **{'d1-yaw-reward26-'+suffix: 'd1-yaw-reward27-'+suffix for suffix in (
        'plan-v1', 'execution-v1', 'stage-v1', 'training-result-v1',
        'training-protocol-v1', 'first-rollout-v1')},
}
ADMISSION = {'minimum_effective_bytes': 8 << 30, 'stable_s': 60, 'poll_s': 5,
             'maximum_wait_s': 300, 'reserve_bytes': 4 << 30, 'total_wall_s': 3900,
             'parent_wall_s': 3600, 'runtime_poll_s': 1}
_BINDING = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_tree(name, root=ROOT):
    path = Path(root)/'tools26'/(name+'.py')
    require(path.is_file() and not path.is_symlink() and sha(path) == FROZEN26[name], 'frozen26 source changed: '+name)
    tree = ast.parse(path.read_text(), filename=str(path))

    class Bind(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is str and node.value in LABELS:
                return ast.copy_location(ast.Constant(LABELS[node.value]), node)
            return node

        def visit_ImportFrom(self, node):
            if node.module in ('tools26', 'tools26.runtime26'):
                node.module = PACKAGE+node.module[len('tools26'):]
            return node

        def visit_keyword(self, node):
            if node.arg == 'D1_STAGE26_NONCE':
                node.arg = 'D1_STAGE27_NONCE'
            return self.generic_visit(node)

    return ast.fix_missing_locations(Bind().visit(tree))


def bindings():
    """Independent globals and defaults, with an explicit finite AST derivation."""
    global _BINDING
    # Recheck all frozen bytes even after a successful cached binding.
    trees = {name: derived_tree(name) for name in FROZEN26}
    if _BINDING is not None:
        return _BINDING
    package = types.ModuleType(PACKAGE)
    package.__path__ = []
    sys.modules[PACKAGE] = package
    loaded = {}
    for name, tree in trees.items():
        module = types.ModuleType(PACKAGE+'.'+name)
        module.__file__ = str(ROOT/'tools26'/(name+'.py'))
        module.__package__ = PACKAGE
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        exec(compile(tree, module.__file__, 'exec'), module.__dict__)  # noqa: S102 -- SHA-pinned source and finite AST binding
        loaded[name] = module
    common = loaded['common26']
    common.SOURCES = tuple(dict.fromkeys((*common.SOURCES, *AUTHORED)))
    original_fixed = common.fixed
    derivation = {
        'frozen26_sha256': {'tools26/'+name+'.py': digest for name, digest in FROZEN26.items()},
        'derived_ast_sha256': {name: hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                               for name, tree in trees.items()},
        'literal_bindings': dict(LABELS), 'internal_import_package': PACKAGE,
        'original_file_identity_retained': True,
    }

    def fixed(base):
        return {**original_fixed(base), 'resource_revision27': dict(ADMISSION), 'derivation27': derivation,
                'scope': 'new independent resource revision of26 scientific pair, from complete25 random; no26 partial or budget reuse'}

    def command(plan_path, stage, output):
        arm, kind = common.split_stage(stage)
        return [sys.executable, '-B', str(ROOT/AUTHORED[0]), '--plan', str(plan_path),
                '--arm', arm, '--stage', kind, '--output', str(output)]

    common.fixed, common.command = fixed, command
    _BINDING = types.SimpleNamespace(common=common, runtime=loaded['runtime26'],
        train=loaded['train_yaw_reward26'], runner=loaded['run_yaw_reward26'], derivation=derivation)
    return _BINDING


def parent_command(plan_path, sidecar):
    return [sys.executable, '-B', str(ROOT/AUTHORED[0]), '--plan', str(plan_path),
            '--parent', '--admission', str(sidecar)]


def validate_admission_samples(rows, start):
    require(bool(rows) and rows[-1].get('phase') == 'before_spawn', 'final27 admission sample missing')
    require(type(start) in (int, float) and rows[-1]['elapsed_s']-start >= ADMISSION['stable_s']
            and 0 <= rows[-1]['elapsed_s'] <= ADMISSION['maximum_wait_s'], 'stable27 interval missing')
    stable = [row for row in rows if row['elapsed_s'] >= start]
    require(len(stable) >= 13 and all(type(row.get('effective_available_bytes')) is int
            and row['effective_available_bytes'] >= ADMISSION['minimum_effective_bytes'] for row in stable)
            and all(0 <= b['elapsed_s']-a['elapsed_s'] <= ADMISSION['poll_s']+.25 for a, b in zip(stable, stable[1:])),
            'continuous27 resource proof differs')


def validate_parent_ticket(plan, plan_path, sidecar, api):
    sidecar = Path(sidecar).absolute()
    admission_root = ROOT/'evidence/yaw_reward27/admissions'
    require(sidecar.parent == admission_root and not sidecar.is_symlink(), 'fixed27 admission sidecar required')
    ticket, receipt = (api.read_json(sidecar/name) for name in ('launch_ticket.json', 'report.json'))
    expected = {'format': 'd1-yaw-reward27-parent-ticket-v1', 'status': 'READY',
        'plan_sha256': api.sha(plan_path), 'source_identity': plan['source_identity'],
        'parent_command': parent_command(plan_path, sidecar), 'admission': ADMISSION}
    require(all(ticket.get(key) == value for key, value in expected.items())
            and type(ticket.get('outer_pid')) is int and ticket['outer_pid'] == os.getppid()
            and ticket.get('nonce') and ticket['nonce'] == os.environ.get('D1_PARENT27_NONCE')
            and api.sha(sidecar/'launch_ticket.json') == os.environ.get('D1_PARENT27_TICKET_SHA256')
            and receipt.get('status') == 'RUNNING_PARENT' and receipt.get('ticket_sha256') == api.sha(sidecar/'launch_ticket.json')
            and receipt.get('nonce') == ticket['nonce']
            and ticket.get('samples_sha256') == api.sha(sidecar/'admission_samples.jsonl'), 'active27 admission ticket required')
    rows = [json.loads(line) for line in (sidecar/'admission_samples.jsonl').read_text().splitlines()]
    validate_admission_samples(rows, ticket.get('stable_started_elapsed_s'))
    return ticket


def execute(plan_path=None, *, bundle=None, loader=None, memory_reader=None, factory=subprocess.Popen,
            cleanup=None, adopter=None, clock=time.monotonic, sleeper=time.sleep, sidecar=None):
    """Wait without a training claim, then spawn the original parent once."""
    bound = bindings() if bundle is None else bundle
    api, runner = bound.common, bound.runner
    loader, memory_reader = loader or api.load_plan, memory_reader or api.read_memory
    cleanup, adopter = cleanup or runner.cleanup_owned_tree, adopter or runner.subreaper
    plan_path = Path(plan_path or ROOT/api.PLAN).absolute()
    started = clock()
    plan = loader(plan_path)
    require(not (ROOT/plan['output_root']).exists(), '27 training attempt already claimed; no retry')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '27 requires external Nice10')
    admission_root = ROOT/'evidence/yaw_reward27/admissions'
    sidecar = Path(sidecar or admission_root/('admission_'+secrets.token_hex(12))).absolute()
    require(sidecar.parent == admission_root and not sidecar.exists(), 'exclusive27 admission directory required')
    cursor = sidecar
    while cursor != ROOT:
        require(not cursor.is_symlink(), 'symlink27 sidecar forbidden')
        cursor = cursor.parent
    sidecar.mkdir(parents=True, exist_ok=False)
    value = {'format': 'd1-yaw-reward27-admission-v1', 'status': 'WAITING', 'outer_pid': os.getpid(),
        'plan_sha256': api.sha(plan_path), 'source_identity': plan['source_identity'], 'admission': dict(ADMISSION),
        'nonce': secrets.token_hex(24), 'execution_started': False, 'spawn_attempted': False, 'cleanup_complete': True,
        'actual_counts': None, 'known_actual_counts': dict(runner.ZERO), 'error': None,
        'minimum_effective_available_bytes': None, 'stop_sample': None}
    process, stable_started, admitted_at, spawn_attempted = None, None, None, False

    def sample(phase, filename):
        row = dict(memory_reader(), elapsed_s=clock()-started, phase=phase)
        with (sidecar/filename).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        ram = row.get('effective_available_bytes')
        if type(ram) is int and ram >= 0:
            old = value['minimum_effective_available_bytes']
            value['minimum_effective_available_bytes'] = ram if old is None else min(old, ram)
        if type(ram) is not int or ram < ADMISSION['reserve_bytes']:
            value['stop_sample'] = row
            raise ValueError('effective27 RAM unknown or below4GiB')
        return row

    with runner.termination_guard() as control:
        try:
            api.atomic_json(sidecar/'report.json', value)
            while True:
                row = sample('admission', 'admission_samples.jsonl')
                elapsed = row['elapsed_s']
                require(elapsed <= ADMISSION['maximum_wait_s'], '27 admission deadline')
                if row['effective_available_bytes'] >= ADMISSION['minimum_effective_bytes']:
                    stable_started = elapsed if stable_started is None else stable_started
                    if elapsed-stable_started >= ADMISSION['stable_s']:
                        break
                else:
                    stable_started = None
                value.update(stable_started_elapsed_s=stable_started, elapsed_s=elapsed)
                api.atomic_json(sidecar/'report.json', value)
                require(elapsed < ADMISSION['maximum_wait_s'], '27 admission exhausted')
                sleeper(min(ADMISSION['poll_s'], ADMISSION['maximum_wait_s']-elapsed))
            require(loader(plan_path) == plan and api.sha(plan_path) == value['plan_sha256'], '27 inputs changed during admission')
            row = sample('before_spawn', 'admission_samples.jsonl')
            require(row['elapsed_s'] <= ADMISSION['maximum_wait_s']
                    and row['effective_available_bytes'] >= ADMISSION['minimum_effective_bytes'], '27 readiness lost before spawn')
            rows = [json.loads(line) for line in (sidecar/'admission_samples.jsonl').read_text().splitlines()]
            validate_admission_samples(rows, stable_started)
            require(control['received'] is None, '27 admission interrupted')
            command = parent_command(plan_path, sidecar)
            ticket = {'format': 'd1-yaw-reward27-parent-ticket-v1', 'status': 'READY',
                'outer_pid': os.getpid(), 'nonce': value['nonce'], 'plan_sha256': value['plan_sha256'],
                'source_identity': plan['source_identity'], 'admission': dict(ADMISSION),
                'stable_started_elapsed_s': stable_started, 'parent_command': command,
                'samples_sha256': api.sha(sidecar/'admission_samples.jsonl')}
            with (sidecar/'launch_ticket.json').open('x') as stream:
                stream.write(json.dumps(ticket, allow_nan=False, indent=2)+'\n')
            value.update(status='RUNNING_PARENT', command=command, ticket_sha256=api.sha(sidecar/'launch_ticket.json'))
            api.atomic_json(sidecar/'report.json', value)
            adopter()
            with (sidecar/'parent.stdout.log').open('xb') as stream:
                control['defer'] = True
                try:
                    spawn_attempted = True
                    admitted_at = clock()
                    value.update(spawn_attempted=True, execution_started=None)
                    process = factory(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, env=dict(os.environ, D1_PARENT27_NONCE=value['nonce'],
                            D1_PARENT27_TICKET_SHA256=value['ticket_sha256'], PYTHONDONTWRITEBYTECODE='1',
                            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                    value['execution_started'] = True
                finally:
                    control['defer'] = False
                require(control['received'] is None, '27 parent spawn interrupted')
                while process.poll() is None:
                    sample('parent_runtime', 'parent_resources.jsonl')
                    require(clock()-started < ADMISSION['total_wall_s']
                            and clock()-admitted_at < ADMISSION['parent_wall_s'], '27 parent/global deadline')
                    sleeper(ADMISSION['runtime_poll_s'])
                sample('parent_exit', 'parent_resources.jsonl')
                require(process.returncode == 0 and clock()-admitted_at < ADMISSION['parent_wall_s'], '27 parent failed/late')
            child = api.read_json(ROOT/plan['output_root']/'report.json')
            require(child.get('format') == api.EXECUTION_FORMAT and child.get('status') == 'COMPLETE'
                    and child.get('plan_sha256') == value['plan_sha256'] and child.get('source_identity') == plan['source_identity']
                    and child.get('cleanup_complete') is True and child.get('error') is None
                    and child.get('actual_counts') is not None, '27 parent receipt rejected')
            value.update(status='COMPLETE', parent_report_sha256=api.sha(ROOT/plan['output_root']/'report.json'))
        except BaseException as exc:
            value.update(status='INCOMPLETE' if spawn_attempted else 'NOT_READY',
                         error=type(exc).__name__+': '+str(exc))
        finally:
            control['defer'] = True
            for number in (signal.SIGINT, signal.SIGTERM):
                signal.signal(number, signal.SIG_IGN)
            try:
                receipt = cleanup(process)
                value['cleanup_complete'] = receipt.get('remaining') == {}
                value['cleanup'] = receipt
            except BaseException as exc:
                value.update(cleanup_complete=False, error=(value['error'] or '')+'; cleanup '+str(exc))
            value['elapsed_s'] = clock()-started
            if (not value['cleanup_complete'] or control['received'] or value['elapsed_s'] >= ADMISSION['total_wall_s']
                    or admitted_at is not None and clock()-admitted_at >= ADMISSION['parent_wall_s']):
                value.update(status='INCOMPLETE', error=value['error'] or '27 termination/cleanup/deadline')
            try:
                require(loader(plan_path) == plan and api.sha(plan_path) == value['plan_sha256'], '27 final sources changed')
                parent = api.read_json(ROOT/plan['output_root']/'report.json') if (ROOT/plan['output_root']/'report.json').is_file() else None
                if parent is not None and parent.get('format') == api.EXECUTION_FORMAT and parent.get('plan_sha256') == value['plan_sha256'] and parent.get('source_identity') == plan['source_identity']:
                    cap = plan['budget']
                    known = runner.projected_counts(parent.get('known_actual_counts'), cap, partial=True)
                    actual = parent.get('actual_counts')
                    if actual is not None:
                        require(runner.same_json(actual, parent['known_actual_counts']), '27 parent actual/known differ')
                        actual = runner.projected_counts(actual, cap, partial=True)
                    value.update(actual_counts=actual, known_actual_counts=known,
                                 parent_report_sha256=api.sha(ROOT/plan['output_root']/'report.json'))
            except BaseException as exc:
                value.update(status='INCOMPLETE', error=(value['error'] or '')+'; final receipt '+str(exc))
            if value['status'] == 'COMPLETE' and value['actual_counts'] is None:
                value.update(status='INCOMPLETE', error=value['error'] or '27 complete parent counts unknown')
            value['elapsed_s'] = clock()-started
            if (value['elapsed_s'] >= ADMISSION['total_wall_s']
                    or admitted_at is not None and clock()-admitted_at >= ADMISSION['parent_wall_s']):
                value.update(status='INCOMPLETE', error=value['error'] or '27 final readback deadline')
            api.atomic_json(sidecar/'report.json', value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'evidence/yaw_reward27/plan_01.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--parent', action='store_true')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--stage', choices=('train', 'export', 'cpp', 'formal', 'holdout'))
    parser.add_argument('--arm', choices=('control', 'yaw3'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    bound = bindings()
    if args.stage:
        require(not (args.execute or args.prepare or args.parent) and args.arm and args.output, '27 supervised stage only')
        value = (bound.train.run(args.plan, args.arm, args.output) if args.stage == 'train'
                 else bound.runner.run_stage(args.plan, args.arm, args.stage, args.output))
    elif args.parent:
        require(args.admission is not None, '27 admission ticket required')
        plan = bound.common.load_plan(args.plan)
        validate_parent_ticket(plan, args.plan, args.admission, bound.common)
        value = bound.runner.execute(args.plan)
    elif args.execute:
        value = execute(args.plan, bundle=bound)
    elif args.prepare:
        value = bound.common.prepare(args.plan)
        value = {'status': 'PREPARED', 'plan': str(args.plan), 'budget': value['budget']}
    else:
        value = {'status': 'PLAN_ONLY', 'budget': dict(bound.common.BUDGET), 'admission': dict(ADMISSION),
            'plan': str(args.plan), 'output': str(ROOT/bound.common.OUTPUT),
            'training_started': False, 'physics_steps_executed': 0, 'cpp_closed_loop_qualified': False}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value.get('status') in ('PLAN_ONLY', 'PREPARED', 'COMPLETE', 'NOT_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
