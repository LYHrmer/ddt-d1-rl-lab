"""Resource revision02 around the unchanged26 scientific pipeline.

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
GUARD_SOURCES = ('tools27/resource_guard27.py', 'tests/test_resource_guard27.py')
PACKAGE = '_d1_owned_cgroup_revision27_02'
REVISION = 'owned_cgroup_v2'
PLAN = 'evidence/yaw_reward27/plan_02.json'
OUTPUT = 'evidence/yaw_reward27/execution_02'
ADMISSIONS = 'evidence/yaw_reward27/admissions_02'
SUPERSESSION = 'evidence/yaw_reward27/superseded_01_by_02.json'
OWNED_SCOPE = (f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/'
               'app.slice/d1-yaw-reward27-owned02-20261018-01.service')
OWNED_POLICY = {'scope': OWNED_SCOPE, 'memory_high_bytes': 2 << 30,
                'memory_max_bytes': 3 << 30, 'memory_swap_max_bytes': 0,
                'memory_oom_group': 1, 'oom_policy': 'kill'}
LEGACY_PLAN = 'evidence/yaw_reward27/plan_01.json'
LEGACY_PLAN_SHA = '3152b2dcd8fa47555a6096998a95dda3820fa2f9fe2cd8a37fbfcb706ff98419'
LEGACY_COMMIT = '14934a40ee74baa3391aecf9faf3663f61ba0e15'
LEGACY_ADMISSIONS = {
    'admission_2edc241f42eaaa1474966ab8': (
        'e33e6f61546cf342b97cf36e366880841e37eaa0b0865095ae6ac5e8139b3c2c',
        'd3b5a9bda19da5d566b5ebe1014dbd6da0359d742ac077b63c26ae7da80ab50e'),
    'admission_c685eeaf693d77fda61c0589': (
        '341bf87dc1b798ed6c3616675dcb88a2fdb9f9a9e235eee6c03317ad1c7f8f72',
        'c69e493cd2da55b9bcc94eb45368ed978390d2324860d9e8f2d193fdb5976951'),
    'admission_fd563a88cc1715b1070e74a4': (
        '55d3901dd56565b3de87efe666278feac201bbe7d02fea6e587077abf3a940b4',
        '56de6a96db5233a98c7df79f631a3d1e881b8190b4d785356be6cb48e39ff101'),
}
LABELS = {
    'evidence/yaw_reward26/execution_01': OUTPUT,
    'evidence/yaw_reward26/plan_01.json': PLAN,
    'D1_STAGE26_NONCE': 'D1_STAGE27_REV02_NONCE',
    'first_rollout26.json': 'first_rollout27_revision02.json',
    **{'d1-yaw-reward26-'+suffix+'-v1': 'd1-yaw-reward27-'+suffix+'-v2' for suffix in (
        'plan', 'execution', 'stage', 'training-result', 'training-protocol', 'first-rollout')},
}
ADMISSION = {'minimum_effective_bytes': 6 << 30, 'stable_s': 10, 'poll_s': 5,
             'minimum_samples': 3,
             'maximum_wait_s': 300, 'reserve_bytes': 4 << 30, 'total_wall_s': 3900,
             'parent_wall_s': 3600, 'runtime_poll_s': 1}
_BINDING = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def legacy_proof02(root=ROOT):
    """The old three observations spent no training budget; never edit them."""
    root = Path(root)
    old_execution = root/'evidence/yaw_reward27/execution_01'
    require(not old_execution.exists() and not old_execution.is_symlink(), 'legacy27 execution exists; no replacement budget')
    old_plan = root/LEGACY_PLAN
    require(old_plan.is_file() and not old_plan.is_symlink() and sha(old_plan) == LEGACY_PLAN_SHA,
            'legacy27 plan identity differs')
    plan = json.loads(old_plan.read_text())
    require(plan['source_identity']['git_commit'] == LEGACY_COMMIT
            and plan['budget'] == {'policy_transitions': 326944, 'native_steps': 3269440, 'pd_updates': 1634720},
            'legacy27 scientific budget differs')
    directory = root/'evidence/yaw_reward27/admissions'
    require(directory.is_dir() and not directory.is_symlink()
            and {p.name for p in directory.iterdir()} == set(LEGACY_ADMISSIONS), 'legacy27 admission set differs')
    evidence = {}
    for name, (report_sha, samples_sha) in LEGACY_ADMISSIONS.items():
        folder = directory/name
        report, samples = folder/'report.json', folder/'admission_samples.jsonl'
        require(folder.is_dir() and not folder.is_symlink() and not report.is_symlink() and not samples.is_symlink()
                and sha(report) == report_sha and sha(samples) == samples_sha,
                'legacy27 admission bytes differ: '+name)
        value = json.loads(report.read_text())
        known = value.get('known_actual_counts')
        require(value.get('status') == 'NOT_READY' and value.get('execution_started') is False
                and value.get('spawn_attempted') is False and value.get('cleanup_complete') is True
                and value.get('actual_counts') is None and isinstance(known, dict)
                and set(known) == set(plan['budget']) and all(type(v) is int and v == 0 for v in known.values())
                and value.get('plan_sha256') == LEGACY_PLAN_SHA
                and value.get('source_identity') == plan['source_identity']
                and not (folder/'launch_ticket.json').exists(), 'legacy27 was spawned or not cleanly unstarted')
        evidence[name] = {'report_sha256': report_sha, 'samples_sha256': samples_sha}
    return {'legacy_plan_path': LEGACY_PLAN, 'legacy_plan_sha256': LEGACY_PLAN_SHA,
            'legacy_source_commit': LEGACY_COMMIT, 'admissions': evidence,
            'legacy_execution_absent': True, 'additional_scientific_budget': 0,
            'scientific_cap': dict(plan['budget']), 'partial26_used': False}


def read_memory02():
    from tools27.resource_guard27 import read_memory
    return read_memory(OWNED_SCOPE)


def supersession02(common):
    value = {'format': 'd1-yaw-reward27-supersession-v1', 'status': 'SUPERSEDED_WITHOUT_EXECUTION',
             'resource_revision': REVISION, 'replacement_plan': PLAN,
             'replacement_output': OUTPUT, 'budget_origin': legacy_proof02(ROOT)}
    path = ROOT/SUPERSESSION
    require(path.is_file() and not path.is_symlink() and common.read_json(path) == value,
            'explicit27 revision02 supersession record required')
    return {'path': SUPERSESSION, 'sha256': sha(path), 'budget_origin': value['budget_origin']}


def prepare02(common, path):
    path = Path(path).absolute()
    require(path == ROOT/PLAN and not path.exists() and not path.is_symlink(), 'revision02 new fixed plan_02 required')
    value = {'format': 'd1-yaw-reward27-supersession-v1', 'status': 'SUPERSEDED_WITHOUT_EXECUTION',
             'resource_revision': REVISION, 'replacement_plan': PLAN,
             'replacement_output': OUTPUT, 'budget_origin': legacy_proof02(ROOT)}
    marker = ROOT/SUPERSESSION
    if marker.exists():
        require(not marker.is_symlink() and common.read_json(marker) == value, 'supersession record changed')
    else:
        with marker.open('x') as stream:
            stream.write(json.dumps(value, allow_nan=False, indent=2)+'\n')
    return common.prepare(path)


def after_import_seam02(tree, *, reverse=False):
    old_message, new_message = 'NOT_READY: RAM after imports', 'RAM_RESERVE_BELOW_4_GIB after imports'
    message = new_message if reverse else old_message
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                  and n.func.id == 'require' and len(n.args) == 2
                  and isinstance(n.args[1], ast.Constant) and n.args[1].value == message]
    require(len(candidates) == 1, 'revision02 exact after-import resource seam missing')
    call = candidates[0]
    key = 'reserve_bytes' if reverse else 'start_min_effective_bytes'
    require(ast.unparse(call.args[0]) == f"memory_ok(memory_reader(), plan['resources']['{key}'])",
            'revision02 after-import expression changed')
    call.args[0].args[1].slice.value = 'start_min_effective_bytes' if reverse else 'reserve_bytes'
    call.args[1].value = old_message if reverse else new_message
    return tree


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
                node.arg = 'D1_STAGE27_REV02_NONCE'
            return self.generic_visit(node)

    tree = Bind().visit(tree)
    if name == 'train_yaw_reward26':
        after_import_seam02(tree)
    return ast.fix_missing_locations(tree)


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
        if name == 'common26':
            # Set before downstream definitions capture read_memory defaults.
            module.read_memory = read_memory02
        loaded[name] = module
    common = loaded['common26']
    common.SOURCES = tuple(dict.fromkeys((*common.SOURCES, *AUTHORED, *GUARD_SOURCES)))
    original_fixed = common.fixed
    derivation = {
        'frozen26_sha256': {'tools26/'+name+'.py': digest for name, digest in FROZEN26.items()},
        'derived_ast_sha256': {name: hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
                               for name, tree in trees.items()},
        'literal_bindings': dict(LABELS), 'internal_import_package': PACKAGE,
        'original_file_identity_retained': True,
        'resource_seams': {'private_common_read_memory': 'tools27.resource_guard27.read_memory(OWNED_SCOPE)',
                           'trainer_after_import_gate': 'start6GiB becomes reserve4GiB only after imports',
                           'cold_start_and_new_stage_gate_unchanged': True},
    }

    def fixed(base):
        return {**original_fixed(base), 'resource_revision': REVISION,
                'resource_revision27': dict(ADMISSION), 'owned_cgroup': dict(OWNED_POLICY), 'derivation27': derivation,
                'supersession27': supersession02(common),
                'scope': 'revision02 replaces wholly unstarted27 plan01; same scientific cap, complete25 random, no26 partial'}

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
    require(len(stable) >= ADMISSION['minimum_samples'] and all(type(row.get('effective_available_bytes')) is int
            and row['effective_available_bytes'] >= ADMISSION['minimum_effective_bytes'] for row in stable)
            and all(0 <= b['elapsed_s']-a['elapsed_s'] <= ADMISSION['poll_s']+.25 for a, b in zip(stable, stable[1:])),
            'continuous27 resource proof differs')


def validate_parent_ticket(plan, plan_path, sidecar, api):
    sidecar = Path(sidecar).absolute()
    admission_root = ROOT/ADMISSIONS
    require(sidecar.parent == admission_root and not sidecar.is_symlink(), 'fixed27 admission sidecar required')
    ticket, receipt = (api.read_json(sidecar/name) for name in ('launch_ticket.json', 'report.json'))
    expected = {'format': 'd1-yaw-reward27-parent-ticket-v2', 'status': 'READY',
        'plan_sha256': api.sha(plan_path), 'source_identity': plan['source_identity'],
        'parent_command': parent_command(plan_path, sidecar), 'admission': ADMISSION}
    require(all(ticket.get(key) == value for key, value in expected.items())
            and type(ticket.get('outer_pid')) is int and ticket['outer_pid'] == os.getppid()
            and ticket.get('nonce') and ticket['nonce'] == os.environ.get('D1_PARENT27_REV02_NONCE')
            and api.sha(sidecar/'launch_ticket.json') == os.environ.get('D1_PARENT27_REV02_TICKET_SHA256')
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
    require(plan_path == ROOT/PLAN, 'revision02 rejects old plan_01 path')
    started = clock()
    plan = loader(plan_path)
    require(not (ROOT/plan['output_root']).exists(), '27 training attempt already claimed; no retry')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, '27 requires external Nice10')
    admission_root = ROOT/ADMISSIONS
    sidecar = Path(sidecar or admission_root/('admission_'+secrets.token_hex(12))).absolute()
    require(sidecar.parent == admission_root and not sidecar.exists(), 'exclusive27 admission directory required')
    cursor = sidecar
    while cursor != ROOT:
        require(not cursor.is_symlink(), 'symlink27 sidecar forbidden')
        cursor = cursor.parent
    sidecar.mkdir(parents=True, exist_ok=False)
    value = {'format': 'd1-yaw-reward27-admission-v2', 'resource_revision': REVISION,
        'status': 'WAITING', 'outer_pid': os.getpid(),
        'plan_sha256': api.sha(plan_path), 'source_identity': plan['source_identity'], 'admission': dict(ADMISSION),
        'nonce': secrets.token_hex(24), 'execution_started': False, 'spawn_attempted': False, 'cleanup_complete': True,
        'actual_counts': None, 'known_actual_counts': dict(runner.ZERO), 'error': None,
        'minimum_effective_available_bytes': None, 'stop_sample': None}
    process, stable_started, admitted_at, spawn_attempted = None, None, None, False

    def sample(phase, filename):
        try:
            data = memory_reader()
        except Exception as exc:
            snapshot = getattr(exc, 'snapshot', None)
            if isinstance(snapshot, dict):
                row = dict(snapshot, elapsed_s=clock()-started, phase=phase)
                with (sidecar/filename).open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                value['stop_sample'] = row
            raise
        row = dict(data, elapsed_s=clock()-started, phase=phase)
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
            ticket = {'format': 'd1-yaw-reward27-parent-ticket-v2', 'status': 'READY',
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
                        start_new_session=True, env=dict(os.environ, D1_PARENT27_REV02_NONCE=value['nonce'],
                            D1_PARENT27_REV02_TICKET_SHA256=value['ticket_sha256'], PYTHONDONTWRITEBYTECODE='1',
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
    parser.add_argument('--plan', type=Path, default=ROOT/PLAN)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--parent', action='store_true')
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--stage', choices=('train', 'export', 'cpp', 'formal', 'holdout'))
    parser.add_argument('--arm', choices=('control', 'yaw3'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    require(args.plan.absolute() == ROOT/PLAN, 'revision02 rejects legacy plan_01; fixed plan_02 only')
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
        value = prepare02(bound.common, args.plan)
        value = {'status': 'PREPARED', 'plan': str(args.plan), 'budget': value['budget']}
    else:
        value = {'status': 'PLAN_ONLY', 'resource_revision': REVISION, 'budget': dict(bound.common.BUDGET),
            'admission': dict(ADMISSION), 'owned_cgroup': dict(OWNED_POLICY),
            'plan': str(args.plan), 'output': str(ROOT/bound.common.OUTPUT),
            'training_started': False, 'physics_steps_executed': 0, 'cpp_closed_loop_qualified': False}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return 0 if value.get('status') in ('PLAN_ONLY', 'PREPARED', 'COMPLETE', 'NOT_READY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
