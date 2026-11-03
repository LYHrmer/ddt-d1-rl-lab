"""One frozen-pair78 retrospective model validation; no optimizer or environment."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / '.python-deps', ROOT / 'training'):
    sys.path.insert(0, str(directory))

from tools53.run_pilot53 import require, sha
from tools74.runtime74 import run_preserving

SOURCE = 'evidence/world_vz75/execution_01'
PLAN75 = 'docs/experiments75/world_vz75_preregistered.json'
PLAN77 = 'docs/experiments77/output_isolation77_preregistered.json'
SEEDS = (64001, 64002, 64003)
ROLES = ('development', 'old_probe', 'old55_local', 'old56', 'old64', 'old69', 'old72', 'fresh')
FIRST = ('old56', 'old64', 'old69', 'old72', 'fresh')


def inputs():
    names = {PLAN75, PLAN77, SOURCE + '/report.json', SOURCE + '/probe_snapshots.pkl',
             'reports/world_vz75_results_20261101.json', 'evidence/output_isolation77/readback_01/report.json',
             'tools78/run78.py', 'tools78/model78.py', 'tools78/compare78.py',
             'tests/test_model78.py', 'tests/test_compare78.py',
             'docs/protected_response78_plan_20261103.md', '.github/workflows/protected-response78.yml'}
    names.update(f'{SOURCE}/grid_{i:02d}_{b:02d}.npz' for i in range(9) for b in range(10))
    for seed in SEEDS:
        for arm in ('control', 'active'):
            names.add(f'{SOURCE}/{arm}_seed{seed}/final_state_dict.pt')
    return names


def verify(plan):
    require(plan['schema'] == 'd1-protected-response78-v1' and plan['seeds'] == list(SEEDS)
            and plan['roles'] == list(ROLES) and plan['first_roles'] == list(FIRST), '78 fixed scope changed')
    require(plan['cpu_seconds'] == 300 and plan['wall_seconds'] == 420
            and plan['new_updates'] == plan['new_physics_steps'] == 0
            and plan['new_composite_models'] == 3 and plan['source_checkpoints'] == 6, '78 budget changed')
    require(inputs() <= plan['source_sha256'].keys(), '78 required input unbound')
    old = json.loads((ROOT / PLAN77).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(k) == v for k, v in old.items()), '77/75 bindings changed')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, '78 input/source changed: ' + name)


def invariants(report, source, output):
    """Arithmetic invariants do not replace strict comparison78 qualification."""
    import numpy as np

    originals = {(m['seed'], m['arm']): m for m in source['models']}
    maximum_body_fd_gap = maximum_h1_ad_gap = 0.
    endpoints = 0
    for model in report['models']:
        base, donor = [originals[(model['seed'], arm)] for arm in ('control', 'active')]
        for role in FIRST:
            for i in range(9):
                new = output / model['directory'] / f'first_step_{role}/point_{i:02d}.npz'
                old_paths = [ROOT / SOURCE / m['directory'] / f'first_step_{role}/point_{i:02d}.npz' for m in (base, donor)]
                with np.load(new, allow_pickle=False) as p, np.load(old_paths[0], allow_pickle=False) as c, np.load(old_paths[1], allow_pickle=False) as a:
                    for ep in range(4):
                        for sign in ('plus', 'minus'):
                            key = f'epsilon_{ep}_model_next_{sign}'
                            require(np.array_equal(p[f'epsilon_{ep}_initial_state'], c[f'epsilon_{ep}_initial_state'])
                                    and np.array_equal(p[f'epsilon_{ep}_actual_action_{sign}'], c[f'epsilon_{ep}_actual_action_{sign}']),
                                    '78 endpoint input differs')
                            keep = [k for k in range(45) if k not in (23, 24, 25)]
                            require(np.array_equal(p[key][:, keep].view(np.uint64), c[key][:, keep].view(np.uint64)),
                                    '78 changed a protected nonvelocity coordinate')
                            require(np.array_equal(p[key][:, 25].view(np.uint64), a[key][:, 25].view(np.uint64)),
                                    '78 world-vz differs from donor')
                            endpoints += 1
                head = model['heads'][role]['points'][i]
                old = base['heads'][role]['points'][i]
                maximum_h1_ad_gap = max(maximum_h1_ad_gap, abs(head['cost_h1']['gradient'] - old['cost_h1']['gradient']))
                for channel in ('vx', 'vy'):
                    for ep in range(4):
                        maximum_body_fd_gap = max(maximum_body_fd_gap,
                            abs(head['body'][channel]['eps'][ep]['model_fd'] - old['body'][channel]['eps'][ep]['model_fd']))
    require(endpoints == 1080, 'incomplete78 endpoint preservation audit')
    return {'endpoints': endpoints, 'nonvelocity42_bit_exact': True, 'world_vz_donor_bit_exact': True,
            'maximum_body_fd_gap_from_control': maximum_body_fd_gap,
            'maximum_H1_autograd_gap_from_control': maximum_h1_ad_gap,
            'arithmetic_invariants_passed': maximum_body_fd_gap <= 1e-8 and maximum_h1_ad_gap <= 1e-10,
            'replaces_original_qualification_gates': False}


def calculate(report, plan, output):
    resource.setrlimit(resource.RLIMIT_CPU, (299, 300))
    require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1, 'Nice10/single CPU required')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'single-thread CPU required')
    verify(plan)
    source = json.loads((ROOT / SOURCE / 'report.json').read_text())
    require(source['status'] == source['verification_after'] == 'COMPLETE' and source['completed_model_fits'] == 6
            and source['model_updates'] == 6000, 'complete six-model75 source required')
    admission = json.loads((ROOT / 'evidence/output_isolation77/readback_01/report.json').read_text())
    require(admission['status'] == admission['verification_after'] == 'COMPLETE'
            and admission['paired_intervals'] == 540 and admission['maximum_body_preservation_gap'] <= 1e-12,
            'complete77 invariant evidence required')
    report.update(schema='d1-protected-response78-result-v1', scope='retrospective only; fresh role means archived75 fresh',
                  updates=0, policy_updates=0, control_parameter_updates=0, physics_steps=0,
                  source_models=6, new_composite_models=3, newly_fitted_models=0, models=[],
                  source_report_sha256=sha(ROOT / SOURCE / 'report.json'), composed_forward_calls=0,
                  consumed53_inputs={}, original_decisions_changed=False, source75_candidate_accepted=False,
                  comparison={'status': 'NOT_RUN'}, invariant_audit={'status': 'NOT_RUN'})
    versions = {name: importlib.metadata.version(name) for name in source['versions']}
    require(versions == source['versions'], 'original numerical runtime required')
    report['versions'] = versions
    import torch

    from tools52 import adapter52, dynamics52
    from tools56.run56 import check_points, evaluate_points
    from tools59.run59 import evaluate_first_steps, load_grid, load_own_points
    from tools69 import outputs69
    from tools75 import fit75, outputs75
    from tools75.run75 import historical75
    from tools78.compare78 import compare
    from tools78.model78 import FrozenPair78

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(torch.__version__ == source['torch_runtime_version'], 'Torch runtime differs')
    report['torch_runtime_version'] = torch.__version__
    original = json.loads((ROOT / 'docs/experiments53/model_gradient53_preregistered.json').read_text())
    summary53 = json.loads((ROOT / 'reports/model_gradient53_results_20261101.json').read_text())
    actor = adapter52.load_actor(ROOT / original['checkpoint']).double().eval().requires_grad_(False)

    def count(key):
        def hook(_module, _inputs, _output):
            report[key] += 1
        return hook

    # Frozen54 calls encoder/policy_net/action_head directly, not actor.forward.
    actor.action_head.register_forward_hook(count('actor_calls'))
    report['actor_call_count_semantics'] = 'one action_head forward per frozen54 actor prediction'
    points, targets, costs, grids, firsts, heads, worlds = historical75(original, summary53, report['consumed53_inputs'])
    points['fresh'] = load_own_points(ROOT / SOURCE / 'probe_snapshots.pkl')
    check_points(points['fresh'], (75301, 75302, 75303))
    grids['fresh'] = load_grid(ROOT / SOURCE, source['physical_probes'])
    targets['fresh'] = [grid[0]['states'].copy() for grid in grids['fresh']]
    for collection, key in ((costs, 'cost_references'), (firsts, 'first_step_references'),
                            (heads, 'head_references'), (worlds, 'world_vz_references')):
        collection['fresh'] = source[key]['fresh']
        require(collection == source[key], 'frozen physical references changed: ' + key)
    labels = fit75.load_labels75(ROOT / 'evidence/world_vz75/labels_01')
    old59 = ROOT / 'evidence/yaw_jvp59/execution_01'
    previous59 = json.loads((old59 / 'report.json').read_text())
    appendix_points = load_own_points(old59 / 'probe_snapshots.pkl')
    appendix_grids = load_grid(old59, previous59['physical_probes'])
    appendix_targets = [g[0]['states'].copy() for g in appendix_grids]
    start = time.monotonic()
    with (output / 'progress.jsonl').open('x', buffering=1) as progress:
        for seed in SEEDS:
            require(time.monotonic() - start < 418. and time.process_time() < 298., '78 budget before model')
            components, component_receipts = [], []
            for arm in ('control', 'active'):
                old = next(m for m in source['models'] if (m['seed'], m['arm']) == (seed, arm))
                require(old['status'] == old['fit']['status'] == 'COMPLETE'
                        and old['fit']['updates'] == old['fit']['updates_attempted'] == 1000, 'partial component')
                relative = old['directory'] + '/final_state_dict.pt'
                path = ROOT / SOURCE / relative
                require(sha(path) == source['sealed_artifacts_sha256'][relative], 'checkpoint file changed')
                model = dynamics52.Dynamics52()
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
                digest = fit75.paired.mapping_sha256(model.state_dict())
                require(digest == old['fit']['final_model_state_sha256'], 'component mapping changed')
                components.append(model.eval().requires_grad_(False))
                component_receipts.append({'arm75': arm, 'path': str(path.relative_to(ROOT)), 'file_sha256': sha(path),
                                           'mapping_sha256': digest, 'inherited_complete_updates': 1000, 'new_updates': 0})
            composed32 = FrozenPair78(*components).eval().requires_grad_(False)
            row = {'seed': seed, 'arm': 'active', 'status': 'INCOMPLETE', 'directory': f'protected_seed{seed}',
                   'source_components': component_receipts, 'new_updates': 0, 'diagnostics': {}, 'first_step': {},
                   'heads': {}, 'world_vz': {}, 'cost_h1': {}, 'regression59': {}}
            report['models'].append(row)
            folder = output / row['directory']
            folder.mkdir(exist_ok=False)
            for module in components:
                module.register_forward_hook(count('model_calls'))
            composed32.register_forward_hook(count('composed_forward_calls'))
            row['final_training_world_vz_diagnostics'] = fit75.final_training_world_vz_diagnostics(
                composed32, labels, folder / 'world_vz_training_readback')
            for module, provenance in zip(components, component_receipts, strict=True):
                require(fit75.paired.mapping_sha256(module.state_dict()) == provenance['mapping_sha256'], 'F32 source changed')
            composed = copy.deepcopy(composed32).double().eval().requires_grad_(False)
            state_before = fit75.paired.mapping_sha256(composed.state_dict())
            for role in ROLES:
                row['diagnostics'][role] = {}
                evaluate_points(composed, actor, points[role], targets[role], folder / role, row['diagnostics'][role], 298.)
                progress.write(json.dumps({'seed': seed, 'phase': 'H32', 'role': role, 'cpu_seconds': time.process_time()}) + '\n')
            for role in FIRST:
                for field in ('first_step', 'cost_h1', 'heads', 'world_vz'):
                    row[field][role] = {}
                evaluate_first_steps(composed, points[role], grids[role], folder / ('first_step_' + role), row['first_step'][role], 298.)
                outputs69.evaluate_h1_costs(composed, actor, points[role], folder / ('cost_h1_' + role), row['cost_h1'][role], 298.)
                outputs69.evaluate_heads(composed, points[role], grids[role], row['cost_h1'][role]['points'],
                    folder / ('heads_' + role), row['heads'][role], 298.)
                outputs75.evaluate_world_vz(composed, points[role], grids[role], folder / ('world_vz_' + role), row['world_vz'][role], 298.)
            regression = row['regression59']
            regression.update(cost_diagnostics={}, first_diagnostics={}, used_for_gate_or_selection=False)
            evaluate_points(composed, actor, appendix_points, appendix_targets, folder / 'old59', regression['cost_diagnostics'], 298.)
            evaluate_first_steps(composed, appendix_points, appendix_grids, folder / 'first_step_old59', regression['first_diagnostics'], 298.)
            require(fit75.paired.mapping_sha256(composed.state_dict()) == state_before, 'evaluation changed model state')
            row.update(status='COMPLETE', composite_mapping_sha256_f64=state_before)
            progress.write(json.dumps({'seed': seed, 'phase': 'MODEL_COMPLETE', 'cpu_seconds': time.process_time()}) + '\n')
    report['invariant_audit'] = invariants(report, source, output)
    report['comparison'] = compare(report['models'], source)
    report['comparison']['arithmetic_invariants_passed'] = report['invariant_audit']['arithmetic_invariants_passed']
    if not report['invariant_audit']['arithmetic_invariants_passed']:
        report['comparison'].update(candidate_accepted=False, complete_repair_validated=False,
                                    candidate_status='FAILED', complete_repair_status='FAILED')
        report['comparison']['candidate_known_failures'].append('arithmetic_preservation_invariant_failed')
        report['comparison']['complete_repair_known_failures'].append({'reason': 'arithmetic_preservation_invariant_failed'})
    report['sealed_artifacts_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    report['artifact_bytes_before_report'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan_hash = sha(args.plan)
    current = {}

    def body(report):
        current['report'] = report
        report['plan_sha256'] = plan_hash
        calculate(report, plan, args.output)

    def after():
        try:
            require(sha(args.plan) == plan_hash, '78 plan changed')
            verify(plan)
            r = current.get('report', {})
            for name, digest in r.get('sealed_artifacts_sha256', {}).items():
                require(sha(args.output / name) == digest, '78 sealed output changed')
            for name, value in r.get('versions', {}).items():
                require(importlib.metadata.version(name) == value, '78 dependency changed')
        except BaseException:
            current.get('report', {}).get('comparison', {}).update(
                status='INCOMPLETE', candidate_status='INCOMPLETE', complete_repair_status='INCOMPLETE',
                candidate_accepted=False, complete_repair_validated=False)
            raise
        finally:
            if current.get('report', {}).get('status') != 'COMPLETE':
                current.get('report', {}).get('comparison', {}).update(
                    status='INCOMPLETE', candidate_status='INCOMPLETE', complete_repair_status='INCOMPLETE',
                    candidate_accepted=False, complete_repair_validated=False)

    report = run_preserving(args.output, body, after)
    print(json.dumps({k: report.get(k) for k in ('status', 'failure', 'verification_after', 'model_calls', 'actor_calls', 'cpu_seconds')}))
    return int(report['status'] != 'COMPLETE')


if __name__ == '__main__':
    raise SystemExit(main())
