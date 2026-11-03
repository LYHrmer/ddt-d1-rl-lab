"""Fixed endpoint interventions on all complete75 pairs; no model or physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools74.geometry74 import endpoint_outputs
from tools74.runtime74 import run_preserving
from tools76.readback76 import f64_array
from tools77.geometry77 import MODES, compose, preservation_receipt

ARCHIVE = 'evidence/world_vz75/execution_01'
PUBLISHED = 'reports/world_vz75_results_20261101.json'
OLD_PLAN = 'docs/experiments75/world_vz75_preregistered.json'
SEEDS = (64001, 64002, 64003)
ROLES = ('old56', 'old64', 'old69', 'old72', 'fresh')
EPS = (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4)
OUTPUTS = ('body_vx', 'body_vy', 'world_vz', 'yaw')
SCALE = 0.30000001192092896


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    names = {OLD_PLAN, PUBLISHED, ARCHIVE + '/report.json', 'tools77/readback77.py',
             'tools77/geometry77.py', 'tests/test_geometry77.py', 'tools74/geometry74.py',
             'tools74/runtime74.py', 'tools76/readback76.py', 'tools76/reasons76.py',
             'tools76/precision76.py', 'docs/output_isolation77_plan_20261103.md',
             '.github/workflows/output-isolation77.yml'}
    for seed in SEEDS:
        for arm in ('control', 'active'):
            for role in ROLES:
                names.update(f'{ARCHIVE}/{arm}_seed{seed}/first_step_{role}/point_{i:02d}.npz' for i in range(9))
    return names


def verify(plan):
    require(plan['schema'] == 'd1-output-isolation77-v1', 'wrong77 plan')
    require(plan['seeds'] == list(SEEDS) and plan['roles'] == list(ROLES)
            and plan['eps'] == list(EPS) and plan['modes'] == list(MODES), 'grid changed')
    require(plan['cpu_seconds'] == 30 and plan['wall_seconds'] == 60 and plan['paired_intervals'] == 540,
            '77 budget changed')
    require(all(plan[k] == 0 for k in ('model_calls', 'actor_calls', 'physics_steps', 'updates')), 'readback only')
    require(inputs() <= plan['source_sha256'].keys(), '77 inputs not bound')
    require(sha(ROOT / OLD_PLAN) == 'ecaad0a625b3e55cadc2fe90f11b27d714b6f87c0e7b73b96f553d3dcd1d4a4c',
            'frozen75 plan changed')
    inherited = json.loads((ROOT / OLD_PLAN).read_text())['source_sha256']
    require(all(plan['source_sha256'].get(k) == v for k, v in inherited.items()), '75 bindings not inherited')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'source/input changed: ' + name)


def values(state):
    body = endpoint_outputs(state)
    return {'body_vx': body['body_vx'], 'body_vy': body['body_vy'], 'world_vz': state[25], 'yaw': state[28]}


def close(a, b):
    require(abs(a - b) <= 1e-10 + 1e-8 * abs(b), 'archived response parity failed')


def aggregate(rows):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for channel in OUTPUTS:
            errors = [row['modes'][mode][channel]['normalized_error'] for row in rows]
            result[mode][channel] = {'normalized_rmse': math.sqrt(math.fsum(x*x for x in errors) / len(errors)),
                'scalar_count': len(errors),
                'endpoint_sign_and_magnitude_matches': sum(row['modes'][mode][channel]['endpoint_match'] is True for row in rows),
                'unresolved_physical_references': sum(row['modes'][mode][channel]['endpoint_match'] is None for row in rows),
                'model_JVP_qualification_for_constructed_modes': 'NOT_MEASURED'}
        body_errors = [row['modes'][mode][c]['normalized_error'] for row in rows for c in ('body_vx', 'body_vy')]
        result[mode]['body_xy_normalized_rmse'] = math.sqrt(math.fsum(x*x for x in body_errors) / len(body_errors))
    return result


def calculate(report, plan):
    resource.setrlimit(resource.RLIMIT_CPU, (28, 30))
    verify(plan)
    source = json.loads((ROOT / ARCHIVE / 'report.json').read_text())
    published = json.loads((ROOT / PUBLISHED).read_text())
    require(source['status'] == source['verification_after'] == 'COMPLETE'
            and source['completed_model_fits'] == 6 and source['model_updates'] == 6000, 'complete75 required')
    require(len(source['models']) == 6, 'six models required')
    by_key = {(m['seed'], m['arm']): m for m in source['models']}
    require(set(by_key) == {(s, a) for s in SEEDS for a in ('control', 'active')}, 'paired models missing')
    report.update(schema='d1-output-isolation77-result-v1', comparisons=[], original_decisions_changed=False,
                  source_candidate_status=source['comparison']['candidate_status'], input_files={},
                  paired_intervals=0, endpoint_interventions=0, maximum_body_preservation_gap=0.,
                  maximum_body_fd_preservation_gap=0., maximum_horizontal_compensation=0.,
                  minimum_abs_Rzz=1., new_candidate_accepted=False,
                  scope='retrospective saved first-step endpoints; no new model JVP, H32 rollout or heldout evaluation')
    wall_start = time.monotonic()
    all_rows = []
    for seed in SEEDS:
        for role in ROLES:
            left, right = [by_key[(seed, arm)] for arm in ('control', 'active')]
            require(left['fit']['status'] == right['fit']['status'] == 'COMPLETE', 'partial model')
            for index in range(9):
                require(time.monotonic() - wall_start < 58., '77 wall budget')
                lpoint, rpoint = [m['first_step'][role]['points'][index] for m in (left, right)]
                require(lpoint['status'] == rpoint['status'] == 'COMPLETE'
                        and (lpoint['seed'], lpoint['tick']) == (rpoint['seed'], rpoint['tick']), 'point pair differs')
                require([e['epsilon'] for e in lpoint['eps']] == [e['epsilon'] for e in rpoint['eps']] == list(EPS),
                        'epsilon grid differs')
                row = {'model_seed': seed, 'role': role, 'seed': lpoint['seed'], 'tick': lpoint['tick'], 'eps': []}
                report['points'].append(row)
                paths = [f"{ARCHIVE}/{m['directory']}/first_step_{role}/point_{index:02d}.npz" for m in (left, right)]
                for name in paths:
                    digest = sha(ROOT / name)
                    require(digest == published['raw_manifest'][name]['sha256'], 'raw NPZ differs')
                    report['input_files'][name] = digest
                with zipfile.ZipFile(ROOT / paths[0]) as cz, zipfile.ZipFile(ROOT / paths[1]) as az:
                    for ep, eps in enumerate(EPS):
                        require(f64_array(cz, f'epsilon_{ep}_initial_state', (45,)) ==
                                f64_array(az, f'epsilon_{ep}_initial_state', (45,)), 'initial state differs')
                        ends = {arm: {sign: list(f64_array(z, f'epsilon_{ep}_model_next_{sign}', (1, 45)))
                                      for sign in ('plus', 'minus')} for arm, z in (('control', cz), ('active', az))}
                        truth = {sign: list(f64_array(cz, f'epsilon_{ep}_actual_next_{sign}', (45,))) for sign in ('plus', 'minus')}
                        for sign in ('plus', 'minus'):
                            require(tuple(truth[sign]) == f64_array(az, f'epsilon_{ep}_actual_next_{sign}', (45,)), 'truth differs')
                            require(f64_array(cz, f'epsilon_{ep}_actual_action_{sign}', (1, 16)) ==
                                    f64_array(az, f'epsilon_{ep}_actual_action_{sign}', (1, 16)), 'actual action differs')
                        true_values = {sign: values(s) for sign, s in truth.items()}
                        item = {'epsilon': eps, 'modes': {}, 'preservation': {}}
                        row['eps'].append(item)
                        refs = {'body_vx': source['head_references'][role][index]['body']['vx'],
                                'body_vy': source['head_references'][role][index]['body']['vy'],
                                'world_vz': source['world_vz_references'][role][index],
                                'yaw': source['first_step_references'][role][index]}
                        for mode in MODES:
                            made = {sign: compose(ends['control'][sign], ends['active'][sign], mode) for sign in ('plus', 'minus')}
                            outs = {sign: values(s) for sign, s in made.items()}
                            item['modes'][mode] = {}
                            for channel in OUTPUTS:
                                predicted = (outs['plus'][channel] - outs['minus'][channel]) / (2 * eps)
                                actual = (true_values['plus'][channel] - true_values['minus'][channel]) / (2 * eps)
                                close(actual, refs[channel]['eps'][ep]['central64'])
                                match = (abs(predicted) >= 1e-5 and predicted * actual > 0) if refs[channel]['resolved'] else None
                                item['modes'][mode][channel] = {'model_fd': predicted, 'true_fd': actual,
                                    'normalized_error': (predicted - actual) / SCALE, 'reference_resolved': refs[channel]['resolved'],
                                    'endpoint_match': match, 'full_model_qualification': 'NOT_REJUDGED_OR_NOT_MEASURED'}
                                if mode in ('control', 'active'):
                                    original = by_key[(seed, mode)]
                                    old = (original['heads'][role]['points'][index]['body'][channel[-2:]] if channel.startswith('body_')
                                           else original['world_vz'][role]['points'][index] if channel == 'world_vz'
                                           else original['first_step'][role]['points'][index])
                                    close(predicted, old['eps'][ep]['model_fd'])
                            if mode == 'body_protected':
                                for sign in ('plus', 'minus'):
                                    p = preservation_receipt(ends['control'][sign], ends['active'][sign], made[sign])
                                    item['preservation'][sign] = p
                                    require(p['unchanged_nonvelocity'] and p['world_vz_exact'], 'protected coordinates differ')
                                    report['maximum_body_preservation_gap'] = max(report['maximum_body_preservation_gap'], *p['body_xy_abs_gap'])
                                    report['minimum_abs_Rzz'] = min(report['minimum_abs_Rzz'], abs(p['Rzz']))
                                    report['maximum_horizontal_compensation'] = max(report['maximum_horizontal_compensation'],
                                        *(abs(made[sign][k] - ends['control'][sign][k]) for k in (23, 24)))
                        item['finite_interval_interaction'] = {}
                        for channel in OUTPUTS:
                            c, a, w, o, protected = [item['modes'][mode][channel]['model_fd'] for mode in MODES]
                            interaction = a - w - o + c
                            total, world, other = a - c, w - c, o - c
                            close(total, world + other + interaction)
                            item['finite_interval_interaction'][channel] = {
                                'total_active_minus_control': total, 'world_only_minus_control': world,
                                'other_only_minus_control': other, 'interaction': interaction,
                                'identity_residual': total - (world + other + interaction)}
                            if channel.startswith('body_'):
                                report['maximum_body_fd_preservation_gap'] = max(
                                    report['maximum_body_fd_preservation_gap'], abs(protected - c))
                        all_rows.append((seed, role, item))
                        report['paired_intervals'] += 1
                        report['endpoint_interventions'] += 2 * len(MODES)
    require(report['paired_intervals'] == 540 and len(report['points']) == 135, 'incomplete registered grid')
    require(report['maximum_body_preservation_gap'] <= 1e-12, 'body invariant exceeded arithmetic audit tolerance')
    for seed in SEEDS:
        rows = [item for s, _, item in all_rows if s == seed]
        summary = aggregate(rows)
        report['comparisons'].append({'seed': seed, 'all': summary,
            'by_role': {role: aggregate([i for s, r, i in all_rows if s == seed and r == role]) for role in ROLES}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    plan_hash = sha(args.plan)

    def body(report):
        report['plan_sha256'] = plan_hash
        calculate(report, plan)

    def after():
        require(sha(args.plan) == plan_hash, '77 plan changed during readback')
        verify(plan)

    report = run_preserving(args.output, body, after)
    print(json.dumps({k: report.get(k) for k in ('status', 'verification_after', 'failure', 'paired_intervals', 'cpu_seconds')}))
    return 0 if report['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
