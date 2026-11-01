"""Read-only73 support geometry of the complete72 response regression grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools73.support73 import (
    action_direction,
    masked_state_distance,
    summarize_distance_association,
)

REPORT72 = 'evidence/yaw_weight72/execution_01/report.json'
PUBLISHED72 = 'reports/yaw_weight72_results_20261101.json'
LABELS = 'evidence/body_response68/admission_01/body_labels96.npz'
TRAIN = 'evidence/model_gradient53/execution_01/train_transitions.npz'
ROLES = ('old56', 'old64', 'old69', 'fresh')
SEEDS = (64001, 64002, 64003)
EPS = (3.125e-5, 6.25e-5, 1.25e-4, 2.5e-4)
ZERO = (0, 1, 10, 14, 18, 22)
GROUPS = {'position': tuple(range(3)), 'quaternion': tuple(range(3, 7)),
          'joint_position': tuple(range(7, 23)), 'base_linear_velocity': tuple(range(23, 26)),
          'base_angular_velocity': tuple(range(26, 29)),
          'leg_velocity': tuple(29+i for i in range(16) if i not in (3, 7, 11, 15)),
          'wheel_velocity': (32, 36, 40, 44)}
AXIS = [0.] * 16
for _i, _v in zip((3, 7, 11, 15), (1., -1., 1., -1.), strict=True):
    AXIS[_i] = _v


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inputs():
    paths = {REPORT72, PUBLISHED72, LABELS, TRAIN, 'docs/experiments72/yaw_weight72_preregistered.json',
             'tools73/run73.py', 'tools73/support73.py', 'tests/test_support73.py',
             'docs/support73_plan_20261101.md', '.github/workflows/support73.yml', 'tools52/dynamics52.py'}
    for role in ROLES:
        for i in range(9):
            for kind in ('first_step', 'heads'):
                paths.add(f'evidence/yaw_weight72/execution_01/control_seed64001/{kind}_{role}/point_{i:02d}.npz')
    return paths


def verify(plan):
    require(plan['schema'] == 'd1-response-support73-v1', 'wrong plan')
    require(plan['roles'] == list(ROLES) and plan['model_seeds'] == list(SEEDS), 'grid changed')
    require(plan['points'] == 36 and plan['training_centers'] == 24 and plan['training_transitions'] == 7200,
            'coverage changed')
    require(plan['cpu_seconds'] == 60 and plan['wall_seconds'] == 120, 'budget changed')
    require(all(plan[k] == 0 for k in ('model_calls', 'actor_calls', 'physics_steps', 'updates')), 'not read-only')
    require(inputs() <= set(plan['source_sha256']), 'missing binding')
    old_path = ROOT / 'docs/experiments72/yaw_weight72_preregistered.json'
    require(sha(old_path) == 'a80e30727ddaedde27d2fa5d52f4dc55b92056cfd04ac08c46c81d5007fab4fa', 'original72 plan changed')
    old = json.loads(old_path.read_text())
    require(all(plan['source_sha256'].get(k) == v for k,v in old['source_sha256'].items()), '72 bindings changed')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'bound source changed: ' + name)


def nearest(state, action, states, actions, identities, scales, np):
    """Two separate distances; no arbitrary state/action mixture or fitted metric."""
    error = (states.astype(np.float64) - state) / scales
    error[:, list(ZERO)] = 0
    group_values = np.stack([np.mean(error[:, indices] ** 2, axis=1) for indices in GROUPS.values()], axis=1)
    distances = np.sqrt(np.mean(group_values, axis=1))
    action_distances = np.sqrt(np.mean((actions.astype(np.float64) - action) ** 2, axis=1))
    si, ai = int(np.argmin(distances)), int(np.argmin(action_distances))
    check = masked_state_distance(state.tolist(), states[si].tolist(), scales.tolist(), GROUPS, ZERO)
    require(math.isclose(check['distance'], float(distances[si]), rel_tol=1e-12, abs_tol=1e-14), 'distance formula differs')
    return {'state_distance': float(distances[si]), 'state_neighbor': identities[si], 'state_neighbor_index': si,
            'state_neighbor_group_mse': dict(zip(GROUPS, group_values[si].tolist(), strict=True)),
            'action_distance_at_state_neighbor': float(action_distances[si]),
            'action_distance': float(action_distances[ai]), 'action_neighbor': identities[ai],
            'state_distance_at_action_neighbor': float(distances[ai]),
            'state_distance_exact_ties': int(np.count_nonzero(distances == distances[si])),
            'action_distance_exact_ties': int(np.count_nonzero(action_distances == action_distances[ai]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'd1-response-support73-result-v1', 'status': 'INCOMPLETE', 'failure': None,
              'model_calls': 0, 'actor_calls': 0, 'physics_steps': 0, 'updates': 0,
              'qualification72_changed': False, 'training_centers': [], 'points': [], 'associations': []}
    cpu, wall = time.process_time(), time.monotonic()
    plan = None
    write(args.output / 'initial_incomplete.json', report)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (59, 60))
        require(os.getpriority(os.PRIO_PROCESS, 0) == 10 and len(os.sched_getaffinity(0)) == 1, 'Nice10/single CPU required')
        require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'single thread required')
        plan = json.loads(args.plan.read_text())
        report['plan_sha256'] = sha(args.plan)
        verify(plan)
        import numpy as np
        report['numpy_version'] = np.__version__
        require(np.__version__ == '2.2.6', 'original NumPy required')
        prior = json.loads((ROOT / REPORT72).read_text())
        published = json.loads((ROOT / PUBLISHED72).read_text())
        require(prior == published['formal'] and prior['status'] == prior['verification_after'] == 'COMPLETE',
                'complete exact72 report required')
        for name in inputs():
            if name.startswith('evidence/yaw_weight72/'):
                require(sha(ROOT/name) == published['raw_manifest'][name]['sha256'], '72 raw binding differs')
        with np.load(ROOT / LABELS, allow_pickle=False) as raw:
            labels = {k: raw[k].copy() for k in raw.files}
        with np.load(ROOT / TRAIN, allow_pickle=False) as raw:
            train = {k: raw[k].copy() for k in ('s', 'a', 'episode', 'tick')}
        require(labels['s'].shape == (96, 45) and train['s'].shape == (7200, 45), 'full input required')
        scales = np.asarray([.1]*7 + [1. if i in (3,7,11,15) else .5 for i in range(16)]
                            + [.3]*6 + [5. if i in (3,7,11,15) else 2. for i in range(16)], dtype=np.float32).astype(np.float64)
        training_ids = [{'seed': 53101+int(e), 'tick': int(t)} for e,t in zip(train['episode'], train['tick'], strict=True)]
        center_ids = []
        for i in range(24):
            k = i*4
            require(all(np.array_equal(labels[key][k], labels[key][k+j]) for key in ('s','a0') for j in range(4)), 'center differs by epsilon')
            identity = {key: int(labels[key][k]) for key in ('source_id','seed','tick')}
            center_ids.append(identity)
            row = {'identity': identity, 'directions': [], 'target_yaw': labels['target_yaw'][k:k+4].tolist()}
            row['train_support'] = nearest(labels['s'][k], labels['a0'][k], train['s'], train['a'], training_ids, scales, np)
            for j in range(4):
                n=k+j
                direction = action_direction(labels['a_plus'][n].tolist(), labels['a_minus'][n].tolist(), float(labels['epsilon'][n]), AXIS)
                direction.update(epsilon=float(labels['epsilon'][n]), stored_training_v32=labels['v'][n].tolist(),
                                 stored_v32_max_abs_gap=float(np.max(np.abs(labels['v'][n].astype(np.float64)-direction['v']))),
                                 projected_axis_normalized_yaw=float(labels['target_yaw'][n])/direction['signed_projection_gain'],
                                 recorded_direction_exactly_parallel_to_axis=direction['orthogonal_norm']==0,
                                 normalized_value_is_finite_interval_proxy=True)
                row['directions'].append(direction)
            report['training_centers'].append(row)
        models = {(m['seed'],m['arm']):m for m in prior['comparison']['models']}
        for role in ROLES:
            for index in range(9):
                require(time.process_time()-cpu < 55 and time.monotonic()-wall < 115, 'diagnostic budget')
                base = ROOT/'evidence/yaw_weight72/execution_01/control_seed64001'
                with np.load(base/f'first_step_{role}/point_{index:02d}.npz', allow_pickle=False) as raw:
                    data = {k:raw[k].copy() for k in raw.files}
                with np.load(base/f'heads_{role}/point_{index:02d}.npz', allow_pickle=False) as raw:
                    heads = {k:raw[k].copy() for k in raw.files}
                reference = models[(64001,'control')]['first_step'][role]['points'][index]
                identity = {'role':role, 'seed':reference['seed'], 'tick':reference['tick']}
                state = data['epsilon_0_initial_state']
                center = ((data['epsilon_0_actual_action_plus']+data['epsilon_0_actual_action_minus'])/2
                          -heads['epsilon_0_action_midpoint_drift'])[0]
                require(np.array_equal(center,center.astype(np.float32).astype(np.float64)), 'center not actual F32')
                row = {'identity': identity, 'directions': [], 'models': []}
                row['label_support'] = nearest(state,center,labels['s'][::4],labels['a0'][::4],center_ids,scales,np)
                row['train_support'] = nearest(state,center,train['s'],train['a'],training_ids,scales,np)
                for ep,epsilon in enumerate(EPS):
                    plus,minus=[data[f'epsilon_{ep}_actual_action_{s}'][0] for s in ('plus','minus')]
                    require(np.array_equal(state,data[f'epsilon_{ep}_initial_state']), 'state differs by epsilon')
                    require(np.array_equal((plus+minus)/2-heads[f'epsilon_{ep}_action_midpoint_drift'][0],center), 'center differs by epsilon')
                    direction = action_direction(plus.tolist(),minus.tolist(),epsilon,AXIS)
                    require(direction['norm'] > 0 and direction['signed_projection_gain'] != 0, 'unresolved action direction; do not filter point')
                    truth = float((data[f'epsilon_{ep}_actual_next_plus'][28]-data[f'epsilon_{ep}_actual_next_minus'][28])/(2*epsilon))
                    require(truth == reference['eps'][ep]['true_fd'], 'true yaw readback differs')
                    direction.update(epsilon=epsilon, true_fd=truth,
                        projected_axis_normalized_yaw=truth/direction['signed_projection_gain'],
                        recorded_direction_exactly_parallel_to_axis=direction['orthogonal_norm']==0,
                        normalized_value_is_finite_interval_proxy=True,
                        nearest_label_target_yaw=float(labels['target_yaw'][4*row['label_support']['state_neighbor_index']+ep]))
                    row['directions'].append(direction)
                for seed in SEEDS:
                    values={}
                    for arm in ('control','active'):
                        point=models[(seed,arm)]['first_step'][role]['points'][index]
                        require((point['seed'],point['tick'])==(identity['seed'],identity['tick']), 'point identity changed')
                        delta=[p['model_fd']-p['true_fd'] for p in point['eps']]
                        unit=[d/v['norm'] for d,v in zip(delta,row['directions'],strict=True)]
                        values[arm]={'yaw_theta_normalized_rmse':math.sqrt(math.fsum(d*d for d in delta)/4)/float(scales[28]),
                                     'yaw_unit_action_rmse':math.sqrt(math.fsum(d*d for d in unit)/4),
                                     'passed':point['passed'], 'eps':point['eps']}
                    row['models'].append({'seed':seed,**values,
                        'unit_action_rmse_change':values['active']['yaw_unit_action_rmse']-values['control']['yaw_unit_action_rmse'],
                        'theta_normalized_rmse_change':values['active']['yaw_theta_normalized_rmse']-values['control']['yaw_theta_normalized_rmse']})
                report['points'].append(row)
        require(len(report['points'])==36 and len(report['training_centers'])==24, 'partial grid')
        for seed in SEEDS:
            for pool in ('label_support','train_support'):
                for metric in ('state_distance','action_distance','action_distance_at_state_neighbor'):
                    rows=[{**p['identity'],'distance':p[pool][metric],
                           'error_change':next(m['unit_action_rmse_change'] for m in p['models'] if m['seed']==seed)} for p in report['points']]
                    report['associations'].append({'seed':seed,'pool':pool,'distance_metric':metric,
                        **summarize_distance_association(rows,'distance','error_change')})
        report.update(status='COMPLETE',completed_points=36,completed_training_centers=24,
                      action_direction_rows=240, point_model_pairs=108,
                      descriptive_only=True, causal_interference_claim=False, candidate_authorized=False)
    except BaseException as exc:  # noqa: BLE001 -- retain all failed or partial diagnostics.
        report['failure']=f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if plan is not None:
                verify(plan)
                require(sha(args.plan)==report['plan_sha256'],'plan changed')
                report['verification_after']='COMPLETE'
        except Exception as exc:  # noqa: BLE001 -- never hide the original failure.
            report.update(status='INCOMPLETE',verification_after='FAILED',verification_failure=str(exc))
        report.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(args.output/'report.json',report)
        print(json.dumps({k:report[k] for k in ('status','failure','cpu_seconds','wall_seconds')},indent=2))
    return int(report['status']!='COMPLETE')


if __name__=='__main__':
    raise SystemExit(main())
