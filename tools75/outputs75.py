"""Actual-action world-vz response checks for75; no body-frame rotation.

Heldout derivatives use the original policy-theta coordinate and recorded F32
actions. They are not the absolute wheel-direction derivative used in training.
All four theta scales, repeated zero and original wheel-ULP gates are retained.
"""
from __future__ import annotations

import math
import time

from tools53.run_pilot53 import require, write_json
from tools55.probe_local55 import EPSILONS, action_delta_ulps
from tools65.diagnose65 import MODEL_EPS, selfcheck
from tools69.outputs69 import BODY_SCALE, scalar_reference

WORLD_VZ_INDEX = 25
WORLD_VZ_SCALE = BODY_SCALE
COORDINATE = 'original policy theta; actual first F32 issued actions at zero delay; world state25'


def world_vz_references(points, grids):
    import numpy as np

    require(len(points) == len(grids) == 9, 'all nine world-vz reference points required')
    references = []
    for start, grid in zip(points, grids, strict=True):
        require(len(grid) == 10 and np.asarray(start['fifo']).shape == (0, 16), 'complete zero-delay grid required')
        require(all(np.asarray(branch['states']).dtype == np.float64 for branch in grid),
                'world-vz physical endpoints require raw float64 states')
        states = np.stack([branch['states'][0] for branch in grid]).astype(np.float64)
        actions = np.stack([branch['observations'][0, 41:57] for branch in grid])
        require(states.shape == (10, 45) and actions.shape == (10, 16), 'world-vz reference seam shape differs')
        require(actions.dtype == np.float32, 'world-vz references require recorded float32 actions')
        require(bool(np.isfinite(states).all() and np.isfinite(actions).all()), 'nonfinite world-vz reference')
        require(states[0, WORLD_VZ_INDEX] == states[1, WORLD_VZ_INDEX]
                and np.array_equal(actions[0], actions[1]), 'repeated zero world-vz/actions differ')
        actions_resolved = all(action_delta_ulps(actions[0], action)['all_four_wheels_nonzero'] for action in actions[2:])
        values = states[:, WORLD_VZ_INDEX]
        ref = scalar_reference(float(values[0]), [(float(values[2 + 2 * i]), float(values[3 + 2 * i]))
                                                for i in range(4)], actions_resolved)
        references.append({'seed': start['seed'], 'tick': start['tick'], **ref, 'coordinate': COORDINATE,
                           'state_index': WORLD_VZ_INDEX, 'rotation_applied': False})
    return references


def evaluate_world_vz(model64, points, grids, output, ledger, cpu_deadline):
    """Read actual endpoint actions; F64 JVP and both frozen model selfchecks."""
    import numpy as np
    import torch

    from tools52.dynamics52 import PHYSICAL_SCALE
    from tools52.run_pilot52 import save_npz
    require(WORLD_VZ_SCALE == float(PHYSICAL_SCALE[WORLD_VZ_INDEX]), 'frozen world-vz scale changed')
    require(len(points) == len(grids) == 9, 'all nine world-vz diagnostic points required')
    output.mkdir(parents=True, exist_ok=False)
    ledger.update(status='INCOMPLETE', points=[], completed_points=0)
    try:
        for index, (start, grid) in enumerate(zip(points, grids, strict=True)):
            require(time.process_time() < cpu_deadline, 'CPU deadline before world-vz diagnostic')
            require(len(grid) == 10 and np.asarray(start['fifo']).shape == (0, 16), 'complete zero-delay world-vz grid required')
            row = {'seed': start['seed'], 'tick': start['tick'], 'status': 'INCOMPLETE', 'eps': [],
                   'coordinate': COORDINATE, 'state_index': WORLD_VZ_INDEX, 'rotation_applied': False}
            ledger['points'].append(row)
            arrays = {}
            try:
                state = torch.tensor(start['state'][None], dtype=torch.float64)
                require(state.shape == (1, 45) and bool(torch.isfinite(state).all()), 'finite state45 required')
                actions = np.stack([branch['observations'][0, 41:57] for branch in grid])
                require(actions.dtype == np.float32 and actions.shape == (10, 16) and bool(np.isfinite(actions).all()),
                        'world-vz prediction requires finite recorded float32 actions')
                center = torch.tensor(grid[0]['observations'][0:1, 41:57], dtype=torch.float64)

                def output_vz(action, state=state):
                    return model64(state, action)[..., WORLD_VZ_INDEX:WORLD_VZ_INDEX + 1]

                for ep, epsilon in enumerate(EPSILONS):
                    require(time.process_time() < cpu_deadline, 'CPU deadline during world-vz diagnostic')
                    plus = torch.tensor(grid[2 + 2 * ep]['observations'][0:1, 41:57], dtype=torch.float64)
                    minus = torch.tensor(grid[3 + 2 * ep]['observations'][0:1, 41:57], dtype=torch.float64)
                    direction = (plus - minus) / (2 * epsilon)
                    primal, jvp = torch.autograd.functional.jvp(output_vz, center, direction, create_graph=False, strict=False)
                    with torch.no_grad():
                        endpoint_plus, endpoint_minus = output_vz(plus), output_vz(minus)
                        fd = (endpoint_plus - endpoint_minus) / (2 * epsilon)
                        line_fd = (output_vz(center + MODEL_EPS * direction)
                                   - output_vz(center - MODEL_EPS * direction)) / (2 * MODEL_EPS)
                    require(all(value.dtype == torch.float64 and bool(torch.isfinite(value).all())
                                for value in (primal, jvp, fd, line_fd)),
                            'nonfinite world-vz model output')
                    arrays.update({f'epsilon_{ep}_{name}': value.detach().numpy() for name, value in (
                        ('actual_action_plus', plus), ('actual_action_minus', minus), ('actual_action_v', direction),
                        ('action_midpoint_drift', (plus + minus) / 2 - center), ('world_vz_zero', primal),
                        ('world_vz_plus', endpoint_plus), ('world_vz_minus', endpoint_minus), ('world_vz_fd', fd),
                        ('world_vz_jvp', jvp), ('world_vz_line_fd', line_fd))})
                    analytic, actual_fd, straight_fd = (float(value[0, 0]) for value in (jvp, fd, line_fd))
                    line_check, endpoint_check = selfcheck(analytic, straight_fd), selfcheck(analytic, actual_fd)
                    row['eps'].append({'epsilon': epsilon, 'model_fd': actual_fd, 'model_jvp': analytic,
                        'model_line_fd': straight_fd, 'line_selfcheck': line_check, 'endpoint_selfcheck': endpoint_check,
                        'numeric_valid': bool(line_check['autograd_agrees'] and endpoint_check['autograd_agrees']
                            and line_check['direction_identifiable'] and endpoint_check['direction_identifiable'])})
                row['status'] = 'COMPLETE'
                ledger['completed_points'] += 1
            except BaseException as exc:
                row['failure'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                save_npz(output / f'point_{index:02d}.npz', arrays)
                write_json(output / f'point_{index:02d}.json', row)
        ledger['status'] = 'COMPLETE'
    finally:
        write_json(output / 'result.json', ledger)
    return ledger


def world_vz_comparison(predictions, refs):
    """Standard-library-only comparison; undefined physical directions stay None."""
    require(len(predictions) == len(refs) == 9, 'all nine world-vz comparison points required')
    result = {'status': 'COMPLETE', 'points': [], 'valid_matches': 0, 'resolved_count': 0, 'numeric_valid': True}
    squared_errors = []
    for prediction, reference in zip(predictions, refs, strict=True):
        require(prediction['status'] == 'COMPLETE'
                and (prediction['seed'], prediction['tick']) == (reference['seed'], reference['tick']), 'world-vz identities differ')
        require([x['epsilon'] for x in prediction['eps']] == [x['epsilon'] for x in reference['eps']] == list(EPSILONS),
                'all four world-vz theta eps required')
        require(type(reference['resolved']) is bool, 'world-vz reference qualification must be boolean')
        rows = []
        for p, q in zip(prediction['eps'], reference['eps'], strict=True):
            require(math.isfinite(p['model_fd']) and math.isfinite(q['central64']), 'nonfinite world-vz comparison')
            require(type(p['numeric_valid']) is bool, 'world-vz numeric qualification must be boolean')
            same = p['model_fd'] * q['central64'] > 0 if reference['resolved'] else None
            magnitude = abs(p['model_fd']) >= 1e-5
            valid = bool(p['numeric_valid'] and magnitude and same) if reference['resolved'] else None
            result['numeric_valid'] &= p['numeric_valid'] and magnitude
            squared_errors.append(((p['model_fd'] - q['central64']) / WORLD_VZ_SCALE) ** 2)
            rows.append({'epsilon': p['epsilon'], 'model_fd': p['model_fd'], 'true_fd': q['central64'],
                         'same_sign': same, 'model_magnitude_admitted': magnitude,
                         'numeric_valid': p['numeric_valid'], 'valid_match': valid})
        passed = all(p['valid_match'] is True for p in rows) if reference['resolved'] else None
        result['points'].append({'seed': prediction['seed'], 'tick': prediction['tick'],
                                'reference_resolved': reference['resolved'], 'eps': rows, 'passed': passed})
        result['resolved_count'] += reference['resolved']
        result['valid_matches'] += passed is True
    require(len(squared_errors) == 36 and all(math.isfinite(value) for value in squared_errors), 'finite36 world-vz errors required')
    mse = math.fsum(squared_errors) / 36
    result.update(normalized_mse=mse, normalized_rmse=math.sqrt(mse), error_scalar_count=36,
                  scale=WORLD_VZ_SCALE, error_includes_unresolved_reference_values=True,
                  passed_9_of_9=result['resolved_count'] == result['valid_matches'] == 9,
                  all_references_resolved=result['resolved_count'] == 9, coordinate=COORDINATE)
    return result
