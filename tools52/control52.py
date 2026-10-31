"""Finite-horizon policy-gradient probe; no logged future actions in imagination."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from tools52 import adapter52 as adapter

HORIZON = 32
POINT_TICKS = (65, 205, 305)


def require(value, message):
    if not value:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def physical(env):
    return np.r_[env.data.qpos, env.data.qvel].copy()


def sequence(index):
    amplitude = (.1, .2, .3)[index % 3]
    sign = 1 if index % 2 == 0 else -1
    vx = 0. if index % 4 < 2 else .1
    return [{'start_s': 0., 'vx': 0., 'wz': 0.},
            {'start_s': 1., 'vx': vx, 'wz': sign * amplitude},
            {'start_s': 4., 'vx': vx, 'wz': -sign * amplitude},
            {'start_s': 6., 'vx': 0., 'wz': 0.}]


def command_at(spec, tick):
    segment = spec[0]
    for item in spec:
        if tick * .02 + 1e-12 >= item['start_s']:
            segment = item
    return np.array([segment['vx'], 0., segment['wz']])


def tensor(array, *, double=False):
    return torch.as_tensor(np.asarray(array).copy(), dtype=torch.float64 if double else torch.float32)


def cache_tensor(shaper):
    if shaper.state is None:
        return None
    return {'state': tensor([shaper.state], double=True),
            'history': tensor(shaper.history[None]), 'raw_history': tensor(shaper.raw_history[None])}


def point(env, shaper, spec, seed):
    return {'snapshot': adapter.capture(env, shaper), 'state': physical(env),
            'frame': env._current_obs.copy(), 'history': env._history.copy(),
            'filtered': env._filtered_gyro.copy(), 'last': env._last_policy_action.copy(),
            'fifo': np.asarray(list(env._delay_fifo)).reshape(-1, 16),
            'shaper': copy.deepcopy(shaper), 'seed': seed, 'tick': env._episode_steps,
            'commands': np.asarray([command_at(spec, env._episode_steps + t)
                                    for t in range(HORIZON + 1)]),
            'nominal_height': env.nominal_base_height_m}


def rotate_inverse(q, vector):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    cross = torch.linalg.cross(q[..., 1:], vector, dim=-1)
    return vector - 2 * q[..., :1] * cross + 2 * torch.linalg.cross(q[..., 1:], cross, dim=-1)


def cost(state, command, action, previous, nominal_height):
    """Local tracking objective, not the frozen PPO reward or qualification J."""
    velocity = rotate_inverse(state[..., 3:7], state[..., 23:26])
    gravity = rotate_inverse(state[..., 3:7], torch.tensor([0., 0., -1.]).expand_as(velocity))
    return (((velocity[..., 0] - command[..., 0]) / .2) ** 2
            + ((state[..., 28] - command[..., 2]) / .2) ** 2
            + (velocity[..., 1] / .2) ** 2 + 6 * (gravity[..., :2] ** 2).sum(-1)
            + 20 * (state[..., 2] - nominal_height) ** 2
            + .1 * ((action - previous) ** 2).mean(-1))


def imaginary(model, actor, start, theta):
    state, frame = tensor(start['state'][None]), tensor(start['frame'][None])
    history = tensor(start['history'][None])
    filtered = tensor(start['filtered'][None], double=True)
    previous = tensor(start['last'][None])
    fifo = tensor(start['fifo'][None])
    cache = cache_tensor(start['shaper'])
    terms = []
    for t in range(HORIZON):
        command = tensor(start['commands'][t][None])
        shaped, cache = adapter.reference_step(frame, history, cache)
        issued = adapter.residual_action(actor(shaped), frame, command, theta)
        applied, fifo = adapter.delay_step(issued, fifo)
        history = adapter.advance_history(history, frame)
        state = model(state, applied)
        terms.append(cost(state, command, issued, previous, start['nominal_height']))
        previous = issued
        frame, filtered = adapter.build_observation(
            state, tensor(start['commands'][t + 1][None]), issued, filtered)
    value = torch.stack(terms).mean()
    require(bool(torch.isfinite(value)), 'nonfinite imagined cost')
    return value


def gradients(model, actor, points):
    rows = []
    for start in points:
        theta = torch.tensor(0., requires_grad=True)
        value = imaginary(model, actor, start, theta)
        derivative, = torch.autograd.grad(value, theta)
        with torch.no_grad():
            plus = imaginary(model, actor, start, torch.tensor(.001))
            minus = imaginary(model, actor, start, torch.tensor(-.001))
            numerical = float((plus - minus) / .002)
        analytic = float(derivative)
        require(np.isfinite(analytic) and np.isfinite(numerical), 'nonfinite policy gradient')
        rows.append({'seed': start['seed'], 'tick': start['tick'], 'cost': float(value.detach()),
                     'gradient': analytic, 'model_central_difference': numerical,
                     'direction_identifiable': abs(analytic) >= 1e-5 and abs(numerical) >= 1e-5
                     and analytic * numerical > 0,
                     'autograd_agrees': abs(analytic - numerical) <= .1 * abs(numerical) + 1e-4})
    return rows


def real_action(env, shaper, actor, theta, noise=None):
    obs = env.current_observation()
    shaped = shaper(obs)
    with torch.no_grad():
        base = actor(tensor(shaped[None]))
        issued = adapter.residual_action(base, tensor(obs[:57][None]),
                                         tensor(env._command[None]), torch.tensor(theta))[0].numpy()
    if noise is not None:
        issued = np.clip(issued + noise, -1., 1.).astype(np.float32)
    return issued


def real_branch(env, shaper, actor, start, theta):
    adapter.restore(env, shaper, start['snapshot'])
    values, states, observations, rewards = [], [], [], []
    for t in range(HORIZON):
        require(np.array_equal(env._command, start['commands'][t]), 'branch command clock differs')
        previous = env._last_policy_action.copy()
        action = real_action(env, shaper, actor, theta)
        obs, reward, terminated, truncated, _info = env.step(action)
        state = physical(env)
        with torch.no_grad():
            value = float(cost(tensor(state[None]), tensor(start['commands'][t][None]),
                               tensor(action[None]), tensor(previous[None]), start['nominal_height']))
        values.append(value); states.append(state); observations.append(obs); rewards.append(reward)
        if terminated or truncated:
            return {'status': 'TERMINATED', 'theta': theta, 'steps': t + 1, 'cost': None,
                    'termination': {'terminated': terminated, 'truncated': truncated}}, states, observations, rewards
    return {'status': 'COMPLETE', 'theta': theta, 'steps': HORIZON,
            'cost': float(np.mean(values))}, states, observations, rewards


def compare_directions(sealed, probes):
    rows = []
    for predictions in sealed['test_gradients']:
        matches = 0
        for predicted, actual in zip(predictions, probes, strict=True):
            if actual['identifiable'] and predicted['direction_identifiable'] and predicted['autograd_agrees'] \
                    and predicted['gradient'] * actual['gradient'] > 0:
                matches += 1
        rows.append({'matches': matches, 'total': len(probes),
                     'passed': matches >= 7 and all(r['autograd_agrees'] for r in predictions)})
    complete = all(b['status'] == 'COMPLETE' for p in probes for b in p['branches'])
    return {'models': rows, 'all_physical_branches_complete': complete,
            'passed': sealed['theta'] is not None and complete
            and all(r['passed'] for r in rows) and all(p['repeated_baseline_exact'] for p in probes)}
