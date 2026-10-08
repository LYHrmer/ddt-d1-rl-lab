#!/usr/bin/env python3
"""Zero-integration comparison of fast diagnostics with the frozen D1 methods."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

from d1_rl.env import D1Env
from d1_rl.fast_checks import D1FastChecksEnv

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome(env):
    env._episode_done = False
    try:
        env._check_finite_state()
        return (None, None, env._episode_done)
    except Exception as exc:
        return (type(exc).__name__, str(exc), env._episode_done)


class Contacts:
    """Injection adapter exposing both old per-contact and new bulk access."""
    def __init__(self, pairs):
        self.geom = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)

    def __getitem__(self, selection):
        return [SimpleNamespace(geom1=int(a), geom2=int(b)) for a, b in self.geom[selection]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [ROOT/'training/d1_rl/env.py', ROOT/'training/d1_rl/interface.py',
                 ROOT/'training/configs/pilot03b.json', ROOT/'training/configs/interface03b.json']
    before = {str(p): sha(p) for p in protected}
    tests = []
    native_attempts = 0

    def forbidden_step(*_args, **_kwargs):
        nonlocal native_attempts
        native_attempts += 1
        raise AssertionError('This test forbids all mj_step calls')

    with patch.object(mujoco, 'mj_step', forbidden_step):
        base = D1Env(config='training/configs/pilot03b.json', seed=7001)
        fast = D1FastChecksEnv(config='training/configs/pilot03b.json', seed=7001)
        for method in ('__init__', 'step', 'reset', '_reward', '_observe', '_delayed_action'):
            assert getattr(D1FastChecksEnv, method) is getattr(D1Env, method)
        tests.append({'name': 'all_rollout_reset_reward_methods_are_original_objects', 'passed': True})
        base_obs, _ = base.reset(seed=7001)
        fast_obs, _ = fast.reset(seed=7001)
        assert np.array_equal(base_obs, fast_obs)
        assert outcome(base) == outcome(fast) == (None, None, False)
        live_views = fast._fast_finite_cache[1:]
        pointers = [a.__array_interface__['data'][0] for a in live_views]
        assert np.shares_memory(live_views[-1], fast.data.warning.number)
        for field in ('qpos', 'qvel', 'qacc'):
            for value in (float('nan'), float('inf'), -float('inf')):
                old = [getattr(env.data, field)[0] for env in (base, fast)]
                for env in (base, fast):
                    getattr(env.data, field)[0] = value
                first, second = outcome(base), outcome(fast)
                assert first == second and first[0] == 'FloatingPointError' and first[2] is True
                for env, previous in zip((base, fast), old):
                    getattr(env.data, field)[0] = previous
                tests.append({'name': f'{field}_{str(value)}_same_exception_and_done_flag', 'passed': True})
        for index in range(len(fast.data.warning.number)):
            for value in (-1, 0, 1, 3):
                base.data.warning.number[:] = 0
                fast.data.warning.number[:] = 0
                base.data.warning.number[index] = value
                fast.data.warning.number[index] = value
                first, second = outcome(base), outcome(fast)
                assert first == second
                assert (first[0] == 'FloatingPointError') is (value > 0)
                tests.append({'name': f'warning_slot{index}_count{value}_live_gt_zero', 'passed': True})
        for seed in (7001, 7002, 7003):
            base_obs, _ = base.reset(seed=seed)
            fast_obs, _ = fast.reset(seed=seed)
            assert np.array_equal(base_obs, fast_obs)
            assert outcome(base) == outcome(fast) == (None, None, False)
            assert [a.__array_interface__['data'][0] for a in fast._fast_finite_cache[1:]] == pointers
            assert np.shares_memory(live_views[-1], fast.data.warning.number)
            assert np.all(live_views[-1] == 0)
            tests.append({'name': f'mj_resetData_seed{seed}_preserves_live_views_and_clears_warning', 'passed': True})
        # Rebinding MjData must not retain stale views even though reset normally
        # preserves the original allocation.
        old_data = fast.data
        fast.data = mujoco.MjData(fast.model)
        fast.data.warning.number[2] = 1
        assert outcome(fast)[0] == 'FloatingPointError'
        assert fast._fast_finite_cache[0] is fast.data
        fast.data = old_data
        fast.reset(seed=7001)
        assert outcome(fast) == (None, None, False)
        tests.append({'name': 'explicit_data_rebinding_refreshes_cached_live_views', 'passed': True})
        # Real original robot contacts and real mj_contactForce, generated by
        # static forward calls only. Never invoke step or a model integrator.
        static_contacts = []
        for height in (fast.nominal_base_height_m+.002, fast.nominal_base_height_m,
                       fast.nominal_base_height_m-.01, .20):
            for env in (base, fast):
                env.reset(seed=7001)
                env.data.qpos[env.free_qpos+2] = height
                mujoco.mj_forward(env.model, env.data)
            actual, expected = fast._contact_diagnostics(), base._contact_diagnostics()
            assert actual == expected
            static_contacts.append({'base_height_m': height, 'contacts': int(fast.data.ncon),
                                    'illegal_support': actual[0], 'self_contacts': actual[1]})
        tests.append({'name': 'actual_robot_static_contact_and_contactForce_parity',
                      'passed': True, 'samples': static_contacts})
        # Synthetic geom-role and force injection covers strict threshold,
        # reversed geom ordering, visuals, both-world, all non-world self pairs,
        # every supported world type and contype !=0 (not merely ==1).
        G = mujoco.mjtGeom
        model = SimpleNamespace(
            geom_bodyid=np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 0]),
            geom_contype=np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 2]),
            geom_type=np.array([G.mjGEOM_PLANE, G.mjGEOM_BOX, G.mjGEOM_HFIELD,
                               G.mjGEOM_SPHERE, G.mjGEOM_PLANE, G.mjGEOM_BOX,
                               G.mjGEOM_CYLINDER, G.mjGEOM_BOX, G.mjGEOM_SPHERE,
                               G.mjGEOM_BOX]))
        old_fake, new_fake = D1Env.__new__(D1Env), D1FastChecksEnv.__new__(D1FastChecksEnv)
        for env in (old_fake, new_fake):
            env.model, env.wheel_geoms = model, {6}
            env.config = {'termination': {'illegal_support_contact_normal_force_above_n': 1.}}
        fake_cases = []
        for pair in ((0, 5), (5, 0)):
            for force in (.999999, 1., 1.000001, -2., float('nan')):
                fake_cases.append(([pair], [force], 'threshold_and_order'))
        pairs = [(a, b) for a in range(10) for b in range(10)]
        fake_cases.extend([(pairs, [2.]*100, 'all_pairs_high_force'),
                           (list(reversed(pairs)), [2.]*100, 'all_pairs_reversed'),
                           ([(0, 5), (5, 8), (6, 8), (8, 5)], [2.]*4, 'self_contacts_after_first_illegal'),
                           ([], [], 'empty_contact_array'),
                           ([(0, 6)], [100.], 'wheel_support_ignored')])
        for pairs, normal_forces, name in fake_cases:
            results, calls = [], []
            for env in (old_fake, new_fake):
                env.data = SimpleNamespace(ncon=len(pairs), contact=Contacts(pairs))
                indices = []

                def force_stub(_model, _data, index, output):
                    assert np.all(output == 0), 'Force buffer must be cleared before every call'
                    indices.append(index)
                    output[:] = 123.
                    output[0] = normal_forces[index]

                with patch.object(mujoco, 'mj_contactForce', force_stub):
                    results.append(env._contact_diagnostics())
                calls.append(indices)
            assert results[0] == results[1] and calls[0] == calls[1]
            if name == 'self_contacts_after_first_illegal':
                assert results[0] == (True, 3)
            tests.append({'name': name, 'passed': True, 'contacts': len(pairs),
                          'result': list(results[1]), 'force_call_indices': calls[1]})
        assert base.counters()['native_steps'] == fast.counters()['native_steps'] == 0
        base.close()
        fast.close()
    assert native_attempts == 0
    assert before == {str(p): sha(p) for p in protected}
    report = {'status': 'PASS', 'physics_steps': 0, 'mj_step_attempts': native_attempts,
              'tests': tests, 'test_count': len(tests), 'mujoco_version': mujoco.__version__,
              'frozen_sources_unchanged': before,
              'fast_checks_sha256': sha(ROOT/'training/d1_rl/fast_checks.py'),
              'test_script_sha256': sha(Path(__file__).resolve()),
              'semantic_limit': 'Cached geom roles assume frozen model topology/contype/type/wheel classification. Dynamic contact arrays are reread every call. This zero-integration test does not establish rollout speed or dynamic parity.'}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'tests': len(tests), 'physics_steps': 0,
                      'fast_checks_sha256': report['fast_checks_sha256']}, indent=2))


if __name__ == '__main__':
    main()
