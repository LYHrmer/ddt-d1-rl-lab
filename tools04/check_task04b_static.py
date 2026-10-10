#!/usr/bin/env python3
"""Zero-integration acceptance checks for contract 04b, run before any training.

Nothing here calls mj_step. Environments are constructed and reset, which uses
mj_resetData/mj_forward only, so the native integration counter must stay at 0.

check_task04a_static.py is imported rather than copied for the parts that are
contract-independent; it is not modified, because its SHA-256 is recorded in
04a's release receipt. The 04b-specific checks are the single added term, the
wheel-magnitude shape, and the injected-entry wiring through factory04b.
"""
import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
sys.path.insert(0, str(ROOT/'tools04'))

from check_task04a_static import check_protection, _raises  # noqa: E402

CONFIG_04B = ROOT/'training/configs/pilot04b.json'
CONFIG_04A = ROOT/'training/configs/pilot04a.json'
CONFIG_03D = ROOT/'training/configs/pilot03d.json'
EXPECTED_NEW_REWARD_KEYS = ('wheel_action_mean_squared_weight',
                            'wheel_action_mean_squared_saturation')
WHEEL_ACTION_INDEX = (3, 7, 11, 15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _worker_resolves_class(queue, config_path):
    try:
        sys.path.insert(0, str(ROOT/'training'))
        from d1_rl.factory04b import environment_class
        queue.put(('ok', environment_class(config_path).__name__, os.getpid()))
    except BaseException as error:  # noqa: BLE001 - the failure identity is the result
        queue.put(('error', f'{type(error).__name__}: {error}', os.getpid()))


def check_config(record):
    from d1_rl.task04b import _expected_04b_config, CHANGE_FROM_PILOT03D
    config04b = json.loads(CONFIG_04B.read_text())
    record('pilot04b_matches_expected_derivation', config04b == _expected_04b_config(), {})

    config03d = json.loads(CONFIG_03D.read_text())
    added = set(config04b['reward']) - set(config03d['reward'])
    removed = set(config03d['reward']) - set(config04b['reward'])
    changed = [k for k in config03d['reward']
               if k in config04b['reward'] and config03d['reward'][k] != config04b['reward'][k]]
    record('reward_adds_exactly_the_one_term',
           added == set(EXPECTED_NEW_REWARD_KEYS) and not removed and not changed,
           {'added': sorted(added), 'removed': sorted(removed), 'changed_in_place': changed})

    top_diff = {k for k in set(config04b) | set(config03d)
                if config04b.get(k) != config03d.get(k)}
    record('no_other_configuration_change',
           top_diff == {'version', 'reward', 'change_from_pilot03c', 'change_from_pilot03d'},
           {'differing_top_level_keys': sorted(top_diff),
            'version': config04b['version'],
            'change_from_pilot03d': config04b['change_from_pilot03d'] == CHANGE_FROM_PILOT03D})

    # 04b is a sibling of 04a on the same 03d base, not a successor.
    config04a = json.loads(CONFIG_04A.read_text())
    record('04b_does_not_inherit_the_04a_anchor_term',
           not any('anchor' in key for key in config04b['reward'])
           and any('anchor' in key for key in config04a['reward']), {})
    return config04b


def check_term_shape(record, config04b):
    from d1_rl.task04b import wheel_action_penalty_raw
    reward = config04b['reward']
    saturation = reward['wheel_action_mean_squared_saturation']
    weight = reward['wheel_action_mean_squared_weight']
    table = []
    # Per-tick mean squares measured in the 03d and 04a native traces, plus the
    # approximate value at initialisation with log_std_init=-1.
    measured = {'03d_forward_11_to_12s': .06516, '03d_reverse_11_to_12s': .05481,
                '04a_turn_11_to_12s': .17363, 'approx_initialisation': .135}
    for name, value in measured.items():
        bounded = min(value, saturation)
        table.append({'case': name, 'measured_mean_square_per_tick': value,
                      'raw': bounded, 'contribution_per_second': bounded*weight,
                      'saturated': value > saturation})
    # Synthetic patterns, to show the term's shape in action units.
    cases = {'ideal_zero_action': [0., 0., 0., 0.],
             '03d_forward_window_mean_pattern': [.386, -.267, .065, .184],
             '04a_turn_window_mean_pattern': [-.041, .064, .150, -.080],
             'dither_rms_1rad_s': [.2, -.2, .2, -.2],
             'full_scale': [1., 1., 1., 1.]}
    for name, actions in cases.items():
        raw = wheel_action_penalty_raw(actions, saturation)
        table.append({'case': name, 'wheel_actions': actions,
                      'net_rad_s': float(np.mean(actions)*5.),
                      'rms_action': float(np.sqrt(np.mean(np.square(actions)))),
                      'raw': raw, 'contribution_per_second': raw*weight})
    record('zero_action_is_free', wheel_action_penalty_raw([0.]*4, saturation) == 0., {})
    record('term_saturates', wheel_action_penalty_raw([1.]*4, saturation) == saturation,
           {'saturation': saturation})
    # The whole measured operating range must stay inside the gradient, or the
    # policy would start training on a flat part of the term.
    record('gradient_live_over_the_measured_range',
           all(value < saturation for value in measured.values()),
           {'measured_mean_square_per_tick': measured, 'saturation': saturation,
            'why': ('an earlier 0.09 saturation put 04a 0.174 and initialisation 0.135 '
                    'in the flat region, which the static check caught before training')})
    record('both_observed_patterns_cost_far_more_than_zero_action',
           all(min(v, saturation)*abs(weight) > .5 for k, v in measured.items()
               if k.startswith(('03d_', '04a_'))),
           {name: min(v, saturation)*weight for name, v in measured.items()})
    record('worse_behaviour_costs_more',
           measured['04a_turn_11_to_12s'] > measured['03d_forward_11_to_12s']
           > measured['03d_reverse_11_to_12s'],
           {'order': '04a dither > 03d forward bias > 03d reverse bias'})
    record('zero_action_is_the_unique_minimum',
           wheel_action_penalty_raw([0.]*4, saturation) == 0.
           and all(wheel_action_penalty_raw(a, saturation) > 0. for a in
                   ([.01, 0., 0., 0.], [0., -.01, 0., 0.], [.05, -.05, .05, -.05])), {})
    worst = saturation*abs(weight)
    floor = (reward['vx_tracking_weight']+reward['wz_tracking_weight']
             + reward['alive_weight'])*.95
    record('worst_case_penalty_below_positive_floor', worst < floor,
           {'worst_penalty_per_second': worst, 'approximate_positive_floor': floor})
    record('term_rejects_bad_parameters',
           _raises(lambda: wheel_action_penalty_raw([0.]*4, 0.))
           and _raises(lambda: wheel_action_penalty_raw([0.]*3, .09)), {})
    return table


def check_factory(record):
    from d1_rl.factory04b import environment_class, environment_sources
    from d1_rl.task04b import D1Task04bEnv
    from d1_rl.task04a import D1Task04aEnv
    from d1_rl.task03d import D1Task03dEnv
    record('factory04b_selects_04b_task', environment_class(CONFIG_04B) is D1Task04bEnv, {})
    record('factory04b_preserves_04a_selection',
           environment_class(CONFIG_04A) is D1Task04aEnv, {})
    record('factory04b_preserves_03d_selection',
           environment_class(CONFIG_03D) is D1Task03dEnv, {})
    sources = [Path(p) for p in environment_sources(CONFIG_04B)]
    names = [p.name for p in sources]
    record('environment_sources_include_04b_chain_without_04a',
           {'task04b.py', 'factory04b.py', 'task03d.py', 'pilot03d.json'} <= set(names)
           and 'task04a.py' not in names and 'pilot04a.json' not in names
           and all(p.is_file() for p in sources) and len(names) == len(set(names)),
           {'sources': names})
    return {str(p.relative_to(ROOT)): sha(p) for p in sources}


def check_injected_entry(record):
    """Exercise the patched state tools04/run_frozen_entry.py actually runs under."""
    import d1_rl.factory03 as factory03
    import d1_rl.factory04b as factory04b
    from d1_rl.task04b import D1Task04bEnv
    from d1_rl.task03d import D1Task03dEnv
    original = (factory03.environment_class, factory03.environment_sources)
    try:
        factory03.environment_class = factory04b.environment_class
        factory03.environment_sources = factory04b.environment_sources
        resolved04b = factory03.environment_class(CONFIG_04B)
        resolved03d = factory03.environment_class(CONFIG_03D)
        sources04b = [Path(p) for p in factory03.environment_sources(CONFIG_04B)]
        sources03d = [Path(p) for p in factory03.environment_sources(CONFIG_03D)]
        ok = (resolved04b is D1Task04bEnv and resolved03d is D1Task03dEnv
              and all(p.is_file() for p in sources04b)
              and 'task04b.py' not in {p.name for p in sources03d})
        detail = {'class_04b': resolved04b.__name__, 'class_03d': resolved03d.__name__,
                  'sources_03d': [p.name for p in sources03d]}
    except RecursionError as error:
        ok, detail = False, {'error': f'RecursionError: {error}'}
    finally:
        factory03.environment_class, factory03.environment_sources = original
    record('injected_entry_resolves_without_recursion', ok, detail)
    record('factory03_restored_after_injection_check',
           (factory03.environment_class, factory03.environment_sources) == original, {})


def check_course_and_reward_parity(record, episodes=6, seed=101):
    from d1_rl.factory04b import environment_class
    env04b = environment_class(CONFIG_04B)(config=CONFIG_04B, arm='B', seed=seed, rank=0)
    env03d = environment_class(CONFIG_03D)(config=CONFIG_03D, arm='B', seed=seed, rank=0)
    schedules_equal = True
    state_equal = False
    field_mismatch, total_mismatch, samples = [], [], []
    try:
        for index in range(episodes):
            env04b.reset(seed=seed if index == 0 else None)
            env03d.reset(seed=seed if index == 0 else None)
            if env04b._training_schedule != env03d._training_schedule:
                schedules_equal = False
            state_equal = bool(np.allclose(env04b.data.qpos, env03d.data.qpos, atol=0, rtol=0))
            for command in ((0., 0., 0.), (.2, 0., 0.), (0., 0., .2), (-.15, 0., 0.)):
                previous = np.zeros(16, dtype=np.float64)
                r4, t4, m4 = env04b._reward(np.array(command), previous, 0., False, 0)
                r3, t3, _ = env03d._reward(np.array(command), previous, 0., False, 0)
                for field in t3['raw']:
                    if t3['raw'][field] != t4['raw'][field]:
                        field_mismatch.append({'episode': index, 'command': command,
                                               'field': field})
                for field in t3['weights']:
                    if t3['weights'][field] != t4['weights'][field]:
                        field_mismatch.append({'episode': index, 'command': command,
                                               'weight_field': field})
                term = t4['contributions_per_second']['wheel_action_mean_squared']
                samples.append({'episode': index, 'command': list(command),
                                'wheel_term_per_second': term,
                                'wheel_action_mean_squared': m4['wheel_action_mean_squared'],
                                'term_active': m4['wheel_action_term_active'],
                                'net_wheel_target_rad_s': m4['net_wheel_target_rad_s'],
                                'r03d': r3, 'r04b': r4})
                moving = command[0] != 0. or command[2] != 0.
                if moving and (term != 0. or abs(r4-r3) > 1e-15):
                    total_mismatch.append({'episode': index, 'command': command,
                                           'r03d': r3, 'r04b': r4, 'term': term})
        counts04b, counts03d = env04b.counters(), env03d.counters()
    finally:
        env04b.close()
        env03d.close()
    record('command_course_identical_to_03d', schedules_equal, {'episodes': episodes})
    record('reset_state_identical_to_03d', state_equal, {})
    record('inherited_reward_fields_identical', not field_mismatch,
           {'mismatches': field_mismatch[:6]})
    record('reward_identical_to_03d_while_commanded_to_move', not total_mismatch,
           {'mismatches': total_mismatch[:6]})
    record('term_gated_off_while_commanded_to_move',
           all(s['term_active'] is False for s in samples if s['command'][0] or s['command'][2]), {})
    record('term_active_at_zero_command',
           all(s['term_active'] is True for s in samples
               if not s['command'][0] and not s['command'][2]), {})
    record('no_native_integration_during_checks',
           counts04b['native_steps'] == 0 and counts03d['native_steps'] == 0,
           {'counts_04b': counts04b, 'counts_03d': counts03d})
    return {'counts_04b': counts04b, 'counts_03d': counts03d, 'reward_samples': samples[:8]}


def check_forkserver_round_trip(record):
    from d1_rl.factory04b import environment_class
    from d1_rl.task04b import D1Task04bEnv
    import cloudpickle

    def build():
        return environment_class(CONFIG_04B)

    restored = cloudpickle.loads(cloudpickle.dumps(build))
    record('cloudpickle_closure_resolves_04b_task', restored() is D1Task04bEnv, {})

    context = mp.get_context('forkserver')
    context.set_forkserver_preload(['d1_rl.forkserver_bootstrap'])
    queue = context.Queue()
    worker = context.Process(target=_worker_resolves_class, args=(queue, str(CONFIG_04B)))
    worker.start()
    status, value, pid = queue.get(timeout=120)
    worker.join(timeout=30)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
    record('forkserver_worker_resolves_04b_task',
           status == 'ok' and value == 'D1Task04bEnv',
           {'status': status, 'class_or_error': value, 'worker_pid': pid,
            'worker_exitcode': worker.exitcode})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    check_protection(record)
    config04b = check_config(record)
    term_table = check_term_shape(record, config04b)
    source_sha = check_factory(record)
    check_injected_entry(record)
    parity = check_course_and_reward_parity(record)
    check_forkserver_round_trip(record)

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract04b-static-checks-v1',
              'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0,
              'transitions_consumed': 0, 'training_started': False,
              'note': ('Static acceptance only. Passing here authorises nothing about control '
                       'behaviour; the 04b hypothesis is decided by the fixed evaluation.'),
              'tool_sha256': sha(__file__),
              'config_04b_sha256': sha(CONFIG_04B),
              'environment_sources_sha256': source_sha,
              'wheel_action_penalty_table': term_table,
              'reward_parity': parity,
              'failed_checks': failed, 'checks': checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'failed_checks': failed,
                      'checks': len(checks), 'native_integration_steps_executed': 0,
                      'output': str(args.output/'report.json')}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
