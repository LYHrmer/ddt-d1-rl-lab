#!/usr/bin/env python3
"""Zero-integration acceptance checks for contract 04a, run before any training.

Nothing here calls mj_step. Environments are constructed and reset, which uses
mj_resetData/mj_forward only, so the native integration counter must stay at 0
and no budget pool is charged. The checks exist to prove, before spending the
single 04a training run, that:

  * every frozen 03 input and protected file is still bit-identical;
  * pilot04a differs from pilot03d by exactly the one added reward term;
  * the 03d command course and every inherited reward field are untouched;
  * the anchor term's shape, segment detection and reset behaviour are correct;
  * the forkserver/cloudpickle path really hands D1Task04aEnv to the workers,
    which is the one way the injected trainer entry could silently train 03d.
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

CONFIG_04A = ROOT/'training/configs/pilot04a.json'
CONFIG_03D = ROOT/'training/configs/pilot03d.json'
EXPECTED_NEW_REWARD_KEYS = ('standstill_anchor_displacement_weight',
                            'standstill_anchor_displacement_knee_m',
                            'standstill_anchor_displacement_saturation_m')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _worker_resolves_class(queue, config_path):
    """Executed in a forkserver child, exactly as SubprocVecEnv would spawn it."""
    try:
        sys.path.insert(0, str(ROOT/'training'))
        from d1_rl.factory04 import environment_class
        queue.put(('ok', environment_class(config_path).__name__, os.getpid()))
    except BaseException as error:  # noqa: BLE001 - the failure identity is the result
        queue.put(('error', f'{type(error).__name__}: {error}', os.getpid()))


def check_protection(record):
    release = json.loads((ROOT/'evidence/career_contract_03/pilot03d_release_01.json').read_text())
    bad = [p for p, h in release['inputs_sha256'].items()
           if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    record('frozen_03d_inputs_unchanged', not bad,
           {'count': len(release['inputs_sha256']), 'mismatched': bad})

    guard = json.loads((ROOT/'evidence/career_contract_03/protected_guard_final03_01.json')
                       .read_text())['guard']
    old = Path('/home/lyh/wheel-legged-control-lab')
    bad77 = [p for p, h in guard['frozen_77_sha256'].items()
             if not (old/p).is_file() or sha(old/p) != h]
    record('protected_77_files_unchanged', not bad77,
           {'count': len(guard['frozen_77_sha256']), 'mismatched': bad77})

    drive = old/'results/my_course_drive_01'
    bad_drive = [p for p, meta in guard['my_course_drive_01'].items()
                 if not (drive/p).is_file() or sha(drive/p) != meta['sha256']]
    record('user_result_drive_unchanged', not bad_drive, {'mismatched': bad_drive})


def check_config(record):
    from d1_rl.task04a import _expected_04a_config, CHANGE_FROM_PILOT03D
    config04a = json.loads(CONFIG_04A.read_text())
    record('pilot04a_matches_expected_derivation', config04a == _expected_04a_config(), {})

    config03d = json.loads(CONFIG_03D.read_text())
    added = set(config04a['reward']) - set(config03d['reward'])
    removed = set(config03d['reward']) - set(config04a['reward'])
    changed = [k for k in config03d['reward']
               if k in config04a['reward'] and config03d['reward'][k] != config04a['reward'][k]]
    record('reward_adds_exactly_the_one_term',
           added == set(EXPECTED_NEW_REWARD_KEYS) and not removed and not changed,
           {'added': sorted(added), 'removed': sorted(removed), 'changed_in_place': changed})

    top_diff = {k for k in set(config04a) | set(config03d)
                if config04a.get(k) != config03d.get(k)}
    record('no_other_configuration_change',
           top_diff == {'version', 'reward', 'change_from_pilot03c', 'change_from_pilot03d'},
           {'differing_top_level_keys': sorted(top_diff),
            'version': config04a['version'],
            'change_from_pilot03d': config04a['change_from_pilot03d'] == CHANGE_FROM_PILOT03D})
    return config04a, config03d


def check_anchor_shape(record, config04a):
    from d1_rl.task04a import anchor_penalty_raw
    reward = config04a['reward']
    knee = reward['standstill_anchor_displacement_knee_m']
    cap = reward['standstill_anchor_displacement_saturation_m']
    weight = reward['standstill_anchor_displacement_weight']
    table = []
    for distance in (0., .005, .01, .02, .0195, .05, .0604, .0806, .09, .10, .1466, .5):
        raw = anchor_penalty_raw(distance, knee, cap)
        table.append({'distance_m': distance, 'raw': raw,
                      'contribution_per_second': raw*weight})
    # Continuity and monotonicity of value and slope at the knee.
    eps = 1e-7
    left = (anchor_penalty_raw(knee, knee, cap)-anchor_penalty_raw(knee-eps, knee, cap))/eps
    right = (anchor_penalty_raw(knee+eps, knee, cap)-anchor_penalty_raw(knee, knee, cap))/eps
    record('anchor_value_continuous_at_knee',
           abs(anchor_penalty_raw(knee, knee, cap)-knee) < 1e-12, {'value_at_knee': knee})
    record('anchor_slope_continuous_at_knee', abs(left-right) < 1e-3,
           {'slope_left': left, 'slope_right': right})
    record('anchor_zero_at_zero_distance', anchor_penalty_raw(0., knee, cap) == 0., {})
    record('anchor_saturates', anchor_penalty_raw(10., knee, cap)
           == anchor_penalty_raw(cap, knee, cap), {'raw_cap': anchor_penalty_raw(cap, knee, cap)})
    worst = anchor_penalty_raw(cap, knee, cap)*abs(weight)
    # Standing still while drifting still earns roughly vx+wz+alive per second.
    positive_floor = (reward['vx_tracking_weight']+reward['wz_tracking_weight']
                      + reward['alive_weight'])*.95
    record('worst_case_penalty_below_positive_floor', worst < positive_floor,
           {'worst_penalty_per_second': worst, 'approximate_positive_floor': positive_floor,
            'why': 'a penalty above the floor would make terminating early look better'})
    record('anchor_rejects_bad_parameters',
           _raises(lambda: anchor_penalty_raw(.1, .1, .05)), {})
    return table


def _raises(call):
    try:
        call()
    except ValueError:
        return True
    return False


def check_factory(record):
    from d1_rl.factory04 import environment_class, environment_sources
    from d1_rl.task04a import D1Task04aEnv
    from d1_rl.task03d import D1Task03dEnv
    record('factory04_selects_04a_task',
           environment_class(CONFIG_04A) is D1Task04aEnv, {})
    record('factory04_preserves_03d_selection',
           environment_class(CONFIG_03D) is D1Task03dEnv, {})
    sources = [Path(p) for p in environment_sources(CONFIG_04A)]
    names = [p.name for p in sources]
    record('environment_sources_include_04a_chain',
           {'task04a.py', 'factory04.py', 'task03d.py', 'task03c.py', 'env.py',
            'interface.py', 'pilot03d.json'} <= set(names)
           and all(p.is_file() for p in sources)
           and len(names) == len(set(names)),
           {'sources': names, 'unique_basenames': len(names) == len(set(names))})
    return {str(p.relative_to(ROOT)): sha(p) for p in sources}


def check_course_and_reward_parity(record, episodes=6, seed=101):
    """03d and 04a must agree on the command course and on every inherited term."""
    from d1_rl.factory04 import environment_class
    env04a = environment_class(CONFIG_04A)(config=CONFIG_04A, arm='B', seed=seed, rank=0)
    env03d = environment_class(CONFIG_03D)(config=CONFIG_03D, arm='B', seed=seed, rank=0)
    schedules_equal = True
    state_equal = False
    reward_field_mismatch = []
    reward_total_mismatch = []
    anchor_values = []
    try:
        for index in range(episodes):
            env04a.reset(seed=seed if index == 0 else None)
            env03d.reset(seed=seed if index == 0 else None)
            if env04a._training_schedule != env03d._training_schedule:
                schedules_equal = False
            # Identical seeds and zero integration mean identical states here.
            state_equal = bool(np.allclose(env04a.data.qpos, env03d.data.qpos, atol=0, rtol=0))
            for command in ((0., 0., 0.), (.2, 0., 0.), (0., 0., .2), (-.15, 0., 0.)):
                previous = np.zeros(16, dtype=np.float64)
                r4, t4, _ = env04a._reward(np.array(command), previous, 0., False, 0)
                r3, t3, _ = env03d._reward(np.array(command), previous, 0., False, 0)
                for field in t3['raw']:
                    if t3['raw'][field] != t4['raw'][field]:
                        reward_field_mismatch.append({'episode': index, 'command': command,
                                                      'field': field, 'r03d': t3['raw'][field],
                                                      'r04a': t4['raw'][field]})
                for field in t3['weights']:
                    if t3['weights'][field] != t4['weights'][field]:
                        reward_field_mismatch.append({'episode': index, 'command': command,
                                                      'weight_field': field})
                anchor = t4['contributions_per_second']['standstill_anchor_displacement']
                anchor_values.append({'episode': index, 'command': list(command),
                                      'anchor_contribution_per_second': anchor,
                                      'r03d': r3, 'r04a': r4})
                moving = command[0] != 0. or command[2] != 0.
                if moving and (anchor != 0. or abs(r4-r3) > 1e-15):
                    reward_total_mismatch.append({'episode': index, 'command': command,
                                                  'r03d': r3, 'r04a': r4, 'anchor': anchor})
        counts04a = env04a.counters()
        counts03d = env03d.counters()
    finally:
        env04a.close()
        env03d.close()
    record('command_course_identical_to_03d', schedules_equal, {'episodes': episodes})
    record('reset_state_identical_to_03d', state_equal, {})
    record('inherited_reward_fields_identical', not reward_field_mismatch,
           {'mismatches': reward_field_mismatch[:6]})
    record('reward_identical_to_03d_while_commanded_to_move', not reward_total_mismatch,
           {'mismatches': reward_total_mismatch[:6]})
    record('no_native_integration_during_checks',
           counts04a['native_steps'] == 0 and counts03d['native_steps'] == 0,
           {'counts_04a': counts04a, 'counts_03d': counts03d})
    return {'counts_04a': counts04a, 'counts_03d': counts03d,
            'reward_samples': anchor_values[:8]}


def check_anchor_segments(record, seed=101):
    """The anchor must latch on the first zero interval and clear when moving."""
    from d1_rl.factory04 import environment_class
    env = environment_class(CONFIG_04A)(config=CONFIG_04A, arm='B', seed=seed, rank=0)
    observed = []
    try:
        env.reset(seed=seed)
        sequence = [(0., 0., 0.)]*3 + [(.2, 0., 0.)]*2 + [(0., 0., 0.)]*3 + [(0., 0., .2)]
        for command in sequence:
            _, _, metrics = env._reward(np.array(command), np.zeros(16), 0., False, 0)
            observed.append({'command': list(command),
                             'anchor_active': metrics['standstill_anchor_active'],
                             'distance_m': metrics['standstill_anchor_distance_m']})
        counts = env.counters()
    finally:
        env.close()
    expected_active = [True, True, True, False, False, True, True, True, False]
    record('anchor_latches_and_clears_with_the_command',
           [o['anchor_active'] for o in observed] == expected_active, {'observed': observed})
    record('anchor_distance_zero_when_inactive',
           all(o['distance_m'] == 0. for o in observed if not o['anchor_active']), {})
    record('segment_checks_integrate_nothing', counts['native_steps'] == 0, {'counts': counts})
    return observed


def check_forkserver_round_trip(record):
    """The one way the injected entry could silently train the 03d task."""
    from d1_rl.factory04 import environment_class
    from d1_rl.task04a import D1Task04aEnv
    import cloudpickle

    def build():
        return environment_class(CONFIG_04A)

    restored = cloudpickle.loads(cloudpickle.dumps(build))
    record('cloudpickle_closure_resolves_04a_task', restored() is D1Task04aEnv, {})

    context = mp.get_context('forkserver')
    context.set_forkserver_preload(['d1_rl.forkserver_bootstrap'])
    queue = context.Queue()
    worker = context.Process(target=_worker_resolves_class, args=(queue, str(CONFIG_04A)))
    worker.start()
    status, value, pid = queue.get(timeout=120)
    worker.join(timeout=30)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
    record('forkserver_worker_resolves_04a_task',
           status == 'ok' and value == 'D1Task04aEnv',
           {'status': status, 'class_or_error': value, 'worker_pid': pid,
            'worker_exitcode': worker.exitcode})


def check_injected_entry(record):
    """Exercise the patched state the trainer actually runs under.

    The first 04a launch attempt crashed here: factory04 looked the 03 fallback
    up through factory03, which the entry had already rebound to factory04, so
    environment_sources recursed until the stack ran out. Checking only the
    unpatched module missed it, so the patch is applied here on purpose.
    """
    import d1_rl.factory03 as factory03
    import d1_rl.factory04 as factory04
    from d1_rl.task04a import D1Task04aEnv
    from d1_rl.task03d import D1Task03dEnv
    original = (factory03.environment_class, factory03.environment_sources)
    try:
        factory03.environment_class = factory04.environment_class
        factory03.environment_sources = factory04.environment_sources
        resolved04a = factory03.environment_class(CONFIG_04A)
        resolved03d = factory03.environment_class(CONFIG_03D)
        sources04a = [Path(p) for p in factory03.environment_sources(CONFIG_04A)]
        sources03d = [Path(p) for p in factory03.environment_sources(CONFIG_03D)]
        ok = (resolved04a is D1Task04aEnv and resolved03d is D1Task03dEnv
              and all(p.is_file() for p in sources04a)
              and {'task04a.py', 'factory04.py'} <= {p.name for p in sources04a}
              and 'task04a.py' not in {p.name for p in sources03d})
        detail = {'class_04a': resolved04a.__name__, 'class_03d': resolved03d.__name__,
                  'sources_04a': [p.name for p in sources04a],
                  'sources_03d': [p.name for p in sources03d]}
    except RecursionError as error:
        ok, detail = False, {'error': f'RecursionError: {error}'}
    finally:
        factory03.environment_class, factory03.environment_sources = original
    record('injected_entry_resolves_without_recursion', ok, detail)
    record('factory03_restored_after_injection_check',
           (factory03.environment_class, factory03.environment_sources) == original, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    check_protection(record)
    config04a, _ = check_config(record)
    anchor_table = check_anchor_shape(record, config04a)
    source_sha = check_factory(record)
    check_injected_entry(record)
    parity = check_course_and_reward_parity(record)
    segments = check_anchor_segments(record)
    check_forkserver_round_trip(record)

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract04a-static-checks-v1',
              'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0,
              'transitions_consumed': 0,
              'training_started': False,
              'note': ('Static acceptance only. Passing here authorises nothing about control '
                       'behaviour; the 04a hypothesis is decided by the fixed evaluation.'),
              'tool_sha256': sha(__file__),
              'config_04a_sha256': sha(CONFIG_04A),
              'environment_sources_sha256': source_sha,
              'anchor_penalty_table': anchor_table,
              'anchor_segment_trace': segments,
              'reward_parity': parity,
              'failed_checks': failed,
              'checks': checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'failed_checks': failed,
                      'checks': len(checks),
                      'native_integration_steps_executed': 0,
                      'output': str(args.output/'report.json')}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
