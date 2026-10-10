#!/usr/bin/env python3
"""Asset-free subset of the contract 04 static checks, runnable in CI.

tools04/check_task04a_static.py and check_task04b_static.py construct the
MuJoCo environment, which needs the D1 robot model that this repository does
not redistribute. This script checks everything that does NOT need the robot
model, so the bounded-experiment discipline is verifiable from a clean clone:

  * each published config is exactly the declared derivation of the frozen
    pilot03d base, and differs from it by exactly the one declared reward term;
  * the frozen base SHA-256 pinned inside each task module matches the file;
  * the task factories resolve each config to the intended class and keep the
    earlier versions intact, and the recorded source manifest is complete;
  * the added reward terms have the shape, continuity, gradient range and
    bounded worst case their contracts claim;
  * the two contract-04 versions are siblings, not a chain: 04b does not
    inherit 04a's term.

No MuJoCo model is loaded, no environment is constructed, no physics runs.
Exit code 0 means every check passed.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))

BASE = ROOT/'training/configs/pilot03d.json'
# Each contract's single change, as its own contract declares it.
VERSIONS = {
    '04a': {'config': ROOT/'training/configs/pilot04a.json',
            'version_string': 'd1-pure-ppo-pilot-04a',
            'module': 'd1_rl.task04a', 'factory': 'd1_rl.factory04',
            'class': 'D1Task04aEnv', 'derive': '_expected_04a_config',
            'reward_keys': ('standstill_anchor_displacement_weight',
                            'standstill_anchor_displacement_knee_m',
                            'standstill_anchor_displacement_saturation_m')},
    '04b': {'config': ROOT/'training/configs/pilot04b.json',
            'version_string': 'd1-pure-ppo-pilot-04b',
            'module': 'd1_rl.task04b', 'factory': 'd1_rl.factory04b',
            'class': 'D1Task04bEnv', 'derive': '_expected_04b_config',
            'reward_keys': ('wheel_action_mean_squared_weight',
                            'wheel_action_mean_squared_saturation')},
}
# Per-tick mean squares measured in the 03d and 04a native traces, plus the
# approximate value at initialisation with log_std_init=-1. The 04b term must
# keep a live gradient across all of them.
MEASURED_WHEEL_MEAN_SQUARE = {'03d_forward_11_to_12s': .06516,
                              '03d_reverse_11_to_12s': .05481,
                              '04a_turn_11_to_12s': .17363,
                              'approx_initialisation': .135}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path,
                        help='optional receipt directory; must not already exist')
    args = parser.parse_args()
    checks = []

    def check(name, ok, detail=None):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL',
                       'detail': detail or {}})

    import importlib
    base_config = json.loads(BASE.read_text())
    check('frozen_base_config_present', BASE.is_file(), {'sha256': sha(BASE)})

    for label, spec in VERSIONS.items():
        task = importlib.import_module(spec['module'])
        factory = importlib.import_module(spec['factory'])
        config = json.loads(spec['config'].read_text())

        check(f'{label}_pinned_base_sha_matches_file',
              task.BASE_CONFIG_03D_SHA256 == sha(BASE),
              {'pinned': task.BASE_CONFIG_03D_SHA256})
        check(f'{label}_config_is_the_declared_derivation',
              config == getattr(task, spec['derive'])())
        check(f'{label}_version_string', config['version'] == spec['version_string'],
              {'version': config['version']})

        added = set(config['reward']) - set(base_config['reward'])
        removed = set(base_config['reward']) - set(config['reward'])
        changed = [k for k in base_config['reward']
                   if k in config['reward'] and base_config['reward'][k] != config['reward'][k]]
        check(f'{label}_adds_exactly_the_one_declared_term',
              added == set(spec['reward_keys']) and not removed and not changed,
              {'added': sorted(added), 'removed': sorted(removed), 'changed': changed})

        top_diff = {k for k in set(config) | set(base_config)
                    if config.get(k) != base_config.get(k)}
        check(f'{label}_no_other_configuration_change',
              top_diff == {'version', 'reward', 'change_from_pilot03c', 'change_from_pilot03d'},
              {'differing_top_level_keys': sorted(top_diff)})

        resolved = factory.environment_class(spec['config'])
        check(f'{label}_factory_resolves_intended_class',
              resolved.__name__ == spec['class'], {'resolved': resolved.__name__})
        check(f'{label}_factory_preserves_03d',
              factory.environment_class(BASE).__name__ == 'D1Task03dEnv')

        sources = [Path(p) for p in factory.environment_sources(spec['config'])]
        names = [p.name for p in sources]
        check(f'{label}_source_manifest_complete',
              all(p.is_file() for p in sources) and len(names) == len(set(names))
              and {'env.py', 'interface.py', 'task03c.py', 'task03d.py',
                   'pilot03d.json'} <= set(names),
              {'sources': names})

    # 04b must be a sibling of 04a on the same base, not its successor.
    config04a = json.loads(VERSIONS['04a']['config'].read_text())
    config04b = json.loads(VERSIONS['04b']['config'].read_text())
    sources04b = {Path(p).name for p in
                  importlib.import_module('d1_rl.factory04b')
                  .environment_sources(VERSIONS['04b']['config'])}
    check('04b_does_not_inherit_04a',
          not any('anchor' in k for k in config04b['reward'])
          and any('anchor' in k for k in config04a['reward'])
          and 'task04a.py' not in sources04b and 'pilot04a.json' not in sources04b)

    # 04a anchor term: continuity at the knee, zero at zero, bounded worst case.
    from d1_rl.task04a import anchor_penalty_raw
    reward04a = config04a['reward']
    knee = reward04a['standstill_anchor_displacement_knee_m']
    cap = reward04a['standstill_anchor_displacement_saturation_m']
    weight04a = reward04a['standstill_anchor_displacement_weight']
    eps = 1e-7
    left = (anchor_penalty_raw(knee, knee, cap)-anchor_penalty_raw(knee-eps, knee, cap))/eps
    right = (anchor_penalty_raw(knee+eps, knee, cap)-anchor_penalty_raw(knee, knee, cap))/eps
    check('04a_anchor_zero_at_zero', anchor_penalty_raw(0., knee, cap) == 0.)
    check('04a_anchor_value_continuous_at_knee',
          abs(anchor_penalty_raw(knee, knee, cap)-knee) < 1e-12)
    check('04a_anchor_slope_continuous_at_knee', abs(left-right) < 1e-3,
          {'slope_left': left, 'slope_right': right})
    check('04a_anchor_saturates',
          anchor_penalty_raw(10., knee, cap) == anchor_penalty_raw(cap, knee, cap))
    check('04a_anchor_monotone',
          all(anchor_penalty_raw(a, knee, cap) < anchor_penalty_raw(b, knee, cap)
              for a, b in zip((0., .01, .03, .06), (.01, .03, .06, .09))))

    # 04b wheel term: unique zero, bounded, and a live gradient over the whole
    # measured operating range. An earlier 0.09 saturation failed this check.
    from d1_rl.task04b import wheel_action_penalty_raw
    reward04b = config04b['reward']
    saturation = reward04b['wheel_action_mean_squared_saturation']
    weight04b = reward04b['wheel_action_mean_squared_weight']
    check('04b_wheel_zero_action_is_free', wheel_action_penalty_raw([0.]*4, saturation) == 0.)
    check('04b_wheel_zero_is_the_unique_minimum',
          all(wheel_action_penalty_raw(a, saturation) > 0. for a in
              ([.01, 0., 0., 0.], [0., -.01, 0., 0.], [.05, -.05, .05, -.05])))
    check('04b_wheel_saturates',
          wheel_action_penalty_raw([1.]*4, saturation) == saturation)
    check('04b_wheel_gradient_live_over_measured_range',
          all(v < saturation for v in MEASURED_WHEEL_MEAN_SQUARE.values()),
          {'measured': MEASURED_WHEEL_MEAN_SQUARE, 'saturation': saturation})
    check('04b_wheel_matches_numpy_reference',
          all(abs(wheel_action_penalty_raw(a, saturation)
                  - min(float(np.mean(np.square(a))), saturation)) < 1e-15
              for a in ([.1, -.2, .3, -.4], [0., 0., 0., 0.], [.5]*4)))

    # Both worst cases must stay below the per-second positive reward floor, or
    # ending an episode early could look cheaper than standing still.
    for label, weight, worst_raw, reward in (('04a', weight04a, cap, reward04a),
                                             ('04b', weight04b, saturation, reward04b)):
        floor = (reward['vx_tracking_weight']+reward['wz_tracking_weight']
                 + reward['alive_weight'])*.95
        worst = worst_raw*abs(weight)
        check(f'{label}_worst_case_penalty_below_positive_floor', worst < floor,
              {'worst_per_second': worst, 'approximate_floor_per_second': floor})
        check(f'{label}_term_is_a_penalty', weight < 0., {'weight': weight})

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract04-ci-static-checks-v1',
              'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': 'asset-free subset; no MuJoCo model loaded, no environment built',
              'native_integration_steps_executed': 0,
              'robot_model_required': False,
              'checks_total': len(checks), 'failed_checks': failed, 'checks': checks,
              'note': ('passing here proves the single-change and term-shape discipline only; '
                       'it says nothing about control behaviour, which needs the robot model '
                       'and the full static checks in tools04/check_task04*_static.py'),
              'tool_sha256': sha(__file__)}
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    width = max(len(c['check']) for c in checks)
    for c in checks:
        print(f"{c['status']:4} {c['check']:{width}}")
    print(f"\n{len(checks)-len(failed)}/{len(checks)} passed, 0 physics steps")
    if failed:
        print('FAILED: '+', '.join(failed))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
