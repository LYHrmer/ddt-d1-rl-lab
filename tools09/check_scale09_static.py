#!/usr/bin/env python3
"""Zero-integration acceptance for contract 09, before any training step is spent.

The new entry must be the frozen trainer with exactly five lines changed; the contract 08
launcher it reuses must be byte-identical to the one frozen into the contract 08 release;
every input of the frozen 03d training must be unchanged; the new entry must refuse
out-of-contract arguments before it creates anything. Builds no environment, no physics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))
sys.path.insert(0, str(ROOT/'tools09'))

import check_scale08_static as static08  # noqa: E402
import run_scale08 as base  # noqa: E402
import run_scale09 as launcher  # noqa: E402

FROZEN_TRAINER = ROOT/'tools/train_d1_ppo.py'
NEW_TRAINER = ROOT/launcher.ENTRY
RELEASE08 = ROOT/'evidence/career_contract_08/scale08_release_01.json'
EXPECTED_CHANGES = dict(static08.EXPECTED_CHANGES) | {
    150: ('           "gamma":.99,"gae_lambda":.95,"clip_range":.2,"ent_coef":.005,"vf_coef":.5,',
          '           "gamma":.99,"gae_lambda":.95,"clip_range":.2,"ent_coef":0.0,"vf_coef":.5,'),
    172: ('                  gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=.005,vf_coef=.5,max_grad_norm=.5,',
          '                  gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.0,vf_coef=.5,max_grad_norm=.5,'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuses(arguments, entry=NEW_TRAINER, python=sys.executable):
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch)/'must_not_exist'
        done = subprocess.run([python, '-B', str(entry), '--output', str(output), *arguments], capture_output=True,
                              text=True, cwd=ROOT, env=base.trainer_env())
        return {'returncode': done.returncode, 'output_created': output.exists(),
                'stderr_tail': done.stderr.strip().splitlines()[-1:] if done.stderr else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    changes = static08.line_changes(FROZEN_TRAINER.read_text(), NEW_TRAINER.read_text())
    record('new_entry_is_the_frozen_trainer_with_exactly_five_lines_changed', changes == EXPECTED_CHANGES,
           {'changed_lines_1_based': None if changes is None else [i+1 for i in sorted(changes)]})

    release08 = json.loads(RELEASE08.read_text())
    record('reused_contract_08_launcher_is_the_one_frozen_into_its_release',
           release08['launcher_sha256'] == sha(base.__file__), {'launcher': base.__file__})

    protocol = json.loads(static08.PROTOCOL_03D.read_text())
    drift = [p for p, h in protocol['inputs_sha256'].items() if not Path(p).is_file() or sha(p) != h]
    record('every_input_of_the_frozen_03d_training_is_unchanged', not drift, {'changed': drift})

    import gymnasium
    import mujoco
    import stable_baselines3
    import torch
    now = {'mujoco': mujoco.__version__, 'gymnasium': gymnasium.__version__,
           'sb3': stable_baselines3.__version__, 'torch': torch.__version__}
    record('library_versions_equal_those_03d_was_trained_with', now == protocol['versions'], now)

    previous = launcher.configure()
    try:
        cmd = base.command('scale', Path('/out'), 'python3')
        record('wrapper_points_the_launcher_at_contract_09',
               cmd[2].endswith(launcher.ENTRY) and base.PLANS is launcher.PLANS
               and cmd[cmd.index('--steps')+1] == '8388608' and cmd[cmd.index('--seed')+1] == '101'
               and all(p['compare_rollouts'] == 1 for p in launcher.PLANS.values())
               and all(p['output'].startswith('evidence/career_contract_09/') for p in launcher.PLANS.values()),
               {'command': cmd, 'plans': launcher.PLANS})
    finally:
        base.ENTRY, base.PLANS = previous['ENTRY'], previous['PLANS']

    common = ['--config', str(ROOT/base.CONFIG), '--arm', 'B', '--envs', '8']
    cases = {'steps_above_the_scale_cap': refuses([*common, '--max-wall', '9000', '--seed', '101', '--steps', str(8388608+4096), '--kind', 'scale']),
             'scale_with_another_seed': refuses([*common, '--max-wall', '9000', '--seed', '7', '--steps', '8388608', '--kind', 'scale']),
             'wall_above_the_scale_cap': refuses([*common, '--max-wall', '9001', '--seed', '101', '--steps', '8388608', '--kind', 'scale'])}
    record('new_entry_refuses_out_of_contract_arguments_before_creating_anything',
           all(c['returncode'] == 2 and not c['output_created'] for c in cases.values()), cases)

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract09-static-checks-v1', 'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0, 'transitions_consumed': 0, 'training_started': False,
              'tool_sha256': sha(__file__), 'new_entry_sha256': sha(NEW_TRAINER),
              'wrapper_sha256': sha(launcher.__file__), 'launcher_sha256': sha(base.__file__),
              'failed_checks': failed, 'checks': checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'failed_checks': failed, 'checks': len(checks),
                      'native_integration_steps_executed': 0}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
