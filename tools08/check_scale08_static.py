#!/usr/bin/env python3
"""Zero-integration acceptance for contract 08, before any training step is spent.

Checks that the new entry is the frozen trainer with exactly three lines changed, that
every input the frozen 03d training recorded is byte-identical today, that the library
versions are the ones 03d was trained with, that the new entry refuses out-of-contract
arguments before it creates anything, and that the launcher's plan matches the contract.
Builds no environment and executes no physics.
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

import run_scale08 as launcher  # noqa: E402

FROZEN_TRAINER = ROOT/'tools/train_d1_ppo.py'
NEW_TRAINER = ROOT/launcher.ENTRY
PROTOCOL_03D = ROOT/'evidence/career_contract_03/pilot03d_01/protocol.json'
EXPECTED_CHANGES = {
    123: ('    p.add_argument("--kind",choices=["preflight","pilot","formal"],required=True)',
          '    p.add_argument("--kind",choices=["preflight","pilot","formal","scale"],required=True)'),
    127: ('    limits={"preflight":(32768,600),"pilot":(1048576,5400),"formal":(4194304,10800)}',
          '    limits={"preflight":(32768,600),"pilot":(1048576,5400),"formal":(4194304,10800),"scale":(8388608,9000)}'),
    134: ('    if a.kind=="pilot" and a.seed!=101: p.error("Development pilot seed is 101")',
          '    if a.kind in ("pilot","scale") and a.seed!=101: p.error("Development pilot seed is 101")'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def line_changes(old_text, new_text):
    """0-based index -> (old, new) for every line that differs; any change in length is a failure."""
    old, new = old_text.split('\n'), new_text.split('\n')
    if len(old) != len(new):
        return None
    return {i: (a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b}


def refuses(arguments, python=sys.executable):
    """Run the new entry with out-of-contract arguments: it must exit 2 and create nothing."""
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch)/'must_not_exist'
        completed = subprocess.run([python, '-B', str(NEW_TRAINER), '--output', str(output), *arguments],
                                   capture_output=True, text=True, cwd=ROOT, env=launcher.trainer_env())
        return {'returncode': completed.returncode, 'output_created': output.exists(),
                'stderr_tail': completed.stderr.strip().splitlines()[-1:] if completed.stderr else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='must not already exist')
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({'check': name, 'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    changes = line_changes(FROZEN_TRAINER.read_text(), NEW_TRAINER.read_text())
    record('new_entry_is_the_frozen_trainer_with_exactly_three_lines_changed',
           changes == EXPECTED_CHANGES,
           {'changed_lines_1_based': None if changes is None else [i+1 for i in sorted(changes)]})

    protocol = json.loads(PROTOCOL_03D.read_text())
    drift = {path.replace(str(ROOT)+'/', ''): (expected[:12], sha(path)[:12] if Path(path).is_file() else None)
             for path, expected in protocol['inputs_sha256'].items()
             if not Path(path).is_file() or sha(path) != expected}
    record('every_input_of_the_frozen_03d_training_is_unchanged', not drift,
           {'inputs': len(protocol['inputs_sha256']), 'changed': drift})

    import gymnasium
    import mujoco
    import stable_baselines3
    import torch
    now = {'mujoco': mujoco.__version__, 'gymnasium': gymnasium.__version__,
           'sb3': stable_baselines3.__version__, 'torch': torch.__version__}
    record('library_versions_equal_those_03d_was_trained_with', now == protocol['versions'],
           {'now': now, 'at_03d_training': protocol['versions']})

    record('03d_training_used_the_same_seed_envs_and_hyperparameters_the_launcher_fixes',
           protocol['seed'] == launcher.SEED and protocol['envs'] == launcher.ENVS
           and protocol['config'] == json.loads((ROOT/launcher.CONFIG).read_text()),
           {'seed': protocol['seed'], 'envs': protocol['envs']})

    frozen = launcher.read_frozen(ROOT/launcher.FROZEN_LEDGER, launcher.PLANS['scale']['compare_rollouts'])
    record('frozen_ledger_holds_the_256_rollouts_to_compare',
           len(frozen) == 256 and all(r['rollout'] == i+1 for i, r in enumerate(frozen)), {'rows': len(frozen)})

    base = ['--config', str(ROOT/launcher.CONFIG), '--arm', 'B', '--envs', '8', '--max-wall', '9000']
    cases = {
        'steps_above_the_scale_cap': refuses([*base, '--seed', '101', '--steps', str(8388608+4096), '--kind', 'scale']),
        'scale_with_another_seed': refuses([*base, '--seed', '7', '--steps', '8388608', '--kind', 'scale']),
        'wall_above_the_scale_cap': refuses(['--config', str(ROOT/launcher.CONFIG), '--arm', 'B', '--envs', '8',
                                             '--max-wall', '9001', '--seed', '101', '--steps', '8388608', '--kind', 'scale']),
        'pilot_cap_unchanged': refuses([*base[:-2], '--max-wall', '5400', '--seed', '101', '--steps', '2097152', '--kind', 'pilot']),
    }
    record('new_entry_refuses_out_of_contract_arguments_before_creating_anything',
           all(c['returncode'] == 2 and not c['output_created'] for c in cases.values()), cases)

    plan = launcher.PLANS
    record('launcher_plan_matches_the_contract',
           plan['scale']['steps'] == 8388608 and plan['scale']['max_wall'] == 9000
           and plan['preflight']['steps'] == 32768 and plan['preflight']['compare_rollouts'] == 8
           and plan['scale']['compare_rollouts'] == 256 and launcher.RELATIVE_TOLERANCE == 1e-9,
           {k: dict(v) for k, v in plan.items()})

    failed = [c['check'] for c in checks if c['status'] == 'FAIL']
    report = {'format': 'd1-contract08-static-checks-v1', 'status': 'PASS' if not failed else 'FAIL',
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_integration_steps_executed': 0, 'transitions_consumed': 0, 'training_started': False,
              'tool_sha256': sha(__file__), 'new_entry_sha256': sha(NEW_TRAINER),
              'launcher_sha256': sha(launcher.__file__), 'failed_checks': failed, 'checks': checks}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'failed_checks': failed, 'checks': len(checks),
                      'native_integration_steps_executed': 0}, indent=2))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
