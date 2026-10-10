#!/usr/bin/env python3
"""Run a frozen 03 entry point with a chosen contract-04 task factory injected.

Generalises tools04/train_d1_ppo04.py and tools04/evaluate_d1_ppo04.py, whose
SHA-256 values are already recorded in 04a's release receipt and therefore must
not change. Future contracts use this runner instead of adding another copy.

The frozen script is executed unmodified, as __main__, via runpy. Running it as
__main__ matters: SubprocVecEnv uses the forkserver start method and cloudpickle
serialises a __main__ closure by value while resolving module-level functions by
reference, so the workers import the injected factory and build the intended
task. tools04/check_task04*_static.py asserts that end to end with no physics.

Usage:
  run_frozen_entry.py --factory d1_rl.factory04b --script tools/train_d1_ppo.py \
      -- --output ... --config ... --arm B --seed 101 --envs 8 ...
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
RELEASE_03D = ROOT/'evidence/career_contract_03/pilot03d_release_01.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--factory', required=True,
                        help='module exposing environment_class/environment_sources, '
                             'e.g. d1_rl.factory04b')
    parser.add_argument('--script', required=True, type=Path,
                        help='repo-relative path of the frozen entry point to execute')
    parser.add_argument('forwarded', nargs=argparse.REMAINDER,
                        help='-- followed by the frozen entry point arguments')
    args = parser.parse_args()

    script = (ROOT/args.script).resolve()
    if not script.is_file():
        raise SystemExit(f'Frozen entry point missing: {script}')
    relative = str(script.relative_to(ROOT))
    frozen = json.loads(RELEASE_03D.read_text())['inputs_sha256']
    if relative not in frozen:
        raise SystemExit(f'{relative} is not a frozen 03d input; refusing to inject into it')
    actual = hashlib.sha256(script.read_bytes()).hexdigest()
    if actual != frozen[relative]:
        raise SystemExit(f'{relative} SHA256 changed: {actual} != {frozen[relative]}')

    forwarded = args.forwarded[1:] if args.forwarded[:1] == ['--'] else args.forwarded
    if not forwarded:
        raise SystemExit('No arguments were forwarded to the frozen entry point')

    # Run directly, sys.path[0] is the script's directory, which is how the
    # frozen evaluator resolves `from export_d1_ppo import check_checkpoint`.
    sys.path.insert(0, str(script.parent))
    sys.path.insert(0, str(ROOT/'training'))
    import d1_rl.factory03 as factory03
    factory = importlib.import_module(args.factory)
    factory03.environment_class = factory.environment_class
    factory03.environment_sources = factory.environment_sources

    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
