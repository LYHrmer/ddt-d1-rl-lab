#!/usr/bin/env python3
"""Contract 04 evaluation entry: the frozen 03 evaluator with factory04 injected.

Same construction as tools04/train_d1_ppo04.py and for the same reason:
tools/evaluate_d1_ppo.py is a frozen input of the 03d release, so it is executed
unmodified as __main__ via runpy after d1_rl.factory03's two lookup functions
are replaced with the factory04 versions. Every scenario, gate, counter and
receipt therefore comes from the frozen evaluator.

A separate file rather than a shared runner because train_d1_ppo04.py's SHA-256
is already recorded in evidence/career_contract_04/pilot04a_release_02.json.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN_EVALUATOR = ROOT/'tools/evaluate_d1_ppo.py'


def main(argv):
    if not FROZEN_EVALUATOR.is_file():
        raise SystemExit(f'Frozen evaluator missing: {FROZEN_EVALUATOR}')
    release = json.loads((ROOT/'evidence/career_contract_03/pilot03d_release_01.json').read_text())
    expected = release['inputs_sha256']['tools/evaluate_d1_ppo.py']
    actual = hashlib.sha256(FROZEN_EVALUATOR.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Frozen evaluator SHA256 changed: {actual} != {expected}')

    # Run directly, sys.path[0] is tools/, which is how the frozen evaluator
    # resolves `from export_d1_ppo import check_checkpoint`. runpy does not set
    # it, so provide it explicitly.
    sys.path.insert(0, str(ROOT/'tools'))
    sys.path.insert(0, str(ROOT/'training'))
    import d1_rl.factory03 as factory03
    import d1_rl.factory04 as factory04
    factory03.environment_class = factory04.environment_class
    factory03.environment_sources = factory04.environment_sources

    sys.argv = [str(FROZEN_EVALUATOR), *argv]
    runpy.run_path(str(FROZEN_EVALUATOR), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
