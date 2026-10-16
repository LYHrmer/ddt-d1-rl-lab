#!/usr/bin/env python3
"""Contract 09 launcher: contract 08's launcher with a different entry, plan and comparison length.

tools08/run_scale08.py is frozen into the contract 08 release and is not edited. Its
run() reads the entry point, plans and frozen ledger from module globals at call time,
so this wrapper points those at contract 09 and calls it. Only the first rollout is
compared with the frozen 03d run: the entropy coefficient changes the loss from the
first parameter update on, so rollout 1, collected by the untouched initial policy, is
the only one that must agree.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools08'))

import run_scale08 as base  # noqa: E402

ENTRY = 'tools09/train_d1_ppo09.py'
PLANS = {
    'preflight': {'steps': 32768, 'max_wall': 600, 'compare_rollouts': 1,
                  'output': 'evidence/career_contract_09/scale09_preflight_01'},
    'scale': {'steps': 8388608, 'max_wall': 9000, 'compare_rollouts': 1,
              'output': 'evidence/career_contract_09/scale09_train_01'},
}


def configure(module=base):
    """Point the contract 08 launcher at contract 09; returns what was replaced, for tests."""
    previous = {'ENTRY': module.ENTRY, 'PLANS': module.PLANS}
    module.ENTRY, module.PLANS = ENTRY, PLANS
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=sorted(PLANS))
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--preflight-receipt', type=Path, help='required for the scale run')
    args = parser.parse_args()
    configure()
    return base.run(args.kind, args.release, args.preflight_receipt)


if __name__ == '__main__':
    raise SystemExit(main())
