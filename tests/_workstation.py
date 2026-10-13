"""Shared marker for tests that read receipts which only exist on the workstation.

evidence/ is gitignored on purpose: it holds weights, step traces and receipts
too large or too local to publish. A test that reads one must therefore run where
the receipt exists and skip, saying why, where it does not. The criterion is
whether the receipt is present, never whether this is CI, so on the workstation
these tests still execute and still guard the real numbers.

Every such test uses this one marker so that a future test cannot forget the
guard, which is how a missing one first reached a clean clone unnoticed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'
# The three historical release receipts and the contract 04 ledger inputs.
SENTINELS = (EVIDENCE/'career_contract_03/pilot03d_release_01.json',
             EVIDENCE/'career_contract_04/pilot04a_release_02.json',
             EVIDENCE/'career_contract_04/pilot04b_release_01.json')

needs_workstation_evidence = pytest.mark.skipif(
    not all(path.is_file() for path in SENTINELS),
    reason='real receipts live on the workstation, not in the repository')
