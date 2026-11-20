"""Registered inputs and file identities for the fixed-action diagnostic."""
from tools91.launch91 import ROOT, digest, read, require, write

BASE = ROOT / "evidence/terrain_replay92"
PLAN = ROOT / "docs/experiments92/fixed_actions.json"
PLAN_SHA = "b40fe92e8ec9bea5c44416c2ea63a666d307800fa478ffaeca84037a662509eb"
# Prefix reader used by both the archived 91 cut and the 92 input exporter.
DATA91_SHA = "e1e39bdf8d7dc14c5389434dffee662d672a4befdbd82ddd10341903b7a91fdc"


def checked_plan():
    require(digest(PLAN) == PLAN_SHA, "fixed-action protocol identity differs")
    require(digest(ROOT / "tools91/data91.py") == DATA91_SHA, "archived row reader changed")
    plan = read(PLAN)
    for relative, expected in plan["source_sha256"].items():
        path = ROOT / relative
        require(not path.is_symlink() and digest(path) == expected, f"registered input changed: {relative}")
    return plan
