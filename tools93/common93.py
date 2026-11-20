"""Pinned inputs and identities for the crossed D1 standing panel."""
from tools91.launch91 import ROOT, digest, read, require, write

BASE = ROOT / "evidence/stand_data93"
PLAN = ROOT / "docs/experiments93/stand_data.json"
PLAN_SHA = "2412b8aa31e78ebb918f30a1890c6319e916d184127470ca063f25ed88964e8d"
PARENTS = (87011, 87023, 87037)
RESETS = (90101, 90102, 90103)
MODELS = (91011, 91023, 91037)
ARMS = ("original", "common")


def checked_plan():
    require(digest(PLAN) == PLAN_SHA, "registered 93 protocol differs")
    plan = read(PLAN)
    for relative, expected in plan["source_sha256"].items():
        path = ROOT / relative
        require(not path.is_symlink() and digest(path) == expected, f"registered source differs: {relative}")
    return plan
