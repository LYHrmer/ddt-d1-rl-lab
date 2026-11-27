"""Bound native references and prospective high-speed D1 protocol."""
from tools91.launch91 import ROOT, digest, read, require
from tools112.surface import SEEDS, bindings as surface_bindings


BASE = ROOT / "evidence/speed_world113"
PUBLIC = ROOT / "reports/speed_world113"
PLAN = ROOT / "docs/experiments113/high_speed.json"
METHOD = ROOT / "docs/high_speed_world113_proposal.md"
GRID_MANIFEST = ROOT / "reports/surface112/results.json"
COMPOUND_MANIFEST = ROOT / "reports/lidar_compound111/native.json"


def registration():
    contract = read(PLAN)
    for name, sha in contract["sources_sha256"].items():
        require(digest(ROOT / name) == sha, "registered source changed: " + name)
    require(read(GRID_MANIFEST)["sources"] == surface_bindings(), "round 112 sources changed")
    require(contract["seeds"] == list(SEEDS), "grid roster changed")
    return digest(PLAN)


def grid_case(seed):
    case, = [r for r in read(GRID_MANIFEST)["cases"] if r["seed"] == seed]
    row = case["episode"]
    require(case["case_status"] == row["status"] == "COMPLETE" and row["motion"]["ticks"] == 600
            and case["resources"]["owned_cgroup_children_empty"], "incomplete grid baseline")
    path = GRID_MANIFEST.parent / case["file"]
    require(digest(path) == row["trace_sha256"], "grid reference digest mismatch")
    return path, row


def roster():
    rows = []
    for case in read(COMPOUND_MANIFEST)["cases"]:
        if case["layout"] != "development" or case["case_status"] != "COMPLETE" or case["unit_status"] != "COMPLETE":
            continue
        row = case["episode"]
        path = COMPOUND_MANIFEST.parent / case["file"]
        require(row["status"] == "COMPLETE" and row["motion"]["ticks"] == 600
                and digest(path) == row["trace_sha256"], "incomplete compound source")
        rows.append((int(case["seed"]), "compound", path))
    require(len(rows) == 7 and {r[0] for r in rows} == {111103,111107}, "changed compound training roster")
    return rows + [(seed,"grid",grid_case(seed)[0]) for seed in SEEDS]
