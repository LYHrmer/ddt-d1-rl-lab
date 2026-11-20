"""Registered paths, resource checks and input receipts for the D1 curriculum smoke."""
import os
from pathlib import Path

from tools87.run87 import ROOT, read, sha, write

PLAN = ROOT / "docs/experiments89/uphill_curriculum.json"
BASE = ROOT / "evidence/uphill89"
PARENTS = ROOT / "evidence/full_body87/training_01"


def registered_plan():
    plan = read(PLAN)
    if (sha(ROOT / plan["parent_protocol"]) != plan["parent_protocol_sha256"]
            or sha(PARENTS / "report.json") != plan["parent_training_report_sha256"]):
        raise ValueError("registered parent protocol or completed training receipt differs")
    return plan


def parent_folder(seed, plan):
    name = f"range_{seed}"
    stage = read(PARENTS / "report.json")
    entries = [r for r in stage["models"] if r["model"] == name]
    folder = PARENTS / name
    record = read(folder / "report.json")
    if (stage["status"] != "COMPLETE" or stage["protocol_sha256"] != plan["parent_protocol_sha256"]
            or len(entries) != 1 or sha(folder / "report.json") != entries[0]["report_sha256"]
            or record["status"] != "COMPLETE" or record["seed"] != seed or record["arm"] != "range"
            or sha(folder / "policy.zip") != record["policy_sha256"]
            or record["policy_sha256"] != plan["parent_policy_sha256"][str(seed)]):
        raise ValueError(f"unqualified perception parent for {seed}")
    return folder


def check_worker(stage):
    from tools27.resource_guard27 import read_memory

    scope = Path("/proc/self/cgroup").read_text().strip().split("::", 1)[-1]
    if (not scope.endswith(f"/d1-curriculum89-{stage}.service") or len(os.sched_getaffinity(0)) != 1
            or os.getpriority(os.PRIO_PROCESS, 0) != 10 or read_memory(scope)["effective_available_bytes"] < 4 << 30):
        raise RuntimeError(f"use the admitted single-CPU systemd unit for {stage}")


def complete_stage(stage):
    report = read(BASE / f"{stage}_01" / "report.json")
    guard = read(BASE / f"guard_{stage}_01" / "report.json")
    if (report["status"] != "COMPLETE" or report["protocol_sha256"] != sha(PLAN)
            or guard["status"] != "COMPLETE" or not guard["owned_cgroup_children_empty"]):
        raise ValueError(f"incomplete {stage} evidence")
    return report
