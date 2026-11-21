"""Immutable registration, stage chain and resource receipts for the endpoint ablation."""
import json

from tools91.launch91 import require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, SEEDS, digest, read
from tools95.run95 import files_for_seal
from tools95.score95 import ARMS, HORIZON
from tools96.common96 import BASE as R96

BASE = ROOT / "evidence/endpoint98"
PLAN = ROOT / "docs/experiments98/endpoint.json"
PLAN_SHA = "4d29b265fa72b5daf4cd63329e91f552f8094cc2016d5c0ceefa6aa1475922ae"
STAGES = ("fit", "predictions", "native", "video")
MODELS = ("control", "endpoint")
RESETS = (98201, 98202, 98203)


def protocol(private=False):
    require(digest(PLAN) == PLAN_SHA, "endpoint training protocol changed")
    plan = read(PLAN)
    require(tuple(plan["training_seeds"]) == SEEDS and tuple(plan["blind_resets"]) == RESETS
            and tuple(plan["arms"]) == MODELS and plan["validation_multipliers"] == ARMS
            and plan["horizon_ticks"] == HORIZON, "registered training or blind roster differs")
    sources = dict(plan["parents_sha256"])
    if private:
        sources.update(plan["local_parent_seals_sha256"])
    for path, expected in sources.items():
        require(digest(ROOT / path) == expected, f"registered parent differs: {path}")
    return plan


def unit_name(stage):
    require(stage in STAGES, "unknown endpoint experiment stage")
    return f"d1-endpoint98-{stage}-01"


def sources_for(stage):
    require(stage in STAGES, "unknown producer")
    paths = ["tools98/common98.py", "tools98/launch98.py"]
    if stage == "video":
        paths += ["tools98/video98.py", "tools96/video96.py", "tools95/video95.py", "tools94/video94.py"]
    else:
        paths += ["tools98/run98.py", "tools98/fit98.py", "tools96/fit96.py", "tools96/common96.py",
                  "tools94/learn94.py", "tools91/model91.py", "tools52/dynamics52.py"]
        if stage != "fit":
            paths += ["tools96/run96.py", "tools95/model95.py", "tools95/run95.py",
                      "tools94/native94.py", "tools95/score95.py", "tools98/score98.py"]
    return {path: digest(ROOT / path) for path in paths}


def previous_seal(stage):
    if stage == "fit":
        return R96 / "fit_01/seal.json"
    return BASE / f"{STAGES[STAGES.index(stage)-1]}_01/seal.json"


def checked_stage(stage):
    folder = BASE / f"{stage}_01"
    seal = read(folder / "seal.json")
    require(seal["stage"] == stage and seal["protocol_sha256"] == PLAN_SHA and seal["sources"] == sources_for(stage)
            and seal["previous_seal_sha256"] == digest(previous_seal(stage)) and seal["files"] == files_for_seal(folder),
            f"unsealed or changed {stage} evidence")
    return folder


def resource_record(stage):
    folder = BASE / f"guard_{stage}_01"
    receipt, log = folder / "report.json", folder / "resources.jsonl"
    row = read(receipt)
    limits = protocol()["resources"]
    scope = scope_for(unit_name(stage))
    require(row["status"] == "COMPLETE" and row["scope"] == scope and row["child_started"]
            and row["owned_cgroup_children_empty"] and row["returncode"] == 0
            and len(row["cpu_affinity"]) == limits["cpu_affinity_count"]
            and row["supervisor_sha256"] == limits["supervisor_sha256"], f"incomplete {stage} supervisor")
    samples = [json.loads(line) for line in log.read_text().splitlines()]
    entered = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    running = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    require(entered and running and entered[-1] >= limits["admission_min_bytes"]
            and min(running) >= limits["runtime_min_bytes"], f"{stage} memory reserve differs")
    for sample in samples:
        group = sample["owned"]
        require(group["path"] == scope and group["high_bytes"] == limits["memory_high_bytes"]
                and group["max_bytes"] == limits["memory_max_bytes"] and group["swap_max_bytes"] == 0
                and group["swap_current_bytes"] == 0, f"{stage} cgroup limits differ")
    fields = ("status", "scope", "cpu_affinity", "peak_group_bytes", "elapsed_s", "supervisor_sha256",
              "returncode", "child_started", "owned_cgroup_children_empty", "final_memory_events")
    return {**{k: row[k] for k in fields}, "limits_verified": True, "startup_effective_bytes": entered[-1],
            "runtime_min_effective_bytes": min(running), "receipt_sha256": digest(receipt), "resources_sha256": digest(log)}
