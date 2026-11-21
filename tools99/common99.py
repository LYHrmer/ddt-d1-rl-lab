"""Round-99 registration, sealed stage order and measured resource limits."""
import json

from tools91.launch91 import require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, SEEDS, digest, read
from tools95.run95 import files_for_seal
from tools95.score95 import ARMS, HORIZON
from tools98.common98 import BASE as BASE98


BASE = ROOT / "evidence/late_response99"
PLAN = ROOT / "docs/experiments99/late_response.json"
PLAN_SHA = "a85674e74619e8cc04fc614e26131fdfb7ade5dc53daa8e73f8721c39699ef59"
STAGES = ("fit", "predictions", "native", "video")
MODELS = ("control", "late")
RESETS = (99201, 99202, 99203)


def protocol(private=False):
    require(digest(PLAN) == PLAN_SHA, "changed late-response registration")
    plan = read(PLAN)
    require(tuple(plan["training_seeds"]) == SEEDS and tuple(plan["blind_resets"]) == RESETS
            and plan["arms"] == {"control": 0, "late": 1} and plan["validation_multipliers"] == ARMS
            and plan["horizon_ticks"] == HORIZON, "round-99 roster or horizon differs")
    sources = dict(plan["parents_sha256"])
    if private:
        sources.update(plan["local_parent_seals_sha256"])
    for path, expected in sources.items():
        require(digest(ROOT / path) == expected, f"changed registered input: {path}")
    return plan


def unit_name(stage):
    require(stage in STAGES, "unknown round-99 stage")
    return f"d1-late-response99-{stage}-01"


def previous_seal(stage):
    if stage == "fit":
        return BASE98 / "fit_01/seal.json"
    prior = STAGES[STAGES.index(stage) - 1]
    return BASE / f"{prior}_01/seal.json"


def sources_for(stage):
    require(stage in STAGES, "unknown round-99 producer")
    paths = ["tools99/common99.py", "tools99/launch99.py"]
    if stage == "video":
        paths.extend(["tools99/video99.py", "tools98/video98.py", "tools96/video96.py",
                      "tools95/video95.py", "tools94/video94.py"])
    else:
        paths.extend(["tools99/run99.py", "tools99/fit99.py", "tools99/loss99.py", "tools99/data99.py",
                      "tools98/fit98.py", "tools96/fit96.py", "tools96/common96.py",
                      "tools94/learn94.py", "tools91/model91.py", "tools52/dynamics52.py"])
        if stage != "fit":
            paths.extend(["tools96/run96.py", "tools95/model95.py", "tools95/run95.py",
                          "tools94/native94.py", "tools95/score95.py", "tools98/score98.py"])
    return {name: digest(ROOT / name) for name in paths}


def checked_stage(stage):
    folder = BASE / f"{stage}_01"
    seal = read(folder / "seal.json")
    require(seal["stage"] == stage and seal["unit"] == unit_name(stage) and seal["protocol_sha256"] == PLAN_SHA
            and seal["sources"] == sources_for(stage) and seal["previous_seal_sha256"] == digest(previous_seal(stage))
            and seal["files"] == files_for_seal(folder), f"changed or unsealed round-99 {stage}")
    return folder


def resource_record(stage):
    limits, scope = protocol()["resources"], scope_for(unit_name(stage))
    folder = BASE / f"guard_{stage}_01"
    receipt, log = folder / "report.json", folder / "resources.jsonl"
    row = read(receipt)
    require(row["status"] == "COMPLETE" and row["scope"] == scope and row["returncode"] == 0 and row["child_started"]
            and row["owned_cgroup_children_empty"] and len(row["cpu_affinity"]) == limits["cpu_affinity_count"]
            and row["supervisor_sha256"] == limits["supervisor_sha256"], f"incomplete {stage} supervisor")
    samples = [json.loads(line) for line in log.read_text().splitlines()]
    starts = [entry["effective_available_bytes"] for entry in samples if entry["stage"] == "startup"]
    running = [entry["effective_available_bytes"] for entry in samples if entry["stage"] == "runtime"]
    require(starts and running and starts[-1] >= limits["admission_min_bytes"]
            and min(running) >= limits["runtime_min_bytes"], "usable memory below registered bound")
    for entry in samples:
        group = entry["owned"]
        require(group["path"] == scope and group["high_bytes"] == limits["memory_high_bytes"]
                and group["max_bytes"] == limits["memory_max_bytes"] and group["swap_max_bytes"] == limits["swap_max_bytes"]
                and group["swap_current_bytes"] == 0, f"round-99 {stage} cgroup limits differ")
    keys = ("status", "scope", "cpu_affinity", "peak_group_bytes", "elapsed_s", "supervisor_sha256", "returncode",
            "child_started", "owned_cgroup_children_empty", "final_memory_events")
    return {**{k: row[k] for k in keys}, "limits_verified": True, "startup_effective_bytes": starts[-1],
            "runtime_min_effective_bytes": min(running), "receipt_sha256": digest(receipt), "resources_sha256": digest(log)}
