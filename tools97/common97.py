"""Registration and provenance of the late D1 world-model replay."""
import json

from tools91.launch91 import require
from tools92.launch92 import scope_for
from tools94.common94 import ROOT, digest, read
from tools95.run95 import files_for_seal
from tools96.common96 import BASE as PARENT


BASE = ROOT / "evidence/suffix_replay97"
PLAN = ROOT / "docs/experiments97/suffix_replay.json"
PLAN_SHA = "73817918e213ca6ba36b599c132cb1bea40a8005358a3d73df2ae37ddc639e99"
STAGES = ("fixed", "anchored", "video")


def protocol(private=False):
    require(digest(PLAN) == PLAN_SHA, "the registered replay protocol has changed")
    plan = read(PLAN)
    for filename, expected in plan["parents_sha256"].items():
        require(digest(ROOT / filename) == expected, f"registered parent differs: {filename}")
    if private:
        for stage, expected in plan["local_parent_seals_sha256"].items():
            require(digest(PARENT / f"{stage}_01/seal.json") == expected, f"round-96 {stage} seal differs")
    return plan


def unit_name(stage):
    require(stage in STAGES, "unknown suffix replay stage")
    return f"d1-suffix97-{stage}-01"


def sources_for(stage):
    require(stage in STAGES, "unknown suffix replay producer")
    names = ["tools97/common97.py", "tools97/launch97.py"]
    if stage in ("fixed", "anchored"):
        names.extend(("tools97/replay97.py", "tools96/fit96.py", "tools96/common96.py",
                      "tools95/run95.py", "tools94/learn94.py", "tools91/model91.py",
                      "tools52/dynamics52.py", "training/configs/pilot03d.json"))
    else:
        names.extend(("tools97/video97.py", "tools96/video96.py", "tools95/video95.py"))
    return {name: digest(ROOT / name) for name in names}


def previous_seal(stage):
    if stage == "fixed":
        return PARENT / "native_01/seal.json"
    return BASE / f"{STAGES[STAGES.index(stage)-1]}_01/seal.json"


def checked_stage(stage):
    folder = BASE / f"{stage}_01"
    seal = read(folder / "seal.json")
    require(seal["stage"] == stage and seal["protocol_sha256"] == PLAN_SHA
            and seal["sources"] == sources_for(stage) and seal["previous_seal_sha256"] == digest(previous_seal(stage))
            and seal["files"] == files_for_seal(folder), f"changed or incomplete {stage} archive")
    return folder


def resource_record(stage):
    folder = BASE / f"guard_{stage}_01"
    report_file, sample_file = folder / "report.json", folder / "resources.jsonl"
    row = read(report_file)
    limits = protocol()["resources"]
    scope = scope_for(unit_name(stage))
    require(row["scope"] == scope and row["status"] == "COMPLETE" and row["child_started"]
            and row["owned_cgroup_children_empty"] and row["returncode"] == 0
            and len(row["cpu_affinity"]) == limits["cpu_affinity_count"]
            and row["supervisor_sha256"] == limits["supervisor_sha256"], f"incomplete {stage} supervisor")
    samples = [json.loads(line) for line in sample_file.read_text().splitlines()]
    startup = [s["effective_available_bytes"] for s in samples if s["stage"] == "startup"]
    running = [s["effective_available_bytes"] for s in samples if s["stage"] == "runtime"]
    require(startup and running and startup[-1] >= limits["admission_min_bytes"]
            and min(running) >= limits["runtime_min_bytes"], f"{stage} memory reserve not met")
    for sample in samples:
        group = sample["owned"]
        require(group["path"] == scope and group["high_bytes"] == limits["memory_high_bytes"]
                and group["max_bytes"] == limits["memory_max_bytes"] and group["swap_max_bytes"] == 0
                and group["swap_current_bytes"] == 0, f"unexpected {stage} cgroup limits")
    fields = ("status", "scope", "cpu_affinity", "peak_group_bytes", "elapsed_s", "supervisor_sha256",
              "returncode", "child_started", "owned_cgroup_children_empty", "final_memory_events")
    return {**{name: row[name] for name in fields}, "limits_verified": True,
            "startup_effective_bytes": startup[-1], "runtime_min_effective_bytes": min(running),
            "receipt_sha256": digest(report_file), "resources_sha256": digest(sample_file)}
