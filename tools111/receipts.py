"""Publish and verify the native static probes and their resource and failure receipts."""
import argparse

from tools91.launch91 import ROOT, digest, read, write
from tools111.run import BASE, PLAN, PLAN_SHA, complete, sources


OUT = ROOT/"reports/lidar_compound111/preflight.json"


def publish():
    folder, stage, usage = complete("preflight")
    failures = []
    for guard in sorted(BASE.glob("guard_preflight_??")):
        raw = read(guard/"report.json")
        if raw["status"] == "COMPLETE":
            continue
        trial = BASE/guard.name.removeprefix("guard_")
        item = dict(attempt=trial.name, status=raw["status"], child_started=raw["child_started"],
                    minimum_effective_bytes=raw["minimum_effective_bytes"], error=raw["error"],
                    guard_sha256=digest(guard/"report.json"), log_sha256=digest(guard/"resources.jsonl"))
        if (trial/"partial.json").is_file():
            item["partial"] = read(trial/"partial.json")
            item["stderr"] = (guard/"stderr.log").read_text().splitlines()[-5:]
        failures.append(item)
    partial = BASE/"archive_export_01/partial.json.fragment"
    guard = BASE/("guard_"+folder.name)
    write(OUT, dict(attempt=folder.name, stage=stage, resource=usage, failures=failures,
                    guard_sha256=digest(guard/"report.json"), resource_log_sha256=digest(guard/"resources.jsonl"),
                    archive_export_failure=dict(reason=(partial.parent/"reason.txt").read_text().strip(),
                                                fragment_sha256=digest(partial), bytes=partial.stat().st_size)))


def verify():
    proof = read(OUT)
    stage, resource = proof["stage"], proof["resource"]
    assert stage["status"] == "COMPLETE" and stage["plan_sha256"] == digest(PLAN) == PLAN_SHA
    assert stage["sources"] == sources() and stage["physical_substeps"] == 0
    assert resource["startup_effective_bytes"] >= 4<<30 and resource["minimum_effective_bytes"] >= 2<<30
    assert resource["owned_cgroup_children_empty"] and len(resource["cpu_affinity"]) == 1
    flat, pitched, compound = stage["live_sensor"]
    for row in (flat, pitched):
        assert row["layout"] == "flat" and row["speed_reduction_m_s"] == 0. and max(row["demand_m"]) == 0.
    assert abs(pitched["pitch_rad"]-.06) < 1e-12 and flat["pitch_rad"] == 0.
    assert compound["layout"] == "development" and compound["speed_reduction_m_s"] >= .020
    assert min(compound["demand_m"][:2]) >= .005 and max(compound["demand_m"][2:]) == 0.
    assert len(stage["terrain_probes"]) == 58
    assert max(abs(physical-expected) for _,_,_,physical,expected in stage["terrain_probes"]) <= .00065
    assert len(stage["kinematic_checks"]) == 6
    assert all(max(r["max_vertical_error_m"], r["max_horizontal_m"]) < .0004 for r in stage["kinematic_checks"])
    assert stage["static_ray_queries"] < 60000 and all(r["status"] == "INCOMPLETE" for r in proof["failures"])
    return stage["static_ray_queries"], compound["rejected_batch_disagreements"], len(proof["failures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    if parser.parse_args().publish:
        publish()
    print(verify())
