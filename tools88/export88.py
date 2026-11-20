"""Export shareable 50 Hz measurements and verified summaries from the D1 experiment."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools87.readback87 import check, digest, load
from tools87.run87 import write
from tools88.probe88 import PLAN


def resources(guard):
    result = load(guard / "report.json")
    check(result["status"] == "COMPLETE" and result["owned_cgroup_children_empty"]
          and result["child_reaped"] and result["returncode"] == 0
          and len(result["cpu_affinity"]) == 1, guard.name)
    with (guard / "resources.jsonl").open() as stream:
        records = [json.loads(line) for line in stream]
    for r in records:
        own = r["owned"]
        check((own["high_bytes"], own["max_bytes"], own["swap_max_bytes"], own["swap_current_bytes"])
              == (2 << 30, 3 << 30, 0, 0) and not r["errors"]
              and r["effective_available_bytes"] >= 2 << 30 and not own["events"]["oom_kill"], guard.name)
    startup = [r["effective_available_bytes"] for r in records if r["stage"] == "startup"][-1]
    check(startup >= 4 << 30, guard.name + " admission")
    return {"startup_effective_bytes": startup, **{k: result[k] for k in
            ("minimum_effective_bytes", "peak_group_bytes", "elapsed_s", "owned_cgroup_children_empty",
             "cpu_affinity", "supervisor_sha256")}}


def export(root, output):
    folder = root / "readback_01"
    verified, drive, contacts, policy = (load(folder / (name + ".json"))
                                         for name in ("verify88", "drivetrain88", "contact88", "policy88"))
    probe = load(root / "probe_01/report.json")
    check(all(r["status"] == "COMPLETE" for r in (probe, verified, drive, contacts, policy))
          and contacts["validation_passed"] and verified["probe_report_sha256"] == digest(root / "probe_01/report.json"),
          "incomplete physical or archive receipts")
    plan = load(PLAN)
    parent = PLAN.parents[2] / plan["parent_protocol"]
    check(digest(PLAN) == probe["protocol_sha256"] and digest(parent) == plan["parent_protocol_sha256"]
          and all(r["protocol87_sha256"] == digest(parent) for r in (drive, contacts, policy)), "protocols")
    toe = load(parent)["course"]["toe_x_m"]
    guards = {s: resources(root / f"guard_{s}_01") for s in ("probe", "archive")}
    output.mkdir(exist_ok=False)
    measurements = output / "uphill88_measurements.csv"
    with measurements.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("seed", "arm", "end_tick", "body_vx_m_s", "loaded_front_past_toe_m", "fallen",
                         "illegal_contact", "wheel_saturation_count", "wheel_offsets_clipped"))
        for row in verified["cases"]:
            arm, seed = row["arm"], row["seed"]
            trace = root / "probe_01" / f"{arm}_{seed}" / "trace.npz"
            check(digest(trace) == row["trace_sha256"], trace.name)
            with np.load(trace, allow_pickle=False) as z:
                vx, x, fallen, illegal, saturation, clipped = (z[k][1:] for k in
                    ("body_vx_m_s", "front_contact_x_m", "fallen", "illegal_body_contact",
                     "wheel_saturation_count", "wheel_offset_clipped"))
            for i in range(len(vx)):
                past = float(x[i] - toe) if np.isfinite(x[i]) else ""
                writer.writerow((seed, arm, i + 1, float(vx[i]), past, int(fallen[i]), int(illegal[i]),
                                 int(saturation[i]), int(clipped[i].sum())))
    dev_contact = [{k: c[k] for k in ("seed", "terrain", "first_loaded_contact_past_toe_s", "max_loaded_contact_x_m")}
                   | {"last_second": c["windows"]["last_second"]} for c in contacts["cases"]
                   if c["split"] == "development"]
    archive = {"cases": len(drive["cases"]),
               "torque_saturated_ticks": sum(c["wheel_saturated_ticks"] for c in drive["cases"]),
               "drivetrain": [{"split": g["split"], "arm": g["arm"], "terrain_summary": g["terrain_summary"]}
                              for g in drive["groups"]],
               "original_contact": dev_contact, "contact_boundaries": contacts["boundary_forward_calls"],
               "contact_position_validation": contacts["validation_passed"],
               "range_training_exposure": policy["training_exposure"],
               "uphill_sensor_same_pose": policy["same_pose_uphill_vs_flat_sensor"],
               "range_uphill_last_second": [{"train_seed": c["training_seed"], "reset_seed": c["reset_seed"],
                                             **c["periods"]["last_second"]} for c in policy["range_cases"] if c["terrain"] == "up"],
               "parent_reconstruction_max_action_error": max(x["max_abs_error"] for x in policy["parent_uphill_cpp_checks"]),
               "ppo_reconstruction_max_action_error": max(x["issued_action_max_abs_error"] for x in policy["range_cases"]),
               "sensor_mask_reconstruction_max_error": max(x["per_tick_mask_max_abs_error"] for x in policy["range_cases"])}
    result = {k: probe[k] for k in ("protocol_sha256", "source_sha256", "primary_pass", "policy_transitions",
                                     "training_transitions", "pd_updates", "native_steps", "pairs")}
    result.update(cases=verified["cases"], archive87=archive, resources=guards,
                  measurements_sha256=digest(measurements), probe_report_sha256=verified["probe_report_sha256"],
                  readback_sha256={p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()})
    write(output / "uphill_diagnosis88.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.output)
