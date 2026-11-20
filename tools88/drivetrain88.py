"""Compare archived D1-87 wheel targets, signed joint speeds and applied motor torques."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools87.readback87 import PROTOCOL, WHEELS, case_readback, digest, load


# robot.xml declares a 6-DOF free base followed by 4 hinges per leg, FL/FR/RL/RR.
WHEEL_DOFS = 6 + np.array(WHEELS)
PHASES = ("pre_toe", "post_toe", "last_second", "commanded")
FIELDS = ("body_vx_m_s", "wheel_target_rad_s", "wheel_actual_rad_s", "wheel_torque_nm")


def trace_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "body_vx_m_s": z["body_vx_m_s"],
            "wheel_target_rad_s": z["wheel_target_rad_s"],
            "wheel_actual_rad_s": z["qvel"][:, WHEEL_DOFS],
            "wheel_torque_nm": z["last_torque_nm"][:, WHEELS],
            "wheel_action": z["applied_action"][:, WHEELS],
            "wheel_saturation_count": z["wheel_saturation_count"],
            "front_contact_x_m": z["front_contact_x_m"],
        }


def window_means(trace, window):
    return {k: np.mean(trace[k][window], axis=0).tolist() for k in FIELDS}


def drive_limits(trace, window):
    action = np.abs(trace["wheel_action"][window])
    torque = np.abs(trace["wheel_torque_nm"][window])
    return {
        "boundaries": len(action),
        "max_abs_wheel_action": np.max(action, axis=0).tolist(),
        "abs_wheel_action_ge_0p99_counts": (action >= 0.99).sum(axis=0).tolist(),
        "abs_wheel_action_at_1_counts": (action >= 1.0).sum(axis=0).tolist(),
        "max_abs_last_pd_wheel_torque_nm": np.max(torque, axis=0).tolist(),
        "any_wheel_torque_saturation_ticks": int(np.count_nonzero(trace["wheel_saturation_count"][window])),
    }


def compare_triple(rows, traces, course):
    toe, hz, n = course["toe_x_m"], course["policy_hz"], course["ticks"]
    crossings = np.flatnonzero(traces["up"]["front_contact_x_m"][1:] >= toe) + 1
    if not len(crossings):
        raise ValueError(f"{rows['up']['case']}: a matched uphill toe crossing is required")
    anchor = int(crossings[0])
    half = hz // 2
    if anchor - half <= course["stand_ticks"] or anchor + half > n:
        raise ValueError(f"{rows['up']['case']}: matched half-second windows outside commanded motion")
    windows = {
        "pre_toe": slice(anchor-half, anchor),
        "post_toe": slice(anchor+1, anchor+half+1),
        "last_second": slice(n-hz+1, n+1),
        "commanded": slice(course["stand_ticks"]+1, n+1),
    }
    for terrain, trace in traces.items():
        row = rows[terrain]
        row["uphill_anchor_tick"] = anchor
        row["uphill_anchor_s"] = anchor / hz
        row["means"] = {name: window_means(trace, window) for name, window in windows.items()}
        row["limits"] = {
            name: drive_limits(trace, window) for name, window in {
                "commanded": windows["commanded"], "from_uphill_anchor": slice(anchor, n+1),
                "last_second": windows["last_second"],
            }.items()
        }
    return {
        "seed": rows["up"]["seed"], "training_seed": rows["up"]["training_seed"],
        "uphill_anchor_s": anchor / hz,
        "up_minus_flat": {
            p: {k: np.subtract(rows["up"]["means"][p][k], rows["flat"]["means"][p][k]).tolist()
                for k in FIELDS} for p in PHASES
        },
    }


def terrain_summary(rows):
    return {
        "episodes": len(rows), "reached_mark": sum(row["reached_mark"] for row in rows),
        "wheel_torque_saturated_ticks": sum(row["wheel_saturated_ticks"] for row in rows),
        "means_across_seeds_and_wheels": {
            p: {k: float(np.mean([row["means"][p][k] for row in rows])) for k in FIELDS} for p in PHASES
        },
    }


def build(root):
    plan, protocol_sha = load(PROTOCOL), digest(PROTOCOL)
    course, evaluation, training = plan["course"], plan["evaluation"], plan["training"]
    cases, groups = [], []
    for split, folder, stage, seeds, arms in (
        ("development", "baseline_01", "baseline", plan["baseline"]["reset_seeds"], ["baseline"]),
        ("heldout", "evaluate_01", "evaluate", evaluation["heldout_reset_seeds"], ["baseline", *training["arms"]]),
    ):
        directory = root / folder
        report = load(directory / "report.json")
        records = {(c["seed"], c["arm"], c["terrain"]): c for c in report["cases"]}
        expected = {(s, a, t) for s in seeds for a in arms for t in course["terrains"]}
        if (report["status"] != "COMPLETE" or report["stage"] != stage or report["protocol_sha256"] != protocol_sha
                or set(records) != expected or len(report["cases"]) != len(expected)):
            raise ValueError(f"{directory}: incomplete or inconsistent registered episode matrix")
        for arm in arms:
            pairs, arm_cases = [], []
            for index, seed in enumerate(seeds):
                rows, traces = {}, {}
                for terrain in course["terrains"]:
                    case = records[seed, arm, terrain]
                    path = directory / case["case"]
                    checked = case_readback(path, case, course, evaluation["course_mark_m_after_toe"])
                    train_seed = None if arm == "baseline" else training["seeds"][index]
                    if not checked["full_horizon"] or checked["training_seed"] != train_seed:
                        raise ValueError(f"{path}: registered full horizon or seed pairing mismatch")
                    rows[terrain] = {k: checked[k] for k in (
                        "case", "arm", "seed", "training_seed", "terrain", "trace_sha256",
                        "course_completed", "fallen", "illegal_contact", "reached_mark",
                        "max_loaded_front_contact_past_toe_m", "wheel_saturated_ticks",
                    )}
                    rows[terrain]["split"] = split
                    traces[terrain] = trace_arrays(path / "trace.npz")
                pairs.append(compare_triple(rows, traces, course))
                arm_cases.extend(rows.values())
            cases.extend(arm_cases)
            groups.append({
                "split": split, "arm": arm, "paired_seeds": pairs,
                "terrain_summary": {t: terrain_summary([r for r in arm_cases if r["terrain"] == t])
                                    for t in course["terrains"]},
            })
    return {
        "status": "COMPLETE", "protocol87_sha256": protocol_sha,
        "method": {
            "wheel_order": ["FL", "FR", "RL", "RR"], "qvel_dof_indices": WHEEL_DOFS.tolist(),
            "action_and_torque_indices": WHEELS, "target_rad_s_per_action": 5.0, "wheel_torque_limit_nm": 12.0,
            "command_forward_m_s": course["command_forward_m_s"], "policy_hz": course["policy_hz"],
            "toe_x_m": course["toe_x_m"], "course_mark_m_after_toe": evaluation["course_mark_m_after_toe"],
            "anchor": "For each arm and reset, first uphill observation endpoint whose farthest front-wheel "
                      "contact with normal force >1 N is at or past the toe. Use its clock for all three terrains.",
            "windows": "pre_toe covers [anchor-0.5 s,anchor), post_toe (anchor,anchor+0.5 s], last_second (5,6], "
                       "commanded (1,6], from_uphill_anchor [anchor,6]. Row zero is excluded.",
            "signals": "body_vx is measured in the body frame. Wheel actual is signed foot-joint qvel at the "
                       "observation endpoint. Torque is sampled at the last PD update 4 ms before the endpoint. "
                       "The archived saturation flag includes all five PD updates in each 20 ms interval.",
            "aggregation": "Each case contains time means per wheel. Terrain summaries equally weight the three "
                           "resets and four wheels. Paired differences retain wheel order and individual reset seeds.",
        },
        "groups": groups, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report = build(args.root)
            payload = json.dumps(report, allow_nan=False, indent=2)
        except Exception as error:
            json.dump({"status": "INCOMPLETE", "error": str(error)}, stream, indent=2)
            raise
        stream.write(payload + "\n")
    print(json.dumps({"status": report["status"], "episodes": len(report["cases"])}))


if __name__ == "__main__":
    main()
