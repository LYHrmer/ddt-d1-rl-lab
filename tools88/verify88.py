"""Independent readback of the six C++/MuJoCo traces, including rear-wheel support."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools87.readback87 import check, compare, digest, load
from tools88.contact88 import analyze_trace
from tools88.probe88 import PLAN
from training.d1_rl.interface import LEG_INDICES, WHEEL_INDICES


def audit(root):
    from tools87.terrain87 import load_model

    plan, report = load(PLAN), load(root / "report.json")
    parent = PLAN.parents[2] / plan["parent_protocol"]
    course, identity = (load(parent)[k] for k in ("course", "parent"))
    check(report["status"] == "COMPLETE" and digest(parent) == plan["parent_protocol_sha256"]
          and report["protocol_sha256"] == digest(PLAN), "probe and parent protocols")
    index = {(c["arm"], c["seed"]): c for c in report["cases"]}
    expected = {(a, s) for a in plan["arm_delta_rad_s"] for s in plan["reset_seeds"]}
    check(set(index) == expected and len(report["cases"]) == len(expected), "case matrix")
    model = load_model(plan["terrain"])
    rows, pairs = [], []
    for seed in plan["reset_seeds"]:
        matrices = {}
        pair_rows = {}
        for arm, delta in plan["arm_delta_rad_s"].items():
            key = f"{arm}_{seed}"
            folder = root / key
            c = index[arm, seed]
            path = folder / "trace.npz"
            check(load(folder / "metrics.json") == c and digest(path) == c["trace_sha256"], key)
            cpp = load(folder / "cpp34/report.json")
            check(cpp["status"] == "CLOSED" and cpp["cleanup_complete"] and cpp["returncode"] == 0
                  and cpp["requests_attempted"] == cpp["requests_completed"] == c["ticks"]
                  and cpp["ready"]["cpu_threads"] == 1 and cpp["identities"]["binary_sha256"] == identity["cpp34_sha256"]
                  and cpp["identities"]["model_sha256"] == identity["onnx_sha256"], key + " C++ actions")
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            t, raw, issued, applied = (a[k] for k in ("time_s", "raw_cpp_action", "issued_action", "applied_action"))
            n = len(t) - 1
            check(all(len(v) == n + 1 for v in a.values()) and n == c["ticks"]
                  and np.allclose(t, np.arange(n + 1) / course["policy_hz"], atol=1e-8, rtol=0), key + " times")
            moving = np.arange(n) >= course["stand_ticks"]
            check(np.allclose(a["command_vx_m_s"][1:], moving * course["command_forward_m_s"], rtol=0, atol=1e-7)
                  and np.array_equal(issued[:-1], applied[1:]), key + " command/FIFO")
            demand = raw[:, WHEEL_INDICES].copy()
            demand[plan["issue_start_tick"] + 1:] += np.float32(delta / plan["wheel_scale_rad_s"])
            check(np.array_equal(issued[:, WHEEL_INDICES], np.clip(demand, -1, 1))
                  and np.array_equal(issued[:, LEG_INDICES], raw[:, LEG_INDICES])
                  and np.array_equal(applied[:, WHEEL_INDICES] * plan["wheel_scale_rad_s"], a["wheel_target_rad_s"]),
                  key + " changed target rows")
            clip_count = int(np.sum(np.abs(demand) > 1))
            compare(c["wheel_offsets_clipped"], clip_count, key + " clipping")
            start, stop = plan["primary"]["interval_end_ticks_exclusive_inclusive"]
            interval = np.arange(n + 1)
            window = (interval > start) & (interval <= stop)
            vx = float(a["body_vx_m_s"][window].mean()) if n >= stop else None
            compare(c["score_vx_m_s"], vx, key + " speed")
            contact = a["front_contact_x_m"][1:]
            finite = contact[np.isfinite(contact)]
            past = float(finite.max() - course["toe_x_m"]) if finite.size else None
            compare(c["max_loaded_front_contact_past_toe_m"], past, key + " contact progress")
            crossed = past is not None and past >= plan["independent_course_mark_m_past_toe"]
            compare(c["reached_mark"], crossed, key + " mark")
            clean = n == course["ticks"] and not (np.any(a["fallen"]) or np.any(a["illegal_body_contact"]))
            compare(c["course_completed"], bool(clean), key + " termination")
            contacts = analyze_trace(model, a, dict(c, case=key, training_seed=None, terrain=plan["terrain"]), course)
            check(all(item["passed"] for item in contacts["validation"].values()), key + " contact geometry")
            measured = {"seed": seed, "arm": arm, "ticks": n, "trace_sha256": c["trace_sha256"], "full_clean": bool(clean),
                        "score_vx_m_s": vx, "max_loaded_front_contact_past_toe_m": past, "reached_mark": crossed,
                        "wheel_offsets_clipped": clip_count, "wheel_saturated_ticks": int((a["wheel_saturation_count"] > 0).sum()),
                        "peak_abs_last_pd_wheel_torque_nm": float(np.abs(a["last_torque_nm"][:, WHEEL_INDICES]).max()),
                        "first_loaded_toe_s": contacts["first_loaded_contact_past_toe_s"],
                        "last_second_contact": contacts["windows"]["last_second"]}
            if vx is not None:
                measured["score_window_four_wheel_means"] = {
                    "target_rad_s": float(a["wheel_target_rad_s"][window].mean()),
                    "actual_rad_s": float(a["qvel"][window][:, 6 + WHEEL_INDICES].mean()),
                    "last_pd_torque_nm": float(a["last_torque_nm"][window][:, WHEEL_INDICES].mean()),
                }
            rows.append(measured)
            pair_rows[arm], matrices[arm] = measured, a
        a, b = (matrices[k] for k in ("original", "wheel_plus"))
        ncommon = min(len(a["time_s"]), len(b["time_s"]))
        issue = plan["issue_start_tick"] + 1
        effect = issue + course["action_delay_ticks"]
        check(np.array_equal(a["issued_action"][:min(issue, ncommon)], b["issued_action"][:min(issue, ncommon)]),
              f"{seed} common issues")
        for k in ("qpos", "qvel", "raw_cpp_action", "applied_action"):
            check(np.array_equal(a[k][:min(effect, ncommon)], b[k][:min(effect, ncommon)]), f"{seed} common {k}")
        if ncommon > effect:
            check(not np.array_equal(a["applied_action"][effect], b["applied_action"][effect])
                  and not np.array_equal(a["qvel"][effect], b["qvel"][effect]), f"{seed} onset")
        orig, plus = pair_rows["original"], pair_rows["wheel_plus"]
        delta = (None if orig["score_vx_m_s"] is None or plus["score_vx_m_s"] is None
                 else plus["score_vx_m_s"] - orig["score_vx_m_s"])
        passed = bool(orig["full_clean"] and plus["full_clean"] and delta is not None
                      and delta >= plan["primary"]["per_pair_min_delta_mean_body_vx_m_s"])
        pairs.append({"seed": seed, "delta_vx_m_s": delta, "passes": passed})
    check(len(pairs) == len(report["pairs"]), "paired count")
    for pair, saved in zip(pairs, report["pairs"], strict=True):
        for k, value in pair.items():
            compare(saved[k], value, f"{pair['seed']} {k}")
    check(all(p["passes"] for p in pairs) == report["primary_pass"], "primary verdict")
    return {"status": "COMPLETE", "protocol_sha256": digest(PLAN), "probe_report_sha256": digest(root / "report.json"),
            "native_steps_in_readback": 0, "cases": rows, "pairs": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as out:
        try:
            result = audit(args.root)
        except Exception as exc:
            json.dump({"status": "INCOMPLETE", "error": str(exc)}, out)
            raise
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
