"""Measure each learned policy's slope response on the same archived 88 state/action history."""
import json

import numpy as np

from tools87.readback87 import WHEELS, check
from tools89.common89 import BASE, PLAN, ROOT, check_worker, complete_stage, read, registered_plan, sha
from tools89.eval89 import checked_models, load_actor


def stats(delta):
    """Each of the three traces contributes exactly 100 decisions and four wheel targets."""
    check(delta.shape == (3, 100, 4) and np.isfinite(delta).all(), "expected complete paired target arrays")
    return {"signed_mean_rad_s": float(delta.mean()), "mean_abs_rad_s": float(np.abs(delta).mean()),
            "per_wheel_signed_mean_rad_s": delta.mean(axis=(0, 1)).tolist(),
            "reference_signed_means_rad_s": delta.mean(axis=(1, 2)).tolist()}


def references():
    from tools88.policy88 import observations_and_rewards, recast_packets

    root = ROOT / "evidence/uphill88/probe_01"
    public = read(ROOT / "reports/uphill_diagnosis88.json")
    protocol = ROOT / "docs/experiments88/uphill_wheel_probe.json"
    plan88, report = read(protocol), read(root / "report.json")
    check(report["status"] == "COMPLETE" and sha(root / "report.json") == public["probe_report_sha256"]
          and report["protocol_sha256"] == public["protocol_sha256"] == sha(protocol), "88 fixed-state source")
    sources, inputs = [], []
    for seed in plan88["reset_seeds"]:
        folder = root / f"original_{seed}"
        row = read(folder / "metrics.json")
        path = folder / "trace.npz"
        check(row in report["cases"] and row["seed"] == seed and row["arm"] == "original" and row["course_completed"]
              and row["ticks"] == 300 and sha(path) == row["trace_sha256"], folder.name + " trace identity")
        with np.load(path, allow_pickle=False) as trace:
            data = {k: trace[k] for k in ("qpos", "qvel", "time_s", "issued_action", "raw_ranges16",
                                          "command_vx_m_s", "body_vx_m_s")}
            check(np.array_equal(trace["raw_cpp_action"], data["issued_action"]), folder.name + " original motor output")
        case = dict(row, case=folder.name, terrain="up")
        bodies, scale, velocity = observations_and_rewards(data, case, 50)
        up, up_deliveries = recast_packets(data, case, "up")
        flat, flat_deliveries = recast_packets(data, case, "flat")
        check(up.shape == (300, 16) and np.array_equal(up, data["raw_ranges16"][1:])
              and up_deliveries == flat_deliveries and np.all(scale == 5.), folder.name + " range replay")
        sources.append({"case": folder.name, "seed": seed, "trace_sha256": sha(path),
                        "metrics_sha256": sha(folder / "metrics.json"), "archived_range_values_matched": up.size,
                        "delivered_captures": len(up_deliveries), "max_archived_vx_error_m_s": velocity["vx_error"],
                        "window_decisions_with_distance_change": int(np.any(up[200:300, :8] != flat[200:300, :8], axis=1).sum()),
                        "window_decisions_with_validity_change": int(np.any(up[200:300, 8:] != flat[200:300, 8:], axis=1).sum())})
        # Observation for decision k comes from boundary k; the issued action integrates [k,k+1).
        inputs.append((bodies[200:300], up[200:300], flat[200:300], scale))
    return sha(root / "report.json"), sources, inputs


def audit():
    plan = registered_plan()
    complete_stage("training")
    training = BASE / "training_01"
    models = checked_models(training, plan)
    source_hash, sources, inputs = references()
    rows = []
    for seed in plan["training"]["seeds"]:
        targets, hashes = {}, {}
        for arm in ("original", "mixed"):
            weights, model_hash = models[arm, seed]
            hashes[arm] = model_hash
            predict = load_actor(weights)
            for surface in ("up", "flat"):
                wheels = []
                for body, up, flat, scale in inputs:
                    packet = up if surface == "up" else flat
                    actions = np.stack([predict(b, z) for b, z in zip(body, packet, strict=True)])
                    wheels.append(actions[:, WHEELS] * scale)
                targets[arm, surface] = np.stack(wheels)
        old_response = targets["original", "up"] - targets["original", "flat"]
        new_response = targets["mixed", "up"] - targets["mixed", "flat"]
        rows.append({"training_seed": seed, "model_sha256": hashes, "comparisons": {
            "mixed_minus_original_on_up": stats(targets["mixed", "up"] - targets["original", "up"]),
            "original_up_minus_flat": stats(old_response), "mixed_up_minus_flat": stats(new_response),
            "curriculum_change_in_up_minus_flat": stats(new_response - old_response)}})
    return {"status": "COMPLETE", "protocol_sha256": sha(PLAN), "source_sha256": sha(__file__),
            "training_report_sha256": sha(training / "report.json"), "probe88_report_sha256": source_hash,
            "native_steps": 0, "optimizer_steps": 0, "fixed_history": "Original 88 C++ decisions, proprioception and issued action history",
            "decision_ticks_inclusive_exclusive": [200, 300], "wheel_order": ["FL", "FR", "RL", "RR"],
            "source_cases": sources, "training_seeds": rows}


if __name__ == "__main__":
    check_worker("sensor")
    with (BASE / "sensor_01.json").open("x") as stream:
        try:
            result = audit()
        except BaseException as exc:
            json.dump({"status": "INCOMPLETE", "error": str(exc)}, stream)
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
