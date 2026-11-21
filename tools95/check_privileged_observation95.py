"""Audit the archived native actor inputs. Only the observation arithmetic runs."""

import argparse
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy

if __name__ == "__main__":
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "evidence/policy_direction95/native_01"
RESULT = ROOT / "reports/policy_direction95.json"
REPORT = ROOT / "reports/policy_direction95_observation.json"
EXAMPLE = ROOT / "reports/policy_direction95_observation_example.npz"
EXAMPLE_ID = (94011, 95101, "zero")
INTERFACE = ROOT / "training/d1_rl/interface.py"
SENSOR = ROOT / "tools87/terrain87.py"
# Import the frozen NumPy interface without importing the environment from d1_rl/__init__.py.
BODY = runpy.run_path(str(INTERFACE))
WHEELS = BODY["WHEEL_INDICES"]


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def packing_check():
    # Load only the original NumPy packet encoder and its two range constants.
    tree = ast.parse(SENSOR.read_bytes())
    selected = [n for n in tree.body if (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                and [t.id for t in n.targets[0].elts] == ["MIN_RANGE_M", "MAX_RANGE_M"])
                or (isinstance(n, ast.FunctionDef) and n.name == "range_features")]
    namespace = {"np": np}
    exec(compile(ast.Module(selected, type_ignores=[]), str(SENSOR), "exec"), namespace)
    distances = [0.14, 1.45, 0.139, 1.451, 0.5, -1.0, np.nan, 0.725]
    blocked = [False, False, False, False, True, False, False, False]
    expected = np.array([0.14/1.45, 1, 0, 0, 0, 0, 0, 0.5, 1, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    require(np.array_equal(namespace["range_features"](distances, blocked), expected), "packet encoder fixture")
    return {"synthetic_beams": 8, "valid": 3, "invalid": 5}


def visible_packets(t):
    n = len(t["obs643"])
    captures = [i for i in range(n) if i == 0 or i*6//50 != (i-1)*6//50]
    require(np.array_equal(np.flatnonzero(t["ray_due"]), captures), "6 Hz capture schedule")
    raw = t["ray_raw16"]
    require(raw.shape == (n, 16) and not np.any(raw[~t["ray_due"]]), "unexpected raw packet outside capture")
    values, flags = raw[captures, :8], raw[captures, 8:]
    valid = flags == 1
    require(np.all(valid | (flags == 0)) and np.all(values[~valid] == 0), "invalid ray flag or invalid ray value")
    require(np.all((values[valid] >= np.float32(0.14/1.45)) & (values[valid] <= 1)), "ray outside encoded range")
    blank = np.zeros(16, dtype=np.float32)
    packets = np.empty_like(raw)
    last = delivery = pending = -1
    for tick in range(n):
        if tick in captures:
            pending = tick
        if pending >= 0 and tick == pending + 2:
            last, delivery, pending = pending, tick, -1
        require(t["sensor_capture_tick"][tick] == last and t["sensor_delivery_tick"][tick] == delivery,
                "delivered capture identity or delivery clock")
        require(t["pending_capture_tick"][tick] == pending and
                np.array_equal(t["pending_ray16"][tick], raw[pending] if pending >= 0 else blank), "pending packet")
        packets[tick] = raw[last] if last >= 0 else blank
    return packets, captures, int(valid.sum())


def rebuild(states, issued, packets):
    # The 32-tick stand still uses a zero command at every endpoint, including the terminal frame.
    previous = np.zeros(3)
    frames, filters = [], []
    for i, state in enumerate(states):
        action = issued[i - 1] if i else np.zeros(16)
        frame, previous = BODY["build_observation"](state[26:29], state[3:7], np.zeros(3),
                                                    state[7:23], state[29:45], action, previous)
        frames.append(frame)
        filters.append(previous)
    frames = np.stack(frames)
    history_ticks = np.maximum(np.arange(len(frames))[:, None] - np.arange(10, 0, -1), 0)
    proprio = np.stack([BODY["pack_actor_input"](frame, frames[indices]) for frame, indices in zip(frames, history_ticks)])
    return np.concatenate((proprio, packets), axis=1), np.stack(filters)


def audit_trace(path):
    fields = ("state45", "obs643", "issued16", "applied16", "command3", "filtered3", "ray_raw16", "ray_due",
              "sensor_capture_tick", "sensor_delivery_tick", "pending_capture_tick", "pending_ray16")
    with np.load(path, allow_pickle=False) as archive:
        t = {name: archive[name] for name in fields}
    states, observed, issued = t["state45"], t["obs643"], t["issued16"]
    n = len(states)
    require(n == 33 and states.shape == (n, 45) and observed.shape == (n, 643) and issued.shape == (n-1, 16),
            "round-95 trace shape")
    require(not t["command3"].any() and not states[0, 23:].any(), "stationary command or reset velocity differs")
    require(not t["applied16"][0].any() and np.array_equal(issued[:-1], t["applied16"][1:]), "one-tick action FIFO")
    packets, captures, valid = visible_packets(t)
    inputs, filtered = rebuild(states, issued, packets)
    require(np.array_equal(inputs, observed), "reconstructed current frame, history or visible packet differs")
    require(not observed[:, 9+WHEELS].any(), "wheel angle position slots must be zero")
    filter_error = float(np.max(np.abs(filtered - t["filtered3"])))
    require(filter_error < 1e-12, "body gyro filter differs")

    mutations = {"world_xy": ([0, 1], [1.0, -2.0]), "world_linear_velocity_xyz": ([23, 24, 25], [3., -2., 1.]),
                 "wheel_angles": (7+WHEELS, [1.25, -2.5, 3.75, -5.0])}
    for name, (indices, delta) in mutations.items():
        altered = states.copy()
        altered[:, indices] += delta
        request, _ = rebuild(altered, issued, packets)
        require(np.array_equal(request, observed), f"{name} changes actor input with packets and issued actions held fixed")
    altered = states.copy()
    altered[:, 29+WHEELS] += 1.0  # Positive control: wheel angular speed is measured in this interface.
    positive, _ = rebuild(altered, issued, packets)
    require(np.all(positive[:, 25+WHEELS] != observed[:, 25+WHEELS]), "wheel-speed positive control")
    return {"endpoints": n, "decisions": n-1, "capture_ticks": captures, "delivery_ticks": [k+2 for k in captures],
            "pre_delivery_endpoints": int((t["sensor_delivery_tick"] < 0).sum()), "valid_capture_beams": valid,
            "invalid_capture_beams": 8*len(captures)-valid, "max_filter_abs_error": filter_error}


def trace_ids():
    cells = json.loads(RESULT.read_bytes())["cells"]
    return {(c["training_seed"], c["reset"], arm): row["native"]["source_sha256"]
            for c in cells for arm, row in c["arms"].items()}


def full_audit():
    ids = trace_ids()
    expected = {(s, r, a) for s in (94011, 94023, 94037) for r in (95101, 95102, 95103)
                for a in ("m1", "mh", "zero", "ph", "p1")}
    require(set(ids) == expected, "incomplete trace manifest")
    rows = {}
    for key, sha in ids.items():
        seed, reset, arm = key
        path = NATIVE / f"seed_{seed}/reset_{reset}/{arm}/native/trace.npz"
        require(digest(path) == sha, f"trace hash {key}")
        rows[key] = audit_trace(path)
    endpoints = sum(r["endpoints"] for r in rows.values())
    require(all(r["capture_ticks"] == [0, 9, 17, 25] and r["delivery_ticks"] == [2, 11, 19, 27] for r in rows.values()),
            "unexpected set of capture and delivery times")
    return {
        "source_identity": {str(p.relative_to(ROOT)): digest(p) for p in (RESULT, INTERFACE, SENSOR)},
        "traces": len(ids), "endpoints": endpoints, "decisions": sum(r["decisions"] for r in rows.values()),
        "exact_reconstructed_current57_history10x57_and_visible16": endpoints,
        "wheel_position_slots_zero": 4*endpoints,
        "unchanged_full_inputs_with_fixed_packets_and_issued_actions": dict.fromkeys(
            ("world_xy", "world_linear_velocity_xyz", "wheel_angles"), endpoints),
        "positive_control_changed_all_four_current_wheel_velocity_slots": endpoints,
        "max_float64_gyro_filter_abs_error": max(r["max_filter_abs_error"] for r in rows.values()),
        "capture_and_delivery_ticks_per_trace": {"capture": [0, 9, 17, 25], "delivery": [2, 11, 19, 27]},
        "packets_captured": sum(len(r["capture_ticks"]) for r in rows.values()),
        "packets_delivered": sum(len(r["delivery_ticks"]) for r in rows.values()),
        "endpoints_before_first_delivery": sum(r["pre_delivery_endpoints"] for r in rows.values()),
        "valid_capture_beams": sum(r["valid_capture_beams"] for r in rows.values()),
        "invalid_capture_beams": sum(r["invalid_capture_beams"] for r in rows.values()),
        "range_encoder_offline_unit_check": packing_check(),
        "public_example": {"seed": EXAMPLE_ID[0], "reset": EXAMPLE_ID[1], "arm": EXAMPLE_ID[2],
                           "source_sha256": ids[EXAMPLE_ID], "metrics": rows[EXAMPLE_ID]},
        "scope": ["Native inputs include the terminal observation. Only the first 32 of 33 observations per arm led to a decision.",
                  "Mutations test the recorded input mapping with the received packets and issued controls fixed. Actual motion can change future measurements, contacts and history.",
                  "Attitude and gyro use an ideal simulated IMU. Robot geometry supplies simulated first-return ranges and modeled self-occlusion.",
                  "All eight captured beams were valid on these starts. Invalid measurements are tested only by the separate encoder fixture.",
                  "This audit does not execute the learned model or actor, measure hardware sensor errors, or evaluate inference from correlations."],
    }


def main():
    args = argparse.ArgumentParser(description=__doc__)
    mode = args.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-example", action="store_true", help="Audit the single public raw trace and the encoder")
    mode.add_argument("--check-all", action="store_true", help="Rebuild the report using all local frozen traces")
    mode.add_argument("--write", action="store_true", help="Create the full local report without overwriting a prior report")
    selected = args.parse_args()
    if selected.check_example:
        report = json.loads(REPORT.read_bytes())
        example = report["public_example"]
        require(report["source_identity"] == {str(p.relative_to(ROOT)): digest(p)
                                              for p in (RESULT, INTERFACE, SENSOR)}, "audited source identity")
        require((example["seed"], example["reset"], example["arm"]) == EXAMPLE_ID, "public example selection")
        require(digest(EXAMPLE) == trace_ids()[EXAMPLE_ID] == example["source_sha256"], "public example identity")
        require(audit_trace(EXAMPLE) == example["metrics"] and packing_check() == report["range_encoder_offline_unit_check"],
                "public example metrics")
        print("Verified public trace, packet timing and counterfactuals.")
    else:
        report = full_audit()
        if selected.check_all:
            require(report == json.loads(REPORT.read_bytes()), "full audit mismatch")
            print("Verified all 45 native traces.")
        else:
            with REPORT.open("x") as out:
                json.dump(report, out, indent=2, allow_nan=False)
                out.write("\n")


if __name__ == "__main__":
    main()
